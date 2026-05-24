from __future__ import annotations

import gc
import json
import warnings
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from exp23_config import WorkflowConfig
from exp23_utils import choose_binary_n_splits, choose_regression_n_splits, slugify

warnings.filterwarnings("ignore", category=ConvergenceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    representation: str | None = None
    feature_mode: str = "full"
    tune: bool = True


def get_main_models(cfg: WorkflowConfig) -> tuple[ModelSpec, ...]:
    return (
        ModelSpec(name="Baseline", representation=None, feature_mode="baseline", tune=False),
        ModelSpec(name="Baseline+Standard", representation="Standard", feature_mode="full", tune=True),
        ModelSpec(name=f"Baseline+Universal-BiCGR d={cfg.universal_depth}", representation=f"Universal-BiCGR d={cfg.universal_depth}", feature_mode="full", tune=True),
    )

def get_minimal_sensitivity_models(cfg: WorkflowConfig) -> tuple[ModelSpec, ...]:
    return get_main_models(cfg)


def _make_unpenalized_logistic(random_state: int) -> LogisticRegression:
    try:
        return LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000, random_state=random_state)
    except Exception:
        return LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000, random_state=random_state)


def load_exposure_frames(cfg: WorkflowConfig) -> dict[str, pd.DataFrame]:
    return {
        "Standard": pd.read_csv(cfg.exposures_dir / "standard_sbs_exposures.tsv", sep="	", index_col=0),
        f"Universal-BiCGR d={cfg.universal_depth}": pd.read_csv(
            cfg.exposures_dir / f"universal_bicgr_d{cfg.universal_depth}_sbs_exposures.tsv",
            sep="	",
            index_col=0,
        ),
        "BiCGR-52": pd.read_csv(cfg.exposures_dir / "bicgr52_sbs_exposures.tsv", sep="	", index_col=0),
    }


def align_exposure(cohort: pd.DataFrame, exp_df: pd.DataFrame) -> pd.DataFrame:
    patient_ids = cohort["patient_id_12"].astype(str).tolist()
    return exp_df.loc[patient_ids].copy()


def intersect_cohort_with_exposures(cohort: pd.DataFrame, exposures: dict[str, pd.DataFrame]) -> pd.DataFrame:
    eligible = set(cohort["patient_id_12"].astype(str))
    for frame in exposures.values():
        eligible &= set(frame.index.astype(str))
    out = cohort[cohort["patient_id_12"].astype(str).isin(eligible)].copy()
    out = out.drop_duplicates(subset=["patient_id_12"]).reset_index(drop=True)
    return out


def build_design_matrix(
    cohort: pd.DataFrame,
    exposures: dict[str, pd.DataFrame],
    model_spec: ModelSpec,
) -> tuple[np.ndarray, list[str]]:
    burden = np.ascontiguousarray(
        cohort["log10_burden"].to_numpy(dtype=np.float64).reshape(-1, 1)
    )
    if model_spec.representation is None:
        return burden, ["log10_burden"]

    exp_df = align_exposure(cohort, exposures[model_spec.representation])
    if model_spec.feature_mode == "sbs3_only":
        if "SBS3" not in exp_df.columns:
            raise KeyError(f"SBS3 not found in {model_spec.representation} exposures")
        exp_df = exp_df[["SBS3"]]
    X = np.ascontiguousarray(np.hstack([burden, exp_df.to_numpy(dtype=np.float64, copy=True)]))
    return X, ["log10_burden", *exp_df.columns.tolist()]


def make_study(cfg: WorkflowConfig, study_name: str, direction: str) -> optuna.Study:
    kwargs = {
        "study_name": study_name,
        "direction": direction,
        "sampler": optuna.samplers.TPESampler(seed=cfg.random_state),
        "pruner": optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    }
    if cfg.optuna_storage:
        kwargs["storage"] = cfg.optuna_storage
        kwargs["load_if_exists"] = True
    return optuna.create_study(**kwargs)


def build_regression_pipeline(cfg: WorkflowConfig, params: dict | None = None, *, baseline: bool = False) -> Pipeline:
    if baseline:
        model = LinearRegression()
    else:
        params = params or {"alpha": 0.1, "l1_ratio": 0.5}
        model = ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=5000,
            tol=1e-3,
            selection="cyclic",
            random_state=cfg.random_state,
        )
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def build_classification_pipeline(cfg: WorkflowConfig, params: dict | None = None, *, baseline: bool = False) -> Pipeline:
    if baseline:
        model = _make_unpenalized_logistic(cfg.random_state)
    else:
        params = params or {"C": 1.0, "l1_ratio": 0.5, "class_weight": None}
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=float(params["C"]),
            l1_ratio=float(params["l1_ratio"]),
            class_weight=params["class_weight"],
            max_iter=2000,
            tol=1e-3,
            random_state=cfg.random_state,
        )
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def tune_regression_model(cfg: WorkflowConfig, X_train: np.ndarray, y_train: np.ndarray, study_name: str):
    n_splits = choose_regression_n_splits(len(y_train), cfg.inner_splits)
    if n_splits < 2:
        model = build_regression_pipeline(cfg, baseline=False)
        model.fit(X_train, y_train)
        return model, None

    inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
    inner_folds = list(inner_cv.split(X_train, y_train))
    study = make_study(cfg, study_name, direction="minimize")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e1, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 1.0),
        }
        scores = []
        for fold_idx, (tr_idx, va_idx) in enumerate(inner_folds):
            model = build_regression_pipeline(cfg, params, baseline=False)
            model.fit(X_train[tr_idx], y_train[tr_idx])
            pred = model.predict(X_train[va_idx])
            score = mean_absolute_error(y_train[va_idx], pred)
            scores.append(score)
            trial.report(float(np.mean(scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study.optimize(
        objective,
        n_trials=cfg.regression_trials,
        n_jobs=cfg.optuna_trial_jobs,
        show_progress_bar=False,
    )
    model = build_regression_pipeline(cfg, study.best_params, baseline=False)
    model.fit(X_train, y_train)
    return model, study


def tune_classification_model(cfg: WorkflowConfig, X_train: np.ndarray, y_train: np.ndarray, study_name: str):
    n_splits = choose_binary_n_splits(y_train, cfg.inner_splits)
    if n_splits < 2:
        model = build_classification_pipeline(cfg, baseline=False)
        model.fit(X_train, y_train)
        return model, None

    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
    inner_folds = list(inner_cv.split(np.zeros(len(y_train)), y_train))
    study = make_study(cfg, study_name, direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
        scores = []
        for fold_idx, (tr_idx, va_idx) in enumerate(inner_folds):
            y_val = y_train[va_idx]
            model = build_classification_pipeline(cfg, params, baseline=False)
            model.fit(X_train[tr_idx], y_train[tr_idx])
            proba = model.predict_proba(X_train[va_idx])[:, 1]
            score = 0.5 if np.unique(y_val).size < 2 else roc_auc_score(y_val, proba)
            scores.append(score)
            trial.report(float(np.mean(scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study.optimize(
        objective,
        n_trials=cfg.classification_trials,
        n_jobs=cfg.optuna_trial_jobs,
        show_progress_bar=False,
    )
    model = build_classification_pipeline(cfg, study.best_params, baseline=False)
    model.fit(X_train, y_train)
    return model, study


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rho, _ = spearmanr(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = np.mean(np.abs(y_true - y_pred))
    return {"spearman": float(rho), "pearson": float(r), "r2": float(r2), "mae": float(mae)}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if np.unique(y_true).size < 2:
        return {"auroc": np.nan, "auprc": np.nan, "balanced_acc": np.nan, "brier": np.nan}
    try:
        auroc = roc_auc_score(y_true, y_pred)
    except Exception:
        auroc = np.nan
    try:
        auprc = average_precision_score(y_true, y_pred)
    except Exception:
        auprc = np.nan
    try:
        balanced_acc = balanced_accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    except Exception:
        balanced_acc = np.nan
    try:
        brier = brier_score_loss(y_true, y_pred)
    except Exception:
        brier = np.nan
    return {
        "auroc": float(auroc) if pd.notna(auroc) else np.nan,
        "auprc": float(auprc) if pd.notna(auprc) else np.nan,
        "balanced_acc": float(balanced_acc) if pd.notna(balanced_acc) else np.nan,
        "brier": float(brier) if pd.notna(brier) else np.nan,
    }


def record_study_summary(
    rows: list[dict],
    *,
    suite: str,
    subset: str,
    endpoint: str,
    task: str,
    model_name: str,
    fold: int,
    scope: str,
    study,
) -> None:
    if study is None:
        return
    rows.append(
        {
            "suite": suite,
            "subset": subset,
            "endpoint": endpoint,
            "task": task,
            "model": model_name,
            "fold": int(fold),
            "scope": scope,
            "study_name": study.study_name,
            "best_value": float(study.best_value),
            "best_params_json": json.dumps(study.best_params, sort_keys=True),
            "n_trials": int(len(study.trials)),
        }
    )


def _fit_and_predict_outer_fold(
    cfg: WorkflowConfig,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    task: str,
    model_spec: ModelSpec,
    study_name: str,
):
    if task == "regression":
        if model_spec.tune:
            model, study = tune_regression_model(cfg, X[train_idx], y[train_idx], study_name)
        else:
            model = build_regression_pipeline(cfg, baseline=True)
            model.fit(X[train_idx], y[train_idx])
            study = None
        pred = model.predict(X[test_idx])
    else:
        if model_spec.tune:
            model, study = tune_classification_model(cfg, X[train_idx], y[train_idx], study_name)
        else:
            model = build_classification_pipeline(cfg, baseline=True)
            model.fit(X[train_idx], y[train_idx])
            study = None
        pred = model.predict_proba(X[test_idx])[:, 1]
    return model, study, pred


def _binary_class_labels(endpoint: str) -> tuple[str, str]:
    if endpoint == "parpi7_binary":
        return "PARPi-high", "PARPi-low"
    return "HRD-high", "HRD-low"


def run_nested_cv_suite(
    cfg: WorkflowConfig,
    cohort: pd.DataFrame,
    exposures: dict[str, pd.DataFrame],
    *,
    suite: str,
    subset: str,
    endpoint: str,
    task: str,
    model_specs: tuple[ModelSpec, ...],
    full_refit: bool = True,
) -> dict[str, pd.DataFrame]:
    data = cohort.copy()
    if task == "classification":
        pos, neg = _binary_class_labels(endpoint)
        data = data[data[endpoint].isin([pos, neg])].copy()
        if data.empty:
            return {key: pd.DataFrame() for key in ["metrics", "fold_metrics", "predictions", "coefficients", "studies"]}
        y = (data[endpoint] == pos).astype(int).to_numpy(dtype=np.int64)
        outer_splits = choose_binary_n_splits(y, cfg.outer_splits)
        if outer_splits < 2:
            return {key: pd.DataFrame() for key in ["metrics", "fold_metrics", "predictions", "coefficients", "studies"]}
        splitter = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=cfg.random_state)
        folds = list(splitter.split(np.zeros(len(y)), y))
    else:
        data = data.dropna(subset=[endpoint]).copy()
        if data.empty:
            return {key: pd.DataFrame() for key in ["metrics", "fold_metrics", "predictions", "coefficients", "studies"]}
        y = data[endpoint].to_numpy(dtype=np.float64)
        outer_splits = choose_regression_n_splits(len(y), cfg.outer_splits)
        if outer_splits < 2:
            return {key: pd.DataFrame() for key in ["metrics", "fold_metrics", "predictions", "coefficients", "studies"]}
        splitter = KFold(n_splits=outer_splits, shuffle=True, random_state=cfg.random_state)
        folds = list(splitter.split(y))

    metric_rows: list[dict] = []
    fold_rows: list[dict] = []
    pred_rows: list[dict] = []
    coef_rows: list[dict] = []
    study_rows: list[dict] = []

    for model_spec in model_specs:
        try:
            X, feature_names = build_design_matrix(data, exposures, model_spec)
        except KeyError:
            continue
        preds = np.full(len(y), np.nan, dtype=np.float64)

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            study_name = slugify(f"{suite}__{subset}__{endpoint}__{model_spec.name}__fold_{fold_idx}")
            model, study, fold_pred = _fit_and_predict_outer_fold(
                cfg,
                X,
                y,
                train_idx,
                test_idx,
                task=task,
                model_spec=model_spec,
                study_name=study_name,
            )
            preds[test_idx] = fold_pred
            record_study_summary(
                study_rows,
                suite=suite,
                subset=subset,
                endpoint=endpoint,
                task=task,
                model_name=model_spec.name,
                fold=fold_idx,
                scope="outer",
                study=study,
            )

            fold_metric = regression_metrics(y[test_idx], fold_pred) if task == "regression" else classification_metrics(y[test_idx], fold_pred)
            fold_metric.update(
                {
                    "suite": suite,
                    "subset": subset,
                    "endpoint": endpoint,
                    "task": task,
                    "model": model_spec.name,
                    "fold": int(fold_idx),
                }
            )
            fold_rows.append(fold_metric)

            for local_idx, sample_idx in enumerate(test_idx):
                pred_rows.append(
                    {
                        "suite": suite,
                        "subset": subset,
                        "endpoint": endpoint,
                        "task": task,
                        "model": model_spec.name,
                        "patient_id": data.iloc[sample_idx]["patient_id_12"],
                        "fold": int(fold_idx),
                        "true_value": float(y[sample_idx]),
                        "pred_value": float(fold_pred[local_idx]),
                    }
                )
            gc.collect()

        overall = regression_metrics(y, preds) if task == "regression" else classification_metrics(y, preds)
        overall.update(
            {
                "suite": suite,
                "subset": subset,
                "endpoint": endpoint,
                "task": task,
                "model": model_spec.name,
                "n": int(len(y)),
            }
        )
        metric_rows.append(overall)

        if full_refit:
            study_name = slugify(f"{suite}__{subset}__{endpoint}__{model_spec.name}__full")
            if task == "regression":
                if model_spec.tune:
                    full_model, full_study = tune_regression_model(cfg, X, y, study_name)
                else:
                    full_model = build_regression_pipeline(cfg, baseline=True)
                    full_model.fit(X, y)
                    full_study = None
            else:
                if model_spec.tune:
                    full_model, full_study = tune_classification_model(cfg, X, y, study_name)
                else:
                    full_model = build_classification_pipeline(cfg, baseline=True)
                    full_model.fit(X, y)
                    full_study = None
            record_study_summary(
                study_rows,
                suite=suite,
                subset=subset,
                endpoint=endpoint,
                task=task,
                model_name=model_spec.name,
                fold=-1,
                scope="full",
                study=full_study,
            )
            model_obj = full_model.named_steps["model"]
            if hasattr(model_obj, "coef_"):
                coef_values = np.ravel(model_obj.coef_)
                for feature, coef in zip(feature_names, coef_values):
                    coef_rows.append(
                        {
                            "suite": suite,
                            "subset": subset,
                            "endpoint": endpoint,
                            "task": task,
                            "model": model_spec.name,
                            "feature": feature,
                            "coefficient": float(coef),
                            "abs_coef": float(abs(coef)),
                        }
                    )
            gc.collect()

    return {
        "metrics": pd.DataFrame(metric_rows),
        "fold_metrics": pd.DataFrame(fold_rows),
        "predictions": pd.DataFrame(pred_rows),
        "coefficients": pd.DataFrame(coef_rows),
        "studies": pd.DataFrame(study_rows),
    }
