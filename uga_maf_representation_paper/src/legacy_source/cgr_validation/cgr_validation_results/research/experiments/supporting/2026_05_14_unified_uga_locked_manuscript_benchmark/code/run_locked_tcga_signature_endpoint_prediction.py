#!/usr/bin/env python3
"""TCGA endpoint prediction from standard and locked UGA reference-signature exposures."""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, balanced_accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier, XGBRegressor

from utils.checkpointing import atomic_write_csv, atomic_write_json, merge_checkpoint_rows, read_completed_keys

from locked_signature_exposure_utils import (
    LOCKED_ID_MODEL,
    LOCKED_SBSDBS_MODEL,
    RANDOM_SEED,
    bh_q_values,
    extract_signature_exposures,
    strip_feature_prefix,
    write_html_table,
)


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent

def find_experiments_root(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if candidate.name == "experiments":
            return candidate
    raise RuntimeError(f"Could not locate experiments root from {path}")


def find_project_root(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if (candidate / "uga_atlas" / "models.py").is_file():
            return candidate
    raise RuntimeError(f"Could not locate cgr_validation project root from {path}")


EXPERIMENTS_ROOT = find_experiments_root(EXPERIMENT_ROOT)
DATA_DIR = EXPERIMENT_ROOT / "data" / "tcga_signature_endpoints"
TABLE_DIR = EXPERIMENT_ROOT / "tables" / "tcga_signature_endpoints"

SOURCE_MC3 = EXPERIMENT_ROOT / "data" / "mc3_source"
FEATURE_DIR = SOURCE_MC3 / "features"
HRD_COHORT_PATH = (
    EXPERIMENTS_ROOT.parent
    / "assets"
    / "EXP023_tcga_brca_hrd"
    / "TCGA-BRCA"
    / "cohort"
    / "final_analysis_cohort.tsv"
)

REPEATS = 5
OUTER_FOLDS = 5
N_ESTIMATORS = 250
TREE_METHOD = "gpu_hist"
BOOTSTRAP = 1000

HRD_REGRESSION_ENDPOINTS = ["HRD_Score", "eCARD", "HRD_TAI", "HRD_LST", "HRD_LOH", "PARPi7"]
HRD_CLASSIFICATION_ENDPOINTS = ["hrd_binary_24", "hrd_binary_33", "hrd_binary_42", "parpi7_binary"]
MC3_ENDPOINTS = {
    "cancer_type_top10": "multiclass",
    "os_event": "binary",
    "high_stage": "binary",
    "smoking_ever": "binary",
    "high_purity": "binary",
    "brca_gene_mutated": "binary",
    "mmr_gene_mutated": "binary",
    "pole_pold1_mutated": "binary",
}

FEATURE_LABELS = {
    "burden_only": "Burden only",
    "standard_sbs_exposure": "Standard SBS exposures",
    "locked_uga_sbs_exposure": "Locked UGA SBS exposures",
    "standard_sbs_id_exposure": "Standard SBS+ID exposures",
    "locked_uga_sbs_id_exposure": "Locked UGA SBS+ID exposures",
}
FEATURE_SETS = list(FEATURE_LABELS)
PRIMARY_STANDARD = "standard_sbs_id_exposure"
PRIMARY_UGA = "locked_uga_sbs_id_exposure"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_standard_feature_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    standard_sbs = pd.read_csv(FEATURE_DIR / "features_standard_sbs96.csv.gz", index_col=0).fillna(0.0)
    standard_id = pd.read_csv(FEATURE_DIR / "features_standard_id83.csv.gz", index_col=0).fillna(0.0)
    burden = pd.read_csv(FEATURE_DIR / "features_burden_only.csv", index_col=0).fillna(0.0)
    standard_sbs.index = standard_sbs.index.astype(str)
    standard_id.index = standard_id.index.astype(str)
    burden.index = burden.index.astype(str)
    return standard_sbs, standard_id, burden


def prefixed(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [f"{prefix}{col}" for col in out.columns]
    return out


def build_exposure_features() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    standard_sbs, standard_id, burden = load_standard_feature_inputs()
    sbs_counts = strip_feature_prefix(standard_sbs, "SBS96__")
    id_counts = strip_feature_prefix(standard_id, "ID83__")

    feature_sets: dict[str, pd.DataFrame] = {"burden_only": burden}
    metadata_rows = []
    extraction_jobs = [
        ("SBS", "standard", sbs_counts, "standard_sbs_exposure", "std_sbs_sig__"),
        ("SBS", "locked_uga", sbs_counts, "locked_uga_sbs_exposure", "uga_sbs_sig__"),
        ("ID", "standard", id_counts, "standard_id_exposure", "std_id_sig__"),
        ("ID", "locked_uga", id_counts, "locked_uga_id_exposure", "uga_id_sig__"),
    ]
    exposures: dict[str, pd.DataFrame] = {}
    for modality, representation, counts, key, prefix in extraction_jobs:
        print(f"Extracting {representation} {modality} COSMIC exposures", flush=True)
        exposure, _, metadata = extract_signature_exposures(counts, modality, representation)
        exposure = prefixed(exposure, prefix)
        exposures[key] = exposure
        metadata["feature_set_key"] = key
        metadata_rows.append(metadata)
        exposure.to_csv(DATA_DIR / f"{key}.csv.gz", compression="gzip")

    feature_sets["standard_sbs_exposure"] = pd.concat([burden, exposures["standard_sbs_exposure"]], axis=1).fillna(0.0)
    feature_sets["locked_uga_sbs_exposure"] = pd.concat([burden, exposures["locked_uga_sbs_exposure"]], axis=1).fillna(0.0)
    feature_sets["standard_sbs_id_exposure"] = pd.concat(
        [burden, exposures["standard_sbs_exposure"], exposures["standard_id_exposure"]],
        axis=1,
    ).fillna(0.0)
    feature_sets["locked_uga_sbs_id_exposure"] = pd.concat(
        [burden, exposures["locked_uga_sbs_exposure"], exposures["locked_uga_id_exposure"]],
        axis=1,
    ).fillna(0.0)

    feature_manifest = []
    for name, frame in feature_sets.items():
        feature_manifest.append(
            {
                "feature_set": name,
                "feature_label": FEATURE_LABELS[name],
                "n_patients": int(frame.shape[0]),
                "n_features": int(frame.shape[1]),
            }
        )
    feature_manifest_df = pd.DataFrame(feature_manifest)
    feature_manifest_df.to_csv(DATA_DIR / "signature_exposure_feature_manifest.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(DATA_DIR / "signature_exposure_extraction_metadata.csv", index=False)
    return feature_sets, feature_manifest_df


def load_endpoint_series() -> list[dict[str, object]]:
    endpoints: list[dict[str, object]] = []
    hrd = pd.read_csv(HRD_COHORT_PATH, sep="\t")
    hrd["patient_id_12"] = hrd["patient_id_12"].astype(str)
    for endpoint in HRD_REGRESSION_ENDPOINTS:
        y = hrd.dropna(subset=[endpoint]).set_index("patient_id_12")[endpoint].astype(float)
        endpoints.append({"suite": "tcga_brca_hrd", "endpoint": endpoint, "task": "regression", "y": y})
    for endpoint in HRD_CLASSIFICATION_ENDPOINTS:
        positive = "PARPi-high" if endpoint == "parpi7_binary" else "HRD-high"
        allowed = ["PARPi-high", "PARPi-low"] if endpoint == "parpi7_binary" else ["HRD-high", "HRD-low"]
        data = hrd[hrd[endpoint].isin(allowed)].copy()
        y = pd.Series((data[endpoint] == positive).astype(int).to_numpy(), index=data["patient_id_12"].astype(str), name=endpoint)
        endpoints.append({"suite": "tcga_brca_hrd", "endpoint": endpoint, "task": "binary", "y": y})

    labels = pd.read_csv(SOURCE_MC3 / "biology_labels.csv", index_col=0)
    labels.index = labels.index.astype(str)
    for endpoint, task in MC3_ENDPOINTS.items():
        y = labels[endpoint].dropna()
        if task == "binary":
            y = y.astype(int)
            counts = y.value_counts()
            if len(counts) != 2 or int(counts.min()) < 25:
                continue
        else:
            counts = y.value_counts()
            y = y[y.isin(counts[counts >= 50].index)].astype(str)
            if y.nunique() < 3:
                continue
        endpoints.append({"suite": "tcga_mc3", "endpoint": endpoint, "task": task, "y": y})
    return endpoints


def class_values(y: pd.Series, task: str) -> np.ndarray:
    if task == "binary":
        return np.array([0, 1], dtype=int)
    return np.array(sorted(y.astype(str).unique()), dtype=object)


def encode_classes(y: pd.Series, classes: np.ndarray, task: str) -> np.ndarray:
    if task == "binary":
        return y.astype(int).to_numpy()
    mapping = {cls: i for i, cls in enumerate(classes)}
    return y.astype(str).map(mapping).astype(int).to_numpy()


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    counts = pd.Series(y).value_counts().to_dict()
    n = float(len(y))
    k = float(len(counts))
    return np.array([n / (k * counts[value]) for value in y], dtype=np.float32)


def classifier_params(y_train: np.ndarray, classes: np.ndarray, task: str, seed: int, tree_method: str) -> dict[str, object]:
    params: dict[str, object] = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "reg_lambda": 2.0,
        "reg_alpha": 0.05,
        "random_state": int(seed),
        "n_jobs": 1,
        "tree_method": tree_method,
        "verbosity": 0,
    }
    if task == "binary":
        positives = float(np.sum(y_train == 1))
        negatives = float(np.sum(y_train == 0))
        params.update(
            {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "scale_pos_weight": negatives / max(positives, 1.0),
            }
        )
    else:
        params.update(
            {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": int(len(classes)),
            }
        )
    if tree_method == "gpu_hist":
        params["predictor"] = "gpu_predictor"
    return params


def regressor_params(seed: int, tree_method: str) -> dict[str, object]:
    params: dict[str, object] = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "reg_lambda": 2.0,
        "reg_alpha": 0.05,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": int(seed),
        "n_jobs": 1,
        "tree_method": tree_method,
        "verbosity": 0,
    }
    if tree_method == "gpu_hist":
        params["predictor"] = "gpu_predictor"
    return params


def fit_predict_classification(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    classes: np.ndarray,
    task: str,
    seed: int,
    tree_method: str,
) -> np.ndarray:
    model = XGBClassifier(**classifier_params(y_train, classes, task, seed, tree_method))
    sample_weight = balanced_sample_weight(y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(x_train, y_train, sample_weight=sample_weight)
        except Exception:
            if tree_method == "gpu_hist":
                model = XGBClassifier(**classifier_params(y_train, classes, task, seed, "hist"))
                model.fit(x_train, y_train, sample_weight=sample_weight)
            else:
                raise
    proba = model.predict_proba(x_test)
    if task == "binary" and proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    return proba.astype(np.float64)


def fit_predict_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int, tree_method: str) -> np.ndarray:
    model = XGBRegressor(**regressor_params(seed, tree_method))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(x_train, y_train)
        except Exception:
            if tree_method == "gpu_hist":
                model = XGBRegressor(**regressor_params(seed, "hist"))
                model.fit(x_train, y_train)
            else:
                raise
    return model.predict(x_test).astype(np.float64)


def classification_score(y: np.ndarray, proba: np.ndarray, classes: np.ndarray, task: str) -> float:
    if task == "binary":
        return float(roc_auc_score(y, proba[:, 1]))
    y_bin = label_binarize(y, classes=np.arange(len(classes)))
    return float(roc_auc_score(y_bin, proba, average="macro"))


def run_oof_prediction(
    y_series: pd.Series,
    features: pd.DataFrame,
    task: str,
    endpoint_idx: int,
    feature_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[dict[str, object]], list[str]]:
    common = y_series.index.astype(str).intersection(features.index.astype(str))
    y_series = y_series.loc[common]
    x_frame = features.loc[common].fillna(0.0)
    x = x_frame.to_numpy(dtype=np.float32)
    fold_rows: list[dict[str, object]] = []
    if task == "regression":
        y = y_series.astype(float).to_numpy(dtype=np.float64)
        pred_sum = np.zeros(len(y), dtype=np.float64)
        for repeat in range(REPEATS):
            seed = RANDOM_SEED + endpoint_idx * 1000 + feature_idx * 100 + repeat
            splitter = KFold(n_splits=min(OUTER_FOLDS, len(y)), shuffle=True, random_state=seed)
            for fold, (train_idx, test_idx) in enumerate(splitter.split(x), start=1):
                pred = fit_predict_regression(x[train_idx], y[train_idx], x[test_idx], seed + fold * 101, TREE_METHOD)
                pred_sum[test_idx] += pred
                fold_rows.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "fold_spearman": float(spearmanr(y[test_idx], pred)[0]),
                        "fold_mae": float(mean_absolute_error(y[test_idx], pred)),
                    }
                )
        return y, pred_sum / float(REPEATS), None, fold_rows, common.tolist()

    classes = class_values(y_series, task)
    y = encode_classes(y_series, classes, task)
    proba_sum = np.zeros((len(y), 2 if task == "binary" else len(classes)), dtype=np.float64)
    n_splits = max(2, min(OUTER_FOLDS, int(pd.Series(y).value_counts().min())))
    for repeat in range(REPEATS):
        seed = RANDOM_SEED + endpoint_idx * 1000 + feature_idx * 100 + repeat
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
            proba = fit_predict_classification(
                x[train_idx],
                y[train_idx],
                x[test_idx],
                classes,
                task,
                seed + fold * 101,
                TREE_METHOD,
            )
            proba_sum[test_idx] += proba
            pred = np.argmax(proba, axis=1)
            fold_rows.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "fold_auroc": classification_score(y[test_idx], proba, classes, task),
                    "fold_balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
                }
            )
    return y, proba_sum / float(REPEATS), classes, fold_rows, common.tolist()


def metric_summary(task: str, y: np.ndarray, pred: np.ndarray, classes: np.ndarray | None) -> dict[str, float]:
    if task == "regression":
        rho = spearmanr(y, pred)[0]
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return {
            "primary_metric": float(rho),
            "primary_metric_name": "spearman",
            "spearman": float(rho),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            "mae": float(mean_absolute_error(y, pred)),
            "auroc": np.nan,
            "auprc": np.nan,
            "balanced_accuracy": np.nan,
        }
    assert classes is not None
    pred_class = np.argmax(pred, axis=1)
    out = {
        "primary_metric": classification_score(y, pred, classes, task),
        "primary_metric_name": "auroc",
        "spearman": np.nan,
        "r2": np.nan,
        "mae": np.nan,
        "auroc": classification_score(y, pred, classes, task),
        "auprc": np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred_class)),
    }
    if task == "binary":
        out["auprc"] = float(average_precision_score(y, pred[:, 1]))
    return out


def bootstrap_regression_delta(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    deltas = np.zeros(BOOTSTRAP, dtype=np.float64)
    n = len(y)
    for i in range(BOOTSTRAP):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        deltas[i] = float(spearmanr(y[idx], pred_a[idx])[0] - spearmanr(y[idx], pred_b[idx])[0])
    p_lower = (np.sum(deltas <= 0.0) + 1.0) / (BOOTSTRAP + 1.0)
    p_upper = (np.sum(deltas >= 0.0) + 1.0) / (BOOTSTRAP + 1.0)
    p_value = float(min(1.0, 2.0 * min(p_lower, p_upper)))
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return p_value, float(ci_low), float(ci_high)


def bootstrap_classification_delta(
    y: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    classes: np.ndarray,
    task: str,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y == cls) for cls in np.unique(y)]
    deltas = np.zeros(BOOTSTRAP, dtype=np.float64)
    for i in range(BOOTSTRAP):
        idx = np.concatenate([rng.choice(stratum, size=len(stratum), replace=True) for stratum in strata])
        deltas[i] = classification_score(y[idx], proba_a[idx], classes, task) - classification_score(
            y[idx],
            proba_b[idx],
            classes,
            task,
        )
    p_lower = (np.sum(deltas <= 0.0) + 1.0) / (BOOTSTRAP + 1.0)
    p_upper = (np.sum(deltas >= 0.0) + 1.0) / (BOOTSTRAP + 1.0)
    p_value = float(min(1.0, 2.0 * min(p_lower, p_upper)))
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return p_value, float(ci_low), float(ci_high)


def pairwise_tests(summary: pd.DataFrame, predictions: dict[str, dict[str, object]]) -> pd.DataFrame:
    summary_index = summary.set_index(["suite", "endpoint", "feature_set"])
    rows = []
    contrasts = [
        ("locked_uga_sbs_exposure", "standard_sbs_exposure", "Locked UGA - Standard (SBS exposures)"),
        (PRIMARY_UGA, PRIMARY_STANDARD, "Locked UGA - Standard (SBS+ID exposures)"),
        (PRIMARY_STANDARD, "standard_sbs_exposure", "Standard SBS+ID - Standard SBS exposures"),
        (PRIMARY_UGA, "locked_uga_sbs_exposure", "Locked UGA SBS+ID - Locked UGA SBS exposures"),
        (PRIMARY_UGA, "burden_only", "Locked UGA SBS+ID exposures - burden only"),
    ]
    for key_a, pred_a in predictions.items():
        suite = str(pred_a["suite"])
        endpoint = str(pred_a["endpoint"])
        task = str(pred_a["task"])
        if not key_a.endswith(f"||{PRIMARY_UGA}"):
            continue
        for contrast_idx, (feature_a, feature_b, label) in enumerate(contrasts, start=1):
            a_key = f"{suite}||{endpoint}||{feature_a}"
            b_key = f"{suite}||{endpoint}||{feature_b}"
            if a_key not in predictions or b_key not in predictions:
                continue
            a = predictions[a_key]
            b = predictions[b_key]
            if task == "regression":
                p_value, ci_low, ci_high = bootstrap_regression_delta(
                    a["y"],
                    a["pred"],
                    b["pred"],
                    RANDOM_SEED + len(rows) * 17 + contrast_idx,
                )
            else:
                p_value, ci_low, ci_high = bootstrap_classification_delta(
                    a["y"],
                    a["pred"],
                    b["pred"],
                    a["classes"],
                    task,
                    RANDOM_SEED + len(rows) * 17 + contrast_idx,
                )
            metric_a = float(summary_index.loc[(suite, endpoint, feature_a), "primary_metric"])
            metric_b = float(summary_index.loc[(suite, endpoint, feature_b), "primary_metric"])
            rows.append(
                {
                    "suite": suite,
                    "endpoint": endpoint,
                    "task": task,
                    "comparison": label,
                    "feature_set_a": feature_a,
                    "feature_set_b": feature_b,
                    "metric_name": "spearman" if task == "regression" else "auroc",
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "delta_metric": metric_a - metric_b,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "p_value": p_value,
                    "n_bootstrap": BOOTSTRAP,
                }
            )
    out = pd.DataFrame(rows)
    out["q_value"] = bh_q_values(out["p_value"].to_numpy())
    return out


def endpoint_comparison(summary: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    summary_index = summary.set_index(["suite", "endpoint", "feature_set"])
    primary = pairwise[pairwise["comparison"] == "Locked UGA - Standard (SBS+ID exposures)"].copy()
    for _, test in primary.iterrows():
        suite = str(test["suite"])
        endpoint = str(test["endpoint"])
        rows.append(
            {
                "suite": suite,
                "endpoint": endpoint,
                "task": test["task"],
                "n": int(summary_index.loc[(suite, endpoint, PRIMARY_STANDARD), "n"]),
                "metric_name": test["metric_name"],
                "standard_sbs_exposure": float(summary_index.loc[(suite, endpoint, "standard_sbs_exposure"), "primary_metric"]),
                "locked_uga_sbs_exposure": float(summary_index.loc[(suite, endpoint, "locked_uga_sbs_exposure"), "primary_metric"]),
                "standard_sbs_id_exposure": float(summary_index.loc[(suite, endpoint, PRIMARY_STANDARD), "primary_metric"]),
                "locked_uga_sbs_id_exposure": float(summary_index.loc[(suite, endpoint, PRIMARY_UGA), "primary_metric"]),
                "delta_locked_uga_minus_standard_sbs_id": float(test["delta_metric"]),
                "bootstrap_ci_low": float(test["bootstrap_ci_low"]),
                "bootstrap_ci_high": float(test["bootstrap_ci_high"]),
                "p_value": float(test["p_value"]),
                "q_value": float(test["q_value"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["suite", "delta_locked_uga_minus_standard_sbs_id"], ascending=[True, False])


def prediction_checkpoint_rows(
    suite: str,
    endpoint: str,
    task: str,
    feature_set: str,
    patients: np.ndarray,
    y_arr: np.ndarray,
    pred: np.ndarray,
    classes: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if task == "regression":
        for patient, y_value, pred_value in zip(patients, y_arr, pred, strict=False):
            rows.append(
                {
                    "suite": suite,
                    "endpoint": endpoint,
                    "task": task,
                    "feature_set": feature_set,
                    "patient": str(patient),
                    "true_value": float(y_value),
                    "class_id": -1,
                    "class_label": "",
                    "pred_value": float(pred_value),
                }
            )
        return rows
    for i, patient in enumerate(patients):
        for class_id in range(pred.shape[1]):
            class_label = str(classes[class_id]) if task == "multiclass" else str(class_id)
            rows.append(
                {
                    "suite": suite,
                    "endpoint": endpoint,
                    "task": task,
                    "feature_set": feature_set,
                    "patient": str(patient),
                    "true_value": int(y_arr[i]),
                    "class_id": int(class_id),
                    "class_label": class_label,
                    "pred_value": float(pred[i, class_id]),
                }
            )
    return rows


def load_prediction_checkpoint(path: Path) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return {}, pd.DataFrame()
    frame = pd.read_csv(path)
    cache: dict[str, dict[str, object]] = {}
    output_frames: list[pd.DataFrame] = []
    for (suite, endpoint, feature_set), group in frame.groupby(["suite", "endpoint", "feature_set"], dropna=False):
        task = str(group["task"].iloc[0])
        patient_order = group[["patient"]].drop_duplicates()["patient"].astype(str).tolist()
        key = f"{suite}||{endpoint}||{feature_set}"
        if task == "regression":
            one = group.drop_duplicates("patient").set_index("patient").loc[patient_order]
            y = one["true_value"].to_numpy(dtype=float)
            pred = one["pred_value"].to_numpy(dtype=float)
            classes = np.array([], dtype=object)
            output_frames.append(
                pd.DataFrame(
                    {
                        "suite": suite,
                        "endpoint": endpoint,
                        "task": task,
                        "feature_set": feature_set,
                        "patient": patient_order,
                        "true_value": y,
                        "pred_value": pred,
                    }
                )
            )
        else:
            wide = group.pivot_table(index="patient", columns="class_id", values="pred_value", aggfunc="first").loc[patient_order].sort_index(axis=1)
            one = group.drop_duplicates("patient").set_index("patient").loc[patient_order]
            y = one["true_value"].to_numpy(dtype=int)
            pred = wide.to_numpy(dtype=float)
            class_table = group[["class_id", "class_label"]].drop_duplicates().sort_values("class_id")
            classes = class_table["class_label"].astype(str).to_numpy(dtype=object)
            out = pd.DataFrame(pred, columns=[f"pred_proba_{i}" for i in range(pred.shape[1])])
            out.insert(0, "true_value", y)
            out.insert(0, "patient", patient_order)
            out.insert(0, "feature_set", feature_set)
            out.insert(0, "task", task)
            out.insert(0, "endpoint", endpoint)
            out.insert(0, "suite", suite)
            output_frames.append(out)
        cache[key] = {
            "suite": suite,
            "endpoint": endpoint,
            "task": task,
            "feature_set": feature_set,
            "patients": np.array(patient_order, dtype=object),
            "y": y,
            "pred": pred,
            "classes": classes,
        }
    output = pd.concat(output_frames, ignore_index=True, sort=False) if output_frames else pd.DataFrame()
    return cache, output


def write_readme(endpoint_summary: pd.DataFrame, metadata: dict[str, object]) -> None:
    hrd = endpoint_summary[endpoint_summary["suite"] == "tcga_brca_hrd"]
    mc3 = endpoint_summary[endpoint_summary["suite"] == "tcga_mc3"]
    lines = [
        "# TCGA Signature-Exposure Endpoint Prediction Benchmark",
        "",
        "## Research Question",
        "Do reference-signature exposures extracted in the locked UGA space improve prediction of TCGA biological and clinical endpoints relative to exposures extracted in the standard COSMIC channel space?",
        "",
        "## Methods",
        f"MC3-derived SBS96 and ID83 mutation-channel profiles were fit to COSMIC v3.5 GRCh37 signatures with NNLS. Standard exposure features were fit in the native SBS96 and ID83 channel bases. Locked UGA exposure features were fit after projecting the same channel profiles and COSMIC signatures into `{LOCKED_SBSDBS_MODEL}` for SBS and `{LOCKED_ID_MODEL}` for ID. DBS was not included because the retained TCGA inputs contain SBS96 and ID83 profiles without explicit DBS/DNP calls. Endpoint prediction used matched {REPEATS} repeats of {OUTER_FOLDS}-fold out-of-fold XGBoost with the same burden covariates in every exposure feature set. Statistical comparisons used paired bootstrap resampling of out-of-fold predictions with Benjamini-Hochberg q values across displayed contrasts.",
        "",
        "## Key Numerical Findings",
    ]
    if not hrd.empty:
        wins = int((hrd["delta_locked_uga_minus_standard_sbs_id"] > 0).sum())
        lines.append(
            f"- TCGA-BRCA HRD endpoints: locked UGA SBS+ID exposures exceeded Standard SBS+ID exposures for {wins} of {len(hrd)} endpoints. Mean primary metric was {hrd['locked_uga_sbs_id_exposure'].mean():.4f} for locked UGA and {hrd['standard_sbs_id_exposure'].mean():.4f} for Standard."
        )
    if not mc3.empty:
        wins = int((mc3["delta_locked_uga_minus_standard_sbs_id"] > 0).sum())
        lines.append(
            f"- MC3 clinical and biology endpoints: locked UGA SBS+ID exposures exceeded Standard SBS+ID exposures for {wins} of {len(mc3)} endpoints. Mean primary metric was {mc3['locked_uga_sbs_id_exposure'].mean():.4f} for locked UGA and {mc3['standard_sbs_id_exposure'].mean():.4f} for Standard."
        )
    lines.extend(
        [
            "",
            "## File Inventory",
            "- `standard_sbs_exposure.csv.gz`, `locked_uga_sbs_exposure.csv.gz`, `standard_id_exposure.csv.gz`, and `locked_uga_id_exposure.csv.gz`: extracted COSMIC exposure features.",
            "- `signature_exposure_feature_manifest.csv`: feature-set dimensions and patient counts.",
            "- `signature_exposure_endpoint_metrics.csv`: endpoint-level out-of-fold metrics for every feature set.",
            "- `signature_exposure_fold_metrics.csv`: repeat and fold metrics.",
            "- `signature_exposure_predictions.csv.gz`: out-of-fold predictions used for paired testing.",
            "- `signature_exposure_pairwise_tests.csv`: paired bootstrap contrasts.",
            "- `signature_exposure_endpoint_summary.csv`: primary locked UGA versus Standard SBS+ID exposure summary.",
            "- `table1_signature_exposure_endpoint_summary.html`: manuscript-ready endpoint summary table.",
            "- `table2_signature_exposure_pairwise_tests.html`: manuscript-ready paired-test table.",
            "- `code/run_locked_tcga_signature_endpoint_prediction.py`: reproducible benchmark script.",
            "",
            "## Reproducibility",
            f"Executed at {metadata['executed_at_utc']} with random seed {RANDOM_SEED}, repeats={REPEATS}, outer_folds={OUTER_FOLDS}, n_estimators={N_ESTIMATORS}, tree_method={TREE_METHOD}, and bootstrap_iterations={BOOTSTRAP}. Runtime was {metadata['elapsed_seconds'] / 60.0:.1f} minutes.",
            "",
        ]
    )
    (DATA_DIR / "README_tcga_signature_endpoints.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    start = time.perf_counter()
    features, feature_manifest = build_exposure_features()
    endpoints = load_endpoint_series()
    metrics_checkpoint = DATA_DIR / "signature_exposure_endpoint_metrics_checkpoint.csv"
    folds_checkpoint = DATA_DIR / "signature_exposure_fold_metrics_checkpoint.csv"
    prediction_checkpoint = DATA_DIR / "signature_exposure_predictions_checkpoint.csv"
    completed = read_completed_keys(metrics_checkpoint, ["suite", "endpoint", "feature_set"])
    total_jobs = len(endpoints) * len(FEATURE_SETS)
    job = 0
    for endpoint_idx, endpoint_info in enumerate(endpoints):
        suite = str(endpoint_info["suite"])
        endpoint = str(endpoint_info["endpoint"])
        task = str(endpoint_info["task"])
        y = endpoint_info["y"]
        print(f"Endpoint {suite}/{endpoint}: n={len(y):,}, task={task}", flush=True)
        for feature_idx, feature_set in enumerate(FEATURE_SETS):
            job += 1
            key_tuple = (suite, endpoint, feature_set)
            if key_tuple in completed:
                print(f"  {job}/{total_jobs} [checkpoint] skip {feature_set}", flush=True)
                continue
            print(f"  {job}/{total_jobs} {feature_set}", flush=True)
            y_arr, pred, classes, folds, patients = run_oof_prediction(y, features[feature_set], task, endpoint_idx, feature_idx)
            metrics = metric_summary(task, y_arr, pred, classes)
            metric_row = {
                "suite": suite,
                "endpoint": endpoint,
                "task": task,
                "feature_set": feature_set,
                "feature_label": FEATURE_LABELS[feature_set],
                "n": int(len(y_arr)),
                "n_features": int(features[feature_set].shape[1]),
                "repeats": REPEATS,
                "outer_folds": OUTER_FOLDS,
                **metrics,
            }
            merge_checkpoint_rows(
                metrics_checkpoint,
                [metric_row],
                key_columns=["suite", "endpoint", "feature_set"],
                sort_columns=["suite", "endpoint", "feature_set"],
            )
            fold_rows = []
            for row in folds:
                row.update({"suite": suite, "endpoint": endpoint, "task": task, "feature_set": feature_set})
                fold_rows.append(row)
            merge_checkpoint_rows(
                folds_checkpoint,
                fold_rows,
                key_columns=["suite", "endpoint", "feature_set", "repeat", "fold"],
                sort_columns=["suite", "endpoint", "feature_set", "repeat", "fold"],
            )
            merge_checkpoint_rows(
                prediction_checkpoint,
                prediction_checkpoint_rows(suite, endpoint, task, feature_set, patients, y_arr, pred, classes),
                key_columns=["suite", "endpoint", "feature_set", "patient", "class_id"],
                sort_columns=["suite", "endpoint", "feature_set", "patient", "class_id"],
            )
            completed.add(key_tuple)
            if task == "regression":
                print(f"    Spearman={metrics['spearman']:.4f}", flush=True)
            else:
                print(f"    AUROC={metrics['auroc']:.4f}", flush=True)

    summary = pd.read_csv(metrics_checkpoint)
    folds = pd.read_csv(folds_checkpoint)
    prediction_cache, predictions = load_prediction_checkpoint(prediction_checkpoint)
    pairwise = pairwise_tests(summary, prediction_cache)
    endpoint_summary = endpoint_comparison(summary, pairwise)

    atomic_write_csv(summary, DATA_DIR / "signature_exposure_endpoint_metrics.csv", index=False)
    atomic_write_csv(folds, DATA_DIR / "signature_exposure_fold_metrics.csv", index=False)
    predictions.to_csv(DATA_DIR / "signature_exposure_predictions.csv.gz", index=False, compression="gzip")
    atomic_write_csv(pairwise, DATA_DIR / "signature_exposure_pairwise_tests.csv", index=False)
    atomic_write_csv(endpoint_summary, DATA_DIR / "signature_exposure_endpoint_summary.csv", index=False)
    atomic_write_csv(endpoint_summary, DATA_DIR / "table1_signature_exposure_endpoint_summary.csv", index=False)
    atomic_write_csv(pairwise, DATA_DIR / "table2_signature_exposure_pairwise_tests.csv", index=False)
    write_html_table(
        endpoint_summary,
        TABLE_DIR / "table1_signature_exposure_endpoint_summary.html",
        "Table 1. TCGA endpoint prediction from standard and locked UGA signature exposures",
        "Primary metric is Spearman correlation for continuous HRD endpoints and AUROC for binary or multiclass endpoints. Delta values compare locked UGA SBS+ID exposures against Standard SBS+ID exposures.",
    )
    write_html_table(
        pairwise,
        TABLE_DIR / "table2_signature_exposure_pairwise_tests.html",
        "Table 2. Paired TCGA signature-exposure endpoint prediction contrasts",
        "P values use paired bootstrap resampling of out-of-fold predictions. Q values use Benjamini-Hochberg correction across displayed contrasts.",
    )
    metadata = {
        "experiment": EXPERIMENT_ROOT.name,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "random_seed": RANDOM_SEED,
        "repeats": REPEATS,
        "outer_folds": OUTER_FOLDS,
        "n_estimators": N_ESTIMATORS,
        "tree_method": TREE_METHOD,
        "bootstrap_iterations": BOOTSTRAP,
        "sbsdbs_model": LOCKED_SBSDBS_MODEL,
        "id_model": LOCKED_ID_MODEL,
        "feature_sets": FEATURE_LABELS,
        "primary_standard": PRIMARY_STANDARD,
        "primary_uga": PRIMARY_UGA,
        "modalities": ["SBS96", "ID83"],
        "dbs_policy": "DBS omitted because retained TCGA inputs contain SBS96 and ID83 profiles without explicit DBS/DNP calls.",
        "n_endpoints": int(endpoint_summary.shape[0]),
        "n_metric_rows": int(summary.shape[0]),
        "feature_manifest_rows": int(feature_manifest.shape[0]),
    }
    atomic_write_json(DATA_DIR / "run_metadata.json", metadata)
    write_readme(endpoint_summary, metadata)
    print(json.dumps({"completed_in_seconds": round(metadata["elapsed_seconds"], 1), "endpoints": int(endpoint_summary.shape[0])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
