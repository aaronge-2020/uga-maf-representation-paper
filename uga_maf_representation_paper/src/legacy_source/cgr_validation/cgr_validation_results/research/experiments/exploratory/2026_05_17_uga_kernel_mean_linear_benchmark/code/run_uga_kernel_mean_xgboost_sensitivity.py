#!/usr/bin/env python3
"""Lightweight XGBoost sensitivity check for UGA RBF kernel-mean features."""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import label_binarize

from utils.checkpointing import atomic_write_csv, atomic_write_json, merge_checkpoint_rows, read_completed_keys

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception as exc:  # pragma: no cover - environment-specific guard
    raise SystemExit(
        "Could not import xgboost. Install the bundle environment, then rerun this script "
        "from the configured experiment workspace.\n\n"
        f"Original import error: {type(exc).__name__}: {exc}"
    )

import run_uga_kernel_mean_linear_benchmark as base


N_ESTIMATORS = 100
MAX_DEPTH = 2
LEARNING_RATE = 0.05
SUBSAMPLE = 0.85
COLSAMPLE_BYTREE = 0.85
MIN_CHILD_WEIGHT = 5
REG_LAMBDA = 2.0
REG_ALPHA = 0.05
TREE_METHOD = "hist"
XGB_N_JOBS = 4


def classifier_params(y_train: np.ndarray, task: str, n_classes: int, seed: int) -> dict[str, object]:
    params: dict[str, object] = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "min_child_weight": MIN_CHILD_WEIGHT,
        "reg_lambda": REG_LAMBDA,
        "reg_alpha": REG_ALPHA,
        "random_state": int(seed),
        "n_jobs": XGB_N_JOBS,
        "tree_method": TREE_METHOD,
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
                "num_class": int(n_classes),
            }
        )
    return params


def regressor_params(seed: int) -> dict[str, object]:
    return {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "min_child_weight": MIN_CHILD_WEIGHT,
        "reg_lambda": REG_LAMBDA,
        "reg_alpha": REG_ALPHA,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": int(seed),
        "n_jobs": XGB_N_JOBS,
        "tree_method": TREE_METHOD,
        "verbosity": 0,
    }


def fit_predict_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int) -> np.ndarray:
    model = XGBRegressor(**regressor_params(seed))
    model.fit(x_train, y_train)
    return model.predict(x_test).astype(np.float64)


def fit_predict_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    task: str,
    n_classes: int,
    seed: int,
) -> np.ndarray:
    model = XGBClassifier(**classifier_params(y_train, task, n_classes, seed))
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)
    if task == "binary" and proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    out = np.zeros((len(x_test), n_classes), dtype=np.float64)
    if proba.shape[1] == n_classes:
        out[:, : proba.shape[1]] = proba
        return out
    for local_col, class_id in enumerate(model.classes_):
        out[:, int(class_id)] = proba[:, local_col]
    return out


def evaluate_endpoint(endpoint: base.Endpoint, feature_set: base.FeatureSet) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    common = endpoint.y.index.astype(str).intersection(feature_set.frame.index.astype(str))
    y_series = endpoint.y.loc[common]
    x = feature_set.frame.loc[common].fillna(0.0).to_numpy(dtype=np.float32)
    y, classes = base.encode_target(y_series, endpoint.task)
    seed = base.stable_seed(endpoint.name)
    splits = base.make_standard_splits(y, endpoint.task, seed)
    folds = np.zeros(len(y), dtype=int)

    if endpoint.task == "regression":
        pred = np.zeros(len(y), dtype=np.float64)
        probability_rows: list[dict[str, object]] = []
        for fold, train_idx, test_idx in splits:
            pred[test_idx] = fit_predict_regression(x[train_idx], y[train_idx], x[test_idx], seed + fold)
            folds[test_idx] = fold
    else:
        n_classes = 2 if endpoint.task == "binary" else len(classes)
        pred = np.zeros((len(y), n_classes), dtype=np.float64)
        for fold, train_idx, test_idx in splits:
            pred[test_idx] = fit_predict_classifier(
                x[train_idx],
                y[train_idx],
                x[test_idx],
                task=endpoint.task,
                n_classes=n_classes,
                seed=seed + fold,
            )
            folds[test_idx] = fold
        probability_rows = probability_long_rows(
            feature_set.benchmark,
            endpoint.name,
            endpoint.family,
            endpoint.task,
            feature_set.name,
            common,
            folds,
            pred,
            classes,
        )

    metric_name, score, aux = base.score_oof(y, pred, endpoint.task, classes)
    row = {
        "benchmark": feature_set.benchmark,
        "endpoint": endpoint.name,
        "family": endpoint.family,
        "task": endpoint.task,
        "representation": feature_set.name,
        "metric": metric_name,
        "score": score,
        "n_samples": int(len(y)),
        "n_classes": int(pd.Series(y).nunique()) if endpoint.task != "regression" else math.nan,
        "n_features": int(feature_set.frame.shape[1]),
        "n_folds": int(len(splits)),
        "model": "XGBRegressor(lightweight)" if endpoint.task == "regression" else "XGBClassifier(lightweight)",
        "split_strategy": "5-fold KFold" if endpoint.task == "regression" else "5-fold StratifiedKFold",
        "tuning": "none",
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "tree_method": TREE_METHOD,
    }
    row.update(aux)
    predictions = prediction_rows(
        feature_set.benchmark,
        endpoint.name,
        endpoint.family,
        endpoint.task,
        feature_set.name,
        common,
        y_series,
        folds,
        pred,
        classes,
    )
    return row, pd.DataFrame(predictions), pd.DataFrame(probability_rows)


def evaluate_kucab(feature_set: base.FeatureSet, endpoint: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    meta = endpoint.loc[feature_set.frame.index.astype(str)]
    x = feature_set.frame.loc[meta.index].fillna(0.0).to_numpy(dtype=np.float32)
    classes = np.array(sorted(meta["damage_class"].unique()), dtype=object)
    mapping = {value: i for i, value in enumerate(classes)}
    y = meta["damage_class"].map(mapping).astype(int).to_numpy()
    groups = meta["agent_core"].to_numpy()
    seed = base.stable_seed("kucab_damage_class")
    splitter = StratifiedGroupKFold(n_splits=base.N_SPLITS, shuffle=True, random_state=seed)
    pred = np.zeros((len(y), len(classes)), dtype=np.float64)
    folds = np.zeros(len(y), dtype=int)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        pred[test_idx] = fit_predict_classifier(
            x[train_idx],
            y[train_idx],
            x[test_idx],
            task="multiclass",
            n_classes=len(classes),
            seed=seed + fold,
        )
        folds[test_idx] = fold

    predicted_ids = np.argmax(pred, axis=1)
    macro_auc = float(roc_auc_score(y, pred, average="macro", multi_class="ovr"))
    row = {
        "benchmark": feature_set.benchmark,
        "endpoint": "damage_class",
        "family": "kucab",
        "task": "multiclass_grouped",
        "representation": feature_set.name,
        "metric": "macro_auroc",
        "score": macro_auc,
        "balanced_accuracy_argmax": float(balanced_accuracy_score(y, predicted_ids)),
        "accuracy": float(accuracy_score(y, predicted_ids)),
        "macro_f1": float(f1_score(y, predicted_ids, average="macro", zero_division=0)),
        "n_samples": int(len(y)),
        "n_classes": int(len(classes)),
        "n_features": int(feature_set.frame.shape[1]),
        "n_folds": base.N_SPLITS,
        "model": "XGBClassifier(lightweight)",
        "split_strategy": "5-fold StratifiedGroupKFold grouped by agent_core",
        "tuning": "none",
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "tree_method": TREE_METHOD,
    }
    predictions = prediction_rows(
        feature_set.benchmark,
        "damage_class",
        "kucab",
        "multiclass_grouped",
        feature_set.name,
        meta.index,
        meta["damage_class"],
        folds,
        pred,
        classes,
    )
    probabilities = probability_long_rows(
        feature_set.benchmark,
        "damage_class",
        "kucab",
        "multiclass_grouped",
        feature_set.name,
        meta.index,
        folds,
        pred,
        classes,
    )
    return row, pd.DataFrame(predictions), pd.DataFrame(probabilities)


def probability_long_rows(
    benchmark: str,
    endpoint: str,
    family: str,
    task: str,
    representation: str,
    samples: pd.Index,
    folds: np.ndarray,
    pred: np.ndarray,
    classes: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for i, sample in enumerate(samples):
        for j in range(pred.shape[1]):
            class_label = str(classes[j]) if task in {"multiclass", "multiclass_grouped"} else str(j)
            rows.append(
                {
                    "benchmark": benchmark,
                    "endpoint": endpoint,
                    "family": family,
                    "task": task,
                    "representation": representation,
                    "sample": sample,
                    "fold": int(folds[i]),
                    "class_label": class_label,
                    "probability": float(pred[i, j]),
                }
            )
    return rows


def prediction_rows(
    benchmark: str,
    endpoint: str,
    family: str,
    task: str,
    representation: str,
    samples: pd.Index,
    y_series: pd.Series,
    folds: np.ndarray,
    pred: np.ndarray,
    classes: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for i, sample in enumerate(samples):
        item: dict[str, object] = {
            "benchmark": benchmark,
            "endpoint": endpoint,
            "family": family,
            "task": task,
            "representation": representation,
            "sample": sample,
            "fold": int(folds[i]),
            "y_true": str(y_series.iloc[i]) if task in {"multiclass", "multiclass_grouped"} else float(y_series.iloc[i]),
        }
        if task == "regression":
            item["prediction"] = float(pred[i])
        elif task == "binary":
            item["prediction"] = float(pred[i, 1])
            item["predicted_label"] = int(pred[i, 1] >= 0.5)
        else:
            pred_id = int(np.argmax(pred[i]))
            item["prediction"] = str(classes[pred_id])
            item["predicted_label"] = str(classes[pred_id])
        rows.append(item)
    return rows


def short_representation(frame: pd.DataFrame) -> pd.DataFrame:
    name_map = {
        "standard_sbs96_id83": "standard",
        "standard_sbs96_dbs78_id83": "standard",
        "previous_uga_mean_pooled": "previous_uga_mean",
        "previous_uga_mean_unified": "previous_uga_mean",
        "uga_rbf_kernel_mean": "uga_rbf_kme",
    }
    out = frame.copy()
    out["rep_short"] = out["representation"].map(name_map)
    return out


def wide_scores(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = short_representation(frame)
    wide = (
        df.pivot_table(
            index=["benchmark", "family", "endpoint", "metric"],
            columns="rep_short",
            values="score",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    wide[f"{prefix}_standard"] = wide["standard"]
    wide[f"{prefix}_previous_uga_mean"] = wide["previous_uga_mean"]
    wide[f"{prefix}_uga_rbf_kme"] = wide["uga_rbf_kme"]
    wide[f"{prefix}_kme_minus_standard"] = wide["uga_rbf_kme"] - wide["standard"]
    wide[f"{prefix}_kme_minus_previous_uga_mean"] = wide["uga_rbf_kme"] - wide["previous_uga_mean"]
    wide[f"{prefix}_previous_uga_mean_minus_standard"] = wide["previous_uga_mean"] - wide["standard"]
    keep = [
        "benchmark",
        "family",
        "endpoint",
        "metric",
        f"{prefix}_standard",
        f"{prefix}_previous_uga_mean",
        f"{prefix}_uga_rbf_kme",
        f"{prefix}_kme_minus_standard",
        f"{prefix}_kme_minus_previous_uga_mean",
        f"{prefix}_previous_uga_mean_minus_standard",
    ]
    return wide[keep]


def sign(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def build_model_sensitivity_summary(linear_results: pd.DataFrame, xgb_results: pd.DataFrame) -> pd.DataFrame:
    linear = wide_scores(linear_results, "linear")
    xgb = wide_scores(xgb_results, "xgb")
    out = linear.merge(xgb, on=["benchmark", "family", "endpoint", "metric"], how="inner")
    comparisons = [
        "kme_minus_standard",
        "kme_minus_previous_uga_mean",
        "previous_uga_mean_minus_standard",
    ]
    for comparison in comparisons:
        out[f"sign_agree_{comparison}"] = [
            sign(lv) == sign(xv)
            for lv, xv in zip(out[f"linear_{comparison}"], out[f"xgb_{comparison}"], strict=False)
        ]
    return out


def build_family_summary(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results[results["representation"] == "uga_rbf_kernel_mean"]
        .groupby(["benchmark", "family"], dropna=False)
        .agg(
            n_endpoints=("endpoint", "nunique"),
            mean_delta_vs_standard=("delta_vs_standard", "mean"),
            median_delta_vs_standard=("delta_vs_standard", "median"),
            wins_vs_standard=("delta_vs_standard", lambda x: int((x > 0).sum())),
            mean_delta_vs_previous_uga_mean=("delta_vs_previous_uga_mean", "mean"),
            median_delta_vs_previous_uga_mean=("delta_vs_previous_uga_mean", "median"),
            wins_vs_previous_uga_mean=("delta_vs_previous_uga_mean", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )


def validate_outputs(results: pd.DataFrame, predictions: pd.DataFrame, probabilities: pd.DataFrame) -> dict[str, object]:
    per_endpoint_rep_counts = results.groupby(["benchmark", "endpoint"], dropna=False)["representation"].nunique()
    missing_scores = int(results["score"].isna().sum())
    fold_mismatches = []
    duplicate_prediction_keys = int(
        predictions.duplicated(["benchmark", "endpoint", "representation", "sample"]).sum()
    )
    for (benchmark, endpoint), group in predictions.groupby(["benchmark", "endpoint"], dropna=False):
        folds = {
            rep: frame.set_index("sample")["fold"].sort_index()
            for rep, frame in group.groupby("representation", dropna=False)
        }
        reps = sorted(folds)
        if len(reps) >= 2:
            first = folds[reps[0]]
            for rep in reps[1:]:
                if not first.equals(folds[rep]):
                    fold_mismatches.append(f"{benchmark}/{endpoint}/{rep}")
    probability_bad_rows = 0
    if not probabilities.empty:
        sums = probabilities.groupby(["benchmark", "endpoint", "representation", "sample"], dropna=False)["probability"].sum()
        probability_bad_rows = int((np.abs(sums - 1.0) > 1e-6).sum())
    return {
        "n_result_rows": int(len(results)),
        "n_endpoint_rows_expected": 51,
        "all_endpoint_representation_counts_are_three": bool((per_endpoint_rep_counts == 3).all()),
        "missing_scores": missing_scores,
        "duplicate_prediction_keys": duplicate_prediction_keys,
        "fold_mismatch_count": int(len(fold_mismatches)),
        "fold_mismatch_examples": fold_mismatches[:10],
        "probability_bad_row_count": probability_bad_rows,
    }


def markdown_endpoint_table(results: pd.DataFrame) -> str:
    wide = wide_scores(results, "xgb")
    table = wide[
        [
            "benchmark",
            "family",
            "endpoint",
            "metric",
            "xgb_standard",
            "xgb_previous_uga_mean",
            "xgb_uga_rbf_kme",
            "xgb_kme_minus_standard",
            "xgb_kme_minus_previous_uga_mean",
            "xgb_previous_uga_mean_minus_standard",
        ]
    ].sort_values(["benchmark", "family", "endpoint"])
    return table.to_markdown(index=False, floatfmt=".6f")


def markdown_sensitivity_table(summary: pd.DataFrame) -> str:
    table = summary[
        [
            "benchmark",
            "family",
            "endpoint",
            "metric",
            "linear_kme_minus_standard",
            "xgb_kme_minus_standard",
            "sign_agree_kme_minus_standard",
            "linear_kme_minus_previous_uga_mean",
            "xgb_kme_minus_previous_uga_mean",
            "sign_agree_kme_minus_previous_uga_mean",
            "linear_previous_uga_mean_minus_standard",
            "xgb_previous_uga_mean_minus_standard",
            "sign_agree_previous_uga_mean_minus_standard",
        ]
    ].sort_values(["benchmark", "family", "endpoint"])
    return table.to_markdown(index=False, floatfmt=".6f")


def update_readme(
    results: pd.DataFrame,
    sensitivity: pd.DataFrame,
    validation: dict[str, object],
    elapsed: float,
) -> None:
    readme = base.EXPERIMENT_ROOT / "README.md"
    marker = "\n## Lightweight XGBoost Sensitivity\n"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else "# UGA RBF Kernel Mean Linear Benchmark\n"
    head = existing.split(marker, 1)[0].rstrip()
    kme = results[results["representation"] == "uga_rbf_kernel_mean"]
    wins_std = int((kme["delta_vs_standard"] > 0).sum())
    wins_prev = int((kme["delta_vs_previous_uga_mean"] > 0).sum())
    total = int(kme["delta_vs_standard"].notna().sum())
    sign_agree_std = int(sensitivity["sign_agree_kme_minus_standard"].sum())
    sign_agree_prev = int(sensitivity["sign_agree_kme_minus_previous_uga_mean"].sum())
    section = [
        "",
        "## Lightweight XGBoost Sensitivity",
        "",
        f"Executed UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Elapsed seconds: {elapsed:.2f}",
        "",
        "This is an exploratory 1x5 lightweight XGBoost sensitivity check. It is not the prior repeated locked XGBoost protocol and should not replace those retained results.",
        "",
        "### Methodology",
        "",
        "- Same 17 endpoints and same three representation families as the corrected linear run.",
        "- One shared endpoint-specific 5-fold OOF split across all representations.",
        "- Kucab uses grouped 5-fold CV by `agent_core`; all other classification endpoints use stratified 5-fold CV; regression endpoints use 5-fold KFold.",
        "- Fixed XGBoost settings: `n_estimators=100`, `max_depth=2`, `learning_rate=0.05`, `subsample=0.85`, `colsample_bytree=0.85`, `min_child_weight=5`, `reg_lambda=2.0`, `reg_alpha=0.05`, `tree_method=hist`, `n_jobs=4`.",
        "- No model tuning, no repeated CV, and no endpoint-specific hyperparameter changes.",
        "",
        "### Headline",
        "",
        f"UGA-RBF KME beat Standard on {wins_std}/{total} XGBoost endpoint comparisons and beat previous UGA mean on {wins_prev}/{total}.",
        f"The sign of `KME - Standard` matched the corrected linear run on {sign_agree_std}/{total} endpoints; the sign of `KME - previous UGA mean` matched on {sign_agree_prev}/{total}.",
        "",
        "### XGBoost Endpoint Results",
        "",
        markdown_endpoint_table(results),
        "",
        "### Linear vs XGBoost Sensitivity",
        "",
        markdown_sensitivity_table(sensitivity),
        "",
        "### Validation",
        "",
        "```json",
        json.dumps(validation, indent=2),
        "```",
    ]
    readme.write_text(head + marker + "\n".join(section[2:]) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    base.ensure_dirs()
    all_results = []
    endpoint_results_path = base.DATA_DIR / "xgboost_endpoint_results.csv"
    prediction_checkpoint_path = base.DATA_DIR / "xgboost_oof_predictions_checkpoint.csv"
    probability_checkpoint_path = base.DATA_DIR / "xgboost_oof_probabilities_long_checkpoint.csv"
    completed = read_completed_keys(endpoint_results_path, base.RESULT_KEY_COLUMNS)

    mc3_feature_sets, _ = base.build_mc3_feature_sets()
    standard_index = mc3_feature_sets[0].frame.index
    endpoints = base.load_hrd_endpoints() + base.load_mc3_clinical_endpoints() + [base.load_kmt2c_endpoint(standard_index)]
    for endpoint in endpoints:
        for feature_set in mc3_feature_sets:
            key = (str(feature_set.benchmark), str(endpoint.name), str(feature_set.name))
            if key in completed:
                print(f"[checkpoint] skip {key}", flush=True)
                continue
            result, predictions, probabilities = evaluate_endpoint(endpoint, feature_set)
            all_results.append(result)
            merge_checkpoint_rows(endpoint_results_path, [result], key_columns=base.RESULT_KEY_COLUMNS, sort_columns=base.RESULT_KEY_COLUMNS)
            merge_checkpoint_rows(
                prediction_checkpoint_path,
                predictions.to_dict("records"),
                key_columns=base.PREDICTION_KEY_COLUMNS,
                sort_columns=base.PREDICTION_KEY_COLUMNS,
            )
            if not probabilities.empty:
                merge_checkpoint_rows(
                    probability_checkpoint_path,
                    probabilities.to_dict("records"),
                    key_columns=base.PROBABILITY_KEY_COLUMNS,
                    sort_columns=base.PROBABILITY_KEY_COLUMNS,
                )
            completed.add(key)
            print(f"[checkpoint] wrote {key}", flush=True)

    kucab_feature_sets, kucab_endpoint, _ = base.build_kucab_feature_sets()
    for feature_set in kucab_feature_sets:
        key = (str(feature_set.benchmark), "damage_class", str(feature_set.name))
        if key in completed:
            print(f"[checkpoint] skip {key}", flush=True)
            continue
        result, predictions, probabilities = evaluate_kucab(feature_set, kucab_endpoint)
        all_results.append(result)
        merge_checkpoint_rows(endpoint_results_path, [result], key_columns=base.RESULT_KEY_COLUMNS, sort_columns=base.RESULT_KEY_COLUMNS)
        merge_checkpoint_rows(
            prediction_checkpoint_path,
            predictions.to_dict("records"),
            key_columns=base.PREDICTION_KEY_COLUMNS,
            sort_columns=base.PREDICTION_KEY_COLUMNS,
        )
        merge_checkpoint_rows(
            probability_checkpoint_path,
            probabilities.to_dict("records"),
            key_columns=base.PROBABILITY_KEY_COLUMNS,
            sort_columns=base.PROBABILITY_KEY_COLUMNS,
        )
        completed.add(key)
        print(f"[checkpoint] wrote {key}", flush=True)

    if not endpoint_results_path.exists():
        raise RuntimeError("No XGBoost endpoint checkpoint rows were written")
    results = base.add_delta_columns(pd.read_csv(endpoint_results_path))
    predictions = pd.read_csv(prediction_checkpoint_path) if prediction_checkpoint_path.exists() else pd.DataFrame()
    probabilities = pd.read_csv(probability_checkpoint_path) if probability_checkpoint_path.exists() else pd.DataFrame()
    family_summary = build_family_summary(results)

    linear_results = pd.read_csv(base.DATA_DIR / "endpoint_results.csv")
    sensitivity = build_model_sensitivity_summary(linear_results, results)
    validation = validate_outputs(results, predictions, probabilities)

    atomic_write_csv(results, base.DATA_DIR / "xgboost_endpoint_results.csv", index=False)
    predictions.to_csv(base.DATA_DIR / "xgboost_oof_predictions.csv.gz", index=False, compression="gzip")
    probabilities.to_csv(base.DATA_DIR / "xgboost_oof_probabilities_long.csv.gz", index=False, compression="gzip")
    atomic_write_csv(family_summary, base.TABLE_DIR / "xgboost_family_summary.csv", index=False)
    atomic_write_csv(sensitivity, base.TABLE_DIR / "model_sensitivity_summary.csv", index=False)

    run_metadata = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - start,
        "random_seed": base.RANDOM_SEED,
        "n_splits": base.N_SPLITS,
        "protocol": "1x5 fixed lightweight XGBoost",
        "xgboost_version": getattr(__import__("xgboost"), "__version__", "unknown"),
        "dyld_library_path": os.environ.get("DYLD_LIBRARY_PATH", ""),
        "params": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "subsample": SUBSAMPLE,
            "colsample_bytree": COLSAMPLE_BYTREE,
            "min_child_weight": MIN_CHILD_WEIGHT,
            "reg_lambda": REG_LAMBDA,
            "reg_alpha": REG_ALPHA,
            "tree_method": TREE_METHOD,
            "n_jobs": XGB_N_JOBS,
        },
        "validation": validation,
    }
    atomic_write_json(base.DATA_DIR / "xgboost_run_metadata.json", run_metadata)
    update_readme(results, sensitivity, validation, time.time() - start)

    kme = results[results["representation"] == "uga_rbf_kernel_mean"]
    payload = {
        "experiment_root": str(base.EXPERIMENT_ROOT),
        "elapsed_seconds": round(time.time() - start, 2),
        "n_endpoint_rows": int(len(results)),
        "kme_wins_vs_standard": int((kme["delta_vs_standard"] > 0).sum()),
        "kme_wins_vs_previous_uga_mean": int((kme["delta_vs_previous_uga_mean"] > 0).sum()),
        "mean_kme_delta_vs_standard": float(kme["delta_vs_standard"].mean()),
        "mean_kme_delta_vs_previous_uga_mean": float(kme["delta_vs_previous_uga_mean"].mean()),
        "validation": validation,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
