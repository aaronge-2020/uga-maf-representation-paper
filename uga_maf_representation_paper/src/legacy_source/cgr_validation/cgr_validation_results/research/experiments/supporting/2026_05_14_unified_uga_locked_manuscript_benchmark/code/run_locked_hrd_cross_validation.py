#!/usr/bin/env python3
"""Locked unified-model TCGA-BRCA HRD benchmark using source-supported modalities."""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

from utils.checkpointing import atomic_write_csv, atomic_write_json, merge_checkpoint_rows, read_completed_keys


def find_project_root() -> Path:
    path = Path(__file__).resolve()
    for candidate in [path.parent, *path.parents]:
        if (candidate / "bench" / "run_pcawg_benchmark.py").is_file():
            return candidate
    raise RuntimeError(f"Could not locate cgr_validation project root from {path}")


PROJECT_ROOT = find_project_root()
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
ASSETS_ROOT = PROJECT_ROOT / "cgr_validation_results" / "research" / "assets" / "EXP023_tcga_brca_hrd" / "TCGA-BRCA"
SOURCE_MC3 = EXPERIMENT_ROOT / "data" / "mc3_source"
FEATURE_DIR = SOURCE_MC3 / "features"

EXP23_SCRIPT_DIR = PROJECT_ROOT / "cgr_validation_results" / "research" / "scripts" / "EXP023_tcga_brca_hrd"
for import_path in (PROJECT_ROOT, EXP23_SCRIPT_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from exp23_config import build_config  # noqa: E402
from exp23_stats import prediction_tests  # noqa: E402
from unified_mc3_helpers import build_mc3_candidate_features  # noqa: E402


RANDOM_STATE = 20260514
OUTER_FOLDS = 5
REPEATS = 5
N_ESTIMATORS = 250
TREE_METHOD = "gpu_hist"

LOCKED_SBSDBS_MODEL = "master_spec_sbs_dbs_d10_dp5"
LOCKED_ID_MODEL = "id83_payload_only_d10_dp5"

FEATURE_LABELS = {
    "burden_only": "Burden only",
    "standard_sbs": "Standard SBS96",
    "locked_uga_sbs": "Locked UGA SBS96",
    "standard_sbs_id": "Standard SBS96+ID83",
    "locked_uga_sbs_id_pooled": "Locked pooled UGA SBS+ID",
    "locked_uga_sbs_id_separate": "Locked separate-block UGA SBS+ID",
}
PRIMARY_STANDARD = "standard_sbs_id"
PRIMARY_UGA = "locked_uga_sbs_id_pooled"

REGRESSION_ENDPOINTS = ["HRD_Score", "eCARD", "HRD_TAI", "HRD_LST", "HRD_LOH", "PARPi7"]
CLASSIFICATION_ENDPOINTS = ["hrd_binary_24", "hrd_binary_33", "hrd_binary_42", "parpi7_binary"]


def ensure_dirs() -> tuple[Path, Path]:
    data_dir = EXPERIMENT_ROOT / "data" / "hrd"
    table_dir = EXPERIMENT_ROOT / "tables" / "hrd"
    data_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, table_dir


def load_cohort() -> pd.DataFrame:
    cohort = pd.read_csv(ASSETS_ROOT / "cohort" / "final_analysis_cohort.tsv", sep="\t")
    cohort["patient_id_12"] = cohort["patient_id_12"].astype(str)
    return cohort


def load_feature_sets() -> dict[str, pd.DataFrame]:
    standard_sbs = pd.read_csv(FEATURE_DIR / "features_standard_sbs96.csv.gz", index_col=0).fillna(0.0)
    standard_id = pd.read_csv(FEATURE_DIR / "features_standard_id83.csv.gz", index_col=0).fillna(0.0)
    standard_sbs_id = pd.read_csv(FEATURE_DIR / "features_standard_sbs96_id83.csv.gz", index_col=0).fillna(0.0)
    burden = pd.read_csv(FEATURE_DIR / "features_burden_only.csv", index_col=0).fillna(0.0)
    candidate = build_mc3_candidate_features(
        standard_sbs,
        standard_id,
        burden,
        LOCKED_SBSDBS_MODEL,
        LOCKED_ID_MODEL,
    )
    return {
        "burden_only": burden,
        "standard_sbs": standard_sbs,
        "locked_uga_sbs": candidate["uga_sbs"],
        "standard_sbs_id": standard_sbs_id,
        "locked_uga_sbs_id_pooled": candidate["uga_combined_pooled"],
        "locked_uga_sbs_id_separate": candidate["uga_combined_separate"],
    }


def feature_audit(cohort: pd.DataFrame, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    hrd_ids = set(cohort["patient_id_12"].astype(str))
    for feature_set, frame in features.items():
        idx = set(frame.index.astype(str))
        rows.append(
            {
                "feature_set": feature_set,
                "label": FEATURE_LABELS[feature_set],
                "n_rows": int(frame.shape[0]),
                "n_features": int(frame.shape[1]),
                "hrd_overlap": int(len(hrd_ids & idx)),
                "hrd_missing": int(len(hrd_ids - idx)),
            }
        )
    return pd.DataFrame(rows)


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rho = spearmanr(y_true, pred)[0]
    r = pearsonr(y_true, pred)[0] if np.std(pred) > 0 and np.std(y_true) > 0 else np.nan
    ss_res = np.sum((y_true - pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "spearman": float(rho),
        "pearson": float(r),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "mae": float(mean_absolute_error(y_true, pred)),
    }


def classification_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if np.unique(y_true).size < 2:
        return {"auroc": np.nan, "auprc": np.nan, "balanced_acc": np.nan, "brier": np.nan}
    return {
        "auroc": float(roc_auc_score(y_true, pred)),
        "auprc": float(average_precision_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, (pred >= 0.5).astype(int))),
        "brier": float(brier_score_loss(y_true, pred)),
    }


def xgb_classifier_params(y_train: np.ndarray, seed: int, tree_method: str) -> dict[str, object]:
    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    params: dict[str, object] = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "reg_lambda": 2.0,
        "reg_alpha": 0.05,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": negatives / max(positives, 1.0),
        "random_state": int(seed),
        "n_jobs": 1,
        "tree_method": tree_method,
        "verbosity": 0,
    }
    if tree_method == "gpu_hist":
        params["predictor"] = "gpu_predictor"
    return params


def xgb_regressor_params(seed: int, tree_method: str) -> dict[str, object]:
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
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
    tree_method: str,
) -> np.ndarray:
    model = XGBClassifier(**xgb_classifier_params(y_train, seed, tree_method))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(X_train, y_train)
        except Exception:
            if tree_method == "gpu_hist":
                model = XGBClassifier(**xgb_classifier_params(y_train, seed, "hist"))
                model.fit(X_train, y_train)
            else:
                raise
    return model.predict_proba(X_test)[:, 1].astype(np.float64)


def fit_predict_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
    tree_method: str,
) -> np.ndarray:
    model = XGBRegressor(**xgb_regressor_params(seed, tree_method))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(X_train, y_train)
        except Exception:
            if tree_method == "gpu_hist":
                model = XGBRegressor(**xgb_regressor_params(seed, "hist"))
                model.fit(X_train, y_train)
            else:
                raise
    return model.predict(X_test).astype(np.float64)


def endpoint_data(cohort: pd.DataFrame, endpoint: str) -> tuple[str, pd.Series]:
    if endpoint in CLASSIFICATION_ENDPOINTS:
        labels = ["PARPi-high", "PARPi-low"] if endpoint == "parpi7_binary" else ["HRD-high", "HRD-low"]
        positive = "PARPi-high" if endpoint == "parpi7_binary" else "HRD-high"
        data = cohort[cohort[endpoint].isin(labels)].copy()
        y = pd.Series((data[endpoint] == positive).astype(int).to_numpy(), index=data["patient_id_12"].astype(str), name=endpoint)
        return "classification", y
    data = cohort.dropna(subset=[endpoint]).copy()
    y = pd.Series(data[endpoint].astype(float).to_numpy(), index=data["patient_id_12"].astype(str), name=endpoint)
    return "regression", y


def run_repeated_oof(
    y: pd.Series,
    features: pd.DataFrame,
    *,
    task: str,
    endpoint_idx: int,
    feature_idx: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    common = y.index.intersection(features.index)
    y = y.loc[common]
    X = features.loc[common].fillna(0.0).to_numpy(dtype=np.float32)
    y_arr = y.to_numpy(dtype=np.float64 if task == "regression" else np.int32)
    repeated_predictions = []
    fold_rows: list[dict[str, object]] = []

    for repeat in range(REPEATS):
        seed = RANDOM_STATE + endpoint_idx * 10_000 + feature_idx * 1000 + repeat * 101
        predictions = np.zeros(len(y_arr), dtype=np.float64)
        if task == "classification":
            min_class = int(pd.Series(y_arr).value_counts().min())
            n_splits = max(2, min(OUTER_FOLDS, min_class))
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_iter = splitter.split(X, y_arr)
        else:
            n_splits = min(OUTER_FOLDS, len(y_arr))
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_iter = splitter.split(X, y_arr)

        for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
            if task == "classification":
                fold_pred = fit_predict_classification(
                    X[train_idx],
                    y_arr[train_idx].astype(np.int32),
                    X[test_idx],
                    seed=seed + fold,
                    tree_method=TREE_METHOD,
                )
                fold_metric = classification_metrics(y_arr[test_idx].astype(np.int32), fold_pred)
            else:
                fold_pred = fit_predict_regression(
                    X[train_idx],
                    y_arr[train_idx].astype(np.float64),
                    X[test_idx],
                    seed=seed + fold,
                    tree_method=TREE_METHOD,
                )
                fold_metric = regression_metrics(y_arr[test_idx].astype(np.float64), fold_pred)
            predictions[test_idx] = fold_pred
            fold_metric.update({"repeat": repeat + 1, "fold": fold, "n_train": int(len(train_idx)), "n_test": int(len(test_idx))})
            fold_rows.append(fold_metric)
        repeated_predictions.append(predictions)

    return y_arr.astype(float), np.mean(repeated_predictions, axis=0), fold_rows


def bh_q_values(p_values: pd.Series) -> pd.Series:
    valid = p_values.dropna().astype(float)
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    if valid.empty:
        return out
    ordered = valid.sort_values()
    running = 1.0
    m = len(ordered)
    adjusted = {}
    for rank, idx in reversed(list(enumerate(ordered.index, start=1))):
        running = min(running, float(ordered.loc[idx]) * m / rank)
        adjusted[idx] = running
    for idx, value in adjusted.items():
        out.loc[idx] = value
    return out


def summary_tables(metrics: pd.DataFrame, primary_stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_index = metrics.set_index(["endpoint", "model"])
    stat_index = primary_stats.set_index(["endpoint", "metric"]) if not primary_stats.empty else pd.DataFrame()
    continuous_rows = []
    binary_rows = []
    for endpoint in REGRESSION_ENDPOINTS:
        standard = metric_index.loc[(endpoint, PRIMARY_STANDARD)]
        uga = metric_index.loc[(endpoint, PRIMARY_UGA)]
        p_value = np.nan
        ci_low = np.nan
        ci_high = np.nan
        if not primary_stats.empty and (endpoint, "spearman") in stat_index.index:
            row = stat_index.loc[(endpoint, "spearman")]
            p_value = float(row["p_value"])
            ci_low = float(row["ci_low"])
            ci_high = float(row["ci_high"])
        continuous_rows.append(
            {
                "Outcome": endpoint,
                "Standard Spearman": standard["spearman"],
                "UGA Spearman": uga["spearman"],
                "Delta rho": uga["spearman"] - standard["spearman"],
                "95% CI lower": ci_low,
                "95% CI upper": ci_high,
                "p value": p_value,
                "Standard R^2": standard["r2"],
                "UGA R^2": uga["r2"],
                "Delta R^2": uga["r2"] - standard["r2"],
            }
        )
    for endpoint in CLASSIFICATION_ENDPOINTS:
        standard = metric_index.loc[(endpoint, PRIMARY_STANDARD)]
        uga = metric_index.loc[(endpoint, PRIMARY_UGA)]
        p_value = np.nan
        ci_low = np.nan
        ci_high = np.nan
        if not primary_stats.empty and (endpoint, "auroc") in stat_index.index:
            row = stat_index.loc[(endpoint, "auroc")]
            p_value = float(row["p_value"])
            ci_low = float(row["ci_low"])
            ci_high = float(row["ci_high"])
        binary_rows.append(
            {
                "Outcome": endpoint,
                "Standard AUROC": standard["auroc"],
                "UGA AUROC": uga["auroc"],
                "Delta AUROC": uga["auroc"] - standard["auroc"],
                "95% CI lower": ci_low,
                "95% CI upper": ci_high,
                "p value": p_value,
                "Standard AUPRC": standard["auprc"],
                "UGA AUPRC": uga["auprc"],
            }
        )
    continuous = pd.DataFrame(continuous_rows)
    binary = pd.DataFrame(binary_rows)
    combined = pd.concat(
        [
            continuous[["Outcome", "p value"]].assign(table="continuous"),
            binary[["Outcome", "p value"]].assign(table="binary"),
        ],
        ignore_index=True,
    )
    combined["q value"] = bh_q_values(combined["p value"])
    continuous.insert(7, "q value", combined.loc[combined["table"] == "continuous", "q value"].to_numpy())
    binary.insert(7, "q value", combined.loc[combined["table"] == "binary", "q value"].to_numpy())
    return continuous, binary


def format_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_html_table(df: pd.DataFrame, path: Path, title: str, footnote: str) -> None:
    header = "".join(f"<th>{escape(str(col))}</th>" for col in df.columns)
    body = []
    for _, row in df.iterrows():
        body.append("".join(f"<td>{escape(format_value(row[col]))}</td>" for col in df.columns))
    html_rows = "".join(f"<tr>{row}</tr>" for row in body)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<style>
body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
caption {{ caption-side: top; text-align: left; font-weight: 700; margin-bottom: 8px; }}
thead th {{ border-top: 1px solid #111; border-bottom: 1px solid #111; padding: 6px 8px; text-align: left; }}
tbody td {{ border-bottom: 0.5px solid #bbb; padding: 5px 8px; text-align: left; }}
tfoot td {{ border-top: 1px solid #111; padding: 6px 8px; font-size: 11px; line-height: 1.35; }}
</style>
</head>
<body>
<table>
<caption>{escape(title)}</caption>
<thead><tr>{header}</tr></thead>
<tbody>{html_rows}</tbody>
<tfoot><tr><td colspan="{len(df.columns)}">{escape(footnote)}</td></tr></tfoot>
</table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_readme(continuous: pd.DataFrame, binary: pd.DataFrame, manifest: dict, data_dir: Path) -> None:
    lines = [
        "# 2026-05-14 TCGA-BRCA HRD Locked UGA Benchmark",
        "",
        "## Research Question",
        "Do locked unified UGA mutation features improve TCGA-BRCA HRD endpoint prediction relative to standard mutation-channel features when only source-supported mutation modalities are used?",
        "",
        "## Methods",
        f"HRD labels from the TCGA-BRCA EXP023 cohort were paired with MC3-derived SBS96 and ID83 mutation features for {manifest['n_hrd_patients']} patients. The primary comparison used Standard SBS96+ID83 mutation-channel frequencies versus locked pooled UGA SBS+ID features. SBS96 was projected with `{LOCKED_SBSDBS_MODEL}`, and ID83 was projected with `{LOCKED_ID_MODEL}`. DBS78 was not included because the retained MC3 and EXP023 inputs do not provide explicit DBS/DNP calls; adjacent-SNV DBS reconstruction was not used in this final benchmark. Predictions were averaged across {REPEATS} repeats of {OUTER_FOLDS}-fold out-of-fold XGBoost. Paired out-of-fold predictions were compared with Steiger tests for continuous endpoints and DeLong tests for binary endpoints, with Benjamini-Hochberg q values across primary Standard-versus-UGA endpoint tests.",
        "",
        "## Key Numerical Findings",
    ]
    for _, row in binary[binary["Outcome"].astype(str).str.startswith("hrd_binary")].iterrows():
        lines.append(
            f"- {row['Outcome']}: UGA AUROC {row['UGA AUROC']:.3f}; Standard AUROC {row['Standard AUROC']:.3f}; delta {row['Delta AUROC']:.3f}; p={row['p value']:.4g}; q={row['q value']:.4g}."
        )
    for _, row in continuous[continuous["Outcome"].isin(["HRD_Score", "HRD_TAI", "HRD_LST", "HRD_LOH"])].iterrows():
        lines.append(
            f"- {row['Outcome']}: UGA Spearman {row['UGA Spearman']:.3f}; Standard Spearman {row['Standard Spearman']:.3f}; delta {row['Delta rho']:.3f}; p={row['p value']:.4g}; q={row['q value']:.4g}."
        )
    lines.extend(
        [
            "",
            "## File Inventory",
            "- `data/hrd/all_metrics.tsv`: repeated out-of-fold endpoint metrics for every feature set.",
            "- `data/hrd/all_fold_metrics.tsv`: per-repeat fold metrics for every feature set.",
            "- `data/hrd/all_oof_predictions.tsv`: averaged out-of-fold predictions for statistical testing.",
            "- `data/hrd/primary_statistical_tests.tsv`: primary Standard SBS96+ID83 versus locked pooled UGA SBS+ID paired tests.",
            "- `data/hrd/feature_audit.csv`: HRD cohort overlap and feature dimensionality for each retained feature set.",
            "- `data/hrd/primary_standard_vs_uga_continuous.csv`: continuous endpoint summary.",
            "- `data/hrd/primary_standard_vs_uga_binary.csv`: binary endpoint summary.",
            "- `tables/hrd/table1_primary_standard_vs_uga_continuous.html`: manuscript-ready continuous endpoint table.",
            "- `tables/hrd/table2_primary_standard_vs_uga_binary.html`: manuscript-ready binary endpoint table.",
            "- `code/run_locked_hrd_cross_validation.py`: reproducible HRD benchmark script.",
            "",
            "## Reproducibility",
            f"Executed at {manifest['executed_at_utc']} with random_state={manifest['random_state']}, repeats={manifest['repeats']}, outer_folds={manifest['outer_folds']}, n_estimators={manifest['n_estimators']}, and tree_method={manifest['tree_method']}.",
            "",
        ]
    )
    (data_dir / "README_hrd.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    data_dir, table_dir = ensure_dirs()
    cohort = load_cohort()
    features = load_feature_sets()
    audit = feature_audit(cohort, features)
    if int(audit["hrd_missing"].max()) != 0:
        raise RuntimeError("At least one feature set is missing HRD cohort patients.")

    metrics_checkpoint = data_dir / "all_metrics_checkpoint.csv"
    fold_checkpoint = data_dir / "all_fold_metrics_checkpoint.csv"
    prediction_checkpoint = data_dir / "all_oof_predictions_checkpoint.csv"
    if not metrics_checkpoint.exists() and (data_dir / "all_metrics.tsv").exists():
        atomic_write_csv(pd.read_csv(data_dir / "all_metrics.tsv", sep="\t"), metrics_checkpoint, index=False)
    if not fold_checkpoint.exists() and (data_dir / "all_fold_metrics.tsv").exists():
        atomic_write_csv(pd.read_csv(data_dir / "all_fold_metrics.tsv", sep="\t"), fold_checkpoint, index=False)
    if not prediction_checkpoint.exists() and (data_dir / "all_oof_predictions.tsv").exists():
        atomic_write_csv(pd.read_csv(data_dir / "all_oof_predictions.tsv", sep="\t"), prediction_checkpoint, index=False)
    completed = read_completed_keys(metrics_checkpoint, ["endpoint", "model"])
    endpoints = [*REGRESSION_ENDPOINTS, *CLASSIFICATION_ENDPOINTS]
    for endpoint_idx, endpoint in enumerate(endpoints):
        task, y = endpoint_data(cohort, endpoint)
        print(f"Endpoint {endpoint}: n={len(y):,}, task={task}", flush=True)
        for feature_idx, (feature_set, frame) in enumerate(features.items()):
            key = (str(endpoint), str(feature_set))
            if key in completed:
                print(f"  [checkpoint] skip {feature_set}", flush=True)
                continue
            y_arr, pred, folds = run_repeated_oof(y, frame, task=task, endpoint_idx=endpoint_idx, feature_idx=feature_idx)
            overall = classification_metrics(y_arr.astype(int), pred) if task == "classification" else regression_metrics(y_arr, pred)
            overall.update(
                {
                    "suite": "main",
                    "subset": "all",
                    "endpoint": endpoint,
                    "task": task,
                    "model": feature_set,
                    "model_label": FEATURE_LABELS[feature_set],
                    "n": int(len(y_arr)),
                    "n_features": int(frame.shape[1]),
                    "repeats": REPEATS,
                    "outer_folds": OUTER_FOLDS,
                }
            )
            merge_checkpoint_rows(metrics_checkpoint, [overall], key_columns=["endpoint", "model"], sort_columns=["endpoint", "model"])
            fold_rows = []
            for row in folds:
                row.update(
                    {
                        "suite": "main",
                        "subset": "all",
                        "endpoint": endpoint,
                        "task": task,
                        "model": feature_set,
                        "model_label": FEATURE_LABELS[feature_set],
                    }
                )
                fold_rows.append(row)
            merge_checkpoint_rows(
                fold_checkpoint,
                fold_rows,
                key_columns=["endpoint", "model", "repeat", "fold"],
                sort_columns=["endpoint", "model", "repeat", "fold"],
            )
            prediction_frame = pd.DataFrame(
                {
                    "suite": "main",
                    "subset": "all",
                    "endpoint": endpoint,
                    "task": task,
                    "model": feature_set,
                    "patient_id": y.index.astype(str).to_numpy(),
                    "true_value": y_arr.astype(float),
                    "pred_value": pred.astype(float),
                }
            )
            merge_checkpoint_rows(
                prediction_checkpoint,
                prediction_frame.to_dict("records"),
                key_columns=["endpoint", "model", "patient_id"],
                sort_columns=["endpoint", "model", "patient_id"],
            )
            metric_name = "auroc" if task == "classification" else "spearman"
            print(f"  {feature_set}: {metric_name}={overall[metric_name]:.4f}", flush=True)
            completed.add(key)

    metrics = pd.read_csv(metrics_checkpoint)
    fold_metrics = pd.read_csv(fold_checkpoint)
    predictions = pd.read_csv(prediction_checkpoint)
    comparisons = [
        ("standard_sbs", "locked_uga_sbs"),
        (PRIMARY_STANDARD, PRIMARY_UGA),
        (PRIMARY_STANDARD, "locked_uga_sbs_id_separate"),
        ("burden_only", PRIMARY_UGA),
    ]
    cfg = build_config(repo_root=PROJECT_ROOT, fasta_walk=False)
    stats = prediction_tests(predictions, cfg, comparisons)
    primary_stats = prediction_tests(
        predictions[predictions["model"].isin([PRIMARY_STANDARD, PRIMARY_UGA])],
        cfg,
        [(PRIMARY_STANDARD, PRIMARY_UGA)],
    )
    continuous, binary = summary_tables(metrics, primary_stats)

    atomic_write_csv(metrics, data_dir / "all_metrics.tsv", sep="\t", index=False)
    atomic_write_csv(fold_metrics, data_dir / "all_fold_metrics.tsv", sep="\t", index=False)
    atomic_write_csv(predictions, data_dir / "all_oof_predictions.tsv", sep="\t", index=False)
    atomic_write_csv(stats, data_dir / "statistical_tests.tsv", sep="\t", index=False)
    atomic_write_csv(primary_stats, data_dir / "primary_statistical_tests.tsv", sep="\t", index=False)
    atomic_write_csv(audit, data_dir / "feature_audit.csv", index=False)
    atomic_write_csv(continuous, data_dir / "primary_standard_vs_uga_continuous.csv", index=False)
    atomic_write_csv(binary, data_dir / "primary_standard_vs_uga_binary.csv", index=False)

    write_html_table(
        continuous,
        table_dir / "table1_primary_standard_vs_uga_continuous.html",
        "Table 1. TCGA-BRCA HRD continuous endpoint prediction: Standard SBS96+ID83 versus locked UGA SBS+ID.",
        "Spearman correlation and R^2 are computed from averaged repeated out-of-fold predictions. Positive delta values favor UGA. Confidence intervals are bootstrap intervals for the paired metric difference; q values use Benjamini-Hochberg correction across primary endpoint tests.",
    )
    write_html_table(
        binary,
        table_dir / "table2_primary_standard_vs_uga_binary.html",
        "Table 2. TCGA-BRCA HRD binary endpoint prediction: Standard SBS96+ID83 versus locked UGA SBS+ID.",
        "AUROC and AUPRC are computed from averaged repeated out-of-fold predictions. Positive delta values favor UGA. Confidence intervals are bootstrap intervals for the paired AUROC difference; q values use Benjamini-Hochberg correction across primary endpoint tests.",
    )

    manifest = {
        "experiment": EXPERIMENT_ROOT.name,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "random_state": RANDOM_STATE,
        "repeats": REPEATS,
        "outer_folds": OUTER_FOLDS,
        "n_estimators": N_ESTIMATORS,
        "tree_method": TREE_METHOD,
        "primary_standard": PRIMARY_STANDARD,
        "primary_uga": PRIMARY_UGA,
        "sbsdbs_model": LOCKED_SBSDBS_MODEL,
        "id_model": LOCKED_ID_MODEL,
        "n_hrd_patients": int(cohort["patient_id_12"].nunique()),
        "modalities": ["SBS96", "ID83"],
        "dbs_policy": "DBS78 omitted because retained inputs contain no explicit DBS/DNP calls; adjacent-SNV DBS reconstruction is not used.",
        "source_cohort": str(ASSETS_ROOT / "cohort" / "final_analysis_cohort.tsv"),
        "source_features": str(FEATURE_DIR),
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions)),
        "statistical_test_rows": int(len(stats)),
    }
    atomic_write_json(data_dir / "run_metadata.json", manifest)
    write_readme(continuous, binary, manifest, data_dir)
    print(json.dumps({"completed_in_seconds": round(manifest["elapsed_seconds"], 1), "metric_rows": len(metrics)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
