#!/usr/bin/env python3
"""Locked unified-model MC3 clinical endpoint benchmark."""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

from utils.checkpointing import atomic_write_csv, atomic_write_json, merge_checkpoint_rows, read_completed_keys


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
DATA_DIR = EXPERIMENT_ROOT / "data" / "mc3_clinical"
TABLE_DIR = EXPERIMENT_ROOT / "tables" / "mc3_clinical"
FIGURE_DIR = EXPERIMENT_ROOT / "figures" / "mc3_clinical"

SOURCE_MC3 = EXPERIMENT_ROOT / "data" / "mc3_source"
FEATURE_DIR = SOURCE_MC3 / "features"

LOCKED_SBSDBS_MODEL = "master_spec_sbs_dbs_d10_dp5"
LOCKED_ID_MODEL = "id83_payload_only_d10_dp5"
RANDOM_SEED = 20260514

CLINICAL_ENDPOINTS = {
    "cancer_type_top10": "multiclass",
    "smoking_ever": "binary",
    "high_purity": "binary",
    "high_stage": "binary",
    "os_event": "binary",
}

FEATURE_SETS = [
    "burden_only",
    "standard_sbs",
    "locked_uga_sbs",
    "standard_sbs_id",
    "locked_uga_sbs_id_pooled",
    "locked_uga_sbs_id_separate",
]


from unified_mc3_helpers import build_mc3_candidate_features  # noqa: E402


def ensure_dirs() -> None:
    for directory in [DATA_DIR, TABLE_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def prepare_endpoint(labels: pd.DataFrame, endpoint: str, kind: str) -> pd.Series | None:
    y = labels[endpoint].dropna()
    if kind == "binary":
        y = y.astype(int)
        counts = y.value_counts()
        if len(counts) != 2 or counts.min() < 25:
            return None
        return y
    counts = y.value_counts()
    y = y[y.isin(counts[counts >= 50].index)].astype(str)
    if y.nunique() < 3:
        return None
    return y


def endpoint_classes(y: pd.Series, kind: str) -> np.ndarray:
    if kind == "binary":
        return np.array([0, 1], dtype=int)
    return np.array(sorted(y.unique()), dtype=object)


def encode_y(y: pd.Series, classes: np.ndarray, kind: str) -> np.ndarray:
    if kind == "binary":
        return y.astype(int).to_numpy()
    mapping = {cls: i for i, cls in enumerate(classes)}
    return y.map(mapping).astype(int).to_numpy()


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    counts = pd.Series(y).value_counts().to_dict()
    n = float(len(y))
    k = float(len(counts))
    return np.array([n / (k * counts[value]) for value in y], dtype=np.float32)


def xgb_params(kind: str, y_train: np.ndarray, classes: np.ndarray, *, n_estimators: int, seed: int, tree_method: str) -> dict[str, object]:
    params: dict[str, object] = {
        "n_estimators": int(n_estimators),
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
        "use_label_encoder": False,
    }
    if kind == "binary":
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


def fit_predict_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    kind: str,
    classes: np.ndarray,
    n_estimators: int,
    seed: int,
    tree_method: str,
) -> np.ndarray:
    model = XGBClassifier(**xgb_params(kind, y_train, classes, n_estimators=n_estimators, seed=seed, tree_method=tree_method))
    sample_weight = balanced_sample_weight(y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        except Exception:
            if tree_method == "gpu_hist":
                model = XGBClassifier(**xgb_params(kind, y_train, classes, n_estimators=n_estimators, seed=seed, tree_method="hist"))
                model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                raise
    proba = model.predict_proba(X_test)
    if kind == "binary" and proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    return proba.astype(np.float64)


def score_auroc(y: np.ndarray, proba: np.ndarray, classes: np.ndarray, kind: str) -> float:
    if kind == "binary":
        return float(roc_auc_score(y, proba[:, 1]))
    y_bin = label_binarize(y, classes=np.arange(len(classes)))
    return float(roc_auc_score(y_bin, proba, average="macro"))


def run_oof_endpoint(
    y_series: pd.Series,
    X_df: pd.DataFrame,
    *,
    kind: str,
    folds: int,
    n_estimators: int,
    seed: int,
    tree_method: str,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    common = y_series.index.intersection(X_df.index)
    y_series = y_series.loc[common]
    classes = endpoint_classes(y_series, kind)
    y = encode_y(y_series, classes, kind)
    X = X_df.loc[common].fillna(0.0).to_numpy(dtype=np.float32)
    n_splits = max(2, min(int(folds), int(pd.Series(y).value_counts().min())))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = np.zeros((len(y), 2 if kind == "binary" else len(classes)), dtype=np.float64)
    pred = np.zeros(len(y), dtype=int)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        fold_proba = fit_predict_xgb(
            X[train_idx],
            y[train_idx],
            X[test_idx],
            kind=kind,
            classes=classes,
            n_estimators=n_estimators,
            seed=seed + fold * 101,
            tree_method=tree_method,
        )
        proba[test_idx] = fold_proba
        pred[test_idx] = np.argmax(fold_proba, axis=1)
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "fold_auroc": score_auroc(y[test_idx], fold_proba, classes, kind),
                "fold_balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred[test_idx])),
            }
        )
    return score_auroc(y, proba, classes, kind), float(balanced_accuracy_score(y, pred)), y, classes, proba, fold_rows


def probability_checkpoint_rows(
    endpoint: str,
    kind: str,
    feature_set: str,
    samples: pd.Index,
    y: np.ndarray,
    classes: np.ndarray,
    proba: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, sample in enumerate(samples.astype(str)):
        for class_id in range(proba.shape[1]):
            class_label = str(classes[class_id]) if kind == "multiclass" else str(class_id)
            rows.append(
                {
                    "endpoint": endpoint,
                    "endpoint_type": kind,
                    "feature_set": feature_set,
                    "sample": sample,
                    "y_encoded": int(y[i]),
                    "class_id": int(class_id),
                    "class_label": class_label,
                    "probability": float(proba[i, class_id]),
                }
            )
    return rows


def load_prediction_checkpoint(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    frame = pd.read_csv(path)
    predictions: dict[str, dict[str, object]] = {}
    for (endpoint, feature_set), group in frame.groupby(["endpoint", "feature_set"], dropna=False):
        kind = str(group["endpoint_type"].iloc[0])
        sample_order = group[["sample"]].drop_duplicates()["sample"].astype(str).tolist()
        wide = group.pivot_table(index="sample", columns="class_id", values="probability", aggfunc="first")
        wide = wide.loc[sample_order].sort_index(axis=1)
        y = (
            group.drop_duplicates("sample")
            .set_index("sample")
            .loc[sample_order, "y_encoded"]
            .astype(int)
            .to_numpy()
        )
        class_table = group[["class_id", "class_label"]].drop_duplicates().sort_values("class_id")
        classes = class_table["class_label"].astype(str).to_numpy(dtype=object)
        predictions[f"{endpoint}||{feature_set}"] = {
            "endpoint": endpoint,
            "kind": kind,
            "feature_set": feature_set,
            "classes": classes,
            "y": y,
            "proba": wide.to_numpy(dtype=np.float64),
        }
    return predictions


def stratified_bootstrap_delta(
    y: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    classes: np.ndarray,
    kind: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y == cls) for cls in np.unique(y)]
    deltas = np.zeros(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample_idx = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in strata])
        deltas[i] = score_auroc(y[sample_idx], proba_a[sample_idx], classes, kind) - score_auroc(
            y[sample_idx],
            proba_b[sample_idx],
            classes,
            kind,
        )
    p_lower = (np.sum(deltas <= 0.0) + 1.0) / (n_bootstrap + 1.0)
    p_upper = (np.sum(deltas >= 0.0) + 1.0) / (n_bootstrap + 1.0)
    p_value = float(min(1.0, 2.0 * min(p_lower, p_upper)))
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return p_value, float(ci_low), float(ci_high)


def bh_q_values(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    q = np.full(len(p), np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return q
    order = valid[np.argsort(p[valid])]
    ranked = p[order]
    adjusted = np.empty_like(ranked)
    running = 1.0
    m = len(ranked)
    for i in range(m - 1, -1, -1):
        running = min(running, ranked[i] * m / (i + 1))
        adjusted[i] = running
    q[order] = np.minimum(adjusted, 1.0)
    return q


def write_html_table(df: pd.DataFrame, path: Path, title: str, footnote: str) -> None:
    css = """
    body{font-family:Arial,Helvetica,sans-serif;margin:28px;color:#111}
    h1{font-size:18px;margin:0 0 12px 0}
    table{border-collapse:collapse;font-size:12px;width:100%;max-width:1280px}
    th,td{padding:6px 8px;border-bottom:1px solid #bbb;text-align:right}
    th:first-child,td:first-child{text-align:left}
    th{border-top:1.5px solid #111;border-bottom:1.5px solid #111;font-weight:600}
    p{font-size:11px;color:#333;max-width:1120px}
    """
    html = [
        "<!doctype html><html><head><meta charset=\"utf-8\"><style>",
        css,
        f"</style><title>{title}</title></head><body>",
        f"<h1>{title}</h1>",
        df.to_html(index=False, escape=True, border=0, float_format=lambda x: f"{x:.4g}"),
        f"<p>{footnote}</p>",
        "</body></html>",
    ]
    path.write_text("\n".join(html), encoding="utf-8")


def write_delta_figure(endpoint_summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plot = endpoint_summary.copy()
    plot.to_csv(DATA_DIR / "figure1_locked_mc3_clinical_delta_auroc.csv", index=False)
    colors = ["#009E73" if x >= 0 else "#D55E00" for x in plot["delta_locked_pooled_vs_standard_sbs_id"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(plot["endpoint"], plot["delta_locked_pooled_vs_standard_sbs_id"], color=colors)
    ax.axhline(0, color="#111111", linewidth=0.8)
    ax.set_ylabel("Delta AUROC")
    ax.set_xlabel("Clinical endpoint")
    ax.tick_params(axis="x", rotation=35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    svg_path = FIGURE_DIR / "figure1_locked_mc3_clinical_delta_auroc.svg"
    png_path = FIGURE_DIR / "figure1_locked_mc3_clinical_delta_auroc.png"
    html_path = FIGURE_DIR / "figure1_locked_mc3_clinical_delta_auroc.html"
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=300)
    svg = svg_path.read_text(encoding="utf-8")
    html_path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Figure 1. Locked MC3 clinical delta AUROC</title>"
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#111}h1{font-size:18px}p{font-size:12px;max-width:1000px;color:#333}</style>"
        "</head><body><h1>Figure 1. Locked MC3 clinical delta AUROC</h1>"
        "<p>Delta AUROC for locked pooled UGA SBS+ID versus Standard SBS+ID across MC3 clinical endpoints.</p>"
        + svg
        + "</body></html>",
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    t0 = time.time()
    labels = pd.read_csv(SOURCE_MC3 / "biology_labels.csv", index_col=0)
    standard_sbs = pd.read_csv(FEATURE_DIR / "features_standard_sbs96.csv.gz", index_col=0).fillna(0.0)
    standard_sbs_id = pd.read_csv(FEATURE_DIR / "features_standard_sbs96_id83.csv.gz", index_col=0).fillna(0.0)
    standard_id = pd.read_csv(FEATURE_DIR / "features_standard_id83.csv.gz", index_col=0).fillna(0.0)
    burden = pd.read_csv(FEATURE_DIR / "features_burden_only.csv", index_col=0).fillna(0.0)
    candidate = build_mc3_candidate_features(
        standard_sbs,
        standard_id,
        burden,
        LOCKED_SBSDBS_MODEL,
        LOCKED_ID_MODEL,
    )
    features = {
        "burden_only": burden,
        "standard_sbs": standard_sbs,
        "locked_uga_sbs": candidate["uga_sbs"],
        "standard_sbs_id": standard_sbs_id,
        "locked_uga_sbs_id_pooled": candidate["uga_combined_pooled"],
        "locked_uga_sbs_id_separate": candidate["uga_combined_separate"],
    }

    folds = 5
    n_estimators = 250
    bootstrap = 1000
    tree_method = "gpu_hist"
    summary_checkpoint = DATA_DIR / "locked_mc3_clinical_feature_summary_checkpoint.csv"
    fold_checkpoint = DATA_DIR / "locked_mc3_clinical_outer_fold_metrics_checkpoint.csv"
    probability_checkpoint = DATA_DIR / "locked_mc3_clinical_oof_probabilities_checkpoint.csv"
    if not summary_checkpoint.exists() and (DATA_DIR / "locked_mc3_clinical_feature_summary.csv").exists():
        atomic_write_csv(pd.read_csv(DATA_DIR / "locked_mc3_clinical_feature_summary.csv"), summary_checkpoint, index=False)
    if not fold_checkpoint.exists() and (DATA_DIR / "locked_mc3_clinical_outer_fold_metrics.csv").exists():
        atomic_write_csv(pd.read_csv(DATA_DIR / "locked_mc3_clinical_outer_fold_metrics.csv"), fold_checkpoint, index=False)
    completed = read_completed_keys(summary_checkpoint, ["endpoint", "feature_set"])
    if not probability_checkpoint.exists():
        completed = set()
    total_jobs = len(CLINICAL_ENDPOINTS) * len(FEATURE_SETS)
    job = 0
    for endpoint_idx, (endpoint, kind) in enumerate(CLINICAL_ENDPOINTS.items()):
        y_series = prepare_endpoint(labels, endpoint, kind)
        if y_series is None:
            continue
        print(f"Endpoint {endpoint}: n={len(y_series):,}, kind={kind}", flush=True)
        for feature_idx, feature_set in enumerate(FEATURE_SETS):
            job += 1
            seed = RANDOM_SEED + endpoint_idx * 1000 + feature_idx * 100
            key = (str(endpoint), str(feature_set))
            if key in completed:
                print(f"  {job}/{total_jobs} [checkpoint] skip {feature_set}", flush=True)
                continue
            print(f"  {job}/{total_jobs} {feature_set}", flush=True)
            auroc, bal_acc, y, classes, proba, rows = run_oof_endpoint(
                y_series,
                features[feature_set],
                kind=kind,
                folds=folds,
                n_estimators=n_estimators,
                seed=seed,
                tree_method=tree_method,
            )
            fold_rows: list[dict[str, object]] = []
            for row in rows:
                row.update({"endpoint": endpoint, "endpoint_type": kind, "feature_set": feature_set})
                fold_rows.append(row)
            summary_row = {
                "endpoint": endpoint,
                "endpoint_type": kind,
                "feature_set": feature_set,
                "n": int(len(y)),
                "n_classes": int(len(classes)),
                "n_features": int(features[feature_set].shape[1]),
                "outer_folds": folds,
                "n_estimators": n_estimators,
                "oof_aggregate_auroc": auroc,
                "oof_balanced_accuracy": bal_acc,
            }
            merge_checkpoint_rows(summary_checkpoint, [summary_row], key_columns=["endpoint", "feature_set"], sort_columns=["endpoint", "feature_set"])
            merge_checkpoint_rows(
                fold_checkpoint,
                fold_rows,
                key_columns=["endpoint", "feature_set", "fold"],
                sort_columns=["endpoint", "feature_set", "fold"],
            )
            common = y_series.index.intersection(features[feature_set].index)
            merge_checkpoint_rows(
                probability_checkpoint,
                probability_checkpoint_rows(endpoint, kind, feature_set, common, y, classes, proba),
                key_columns=["endpoint", "feature_set", "sample", "class_id"],
                sort_columns=["endpoint", "feature_set", "sample", "class_id"],
            )
            completed.add(key)
            print(f"    AUROC={auroc:.4f}, balanced_accuracy={bal_acc:.4f}", flush=True)

    summary = pd.read_csv(summary_checkpoint)
    folds_df = pd.read_csv(fold_checkpoint)
    predictions = load_prediction_checkpoint(probability_checkpoint)
    pairs = [
        ("locked_uga_sbs", "standard_sbs", "Locked UGA - Standard (SBS)"),
        ("locked_uga_sbs_id_pooled", "standard_sbs_id", "Locked pooled UGA - Standard (SBS+ID)"),
        ("locked_uga_sbs_id_separate", "standard_sbs_id", "Locked separate-block UGA - Standard (SBS+ID)"),
        ("locked_uga_sbs_id_pooled", "burden_only", "Locked pooled UGA - Burden only"),
    ]
    auc = summary.set_index(["endpoint", "feature_set"])["oof_aggregate_auroc"].to_dict()
    q_rows = []
    for endpoint_idx, (endpoint, kind) in enumerate(CLINICAL_ENDPOINTS.items()):
        if f"{endpoint}||standard_sbs_id" not in predictions:
            continue
        for pair_idx, (feature_a, feature_b, comparison) in enumerate(pairs, start=1):
            pred_a = predictions[f"{endpoint}||{feature_a}"]
            pred_b = predictions[f"{endpoint}||{feature_b}"]
            p_value, ci_low, ci_high = stratified_bootstrap_delta(
                pred_a["y"],
                pred_a["proba"],
                pred_b["proba"],
                pred_a["classes"],
                pred_a["kind"],
                n_bootstrap=bootstrap,
                seed=RANDOM_SEED + endpoint_idx * 1000 + pair_idx,
            )
            q_rows.append(
                {
                    "endpoint": endpoint,
                    "endpoint_type": kind,
                    "comparison": comparison,
                    "feature_set_a": feature_a,
                    "feature_set_b": feature_b,
                    "auroc_a": float(auc[(endpoint, feature_a)]),
                    "auroc_b": float(auc[(endpoint, feature_b)]),
                    "delta_auroc": float(auc[(endpoint, feature_a)] - auc[(endpoint, feature_b)]),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "p_value": p_value,
                    "n_bootstrap": bootstrap,
                }
            )
    q_values = pd.DataFrame(q_rows)
    q_values["q_value"] = bh_q_values(q_values["p_value"].to_numpy())

    endpoint_rows = []
    for endpoint, kind in CLINICAL_ENDPOINTS.items():
        group = summary[summary["endpoint"] == endpoint].set_index("feature_set")
        if group.empty:
            continue
        pooled_q = q_values[
            (q_values["endpoint"] == endpoint)
            & (q_values["comparison"] == "Locked pooled UGA - Standard (SBS+ID)")
        ].iloc[0]
        endpoint_rows.append(
            {
                "endpoint": endpoint,
                "endpoint_type": kind,
                "n": int(group["n"].iloc[0]),
                "standard_sbs_auroc": float(group.loc["standard_sbs", "oof_aggregate_auroc"]),
                "locked_uga_sbs_auroc": float(group.loc["locked_uga_sbs", "oof_aggregate_auroc"]),
                "delta_locked_sbs_vs_standard_sbs": float(group.loc["locked_uga_sbs", "oof_aggregate_auroc"] - group.loc["standard_sbs", "oof_aggregate_auroc"]),
                "standard_sbs_id_auroc": float(group.loc["standard_sbs_id", "oof_aggregate_auroc"]),
                "locked_pooled_sbs_id_auroc": float(group.loc["locked_uga_sbs_id_pooled", "oof_aggregate_auroc"]),
                "delta_locked_pooled_vs_standard_sbs_id": float(pooled_q["delta_auroc"]),
                "pooled_q_value": float(pooled_q["q_value"]),
            }
        )
    endpoint_summary = pd.DataFrame(endpoint_rows).sort_values("delta_locked_pooled_vs_standard_sbs_id", ascending=False)

    atomic_write_csv(summary, DATA_DIR / "locked_mc3_clinical_feature_summary.csv", index=False)
    atomic_write_csv(folds_df, DATA_DIR / "locked_mc3_clinical_outer_fold_metrics.csv", index=False)
    atomic_write_csv(q_values, DATA_DIR / "locked_mc3_clinical_pairwise_q_values.csv", index=False)
    atomic_write_csv(endpoint_summary, DATA_DIR / "locked_mc3_clinical_endpoint_summary.csv", index=False)
    atomic_write_csv(endpoint_summary, DATA_DIR / "table1_locked_mc3_clinical_endpoint_summary.csv", index=False)
    atomic_write_csv(q_values, DATA_DIR / "table2_locked_mc3_clinical_pairwise_q_values.csv", index=False)
    write_html_table(
        endpoint_summary,
        TABLE_DIR / "table1_locked_mc3_clinical_endpoint_summary.html",
        "Table 1. Locked unified-model MC3 clinical endpoint summary",
        "AUROC is computed from matched five-fold out-of-fold XGBoost predictions. Delta values compare locked pooled UGA SBS+ID against Standard SBS+ID.",
    )
    write_html_table(
        q_values,
        TABLE_DIR / "table2_locked_mc3_clinical_pairwise_q_values.html",
        "Table 2. Locked unified-model MC3 clinical endpoint paired bootstrap q values",
        "P values use stratified paired bootstrap resampling of out-of-fold predictions. Q values use Benjamini-Hochberg correction across displayed contrasts.",
    )
    write_delta_figure(endpoint_summary)
    elapsed = time.time() - t0
    metadata = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "random_seed": RANDOM_SEED,
        "sbsdbs_model": LOCKED_SBSDBS_MODEL,
        "id_model": LOCKED_ID_MODEL,
        "clinical_endpoints": CLINICAL_ENDPOINTS,
        "folds": folds,
        "n_estimators": n_estimators,
        "bootstrap_iterations": bootstrap,
        "tree_method": tree_method,
    }
    atomic_write_json(DATA_DIR / "run_metadata.json", metadata)
    readme = f"""# Locked Unified-Model MC3 Clinical Endpoint Benchmark

## Research Question

Do locked unified UGA mutation features improve prediction of MC3 clinical endpoints relative to standard mutation-channel features?

## Methods

The benchmark evaluated `cancer_type_top10`, `smoking_ever`, `high_purity`, `high_stage`, and `os_event` using matched five-fold out-of-fold XGBoost. The locked UGA model used `{LOCKED_SBSDBS_MODEL}` for SBS96 and `{LOCKED_ID_MODEL}` for ID83. The primary combined feature was a pooled same-space SBS+ID projection, because SBS96 and ID83 both use a 70-dimensional UGA basis. Statistical testing used stratified paired-bootstrap resampling of out-of-fold predictions with Benjamini-Hochberg q values across displayed contrasts.

## Key Findings

Mean AUROC across the five clinical endpoints was {endpoint_summary['locked_pooled_sbs_id_auroc'].mean():.4f} for locked pooled UGA SBS+ID and {endpoint_summary['standard_sbs_id_auroc'].mean():.4f} for Standard SBS+ID. Locked pooled UGA SBS+ID exceeded Standard SBS+ID for {(endpoint_summary['delta_locked_pooled_vs_standard_sbs_id'] > 0).sum()} of {len(endpoint_summary)} endpoints. No locked pooled UGA versus Standard SBS+ID clinical contrast passed FDR correction.

## File Inventory

| File | Purpose |
|---|---|
| `locked_mc3_clinical_feature_summary.csv` | Feature-level out-of-fold AUROC and balanced accuracy for every endpoint. |
| `locked_mc3_clinical_endpoint_summary.csv` | Endpoint-level Standard versus locked UGA summary. |
| `locked_mc3_clinical_pairwise_q_values.csv` | Paired bootstrap p values and q values. |
| `locked_mc3_clinical_outer_fold_metrics.csv` | Fold-level AUROC and balanced accuracy. |
| `table1_locked_mc3_clinical_endpoint_summary.csv` and `.html` | Manuscript-ready endpoint summary table. |
| `table2_locked_mc3_clinical_pairwise_q_values.csv` and `.html` | Manuscript-ready statistical contrast table. |
| `figure1_locked_mc3_clinical_delta_auroc.*` | Clinical endpoint delta-AUROC figure. |

## Reproducibility

Executed at {metadata['executed_at_utc']} with random seed {RANDOM_SEED}, XGBoost `tree_method={tree_method}`, {folds}-fold cross-validation, {n_estimators} trees per model, and {bootstrap} bootstrap iterations. Runtime was {elapsed / 60.0:.1f} minutes.
"""
    (DATA_DIR / "README_mc3_clinical.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"completed_in_seconds": round(elapsed, 1), "endpoints": len(endpoint_summary), "q_rows": len(q_values)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
