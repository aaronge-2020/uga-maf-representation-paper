#!/usr/bin/env python3
"""Locked unified-model MC3 LUAD KMT2C validation."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from utils.checkpointing import atomic_write_csv, atomic_write_json, atomic_write_text, merge_checkpoint_rows, read_completed_keys


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
DATA_DIR = EXPERIMENT_ROOT / "data" / "mc3_driver"
TABLE_DIR = EXPERIMENT_ROOT / "tables" / "mc3_driver"
FIGURE_DIR = EXPERIMENT_ROOT / "figures" / "mc3_driver"

SOURCE_MC3 = EXPERIMENT_ROOT / "data" / "mc3_source"
FEATURE_DIR = SOURCE_MC3 / "features"
RAW_DIR = SOURCE_MC3 / "raw"

LOCKED_SBSDBS_MODEL = "master_spec_sbs_dbs_d10_dp5"
LOCKED_ID_MODEL = "id83_payload_only_d10_dp5"
RANDOM_SEED = 20260514
SUMMARY_KEY_COLUMNS = ["scenario", "endpoint", "cancer_type", "feature_set"]
FOLD_KEY_COLUMNS = ["scenario", "endpoint", "cancer_type", "feature_set", "repeat", "fold"]
PREDICTION_KEY_COLUMNS = ["scenario", "endpoint", "cancer_type", "feature_set", "sample"]

from unified_mc3_helpers import (  # noqa: E402
    bh_q_values,
    build_driver_labels,
    build_mc3_candidate_features,
    load_cancer_types,
    patients_for_task,
    run_oof,
    stratified_bootstrap_delta,
)


def ensure_dirs() -> None:
    for directory in [DATA_DIR, TABLE_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def write_html_table(df: pd.DataFrame, path: Path, title: str, footnote: str) -> None:
    css = """
    body{font-family:Arial,Helvetica,sans-serif;margin:28px;color:#111}
    h1{font-size:18px;margin:0 0 12px 0}
    table{border-collapse:collapse;font-size:12px;width:100%;max-width:1280px}
    th,td{padding:6px 8px;border-bottom:1px solid #bbb;text-align:right}
    th:first-child,td:first-child{text-align:left}
    th{border-top:1.5px solid #111;border-bottom:1.5px solid #111;font-weight:600}
    p{font-size:11px;color:#333;max-width:1120px}
    </style>
    """
    html = [
        "<!doctype html><html><head><meta charset=\"utf-8\"><style>",
        css,
        f"<title>{title}</title></head><body>",
        f"<h1>{title}</h1>",
        df.to_html(index=False, escape=True, border=0, float_format=lambda x: f"{x:.4g}"),
        f"<p>{footnote}</p>",
        "</body></html>",
    ]
    atomic_write_text(path, "\n".join(html))


def write_figure(summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plot = summary.sort_values("oof_auroc", ascending=False).copy()
    atomic_write_csv(plot, DATA_DIR / "figure1_locked_luad_kmt2c_xgboost_auroc.csv", index=False)
    colors = []
    for feature_set in plot["feature_set"]:
        if str(feature_set).startswith("locked_uga"):
            colors.append("#D55E00")
        elif feature_set == "burden_only":
            colors.append("#999999")
        else:
            colors.append("#0072B2")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(plot["feature_set"], plot["oof_auroc"], color=colors)
    ax.set_ylabel("Repeated out-of-fold AUROC")
    ax.set_xlabel("Feature set")
    ax.set_ylim(max(0.45, float(plot["oof_auroc"].min()) - 0.05), min(1.0, float(plot["oof_auroc"].max()) + 0.05))
    ax.tick_params(axis="x", rotation=35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    svg_path = FIGURE_DIR / "figure1_locked_luad_kmt2c_xgboost_auroc.svg"
    png_path = FIGURE_DIR / "figure1_locked_luad_kmt2c_xgboost_auroc.png"
    html_path = FIGURE_DIR / "figure1_locked_luad_kmt2c_xgboost_auroc.html"
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=300)
    svg = svg_path.read_text(encoding="utf-8")
    atomic_write_text(
        html_path,
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Figure 1. Locked LUAD KMT2C repeated XGBoost AUROC</title>"
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#111}h1{font-size:18px}p{font-size:12px;max-width:1000px;color:#333}</style>"
        "</head><body><h1>Figure 1. Locked LUAD KMT2C repeated XGBoost AUROC</h1>"
        "<p>Repeated 5-fold out-of-fold AUROC for KMT2C mutation prediction within TCGA-LUAD using the locked unified UGA model.</p>"
        + svg
        + "</body></html>",
    )


def main() -> None:
    ensure_dirs()
    t0 = time.time()

    standard_sbs = pd.read_csv(FEATURE_DIR / "features_standard_sbs96.csv.gz", index_col=0).fillna(0.0)
    standard_combined = pd.read_csv(FEATURE_DIR / "features_standard_sbs96_id83.csv.gz", index_col=0).fillna(0.0)
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
        "standard_sbs_id": standard_combined,
        "locked_uga_sbs": candidate["uga_sbs"],
        "locked_uga_sbs_id_pooled": candidate["uga_combined_pooled"],
        "locked_uga_sbs_id_separate": candidate["uga_combined_separate"],
    }

    patients = burden.index.astype(str)
    cancer_type = load_cancer_types(RAW_DIR, pd.Index(patients))
    driver_labels = build_driver_labels(SOURCE_MC3, pd.Index(patients))
    scenario = "within_cancer_type_LUAD"
    endpoint = "kmt2c_mutated"
    ct = "LUAD"
    task = {
        "scenario": scenario,
        "scenario_label": "Within LUAD",
        "endpoint": endpoint,
        "cancer_type": ct,
    }
    patients_idx = patients_for_task(pd.Series(task), driver_labels, cancer_type, features)
    y = driver_labels.loc[driver_labels.index.intersection(patients_idx), endpoint].astype(int)
    y_arr = y.to_numpy(dtype=int)

    repeats = int(os.environ.get("CV_REPEATS", "1"))
    folds = int(os.environ.get("CV_FOLDS", "5"))
    n_estimators = int(os.environ.get("XGB_N_ESTIMATORS", "160"))
    bootstrap = int(os.environ.get("BOOTSTRAP_ITERATIONS", "200"))
    summary_checkpoint = DATA_DIR / "locked_mc3_luad_kmt2c_feature_summary_checkpoint.csv"
    fold_checkpoint = DATA_DIR / "locked_mc3_luad_kmt2c_fold_metrics_checkpoint.csv"
    prediction_checkpoint = DATA_DIR / "locked_mc3_luad_kmt2c_oof_predictions_checkpoint.csv"
    final_summary_path = DATA_DIR / "locked_mc3_luad_kmt2c_feature_summary.csv"
    final_fold_path = DATA_DIR / "locked_mc3_luad_kmt2c_fold_metrics.csv"
    if not summary_checkpoint.exists() and final_summary_path.exists():
        atomic_write_csv(pd.read_csv(final_summary_path), summary_checkpoint, index=False)
    if not fold_checkpoint.exists() and final_fold_path.exists():
        atomic_write_csv(pd.read_csv(final_fold_path), fold_checkpoint, index=False)
    completed = read_completed_keys(summary_checkpoint, SUMMARY_KEY_COLUMNS)
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    pred_store: dict[str, np.ndarray] = {}
    if prediction_checkpoint.exists():
        pred_checkpoint = pd.read_csv(prediction_checkpoint, low_memory=False)
        for feature_set, group in pred_checkpoint.groupby("feature_set"):
            ordered = group.set_index("sample").reindex(y.index.astype(str))
            if ordered["predicted_probability"].notna().all():
                pred_store[str(feature_set)] = ordered["predicted_probability"].to_numpy(dtype=float)
    for feature_idx, (feature_set, frame) in enumerate(features.items()):
        key = tuple(str(value) for value in [scenario, endpoint, ct, feature_set])
        if key in completed and feature_set in pred_store:
            print(f"[checkpoint] skip {key}", flush=True)
            continue
        repeat_probas = []
        for repeat in range(repeats):
            seed = RANDOM_SEED + repeat * 10_000 + feature_idx * 100
            auroc, bal_acc, proba, fold_info = run_oof(
                y,
                frame,
                folds=folds,
                n_estimators=n_estimators,
                seed=seed,
                tree_method="gpu_hist",
            )
            repeat_probas.append(proba)
            for row in fold_info:
                row.update(
                    {
                        "repeat": repeat + 1,
                        "scenario": scenario,
                        "endpoint": endpoint,
                        "cancer_type": ct,
                        "feature_set": feature_set,
                        "repeat_auroc": auroc,
                        "repeat_balanced_accuracy": bal_acc,
                    }
                )
                fold_rows.append(row)
        avg_proba = np.mean(repeat_probas, axis=0)
        pred_store[feature_set] = avg_proba
        summary_row = {
            "scenario": scenario,
            "scenario_label": "Within LUAD",
            "endpoint": endpoint,
            "cancer_type": ct,
            "feature_set": feature_set,
            "n": int(len(y_arr)),
            "positive": int(y_arr.sum()),
            "negative": int(len(y_arr) - y_arr.sum()),
            "n_features": int(frame.shape[1]),
            "repeats": repeats,
            "folds": folds,
            "n_estimators": n_estimators,
            "oof_auroc": float(roc_auc_score(y_arr, avg_proba)),
            "oof_balanced_accuracy": float(balanced_accuracy_score(y_arr, (avg_proba >= 0.5).astype(int))),
        }
        prediction_rows = pd.DataFrame(
            {
                "scenario": scenario,
                "endpoint": endpoint,
                "cancer_type": ct,
                "feature_set": feature_set,
                "sample": y.index.astype(str),
                "true_label": y_arr,
                "predicted_probability": avg_proba,
            }
        )
        merge_checkpoint_rows(summary_checkpoint, [summary_row], key_columns=SUMMARY_KEY_COLUMNS, sort_columns=SUMMARY_KEY_COLUMNS)
        merge_checkpoint_rows(fold_checkpoint, fold_rows, key_columns=FOLD_KEY_COLUMNS, sort_columns=FOLD_KEY_COLUMNS)
        merge_checkpoint_rows(prediction_checkpoint, prediction_rows.to_dict("records"), key_columns=PREDICTION_KEY_COLUMNS, sort_columns=PREDICTION_KEY_COLUMNS)
        completed.add(key)
        summary_rows.append(summary_row)
        print(f"[checkpoint] wrote {key} AUROC={summary_row['oof_auroc']:.4f}", flush=True)

    summary = pd.read_csv(summary_checkpoint, low_memory=False) if summary_checkpoint.exists() else pd.DataFrame(summary_rows)
    fold_metrics = pd.read_csv(fold_checkpoint, low_memory=False) if fold_checkpoint.exists() else pd.DataFrame(fold_rows)
    pairs = [
        ("locked_uga_sbs", "standard_sbs", "Locked UGA - Standard (SBS)"),
        ("locked_uga_sbs_id_pooled", "standard_sbs_id", "Locked UGA pooled - Standard (SBS+ID)"),
        ("locked_uga_sbs_id_separate", "standard_sbs_id", "Locked UGA separate-block sensitivity - Standard (SBS+ID)"),
        ("locked_uga_sbs_id_pooled", "locked_uga_sbs", "Locked pooled SBS+ID - Locked SBS"),
        ("locked_uga_sbs_id_pooled", "burden_only", "Locked pooled SBS+ID - Burden only"),
    ]
    auc = summary.set_index("feature_set")["oof_auroc"].to_dict()
    q_rows = []
    for pair_idx, (feature_a, feature_b, comparison) in enumerate(pairs, start=1):
        p_value, ci_low, ci_high = stratified_bootstrap_delta(
            y_arr,
            pred_store[feature_a],
            pred_store[feature_b],
            n_bootstrap=bootstrap,
            seed=RANDOM_SEED + pair_idx,
        )
        q_rows.append(
            {
                "comparison": comparison,
                "feature_set_a": feature_a,
                "feature_set_b": feature_b,
                "auroc_a": float(auc[feature_a]),
                "auroc_b": float(auc[feature_b]),
                "delta_auroc": float(auc[feature_a] - auc[feature_b]),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "p_value": p_value,
                "n_bootstrap": bootstrap,
            }
        )
    q_values = pd.DataFrame(q_rows)
    q_values["q_value"] = bh_q_values(q_values["p_value"].to_numpy())

    atomic_write_csv(summary, DATA_DIR / "locked_mc3_luad_kmt2c_feature_summary.csv", index=False)
    atomic_write_csv(fold_metrics, DATA_DIR / "locked_mc3_luad_kmt2c_fold_metrics.csv", index=False)
    atomic_write_csv(q_values, DATA_DIR / "locked_mc3_luad_kmt2c_pairwise_q_values.csv", index=False)
    atomic_write_csv(summary.sort_values("oof_auroc", ascending=False), DATA_DIR / "table1_locked_mc3_luad_kmt2c_feature_summary.csv", index=False)
    atomic_write_csv(q_values, DATA_DIR / "table2_locked_mc3_luad_kmt2c_pairwise_q_values.csv", index=False)
    write_html_table(
        summary.sort_values("oof_auroc", ascending=False),
        TABLE_DIR / "table1_locked_mc3_luad_kmt2c_feature_summary.html",
        "Table 1. Locked unified-model TCGA-LUAD KMT2C prediction",
        "Repeated 5-fold out-of-fold XGBoost result. AUROC is computed from predictions averaged across five repeated cross-validation runs.",
    )
    write_html_table(
        q_values,
        TABLE_DIR / "table2_locked_mc3_luad_kmt2c_pairwise_q_values.html",
        "Table 2. Locked unified-model TCGA-LUAD KMT2C paired bootstrap q values",
        "P values use stratified paired bootstrap resampling of averaged repeated out-of-fold predictions. Q values use Benjamini-Hochberg correction across contrasts.",
    )
    write_figure(summary)
    elapsed = time.time() - t0
    metadata = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "random_seed": RANDOM_SEED,
        "sbsdbs_model": LOCKED_SBSDBS_MODEL,
        "id_model": LOCKED_ID_MODEL,
        "scenario": scenario,
        "endpoint": endpoint,
        "cancer_type": ct,
        "n": int(len(y_arr)),
        "positive": int(y_arr.sum()),
        "negative": int(len(y_arr) - y_arr.sum()),
        "repeats": repeats,
        "folds": folds,
        "n_estimators": n_estimators,
        "bootstrap_iterations": bootstrap,
    }
    atomic_write_json(DATA_DIR / "run_metadata.json", metadata)
    full_table = summary.sort_values("oof_auroc", ascending=False)[
        ["feature_set", "n", "positive", "negative", "n_features", "oof_auroc", "oof_balanced_accuracy"]
    ]
    q_table = q_values[["comparison", "delta_auroc", "bootstrap_ci_low", "bootstrap_ci_high", "p_value", "q_value"]]
    readme = f"""# Locked Unified-Model MC3 LUAD KMT2C Validation

## Research Question

Do locked unified UGA mutation features improve TCGA-LUAD KMT2C functional mutation prediction relative to standard mutation-channel features?

## Methods

Functional KMT2C mutation labels were derived from the MC3 MAF using missense, truncating, in-frame, splice-site, and translation-start variants. The locked UGA model used `{LOCKED_SBSDBS_MODEL}` for SBS96 and `{LOCKED_ID_MODEL}` for ID83. The primary combined UGA feature was a pooled same-space SBS+ID projection, because both components have the same 70-dimensional UGA basis. XGBoost predictions were averaged across five repeats of five-fold out-of-fold cross-validation before AUROC calculation. Statistical testing used stratified paired-bootstrap resampling of averaged out-of-fold predictions with Benjamini-Hochberg q values across contrasts.

## Full Run Feature Summary

{full_table.to_markdown(index=False)}

## Full Run Statistical Contrasts

{q_table.to_markdown(index=False)}

## Reproducibility

Executed at {metadata['executed_at_utc']} with random seed {RANDOM_SEED}, XGBoost `tree_method=gpu_hist`, {repeats} repeats of {folds}-fold cross-validation, {n_estimators} trees per model, and {bootstrap} bootstrap iterations. Runtime was {elapsed / 60.0:.1f} minutes.
"""
    atomic_write_text(DATA_DIR / "README_mc3_driver.md", readme)
    print(json.dumps({"completed_in_seconds": round(elapsed, 1), "summary_rows": len(summary), "q_rows": len(q_values)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
