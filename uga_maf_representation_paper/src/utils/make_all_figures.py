"""Build strict manuscript tables and figures from regenerated bundle outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import BUNDLE_ROOT, load_yaml, resolve_paths_map
from utils.checkpointing import atomic_write_csv, atomic_write_json
from utils.stat_tests import bh_qvalues, paired_bootstrap_delta, paired_delong_auc


MAIN_ENDPOINTS = ["damage_class", "HRD_Score", "hrd_binary_33", "cancer_type_top10", "os_event"]
MAIN_REPRESENTATIONS = [
    "burden_only",
    "signatures_only",
    "MAF_stack_only",
    "signatures_plus_MAF_stack",
    "one_hot_event_KME",
]
MODEL_FAMILIES = ["elastic_net", "XGBoost"]
CANONICAL_SOURCE_BY_FAMILY = {
    "burden_only": "main_manuscript_complete_panel",
    "signatures_only": "main_manuscript_complete_panel",
    "MAF_stack_only": "main_manuscript_complete_panel",
    "signatures_plus_MAF_stack": "main_manuscript_complete_panel",
    "one_hot_event_KME": "one_hot_event_kme_scout",
}
MAIN_COMPARISONS = [
    ("figure_2", "signatures_vs_burden", "signatures_only", "burden_only"),
    ("figure_3", "one_hot_kme_vs_signatures", "one_hot_event_KME", "signatures_only"),
    ("figure_4", "maf_stack_vs_signatures", "MAF_stack_only", "signatures_only"),
    ("figure_4", "sig_maf_vs_signatures", "signatures_plus_MAF_stack", "signatures_only"),
    ("figure_4", "sig_maf_vs_maf_stack", "signatures_plus_MAF_stack", "MAF_stack_only"),
    ("figure_5", "signatures_vs_burden", "signatures_only", "burden_only"),
    ("figure_5", "one_hot_kme_vs_signatures", "one_hot_event_KME", "signatures_only"),
    ("figure_5", "maf_stack_vs_signatures", "MAF_stack_only", "signatures_only"),
    ("figure_5", "sig_maf_vs_signatures", "signatures_plus_MAF_stack", "signatures_only"),
]
REQUIRED_ENDPOINT_FILES = [
    "unified_locked_endpoint_results.csv",
    "uga_kme_endpoint_results.csv",
    "maf_event_gene_locus_endpoint_results.csv",
    "main_manuscript_complete_panel_endpoint_results.csv",
    "one_hot_event_kme_scout_endpoint_results.csv",
]
SOURCE_PRIORITY = {
    "main_manuscript_complete_panel": 60,
    "one_hot_event_kme_scout": 55,
    "maf_event_gene_locus": 35,
    "unified_locked": 25,
    "uga_kme": 20,
    "mechanistic_representation": 10,
}
REQUIRED_MANUSCRIPT_FILES = [
    "tables/table_1_datasets_endpoints.csv",
    "tables/table_2_full_performance_metrics.csv",
    "tables/table_3_hyperparameters_feature_dimensionality.csv",
    "tables/table_4_label_mapping.csv",
    "text/manuscript_captions_and_results.md",
    "text/label_mapping_notes.md",
    "supplement/table_s1_class_distribution_baselines.csv",
    "supplement/table_s2_sensitivity_analyses.csv",
    "figures/figure_1_conceptual_overview.png",
    "figures/figure_2_signature_baselines.png",
    "figures/figure_3_geometry_vs_signatures.png",
    "figures/figure_4_maf_stack_vs_signatures.png",
    "figures/figure_5_cross_endpoint_summary.png",
    "supplement/figure_s1_representation_construction.png",
    "supplement/figure_s2_calibration_thresholds.png",
    "supplement/figure_s3_feature_importance.png",
]
LABEL_REGISTRY_PATH = BUNDLE_ROOT / "src" / "utils" / "label_registry.json"
DISPLAY_COLUMN_SPECS = [
    ("endpoint", "endpoint_display", "endpoint"),
    ("endpoint_tier", "endpoint_tier_display", "endpoint_tier"),
    ("task", "task_display", "task"),
    ("representation_family", "representation_family_display", "representation_family"),
    ("representation", "representation_display", "representation"),
    ("atlas_status", "atlas_status_display", "atlas_status"),
    ("model_family", "model_display", "model_family"),
    ("model_label", "model_label_display", "model_label"),
    ("display_model", "display_model_display", "model_label"),
    ("metric", "metric_display", "metric"),
    ("candidate_representation", "candidate_representation_display", "representation_family"),
    ("baseline_representation", "baseline_representation_display", "representation_family"),
    ("comparison_name", "comparison_display", "comparison_name"),
    ("calibration_mode", "calibration_mode_display", "calibration_mode"),
    ("analysis_family", "analysis_family_display", "analysis_family"),
]
DISPLAY_BY_SOURCE = {source: display for source, display, _ in DISPLAY_COLUMN_SPECS}
DISPLAY_DOMAIN_BY_SOURCE = {source: domain for source, _, domain in DISPLAY_COLUMN_SPECS}


def _load_label_registry() -> dict[str, dict[str, dict[str, str]]]:
    if not LABEL_REGISTRY_PATH.exists():
        return {}
    with LABEL_REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


LABEL_REGISTRY = _load_label_registry()


def _fallback_label(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    aliases = {"xgboost": "XGBoost", "auroc": "AUROC", "auprc": "AUPRC"}
    if text.lower() in aliases:
        return aliases[text.lower()]
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(word.upper() if word.upper() in {"HRD", "LOH", "LST", "TAI", "SBS", "DBS", "ID", "KME", "UGA", "MAF", "NNLS", "VAF", "MC3", "LUAD", "TCGA", "BRCA", "MMR", "POLE", "POLD1"} else word.capitalize() for word in text.split())


def _display_label(domain: str, value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    entry = (LABEL_REGISTRY.get(domain) or {}).get(text)
    if entry:
        return str(entry.get("display") or text)
    return _fallback_label(text)


def _label_description(domain: str, value: object) -> str:
    text = str(value or "").strip()
    entry = (LABEL_REGISTRY.get(domain) or {}).get(text)
    if entry:
        return str(entry.get("description") or entry.get("display") or text)
    return _display_label(domain, text)


def _add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for source, display, domain in DISPLAY_COLUMN_SPECS:
        if source in out.columns:
            values = out[source].map(lambda value, d=domain: _display_label(d, value))
            if display in out.columns:
                out[display] = values
            else:
                source_idx = list(out.columns).index(source)
                out.insert(source_idx + 1, display, values)
    return out


def _publication_column_order(df: pd.DataFrame) -> list[str]:
    display_cols = [display for source, display, _ in DISPLAY_COLUMN_SPECS if display in df.columns]
    score_cols = [col for col in ["primary_score", "auroc", "auprc", "accuracy", "f1", "balanced_accuracy", "p_value", "q_value"] if col in df.columns]
    machine_cols = [source for source, _, _ in DISPLAY_COLUMN_SPECS if source in df.columns]
    provenance_cols = [
        col
        for col in [
            "run_id",
            "cache_key",
            "oof_prediction_file",
            "fold_metrics_file",
            "experiment_id",
            "source_file",
            "bundle_table",
            "canonical_slot_id",
        ]
        if col in df.columns
    ]
    used = set(display_cols + score_cols + machine_cols + provenance_cols)
    middle = [col for col in df.columns if col not in used]
    return display_cols + score_cols + middle + machine_cols + provenance_cols


def _write_publication_copy(df: pd.DataFrame, path: Path, public_dir: Path) -> None:
    public_dir.mkdir(parents=True, exist_ok=True)
    public = df.loc[:, _publication_column_order(df)]
    public.to_csv(public_dir / path.name, index=False)


def _read_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix == ".tsv":
            return pd.read_csv(path, sep="\t", low_memory=False)
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def _load_endpoint_results(tables_dir: Path, *, strict: bool) -> pd.DataFrame:
    missing = [name for name in REQUIRED_ENDPOINT_FILES if not (tables_dir / name).exists()]
    if strict and missing:
        raise FileNotFoundError(f"Missing required fresh endpoint outputs: {', '.join(missing)}")
    frames: list[pd.DataFrame] = []
    for path in sorted(tables_dir.glob("*endpoint_results*.csv")):
        frame = _read_table(path)
        if frame is None:
            continue
        frame.insert(0, "bundle_table", path.name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return _add_display_columns(pd.concat(frames, ignore_index=True, sort=False))


def _load_side_tables(tables_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(tables_dir.glob("*.csv")):
        if "endpoint_results" in path.name:
            continue
        frame = _read_table(path)
        if frame is not None:
            out[path.name] = frame
    return out


def _text(row: pd.Series, names: list[str], default: str = "") -> str:
    for name in names:
        if name in row.index and pd.notna(row[name]) and str(row[name]).strip():
            return str(row[name]).strip()
    return default


def _num(row: pd.Series, names: list[str]) -> float:
    for name in names:
        if name in row.index:
            value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return float("nan")


def _clean_endpoint(value: str) -> str:
    value = str(value or "").strip()
    aliases = {
        "kucab_damage_class": "damage_class",
        "mc3_hrd33": "hrd_binary_33",
    }
    return aliases.get(value, value)


def _representation_family(raw: str, experiment_id: str, source_file: str) -> str:
    raw_value = str(raw or "").lower()
    value = f"{raw} {source_file}".lower()
    if "maf_only" in value or "maf_stack_only" in value:
        return "MAF_stack_only"
    if "id_plus_best_gene_locus" in value or "signatures_plus_maf" in value:
        return "signatures_plus_MAF_stack"
    if "one_hot_event_kme" in raw_value:
        return "one_hot_event_KME"
    if "uga_rbf_kernel_mean" in value or "tuned_kme" in value or "channel_kme" in value:
        return "channel_KME"
    if "locked_uga" in value or "previous_uga" in value or "uga_mean" in value:
        return "UGA_geometry"
    if "exposure" in value or "nnls" in value:
        return "COSMIC_NNLS_exposures"
    if "burden" in value:
        return "burden_only"
    if "standard" in value or "sbs" in value or "id83" in value or "dbs" in value:
        return "signatures_only"
    if "mechanistic" in value or "payload" in value or "shared_space" in value:
        return "mechanistic_control"
    return "other"


def _atlas_status(family: str) -> str:
    if family in {"UGA_geometry", "channel_KME"}:
        return "uses UGA channel atlas for SBS/DBS; ID payload encoder for ID83"
    if family == "one_hot_event_KME":
        return "uses GRCh37 FASTA windows; no UGA atlas"
    if family in {"burden_only", "signatures_only", "MAF_stack_only", "signatures_plus_MAF_stack"}:
        return "no UGA atlas"
    if family == "COSMIC_NNLS_exposures":
        return "uses COSMIC reference signatures; no UGA atlas for standard exposures"
    return "not applicable"


def _model_family(raw: str, experiment_id: str) -> str:
    text = f"{raw} {experiment_id}".lower()
    if "xgb" in text or "xgboost" in text:
        return "XGBoost"
    if "maf_event_gene_locus" in text:
        return "XGBoost"
    if "linear" in text or "ridge" in text or "logistic" in text or "elastic" in text:
        return "elastic_net"
    if "kucab" in text:
        return "elastic_net"
    return "unspecified"


def _metric_name(row: pd.Series) -> str:
    metric = _text(row, ["metric", "metric_name", "primary_metric_name", "primary_metric", "Metric"], "")
    if metric:
        return metric
    if pd.notna(_num(row, ["oof_auroc", "auroc"])):
        return "auroc"
    if pd.notna(_num(row, ["spearman"])):
        return "spearman"
    if pd.notna(_num(row, ["oof_balanced_accuracy", "balanced_accuracy"])):
        return "balanced_accuracy"
    return "score"


def _primary_score(row: pd.Series) -> float:
    return _num(
        row,
        [
            "score",
            "candidate_score",
            "oof_auroc",
            "auroc",
            "spearman",
            "oof_balanced_accuracy",
            "balanced_accuracy",
            "macro_auroc",
            "mean_cosine",
        ],
    )


def _normalize(endpoint_results: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in endpoint_results.iterrows():
        endpoint = _clean_endpoint(_text(row, ["endpoint", "Endpoint", "benchmark", "analysis_set"], ""))
        raw_rep = _text(row, ["representation", "feature_set", "model_id", "candidate", "Candidate", "model", "model_label"], "")
        experiment_id = _text(row, ["experiment_id"], _text(row, ["bundle_table"], "unknown").replace("_endpoint_results.csv", ""))
        source_file = _text(row, ["source_file", "bundle_table"], "")
        family = _representation_family(raw_rep, experiment_id, source_file)
        score = _primary_score(row)
        if not endpoint or not np.isfinite(score):
            continue
        model_raw = _text(row, ["learner", "model", "model_label", "algorithm"], experiment_id)
        optuna_trials_completed = _num(row, ["optuna_trials_completed"])
        source_priority = SOURCE_PRIORITY.get(experiment_id, 0)
        if family == "one_hot_event_KME":
            source_priority += 5 if pd.notna(optuna_trials_completed) and int(optuna_trials_completed) == 10 else -5
        rows.append(
            {
                "run_id": run_id,
                "endpoint": endpoint,
                "endpoint_tier": "main" if endpoint in MAIN_ENDPOINTS else "supplement",
                "endpoint_family": _text(row, ["endpoint_family", "family", "suite", "benchmark"], ""),
                "task": _text(row, ["task", "endpoint_type"], ""),
                "representation": raw_rep or family,
                "representation_family": family,
                "atlas_status": _atlas_status(family),
                "model_family": _model_family(model_raw, experiment_id),
                "model_label": model_raw,
                "metric": _metric_name(row),
                "primary_score": score,
                "auroc": _num(row, ["auroc", "oof_auroc"]),
                "auprc": _num(row, ["auprc"]),
                "accuracy": _num(row, ["accuracy", "mean_accuracy"]),
                "f1": _num(row, ["macro_f1", "mean_macro_f1"]),
                "balanced_accuracy": _num(row, ["balanced_accuracy", "oof_balanced_accuracy", "mean_balanced_accuracy"]),
                "delta_vs_signatures": _num(row, ["delta_vs_standard", "delta_locked_uga_minus_standard", "delta_balanced_accuracy", "delta"]),
                "ci_low": _num(row, ["ci_low", "delta_ci_low", "bootstrap_ci_low", "delta_95ci_low"]),
                "ci_high": _num(row, ["ci_high", "delta_ci_high", "bootstrap_ci_high", "delta_95ci_high"]),
                "p_value": _num(row, ["p_value", "p_balanced_accuracy"]),
                "q_value": _num(row, ["q_value", "q_balanced_accuracy"]),
                "n_samples": _num(row, ["n_samples", "n", "n_patients", "n_test"]),
                "n_features": _num(row, ["n_features", "feature_dimension", "standard_features", "standard_plus_kme_features"]),
                "folds": _num(row, ["n_folds", "folds", "outer_folds"]),
                "repeats": _num(row, ["repeats", "n_repeats"]),
                "xgb_estimators": _num(row, ["n_estimators"]),
                "optuna_trials_completed": optuna_trials_completed,
                "cache_key": _text(row, ["cache_key"], ""),
                "oof_prediction_file": _text(row, ["oof_prediction_file"], ""),
                "fold_metrics_file": _text(row, ["fold_metrics_file"], ""),
                "na_reason": _text(row, ["na_reason"], ""),
                "split_strategy": _text(row, ["split_strategy"], "5-fold OOF CV where available"),
                "experiment_id": experiment_id,
                "source_file": source_file,
                "bundle_table": _text(row, ["bundle_table"], ""),
                "source_priority": source_priority,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "endpoint",
                "endpoint_tier",
                "representation_family",
                "model_family",
                "metric",
                "primary_score",
            ]
        )
    out = out.drop_duplicates()
    out["folds"] = out["folds"].fillna(5)
    out["repeats"] = out["repeats"].fillna(1)
    return _add_display_columns(out)


def _best_scores(df: pd.DataFrame, *, main_only: bool = False, families: list[str] | None = None) -> pd.DataFrame:
    work = df.copy()
    if main_only:
        work = work[work["endpoint_tier"].eq("main")]
    if families:
        work = work[work["representation_family"].isin(families)]
    if work.empty:
        return work
    if "source_priority" not in work.columns:
        work["source_priority"] = 0
    work = work.sort_values(
        ["endpoint", "representation_family", "model_family", "source_priority", "primary_score"],
        ascending=[True, True, True, False, False],
    )
    return work.groupby(["endpoint", "representation_family", "model_family"], as_index=False).head(1)


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=260 if suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)


def _write_text_panel(stem: Path, title: str, boxes: list[tuple[float, float, str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    ax.text(0.02, 0.96, title, fontsize=18, weight="bold", va="top")
    for x, y, header, body in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.28, 0.18, facecolor="#f6f7f9", edgecolor="#333333", linewidth=1.0))
        ax.text(x + 0.015, y + 0.145, header, fontsize=11, weight="bold", va="top")
        ax.text(x + 0.015, y + 0.105, body, fontsize=9.5, va="top", wrap=True)
    for x1, y1, x2, y2 in [(0.30, 0.66, 0.37, 0.66), (0.65, 0.66, 0.72, 0.66), (0.30, 0.36, 0.37, 0.36), (0.65, 0.36, 0.72, 0.36)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#333333"})
    _save_figure(fig, stem)


def _figure_1(figures_dir: Path) -> None:
    _write_text_panel(
        figures_dir / "figure_1_conceptual_overview",
        "Figure 1. Mutation Catalogues to Tabular Representations",
        [
            (0.03, 0.58, "A. Sample Catalogue", "MAF/VCF events: loci, alleles, genes, consequences, VAF, local sequence context."),
            (0.38, 0.72, "Signatures", "Burden plus SBS96/ID83/DBS78 channel spectra; canonical spectra baseline."),
            (0.38, 0.48, "Geometry", "Main: one-hot event KME from FASTA windows. Supplement: UGA and channel KME atlas variants."),
            (0.38, 0.24, "MAF Stack", "Gene, pathway, consequence, locus, VAF, and locus-topography aggregations."),
            (0.73, 0.58, "Models and Endpoints", "Elastic-net and XGBoost. Main endpoints: Kucab, HRD_Score, HRD33, cancer type, OS event."),
            (0.73, 0.30, "End-to-End Alternatives", "MuAt/ATGC operate directly on event sets; included as conceptual alternatives, not benchmarked here."),
        ],
    )


def _barplot(df: pd.DataFrame, stem: Path, title: str, families: list[str]) -> None:
    plot = _best_scores(df, main_only=True, families=families)
    if plot.empty:
        plot = pd.DataFrame({"endpoint": ["missing"], "representation_family": ["missing"], "model_family": ["missing"], "primary_score": [0.0]})
    plot["label"] = plot["endpoint"].astype(str) + "\n" + plot["model_family"].astype(str)
    pivot = plot.pivot_table(index="label", columns="representation_family", values="primary_score", aggfunc="max").reindex(columns=families)
    fig, ax = plt.subplots(figsize=(max(9, len(pivot) * 0.9), 5.4))
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title(title)
    ax.set_ylabel("Primary OOF metric")
    ax.set_xlabel("")
    ax.set_ylim(0, min(1.05, max(0.75, float(np.nanmax(pivot.to_numpy())) + 0.1)) if np.isfinite(pivot.to_numpy()).any() else 1)
    ax.legend(title="", frameon=False, ncol=2)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_figure(fig, stem)


def _heatmap(df: pd.DataFrame, stem: Path, title: str, families: list[str], *, main_only: bool) -> None:
    plot = _best_scores(df, main_only=main_only, families=families)
    if plot.empty:
        plot = pd.DataFrame({"endpoint": ["missing"], "representation_family": ["missing"], "primary_score": [np.nan]})
    pivot = plot.pivot_table(index="endpoint", columns="representation_family", values="primary_score", aggfunc="max").reindex(columns=families)
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(families) + 3), max(4.5, 0.35 * len(pivot) + 2)))
    arr = pivot.to_numpy(dtype=float)
    im = ax.imshow(arr, cmap="viridis", vmin=np.nanmin(arr) if np.isfinite(arr).any() else 0, vmax=np.nanmax(arr) if np.isfinite(arr).any() else 1, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", color="white" if arr[i, j] < np.nanmean(arr) else "black", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Primary OOF metric")
    fig.tight_layout()
    _save_figure(fig, stem)


def _write_tables(df: pd.DataFrame, side_tables: dict[str, pd.DataFrame], manuscript_dir: Path) -> None:
    tables_dir = manuscript_dir / "tables"
    public_dir = tables_dir / "publication"
    supp_dir = manuscript_dir / "supplement"
    tables_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = []
    for endpoint in sorted(df["endpoint"].dropna().unique()):
        sub = df[df["endpoint"].eq(endpoint)]
        dataset_rows.append(
            {
                "endpoint": endpoint,
                "endpoint_tier": "main" if endpoint in MAIN_ENDPOINTS else "supplement",
                "n_samples_max": int(pd.to_numeric(sub["n_samples"], errors="coerce").max()) if pd.to_numeric(sub["n_samples"], errors="coerce").notna().any() else "",
                "task": "; ".join(sorted({str(v) for v in sub["task"].dropna().unique() if str(v)})),
                "assay_or_source": _source_label(endpoint),
                "label_definition": _label_definition(endpoint),
                "splitting_scheme": "single 5-fold CV with aggregated out-of-fold test predictions where model-based",
            }
        )
    table1 = _add_display_columns(pd.DataFrame(dataset_rows))
    table1_path = tables_dir / "table_1_datasets_endpoints.csv"
    table1.to_csv(table1_path, index=False)
    _write_publication_copy(table1, table1_path, public_dir)

    ordered_cols = [
        "run_id",
        "endpoint_tier",
        "endpoint_tier_display",
        "endpoint",
        "endpoint_display",
        "task",
        "task_display",
        "representation_family",
        "representation_family_display",
        "representation",
        "representation_display",
        "atlas_status",
        "atlas_status_display",
        "model_family",
        "model_display",
        "metric",
        "metric_display",
        "primary_score",
        "auroc",
        "auprc",
        "accuracy",
        "f1",
        "balanced_accuracy",
        "delta_vs_signatures",
        "ci_low",
        "ci_high",
        "p_value",
        "q_value",
        "n_samples",
        "n_features",
        "folds",
        "repeats",
        "xgb_estimators",
        "optuna_trials_completed",
        "cache_key",
        "oof_prediction_file",
        "fold_metrics_file",
        "na_reason",
        "split_strategy",
        "experiment_id",
        "source_file",
    ]
    table2 = _add_display_columns(df.loc[:, [col for col in ordered_cols if col in df.columns]])
    table2_path = tables_dir / "table_2_full_performance_metrics.csv"
    table2.to_csv(table2_path, index=False)
    _write_publication_copy(table2, table2_path, public_dir)

    hyper = (
        df.groupby(["representation_family", "model_family", "atlas_status"], dropna=False)
        .agg(
            rows=("primary_score", "size"),
            median_features=("n_features", "median"),
            max_features=("n_features", "max"),
            folds=("folds", "max"),
            repeats=("repeats", "max"),
            xgb_estimators=("xgb_estimators", "max"),
        )
        .reset_index()
    )
    hyper["linear_model"] = np.where(hyper["model_family"].eq("elastic_net"), "elastic-net linear model", "")
    hyper["tuning"] = "10 Optuna trials where endpoint-specific tuning is used; otherwise frozen settings"
    hyper = _add_display_columns(hyper)
    table3_path = tables_dir / "table_3_hyperparameters_feature_dimensionality.csv"
    hyper.to_csv(table3_path, index=False)
    _write_publication_copy(hyper, table3_path, public_dir)

    class_dist = (
        df.groupby(["endpoint_tier", "endpoint"], dropna=False)
        .agg(n_samples_max=("n_samples", "max"), metrics_seen=("metric", lambda x: "; ".join(sorted(set(map(str, x))))), representations=("representation_family", lambda x: "; ".join(sorted(set(map(str, x))))))
        .reset_index()
    )
    class_dist["naive_baseline_context"] = "prevalence/all-negative baselines should be interpreted from endpoint class balance; model rows use OOF predictions"
    class_dist = _add_display_columns(class_dist)
    class_dist.to_csv(supp_dir / "table_s1_class_distribution_baselines.csv", index=False)

    sensitivity = df[df["endpoint_tier"].eq("supplement") | df["representation_family"].isin(["UGA_geometry", "channel_KME", "COSMIC_NNLS_exposures", "mechanistic_control"])].copy()
    sensitivity = _add_display_columns(sensitivity)
    sensitivity.to_csv(supp_dir / "table_s2_sensitivity_analyses.csv", index=False)

    pd.DataFrame(
        [{"source_table": name, "rows": len(frame), "columns": len(frame.columns)} for name, frame in side_tables.items()]
    ).to_csv(supp_dir / "table_s0_source_inventory.csv", index=False)


def _source_label(endpoint: str) -> str:
    if endpoint == "damage_class":
        return "Kucab mutagen-treated clone mutation catalogues"
    if endpoint.startswith("hrd") or endpoint in {"HRD_Score", "eCARD", "HRD_TAI", "HRD_LST", "HRD_LOH", "PARPi7", "parpi7_binary"}:
        return "TCGA-BRCA HRD labels with MC3 mutation features"
    if endpoint in {"cancer_type_top10", "smoking_ever", "high_purity", "high_stage", "os_event"}:
        return "TCGA MC3 clinical endpoint panel"
    if "kmt2c" in endpoint.lower():
        return "TCGA MC3 LUAD driver endpoint"
    if endpoint in {"SBS", "DBS", "ID"}:
        return "PCAWG/COSMIC attribution recovery"
    return "Regenerated benchmark output"


def _label_definition(endpoint: str) -> str:
    labels = {
        "damage_class": "curated DNA damage class from Kucab treatment metadata",
        "HRD_Score": "continuous HRD score",
        "hrd_binary_33": "HRD-high versus HRD-low at threshold 33",
        "cancer_type_top10": "top-10 TCGA cancer type classification",
        "os_event": "overall-survival event indicator",
    }
    return labels.get(endpoint, "see regenerated source table and experiment manifest")


def _completion_grid(df: pd.DataFrame, *, endpoints: list[str], families: list[str], model_families: list[str] | None = None) -> pd.DataFrame:
    model_families = model_families or ["elastic_net", "XGBoost"]
    best = _best_scores(df, main_only=False, families=families)
    rows: list[dict[str, object]] = []
    for endpoint in endpoints:
        for model in model_families:
            for family in families:
                match = best[
                    best["endpoint"].eq(endpoint)
                    & best["model_family"].eq(model)
                    & best["representation_family"].eq(family)
                ]
                if len(match):
                    row = match.sort_values("primary_score", ascending=False).iloc[0].to_dict()
                    row.update({"status": "measured", "na_reason": ""})
                else:
                    row = {
                        "endpoint": endpoint,
                        "endpoint_tier": "main" if endpoint in MAIN_ENDPOINTS else "supplement",
                        "representation_family": family,
                        "model_family": model,
                        "metric": "",
                        "primary_score": np.nan,
                        "status": "not_applicable",
                        "na_reason": "no finalized OOF result for this endpoint/representation/model slot",
                    }
                rows.append(row)
    return pd.DataFrame(rows)


def _plot_grid(df: pd.DataFrame, *, endpoints: list[str], families: list[str], model_families: list[str] | None = None) -> pd.DataFrame:
    cols = [
        "endpoint",
        "endpoint_tier",
        "representation_family",
        "model_family",
        "metric",
        "primary_score",
        "status",
        "na_reason",
        "n_samples",
        "n_features",
        "folds",
        "repeats",
        "optuna_trials_completed",
        "cache_key",
        "oof_prediction_file",
    ]
    grid = _completion_grid(df, endpoints=endpoints, families=families, model_families=model_families)
    for col in cols:
        if col not in grid.columns:
            grid[col] = ""
    return grid.loc[:, cols]


def _valid_text(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text and text.lower() not in {"nan", "none", "na", "n/a"})


def _canonical_main_results(df: pd.DataFrame, tables_dir: Path, *, strict: bool) -> pd.DataFrame:
    work = df.copy()
    if work.empty:
        if strict:
            raise ValueError("No normalized rows available for canonical main-panel construction")
        return work
    work = work[
        work["endpoint"].isin(MAIN_ENDPOINTS)
        & work["representation_family"].isin(MAIN_REPRESENTATIONS)
        & work["model_family"].isin(MODEL_FAMILIES)
    ].copy()
    work["expected_source"] = work["representation_family"].map(CANONICAL_SOURCE_BY_FAMILY)
    work = work[work["experiment_id"].astype(str).eq(work["expected_source"].astype(str))]
    work["primary_score"] = pd.to_numeric(work["primary_score"], errors="coerce")
    work["folds"] = pd.to_numeric(work["folds"], errors="coerce")
    work["repeats"] = pd.to_numeric(work["repeats"], errors="coerce")
    work = work[np.isfinite(work["primary_score"]) & work["folds"].eq(5) & work["repeats"].eq(1)]
    work = work[work["oof_prediction_file"].map(_valid_text)]
    work = work[work["oof_prediction_file"].map(lambda name: (tables_dir / str(name)).exists())]
    sort_cols = ["endpoint", "representation_family", "model_family", "source_priority", "primary_score"]
    work = work.sort_values(sort_cols, ascending=[True, True, True, False, False])
    canonical = work.groupby(["endpoint", "representation_family", "model_family"], as_index=False).head(1).copy()
    canonical["status"] = "measured"
    canonical["canonical_source"] = canonical["experiment_id"]
    canonical["canonical_slot_id"] = (
        canonical["endpoint"].astype(str)
        + "|"
        + canonical["representation_family"].astype(str)
        + "|"
        + canonical["model_family"].astype(str)
    )
    expected = {
        (endpoint, family, model)
        for endpoint in MAIN_ENDPOINTS
        for family in MAIN_REPRESENTATIONS
        for model in MODEL_FAMILIES
    }
    observed = set(zip(canonical["endpoint"], canonical["representation_family"], canonical["model_family"]))
    missing = sorted(expected - observed)
    if strict and missing:
        detail = [
            {"endpoint": endpoint, "representation_family": family, "model_family": model}
            for endpoint, family, model in missing
        ]
        raise ValueError(f"Canonical main panel is missing OOF-backed measured rows: {json.dumps(detail, indent=2)}")
    return canonical.reset_index(drop=True)


def _read_prediction_file(tables_dir: Path, name: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if name not in cache:
        path = tables_dir / name
        cache[name] = pd.read_csv(path, low_memory=False)
    return cache[name]


def _oof_representation_aliases(representation: str, family: str) -> list[str]:
    aliases = [str(representation)]
    if family == "one_hot_event_KME":
        aliases.append("one_hot_event_kme")
    return sorted(set(aliases))


def _canonical_oof_predictions(canonical: pd.DataFrame, tables_dir: Path, *, strict: bool) -> pd.DataFrame:
    cache: dict[str, pd.DataFrame] = {}
    frames: list[pd.DataFrame] = []
    missing: list[dict[str, str]] = []
    for _, row in canonical.iterrows():
        oof_name = str(row.get("oof_prediction_file", ""))
        pred = _read_prediction_file(tables_dir, oof_name, cache)
        learner = str(row.get("model_label", "")).strip()
        if not learner or learner.lower() in {"nan", "none"}:
            learner = "linear" if str(row.get("model_family")) == "elastic_net" else "xgboost"
        aliases = _oof_representation_aliases(str(row["representation"]), str(row["representation_family"]))
        slot = pred[
            pred["endpoint"].astype(str).eq(str(row["endpoint"]))
            & pred["representation"].astype(str).isin(aliases)
            & pred["learner"].astype(str).eq(learner)
        ].copy()
        if slot.empty:
            missing.append(
                {
                    "endpoint": str(row["endpoint"]),
                    "representation": str(row["representation"]),
                    "representation_family": str(row["representation_family"]),
                    "model_family": str(row["model_family"]),
                    "oof_prediction_file": oof_name,
                }
            )
            continue
        slot["representation_family"] = row["representation_family"]
        slot["model_family"] = row["model_family"]
        slot["metric"] = row["metric"]
        slot["canonical_slot_id"] = row["canonical_slot_id"]
        frames.append(slot)
    if strict and missing:
        raise ValueError(f"Canonical rows lack matching OOF predictions: {json.dumps(missing, indent=2)}")
    return _add_display_columns(pd.concat(frames, ignore_index=True, sort=False)) if frames else pd.DataFrame()


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _prediction_matrix(merged: pd.DataFrame, suffix: str) -> np.ndarray:
    class_cols = []
    for col in merged.columns:
        match = re.match(rf"pred_class_(\d+)_{suffix}$", col)
        if match and pd.to_numeric(merged[col], errors="coerce").notna().any():
            class_cols.append((int(match.group(1)), col))
    class_cols = sorted(class_cols)
    if not class_cols:
        raise ValueError(f"No class probability columns found for suffix={suffix}")
    matrix = merged[[col for _, col in class_cols]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Class probability matrix contains NaN for suffix={suffix}")
    return matrix


def _comparison_arrays(oof: pd.DataFrame, candidate: pd.Series, baseline: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    cand = oof[
        oof["endpoint"].astype(str).eq(str(candidate["endpoint"]))
        & oof["representation_family"].astype(str).eq(str(candidate["representation_family"]))
        & oof["model_family"].astype(str).eq(str(candidate["model_family"]))
    ].copy()
    base = oof[
        oof["endpoint"].astype(str).eq(str(baseline["endpoint"]))
        & oof["representation_family"].astype(str).eq(str(baseline["representation_family"]))
        & oof["model_family"].astype(str).eq(str(baseline["model_family"]))
    ].copy()
    merged = cand.merge(base, on="sample", suffixes=("_candidate", "_baseline"))
    if merged.empty:
        raise ValueError("No paired OOF samples after merging candidate and baseline predictions")
    y = merged["true_value_candidate"].to_numpy()
    metric = str(candidate.get("metric") or baseline.get("metric") or "").lower()
    if metric == "spearman":
        return y.astype(float), merged["pred_value_candidate"].to_numpy(dtype=float), merged["pred_value_baseline"].to_numpy(dtype=float), metric
    if metric == "auroc":
        return y.astype(float), merged["pred_class_1_candidate"].to_numpy(dtype=float), merged["pred_class_1_baseline"].to_numpy(dtype=float), metric
    if metric == "macro_auroc":
        return y, _prediction_matrix(merged, "candidate"), _prediction_matrix(merged, "baseline"), metric
    raise ValueError(f"Unsupported canonical comparison metric: {metric}")


def _comparison_registry_frame() -> pd.DataFrame:
    rows = []
    for figure_id, comparison_name, candidate, baseline in MAIN_COMPARISONS:
        for endpoint in MAIN_ENDPOINTS:
            for model_family in MODEL_FAMILIES:
                rows.append(
                    {
                        "figure_id": figure_id,
                        "comparison_name": comparison_name,
                        "endpoint": endpoint,
                        "model_family": model_family,
                        "candidate_representation": candidate,
                        "baseline_representation": baseline,
                    }
                )
    return _add_display_columns(pd.DataFrame(rows))


def _compute_pairwise_tests(canonical: pd.DataFrame, oof: pd.DataFrame, *, bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, spec in _comparison_registry_frame().iterrows():
        candidate_match = canonical[
            canonical["endpoint"].eq(spec["endpoint"])
            & canonical["model_family"].eq(spec["model_family"])
            & canonical["representation_family"].eq(spec["candidate_representation"])
        ]
        baseline_match = canonical[
            canonical["endpoint"].eq(spec["endpoint"])
            & canonical["model_family"].eq(spec["model_family"])
            & canonical["representation_family"].eq(spec["baseline_representation"])
        ]
        row = spec.to_dict()
        if candidate_match.empty or baseline_match.empty:
            row.update({"test_status": "not_tested", "not_tested_reason": "missing canonical candidate or baseline"})
            rows.append(row)
            continue
        candidate = candidate_match.iloc[0]
        baseline = baseline_match.iloc[0]
        row.update(
            {
                "candidate_score": float(candidate["primary_score"]),
                "baseline_score": float(baseline["primary_score"]),
                "metric": candidate["metric"],
                "test_status": "tested",
                "not_tested_reason": "",
            }
        )
        try:
            y, cand_pred, base_pred, metric = _comparison_arrays(oof, candidate, baseline)
            if metric == "auroc" and len(np.unique(y)) == 2:
                result = paired_delong_auc(y, cand_pred, base_pred)
            else:
                result = paired_bootstrap_delta(
                    y,
                    cand_pred,
                    base_pred,
                    metric,
                    n_bootstrap=int(bootstrap),
                    seed=_stable_seed(spec["figure_id"], spec["comparison_name"], spec["endpoint"], spec["model_family"]),
                    stratify=metric == "macro_auroc",
                )
            row.update(
                {
                    "delta": result.delta,
                    "p_value": result.p_value,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "test_name": result.test_name,
                    "n_resamples": result.n_resamples,
                    "n_paired_samples": int(len(y)),
                }
            )
        except Exception as exc:
            row.update({"test_status": "not_tested", "not_tested_reason": str(exc), "delta": np.nan, "p_value": np.nan})
        rows.append(row)
    tests = pd.DataFrame(rows)
    tested = tests["test_status"].eq("tested") if "test_status" in tests.columns else pd.Series(False, index=tests.index)
    tests["q_value"] = np.nan
    tests.loc[tested, "q_value"] = bh_qvalues(pd.to_numeric(tests.loc[tested, "p_value"], errors="coerce").to_numpy())
    tests["q_value_figure"] = np.nan
    for figure_id, idx in tests[tested].groupby("figure_id").groups.items():
        tests.loc[idx, "q_value_figure"] = bh_qvalues(pd.to_numeric(tests.loc[idx, "p_value"], errors="coerce").to_numpy())
    q = pd.to_numeric(tests["q_value"], errors="coerce")
    tests["significance_label"] = np.select([q < 0.001, q < 0.01, q < 0.05, q < 0.10], ["***", "**", "*", "."], default="")
    return _add_display_columns(tests)


def _attach_plot_tests(frame: pd.DataFrame, tests: pd.DataFrame, figure_id: str) -> pd.DataFrame:
    out = frame.copy()
    out["comparison_id"] = ""
    out["baseline_representation"] = ""
    out["candidate_representation"] = ""
    out["delta"] = np.nan
    out["test_name"] = ""
    out["p_value"] = np.nan
    out["q_value"] = np.nan
    out["q_value_figure"] = np.nan
    out["significance_label"] = ""
    if tests.empty:
        return out
    subset = tests[tests["figure_id"].eq(figure_id) & tests["test_status"].eq("tested")].copy()
    for idx, row in out.iterrows():
        match = subset[
            subset["endpoint"].eq(row["endpoint"])
            & subset["model_family"].eq(row["model_family"])
            & subset["candidate_representation"].eq(row["representation_family"])
        ]
        if match.empty:
            continue
        test = match.iloc[0]
        out.loc[idx, "comparison_id"] = f"{test['figure_id']}:{test['comparison_name']}"
        out.loc[idx, "baseline_representation"] = test["baseline_representation"]
        out.loc[idx, "candidate_representation"] = test["candidate_representation"]
        for col in ["delta", "test_name", "p_value", "q_value", "q_value_figure", "significance_label"]:
            out.loc[idx, col] = test[col]
    return out


def _canonical_plot_grid(canonical: pd.DataFrame, *, endpoints: list[str], families: list[str], model_families: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for endpoint in endpoints:
        for model in model_families:
            for family in families:
                match = canonical[
                    canonical["endpoint"].eq(endpoint)
                    & canonical["model_family"].eq(model)
                    & canonical["representation_family"].eq(family)
                ]
                if not len(match):
                    rows.append(
                        {
                            "endpoint": endpoint,
                            "endpoint_tier": "main",
                            "representation_family": family,
                            "model_family": model,
                            "status": "missing",
                            "na_reason": "missing canonical measured OOF row",
                        }
                    )
                else:
                    row = match.iloc[0].to_dict()
                    row["status"] = "measured"
                    row["na_reason"] = ""
                    rows.append(row)
    return pd.DataFrame(rows)


def _calibration_plot_data(canonical: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    endpoints = ["damage_class", "hrd_binary_33", "cancer_type_top10", "os_event"]
    rows: list[dict[str, object]] = []
    for endpoint in endpoints:
        match = canonical[
            canonical["endpoint"].eq(endpoint)
            & canonical["representation_family"].eq("signatures_plus_MAF_stack")
            & canonical["model_family"].eq("XGBoost")
        ]
        if match.empty:
            continue
        slot = oof[
            oof["endpoint"].astype(str).eq(endpoint)
            & oof["representation_family"].astype(str).eq("signatures_plus_MAF_stack")
            & oof["model_family"].astype(str).eq("XGBoost")
        ].copy()
        if slot.empty:
            continue
        y = pd.to_numeric(slot["true_value"], errors="coerce")
        class_cols = sorted([col for col in slot.columns if re.match(r"pred_class_\d+$", col)], key=lambda c: int(c.split("_")[-1]))
        if not class_cols:
            continue
        probs = slot[class_cols].apply(pd.to_numeric, errors="coerce")
        if endpoint in {"hrd_binary_33", "os_event"} and "pred_class_1" in probs.columns:
            confidence = probs["pred_class_1"].to_numpy(dtype=float)
            observed = y.to_numpy(dtype=float)
            mode = "binary_positive_probability"
        else:
            confidence = probs.max(axis=1).to_numpy(dtype=float)
            predicted = probs.to_numpy(dtype=float).argmax(axis=1)
            observed = (predicted == y.to_numpy(dtype=int)).astype(float)
            mode = "multiclass_confidence_accuracy"
        valid = np.isfinite(confidence) & np.isfinite(observed)
        confidence = confidence[valid]
        observed = observed[valid]
        if len(confidence) < 10:
            continue
        bins = np.linspace(0.0, 1.0, 11)
        for bin_idx in range(10):
            lo, hi = bins[bin_idx], bins[bin_idx + 1]
            mask = (confidence >= lo) & (confidence <= hi if bin_idx == 9 else confidence < hi)
            if not np.any(mask):
                continue
            rows.append(
                {
                    "endpoint": endpoint,
                    "representation_family": "signatures_plus_MAF_stack",
                    "model_family": "XGBoost",
                    "calibration_mode": mode,
                    "bin": bin_idx + 1,
                    "bin_left": lo,
                    "bin_right": hi,
                    "mean_predicted": float(np.mean(confidence[mask])),
                    "observed_frequency": float(np.mean(observed[mask])),
                    "n_samples": int(np.sum(mask)),
                }
            )
    return pd.DataFrame(rows)


def _s3_measured_plot_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = ["UGA_geometry", "channel_KME", "COSMIC_NNLS_exposures", "mechanistic_control"]
    measured = df[
        df["endpoint_tier"].eq("supplement")
        & df["representation_family"].isin(families)
        & np.isfinite(pd.to_numeric(df["primary_score"], errors="coerce"))
    ].copy()
    if measured.empty:
        return measured, pd.DataFrame()
    measured["analysis_family"] = np.select(
        [
            measured["representation_family"].isin(["UGA_geometry", "channel_KME"]),
            measured["representation_family"].eq("COSMIC_NNLS_exposures"),
            measured["representation_family"].eq("mechanistic_control"),
        ],
        ["Alternative geometry", "COSMIC/NNLS exposure checks", "Mechanistic controls"],
        default="Other",
    )
    measured["display_model"] = np.where(
        measured["model_family"].isin(MODEL_FAMILIES),
        measured["model_family"],
        measured["model_label"].fillna("").astype(str),
    )
    measured["display_model"] = measured["display_model"].replace({"": "source model reported in table"})
    measured = measured.sort_values(["analysis_family", "endpoint", "representation_family", "primary_score"], ascending=[True, True, True, False])
    measured = measured.groupby(["analysis_family", "endpoint", "representation_family"], as_index=False).head(1)
    measured["status"] = "measured"
    measured["selection_policy"] = "best measured supplementary row for endpoint and representation family; selected model shown"
    endpoints = sorted(df.loc[df["endpoint_tier"].eq("supplement"), "endpoint"].dropna().astype(str).unique())
    observed = set(zip(measured["endpoint"].astype(str), measured["representation_family"].astype(str)))
    missing_rows = [
        {
            "endpoint": endpoint,
            "representation_family": family,
            "na_reason": "no measured supplementary result for this analysis family/endpoint",
        }
        for endpoint in endpoints
        for family in families
        if (endpoint, family) not in observed
    ]
    return measured, pd.DataFrame(missing_rows)


def _write_canonical_outputs(
    normalized: pd.DataFrame,
    tables_dir: Path,
    manuscript_dir: Path,
    *,
    strict: bool,
    bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canonical_dir = manuscript_dir / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical = _canonical_main_results(normalized, tables_dir, strict=strict)
    oof = _canonical_oof_predictions(canonical, tables_dir, strict=strict)
    tests = _compute_pairwise_tests(canonical, oof, bootstrap=bootstrap) if not canonical.empty and not oof.empty else pd.DataFrame()
    atomic_write_csv(canonical, canonical_dir / "main_panel_results.csv", index=False)
    atomic_write_csv(oof, canonical_dir / "main_panel_oof_predictions.csv", index=False)
    atomic_write_csv(tests, canonical_dir / "main_panel_pairwise_tests.csv", index=False)
    atomic_write_csv(_comparison_registry_frame(), canonical_dir / "main_panel_comparison_registry.csv", index=False)
    if strict:
        bad_tests = tests[tests["test_status"].ne("tested")] if not tests.empty and "test_status" in tests.columns else pd.DataFrame()
        if not bad_tests.empty:
            cols = ["figure_id", "comparison_name", "endpoint", "model_family", "candidate_representation", "baseline_representation", "not_tested_reason"]
            raise ValueError(f"Some main comparisons were not tested: {json.dumps(bad_tests.loc[:, cols].to_dict('records'), indent=2, default=str)}")
    return canonical, oof, tests


def _assert_main_plot_measured(name: str, frame: pd.DataFrame) -> None:
    if not re.match(r"figure_[2-5]_", name):
        return
    if "status" not in frame.columns:
        raise ValueError(f"{name} is missing status column for measured-only main figure validation")
    bad = frame[~frame["status"].astype(str).eq("measured")].copy()
    if bad.empty:
        return
    cols = [col for col in ["endpoint", "representation_family", "model_family", "status", "na_reason"] if col in bad.columns]
    detail = bad.loc[:, cols].to_dict("records")
    raise ValueError(f"{name} contains non-measured visible main slots: {json.dumps(detail, indent=2, default=str)}")


def _write_plot_data(
    df: pd.DataFrame,
    canonical: pd.DataFrame,
    canonical_oof: pd.DataFrame,
    tests: pd.DataFrame,
    manuscript_dir: Path,
    *,
    measured_only_main: bool = False,
) -> None:
    plot_dir = manuscript_dir / "plot_data"
    plot_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "figure_2_signature_baselines.csv": _attach_plot_tests(
            _canonical_plot_grid(canonical, endpoints=MAIN_ENDPOINTS, families=["burden_only", "signatures_only"], model_families=MODEL_FAMILIES),
            tests,
            "figure_2",
        ),
        "figure_3_geometry_vs_signatures.csv": _attach_plot_tests(
            _canonical_plot_grid(canonical, endpoints=MAIN_ENDPOINTS, families=["signatures_only", "one_hot_event_KME"], model_families=MODEL_FAMILIES),
            tests,
            "figure_3",
        ),
        "figure_4_maf_stack_vs_signatures.csv": _attach_plot_tests(
            _canonical_plot_grid(canonical, endpoints=MAIN_ENDPOINTS, families=["signatures_only", "MAF_stack_only", "signatures_plus_MAF_stack"], model_families=MODEL_FAMILIES),
            tests,
            "figure_4",
        ),
        "figure_5_cross_endpoint_summary.csv": _attach_plot_tests(
            _canonical_plot_grid(canonical, endpoints=MAIN_ENDPOINTS, families=MAIN_REPRESENTATIONS, model_families=MODEL_FAMILIES),
            tests,
            "figure_5",
        ),
        "figure_s2_calibration_thresholds.csv": _calibration_plot_data(canonical, canonical_oof),
    }
    s3_measured, s3_missing = _s3_measured_plot_data(df)
    files["figure_s3_feature_importance.csv"] = s3_measured
    completeness = []
    for name, frame in files.items():
        frame = _add_display_columns(frame.copy())
        if measured_only_main:
            _assert_main_plot_measured(name, frame)
        if name == "figure_s3_feature_importance.csv":
            bad = frame[~frame.get("status", pd.Series("measured", index=frame.index)).astype(str).eq("measured")]
            if measured_only_main and not bad.empty:
                raise ValueError("Figure S3 contains non-measured rows")
            if measured_only_main and "model_family" in frame.columns and frame["model_family"].astype(str).eq("best").any():
                raise ValueError("Figure S3 contains unlabeled model_family=best rows")
        atomic_write_csv(frame, plot_dir / name, index=False)
        temp = frame.copy()
        temp.insert(0, "plot_data_file", name)
        completeness.append(temp)
    benchmark_completeness = pd.concat(completeness, ignore_index=True, sort=False)
    atomic_write_csv(benchmark_completeness, plot_dir / "benchmark_completeness.csv", index=False)
    if not s3_missing.empty:
        atomic_write_csv(_add_display_columns(s3_missing), manuscript_dir / "supplement" / "table_s3_completeness_and_na_reasons.csv", index=False)
    atomic_write_json(
        plot_dir / "plot_data_manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "files": sorted(files),
            "benchmark_completeness": "benchmark_completeness.csv",
            "main_endpoints": MAIN_ENDPOINTS,
            "main_representations": MAIN_REPRESENTATIONS,
            "canonical_results": "canonical/main_panel_results.csv",
            "canonical_pairwise_tests": "canonical/main_panel_pairwise_tests.csv",
        },
    )


def _manuscript_csvs_for_label_mapping(manuscript_dir: Path) -> list[Path]:
    files: list[Path] = []
    for rel in ["tables", "supplement", "canonical", "plot_data"]:
        base = manuscript_dir / rel
        if not base.exists():
            continue
        for path in sorted(base.glob("*.csv")):
            if path.name == "table_4_label_mapping.csv":
                continue
            files.append(path)
    return files


def _write_label_mapping(manuscript_dir: Path) -> pd.DataFrame:
    tables_dir = manuscript_dir / "tables"
    public_dir = tables_dir / "publication"
    text_dir = manuscript_dir / "text"
    tables_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    usage: dict[tuple[str, str], dict[str, bool]] = {}
    for path in _manuscript_csvs_for_label_mapping(manuscript_dir):
        frame = _read_table(path)
        if frame is None or frame.empty:
            continue
        is_figure = "plot_data" in path.parts
        is_table = not is_figure
        for source, _, domain in DISPLAY_COLUMN_SPECS:
            if source not in frame.columns:
                continue
            for value in frame[source].dropna().astype(str).map(str.strip).unique():
                if not value or value.lower() in {"nan", "none"}:
                    continue
                slot = usage.setdefault((domain, value), {"used_in_figures": False, "used_in_tables": False})
                slot["used_in_figures"] = slot["used_in_figures"] or is_figure
                slot["used_in_tables"] = slot["used_in_tables"] or is_table

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for domain, values in sorted(LABEL_REGISTRY.items()):
        for machine_name, entry in sorted(values.items()):
            key = (domain, str(machine_name))
            seen.add(key)
            used = usage.get(key, {"used_in_figures": False, "used_in_tables": False})
            rows.append(
                {
                    "label_domain": domain,
                    "machine_name": machine_name,
                    "display_name": entry.get("display", machine_name),
                    "long_description": entry.get("description", entry.get("display", machine_name)),
                    "used_in_figures": bool(used["used_in_figures"]),
                    "used_in_tables": bool(used["used_in_tables"]),
                }
            )
    for (domain, machine_name), used in sorted(usage.items()):
        if (domain, machine_name) in seen:
            continue
        rows.append(
            {
                "label_domain": domain,
                "machine_name": machine_name,
                "display_name": _display_label(domain, machine_name),
                "long_description": _label_description(domain, machine_name),
                "used_in_figures": bool(used["used_in_figures"]),
                "used_in_tables": bool(used["used_in_tables"]),
            }
        )
    mapping = pd.DataFrame(rows).sort_values(["label_domain", "display_name", "machine_name"]).reset_index(drop=True)
    table4_path = tables_dir / "table_4_label_mapping.csv"
    mapping.to_csv(table4_path, index=False)
    _write_publication_copy(mapping, table4_path, public_dir)
    notes = (
        "# Label Mapping Notes\n\n"
        "Machine-readable identifiers are retained in the technical CSVs so results can be traced back to cached features, "
        "model slots, and source runner outputs. Manuscript-facing figures, captions, and publication-friendly tables use "
        "the paired display labels from `table_4_label_mapping.csv`. When a value is not explicitly listed in the registry, "
        "the pipeline falls back to a deterministic title-cased label and records that value in the mapping table.\n"
    )
    (text_dir / "label_mapping_notes.md").write_text(notes, encoding="utf-8")
    return mapping


def _run_d3_renderer(manuscript_dir: Path, *, strict: bool) -> None:
    node = shutil.which("node")
    if node is None:
        bundled_node = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "bin" / ("node.exe" if sys.platform.startswith("win") else "node")
        if bundled_node.exists():
            node = str(bundled_node)
    script = BUNDLE_ROOT / "src" / "visualization" / "render_manuscript_figures.mjs"
    d3_vendor = BUNDLE_ROOT / "src" / "legacy_source" / "cgr_validation" / "cgr_validation_results" / "research" / "experiments" / "supporting" / "2026_05_14_unified_uga_locked_manuscript_benchmark" / "code" / "d3.v7.min.js"
    if node is None:
        message = "D3 renderer requested but node executable was not found on PATH."
        if strict:
            raise RuntimeError(message)
        (manuscript_dir / "d3_render_manifest.json").write_text(json.dumps({"status": "skipped", "reason": message}, indent=2), encoding="utf-8")
        return
    env = dict(os.environ)
    bundled_node_modules = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "node_modules"
    if bundled_node_modules.exists():
        node_paths = [str(bundled_node_modules)]
        pnpm_flat = bundled_node_modules / ".pnpm" / "node_modules"
        if pnpm_flat.exists():
            node_paths.append(str(pnpm_flat))
        env["NODE_PATH"] = os.pathsep.join(node_paths + ([env["NODE_PATH"]] if env.get("NODE_PATH") else []))
    subprocess.run([node, str(script), str(manuscript_dir), str(d3_vendor)], cwd=str(BUNDLE_ROOT), check=True, env=env)


def _write_figures(df: pd.DataFrame, manuscript_dir: Path) -> None:
    figures_dir = manuscript_dir / "figures"
    supp_dir = manuscript_dir / "supplement"
    figures_dir.mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)
    _figure_1(figures_dir)
    _barplot(df, figures_dir / "figure_2_signature_baselines", "Figure 2. Burden and Signature Baselines", ["burden_only", "signatures_only"])
    _barplot(df, figures_dir / "figure_3_geometry_vs_signatures", "Figure 3. One-Hot Event KME vs Signatures", ["signatures_only", "one_hot_event_KME"])
    _barplot(df, figures_dir / "figure_4_maf_stack_vs_signatures", "Figure 4. MAF-Stack Biology Adds to Spectra", ["signatures_only", "MAF_stack_only", "signatures_plus_MAF_stack"])
    _heatmap(df, figures_dir / "figure_5_cross_endpoint_summary", "Figure 5. Main Endpoint Representation Summary", MAIN_REPRESENTATIONS, main_only=True)
    _write_text_panel(
        supp_dir / "figure_s1_representation_construction",
        "Supplementary Figure S1. Representation Construction Details",
        [
            (0.03, 0.58, "Spectra", "Count SBS96/DBS78/ID83 channels, normalize, append burden covariates where specified."),
            (0.38, 0.58, "One-Hot KME", "Fetch local GRCh37 sequence windows, one-hot encode context/payload, average RBF kernels over events."),
            (0.73, 0.58, "MAF Stack", "Aggregate genes, pathways, consequences, VAF summaries, and genomic locus/topography bins."),
            (0.03, 0.25, "UGA Variants", "Supplementary channel projections use the locked UGA channel atlas and ID payload encoder."),
            (0.38, 0.25, "Models", "Elastic-net linear and XGBoost trained with single 5-fold OOF CV."),
            (0.73, 0.25, "Outputs", "OOF metrics, feature dimensions, atlas status, and endpoint-tier labels are normalized into manuscript tables."),
        ],
    )
    _heatmap(df, supp_dir / "figure_s2_calibration_thresholds", "Supplementary Figure S2. OOF Metric Coverage for Calibration Review", MAIN_REPRESENTATIONS, main_only=True)
    _heatmap(df, supp_dir / "figure_s3_feature_importance", "Supplementary Figure S3. Supplementary Representation Sensitivity", ["UGA_geometry", "channel_KME", "COSMIC_NNLS_exposures", "mechanistic_control"], main_only=False)


def _validate_plot_data(manuscript_dir: Path) -> None:
    plot_dir = manuscript_dir / "plot_data"
    canonical_path = manuscript_dir / "canonical" / "main_panel_results.csv"
    tests_path = manuscript_dir / "canonical" / "main_panel_pairwise_tests.csv"
    if not canonical_path.exists() or not tests_path.exists():
        raise FileNotFoundError("Canonical main-panel results or pairwise tests are missing")
    canonical = pd.read_csv(canonical_path)
    tests = pd.read_csv(tests_path)
    for name in [
        "figure_2_signature_baselines.csv",
        "figure_3_geometry_vs_signatures.csv",
        "figure_4_maf_stack_vs_signatures.csv",
        "figure_5_cross_endpoint_summary.csv",
    ]:
        frame = pd.read_csv(plot_dir / name)
        bad = frame[~frame["status"].astype(str).eq("measured")]
        if not bad.empty:
            cols = ["endpoint", "representation_family", "model_family", "status", "na_reason"]
            raise ValueError(f"{name} contains non-measured main slots: {bad.loc[:, [c for c in cols if c in bad.columns]].to_dict('records')}")
        merged = frame.merge(
            canonical[["endpoint", "representation_family", "model_family", "primary_score"]],
            on=["endpoint", "representation_family", "model_family"],
            suffixes=("_plot", "_canonical"),
            how="left",
        )
        missing = merged["primary_score_canonical"].isna()
        if missing.any():
            raise ValueError(f"{name} contains rows not found in canonical main-panel results")
        delta = (pd.to_numeric(merged["primary_score_plot"], errors="coerce") - pd.to_numeric(merged["primary_score_canonical"], errors="coerce")).abs()
        if (delta > 1e-12).any():
            raise ValueError(f"{name} values differ from canonical main-panel results")
    fig5 = pd.read_csv(plot_dir / "figure_5_cross_endpoint_summary.csv")
    for name in ["figure_2_signature_baselines.csv", "figure_3_geometry_vs_signatures.csv", "figure_4_maf_stack_vs_signatures.csv"]:
        frame = pd.read_csv(plot_dir / name)
        merged = frame.merge(
            fig5[["endpoint", "representation_family", "model_family", "primary_score"]],
            on=["endpoint", "representation_family", "model_family"],
            suffixes=("_figure", "_figure5"),
            how="left",
        )
        delta = (pd.to_numeric(merged["primary_score_figure"], errors="coerce") - pd.to_numeric(merged["primary_score_figure5"], errors="coerce")).abs()
        if (delta > 1e-12).any():
            raise ValueError(f"{name} and Figure 5 disagree for at least one shared slot")
    s3 = pd.read_csv(plot_dir / "figure_s3_feature_importance.csv")
    if not s3.empty:
        if "status" in s3.columns and (~s3["status"].astype(str).eq("measured")).any():
            raise ValueError("Figure S3 contains non-measured rows")
        if "model_family" in s3.columns and s3["model_family"].astype(str).eq("best").any():
            raise ValueError("Figure S3 contains unlabeled model_family=best rows")
        if "display_model" not in s3.columns or s3["display_model"].astype(str).str.strip().eq("").any():
            raise ValueError("Figure S3 rows must include explicit display_model provenance")
    if tests.empty or tests["test_status"].ne("tested").any():
        raise ValueError("Main pairwise comparison tests are missing or incomplete")
    for col in ["p_value", "q_value"]:
        values = pd.to_numeric(tests[col], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Main pairwise tests contain missing {col}")


def _score_lookup(canonical: pd.DataFrame, endpoint: str, model_family: str, representation: str) -> float:
    rows = canonical[
        canonical["endpoint"].astype(str).eq(endpoint)
        & canonical["model_family"].astype(str).eq(model_family)
        & canonical["representation_family"].astype(str).eq(representation)
    ]
    if rows.empty:
        return float("nan")
    return float(pd.to_numeric(rows.iloc[0]["primary_score"], errors="coerce"))


def _fmt_score(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def _sig_count(
    tests: pd.DataFrame,
    comparison_name: str,
    *,
    positive: bool | None = None,
    model_family: str | None = None,
) -> int:
    rows = tests[tests["comparison_name"].astype(str).eq(comparison_name)].copy()
    if model_family is not None:
        rows = rows[rows["model_family"].astype(str).eq(model_family)]
    if positive is True:
        rows = rows[pd.to_numeric(rows["delta"], errors="coerce") > 0]
    elif positive is False:
        rows = rows[pd.to_numeric(rows["delta"], errors="coerce") < 0]
    rows = rows.drop_duplicates(
        subset=["comparison_name", "endpoint", "model_family", "candidate_representation", "baseline_representation"]
    )
    return int((pd.to_numeric(rows["q_value"], errors="coerce") < 0.05).sum())


def _sig_endpoints(tests: pd.DataFrame, comparison_name: str, *, model_family: str, positive: bool = True) -> list[str]:
    rows = tests[
        tests["comparison_name"].astype(str).eq(comparison_name)
        & tests["model_family"].astype(str).eq(model_family)
    ].copy()
    delta = pd.to_numeric(rows["delta"], errors="coerce")
    q_value = pd.to_numeric(rows["q_value"], errors="coerce")
    if positive:
        rows = rows[(delta > 0) & (q_value < 0.05)]
    else:
        rows = rows[(delta < 0) & (q_value < 0.05)]
    rows = rows.drop_duplicates(
        subset=["comparison_name", "endpoint", "model_family", "candidate_representation", "baseline_representation"]
    )
    endpoints = [str(x) for x in rows["endpoint"].tolist()]
    return [endpoint for endpoint in MAIN_ENDPOINTS if endpoint in endpoints] + sorted(
        endpoint for endpoint in endpoints if endpoint not in MAIN_ENDPOINTS
    )


def _write_manuscript_text(manuscript_dir: Path, canonical: pd.DataFrame, pairwise_tests: pd.DataFrame) -> None:
    text_dir = manuscript_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    figure_tests = pairwise_tests[pairwise_tests["figure_id"].astype(str).isin(["figure_2", "figure_3", "figure_4"])].copy()
    table_shapes: dict[str, tuple[int, int]] = {}
    for name in [
        "tables/table_1_datasets_endpoints.csv",
        "tables/table_2_full_performance_metrics.csv",
        "tables/table_3_hyperparameters_feature_dimensionality.csv",
        "supplement/table_s1_class_distribution_baselines.csv",
        "supplement/table_s2_sensitivity_analyses.csv",
        "supplement/table_s3_completeness_and_na_reasons.csv",
    ]:
        path = manuscript_dir / name
        if path.exists():
            frame = pd.read_csv(path)
            table_shapes[name] = frame.shape

    s = lambda endpoint, model, rep: _fmt_score(_score_lookup(canonical, endpoint, model, rep))
    sig_maf_over_maf = _sig_count(figure_tests, "sig_maf_vs_maf_stack", positive=True)
    kme_under = _sig_count(figure_tests, "one_hot_kme_vs_signatures", positive=False)
    kme_over = _sig_count(figure_tests, "one_hot_kme_vs_signatures", positive=True)
    sig_over_burden_xgb = _sig_count(figure_tests, "signatures_vs_burden", positive=True, model_family="XGBoost")
    sig_maf_over_sig_xgb = _sig_endpoints(figure_tests, "sig_maf_vs_signatures", model_family="XGBoost", positive=True)

    table2_rows = table_shapes.get("tables/table_2_full_performance_metrics.csv", (len(canonical), 0))[0]
    table_s2_rows = table_shapes.get("supplement/table_s2_sensitivity_analyses.csv", (0, 0))[0]
    burden = _display_label("representation_family", "burden_only")
    signatures = _display_label("representation_family", "signatures_only")
    one_hot_kme = _display_label("representation_family", "one_hot_event_KME")
    maf_stack = _display_label("representation_family", "MAF_stack_only")
    sig_maf = _display_label("representation_family", "signatures_plus_MAF_stack")
    xgboost = _display_label("model_family", "XGBoost")
    elastic_net = _display_label("model_family", "elastic_net")
    damage = _display_label("endpoint", "damage_class")
    hrd_score = _display_label("endpoint", "HRD_Score")
    hrd33 = _display_label("endpoint", "hrd_binary_33")
    cancer_type = _display_label("endpoint", "cancer_type_top10")
    os_event = _display_label("endpoint", "os_event")

    lines = [
        "# Manuscript Captions And Results Text",
        "",
        f"Author note: this text describes the generated manuscript outputs currently produced by the pipeline. "
        f"Figure 3 uses {one_hot_kme} as the main geometry comparison, while UGA and channel-KME variants are handled in Supplementary Figure S3.",
        "",
        "## Captions",
        "",
        "### Figure 1. Conceptual overview of mutation-catalogue representations.",
        "",
        "Each sample is represented as a catalogue of somatic mutation events, which can be transformed into complementary tabular feature families. "
        "Signature features summarize mutation spectra, geometry features encode sequence-context distributions from FASTA-derived windows or UGA/channel encodings, "
        "and MAF-stack features aggregate event-level biological annotations such as gene, locus, consequence, and burden summaries. "
        "Combined representations concatenate process-level spectra with event-level biology. These tabular representations are evaluated with elastic-net and XGBoost models "
        "across mechanistic, HRD, cancer-type, and clinical endpoints. MuAt and ATGC are shown as conceptually distinct end-to-end alternatives that operate directly on mutation event sets rather than on precomputed tabular summaries.",
        "",
        "### Figure 2. Signature baselines compared with mutational burden.",
        "",
        f"Single 5-fold out-of-fold performance is shown for {burden} and {signatures} across the five main endpoints and two model families. "
        f"Primary metrics are Spearman correlation for continuous {hrd_score}, AUROC for binary endpoints, and macro-AUROC for multiclass endpoints. "
        f"{signatures} improve over {burden} for {xgboost} on {damage} ({s('damage_class', 'XGBoost', 'signatures_only')} vs {s('damage_class', 'XGBoost', 'burden_only')}), "
        f"{hrd_score} ({s('HRD_Score', 'XGBoost', 'signatures_only')} vs {s('HRD_Score', 'XGBoost', 'burden_only')}), "
        f"{hrd33} ({s('hrd_binary_33', 'XGBoost', 'signatures_only')} vs {s('hrd_binary_33', 'XGBoost', 'burden_only')}), "
        f"{cancer_type} ({s('cancer_type_top10', 'XGBoost', 'signatures_only')} vs {s('cancer_type_top10', 'XGBoost', 'burden_only')}), "
        f"and {os_event} ({s('os_event', 'XGBoost', 'signatures_only')} vs {s('os_event', 'XGBoost', 'burden_only')}). "
        f"{elastic_net} shows the same pattern for Kucab and cancer type, but {burden} remains stronger for {elastic_net} {hrd33} and {os_event}.",
        "",
        "### Figure 3. Geometry-only one-hot event KME compared with signatures.",
        "",
        f"Performance of FASTA-derived {one_hot_kme} is compared with {signatures} using the same canonical out-of-fold results. "
        f"{one_hot_kme} is strongest for {damage}, where it slightly exceeds {signatures} for {xgboost} ({s('damage_class', 'XGBoost', 'one_hot_event_KME')} vs {s('damage_class', 'XGBoost', 'signatures_only')}) "
        f"and {elastic_net} ({s('damage_class', 'elastic_net', 'one_hot_event_KME')} vs {s('damage_class', 'elastic_net', 'signatures_only')}). "
        f"Across the remaining endpoints, {one_hot_kme} is generally competitive but does not consistently outperform {signatures}; it is lower than {signatures} for {xgboost} {cancer_type} "
        f"({s('cancer_type_top10', 'XGBoost', 'one_hot_event_KME')} vs {s('cancer_type_top10', 'XGBoost', 'signatures_only')}), "
        f"{hrd33} ({s('hrd_binary_33', 'XGBoost', 'one_hot_event_KME')} vs {s('hrd_binary_33', 'XGBoost', 'signatures_only')}), "
        f"and {os_event} ({s('os_event', 'XGBoost', 'one_hot_event_KME')} vs {s('os_event', 'XGBoost', 'signatures_only')}). "
        "This supports the conclusion that sequence-context geometry can be useful for mechanistic process classification, but is not a general replacement for spectra.",
        "",
        "### Figure 4. Event-level MAF-stack features and combined signature-plus-event representations.",
        "",
        f"This figure compares {signatures}, {maf_stack}, and {sig_maf} for each endpoint and model family. "
        f"{xgboost} with {sig_maf} gives the strongest results for {hrd_score} ({s('HRD_Score', 'XGBoost', 'signatures_plus_MAF_stack')}), "
        f"{hrd33} ({s('hrd_binary_33', 'XGBoost', 'signatures_plus_MAF_stack')}), "
        f"{cancer_type} ({s('cancer_type_top10', 'XGBoost', 'signatures_plus_MAF_stack')}), "
        f"and {os_event} ({s('os_event', 'XGBoost', 'signatures_plus_MAF_stack')}). "
        f"{maf_stack} alone improves over {signatures} for {xgboost} {cancer_type} ({s('cancer_type_top10', 'XGBoost', 'MAF_stack_only')} vs {s('cancer_type_top10', 'XGBoost', 'signatures_only')}), "
        f"but underperforms {signatures} for {damage} ({s('damage_class', 'XGBoost', 'MAF_stack_only')} vs {s('damage_class', 'XGBoost', 'signatures_only')}). "
        f"The combined representation improves over {maf_stack} in {sig_maf_over_maf} of 10 tested Figure 4 comparisons at q < 0.05, showing that process-level spectra and event-level biology are complementary.",
        "",
        "### Figure 5. Cross-endpoint summary of representation tradeoffs.",
        "",
        f"A canonical heatmap summarizes all five main representations across the five main endpoints and two model families. Values exactly match the canonical rows used in Figures 2-4. "
        f"{xgboost} dominates the best overall results, with {sig_maf} winning four of five endpoints: {hrd_score}, {hrd33}, {cancer_type}, and {os_event}. "
        f"The exception is {damage}, where {one_hot_kme} is best. No single representation wins everywhere, but the combined signature-plus-MAF representation is the most consistently strong practical default for tabular models.",
        "",
        "### Table 1. Datasets, endpoints, and evaluation design.",
        "",
        "This table summarizes the benchmark endpoints, sample counts, endpoint tiers, task types, data sources, label definitions, and splitting scheme. "
        f"The main panel includes {damage} (n = 259), {hrd_score} (n = 772), {hrd33} (n = 772), MC3 {cancer_type} (n = 5,462), and MC3 {os_event} (n = 10,139). "
        "Supplementary endpoints include additional HRD metrics and thresholds, MC3 clinical and driver endpoints, LUAD KMT2C status, and other validation tasks. "
        "Model-based results use a single 5-fold cross-validation design with aggregated out-of-fold predictions.",
        "",
        "### Table 2. Full performance metrics by endpoint, representation, and model.",
        "",
        f"This table is the numeric backbone for the manuscript figures and supplement, containing {table2_rows} benchmark rows. "
        "Rows report endpoint, representation, model family, primary metric, AUROC or macro-AUROC where applicable, AUPRC, accuracy/F1-style metrics where available, fold metadata, feature counts, runtime/provenance fields, and run identifiers. "
        "Main-figure values are drawn only from canonical measured rows with valid out-of-fold prediction provenance.",
        "",
        "### Table 3. Hyperparameters and feature dimensionality.",
        "",
        "This table summarizes representation dimensionality, model family, atlas status, folds/repeats, XGBoost estimator settings where applicable, linear model type, and tuning policy. "
        f"{burden} features are compact with a median of 3 features; {signatures} have a median of 182 features; {one_hot_kme} has 68-132 features depending on model family; "
        f"{maf_stack} features are substantially larger, with median dimensionality around 1,823 features; and {sig_maf} reaches a median of about 2,005 features. "
        "These values make the performance/complexity tradeoff explicit.",
        "",
        "### Table 4. Machine-readable to manuscript label mapping.",
        "",
        "This table maps internal endpoint, representation, model, metric, task, and atlas-status identifiers to manuscript-facing display labels. "
        "Machine-readable identifiers are retained in technical CSVs for reproducibility, while display labels are used in figures, captions, text, and publication-friendly table copies.",
        "",
        "### Supplementary Figure S1. Representation construction and reproducibility workflow.",
        "",
        "This schematic details how raw mutation catalogues are transformed into spectra, FASTA-window one-hot KME features, UGA/channel-KME variants, MAF-stack aggregates, and combined representations. "
        "It also illustrates the cache/checkpoint workflow used to make feature generation reusable and restartable. Context-derived features use GRCh37 FASTA windows where appropriate, while atlas-based UGA/channel features are treated as supplementary geometry variants.",
        "",
        "### Supplementary Figure S2. Calibration of selected main models.",
        "",
        f"Reliability curves are shown for classification endpoints using the selected {sig_maf} {xgboost} models. Calibration is evaluated from out-of-fold predictions for {damage}, {hrd33}, {cancer_type}, and {os_event}. "
        "These plots check whether the strongest models' predicted probabilities are broadly aligned with observed event frequencies rather than merely improving rank-based metrics.",
        "",
        "### Supplementary Figure S3. Supplementary measured representation panels.",
        "",
        "Measured supplementary results are shown for alternative geometry encodings, COSMIC/NNLS exposure checks, and mechanistic-control benchmarks. Visible marks are measured only and specify the model family or analysis family used. "
        "Unsupported or intentionally omitted combinations are excluded from the figure and documented separately in Supplementary Table S3.",
        "",
        "### Supplementary Table S1. Class distribution and baseline rates.",
        "",
        "This table reports endpoint-level sample counts, class distributions, prevalence, and naive baseline context. It provides the denominator and imbalance information needed to interpret AUROC, macro-AUROC, AUPRC, and accuracy-like metrics across binary, multiclass, and continuous tasks.",
        "",
        "### Supplementary Table S2. Sensitivity analyses and supplementary endpoint results.",
        "",
        f"This table reports {table_s2_rows} supplementary benchmark rows across additional endpoints, representations, and analysis families. It extends the main conclusions to extra HRD metrics/thresholds, additional MC3 clinical or driver endpoints, geometry variants, COSMIC/NNLS exposure checks, and mechanistic-control analyses.",
        "",
        "### Supplementary Table S3. Completeness and non-applicability registry.",
        "",
        "This table records combinations that are unsupported, intentionally omitted, or not applicable for supplementary analyses. Main manuscript figures contain no N/A rows; missing or unsupported supplementary combinations are documented here rather than rendered as visual placeholders.",
        "",
        "## Results Section Text",
        "",
        f"We first established the strength of conventional mutational spectra relative to a minimal burden baseline. In the canonical main panel, {xgboost} models using {signatures} outperformed {burden} across all five main endpoints: "
        f"{damage} improved from {s('damage_class', 'XGBoost', 'burden_only')} to {s('damage_class', 'XGBoost', 'signatures_only')} macro-AUROC, "
        f"{hrd_score} from {s('HRD_Score', 'XGBoost', 'burden_only')} to {s('HRD_Score', 'XGBoost', 'signatures_only')} Spearman correlation, "
        f"{hrd33} from {s('hrd_binary_33', 'XGBoost', 'burden_only')} to {s('hrd_binary_33', 'XGBoost', 'signatures_only')} AUROC, "
        f"{cancer_type} from {s('cancer_type_top10', 'XGBoost', 'burden_only')} to {s('cancer_type_top10', 'XGBoost', 'signatures_only')} macro-AUROC, "
        f"and {os_event} from {s('os_event', 'XGBoost', 'burden_only')} to {s('os_event', 'XGBoost', 'signatures_only')} AUROC. "
        f"These gains were statistically significant for {sig_over_burden_xgb} of 5 {xgboost} comparisons after FDR correction. {elastic_net} models showed the same qualitative gain for Kucab and cancer-type prediction, but not for every clinical or HRD endpoint; "
        f"in particular, {burden} exceeded {signatures} for {elastic_net} {hrd33} and {os_event}. Thus, signatures are a strong baseline, but their advantage depends on both endpoint and model class.",
        "",
        f"We next tested whether geometry-only sequence-context encodings provide a general replacement for spectra. The main geometry comparison used FASTA-derived {one_hot_kme}. "
        f"This representation performed best on {damage}, reaching {s('damage_class', 'XGBoost', 'one_hot_event_KME')} macro-AUROC with {xgboost} and {s('damage_class', 'elastic_net', 'one_hot_event_KME')} with {elastic_net}, slightly above the corresponding signature models. "
        f"However, {one_hot_kme} did not consistently outperform {signatures} elsewhere. For {xgboost}, it was lower than {signatures} on {cancer_type} ({s('cancer_type_top10', 'XGBoost', 'one_hot_event_KME')} vs {s('cancer_type_top10', 'XGBoost', 'signatures_only')}), "
        f"{hrd33} ({s('hrd_binary_33', 'XGBoost', 'one_hot_event_KME')} vs {s('hrd_binary_33', 'XGBoost', 'signatures_only')}), "
        f"and {os_event} ({s('os_event', 'XGBoost', 'one_hot_event_KME')} vs {s('os_event', 'XGBoost', 'signatures_only')}). "
        f"Pairwise testing showed significant underperformance of {one_hot_kme} versus {signatures} in {kme_under} of 10 main comparisons, with {kme_over} significant positive comparisons. "
        "These results support a conditional role for geometry encodings, especially in mechanistic damage-class prediction, rather than a universal replacement for spectra.",
        "",
        f"Event-level MAF-stack features provided a complementary source of biological information. {maf_stack} alone was particularly useful for cancer-type prediction with {xgboost}, improving over {signatures} from {s('cancer_type_top10', 'XGBoost', 'signatures_only')} to {s('cancer_type_top10', 'XGBoost', 'MAF_stack_only')} macro-AUROC. "
        f"However, it was not uniformly better than spectra: for {damage}, {maf_stack} alone was lower than {signatures} with {xgboost} ({s('damage_class', 'XGBoost', 'MAF_stack_only')} vs {s('damage_class', 'XGBoost', 'signatures_only')}), "
        "consistent with the idea that mechanistic mutagen exposure is better captured by sequence-context or spectral information than by event-level gene/locus aggregates alone.",
        "",
        f"The strongest overall pattern emerged from combining spectra with event-level MAF features. {sig_maf} was the best overall representation for four of five main endpoints with {xgboost}: "
        f"{hrd_score} reached {s('HRD_Score', 'XGBoost', 'signatures_plus_MAF_stack')} Spearman correlation, "
        f"{hrd33} reached {s('hrd_binary_33', 'XGBoost', 'signatures_plus_MAF_stack')} AUROC, "
        f"{cancer_type} reached {s('cancer_type_top10', 'XGBoost', 'signatures_plus_MAF_stack')} macro-AUROC, "
        f"and {os_event} reached {s('os_event', 'XGBoost', 'signatures_plus_MAF_stack')} AUROC. "
        f"The only main endpoint where it did not win was {damage}, where {one_hot_kme} was slightly higher. "
        f"In pairwise tests, {sig_maf} significantly improved over {maf_stack} alone in {sig_maf_over_maf} of 10 Figure 4 comparisons and significantly improved over {signatures} alone for {xgboost} {', '.join(_display_label('endpoint', endpoint) for endpoint in sig_maf_over_sig_xgb)}.",
        "",
        f"Taken together, the cross-endpoint summary shows that there is no single magic representation. Geometry-only features are useful for some mechanistic settings, {signatures} remain a strong and efficient baseline, and {maf_stack} features capture endpoint-relevant biology that spectra alone can miss. "
        f"Across the main panel, the most robust practical default for tabular models is {sig_maf}, particularly when paired with {xgboost}. "
        "The supplementary analyses document additional geometry variants, exposure checks, endpoint extensions, and completeness metadata without introducing N/A placeholders into the main manuscript figures.",
        "",
        "## Statistical Notes",
        "",
        f"Primary metrics are Spearman correlation for {hrd_score}, AUROC for binary endpoints, and macro-AUROC for multiclass endpoints. "
        "Statistical statements refer to canonical/main_panel_pairwise_tests.csv, using paired DeLong tests for binary AUROC and paired bootstrap tests for macro-AUROC or Spearman correlation.",
        "",
    ]
    text = "\n".join(lines)
    (text_dir / "manuscript_captions_and_results.md").write_text(text, encoding="utf-8")
    (text_dir / "figure_table_captions.md").write_text(text.split("## Results Section Text", 1)[0].rstrip() + "\n", encoding="utf-8")
    (text_dir / "results_section.md").write_text("## Results Section Text" + text.split("## Results Section Text", 1)[1], encoding="utf-8")


def _validate_d3_manifest(manuscript_dir: Path) -> None:
    manifest_path = manuscript_dir / "d3_render_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("D3 render manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qa = manifest.get("visual_qa") or []
    failures = [item for item in qa if item.get("status") != "passed"]
    if failures:
        raise ValueError(f"D3 visual QA failed: {json.dumps(failures, indent=2)}")


def _validate_display_columns(manuscript_dir: Path) -> None:
    errors: list[str] = []
    for path in _manuscript_csvs_for_label_mapping(manuscript_dir):
        frame = _read_table(path)
        if frame is None:
            continue
        for source, display, _ in DISPLAY_COLUMN_SPECS:
            if source in frame.columns and display not in frame.columns:
                errors.append(f"{path.relative_to(manuscript_dir)} has {source} but lacks {display}")
    if errors:
        raise ValueError("Missing manuscript display columns:\n" + "\n".join(errors))


def _validate_label_mapping_coverage(manuscript_dir: Path) -> None:
    table4 = manuscript_dir / "tables" / "table_4_label_mapping.csv"
    if not table4.exists():
        raise FileNotFoundError("Label mapping table is missing")
    mapping = pd.read_csv(table4)
    keys = set(zip(mapping["label_domain"].astype(str), mapping["machine_name"].astype(str)))
    missing: list[dict[str, str]] = []
    for path in _manuscript_csvs_for_label_mapping(manuscript_dir):
        frame = _read_table(path)
        if frame is None:
            continue
        for source, _, domain in DISPLAY_COLUMN_SPECS:
            if source not in frame.columns:
                continue
            for value in frame[source].dropna().astype(str).map(str.strip).unique():
                if not value or value.lower() in {"nan", "none"}:
                    continue
                if (domain, value) not in keys:
                    missing.append({"file": str(path.relative_to(manuscript_dir)), "domain": domain, "machine_name": value})
    if missing:
        raise ValueError(f"Label mapping table does not cover all manuscript labels: {json.dumps(missing[:50], indent=2)}")


def _validate_visible_svg_labels(manuscript_dir: Path) -> None:
    raw_values = [
        machine_name
        for domain, values in LABEL_REGISTRY.items()
        for machine_name in values
        if any(ch in machine_name for ch in "_")
    ]
    if not raw_values:
        return
    offenders: list[dict[str, str]] = []
    for folder in ["figures", "supplement"]:
        for path in sorted((manuscript_dir / folder).glob("*.svg")):
            svg = path.read_text(encoding="utf-8", errors="ignore")
            visible_text = " ".join(re.findall(r"<text[^>]*>(.*?)</text>", svg, flags=re.DOTALL))
            visible_text = re.sub(r"<[^>]+>", "", visible_text)
            for raw in raw_values:
                if raw in visible_text:
                    offenders.append({"file": str(path.relative_to(manuscript_dir)), "raw_label": raw})
    if offenders:
        raise ValueError(f"Rendered SVGs contain visible machine labels: {json.dumps(offenders[:50], indent=2)}")


def _validate(manuscript_dir: Path, df: pd.DataFrame, *, strict: bool) -> None:
    missing = [rel for rel in REQUIRED_MANUSCRIPT_FILES if not (manuscript_dir / rel).exists()]
    if (manuscript_dir / "d3_render_manifest.json").exists():
        for folder, stems in {
            "figures": [
                "figure_1_conceptual_overview",
                "figure_2_signature_baselines",
                "figure_3_geometry_vs_signatures",
                "figure_4_maf_stack_vs_signatures",
                "figure_5_cross_endpoint_summary",
            ],
            "supplement": [
                "figure_s1_representation_construction",
                "figure_s2_calibration_thresholds",
                "figure_s3_feature_importance",
            ],
        }.items():
            for stem in stems:
                for suffix in (".html", ".svg", ".pdf", ".png"):
                    rel = f"{folder}/{stem}{suffix}"
                    if not (manuscript_dir / rel).exists():
                        missing.append(rel)
    if missing:
        raise FileNotFoundError(f"Missing manuscript artifacts: {', '.join(missing)}")
    if strict:
        main = df[df["endpoint_tier"].eq("main")]
        missing_endpoints = [endpoint for endpoint in MAIN_ENDPOINTS if endpoint not in set(main["endpoint"])]
        if missing_endpoints:
            raise ValueError(f"Missing main endpoint rows: {', '.join(missing_endpoints)}")
        required_families = set(MAIN_REPRESENTATIONS)
        seen = set(main["representation_family"])
        missing_families = sorted(required_families - seen)
        if missing_families:
            raise ValueError(f"Missing main representation families: {', '.join(missing_families)}")
        if not np.isfinite(pd.to_numeric(df["primary_score"], errors="coerce")).any():
            raise ValueError("No finite primary scores found in normalized results")
        _validate_display_columns(manuscript_dir)
        _validate_label_mapping_coverage(manuscript_dir)
        _validate_plot_data(manuscript_dir)
        _validate_d3_manifest(manuscript_dir)
        _validate_visible_svg_labels(manuscript_dir)


def make_all_figures(*, settings: dict[str, Any] | None = None, paths: dict[str, Any] | None = None, strict: bool | None = None) -> None:
    settings = settings or {}
    if paths is None:
        paths_file = BUNDLE_ROOT / "config" / "paths.yaml"
        if not paths_file.exists():
            paths_file = BUNDLE_ROOT / "config" / "paths_example.yaml"
        paths = resolve_paths_map(load_yaml(paths_file))
    if strict is None:
        strict = bool((settings.get("outputs") or {}).get("strict_manuscript", False))
    tables_dir = Path(paths["workspace"]["results_tables_dir"])
    manuscript_dir = BUNDLE_ROOT / "results" / "manuscript"
    run_id = str(settings.get("_run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    endpoint_results = _load_endpoint_results(tables_dir, strict=strict)
    side_tables = _load_side_tables(tables_dir)
    normalized = _normalize(endpoint_results, run_id=run_id)
    if strict and normalized.empty:
        raise ValueError("No normalized endpoint rows were produced")
    canonical, canonical_oof, pairwise_tests = _write_canonical_outputs(
        normalized,
        tables_dir,
        manuscript_dir,
        strict=strict,
        bootstrap=int(settings.get("bootstrap", 200)),
    )
    table_df = pd.concat(
        [
            canonical,
            normalized[normalized["endpoint_tier"].eq("supplement")],
        ],
        ignore_index=True,
        sort=False,
    )
    _write_tables(table_df, side_tables, manuscript_dir)
    measured_only_main = bool(strict or (settings.get("outputs") or {}).get("strict_plot_completeness", False))
    _write_plot_data(normalized, canonical, canonical_oof, pairwise_tests, manuscript_dir, measured_only_main=measured_only_main)
    _write_label_mapping(manuscript_dir)
    _write_manuscript_text(manuscript_dir, canonical, pairwise_tests)
    renderer = str((settings.get("outputs") or {}).get("visualization_renderer", "matplotlib")).lower()
    if renderer == "d3":
        _run_d3_renderer(manuscript_dir, strict=strict)
    else:
        _write_figures(table_df, manuscript_dir)
    _validate(manuscript_dir, table_df, strict=strict)
    manifest = {
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "run_id": run_id,
        "strict": strict,
        "normalized_rows": int(len(normalized)),
        "canonical_main_rows": int(len(canonical)),
        "pairwise_tests": int(len(pairwise_tests)),
        "main_endpoint_rows": int(table_df["endpoint_tier"].eq("main").sum()) if not table_df.empty else 0,
        "main_representations": MAIN_REPRESENTATIONS,
        "main_endpoints": MAIN_ENDPOINTS,
        "required_files": REQUIRED_MANUSCRIPT_FILES,
        "manuscript_text_files": [
            "text/manuscript_captions_and_results.md",
            "text/figure_table_captions.md",
            "text/results_section.md",
            "text/label_mapping_notes.md",
        ],
    }
    (manuscript_dir / "manuscript_asset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment_settings.cpu.yaml")
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    settings = load_yaml(args.config)
    paths = resolve_paths_map(load_yaml(args.paths))
    make_all_figures(settings=settings, paths=paths, strict=bool(args.strict or (settings.get("outputs") or {}).get("strict_manuscript", False)))


if __name__ == "__main__":
    main()
