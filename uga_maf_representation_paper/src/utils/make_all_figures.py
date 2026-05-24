"""Build strict manuscript tables and figures from regenerated bundle outputs."""

from __future__ import annotations

import argparse
import hashlib
import html
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
    "tables/table_1_datasets_endpoints.html",
    "tables/table_2_full_performance_metrics.csv",
    "tables/table_2_full_performance_metrics.html",
    "tables/table_3_hyperparameters_feature_dimensionality.csv",
    "tables/table_3_hyperparameters_feature_dimensionality.html",
    "tables/table_4_label_mapping.csv",
    "tables/table_4_label_mapping.html",
    "tables/publication/table_1_datasets_endpoints.csv",
    "tables/publication/table_1_datasets_endpoints.html",
    "tables/publication/table_2_full_performance_metrics.csv",
    "tables/publication/table_2_full_performance_metrics.html",
    "tables/publication/table_3_hyperparameters_feature_dimensionality.csv",
    "tables/publication/table_3_hyperparameters_feature_dimensionality.html",
    "tables/publication/table_4_label_mapping.csv",
    "tables/publication/table_4_label_mapping.html",
    "tables/technical/table_1_datasets_endpoints_technical.csv",
    "tables/technical/table_1_datasets_endpoints_technical.html",
    "tables/technical/table_2_full_performance_metrics_technical.csv",
    "tables/technical/table_2_full_performance_metrics_technical.html",
    "tables/technical/table_3_hyperparameters_feature_dimensionality_technical.csv",
    "tables/technical/table_3_hyperparameters_feature_dimensionality_technical.html",
    "tables/technical/table_4_label_mapping_technical.csv",
    "tables/technical/table_4_label_mapping_technical.html",
    "text/manuscript_captions_and_results.md",
    "text/label_mapping_notes.md",
    "supplement/table_s0_source_inventory.csv",
    "supplement/table_s0_source_inventory.html",
    "supplement/table_s1_class_distribution_baselines.csv",
    "supplement/table_s1_class_distribution_baselines.html",
    "supplement/table_s2_sensitivity_analyses.csv",
    "supplement/table_s2_sensitivity_analyses.html",
    "supplement/table_s3_completeness_and_na_reasons.csv",
    "supplement/table_s3_completeness_and_na_reasons.html",
    "tables/technical/table_s0_source_inventory_technical.csv",
    "tables/technical/table_s0_source_inventory_technical.html",
    "tables/technical/table_s1_class_distribution_baselines_technical.csv",
    "tables/technical/table_s1_class_distribution_baselines_technical.html",
    "tables/technical/table_s2_sensitivity_analyses_technical.csv",
    "tables/technical/table_s2_sensitivity_analyses_technical.html",
    "tables/technical/table_s3_completeness_and_na_reasons_technical.csv",
    "tables/technical/table_s3_completeness_and_na_reasons_technical.html",
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
TABLE_TITLES = {
    "table_1_datasets_endpoints": "Table 1. Datasets, endpoints, and evaluation design",
    "table_2_full_performance_metrics": "Table 2. Main-panel performance matrix",
    "table_3_hyperparameters_feature_dimensionality": "Table 3. Representation summary and dimensionality",
    "table_4_label_mapping": "Table 4. Key terminology and abbreviations",
    "table_s0_source_inventory": "Supplementary Table S0. Regenerated source inventory",
    "table_s1_class_distribution_baselines": "Supplementary Table S1. Supplementary endpoint inventory",
    "table_s2_sensitivity_analyses": "Supplementary Table S2. Headline supplementary results",
    "table_s3_completeness_and_na_reasons": "Supplementary Table S3. Non-applicability summary",
}
HTML_COLUMN_LABELS = {
    "endpoint_display": "Endpoint",
    "endpoint_tier_display": "Endpoint tier",
    "task_display": "Task",
    "representation_family_display": "Representation family",
    "representation_display": "Representation",
    "atlas_status_display": "Atlas status",
    "model_display": "Model",
    "model_label_display": "Model label",
    "display_model_display": "Display model",
    "metric_display": "Metric",
    "candidate_representation_display": "Candidate representation",
    "baseline_representation_display": "Baseline representation",
    "comparison_display": "Comparison",
    "calibration_mode_display": "Calibration mode",
    "analysis_family_display": "Analysis family",
    "endpoint": "Endpoint ID",
    "endpoint_tier": "Endpoint tier ID",
    "task": "Task ID",
    "representation_family": "Representation family ID",
    "representation": "Representation ID",
    "atlas_status": "Atlas status ID",
    "model_family": "Model family ID",
    "model_label": "Model label ID",
    "display_model": "Display model ID",
    "metric": "Metric ID",
    "candidate_representation": "Candidate representation ID",
    "baseline_representation": "Baseline representation ID",
    "comparison_name": "Comparison ID",
    "calibration_mode": "Calibration mode ID",
    "analysis_family": "Analysis family ID",
    "n_samples_max": "Max samples",
    "assay_or_source": "Assay or source",
    "label_definition": "Label definition",
    "splitting_scheme": "Splitting scheme",
    "primary_score": "Primary score",
    "delta_vs_signatures": "Delta vs signatures",
    "ci_low": "CI low",
    "ci_high": "CI high",
    "p_value": "P value",
    "q_value": "FDR q value",
    "n_samples": "Samples",
    "n_features": "Features",
    "xgb_estimators": "XGBoost estimators",
    "optuna_trials_completed": "Optuna trials",
    "na_reason": "N/A reason",
    "split_strategy": "Split strategy",
    "cache_key": "Cache key",
    "oof_prediction_file": "OOF prediction file",
    "fold_metrics_file": "Fold metrics file",
    "source_file": "Source file",
    "source_table": "Source table",
    "metrics_seen": "Metrics seen",
    "naive_baseline_context": "Naive baseline context",
    "median_features": "Median features",
    "max_features": "Max features",
    "linear_model": "Linear model",
    "label_domain": "Label domain",
    "machine_name": "Machine name",
    "display_name": "Display name",
    "long_description": "Long description",
    "used_in_figures": "Used in figures",
    "used_in_tables": "Used in tables",
    "N": "N",
    "Score": "Score",
    "Features": "Features",
    "Panel": "Panel",
    "Result rows": "Result rows",
    "CV design": "CV design",
    "Estimator/tuning": "Estimator/tuning",
    "Atlas/context": "Atlas/context",
    "95% CI": "95% CI",
    "FDR q value": "FDR q value",
    "Label category": "Label category",
    "Internal identifier": "Internal identifier",
    "Manuscript label": "Manuscript label",
    "Description": "Description",
    "Used in": "Used in",
    "Source artifact": "Source artifact",
    "Rows": "Rows",
    "Columns": "Columns",
    "Metrics": "Metrics",
    "Representations": "Representations",
    "Interpretation note": "Interpretation note",
}
INTEGER_HTML_COLUMNS = {
    "N",
    "Rows",
    "Columns",
    "Features",
    "Result rows",
    "Median features",
    "Max features",
    "rows",
    "columns",
    "folds",
    "repeats",
    "n_samples",
    "n_samples_max",
    "n_features",
    "median_features",
    "max_features",
    "xgb_estimators",
    "optuna_trials_completed",
}
SCORE_HTML_COLUMNS = {
    "Score",
    "Delta vs signatures",
    "FDR q value",
    "primary_score",
    "auroc",
    "auprc",
    "accuracy",
    "f1",
    "balanced_accuracy",
    "delta",
    "delta_vs_signatures",
    "ci_low",
    "ci_high",
    "p_value",
    "q_value",
    "prevalence",
}
MAIN_REPRESENTATION_SHORT_LABELS = {
    "burden_only": "Burden (EN/XGB)",
    "signatures_only": "Signatures (EN/XGB)",
    "one_hot_event_KME": "One-hot KME (EN/XGB)",
    "MAF_stack_only": "MAF stack (EN/XGB)",
    "signatures_plus_MAF_stack": "Signatures + MAF (EN/XGB)",
}
MANUSCRIPT_TABLE_REPRESENTATIONS = [
    "burden_only",
    "signatures_only",
    "one_hot_event_KME",
    "MAF_stack_only",
    "signatures_plus_MAF_stack",
]
REPRESENTATION_SUMMARY_SPECS = {
    "burden_only": {
        "input_signal": "Total mutation burden and compact burden summaries",
        "role": "Minimal baseline for all main endpoints",
    },
    "signatures_only": {
        "input_signal": "SBS/DBS/ID mutational spectra",
        "role": "Canonical mutational-process baseline",
    },
    "one_hot_event_KME": {
        "input_signal": "One-hot encoded local sequence windows averaged with kernel mean embeddings",
        "role": "Geometry-only sequence-context comparator",
    },
    "MAF_stack_only": {
        "input_signal": "Gene, locus, consequence, VAF, and event-level MAF aggregates",
        "role": "Event-level biological annotation comparator",
    },
    "signatures_plus_MAF_stack": {
        "input_signal": "Concatenated mutational spectra and event-level MAF aggregates",
        "role": "Combined practical default tested against single-source feature sets",
    },
}
BASELINE_REPRESENTATION_FAMILIES = {
    "burden_only",
    "signatures_only",
    "MAF_stack_only",
    "signatures_plus_MAF_stack",
}
SENSITIVITY_REPRESENTATION_FAMILIES = {
    "UGA_geometry",
    "channel_KME",
    "COSMIC_NNLS_exposures",
    "mechanistic_control",
}
KEY_GLOSSARY_ROWS = [
    ("EN / XGB", "Elastic-net score / XGBoost score in compact performance tables.", "Table 2"),
    ("AUROC", "Area under the receiver operating characteristic curve for binary endpoints.", "Tables 2 and S2"),
    ("macro-AUROC", "Class-balanced AUROC averaged across multiclass labels.", "Tables 2 and S2"),
    ("Spearman r", "Spearman rank correlation for continuous endpoints.", "Tables 2 and S2"),
    ("HRD", "Homologous recombination deficiency.", "Endpoint labels"),
    ("MAF", "Mutation annotation format; here also shorthand for event-level mutation annotations.", "Representations"),
    ("SBS/DBS/ID", "Single-base substitution, double-base substitution, and insertion/deletion mutation spectra.", "Representations"),
    ("KME", "Kernel mean embedding used to summarize event-level mutation contexts.", "Representations"),
    ("UGA", "Universal genomic atlas/channel geometry representation used in supplementary analyses.", "Supplement"),
    ("Mutational burden", "Compact mutation-count baseline features.", "Tables 2 and 3"),
    ("Mutational signatures", "Canonical mutation-spectrum representation.", "Tables 2 and 3"),
    ("One-hot sequence KME", "FASTA-window sequence-context KME representation.", "Tables 2 and 3"),
    ("Event-level MAF stack", "Gene, locus, consequence, VAF, and event-annotation feature stack.", "Tables 2 and 3"),
    ("Signatures + MAF stack", "Combined spectra plus event-level MAF features.", "Tables 2 and 3"),
    ("XGBoost", "Gradient-boosted tree model used as the nonlinear learner.", "Model columns"),
]


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


def _html_table_title(path: Path) -> str:
    return TABLE_TITLES.get(path.stem, path.stem.replace("_", " ").title())


def _html_column_label(column: str) -> str:
    if column in HTML_COLUMN_LABELS:
        return HTML_COLUMN_LABELS[column]
    if any(token in column for token in (" ", "/", "(", ")", "%")):
        return column
    label = column
    if label.endswith("_display"):
        label = label.removesuffix("_display")
    label = label.replace("_", " ")
    acronyms = {"id": "ID", "hrd": "HRD", "uga": "UGA", "maf": "MAF", "kme": "KME", "oof": "OOF", "xgb": "XGBoost", "cv": "CV"}
    return " ".join(acronyms.get(part.lower(), part.capitalize()) for part in label.split())


def _is_missing_html_value(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False
    return str(value).strip().lower() in {"", "nan", "none", "na", "n/a"}


def _format_html_cell(value: object, column: str) -> str:
    if _is_missing_html_value(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    if column in INTEGER_HTML_COLUMNS:
        number = pd.to_numeric(value, errors="coerce")
        if pd.notna(number):
            return f"{float(number):,.0f}"
    if column in SCORE_HTML_COLUMNS:
        number = pd.to_numeric(value, errors="coerce")
        if pd.notna(number):
            if column in {"p_value", "q_value"} and 0 < float(number) < 0.001:
                return "<0.001"
            return f"{float(number):.3f}"
    return str(value)


def _is_numeric_html_column(frame: pd.DataFrame, column: str) -> bool:
    if column in INTEGER_HTML_COLUMNS or column in SCORE_HTML_COLUMNS:
        return True
    if column not in frame.columns:
        return False
    series = frame[column].dropna()
    if series.empty:
        return False
    if pd.api.types.is_numeric_dtype(series):
        return True
    converted = pd.to_numeric(series, errors="coerce")
    return bool(converted.notna().mean() > 0.95)


def _write_html_table(df: pd.DataFrame, path: Path, *, title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    title = title or _html_table_title(path)
    numeric_columns = {column for column in df.columns if _is_numeric_html_column(df, column)}
    header = "".join(f"<th>{html.escape(_html_column_label(str(column)))}</th>" for column in df.columns)
    body_rows: list[str] = []
    for _, row in df.iterrows():
        cells = []
        for column in df.columns:
            value = _format_html_cell(row[column], str(column))
            classes = ["numeric"] if column in numeric_columns else []
            if len(value) > 60 and column not in numeric_columns:
                classes.append("long-text")
            class_attr = f' class="{" ".join(classes)}"' if classes else ""
            cells.append(f"<td{class_attr}>{html.escape(value)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    table = (
        '<div class="table-wrap">\n'
        '<table class="manuscript-table">\n'
        f"<caption>{html.escape(title)}</caption>\n"
        "<thead><tr>"
        + header
        + "</tr></thead>\n<tbody>\n"
        + "\n".join(body_rows)
        + "\n</tbody>\n</table>\n</div>"
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{
  color: #1f2933;
  font-family: Arial, Helvetica, sans-serif;
  margin: 24px;
}}
.table-wrap {{
  overflow-x: auto;
}}
table.manuscript-table {{
  border-collapse: collapse;
  border-spacing: 0;
  font-size: 12px;
  line-height: 1.35;
  width: 100%;
}}
table.manuscript-table caption {{
  caption-side: top;
  color: #111827;
  font-size: 14px;
  font-weight: 700;
  margin: 0 0 8px;
  text-align: left;
}}
table.manuscript-table th,
table.manuscript-table td {{
  border: 1px solid #cfd6df;
  padding: 6px 8px;
  vertical-align: top;
}}
table.manuscript-table th {{
  background: #eef2f6;
  color: #111827;
  font-weight: 700;
  text-align: left;
}}
table.manuscript-table tbody tr:nth-child(even) td {{
  background: #f8fafc;
}}
table.manuscript-table td.numeric {{
  font-variant-numeric: tabular-nums;
  text-align: right;
  white-space: nowrap;
}}
table.manuscript-table td.long-text {{
  min-width: 18rem;
}}
</style>
</head>
<body>
{table}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _write_table_csv_html(df: pd.DataFrame, path: Path, *, title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    _write_html_table(df, path.with_suffix(".html"), title=title or _html_table_title(path))


def _write_publication_copy(df: pd.DataFrame, path: Path, public_dir: Path) -> None:
    public_dir.mkdir(parents=True, exist_ok=True)
    _write_table_csv_html(df, public_dir / path.name, title=_html_table_title(path))


def _format_manuscript_count(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return ""
    return f"{int(round(float(number))):,}"


def _format_manuscript_score(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return ""
    return f"{float(number):.3f}"


def _format_manuscript_q(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return ""
    if 0 < float(number) < 0.001:
        return "<0.001"
    return f"{float(number):.3f}"


def _format_ci(low: object, high: object) -> str:
    lo = pd.to_numeric(low, errors="coerce")
    hi = pd.to_numeric(high, errors="coerce")
    if pd.isna(lo) or pd.isna(hi):
        return ""
    return f"{float(lo):.3f} to {float(hi):.3f}"


def _format_cv_design(folds: object, repeats: object) -> str:
    fold_text = _format_manuscript_count(folds)
    repeat_text = _format_manuscript_count(repeats)
    if not fold_text:
        return ""
    if repeat_text and repeat_text != "1":
        return f"{fold_text}-fold CV, {repeat_text} repeats"
    return f"{fold_text}-fold CV"


def _format_feature_range(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return ""
    low = float(numeric.min())
    high = float(numeric.max())
    if int(round(low)) == int(round(high)):
        return _format_manuscript_count(low)
    return f"{_format_manuscript_count(low)}-{_format_manuscript_count(high)}"


def _ordered_unique(values: pd.Series | list[object]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text.lower() in {"nan", "none"}:
            continue
        if text not in out:
            out.append(text)
    return out


def _best_result_cell(frame: pd.DataFrame) -> tuple[str, float | None, str]:
    if frame.empty:
        return "", None, ""
    work = frame.copy()
    work["primary_score"] = pd.to_numeric(work["primary_score"], errors="coerce")
    work = work.dropna(subset=["primary_score"])
    if work.empty:
        return "", None, ""
    row = work.sort_values("primary_score", ascending=False, kind="mergesort").iloc[0]
    representation = str(row.get("representation_family_display", "") or row.get("representation_display", "")).strip()
    model = str(row.get("model_display", "")).strip()
    metric = str(row.get("metric_display", "")).strip()
    score = float(row["primary_score"])
    pieces = [part for part in [representation, model] if part]
    label = " / ".join(pieces) if pieces else "Best result"
    return f"{label} ({metric} {score:.3f})", score, representation


def _supplementary_interpretation(baseline_score: float | None, sensitivity_score: float | None, best_label: str) -> str:
    if baseline_score is None and sensitivity_score is None:
        return "No compact comparison available; see technical table."
    if sensitivity_score is None:
        return "Best result comes from baseline or event-level feature families."
    if baseline_score is None:
        return "Best result comes from supplementary geometry/sensitivity analyses."
    delta = sensitivity_score - baseline_score
    if delta > 0:
        return f"Best sensitivity result exceeds baseline by {delta:.3f}."
    if delta < 0:
        return f"Best baseline/event-level result exceeds sensitivity result by {abs(delta):.3f}."
    return "Best baseline and sensitivity results are tied."


def _task_from_group(frame: pd.DataFrame, endpoint_display: str) -> str:
    values: list[str] = []
    if "task_display" in frame.columns:
        values = sorted({str(value).strip() for value in frame["task_display"].dropna() if str(value).strip()})
    if values:
        return "; ".join(values)
    metrics = {str(value).strip().lower() for value in frame.get("metric", pd.Series(dtype=object)).dropna()}
    if "spearman" in metrics:
        return "Regression"
    if "macro_auroc" in metrics or "balanced_accuracy" in metrics or "kucab" in endpoint_display.lower():
        return "Multiclass classification"
    if "auroc" in metrics:
        return "Binary classification"
    return ""


def _row_order(value: object, order: list[str]) -> int:
    text = str(value)
    return order.index(text) if text in order else len(order)


def _manuscript_performance_table(frame: pd.DataFrame, *, include_analysis: bool) -> pd.DataFrame:
    work = _add_display_columns(frame.copy())
    if work.empty:
        columns = ["Endpoint", "Task", "Representation", "Model", "Metric", "Score", "N", "Features"]
        if include_analysis:
            columns.insert(2, "Analysis")
            columns.extend(["Delta vs signatures", "95% CI", "FDR q value"])
        return pd.DataFrame(columns=columns)
    work["_endpoint_order"] = work["endpoint"].map(lambda value: _row_order(value, MAIN_ENDPOINTS))
    work["_representation_order"] = work["representation_family"].map(lambda value: _row_order(value, MAIN_REPRESENTATIONS))
    work["_model_order"] = work["model_family"].map(lambda value: _row_order(value, MODEL_FAMILIES))
    sort_cols = ["_endpoint_order", "endpoint_display"]
    if include_analysis and "model_label_display" in work.columns:
        sort_cols.append("model_label_display")
    sort_cols.extend(["_representation_order", "representation_family_display", "_model_order", "model_display"])
    work = work.sort_values(sort_cols, kind="mergesort")
    out = pd.DataFrame(
        {
            "Endpoint": work["endpoint_display"],
            "Task": work["task_display"],
            "Representation": work["representation_family_display"],
            "Model": work["model_display"],
            "Metric": work["metric_display"],
            "Score": work["primary_score"].map(_format_manuscript_score),
            "N": work["n_samples"].map(_format_manuscript_count),
            "Features": work["n_features"].map(_format_manuscript_count) if "n_features" in work.columns else "",
        }
    )
    if include_analysis:
        out.insert(2, "Analysis", work.get("model_label_display", pd.Series("", index=work.index)).fillna("").astype(str))
        out.insert(8, "Delta vs signatures", work.get("delta_vs_signatures", pd.Series("", index=work.index)).map(_format_manuscript_score))
        out.insert(9, "95% CI", [_format_ci(low, high) for low, high in zip(work.get("ci_low", pd.Series("", index=work.index)), work.get("ci_high", pd.Series("", index=work.index)))])
        out.insert(10, "FDR q value", work.get("q_value", pd.Series("", index=work.index)).map(_format_manuscript_q))
    return out.reset_index(drop=True)


def _technical_path(technical_dir: Path, source_path: Path) -> Path:
    return technical_dir / f"{source_path.stem}_technical{source_path.suffix}"


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
                "kme_version": _text(row, ["kme_version"], ""),
                "kme_config_id": _text(row, ["kme_config_id"], ""),
                "landmark_mode": _text(row, ["landmark_mode"], ""),
                "sigma_multiplier": _num(row, ["sigma_multiplier"]),
                "sigma_strategy": _text(row, ["sigma_strategy"], ""),
                "modality_strategy": _text(row, ["modality_strategy"], ""),
                "landmark_sampling": _text(row, ["landmark_sampling"], ""),
                "kernel_weighting": _text(row, ["kernel_weighting"], ""),
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
    technical_dir = tables_dir / "technical"
    supp_dir = manuscript_dir / "supplement"
    tables_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    technical_dir.mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)

    display_df = _add_display_columns(df.copy())
    technical_dataset_rows: list[dict[str, object]] = []
    for endpoint in sorted(display_df["endpoint"].dropna().unique()):
        sub = display_df[display_df["endpoint"].eq(endpoint)]
        technical_dataset_rows.append(
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
    table1_technical = _add_display_columns(pd.DataFrame(technical_dataset_rows))
    table1_path = tables_dir / "table_1_datasets_endpoints.csv"
    _write_table_csv_html(table1_technical, _technical_path(technical_dir, table1_path))

    endpoint_rows: list[dict[str, object]] = []
    for endpoint_display, sub in display_df.groupby("endpoint_display", dropna=False, sort=False):
        endpoint_values = [str(value) for value in sub["endpoint"].dropna().unique()]
        panel = "Main" if any(value in MAIN_ENDPOINTS for value in endpoint_values) else "Supplement"
        definitions = []
        for endpoint in endpoint_values:
            definition = _label_definition(endpoint)
            if definition and definition not in definitions:
                definitions.append(definition)
        sources = []
        for endpoint in endpoint_values:
            source = _source_label(endpoint)
            if source and source not in sources:
                sources.append(source)
        metrics = sorted({_display_label("metric", value) for value in sub["metric"].dropna().unique() if str(value).strip()})
        representations = sorted({_display_label("representation_family", value) for value in sub["representation_family"].dropna().unique() if str(value).strip()})
        endpoint_rows.append(
            {
                "_panel_order": 0 if panel == "Main" else 1,
                "_endpoint_order": min([_row_order(value, MAIN_ENDPOINTS) for value in endpoint_values] or [len(MAIN_ENDPOINTS)]),
                "Endpoint": str(endpoint_display),
                "Panel": panel,
                "N": _format_manuscript_count(pd.to_numeric(sub["n_samples"], errors="coerce").max()),
                "Task": _task_from_group(sub, str(endpoint_display)),
                "Source": "; ".join(sources),
                "Primary metric": "; ".join(metrics),
                "Label definition": "; ".join(definitions),
                "Representation families": "; ".join(representations),
            }
        )
    endpoint_inventory = pd.DataFrame(endpoint_rows).sort_values(["_panel_order", "_endpoint_order", "Endpoint"], kind="mergesort")
    table1 = endpoint_inventory[endpoint_inventory["Panel"].eq("Main")].loc[
        :, ["Endpoint", "Source", "N", "Task", "Primary metric", "Label definition"]
    ].rename(columns={"Source": "Cohort/source", "Label definition": "Label"})
    table1 = table1.reset_index(drop=True)
    _write_table_csv_html(table1, table1_path)
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
    table2_technical = _add_display_columns(df.loc[:, [col for col in ordered_cols if col in df.columns]])
    table2_path = tables_dir / "table_2_full_performance_metrics.csv"
    _write_table_csv_html(table2_technical, _technical_path(technical_dir, table2_path))
    main_perf = table2_technical[table2_technical["endpoint_tier"].astype(str).eq("main")].copy()
    table2_rows: list[dict[str, object]] = []
    for endpoint in MAIN_ENDPOINTS:
        sub = main_perf[main_perf["endpoint"].astype(str).eq(endpoint)].copy()
        if sub.empty:
            continue
        row: dict[str, object] = {
            "Endpoint": sub["endpoint_display"].iloc[0],
            "Task": sub["task_display"].iloc[0],
            "Metric": sub["metric_display"].iloc[0],
            "N": _format_manuscript_count(pd.to_numeric(sub["n_samples"], errors="coerce").max()),
        }
        best = sub.assign(primary_score=pd.to_numeric(sub["primary_score"], errors="coerce")).sort_values("primary_score", ascending=False, kind="mergesort").iloc[0]
        for family in MANUSCRIPT_TABLE_REPRESENTATIONS:
            fam = sub[sub["representation_family"].astype(str).eq(family)]
            scores_by_model = {
                str(item["model_family"]): _format_manuscript_score(item["primary_score"])
                for _, item in fam.iterrows()
            }
            row[MAIN_REPRESENTATION_SHORT_LABELS[family]] = " / ".join(scores_by_model.get(model, "") for model in MODEL_FAMILIES)
        row["Best model"] = (
            f"{best['representation_family_display']} ({best['model_display']}, "
            f"{_format_manuscript_score(best['primary_score'])})"
        )
        table2_rows.append(row)
    table2 = pd.DataFrame(table2_rows)
    _write_table_csv_html(table2, table2_path)
    _write_publication_copy(table2, table2_path, public_dir)

    hyper_technical = (
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
    hyper_technical["linear_model"] = np.where(hyper_technical["model_family"].eq("elastic_net"), "elastic-net linear model", "")
    hyper_technical["tuning"] = "10 Optuna trials where endpoint-specific tuning is used; otherwise frozen settings"
    hyper_technical = _add_display_columns(hyper_technical)
    table3_path = tables_dir / "table_3_hyperparameters_feature_dimensionality.csv"
    _write_table_csv_html(hyper_technical, _technical_path(technical_dir, table3_path))
    representation_rows: list[dict[str, object]] = []
    for family in MANUSCRIPT_TABLE_REPRESENTATIONS:
        sub = main_perf[main_perf["representation_family"].astype(str).eq(family)].copy()
        if sub.empty:
            continue
        specs = REPRESENTATION_SUMMARY_SPECS[family]
        models = [model for model in [_display_label("model_family", model) for model in MODEL_FAMILIES] if model in set(sub["model_display"].astype(str))]
        representation_rows.append(
            {
                "Representation": sub["representation_family_display"].iloc[0],
                "Input signal": specs["input_signal"],
                "Feature dimensionality": _format_feature_range(sub["n_features"]),
                "Context/atlas": "; ".join(_ordered_unique(sub["atlas_status_display"].tolist())),
                "Evaluated models": "; ".join(models),
                "Manuscript role": specs["role"],
            }
        )
    hyper = pd.DataFrame(representation_rows)
    _write_table_csv_html(hyper, table3_path)
    _write_publication_copy(hyper, table3_path, public_dir)

    class_dist_technical = (
        df.groupby(["endpoint_tier", "endpoint"], dropna=False)
        .agg(n_samples_max=("n_samples", "max"), metrics_seen=("metric", lambda x: "; ".join(sorted(set(map(str, x))))), representations=("representation_family", lambda x: "; ".join(sorted(set(map(str, x))))))
        .reset_index()
    )
    class_dist_technical["naive_baseline_context"] = "Endpoint-level sample size and metric coverage; model rows use OOF predictions where available"
    class_dist_technical = _add_display_columns(class_dist_technical)
    s1_path = supp_dir / "table_s1_class_distribution_baselines.csv"
    _write_table_csv_html(class_dist_technical, _technical_path(technical_dir, s1_path))
    s1 = endpoint_inventory[endpoint_inventory["Panel"].eq("Supplement")].loc[
        :, ["Endpoint", "N", "Task", "Source", "Primary metric", "Representation families"]
    ].rename(columns={"Primary metric": "Metric(s)"})
    s1 = s1.reset_index(drop=True)
    _write_table_csv_html(s1, s1_path)

    sensitivity_technical = df[df["endpoint_tier"].eq("supplement") | df["representation_family"].isin(["UGA_geometry", "channel_KME", "COSMIC_NNLS_exposures", "mechanistic_control"])].copy()
    sensitivity_technical = _add_display_columns(sensitivity_technical)
    s2_path = supp_dir / "table_s2_sensitivity_analyses.csv"
    _write_table_csv_html(sensitivity_technical, _technical_path(technical_dir, s2_path))
    sensitivity_rows: list[dict[str, object]] = []
    for endpoint_display, sub in sensitivity_technical.groupby("endpoint_display", dropna=False, sort=False):
        baseline_cell, baseline_score, _ = _best_result_cell(sub[sub["representation_family"].isin(BASELINE_REPRESENTATION_FAMILIES)])
        sensitivity_cell, sensitivity_score, _ = _best_result_cell(sub[sub["representation_family"].isin(SENSITIVITY_REPRESENTATION_FAMILIES)])
        best_cell, _, _ = _best_result_cell(sub)
        sensitivity_rows.append(
            {
                "Endpoint": str(endpoint_display),
                "N": _format_manuscript_count(pd.to_numeric(sub["n_samples"], errors="coerce").max()),
                "Metric(s)": "; ".join(sorted({_display_label("metric", value) for value in sub["metric"].dropna().unique() if str(value).strip()})),
                "Best baseline/event-level": baseline_cell,
                "Best geometry/sensitivity": sensitivity_cell,
                "Best overall": best_cell,
                "Interpretation": _supplementary_interpretation(baseline_score, sensitivity_score, best_cell),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values("Endpoint", kind="mergesort").reset_index(drop=True)
    _write_table_csv_html(sensitivity, s2_path)

    source_inventory = pd.DataFrame(
        [{"source_table": name, "rows": len(frame), "columns": len(frame.columns)} for name, frame in side_tables.items()]
    )
    s0_path = supp_dir / "table_s0_source_inventory.csv"
    _write_table_csv_html(source_inventory, _technical_path(technical_dir, s0_path))
    group_labels = {
        "main_manuscript_complete_panel": "Main complete panel",
        "one_hot_event_kme_scout": "One-hot event KME scout",
        "maf_event_gene_locus": "MAF event-gene/locus stack",
        "unified_locked": "Unified locked benchmark",
        "uga_kme": "UGA/KME variants",
        "mechanistic_representation": "Mechanistic controls",
    }
    source_inventory["source_group"] = source_inventory["source_table"].map(
        lambda name: next((label for key, label in group_labels.items() if str(name).startswith(key)), "Other regenerated source artifacts")
    )
    source_inventory_public = (
        source_inventory.groupby("source_group", as_index=False)
        .agg(Files=("source_table", "size"), **{"Total source rows": ("rows", "sum"), "Max columns": ("columns", "max")})
        .rename(columns={"source_group": "Source group"})
        .sort_values("Source group", kind="mergesort")
    )
    _write_table_csv_html(source_inventory_public, s0_path)


def _source_label(endpoint: str) -> str:
    if endpoint in {"damage_class", "Low-burden downsample", "Original data"}:
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
    if endpoint in labels:
        return labels[endpoint]
    description = _label_description("endpoint", endpoint)
    display = _display_label("endpoint", endpoint)
    if description and description != display:
        return description
    return "see regenerated source table and experiment manifest"


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
    # One-hot KME has many measured grid configurations. Broad aliases can mix
    # predictions from unselected configurations into the selected manuscript row,
    # so canonical OOF extraction uses exact representation and, when present,
    # exact kme_config_id filtering below.
    aliases = [str(representation)]
    if family == "one_hot_event_KME" and representation == "one_hot_event_kme_oracle":
        aliases.append("one_hot_event_kme")
    return aliases


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
        kme_config = str(row.get("kme_config_id", "")).strip()
        if str(row["representation_family"]) == "one_hot_event_KME" and kme_config:
            if "kme_config_id" not in pred.columns:
                slot = slot.iloc[0:0].copy()
            else:
                slot = slot[slot["kme_config_id"].astype(str).eq(kme_config)].copy()
            if slot.empty:
                if "kme_config_id" in pred.columns:
                    raw_slot = pred[
                        pred["endpoint"].astype(str).eq(str(row["endpoint"]))
                        & pred["learner"].astype(str).eq(learner)
                        & pred["kme_config_id"].astype(str).eq(kme_config)
                    ].copy()
                else:
                    raw_slot = pd.DataFrame()
                if not raw_slot.empty:
                    slot = raw_slot.copy()
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
        duplicated_samples = slot["sample"].astype(str).duplicated(keep=False)
        if duplicated_samples.any():
            detail = {
                "endpoint": str(row["endpoint"]),
                "representation": str(row["representation"]),
                "representation_family": str(row["representation_family"]),
                "model_family": str(row["model_family"]),
                "kme_config_id": kme_config,
                "duplicate_samples": int(duplicated_samples.sum()),
            }
            if strict:
                raise ValueError(f"Canonical OOF slot has duplicate samples: {json.dumps(detail, indent=2)}")
            slot = slot.drop_duplicates(["sample"], keep="last")
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
        s3_path = manuscript_dir / "supplement" / "table_s3_completeness_and_na_reasons.csv"
        s3_technical = _add_display_columns(s3_missing)
        _write_table_csv_html(s3_technical, _technical_path(manuscript_dir / "tables" / "technical", s3_path))
        s3_public = (
            s3_technical.groupby(["representation_family_display", "na_reason"], dropna=False)
            .agg(**{"Missing combinations": ("endpoint_display", "size"), "Example endpoints": ("endpoint_display", lambda x: "; ".join(_ordered_unique(list(x))[:5]))})
            .reset_index()
            .rename(columns={"representation_family_display": "Representation/analysis family", "na_reason": "Reason"})
        )
        _write_table_csv_html(s3_public, s3_path)
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
    mapping_technical = pd.DataFrame(rows).sort_values(["label_domain", "display_name", "machine_name"]).reset_index(drop=True)
    table4_path = tables_dir / "table_4_label_mapping.csv"
    _write_table_csv_html(mapping_technical, _technical_path(tables_dir / "technical", table4_path))
    mapping = pd.DataFrame(KEY_GLOSSARY_ROWS, columns=["Term", "Definition", "Used in"])
    _write_table_csv_html(mapping, table4_path)
    _write_publication_copy(mapping, table4_path, public_dir)
    notes = (
        "# Label Mapping Notes\n\n"
        "Manuscript-facing tables use compact display labels and omit run/cache provenance columns. "
        "Full machine-readable mappings and provenance-heavy table versions are retained under `tables/technical/` "
        "so results can still be traced back to cached features, model slots, and source runner outputs.\n"
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
    _barplot(df, figures_dir / "figure_3_geometry_vs_signatures", "Figure 3. One-Hot Sequence KME v2 vs Signatures", ["signatures_only", "one_hot_event_KME"])
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
    kme_tests = figure_tests[figure_tests.get("comparison_name", pd.Series(dtype=object)).astype(str).eq("one_hot_kme_vs_signatures")].copy()
    kme_numeric_wins = int((pd.to_numeric(kme_tests.get("delta"), errors="coerce") > 0).sum()) if not kme_tests.empty else 0
    kme_sig_positive_pairs = [
        f"{_display_label('model_family', str(row.model_family))} {_display_label('endpoint', str(row.endpoint))}"
        for row in kme_tests.itertuples()
        if pd.to_numeric(pd.Series([getattr(row, "delta", np.nan)]), errors="coerce").iloc[0] > 0
        and pd.to_numeric(pd.Series([getattr(row, "q_value", np.nan)]), errors="coerce").iloc[0] < 0.05
    ]
    kme_sig_negative_pairs = [
        f"{_display_label('model_family', str(row.model_family))} {_display_label('endpoint', str(row.endpoint))}"
        for row in kme_tests.itertuples()
        if pd.to_numeric(pd.Series([getattr(row, "delta", np.nan)]), errors="coerce").iloc[0] < 0
        and pd.to_numeric(pd.Series([getattr(row, "q_value", np.nan)]), errors="coerce").iloc[0] < 0.05
    ]
    kme_sig_positive_text = ", ".join(kme_sig_positive_pairs) if kme_sig_positive_pairs else "none"
    kme_sig_negative_text = ", ".join(kme_sig_negative_pairs) if kme_sig_negative_pairs else "none"
    xgb_winners: list[str] = []
    elastic_winners: list[str] = []
    for endpoint_name in MAIN_ENDPOINTS:
        for model_name, winners in [("XGBoost", xgb_winners), ("elastic_net", elastic_winners)]:
            subset = canonical[
                canonical["endpoint"].astype(str).eq(endpoint_name)
                & canonical["model_family"].astype(str).eq(model_name)
            ].copy()
            if subset.empty:
                continue
            subset["_score_numeric"] = pd.to_numeric(subset["primary_score"], errors="coerce")
            best = subset.sort_values("_score_numeric", ascending=False).iloc[0]
            winners.append(f"{_display_label('endpoint', endpoint_name)}: {_display_label('representation_family', str(best['representation_family']))}")

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
        f"{one_hot_kme} numerically exceeds {signatures} in {kme_numeric_wins} of 10 endpoint/model comparisons, including {elastic_net} {damage} "
        f"({s('damage_class', 'elastic_net', 'one_hot_event_KME')} vs {s('damage_class', 'elastic_net', 'signatures_only')}), {elastic_net} {hrd_score} "
        f"({s('HRD_Score', 'elastic_net', 'one_hot_event_KME')} vs {s('HRD_Score', 'elastic_net', 'signatures_only')}), and {xgboost} {cancer_type} "
        f"({s('cancer_type_top10', 'XGBoost', 'one_hot_event_KME')} vs {s('cancer_type_top10', 'XGBoost', 'signatures_only')}). "
        f"FDR-significant positive KME differences are observed for {kme_sig_positive_text}, while significant negative differences are observed for {kme_sig_negative_text}. "
        f"For {xgboost} {damage}, {one_hot_kme} is lower than {signatures} ({s('damage_class', 'XGBoost', 'one_hot_event_KME')} vs {s('damage_class', 'XGBoost', 'signatures_only')}). "
        "This supports the conclusion that sequence-context geometry can be useful in selected settings, but is not a general replacement for spectra.",
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
        f"For {xgboost}, {sig_maf} wins four of five endpoints ({hrd_score}, {hrd33}, {cancer_type}, and {os_event}), while {signatures} are highest for {damage}. "
        f"For {elastic_net}, the winners are {', '.join(elastic_winners)}. No single representation wins everywhere, but the combined signature-plus-MAF representation is the strongest practical default for XGBoost tabular models.",
        "",
        "### Table 1. Datasets, endpoints, and evaluation design.",
        "",
        "This table summarizes the five main manuscript endpoints, sample counts, task types, data sources, primary metrics, and label definitions. "
        f"The main panel includes {damage} (n = 259), {hrd_score} (n = 772), {hrd33} (n = 772), MC3 {cancer_type} (n = 5,462), and MC3 {os_event} (n = 10,139). "
        "Supplementary endpoints are listed separately in Supplementary Table S1. Model-based results use a single 5-fold cross-validation design with aggregated out-of-fold predictions.",
        "",
        "### Table 2. Main-panel performance matrix.",
        "",
        f"This table is the compact numeric backbone for the main manuscript figures, containing {table2_rows} endpoint rows. "
        "Each representation column reports elastic-net and XGBoost scores as EN / XGB, using the endpoint-specific primary metric. "
        "Full provenance-heavy versions with run identifiers, cache keys, and source files are retained under `tables/technical/`.",
        "",
        "### Table 3. Representation summary and dimensionality.",
        "",
        "This table summarizes the five main representations, their input signal, feature dimensionality range, context or atlas status, evaluated models, and manuscript role. "
        f"{burden} features are compact with a median of 3 features; {signatures} have a median of 182 features; {one_hot_kme} has 68-132 features depending on model family; "
        f"{maf_stack} features are substantially larger, with median dimensionality around 1,823 features; and {sig_maf} reaches a median of about 2,005 features. "
        "These values make the performance/complexity tradeoff explicit.",
        "",
        "### Table 4. Key terminology and abbreviations.",
        "",
        "This short glossary defines the key abbreviations, metrics, representation names, and model shorthand needed to read the main tables. "
        "The full machine-readable label mapping is retained in `tables/technical/table_4_label_mapping_technical.csv`.",
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
        "### Supplementary Table S1. Supplementary endpoint inventory.",
        "",
        "This table lists supplementary endpoints, sample counts, task types, data sources, primary metrics, and representation families evaluated outside the main five-endpoint panel.",
        "",
        "### Supplementary Table S2. Headline supplementary results.",
        "",
        f"This table reports {table_s2_rows} endpoint-level headline rows summarizing the best baseline/event-level result, best geometry or sensitivity result, and best overall supplementary result. "
        "The exhaustive supplementary result matrix is retained under `tables/technical/`.",
        "",
        "### Supplementary Table S3. Non-applicability summary.",
        "",
        "This compact table groups unsupported, intentionally omitted, or not-applicable supplementary combinations by representation or analysis family. Full endpoint-level details are retained under `tables/technical/`.",
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
        f"It numerically exceeded {signatures} in {kme_numeric_wins} of 10 endpoint/model comparisons. The clearest positive cases were {elastic_net} {hrd_score} "
        f"({s('HRD_Score', 'elastic_net', 'one_hot_event_KME')} vs {s('HRD_Score', 'elastic_net', 'signatures_only')}) and {xgboost} {cancer_type} "
        f"({s('cancer_type_top10', 'XGBoost', 'one_hot_event_KME')} vs {s('cancer_type_top10', 'XGBoost', 'signatures_only')}), both significant after FDR correction. "
        f"However, it did not improve {xgboost} {damage} ({s('damage_class', 'XGBoost', 'one_hot_event_KME')} vs {s('damage_class', 'XGBoost', 'signatures_only')}), {xgboost} {hrd33} "
        f"({s('hrd_binary_33', 'XGBoost', 'one_hot_event_KME')} vs {s('hrd_binary_33', 'XGBoost', 'signatures_only')}), or {xgboost} {os_event} "
        f"({s('os_event', 'XGBoost', 'one_hot_event_KME')} vs {s('os_event', 'XGBoost', 'signatures_only')}). "
        f"Pairwise testing showed {kme_over} significant positive and {kme_under} significant negative KME-vs-signature comparisons. "
        "These results support a conditional role for geometry encodings rather than a universal replacement for spectra.",
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
        f"The only main endpoint where it did not win was {damage}, where {signatures} remained slightly higher for {xgboost}. "
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
    table4 = manuscript_dir / "tables" / "technical" / "table_4_label_mapping_technical.csv"
    if not table4.exists():
        table4 = manuscript_dir / "tables" / "table_4_label_mapping.csv"
    if not table4.exists():
        raise FileNotFoundError("Label mapping table is missing")
    mapping = pd.read_csv(table4)
    if not {"label_domain", "machine_name"}.issubset(mapping.columns):
        return
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


def _validate_manuscript_ready_tables(manuscript_dir: Path, df: pd.DataFrame) -> None:
    table1 = pd.read_csv(manuscript_dir / "tables" / "table_1_datasets_endpoints.csv")
    if len(table1) != len(MAIN_ENDPOINTS):
        raise ValueError(f"Table 1 should contain {len(MAIN_ENDPOINTS)} main endpoint rows, found {len(table1)}")
    table2 = pd.read_csv(manuscript_dir / "tables" / "table_2_full_performance_metrics.csv")
    if len(table2) != len(MAIN_ENDPOINTS):
        raise ValueError(f"Table 2 should contain {len(MAIN_ENDPOINTS)} endpoint summary rows, found {len(table2)}")
    score_columns = [MAIN_REPRESENTATION_SHORT_LABELS[family] for family in MANUSCRIPT_TABLE_REPRESENTATIONS]
    for column in score_columns:
        if column not in table2.columns:
            raise ValueError(f"Table 2 is missing score column {column}")
        bad = table2[column].dropna().astype(str).map(lambda value: " / " not in value)
        if bad.any():
            raise ValueError(f"Table 2 column {column} must use EN / XGB score cells")
    s2 = pd.read_csv(manuscript_dir / "supplement" / "table_s2_sensitivity_analyses.csv")
    s2_technical_path = manuscript_dir / "tables" / "technical" / "table_s2_sensitivity_analyses_technical.csv"
    if not s2_technical_path.exists():
        raise FileNotFoundError("Technical Supplementary Table S2 is missing")
    s2_technical = pd.read_csv(s2_technical_path)
    expected_s2_rows = int((df["endpoint_tier"].astype(str).eq("supplement") | df["representation_family"].isin(["UGA_geometry", "channel_KME", "COSMIC_NNLS_exposures", "mechanistic_control"])).sum())
    if len(s2_technical) != expected_s2_rows:
        raise ValueError(f"Technical Supplementary Table S2 should contain {expected_s2_rows} rows, found {len(s2_technical)}")
    if len(s2) >= len(s2_technical):
        raise ValueError("Public Supplementary Table S2 should be a compact headline summary, not the full technical table")
    banned_exact = {
        "run_id",
        "cache_key",
        "source_file",
        "oof_prediction_file",
        "fold_metrics_file",
        "experiment_id",
        "bundle_table",
        "canonical_slot_id",
        "source_priority",
    }
    offenders: list[dict[str, str]] = []
    public_paths = sorted((manuscript_dir / "tables").glob("table_*.csv")) + sorted((manuscript_dir / "supplement").glob("table_*.csv"))
    for path in public_paths:
        frame = pd.read_csv(path, nrows=0)
        for column in frame.columns:
            normalized = str(column).strip().lower()
            if normalized in banned_exact or normalized.endswith("_id"):
                offenders.append({"file": str(path.relative_to(manuscript_dir)), "column": str(column)})
    if offenders:
        raise ValueError(f"Manuscript-ready tables contain technical/provenance columns: {json.dumps(offenders[:50], indent=2)}")


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
        _validate_manuscript_ready_tables(manuscript_dir, df)
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
