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
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import BUNDLE_ROOT, DATASETS_ROOT, enabled as experiment_enabled, load_yaml, resolve_paths_map
from utils.checkpointing import atomic_write_csv, atomic_write_json
from utils.stat_tests import bh_qvalues, paired_bootstrap_delta, paired_delong_auc


MAIN_ENDPOINTS = ["damage_class", "HRD_Score", "hrd_binary_24", "hrd_binary_33", "hrd_binary_42", "cancer_type_top20", "OS"]
SURVIVAL_MAIN_ENDPOINTS = {"OS"}
FIGURE_MODEL_COMPARISON_ENDPOINTS = [endpoint for endpoint in MAIN_ENDPOINTS if endpoint not in SURVIVAL_MAIN_ENDPOINTS]
TCGA_TOP20_ENDPOINTS = {"cancer_type_top20", "tcga_20_type"}
SUPPLEMENTARY_ENDPOINTS = ["HRD_TAI", "HRD_LST", "HRD_LOH", "PARPi7", "eCARD", "PFI"]
MANUSCRIPT_ENDPOINTS = MAIN_ENDPOINTS + SUPPLEMENTARY_ENDPOINTS
MANUSCRIPT_ENDPOINT_SET = set(MANUSCRIPT_ENDPOINTS)
MANUSCRIPT_EXCLUDED_ENDPOINT_LABEL_TERMS = [
    "parpi7_binary",
    "PARPi7 binary",
    "high_purity",
    "High tumor purity",
    "smoking_ever",
    "Ever smoker",
    "high_stage",
    "High stage",
    "DSS",
    "DFI",
    "brca_gene_mutated",
    "BRCA-pathway mutated",
    "kmt2c_mutated",
    "luad_kmt2c_mutated",
    "LUAD KMT2C mutated",
    "mmr_gene_mutated",
    "MMR-pathway mutated",
    "pole_pold1_mutated",
    "POLE/POLD1 mutated",
    "tcga_20_type",
    "TCGA 20 Type",
    "Low-burden Kucab",
    "Original Kucab",
]
MANUSCRIPT_EXCLUDED_ENDPOINT_PATTERN = re.compile(
    "|".join(re.escape(term) for term in MANUSCRIPT_EXCLUDED_ENDPOINT_LABEL_TERMS),
    flags=re.IGNORECASE,
)
MUAT_MAIN_ENDPOINTS = ["HRD_Score", "hrd_binary_33"]
TABULAR_MAIN_REPRESENTATIONS = [
    "burden_only",
    "signatures_only",
    "MAF_stack_only",
    "signatures_plus_MAF_stack",
]
MAIN_REPRESENTATIONS = [
    *TABULAR_MAIN_REPRESENTATIONS,
    "MuAt_style_attention_MIL",
]
MODEL_FAMILIES = ["elastic_net", "XGBoost", "cox_ph", "MuAt-compatible reimplementation", "MuAt-style attention MIL"]
CANONICAL_SOURCE_BY_FAMILY = {
    "burden_only": "main_manuscript_complete_panel",
    "signatures_only": "main_manuscript_complete_panel",
    "MAF_stack_only": "main_manuscript_complete_panel",
    "signatures_plus_MAF_stack": "main_manuscript_complete_panel",
    "MuAt_style_attention_MIL": "muat_style_tcga_comparator",
}
MAIN_COMPARISONS = [
    ("figure_2", "signatures_vs_burden", "signatures_only", "burden_only"),
    ("figure_4", "maf_stack_vs_signatures", "MAF_stack_only", "signatures_only"),
    ("figure_4", "sig_maf_vs_signatures", "signatures_plus_MAF_stack", "signatures_only"),
    ("figure_4", "sig_maf_vs_maf_stack", "signatures_plus_MAF_stack", "MAF_stack_only"),
]
REQUIRED_ENDPOINT_FILES = [
    "main_manuscript_complete_panel_endpoint_results.csv",
]
ACTIVE_EXPERIMENT_IDS = {"main_manuscript_complete_panel", "muat_style_tcga_comparator"}
SOURCE_PRIORITY = {
    "main_manuscript_complete_panel": 60,
    "muat_style_tcga_comparator": 54,
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
    "tables/publication/table_s0_source_inventory.csv",
    "tables/publication/table_s0_source_inventory.html",
    "tables/publication/table_s1_class_distribution_baselines.csv",
    "tables/publication/table_s1_class_distribution_baselines.html",
    "tables/publication/table_s2_sensitivity_analyses.csv",
    "tables/publication/table_s2_sensitivity_analyses.html",
    "tables/publication/table_s3_completeness_and_na_reasons.csv",
    "tables/publication/table_s3_completeness_and_na_reasons.html",
    "tables/publication/table_s5_bio_maf_v4_feature_guide.csv",
    "tables/publication/table_s5_bio_maf_v4_feature_guide.html",
    "tables/publication/table_s6_bio_maf_v4_nested_feature_selection.csv",
    "tables/publication/table_s6_bio_maf_v4_nested_feature_selection.html",
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
    "tables/table_s0_source_inventory.csv",
    "tables/table_s0_source_inventory.html",
    "tables/table_s1_class_distribution_baselines.csv",
    "tables/table_s1_class_distribution_baselines.html",
    "tables/table_s2_sensitivity_analyses.csv",
    "tables/table_s2_sensitivity_analyses.html",
    "tables/table_s3_completeness_and_na_reasons.csv",
    "tables/table_s3_completeness_and_na_reasons.html",
    "tables/table_s5_bio_maf_v4_feature_guide.csv",
    "tables/table_s5_bio_maf_v4_feature_guide.html",
    "tables/table_s6_bio_maf_v4_nested_feature_selection.csv",
    "tables/table_s6_bio_maf_v4_nested_feature_selection.html",
    "supplement/table_s0_source_inventory.csv",
    "supplement/table_s0_source_inventory.html",
    "supplement/table_s1_class_distribution_baselines.csv",
    "supplement/table_s1_class_distribution_baselines.html",
    "supplement/table_s2_sensitivity_analyses.csv",
    "supplement/table_s2_sensitivity_analyses.html",
    "supplement/table_s3_completeness_and_na_reasons.csv",
    "supplement/table_s3_completeness_and_na_reasons.html",
    "supplement/table_s5_bio_maf_v4_feature_guide.csv",
    "supplement/table_s5_bio_maf_v4_feature_guide.html",
    "supplement/table_s6_bio_maf_v4_nested_feature_selection.csv",
    "supplement/table_s6_bio_maf_v4_nested_feature_selection.html",
    "tables/technical/table_s0_source_inventory_technical.csv",
    "tables/technical/table_s0_source_inventory_technical.html",
    "tables/technical/table_s1_class_distribution_baselines_technical.csv",
    "tables/technical/table_s1_class_distribution_baselines_technical.html",
    "tables/technical/table_s2_sensitivity_analyses_technical.csv",
    "tables/technical/table_s2_sensitivity_analyses_technical.html",
    "tables/technical/table_s3_completeness_and_na_reasons_technical.csv",
    "tables/technical/table_s3_completeness_and_na_reasons_technical.html",
    "tables/technical/table_s5_bio_maf_v4_feature_guide_technical.csv",
    "tables/technical/table_s5_bio_maf_v4_feature_guide_technical.html",
    "tables/technical/table_s6_bio_maf_v4_nested_feature_selection_by_fold.csv",
    "tables/technical/table_s6_bio_maf_v4_nested_feature_selection_by_fold.html",
    "figures/figure_1_conceptual_overview.png",
    "figures/figure_2_signature_baselines.png",
    "figures/figure_3_geometry_vs_signatures.png",
    "figures/figure_4_maf_stack_vs_signatures.png",
    "figures/figure_5_overall_survival_cox.png",
    "supplement/figure_s1_representation_construction.png",
    "supplement/figure_s2_calibration_thresholds.png",
    "supplement/figure_s3_feature_importance.png",
    "supplement/figure_s4_bio_maf_v4_feature_map.png",
    "supplement/figure_s4b_bio_maf_v4_feature_distributions.png",
    "supplement/figure_s4c_bio_maf_v4_feature_distributions_by_cancer_type.png",
    "supplement/figure_s5_bio_maf_v4_nested_feature_selection.png",
    "tables/technical/table_s4b_bio_maf_v4_feature_distribution_summary.csv",
    "tables/technical/table_s4b_bio_maf_v4_feature_distribution_summary.html",
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
    "table_s4_cancer_type_top20_metrics": "Supplementary Table S4. TCGA top-20 cancer-type metrics",
    "table_s5_bio_maf_v4_feature_guide": "Supplementary Table S5. Bio MAF v4 feature guide",
    "table_s6_bio_maf_v4_nested_feature_selection": "Supplementary Table S6. Nested-CV Bio MAF v4 feature-block selections",
}
TABLE_NOTES = {
    "table_1_datasets_endpoints": [
        "Main endpoints are evaluated with five outer folds, inner validation tuning, and pooled out-of-fold predictions where model-based.",
    ],
    "table_2_full_performance_metrics": [
        "EN = elastic net; XGB = XGBoost. Non-survival rows report EN / XGB scores; survival rows report CoxNet C-index separately.",
    ],
    "table_3_hyperparameters_feature_dimensionality": [
        "Feature-dimensionality ranges are observed across endpoint/model rows because spectrum channels vary by endpoint data source and Bio MAF v4 blocks are selected within each outer fold's inner-validation split; MuAt-compatible is an event-bag model rather than a fixed tabular vector.",
    ],
    "table_4_label_mapping": [
        "AUROC = area under the receiver operating characteristic curve; HRD = homologous recombination deficiency; KME = kernel mean embedding; MAF = mutation annotation format.",
    ],
    "table_s0_source_inventory": [
        "Rows summarize regenerated source artifacts used to build the manuscript tables and figures.",
    ],
    "table_s1_class_distribution_baselines": [
        "Supplementary endpoints are listed separately from the seven main manuscript endpoints.",
    ],
    "table_s2_sensitivity_analyses": [
        "Headline rows summarize the best baseline or event-level result, best geometry or sensitivity result, and best overall supplementary result.",
    ],
    "table_s3_completeness_and_na_reasons": [
        "Counts summarize unavailable or intentionally omitted supplementary analysis combinations; endpoint-level details are retained in the technical table.",
    ],
    "table_s4_cancer_type_top20_metrics": [
        "Rows report pooled out-of-fold metrics for the fixed 20-class TCGA cancer-type endpoint. Balanced accuracy is the primary metric; other columns summarize classification behavior from the same predictions.",
    ],
    "table_s5_bio_maf_v4_feature_guide": [
        "Rows summarize the predeclared Bio MAF v4 feature blocks. The technical table gives the exact feature-level glossary used by the benchmark.",
    ],
    "table_s6_bio_maf_v4_nested_feature_selection": [
        "Rows summarize which predeclared Bio MAF v4 feature block was selected inside the inner loop of nested CV. This is block-level selection, not post-hoc individual gene selection.",
    ],
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
    "macro_auroc": "Macro-AUROC",
    "micro_auroc": "Micro-AUROC",
    "delta_vs_signatures": "Delta vs signatures",
    "macro_f1": "Macro-F1",
    "cohen_kappa": "Cohen's kappa",
    "top3_accuracy": "Top-3 accuracy",
    "top5_accuracy": "Top-5 accuracy",
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
    "macro_auroc",
    "micro_auroc",
    "macro_f1",
    "f1",
    "balanced_accuracy",
    "cohen_kappa",
    "top3_accuracy",
    "top5_accuracy",
    "delta",
    "delta_vs_signatures",
    "ci_low",
    "ci_high",
    "p_value",
    "q_value",
    "prevalence",
}
MAIN_REPRESENTATION_SHORT_LABELS = {
    "burden_only": "Burden",
    "signatures_only": "Signatures",
    "MAF_stack_only": "Bio MAF v4",
    "signatures_plus_MAF_stack": "Signatures + bio MAF",
    "MuAt_style_attention_MIL": "MuAt-compatible",
}
MANUSCRIPT_TABLE_REPRESENTATIONS = [
    "burden_only",
    "signatures_only",
    "MAF_stack_only",
    "signatures_plus_MAF_stack",
    "MuAt_style_attention_MIL",
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
    "MAF_stack_only": {
        "input_signal": "Predeclared Bio MAF v4 blocks selected inside each outer fold's inner validation split",
        "role": "Interpretable event-level biological annotation comparator with nested feature-block selection",
    },
    "signatures_plus_MAF_stack": {
        "input_signal": "Mutational spectra plus predeclared Bio MAF v4 blocks selected by nested validation",
        "role": "Combined practical default with inner-loop feature-block selection",
    },
    "MuAt_style_attention_MIL": {
        "input_signal": "Per-mutation MC3 event bags with motif, position-bin, and genic/exonic/strand tokens",
        "role": "TCGA-WES MuAt-compatible smoke comparator; official MuAt only when package/checkpoints are configured",
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
    "COSMIC_NNLS_exposures",
}
KEY_GLOSSARY_ROWS = [
    ("EN / XGB", "Elastic-net score / XGBoost score in compact performance tables.", "Table 2"),
    ("AUROC", "Area under the receiver operating characteristic curve for binary endpoints.", "Tables 2 and S2"),
    ("macro-AUROC", "Class-balanced AUROC averaged across multiclass labels.", "Tables 2 and S2"),
    ("Balanced accuracy", "Mean per-class recall; the primary metric for the fixed top-20 TCGA cancer-type endpoint.", "Tables 2 and S4"),
    ("Spearman r", "Spearman rank correlation for continuous endpoints.", "Tables 2 and S2"),
    ("HRD", "Homologous recombination deficiency.", "Endpoint labels"),
    ("MAF", "Mutation annotation format; here also shorthand for event-level mutation annotations.", "Representations"),
    ("SBS/DBS/ID", "Single-base substitution, double-base substitution, and insertion/deletion mutation spectra.", "Representations"),
    ("KME", "Kernel mean embedding used to summarize event-level mutation contexts.", "Representations"),
    ("UGA", "Universal genomic atlas/channel geometry representation used in supplementary analyses.", "Supplement"),
    ("Mutational burden", "Compact mutation-count baseline features.", "Tables 2 and 3"),
    ("Mutational signatures", "Canonical mutation-spectrum representation.", "Tables 2 and 3"),
    ("Bio MAF v4", "Predeclared named cancer-gene, evidence-confidence, hotspot, consequence, allele-fraction, and event-annotation feature blocks selected inside nested CV.", "Tables 2 and 3"),
    ("Signatures + bio MAF", "Combined spectra plus frozen named event-level MAF features.", "Tables 2 and 3"),
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


def _manuscript_endpoint_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "endpoint" not in df.columns:
        return df.copy()
    return df[df["endpoint"].astype(str).isin(MANUSCRIPT_ENDPOINT_SET)].copy()


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
    stem = path.stem
    if stem.endswith("_technical"):
        base = stem.removesuffix("_technical")
        return f"{TABLE_TITLES.get(base, base.replace('_', ' ').title())} (technical detail)"
    return TABLE_TITLES.get(stem, stem.replace("_", " ").title())


def _html_table_note(path: Path) -> list[str]:
    stem = path.stem.removesuffix("_technical")
    notes = TABLE_NOTES.get(stem, [])
    if path.stem.endswith("_technical"):
        return notes + ["Technical detail table retained for reproducibility; manuscript-facing summary tables omit run, cache, and source provenance columns."]
    return notes


def _split_table_heading(title: str) -> tuple[str, str]:
    match = re.match(r"^(Supplementary\s+Table\s+\S+|Table\s+\S+)\.\s*(.+)$", title)
    if match:
        return match.group(1), match.group(2)
    return "", title


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
    table_number, table_title = _split_table_heading(title)
    notes = _html_table_note(path)
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
    table_number_html = f'<div class="table-number">{html.escape(table_number)}</div>\n' if table_number else ""
    note_html = ""
    if notes:
        note_rows = []
        for index, note in enumerate(notes):
            prefix = '<span class="note-label">Note.</span> ' if index == 0 else ""
            note_rows.append(f"<p>{prefix}{html.escape(note)}</p>")
        note_html = '<div class="table-notes">\n' + "\n".join(note_rows) + "\n</div>"
    table = (
        '<div class="manuscript-table-block">\n'
        f"{table_number_html}"
        f'<div class="table-title">{html.escape(table_title)}</div>\n'
        '<div class="table-wrap">\n'
        '<table class="manuscript-table">\n'
        "<thead><tr>"
        + header
        + "</tr></thead>\n<tbody>\n"
        + "\n".join(body_rows)
        + "\n</tbody>\n</table>\n</div>\n"
        + note_html
        + "\n</div>"
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{
  background: #ffffff;
  color: #111111;
  font-family: "Times New Roman", Times, serif;
  margin: 28px auto;
  max-width: 940px;
  padding: 0 18px;
}}
.manuscript-table-block {{
  margin: 0 auto;
}}
.table-number {{
  font-size: 16px;
  font-weight: 700;
  line-height: 1.1;
  margin: 0 0 22px;
  text-align: center;
}}
.table-title {{
  font-size: 15px;
  font-variant: small-caps;
  letter-spacing: 0.04em;
  line-height: 1.25;
  margin: 0 0 24px;
  text-align: center;
}}
.table-wrap {{
  overflow-x: auto;
}}
table.manuscript-table {{
  border-top: 1.5px solid #111111;
  border-collapse: collapse;
  border-spacing: 0;
  font-size: 14px;
  line-height: 1.12;
  width: 100%;
}}
table.manuscript-table thead {{
  border-bottom: 1.5px solid #111111;
}}
table.manuscript-table th {{
  font-weight: 400;
  padding: 8px 10px 10px;
  text-align: center;
  vertical-align: bottom;
}}
table.manuscript-table td {{
  border: 0;
  padding: 2px 10px;
  vertical-align: top;
}}
table.manuscript-table tbody tr:last-child td {{
  border-bottom: 1.5px solid #111111;
  padding-bottom: 7px;
}}
table.manuscript-table td:first-child {{
  padding-left: 0;
  text-align: left;
}}
table.manuscript-table td.numeric {{
  font-variant-numeric: tabular-nums;
  text-align: center;
  white-space: nowrap;
}}
table.manuscript-table td.long-text {{
  min-width: 12rem;
}}
.table-notes {{
  font-size: 13px;
  line-height: 1.18;
  margin-top: 14px;
  text-align: left;
}}
.table-notes p {{
  margin: 0 0 5px;
}}
.note-label {{
  font-variant: small-caps;
  letter-spacing: 0.03em;
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


def _write_supplement_public_table(df: pd.DataFrame, supplement_path: Path, tables_dir: Path, public_dir: Path) -> None:
    _write_table_csv_html(df, supplement_path)
    table_path = tables_dir / supplement_path.name
    _write_table_csv_html(df, table_path)
    _write_publication_copy(df, table_path, public_dir)


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


def _bio_maf_v4_feature_guide() -> tuple[pd.DataFrame, pd.DataFrame]:
    tables_dir = BUNDLE_ROOT / "results" / "tables"
    exact_path = tables_dir / "proposed_clinical_bio_v4_feature_table_exact.csv"
    panel_path = tables_dir / "proposed_clinical_bio_v4_driver_gene_panel.csv"
    summary_path = tables_dir / "proposed_clinical_bio_v4_feature_summary.csv"

    exact = pd.read_csv(exact_path) if exact_path.exists() else pd.DataFrame()
    if exact.empty:
        counts = {
            "v3_core_compact_biology": 121,
            "external_driver_evidence_tier_aggregates": 16,
            "exact_consensus_driver_gene_identity": 802,
            "cancer_hotspot_summary": 4,
            "optional_burden_and_event_composition_controls": 5,
        }
    else:
        counts = exact["Bio MAF v4 block"].value_counts().to_dict()
    panel_count = 401
    if panel_path.exists():
        panel_count = int(pd.read_csv(panel_path)["gene"].nunique())

    compact_count = int(counts.get("v3_core_compact_biology", 121))
    tier_count = int(counts.get("external_driver_evidence_tier_aggregates", 16))
    exact_count = int(counts.get("exact_consensus_driver_gene_identity", 802))
    hotspot_count = int(counts.get("cancer_hotspot_summary", 4))
    optional_count = int(counts.get("optional_burden_and_event_composition_controls", 5))
    strict_core_count = compact_count + tier_count + exact_count + hotspot_count
    full_count = strict_core_count + optional_count
    candidate_size_derivations = (
        f"Nested-CV Bio MAF candidate feature-set sizes: compact only = {compact_count} "
        "(100 VEP effect-label features + 5 allele-fraction summary features + 12 OncoKB role-summary features + "
        "4 non-OncoKB cancer-gene summary features); "
        f"compact + evidence-confidence controls = {compact_count} + {tier_count} = {compact_count + tier_count}; "
        f"compact + hotspots = {compact_count} + {hotspot_count} = {compact_count + hotspot_count}; "
        f"compact + exact cancer-gene identity = {compact_count} + {exact_count} = {compact_count + exact_count}; "
        f"full strict core = {compact_count} + {tier_count} + {exact_count} + {hotspot_count} = {strict_core_count}; "
        f"full + optional controls = {strict_core_count} + {optional_count} = {full_count}. "
        "Mutational signatures are separate from Bio MAF; signatures + Bio MAF models add signature-matrix features on top of the selected Bio MAF candidate."
    )

    rows = [
        {
            "Feature block": "Compact biological annotation core",
            "Features": compact_count,
            "Strict Bio MAF v4 core": "Yes",
            "Where the count comes from": "121 = 100 Ensembl VEP annotation features + 5 allele-fraction measurements + 12 OncoKB cancer-role measurements + 4 non-OncoKB cancer-gene measurements. The 100 VEP features are 50 fixed VEP labels x 2 value types: a log-count and a fraction. The 50 VEP labels are 5 IMPACT labels, 26 consequence labels, 3 canonical-transcript labels, 7 transcript-biotype labels, 5 SIFT labels, and 4 PolyPhen labels.",
            "Plain-English meaning": "Describes what kinds of mutations the tumor has, how disruptive they look, what cancer-gene groups they hit, and how large those mutations appear in the tumor DNA.",
            "How to read one feature": "`vep_impact_count__high` means the tumor has more mutations predicted to strongly disrupt a gene; `vaf_max` means the largest mutation is present in a larger share of tumor DNA reads.",
            "How feature values are calculated": "VEP count features are log1p(number of matching mutations). VEP fraction features are matching mutations divided by total annotated mutations. Allele-fraction features are mean, standard deviation, median, 90th percentile, and maximum. Cancer-role count and unique-gene features use log1p(count); max-VAF features use the largest allele fraction.",
            "Source / grounding": "Ensembl VEP fields, allele-fraction fields in the MAF, and the vendored OncoKB Cancer Gene List.",
            "Key terms decoded": "Bio MAF = a biologically annotated feature matrix built from MAF fields plus fixed external references; it is not the raw MAF file and it is not mutational signatures. MAF = Mutation Annotation Format, the mutation table for a tumor. VEP = Ensembl Variant Effect Predictor, a standard tool that labels the likely effect of each mutation. A VEP label/category is one allowed value from a fixed VEP field. SIFT and PolyPhen are VEP-reported predictions about whether a protein change is likely damaging. Biotype means transcript class, such as protein-coding or long non-coding RNA. Canonical transcript means the main/reference transcript used for a gene. Allele fraction/VAF means the fraction of tumor DNA reads carrying the mutation. OncoKB roles are curated cancer-gene roles such as oncogene or tumor suppressor.",
            "Why included": "These are standard mutation-file annotations and do not require choosing genes based on the endpoint being predicted.",
            "Nested-CV use": "Included in every Bio MAF v4 candidate block.",
            "Example feature name": "`vep_consequence_count__missense_variant`; paired feature `vep_consequence_fraction__missense_variant`",
            "Toy MAF input for example": "A toy tumor has 7 annotated mutations. VEP labels 3 of them as `missense_variant`, meaning the mutation changes one amino acid in the protein.",
            "Exact example calculation": "`vep_consequence_count__missense_variant` = log1p(3) = 1.386. `vep_consequence_fraction__missense_variant` = 3 / 7 = 0.429.",
            "How to interpret the example": "The tumor has 3 missense mutations, and missense mutations make up 42.9% of its annotated mutation list. The count tells the model absolute amount; the fraction tells the model composition.",
        },
        {
            "Feature block": "External evidence-confidence controls",
            "Features": tier_count,
            "Strict Bio MAF v4 core": "Yes, evidence-control block",
            "Where the count comes from": "16 = 4 evidence-confidence groups x 4 named measurements. A gene's evidence-confidence group is the number of external cancer references that include the gene: 1 source, 2 sources, 3 sources, or 4+ sources. The 4 measurements per group are: high-or-moderate VEP-impact mutation count, high VEP-impact mutation count, unique genes with high-or-moderate VEP-impact mutations, and maximum allele fraction among high-or-moderate VEP-impact mutations.",
            "Plain-English meaning": "This is not a biological mechanism by itself. It is a low-dimensional control that tests whether mutations in better-established cancer genes add signal without naming the exact gene.",
            "How to read one feature": "`driver_evidence_high_impact_log_count__source_count_3` means the tumor has more strongly disruptive mutations in genes supported by 3 trusted sources.",
            "How feature values are calculated": "Count and unique-gene features use log1p(count). The max-VAF feature is the largest allele fraction among matching mutations in that evidence-confidence group.",
            "Source / grounding": "OncoKB, COSMIC CGC v99 flag from the OncoKB list, IntOGen 2024 driver compendium, Vogelstein 2013, plus Cancer Hotspots v3 for hotspot genes.",
            "Key terms decoded": "Evidence-confidence group means how many independent external cancer resources include a gene. It should not be interpreted as biology inside the tumor; it is a way to summarize external confidence in whether the gene is a cancer gene. Vogelstein refers to the curated cancer-driver gene list from Vogelstein and colleagues' cancer genome landscape work. Nested CV means nested cross-validation: feature-block choice happens inside training folds, while held-out outer folds estimate performance.",
            "Why included": "Included as a predeclared sensitivity/control block: it lets nested CV test a collapsed evidence-confidence signal against compact biology, exact-gene features, hotspot features, and full-core combinations. It is not selected using held-out labels.",
            "Nested-CV use": "Competes as the compact + evidence-confidence candidate; also part of the full-core and full-core + optional-control candidates.",
            "Example feature name": "`driver_evidence_high_or_moderate_impact_log_count__source_count_4`",
            "Toy MAF input for example": "In the fixed reference table, TP53 and BRCA1 are in the 4-source evidence-confidence group. A toy tumor has one high/moderate-impact TP53 mutation and one high/moderate-impact BRCA1 mutation.",
            "Exact example calculation": "`driver_evidence_high_or_moderate_impact_log_count__source_count_4` = log1p(2) = 1.099 because 2 mutations match the 4-source group and the high/moderate-impact rule.",
            "How to interpret the example": "The tumor has two disruptive mutations in genes that many external cancer resources agree are cancer genes. This feature does not say TP53 or BRCA1 specifically; it summarizes the confidence level of the genes hit.",
        },
        {
            "Feature block": "Exact consensus driver-gene identity",
            "Features": exact_count,
            "Strict Bio MAF v4 core": "Yes",
            "Where the count comes from": f"{exact_count} = {panel_count} externally supported cancer genes x 2 named measurements per gene. The {panel_count} genes are fixed before modeling by a written external-resource rule: selected genes have at least 3 sources of support among OncoKB, COSMIC CGC, IntOGen, and Vogelstein, or are Cancer Hotspots v3 genes. The 2 measurements are a protein-affecting mutation count and a cancer-role-fit mutation count.",
            "Plain-English meaning": "Records which specific known cancer genes are changed in the tumor, using a fixed gene list created before modeling without endpoint labels or model performance.",
            "How to read one feature": "`driver_gene_functional_event_log_count__TP53` means TP53 has a mutation likely to affect the TP53 protein. `driver_gene_role_matched_event_log_count__TP53` means the mutation fits TP53's usual cancer role as a gene that is often damaged or lost.",
            "How feature values are calculated": "Both exact-gene features are per-tumor, per-gene log-counts, not averages. Protein-affecting mutation count = log1p(number of mutations in that patient's tumor and that gene that are likely to change or disrupt the encoded protein). A mutation is counted as protein-affecting if VEP IMPACT is HIGH or MODERATE, or if the VEP consequence is one of these predefined coding/protein-affecting terms: stop gained/lost, frameshift, splice acceptor/donor, start lost, transcript ablation, missense, in-frame insertion/deletion, protein_altering_variant, or coding_sequence_variant. Cancer-role-fit mutation count = log1p(number of mutations in that patient's tumor and that gene whose mutation type fits the gene's externally curated cancer role): loss-like events for tumor suppressor genes, activating-like or hotspot events for oncogenes, and either pattern for both-role/ambiguous genes.",
            "Source / grounding": "Fixed gene panel from OncoKB, COSMIC CGC v99 flag in OncoKB, IntOGen 2024, Vogelstein 2013, and Cancer Hotspots v3.",
            "Key terms decoded": "Fixed before modeling means the gene list is created once from external resources before cross-validation begins; no endpoint labels, fold results, or model performance are used to choose genes. Protein-affecting means likely to change or disrupt the gene's protein product by fixed VEP rules. Cancer-role-fit means the mutation type is consistent with how that gene is known to act in cancer. COSMIC CGC is the COSMIC Cancer Gene Census. IntOGen is an external cancer-driver gene compendium. Cancer Hotspots is a catalogue of recurrently mutated cancer sites.",
            "Why included": "Many cancer-type and HRD signals depend on which cancer gene is altered, not just on the generic consequence category.",
            "Nested-CV use": "Competes as `v4_compact_plus_exact_driver_genes`; also part of `v4_full_core` and `v4_full_core_plus_optional_controls`.",
            "Example feature name": "`driver_gene_functional_event_log_count__TP53`; paired feature `driver_gene_role_matched_event_log_count__TP53`",
            "Toy MAF input for example": "A toy tumor has one TP53 frameshift mutation and one TP53 synonymous mutation. The frameshift disrupts the protein; the synonymous mutation does not change the protein sequence.",
            "Exact example calculation": "`driver_gene_functional_event_log_count__TP53` = log1p(1) = 0.693 because only the frameshift is protein-affecting. `driver_gene_role_matched_event_log_count__TP53` = log1p(1) = 0.693 because TP53 is curated as a tumor-suppressor gene and a frameshift is a loss-like event.",
            "How to interpret the example": "The tumor has a TP53 mutation that both affects the protein and fits TP53's known cancer role as a gene commonly damaged or lost in cancer. The synonymous TP53 mutation does not add to either example count.",
        },
        {
            "Feature block": "Cancer hotspot summaries",
            "Features": hotspot_count,
            "Strict Bio MAF v4 core": "Yes",
            "Where the count comes from": "4 named hotspot measurements: exact hotspot count, recurrent protein-site count, number of genes with hotspot hits, and largest allele fraction among hotspot mutations.",
            "Plain-English meaning": "Measures whether mutations fall at protein sites repeatedly seen in cancer.",
            "How to read one feature": "`cancer_hotspot_residue_log_count` means the tumor has more mutations at protein positions repeatedly seen in cancer.",
            "How feature values are calculated": "Exact hotspot, recurrent protein-site, and unique-gene features use log1p(count). The hotspot max-VAF feature is the largest allele fraction among hotspot mutations.",
            "Source / grounding": "Cancer Hotspots v3 / cBioPortal Cancer Hotspots resource.",
            "Key terms decoded": "A cancer hotspot is a position in a protein that is recurrently mutated across cancer datasets. Recurrent does not mean the mutation occurs repeatedly in the same patient; it means the same site is observed across tumors.",
            "Why included": "Hotspots capture activating driver-like signal that may be missed by generic high/moderate-impact consequence labels.",
            "Nested-CV use": "Competes as `v4_compact_plus_hotspot_summary`; also part of `v4_full_core` and `v4_full_core_plus_optional_controls`.",
            "Example feature name": "`cancer_hotspot_residue_log_count`; paired feature `cancer_hotspot_max_vaf`",
            "Toy MAF input for example": "A toy tumor has two mutations whose gene/protein positions match Cancer Hotspots recurrent residues. Their allele fractions are 0.21 and 0.47.",
            "Exact example calculation": "`cancer_hotspot_residue_log_count` = log1p(2) = 1.099. `cancer_hotspot_max_vaf` = max(0.21, 0.47) = 0.47.",
            "How to interpret the example": "The tumor has two mutations at protein positions repeatedly observed across cancer datasets. The maximum allele fraction says the strongest hotspot mutation is present in 47% of tumor DNA reads covering that site.",
        },
        {
            "Feature block": "Optional burden and event-composition controls",
            "Features": optional_count,
            "Strict Bio MAF v4 core": "No",
            "Where the count comes from": "5 named mutation-count controls: single-base mutation count, double-base mutation count, insertion/deletion count, double-base mutation fraction, and insertion/deletion fraction.",
            "Plain-English meaning": "Measures broad mutation load and the basic mix of mutation types directly from the MAF.",
            "How to read one feature": "`log10_id_burden` means the tumor has more insertion/deletion events after compressing very large counts.",
            "How feature values are calculated": "Burden controls use log10(1 + event count). Fraction controls are event-class count divided by the total number of single-base, double-base, and insertion/deletion events.",
            "Source / grounding": "COSMIC SBS/DBS/ID event-class definitions applied directly to MAF event types.",
            "Key terms decoded": "SBS means single-base substitution, DBS means double-base substitution, and ID means insertion/deletion. These controls count event classes from the MAF; they are separate from COSMIC mutational-signature exposures.",
            "Why included": "Provides a sensitivity/control block because broad mutation burden can be predictive but can also dominate biology.",
            "Nested-CV use": "Only appears in `v4_full_core_plus_optional_controls`.",
            "Example feature name": "`log10_id_burden`; paired feature `id_fraction`",
            "Toy MAF input for example": "A toy tumor has 80 single-base substitutions, 2 double-base substitutions, and 18 insertion/deletion events.",
            "Exact example calculation": "`log10_id_burden` = log10(18 + 1) = 1.279. `id_fraction` = 18 / (80 + 2 + 18) = 0.180.",
            "How to interpret the example": "The tumor has 18 insertion/deletion events after compressing the count so very high-burden tumors do not dominate. Insertions/deletions make up 18.0% of the simple event mix.",
        },
    ]
    guide = pd.DataFrame(rows)
    guide["Share of full Bio MAF v4 universe"] = guide["Features"].map(
        lambda value: f"{float(value) / float(full_count) * 100:.1f}%" if full_count else ""
    )
    guide["Core feature total"] = strict_core_count
    guide["Full feature total including optional controls"] = full_count
    guide["Candidate feature-set size derivations"] = candidate_size_derivations

    if exact.empty:
        exact = guide.copy()
    else:
        exact = exact.copy()
        exact["Strict Bio MAF v4 core"] = np.where(
            exact["Bio MAF v4 block"].eq("optional_burden_and_event_composition_controls"),
            "No",
            "Yes",
        )
        exact["Core feature total"] = strict_core_count
        exact["Full feature total including optional controls"] = full_count
        if summary_path.exists():
            exact["Feature summary source"] = str(summary_path.relative_to(BUNDLE_ROOT))
    return guide, exact


def _bio_maf_v4_feature_set_details(feature_set: object, selected_feature_count: object = None) -> dict[str, object]:
    feature_set_id = str(feature_set or "").strip()
    total_count = pd.to_numeric(pd.Series([selected_feature_count]), errors="coerce").iloc[0]
    has_signatures = feature_set_id == "signatures_only" or feature_set_id.startswith("signatures_plus_")
    bio_set_id = feature_set_id.replace("signatures_plus_", "", 1) if feature_set_id.startswith("signatures_plus_") else feature_set_id
    definitions: dict[str, dict[str, object]] = {
        "signatures_only": {
            "label": "Signatures only; no Bio MAF block",
            "bio_features": 0,
            "blocks": "None",
            "compact": "No",
            "tiers": "No",
            "exact": "No",
            "hotspots": "No",
            "optional": "No",
        },
        "v4_compact_biology_only": {
            "label": "Compact biology only",
            "bio_features": 121,
            "blocks": "Compact biological annotation core",
            "compact": "Yes",
            "tiers": "No",
            "exact": "No",
            "hotspots": "No",
            "optional": "No",
        },
        "v4_compact_plus_driver_evidence_tiers": {
            "label": "Compact biology + evidence-confidence controls",
            "bio_features": 137,
            "blocks": "Compact biological annotation core; external evidence-confidence controls",
            "compact": "Yes",
            "tiers": "Yes",
            "exact": "No",
            "hotspots": "No",
            "optional": "No",
        },
        "v4_compact_plus_exact_driver_genes": {
            "label": "Compact biology + exact driver genes",
            "bio_features": 923,
            "blocks": "Compact biological annotation core; exact consensus driver-gene identity",
            "compact": "Yes",
            "tiers": "No",
            "exact": "Yes",
            "hotspots": "No",
            "optional": "No",
        },
        "v4_compact_plus_hotspot_summary": {
            "label": "Compact biology + hotspot summaries",
            "bio_features": 125,
            "blocks": "Compact biological annotation core; cancer hotspot summaries",
            "compact": "Yes",
            "tiers": "No",
            "exact": "No",
            "hotspots": "Yes",
            "optional": "No",
        },
        "v4_full_core": {
            "label": "Full strict Bio MAF core",
            "bio_features": 943,
            "blocks": "Compact biological annotation core; external evidence-confidence controls; exact consensus driver-gene identity; cancer hotspot summaries",
            "compact": "Yes",
            "tiers": "Yes",
            "exact": "Yes",
            "hotspots": "Yes",
            "optional": "No",
        },
        "v4_full_core_plus_optional_controls": {
            "label": "Full core + optional burden/composition controls",
            "bio_features": 948,
            "blocks": "Compact biological annotation core; external evidence-confidence controls; exact consensus driver-gene identity; cancer hotspot summaries; optional burden and event-composition controls",
            "compact": "Yes",
            "tiers": "Yes",
            "exact": "Yes",
            "hotspots": "Yes",
            "optional": "Yes",
        },
    }
    details = dict(definitions.get(bio_set_id, {}))
    if not details:
        bio_features = max(0, int(total_count - 182)) if has_signatures and np.isfinite(total_count) else int(total_count) if np.isfinite(total_count) else ""
        details = {
            "label": _fallback_label(bio_set_id),
            "bio_features": bio_features,
            "blocks": "",
            "compact": "",
            "tiers": "",
            "exact": "",
            "hotspots": "",
            "optional": "",
        }
    label = str(details["label"])
    if has_signatures and bio_set_id != "signatures_only":
        label = f"Signatures + {label}"
    return {
        "selected_feature_set_id": feature_set_id,
        "selected_feature_set_label": label,
        "selected_bio_maf_block_id": bio_set_id if bio_set_id != "signatures_only" else "",
        "selected_bio_maf_blocks": details["blocks"],
        "selected_bio_maf_features": int(details["bio_features"]) if str(details["bio_features"]).strip() != "" else "",
        "includes_mutational_signatures": "Yes" if has_signatures else "No",
        "includes_compact_biology_core": details["compact"],
        "includes_driver_evidence_tiers": details["tiers"],
        "includes_exact_driver_genes": details["exact"],
        "includes_hotspot_summaries": details["hotspots"],
        "includes_optional_burden_controls": details["optional"],
    }


def _bio_maf_v4_nested_selection_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_path = BUNDLE_ROOT / "results" / "tables" / "main_manuscript_complete_panel_fold_metrics.csv"
    if not fold_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    folds = pd.read_csv(fold_path)
    if "selected_feature_set" not in folds.columns:
        return pd.DataFrame(), pd.DataFrame()
    selected = folds[
        folds["representation"].astype(str).isin(["MAF_stack_only", "signatures_plus_MAF_stack"])
        & folds["selected_feature_set"].notna()
    ].copy()
    selected = selected[selected["endpoint"].astype(str).isin(MANUSCRIPT_ENDPOINT_SET)].copy()
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()
    selected = selected.drop_duplicates(
        ["endpoint", "representation", "learner", "repeat", "fold", "metric"],
        keep="first",
    )
    detail_rows = []
    for _, row in selected.iterrows():
        details = _bio_maf_v4_feature_set_details(row.get("selected_feature_set"), row.get("selected_feature_count"))
        model_label = {"linear": "Elastic net", "xgboost": "XGBoost", "cox_ph": "CoxNet"}.get(
            str(row.get("learner")),
            _display_label("model_family", row.get("learner")),
        )
        detail_rows.append(
            {
                "Endpoint tier": "Main" if str(row.get("endpoint")) in MAIN_ENDPOINTS else "Supplementary",
                "Endpoint": _display_label("endpoint", row.get("endpoint")),
                "Endpoint ID": row.get("endpoint"),
                "Representation": _display_label("representation_family", row.get("representation")),
                "Representation ID": row.get("representation"),
                "Model": model_label,
                "Model ID": row.get("learner"),
                "Primary metric": _display_label("metric", row.get("metric")),
                "Outer fold": int(row.get("fold")) if pd.notna(row.get("fold")) else "",
                "Selected feature-set label": details["selected_feature_set_label"],
                "Selected feature-set ID": details["selected_feature_set_id"],
                "Selected Bio MAF block ID": details["selected_bio_maf_block_id"],
                "Selected total features": int(row.get("selected_feature_count")) if pd.notna(row.get("selected_feature_count")) else "",
                "Selected Bio MAF features": details["selected_bio_maf_features"],
                "Selected Bio MAF blocks": details["selected_bio_maf_blocks"],
                "Includes mutational signatures": details["includes_mutational_signatures"],
                "Includes compact biology core": details["includes_compact_biology_core"],
                "Includes driver-evidence tiers": details["includes_driver_evidence_tiers"],
                "Includes exact driver genes": details["includes_exact_driver_genes"],
                "Includes hotspot summaries": details["includes_hotspot_summaries"],
                "Includes optional burden controls": details["includes_optional_burden_controls"],
                "Inner validation score": row.get("selected_inner_score"),
                "Outer fold score": row.get("score"),
                "Candidate feature-set count": int(row.get("candidate_feature_set_count")) if pd.notna(row.get("candidate_feature_set_count")) else "",
                "Candidate feature sets considered": row.get("candidate_feature_sets_json"),
                "Split strategy": row.get("split_strategy"),
            }
        )
    by_fold = pd.DataFrame(detail_rows)
    group_cols = [
        "Endpoint tier",
        "Endpoint",
        "Endpoint ID",
        "Representation",
        "Representation ID",
        "Model",
        "Model ID",
        "Primary metric",
        "Selected feature-set label",
        "Selected feature-set ID",
        "Selected Bio MAF block ID",
        "Selected total features",
        "Selected Bio MAF features",
        "Selected Bio MAF blocks",
        "Includes mutational signatures",
        "Includes compact biology core",
        "Includes driver-evidence tiers",
        "Includes exact driver genes",
        "Includes hotspot summaries",
        "Includes optional burden controls",
    ]
    denominators = (
        by_fold.groupby(["Endpoint ID", "Representation ID", "Model ID"], dropna=False)["Outer fold"]
        .nunique()
        .rename("Outer folds evaluated")
        .reset_index()
    )
    summary = (
        by_fold.groupby(group_cols, dropna=False)
        .agg(
            Outer_folds_selected=("Outer fold", "nunique"),
            Mean_inner_validation_score=("Inner validation score", "mean"),
            Mean_outer_fold_score=("Outer fold score", "mean"),
            Candidate_feature_set_count=("Candidate feature-set count", "max"),
        )
        .reset_index()
        .merge(denominators, on=["Endpoint ID", "Representation ID", "Model ID"], how="left")
    )
    summary["Fold selection fraction"] = summary["Outer_folds_selected"] / summary["Outer folds evaluated"].replace(0, np.nan)
    summary["Selection interpretation"] = summary.apply(
        lambda row: f"Selected in {int(row['Outer_folds_selected'])} of {int(row['Outer folds evaluated'])} outer folds.",
        axis=1,
    )
    summary = summary.rename(
        columns={
            "Outer_folds_selected": "Outer folds selected",
            "Mean_inner_validation_score": "Mean inner-validation score",
            "Mean_outer_fold_score": "Mean outer-fold score",
            "Candidate_feature_set_count": "Candidate feature-set count",
        }
    )
    tier_order = {"Main": 0, "Supplementary": 1}
    endpoint_order = {endpoint: idx for idx, endpoint in enumerate(MANUSCRIPT_ENDPOINTS)}
    summary["_tier_order"] = summary["Endpoint tier"].map(tier_order).fillna(9)
    summary["_endpoint_order"] = summary["Endpoint ID"].map(endpoint_order).fillna(999)
    summary = summary.sort_values(
        ["_tier_order", "_endpoint_order", "Representation", "Model", "Selected feature-set label"],
        kind="mergesort",
    ).drop(columns=["_tier_order", "_endpoint_order"])
    by_fold["_tier_order"] = by_fold["Endpoint tier"].map(tier_order).fillna(9)
    by_fold["_endpoint_order"] = by_fold["Endpoint ID"].map(endpoint_order).fillna(999)
    by_fold = by_fold.sort_values(
        ["_tier_order", "_endpoint_order", "Representation", "Model", "Outer fold"],
        kind="mergesort",
    ).drop(columns=["_tier_order", "_endpoint_order"])
    return summary, by_fold


def _figure_bio_maf_v4_feature_map_classic_workflow(stem: Path) -> None:
    guide, _exact = _bio_maf_v4_feature_guide()
    if guide.empty:
        return
    counts = {
        str(row["Feature block"]): int(row["Features"])
        for _, row in guide.iterrows()
    }
    full_total = int(pd.to_numeric(guide["Full feature total including optional controls"], errors="coerce").dropna().max())
    core_total = int(pd.to_numeric(guide["Core feature total"], errors="coerce").dropna().max())

    fig, ax = plt.subplots(figsize=(17, 9.5))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FBFCFE")

    def wrap(text: str, width: int) -> str:
        return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False, replace_whitespace=False))

    def add_box(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        *,
        body: str = "",
        face: str = "#FFFFFF",
        edge: str = "#D6DEE9",
        title_color: str = "#1F2430",
        body_color: str = "#333844",
        title_size: float = 9.0,
        body_size: float = 7.3,
        wrap_width: int = 30,
        badge: str | None = None,
        badge_edge: str | None = None,
        lw: float = 0.9,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.009,rounding_size=0.012",
            linewidth=lw,
            edgecolor=edge,
            facecolor=face,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            x + 0.011,
            y + h - 0.020,
            wrap(title, wrap_width),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=title_size,
            fontweight="bold",
            color=title_color,
            linespacing=1.05,
        )
        if badge:
            ax.text(
                x + w - 0.010,
                y + h - 0.018,
                badge,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.2,
                fontweight="bold",
                color="#1F2430",
                bbox={
                    "boxstyle": "round,pad=0.20",
                    "facecolor": "#FFFFFF",
                    "edgecolor": badge_edge or edge,
                    "linewidth": 0.8,
                },
            )
        if body:
            ax.text(
                x + 0.011,
                y + h - 0.052,
                wrap(body, wrap_width),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=body_size,
                color=body_color,
                linespacing=1.12,
            )

    def chip(x: float, y: float, text: str, *, w: float, h: float = 0.040, face: str = "#F3F6FA", edge: str = "#CBD5E1", color: str = "#273142") -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.004,rounding_size=0.010",
                linewidth=0.8,
                edgecolor=edge,
                facecolor=face,
                transform=ax.transAxes,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.0,
            color=color,
            fontweight="bold",
        )

    def arrow(x1: float, x2: float, *, y: float = 0.735, color: str = "#8B95A7") -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y),
                (x2, y),
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.25,
                color=color,
                shrinkA=2,
                shrinkB=2,
            )
        )

    fig.suptitle(
        "Bio MAF v4: fixed biological annotations selected inside nested CV",
        x=0.035,
        y=0.985,
        ha="left",
        fontsize=18.5,
        fontweight="bold",
        color="#1F2430",
    )
    fig.text(
        0.035,
        0.940,
        "A leakage-safe schema: patient MAF fields plus fixed external resources define biological feature blocks; nested CV chooses among predeclared blocks inside training folds.",
        fontsize=10.2,
        color="#6F768A",
        va="top",
    )

    panel_y = 0.252
    panel_h = 0.620
    panels = [
        (0.035, 0.145, "1", "Patient MAF"),
        (0.205, 0.165, "2", "External resources"),
        (0.395, 0.265, "3", "Bio MAF v4 feature blocks"),
        (0.690, 0.140, "4", "Nested-CV candidate sets"),
        (0.855, 0.120, "5", "Model inputs"),
    ]
    for x, w, number, label in panels:
        ax.text(
            x,
            0.888,
            number,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="#FFFFFF",
            bbox={"boxstyle": "circle,pad=0.30", "facecolor": "#3E5F9F", "edgecolor": "#3E5F9F"},
        )
        ax.text(
            x + 0.024,
            0.888,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10.4,
            fontweight="bold",
            color="#1F2430",
        )
        ax.add_patch(
            FancyBboxPatch(
                (x, panel_y),
                w,
                panel_h,
                boxstyle="round,pad=0.010,rounding_size=0.014",
                linewidth=0.9,
                edgecolor="#D7DEE8",
                facecolor="#FFFFFF",
                transform=ax.transAxes,
                zorder=0,
            )
        )

    arrow(0.181, 0.203)
    arrow(0.371, 0.393)
    arrow(0.663, 0.689)
    arrow(0.831, 0.854)

    # Panel 1: Patient MAF.
    maf_chips = [
        ("gene", 0.051, 0.780, 0.050),
        ("consequence", 0.105, 0.780, 0.080),
        ("amino-acid site", 0.051, 0.720, 0.091),
        ("allele fraction", 0.146, 0.720, 0.076),
        ("event type", 0.051, 0.660, 0.069),
        ("hotspot status", 0.125, 0.660, 0.087),
    ]
    for text, x, y, w in maf_chips:
        chip(x, y, text, w=w, face="#F3F6FA", edge="#C7D2E0")
    add_box(
        0.052,
        0.355,
        0.103,
        0.165,
        "Portable input",
        body="Available for any annotated MAF; labels are not used to create these fields.",
        face="#F8FAFC",
        edge="#D7DEE8",
        title_size=8.8,
        body_size=7.4,
        wrap_width=23,
    )

    # Panel 2: External resources.
    source_rows = [
        ("Ensembl VEP", "consequence severity"),
        ("OncoKB", "clinical cancer genes"),
        ("COSMIC CGC", "curated cancer roles"),
        ("IntOGen", "driver compendium"),
        ("Vogelstein genes", "classic drivers"),
        ("Cancer Hotspots", "recurrent sites"),
    ]
    for idx, (name, role) in enumerate(source_rows):
        y = 0.785 - idx * 0.075
        chip(0.222, y, name, w=0.078, face="#F3F6FA", edge="#C7D2E0")
        ax.text(0.307, y + 0.020, role, transform=ax.transAxes, ha="left", va="center", fontsize=6.8, color="#4B5563")
    add_box(
        0.222,
        0.285,
        0.125,
        0.100,
        "Fixed before CV",
        body="Resources are versioned and not learned from endpoint labels.",
        face="#F8FAFC",
        edge="#D7DEE8",
        title_size=8.6,
        body_size=7.2,
        wrap_width=27,
    )

    # Panel 3: feature blocks.
    core_edge = "#4F73BF"
    optional_edge = "#B89D2C"
    feature_cards = [
        (
            "Compact annotation core",
            counts.get("Compact biological annotation core", 121),
            "Measures: consequence, VAF, and cancer-gene role summaries.",
            "Why: broad mutation damage without endpoint-specific genes.",
            "#EEF4FF",
            core_edge,
            None,
        ),
        (
            "Driver-evidence tiers",
            counts.get("External driver-evidence tiers", 16),
            "Measures: mutations in genes supported by 1-4 sources.",
            "Why: literature support without exact-gene identity.",
            "#EEF4FF",
            core_edge,
            None,
        ),
        (
            "Exact consensus driver-gene identity",
            counts.get("Exact consensus driver-gene identity", 802),
            "Measures: 401 genes x two log-counts: protein-affecting and cancer-role-fit mutations.",
            "Why: cancer type and HRD can depend on which gene is hit.",
            "#EEF4FF",
            core_edge,
            "401 genes x 2",
        ),
        (
            "Cancer hotspot summaries",
            counts.get("Cancer hotspot summaries", 4),
            "Measures: exact/residue hotspot hits, genes, and VAF.",
            "Why: recurrent sites mark driver-like biology.",
            "#EEF4FF",
            core_edge,
            None,
        ),
        (
            "Optional burden/composition controls",
            counts.get("Optional burden and event-composition controls", 5),
            "Measures: SBS/DBS/indel burden and simple fractions.",
            "Why: sensitivity controls; can proxy technical factors.",
            "#FFFDF0",
            optional_edge,
            "optional",
        ),
    ]
    card_x, card_w, card_h = 0.412, 0.232, 0.090
    for idx, (title, count, measures, why, face, edge, micro) in enumerate(feature_cards):
        y = 0.775 - idx * 0.105
        body = f"{measures}\n{why}"
        badge = f"{count:,}"
        if micro and micro != "optional":
            badge = f"{count:,} ({micro})"
        elif micro == "optional":
            badge = f"{count:,} optional"
        add_box(
            card_x,
            y,
            card_w,
            card_h,
            title,
            body=body,
            face=face,
            edge=edge,
            title_size=7.9,
            body_size=6.35,
            wrap_width=48,
            badge=badge,
            badge_edge=edge,
            lw=1.1,
        )
    ax.text(
        0.412,
        0.290,
        f"Strict core = {core_total:,} features      Full with optional controls = {full_total:,}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color="#273142",
    )

    # Panel 4: nested-CV candidate ladder.
    ladder = [
        ("compact only", "121", "#EEF4FF", core_edge),
        ("compact + evidence tiers", "137", "#EEF4FF", core_edge),
        ("compact + hotspots", "125", "#EEF4FF", core_edge),
        ("compact + exact drivers", "923", "#EEF4FF", core_edge),
        ("full strict core", "943", "#EEF4FF", core_edge),
        ("full + optional controls", "948", "#FFFDF0", optional_edge),
    ]
    for idx, (label, n, face, edge) in enumerate(ladder):
        y = 0.780 - idx * 0.071
        ax.add_patch(
            FancyBboxPatch(
                (0.707, y),
                0.103,
                0.046,
                boxstyle="round,pad=0.005,rounding_size=0.008",
                linewidth=0.85,
                edgecolor=edge,
                facecolor=face,
                transform=ax.transAxes,
            )
        )
        ax.text(0.713, y + 0.023, label, transform=ax.transAxes, ha="left", va="center", fontsize=6.35, color="#273142")
        ax.text(0.803, y + 0.023, n, transform=ax.transAxes, ha="right", va="center", fontsize=6.9, fontweight="bold", color="#273142")
    ax.text(
        0.707,
        0.345,
        "Selected in inner loop only",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.7,
        fontweight="bold",
        color="#273142",
    )
    ax.text(
        0.707,
        0.315,
        "Outer folds remain held out.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.0,
        color="#6F768A",
    )

    # Panel 5: model inputs.
    add_box(
        0.873,
        0.700,
        0.086,
        0.100,
        "Bio MAF-only model",
        body="Tests biology features without signatures.",
        face="#EEF4FF",
        edge=core_edge,
        title_size=7.6,
        body_size=6.5,
        wrap_width=18,
    )
    add_box(
        0.873,
        0.565,
        0.086,
        0.110,
        "Signatures + Bio MAF",
        body="Tests added annotation signal beyond mutational processes.",
        face="#EEF4FF",
        edge=core_edge,
        title_size=7.6,
        body_size=6.5,
        wrap_width=18,
    )
    add_box(
        0.873,
        0.350,
        0.086,
        0.145,
        "Signatures separate",
        body="COSMIC SBS/DBS/ID signature exposures are not Bio MAF features.",
        face="#F8FAFC",
        edge="#C7D2E0",
        title_size=7.6,
        body_size=6.5,
        wrap_width=18,
    )

    # Bottom reviewer guardrail strip.
    guard_x, guard_y, guard_w, guard_h = 0.035, 0.055, 0.940, 0.135
    ax.add_patch(
        FancyBboxPatch(
            (guard_x, guard_y),
            guard_w,
            guard_h,
            boxstyle="round,pad=0.010,rounding_size=0.014",
            linewidth=1.0,
            edgecolor="#C7D2E0",
            facecolor="#F8FAFC",
            transform=ax.transAxes,
        )
    )
    ax.text(
        guard_x + 0.014,
        guard_y + guard_h - 0.030,
        "Reviewer guardrails",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color="#1F2430",
    )
    guardrails = [
        "Not a post-hoc top-gene screen.",
        "No held-out fold labels used for feature-block selection.",
        "External annotations are versioned and fixed.",
        "Exact feature glossary in Table S5; selected blocks in Table S6.",
    ]
    gx = [0.052, 0.285, 0.548, 0.755]
    gw = [0.195, 0.225, 0.180, 0.200]
    for text, x, w in zip(guardrails, gx, gw):
        ax.text(
            x,
            guard_y + 0.054,
            u"\u2022 " + wrap(text, 34),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.6,
            color="#333844",
            linespacing=1.12,
        )
    _save_figure(fig, stem)


def _figure_bio_maf_v4_feature_map_plain_v1(stem: Path) -> None:
    guide, _exact = _bio_maf_v4_feature_guide()
    if guide.empty:
        return

    counts = {
        str(row["Feature block"]): int(row["Features"])
        for _, row in guide.iterrows()
    }
    full_total = int(pd.to_numeric(guide["Full feature total including optional controls"], errors="coerce").dropna().max())
    core_total = int(pd.to_numeric(guide["Core feature total"], errors="coerce").dropna().max())

    compact_count = counts.get("Compact biological annotation core", 121)
    tier_count = counts.get("External driver-evidence tiers", 16)
    exact_count = counts.get("Exact consensus driver-gene identity", 802)
    hotspot_count = counts.get("Cancer hotspot summaries", 4)
    optional_count = counts.get("Optional burden and event-composition controls", 5)
    panel_count = int(exact_count / 2) if exact_count else 401

    fig, ax = plt.subplots(figsize=(16, 10.2))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FBFCFE")

    blue_edge = "#4F73BF"
    blue_face = "#EEF5FF"
    gold_edge = "#B89D2C"
    gold_face = "#FFF9E6"
    ink = "#1F2430"
    muted = "#596273"
    line = "#CBD5E1"
    neutral_face = "#FFFFFF"
    soft_face = "#F8FAFC"

    def wrap(text: str, width: int) -> str:
        return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False, replace_whitespace=False))

    def add_card(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: str,
        *,
        face: str = neutral_face,
        edge: str = line,
        title_size: float = 11.2,
        body_size: float = 8.6,
        wrap_width: int = 48,
        badge: str | None = None,
        badge_face: str = "#FFFFFF",
        badge_edge: str | None = None,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.010,rounding_size=0.014",
                linewidth=1.0,
                edgecolor=edge,
                facecolor=face,
                transform=ax.transAxes,
            )
        )
        ax.text(
            x + 0.018,
            y + h - 0.026,
            wrap(title, wrap_width),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=title_size,
            fontweight="bold",
            color=ink,
            linespacing=1.08,
        )
        if badge:
            ax.text(
                x + w - 0.018,
                y + h - 0.024,
                badge,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9.0,
                fontweight="bold",
                color=ink,
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": badge_face,
                    "edgecolor": badge_edge or edge,
                    "linewidth": 0.9,
                },
            )
        ax.text(
            x + 0.018,
            y + h - 0.068,
            wrap(body, wrap_width),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=body_size,
            color=muted,
            linespacing=1.20,
        )

    def chip(
        x: float,
        y: float,
        text: str,
        *,
        w: float,
        h: float = 0.040,
        face: str = "#F3F6FA",
        edge: str = line,
        color: str = "#273142",
        fontsize: float = 8.0,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.004,rounding_size=0.010",
                linewidth=0.8,
                edgecolor=edge,
                facecolor=face,
                transform=ax.transAxes,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            fontweight="bold",
        )

    def small_label(x: float, y: float, label: str, text: str, *, width: int = 38) -> None:
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.1,
            fontweight="bold",
            color=ink,
        )
        ax.text(
            x + 0.078,
            y,
            wrap(text, width),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.1,
            color=muted,
            linespacing=1.15,
        )

    fig.suptitle(
        "Supplementary Figure S4A. What Bio MAF v4 features mean",
        x=0.040,
        y=0.975,
        ha="left",
        fontsize=19.0,
        fontweight="bold",
        color=ink,
    )
    fig.text(
        0.040,
        0.925,
        "Bio MAF turns a patient's mutation file plus fixed cancer references into five plain-language groups of features.",
        fontsize=10.8,
        color=muted,
        va="top",
    )
    fig.text(
        0.040,
        0.895,
        f"Strict Bio MAF core: {core_total:,} features. Optional mutation-count controls add {optional_count:,}, for {full_total:,} total candidate Bio MAF features.",
        fontsize=9.6,
        color="#41506A",
        va="top",
    )

    # Left: what the method is allowed to read.
    add_card(
        0.040,
        0.595,
        0.315,
        0.250,
        "1. Patient mutation file",
        "A MAF is a tumor mutation table. Bio MAF reads only fields available from that file.",
        face=neutral_face,
        edge=line,
        title_size=11.0,
        body_size=8.4,
        wrap_width=44,
    )
    maf_chips = [
        ("gene changed", 0.062, 0.705, 0.090),
        ("mutation type", 0.160, 0.705, 0.094),
        ("gene effect", 0.262, 0.705, 0.072),
        ("protein site", 0.062, 0.650, 0.082),
        ("allele fraction", 0.152, 0.650, 0.104),
        ("hotspot flag", 0.264, 0.650, 0.078),
    ]
    for text, x, y, w in maf_chips:
        chip(x, y, text, w=w)
    ax.text(
        0.062,
        0.615,
        "Allele fraction means the share of tumor DNA reads carrying that mutation.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.1,
        color=muted,
    )

    add_card(
        0.040,
        0.235,
        0.315,
        0.310,
        "2. Fixed cancer references",
        "These sources are vendored and fixed before model training. They are not learned from outcome labels.",
        face=neutral_face,
        edge=line,
        title_size=11.0,
        body_size=8.3,
        wrap_width=44,
    )
    sources = [
        ("Ensembl VEP", "predicts the likely gene effect"),
        ("OncoKB", "clinically curated cancer genes"),
        ("COSMIC CGC", "cancer gene roles"),
        ("IntOGen + Vogelstein", "independent cancer-gene lists"),
        ("Cancer Hotspots", "recurrent cancer mutation sites"),
    ]
    y0 = 0.405
    for idx, (name, meaning) in enumerate(sources):
        y = y0 - idx * 0.039
        chip(0.062, y, name, w=0.124, h=0.032, face="#F3F6FA", fontsize=7.2)
        ax.text(
            0.198,
            y + 0.016,
            meaning,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.8,
            color=muted,
        )

    # Connection arrow.
    ax.add_patch(
        FancyArrowPatch(
            (0.365, 0.520),
            (0.405, 0.520),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.3,
            color="#8B95A7",
        )
    )
    ax.text(
        0.365,
        0.545,
        "converted into",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#6F768A",
    )

    # Right: the actual Bio MAF feature groups.
    ax.text(
        0.420,
        0.842,
        "3. Five types of Bio MAF inputs",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13.5,
        fontweight="bold",
        color=ink,
    )
    ax.text(
        0.420,
        0.805,
        "Each card says what is measured, where it comes from, and why it belongs in the biology feature set.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.9,
        color=muted,
    )

    feature_cards = [
        (
            "Basic mutation details",
            f"{compact_count:,}",
            "From: mutation file + Ensembl VEP\nCount: 100 effect summaries + 21 role/allele summaries.\nMeans: what kinds of mutations are present and how large they look.\nWhy: broad biology without choosing genes.",
            blue_face,
            blue_edge,
            "#FFFFFF",
        ),
        (
            "Cancer-gene support",
            f"{tier_count:,}",
            "From: cancer gene references\nCount: 4 support levels x 4 named measurements.\nMeans: whether mutated genes appear in trusted cancer-gene lists.\nWhy: externally fixed consensus signal.",
            blue_face,
            blue_edge,
            "#FFFFFF",
        ),
        (
            "Specific cancer genes",
            f"{exact_count:,}",
            f"From: fixed {panel_count:,}-gene cancer panel\nCount: {panel_count:,} genes x 2 named measurements.\nMeans: which exact known cancer genes changed.\nWhy: exact genes can matter.",
            blue_face,
            blue_edge,
            "#FFFFFF",
        ),
        (
            "Recurrent cancer sites",
            f"{hotspot_count:,}",
            "From: Cancer Hotspots\nCount: 4 hotspot summaries.\nMeans: mutations at recurrent cancer sites.\nWhy: marks changes likely to help cancer grow.",
            blue_face,
            blue_edge,
            "#FFFFFF",
        ),
        (
            "Optional mutation-count controls",
            f"{optional_count:,}",
            "From: simple counts in the mutation file\nCount: 5 mutation-load controls.\nMeans: mutation load and broad mutation-type mix.\nWhy: useful controls, kept separate from the strict core.",
            gold_face,
            gold_edge,
            "#FFFFFF",
        ),
    ]
    card_positions = [
        (0.420, 0.620, 0.260, 0.155, 34),
        (0.700, 0.620, 0.260, 0.155, 34),
        (0.420, 0.405, 0.260, 0.165, 34),
        (0.700, 0.405, 0.260, 0.165, 34),
        (0.420, 0.185, 0.540, 0.170, 76),
    ]
    for (title, badge, body, face, edge, badge_face), (card_x, card_y, card_w, card_h, wrap_width) in zip(feature_cards, card_positions):
        add_card(
            card_x,
            card_y,
            card_w,
            card_h,
            title,
            body,
            face=face,
            edge=edge,
            title_size=10.6,
            body_size=8.05,
            wrap_width=wrap_width,
            badge=f"{badge} features",
            badge_face=badge_face,
            badge_edge=edge,
        )

    # Bottom: short guardrails without making them the main reading task.
    ax.add_patch(
        FancyBboxPatch(
            (0.040, 0.050),
            0.920,
            0.105,
            boxstyle="round,pad=0.010,rounding_size=0.014",
            linewidth=1.0,
            edgecolor=line,
            facecolor=soft_face,
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.062,
        0.122,
        "Important separations",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=ink,
    )
    bottom_notes = [
        ("Signatures are separate.", "COSMIC mutational signatures are not counted as Bio MAF features."),
        ("Selection is inside training folds.", "Nested CV chooses among pre-declared Bio MAF groups without using held-out labels."),
        ("Details live in tables.", "Table S5 defines the feature groups; Table S6 reports which groups were selected."),
    ]
    note_xs = [0.062, 0.360, 0.690]
    for (label, text), x in zip(bottom_notes, note_xs):
        ax.text(
            x,
            0.097,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color=ink,
        )
        ax.text(
            x,
            0.072,
            wrap(text, 42),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.9,
            color=muted,
            linespacing=1.15,
        )

    _save_figure(fig, stem)


def _figure_bio_maf_v4_feature_map_detailed_table_v1(stem: Path) -> None:
    guide, _exact = _bio_maf_v4_feature_guide()
    if guide.empty:
        return

    counts = {str(row["Feature block"]): int(row["Features"]) for _, row in guide.iterrows()}
    full_total = int(pd.to_numeric(guide["Full feature total including optional controls"], errors="coerce").dropna().max())
    core_total = int(pd.to_numeric(guide["Core feature total"], errors="coerce").dropna().max())
    compact_count = counts.get("Compact biological annotation core", 121)
    tier_count = counts.get("External driver-evidence tiers", 16)
    exact_count = counts.get("Exact consensus driver-gene identity", 802)
    hotspot_count = counts.get("Cancer hotspot summaries", 4)
    optional_count = counts.get("Optional burden and event-composition controls", 5)
    panel_count = int(exact_count / 2) if exact_count else 401

    fig, ax = plt.subplots(figsize=(16, 9.0))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FBFCFE")

    ink = "#1F2430"
    muted = "#5C6678"
    pale = "#F8FAFC"
    line = "#CBD5E1"
    blue = "#4F73BF"
    blue_pale = "#EEF5FF"
    gold = "#B89D2C"
    gold_pale = "#FFF9E6"
    green = "#2F7A67"
    green_pale = "#EFF8F5"

    def wrap(text: str, width: int) -> str:
        return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False, replace_whitespace=False))

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        face: str = "#FFFFFF",
        edge: str = line,
        lw: float = 1.0,
        radius: float = 0.012,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0.007,rounding_size={radius}",
                linewidth=lw,
                edgecolor=edge,
                facecolor=face,
                transform=ax.transAxes,
            )
        )

    def text(
        x: float,
        y: float,
        value: str,
        *,
        size: float = 8.5,
        color: str = muted,
        weight: str = "normal",
        width: int | None = None,
        va: str = "top",
        ha: str = "left",
        linespacing: float = 1.14,
    ) -> None:
        rendered = wrap(value, width) if width else value
        ax.text(
            x,
            y,
            rendered,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=size,
            color=color,
            fontweight=weight,
            linespacing=linespacing,
        )

    def step_card(x: float, label: str, title: str, body: str, *, face: str, edge: str) -> None:
        box(x, 0.705, 0.282, 0.150, face=face, edge=edge, lw=1.15)
        ax.text(
            x + 0.020,
            0.823,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=9.4,
            fontweight="bold",
            color="#FFFFFF",
            bbox={"boxstyle": "round,pad=0.32", "facecolor": edge, "edgecolor": edge},
        )
        text(x + 0.065, 0.837, title, size=10.7, color=ink, weight="bold", width=26)
        text(x + 0.020, 0.780, body, size=8.2, color=muted, width=46)

    fig.suptitle(
        "Supplementary Figure S4A. What Goes Into Bio MAF v4",
        x=0.045,
        y=0.965,
        ha="left",
        fontsize=18.8,
        fontweight="bold",
        color=ink,
    )
    text(
        0.045,
        0.915,
        "A tumor mutation file plus fixed cancer references becomes five feature groups. Counts, sources, and meanings are shown in the table.",
        size=10.2,
        color=muted,
    )
    text(
        0.045,
        0.885,
        f"Strict Bio MAF core: {core_total:,} features. Optional controls add {optional_count:,}, for {full_total:,} possible Bio MAF features.",
        size=9.3,
        color="#41506A",
    )

    step_card(
        0.045,
        "1",
        "Patient mutation file",
        "MAF means Mutation Annotation Format: a table of mutations found in one tumor.",
        face=pale,
        edge="#7A8798",
    )
    step_card(
        0.362,
        "2",
        "Fixed outside references",
        "Ensembl VEP, OncoKB, COSMIC CGC, IntOGen, Vogelstein genes, and Cancer Hotspots.",
        face=green_pale,
        edge=green,
    )
    step_card(
        0.679,
        "3",
        "Bio MAF feature groups",
        "The model receives grouped measurements, not a post-hoc list chosen from the endpoint labels.",
        face=blue_pale,
        edge=blue,
    )
    for x1, x2 in [(0.330, 0.360), (0.647, 0.677)]:
        ax.add_patch(
            FancyArrowPatch(
                (x1, 0.780),
                (x2, 0.780),
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=17,
                linewidth=1.2,
                color="#8B95A7",
            )
        )

    text(
        0.055,
        0.645,
        "Mutation-file fields used: gene changed, mutation type, protein site, predicted gene effect, allele fraction, and known recurrent cancer site.",
        size=8.7,
        color=ink,
        weight="bold",
        width=150,
    )
    text(
        0.055,
        0.618,
        "Feature values: counts = log1p(number matching); fractions = matching / total annotated; allele fraction = share of tumor DNA reads carrying the mutation.",
        size=7.9,
        color=muted,
    )

    # Main feature table.
    table_x = 0.045
    table_y = 0.130
    table_w = 0.910
    header_h = 0.052
    row_h = 0.082
    col = {
        "group": table_x + 0.020,
        "count": table_x + 0.270,
        "source": table_x + 0.440,
        "meaning": table_x + 0.650,
    }
    box(table_x, table_y, table_w, header_h + 5 * row_h, face="#FFFFFF", edge=line, lw=1.05, radius=0.012)
    box(table_x, table_y + 5 * row_h, table_w, header_h, face="#F3F6FA", edge=line, lw=1.05, radius=0.012)
    text(col["group"], table_y + 5 * row_h + 0.034, "Feature group", size=8.4, color=ink, weight="bold", va="center")
    text(col["count"], table_y + 5 * row_h + 0.034, "Count math", size=8.4, color=ink, weight="bold", va="center")
    text(col["source"], table_y + 5 * row_h + 0.034, "Where it comes from", size=8.4, color=ink, weight="bold", va="center")
    text(col["meaning"], table_y + 5 * row_h + 0.034, "What it means in plain English", size=8.4, color=ink, weight="bold", va="center")

    rows = [
        {
            "group": "Basic mutation details",
            "n": f"{compact_count:,}",
            "count": "100 = 50 fixed VEP labels x 2 values; +5 allele-fraction summaries; +12 OncoKB role summaries; +4 non-OncoKB cancer-gene summaries",
            "source": "MAF fields; Ensembl VEP effect labels; OncoKB cancer-gene roles",
            "meaning": "Fixed annotation vocabulary, not selected from labels. Captures mutation type, predicted effect, allele fraction, and cancer-gene role.",
            "face": blue_pale,
            "edge": blue,
        },
        {
            "group": "External evidence-confidence controls",
            "n": f"{tier_count:,}",
            "count": "16 = 4 evidence-confidence groups x 4 mutation/allele-fraction measures",
            "source": "Evidence-confidence group = genes named by 1, 2, 3, or 4+ external cancer references",
            "meaning": "For genes supported by 1, 2, 3, or 4+ sources: high/moderate-impact count, high-impact count, unique high/mod genes, and max allele fraction.",
            "face": blue_pale,
            "edge": blue,
        },
        {
            "group": "Specific cancer genes",
            "n": f"{exact_count:,}",
            "count": f"{panel_count:,} fixed genes x 2 log-count features = {exact_count:,}",
            "source": "Genes fixed by rule: >=3 sources among OncoKB/COSMIC/IntOGen/Vogelstein, or Cancer Hotspots gene",
            "meaning": "For each gene: protein-affecting mutation count, and cancer-role-fit mutation count. Both are log-counts, not averages.",
            "face": blue_pale,
            "edge": blue,
        },
        {
            "group": "Recurrent cancer sites",
            "n": f"{hotspot_count:,}",
            "count": "4 hotspot measurements",
            "source": "Cancer Hotspots",
            "meaning": "Whether mutations hit exact or nearby protein sites repeatedly seen in cancer, how many genes are hit, and the largest hotspot allele fraction.",
            "face": blue_pale,
            "edge": blue,
        },
        {
            "group": "Optional mutation-count controls",
            "n": f"{optional_count:,}",
            "count": "5 mutation-load controls",
            "source": "Simple counts from the MAF",
            "meaning": "Single-base, double-base, and insertion/deletion mutation counts and fractions. Kept separate from the strict core.",
            "face": gold_pale,
            "edge": gold,
        },
    ]
    for idx, row in enumerate(rows):
        y = table_y + (4 - idx) * row_h
        if idx < 4:
            ax.plot([table_x, table_x + table_w], [y, y], transform=ax.transAxes, color="#E2E8F0", linewidth=0.8)
        ax.add_patch(
            Rectangle(
                (table_x, y),
                0.010,
                row_h,
                transform=ax.transAxes,
                facecolor=row["edge"],
                edgecolor=row["edge"],
                linewidth=0,
            )
        )
        ax.text(
            col["group"],
            y + row_h * 0.63,
            row["group"],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.7,
            fontweight="bold",
            color=ink,
        )
        ax.text(
            col["group"],
            y + row_h * 0.30,
            f"{row['n']} features",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.7,
            fontweight="bold",
            color=row["edge"],
        )
        text(col["count"], y + row_h * 0.66, row["count"], size=7.7, color=muted, width=28, va="center")
        text(col["source"], y + row_h * 0.66, row["source"], size=7.7, color=muted, width=30, va="center")
        text(col["meaning"], y + row_h * 0.66, row["meaning"], size=7.7, color=muted, width=52, va="center")

    # Minimal footer guardrails.
    box(0.045, 0.055, 0.910, 0.075, face=pale, edge=line, lw=0.95, radius=0.012)
    footer_items = [
        ("Blue", "strict Bio MAF core"),
        ("Gold", "optional mutation-count controls"),
        ("Separate", "COSMIC mutational signatures are not Bio MAF features"),
        ("Training only", "feature groups are selected inside the training folds"),
    ]
    footer_xs = [0.065, 0.230, 0.410, 0.690]
    footer_colors = [blue, gold, ink, ink]
    for (label, body), x, color in zip(footer_items, footer_xs, footer_colors):
        text(x, 0.105, label, size=8.0, color=color, weight="bold")
        text(x, 0.082, body, size=7.5, color=muted, width=34)

    _save_figure(fig, stem)


def _figure_bio_maf_v4_feature_map(stem: Path) -> None:
    guide, _exact = _bio_maf_v4_feature_guide()
    if guide.empty:
        return

    counts = {str(row["Feature block"]): int(row["Features"]) for _, row in guide.iterrows()}
    full_total = int(pd.to_numeric(guide["Full feature total including optional controls"], errors="coerce").dropna().max())
    core_total = int(pd.to_numeric(guide["Core feature total"], errors="coerce").dropna().max())
    compact_count = counts.get("Compact biological annotation core", 121)
    tier_count = counts.get("External evidence-confidence controls", counts.get("External driver-evidence tiers", 16))
    exact_count = counts.get("Exact consensus driver-gene identity", 802)
    hotspot_count = counts.get("Cancer hotspot summaries", 4)
    optional_count = counts.get("Optional burden and event-composition controls", 5)
    panel_count = int(exact_count / 2) if exact_count else 401

    fig, ax = plt.subplots(figsize=(16, 12.2))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FBFCFE")

    ink = "#1F2430"
    muted = "#5C6678"
    line = "#CBD5E1"
    pale = "#F8FAFC"
    blue = "#4F73BF"
    blue_pale = "#EEF5FF"
    gold = "#B89D2C"
    gold_pale = "#FFF9E6"
    green = "#2F7A67"
    green_pale = "#EFF8F5"
    gray = "#7A8798"

    def wrap(value: str, width: int) -> str:
        return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False, replace_whitespace=False))

    def box(x: float, y: float, w: float, h: float, *, face: str = "#FFFFFF", edge: str = line, lw: float = 1.0) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.014",
                linewidth=lw,
                edgecolor=edge,
                facecolor=face,
                transform=ax.transAxes,
            )
        )

    def label(x: float, y: float, value: str, *, size: float = 8.0, color: str = muted, weight: str = "normal", width: int | None = None, va: str = "top", ha: str = "left") -> None:
        ax.text(
            x,
            y,
            wrap(value, width) if width else value,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=size,
            color=color,
            fontweight=weight,
            linespacing=1.13,
        )

    def step(x: float, n: str, title: str, body: str, *, edge: str, face: str) -> None:
        box(x, 0.745, 0.282, 0.110, face=face, edge=edge, lw=1.1)
        ax.text(
            x + 0.018,
            0.830,
            n,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#FFFFFF",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": edge, "edgecolor": edge},
        )
        label(x + 0.052, 0.840, title, size=10.2, color=ink, weight="bold", width=28)
        label(x + 0.020, 0.795, body, size=7.8, color=muted, width=48)

    fig.suptitle(
        "Supplementary Figure S4A. Bio MAF v4 Feature Groups, Without the Jargon",
        x=0.045,
        y=0.965,
        ha="left",
        fontsize=18.2,
        fontweight="bold",
        color=ink,
    )
    label(
        0.045,
        0.920,
        "Bio MAF v4 converts an annotated tumor mutation file into fixed, predeclared feature groups. Table S5 gives worked examples, formulas, and definitions.",
        size=10.0,
        color=muted,
    )
    label(
        0.045,
        0.892,
        f"Strict Bio MAF core: {core_total:,} features. Optional controls add {optional_count:,}, for {full_total:,} possible Bio MAF features.",
        size=9.0,
        color="#41506A",
    )

    step(
        0.045,
        "1",
        "Mutation file",
        "For each tumor: gene changed, mutation type, protein site, predicted effect, and allele fraction.",
        edge=gray,
        face=pale,
    )
    step(
        0.362,
        "2",
        "Fixed references",
        "External resources are vendored and fixed before modeling; outcome labels are not used to create them.",
        edge=green,
        face=green_pale,
    )
    step(
        0.679,
        "3",
        "Feature groups",
        "Training folds choose among predeclared groups; held-out folds only evaluate performance.",
        edge=blue,
        face=blue_pale,
    )
    for x1, x2 in [(0.330, 0.360), (0.647, 0.677)]:
        ax.add_patch(
            FancyArrowPatch(
                (x1, 0.800),
                (x2, 0.800),
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.15,
                color="#8B95A7",
            )
        )

    table_x, table_y, table_w = 0.045, 0.185, 0.910
    header_h, row_h = 0.052, 0.098
    box(table_x, table_y, table_w, header_h + 5 * row_h, face="#FFFFFF", edge=line, lw=1.0)
    box(table_x, table_y + 5 * row_h, table_w, header_h, face="#F3F6FA", edge=line, lw=1.0)

    col_group = table_x + 0.018
    col_count = table_x + 0.235
    col_example = table_x + 0.420
    col_worked = table_x + 0.610
    label(col_group, table_y + 5 * row_h + 0.032, "Feature group", size=8.2, color=ink, weight="bold", va="center")
    label(col_count, table_y + 5 * row_h + 0.032, "Count derivation", size=8.2, color=ink, weight="bold", va="center")
    label(col_example, table_y + 5 * row_h + 0.032, "Example feature", size=8.2, color=ink, weight="bold", va="center")
    label(col_worked, table_y + 5 * row_h + 0.032, "Worked example: toy MAF -> feature value", size=8.2, color=ink, weight="bold", va="center")

    rows = [
        (
            "Mutation-effect and allele-fraction basics",
            f"{compact_count:,} features",
            "100 = 50 fixed VEP labels x 2 values; +5 allele-fraction summaries; +12 OncoKB role summaries; +4 non-OncoKB cancer-gene summaries.",
            "`vep_consequence_count__\nmissense_variant`",
            "Toy tumor: 3 missense mutations among 7 annotated mutations. Count = log1p(3) = 1.386. Paired fraction = 3 / 7 = 0.429.",
            blue,
        ),
        (
            "External evidence-confidence controls",
            f"{tier_count:,} features",
            "16 = 4 groups x 4 measures. Groups are genes named by 1, 2, 3, or 4+ external cancer references.",
            "`driver_evidence_high_or_moderate_\nimpact_log_count__source_count_4`",
            "Toy tumor: one TP53 and one BRCA1 high/moderate-impact mutation. Both genes are in the 4-source group. Value = log1p(2) = 1.099.",
            blue,
        ),
        (
            "Fixed cancer-gene identity",
            f"{exact_count:,} features",
            f"{exact_count:,} = {panel_count:,} externally fixed cancer genes x 2 log-counts: protein-affecting + cancer-role-fit.",
            "`driver_gene_functional_event_log_count__TP53`\n`driver_gene_role_matched_event_log_count__TP53`",
            "Toy tumor: one TP53 frameshift and one TP53 synonymous mutation. Frameshift counts; synonymous does not. Each TP53 example value = log1p(1) = 0.693.",
            blue,
        ),
        (
            "Known recurrent cancer sites",
            f"{hotspot_count:,} features",
            "4 hotspot measures.",
            "`cancer_hotspot_residue_log_count`\n`cancer_hotspot_max_vaf`",
            "Toy tumor: two recurrent hotspot-residue hits with allele fractions 0.21 and 0.47. Count = log1p(2) = 1.099; max allele fraction = 0.47.",
            blue,
        ),
        (
            "Optional mutation-count controls",
            f"{optional_count:,} features",
            "5 mutation-load and event-mix controls.",
            "`log10_id_burden`\n`id_fraction`",
            "Toy tumor: 80 single-base, 2 double-base, and 18 insertion/deletion events. Burden = log10(18 + 1) = 1.279; fraction = 18 / 100 = 0.180.",
            gold,
        ),
    ]
    for idx, (group, n, count_text, example, worked, edge) in enumerate(rows):
        y = table_y + (4 - idx) * row_h
        if idx < 4:
            ax.plot([table_x, table_x + table_w], [y, y], transform=ax.transAxes, color="#E2E8F0", linewidth=0.8)
        ax.add_patch(Rectangle((table_x, y), 0.010, row_h, transform=ax.transAxes, facecolor=edge, edgecolor=edge, linewidth=0))
        label(col_group, y + row_h * 0.65, group, size=8.3, color=ink, weight="bold", width=32, va="center")
        label(col_group, y + row_h * 0.28, n, size=7.5, color=edge, weight="bold", va="center")
        label(col_count, y + row_h * 0.60, count_text, size=6.8, color=muted, width=31, va="center")
        label(col_example, y + row_h * 0.60, example, size=6.8, color=ink, width=30, va="center")
        label(col_worked, y + row_h * 0.60, worked, size=6.8, color=muted, width=58, va="center")

    # Jargon decoder.
    box(0.045, 0.015, 0.910, 0.160, face=pale, edge=line, lw=0.95)
    label(0.065, 0.156, "Jargon decoder", size=9.1, color=ink, weight="bold")
    glossary = [
        ("VEP label", "Standard Ensembl effect term assigned to a mutation."),
        ("Allele fraction", "Share of tumor DNA reads carrying the mutation."),
        ("Log-count", "log1p(number of matching mutations), not an average."),
        ("Protein-affecting", "VEP rules say the mutation likely changes/disrupts protein."),
        ("Cancer-role-fit", "Mutation type fits the gene's known cancer role."),
        ("Fixed before training", "Created before model training; no endpoint labels used."),
    ]
    gx = [0.065, 0.205, 0.350, 0.495, 0.650, 0.805]
    gw = [0.118, 0.122, 0.118, 0.128, 0.128, 0.118]
    for (term, definition), x, width in zip(glossary, gx, gw):
        label(x, 0.116, term, size=7.3, color=ink, weight="bold", width=18)
        label(x, 0.088, definition, size=6.8, color=muted, width=int(width * 150))

    _save_figure(fig, stem)


BIO_MAF_DISTRIBUTION_FEATURES: list[dict[str, str]] = [
    {
        "feature": "vep_consequence_count__missense_variant",
        "short": "Missense mutation count",
        "family": "Compact VEP annotation",
        "plain": "log1p count of mutations where VEP says one amino acid changed.",
    },
    {
        "feature": "vep_impact_fraction__high",
        "short": "High-impact fraction",
        "family": "Compact VEP annotation",
        "plain": "fraction of annotated mutations with VEP HIGH impact.",
    },
    {
        "feature": "vaf_max",
        "short": "Maximum allele fraction",
        "family": "Allele-fraction summary",
        "plain": "largest share of tumor DNA reads carrying any mutation in the sample.",
    },
    {
        "feature": "driver_evidence_high_or_moderate_impact_log_count__source_count_4",
        "short": "4-source cancer-gene hits",
        "family": "Evidence-confidence control",
        "plain": "log1p count of high/moderate-impact mutations in genes named by 4+ external cancer resources.",
    },
    {
        "feature": "driver_gene_functional_event_log_count__TP53",
        "short": "TP53 protein-affecting count",
        "family": "Exact cancer-gene identity",
        "plain": "log1p count of TP53 mutations predicted to change or disrupt TP53 protein.",
    },
    {
        "feature": "driver_gene_role_matched_event_log_count__TP53",
        "short": "TP53 cancer-role-fit count",
        "family": "Exact cancer-gene identity",
        "plain": "log1p count of TP53 mutations fitting TP53's tumor-suppressor role.",
    },
    {
        "feature": "cancer_hotspot_residue_log_count",
        "short": "Hotspot residue count",
        "family": "Cancer hotspot summary",
        "plain": "log1p count of mutations hitting recurrent Cancer Hotspots protein residues.",
    },
    {
        "feature": "cancer_hotspot_max_vaf",
        "short": "Maximum hotspot allele fraction",
        "family": "Cancer hotspot summary",
        "plain": "largest allele fraction among hotspot-matched mutations.",
    },
    {
        "feature": "log10_id_burden",
        "short": "Insertion/deletion burden",
        "family": "Optional burden control",
        "plain": "log10(1 + insertion/deletion event count).",
    },
    {
        "feature": "id_fraction",
        "short": "Insertion/deletion fraction",
        "family": "Optional burden control",
        "plain": "insertion/deletion events divided by SBS + DBS + ID events.",
    },
]


def _bio_maf_top20_labels() -> pd.DataFrame:
    oof_path = BUNDLE_ROOT / "results" / "tables" / "main_manuscript_complete_panel_oof_predictions.csv"
    if not oof_path.exists():
        return pd.DataFrame(columns=["sample", "cancer_type_top20"])
    label_cols = [f"class_label_{idx}" for idx in range(20)]
    needed = {"endpoint", "representation", "learner", "sample", "true_value", *label_cols}
    oof = pd.read_csv(oof_path, usecols=lambda column: column in needed, low_memory=False)
    rows = oof[
        oof["endpoint"].astype(str).eq("cancer_type_top20")
        & oof["representation"].astype(str).eq("MAF_stack_only")
        & oof["learner"].astype(str).str.lower().eq("xgboost")
    ].copy()
    if rows.empty:
        rows = oof[oof["endpoint"].astype(str).eq("cancer_type_top20")].copy()
    if rows.empty:
        return pd.DataFrame(columns=["sample", "cancer_type_top20"])

    def true_label(row: pd.Series) -> str:
        idx = pd.to_numeric(row.get("true_value"), errors="coerce")
        if pd.isna(idx):
            return ""
        column = f"class_label_{int(idx)}"
        return str(row.get(column, "") or "")

    rows["cancer_type_top20"] = rows.apply(true_label, axis=1)
    labels = rows[["sample", "cancer_type_top20"]].dropna().drop_duplicates("sample")
    labels = labels[labels["cancer_type_top20"].astype(str).ne("")]
    return labels.reset_index(drop=True)


def _load_bio_maf_v4_top20_distribution_frame() -> tuple[pd.DataFrame, Path | None]:
    manifest_path = BUNDLE_ROOT / "results" / "tables" / "main_manuscript_complete_panel_feature_manifest.csv"
    labels = _bio_maf_top20_labels()
    if labels.empty or not manifest_path.exists():
        return pd.DataFrame(), None
    manifest = pd.read_csv(manifest_path)
    match = manifest[
        manifest["benchmark"].astype(str).eq("mc3_main")
        & manifest["representation"].astype(str).eq("MAF_stack_only")
    ].copy()
    if match.empty:
        return pd.DataFrame(), None
    cache_key = str(match.iloc[0]["cache_key"])
    feature_path = BUNDLE_ROOT / "results" / "cache" / "features" / cache_key / "features.csv.gz"
    if not feature_path.exists():
        return pd.DataFrame(), None
    requested = {"sample", *[item["feature"] for item in BIO_MAF_DISTRIBUTION_FEATURES]}
    features = pd.read_csv(feature_path, usecols=lambda column: column in requested)
    keep = ["sample"] + [item["feature"] for item in BIO_MAF_DISTRIBUTION_FEATURES if item["feature"] in features.columns]
    features = features.loc[:, keep].copy()
    merged = labels.merge(features, on="sample", how="inner")
    return merged, feature_path


def _write_bio_maf_v4_distribution_summary(frame: pd.DataFrame, technical_dir: Path, source_path: Path | None) -> None:
    rows: list[dict[str, object]] = []
    for item in BIO_MAF_DISTRIBUTION_FEATURES:
        feature = item["feature"]
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "Feature": feature,
                "Display name": item["short"],
                "Feature family": item["family"],
                "Plain-English meaning": item["plain"],
                "N top-20 samples": int(values.shape[0]),
                "Nonzero samples": int((values > 0).sum()),
                "Nonzero percent": round(float((values > 0).mean() * 100.0), 2),
                "Mean": float(values.mean()),
                "Median": float(values.median()),
                "P25": float(values.quantile(0.25)),
                "P75": float(values.quantile(0.75)),
                "P95": float(values.quantile(0.95)),
                "Max": float(values.max()),
                "Source feature matrix": str(source_path.relative_to(BUNDLE_ROOT)) if source_path else "",
            }
        )
    fields = [
        "Feature",
        "Display name",
        "Feature family",
        "Plain-English meaning",
        "N top-20 samples",
        "Nonzero samples",
        "Nonzero percent",
        "Mean",
        "Median",
        "P25",
        "P75",
        "P95",
        "Max",
        "Source feature matrix",
    ]
    summary = pd.DataFrame(rows, columns=fields)
    _write_table_csv_html(
        summary,
        technical_dir / "table_s4b_bio_maf_v4_feature_distribution_summary.csv",
        title="Supplementary Figure S4B technical feature-distribution summary",
    )


def _figure_bio_maf_v4_feature_distributions(stem: Path, technical_dir: Path) -> None:
    frame, source_path = _load_bio_maf_v4_top20_distribution_frame()
    _write_bio_maf_v4_distribution_summary(frame, technical_dir, source_path)
    if frame.empty:
        return

    available = [item for item in BIO_MAF_DISTRIBUTION_FEATURES if item["feature"] in frame.columns]
    if not available:
        return
    fig, axes = plt.subplots(2, 5, figsize=(17, 8.4))
    axes = np.ravel(axes)
    fig.patch.set_facecolor("#FBFCFE")
    for ax, item in zip(axes, available):
        values = pd.to_numeric(frame[item["feature"]], errors="coerce").dropna()
        finite = values[np.isfinite(values)]
        ax.set_facecolor("#FFFFFF")
        ax.hist(finite, bins=34, color="#4F73BF", alpha=0.82, edgecolor="#FFFFFF", linewidth=0.6)
        median = float(finite.median()) if not finite.empty else np.nan
        p95 = float(finite.quantile(0.95)) if not finite.empty else np.nan
        nonzero_pct = float((finite > 0).mean() * 100.0) if not finite.empty else np.nan
        if np.isfinite(median):
            ax.axvline(median, color="#1F2430", linewidth=1.1)
        ax.set_title(item["short"], fontsize=10.0, fontweight="bold", loc="left", pad=7, color="#1F2430")
        ax.text(
            0.98,
            0.92,
            f"nonzero {nonzero_pct:.1f}%\nmedian {median:.3g}\np95 {p95:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.4,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1", "linewidth": 0.6},
        )
        ax.tick_params(axis="both", labelsize=7.5, colors="#475569")
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.6)
        ax.set_xlabel("Feature value", fontsize=7.5, color="#475569")
        ax.set_ylabel("Samples", fontsize=7.5, color="#475569")
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
            spine.set_linewidth(0.7)
    for ax in axes[len(available) :]:
        ax.axis("off")
    fig.suptitle(
        "Supplementary Figure S4B. Distribution of Representative Bio MAF v4 Features in TCGA Top-20 Samples",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=15.5,
        fontweight="bold",
        color="#1F2430",
    )
    fig.text(
        0.02,
        0.945,
        f"Each panel shows one real Bio MAF v4 feature across {len(frame):,} fixed top-20 TCGA cancer-type samples. "
        "Many biological event features are intentionally zero-inflated because most tumors do not hit a given gene or hotspot.",
        ha="left",
        va="top",
        fontsize=9.2,
        color="#5C6678",
    )
    fig.tight_layout(rect=[0.015, 0.02, 0.995, 0.915], h_pad=1.1, w_pad=1.0)
    _save_figure(fig, stem)


def _figure_bio_maf_v4_feature_distributions_by_cancer_type(stem: Path) -> None:
    frame, _source_path = _load_bio_maf_v4_top20_distribution_frame()
    if frame.empty:
        return
    heatmap_features = [
        "vep_consequence_count__missense_variant",
        "driver_evidence_high_or_moderate_impact_log_count__source_count_4",
        "driver_gene_functional_event_log_count__TP53",
        "driver_gene_role_matched_event_log_count__TP53",
        "cancer_hotspot_residue_log_count",
        "log10_id_burden",
    ]
    display = {item["feature"]: item["short"] for item in BIO_MAF_DISTRIBUTION_FEATURES}
    heatmap_features = [feature for feature in heatmap_features if feature in frame.columns]
    if not heatmap_features:
        return
    grouped = []
    for cancer_type, sub in frame.groupby("cancer_type_top20", sort=True):
        row = {"Cancer type": cancer_type, "N": int(len(sub))}
        for feature in heatmap_features:
            values = pd.to_numeric(sub[feature], errors="coerce")
            row[feature] = float((values > 0).mean() * 100.0)
        grouped.append(row)
    matrix = pd.DataFrame(grouped).set_index("Cancer type")
    counts = matrix.pop("N")
    matrix = matrix.sort_values("vep_consequence_count__missense_variant", ascending=False)
    counts = counts.reindex(matrix.index)
    values = matrix.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(13.5, 9.5))
    fig.patch.set_facecolor("#FBFCFE")
    im = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0, vmax=max(1.0, float(np.nanmax(values))))
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([display.get(feature, feature) for feature in matrix.columns], rotation=28, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([f"{idx} (n={int(counts.loc[idx])})" for idx in matrix.index], fontsize=8.5)
    fig.suptitle(
        "Supplementary Figure S4C. Nonzero Rates for Representative Bio MAF v4 Features by Cancer Type",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=13.4,
        fontweight="bold",
        color="#1F2430",
    )
    fig.text(
        0.02,
        0.945,
        "Cell values are the percent of tumors in that cancer type with a nonzero value for the feature. "
        "This emphasizes sparsity and cancer-type concentration for gene and hotspot features.",
        ha="left",
        va="top",
        fontsize=9.0,
        color="#5C6678",
    )
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            color = "#FFFFFF" if value > np.nanmax(values) * 0.55 else "#1F2430"
            ax.text(x, y, f"{value:.0f}%", ha="center", va="center", fontsize=7.3, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.018)
    cbar.set_label("Percent of tumors with nonzero feature value", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(rect=[0.02, 0.02, 0.985, 0.900])
    _save_figure(fig, stem)


def _figure_bio_maf_v4_nested_selection(stem: Path) -> None:
    _summary, by_fold = _bio_maf_v4_nested_selection_tables()
    if by_fold.empty:
        return
    plot = by_fold[by_fold["Endpoint tier"].eq("Main")].copy()
    if plot.empty:
        plot = by_fold.copy()
    base_labels = {
        "": "No Bio MAF\n(signatures only)",
        "v4_compact_biology_only": "Compact\nbiology only",
        "v4_compact_plus_driver_evidence_tiers": "Compact +\nevidence controls",
        "v4_compact_plus_hotspot_summary": "Compact +\nhotspots",
        "v4_compact_plus_exact_driver_genes": "Compact +\nexact driver genes",
        "v4_full_core": "Full strict\ncore",
        "v4_full_core_plus_optional_controls": "Full core +\noptional controls",
    }
    plot["Selected block"] = plot["Selected Bio MAF block ID"].map(base_labels).fillna(plot["Selected feature-set label"])
    plot["Model group"] = plot["Representation"].astype(str) + "\n" + plot["Model"].astype(str)
    row_order = [label for label in base_labels.values() if label in set(plot["Selected block"])]
    column_order = [
        "Bio MAF v4\nElastic net",
        "Bio MAF v4\nXGBoost",
        "Signatures + Bio MAF v4\nElastic net",
        "Signatures + Bio MAF v4\nXGBoost",
        "Bio MAF v4\nCoxNet",
        "Signatures + Bio MAF v4\nCoxNet",
    ]
    column_order = [value for value in column_order if value in set(plot["Model group"])] + [
        value for value in _ordered_unique(plot["Model group"].tolist()) if value not in column_order
    ]
    pivot = (
        plot.pivot_table(index="Selected block", columns="Model group", values="Outer fold", aggfunc="count", fill_value=0)
        .reindex(index=row_order, columns=column_order, fill_value=0)
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8.5, 1.45 * len(pivot.columns)), max(5.6, 0.72 * len(pivot.index) + 2.2)))
    matrix = pivot.to_numpy(dtype=float)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(5, int(np.nanmax(matrix))))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = int(pivot.iat[i, j])
            ax.text(j, i, str(value) if value else "", ha="center", va="center", fontsize=10, color="#1F2430")
    ax.set_xlabel("Representation and model")
    ax.set_ylabel("Feature block selected by inner-loop validation")
    fig.suptitle(
        "Supplementary Figure S5. Nested-CV Bio MAF v4 Feature-Block Selections",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color="#1F2430",
    )
    fig.text(
        0.02,
        0.925,
        "Cell values are outer-fold counts for manuscript endpoints with Bio MAF v4 feature-block tuning; selection used training data only.",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#6F768A",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.84)
    cbar.set_label("Outer folds selected")
    ax.grid(False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    _save_figure(fig, stem)


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
        if "endpoint" in frame.columns:
            frame = frame[~frame["endpoint"].astype(str).eq("os_event")].copy()
        if frame.empty:
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


def _endpoint_result_experiment_id(row: pd.Series) -> str:
    experiment_id = _text(row, ["experiment_id"], "")
    if experiment_id:
        return experiment_id
    bundle_table = _text(row, ["bundle_table"], "")
    if bundle_table:
        return re.sub(r"_endpoint_results.*$", "", bundle_table)
    return ""


def _filter_enabled_endpoint_results(endpoint_results: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    configured = set((settings.get("experiments") or {}).keys())
    if endpoint_results.empty:
        return endpoint_results
    enabled_ids = ACTIVE_EXPERIMENT_IDS if not configured else {experiment_id for experiment_id in configured if experiment_enabled(settings, experiment_id)}
    source_ids = endpoint_results.apply(_endpoint_result_experiment_id, axis=1)
    keep = source_ids.isin(enabled_ids & ACTIVE_EXPERIMENT_IDS)
    filtered = endpoint_results.loc[keep].copy()
    return _filter_current_muat_outputs(filtered, settings)


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_")


def _muat_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return dict(((settings.get("experiments") or {}).get("muat_style_tcga_comparator") or {}))


def _configured_muat_output_prefix(settings: dict[str, Any]) -> str:
    tag = _safe_name(_muat_settings(settings).get("output_tag"))
    return "muat_style_tcga_comparator" if not tag else f"muat_style_tcga_comparator_{tag}"


def _accepted_muat_output_prefixes(settings: dict[str, Any]) -> list[str]:
    prefixes = [
        _configured_muat_output_prefix(settings),
        "muat_style_tcga_comparator_main_endpoints_comparable",
    ]
    out: list[str] = []
    seen: set[str] = set()
    for prefix in prefixes:
        if prefix and prefix not in seen:
            out.append(prefix)
            seen.add(prefix)
    return out


def _source_matches_prefix(source_name: str, prefix: str) -> bool:
    source_name = str(source_name or "")
    return source_name == prefix or source_name.startswith(f"{prefix}_")


def _configured_muat_endpoints(settings: dict[str, Any] | None) -> list[str]:
    if not settings or not experiment_enabled(settings, "muat_style_tcga_comparator"):
        return []
    local = _muat_settings(settings)
    endpoints = [str(endpoint) for endpoint in local.get("endpoints", []) if str(endpoint).strip()]
    if endpoints:
        return endpoints
    primary = str(local.get("primary_endpoint") or local.get("primary_task") or "").strip()
    secondary = [str(endpoint) for endpoint in local.get("secondary_endpoints", []) if str(endpoint).strip()]
    if primary:
        return [primary, *[endpoint for endpoint in secondary if endpoint != primary]]
    if local.get("run_manuscript_endpoints", True):
        return list(MUAT_MAIN_ENDPOINTS)
    return []


def _measured_muat_endpoints(
    canonical: pd.DataFrame,
    settings: dict[str, Any] | None = None,
    *,
    include_survival: bool = False,
) -> list[str]:
    configured = set(_configured_muat_endpoints(settings))
    measured = canonical[
        canonical["representation_family"].astype(str).eq("MuAt_style_attention_MIL")
        & canonical["model_family"].astype(str).eq("MuAt-compatible reimplementation")
        & canonical["status"].astype(str).eq("measured")
    ] if not canonical.empty else pd.DataFrame()
    measured_set = set(measured.get("endpoint", pd.Series(dtype=str)).astype(str))
    preferred = _ordered_unique([*_configured_muat_endpoints(settings), *MUAT_MAIN_ENDPOINTS, *MAIN_ENDPOINTS])
    endpoints = [endpoint for endpoint in preferred if endpoint in measured_set]
    if not endpoints and not configured and not measured.empty:
        endpoints = _ordered_unique(measured["endpoint"].astype(str).tolist())
    endpoints = [endpoint for endpoint in endpoints if endpoint in MAIN_ENDPOINTS]
    if not include_survival:
        endpoints = [endpoint for endpoint in endpoints if endpoint not in SURVIVAL_MAIN_ENDPOINTS]
    return endpoints


def _filter_current_muat_outputs(endpoint_results: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    if endpoint_results.empty or "endpoint" not in endpoint_results.columns:
        return endpoint_results
    source_ids = endpoint_results.apply(_endpoint_result_experiment_id, axis=1)
    muat_mask = source_ids.eq("muat_style_tcga_comparator")
    if not muat_mask.any():
        return endpoint_results
    accepted_prefixes = _accepted_muat_output_prefixes(settings)
    expected_endpoints = set(_configured_muat_endpoints(settings)) | set(MUAT_MAIN_ENDPOINTS)
    source_names = endpoint_results.get("source_file", endpoint_results.get("bundle_table", pd.Series("", index=endpoint_results.index))).fillna("").astype(str)
    if "bundle_table" in endpoint_results.columns:
        source_names = source_names.mask(source_names.eq(""), endpoint_results["bundle_table"].fillna("").astype(str))
    current_muat = source_names.map(lambda name: any(_source_matches_prefix(name, prefix) for prefix in accepted_prefixes)) & endpoint_results["endpoint"].astype(str).isin(expected_endpoints)
    return endpoint_results.loc[~muat_mask | current_muat].copy()


def _is_current_source_inventory_table(name: str, settings: dict[str, Any]) -> bool:
    if name.startswith("main_manuscript_complete_panel"):
        return experiment_enabled(settings, "main_manuscript_complete_panel")
    if name.startswith("muat_style_tcga_comparator"):
        return any(_source_matches_prefix(name, prefix) for prefix in _accepted_muat_output_prefixes(settings))
    return name.startswith(("proposed_clinical_bio_v4", "quick_bio_v4"))


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
    if "muat" in value or "attention mil" in value or "attention_mil" in value:
        return "MuAt_style_attention_MIL"
    if "maf_only" in value or "maf_stack_only" in value:
        return "MAF_stack_only"
    if "id_plus_best_gene_locus" in value or "signatures_plus_maf" in value:
        return "signatures_plus_MAF_stack"
    if "uga_rbf_kernel_mean" in value or "tuned_kme" in value or "channel_kme" in value:
        return "channel_KME"
    if "uga_unified" in value or "uga_geometry" in value:
        return "UGA_geometry"
    if "exposure" in value or "nnls" in value:
        return "COSMIC_NNLS_exposures"
    if "burden" in value:
        return "burden_only"
    if "standard" in value or "sbs" in value or "id83" in value or "dbs" in value:
        return "signatures_only"
    return "other"


def _atlas_status(family: str) -> str:
    if family in {"UGA_geometry", "channel_KME"}:
        return "uses UGA channel atlas for SBS/DBS; ID payload encoder for ID83"
    if family in {"burden_only", "signatures_only", "MAF_stack_only", "signatures_plus_MAF_stack"}:
        return "no UGA atlas"
    if family == "MuAt_style_attention_MIL":
        return "no UGA atlas"
    if family == "COSMIC_NNLS_exposures":
        return "uses COSMIC reference signatures; no UGA atlas for standard exposures"
    return "not applicable"


def _model_family(raw: str, experiment_id: str) -> str:
    text = f"{raw} {experiment_id}".lower()
    if "cox" in text or "survival" in text:
        return "cox_ph"
    if "muat-compatible" in text or "muat_compatible" in text:
        return "MuAt-compatible reimplementation"
    if "muat" in text or "attention_mil" in text:
        return "MuAt-style attention MIL"
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
    endpoint = _clean_endpoint(_text(row, ["endpoint", "Endpoint", "benchmark", "analysis_set"], ""))
    task = _text(row, ["task", "endpoint_type"], "").lower()
    has_balanced_accuracy = pd.notna(_num(row, ["oof_balanced_accuracy", "balanced_accuracy", "mean_balanced_accuracy"]))
    if has_balanced_accuracy and (endpoint in TCGA_TOP20_ENDPOINTS or "multiclass" in task):
        return "balanced_accuracy"
    metric = _text(row, ["metric", "metric_name", "primary_metric_name", "primary_metric", "Metric"], "")
    if metric:
        return metric
    if pd.notna(_num(row, ["oof_auroc", "auroc"])):
        return "auroc"
    if pd.notna(_num(row, ["c_index"])):
        return "c_index"
    if pd.notna(_num(row, ["spearman"])):
        return "spearman"
    if pd.notna(_num(row, ["oof_balanced_accuracy", "balanced_accuracy"])):
        return "balanced_accuracy"
    return "score"


def _primary_score(row: pd.Series) -> float:
    endpoint = _clean_endpoint(_text(row, ["endpoint", "Endpoint", "benchmark", "analysis_set"], ""))
    task = _text(row, ["task", "endpoint_type"], "").lower()
    if endpoint in TCGA_TOP20_ENDPOINTS or "multiclass" in task:
        balanced = _num(row, ["oof_balanced_accuracy", "balanced_accuracy", "mean_balanced_accuracy"])
        if pd.notna(balanced):
            return balanced
    return _num(
        row,
        [
            "score",
            "candidate_score",
            "oof_auroc",
            "auroc",
            "c_index",
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
        oof_prediction_file = _text(row, ["oof_prediction_file"], "")
        fold_metrics_file = _text(row, ["fold_metrics_file"], "")
        folds = _num(row, ["n_folds", "folds", "outer_folds"])
        repeats = _num(row, ["repeats", "n_repeats"])
        if experiment_id == "muat_style_tcga_comparator":
            source_stem = source_file.removesuffix("_endpoint_results.csv") if source_file else "muat_style_tcga_comparator"
            oof_prediction_file = oof_prediction_file or f"{source_stem}_oof_predictions.csv"
            fold_metrics_file = fold_metrics_file or f"{source_stem}_fold_metrics.csv"
            folds = 5.0 if not np.isfinite(folds) else folds
            repeats = 1.0 if not np.isfinite(repeats) else repeats
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
                "macro_auroc": _num(row, ["macro_auroc"]),
                "auroc": _num(row, ["auroc", "oof_auroc"]),
                "micro_auroc": _num(row, ["micro_auroc"]),
                "auprc": _num(row, ["auprc"]),
                "accuracy": _num(row, ["accuracy", "mean_accuracy"]),
                "macro_f1": _num(row, ["macro_f1", "mean_macro_f1"]),
                "f1": _num(row, ["macro_f1", "mean_macro_f1"]),
                "balanced_accuracy": _num(row, ["balanced_accuracy", "oof_balanced_accuracy", "mean_balanced_accuracy"]),
                "cohen_kappa": _num(row, ["cohen_kappa"]),
                "top3_accuracy": _num(row, ["top3_accuracy"]),
                "top5_accuracy": _num(row, ["top5_accuracy"]),
                "delta_vs_signatures": _num(row, ["delta_vs_standard", "delta_balanced_accuracy", "delta"]),
                "ci_low": _num(row, ["ci_low", "delta_ci_low", "bootstrap_ci_low", "delta_95ci_low"]),
                "ci_high": _num(row, ["ci_high", "delta_ci_high", "bootstrap_ci_high", "delta_95ci_high"]),
                "p_value": _num(row, ["p_value", "p_balanced_accuracy"]),
                "q_value": _num(row, ["q_value", "q_balanced_accuracy"]),
                "n_samples": _num(row, ["n_samples", "n", "n_patients", "n_test"]),
                "n_features": _num(row, ["n_features", "feature_dimension", "standard_features", "standard_plus_kme_features"]),
                "folds": folds,
                "repeats": repeats,
                "xgb_estimators": _num(row, ["n_estimators"]),
                "optuna_trials_completed": optuna_trials_completed,
                "kme_version": _text(row, ["kme_version"], ""),
                "landmark_mode": _text(row, ["landmark_mode"], ""),
                "sigma_multiplier": _num(row, ["sigma_multiplier"]),
                "sigma_strategy": _text(row, ["sigma_strategy"], ""),
                "modality_strategy": _text(row, ["modality_strategy"], ""),
                "landmark_sampling": _text(row, ["landmark_sampling"], ""),
                "kernel_weighting": _text(row, ["kernel_weighting"], ""),
                "cache_key": _text(row, ["cache_key"], ""),
                "oof_prediction_file": oof_prediction_file,
                "fold_metrics_file": fold_metrics_file,
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
            (0.38, 0.48, "Geometry", "Supplementary UGA and channel KME atlas variants are kept separate from the main production panel."),
            (0.38, 0.24, "MAF Stack", "Gene, pathway, consequence, locus, VAF, and locus-topography aggregations."),
            (0.73, 0.58, "Models and Endpoints", "Nested elastic-net/logistic and XGBoost for non-survival endpoints; CoxNet survival is reported separately in Figure 5 and tables."),
            (0.73, 0.30, "Event-Set Comparator", "The MuAt-compatible reimplementation operates directly on mutation event bags for measured comparator endpoints."),
        ],
    )


def _barplot(df: pd.DataFrame, stem: Path, title: str, families: list[str], *, endpoints: list[str] | None = None) -> None:
    plot = _best_scores(df, main_only=True, families=families)
    if endpoints is not None and not plot.empty:
        plot = plot[plot["endpoint"].astype(str).isin(endpoints)].copy()
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


def _write_tables(df: pd.DataFrame, side_tables: dict[str, pd.DataFrame], manuscript_dir: Path, *, settings: dict[str, Any]) -> None:
    tables_dir = manuscript_dir / "tables"
    public_dir = tables_dir / "publication"
    technical_dir = tables_dir / "technical"
    supp_dir = manuscript_dir / "supplement"
    tables_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    technical_dir.mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)

    manuscript_df = _manuscript_endpoint_frame(df)
    display_df = _add_display_columns(manuscript_df.copy())
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
                "splitting_scheme": "5 outer folds with inner validation tuning and pooled out-of-fold test predictions where model-based",
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
        "macro_auroc",
        "auroc",
        "micro_auroc",
        "auprc",
        "accuracy",
        "macro_f1",
        "f1",
        "balanced_accuracy",
        "cohen_kappa",
        "top3_accuracy",
        "top5_accuracy",
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
    table2_technical = _add_display_columns(display_df.loc[:, [col for col in ordered_cols if col in display_df.columns]])
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
            if family == "MuAt_style_attention_MIL":
                measured = fam.assign(primary_score=pd.to_numeric(fam["primary_score"], errors="coerce"))
                measured = measured[np.isfinite(measured["primary_score"])]
                row[MAIN_REPRESENTATION_SHORT_LABELS[family]] = (
                    _format_manuscript_score(measured.sort_values("primary_score", ascending=False, kind="mergesort").iloc[0]["primary_score"])
                    if not measured.empty
                    else ""
                )
            else:
                scores_by_model = {
                    str(item["model_family"]): _format_manuscript_score(item["primary_score"])
                    for _, item in fam.iterrows()
                }
                cell_models = ["cox_ph"] if endpoint == "OS" else ["elastic_net", "XGBoost"]
                row[MAIN_REPRESENTATION_SHORT_LABELS[family]] = " / ".join(scores_by_model.get(model, "") for model in cell_models)
        row["Best model"] = (
            f"{best['representation_family_display']} ({best['model_display']}, "
            f"{_format_manuscript_score(best['primary_score'])})"
        )
        table2_rows.append(row)
    table2 = pd.DataFrame(table2_rows)
    _write_table_csv_html(table2, table2_path)
    _write_publication_copy(table2, table2_path, public_dir)

    cancer_type_metrics_path = supp_dir / "table_s4_cancer_type_top20_metrics.csv"
    cancer_type_metrics = main_perf[main_perf["endpoint"].astype(str).eq("cancer_type_top20")].copy()
    cancer_metric_rows: list[dict[str, object]] = []
    if not cancer_type_metrics.empty:
        for _, item in cancer_type_metrics.sort_values(["representation_family_display", "model_display"], kind="mergesort").iterrows():
            macro_auroc = _num(item, ["macro_auroc", "auroc"])
            if pd.isna(macro_auroc) and str(item.get("metric", "")).lower() != "balanced_accuracy":
                macro_auroc = _num(item, ["primary_score"])
            cancer_metric_rows.append(
                {
                    "Representation": item.get("representation_family_display", ""),
                    "Model": item.get("model_display", ""),
                    "N": _format_manuscript_count(item.get("n_samples")),
                    "Macro-AUROC": _format_manuscript_score(macro_auroc),
                    "Micro-AUROC": _format_manuscript_score(item.get("micro_auroc")),
                    "Accuracy": _format_manuscript_score(item.get("accuracy")),
                    "Balanced accuracy": _format_manuscript_score(item.get("balanced_accuracy")),
                    "Macro-F1": _format_manuscript_score(item.get("macro_f1", item.get("f1"))),
                    "Cohen's kappa": _format_manuscript_score(item.get("cohen_kappa")),
                    "Top-3 accuracy": _format_manuscript_score(item.get("top3_accuracy")),
                    "Top-5 accuracy": _format_manuscript_score(item.get("top5_accuracy")),
                }
            )
    cancer_metric_table = pd.DataFrame(
        cancer_metric_rows,
        columns=[
            "Representation",
            "Model",
            "N",
            "Macro-AUROC",
            "Micro-AUROC",
            "Accuracy",
            "Balanced accuracy",
            "Macro-F1",
            "Cohen's kappa",
            "Top-3 accuracy",
            "Top-5 accuracy",
        ],
    )
    _write_supplement_public_table(cancer_metric_table, cancer_type_metrics_path, tables_dir, public_dir)

    bio_maf_guide, bio_maf_exact = _bio_maf_v4_feature_guide()
    bio_maf_guide_path = supp_dir / "table_s5_bio_maf_v4_feature_guide.csv"
    _write_supplement_public_table(bio_maf_guide, bio_maf_guide_path, tables_dir, public_dir)
    _write_table_csv_html(bio_maf_exact, _technical_path(technical_dir, bio_maf_guide_path))

    bio_maf_selection, bio_maf_selection_by_fold = _bio_maf_v4_nested_selection_tables()
    selection_path = supp_dir / "table_s6_bio_maf_v4_nested_feature_selection.csv"
    selection_public_columns = [
        "Endpoint tier",
        "Endpoint",
        "Representation",
        "Model",
        "Primary metric",
        "Selected feature-set label",
        "Selected total features",
        "Selected Bio MAF features",
        "Selected Bio MAF blocks",
        "Includes mutational signatures",
        "Includes exact driver genes",
        "Includes optional burden controls",
        "Outer folds selected",
        "Outer folds evaluated",
        "Fold selection fraction",
        "Mean inner-validation score",
        "Mean outer-fold score",
        "Selection interpretation",
    ]
    if not bio_maf_selection.empty:
        bio_maf_selection_public = bio_maf_selection.loc[
            :, [col for col in selection_public_columns if col in bio_maf_selection.columns]
        ].copy()
    else:
        bio_maf_selection_public = pd.DataFrame(columns=selection_public_columns)
    _write_supplement_public_table(bio_maf_selection_public, selection_path, tables_dir, public_dir)
    _write_table_csv_html(
        bio_maf_selection_by_fold,
        technical_dir / "table_s6_bio_maf_v4_nested_feature_selection_by_fold.csv",
        title="Supplementary Table S6 technical by-fold feature-block selections",
    )

    hyper_technical = (
        display_df.groupby(["representation_family", "model_family", "atlas_status"], dropna=False)
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
                "Feature dimensionality": _format_feature_range(sub["n_features"]) or "Event bag",
                "Context/atlas": "; ".join(_ordered_unique(sub["atlas_status_display"].tolist())),
                "Evaluated models": "; ".join(models),
                "Manuscript role": specs["role"],
            }
        )
    hyper = pd.DataFrame(representation_rows)
    _write_table_csv_html(hyper, table3_path)
    _write_publication_copy(hyper, table3_path, public_dir)

    class_dist_technical = (
        display_df.groupby(["endpoint_tier", "endpoint"], dropna=False)
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
    _write_supplement_public_table(s1, s1_path, tables_dir, public_dir)

    s2_sensitivity_families = ["UGA_geometry", "COSMIC_NNLS_exposures"]
    sensitivity_technical = display_df[
        (display_df["endpoint_tier"].eq("supplement") | display_df["representation_family"].isin(s2_sensitivity_families))
        & ~display_df["representation_family"].eq("channel_KME")
    ].copy()
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
    _write_supplement_public_table(sensitivity, s2_path, tables_dir, public_dir)

    source_inventory = pd.DataFrame(
        [{"source_table": name, "rows": len(frame), "columns": len(frame.columns)} for name, frame in side_tables.items()]
    )
    if not source_inventory.empty:
        keep = ~source_inventory["source_table"].astype(str).str.contains(MANUSCRIPT_EXCLUDED_ENDPOINT_PATTERN, na=False)
        keep &= source_inventory["source_table"].astype(str).map(lambda name: _is_current_source_inventory_table(name, settings))
        source_inventory = source_inventory.loc[keep].reset_index(drop=True)
    s0_path = supp_dir / "table_s0_source_inventory.csv"
    _write_table_csv_html(source_inventory, _technical_path(technical_dir, s0_path))
    group_labels = {
        "main_manuscript_complete_panel": "Main complete panel",
        "muat_style_tcga_comparator": "MuAt-compatible comparator",
        "proposed_clinical_bio_v4": "Bio MAF v4 source artifacts",
        "quick_bio_v4": "Bio MAF v4 source artifacts",
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
    _write_supplement_public_table(source_inventory_public, s0_path, tables_dir, public_dir)


def _source_label(endpoint: str) -> str:
    if endpoint in {"damage_class", "Low-burden downsample", "Original data"}:
        return "Kucab mutagen-treated clone mutation catalogues"
    if endpoint.startswith("hrd") or endpoint in {"HRD_Score", "eCARD", "HRD_TAI", "HRD_LST", "HRD_LOH", "PARPi7", "parpi7_binary"}:
        return "TCGA-BRCA HRD labels with MC3 mutation features"
    if endpoint in {"cancer_type_top20", "smoking_ever", "high_purity", "high_stage", "OS", "DSS", "PFI", "DFI"}:
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
        "hrd_binary_24": "HRD-high versus HRD-low at threshold 24",
        "hrd_binary_33": "HRD-high versus HRD-low at threshold 33",
        "hrd_binary_42": "HRD-high versus HRD-low at threshold 42",
        "cancer_type_top20": "fixed top-20 TCGA cancer type classification",
        "OS": "overall survival time-to-event endpoint",
        "DSS": "disease-specific survival time-to-event endpoint",
        "PFI": "progression-free interval time-to-event endpoint",
        "DFI": "disease-free interval time-to-event endpoint",
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


def _expected_canonical_slots(settings: dict[str, Any] | None = None) -> set[tuple[str, str, str]]:
    tabular_families = [
        "burden_only",
        "signatures_only",
        "MAF_stack_only",
        "signatures_plus_MAF_stack",
    ]
    muat_endpoints = set(_configured_muat_endpoints(settings)) if settings is not None else set()
    survival_endpoints = SURVIVAL_MAIN_ENDPOINTS
    slots: set[tuple[str, str, str]] = set()
    for endpoint in MAIN_ENDPOINTS:
        if endpoint in survival_endpoints:
            for family in tabular_families:
                slots.add((endpoint, family, "cox_ph"))
        else:
            for family in tabular_families:
                for model in ["elastic_net", "XGBoost"]:
                    slots.add((endpoint, family, model))
        if endpoint in muat_endpoints:
            slots.add((endpoint, "MuAt_style_attention_MIL", "MuAt-compatible reimplementation"))
    return slots


def _canonical_main_results(df: pd.DataFrame, tables_dir: Path, *, strict: bool, settings: dict[str, Any] | None = None) -> pd.DataFrame:
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
    multiclass_non_balanced = canonical[
        canonical["task"].astype(str).str.contains("multiclass", case=False, na=False)
        & pd.to_numeric(canonical["balanced_accuracy"], errors="coerce").notna()
        & ~canonical["metric"].astype(str).eq("balanced_accuracy")
    ]
    if strict and not multiclass_non_balanced.empty:
        detail = multiclass_non_balanced.loc[:, ["endpoint", "representation_family", "model_family", "metric"]].to_dict("records")
        raise ValueError(f"Canonical multiclass rows with balanced accuracy must use balanced_accuracy: {json.dumps(detail, indent=2)}")
    canonical["status"] = "measured"
    canonical["canonical_source"] = canonical["experiment_id"]
    canonical["canonical_slot_id"] = (
        canonical["endpoint"].astype(str)
        + "|"
        + canonical["representation_family"].astype(str)
        + "|"
        + canonical["model_family"].astype(str)
    )
    expected = _expected_canonical_slots(settings)
    observed = set(zip(canonical["endpoint"], canonical["representation_family"], canonical["model_family"]))
    missing = sorted(expected - observed)
    if strict and missing:
        detail = [
            {"endpoint": endpoint, "representation_family": family, "model_family": model}
            for endpoint, family, model in missing
        ]
        raise ValueError(f"Canonical main panel is missing OOF-backed measured rows: {json.dumps(detail, indent=2)}")
    return canonical.reset_index(drop=True)


def _prediction_file_candidates(tables_dir: Path, name: str) -> list[Path]:
    """Return local and dataset-backed prediction-file candidates.

    Large OOF tables may be kept outside the Git checkout for upload/reviewer
    workflows. Figure regeneration should still work from those declared
    restore locations without re-running model training.
    """

    candidates = [tables_dir / name]
    manifest_path = BUNDLE_ROOT / "manifests" / "dataset_assets_manifest.csv"
    if manifest_path.exists():
        try:
            manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
        except Exception:
            manifest = pd.DataFrame()
        if not manifest.empty:
            destination = f"results/tables/{name}".replace("\\", "/")
            restore_matches = manifest[
                manifest.get("restore_destination", pd.Series(dtype=str)).astype(str).str.replace("\\", "/", regex=False).eq(destination)
            ]
            for _, row in restore_matches.iterrows():
                relative = str(row.get("dataset_relative_path") or "").strip()
                if relative:
                    candidates.append(DATASETS_ROOT / relative)
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def _read_prediction_file(path: Path, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    key = str(path.resolve()) if path.exists() else str(path)
    if key not in cache:
        cache[key] = pd.read_csv(path, low_memory=False)
    return cache[key]


def _oof_representation_aliases(representation: str, family: str) -> list[str]:
    return [str(representation)]


def _canonical_oof_predictions(canonical: pd.DataFrame, tables_dir: Path, *, strict: bool) -> pd.DataFrame:
    cache: dict[str, pd.DataFrame] = {}
    frames: list[pd.DataFrame] = []
    missing: list[dict[str, str]] = []
    for _, row in canonical.iterrows():
        oof_name = str(row.get("oof_prediction_file", ""))
        learner = str(row.get("model_label", "")).strip()
        if not learner or learner.lower() in {"nan", "none"}:
            learner = "linear" if str(row.get("model_family")) == "elastic_net" else "xgboost"
        aliases = _oof_representation_aliases(str(row["representation"]), str(row["representation_family"]))
        slot = pd.DataFrame()
        searched_paths: list[str] = []
        for prediction_path in _prediction_file_candidates(tables_dir, oof_name):
            if not prediction_path.exists():
                searched_paths.append(str(prediction_path))
                continue
            pred = _read_prediction_file(prediction_path, cache)
            searched_paths.append(str(prediction_path))
            slot = pred[
                pred["endpoint"].astype(str).eq(str(row["endpoint"]))
                & pred["representation"].astype(str).isin(aliases)
                & pred["learner"].astype(str).eq(learner)
            ].copy()
            if not slot.empty:
                break
        if slot.empty:
            missing.append(
                {
                    "endpoint": str(row["endpoint"]),
                    "representation": str(row["representation"]),
                    "representation_family": str(row["representation_family"]),
                    "model_family": str(row["model_family"]),
                    "oof_prediction_file": oof_name,
                    "searched_paths": ";".join(searched_paths),
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


def _predicted_classes(merged: pd.DataFrame, suffix: str) -> np.ndarray:
    class_col = f"predicted_class_{suffix}"
    if class_col in merged.columns:
        values = pd.to_numeric(merged[class_col], errors="coerce")
        if values.notna().all():
            return values.to_numpy(dtype=int)
    return np.argmax(_prediction_matrix(merged, suffix), axis=1).astype(int)


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
    if merged["sample"].astype(str).duplicated().any():
        raise ValueError("Paired OOF merge produced duplicate samples")
    true_candidate = pd.to_numeric(merged["true_value_candidate"], errors="coerce")
    true_baseline = pd.to_numeric(merged["true_value_baseline"], errors="coerce")
    if true_candidate.notna().all() and true_baseline.notna().all():
        mismatch = ~np.isclose(true_candidate.to_numpy(dtype=float), true_baseline.to_numpy(dtype=float), equal_nan=True)
    else:
        mismatch = merged["true_value_candidate"].astype(str).to_numpy() != merged["true_value_baseline"].astype(str).to_numpy()
    if bool(np.any(mismatch)):
        raise ValueError(f"Paired OOF true labels differ for {int(np.sum(mismatch))} samples")
    y = merged["true_value_candidate"].to_numpy()
    metric = str(candidate.get("metric") or baseline.get("metric") or "").lower()
    if metric == "spearman":
        return y.astype(float), merged["pred_value_candidate"].to_numpy(dtype=float), merged["pred_value_baseline"].to_numpy(dtype=float), metric
    if metric == "auroc":
        return y.astype(float), merged["pred_class_1_candidate"].to_numpy(dtype=float), merged["pred_class_1_baseline"].to_numpy(dtype=float), metric
    if metric == "macro_auroc":
        return y, _prediction_matrix(merged, "candidate"), _prediction_matrix(merged, "baseline"), metric
    if metric == "balanced_accuracy":
        return y, _predicted_classes(merged, "candidate"), _predicted_classes(merged, "baseline"), metric
    raise ValueError(f"Unsupported canonical comparison metric: {metric}")


def _comparison_registry_frame(settings: dict[str, Any] | None = None) -> pd.DataFrame:
    rows = []
    expected = _expected_canonical_slots(settings)
    for figure_id, comparison_name, candidate, baseline in MAIN_COMPARISONS:
        for endpoint in MAIN_ENDPOINTS:
            if endpoint == "OS":
                continue
            for model_family in MODEL_FAMILIES:
                if (endpoint, candidate, model_family) not in expected:
                    continue
                if (endpoint, baseline, model_family) not in expected:
                    continue
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


def _compute_pairwise_tests(canonical: pd.DataFrame, oof: pd.DataFrame, *, bootstrap: int, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, spec in _comparison_registry_frame(settings).iterrows():
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
                    stratify=metric in {"macro_auroc", "balanced_accuracy"},
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


def _canonical_plot_grid(canonical: pd.DataFrame, *, endpoints: list[str], families: list[str], model_families: list[str], settings: dict[str, Any] | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expected = _expected_canonical_slots(settings)
    for endpoint in endpoints:
        for model in model_families:
            for family in families:
                if (endpoint, family, model) not in expected:
                    continue
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


def _survival_cox_plot_data(canonical: pd.DataFrame) -> pd.DataFrame:
    rows = canonical[
        canonical["endpoint"].astype(str).isin(SURVIVAL_MAIN_ENDPOINTS)
        & canonical["model_family"].astype(str).eq("cox_ph")
        & canonical["representation_family"].astype(str).isin(TABULAR_MAIN_REPRESENTATIONS)
    ].copy()
    order = {name: idx for idx, name in enumerate(TABULAR_MAIN_REPRESENTATIONS)}
    if rows.empty:
        rows = pd.DataFrame(
            [
                {
                    "endpoint": endpoint,
                    "endpoint_tier": "main",
                    "representation_family": family,
                    "model_family": "cox_ph",
                    "metric": "c_index",
                    "primary_score": np.nan,
                    "status": "missing",
                    "na_reason": "missing canonical measured Cox survival row",
                }
                for endpoint in sorted(SURVIVAL_MAIN_ENDPOINTS)
                for family in TABULAR_MAIN_REPRESENTATIONS
            ]
        )
    else:
        rows["status"] = "measured"
        rows["na_reason"] = ""
    rows["_order"] = rows["representation_family"].map(order).fillna(99)
    return rows.sort_values(["endpoint", "_order", "representation_family"], kind="mergesort").drop(columns=["_order"], errors="ignore").reset_index(drop=True)


def _calibration_plot_data(canonical: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    endpoints = ["damage_class", "hrd_binary_24", "hrd_binary_33", "hrd_binary_42", "cancer_type_top20"]
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
        probs = probs.loc[:, probs.notna().any(axis=0)]
        if probs.empty:
            continue
        if endpoint in {"hrd_binary_24", "hrd_binary_33", "hrd_binary_42"} and "pred_class_1" in probs.columns:
            confidence = probs["pred_class_1"].to_numpy(dtype=float)
            observed = y.to_numpy(dtype=float)
            mode = "binary_positive_probability"
            y_label = "Observed positive fraction"
        else:
            prob_values = probs.to_numpy(dtype=float)
            finite_prob_row = np.isfinite(prob_values).any(axis=1)
            prob_for_argmax = np.where(np.isfinite(prob_values), prob_values, -np.inf)
            confidence = np.max(prob_for_argmax, axis=1)
            predicted = np.argmax(prob_for_argmax, axis=1)
            true_numeric = y.to_numpy(dtype=float)
            observed = (predicted == true_numeric).astype(float)
            observed[~np.isfinite(true_numeric)] = np.nan
            confidence[~finite_prob_row] = np.nan
            mode = "multiclass_confidence_accuracy"
            y_label = "Observed accuracy"
        valid = np.isfinite(confidence) & np.isfinite(observed)
        confidence = confidence[valid]
        observed = observed[valid]
        if len(confidence) < 10:
            continue
        bins = np.linspace(0.0, 1.0, 11)
        endpoint_rows: list[dict[str, object]] = []
        for bin_idx in range(10):
            lo, hi = bins[bin_idx], bins[bin_idx + 1]
            mask = (confidence >= lo) & (confidence <= hi if bin_idx == 9 else confidence < hi)
            if not np.any(mask):
                continue
            endpoint_rows.append(
                {
                    "endpoint": endpoint,
                    "representation_family": "signatures_plus_MAF_stack",
                    "model_family": "XGBoost",
                    "calibration_mode": mode,
                    "calibration_y_label": y_label,
                    "bin": bin_idx + 1,
                    "bin_left": lo,
                    "bin_right": hi,
                    "mean_predicted": float(np.mean(confidence[mask])),
                    "observed_frequency": float(np.mean(observed[mask])),
                    "n_samples": int(np.sum(mask)),
                    "n_total": int(len(confidence)),
                    "n_classes": int(probs.shape[1]) if mode == "multiclass_confidence_accuracy" else 2,
                }
            )
        total = sum(int(row["n_samples"]) for row in endpoint_rows)
        if total:
            ece = sum(
                int(row["n_samples"]) * abs(float(row["observed_frequency"]) - float(row["mean_predicted"]))
                for row in endpoint_rows
            ) / total
            for row in endpoint_rows:
                row["endpoint_ece"] = float(ece)
            rows.extend(endpoint_rows)
    return pd.DataFrame(rows)


def _s3_measured_plot_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _manuscript_endpoint_frame(df)
    families = ["UGA_geometry", "channel_KME", "COSMIC_NNLS_exposures"]
    measured = df[
        df["representation_family"].isin(families)
        & np.isfinite(pd.to_numeric(df["primary_score"], errors="coerce"))
    ].copy()
    if measured.empty:
        return measured, pd.DataFrame()
    measured["analysis_family"] = np.select(
        [
            measured["representation_family"].isin(["UGA_geometry", "channel_KME"]),
            measured["representation_family"].eq("COSMIC_NNLS_exposures"),
        ],
        ["Alternative geometry", "COSMIC/NNLS exposure checks"],
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
    settings: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canonical_dir = manuscript_dir / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical = _canonical_main_results(normalized, tables_dir, strict=strict, settings=settings)
    oof = _canonical_oof_predictions(canonical, tables_dir, strict=strict)
    tests = _compute_pairwise_tests(canonical, oof, bootstrap=bootstrap, settings=settings) if not canonical.empty and not oof.empty else pd.DataFrame()
    atomic_write_csv(canonical, canonical_dir / "main_panel_results.csv", index=False)
    atomic_write_csv(oof, canonical_dir / "main_panel_oof_predictions.csv", index=False)
    atomic_write_csv(tests, canonical_dir / "main_panel_pairwise_tests.csv", index=False)
    atomic_write_csv(_comparison_registry_frame(settings), canonical_dir / "main_panel_comparison_registry.csv", index=False)
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
    settings: dict[str, Any] | None = None,
) -> None:
    plot_dir = manuscript_dir / "plot_data"
    plot_dir.mkdir(parents=True, exist_ok=True)
    muat_figure_endpoints = _measured_muat_endpoints(canonical, settings)
    files = {
        "figure_2_signature_baselines.csv": _attach_plot_tests(
            _canonical_plot_grid(canonical, endpoints=FIGURE_MODEL_COMPARISON_ENDPOINTS, families=["burden_only", "signatures_only"], model_families=MODEL_FAMILIES, settings=settings),
            tests,
            "figure_2",
        ),
        "figure_3_geometry_vs_signatures.csv": _attach_plot_tests(
            _canonical_plot_grid(canonical, endpoints=muat_figure_endpoints, families=MAIN_REPRESENTATIONS, model_families=MODEL_FAMILIES, settings=settings),
            tests,
            "figure_3",
        ),
        "figure_4_maf_stack_vs_signatures.csv": _attach_plot_tests(
            _canonical_plot_grid(canonical, endpoints=FIGURE_MODEL_COMPARISON_ENDPOINTS, families=["signatures_only", "MAF_stack_only", "signatures_plus_MAF_stack"], model_families=MODEL_FAMILIES, settings=settings),
            tests,
            "figure_4",
        ),
        "figure_5_overall_survival_cox.csv": _survival_cox_plot_data(canonical),
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
        tables_dir = manuscript_dir / "tables"
        _write_supplement_public_table(s3_public, s3_path, tables_dir, tables_dir / "publication")
    atomic_write_json(
        plot_dir / "plot_data_manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "files": sorted(files),
            "benchmark_completeness": "benchmark_completeness.csv",
            "main_endpoints": MAIN_ENDPOINTS,
            "figure_model_comparison_endpoints": FIGURE_MODEL_COMPARISON_ENDPOINTS,
            "survival_figure": "figure_5_overall_survival_cox.csv",
            "survival_endpoints_reported_separately_from_model_comparison_figures": sorted(SURVIVAL_MAIN_ENDPOINTS),
            "muat_figure_endpoints": muat_figure_endpoints,
            "main_representations": MAIN_REPRESENTATIONS,
            "tabular_main_representations": TABULAR_MAIN_REPRESENTATIONS,
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
            if not used["used_in_figures"] and not used["used_in_tables"]:
                continue
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
    subprocess.run([node, str(script), str(manuscript_dir)], cwd=str(BUNDLE_ROOT), check=True, env=env)


def _write_figures(df: pd.DataFrame, manuscript_dir: Path) -> None:
    figures_dir = manuscript_dir / "figures"
    supp_dir = manuscript_dir / "supplement"
    figures_dir.mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)
    _figure_1(figures_dir)
    _barplot(df, figures_dir / "figure_2_signature_baselines", "Figure 2. Burden and Signature Baselines", ["burden_only", "signatures_only"], endpoints=FIGURE_MODEL_COMPARISON_ENDPOINTS)
    _barplot(df, figures_dir / "figure_3_geometry_vs_signatures", "Figure 3. MuAt-Compatible Event-Bag Comparator", MAIN_REPRESENTATIONS, endpoints=MUAT_MAIN_ENDPOINTS)
    _barplot(df, figures_dir / "figure_4_maf_stack_vs_signatures", "Figure 4. MAF-Stack Biology Adds to Spectra", ["signatures_only", "MAF_stack_only", "signatures_plus_MAF_stack"], endpoints=FIGURE_MODEL_COMPARISON_ENDPOINTS)
    _barplot(df, figures_dir / "figure_5_overall_survival_cox", "Figure 5. Overall Survival CoxNet C-index", TABULAR_MAIN_REPRESENTATIONS, endpoints=sorted(SURVIVAL_MAIN_ENDPOINTS))
    _write_text_panel(
        supp_dir / "figure_s1_representation_construction",
        "Supplementary Figure S1. Representation Construction Details",
        [
            (0.03, 0.58, "Spectra", "Count SBS96/DBS78/ID83 channels, normalize, append burden covariates where specified."),
            (0.38, 0.58, "MuAt-Compatible", "Encode mutation motif, 1-Mb position, and annotation tokens for the event-bag attention comparator."),
            (0.73, 0.58, "MAF Stack", "Aggregate genes, pathways, consequences, VAF summaries, and genomic locus/topography bins."),
            (0.03, 0.25, "UGA Variants", "Supplementary channel projections use the production UGA channel atlas and ID payload encoder."),
            (0.38, 0.25, "Models", "Nested elastic-net/logistic and XGBoost for non-survival model-comparison figures; CoxNet survival is reported separately in Figure 5 and tables."),
            (0.73, 0.25, "Outputs", "OOF metrics, feature dimensions, atlas status, and endpoint-tier labels are normalized into manuscript tables."),
        ],
    )
    _heatmap(df, supp_dir / "figure_s2_calibration_thresholds", "Supplementary Figure S2. OOF Metric Coverage for Calibration Review", MAIN_REPRESENTATIONS, main_only=True)
    _heatmap(df, supp_dir / "figure_s3_feature_importance", "Supplementary Figure S3. Supplementary Representation Sensitivity", ["UGA_geometry", "channel_KME", "COSMIC_NNLS_exposures"], main_only=False)


def _validate_plot_data(manuscript_dir: Path) -> None:
    plot_dir = manuscript_dir / "plot_data"
    canonical_path = manuscript_dir / "canonical" / "main_panel_results.csv"
    tests_path = manuscript_dir / "canonical" / "main_panel_pairwise_tests.csv"
    if not canonical_path.exists() or not tests_path.exists():
        raise FileNotFoundError("Canonical main-panel results or pairwise tests are missing")
    canonical = pd.read_csv(canonical_path)
    tests = pd.read_csv(tests_path)
    model_comparison_plot_names = [
        "figure_2_signature_baselines.csv",
        "figure_3_geometry_vs_signatures.csv",
        "figure_4_maf_stack_vs_signatures.csv",
    ]
    for name in model_comparison_plot_names:
        frame = pd.read_csv(plot_dir / name)
        bad = frame[~frame["status"].astype(str).eq("measured")]
        if not bad.empty:
            cols = ["endpoint", "representation_family", "model_family", "status", "na_reason"]
            raise ValueError(f"{name} contains non-measured main slots: {bad.loc[:, [c for c in cols if c in bad.columns]].to_dict('records')}")
        survival_rows = frame[
            frame["endpoint"].astype(str).isin(SURVIVAL_MAIN_ENDPOINTS)
            | frame["model_family"].astype(str).eq("cox_ph")
        ]
        if not survival_rows.empty:
            cols = ["endpoint", "representation_family", "model_family"]
            raise ValueError(f"{name} contains survival/Cox rows that belong in tables, not model-comparison figures: {survival_rows.loc[:, [c for c in cols if c in survival_rows.columns]].to_dict('records')}")
        canonical_muat = set(
            canonical.loc[
                canonical["representation_family"].astype(str).eq("MuAt_style_attention_MIL")
                & canonical["model_family"].astype(str).eq("MuAt-compatible reimplementation"),
                "endpoint",
            ].astype(str)
        )
        figure_muat = set(frame.loc[frame["representation_family"].astype(str).eq("MuAt_style_attention_MIL"), "endpoint"].astype(str))
        unsupported_muat = sorted(figure_muat - canonical_muat)
        if unsupported_muat:
            raise ValueError(f"{name} contains MuAt rows for endpoints without measured MuAt canonical results: {unsupported_muat}")
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
    survival_plot = pd.read_csv(plot_dir / "figure_5_overall_survival_cox.csv")
    if survival_plot.empty:
        raise ValueError("figure_5_overall_survival_cox.csv is empty")
    survival_bad_status = survival_plot[~survival_plot["status"].astype(str).eq("measured")]
    if not survival_bad_status.empty:
        cols = ["endpoint", "representation_family", "model_family", "status", "na_reason"]
        raise ValueError(f"figure_5_overall_survival_cox.csv contains non-measured rows: {survival_bad_status.loc[:, [c for c in cols if c in survival_bad_status.columns]].to_dict('records')}")
    bad_survival = survival_plot[
        ~survival_plot["endpoint"].astype(str).isin(SURVIVAL_MAIN_ENDPOINTS)
        | ~survival_plot["model_family"].astype(str).eq("cox_ph")
    ]
    if not bad_survival.empty:
        cols = ["endpoint", "representation_family", "model_family", "status"]
        raise ValueError(f"figure_5_overall_survival_cox.csv contains non-survival rows: {bad_survival.loc[:, [c for c in cols if c in bad_survival.columns]].to_dict('records')}")
    survival_merged = survival_plot.merge(
        canonical[["endpoint", "representation_family", "model_family", "primary_score"]],
        on=["endpoint", "representation_family", "model_family"],
        suffixes=("_plot", "_canonical"),
        how="left",
    )
    if survival_merged["primary_score_canonical"].isna().any():
        raise ValueError("figure_5_overall_survival_cox.csv contains rows not found in canonical main-panel results")
    survival_delta = (
        pd.to_numeric(survival_merged["primary_score_plot"], errors="coerce")
        - pd.to_numeric(survival_merged["primary_score_canonical"], errors="coerce")
    ).abs()
    if (survival_delta > 1e-12).any():
        raise ValueError("figure_5_overall_survival_cox.csv values differ from canonical main-panel results")
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


def _tested_count(tests: pd.DataFrame, comparison_name: str, *, model_family: str | None = None) -> int:
    rows = tests[tests["comparison_name"].astype(str).eq(comparison_name)].copy()
    if model_family is not None:
        rows = rows[rows["model_family"].astype(str).eq(model_family)]
    rows = rows[rows["test_status"].astype(str).eq("tested")]
    rows = rows.drop_duplicates(
        subset=["comparison_name", "endpoint", "model_family", "candidate_representation", "baseline_representation"]
    )
    return int(len(rows))


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


def _write_manuscript_text(
    manuscript_dir: Path,
    canonical: pd.DataFrame,
    pairwise_tests: pd.DataFrame,
    settings: dict[str, Any] | None = None,
) -> None:
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
        "supplement/table_s5_bio_maf_v4_feature_guide.csv",
        "supplement/table_s6_bio_maf_v4_nested_feature_selection.csv",
    ]:
        path = manuscript_dir / name
        if path.exists():
            frame = pd.read_csv(path)
            table_shapes[name] = frame.shape

    s = lambda endpoint, model, rep: _fmt_score(_score_lookup(canonical, endpoint, model, rep))
    sig_maf_over_maf = _sig_count(figure_tests, "sig_maf_vs_maf_stack", positive=True)
    sig_over_burden_xgb = _sig_count(figure_tests, "signatures_vs_burden", positive=True, model_family="XGBoost")
    sig_over_burden_xgb_n = _tested_count(figure_tests, "signatures_vs_burden", model_family="XGBoost")
    sig_maf_over_sig_xgb = _sig_endpoints(figure_tests, "sig_maf_vs_signatures", model_family="XGBoost", positive=True)
    table2_rows = table_shapes.get("tables/table_2_full_performance_metrics.csv", (len(canonical), 0))[0]
    table_s2_rows = table_shapes.get("supplement/table_s2_sensitivity_analyses.csv", (0, 0))[0]
    table_s5_rows = table_shapes.get("supplement/table_s5_bio_maf_v4_feature_guide.csv", (0, 0))[0]
    table_s6_rows = table_shapes.get("supplement/table_s6_bio_maf_v4_nested_feature_selection.csv", (0, 0))[0]
    burden = _display_label("representation_family", "burden_only")
    signatures = _display_label("representation_family", "signatures_only")
    muat_label = _display_label("representation_family", "MuAt_style_attention_MIL")
    maf_stack = _display_label("representation_family", "MAF_stack_only")
    sig_maf = _display_label("representation_family", "signatures_plus_MAF_stack")
    xgboost = _display_label("model_family", "XGBoost")
    elastic_net = _display_label("model_family", "elastic_net")
    damage = _display_label("endpoint", "damage_class")
    hrd_score = _display_label("endpoint", "HRD_Score")
    hrd24 = _display_label("endpoint", "hrd_binary_24")
    hrd33 = _display_label("endpoint", "hrd_binary_33")
    hrd42 = _display_label("endpoint", "hrd_binary_42")
    hrd_thresholds = f"{hrd24}, {hrd33}, and {hrd42}"
    cancer_type = _display_label("endpoint", "cancer_type_top20")
    os_cox = _display_label("endpoint", "OS")
    muat_measured_non_survival = _measured_muat_endpoints(canonical, settings, include_survival=False)
    muat_measured_all = _measured_muat_endpoints(canonical, settings, include_survival=True)
    muat_measured_labels = [_display_label("endpoint", endpoint) for endpoint in muat_measured_non_survival]
    muat_measured_clause = ", ".join(muat_measured_labels) if muat_measured_labels else "the measured comparator endpoints"
    muat_missing_main = [
        endpoint
        for endpoint in FIGURE_MODEL_COMPARISON_ENDPOINTS
        if endpoint not in set(muat_measured_non_survival)
    ]
    muat_missing_clause = "non-HRD endpoints in the tabular panel" if muat_missing_main else "no active endpoints"
    muat_score_clauses = [
        f"{_display_label('endpoint', endpoint)} {s(endpoint, 'MuAt-compatible reimplementation', 'MuAt_style_attention_MIL')}"
        for endpoint in muat_measured_non_survival
    ]
    muat_score_sentence = "; ".join(muat_score_clauses)
    muat_baseline_clauses = [
        f"{_display_label('endpoint', endpoint)}: {s(endpoint, 'MuAt-compatible reimplementation', 'MuAt_style_attention_MIL')} for MuAt-compatible versus {s(endpoint, 'XGBoost', 'signatures_plus_MAF_stack')} for {sig_maf} {xgboost}"
        for endpoint in muat_measured_non_survival
    ]
    muat_baseline_sentence = "; ".join(muat_baseline_clauses)

    lines = [
        "# Manuscript Captions And Results Text",
        "",
        f"Author note: this text describes the generated manuscript outputs currently produced by the pipeline. "
        f"Figure 3 uses the MuAt-compatible event-bag comparator as the direct neural-model comparison, while UGA and channel-KME variants are handled as supplementary geometry analyses.",
        "",
        "## Captions",
        "",
        "### Figure 1. Conceptual overview of mutation-catalogue representations.",
        "",
        "Each sample is represented as a catalogue of somatic mutation events, which can be transformed into complementary tabular feature families. "
        "Signature features summarize mutation spectra, geometry features encode sequence-context distributions from FASTA-derived windows or UGA/channel encodings, "
        "and MAF-stack features aggregate event-level biological annotations such as gene, locus, consequence, and burden summaries. "
        "Combined representations concatenate process-level spectra with event-level biology. Non-survival tabular endpoints are evaluated with nested elastic-net/logistic and XGBoost models, while survival endpoints are reported separately with scikit-survival CoxNet C-index. "
        f"A MuAt-compatible event-bag comparator is measured directly on mutation events for {muat_measured_clause} rather than presented only as a conceptual alternative.",
        "",
        "### Figure 2. Signature baselines compared with mutational burden.",
        "",
        f"Nested five-fold out-of-fold performance is shown for {burden} and {signatures} across the non-survival model-comparison endpoints. "
        f"Primary metrics are pooled Spearman correlation for continuous {hrd_score}, AUROC for binary endpoints, and balanced accuracy for multiclass endpoints ({damage} and {cancer_type}). "
        f"{signatures} improve over {burden} for {xgboost} on {damage} ({s('damage_class', 'XGBoost', 'signatures_only')} vs {s('damage_class', 'XGBoost', 'burden_only')}), "
        f"{hrd_score} ({s('HRD_Score', 'XGBoost', 'signatures_only')} vs {s('HRD_Score', 'XGBoost', 'burden_only')}), "
        f"{hrd33} ({s('hrd_binary_33', 'XGBoost', 'signatures_only')} vs {s('hrd_binary_33', 'XGBoost', 'burden_only')}), "
        f"and {cancer_type} ({s('cancer_type_top20', 'XGBoost', 'signatures_only')} vs {s('cancer_type_top20', 'XGBoost', 'burden_only')}). "
        f"CoxNet survival rows are reported separately with {os_cox} C-index.",
        "",
        "### Figure 3. MuAt-compatible event-bag comparator.",
        "",
        f"The MuAt-compatible reimplementation is evaluated on measured comparator endpoints with the same held-out folds used by the corresponding tabular benchmark: {muat_measured_clause}. "
        f"The non-survival MuAt-compatible primary scores are {muat_score_sentence}. "
        f"The directly comparable tabular baselines are {muat_baseline_sentence}. "
        f"MuAt-compatible rows are shown only where measured; no MuAt-compatible result is reported for {muat_missing_clause}.",
        "",
        "### Figure 4. Event-level MAF-stack features and combined signature-plus-event representations.",
        "",
        f"This figure compares {signatures}, {maf_stack}, and {sig_maf} for each endpoint and model family. "
        f"{xgboost} with {sig_maf} gives the strongest non-survival tabular results for {hrd_score} ({s('HRD_Score', 'XGBoost', 'signatures_plus_MAF_stack')}), "
        f"{hrd33} ({s('hrd_binary_33', 'XGBoost', 'signatures_plus_MAF_stack')}), "
        f"and {cancer_type} ({s('cancer_type_top20', 'XGBoost', 'signatures_plus_MAF_stack')}). "
        f"{maf_stack} alone improves over {signatures} for {xgboost} {cancer_type} ({s('cancer_type_top20', 'XGBoost', 'MAF_stack_only')} vs {s('cancer_type_top20', 'XGBoost', 'signatures_only')}), "
        f"but underperforms {signatures} for {damage} ({s('damage_class', 'XGBoost', 'MAF_stack_only')} vs {s('damage_class', 'XGBoost', 'signatures_only')}). "
        f"The combined representation improves over {maf_stack} in {sig_maf_over_maf} of 10 tested Figure 4 comparisons at q < 0.05, showing that process-level spectra and event-level biology are complementary.",
        "",
        "### Figure 5. Overall survival CoxNet benchmark.",
        "",
        f"The survival benchmark is shown separately as Harrell C-index for scikit-survival CoxNet risk ranking. "
        f"For {os_cox}, {burden} reached {s('OS', 'cox_ph', 'burden_only')}, {signatures} reached {s('OS', 'cox_ph', 'signatures_only')}, "
        f"{maf_stack} reached {s('OS', 'cox_ph', 'MAF_stack_only')}, and {sig_maf} reached {s('OS', 'cox_ph', 'signatures_plus_MAF_stack')}. "
        "Keeping survival in its own panel avoids mixing time-to-event C-index with the non-survival model-comparison metrics.",
        "",
        "### Table 1. Datasets, endpoints, and evaluation design.",
        "",
        "This table summarizes the seven main manuscript endpoints, sample counts, task types, data sources, primary metrics, and label definitions. "
        f"The main panel includes {damage}, {hrd_score}, {hrd_thresholds}, MC3 {cancer_type}, and TCGA CDR {os_cox}. "
        "Supplementary endpoints are listed separately in Supplementary Table S1. Model-based results use five outer folds with one inner validation split and pooled global out-of-fold metrics; the single inner split is retained as a leakage-prevention and runtime compromise for the current expensive rerun.",
        "",
        "### Table 2. Main-panel performance matrix.",
        "",
        f"This table is the compact numeric backbone for the main manuscript figures, containing {table2_rows} endpoint rows. "
        "Non-survival representation columns report elastic-net and XGBoost scores as EN / XGB, using the endpoint-specific primary metric; survival rows report CoxNet C-index separately. "
        "Full provenance-heavy versions with run identifiers, cache keys, and source files are retained under `tables/technical/`.",
        "",
        "### Table 3. Representation summary and dimensionality.",
        "",
        "This table summarizes the main representations, their input signal, feature dimensionality range, context or atlas status, evaluated models, and manuscript role. "
        f"{burden} features are compact with a median of 3 features; {signatures} have a median of 182 features; "
        f"{maf_stack} dimensionality varies because predeclared Bio MAF v4 feature blocks are selected inside each outer fold's inner-validation split, and {sig_maf} adds the same nested-selected biology blocks to mutational spectra. "
        f"The {muat_label} comparator is reported separately as an event-bag neural model with learned tumour-level features. These values make the performance/complexity tradeoff explicit.",
        "",
        "### Table 4. Key terminology and abbreviations.",
        "",
        "This short glossary defines the key abbreviations, metrics, representation names, and model shorthand needed to read the main tables. "
        "The full machine-readable label mapping is retained in `tables/technical/table_4_label_mapping_technical.csv`.",
        "",
        "### Supplementary Figure S1. Representation construction and reproducibility workflow.",
        "",
        "This schematic details how raw mutation catalogues are transformed into spectra, UGA/channel-KME variants, MAF-stack aggregates, combined tabular representations, and MuAt-compatible event bags. "
        "It also illustrates the cache/checkpoint workflow used to make feature generation reusable and restartable. Context-derived features use GRCh37 FASTA windows where appropriate, while atlas-based UGA/channel features are treated as supplementary geometry variants.",
        "",
        "### Supplementary Figure S2. Calibration of selected main models.",
        "",
        f"Reliability curves are shown for classification endpoints using the selected {sig_maf} {xgboost} models. Calibration is evaluated from out-of-fold predictions for {damage}, {hrd_thresholds}, and {cancer_type}. "
        "These plots check whether the strongest models' predicted probabilities are broadly aligned with observed event frequencies rather than merely improving rank-based metrics.",
        "",
        "### Supplementary Figure S3. Supplementary measured representation panels.",
        "",
        "Measured supplementary results are shown for alternative geometry encodings and COSMIC/NNLS exposure checks. Visible marks are measured only and specify the model family or analysis family used. "
        "Unsupported or intentionally omitted combinations are excluded from the figure and documented separately in Supplementary Table S3.",
        "",
        "### Supplementary Figure S4A. Bio MAF v4 feature groups, without the jargon.",
        "",
        "This plain-language overview explains how Bio MAF v4 turns a patient's mutation file and fixed cancer reference resources into five groups of biological features. "
        "The figure separates the readable overview from the technical audit trail: VEP annotation features come from a fixed VEP label vocabulary, external evidence-confidence controls summarize reference support without treating source count as a tumor biology mechanism, exact-gene features come from a fixed 401-gene external rule, and hotspot features come from Cancer Hotspots. "
        "Feature values are per-sample aggregates: count-like features use log1p(count), fraction features divide by total annotated mutations, and allele-fraction features use summary statistics or maxima. "
        "The bottom jargon decoder defines VEP labels, allele fraction, log-counts, protein-affecting mutation counts, cancer-role-fit mutation counts, and fixed-before-training language; Supplementary Table S5 gives worked feature examples, exact feature definitions, and candidate-size derivations, and Supplementary Table S6 reports nested-CV-selected feature groups.",
        "",
        "### Supplementary Figure S4B. Distribution of representative Bio MAF v4 features.",
        "",
        "This figure shows empirical distributions for representative Bio MAF v4 features in the fixed 8,800-sample TCGA top-20 cancer-type cohort. "
        "Panels include compact VEP annotation features, allele-fraction summaries, evidence-confidence controls, exact TP53 gene features, Cancer Hotspots summaries, and optional mutation-burden controls. "
        "The annotations report nonzero percentage, median, and 95th percentile so readers can see which features are continuous, sparse, or strongly zero-inflated.",
        "",
        "### Supplementary Figure S4C. Representative Bio MAF v4 feature nonzero rates by cancer type.",
        "",
        "This heatmap summarizes the same feature-family logic by cancer type. "
        "Each cell is the percent of tumors in a TCGA top-20 cancer type with a nonzero value for the representative feature, highlighting that exact-gene and hotspot features are intentionally sparse while broad VEP and burden features are present in most tumors. "
        "The figure is descriptive only and is not used for feature selection.",
        "",
        "### Supplementary Figure S5. Nested-CV Bio MAF v4 feature-block selections.",
        "",
        "This heatmap reports which predeclared Bio MAF v4 feature block was selected by the inner-loop validation search within each outer fold. "
        "Cell values count outer folds across manuscript endpoints with Bio MAF v4 feature-block tuning. "
        "The figure separates Bio MAF-only models from signatures-plus-biology models so readers can see when the inner loop selected compact biology, exact driver-gene features, the full strict core, optional controls, or signatures alone.",
        "",
        "### Supplementary Table S1. Supplementary endpoint inventory.",
        "",
        "This table lists supplementary endpoints, sample counts, task types, data sources, primary metrics, and representation families evaluated outside the simplified main panel.",
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
        "### Supplementary Table S4. TCGA top-20 cancer-type metrics.",
        "",
        f"This table reports complementary classification metrics for {cancer_type}, including balanced accuracy, macro-AUROC, micro-AUROC, accuracy, macro-F1, Cohen's kappa, and top-k accuracy where available. Balanced accuracy is the primary ranking metric.",
        "",
        "### Supplementary Table S5. Bio MAF v4 feature guide.",
        "",
        f"This {table_s5_rows}-row guide explains what the Bio MAF v4 feature blocks contain, how each block should be interpreted, where the feature counts came from, how feature values are calculated, and why each block is scientifically grounded. "
        "For each feature block, it includes a representative real feature name, a toy MAF input, the exact arithmetic used to turn that toy input into a feature value, and a plain-English interpretation of the resulting number. "
        "It explicitly defines VEP labels, SIFT, PolyPhen, transcript biotype, canonical transcript, OncoKB roles, Vogelstein genes, fixed-before-modeling rules, and the external evidence-confidence control block. "
        "It also defines protein-affecting mutation counts and cancer-role-fit mutation counts, then lists every nested-CV Bio MAF candidate feature-set size derivation: compact only, compact plus evidence-confidence controls, compact plus hotspots, compact plus exact cancer-gene identity, full strict core, and full plus optional controls. "
        "The accompanying technical table gives the exact feature-level glossary for all 948 Bio MAF v4 features, including the 943 strict-core features and 5 optional controls. "
        "Exact consensus driver-gene features come from a fixed external rule before cross-validation; nested CV selects among predeclared blocks inside the inner loop rather than using held-out endpoint performance.",
        "",
        "### Supplementary Table S6. Nested-CV Bio MAF v4 feature-block selections.",
        "",
        f"This {table_s6_rows}-row table summarizes the feature-block candidate selected inside the inner loop for each endpoint, representation, and model. "
        "The corresponding technical table reports the exact by-fold selection decisions. "
        "Because selection occurs within the training partition of each outer fold, these rows document the tuning choices without leaking held-out test-fold information.",
        "",
        "## Results Section Text",
        "",
        f"We first established the strength of conventional mutational spectra relative to a minimal burden baseline. In the canonical main panel, {xgboost} models using {signatures} outperformed {burden} across the non-survival model-comparison endpoints: "
        f"{damage} improved from {s('damage_class', 'XGBoost', 'burden_only')} to {s('damage_class', 'XGBoost', 'signatures_only')} balanced accuracy, "
        f"{hrd_score} from {s('HRD_Score', 'XGBoost', 'burden_only')} to {s('HRD_Score', 'XGBoost', 'signatures_only')} Spearman correlation, "
        f"{hrd33} from {s('hrd_binary_33', 'XGBoost', 'burden_only')} to {s('hrd_binary_33', 'XGBoost', 'signatures_only')} AUROC, "
        f"{cancer_type} from {s('cancer_type_top20', 'XGBoost', 'burden_only')} to {s('cancer_type_top20', 'XGBoost', 'signatures_only')} balanced accuracy, "
        f"Survival is reported separately as CoxNet C-index rather than being mixed into the elastic-net/XGBoost comparison figures. "
        f"These gains were statistically significant for {sig_over_burden_xgb} of {sig_over_burden_xgb_n} tested {xgboost} comparisons after FDR correction. {elastic_net} models showed the same qualitative gain for Kucab and cancer-type prediction, but not for every clinical or HRD endpoint; "
        f"in particular, {burden} exceeded {signatures} for {elastic_net} {hrd33}. Thus, signatures are a strong baseline, but their advantage depends on both endpoint and model class.",
        "",
        f"We also added a direct neural event-bag comparator. The MuAt-compatible reimplementation uses mutation motif, 1-Mb position-bin, and annotation tokens with a Q/K/V attention architecture, and is evaluated on measured comparator endpoints ({muat_measured_clause}) using the corresponding manuscript outer folds. "
        f"Across these HRD endpoints, the directly comparable scores are {muat_baseline_sentence}. This result should be interpreted as a faithful, comparable local reimplementation rather than as a claim about the official pretrained MuAt model; endpoints without measured MuAt-compatible OOF outputs are not displayed as comparator results.",
        "",
        f"Event-level MAF-stack features provided a complementary source of biological information. {maf_stack} alone was particularly useful for cancer-type prediction with {xgboost}, improving over {signatures} from {s('cancer_type_top20', 'XGBoost', 'signatures_only')} to {s('cancer_type_top20', 'XGBoost', 'MAF_stack_only')} balanced accuracy. "
        f"However, it was not uniformly better than spectra: for {damage}, {maf_stack} alone was lower than {signatures} with {xgboost} ({s('damage_class', 'XGBoost', 'MAF_stack_only')} vs {s('damage_class', 'XGBoost', 'signatures_only')}), "
        "consistent with the idea that mechanistic mutagen exposure is better captured by sequence-context or spectral information than by event-level gene/locus aggregates alone.",
        "",
        f"The strongest overall pattern emerged from combining spectra with event-level MAF features. {sig_maf} was the best overall representation for most non-survival main endpoints with {xgboost}: "
        f"{hrd_score} reached {s('HRD_Score', 'XGBoost', 'signatures_plus_MAF_stack')} Spearman correlation, "
        f"{hrd33} reached {s('hrd_binary_33', 'XGBoost', 'signatures_plus_MAF_stack')} AUROC, "
        f"and {cancer_type} reached {s('cancer_type_top20', 'XGBoost', 'signatures_plus_MAF_stack')} balanced accuracy. "
        f"In the separately reported survival analysis, CoxNet {os_cox} reached {s('OS', 'cox_ph', 'signatures_plus_MAF_stack')} C-index. "
        f"The only main endpoint where it did not win was {damage}, where {signatures} remained slightly higher for {xgboost}. "
        f"In pairwise tests, {sig_maf} significantly improved over {maf_stack} alone in {sig_maf_over_maf} of 10 Figure 4 comparisons and significantly improved over {signatures} alone for {xgboost} {', '.join(_display_label('endpoint', endpoint) for endpoint in sig_maf_over_sig_xgb)}.",
        "",
        f"Taken together, the cross-endpoint summary shows that there is no single magic representation. {signatures} remain a strong and efficient baseline, {maf_stack} features capture endpoint-relevant biology that spectra alone can miss, and the MuAt-compatible event-bag comparator does not automatically exceed a tuned tabular baseline under this directly comparable evaluation. "
        f"Across the main panel, the most robust practical default for tabular models is {sig_maf}, particularly when paired with {xgboost}. "
        "The supplementary analyses document additional geometry variants, exposure checks, endpoint extensions, and completeness metadata without introducing N/A placeholders into the main manuscript figures.",
        "",
        "## Statistical Notes",
        "",
        f"Primary metrics are pooled Spearman correlation for {hrd_score}, AUROC for binary endpoints, balanced accuracy for multiclass endpoints ({damage} and {cancer_type}), and Harrell C-index for CoxNet survival endpoints. "
        "Statistical statements refer to canonical/main_panel_pairwise_tests.csv, using paired DeLong tests for binary AUROC and paired bootstrap tests for balanced accuracy, C-index, or Spearman correlation.",
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
        raise ValueError("Missing manuscript display columns\n" + "\n".join(errors))


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
        values = table2[column].fillna("").astype(str)
        if column == MAIN_REPRESENTATION_SHORT_LABELS["MuAt_style_attention_MIL"]:
            bad = values.map(lambda value: " / " in value)
            if bad.any():
                raise ValueError(f"Table 2 column {column} must use single-score MuAt cells")
            continue
        non_survival = table2["Endpoint"].astype(str).ne(_display_label("endpoint", "OS"))
        bad_non_survival = values.loc[non_survival].map(lambda value: bool(value) and " / " not in value)
        bad_survival = values.loc[~non_survival].map(lambda value: " / " in value)
        if bad_non_survival.any() or bad_survival.any():
            raise ValueError(f"Table 2 column {column} has malformed score cells")
    s2 = pd.read_csv(manuscript_dir / "supplement" / "table_s2_sensitivity_analyses.csv")
    s2_technical_path = manuscript_dir / "tables" / "technical" / "table_s2_sensitivity_analyses_technical.csv"
    if not s2_technical_path.exists():
        raise FileNotFoundError("Technical Supplementary Table S2 is missing")
    s2_technical = pd.read_csv(s2_technical_path)
    manuscript_df = _manuscript_endpoint_frame(df)
    expected_s2_rows = int(
        (
            manuscript_df["endpoint_tier"].astype(str).eq("supplement")
            | manuscript_df["representation_family"].isin(["UGA_geometry", "COSMIC_NNLS_exposures"])
        )
        .loc[~manuscript_df["representation_family"].eq("channel_KME")]
        .sum()
    )
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
    value_offenders: list[dict[str, str]] = []
    value_paths = (
        sorted((manuscript_dir / "tables").glob("**/*.csv"))
        + sorted((manuscript_dir / "supplement").glob("*.csv"))
        + sorted((manuscript_dir / "plot_data").glob("*.csv"))
        + sorted((manuscript_dir / "canonical").glob("*.csv"))
    )
    for path in value_paths:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        text_columns = [col for col in frame.columns if frame[col].dtype == object]
        for term in MANUSCRIPT_EXCLUDED_ENDPOINT_LABEL_TERMS:
            for column in text_columns:
                if frame[column].str.contains(re.escape(term), case=False, na=False).any():
                    value_offenders.append({"file": str(path.relative_to(manuscript_dir)), "term": term, "column": column})
                    break
    if value_offenders:
        raise ValueError(f"Manuscript-ready outputs contain excluded endpoint labels: {json.dumps(value_offenders[:50], indent=2)}")


def _validate(manuscript_dir: Path, df: pd.DataFrame, *, strict: bool) -> None:
    missing = [rel for rel in REQUIRED_MANUSCRIPT_FILES if not (manuscript_dir / rel).exists()]
    if (manuscript_dir / "d3_render_manifest.json").exists():
        for folder, stems in {
            "figures": [
                "figure_1_conceptual_overview",
                "figure_2_signature_baselines",
                "figure_3_geometry_vs_signatures",
                "figure_4_maf_stack_vs_signatures",
                "figure_5_overall_survival_cox",
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
    endpoint_results = _filter_enabled_endpoint_results(endpoint_results, settings)
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
        settings=settings,
    )
    table_df = pd.concat(
        [
            canonical,
            normalized[normalized["endpoint_tier"].eq("supplement")],
        ],
        ignore_index=True,
        sort=False,
    )
    _write_tables(table_df, side_tables, manuscript_dir, settings=settings)
    measured_only_main = bool(strict or (settings.get("outputs") or {}).get("strict_plot_completeness", False))
    _write_plot_data(normalized, canonical, canonical_oof, pairwise_tests, manuscript_dir, measured_only_main=measured_only_main, settings=settings)
    _write_label_mapping(manuscript_dir)
    _write_manuscript_text(manuscript_dir, canonical, pairwise_tests, settings=settings)
    renderer = str((settings.get("outputs") or {}).get("visualization_renderer", "matplotlib")).lower()
    if renderer == "d3":
        _run_d3_renderer(manuscript_dir, strict=strict)
    else:
        _write_figures(table_df, manuscript_dir)
    _figure_bio_maf_v4_feature_map(manuscript_dir / "supplement" / "figure_s4_bio_maf_v4_feature_map")
    _figure_bio_maf_v4_feature_distributions(
        manuscript_dir / "supplement" / "figure_s4b_bio_maf_v4_feature_distributions",
        manuscript_dir / "tables" / "technical",
    )
    _figure_bio_maf_v4_feature_distributions_by_cancer_type(
        manuscript_dir / "supplement" / "figure_s4c_bio_maf_v4_feature_distributions_by_cancer_type"
    )
    _figure_bio_maf_v4_nested_selection(manuscript_dir / "supplement" / "figure_s5_bio_maf_v4_nested_feature_selection")
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
    parser.add_argument("--config", default="config/experiment_settings.strict_no_leakage.yaml")
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    settings = load_yaml(args.config)
    paths = resolve_paths_map(load_yaml(args.paths))
    make_all_figures(settings=settings, paths=paths, strict=bool(args.strict or (settings.get("outputs") or {}).get("strict_manuscript", False)))


if __name__ == "__main__":
    main()
