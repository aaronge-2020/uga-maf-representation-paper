"""Bio MAF v4 feature-selection and SHAP-style interpretation report.

This script is intentionally post hoc. It does not estimate predictive
performance; performance remains the nested out-of-fold benchmark output.
Here we summarize which predeclared Bio MAF v4 blocks were selected inside
the nested-CV tuning loop, then fit a bounded interpretation model to explain
which biological annotation features drive the top-20 cancer-type signal.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

try:
    import shap
except ImportError:  # pragma: no cover - dependency is present in the manuscript env.
    shap = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - dependency is present in the manuscript env.
    xgb = None

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import BUNDLE_ROOT, load_yaml, resolve_paths_map
from utils.endpoint_registry import TCGA_CANCER_TYPE_TOP20, build_cdr_cancer_type_labels
from utils.quick_bio_v4_nested_feature_selection import v4_feature_columns


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "blue": "#5477C4",
    "gold": "#B8A037",
    "orange": "#CC6F47",
    "olive": "#71B436",
    "pink": "#BD569B",
}


def _style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.titlecolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "font.family": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        }
    )


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _write_table(frame: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    html = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#1F2430}"
        "table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{border-bottom:1px solid #E6E8F0;padding:7px 9px;text-align:left;vertical-align:top}"
        "th{background:#F4F5F7}td.num{text-align:right;font-variant-numeric:tabular-nums}</style>",
        f"<h1>{title}</h1>",
        frame.to_html(index=False, escape=False, classes="data"),
    ]
    path.with_suffix(".html").write_text("\n".join(html), encoding="utf-8")


def _display_feature_set(name: object) -> str:
    text = "" if pd.isna(name) else str(name)
    mapping = {
        "v4_compact_biology_only": "Compact biology only",
        "v4_compact_plus_driver_evidence_tiers": "Compact + evidence-confidence controls",
        "v4_compact_plus_exact_driver_genes": "Compact + exact driver genes",
        "v4_compact_plus_hotspot_summary": "Compact + hotspot summaries",
        "v4_full_core": "Full core biology",
        "v4_full_core_plus_optional_controls": "Full core + optional controls",
        "signatures_only": "Signatures only",
        "signatures_plus_v4_compact_biology_only": "Signatures + compact biology",
        "signatures_plus_v4_compact_plus_driver_evidence_tiers": "Signatures + evidence-confidence controls",
        "signatures_plus_v4_compact_plus_exact_driver_genes": "Signatures + exact driver genes",
        "signatures_plus_v4_compact_plus_hotspot_summary": "Signatures + hotspot summaries",
        "signatures_plus_v4_full_core": "Signatures + full core biology",
        "signatures_plus_v4_full_core_plus_optional_controls": "Signatures + full core + optional controls",
    }
    return mapping.get(text, text.replace("_", " ").strip().title())


def _feature_family(feature: str) -> str:
    if feature.startswith("signature_matrix__"):
        return "Mutational signature context"
    if feature.startswith("driver_gene_role_matched_event_log_count__"):
        return "Exact cancer gene, cancer-role-fit mutation count"
    if feature.startswith("driver_gene_functional_event_log_count__"):
        return "Exact cancer gene, protein-affecting mutation count"
    if feature.startswith("driver_evidence_"):
        return "External evidence-confidence control"
    if feature.startswith("cancer_hotspot_"):
        return "Cancer hotspot summary"
    if feature.startswith("oncokb_role_"):
        return "OncoKB role aggregate"
    if feature.startswith("residual_non_oncokb"):
        return "Non-OncoKB cancer-gene aggregate"
    if feature.startswith("vep_consequence_"):
        return "VEP consequence composition"
    if feature.startswith("vep_impact_"):
        return "VEP impact composition"
    if feature.startswith("vep_sift_") or feature.startswith("vep_polyphen_"):
        return "Predicted functional damage"
    if feature.startswith("vep_biotype_") or feature.startswith("vep_canonical_"):
        return "Transcript annotation composition"
    if feature.startswith("vaf_"):
        return "Variant allele fraction summary"
    return "Other biological annotation"


def _plain_feature_meaning(feature: str) -> str:
    if feature.startswith("signature_matrix__"):
        return "A mutational-signature channel count/fraction carried only in combined models."
    if feature.startswith("driver_gene_role_matched_event_log_count__"):
        gene = feature.rsplit("__", 1)[-1]
        return f"How many mutations affect {gene} in a way consistent with that gene's curated cancer role."
    if feature.startswith("driver_gene_functional_event_log_count__"):
        gene = feature.rsplit("__", 1)[-1]
        return f"How many mutations in {gene} are predicted by fixed VEP rules to change or disrupt the encoded protein."
    if feature.startswith("driver_evidence_"):
        tier = feature.rsplit("_", 1)[-1]
        return f"Aggregate mutation signal among genes named by {tier} external cancer-reference source(s)."
    if feature.startswith("cancer_hotspot_"):
        return "Mutation count or maximum allele fraction for mutations at known cancer hotspot residues or exact hotspot variants."
    if feature.startswith("oncokb_role_"):
        return "Aggregate burden in OncoKB-curated oncogene/tumor-suppressor role groups."
    if feature.startswith("residual_non_oncokb"):
        return "Aggregate burden in externally supported cancer genes not covered by the OncoKB role group."
    if feature.startswith("vep_consequence_"):
        return "Composition of mutation consequence types assigned by VEP."
    if feature.startswith("vep_impact_"):
        return "Composition of VEP impact levels such as high, moderate, low, or modifier."
    if feature.startswith("vep_sift_") or feature.startswith("vep_polyphen_"):
        return "Composition of predicted protein-damaging versus tolerated variants."
    if feature.startswith("vep_biotype_") or feature.startswith("vep_canonical_"):
        return "Transcript-level annotation composition from VEP."
    if feature.startswith("vaf_"):
        return "A sample-level summary of allele fractions, meaning the share of tumor DNA reads carrying each mutation; useful but should be checked as a purity/proxy sensitivity."
    return "A fixed Bio MAF v4 annotation aggregate."


def summarize_feature_selection(tables_dir: Path, out_dir: Path) -> pd.DataFrame:
    folds = pd.read_csv(tables_dir / "main_manuscript_complete_panel_fold_metrics.csv", low_memory=False)
    rows = folds[
        folds["representation"].astype(str).isin(["MAF_stack_only", "signatures_plus_MAF_stack"])
        & folds["selected_feature_set"].notna()
    ].copy()
    rows["feature_set_display"] = rows["selected_feature_set"].map(_display_feature_set)
    rows["selected_feature_count"] = pd.to_numeric(rows["selected_feature_count"], errors="coerce")
    grouped = (
        rows.groupby(["endpoint", "representation", "learner", "selected_feature_set", "feature_set_display"], dropna=False)
        .agg(
            outer_folds_selected=("fold", "nunique"),
            mean_outer_score=("score", "mean"),
            median_selected_features=("selected_feature_count", "median"),
            candidate_sets=("candidate_feature_set_count", "max"),
        )
        .reset_index()
        .sort_values(["endpoint", "representation", "learner", "outer_folds_selected", "mean_outer_score"], ascending=[True, True, True, False, False])
    )
    _write_table(grouped, out_dir / "bio_maf_v4_nested_feature_selection_by_block.csv", "Bio MAF v4 nested feature-selection by block")

    endpoint_summary = (
        rows.groupby(["endpoint", "representation", "learner"], dropna=False)
        .agg(
            folds=("fold", "nunique"),
            selected_blocks=("feature_set_display", lambda x: "; ".join(f"{k} ({v}/5)" for k, v in Counter(x).most_common())),
            median_selected_features=("selected_feature_count", "median"),
            min_selected_features=("selected_feature_count", "min"),
            max_selected_features=("selected_feature_count", "max"),
        )
        .reset_index()
    )
    _write_table(endpoint_summary, out_dir / "bio_maf_v4_nested_feature_selection_summary.csv", "Bio MAF v4 nested feature-selection summary")

    top20 = grouped[grouped["endpoint"].eq("cancer_type_top20")].copy()
    if not top20.empty:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        plot = top20.copy()
        plot["model"] = plot["representation"] + " / " + plot["learner"]
        sns.barplot(
            data=plot,
            y="feature_set_display",
            x="outer_folds_selected",
            hue="model",
            palette=[TOKENS["blue"], TOKENS["orange"], TOKENS["olive"], TOKENS["pink"]],
            ax=ax,
        )
        ax.set_title("Top-20 cancer-type feature blocks selected inside nested CV")
        ax.set_xlabel("Outer folds selecting the block")
        ax.set_ylabel("")
        ax.set_xlim(0, 5)
        ax.legend(title="", loc="lower right", frameon=False)
        _save_plot(fig, out_dir / "figure_bio_maf_v4_top20_feature_block_selection")
    return grouped


def summarize_top20_performance(tables_dir: Path, out_dir: Path) -> pd.DataFrame:
    results = pd.read_csv(tables_dir / "main_manuscript_complete_panel_endpoint_results.csv", low_memory=False)
    top20 = results[results["endpoint"].eq("cancer_type_top20")].copy()
    if top20.empty:
        return pd.DataFrame()
    keep = [
        "representation",
        "learner",
        "score",
        "auroc",
        "micro_auroc",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "cohen_kappa",
        "top3_accuracy",
        "top5_accuracy",
        "runtime_seconds",
        "selected_feature_count_median",
        "selected_feature_sets",
    ]
    keep = [col for col in keep if col in top20.columns]
    top20 = top20.loc[:, keep].sort_values(["balanced_accuracy", "score"], ascending=False)
    _write_table(top20, out_dir / "bio_maf_v4_top20_performance_metrics.csv", "TCGA top-20 performance metrics")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    plot = top20.copy()
    plot["model"] = plot["representation"] + " / " + plot["learner"]
    sns.barplot(data=plot, y="model", x="balanced_accuracy", color=TOKENS["blue"], ax=ax)
    ax.set_title("Bio MAF v4 top-20 cancer-type performance with balanced accuracy")
    ax.set_xlabel("Balanced accuracy")
    ax.set_ylabel("")
    ax.set_xlim(0, max(0.72, float(plot["balanced_accuracy"].max()) + 0.04))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=4, fontsize=9)
    _save_plot(fig, out_dir / "figure_bio_maf_v4_top20_balanced_accuracy")
    return top20


def _load_top20_features(paths: dict[str, Any], representation: str, feature_set: str, tables_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    bio = pd.read_csv(tables_dir / "quick_bio_v4_maf_features.csv.gz", index_col=0).fillna(0.0).astype(np.float32)
    bio.index = bio.index.astype(str)
    _core, _all, blocks = v4_feature_columns()
    if representation == "MAF_stack_only":
        if feature_set not in blocks:
            raise ValueError(f"{feature_set!r} is not a Bio MAF-only candidate block")
        features = bio.loc[:, list(blocks[feature_set])].copy()
    else:
        standard = pd.read_csv(Path(paths["raw_data"]["mc3_source_dir"]) / "features" / "features_standard_sbs96_id83.csv.gz", index_col=0).fillna(0.0).astype(np.float32)
        standard.index = standard.index.astype(str)
        sig = standard.add_prefix("signature_matrix__")
        if feature_set == "signatures_only":
            features = sig.copy()
        else:
            suffix = feature_set.replace("signatures_plus_", "", 1)
            if suffix not in blocks:
                raise ValueError(f"{feature_set!r} is not a signatures+Bio MAF candidate block")
            features = pd.concat([sig, bio.loc[:, list(blocks[suffix])]], axis=1).fillna(0.0).astype(np.float32)

    patients = pd.read_csv(Path(paths["raw_data"]["mc3_source_dir"]) / "features" / "features_burden_only.csv", usecols=[0], index_col=0).index.astype(str)
    labels = build_cdr_cancer_type_labels(
        Path(paths["raw_data"]["mc3_source_dir"]),
        patients=patients,
        classes=TCGA_CANCER_TYPE_TOP20,
        endpoint_name="cancer_type_top20",
    )
    common = features.index.astype(str).intersection(labels.index.astype(str))
    return features.loc[common].astype(np.float32), labels.loc[common].astype(str)


def _mode_selected_feature_set(tables_dir: Path, representation: str, learner: str) -> str:
    folds = pd.read_csv(tables_dir / "main_manuscript_complete_panel_fold_metrics.csv", low_memory=False)
    sub = folds[
        folds["endpoint"].eq("cancer_type_top20")
        & folds["representation"].eq(representation)
        & folds["learner"].eq(learner)
        & folds["selected_feature_set"].notna()
    ].copy()
    if sub.empty:
        raise ValueError(f"No selected feature-set rows found for cancer_type_top20/{representation}/{learner}")
    return Counter(sub["selected_feature_set"].astype(str)).most_common(1)[0][0]


def _fit_interpretation_xgb(x: pd.DataFrame, y: pd.Series, seed: int, params: dict[str, Any] | None = None) -> tuple[Any, LabelEncoder]:
    if xgb is None:
        raise RuntimeError("xgboost is not installed")
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y.astype(str))
    params = dict(params or {})
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        n_estimators=int(params.get("n_estimators", 220)),
        max_depth=int(params.get("max_depth", 4)),
        learning_rate=float(params.get("learning_rate", 0.06)),
        subsample=float(params.get("subsample", 0.85)),
        colsample_bytree=float(params.get("colsample_bytree", 0.85)),
        min_child_weight=float(params.get("min_child_weight", 3.0)),
        reg_lambda=float(params.get("reg_lambda", 2.0)),
        reg_alpha=float(params.get("reg_alpha", 0.1)),
        tree_method=str(params.get("tree_method", "hist")),
        eval_metric="mlogloss",
        n_jobs=int(params.get("n_jobs", 8)),
        random_state=int(seed),
    )
    model.fit(x.to_numpy(dtype=np.float32), y_enc)
    return model, encoder


def _stratified_sample(labels: pd.Series, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices: list[int] = []
    per_class = max(10, int(math.ceil(n / max(labels.nunique(), 1))))
    for _label, group in labels.reset_index(drop=True).groupby(labels.reset_index(drop=True), sort=True):
        choices = group.index.to_numpy(dtype=int)
        take = min(len(choices), per_class)
        indices.extend(rng.choice(choices, size=take, replace=False).tolist())
    if len(indices) > n:
        indices = rng.choice(np.array(indices, dtype=int), size=n, replace=False).tolist()
    return np.array(sorted(indices), dtype=int)


def run_shap_analysis(paths: dict[str, Any], tables_dir: Path, out_dir: Path, representation: str, learner: str, shap_n: int, seed: int) -> dict[str, Any]:
    if shap is None:
        raise RuntimeError("shap is not installed")
    feature_set = _mode_selected_feature_set(tables_dir, representation, learner)
    features, labels = _load_top20_features(paths, representation, feature_set, tables_dir)
    model, encoder = _fit_interpretation_xgb(features, labels, seed)
    y_pred = encoder.inverse_transform(np.asarray(model.predict(features.to_numpy(dtype=np.float32)), dtype=int))
    train_metrics = {
        "interpretation_fit_accuracy": float(accuracy_score(labels, y_pred)),
        "interpretation_fit_balanced_accuracy": float(balanced_accuracy_score(labels, y_pred)),
        "interpretation_fit_macro_f1": float(f1_score(labels, y_pred, average="macro")),
    }

    sample_idx = _stratified_sample(labels, int(shap_n), seed)
    sample = features.iloc[sample_idx].copy()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample.to_numpy(dtype=np.float32))
    if isinstance(shap_values, list):
        arr = np.stack([np.asarray(v, dtype=np.float64) for v in shap_values], axis=2)
    else:
        arr = np.asarray(shap_values, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        elif arr.shape[1] == len(encoder.classes_) and arr.shape[2] == sample.shape[1]:
            arr = np.transpose(arr, (0, 2, 1))
    mean_abs = np.nanmean(np.abs(arr), axis=(0, 2))
    top = pd.DataFrame(
        {
            "feature": sample.columns.astype(str),
            "feature_family": [_feature_family(str(col)) for col in sample.columns],
            "mean_abs_shap": mean_abs,
            "plain_english_meaning": [_plain_feature_meaning(str(col)) for col in sample.columns],
        }
    ).sort_values("mean_abs_shap", ascending=False)
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    _write_table(top.head(100), out_dir / f"bio_maf_v4_top20_{representation}_{learner}_top100_shap_features.csv", "Top Bio MAF v4 SHAP features for TCGA top-20")

    family = (
        top.groupby("feature_family", as_index=False)
        .agg(total_mean_abs_shap=("mean_abs_shap", "sum"), mean_abs_shap=("mean_abs_shap", "mean"), features=("feature", "count"))
        .sort_values("total_mean_abs_shap", ascending=False)
    )
    family["share_of_total"] = family["total_mean_abs_shap"] / family["total_mean_abs_shap"].sum()
    _write_table(family, out_dir / f"bio_maf_v4_top20_{representation}_{learner}_shap_feature_families.csv", "Bio MAF v4 SHAP attribution by feature family")

    class_rows = []
    class_mean_abs = np.nanmean(np.abs(arr), axis=0)
    for class_id, class_name in enumerate(encoder.classes_):
        values = class_mean_abs[:, class_id]
        order = np.argsort(-values)[:15]
        for rank, idx in enumerate(order, start=1):
            feature = str(sample.columns[idx])
            class_rows.append(
                {
                    "cancer_type": str(class_name),
                    "rank": rank,
                    "feature": feature,
                    "feature_family": _feature_family(feature),
                    "mean_abs_shap": float(values[idx]),
                    "plain_english_meaning": _plain_feature_meaning(feature),
                }
            )
    class_top = pd.DataFrame(class_rows)
    _write_table(class_top, out_dir / f"bio_maf_v4_top20_{representation}_{learner}_class_specific_shap.csv", "Class-specific Bio MAF v4 SHAP features")

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    plot = top.head(20).iloc[::-1]
    sns.barplot(data=plot, y="feature", x="mean_abs_shap", hue="feature_family", dodge=False, ax=ax)
    ax.set_title("Top Bio MAF v4 features driving 20-class cancer-type prediction")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("")
    ax.legend(title="", loc="lower right", frameon=False, fontsize=8)
    _save_plot(fig, out_dir / f"figure_bio_maf_v4_top20_{representation}_{learner}_top_shap_features")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    plot = family.head(12).iloc[::-1]
    sns.barplot(data=plot, y="feature_family", x="share_of_total", color=TOKENS["olive"], ax=ax)
    ax.set_title("Biological annotation families carrying the top-20 signal")
    ax.set_xlabel("Share of total mean absolute SHAP")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=4, fontsize=9)
    _save_plot(fig, out_dir / f"figure_bio_maf_v4_top20_{representation}_{learner}_shap_families")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "endpoint": "cancer_type_top20",
        "representation": representation,
        "learner": learner,
        "selected_feature_set_used_for_interpretation": feature_set,
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "shap_sample_n": int(len(sample)),
        "class_count": int(len(encoder.classes_)),
        "classes": [str(value) for value in encoder.classes_],
        "interpretation_fit_note": "Full-data fit used only for attribution; predictive performance is reported from nested out-of-fold benchmark rows.",
        **train_metrics,
    }
    (out_dir / f"bio_maf_v4_top20_{representation}_{learner}_shap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment_settings.explicit_bio_maf.yaml")
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--output-dir", default="results/manuscript/interpretability")
    parser.add_argument("--representation", default="MAF_stack_only")
    parser.add_argument("--learner", default="xgboost")
    parser.add_argument("--shap-sample", type=int, default=1600)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    _style()
    _settings = load_yaml(args.config)
    paths = resolve_paths_map(load_yaml(args.paths))
    tables_dir = BUNDLE_ROOT / "results" / "tables"
    out_dir = BUNDLE_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = summarize_feature_selection(tables_dir, out_dir)
    performance = summarize_top20_performance(tables_dir, out_dir)
    manifest: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "selection_rows": int(len(selection)),
        "top20_performance_rows": int(len(performance)),
        "shap_status": "skipped",
    }
    if not args.skip_shap:
        manifest["shap_status"] = "completed"
        manifest["shap"] = run_shap_analysis(
            paths,
            tables_dir,
            out_dir,
            representation=str(args.representation),
            learner=str(args.learner),
            shap_n=int(args.shap_sample),
            seed=int(args.seed),
        )
    (out_dir / "bio_maf_v4_interpretability_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
