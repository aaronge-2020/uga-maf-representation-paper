#!/usr/bin/env python3
"""PCAWG reference-signature attribution benchmark for the locked manuscript model."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from utils.checkpointing import atomic_write_csv, atomic_write_json, atomic_write_text, merge_checkpoint_rows, read_completed_keys

from locked_signature_exposure_utils import (
    CGR_ROOT,
    LOCKED_ID_MODEL,
    LOCKED_SBSDBS_MODEL,
    bh_q_values,
    cosine_similarity,
    extract_signature_exposures,
    write_html_table,
)


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
DATA_DIR = EXPERIMENT_ROOT / "data" / "pcawg_signature_attribution"
TABLE_DIR = EXPERIMENT_ROOT / "tables" / "pcawg_signature_attribution"

PCAWG_DIR = CGR_ROOT / "cgr_validation_results" / "research" / "data" / "pancan_pcawg_2020"
MODALITIES = ["SBS", "DBS", "ID"]
REPRESENTATIONS = ["standard", "locked_uga"]
METRIC_KEY_COLUMNS = ["modality", "representation", "patient"]
METADATA_KEY_COLUMNS = ["modality", "representation"]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def pcawg_counts_path(modality: str) -> Path:
    return PCAWG_DIR / f"data_mutational_signatures_counts_{modality}.txt"


def pcawg_truth_path(modality: str) -> Path:
    return PCAWG_DIR / f"data_mutational_signatures_contribution_{modality}.txt"


def load_counts(modality: str) -> pd.DataFrame:
    path = pcawg_counts_path(modality)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t")
    channel_col = "NAME" if "NAME" in df.columns else "Type"
    patient_cols = [col for col in df.columns if str(col).startswith("SP")]
    counts = df.set_index(channel_col)[patient_cols].T
    counts.index = counts.index.astype(str)
    counts.columns = counts.columns.astype(str)
    return counts.fillna(0.0).astype(float)


def truth_signature_name(value: object) -> str:
    text = str(value).strip()
    if not text:
        return text
    return text.split()[0]


def load_truth(modality: str, sigs: list[str]) -> pd.DataFrame:
    path = pcawg_truth_path(modality)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t")
    name_col = "NAME" if "NAME" in df.columns else df.columns[0]
    patient_cols = [col for col in df.columns if str(col).startswith("SP")]
    df = df.copy()
    df["signature"] = df[name_col].map(truth_signature_name)
    truth = df.set_index("signature")[patient_cols].T
    truth = truth.reindex(columns=sigs).fillna(0.0).astype(float)
    sums = truth.sum(axis=1)
    truth.loc[sums > 1e-15] = truth.loc[sums > 1e-15].div(sums[sums > 1e-15], axis=0)
    truth.index = truth.index.astype(str)
    return truth


def evaluate_exposures(
    truth: pd.DataFrame,
    exposures: pd.DataFrame,
    modality: str,
    representation: str,
    burden: pd.Series,
) -> pd.DataFrame:
    common = truth.index.intersection(exposures.index)
    rows = []
    for patient in common:
        y = truth.loc[patient].to_numpy(dtype=np.float64)
        pred = exposures.loc[patient, truth.columns].to_numpy(dtype=np.float64)
        if burden.loc[patient] <= 0.0 or y.sum() <= 1e-15:
            continue
        rows.append(
            {
                "modality": modality,
                "representation": representation,
                "algorithm": "NNLS",
                "patient": patient,
                "mutation_burden": float(burden.loc[patient]),
                "cosine": cosine_similarity(y, pred),
                "mae": float(np.mean(np.abs(y - pred))),
                "n_active_truth_signatures": int(np.sum(y > 1e-12)),
                "n_active_predicted_signatures": int(np.sum(pred > 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame, metadata: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        metrics.groupby(["modality", "representation", "algorithm"], as_index=False)
        .agg(
            n_patients=("patient", "nunique"),
            mean_cosine=("cosine", "mean"),
            median_cosine=("cosine", "median"),
            mean_mae=("mae", "mean"),
            median_mae=("mae", "median"),
            mean_active_truth_signatures=("n_active_truth_signatures", "mean"),
            mean_active_predicted_signatures=("n_active_predicted_signatures", "mean"),
        )
        .sort_values(["modality", "representation"])
    )
    meta = pd.DataFrame(metadata)
    summary = summary.merge(
        meta[["modality", "representation", "feature_dimension", "n_signatures", "n_matched_channels", "n_encoded_channels"]],
        on=["modality", "representation"],
        how="left",
    )

    pair_rows = []
    for modality in MODALITIES:
        sub = metrics[metrics["modality"] == modality]
        for metric in ["cosine", "mae"]:
            wide = sub.pivot_table(index="patient", columns="representation", values=metric, aggfunc="first")
            if {"standard", "locked_uga"} - set(wide.columns):
                continue
            wide = wide.dropna(subset=["standard", "locked_uga"])
            delta = wide["locked_uga"] - wide["standard"]
            if len(wide) == 0:
                continue
            try:
                p_value = float(wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)
            except ValueError:
                p_value = np.nan
            ci_low, ci_high = np.percentile(delta.to_numpy(dtype=float), [2.5, 97.5])
            pair_rows.append(
                {
                    "modality": modality,
                    "algorithm": "NNLS",
                    "metric": metric,
                    "n_patients": int(len(wide)),
                    "standard_mean": float(wide["standard"].mean()),
                    "locked_uga_mean": float(wide["locked_uga"].mean()),
                    "delta_locked_uga_minus_standard": float(delta.mean()),
                    "delta_95pct_interval_low": float(ci_low),
                    "delta_95pct_interval_high": float(ci_high),
                    "p_value": p_value,
                    "better_direction": "positive" if metric == "cosine" else "negative",
                }
            )
    pairwise = pd.DataFrame(pair_rows)
    pairwise["q_value"] = bh_q_values(pairwise["p_value"].to_numpy())
    return summary, pairwise


def write_readme(summary: pd.DataFrame, pairwise: pd.DataFrame, metadata: dict[str, object]) -> None:
    primary = pairwise[pairwise["metric"] == "cosine"].copy()
    lines = [
        "# PCAWG Reference-Signature Attribution Benchmark",
        "",
        "## Research Question",
        "Does locked UGA projection improve recovery of PCAWG reference-signature exposures relative to standard COSMIC channel-space exposure extraction when the fitting algorithm and evaluation labels are held constant?",
        "",
        "## Methods",
        f"SBS96, DBS78, and ID83 PCAWG count matrices were fit against COSMIC v3.5 GRCh37 reference signatures using nonnegative least squares. Standard exposures were fit in the native COSMIC channel basis. Locked UGA exposures were fit after projecting the same channel-count matrix and the same COSMIC reference signatures into `{LOCKED_SBSDBS_MODEL}` for SBS/DBS and `{LOCKED_ID_MODEL}` for ID83. PCAWG contribution matrices were used as computational reference labels, normalized within modality before scoring. Attribution accuracy was measured by patient-level cosine similarity and mean absolute error.",
        "",
        "## Key Numerical Findings",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"- {row['modality']}: Standard mean cosine {row['standard_mean']:.4f}; locked UGA mean cosine {row['locked_uga_mean']:.4f}; delta {row['delta_locked_uga_minus_standard']:.4f}; q={row['q_value']:.4g}."
        )
    lines.extend(
        [
            "",
            "## File Inventory",
            "- `patient_metrics.csv`: patient-level cosine similarity and mean absolute error for each modality and representation.",
            "- `summary_metrics.csv`: modality-level mean and median attribution metrics.",
            "- `pairwise_tests.csv`: paired Standard versus locked UGA Wilcoxon tests with q values.",
            "- `exposure_feature_metadata.csv`: channel, signature, and UGA feature dimensions.",
            "- `table1_pcawg_signature_summary.html`: manuscript-ready summary table.",
            "- `table2_pcawg_signature_pairwise_tests.html`: manuscript-ready paired-test table.",
            "- `code/run_locked_pcawg_signature_attribution.py`: reproducible benchmark script.",
            "",
            "## Reproducibility",
            f"Executed at {metadata['executed_at_utc']} with locked SBS/DBS model `{LOCKED_SBSDBS_MODEL}`, locked ID model `{LOCKED_ID_MODEL}`, and NNLS exposure extraction. Runtime was {metadata['elapsed_seconds'] / 60.0:.1f} minutes.",
            "",
        ]
    )
    atomic_write_text(DATA_DIR / "README_pcawg_signature_attribution.md", "\n".join(lines))


def main() -> None:
    ensure_dirs()
    t0 = time.perf_counter()
    metric_checkpoint = DATA_DIR / "patient_metrics_checkpoint.csv"
    metadata_checkpoint = DATA_DIR / "exposure_feature_metadata_checkpoint.csv"
    final_metrics_path = DATA_DIR / "patient_metrics.csv"
    final_metadata_path = DATA_DIR / "exposure_feature_metadata.csv"
    if not metric_checkpoint.exists() and final_metrics_path.exists():
        atomic_write_csv(pd.read_csv(final_metrics_path), metric_checkpoint, index=False)
    if not metadata_checkpoint.exists() and final_metadata_path.exists():
        atomic_write_csv(pd.read_csv(final_metadata_path), metadata_checkpoint, index=False)
    completed_metrics = read_completed_keys(metric_checkpoint, ["modality", "representation"])
    metric_frames = []
    metadata_rows = []
    for modality in MODALITIES:
        print(f"PCAWG {modality}: loading counts", flush=True)
        counts = load_counts(modality)
        burden = counts.sum(axis=1)
        exposures_by_rep = {}
        for representation in REPRESENTATIONS:
            exposure_path = DATA_DIR / f"pcawg_{modality.lower()}_{representation}_nnls_exposures.csv"
            if exposure_path.exists():
                print(f"  [checkpoint] reuse {representation} NNLS exposures", flush=True)
                prefix = f"{modality}_{representation}_"
                exposures = pd.read_csv(exposure_path, index_col=0).rename(
                    columns=lambda col: str(col)[len(prefix) :] if str(col).startswith(prefix) else str(col)
                )
                _, _, metadata = extract_signature_exposures(counts, modality, representation)
            else:
                print(f"  extracting {representation} NNLS exposures", flush=True)
                exposures, _, metadata = extract_signature_exposures(counts, modality, representation)
                exposure_out = exposures.add_prefix(f"{modality}_{representation}_")
                atomic_write_csv(exposure_out, exposure_path)
            exposures_by_rep[representation] = exposures
            metadata_rows.append(metadata)
            merge_checkpoint_rows(metadata_checkpoint, [metadata], key_columns=METADATA_KEY_COLUMNS, sort_columns=METADATA_KEY_COLUMNS)
        sigs = exposures_by_rep["standard"].columns.tolist()
        truth = load_truth(modality, sigs)
        atomic_write_csv(truth, DATA_DIR / f"pcawg_{modality.lower()}_truth_normalized.csv")
        for representation, exposures in exposures_by_rep.items():
            key = tuple(str(value) for value in [modality, representation])
            if key in completed_metrics:
                print(f"  [checkpoint] skip metrics {key}", flush=True)
                continue
            frame = evaluate_exposures(truth, exposures, modality, representation, burden)
            merge_checkpoint_rows(metric_checkpoint, frame.to_dict("records"), key_columns=METRIC_KEY_COLUMNS, sort_columns=METRIC_KEY_COLUMNS)
            completed_metrics.add(key)
            metric_frames.append(frame)
            print(f"  [checkpoint] wrote metrics {key}", flush=True)

    metrics = pd.read_csv(metric_checkpoint, low_memory=False) if metric_checkpoint.exists() else pd.concat(metric_frames, ignore_index=True)
    meta_df = pd.read_csv(metadata_checkpoint, low_memory=False) if metadata_checkpoint.exists() else pd.DataFrame(metadata_rows)
    summary, pairwise = summarize_metrics(metrics, meta_df.to_dict("records"))
    atomic_write_csv(metrics, DATA_DIR / "patient_metrics.csv", index=False)
    atomic_write_csv(summary, DATA_DIR / "summary_metrics.csv", index=False)
    atomic_write_csv(pairwise, DATA_DIR / "pairwise_tests.csv", index=False)
    atomic_write_csv(meta_df, DATA_DIR / "exposure_feature_metadata.csv", index=False)
    write_html_table(
        summary,
        TABLE_DIR / "table1_pcawg_signature_summary.html",
        "Table 1. PCAWG reference-signature attribution by representation",
        "Cosine similarity and mean absolute error compare NNLS-fitted exposures against normalized PCAWG contribution labels. Feature dimension is the number of standard channels or UGA coordinates used for fitting.",
    )
    write_html_table(
        pairwise,
        TABLE_DIR / "table2_pcawg_signature_pairwise_tests.html",
        "Table 2. Paired PCAWG attribution contrasts: locked UGA versus standard",
        "Delta is locked UGA minus Standard. Positive delta favors UGA for cosine similarity; negative delta favors UGA for mean absolute error. P values use paired Wilcoxon signed-rank tests, and q values use Benjamini-Hochberg correction.",
    )
    metadata = {
        "experiment": EXPERIMENT_ROOT.name,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - t0,
        "modalities": MODALITIES,
        "sbsdbs_model": LOCKED_SBSDBS_MODEL,
        "id_model": LOCKED_ID_MODEL,
        "algorithm": "NNLS",
        "truth_source": "PCAWG SigProfiler contribution matrices",
        "cosmic_reference": "COSMIC v3.5 GRCh37",
        "metric_rows": int(len(metrics)),
        "summary_rows": int(len(summary)),
        "pairwise_rows": int(len(pairwise)),
    }
    atomic_write_json(DATA_DIR / "run_metadata.json", metadata)
    write_readme(summary, pairwise, metadata)
    print(json.dumps({"completed_in_seconds": round(metadata["elapsed_seconds"], 1), "metric_rows": len(metrics)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
