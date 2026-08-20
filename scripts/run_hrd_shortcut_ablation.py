#!/usr/bin/env python3
"""Does Bio MAF v4 predict HRD, or is it exploiting obvious shortcuts?

A reviewer can reasonably ask whether the HRD result reflects learned mutation biology or two
much simpler signals:

  1. Direct hits in HRD / homologous-recombination-repair genes. If the model is largely reading
     "BRCA1 or BRCA2 is mutated", it is recovering a known clinical rule rather than a
     representation-level finding.
  2. Variant-allele-fraction features, which can proxy tumour purity, clonality or copy-number
     state -- and HRD scores are themselves derived from copy-number data, so this is a route to
     partial circularity rather than biology.

This runs the same nested out-of-fold HRD benchmark three ways and reports the cost of removing
each shortcut:

    full          signatures + all Bio MAF v4 features
    no_hrd_genes  minus every per-gene column for the 16 HRD/HRR genes  (32 columns)
    no_vaf        minus every VAF-derived column                        (14 columns)

    python scripts/run_hrd_shortcut_ablation.py
    python scripts/run_hrd_shortcut_ablation.py --endpoints HRD_Score hrd_binary_33
    python scripts/run_hrd_shortcut_ablation.py --arms full no_hrd_genes

Fold pairing
------------
Every arm uses one fixed seed, so all arms see identical outer folds and the deltas are paired.
This differs from the main panel, where the seed is derived from the representation name and each
representation therefore gets different folds. Paired folds are required here because the entire
question is a difference between arms.

CPU only -- safe to run alongside a GPU job.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.config import load_yaml, resolve_paths_map  # noqa: E402
from utils.nested_oof import evaluate_nested_oof  # noqa: E402

PAIRED_SEED = 20260820

BIO_MAF_CACHE = REPO_ROOT / "results/tables/quick_bio_v4_maf_features.csv.gz"
SIGNATURE_REL = "datasets/tcga_mc3/features/features_standard_sbs96_id83.csv.gz"
COHORT_REL = "datasets/tcga_brca_hrd/cohort/final_analysis_cohort.tsv"

# The 16 HRD/HRR genes flagged in the project's own driver-gene CSV. Per-gene Bio MAF columns for
# these are removed in the no_hrd_genes arm; all global features (burden, spectra, consequence and
# driver-tier counts, hotspots) are retained.
HRD_GENES = [
    "ABL1", "ATM", "ATR", "BARD1", "BRCA1", "BRCA2", "MDC1", "NSD2",
    "PALB2", "PARP1", "POLD1", "POLE", "PPP4R2", "RAD50", "RAD51C", "RAD52",
]

CONTINUOUS = {"HRD_Score", "HRD_TAI", "HRD_LST", "HRD_LOH", "PARPi7", "eCARD"}
BINARY = {"hrd_binary_24", "hrd_binary_33", "hrd_binary_42"}

ARMS = ("full", "no_hrd_genes", "no_vaf")


def load_features(datasets_dir: Path) -> pd.DataFrame:
    """Signatures + Bio MAF v4, indexed by 12-character patient barcode."""
    bio = pd.read_csv(BIO_MAF_CACHE)
    index_col = "sample" if "sample" in bio.columns else bio.columns[0]
    bio[index_col] = bio[index_col].astype(str).str[:12]
    bio = bio.drop_duplicates(index_col).set_index(index_col)
    bio = bio.select_dtypes(include=[np.number]).astype(np.float32)

    sig = pd.read_csv(datasets_dir / SIGNATURE_REL)
    sig_index = sig.columns[0]
    sig[sig_index] = sig[sig_index].astype(str).str[:12]
    sig = sig.drop_duplicates(sig_index).set_index(sig_index)
    sig = sig.select_dtypes(include=[np.number]).astype(np.float32)
    # Disambiguate any column name shared between the two blocks.
    sig = sig.rename(columns={c: f"sig__{c}" for c in sig.columns if c in set(bio.columns)})

    common = bio.index.intersection(sig.index)
    return pd.concat([sig.loc[common], bio.loc[common]], axis=1)


def load_labels(datasets_dir: Path, endpoint: str) -> tuple[pd.Series, str]:
    """Reproduce the main runner's HRD label construction exactly."""
    cohort = pd.read_csv(datasets_dir / COHORT_REL, sep="\t")
    if endpoint in CONTINUOUS:
        data = cohort.dropna(subset=[endpoint]).copy()
        labels = pd.Series(
            data[endpoint].astype(float).to_numpy(),
            index=data["patient_id_12"].astype(str),
            name=endpoint,
        )
        return labels, "regression"
    if endpoint in BINARY:
        # The cohort stores these as the strings "HRD-high" / "HRD-low", not 0/1.
        data = cohort[cohort[endpoint].isin(["HRD-high", "HRD-low"])].copy()
        labels = pd.Series(
            (data[endpoint] == "HRD-high").astype(int).to_numpy(),
            index=data["patient_id_12"].astype(str),
            name=endpoint,
        )
        return labels, "binary"
    raise SystemExit(f"endpoint '{endpoint}' is not an HRD endpoint")


def shortcut_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    gene_cols = [c for c in columns if any(c.endswith("__" + g) for g in HRD_GENES)]
    vaf_cols = [c for c in columns if "vaf" in c.lower()]
    return gene_cols, vaf_cols


def paired_bootstrap(
    task: str,
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = PAIRED_SEED,
) -> dict[str, float]:
    """Bootstrap the paired metric difference (b - a) over samples."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    def metric(idx: np.ndarray, pred: np.ndarray) -> float:
        yy, pp = y[idx], pred[idx]
        if task == "regression":
            if len(np.unique(yy)) < 2 or len(np.unique(pp)) < 2:
                return float("nan")
            return float(spearmanr(yy, pp).correlation)
        if len(np.unique(yy)) < 2:
            return float("nan")
        return float(roc_auc_score(yy, pp))

    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = metric(idx, pred_b) - metric(idx, pred_a)
    deltas = deltas[np.isfinite(deltas)]
    allidx = np.arange(n)
    observed = metric(allidx, pred_b) - metric(allidx, pred_a)
    if deltas.size == 0:
        return {"delta": float(observed), "ci_low": float("nan"), "ci_high": float("nan"), "p_two_sided": float("nan")}
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "delta": float(observed),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_two_sided": float(min(1.0, p)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--config", default="config/experiment_settings.strict_no_leakage.yaml")
    parser.add_argument("--endpoints", nargs="+",
                        default=["HRD_Score", "hrd_binary_24", "hrd_binary_33", "hrd_binary_42"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--learner", default="xgboost", choices=["xgboost", "linear"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    paths = resolve_paths_map(load_yaml(args.paths))
    datasets_dir = Path(paths["raw_data"]["mc3_source_dir"]).parents[1]
    if not (datasets_dir / SIGNATURE_REL).exists():
        raise SystemExit(f"signature features not found: {datasets_dir / SIGNATURE_REL}")

    full = load_yaml(args.config)
    panel = ((full.get("experiments") or {}).get("main_manuscript_complete_panel") or {})
    settings = {k: v for k, v in {**full, **panel}.items() if not isinstance(v, dict)}

    features = load_features(datasets_dir)
    gene_cols, vaf_cols = shortcut_columns(list(features.columns))
    print(f"features: {features.shape[1]} columns, {features.shape[0]} patients")
    print(f"  HRD/HRR per-gene columns : {len(gene_cols)}")
    print(f"  VAF-derived columns      : {len(vaf_cols)}")

    designs = {
        "full": list(features.columns),
        "no_hrd_genes": [c for c in features.columns if c not in set(gene_cols)],
        "no_vaf": [c for c in features.columns if c not in set(vaf_cols)],
    }

    summary_rows: list[dict] = []
    comparison_rows: list[dict] = []

    for endpoint in args.endpoints:
        labels, task = load_labels(datasets_dir, endpoint)
        common = labels.index.intersection(features.index)
        labels = labels.loc[common]
        print(f"\n=== {endpoint} ({task}) n={len(common)} ===", flush=True)

        preds: dict[str, np.ndarray] = {}
        for arm in args.arms:
            cols = designs[arm]
            x = features.loc[common, cols]
            result = evaluate_nested_oof(
                samples=common.astype(str),
                x=x.to_numpy(dtype=np.float32),
                labels=labels,
                task=task,
                benchmark="mc3_main",
                endpoint=endpoint,
                representation=f"signatures_plus_MAF_stack__{arm}",
                learner=args.learner,
                settings=settings,
                n_splits=args.folds,
                seed=PAIRED_SEED,
            )
            row = dict(result.summary)
            row["arm"] = arm
            row["n_features"] = len(cols)
            summary_rows.append(row)
            frame = result.predictions.set_index("sample")
            col = "pred_value" if task == "regression" else "pred_class_1"
            preds[arm] = frame[col].reindex(common.astype(str)).to_numpy(dtype=float)
            print(f"  {arm:14s} {row['metric']:9s} {float(row['score']):.4f}  ({len(cols)} features)", flush=True)

        y = labels.to_numpy(dtype=float)
        for arm in args.arms:
            if arm == "full" or "full" not in preds:
                continue
            stats = paired_bootstrap(task, y, preds["full"], preds[arm], n_boot=args.bootstrap)
            comparison_rows.append({"endpoint": endpoint, "task": task, "reference": "full", "candidate": arm, **stats})
            print(f"    {arm} vs full: {stats['delta']:+.4f}  95% CI [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]  p={stats['p_two_sided']:.3f}", flush=True)

    tables = REPO_ROOT / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(tables / "hrd_shortcut_ablation_summary.csv", index=False)
    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        comparisons.to_csv(tables / "hrd_shortcut_ablation_comparisons.csv", index=False)

    print("\n" + "=" * 78)
    print("HRD SHORTCUT ABLATION")
    print("=" * 78)
    frame = pd.DataFrame(summary_rows)
    for endpoint in frame["endpoint"].unique():
        block = frame[frame["endpoint"] == endpoint]
        print(f"\n{endpoint}  ({block['metric'].iloc[0]})")
        print("-" * 62)
        for _, row in block.iterrows():
            print(f"  {row['arm']:14s} {float(row['score']):.4f}   ({int(row['n_features'])} features)")
        if not comparisons.empty:
            for _, row in comparisons[comparisons["endpoint"] == endpoint].iterrows():
                flag = "" if row["p_two_sided"] >= 0.05 else "  *"
                print(f"    {row['candidate']} vs full: {row['delta']:+.4f}  [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  p={row['p_two_sided']:.3f}{flag}")

    print("\nReading the result:")
    print("  Deltas near zero mean the representation does NOT depend on that shortcut, which is")
    print("  the defensible outcome. Large negative deltas mean the HRD result was substantially")
    print("  carried by direct HRD-gene hits or by VAF-derived purity/clonality proxies.")
    print("\nwritten -> results/tables/hrd_shortcut_ablation_{summary,comparisons}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
