#!/usr/bin/env python3
"""Does the genomic representation predict survival beyond cancer type alone?

Requested by Jeya (1 Aug 2026 meeting):

    "If knowing the cancer type yields the same C-index as the genomic features, the genomic
     model isn't adding new predictive power for that specific task."

Mortality varies enormously by cancer type, and the benchmark has separately shown that the Bio
MAF v4 representation predicts cancer type well. So a survival model built on that representation
could be scoring highly by rediscovering cancer type rather than by capturing survival-relevant
biology. This script measures which.

    python scripts/run_survival_cancer_type_baseline.py                    # all arms, OS
    python scripts/run_survival_cancer_type_baseline.py --arms cancer_type # the cheap arm only
    python scripts/run_survival_cancer_type_baseline.py --endpoints OS PFI

Arms
----
    cancer_type              one-hot cancer type only            -> baseline C-index
    bio_maf                  Bio MAF v4 features only            -> the manuscript's claim
    cancer_type_plus_bio_maf both, concatenated                   -> incremental value

The third arm is the one that answers the question. If it does not beat `cancer_type`, then Bio
MAF v4 adds nothing to survival prediction once cancer type is known.

Fold pairing
------------
All arms use one fixed seed, so every arm sees identical outer folds and the C-indices are
directly paired. This differs from the main panel, where `stable_seed` mixes the representation
name into the seed and therefore gives each representation different folds. Paired folds are
required here because the whole question is a difference between arms, not their absolute values.
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
from utils.endpoint_registry import build_cdr_cancer_type_labels, build_cdr_survival_labels  # noqa: E402
from utils.nested_oof import evaluate_nested_oof  # noqa: E402

# One seed for every arm, so the outer folds are identical and the comparison is paired.
PAIRED_SEED = 20260801

BIO_MAF_CACHE = REPO_ROOT / "results/tables/quick_bio_v4_maf_features.csv.gz"

ARMS = ("cancer_type", "bio_maf", "cancer_type_plus_bio_maf")


def load_bio_maf_features() -> pd.DataFrame:
    if not BIO_MAF_CACHE.exists():
        raise SystemExit(
            f"Bio MAF v4 feature table not found at {BIO_MAF_CACHE}.\n"
            "Regenerate it with the main panel runner, or run with --arms cancer_type to get the\n"
            "baseline alone (which needs only the CDR table)."
        )
    frame = pd.read_csv(BIO_MAF_CACHE)
    index_col = "sample" if "sample" in frame.columns else frame.columns[0]
    frame[index_col] = frame[index_col].astype(str).str[:12]
    frame = frame.drop_duplicates(index_col).set_index(index_col)
    return frame.select_dtypes(include=[np.number]).astype(np.float32)


def one_hot_cancer_type(labels: pd.Series) -> pd.DataFrame:
    """One-hot encode cancer type, keeping every level.

    Cox models carry no intercept, so the full indicator set is the standard parameterisation and
    does not introduce the usual dummy-variable collinearity.
    """

    dummies = pd.get_dummies(labels.astype(str), prefix="cancer_type", dtype=np.float32)
    return dummies.reindex(sorted(dummies.columns), axis=1)


def concordance(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    from sksurv.metrics import concordance_index_censored

    valid = np.isfinite(risk)
    if valid.sum() < 2:
        return float("nan")
    return float(concordance_index_censored(event[valid].astype(bool), time[valid], risk[valid])[0])


def paired_bootstrap(
    time: np.ndarray,
    event: np.ndarray,
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 20260801,
) -> dict[str, float]:
    """Bootstrap the paired C-index difference (b - a) over patients."""

    rng = np.random.default_rng(seed)
    n = len(time)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if event[idx].sum() < 2:
            deltas[i] = np.nan
            continue
        deltas[i] = concordance(time[idx], event[idx], risk_b[idx]) - concordance(time[idx], event[idx], risk_a[idx])
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_two_sided": float("nan")}
    observed = concordance(time, event, risk_b) - concordance(time, event, risk_a)
    # Two-sided bootstrap p-value for H0: delta = 0.
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "delta": float(observed),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_two_sided": float(min(1.0, p)),
    }


def run_arm(
    arm: str,
    endpoint: str,
    features: pd.DataFrame,
    survival: pd.DataFrame,
    settings: dict,
    n_splits: int,
) -> tuple[dict, pd.DataFrame]:
    samples = features.index.astype(str)
    print(f"  [{arm}] n={len(samples)} features={features.shape[1]} events={int(survival['event'].sum())}", flush=True)
    result = evaluate_nested_oof(
        samples=samples,
        x=features.to_numpy(dtype=np.float32),
        labels=survival,
        task="survival",
        benchmark="tcga_cdr_survival",
        endpoint=endpoint,
        representation=arm,
        learner="cox_ph",
        settings=settings,
        n_splits=n_splits,
        seed=PAIRED_SEED,
    )
    row = dict(result.summary)
    row["arm"] = arm
    row["n_features"] = int(features.shape[1])
    return row, result.predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--config", default="config/experiment_settings.strict_no_leakage.yaml")
    parser.add_argument("--endpoints", nargs="+", default=["OS"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    paths = resolve_paths_map(load_yaml(args.paths))
    mc3_dir = Path(paths["raw_data"]["mc3_source_dir"])
    if not mc3_dir.exists():
        raise SystemExit(
            f"mc3_source_dir does not exist: {mc3_dir}\n"
            "Set CGR_DATASETS_DIR, or run scripts/fetch_datasets.py --minimal"
        )

    full = load_yaml(args.config)
    panel = ((full.get("experiments") or {}).get("main_manuscript_complete_panel") or {})
    settings = {key: value for key, value in {**full, **panel}.items() if not isinstance(value, dict)}

    # Always load the Bio MAF table, even when only the cancer_type arm was requested. The cohort
    # is defined as the intersection of all three inputs, so if the Bio MAF index were excluded
    # when that arm is skipped, the cancer_type arm would be scored on a larger cohort (10,986)
    # than the Bio MAF arms (9,986) and the C-indices would not be comparable.
    bio_maf = load_bio_maf_features()

    survival_labels, _audit = build_cdr_survival_labels(mc3_dir, args.endpoints)
    if not survival_labels:
        raise SystemExit(f"no survival labels built for {args.endpoints}")

    summary_rows: list[dict] = []
    comparison_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for endpoint, labels in survival_labels.items():
        print(f"\n=== {endpoint} ===", flush=True)
        cancer_type = build_cdr_cancer_type_labels(mc3_dir, endpoint_name="cancer_type_all")

        # Restrict to patients present in ALL THREE inputs regardless of which arms were
        # requested, so every arm is fitted and scored on exactly the same cohort as well as the
        # same folds. This also reproduces the main panel's OS cohort (n = 9,986), which makes the
        # baseline directly comparable to the published 0.647.
        common = labels.index.astype(str).intersection(cancer_type.index.astype(str))
        common = common.intersection(bio_maf.index.astype(str))
        common = pd.Index(sorted(set(map(str, common))))
        # ASCII only: the default Windows console codepage (cp1252) cannot encode characters like
        # the set-intersection sign, and an UnicodeEncodeError in a progress message is enough to
        # abort the whole stage.
        print(f"  cohort after intersection: n={len(common)} (survival AND cancer type AND Bio MAF v4)", flush=True)

        surv = labels.loc[common]
        ct_design = one_hot_cancer_type(cancer_type.loc[common])
        print(f"  cancer types present: {ct_design.shape[1]}", flush=True)

        designs = {}
        if "cancer_type" in args.arms:
            designs["cancer_type"] = ct_design
        if "bio_maf" in args.arms:
            designs["bio_maf"] = bio_maf.loc[common]
        if "cancer_type_plus_bio_maf" in args.arms:
            designs["cancer_type_plus_bio_maf"] = pd.concat([ct_design, bio_maf.loc[common]], axis=1)

        risks: dict[str, pd.Series] = {}
        for arm in args.arms:
            row, preds = run_arm(arm, endpoint, designs[arm], surv, settings, args.folds)
            summary_rows.append(row)
            preds = preds.copy()
            preds["arm"] = arm
            prediction_frames.append(preds)
            score_col = "risk_score" if "risk_score" in preds.columns else "pred_value"
            risks[arm] = preds.set_index("sample")[score_col].reindex(common)
            print(f"  [{arm}] C-index = {row.get('score'):.4f}", flush=True)

        time = surv["time"].to_numpy(dtype=float)
        event = surv["event"].to_numpy(dtype=int)
        for reference, candidate in (
            ("cancer_type", "bio_maf"),
            ("cancer_type", "cancer_type_plus_bio_maf"),
            ("bio_maf", "cancer_type_plus_bio_maf"),
        ):
            if reference not in risks or candidate not in risks:
                continue
            stats = paired_bootstrap(
                time, event,
                risks[reference].to_numpy(dtype=float),
                risks[candidate].to_numpy(dtype=float),
                n_boot=args.bootstrap,
                seed=PAIRED_SEED,
            )
            comparison_rows.append({"endpoint": endpoint, "reference": reference, "candidate": candidate, **stats})

    tables = REPO_ROOT / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables / "survival_cancer_type_baseline_summary.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            tables / "survival_cancer_type_baseline_predictions.csv.gz", index=False, compression="gzip"
        )
    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        comparisons.to_csv(tables / "survival_cancer_type_baseline_comparisons.csv", index=False)

    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    for endpoint in summary["endpoint"].unique():
        block = summary[summary["endpoint"] == endpoint]
        print(f"\n{endpoint}  (Harrell C-index, pooled out-of-fold, identical folds across arms)")
        print("-" * 74)
        for _, row in block.iterrows():
            print(f"  {row['arm']:26s} {float(row['score']):.4f}   ({int(row['n_features'])} features)")
        if not comparisons.empty:
            sub = comparisons[comparisons["endpoint"] == endpoint]
            print()
            for _, row in sub.iterrows():
                sig = "" if row["p_two_sided"] >= 0.05 else "  *"
                print(
                    f"  {row['candidate']} vs {row['reference']}: "
                    f"delta = {row['delta']:+.4f}  95% CI [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  "
                    f"p = {row['p_two_sided']:.3f}{sig}"
                )

    print("\nReading the result:")
    print("  cancer_type_plus_bio_maf vs cancer_type is the answer to Jeya's question.")
    print("  If that delta is not clearly positive, Bio MAF v4 adds nothing to survival")
    print("  prediction once cancer type is known, and the survival claim needs rewording.")
    print(f"\nwritten -> results/tables/survival_cancer_type_baseline_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
