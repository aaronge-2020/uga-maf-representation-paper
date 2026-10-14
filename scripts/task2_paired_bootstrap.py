#!/usr/bin/env python3
"""Task 2: paired bootstrap inference for the headline model comparisons.

Runs the one inferential procedure specified in
``results/tables/task2_notes.md``: paired tumour-resample bootstrap on saved
out-of-fold (OOF) predictions, percentile 95% intervals, two-sided bootstrap
p-values, and Benjamini-Hochberg correction over the computable contrasts.

Method (from task2_notes.md)
----------------------------
- ``--n_replicates`` replicates (spec: 10,000). Tumours are resampled with
  replacement in each replicate.
- Each contrast gets its own ``numpy.random.default_rng(--seed)`` stream
  (spec: seed 42 for every contrast).
- C1b (cancer typing, pooled-OOF balanced accuracy): BA(Sig+MAF) -
  BA(Bio MAF). Balanced accuracy = nan-mean of per-class recall over the
  20 classes; classes absent from a replicate are skipped. The same resample
  indices are used for both models (paired by tumour barcode).
- C4a (HRD42, Sig+MAF only, not a contrast): fold-mean AUROC over the 5
  archived outer folds. Per-fold AUROC on the resampled tumours in each
  fold; folds with a single class in a replicate are skipped. Score =
  P(class 1) from the nested-fold checkpoint.
- Percentile 95% interval: 2.5th and 97.5th percentiles of the replicate
  statistics.
- Two-sided bootstrap p-value: let d be the observed point difference and B
  the bootstrap differences. If d >= 0, p = 2 * fraction(B <= 0); if
  d < 0, p = 2 * fraction(B >= 0); capped at 1.0.
- BH correction over the raw p-values of exactly the computable contrasts.

Filename mapping (the notes predate a repo rename; no file with "archived"
in its name exists)
---------------------------------------------------------------
- notes ``task1_oof_a_sigmaf_full_archived.csv`` (Sig+MAF, pooled BA
  0.737784)  ->  ``task1_oof_sigmaf_a_full.csv`` (verified 0.737784).
- notes ``task1_oof_a_maf_full_archived.csv`` (archived Bio MAF OOF, pooled
  BA 0.661830)  ->  ``task1_oof_maf_a_full.csv`` (verified 0.661830).
  Used for C1b.
- notes ``task1_oof_a_maf_full.csv`` (Bio MAF Task-1 control (a) *refit*,
  pooled BA 0.665743)  ->  NOT FOUND in the repo. No file in
  ``results/tables/`` reproduces pooled BA 0.665743 (all
  ``task1_oof_maf_*.csv`` files were checked; the closest is
  ``task1_oof_maf_a_full.csv`` at 0.661830). Therefore C1a CANNOT be
  recomputed from available repo data and is written as blocked. No data
  is invented: the refit OOF simply is not in the repo.
- C4a checkpoint:
  ``..._nested_fold_checkpoints/hrd_binary_42__signatures_plus_MAF_stack__xgboost*.npz``
  (``pred`` (772,2), score = P(class 1), ``sample_folds`` 1-5). Sample
  order from ``data/task2/order_hrd_binary_42.csv``; true labels
  (622 negative / 150 positive) from
  ``results/tables/task3_hrd_attribution_oof_sig3_only.csv`` (barcode
  first-12 match; labels cross-checked against the other two task3
  attribution files). Per-fold AUROC recomputed from this OOF reproduces
  the archived ``main_manuscript_complete_panel_fold_metrics.csv`` values
  exactly (0.902400, 0.897733, 0.953763, 0.934409, 0.937634; fold-mean
  0.925188) - asserted in this script.

Blocked contrasts: C2, C3, C4b, C5, C6 (vs patterns alone) - no archived
patterns-alone OOF predictions exist in the repo (see task2_notes.md
"Blockers"). They are listed with blocked status and do not enter BH.

Expected results (spec table; this script must reproduce them exactly at
--n-replicates 10000 --seed 42)
---------------------------------------------------------------
C1b  +0.075954  [0.066996, 0.084998]  p = 0.0  q = 0.0
C4a   0.925188  [0.902291, 0.945974]  (not a contrast; no p-value)
C1a  blocked (refit Bio MAF OOF missing; expected +0.072041
     [0.063315, 0.080799] from the notes, not recomputable here)

Usage
-----
python scripts/task2_paired_bootstrap.py [--n-replicates 20 --output /tmp/smoke.csv]
All paths default to repo-relative locations.
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Archived HRD42 per-fold AUROC (Sig+MAF, XGBoost), from
# main_manuscript_complete_panel_fold_metrics.csv. Recomputed per-fold AUROC
# must match these exactly (1e-9) for the npz OOF to be accepted.
ARCHIVED_HRD42_FOLD_AUROC = [0.902400, 0.897733, 0.953763, 0.934409, 0.937634]

HEADER = ["contrast_id", "endpoint", "metric", "estimator", "model_A", "model_B",
          "n_tumours", "point_estimate", "ci_low_95", "ci_high_95", "raw_p",
          "q_bh", "status"]


def balanced_accuracy(true_codes, pred_codes, n_classes):
    """Nan-mean of per-class recall over classes present in the replicate."""
    recalls = []
    for c in range(n_classes):
        mask = true_codes == c
        n = int(mask.sum())
        if n > 0:
            recalls.append(float((pred_codes[mask] == c).sum()) / n)
    return float(np.mean(recalls)) if recalls else float("nan")


def percentile_ci(draws):
    lo = float(np.percentile(draws, 2.5))
    hi = float(np.percentile(draws, 97.5))
    return lo, hi


def two_sided_bootstrap_p(d_obs, draws):
    draws = np.asarray(draws)
    if d_obs >= 0:
        p = 2.0 * float((draws <= 0).mean())
    else:
        p = 2.0 * float((draws >= 0).mean())
    return min(p, 1.0)


def benjamini_hochberg(pvals):
    """BH q-values in the order of the input p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]  # monotonicity
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(m)
    out[order] = q
    return out


def fmt(x):
    return repr(round(float(x), 6))


def load_task1_oof(path):
    df = pd.read_csv(path, usecols=["sample", "true_label", "pred_label"])
    return df.set_index("sample")


def run_c1b(tables_dir, n_replicates, seed):
    """C1b: paired bootstrap of pooled-OOF balanced accuracy difference.

    Sig+MAF (task1_oof_sigmaf_a_full.csv) vs archived Bio MAF OOF
    (task1_oof_maf_a_full.csv), paired by tumour barcode.
    """
    sig = load_task1_oof(os.path.join(tables_dir, "task1_oof_sigmaf_a_full.csv"))
    maf = load_task1_oof(os.path.join(tables_dir, "task1_oof_maf_a_full.csv"))
    assert (sig.index == maf.index).all(), "sample order mismatch between OOF files"
    assert (sig["true_label"] == maf["true_label"]).all(), "true-label mismatch"
    n = len(sig)

    classes = sorted(sig["true_label"].unique())
    assert len(classes) == 20, f"expected 20 classes, got {len(classes)}"
    lmap = {label: i for i, label in enumerate(classes)}
    true_c = sig["true_label"].map(lmap).to_numpy()
    pred_sig = sig["pred_label"].map(lmap).to_numpy()
    pred_maf = maf["pred_label"].map(lmap).to_numpy()

    d_obs = balanced_accuracy(true_c, pred_sig, 20) - balanced_accuracy(true_c, pred_maf, 20)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_replicates)
    undef = 0
    for b in range(n_replicates):
        idx = rng.integers(0, n, n)
        t = true_c[idx]
        ba_a = balanced_accuracy(t, pred_sig[idx], 20)
        ba_b = balanced_accuracy(t, pred_maf[idx], 20)
        if np.isnan(ba_a) or np.isnan(ba_b):
            undef += 1
        diffs[b] = ba_a - ba_b
    lo, hi = percentile_ci(diffs)
    p = two_sided_bootstrap_p(d_obs, diffs)
    print(f"C1b: n={n} d_obs={d_obs:.6f} CI=[{lo:.6f},{hi:.6f}] p={p:.6g} "
          f"(undefined-BA replicates: {undef})", flush=True)
    return n, d_obs, lo, hi, p


def foldmean_auroc(yb, pb, fb):
    aucs = []
    for k in np.unique(fb):
        mask = fb == k
        yt = yb[mask]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt.astype(int), pb[mask]))
    return float(np.mean(aucs)) if aucs else float("nan")


def run_c4a(tables_dir, hrd_order_path, n_replicates, seed):
    """C4a: bootstrap interval for HRD42 Sig+MAF fold-mean AUROC.

    Uses the nested-fold checkpoint npz (pred (772,2); score = P(class 1))
    and true labels from the task3 HRD attribution OOF table.
    """
    ckpt_dir = os.path.join(tables_dir,
                            "main_manuscript_complete_panel_nested_fold_checkpoints")
    matches = glob.glob(os.path.join(
        ckpt_dir, "hrd_binary_42__signatures_plus_MAF_stack__xgboost*.npz"))
    assert len(matches) == 1, f"expected 1 HRD42 Sig+MAF xgboost npz, got {matches}"
    z = np.load(matches[0])
    pred = z["pred"][:, 1].astype(float)
    folds = z["sample_folds"].astype(int)
    n = len(pred)

    order = pd.read_csv(hrd_order_path)["patient_id_12"].astype(str).tolist()
    assert len(order) == n, f"order length {len(order)} != npz length {n}"

    lab_path = os.path.join(tables_dir, "task3_hrd_attribution_oof_sig3_only.csv")
    d42 = pd.read_csv(lab_path)
    d42 = d42[d42["endpoint"] == "hrd_binary_42"]
    by_barcode = dict(zip(d42["tumour_barcode"].str[:12].astype(str), d42["true"].astype(float)))
    y = np.array([by_barcode[s] for s in order], dtype=float)
    assert not np.isnan(y).any(), "missing true labels for ordered samples"
    assert int((y == 0).sum()) == 622 and int((y == 1).sum()) == 150, \
        f"class balance != 622/150: {(y == 0).sum()}/{(y == 1).sum()}"
    print(f"C4a: n={n} class balance neg=622 pos=150", flush=True)

    # Validation: recomputed per-fold AUROC must match archived fold metrics.
    arch = {}
    with open(os.path.join(tables_dir, "main_manuscript_complete_panel_fold_metrics.csv")) as f:
        for r in csv.DictReader(f):
            if (r["endpoint"] == "hrd_binary_42"
                    and r["representation"] == "signatures_plus_MAF_stack"
                    and r["learner"] == "xgboost" and r["metric"] == "auroc"):
                arch[int(r["fold"])] = float(r["score"])
    assert [arch[k] for k in sorted(arch)] == ARCHIVED_HRD42_FOLD_AUROC or \
        all(abs(arch[k] - a) < 1e-6 for k, a in zip(sorted(arch), ARCHIVED_HRD42_FOLD_AUROC)), \
        "archived fold metrics do not match expected values"
    per_fold = [roc_auc_score(y[folds == k].astype(int), pred[folds == k])
                for k in sorted(set(folds.tolist()))]
    assert all(abs(a - b) < 1e-9 for a, b in
               zip(per_fold, [arch[k] for k in sorted(arch)])), \
        "npz OOF does not reproduce archived per-fold AUROC"
    point = float(np.mean(per_fold))
    print(f"C4a: per-fold AUROC={[round(float(v), 6) for v in per_fold]} "
          f"fold-mean={point:.6f}", flush=True)

    rng = np.random.default_rng(seed)
    fm_boots = np.empty(n_replicates)
    undef = 0
    for b in range(n_replicates):
        idx = rng.integers(0, n, n)
        v = foldmean_auroc(y[idx], pred[idx], folds[idx])
        if np.isnan(v):
            undef += 1
        fm_boots[b] = v
    lo, hi = percentile_ci(fm_boots)
    print(f"C4a: CI=[{lo:.6f},{hi:.6f}] (undefined replicates: {undef})", flush=True)
    return n, point, lo, hi


def blocked_row(contrast_id, endpoint, metric, estimator, model_a, model_b,
                n_tumours, status):
    return [contrast_id, endpoint, metric, estimator, model_a, model_b,
            n_tumours, "", "", "", "", "", status]


def main():
    ap = argparse.ArgumentParser(
        description="Task 2: paired bootstrap inference on saved OOF predictions "
                    "(computable contrasts C1b and C4a; blocked contrasts listed "
                    "with status). See module docstring for the filename mapping.")
    ap.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "results", "tables"),
                    help="Directory holding the OOF tables and checkpoints "
                         "(default: repo results/tables).")
    ap.add_argument("--hrd-order", default=os.path.join(REPO_ROOT, "data", "task2",
                                                        "order_hrd_binary_42.csv"),
                    help="Sample order for the HRD42 npz (default: repo "
                         "data/task2/order_hrd_binary_42.csv).")
    ap.add_argument("--output", default=os.path.join(REPO_ROOT, "results", "tables",
                                                     "task2_inference_results.csv"),
                    help="Output CSV path (default: repo "
                         "results/tables/task2_inference_results.csv).")
    ap.add_argument("--n-replicates", type=int, default=10000,
                    help="Bootstrap replicates per contrast (default 10000).")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed; each contrast gets its own default_rng(seed) "
                         "stream (default 42).")
    args = ap.parse_args()

    tables = os.path.abspath(args.data_dir)
    assert os.path.isdir(tables), f"data dir not found: {tables}"
    assert os.path.isfile(os.path.abspath(args.hrd_order)), \
        f"HRD order file not found: {args.hrd_order}"

    sig_maf = "signatures_plus_MAF_stack (XGBoost)"

    # ---- C1b: computable ----
    n_c1b, d_c1b, lo_c1b, hi_c1b, p_c1b = run_c1b(
        tables, args.n_replicates, args.seed)

    # ---- C4a: computable (not a contrast; no p-value) ----
    n_c4a, pt_c4a, lo_c4a, hi_c4a = run_c4a(
        tables, os.path.abspath(args.hrd_order), args.n_replicates, args.seed)

    # ---- BH over raw p-values of exactly the computable contrasts ----
    # C1a is blocked (its refit Bio MAF OOF is missing from the repo), so the
    # only computable p-value is C1b's.
    computable_p = [p_c1b]
    q = benjamini_hochberg(computable_p)
    q_c1b = q[0]

    rows = [
        # C1a: blocked - the refit Bio MAF OOF (pooled BA 0.665743) is not in
        # the repo; the only available MAF OOF reproduces the archived value
        # (0.661830). Do not invent data.
        blocked_row(
            "C1a", "cancer_type_top20", "balanced_accuracy", "pooled OOF",
            sig_maf, "MAF_stack_only (XGBoost) Task-1 control (a) refit", 8800,
            "blocked: Bio MAF Task-1 control (a) refit OOF (notes pooled BA "
            "0.665743) not present in repo; the available MAF OOF file "
            "reproduces the archived Bio MAF value 0.661830"),
        ["C1b", "cancer_type_top20", "balanced_accuracy", "pooled OOF",
         sig_maf, "MAF_stack_only (XGBoost) archived OOF", n_c1b,
         fmt(d_c1b), fmt(lo_c1b), fmt(hi_c1b), fmt(p_c1b), fmt(q_c1b),
         "computable"],
        blocked_row(
            "C2", "cancer_type_top20", "balanced_accuracy", "pooled OOF",
            sig_maf, "standard_sbs96_id83 (XGBoost) patterns alone", 8800,
            "blocked: archived patterns-alone OOF unavailable"),
        blocked_row(
            "C3", "HRD_Score", "spearman", "pooled OOF",
            sig_maf, "standard_sbs96_id83 (XGBoost) patterns alone", 772,
            "blocked: archived patterns-alone HRD OOF unavailable"),
        ["C4a", "hrd_binary_42", "auroc", "fold-mean over 5 outer folds",
         sig_maf + " single model", "", n_c4a,
         fmt(pt_c4a), fmt(lo_c4a), fmt(hi_c4a), "", "",
         "computable (not a contrast; no p-value)"],
        blocked_row(
            "C4b", "hrd_binary_42", "auroc", "fold-mean over 5 outer folds",
            sig_maf, "standard_sbs96_id83 (XGBoost) patterns alone", 772,
            "blocked: archived patterns-alone HRD OOF unavailable"),
        blocked_row(
            "C5", "hrd_binary_33", "auroc", "fold-mean over 5 outer folds",
            sig_maf, "standard_sbs96_id83 (XGBoost) patterns alone", 772,
            "blocked: archived patterns-alone HRD OOF unavailable"),
        blocked_row(
            "C6", "hrd_binary_24", "auroc", "fold-mean over 5 outer folds",
            sig_maf, "standard_sbs96_id83 (XGBoost) patterns alone", 772,
            "blocked: archived patterns-alone HRD OOF unavailable "
            "(HRD24 was run in the archived analysis)"),
    ]

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
