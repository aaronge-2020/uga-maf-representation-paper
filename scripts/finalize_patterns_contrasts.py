#!/usr/bin/env python3
"""Compute C2/C3/C4b/C5 and update task2_inference_results.csv."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "results" / "tables"

N_REP = 10000
SEED = 42

def balanced_accuracy(true, pred, n_classes):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true, pred, labels=list(range(n_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(cm) / cm.sum(axis=1)
    per_class = per_class[~np.isnan(per_class)]
    return float(np.mean(per_class)) if len(per_class) > 0 else float('nan')

def percentile_ci(diffs, alpha=0.05):
    return float(np.percentile(diffs, 100*alpha/2)), float(np.percentile(diffs, 100*(1-alpha/2)))

def two_sided_p(d_obs, draws):
    draws = np.asarray(draws)
    if d_obs >= 0:
        p = 2.0 * float((draws <= 0).mean())
    else:
        p = 2.0 * float((draws >= 0).mean())
    return min(p, 1.0)

def benjamini_hochberg(pvals):
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = np.empty(n)
    prev = 1.0
    for i in range(n-1, -1, -1):
        q[order[i]] = min(prev, ranked[i] * n / (i+1))
        prev = q[order[i]]
    return q

def main():
    # Load data
    sigmaf = pd.read_csv(TABLES / "task1_oof_sigmaf_a_full.csv")
    pa_ct = pd.read_csv(TABLES / "patterns_alone_oof_cancer_type.csv", index_col="sample")
    
    # C2: cancer typing
    print("C2...")
    classes = sorted(sigmaf["true_label"].unique())
    lmap = {l: i for i, l in enumerate(classes)}
    true_c = sigmaf["true_label"].map(lmap).to_numpy()
    pred_sig = sigmaf["pred_label"].map(lmap).to_numpy()
    proba_cols = [c for c in pa_ct.columns if c.startswith("proba_")]
    pa_classes = [c.replace("proba_", "") for c in proba_cols]
    pa_pred = pd.Series([pa_classes[i] for i in pa_ct[proba_cols].values.argmax(axis=1)], index=pa_ct.index)
    pa_aligned = pa_pred.loc[sigmaf.set_index("sample").index].map(lmap).to_numpy()
    
    d_obs = balanced_accuracy(true_c, pred_sig, 20) - balanced_accuracy(true_c, pa_aligned, 20)
    rng = np.random.default_rng(SEED)
    n = len(true_c)
    diffs = np.array([balanced_accuracy(true_c[idx := rng.integers(0, n, n)], pred_sig[idx], 20) - 
                      balanced_accuracy(true_c[idx], pa_aligned[idx], 20) for _ in range(N_REP)])
    lo, hi = percentile_ci(diffs)
    p = two_sided_p(d_obs, diffs)
    c2 = (n, d_obs, lo, hi, p)
    print(f"  C2: {d_obs:.6f} [{lo:.6f},{hi:.6f}] p={p:.6g}")
    
    # For C3/C4b/C5, need Sig+MAF HRD OOF - use the existing task2 approach
    # For now, compute from the re-run files with paired samples
    
    print("\nDone")
    # Save results for manual CSV update
    import json
    out = {
        "C2": {"n": c2[0], "d": c2[1], "lo": c2[2], "hi": c2[3], "p": c2[4]},
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
