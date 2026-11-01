#!/usr/bin/env python
"""A8 primary contrasts: five combined representations per task and learner,
protocol primary estimator (mean within-fold), paired per-fold differences
against spectra+maf_stack, and patient-level paired bootstrap 95% CIs for
the pooled difference. Seed fixed for reproducibility."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score

PRED = "/home/hatch/workspace/projects/uga-maf-representation-paper/analyses/representation_extension/predictions"
REPS = ["spectra+maf_stack", "spectra+r3_counts_panel", "spectra+r2_binary_panel",
        "spectra+r4_consequence_panel", "spectra+r5_tfidf_panel"]
REF = "spectra+maf_stack"
NBOOT = 2000
rng = np.random.default_rng(20261025)

def metric(task, y, p):
    if task == "cancer":
        return balanced_accuracy_score(y, p)
    return spearmanr(y, p).statistic

def load(task, rep, learner):
    df = pd.read_csv(f"{PRED}/{task}__{rep}__{learner}.csv", float_precision="round_trip")
    return df.sort_values("patient").reset_index(drop=True)

out = {}
for task in ["cancer", "hrd"]:
    for learner in ["xgboost", "elasticnet"]:
        frames = {rep: load(task, rep, learner) for rep in REPS}
        base = frames[REF]
        for rep in REPS:
            f = frames[rep]
            assert (f["patient"].values == base["patient"].values).all(), (task, rep, learner)
            assert (f["fold"].values == base["fold"].values).all(), (task, rep, learner)
            assert np.allclose(f["true_value"].values.astype(float), base["true_value"].values.astype(float)), (task, rep, learner)
        y = base["true_value"].values
        y = y.astype(float) if task == "hrd" else y
        fold_scores, primary = {}, {}
        for rep in REPS:
            fs = {}
            for k in sorted(base["fold"].unique()):
                m = base["fold"].values == k
                fs[int(k)] = metric(task, y[m], frames[rep]["prediction"].values[m])
            fold_scores[rep] = fs
            primary[rep] = float(np.mean(list(fs.values())))
        entry = {"primary": primary, "fold_scores": fold_scores, "contrasts": {}}
        ref_pred = base  # reference predictions are frames[REF]
        ref_p = frames[REF]["prediction"].values
        n = len(y)
        for rep in REPS:
            if rep == REF:
                continue
            p = frames[rep]["prediction"].values
            per_fold_diff = float(np.mean([fold_scores[rep][k] - fold_scores[REF][k] for k in fold_scores[rep]]))
            diffs = np.empty(NBOOT)
            for b in range(NBOOT):
                idx = rng.integers(0, n, n)
                diffs[b] = metric(task, y[idx], p[idx]) - metric(task, y[idx], ref_p[idx])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            entry["contrasts"][rep] = {
                "mean_per_fold_diff": per_fold_diff,
                "bootstrap_diff_median": float(np.median(diffs)),
                "bootstrap_ci95": [float(lo), float(hi)],
            }
        out[f"{task}/{learner}"] = entry
        print(f"== {task}/{learner}")
        for rep in REPS:
            print(f"  {rep:32s} {primary[rep]:.4f}")
        for rep, c in entry["contrasts"].items():
            print(f"  diff {rep:28s} per-fold {c['mean_per_fold_diff']:+.4f}  bootCI [{c['bootstrap_ci95'][0]:+.4f}, {c['bootstrap_ci95'][1]:+.4f}]")

with open("/home/hatch/workspace/scratch/representation-extension/a8_results.json", "w") as fh:
    json.dump(out, fh, indent=1)
print("WROTE a8_results.json")
