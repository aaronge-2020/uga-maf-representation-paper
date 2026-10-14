#!/usr/bin/env python3
"""Task 5 (PCAWG audit): per-type recall with bootstrap 95% CIs for the two
frozen cancer-type models on the PCAWG 913-tumour cohort.

Models (frozen, no retraining):
  - Sig+MAF: results/frozen_models/cancer_type_top20__signatures_plus_MAF_stack
    predictions: external_validation/pcawg_transfer/predictions/pcawg_sigmaf_predictions.csv
  - patterns-only (standard_sbs96_id83):
    results/frozen_models/cancer_type_top20__standard_sbs96_id83
    predictions: results/external_validation/aaron_email/pcawg/pcawg_frozen_model_predictions.csv

Bootstrap: 10,000 replicates, fixed seed 42, resampling tumours within each
true cancer type with replacement. The per-type gain (Sig+MAF minus
patterns-only) is computed on the SAME resample (paired), so its CI reflects
the within-type paired difference.

Output: results/tables/task5_pcawg_audit_per_type.csv
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SIGMAF_PRED = REPO / "external_validation/pcawg_transfer/predictions/pcawg_sigmaf_predictions.csv"
SPECTRA_PRED = REPO / "results/external_validation/aaron_email/pcawg/pcawg_frozen_model_predictions.csv"
OUT = REPO / "results/tables/task5_pcawg_audit_per_type.csv"

SEED = 42
N_BOOT = 10_000

sig = pd.read_csv(SIGMAF_PRED).set_index("sample_id")
spec = pd.read_csv(SPECTRA_PRED).set_index("sample_id")

assert (sig.index == spec.index).all() or set(sig.index) == set(spec.index), "sample id mismatch"
spec = spec.loc[sig.index]
assert (sig["true_label"].values == spec["true_label"].values).all(), "true_label mismatch"
y = sig["true_label"].values
pred_sig = sig["pred_label"].values
pred_spec = spec["pred_label"].values

rng = np.random.default_rng(SEED)
rows = []
for cls in sorted(set(y)):
    idx = np.flatnonzero(y == cls)
    n = len(idx)
    hit_sig = (pred_sig[idx] == cls).astype(float)
    hit_spec = (pred_spec[idx] == cls).astype(float)
    recall_sig = float(hit_sig.mean())
    recall_spec = float(hit_spec.mean())
    gain = recall_sig - recall_spec
    b_sig, b_spec, b_gain = [], [], []
    for _ in range(N_BOOT):
        r = rng.integers(0, n, n)
        rs = hit_sig[r].mean()
        rp = hit_spec[r].mean()
        b_sig.append(rs)
        b_spec.append(rp)
        b_gain.append(rs - rp)
    rows.append({
        "cancer_type": cls,
        "n": n,
        "recall_sigmaf": recall_sig,
        "recall_sigmaf_ci_low": float(np.quantile(b_sig, 0.025)),
        "recall_sigmaf_ci_high": float(np.quantile(b_sig, 0.975)),
        "recall_patterns_only": recall_spec,
        "recall_patterns_only_ci_low": float(np.quantile(b_spec, 0.025)),
        "recall_patterns_only_ci_high": float(np.quantile(b_spec, 0.975)),
        "gain_sigmaf_minus_patterns": gain,
        "gain_ci_low": float(np.quantile(b_gain, 0.025)),
        "gain_ci_high": float(np.quantile(b_gain, 0.975)),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"wrote {OUT} ({len(df)} rows)")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"total n = {df['n'].sum()}")
