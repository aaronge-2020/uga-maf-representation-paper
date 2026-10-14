#!/usr/bin/env python3
"""Task 3 (conditional POG570 item): compute AUROC, calibration slope and
intercept, and bootstrap 95% intervals from the saved POG570 external-validation
predictions (frozen HRD_Score__signatures_plus_MAF_stack model).

Inputs:  results/external_validation/aaron_email/pog570/pog570_frozen_model_predictions.csv
Outputs: appended rows in results/tables/task3_hrd_attribution_baselines.csv
         (analysis = pog570_external) and detail in task3_notes.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

N_BOOT = 2000
SEED = 20261013

DEFAULT_INPUT = (REPO_ROOT / "results/external_validation/aaron_email/pog570"
                 / "pog570_frozen_model_predictions.csv")
DEFAULT_OUTPUT = REPO_ROOT / "results/tables/task3_pog570_external_metrics.csv"


def parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Task 3: POG570 external-validation metrics from saved predictions")
    p.add_argument("--input", dest="input", type=Path, default=DEFAULT_INPUT,
                   help=f"POG570 frozen-model predictions CSV (default: {DEFAULT_INPUT})")
    p.add_argument("--output", dest="output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"metrics CSV to write (default: {DEFAULT_OUTPUT})")
    return p.parse_args(argv)


def bootstrap_ci(stat_fn, y, p, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(stat_fn(y[idx], p[idx]))
    vals = np.array([v for v in vals if np.isfinite(v)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    frame = frame.dropna(subset=["HRD_SCORE", "predicted_HRD_SCORE"]).copy()
    y_true = frame["HRD_SCORE"].to_numpy(dtype=float)
    y_pred = frame["predicted_HRD_SCORE"].to_numpy(dtype=float)

    rows: list[dict] = []
    for cohort_label, mask in (
        ("primary_breast", frame["analysis_group"].eq("primary_breast").to_numpy()),
        ("all_pog570", np.ones(len(frame), dtype=bool)),
    ):
        yt, yp = y_true[mask], y_pred[mask]
        n = len(yt)
        # Continuous calibration: regress true on predicted.
        lin = LinearRegression().fit(yp.reshape(-1, 1), yt)
        slope, intercept = float(lin.coef_[0]), float(lin.intercept_)
        sp = float(spearmanr(yt, yp).correlation)
        sp_lo, sp_hi = bootstrap_ci(lambda a, b: float(spearmanr(a, b).correlation), yt, yp)
        sl_lo, sl_hi = bootstrap_ci(lambda a, b: float(LinearRegression().fit(b.reshape(-1, 1), a).coef_[0]), yt, yp)
        in_lo, in_hi = bootstrap_ci(lambda a, b: float(LinearRegression().fit(b.reshape(-1, 1), a).intercept_), yt, yp)
        rows.append(dict(cohort=cohort_label, endpoint="HRD_Score", metric="spearman", score=sp, ci_low=sp_lo, ci_high=sp_hi, n=n))
        rows.append(dict(cohort=cohort_label, endpoint="HRD_Score", metric="calibration_slope_linear", score=slope, ci_low=sl_lo, ci_high=sl_hi, n=n))
        rows.append(dict(cohort=cohort_label, endpoint="HRD_Score", metric="calibration_intercept_linear", score=intercept, ci_low=in_lo, ci_high=in_hi, n=n))
        for cutoff in (33, 42):
            yb = (yt >= cutoff).astype(int)
            auc = float(roc_auc_score(yb, yp))
            a_lo, a_hi = bootstrap_ci(lambda a, b: float(roc_auc_score(a, b)), yb, yp)
            rows.append(dict(cohort=cohort_label, endpoint=f"hrd_binary_{cutoff}", metric="auroc", score=auc, ci_low=a_lo, ci_high=a_hi, n=n))
            # Logistic calibration: logit P(y=1) = alpha + beta * predicted score.
            logr = LogisticRegression(C=1e6, max_iter=5000).fit(yp.reshape(-1, 1), yb)
            b_slope, b_inter = float(logr.coef_[0, 0]), float(logr.intercept_[0])
            bs_lo, bs_hi = bootstrap_ci(
                lambda a, b: float(LogisticRegression(C=1e6, max_iter=5000).fit(b.reshape(-1, 1), a).coef_[0, 0]), yb, yp)
            rows.append(dict(cohort=cohort_label, endpoint=f"hrd_binary_{cutoff}", metric="calibration_slope_logistic", score=b_slope, ci_low=bs_lo, ci_high=bs_hi, n=n))
            bi_lo, bi_hi = bootstrap_ci(
                lambda a, b: float(LogisticRegression(C=1e6, max_iter=5000).fit(b.reshape(-1, 1), a).intercept_[0]), yb, yp)
            rows.append(dict(cohort=cohort_label, endpoint=f"hrd_binary_{cutoff}", metric="calibration_intercept_logistic", score=b_inter, ci_low=bi_lo, ci_high=bi_hi, n=n))
            print(f"{cohort_label} HRD{cutoff}: AUROC={auc:.4f} [{a_lo:.4f},{a_hi:.4f}] "
                  f"cal_slope={b_slope:.4f} [{bs_lo:.4f},{bs_hi:.4f}] cal_inter={b_inter:.4f} [{bi_lo:.4f},{bi_hi:.4f}] (n={n})", flush=True)
        print(f"{cohort_label} HRD_Score: spearman={sp:.4f} [{sp_lo:.4f},{sp_hi:.4f}] "
              f"slope={slope:.4f} [{sl_lo:.4f},{sl_hi:.4f}] intercept={intercept:.4f} [{in_lo:.4f},{in_hi:.4f}] (n={n})", flush=True)

    out = pd.DataFrame(rows)
    out["analysis"] = "pog570_external"
    out["model"] = "frozen HRD_Score__signatures_plus_MAF_stack XGBoost regression (1,130 features)"
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
