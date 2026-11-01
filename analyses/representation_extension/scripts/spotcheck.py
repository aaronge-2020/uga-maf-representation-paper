#!/usr/bin/env python
"""Bounded spot check: refit four banked folds end to end and compare the
refit predictions with the banked predictions file.

Folds (chosen for recorded cost and coverage of all four task/learner
combinations):
  hrd__spectra+maf_stack__elasticnet fold 1  (the October kill-and-resume
      proof cell; exact equality expected)
  hrd__spectra+maf_stack__xgboost fold 1
  cancer__spectra+maf_stack__xgboost fold 5
  cancer__spectra__elasticnet fold 2

Trial configurations are reconstructed from the saved trial tables (the
per-fold trial_configs.json files were deleted when the cells completed).
The refit mirrors runner.run_cell's fold path exactly: same canonical
split, same inner holdout (StratifiedShuffleSplit, seed 20261025), trial
seeds SEED + t, final fit seed SEED, RUNNER_N_JOBS=2 as on the grid runs.

Pass rule, set from measurement and disclosed in README_REPRODUCE.md:
every refit trial inner score within 1e-9 of the trial table (observed:
exactly 0 on all four folds), the FINAL record's configuration identical
and its inner score within 1e-9, and refit predictions within 1e-5
absolute of the banked predictions. Classification predictions are class
labels, so the 1e-5 bound still requires identical labels. The prediction
tolerance is not zero: a cold refit in a fresh process reproduces the
tuning layer bit-exactly but carries small floating-point deviations in
the final fit's leaf and summation arithmetic (observed maxima: 3.6e-15
elastic net, 3.8e-06 XGBoost on the HRD folds), far too small to move any
fold score at the printed precision. Byte identity after kill-and-resume
was proven in October for the checkpoint machinery itself (deviations log
entries 9 and 10, md5 2bce989bb717ddc487962868d306f7e3); this spot check
proves the stronger statement that a cold refit from the saved trial
tables alone lands on the same tuning outcome and the same predictions
to 1e-5.
"""
import ast
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RUNNER_N_JOBS", "2")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repro_common as rc

sys.path.insert(0, str(HERE))
import runner

NS = rc.NS
SEED = runner.SEED
SPOT = [("hrd", "spectra+maf_stack", "elasticnet", 1, True),
        ("hrd", "spectra+maf_stack", "xgboost", 1, False),
        ("cancer", "spectra+maf_stack", "xgboost", 5, False),
        ("cancer", "spectra", "elasticnet", 2, False)]
TOL = 1e-9
PRED_TOL = 1e-5


def refit_fold(task, rep, learner, fold):
    cell = rc.cell_name(task, rep, learner)
    split = pd.read_csv(NS / "config" / f"canonical_split_{task}.csv",
                        dtype={"patient": str})
    patients = list(split["patient"])
    y_all = split["label"].values
    X_df, _ = runner.assemble(task, rep, patients)
    X_all = X_df.values.astype(np.float64)
    te_mask = split["fold"].values == fold
    tr_idx = np.where(~te_mask)[0]
    sss = StratifiedShuffleSplit(1, test_size=0.2, random_state=SEED)
    strat = y_all[tr_idx] if task == "cancer" else pd.qcut(y_all[tr_idx], 5, labels=False)
    inner_tr_pos, inner_va_pos = next(sss.split(np.zeros(len(tr_idx)), strat))
    itr, iva = tr_idx[inner_tr_pos], tr_idx[inner_va_pos]

    table = pd.read_csv(NS / "metrics" / "trials" / f"{cell}.csv", dtype={"trial": str})
    sub = table[table["fold"] == fold].copy()
    trial_rows = sub[sub["trial"] != "FINAL"].sort_values(by="trial", key=lambda s: s.astype(int))
    final_row = sub[sub["trial"] == "FINAL"].iloc[0]

    best = None
    max_inner_dev = 0.0
    for _, r in trial_rows.iterrows():
        t = int(r["trial"])
        cfg = ast.literal_eval(r["config"])
        if r["status"] != "ok":
            continue
        p_va, _, _ = runner.fit_predict(learner, task, cfg, X_all[itr], y_all[itr],
                                        X_all[iva], y_all[iva], X_all[iva], SEED + t)
        s = runner.score(task, y_all[iva], p_va)
        max_inner_dev = max(max_inner_dev, abs(s - float(r["inner_score"])))
        if best is None or s > best[0]:
            best = (s, cfg)
    cfg = best[1]
    p_te, proba, _ = runner.fit_predict(learner, task, cfg, X_all[tr_idx], y_all[tr_idx],
                                        X_all[iva], y_all[iva], X_all[te_mask], SEED)
    refit = pd.DataFrame({"patient": np.array(patients)[te_mask], "prediction": p_te})
    if proba is not None:
        for k in range(proba.shape[1]):
            refit[f"proba_{k}"] = proba[:, k]
    banked = rc.load_predictions(task, rep, learner)
    banked = banked[banked["fold"] == fold][["patient", "prediction"]
              + [c for c in banked.columns if c.startswith("proba_")]]
    m = banked.merge(refit, on="patient", suffixes=("_banked", "_refit"))
    assert len(m) == int(te_mask.sum()), (cell, fold, len(m), int(te_mask.sum()))
    max_pred_dev = float(np.abs(m["prediction_banked"].astype(float)
                                - m["prediction_refit"].astype(float)).max())
    max_proba_dev = 0.0
    for c in [c for c in m.columns if c.startswith("proba_") and c.endswith("_banked")]:
        base = c[:-len("_banked")]
        max_proba_dev = max(max_proba_dev, float(np.abs(
            m[c].astype(float) - m[base + "_refit"].astype(float)).max()))
    final_cfg_ok = ast.literal_eval(final_row["config"]) == cfg
    final_score_dev = abs(float(final_row["inner_score"]) - best[0])
    return {"cell": cell, "fold": fold, "max_pred_dev": max_pred_dev,
            "max_proba_dev": max_proba_dev, "max_inner_dev": max_inner_dev,
            "final_cfg_ok": final_cfg_ok, "final_score_dev": final_score_dev}


def main():
    failures = 0
    for task, rep, learner, fold, proven in SPOT:
        t0 = time.time()
        r = refit_fold(task, rep, learner, fold)
        dev = max(r["max_pred_dev"], r["max_proba_dev"])
        ok = (dev <= PRED_TOL and r["max_inner_dev"] <= TOL and r["final_cfg_ok"]
              and r["final_score_dev"] <= TOL)
        failures += 0 if ok else 1
        print(("PASS " if ok else "FAIL ")
              + f"spotcheck {r['cell']} fold {fold}: max pred dev {r['max_pred_dev']:.3g}, "
              + f"max proba dev {r['max_proba_dev']:.3g}, max inner-score dev "
              + f"{r['max_inner_dev']:.3g}, FINAL config match {r['final_cfg_ok']}, "
              + f"FINAL score dev {r['final_score_dev']:.3g} "
              + f"({time.time() - t0:.0f}s)", flush=True)
    print("SPOTCHECK RESULT:", "PASS" if failures == 0 else "FAIL")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
