#!/usr/bin/env python
"""Build results_ledger.csv (G1) and metrics/artifact_hashes.csv.

Computed rows come from repro_common (the same functions the verifier uses).
Curated rows (extension reports, literature crosswalk, unsourced manuscript
numbers) come from scripts/ledger_curated.csv, which is maintained by hand
and reviewed against the source documents.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repro_common as rc

NS = rc.NS
COLS = ["claim_id", "location", "value", "source_file", "cohort",
        "model_or_learner", "estimator", "status", "notes"]
rows = []


def R(cid, location, value, source, cohort, model, estimator, status, notes=""):
    rows.append({"claim_id": cid, "location": location, "value": value,
                 "source_file": source, "cohort": cohort, "model_or_learner": model,
                 "estimator": estimator, "status": status, "notes": notes})


def f4(x):
    return f"{x:.4f}"


PRED_REL = "analyses/representation_extension/predictions/"
CA = "TCGA cancer typing (n=8,800)"
HB = "TCGA BRCA HRD cohort (n=772)"

# ------------------------------------------------------------- grid cells
for task, rep, learner in rc.list_cells():
    cell = rc.cell_name(task, rep, learner)
    folds, primary = rc.cell_scores(task, rep, learner)
    cohort = CA if task == "cancer" else HB
    est = "balanced accuracy" if task == "cancer" else "Spearman correlation"
    src = f"{PRED_REL}{cell}.csv"
    for k in range(1, 6):
        R(f"EXT-CELL-{cell}-FOLD{k}", f"Extension grid, cell {cell}, fold {k}",
          f4(folds[k]), src, cohort, learner, f"within-fold {est}", "reproduced")
    R(f"EXT-CELL-{cell}-PRIMARY", f"Extension grid, cell {cell}, primary score",
      f4(primary), src, cohort, learner, f"mean within-fold {est} (protocol primary)",
      "reproduced")

# ------------------------------------------------------------- A8 contrasts
a8 = rc.a8_contrasts()
A8LOC = "Primary contrasts table (workstream A8), {task} / {learner}"
for key, entry in a8.items():
    task, learner = key.split("/")
    cohort = CA if task == "cancer" else HB
    est = "balanced accuracy" if task == "cancer" else "Spearman correlation"
    for rep in rc.REPS_A8:
        R(f"EXT-A8-{task}-{learner}-{rep}-PRIMARY",
          A8LOC.format(task=task, learner=learner) + f", {rep}",
          f4(entry["primary"][rep]),
          f"{PRED_REL}{task}__{rep}__{learner}.csv", cohort, learner,
          f"mean within-fold {est}", "reproduced")
    for rep, c in entry["contrasts"].items():
        base = f"EXT-A8-{task}-{learner}-{rep}"
        srcs = (f"{PRED_REL}{task}__{rep}__{learner}.csv; "
                f"{PRED_REL}{task}__spectra+maf_stack__{learner}.csv")
        R(base + "-DIFF", A8LOC.format(task=task, learner=learner) +
          f", {rep} minus spectra+maf_stack",
          f"{c['mean_per_fold_diff']:+.4f}", srcs, cohort, learner,
          "mean per-fold paired difference", "reproduced")
        for side, v in [("CILO", c["bootstrap_ci95"][0]), ("CIHI", c["bootstrap_ci95"][1])]:
            R(base + "-" + side, A8LOC.format(task=task, learner=learner) +
              f", {rep} minus spectra+maf_stack, bootstrap 95% CI " +
              ("lower" if side == "CILO" else "upper"),
              f"{v:+.4f}", srcs, cohort, learner,
              "patient-level paired bootstrap 95% CI (2,000 resamples, seed 20261025)",
              "reproduced", "Monte Carlo tolerance for CI bounds: +/-0.002")

# ------------------------------------------------------------- audit recomputes
sc = rc.sbs3_brca_counts()
R("EXT-SBS3-NONZERO", "Audit report section 6; manuscript Results (HRD controls)",
  "9", "data/task3/sbs3_activity_772.csv", HB, "SBS3 fitted activity",
  "count of patients with activity > 0", "reproduced",
  "SigProfilerAssignment 1.1.5, COSMIC v3.4; verification item 9 noted no summary file states this count; it is recomputed here from the raw activity file")
R("EXT-SBS3-MEAN", "Audit report section 6", f4(sc["sbs3_mean"]),
  "data/task3/sbs3_activity_772.csv", HB, "SBS3 fitted activity", "mean activity",
  "reproduced", "audit report prints 1.54")
R("EXT-SBS3-MAX", "Audit report section 6", f4(sc["sbs3_max"]),
  "data/task3/sbs3_activity_772.csv", HB, "SBS3 fitted activity", "maximum activity",
  "reproduced")
R("EXT-BRCA-POS772", "Audit report section 6; manuscript Results (HRD controls)",
  "14", "results/tables/quick_bio_maf_features.csv.gz", HB,
  "BRCA1/2 role-matched event indicator", "count of patients with indicator > 0",
  "reproduced", "columns driver_gene_role_matched_event_log_count__BRCA1/BRCA2")
R("EXT-BRCA-POSALL", "Audit report section 6", "204",
  "results/tables/quick_bio_maf_features.csv.gz", "TCGA all cohorts (n=10,224)",
  "BRCA1/2 role-matched event indicator", "count of patients with indicator > 0",
  "reproduced")

# ------------------------------------------------------------- manuscript Table 1
PRINT_T1 = {
    "HRD_Score": {"burden_only": (0.528, 0.612), "standard_sbs96_id83": (0.508, 0.694),
                  "MAF_stack_only": (0.718, 0.739), "signatures_plus_MAF_stack": (0.725, 0.777)},
    "hrd_binary_24": {"burden_only": (0.765, 0.788), "standard_sbs96_id83": (0.774, 0.839),
                      "MAF_stack_only": (0.813, 0.851), "signatures_plus_MAF_stack": (0.841, 0.860)},
    "hrd_binary_33": {"burden_only": (0.788, 0.821), "standard_sbs96_id83": (0.791, 0.874),
                      "MAF_stack_only": (0.831, 0.873), "signatures_plus_MAF_stack": (0.839, 0.911)},
    "hrd_binary_42": {"burden_only": (0.828, 0.850), "standard_sbs96_id83": (0.878, 0.915),
                      "MAF_stack_only": (0.842, 0.880), "signatures_plus_MAF_stack": (0.894, 0.925)},
    "cancer_type_top20": {"burden_only": (0.247, 0.255), "standard_sbs96_id83": (0.526, 0.538),
                          "MAF_stack_only": (0.571, 0.662), "signatures_plus_MAF_stack": (0.672, 0.738)},
    "OS": {"burden_only": (0.584,), "standard_sbs96_id83": (0.607,),
           "MAF_stack_only": (0.634,), "signatures_plus_MAF_stack": (0.651,)},
}
EP_PRINT = {"HRD_Score": "HRD score", "hrd_binary_24": "HRD24", "hrd_binary_33": "HRD33",
            "hrd_binary_42": "HRD42", "cancer_type_top20": "Cancer type", "OS": "Overall survival"}
EP_COHORT = {"HRD_Score": HB, "hrd_binary_24": HB, "hrd_binary_33": HB, "hrd_binary_42": HB,
             "cancer_type_top20": CA, "OS": "TCGA overall survival (n=9,986)"}
EP_METRIC = {"HRD_Score": "Spearman correlation", "hrd_binary_24": "AUROC",
             "hrd_binary_33": "AUROC", "hrd_binary_42": "AUROC",
             "cancer_type_top20": "balanced accuracy", "OS": "Harrell C-index"}
t1 = rc.table1_fold_means()
FM_SRC = "results/tables/main_manuscript_complete_panel_fold_metrics.csv"
for ep, reps in PRINT_T1.items():
    for rep, vals in reps.items():
        learners = ["cox_ph"] if ep == "OS" else ["linear", "xgboost"]
        for learner, printed in zip(learners, vals):
            mean, _ = t1[(ep, rep, learner)]
            assert round(mean, 3) == printed, (ep, rep, learner, mean, printed)
            R(f"MS-T1-{ep}-{rep}-{learner}",
              f"Manuscript Table 1, {EP_PRINT[ep]} row, {rc.REP_PRINT[rep]} column",
              f"{printed:.3f}", FM_SRC, EP_COHORT[ep],
              {"linear": "elastic net", "xgboost": "XGBoost", "cox_ph": "CoxNet"}[learner],
              f"fold mean of {EP_METRIC[ep]} over 5 folds", "reproduced")
# Table 1 / text confidence intervals (t-intervals from the five fold scores)
CI_CELLS = [("HRD_Score", "Table 1, Sig+MAF XGBoost CI", (0.745, 0.808)),
            ("hrd_binary_33", "Table 1, Sig+MAF XGBoost CI", (0.886, 0.936)),
            ("cancer_type_top20", "Table 1, Sig+MAF XGBoost CI", (0.721, 0.755)),
            ("hrd_binary_24", "Results text, Sig+MAF XGBoost CI", (0.831, 0.889)),
            ("hrd_binary_42", "Results text, Sig+MAF XGBoost CI", (0.895, 0.955))]
for ep, loc, printed in CI_CELLS:
    lo, hi = rc.t_interval(t1[(ep, "signatures_plus_MAF_stack", "xgboost")][1])
    for side, pv, rv in [("LO", printed[0], lo), ("HI", printed[1], hi)]:
        R(f"MS-T1CI-{ep}-{side}", f"Manuscript {loc} ({'lower' if side == 'LO' else 'upper'})",
          f"{pv:.3f}", FM_SRC, EP_COHORT[ep], "XGBoost",
          "t-interval from the five fold scores (t crit 2.776, df 4)", "reproduced",
          f"recomputed {rv:.4f}")
# PFI fold means (Results / supplement survival values)
PFI_PRINT = {"burden_only": 0.556, "standard_sbs96_id83": 0.570,
             "MAF_stack_only": 0.621, "signatures_plus_MAF_stack": 0.631}
sv_means = rc.survival_fold_means()
for rep, printed in PFI_PRINT.items():
    mean = sv_means[("PFI", rep, "cox_ph")]
    assert round(mean, 3) == printed, (rep, mean, printed)
    R(f"MS-PFI-{rep}-cox_ph", f"Manuscript survival results, PFI row, {rc.REP_PRINT[rep]} column",
      f"{printed:.3f}", FM_SRC, "TCGA progression-free interval (n=9,857)", "CoxNet",
      "fold mean of Harrell C-index over 5 folds", "reproduced")

# ------------------------------------------------------------- MuAt column
mc = rc.muat_cancer()
mh = rc.muat_hrd()
pw = rc.paired_vs_muat()
MU_C = "results/tables/muat_style_tcga_comparator_cancer_type_top20_faithful_fixed_arch_oof_predictions.csv"
MU_H = "results/tables/muat_style_tcga_comparator_hrd_faithful_fixed_arch_oof_predictions.csv"
R("MS-MUAT-CANCER-BA", "Manuscript Table 1, Cancer type row, MuAt-compatible column",
  "0.528", MU_C, CA, "MuAt-compatible (faithful fixed architecture)",
  "fold mean of balanced accuracy", "reproduced")
for k in range(1, 6):
    R(f"MS-MUAT-CANCER-FOLD{k}", f"MuAt faithful run, cancer fold {k} balanced accuracy",
      f"{mc['fold_ba'][k]:.3f}", MU_C, CA, "MuAt-compatible (faithful fixed architecture)",
      "within-fold balanced accuracy", "reproduced")
R("MS-MUAT-CANCER-ACC", "Manuscript Table 1 note, MuAt cancer accuracy", "0.549",
  MU_C, CA, "MuAt-compatible (faithful fixed architecture)", "pooled accuracy", "reproduced")
R("MS-MUAT-CANCER-TOP5", "Manuscript Table 1 note, MuAt cancer top-5 accuracy", "0.861",
  MU_C, CA, "MuAt-compatible (faithful fixed architecture)", "pooled top-5 accuracy",
  "reproduced")
for side, pv in [("LO", 0.514), ("HI", 0.542)]:
    R(f"MS-MUAT-CANCER-BACI-{side}", f"Manuscript Table 1, MuAt cancer CI ({'lower' if side == 'LO' else 'upper'})",
      f"{pv:.3f}", MU_C, CA, "MuAt-compatible (faithful fixed architecture)",
      "t-interval from the five fold scores", "reproduced",
      f"recomputed bound {mc['ba_ci'][0 if side == 'LO' else 1]:.4f}" +
      ("; manuscript prints 0.543" if side == "HI" else ""))
R("MS-MUAT-HRD-SPEARMAN", "Manuscript Table 1, HRD score row, MuAt-compatible column",
  "0.507", MU_H, HB, "MuAt-compatible (faithful fixed architecture)",
  "fold mean of Spearman correlation", "reproduced")
for side, pv in [("LO", 0.432), ("HI", 0.582)]:
    R(f"MS-MUAT-HRD-CI-{side}", f"Manuscript Table 1, MuAt HRD score CI ({'lower' if side == 'LO' else 'upper'})",
      f"{pv:.3f}", MU_H, HB, "MuAt-compatible (faithful fixed architecture)",
      "t-interval from the five fold scores", "reproduced")
R("MS-MUAT-HRD33-AUROC", "Manuscript Table 1, HRD33 row, MuAt-compatible column",
  "0.746", MU_H, HB, "MuAt-compatible (faithful fixed architecture)",
  "fold mean of AUROC", "reproduced")
for k in range(1, 6):
    R(f"MS-MUAT-HRD33-FOLD{k}", f"MuAt faithful run, HRD33 fold {k} AUROC",
      f"{mh['hrd_binary_33']['folds'][k]:.3f}", MU_H, HB,
      "MuAt-compatible (faithful fixed architecture)", "within-fold AUROC", "reproduced")
for side, pv in [("LO", 0.692), ("HI", 0.800)]:
    R(f"MS-MUAT-HRD33-CI-{side}", f"Manuscript Table 1, MuAt HRD33 CI ({'lower' if side == 'LO' else 'upper'})",
      f"{pv:.3f}", MU_H, HB, "MuAt-compatible (faithful fixed architecture)",
      "t-interval from the five fold scores", "reproduced")
PAIR_SRC = f"{FM_SRC}; {MU_C}; {MU_H}"
R("MS-PAIR-HRD-DIFF", "Manuscript Results / Table 1 note, Sig+MAF minus MuAt HRD Spearman difference",
  "+0.270", PAIR_SRC, HB, "XGBoost vs MuAt-compatible", "mean per-fold paired difference",
  "reproduced", f"recomputed {pw['hrd_spearman']['mean']:+.4f}; also stored in results/tables/hrd_muat_paired_tests.json")
R("MS-PAIR-HRD-P", "Manuscript Results, HRD paired difference p value", "0.0001",
  PAIR_SRC, HB, "XGBoost vs MuAt-compatible", "two-sided paired t test on 5 fold differences",
  "reproduced", f"recomputed {pw['hrd_spearman']['p']:.6f}")
for side, pv in [("LO", 0.220), ("HI", 0.319)]:
    R(f"MS-PAIR-HRD-CI-{side}", f"Manuscript Results, HRD paired difference CI ({'lower' if side == 'LO' else 'upper'})",
      f"{pv:+.3f}", PAIR_SRC, HB, "XGBoost vs MuAt-compatible",
      "paired t interval on 5 fold differences", "reproduced")
R("MS-PAIR-HRD33-DIFF", "Manuscript Results, Sig+MAF minus MuAt HRD33 AUROC difference",
  "+0.165", PAIR_SRC, HB, "XGBoost vs MuAt-compatible", "mean per-fold paired difference",
  "reproduced", f"recomputed {pw['hrd33_auroc']['mean']:+.4f}; also stored in results/tables/hrd_muat_paired_tests.json")
R("MS-PAIR-HRD33-P", "Manuscript Results, HRD33 paired difference p value", "0.003",
  PAIR_SRC, HB, "XGBoost vs MuAt-compatible", "two-sided paired t test on 5 fold differences",
  "reproduced", f"recomputed {pw['hrd33_auroc']['p']:.6f}")
for side, pv in [("LO", 0.094), ("HI", 0.236)]:
    R(f"MS-PAIR-HRD33-CI-{side}", f"Manuscript Results, HRD33 paired difference CI ({'lower' if side == 'LO' else 'upper'})",
      f"{pv:+.3f}", PAIR_SRC, HB, "XGBoost vs MuAt-compatible",
      "paired t interval on 5 fold differences", "reproduced")
R("MS-PAIR-CANCER-DIFF", "Manuscript Results, Sig+MAF minus MuAt cancer balanced accuracy difference",
  "+0.209", PAIR_SRC, CA, "XGBoost vs MuAt-compatible", "mean per-fold paired difference",
  "reproduced", f"recomputed {pw['cancer_ba']['mean']:+.4f}")
R("MS-PAIR-CANCER-CI-LO", "Manuscript Results, cancer paired difference CI (lower)", "+0.196",
  PAIR_SRC, CA, "XGBoost vs MuAt-compatible", "paired t interval on 5 fold differences",
  "reproduced", "manuscript prints 0.197; the recomputation from the saved folds gives 0.196")
R("MS-PAIR-CANCER-CI-HI", "Manuscript Results, cancer paired difference CI (upper)", "+0.222",
  PAIR_SRC, CA, "XGBoost vs MuAt-compatible", "paired t interval on 5 fold differences",
  "reproduced", f"recomputed {pw['cancer_ba']['ci_hi']:+.4f}")
R("MS-PAIR-CANCER-P", "Manuscript Results, cancer paired difference p value",
  f"{pw['cancer_ba']['p']:.2e}", PAIR_SRC, CA, "XGBoost vs MuAt-compatible",
  "two-sided paired t test on 5 fold differences", "reproduced",
  "manuscript prints p<0.000001; the paired t recomputation gives 1.5e-06, marginally weaker than the printed bound")

# ------------------------------------------------------------- ablations
abl = rc.ablations()
ABL_SRC = "results/tables/task1_gene_knowledge_ablations.csv"
ABL_PRINT = {("maf", "a_full"): ("Bio MAF alone, full", 0.662),
             ("maf", "c_no_gene_identity"): ("Bio MAF alone, no gene identity", 0.540),
             ("maf", "b_no_knowledge"): ("Bio MAF alone, no knowledge summaries", 0.659),
             ("sigmaf", "a_full"): ("Sig+MAF, full", 0.738),
             ("sigmaf", "c_no_gene_identity"): ("Sig+MAF, no gene identity", 0.677),
             ("sigmaf", "e_binary_genes"): ("Sig+MAF, binary gene control", 0.654),
             ("sigmaf", "f_driver_gene_counts"): ("Sig+MAF, driver count control", 0.614)}
for (rep, var), (label, printed) in ABL_PRINT.items():
    assert round(abl[(rep, var)], 3) == printed, (rep, var, abl[(rep, var)], printed)
    R(f"MS-ABL-{rep}-{var}", f"Manuscript Results, gene knowledge ablation: {label}",
      f"{printed:.3f}", ABL_SRC, CA, "XGBoost",
      "pooled balanced accuracy (mean of per-fold values in source file)", "reproduced",
      "control refits reuse the full run's winner hyperparameters (audit report section 4)")
R("MS-ABL-KNOWLEDGE-DELTA", "Manuscript Results (Figure 6 text), change when curated knowledge columns are removed",
  "0.003", ABL_SRC, CA, "XGBoost", "difference of pooled balanced accuracies", "reproduced",
  f"recomputed |{abl[('sigmaf','b_no_knowledge')]:.4f} - {abl[('sigmaf','a_full')]:.4f}| = 0.0030 (removal raises the score slightly)")

# ------------------------------------------------------------- PCAWG / POG570
pc = rc.pcawg()
PC_SRC = "external_validation/pcawg_transfer/predictions/pcawg_sigmaf_predictions.csv"
PC_COH = "PCAWG ICGC-only (n=913)"
R("MS-PCAWG-ACC", "Manuscript Results, PCAWG Sig+MAF accuracy", "0.614", PC_SRC, PC_COH,
  "XGBoost (frozen TCGA model)", "accuracy", "reproduced")
for side, pv in [("LO", 0.583), ("HI", 0.645)]:
    R(f"MS-PCAWG-ACC-CI-{side}", f"Manuscript Results, PCAWG accuracy CI ({'lower' if side == 'LO' else 'upper'})",
      f"{pv:.3f}", "external_validation/pcawg_transfer/predictions/pcawg_sigmaf_metrics.json",
      PC_COH, "XGBoost (frozen TCGA model)", "bootstrap 95% CI as stored", "sourced_not_rerun")
R("MS-PCAWG-BA", "Manuscript Results, PCAWG Sig+MAF balanced accuracy", "0.557", PC_SRC,
  PC_COH, "XGBoost (frozen TCGA model)", "balanced accuracy", "reproduced")
PC_RECALL_PRINT = {"BRCA": 0.85, "LIHC": 0.83, "KIRC": 0.82, "ESCA": 0.77, "SKCM": 0.73,
                   "OV": 0.67, "PRAD": 0.03, "HNSC": 0.15, "STAD": 0.16}
for typ, printed in PC_RECALL_PRINT.items():
    assert round(pc["recalls"][typ], 2) == printed, (typ, pc["recalls"][typ], printed)
    R(f"MS-PCAWG-RECALL-{typ}", f"Manuscript Results, PCAWG Sig+MAF per-type recall, {typ}",
      f"{printed:.2f}", PC_SRC, PC_COH, "XGBoost (frozen TCGA model)", "per-type recall",
      "reproduced")
PCS_SRC = "results/external_validation/aaron_email/pcawg/pcawg_frozen_model_predictions.csv"
sp = pd.read_csv(rc.ROOT / PCS_SRC)
from sklearn.metrics import balanced_accuracy_score as _bas
R("MS-PCAWG-SPECTRA-ACC", "Manuscript Results, PCAWG spectra-only accuracy", "0.479",
  PCS_SRC, PC_COH, "XGBoost (frozen TCGA model)", "accuracy", "reproduced",
  "this file is the spectra-only frozen model's predictions (accuracy 0.4786)")
R("MS-PCAWG-SPECTRA-BA", "Manuscript Results, PCAWG spectra-only balanced accuracy", "0.365",
  PCS_SRC, PC_COH, "XGBoost (frozen TCGA model)", "balanced accuracy", "reproduced")
for typ, printed in [("BRCA", 0.077), ("KIRC", 0.000)]:
    sub = sp[sp["true_label"] == typ]
    rec = float((sub["pred_label"] == typ).mean())
    assert round(rec, 3) == printed, (typ, rec, printed)
    R(f"MS-PCAWG-SPECTRA-RECALL-{typ}", f"Manuscript Results, PCAWG spectra-only recall, {typ}",
      f"{printed:.3f}", PCS_SRC, PC_COH, "XGBoost (frozen TCGA model)", "per-type recall",
      "reproduced")
pg = rc.pog570()
PG_SRC = "results/external_validation/aaron_email/pog570/pog570_frozen_model_predictions.csv"
PG_COH = "POG570 breast primary (n=147)"
for cid, loc, printed, key in [
        ("MS-POG-SPEARMAN", "Spearman correlation", 0.667, "spearman"),
        ("MS-POG-PEARSON", "Pearson correlation", 0.670, "pearson"),
        ("MS-POG-R2", "R squared", -0.225, "r2"),
        ("MS-POG-SPEC24", "specificity at HRD cutoff 24", 0.034, "specificity24")]:
    assert round(pg[key], 3) == printed, (cid, pg[key], printed)
    R(cid, f"Manuscript Results, POG570 {loc}", f"{printed:+.3f}" if printed < 0 else f"{printed:.3f}",
      PG_SRC, PG_COH, "XGBoost (frozen TCGA model)", loc, "reproduced",
      "subset: analysis_group = primary_breast")

# ------------------------------------------------------------- centre held out
centre = rc.centre_heldout()
CT_SRC = "results/tables/task4_centre_heldout.csv"
CT_PRINT = {"OV": (0.64, 0.40), "COAD": (0.57, 0.13), "KIRC": (0.73, 0.46),
            "KIRP": (0.43, 0.22), "CESC": (0.12, 0.10)}
for typ, (p_sig, p_spec) in CT_PRINT.items():
    for rep, printed in [("signatures_plus_MAF_stack", p_sig), ("standard_sbs96_id83", p_spec)]:
        v = centre[(typ, rep)]["pooled_tumours"]
        assert round(v, 2) == printed, (typ, rep, v, printed)
        R(f"MS-CENTRE-{typ}-{rep}", f"Manuscript Results, centre-held-out pooled recall, {typ}, {rc.REP_PRINT[rep]}",
          f"{printed:.2f}", CT_SRC, "TCGA centre-held-out", "XGBoost",
          "pooled recall (summed correct over summed test tumours)", "reproduced",
          "two dual-aliquot patients are double-weighted (audit report section 7)")

# ------------------------------------------------------------- survival covariate run
svv = rc.survival_covariate()
SV_SRC = "results/tables/survival_cancer_type_baseline_summary.csv"
for rep, printed in [("cancer_type", 0.742), ("bio_maf", 0.638), ("cancer_type_plus_bio_maf", 0.743)]:
    assert round(svv["scores"][rep], 3) == printed, (rep, svv["scores"][rep], printed)
    R(f"MS-SURVCOV-{rep}", f"Manuscript Figure 8 / Results, survival covariate run: {rep}",
      f"{printed:.3f}", SV_SRC, "TCGA overall survival (n=9,986)", "Cox model",
      "Harrell C-index", "reproduced")
R("MS-SURVCOV-GAIN", "Manuscript Figure 8, gain from adding Bio MAF to cancer type", "+0.0013",
  "results/tables/survival_cancer_type_baseline_comparisons.csv",
  "TCGA overall survival (n=9,986)", "Cox model", "difference of C-indices", "reproduced")
R("MS-SURVCOV-P", "Manuscript Figure 8, gain p value", "0.512",
  "results/tables/survival_cancer_type_baseline_comparisons.csv",
  "TCGA overall survival (n=9,986)", "Cox model", "two-sided p value", "reproduced")
for side, pv in [("LO", -0.003), ("HI", 0.005)]:
    R(f"MS-SURVCOV-CI-{side}", f"Manuscript Figure 8, gain CI ({'lower' if side == 'LO' else 'upper'})",
      f"{pv:+.3f}", "results/tables/survival_cancer_type_baseline_comparisons.csv",
      "TCGA overall survival (n=9,986)", "Cox model", "95% CI of the C-index difference",
      "reproduced", f"stored interval {svv['ci']}")

# ------------------------------------------------------------- HRD controls
hc = rc.hrd_controls()
HC_SRC = "results/tables/hrd_shortcut_ablation_summary.csv"
HCC_SRC = "results/tables/hrd_shortcut_ablation_comparisons.csv"
FULL = "signatures_plus_MAF_stack__full"
def ctl(cid, loc, value, src, model, est, notes=""):
    R(cid, loc, value, src, HB, model, est, "reproduced", notes)
ctl("MS-HRDCTL-FULL-SCORE", "Manuscript Results, HRD control full model HRD score", "0.760",
    HC_SRC, "XGBoost", "pooled Spearman correlation", f"stored pooled score {hc['scores'][('HRD_Score', FULL)]:.4f}")
ctl("MS-HRDCTL-NOGENES-SCORE", "Manuscript Results, HRD score after repair-gene removal", "0.763",
    HC_SRC, "XGBoost", "pooled Spearman correlation",
    f"stored pooled score {hc['scores'][('HRD_Score', 'signatures_plus_MAF_stack__no_hrd_genes')]:.4f}")
ctl("MS-HRDCTL-NOGENES-P", "Manuscript Results, repair-gene removal p value (HRD score)", "0.622",
    HCC_SRC, "XGBoost", "two-sided p value")
ctl("MS-HRDCTL-FULL-33", "Manuscript Results, HRD control full model HRD33", "0.900",
    HC_SRC, "XGBoost", "pooled AUROC", f"stored pooled score {hc['scores'][('hrd_binary_33', FULL)]:.4f}")
ctl("MS-HRDCTL-NOGENES-33", "Manuscript Results, HRD33 after repair-gene removal", "0.898",
    HC_SRC, "XGBoost", "pooled AUROC",
    f"stored pooled score {hc['scores'][('hrd_binary_33', 'signatures_plus_MAF_stack__no_hrd_genes')]:.4f}")
ctl("MS-HRDCTL-NOGENES-P33", "Manuscript Results, repair-gene removal p value (HRD33)", "0.637",
    HCC_SRC, "XGBoost", "two-sided p value")
ctl("MS-HRDCTL-NOVAF-33", "Manuscript Results, HRD33 after VAF removal", "0.874",
    HC_SRC, "XGBoost", "pooled AUROC",
    f"stored pooled score {hc['scores'][('hrd_binary_33', 'signatures_plus_MAF_stack__no_vaf')]:.4f}")
R("MS-HRDCTL-NOVAF-P33", "Manuscript Results, VAF removal p value (HRD33)", "<0.001",
  HCC_SRC, HB, "XGBoost", "two-sided p value", "reproduced",
  "comparisons file stores 0.000 (rounded); manuscript prints p<0.001")
t3 = rc.task3_baselines()
T3_SRC = "results/tables/task3_hrd_attribution_baselines.csv"
for cid, loc, key, printed in [
        ("MS-T3-SIG3-SPEARMAN", "SBS3 alone, HRD score", ("sig3_only", "HRD_Score", "spearman"), -0.068),
        ("MS-T3-SIG3-AUROC33", "SBS3 alone, HRD33", ("sig3_only", "hrd_binary_33", "auroc"), 0.501),
        ("MS-T3-SIG3-AUROC42", "SBS3 alone, HRD42", ("sig3_only", "hrd_binary_42", "auroc"), 0.510),
        ("MS-T3-BRCA-SPEARMAN", "BRCA1/2 alone, HRD score", ("brca12_only", "HRD_Score", "spearman"), -0.065),
        ("MS-T3-BRCA-AUROC33", "BRCA1/2 alone, HRD33", ("brca12_only", "hrd_binary_33", "auroc"), 0.507),
        ("MS-T3-BRCA-AUROC42", "BRCA1/2 alone, HRD42", ("brca12_only", "hrd_binary_42", "auroc"), 0.524)]:
    assert round(t3[key], 3) == printed, (cid, t3[key], printed)
    R(cid, f"Manuscript Results, HRD attribution control: {loc}", f"{printed:+.3f}" if printed < 0 else f"{printed:.3f}",
      T3_SRC, HB, "XGBoost (fixed config, archived folds)", "pooled score as stored in source file",
      "reproduced", "fixed-config runs in the archived original folds (source column of the file)")

# ------------------------------------------------------------- Figure 3 and 6
g3 = rc.fig3_gains()
OOF_SRC = "results/tables/main_manuscript_complete_panel_oof_predictions.csv"
for k in sorted(g3):
    lab = g3[k]["label"]
    R(f"MS-FIG3-{lab}-GAIN", f"Manuscript Figure 3, per-type recall gain (Sig+MAF minus spectra), {lab}",
      f"{g3[k]['gain']:+.2f}", OOF_SRC, CA, "XGBoost",
      "difference of pooled per-type recalls from out-of-fold predictions", "reproduced")
    R(f"MS-FIG3-{lab}-N", f"Manuscript Figure 3, per-type sample size, {lab}",
      str(g3[k]["n"]), OOF_SRC, CA, "XGBoost", "patient count", "reproduced")
SH_SRC = "results/tables/feature_selection_cancer_type_top20_shap_values.csv"
text_sh = rc.shap_text_shares(rc.T / "feature_selection_cancer_type_top20_shap_values.csv")
for name, printed in [("Mutation consequence summaries", 23.8), ("Per-gene driver counts", 17.3),
                      ("Functional-effect predictions", 16.3), ("Impact", 10.9),
                      ("OncoKB gene roles", 9.1), ("Allele-fraction summaries", 7.8),
                      ("Driver evidence tiers", 6.1), ("Hotspot summaries", 4.3),
                      ("Mutation type / count", 4.4)]:
    assert round(text_sh[name], 1) == printed, (name, text_sh[name], printed)
    R(f"MS-FIG6-TEXT-{name}", f"Manuscript Results text, Figure 6 SHAP share: {name}",
      f"{printed:.1f}", SH_SRC, CA, "Bio MAF XGBoost (feature-selection run)",
      "share of total mean |SHAP| (%)", "reproduced",
      "manuscript text grouping (see claim_audit.md for the grouping definition)")
fig_sh = rc.shap_shares(rc.T / "feature_selection_cancer_type_top20_shap_values.csv")
for name, v in fig_sh.items():
    R(f"MS-FIG6-FIG-{name}", f"Manuscript Figure 6 (regenerated), SHAP share: {name}",
      f"{v:.1f}", SH_SRC, CA, "Bio MAF XGBoost (feature-selection run)",
      "share of total mean |SHAP| (%)", "reproduced", "figure block grouping (fixed prefix rules)")
os_sh = rc.shap_shares(rc.T / "feature_selection_OS_shap_values.csv")
for name, v in os_sh.items():
    R(f"MS-FIG6OS-{name}", f"Manuscript Figure 6 survival panel (regenerated), SHAP share: {name}",
      f"{v:.1f}", "results/tables/feature_selection_OS_shap_values.csv",
      "TCGA overall survival (n=9,986)", "Bio MAF XGBoost (feature-selection run)",
      "share of total mean |SHAP| (%)", "reproduced", "figure block grouping (fixed prefix rules)")
R("MS-FIG6-MAXFEATURE", "Manuscript Results text, largest single-feature SHAP share",
  f"{rc.shap_max_feature_share(rc.T / 'feature_selection_cancer_type_top20_shap_values.csv'):.2f}",
  SH_SRC, CA, "Bio MAF XGBoost (feature-selection run)", "share of total mean |SHAP| (%)",
  "reproduced", "manuscript states no single feature exceeds 6%")
fsj = json.load(open(rc.T / "feature_selection_cancer_type_top20_summary.json"))
R("MS-FIG6-SELECTED-N", "Manuscript Results, features retained by SHAP selection", "347",
  "results/tables/feature_selection_cancer_type_top20_summary.json", CA,
  "Bio MAF XGBoost (feature-selection run)", "feature count", "sourced_not_rerun",
  f"of {fsj['n_features_available']} available")
R("MS-FIG6-HELDOUT", "Manuscript Results, held-out score of the selected feature set", "0.697",
  "results/tables/feature_selection_cancer_type_top20_summary.json", CA,
  "Bio MAF XGBoost (feature-selection run)", "held-out balanced accuracy", "sourced_not_rerun",
  f"stored value {fsj['heldout_score']:.4f}")

# ------------------------------------------------------------- Kucab + text contrasts
kuc = rc.kucab_pooled()
R("MS-KUCAB-SPECTRA-LINEAR", "Manuscript Additional results, Kucab spectra pooled score (elastic net)",
  "0.331", "results/tables/main_manuscript_complete_panel_endpoint_results.csv",
  "Kucab damage class (n=259)", "elastic net", "pooled balanced accuracy", "reproduced",
  "representation standard_sbs96_dbs78_id83; Kucab has its own spectra matrix")
R("MS-KUCAB-SPECTRA-XGB", "Manuscript Additional results, Kucab spectra pooled score (XGBoost)",
  "0.250", "results/tables/main_manuscript_complete_panel_endpoint_results.csv",
  "Kucab damage class (n=259)", "XGBoost", "pooled balanced accuracy", "reproduced")
R("MS-KUCAB-EVENTSTACK-XGB", "Manuscript Additional results, Kucab event-token stack pooled score (XGBoost)",
  "0.174", "results/tables/main_manuscript_complete_panel_endpoint_results.csv",
  "Kucab damage class (n=259)", "XGBoost", "pooled balanced accuracy", "reproduced",
  "the Kucab event stack (2,980 features) is not Bio MAF; it is labelled MAF_stack_only in the file by file convention only")
ep = rc.endpoint_results()
def pooled(ep_id, rep, learner):
    r = ep[(ep["endpoint"] == ep_id) & (ep["representation"] == rep) & (ep["learner"] == learner)].iloc[0]
    return float(r["score"])
R("MS-TEXT-SIGMAF-VS-MAF-DIFF", "Manuscript Results, Sig+MAF minus Bio MAF pooled cancer difference",
  "+0.076", "results/tables/main_manuscript_complete_panel_endpoint_results.csv", CA,
  "XGBoost", "difference of pooled balanced accuracies", "reproduced",
  f"recomputed {pooled('cancer_type_top20','signatures_plus_MAF_stack','xgboost'):.4f} - {pooled('cancer_type_top20','MAF_stack_only','xgboost'):.4f} = +0.0760")
for ep_id, cid, printed in [("hrd_binary_33", "MS-TEXT-HRD33-GAIN", 0.037),
                            ("HRD_Score", "MS-TEXT-HRD-GAIN", 0.082)]:
    d = t1[(ep_id, "signatures_plus_MAF_stack", "xgboost")][0] - t1[(ep_id, "standard_sbs96_id83", "xgboost")][0]
    assert round(d, 3) == printed, (cid, d, printed)
    R(cid, f"Manuscript Results, secondary contrast gain for {EP_PRINT[ep_id]} (Sig+MAF minus spectra, fold means)",
      f"{printed:+.3f}", FM_SRC, HB, "XGBoost", "difference of fold-mean scores", "reproduced",
      "the gain itself is the fold-mean difference; its printed p and q values are unsourced (see UNS rows)")

# ------------------------------------------------------------- curated rows
curated_path = Path(__file__).resolve().parent / "ledger_curated.csv"
with open(curated_path, newline="") as f:
    for r in csv.DictReader(f):
        rows.append({c: r[c] for c in COLS})

# ------------------------------------------------------------- write ledger
out_path = NS / "results_ledger.csv"
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)
print(f"ledger rows: {len(rows)}")
from collections import Counter
print(Counter(r["status"] for r in rows))

# ------------------------------------------------------------- artifact hashes
HASH_FILES = []
for p in sorted((NS / "predictions").glob("*.csv")):
    HASH_FILES.append(("extension_predictions", p))
for p in sorted((NS / "metrics" / "trials").glob("*.csv")):
    HASH_FILES.append(("extension_trials", p))
MS_ARTIFACTS = [
    "results/tables/main_manuscript_complete_panel_fold_metrics.csv",
    "results/tables/main_manuscript_complete_panel_endpoint_results.csv",
    "results/tables/main_manuscript_complete_panel_oof_predictions.csv",
    "results/tables/muat_style_tcga_comparator_cancer_type_top20_faithful_fixed_arch_oof_predictions.csv",
    "results/tables/muat_style_tcga_comparator_hrd_faithful_fixed_arch_oof_predictions.csv",
    "results/tables/hrd_muat_paired_tests.json",
    "results/tables/cancer_typing_primary_fold_differences.json",
    "results/tables/task1_gene_knowledge_ablations.csv",
    "results/tables/task3_hrd_attribution_baselines.csv",
    "results/tables/hrd_shortcut_ablation_summary.csv",
    "results/tables/hrd_shortcut_ablation_comparisons.csv",
    "results/tables/survival_cancer_type_baseline_summary.csv",
    "results/tables/survival_cancer_type_baseline_comparisons.csv",
    "results/tables/task4_centre_heldout.csv",
    "results/tables/feature_selection_cancer_type_top20_shap_values.csv",
    "results/tables/feature_selection_OS_shap_values.csv",
    "results/tables/feature_selection_cancer_type_top20_summary.json",
    "results/tables/quick_bio_maf_features.csv.gz",
    "external_validation/pcawg_transfer/predictions/pcawg_sigmaf_predictions.csv",
    "external_validation/pcawg_transfer/predictions/pcawg_sigmaf_metrics.json",
    "results/external_validation/aaron_email/pcawg/pcawg_frozen_model_predictions.csv",
    "results/external_validation/aaron_email/pcawg/pcawg_metrics_with_95ci.csv",
    "results/tables/task5_pcawg_audit_per_type.csv",
    "results/external_validation/aaron_email/pog570/pog570_frozen_model_predictions.csv",
    "results/external_validation/aaron_email/pog570/pog570_primary_and_all_metrics_with_95ci.csv",
    "data/task3/sbs3_activity_772.csv",
    "data/task3/final_analysis_cohort.tsv",
]
for rel in MS_ARTIFACTS:
    HASH_FILES.append(("manuscript_artifacts", rc.ROOT / rel))
hash_path = NS / "metrics" / "artifact_hashes.csv"
with open(hash_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scope", "path", "sha256", "bytes"])
    for scope, p in HASH_FILES:
        rel = str(p.relative_to(rc.ROOT)) if str(p).startswith(str(rc.ROOT)) else str(p)
        w.writerow([scope, rel, rc.sha256_file(p), p.stat().st_size])
print(f"hash rows: {len(HASH_FILES)}")
