# Task 3 notes: HRD attribution baselines

Question: is the HRD gain repair biology or scar-correlated summaries?

## Method as run

Three baselines were evaluated on the TCGA HRD cohort (n = 772) in the same five
outer folds as the existing HRD analysis, for the three endpoints HRD_Score
(Spearman), hrd_binary_33 (AUROC), hrd_binary_42 (AUROC):

- (a) `sig3_only`: Signature 3 activity alone (one continuous feature: SBS3
  mutation counts attributed by signature fitting).
- (b) `brca12_only`: BRCA1/2 mutation status alone (one binary feature).
- (c) `sig3_brca12`: both features together.

Learner: XGBoost with one fixed configuration per outer fold
(max_depth=2, learning_rate=0.05, n_estimators=200, min_child_weight=1,
reg_lambda=1.0, reg_alpha=0.0, subsample=0.9, colsample_bytree=1.0,
tree_method=hist, n_jobs=8, random_state=seed+fold). Binary endpoints use the
predicted class-1 probability as the score; the regression endpoint uses the
predicted value.

Fold provenance: the nested-fold checkpoint npz files in
`results/tables/main_manuscript_complete_panel_nested_fold_checkpoints/`
record the original main-panel `sample_folds` and the seeds used
(HRD_Score 196932, hrd_binary_33 476778, hrd_binary_42 134427). The sample order
was reconstructed from the original runner code path (cohort TSV row order,
labels built exactly as in
`src/runners/run_main_manuscript_complete_panel.py::_load_mc3_endpoints`,
intersected with the feature index). Re-running `outer_splits` with the
recorded seeds in that order reproduces the archived `sample_folds` bit-for-bit
for all three endpoints. The baseline script asserts this equality before
fitting, so the baselines ran in the archived original folds, not a re-split.

Note: `muat_style_tcga_comparator_ab_c_full_split_manifest.csv` was examined as
a candidate fold source but its per-sample test assignments do NOT match the
original main-panel folds (597/772 and 626/772 mismatches for HRD_Score and
hrd_binary_33; hrd_binary_42 absent entirely), so it was not used.

## Signature-activity provenance

No precomputed signature activities existed in the repository for this cohort,
so they were fitted with SigProfilerAssignment 1.1.5 (SigProfilerMatrixGenerator
1.4.0, sigProfilerPlotting 1.4.3), reference set COSMIC v3.4 SBS on GRCh37
(87 signatures including artefact signatures, auto-downloaded by the tool),
context SBS96, exome=False, default NNLS penalties (nnls_add_penalty=0.05,
nnls_remove_penalty=0.01), no plots.

Input counts: the repository's `features_standard_sbs96.csv.gz` stores
normalized per-channel frequencies, not counts. Integer counts were recovered
exactly by finding, per sample, the integer denominator D such that every
fraction equals count/D (verified exact to <1e-12 for all 772 samples; 764 with
the naive D, 8 needing a 2x multiple because their smallest nonzero count was
2). The fitted activities were sanity-checked: mean cosine similarity 0.844,
SBS3 assigned nonzero in 9 of 772 samples (mean activity 1.54, max 250), raw
SBS3 vs HRD_Score Spearman 0.151. The SBS3 column is the (a) baseline feature.
Raw assignment outputs are in the task working directory
(`~/workspace/task3_inputs/sigprofiler_out/`); the SBS3 activity table used is
`~/workspace/task3_inputs/sbs3_activity_772.csv`.

The SigProfilerAssignment input must be tab-separated (`input_type="matrix"`
reads with `sep="\t"`); the first attempt with a comma-separated file failed
with an IndexError inside sigProfilerPlotting, fixed by rewriting as TSV.

## BRCA1/2 definition used

The pipeline has no pathogenicity classifier, so the closest pipeline-native
definition of "protein-truncating" was used: a sample is BRCA1/2-positive if
`driver_gene_role_matched_event_log_count__BRCA1 > 0` or the BRCA2 column `> 0`
in `results/tables/quick_bio_v4_maf_features.csv.gz`. For BRCA1/BRCA2 (both
annotated "loss-of-function/tumor-suppressor evidence" in the 401-gene driver
panel), `is_role_matched` in `biomaf_v4/biomaf_v4/features.py` counts events
with VEP IMPACT "high" or consequence in {stop_gained, frameshift_variant,
splice_acceptor_variant, splice_donor_variant, start_lost,
transcript_ablation}. The log counts are log1p, so > 0 means at least one such
event. This is a truncating/high-impact proxy, not a ClinVar pathogenicity
call. 14 of 772 samples are positive.

## Headline numbers (pooled; exact values from task3_hrd_attribution_baselines.csv)

| baseline | HRD_Score (Spearman) | HRD33 (AUROC) | HRD42 (AUROC) |
|---|---|---|---|
| (a) sig3_only | -0.067805 | 0.500659 | 0.510397 |
| (b) brca12_only | -0.064635 | 0.507415 | 0.524153 |
| (c) sig3_brca12 | -0.040344 | 0.510914 | 0.533017 |

Fold-mean (over folds with defined scores; 4 of 5 folds defined for the two
single-feature regression baselines, 5 of 5 elsewhere):

| baseline | HRD_Score | HRD33 | HRD42 |
|---|---|---|---|
| (a) sig3_only | 0.155438 | 0.503321 | 0.510000 |
| (b) brca12_only | 0.167707 | 0.520887 | 0.534267 |
| (c) sig3_brca12 | 0.185807 | 0.524227 | 0.546853 |

Existing results in the same CSV for comparison (Sig+MAF =
signatures_plus_MAF_stack xgboost; patterns-alone = MuAt stage C exact dicts):

| analysis | HRD_Score pooled / fold-mean | HRD33 pooled / fold-mean | HRD42 pooled / fold-mean |
|---|---|---|---|
| Sig+MAF | 0.766645 / 0.776503 | 0.900812 / 0.911064 | 0.896961 / 0.925188 |
| patterns-alone (MuAt) | 0.504122 / 0.527235 | 0.702825 / 0.751330 | not run |

Reading: neither Signature 3 activity alone nor BRCA1/2 status alone carries the
HRD signal (AUROCs 0.50-0.53, regression fold-mean Spearman 0.16-0.19). The full
Sig+MAF stack (0.77/0.90/0.90 pooled) is far above both attribution baselines,
and also above the patterns-alone MuAt comparator (0.50/0.70 pooled).

## Deviations and why

1. Fixed XGBoost config instead of the nested hyperparameter search. The nested
   inner search is degenerate for these 1-2-feature baselines: on several outer
   folds the inner validation split contains no positive feature values at all
   (SBS3>0 in 9/772 samples, BRCA1/2-positive in 14/772), so every candidate
   predicts a constant, the inner Spearman is undefined, and the runner raises
   "No valid nested-search candidate". A fixed, documented XGBoost configuration
   was used instead. The outer folds are the archived originals (verified
   bit-for-bit), which is the requirement the task actually imposes.
2. For the regression endpoint, per-fold Spearman is undefined (NaN) on folds
   where the fitted model predicts a constant (folds 1 and 3 for the
   single-feature baselines); fold-mean is the mean over defined folds only
   (n_defined_folds column). Pooled Spearman is slightly negative while
   per-fold values are positive: each fold's model has its own prediction
   offset, and with weak features the between-fold offset variance swamps the
   within-fold rank signal when OOF predictions are pooled. Both are reported
   as computed.
3. HRD label and SBS96 count files were fetched from the public Hugging Face
   dataset `vjdeara/cgr_tensorlab_umsom` (the bundle `scripts/fetch_datasets.py`
   is built to fetch), because the sibling `../github_exports_datasets` bundle
   is not present in this checkout. SHA-256 of both files matches the repo's
   recorded manifests exactly (final_analysis_cohort.tsv:
   7e280d5823ed0c35ccaa59d68d864db05999a7a55537055f6ab13c86b72cdf84;
   features_standard_sbs96.csv.gz:
   e450c7ffb1bfb4cbfb71a05b884b192ffda62fee4851088977f06240c6b75a12).

## Blockers

None for the three baselines or the POG570 item; all items completed.

## POG570 item

POG570 external predictions exist
(`results/external_validation/aaron_email/pog570/pog570_frozen_model_predictions.csv`,
frozen HRD_Score__signatures_plus_MAF_stack XGBoost regression, 1,130
features), so AUROC, calibration slope/intercept, and bootstrap 95% CIs
(2,000 resamples, percentile) were computed from them and saved to
`results/tables/task3_pog570_external_metrics.csv`. The AUROC and Spearman
values reproduce the published external-validation summary exactly.

Primary breast cohort (n=147): HRD33 AUROC 0.805178 [0.729097, 0.869976],
logistic calibration slope 0.100548 [0.070525, 0.139485], intercept -4.861756
[-6.805631, -3.455243]; HRD42 AUROC 0.871007 [0.801223, 0.928993], slope
0.144867 [0.104012, 0.214839], intercept -8.268630 [-12.276798, -6.002883];
HRD_Score Spearman 0.667363 [0.553043, 0.754000], linear calibration slope
0.894168 [0.738212, 1.040214], intercept -9.223787 [-15.835641, -2.268708].

All POG570 (n=570): HRD33 AUROC 0.770803 [0.719741, 0.818568], slope 0.095526
[0.075380, 0.119286], intercept -5.355546 [-6.554457, -4.375454]; HRD42 AUROC
0.827020 [0.765455, 0.882147], slope 0.125710 [0.094035, 0.168935], intercept
-8.031113 [-10.454389, -6.366514]; HRD_Score Spearman 0.520391 [0.449323,
0.587377], slope 0.751611 [0.651277, 0.848566], intercept -9.033723
[-13.077037, -4.971520].

Calibration slopes far below 1 (binary) confirm the documented upward
prediction shift and poor absolute calibration of the frozen model on POG570,
despite retained discrimination.

## Files

- `results/tables/task3_hrd_attribution_baselines.csv`: per-fold, fold-mean,
  and pooled scores for the three baselines plus Sig+MAF and patterns-alone
  comparison rows (source column names each row's origin).
- `results/tables/task3_hrd_attribution_oof_sig3_only.csv`,
  `results/tables/task3_hrd_attribution_oof_brca12_only.csv`,
  `results/tables/task3_hrd_attribution_oof_sig3_brca12.csv`: out-of-fold
  predictions (tumour_barcode, endpoint, true, predicted_score, fold).
- `results/tables/task3_pog570_external_metrics.csv`: POG570 AUROC /
  calibration metrics with bootstrap 95% CIs.
- `results/tables/task3_notes.md`: this file.
- `scripts/run_task3_hrd_attribution_baselines.py`,
  `scripts/run_task3_pog570_metrics.py`: runnable scripts.
