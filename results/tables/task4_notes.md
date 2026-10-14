# Task 4 — Centre-held-out evaluation: notes

Question: does the cancer-typing result survive holding out whole sequencing
centres, for the types where that test is possible?

## Method as run

Design: leave-one-centre-out WITHIN each of the five multi-centre types
(CESC, OV, COAD, KIRC, KIRP). For each type T and each held-out centre C:
train the headline 20-class cancer-typing XGBoost model on all samples NOT
from centre C (the whole centre is excluded from training, all cancer types),
test on the type-T samples from centre C. Metric per fold: recall of the
held-out type (test sets are single-class, so balanced accuracy is undefined
there). Pooled metric per (type, representation): total correct / total
held-out tumours across that type's centre folds.

Script: `scripts/task4_centre_heldout.py`. It reuses the headline nested
validation machinery directly
(`src/utils/nested_oof.py::_search_and_predict_feature_set_fold`):
per fold, an inner stratified 75/25 train/validation split of the
centre-filtered training data; candidates = feature set x 5 XGBoost
hyperparameter draws (headline `random_search_trials` /
`xgb_feature_set_random_search_trials`); best (feature set, params) by
balanced accuracy on the inner validation split; refit on the full
outer-train data; predict the held-out centre. The feature set is the frozen
1130-col Bio MAF v4 block for Sig+MAF (see deviation 6) and the single
182-col set for patterns (which never had a block search). Seeds via
`stable_seed("task4_centre_heldout", type, heldout_group, representation)`.

Headline configuration matched:
- Endpoint `cancer_type_top20`, task multiclass, 20 classes, n=8800 (CDR
  labels restricted to patients with matched MC3 feature rows; cohort
  verified at exactly n=8800, TCGA-36-2539 absent).
- Learner XGBoost, `xgb_estimators=160`, 5 hyperparameter candidates from
  the pipeline's `_xgb_candidates` sampler, `multi:softprob`, `mlogloss`.
- Representations: `signatures_plus_MAF_stack` (frozen 1130-col block,
  deviation 6) vs `standard_sbs96_id83` (182 features, single feature set,
  no block selection).
- Feature matrices: 948-col MAF stack
  (`results/tables/quick_bio_v4_maf_features.csv.gz`) + 182-col SBS96+ID83
  (`feats_std.csv.gz`, downloaded from the companion HF dataset bundle
  `vjdeara/cgr_tensorlab_umsom`; column set verified identical to the frozen
  model's `feature_order.txt`). Reconstructed headline candidate-block
  column lists give exactly the headline counts 182/303/319/307/1105/1125/1130
  (the 948 matrix = current 910 v4 columns + 38 role-matched driver-gene
  columns whose genes became OncoKB INSUFFICIENT_EVIDENCE after the headline
  run; they belong to the exact-driver-genes family).

Centre assignment: tumour-sample barcodes via
`results/tables/batch_center_sample_mapping.csv` (last two barcode
characters), same counting convention as
`scripts/run_batch_center_analysis.py`: the two dual-aliquot patients
(TCGA-DV-A4W0 KIRC, TCGA-UZ-A9PS KIRP, both centre 10) contribute two
tumour-barcode rows each; features and model training are patient-level.
All 8800 cohort patients have centre mapping; no patient has tumours at two
centres. KIRC centre-10 (163) and KIRP centre-10 (168) reproduce the
manuscript's within-type counts exactly at tumour-barcode level.

## Small-centre rule (fixed before running)

Centres with fewer than 20 tumours of the type are pooled into a single
"small" test fold per type (train on everything else). Uniform across the
five types; no data discarded. In this cohort only OV has a sub-20 centre
(centre 01, 12 tumours). All other centres in the five types have >=41
tumours.

## Deviations from the headline pipeline

1. `tree_method=hist` instead of `gpu_hist` (no GPU on this VM). The
   pipeline's own `_fit_xgb_with_optional_fallback` falls back
   gpu_hist->hist anyway.
2. `xgb_n_jobs=1` (config: 32; 2-core VM shared with other tasks; 2-3
   concurrent single-thread workers).
3. Precomputed feature matrices + reconstructed headline candidate blocks
   instead of rebuilding features from the MAF (column counts verified to
   match the headline exactly, see above).
4. Outer folds are leave-one-centre-out by design (not the 5 stratified
   folds); pooled held-out metric is recall, not balanced accuracy.
5. The whole held-out centre is excluded from training, including other
   cancer types. Consequence: folds holding out centre 08 train on only 11
   classes (BLCA/GBM/HNSC/LGG/LUSC/PRAD/SKCM/STAD/THCA are single-centre-08);
   folds holding out centre 09 train on 16 classes (BRCA/ESCA/SARC/UCEC are
   single-centre-09); folds holding out centre 10 train on 19 classes (LIHC
   is single-centre-10). Per-fold training n and class counts are recorded
   in `task4_centre_heldout.csv`. Both representations face the same shift,
   so the Sig+MAF vs patterns comparison stays valid; per-type recall reads
   as generalisation to a never-seen centre.
6. FROZEN-BLOCK DESIGN for `signatures_plus_MAF_stack` (the one substantial
   pipeline simplification): instead of the nested 7-block search (35
   candidates per fold), every Sig+MAF fold uses the single 1130-col block
   `signatures_plus_v4_full_core_plus_optional_controls`, keeping the inner
   5-candidate XGBoost hyperparameter search intact. Why: (a) the one
   faithful Task 4 job completed (CESC held-out 08) selected the 1130 block
   on the inner validation split (0.6570 vs 0.6555/0.6533 for the other two
   large blocks); (b) the headline run's selected blocks for this endpoint
   were all in the 1105/1125/1130 family, of which 1130 is the superset;
   (c) the shared 2-core VM made the 35-candidate search infeasible
   (~20+ h remaining with other tasks' load). The `design` column in
   `task4_centre_heldout.csv` records `frozen_1130_block` vs
   `faithful_nested_block_search` per row. `standard_sbs96_id83` never had a
   block search (single 182-col set), so it is unaffected.

## Headline numbers

Pooled held-out-centre recall per type (tumour-barcode level; exact values in
`task4_centre_heldout.csv`, verified against `task4_centre_heldout_predictions.csv`):

| type | Sig+MAF pooled recall | patterns pooled recall |
|------|----------------------|------------------------|
| CESC | 0.1176 (34/289)      | 0.1038 (30/289)        |
| OV   | 0.6423 (264/411)     | 0.3966 (163/411)       |
| COAD | 0.5718 (231/404)     | 0.1337 (54/404)        |
| KIRC | 0.7297 (270/370)     | 0.4568 (169/370)       |
| KIRP | 0.4291 (121/282)     | 0.2199 (62/282)        |

Per-fold recall (held-out centre):

- CESC: 08: Sig+MAF 0.0244 (1/41), patterns 0.6341 (26/41);
  09: Sig+MAF 0.1331 (33/248), patterns 0.0161 (4/248)
- OV: 08: Sig+MAF 0.6117 (126/206), patterns 0.3252 (67/206);
  09: Sig+MAF 0.6788 (131/193), patterns 0.4767 (92/193);
  small(01): Sig+MAF 0.5833 (7/12), patterns 0.3333 (4/12)
- COAD: 09: Sig+MAF 0.8542 (41/48), patterns 0.5000 (24/48);
  10: Sig+MAF 0.5337 (190/356), patterns 0.0843 (30/356)
- KIRC: 08: Sig+MAF 0.7923 (164/207), patterns 0.4203 (87/207);
  10: Sig+MAF 0.6503 (106/163), patterns 0.5031 (82/163)
- KIRP: 08: Sig+MAF 0.3421 (39/114), patterns 0.2632 (30/114);
  10: Sig+MAF 0.4881 (82/168), patterns 0.1905 (32/168)

Reading: on four of five types (OV, COAD, KIRC, KIRP) Sig+MAF holds a clear
margin over patterns alone on unseen centres (pooled gaps +0.25, +0.44,
+0.27, +0.21). CESC is the exception: both representations fail on unseen
centres (pooled 0.118 vs 0.104), and the folds disagree in direction
(Sig+MAF 0.024 vs patterns 0.634 on held-out 08; Sig+MAF 0.133 vs patterns
0.016 on held-out 09). The CESC/08 Sig+MAF model confidently misclassifies
37/41 tumours as BRCA (mean predicted proba 0.88), a genuine centre-transfer
failure, not a label bug (test barcodes verified as 41 CESC tumours at
centre 08).

## Blockers

- `/tmp` was wiped by a service restart mid-task (2026-10-13 ~11:28); the
  downloaded assets (CDR xlsx, burden features, signature features) were
  re-downloaded into `~/workspace/task4_assets/` and the script reads them
  from there (`TASK4_ASSETS` env override).
- The runtime restarted 3 more times during the compute (~12:00, ~13:33,
  ~14:55 UTC), killing worker processes each time. No work was lost: every
  fold writes per-candidate checkpoints
  (`results/logs/task4_centre_heldout_checkpoints/candidate_rows/`) and
  per-job result/prediction files, and relaunched workers resume
  automatically ("resume candidate search ... completed N/M" in the logs).
- GitHub raw token unavailable (standing); the 182-col signature matrix is
  not in the checkout or the sibling bundle, so it was fetched from the
  public companion HF dataset (`vjdeara/cgr_tensorlab_umsom`,
  `github_exports_datasets/datasets/tcga_mc3/features/features_standard_sbs96_id83.csv.gz`,
  182 cols, sample IDs TCGA patient barcodes, column set verified identical
  to the frozen model's `feature_order.txt`).
