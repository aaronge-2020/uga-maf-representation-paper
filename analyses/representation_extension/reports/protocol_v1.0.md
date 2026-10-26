# Fixed analysis protocol, version 1.0 (2026-10-25)

Status: fixed before any new outcome comparison is run under this protocol. This is a follow-up analysis protocol for an existing study. It is not a preregistration. The TCGA test folds, PCAWG, and POG570 results have all been examined before (see the audit report, section on prior exposure), so no cohort in this project counts as untouched validation data.

Source commit: ca13132462fab9d1be364a2477ea8e753d7d5050 (main), worked on branch `analysis/representation-extension`. Companion document: `audit_report.md` in this folder.

## 1. Primary questions

Q1 (Priority A). On matched patients, matched outer folds, and independently tuned learners, does spectra plus Bio MAF beat spectra plus each simpler gene encoding (binary presence, counts, consequence-aware counts, TF-IDF weighting) on (a) 20-type cancer classification and (b) continuous HRD score prediction in 772 TCGA breast cancers?

Q2 (Priority B). Which Bio MAF feature blocks carry any advantage: gene event information, consequence and functional prediction summaries, VAF summaries, curated role/hotspot/driver summaries?

Q3 (Priority C). Do the leading representations keep their ranking under frozen transfer to PCAWG (ICGC-only, 913 samples) and POG570 breast (n=147), and does consistent feature construction change the answer?

Q4 (Priority D). Are HRD findings robust to the sparse-control audit (SBS3, BRCA indicators) and to adjustment for burden, purity, ploidy, and subtype on matched subsets?

## 2. Cohorts and endpoints (frozen definitions)

- Cancer typing: the 8,800 patients with a top-20 cancer type label in the main panel OOF file, fold assignment taken from that same file (canonical shared splits). Label set: the 20 class labels recorded there.
- HRD regression: the 772 patients in `data/task3/final_analysis_cohort.tsv`, outcome `HRD_Score` (continuous). Secondary: HRD24/33/42 binary labels from the same cohort file family.
- Survival (later, conditional): 9,986 patients, OS primary, from the CDR table already in the repo.
- External: PCAWG ICGC-only 913 samples (nine mapped types, full 20-label prediction space retained); POG570 breast primary cohort n=147 with HRD outcome. No external sample enters any fitting, tuning, feature selection, or threshold choice.

Patient identity: TCGA 12-character patient barcode for TCGA; dataset-native identifiers for PCAWG and POG570, never truncated with the TCGA rule.

## 3. Representations

All gene representations use mutation events from the repo's MuAt-compatible events files (the only mutation-level data stored in the repo), restricted as follows, fixed here: an event is eligible if its variant_classification is one of the coding set: Missense_Mutation, Silent, Nonsense_Mutation, Frame_Shift_Del, Frame_Shift_Ins, In_Frame_Del, In_Frame_Ins, Splice_Site, Nonstop_Mutation, Translation_Start_Site. UTR, intron, RNA, and flank events in the MuAt token stream are excluded, matching the manuscript's coding-mutation cohort framing. Bio MAF itself consumes all annotated rows for its aggregate features (its code applies no classification filter), so this eligibility rule makes the simple encodings a coding-event control, and the difference is recorded in the deviations log. One event row is one mutation call; duplicate rows for the same patient, position, ref, and alt are counted once.

- R0 burden: the repo's 3-feature matrix (`features_burden_only.csv`), as archived.
- R1 spectra: the repo's 182-feature matrix (`features_standard_sbs96_id83.csv.gz`: SBS96 + ID83 + count summaries as constructed by the original pipeline), as archived. Raw channel counts and fitted signature activities are not mixed; this matrix is used exactly as the manuscript used it.
- R2 binary gene presence: per patient, 1 if at least one eligible event in the gene.
- R3 gene counts: per patient, number of eligible events in the gene.
- R4 consequence-aware counts: per patient and gene, counts split by consequence group (missense, truncating [nonsense, frameshift, splice], other coding). A variant with multiple transcript consequences contributes once, under the single classification stored in the events file; there is no transcript re-picking in this benchmark, and this is recorded as a limitation.
- R5 TF-IDF gene weighting, Ozcan Simsek 2019 Eq. 1 and Eq. 2 exactly: document = patient, term = gene, tf(g,s) = raw count of eligible events in gene g, idf(g) = log(N_train / df_train(g)) with natural log (the paper does not state the base; the choice is recorded here), fitted on the training partition only at every level (outer and inner). Genes with df_train = 0 are excluded from the vocabulary. Amended pre-execution from a smoothed draft variant to match the published equations (see deviations log). This is the paper's own inference-safe method; it is not the paper's headline tf-rf or BM25-tf-rf, whose transform depends on the patient's true class and which therefore cannot be applied to an unlabeled patient as published (see crosswalk section B).
- R6 Bio MAF: the archived 948-feature representation. Its feature matrix is not stored in the repo (feature caches were excluded), so R6 enters through (a) its archived per-fold metrics and OOF predictions, flagged with the known OOF/table mismatch for a subset of groups, and (b) where a fold-level refit is required and impossible without the matrix, the contrast is reported as blocked with that exact prerequisite. No silent substitute for R6 is allowed.

Gene scope, two arms, reported separately:
- Panel arm: the Bio MAF per-gene panel (the genes behind the 802 per-gene columns, recovered from the Bio MAF feature guide table in the repo).
- All-gene arm: every gene with at least one eligible event in the training partition; prevalence filter (gene mutated in at least 5 training patients) fitted inside training only. R2 and R3 only, per the brief.

Combinations: each of R2 to R5 is run alone and concatenated with R1. R6 combinations are the archived Sig+MAF results. The primary question is added value beyond spectra, so the combined arm is primary and the alone arm is secondary.

Normalization sensitivity (fixed here, not tuned): R3 and R4 also run as log1p counts in a sensitivity arm. R2 and R5 need no transform. Burden-normalized counts are not run, to keep the sensitivity family small; the deviation log records this choice.

CADD-weighted representation: conditional. It runs only if per-variant CADD scores with verified provenance and coverage logging exist locally. The audit found none in the repo, so this arm starts as blocked (missing prerequisite: a licensed CADD score source matched to the event calls). R4 is the available biological encoding control and is not described as equivalent to CADD weighting.

## 4. Learners and tuning

Learners: elastic net (logistic for classification with balanced accuracy as the tuning objective; linear with Spearman as the tuning objective for HRD regression) and XGBoost, matching the manuscript's learners.

Tuning: per representation, per learner, per outer fold, an independent random search of 10 trials (the manuscript's original budget) over fixed learner-specific spaces (recorded in `config/search_spaces.yaml` when the runner is written, before first execution). Inner validation: the canonical inner folds where recoverable, else a fixed stratified 5-fold split of the outer training partition with seed 20261025, fitted transforms inside each inner fold. Early stopping for XGBoost uses the inner validation split only. Class weighting: XGBoost scale handled by sample weights computed from the training partition only; elastic net uses class_weight balanced for classification, none for regression. Every trial, seed, selected configuration, fit time, and failure is written to the tuning trial table; failed trials are kept in the table, not dropped.

Seed policy: one seed (20261025) for the primary runs. For the leading contrast (best simple representation vs the archived Sig+MAF reference), a 3-seed stability check on XGBoost is run if the primary difference is under 0.02 balanced accuracy or 0.02 Spearman.

Compute envelope: this machine has 2 CPU cores and 7 GB RAM, no GPU. XGBoost runs with nthread=2, hist tree method. Estimated cost is in the audit report; if the full grid exceeds the envelope, the reduction order fixed here is: (1) drop the alone arms for R4 and R5, (2) drop the log1p sensitivity for R4, (3) reduce XGBoost trials to 6 for the all-gene arm only. The primary combined panel arm is never reduced.

## 5. Estimators and statistics

- Cancer typing primary: balanced accuracy computed from the complete pooled outer predictions; secondary: macro F1, accuracy, per-type recall, calibration (log loss, Brier). Fold-mean values are reported alongside and labelled, never mixed with pooled values in one number.
- HRD regression primary: mean within-fold Spearman; secondary: pooled Spearman (labelled sensitivity), MAE, R squared.
- HRD classification secondary: mean within-fold AUROC and AUPRC, with prevalence stated.
- Contrasts: the four primary contrasts of the brief (Sig+MAF vs spectra plus each of R2, R3, R4, R5), evaluated as paired patient bootstrap differences (10,000 resamples, stratified by cancer type for typing and by fold for HRD, seed fixed), conditional on the fitted models and labelled as such. Multiplicity: Benjamini-Hochberg within the primary family of 8 (4 contrasts x 2 tasks). Bootstrap results are never reported as exactly zero; the smallest reportable value is 1/10,000.
- Paired patient matching for every contrast is by explicit join on patient identifier; exclusions are counted and reported.
- Equivalence: no equivalence claim without a prespecified margin. The margin fixed here for "practically similar" is 0.01 balanced accuracy and 0.02 Spearman, assessed by whether the bootstrap interval lies inside it.

## 6. Priorities B, C, D rules

- B: feature blocks are defined from the Bio MAF feature guide (gene event, consequence/functional, VAF, curated). Additions go onto spectra plus R3 (the plain count control) and removals come off the archived Bio MAF block structure where the matrix allows; where the missing Bio MAF matrix blocks a removal refit, the archived task1 ablation OOF files are the evidence and the limitation is stated. Any compact subset is selected inside training data only (the repo's earlier 347-feature selection used full-cohort SHAP and is treated as exploratory only).
- C: frozen models only; PCAWG scored over the full 20-label space with balanced accuracy over the nine observed true types, per-type recall and the confusion matrix reported; POG570 with Spearman, MAE, R squared, calibration slope/intercept, and threshold metrics at 24/33/42. One factor changed at a time; any redefined feature set gets its own TCGA refit and internal evaluation before freezing.
- D: SBS3 and BRCA sparsity audited from `data/task3/sbs3_activity_772.csv` and the cohort table before any interpretation. Covariate analyses (burden, purity, ploidy, subtype) on matched complete-case subsets with missingness reported. SigMA is the candidate HRD comparator, subject to input and training-overlap checks; no WGS-only method is forced onto exome data.

## 7. Complementary endpoint (section 9A)

Feasibility review only, in the brief's priority order (LUAD smoking, PAM50, MSI, HPV), from local label files. One endpoint is adopted only if labels and class sizes support a complete core comparison, and the choice is made from label quality and counts, never from outer test performance.

## 8. Reproducibility

Every run writes: config hash, input file checksums (sha256), library versions, seed, wall time, exit status, and failures, into `analyses/representation_extension/metrics/run_log.csv`. Predictions are saved as CSV with dataset, task, patient key, true outcome, fold, seed, representation, learner, prediction, and class probabilities where applicable. A fresh-process metric rebuild check is run on one representative completed contrast before results are reported.

## Amendment v1.2 (2026-10-25, before primary outcomes)

Measured compute on this machine (one full-train XGBoost fit on the cancer Sig+MAF matrix: 147 s) showed the original tree cap made the grid non-executable. XGBoost n_estimators cap is 400 with early stopping at 40 rounds. Trials per outer fold: 10 for all primary combined and reference cells and all HRD cells; 6 for cancer alone arms and extension cells. Full reasoning and the disclosure of the one interrupted run are in the deviations log, entry 8. No primary outcome had been computed under the earlier settings.
