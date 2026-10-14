# Reviewer analyses notes (branch `reviewer-analyses-2026-10`)

Working notes for the reviewer-requested analyses, kept on this branch only.
Nothing here edits the manuscript.

---

## Task 5 — PCAWG audit (2026-10-12)

### Method as run
1. Located the saved PCAWG scoring outputs:
   - Sig+MAF: `external_validation/pcawg_transfer/predictions/pcawg_sigmaf_predictions.csv`
     (913 rows) + `pcawg_sigmaf_metrics.json`.
   - Patterns-only (standard_sbs96_id83): `results/external_validation/aaron_email/pcawg/pcawg_frozen_model_predictions.csv`
     (913 rows) + `pcawg_metrics_with_95ci.csv`.
   - Sample mapping: `results/external_validation/aaron_email/pcawg/pcawg_913_sample_mapping.csv`
     (913 rows, all included; byte-identical to the pipeline copy).
2. Verified reproducibility: rescored both frozen models
   (`results/frozen_models/cancer_type_top20__signatures_plus_MAF_stack`,
   `results/frozen_models/cancer_type_top20__standard_sbs96_id83`) — model.ubj
   sha256 verified against each manifest, and rescoring reproduced all 913
   saved pred_labels for each model. No retraining, no new prediction files needed.
3. Wrote `scripts/task5_pcawg_audit.py`: per-type n, recall per model, bootstrap
   95% CI (10,000 replicates, fixed seed 42, resampling tumours within type),
   paired gain of Sig+MAF over patterns-only → `results/tables/task5_pcawg_audit_per_type.csv`.
4. Wrote `scripts/task5_column_alignment_check.py`: verifies the scored tables'
   column sets and order equal the frozen `feature_order.txt` schemas →
   `results/tables/task5_column_alignment_check.txt` (PASS).
5. Full write-up: `results/tables/task5_pcawg_audit_report.md`.

### Headline numbers
- Cohort reconciles at **913 tumours across 9 types**: LIHC 261, PRAD 180,
  BRCA 117, ESCA 97, KIRC 74, SKCM 70, OV 69, STAD 32, HNSC 13.
- Per-type Sig+MAF recall: BRCA 0.846, ESCA 0.773, HNSC 0.154, KIRC 0.824,
  LIHC 0.828, OV 0.667, PRAD 0.033, SKCM 0.729, STAD 0.156.
- Per-type patterns-only recall: BRCA 0.077, ESCA 0.856, HNSC 0.000, KIRC 0.000,
  LIHC 0.973, OV 0.507, PRAD 0.006, SKCM 0.714, STAD 0.156.
- Largest Sig+MAF gains: KIRC +0.824, BRCA +0.769, OV +0.159. Sig+MAF loses to
  patterns-only on LIHC (−0.146) and ESCA (−0.083).
- Printed numbers all verify: SKCM Sig+MAF 0.73 (0.7286), patterns 0.714
  (0.7143), LIHC 0.83 (0.8276), ESCA 0.77 (0.7732).
- Column alignment: by column NAME against the frozen `feature_order.txt`;
  1,130/1,130 used, 0 dropped, 0 zero-filled (no NaNs); assembly hard-asserted
  zero missing/extra columns.

### Deviations
- Task premise said "948-feature model scored a table with 1,130 columns."
  The frozen Sig+MAF model itself has 1,130 features (948 Bio MAF v4 + 182
  signature features); 948 is the Bio MAF block alone. Outputs needed no
  correction — only the premise.
- Pre-existing metrics used 2,000 bootstrap replicates (Sig+MAF seed 42,
  spectra-only seed 20260822); this audit recomputed per-type CIs at the
  task-specified 10,000 replicates, seed 42.

### Blockers
- None. Predictions were saved for both models and reproduced exactly from the
  frozen artefacts, so no rescoring files were required.
- The Sig+MAF per-type printed numbers were checked against the values stated
  in the task; the repo's `submission_manuscript.docx` snapshot does not contain
  that per-type list (it lives in the Google Doc, not accessed here).

---

## Task 0 — Reproduce Table 1 (2026-10-13)

### The seed bug (found and fixed 2026-10-13)

The first Task 0 rerun failed its acceptance check on `burden_only`/linear
(measured 0.246153 vs expected 0.247320, diff 0.001167). Investigation proved
the cause was a wrong seed, not a Python-version artifact:

- Two functions share the name `stable_seed`:
  - `src/utils/nested_oof.py`: `1729 + sha256(...) % 1_000_000`
  - `src/utils/feature_evaluation.py`: `20260518 + sha256(...) % 100_000`
    ("Preserve the historical manuscript seed schedule.")
- The August production runner
  (`src/runners/run_main_manuscript_complete_panel.py`) uses
  `feature_evaluation.evaluate_features` for the four single-matrix slots
  (`burden_only`, `standard_sbs96_id83`), which internally calls the
  20260518-based `stable_seed`. The `nested_oof` import is used only for the
  feature-set path.
- The rerun script imported `nested_oof.stable_seed` and applied it to all
  eight slots, so the four single-matrix slots ran with wrong seeds
  (e.g. 525278 instead of 20284067 for burden_only/linear).
- A full 5-fold rerun of burden_only/linear with the correct seed (20284067)
  reproduced all five archived fold scores bit-for-bit
  (mean 0.24733881249806133). The Python-version hypothesis is refuted.

Fix: `~/workspace/task0_rerun/rerun_table1.py` now imports
`stable_seed as fe_stable_seed` from `utils.feature_evaluation` for the four
single-matrix slots, with asserts against the proven literal seeds:

| slot | seed |
|---|---|
| burden_only/linear | 20284067 |
| burden_only/xgboost | 20287593 |
| standard_sbs96_id83/linear | 20295293 |
| standard_sbs96_id83/xgboost | 20311142 |
| MAF_stack_only/linear | 243603 |
| MAF_stack_only/xgboost | 615371 |
| signatures_plus_MAF_stack/linear | 566310 |
| signatures_plus_MAF_stack/xgboost | 750978 |

Note: archived fold-mean for burden_only/linear is 0.24733881249806133 while
the manuscript prints 0.247320 (delta 1.9e-5); both round to 0.2473, which
satisfies "within rounding". The gap is manuscript-table rounding, not a
rerun issue.

### Recommendation (pre-submission)

`src/utils/nested_oof.py` and `src/utils/feature_evaluation.py` define two
different functions both named `stable_seed`. This duplication caused the bug
above. Before submission, rename one of them or consolidate to a single
definition. Production code was NOT changed for this work package.

### Environment correction (2026-10-13)

The pinned Table 1 environment (xgboost 1.6.2, sklearn 1.7.2, numpy 1.26.4,
pandas 2.3.1, scipy 1.15.3) was rebuilt on Python 3.12.3 at
`~/workspace/task0_rerun/venv312/` after the original `venv310` broke (its
`/opt/python310` interpreter was wiped in a runtime restart). An earlier note
claimed "xgboost 1.6.2 has no cp312 wheels" — that is wrong: xgboost 1.6.2
ships a `py3-none-manylinux2014` wheel that installs on 3.12, but it needs
`setuptools==75.8.0` (not 65.5.0, whose `pkg_resources` uses the
3.12-removed `pkgutil.ImpImporter`) for its dask-availability check to import.
The seed fix was already proven bit-for-bit on Python 3.12, so the interpreter
version does not affect the numbers.

### Reproduction results

| slot | seed | measured | expected | status |
|---|---|---|---|---|
| burden_only/linear | 20284067 | 0.247320 | 0.2473 | MATCH |
| burden_only/xgboost | 20287593 | 0.253957 | 0.2549 | DIFFER (see GPU note) |
| standard_sbs96_id83/linear | 20295293 | running | 0.5263 | — |
| standard_sbs96_id83/xgboost | 20311142 | pending | 0.5381 | — |
| MAF_stack_only/linear | 243603 | pending | 0.5712 | — |
| MAF_stack_only/xgboost | 615371 | pending | 0.6618 | — |
| signatures_plus_MAF_stack/linear | 566310 | pending | 0.6723 | — |
| signatures_plus_MAF_stack/xgboost | 750978 | pending | 0.7378 | — |

### GPU vs CPU note (burden_only/xgboost, 2026-10-13)

The burden_only/xgboost DIFFER (0.253957 vs 0.254945) is an environmental
difference, not a code/seed/data issue:

- Production ran on GPU with `tree_method=gpu_hist` (65s runtime, recorded in
  `xgb_tree_method` column of endpoint_results.csv).
- This machine has no GPU; the pipeline's `_fit_xgb_with_optional_fallback`
  catches the gpu_hist rejection and falls back to `tree_method=hist` on CPU
  (2338s runtime, 36x slower).
- Seed 20287593 is correct: fold 1 selected IDENTICAL hyperparameters in both
  runs (`colsample_bytree=0.7, lr=0.08, max_depth=3, mcw=5.0, alpha=0.05,
  lambda=10.0, subsample=1.0`), and fold-1 scores differ by only 0.0007
  (GPU vs CPU tree building with same params).
- Fold 2 selected different hyperparameters (GPU vs CPU inner-validation
  scores differ), causing the larger fold-2 gap (-0.0088).
- XGBoost 1.6.2 is correct in both. Python 3.10 is not available on this
  machine; the linear exact match on 3.12.3 proves Python version is not
  the cause.

Full investigation: `/tmp/task0_investigation_notes.md` (recreated post-restart).

### Acceptance (Aaron, 2026-10-13)

Task 0 accepted as verified with the GPU caveat documented above:
- burden_only/linear matches bit-for-bit (0.24732045 vs 0.2473).
- burden_only/xgboost differs by 0.001 (0.253957 vs 0.254945) due to GPU (production) vs CPU (this machine) tree building. Seed verified correct; this is environmental, not a reproducibility failure.
- The six remaining slots (standard_sbs96_id83 linear/xgboost, MAF_stack_only linear/xgboost, signatures_plus_MAF_stack linear/xgboost) are not practically rerunnable on this 2-core CPU machine (60+ hours estimated, no GPU; standard_sbs96_id83/linear died silently 4 times).
- All input digests match archived production values. All eight seeds verified via formula. The linear exact match proves the pipeline is reproducible.

Tasks 1, 2, 4 proceed using the verified burden_only OOF predictions plus production OOF predictions for other representations, with the GPU caveat documented for xgboost comparisons.

---

## Task 3 — HRD attribution baselines (accepted as run 2026-10-13)

Three baselines were evaluated on the TCGA HRD cohort (n = 772) in the same
five outer folds as the existing HRD analysis (fold assignments verified
bit-for-bit against the archived `sample_folds` before fitting):

- (a) Signature 3 activity alone, fitted with SigProfilerAssignment 1.1.5
  (COSMIC v3.4 SBS, GRCh37); not raw SBS channel counts.
- (b) BRCA1/2 mutation status alone: sample positive if
  `driver_gene_role_matched_event_log_count__BRCA1 > 0` or the BRCA2 column
  `> 0` (pipeline-native truncating/high-impact proxy; 14/772 positive).
  Documented as a proxy, not a ClinVar pathogenicity call.
- (c) Both together.

Pooled results (`results/tables/task3_hrd_attribution_baselines.csv`):

| baseline | HRD score (Spearman) | HRD33 (AUROC) | HRD42 (AUROC) |
|---|---|---|---|
| (a) sig3_only | -0.067805 | 0.500659 | 0.510397 |
| (b) brca12_only | -0.064635 | 0.507415 | 0.524153 |
| (c) sig3_brca12 | -0.040344 | 0.510914 | 0.533017 |

For comparison, in the same folds: Sig+MAF 0.766645 / 0.900812 / 0.896961
(pooled); patterns-alone (MuAt) 0.504122 / 0.702825 / not run.

Deviation: fixed XGBoost config (max_depth=2, lr=0.05, n_estimators=200)
instead of the nested hyperparameter search, because the nested inner search
is degenerate for these 1–2-feature baselines (inner validation splits with
no positive feature values produce constant predictions and undefined inner
Spearman; the runner raises "No valid nested-search candidate"). Documented
in `results/tables/task3_notes.md`.

POG570 conditional item: frozen-model predictions on POG570 WERE found and
scored — AUROC, calibration slope/intercept, and bootstrap 95% CIs are in
`results/tables/task3_pog570_external_metrics.csv` (n=147 primary breast,
n=570 all POG570).

---

## Task 1 — Gene identity and knowledge controls (2026-10-14)

### Method as run
New script `scripts/task1_gene_knowledge_ablations_v2.py` (the superseded
`task1_gene_knowledge_ablations.py` / `task1_part2_ef_and_summary.py` used
xgboost 3.4.1 and a non-native harness; left untouched, not used).

1. Rebuilt the 8,800-tumour cohort, labels, the 948-col Bio MAF v4 matrix and
   the 182-col signature matrix from `results/tables/` (+ the Task 0 rerun's
   `features_standard_sbs96_id83.csv.gz` for signatures). All three matrices
   digest-verified byte-for-byte against the archived checkpoint manifests
   (`_matrix_digest`: sha256 of shape + float32 bytes), including the
   `signatures_only` and `signatures_plus_v4_full_core*` sets — so the
   Sig+MAF ablations CAN run on this machine (the old note claiming the
   signature matrix was unavailable was wrong).
2. Used the archived outer folds per representation (MAF seed 615371,
   Sig+MAF seed 750978) and the archived per-fold nested-search winners
   (feature set, hyperparameters, m_star) with NO re-tuning. (a) controls are
   the archived OOF predictions verbatim — they match Task 0 by construction
   (MAF 0.661830 vs manuscript 0.6618; Sig+MAF 0.737784 vs 0.7378).
3. (b)-(f) refit per fold with the production final-refit seed schedule
   (split seed + fold*1009 + 99991), pinned Table 1 env
   (~/workspace/task0_rerun/venv312/: xgboost 1.6.2, sklearn 1.7.2,
   numpy 1.26.4, pandas 2.3.1, scipy 1.15.3), tree_method='hist' (production's
   own CPU fallback for gpu_hist), NaN->0 float32, proba row-renormalized as
   in `_coerce_proba`. 50 refits, 1914 s total.
4. (e)/(f) use the same per-fold winner params/m_star as (a) ("no tuning");
   for Sig+MAF they keep the 182-signature block (otherwise they would be
   identical to the MAF-alone runs).

### MECHANICAL feature-classification rules (no case-by-case calls)
- **Gene identity**: column name starts with
  `driver_gene_functional_event_log_count__` or
  `driver_gene_role_matched_event_log_count__` → 802 columns. Verified
  byte-identical to the feature table's `exact_consensus_driver_gene_identity`
  block, and the 401 gene names verified identical to the 401-gene Bio MAF v4
  panel (`proposed_clinical_bio_v4_driver_gene_panel.csv`, all 401 selected).
- **Knowledge**: column name starts with `oncokb_role_` (12), `cancer_hotspot_`
  (4), or `driver_evidence_` (16) → 32 columns, zero overlap with gene identity.

Full column lists: `results/tables/task1_feature_lists.md`
(32 knowledge columns enumerated; 802 gene-identity columns enumerated,
401 genes x 2 prefixes).

### Per-fold feature counts (winner set minus ablated columns)
MAF: f1 full=923 b=911 c=121 d=109 | f2 full=948 b=916 c=146 d=114 |
f3 full=943 b=911 c=141 d=109 | f4 full=948 b=916 c=146 d=114 |
f5 full=948 b=916 c=146 d=114. (e)/(f): 401 each.
Sig+MAF (182 sig cols never removed): f1 full=1130 b=1098 c=328 d=296 |
f2 full=1125 b=1093 c=323 d=291 | f3 1130/1098/328/296 | f4 1125/1093/323/291 |
f5 1125/1093/323/291. (e)/(f): 583 each (182 sig + 401).

### Headline numbers (pooled balanced accuracy; accuracy in CSV)
MAF alone: (a) full 0.6618 | (b) no knowledge 0.6586 (−0.003) |
(c) no gene identity 0.5398 (−0.122) | (d) neither 0.4821 (−0.180) |
(e) binary gene indicators 0.4556 | (f) per-gene driver counts 0.3068.
Sig+MAF: (a) full 0.7378 | (b) no knowledge 0.7409 (+0.003) |
(c) no gene identity 0.6775 (−0.060) | (d) neither 0.6505 (−0.087) |
(e) sig + binary genes 0.6540 | (f) sig + driver counts 0.6140.
Reference: mutation patterns alone 0.5381 (manuscript Table 1).
Per-fold + pooled, balanced accuracy and accuracy:
`results/tables/task1_gene_knowledge_ablations.csv`. OOF predictions
(sample, true_label, pred_label, fold + 20 proba cols):
`results/tables/task1_oof_<maf|sigmaf>_<variant>.csv` (12 files, 8,800 rows each).

### Deviations / corrections
- Fold assignments: used each representation's own archived Task 0 folds,
  not the burden_only fold file named in the task brief — the burden_only
  folds are a different split (seed 20284067), and the (a) control can only
  match Task 0 on the MAF/Sig+MAF folds.
- "397 Bio MAF genes" in the task brief and in
  `scripts/make_new_analysis_figures.py` ("794 columns, 397 genes") is stale:
  the manuscript text and the feature table both have the 401-gene panel /
  802 per-gene columns. Used the mechanical rule on the actual columns
  (401 genes); the manuscript's "397" figure label needs correcting.
- The manuscript's current ablation numbers (0.514 vs 0.538) came from
  different configurations and are superseded by the same-config reruns above.

## Tasks 2, 4, 6 — pending Task 0

Task 1 (gene identity/knowledge controls), Task 2 (inferential procedure),
Task 4 (centre-held-out), and Task 6 (reproducibility deposit) will run after
Task 0 passes, using the Task 0 folds, OOF predictions, and pinned
environment. An earlier exploratory session (2026-10-13, 11:44–14:46 UTC)
produced partial outputs for these tasks using xgboost 3.4.1 and a
non-nested refit harness; those are superseded (backed up to
/tmp/superseded_2026-10-13/) because the Task 1 control did not match Task 0
and the Task 6 lockfile misidentified the pinned environment.
