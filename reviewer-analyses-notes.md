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
