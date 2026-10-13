# Task 5 — PCAWG audit report

Date: 2026-10-12. Analyst: subagent (code agent), branch `reviewer-analyses-2026-10`.
Scope: reconcile the PCAWG cohort, document column alignment, report per-type
results for the two frozen cancer-type models, and verify the printed per-type
numbers — all from the repository's own scoring outputs. No model was retrained;
no manuscript text was edited.

## 1. Cohort reconciliation: 913 vs 844

**Finding: the cohort total of 913 is correct. The manuscript text's per-type
list omits one type — OV (ovarian), n = 69 — which exactly accounts for the
913 − 844 = 69 gap.**

Per-type counts measured directly from the saved scoring outputs (both models
scored the identical 913 tumours; true labels agree on all 913):

| Cancer type | n | PCAWG source file label |
|---|---|---|
| LIHC | 261 | Liver-HCC |
| PRAD | 180 | Prost-AdenoCA |
| BRCA | 117 | Breast-AdenoCa / Breast-LobularCa |
| ESCA | 97 | Eso-AdenoCa |
| KIRC | 74 | Kidney-RCC |
| SKCM | 70 | Skin-Melanoma |
| OV | 69 | Ovary-AdenoCA |
| STAD | 32 | Stomach-AdenoCa |
| HNSC | 13 | Head-SCC |
| **Total** | **913** | |

Evidence:
- `external_validation/pcawg_transfer/predictions/pcawg_sigmaf_predictions.csv` — 913 rows, per-type counts above.
- `results/external_validation/aaron_email/pcawg/pcawg_913_sample_mapping.csv` — 913 rows, `included=True` on all 913, identical (byte-diff clean) to the pipeline's `raw/pcawg_913_sample_mapping.csv`.
- `external_validation/pcawg_transfer/predictions/pcawg_sigmaf_metrics.json` — `"n_samples": 913`, per-type recall dict with exactly these 9 keys.
- `results/external_validation/aaron_email/pcawg/pcawg_frozen_model_predictions.csv` — 913 rows, same sample ids, same true labels.
- `results/external_validation/aaron_email/pcawg/pcawg_exclusions_by_design.csv` — only a-priori file-level exclusions (Breast-DCIS; Pancan/combined files); 0 sample-level exclusions, consistent with the one-page summary's "913/913 usable across 9 mappings".
- Arithmetic: the 9 types sum to 913. The manuscript text's 8-type list sums to 844; 913 − 844 = 69, and OV is the only type with n = 69. The omitted type is therefore OV.

Note: the frozen Sig+MAF predictions file was scored from the PCAWG
per-type list printed in the manuscript's supplement (Table S2, spectra-only
model) already contains all 9 types including OV (0.507) — the omission is
only in the Sig+MAF per-type list in the text.

## 2. Column alignment: how the model scored the 1,130-column table

**Correction to the task premise: the frozen Sig+MAF cancer-type model itself
has 1,130 features — it is not a 948-feature model.** The 948 figure is the
Bio MAF v4 block alone (the manuscript's original 948-feature matrix, before
the 38 undocumented role-matched columns were removed for the 910-feature
public release). The frozen model's feature schema is:

- 948 Bio MAF v4 features + 182 signature features (96 SBS96 + 83 ID83 + 3
  burden summaries renamed `sig__log10_sbs_burden`, `sig__log10_id_burden`,
  `sig__id_fraction`) = **1,130 features total**.
- Manifest: `results/frozen_models/cancer_type_top20__signatures_plus_MAF_stack/manifest.json`
  (`"n_features": 1130`, preprocessing note: "Apply column reordering and
  NaN->0 fill only", `"nan_fill": 0.0`).

**Alignment rule: by column NAME, against the frozen `feature_order.txt`.**
Assembly (`/home/hatch/workspace/pcawg_sigmaf_transfer/stage6_assemble.py`,
also mirrored at `external_validation/pcawg_transfer/stage6_assemble.py` in
the repo work dir listing):

1. Concatenate the Bio MAF and signature tables on common sample ids.
2. Read the 1,130-name schema from the frozen model's `feature_order.txt`.
3. Compute `missing` (schema names absent from the table) and `extra` (table
   columns absent from the schema); **hard-assert both are empty** — a schema
   mismatch fails loudly instead of scoring.
4. `combined.reindex(columns=order).fillna(0.0).astype(np.float32)` — name-based
   reorder; NaN→0 is the manifest-specified fill.

Measured on the actual scored table:
- Table: `/home/hatch/workspace/pcawg_sigmaf_transfer/features/pcawg_sigmaf_1130_features.csv.gz`
  — shape (913, 1,130).
- **1,130 of 1,130 columns used; 0 dropped; 0 zero-filled** (no NaNs present in
  the scored table, so the NaN→0 fill was a no-op; the table's column order
  already equals the schema order exactly, so the name-based reindex was also
  a no-op — name alignment and positional alignment coincide here).
- Assembly log `/home/hatch/workspace/pcawg_sigmaf_transfer/logs/stage6_assemble.json`:
  `{"n_samples": 913, "n_features": 1130, "missing": [], "extra": []}`.

The patterns-only (spectra) model was aligned the same way: 182-name schema
(`results/frozen_models/cancer_type_top20__standard_sbs96_id83/feature_order.txt`)
vs `results/external_validation/aaron_email/pcawg/pcawg_standard_sbs96_id83_features.csv.gz`
(913 × 182); 182/182 used, 0 dropped, 0 zero-filled. Its `predict.py` states
the same rule explicitly: "Column ORDER does not matter (this script
reorders), but the column NAMES must match feature_order.txt."

Check script + saved output:
- `scripts/task5_column_alignment_check.py`
- `results/tables/task5_column_alignment_check.txt` (OVERALL: PASS)

## 3. Results: per-type recall, bootstrap 95% CIs, Sig+MAF gain

Method: computed from the saved frozen-model predictions (no retraining).
Bootstrap: 10,000 replicates, fixed seed 42, resampling tumours within each
true cancer type with replacement; the gain (Sig+MAF minus patterns-only) is
computed on the same resample each replicate (paired). Full table:
`results/tables/task5_pcawg_audit_per_type.csv` (script: `scripts/task5_pcawg_audit.py`).

| Type | n | Sig+MAF recall (95% CI) | patterns-only recall (95% CI) | gain, Sig+MAF − patterns (95% CI) |
|---|---|---|---|---|
| BRCA | 117 | 0.8462 (0.7778–0.9060) | 0.0769 (0.0342–0.1282) | +0.7692 (0.6923–0.8462) |
| ESCA | 97 | 0.7732 (0.6907–0.8557) | 0.8557 (0.7835–0.9175) | −0.0825 (−0.1753–0.0103) |
| HNSC | 13 | 0.1538 (0.0000–0.3846) | 0.0000 (0.0000–0.0000) | +0.1538 (0.0000–0.3846) |
| KIRC | 74 | 0.8243 (0.7297–0.9054) | 0.0000 (0.0000–0.0000) | +0.8243 (0.7297–0.9054) |
| LIHC | 261 | 0.8276 (0.7816–0.8736) | 0.9732 (0.9540–0.9923) | −0.1456 (−0.1916–−0.1034) |
| OV | 69 | 0.6667 (0.5507–0.7681) | 0.5072 (0.3913–0.6232) | +0.1594 (0.0290–0.3043) |
| PRAD | 180 | 0.0333 (0.0111–0.0611) | 0.0056 (0.0000–0.0167) | +0.0278 (0.0000–0.0556) |
| SKCM | 70 | 0.7286 (0.6286–0.8286) | 0.7143 (0.6000–0.8143) | +0.0143 (0.0000–0.0429) |
| STAD | 32 | 0.1562 (0.0312–0.2812) | 0.1562 (0.0312–0.2812) | +0.0000 (−0.1875–0.1875) |

Context (overall, from the saved metrics): Sig+MAF accuracy 0.6145
(95% CI 0.5827–0.6451, bootstrap n=2000 seed 42 per stage7), balanced accuracy
0.5567; patterns-only accuracy 0.4786, balanced accuracy 0.3655 (bootstrap
n=2000 seed 20260822 per `scripts/run_pcawg_email_external_validation.py`).
The per-type CIs above use the task-specified 10,000 replicates / seed 42 and
supersede the earlier per-type intervals for this audit.

Reproducibility: predictions were already saved for both models, and both
reproduce exactly from the frozen artefacts — rescored 913/913 pred_label
matches for each model, with model.ubj sha256 verified against each
manifest. No rescoring output was needed; nothing was retrained.

## 4. Verification of the printed per-type numbers

| Printed in manuscript | Measured from scoring outputs | Verdict |
|---|---|---|
| SKCM Sig+MAF 0.73 | 0.7286 (51/70) | MATCHES (rounds to 0.73) |
| SKCM patterns alone 0.714 | 0.7143 (50/70) | MATCHES (rounds to 0.714) |
| LIHC 0.83 (Sig+MAF) | 0.8276 (216/261) | MATCHES (rounds to 0.83) |
| ESCA 0.77 (Sig+MAF) | 0.7732 (75/97) | MATCHES (rounds to 0.77) |

All four printed per-type recalls verify against the frozen-model scoring
outputs. (These Sig+MAF per-type figures appear in the Google Doc; the repo's
`submission_manuscript.docx` snapshot only prints the spectra-only per-type
list, which also matches its outputs — e.g. LIHC 0.973, ESCA 0.856,
SKCM 0.714, OV 0.507.)

## Deliverables and evidence paths

- Audit CSV: `results/tables/task5_pcawg_audit_per_type.csv`
- This report: `results/tables/task5_pcawg_audit_report.md`
- Column-alignment check script: `scripts/task5_column_alignment_check.py`
- Column-alignment saved output: `results/tables/task5_column_alignment_check.txt`
- Audit analysis script: `scripts/task5_pcawg_audit.py`
- Saved predictions (pre-existing, verified reproducible):
  `external_validation/pcawg_transfer/predictions/pcawg_sigmaf_predictions.csv` (+ `pcawg_sigmaf_metrics.json`),
  `results/external_validation/aaron_email/pcawg/pcawg_frozen_model_predictions.csv`
- Sample mapping: `results/external_validation/aaron_email/pcawg/pcawg_913_sample_mapping.csv`
- Frozen models: `results/frozen_models/cancer_type_top20__signatures_plus_MAF_stack/`,
  `results/frozen_models/cancer_type_top20__standard_sbs96_id83/`

## Deviations and blockers

- **Deviation (task premise):** the task describes "the 948-feature model"
  scoring "a table with 1,130 columns." The frozen Sig+MAF model has 1,130
  features; 948 is the Bio MAF v4 block alone. The scored table has exactly the
  1,130 schema columns (0 missing, 0 extra). No correction to the outputs was
  needed — only the premise.
- **Deviation (bootstrap spec):** the pre-existing metrics used 2,000
  replicates (Sig+MAF: seed 42; spectra-only: seed 20260822). This audit
  recomputed per-type CIs at the task-specified 10,000 replicates, fixed seed 42.
- **No blockers.** Predictions were saved for both models and both reproduce
  bit-for-bit from the frozen models, so no rescoring files were required.
- The Sig+MAF per-type printed numbers were checked against the values stated
  in the task; the repo's docx snapshot does not contain that per-type list
  (it lives in the Google Doc, which was not accessed).
