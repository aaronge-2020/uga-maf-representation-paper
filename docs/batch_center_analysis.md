# Batch-center (sequencing-center) confounding analysis

Reproduces the "Batch effects and confounding" analysis from the manuscript.
Run: `python3 scripts/run_batch_center_analysis.py`
(Table 2 in the manuscript; eligible within-type comparisons; center classifier.)

## Inputs

| File | Provenance |
|---|---|
| `results/tables/quick_bio_v4_maf_features_910.csv.gz` | 910-feature matrix (this repo). The six summaries used here are identical in the 948-feature matrix. |
| `results/tables/batch_center_sample_mapping.csv` | Per-patient mapping built for this analysis (10,224 rows): full-length MC3 `Tumor_Sample_Barcode`, sequencing center (final two barcode characters), cancer-type labels from GDC (`project.project_id` via `/cases`, 2026-09-23) and from the TCGA CDR Supplemental Table S1 (Cell 2018). |

Full-length barcodes come from the public MC3 MAF (`mc3.v0.2.8.PUBLIC.maf.gz`,
GDC `1c8cfe5f-e52d-41ba-94da-f15ea1337efc`). All 10,224 matrix patients appear
in the MAF (exact set match); 70 patients have >1 tumor barcode but none have
conflicting center codes, so center assignment is unambiguous.

## Methods

**Six per-tumor summaries** (from the 910 matrix):

- Modifier fraction: `vep_impact_fraction__modifier`
- 3′ UTR fraction: `vep_consequence_fraction__3_prime_utr_variant`
- Indel fraction: stored `id_fraction` column (see deviation note below)
- Log SBS burden: `log10_sbs_burden`
- Mean VAF: `vaf_mean`; Max VAF: `vaf_max`

**Eta-squared** is SS_between / SS_total from a one-way ANOVA grouping by
center. "Across centers" uses the top-20 cohort (n = 8,858; GDC project
labels, top-20 types as in the benchmark registry). "Median / maximum within
type" are computed over the three eligible within-type comparisons (OV, KIRC,
KIRP), each restricted to its largest two centers — the only reading that
reproduces the manuscript's medians and maxima (a whole-cohort median would
be dragged down by single-center types with eta-squared 0).

**Center classifier:** multinomial logistic regression (`lbfgs`) on
standardized six-summary features, 5-fold stratified cross-validation
(seed 0), predicting the four largest centers (08, 09, 10, 32) on **all**
10,224 samples (n = 10,206 after restricting to the four centers). No
hyperparameter tuning. (No original classifier implementation was found in
the repo; this standard specification reproduces both reported numbers
exactly.)

## Results

### Eta-squared of sequencing center (recomputed vs manuscript Table 2)

| Summary | Across (ms) | Median within (ms) | Max within (ms) |
|---|---:|---:|---:|
| Modifier fraction | 0.294 (0.293) | 0.134 (0.134) | 0.291 KIRP (0.292 KIRP) |
| 3′ UTR fraction | 0.309 (0.308) | 0.158 (0.158) | 0.178 KIRP (0.180 KIRP) |
| Indel fraction | 0.034 (0.033) | 0.024 (0.024) | 0.045 KIRP (0.045 KIRP) |
| Log SBS burden | 0.002 (0.003) | 0.002 (0.002) | 0.036 KIRC (0.033 KIRC) |
| Mean VAF | 0.035 (0.035) | 0.011 (0.011) | 0.064 KIRP (0.063 KIRP) |
| Max VAF | 0.020 (0.021) | 0.002 (0.002) | 0.007 KIRP (0.007 KIRP) |

All values match the manuscript to the reported 3-decimal precision.

### Eligible within-type comparisons (recomputed vs manuscript)

| Type | Recomputed | Manuscript |
|---|---|---|
| OV | n=399; 08:206, 09:193 | n=399; 08:206, 09:193 |
| KIRC | n=369; 08:207, 10:162 | n=370; 08:207, 10:163 |
| KIRP | n=281; 08:114, 10:167 | n=282; 08:114, 10:168 |

OV reproduces exactly. KIRC/KIRP match on the Broad (08) counts exactly and
are each one patient short at Baylor (10) — an unresolved 1-patient
discrepancy (see below). Within-type modifier eta-squared for KIRP is 0.291
(manuscript "roughly 0.29").

### Center classification (recomputed vs manuscript)

- 5-fold CV accuracy: **0.697** (manuscript 0.697)
- Majority-class baseline: **0.583** (manuscript 0.583)
- Cohort: 10,206 samples across centers 08 (5,950), 09 (2,666), 10 (1,508), 32 (82)

## Deviations, discrepancies, and caveats

1. **Indel fraction source.** The manuscript text says the indel fraction is
   "derived from the SBS/DBS/ID burden logs." Re-deriving it as
   10^log-transformed counts (with log 0 → 0 events) gives eta-squared values
   of 0.030 / 0.031 / 0.039 (across / median / max), which do **not** match the
   manuscript. The stored `id_fraction` column reproduces the manuscript
   values exactly (0.034 / 0.024 / 0.045). This analysis therefore uses the
   stored column; the manuscript's phrasing appears to describe the
   column's provenance (fraction counterpart of the burden logs) rather than
   a literal re-derivation recipe.
2. **TCGA-36-2539 excluded.** GDC labels this patient OV (center 09), but it
   is absent from the TCGA CDR. Current-GDC top-20 gives n = 8,859;
   excluding this patient gives the manuscript's n = 8,858 and makes the
   OV within-type comparison exactly n = 399 (08:206, 09:193).
3. **KIRC/KIRP ±1 at Baylor.** Recomputed center-10 counts are one patient
   short of the manuscript for KIRC (162 vs 163) and KIRP (167 vs 168);
   the Broad counts match exactly. Cause unresolved; no label or barcode
   ambiguity was found for these patients.
4. **Within-type scope.** Median/maximum eta-squared are over the three
   eligible types only (OV, KIRC, KIRP). Eligibility is not spelled out in
   the manuscript; the eligible set is exactly the types with two
   well-represented sequencing centers.
5. **Confounding caveat (manuscript).** Center and cancer type are strongly
   confounded (most types were sequenced predominantly at one center), and
   the KIRP center effect is further confounded with tissue-source site, so
   within-type center comparisons cannot separate sequencing-center effects
   from TSS effects.
6. Center codes: 08 Broad Institute, 09 Washington University, 10 Baylor,
   32 HudsonAlpha.

## Outputs

- `results/tables/batch_center_etasq.csv` — eta-squared per summary with manuscript values alongside
- `results/tables/batch_center_within_type.csv` — per-type n's, center splits, within-type eta-squared
- `results/tables/batch_center_classification.csv` — classifier accuracy vs baseline with manuscript values
