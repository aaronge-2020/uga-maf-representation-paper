# Provenance: oncokb_cancer_genes.tsv

## What this file is
Per-gene OncoKB role annotations (oncogene / tumor suppressor) used by the
Bio MAF feature builder (`src/utils/bio_maf_base_features.py::load_oncokb_roles`)
to assign each mutated gene to a role group for the 16 `oncokb_role_*`
aggregate features.

## Source
Downloaded 2026-09-23 from the public OncoKB API, the same source URL recorded
in the previous stub version of this file:

    https://public.api.oncokb.org/api/v1/utils/cancerGeneList

No API key required. Response is JSON; `geneType` values were mapped to the
`oncogene`/`tsg` boolean columns as follows:

- `ONCOGENE` -> oncogene=True
- `TSG` -> tsg=True
- `ONCOGENE_AND_TSG` -> both True
- `INSUFFICIENT_EVIDENCE`, `NEITHER`, or null -> both False

This yields 1,246 genes: 445 oncogene, 340 tumor suppressor, 68 both,
393 without a positive role assignment.

## Consistency checks performed 2026-09-23
- All 401 genes of the analysis driver panel
  (`results/tables/proposed_clinical_bio_v4_driver_gene_panel.csv`) are present
  after the pipeline's gene-symbol normalization, and their OncoKB gene types
  agree 401/401 with the panel's documented `oncokb_gene_type` values.
- Rebuilding the 16 `oncokb_role_*` columns from the MC3 MAF with this file
  (script: `~/workspace/analyses/verify_910/rebuild_role_cols_check.py`,
  standalone re-run 2026-09-23 with identical role logic) does NOT reproduce the
  published feature matrix (`results/tables/quick_bio_v4_maf_features.csv.gz`)
  bit-identically: 15 of the 16 columns differ, and exactly one column matches
  (`oncokb_role_max_vaf_high_or_moderate_impact__tumor_suppressor`, 0 differing
  samples). Per-column Pearson correlations between rebuilt and published columns
  range from 0.95 to 1.00 (for example, r = 0.9590 for the oncogene
  high-or-moderate-impact log-count column, r = 0.9599 for the tumor-suppressor
  equivalent; maximum-VAF columns correlate at 0.99 to 1.00). The previously
  quoted r = 0.96 described that one oncogene count column only, not all 16
  columns combined.
- For that one oncogene high-or-moderate-impact count column, the rebuilt values
  never fall below the published values across all 10,224 samples (rebuilt mean
  1.31 vs published mean 0.75; 8,588 samples differ upward), consistent with this
  file's role assignments being a superset of the historical ones for that
  column. This superset property was checked for that column only and is not
  established for the other 15 columns.

## Known limitation: version drift
OncoKB updates its gene list continuously and the API is not versioned. The
published feature matrix was built with an earlier OncoKB snapshot. A from-scratch
rebuild of the 16 role-aggregate columns with this file produces columns that
closely track (per-column r 0.95 to 1.00) but do not exactly equal the published
ones (15 of 16 differ). The exact historical snapshot is not recoverable
from public sources (no git history, no web archive of the API endpoint).

Consequence: a from-scratch rebuild of the feature matrix with this file will
produce role-aggregate columns that closely track but do not exactly equal the
published ones. The published matrix
(`results/tables/quick_bio_v4_maf_features.csv.gz`) remains the artifact of
record for every number reported in the manuscript. If bit-identical
reproduction of the original role columns is ever required, the exact
historical OncoKB snapshot must be obtained from the analyst who built the
published matrix (Vijay Vankadaru).

## Previous state
Before 2026-09-23 this file was a stub: all 1,240 genes had oncogene=False and
tsg=False (including KRAS, TP53, PIK3CA), which silently zeroed every
`oncokb_role_*` aggregate in any from-scratch rebuild. A backup of the stub is
kept at `/tmp/oncokb_cancer_genes_stub_backup.tsv` (ephemeral).
