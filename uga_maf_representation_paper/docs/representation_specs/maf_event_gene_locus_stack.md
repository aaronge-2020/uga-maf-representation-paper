# MAF Event-Gene/Locus Stack

## Inputs

- MC3 MAF file `mc3.v0.2.8.PUBLIC.maf.gz`.
- Standard MC3 SBS96+ID83 feature matrix for patient alignment and baseline comparison.
- MC3 clinical/driver labels and TCGA-BRCA HRD cohort labels.

## Construction

1. Read the MAF in chunks and normalize patient identifiers to 12-character TCGA barcodes.
2. Remove rows for the target leakage gene `KMT2C` before event-gene and locus feature construction.
3. Build gene, damaging-gene, pathway, pathway-impact, variant-class, variant-type, consequence, and VAF summary blocks.
4. Build locus-topography blocks from chromosome, chromosome-modality, variant-class, megabase bins, and density summaries using `src/utils/maf_features.py`.
5. Transform selected sparse count blocks with log1p and low-rank SVD where specified by the candidate model.
6. Concatenate Standard SBS96+ID83 with the retained event-gene/locus multiscale blocks for `id_plus_best_gene_locus_multiscale_stack`.

## Dimensionality

Dimensionality is candidate-specific and is written to `feature_dimension_audit.csv` and `model_manifest.csv` during regeneration. Locus `top512` blocks contain at most 512 high-variance megabase-bin features before downstream concatenation.

## Preserved, Added, Lost

- Preserved: Standard SBS96+ID83 baseline features when included in the candidate.
- Added: gene, pathway, consequence, damaging status, broad locus topology, mutation class, and VAF summaries.
- Lost or smoothed: exact event order and exact base-level coordinates are collapsed into aggregate feature blocks; SVD blocks further smooth sparse gene signals.

## Leakage Controls

`KMT2C` rows are excluded before MAF event-gene and locus features are built, protecting the LUAD KMT2C mutation endpoint from direct target leakage. Feature construction is performed before folds and does not use endpoint labels.
