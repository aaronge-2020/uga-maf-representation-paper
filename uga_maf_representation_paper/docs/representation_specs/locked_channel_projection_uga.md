# Locked Channel Projection UGA

## Inputs

- Standard SBS96 and ID83 feature matrices.
- Locked UGA atlas/spec files vendored under `src/legacy_source/cgr_validation/uga_atlas`.
- Genome-wide context atlas `EXP022_atlas_genome_wide_45mer_universal_d22.json`.

## Construction

1. Map each SBS96 or ID83 channel to a locked UGA event vector.
2. SBS/DBS channels use `master_spec_sbs_dbs_d10_dp5`.
3. ID83 channels use `id83_payload_only_d10_dp5`.
4. Project each patient's channel counts into UGA space by count-weighted averaging over the locked channel basis.
5. Benchmarks evaluate pooled SBS+ID UGA means and, where present, separate SBS and ID UGA blocks.

## Dimensionality

The locked d10/dp5 masked UGA vector is 70D: 40 context coordinates plus two 15D masked payload blocks. A pooled SBS+ID mean is 70D; separate SBS and ID blocks are 140D before burden covariates.

## Preserved, Added, Lost

- Preserved exactly: total channel mass used in the weighted projection and optional burden covariates.
- Added: a common coordinate geometry shared across mutation classes.
- Lost or smoothed: individual SBS96/ID83 channel identities are averaged into a lower-dimensional mean vector.

## Leakage Controls

The UGA basis is locked before endpoint modeling. Projection does not use labels, test folds, or endpoint-specific tuning.
