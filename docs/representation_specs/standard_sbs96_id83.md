# Standard SBS96 + ID83

## Inputs

- Per-patient SBS96 count or frequency table.
- Per-patient ID83 count or frequency table.
- Optional burden covariates used by the benchmark runners.

## Construction

1. Load `features_standard_sbs96.csv.gz` and `features_standard_id83.csv.gz`.
2. Align rows by patient identifier.
3. Concatenate SBS96 and ID83 columns to form `features_standard_sbs96_id83.csv.gz`.
4. Fill missing channels with zero.

## Dimensionality

179 mutation-channel features: 96 SBS channels plus 83 indel channels. Some models also append shared burden covariates outside this representation.

## Preserved, Added, Lost

- Preserved exactly: channel-level SBS96 and ID83 counts or normalized frequencies.
- Added: no event-level information beyond optional burden covariates.
- Lost: gene, pathway, genomic coordinate, allele fraction, and event-level locus context.

## Leakage Controls

The representation is endpoint-agnostic. Feature construction uses mutation channels only and does not use labels, folds, or endpoint definitions.
