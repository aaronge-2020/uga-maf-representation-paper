# MuAt-Compatible Fidelity Report

- Official MuAt CLI available: `False`.
- Local run mode: `full_local`.
- Active manuscript task scope: TCGA/MC3 HRD endpoints only, `HRD_Score` and `hrd_binary_33`.

## Supported

- TCGA WES SNV/MNV/indel mutation rows from bundled MC3.
- TCGA-BRCA HRD labels from the bundled HRD cohort inputs.
- Explicit motif, 1-Mb position, and genic/exonic/strand dictionaries.
- Separate modality embeddings, Q/K/V self-attention, skip connection, normalization, average pooling, and a 24-dimensional tumour-feature layer.
- Held-out outer folds matched to the corresponding manuscript tabular benchmark splits.

## Partial

- Motif recovery uses MC3 `CONTEXT` where available; rows without reference context fall back to `N` flanks.
- This is a local MuAt-compatible reimplementation, not an official pretrained MuAt checkpoint evaluation.

## Unsupported In Bundled TCGA WES

- PCAWG WGS training, GEL/ICGC/CRC external validation, SV/MEI modalities, and official pretrained checkpoint claims unless the official MuAt CLI/checkpoints are configured and executed.
- Active manuscript MuAt comparator claims for cancer type, KUCAB damage class, HRD24, HRD42, OS, PFI, or other endpoints.

## Naming Rule

- Results from this local model must be labelled `MuAt-compatible reimplementation`, not official `MuAt`.
