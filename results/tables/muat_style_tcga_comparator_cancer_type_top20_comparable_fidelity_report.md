# MuAt-Compatible Fidelity Report

- Official MuAt CLI available: `False`.
- Local run mode: `full_local`.
- TCGA task: `20` tumour types, `8800` patients in this run.

## Supported
- TCGA WES SNV/MNV/indel mutation rows from bundled MC3.
- Explicit motif, 1-Mb position, and genic/exonic/strand dictionaries.
- Separate modality embeddings, Q/K/V self-attention, skip connection, normalization, average pooling, and a 24-dimensional tumour-feature layer.

## Partial
- Motif recovery uses MC3 `CONTEXT` where available; rows without reference context fall back to `N` flanks.
- `cancer_type_top20` uses the fixed manuscript class list and canonical matched-feature folds; `tcga_20_type` uses the MuAt-style top-20-by-count label loader.

## Unsupported In Bundled TCGA WES
- PCAWG WGS training, GEL/ICGC/CRC external validation, SV/MEI modalities, and official pretrained checkpoint claims unless the official MuAt CLI/checkpoints are configured and executed.

## MuAt Paper Reference
- TCGA-WES reference: `7352` tumours, `20` tumour types, accuracy `0.641`, top-5 accuracy `0.906`.
- The local result is a same-fold MuAt-compatible comparator, not an official MuAt replication, unless the official CLI status above is true.

## Naming Rule
- Results from this local model must be labelled `MuAt-compatible reimplementation`, not official `MuAt`.
