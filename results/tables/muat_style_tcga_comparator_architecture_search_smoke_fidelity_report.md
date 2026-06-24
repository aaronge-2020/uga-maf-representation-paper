# MuAt-Compatible Fidelity Report

- Official MuAt CLI available: `False`.
- Local run mode: `smoke`.
- TCGA task: `20` tumour types, `60` patients in this run.

## Supported
- TCGA WES SNV/MNV/indel mutation rows from bundled MC3.
- Explicit motif, 1-Mb position, and genic/exonic/strand dictionaries.
- Separate modality embeddings, stacked Q/K/V self-attention, residual/normalization blocks, attention-weighted set pooling, and a 24-dimensional tumour-feature layer.
- Limited architecture search over embedding size, self-attention layer count, and attention-head count before outer-fold refitting.

## Partial
- Motif recovery uses MC3 `CONTEXT` where available; rows without reference context fall back to `N` flanks.
- `cancer_type_top20` uses the fixed manuscript class list and canonical matched-feature folds; `tcga_20_type` uses the MuAt-style top-20-by-count label loader.
- The source paper clearly describes sequence-context encodings, but the exact auxiliary biological annotation vocabulary is not fully specified in the bundled manuscript sources. This implementation uses deterministic MAF/VEP-derived genic, exonic, and strand tokens; broader Bio MAF v4 external-resource features are part of the tabular benchmark rather than this MuAt-compatible event bag.

## Unsupported In Bundled TCGA WES
- PCAWG WGS training, GEL/ICGC/CRC external validation, SV/MEI modalities, and official pretrained checkpoint claims unless the official MuAt CLI/checkpoints are configured and executed.

## MuAt Paper Reference
- TCGA-WES reference: `7352` tumours, `20` tumour types, accuracy `0.641`, top-5 accuracy `0.906`.
- The local result is a same-fold MuAt-compatible comparator, not an official MuAt replication, unless the official CLI status above is true.

## Naming Rule
- Results from this local model must be labelled `MuAt-compatible reimplementation`, not official `MuAt`.
