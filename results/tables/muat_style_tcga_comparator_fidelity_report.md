# MuAt-Compatible Fidelity Report

- Official MuAt CLI available: `False`.
- Local run mode: `smoke`.
- TCGA task: `20` tumour types, `60` patients in this run.

## Supported
- TCGA WES SNV/MNV/indel mutation rows from bundled MC3.
- Explicit motif, 1-Mb position, and genic/exonic/strand dictionaries.
- Separate modality embeddings, Q/K/V self-attention, skip connection, normalization, average pooling, and a 24-dimensional tumour-feature layer.

## Partial
- Motif recovery uses MC3 `CONTEXT` where available; rows without reference context fall back to `N` flanks.
- The bundled clinical/MAF assets contain more than 20 eligible TCGA acronyms, so the paper-facing task uses the top 20 eligible types by MC3 patient count and audits the rest.

## Unsupported In Bundled TCGA WES
- PCAWG WGS training, GEL/ICGC/CRC external validation, SV/MEI modalities, and official pretrained checkpoint claims unless the official MuAt CLI/checkpoints are configured and executed.

## Naming Rule
- Results from this local model must be labelled `MuAt-compatible reimplementation`, not official `MuAt`.
