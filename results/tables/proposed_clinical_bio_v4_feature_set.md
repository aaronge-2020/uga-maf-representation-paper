# Proposed Clinical Bio MAF v4 Feature Set

Bio MAF v4 keeps mutational signatures out of the Bio MAF matrix. Signatures remain a separate input matrix for the signatures + biology model.

## Feature counts

| Block | Count | Status | Why included |
|---|---:|---|---|
| v3_core_compact_biology | 121 | core | Retains standard VEP, VAF, OncoKB role, and residual aggregates. |
| external_driver_evidence_tier_aggregates | 16 | core | Adds graded external support without exact gene identity. |
| exact_consensus_driver_gene_identity | 802 | core selectable block | Recovers exact identity for 401 externally supported cancer genes. |
| cancer_hotspot_summary | 4 | core | Adds exact recurrent hotspot signal. |
| optional_burden_and_event_composition_controls | 5 | optional control | Sensitivity controls; not part of strict core Bio MAF v4. |

Core Bio MAF v4 feature count: **943**.
Full Bio MAF v4 universe including optional controls: **948**.
Exact driver-gene block: **401 genes x 2 features = 802 features**.

## Exact driver gene selection rule

A gene enters the exact-gene block if it has either:

- support from at least 3 of 4 external evidence channels: OncoKB annotated, COSMIC CGC v99 flag in the OncoKB cancer gene list, IntOGen 2024 driver compendium, Vogelstein 2013; or
- Cancer Hotspots v3 recurrent hotspot evidence.

This rule selects genes before cross-validation and does not inspect any endpoint labels.

## New v4 blocks

### External Driver Evidence Tier Aggregates

Sixteen features summarize mutations in genes supported by 1, 2, 3, or 4 external evidence channels. Each tier gets four summaries: high/moderate-impact mutation count, high-impact mutation count, number of distinct mutated genes, and maximum VAF.

### Exact Consensus Driver Gene Identity

For each selected gene, v4 adds two features:

- `driver_gene_functional_event_log_count__GENE`: protein-affecting mutation signal in that exact gene.
- `driver_gene_role_matched_event_log_count__GENE`: mutation signal that better matches the known mechanism of that gene, such as loss-of-function for tumor suppressors or hotspot evidence for oncogenes.

### Cancer Hotspot Summary

Four features summarize exact Cancer Hotspots v3 matches: exact variant count, residue-level count, number of hotspot-hit genes, and maximum hotspot VAF.

## Nested-CV candidate blocks

Recommended inner-loop feature-set candidates:

- `v4_compact_biology_only`: 121 features
- `v4_compact_plus_driver_evidence_tiers`: 137 features
- `v4_compact_plus_hotspot_summary`: 125 features
- `v4_compact_plus_exact_driver_genes`: 923 features
- `v4_full_core`: 943 features
- `v4_full_core_plus_optional_controls`: 948 features

The exact-gene block is therefore allowed to compete with the compact aggregate block inside nested CV; it is not chosen using held-out data.

## Source files

- OncoKB Cancer Gene List: `config/feature_resources/oncokb_cancer_genes.tsv`
- IntOGen driver compendium: `config/feature_resources/clinical_bio_v4_sources/2024-06-18_IntOGen-Drivers/Compendium_Cancer_Genes.tsv`
- Cancer Hotspots v3: `config/feature_resources/clinical_bio_v4_sources/v3_multi_type_residue.txt`, `v3_multi_type_variant_file.txt`, `hotspots_v3.xlsx`
- COSMIC CGC: direct current COSMIC download is login-gated; v4 uses the `COSMIC CGC (v99)` flag in the downloaded OncoKB Cancer Gene List as the CGC evidence channel until direct COSMIC credentials are available.


