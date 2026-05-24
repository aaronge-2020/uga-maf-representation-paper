# MC3 Source Package

## Purpose

This folder contains the MC3 source inputs used by the locked unified UGA benchmark scripts. The retained MC3 analyses do not read from superseded experiment folders.

## File Inventory

| File or folder | Purpose |
|---|---|
| `features/features_burden_only.csv` | Burden covariates used in MC3 driver and clinical endpoint models. |
| `features/features_standard_sbs96.csv.gz` | Standard SBS96 patient feature matrix. |
| `features/features_standard_id83.csv.gz` | Standard ID83 patient feature matrix. |
| `features/features_standard_sbs96_id83.csv.gz` | Standard SBS96+ID83 patient feature matrix. |
| `biology_labels.csv` | Clinical and biology endpoint labels used by the MC3 clinical endpoint panel. |
| `gene_group_labels.csv` | Gene-group labels retained from the MC3 direct-feature source package. |
| `driver_gene_labels_functional.csv` | Functional driver-gene mutation labels derived from the MC3 MAF. |
| `raw/mc3.v0.2.8.PUBLIC.maf.gz` | MC3 GRCh37 MAF used to derive functional driver-gene labels. |
| `raw/TCGA-CDR-SupplementalTableS1.xlsx` | TCGA CDR clinical table used for cancer type and clinical endpoint derivation. |
| `raw/clinical_PANCAN_patient_with_followup.tsv` | PanCanAtlas follow-up table used in clinical label derivation. |
| `raw/TCGA_mastercalls.abs_tables_JSedit.fixed.txt` | ABSOLUTE purity table used in clinical label derivation. |

## Reproducibility Notes

The retained MC3 scripts construct locked UGA features from the standard SBS96 and ID83 feature matrices in this folder using `master_spec_sbs_dbs_d10_dp5` and `id83_payload_only_d10_dp5`. The retained MC3 feature cache does not contain DBS78 matrices, and the retained MC3 MAF contains no explicit DNP records or direct two-base REF-to-ALT substitution records. MC3 retained analyses therefore evaluate SBS96 and ID83 only. Adjacent-SNV DBS reconstruction is not used in the retained final benchmark. Kucab is the retained all-modality SBS96+DBS78+ID83 benchmark because explicit DBS events are available there.
