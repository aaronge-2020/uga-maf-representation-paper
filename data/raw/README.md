# Raw Data Placement

Place raw or restricted data here only if your local policy allows it. The
bundle does not require raw files to be committed. The default `config/paths.yaml`
expects subdirectories with these names:

- `mc3_source/`: MC3 raw files, labels, and small standard feature caches.
- `tcga_brca_hrd/`: TCGA-BRCA HRD cohort and label assets.
- `kucab/`: Kucab mutation tables and treatment metadata.
- `pancan_pcawg_2020/`: PCAWG/cBioPortal signature attribution inputs.
- `GRCH37/`: optional GRCh37 FASTA assets for workflows that require reference context.

Edit `config/paths.yaml` if your files live elsewhere.

