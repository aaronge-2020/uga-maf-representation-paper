# GitHub and Hugging Face Upload Checklist

This checklist assumes commands are run from the repository root.

## Required preflight

1. Run `python scripts/reviewer_workflow.py check --datasets-dir ../github_exports_datasets`.
2. Confirm `results/logs/reviewer_setup_check.json` has no blocking missing files or package issues.
3. Run `python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets --dry-run`.
4. Confirm the strict portion reports `status: passed`.
5. Confirm `results/logs/strict_validation_latest.json` reports `outside_root_reads: 0`.
6. Confirm `manifests/manuscript_artifact_manifest.csv`, `manifests/dataset_assets_manifest.csv`, `manifests/huggingface_asset_manifest.csv`, `DATA_MANIFEST.csv`, and `LFS_MANIFEST.csv` are current.

## GitHub repository

1. Commit active source, configs, docs, manifests, environment files, scripts, small data/resources, and manuscript outputs.
2. Commit only active production code, configuration, documentation, manifests, and release artifacts; do not commit obsolete payload mirrors, scratch outputs, generated caches, checkpoints, or local runtime folders ignored by `.gitignore`.
3. Review `.gitattributes` and `.gitignore` before upload. Large binary or compressed assets listed in `manifests/large_assets_manifest.csv` should remain in the sibling dataset folder or use Git LFS, not ordinary Git history.
4. Run `python scripts/reproduce_manuscript.py --prepare-github-upload --datasets-dir ../github_exports_datasets` before final GitHub status review.
5. After clone, download the companion dataset and run `python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets --dry-run` before making scientific claims from the checkout.

## Hugging Face or Git LFS assets

The authoritative dataset list is `manifests/dataset_assets_manifest.csv`, mirrored at `../github_exports_datasets/dataset_assets_manifest.csv`. The reader-facing upload manifest is `manifests/huggingface_asset_manifest.csv`. The current sibling bundle contains 1,257 manifest payloads and 48 active dataset-backed destinations; `manifests/large_assets_manifest.csv` is a compact large-file planning view. Upload `../github_exports_datasets/` as the companion dataset, or keep the active large files in Git LFS.

The bundle uses readable paths instead of an opaque SHA-256 tree:

- `datasets/tcga_mc3/`, `datasets/tcga_brca_hrd/`, `datasets/kucab_mendeley/`, `datasets/pcawg_pancan/`, and other public dataset folders.
- `references/grch37/` and `references/grch38/`.
- `resources/bio_maf_v4/` and `resources/signatures_cosmic/`.
- `results/manuscript_large/`, `results/large_tables/`, and `caches/` for generated assets that are read from the dataset bundle.

Current asset groups:

| Group | Count | Notes |
|---|---:|---|
| TCGA MC3 public mutation calls | 1 | Public MAF source input. |
| GRCh37 reference FASTA and metadata | 2 | Plain and gzipped FASTA assets. |
| MuAt-compatible `cancer_type_top20` fold checkpoint cache | 35 | Validated vendored fold outputs used by release validation. |
| MuAt-compatible `cancer_type_top20` event-token cache | 1 | Dataset-backed comparator event cache. |
| Bundled manuscript/result inputs | 5 | Prediction tables and comparator event tables. |
| Optional external-project datasets | 514 | Includes discovered raw/downloaded resources retained outside ordinary Git history. |

## Dataset Bundle Use

1. Download or keep the companion dataset as `../github_exports_datasets/`.
2. Verify the dataset folder with `python scripts/reproduce_manuscript.py --check-datasets --datasets-dir ../github_exports_datasets`.
3. Run the reviewer setup check with `python scripts/reviewer_workflow.py check --datasets-dir ../github_exports_datasets`.
4. Validate and dry run with `python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets --dry-run`.
5. Treat any missing, stale, or checksum-mismatched asset as a failed dataset-bundle validation.
