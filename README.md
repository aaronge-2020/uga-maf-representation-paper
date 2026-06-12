# UGA/MAF Representation Production Pipeline

This repository contains the definitive, end-to-end manuscript pipeline: production source code, active configuration, canonical result artifacts, and strict validation checks. The large data bundle is expected next to this checkout as `../github_exports_datasets`.

## Quick Start

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_manuscript.py --datasets-dir ../github_exports_datasets --strict
```

The strict command validates repository files, manuscript artifacts, dataset-bundle checksums, endpoint guards, Bio MAF v4 artifacts, and source scans. It writes the latest report to `results/logs/strict_validation_latest.json`.

## Production Pipeline

Run the complete active pipeline:

```bash
python src/run_all_experiments.py --config config/experiment_settings.strict_no_leakage.yaml --paths config/paths.yaml
```

The active experiment graph is intentionally small:

- `main_manuscript_complete_panel`: nested out-of-fold evaluation for burden, mutational signatures, Bio MAF v4, and signatures + Bio MAF v4 across the manuscript endpoints.
- `muat_style_tcga_comparator`: MuAt-compatible event-bag comparator on configured TCGA endpoints.
- `src/utils/make_all_figures.py`: canonical manuscript tables, figures, supplements, plot data, and text.

For a preflight without expensive jobs:

```bash
python src/run_all_experiments.py --config config/experiment_settings.strict_no_leakage.yaml --paths config/paths.yaml --dry-run
```

Reviewer workflow wrappers are still available:

```bash
python scripts/reviewer_workflow.py start --datasets-dir ../github_exports_datasets
python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets --dry-run
python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets
```

## Layout

- `src/`: production runners and utilities.
- `config/`: active path/config profiles and Bio MAF v4 resources.
- `data/`: small bundled references.
- `results/tables/`: source tables consumed by manuscript generation.
- `results/manuscript/`: canonical manuscript tables, figures, text, supplements, and interpretability outputs.
- `docs/`: reviewer notes, upload checklist, and QC reports.
- `manifests/`: reproducibility, data, large-asset, dataset, and manuscript manifests.

## Dataset Bundle

Required sibling bundle paths:

- `../github_exports_datasets/datasets/tcga_mc3/`
- `../github_exports_datasets/datasets/tcga_brca_hrd/`
- `../github_exports_datasets/datasets/kucab_mendeley/`
- `../github_exports_datasets/datasets/kucab_mendeley/raw_mutation_tables/`
- `../github_exports_datasets/datasets/pcawg_pancan/`
- `../github_exports_datasets/references/grch37/`
- `../github_exports_datasets/resources/bio_maf_v4/`
- `../github_exports_datasets/resources/signatures_cosmic/`

Before publishing the code checkout, optional upload prep removes accidental large duplicates after validating the dataset bundle:

```bash
python scripts/reproduce_manuscript.py --prepare-github-upload --datasets-dir ../github_exports_datasets
```
