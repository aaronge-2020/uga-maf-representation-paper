# Reproducibility

This repository is a standalone, curated manuscript export. Run commands from the repository root.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

Use an equivalent Python environment if preferred. Active code paths resolve through `config/paths.yaml` and `src/utils/config.py`; raw/large datasets default to the sibling `../github_exports_datasets` bundle through the `${datasets_root}` placeholder.

For a reviewer-oriented setup report that checks packages, required repo files, the full companion dataset bundle, Node.js, and dataset-backed asset state:

```bash
python scripts/reviewer_workflow.py check --datasets-dir ../github_exports_datasets
```

The command prints a short human-readable checklist and writes the full report to `results/logs/reviewer_setup_check.json`. It includes exact fix commands for missing packages, files, or dataset assets. Add `--json` to print the full report in the terminal.

## Strict Manuscript Validation

```bash
python scripts/reproduce_manuscript.py --datasets-dir ../github_exports_datasets --strict
```

The strict command validates every active file under `results/manuscript`, uses large active artifacts directly from the readable sibling bundle when they are intentionally absent from Git, checks Bio MAF v4 resources, confirms top-20 and HRD binary endpoints, verifies the Bio MAF v4 S4/S5/S6 outputs, scans active configs/source/docs/manifests for disallowed external path dependencies, and writes repo plus dataset read audits under `results/logs/`.

To check the sibling dataset folder without copying files back into the checkout:

```bash
python scripts/reproduce_manuscript.py --check-datasets --datasets-dir ../github_exports_datasets
```

## Full Regeneration

The one-line reviewer command for full regeneration is:

```bash
python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets
```

It verifies required dataset assets, runs strict validation, runs the active manuscript experiments, and regenerates manuscript tables/figures.

The underlying active full-run driver is:

```bash
python src/run_all_experiments.py --config config/experiment_settings.strict_no_leakage.yaml --paths config/paths.yaml
```

Before launching the expensive run, execute the preflight dry run:

```bash
python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets --dry-run
```

The dry run writes `results/logs/dry_run_environment_checks.csv` and `results/logs/dry_run_all_experiments_manifest.json`; paths in those logs are portable labels rooted at `${BUNDLE_ROOT}` or `${DATASETS_ROOT}`.
Dry run checks the dataset-bundle reference FASTA and configured raw-data paths directly. It does not import the expensive runner modules. If the full runtime stack has not been installed yet, it reports missing Python packages as a module advisory while still failing on missing required paths or assets.

Before uploading the code repository to GitHub, optionally remove accidental large duplicates from the checkout while keeping the sibling dataset bundle intact:

```bash
python scripts/reproduce_manuscript.py --prepare-github-upload --datasets-dir ../github_exports_datasets
```

Full regeneration writes `results/logs/run_all_experiments_manifest.json`. The MuAt-compatible top-20 comparator uses validated vendored fold checkpoint caches during release validation; those caches are listed in `manifests/large_assets_manifest.csv` and read from `../github_exports_datasets/`.

This is computationally expensive. The GitHub-ready validation path above is intended for release QA and artifact integrity checks.
