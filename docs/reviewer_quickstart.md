# Reviewer Quickstart

This guide is for a reviewer starting from a fresh clone plus the companion dataset bundle.

## Folder Setup

Place the code repository and dataset bundle side by side:

```text
parent_folder/
  github_exports/
  github_exports_datasets/
```

Run all commands from `github_exports`.

## 1. Create the Python Environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Using another Python environment is fine. The important requirement is that `python -m pip install -r requirements.txt` succeeds before the full run.

## 2. Check Files, Dataset Bundle, and Packages

```bash
python scripts/reviewer_workflow.py start --datasets-dir ../github_exports_datasets
```

`python scripts/reviewer_workflow.py` also works; it defaults to the same setup check. This checks:

- required repository files and configs;
- all payloads in the companion dataset bundle;
- required Python packages;
- Node.js for figure rendering;
- whether large active assets are available through the dataset bundle.

The command prints a short checklist: what is okay, where the datasets are, what to fix, and what to run next. It also writes the full machine-readable report to `results/logs/reviewer_setup_check.json`. If something is missing, the checklist includes the exact fix, usually one of:

- install packages with `python -m pip install -r requirements.txt`;
- download or re-download the companion dataset bundle as `../github_exports_datasets`;
- rerun the validation command below so checksums and dataset-backed artifact links are rechecked.

For the full JSON in the terminal, add `--json`.

Expected dataset locations in the companion bundle:

- `datasets/tcga_mc3/`: TCGA/MC3 mutation calls, labels, and feature tables.
- `datasets/tcga_brca_hrd/`: TCGA-BRCA HRD labels and cohort inputs.
- `datasets/kucab_mendeley/`: KUCAB/Mendeley mutagenesis inputs.
- `datasets/kucab_mendeley/raw_mutation_tables/`: KUCAB raw mutation tables used by active runners.
- `datasets/pcawg_pancan/`: PCAWG/PanCancer signature data.
- `references/grch37/`: GRCh37 reference genome.
- `resources/bio_maf_v4/`: Bio MAF v4 feature resources.
- `resources/signatures_cosmic/`: COSMIC and other signature resources.

## 3. Fast Reviewer Validation

```bash
python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets --dry-run
```

This verifies required dataset-backed assets, strictly validates manuscript artifacts and science guards, and dry-runs the full manuscript driver without launching expensive model jobs.

Expected proof files:

- `results/logs/strict_validation_latest.json`
- `results/logs/dry_run_environment_checks.csv`
- `results/logs/dry_run_all_experiments_manifest.json`

## 4. One-Line Full Reproduction

```bash
python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets
```

This single command verifies assets, runs strict validation, runs the active manuscript experiments, and regenerates manuscript tables/figures through `src/run_all_experiments.py`.

The full run is computationally expensive. If it stops before the expensive jobs begin, first inspect `results/logs/reviewer_setup_check.json`; it will list the missing dependency, file, or dataset path and the command needed to fix it.

## 5. After GitHub Upload Preparation

If you are preparing the code repository for upload after validation, this optional command removes accidental large duplicates from the Git checkout while keeping the dataset bundle intact:

```bash
python scripts/reproduce_manuscript.py --prepare-github-upload --datasets-dir ../github_exports_datasets
```
