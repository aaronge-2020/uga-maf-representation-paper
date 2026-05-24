# EXP-024 Shared-Space Benchmark Runner

This folder contains the reproducible runner for the UGA shared-coordinate benchmark.

## Run

From the repository root:

```powershell
python cgr_validation_results\research\scripts\EXP024_shared_space\run_shared_space_benchmark.py
```

The compatibility wrapper below is kept for convenience:

```powershell
python bench\run_shared_space_benchmark.py
```

## Outputs

The default output package is:

```text
cgr_validation_results/research/reports/exp024_shared_space/
```

The runner writes a structured package with:

- `README.md`: generated experiment documentation and headline results.
- `index.html`: local landing page linking the rendered artifacts.
- `html/`: narrative walkthrough, publication-ready tables, and copy/paste HTML fragments.
- `figures/`: D3 figure page.
- `tables/`: CSV summaries and patient-level metrics.
- `data/`: JSON data used by the D3 figures.
- `manifest.json`: run parameters and artifact paths.

## Purpose

The benchmark tests whether UGA's shared SBS/DBS/ID coordinate space provides useful cross-modality structure even when exposure estimation is better solved separately by modality.

The central comparison is intentionally two-part:

- separate modality solves vs a naive pooled UGA solve;
- cross-modality retrieval and held-out imputation enabled by shared UGA coordinates.
