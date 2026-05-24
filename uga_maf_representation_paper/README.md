# UGA/MAF Representation Paper Reproducibility Bundle

This folder contains the reproducible bundle for the UGA, KME, and MAF representation benchmarking manuscript. It is designed to be moved to another machine and rerun from one command after the raw data paths are configured.

The bundle does not include raw patient-scale data, raw MAFs, FASTA/BAM files, or archived result artifacts. Existing experiment code and small reference/helper assets are vendored under `src/legacy_source/`; all tables, figures, logs, and working files are regenerated under `results/`.

The MAF event-gene/locus stack is preserved as the strongest exploratory lead. It is not documented here as a locked primary validation.

## Quick Start

```bash
conda env create -f environment/environment.yml
conda activate uga-maf
cp config/paths_example.yaml config/paths.yaml
# Edit config/paths.yaml so every raw_data path points to local inputs.
python src/run_all_experiments.py --config config/experiment_settings.cpu.yaml --paths config/paths.yaml
```

GPU profile:

```bash
python src/run_all_experiments.py --config config/experiment_settings.gpu.yaml --paths config/paths.yaml
```

Fresh manuscript rerun profile. The portable reproduction root is the parent
`projects/cgr_validation` folder; `config/paths.yaml` uses `${repro_root}` so
raw inputs, GRCh37, caches, checkpoints, and manuscript outputs stay inside
that single folder.

```bash
python src/utils/clean_generated_outputs.py --paths config/paths.yaml
python src/utils/clean_generated_outputs.py --paths config/paths.yaml --execute
python src/run_all_experiments.py --config config/experiment_settings.fresh_manuscript.yaml --paths config/paths.yaml --dry-run
python src/run_all_experiments.py --config config/experiment_settings.fresh_manuscript.yaml --paths config/paths.yaml
python src/utils/make_all_figures.py --config config/experiment_settings.fresh_manuscript.yaml --paths config/paths.yaml --strict
```

Intentional cache rebuild:

```bash
python src/run_all_experiments.py --config config/experiment_settings.fresh_manuscript.yaml --paths config/paths.yaml --refresh-cache
```

The fresh manuscript profile uses a single 5-fold cross-validation protocol with aggregated out-of-fold predictions, 10 Optuna trials for KME tuning, elastic-net linear models where the legacy code exposes linear learners, GPU-first XGBoost with CPU fallback where supported by the underlying runner, and resource-aware parallelism. Its main manuscript panel is intentionally tight: `burden_only`, `signatures_only`, `MAF_stack_only`, `signatures_plus_MAF_stack`, and `one_hot_event_KME`.

Restart/resume rule:

```bash
# Resume after interruption or power loss. Do not run cleanup first.
python src/run_all_experiments.py --config config/experiment_settings.fresh_manuscript.yaml --paths config/paths.yaml
```

Long-running scripts write step-level checkpoint CSVs as they progress, and the runner writes script-completion checkpoints under `results/checkpoints/scripts`. Persistent feature caches live under `results/cache/features/`, including FASTA one-hot event inventories, KME feature matrices, and snapshots of legacy generated feature artifacts. Cleanup preserves checkpoints and feature caches; use `--refresh-cache` only when you intentionally want to rebuild cached features.

Obsolete generated artifacts are reviewed before deletion:

```bash
python src/utils/clean_generated_outputs.py --paths config/paths.yaml --include-legacy --obsolete
python src/utils/clean_generated_outputs.py --paths config/paths.yaml --include-legacy --obsolete --execute
```

Regenerate manuscript tables and figures from already regenerated result CSVs:

```bash
python src/utils/make_all_figures.py --config config/experiment_settings.cpu.yaml
```

Dry-run path and command check:

```bash
python src/run_all_experiments.py --config config/experiment_settings.cpu.yaml --paths config/paths.yaml --dry-run
```

## Data Preparation

Copy `config/paths_example.yaml` to `config/paths.yaml` and edit paths. Relative paths are resolved from this bundle root.

Expected MC3 source directory:

```text
mc3_source/
  biology_labels.csv
  driver_gene_labels_functional.csv              # optional cache, rebuilt if absent
  features/
    features_burden_only.csv
    features_standard_sbs96.csv.gz
    features_standard_id83.csv.gz
    features_standard_sbs96_id83.csv.gz
  raw/
    mc3.v0.2.8.PUBLIC.maf.gz
    TCGA-CDR-SupplementalTableS1.xlsx
```

Expected TCGA-BRCA HRD assets:

```text
tcga_brca_hrd/
  cohort/
    final_analysis_cohort.tsv
```

Expected Kucab low-burden raw directory:

```text
kucab/
  README.txt
  denovo_subclone_subs_final.txt
  denovo_subclone_doublesub_final.txt
  denovo_subclone_indels.final.txt
```

Expected PCAWG signature attribution directory:

```text
pancan_pcawg_2020/
  data_mutational_signatures_counts_SBS.txt
  data_mutational_signatures_counts_DBS.txt
  data_mutational_signatures_counts_ID.txt
  data_mutational_signatures_contribution_SBS.txt
  data_mutational_signatures_contribution_DBS.txt
  data_mutational_signatures_contribution_ID.txt
```

`data/raw/` is intentionally empty except for instructions. Raw data may live elsewhere; put those locations in `config/paths.yaml`.

## Runtime Profiles

`config/experiment_settings.cpu.yaml` uses `tree_method: hist` for portable CPU reruns.

`config/experiment_settings.gpu.yaml` uses `tree_method: gpu_hist` to preserve the retained GPU-style XGBoost profile.

`config/experiment_settings.fresh_manuscript.yaml` is the strict regeneration profile for the revised manuscript plan. Run the cleanup utility only when intentionally starting from zero; otherwise the profile preserves `results/work/` and `results/checkpoints/` so interrupted runs can resume. Final paper assets are written under `results/manuscript/{figures,tables,supplement}`.

Both configs call the same runners and write the same normalized output filenames.

## Outputs

Main normalized outputs:

```text
results/tables/unified_locked_endpoint_results.csv
results/tables/unified_locked_summary.csv
results/tables/uga_kme_endpoint_results.csv
results/tables/uga_kme_summary.csv
results/tables/maf_event_gene_locus_endpoint_results.csv
results/tables/maf_event_gene_locus_summary.csv
results/tables/mechanistic_representation_endpoint_results.csv
results/tables/mechanistic_representation_summary.csv
results/tables/manuscript_benchmark_endpoint_table.csv
results/tables/manuscript_endpoint_delta_long.csv
results/tables/manuscript_panel_file_map.csv
results/figures/manuscript_auroc_spearman_delta_bars.png
results/figures/manuscript_representation_learner_heatmap.png
results/figures/manuscript_mechanistic_summary.png
results/logs/run_all_experiments_manifest.json
```

Fresh manuscript outputs:

```text
results/manuscript/tables/table_1_datasets_endpoints.csv
results/manuscript/tables/table_2_full_performance_metrics.csv
results/manuscript/tables/table_3_hyperparameters_feature_dimensionality.csv
results/manuscript/figures/figure_1_conceptual_overview.{png,svg,pdf}
results/manuscript/figures/figure_1_conceptual_overview.html
results/manuscript/figures/figure_2_signature_baselines.{png,svg,pdf}
results/manuscript/figures/figure_3_geometry_vs_signatures.{png,svg,pdf}
results/manuscript/figures/figure_4_maf_stack_vs_signatures.{png,svg,pdf}
results/manuscript/figures/figure_5_cross_endpoint_summary.{png,svg,pdf}
results/manuscript/supplement/figure_s1_representation_construction.{png,svg,pdf}
results/manuscript/supplement/figure_s2_calibration_thresholds.{png,svg,pdf}
results/manuscript/supplement/figure_s3_feature_importance.{png,svg,pdf}
results/manuscript/plot_data/benchmark_completeness.csv
results/manuscript/plot_data/plot_data_manifest.json
results/manuscript/d3_render_manifest.json
results/manuscript/supplement/table_s1_class_distribution_baselines.csv
results/manuscript/supplement/table_s2_sensitivity_analyses.csv
results/manuscript/manuscript_asset_manifest.json
results/logs/fresh_manuscript_final_manifest.json
```

The runners also copy regenerated source CSV, TSV, HTML, OOF prediction, and figure artifacts into `results/tables/` and `results/figures/` with experiment-specific prefixes.

## Manuscript Mapping

| Manuscript item | Regenerated file |
|---|---|
| Locked Standard vs UGA benchmark across HRD, Kucab, PCAWG, MC3, and TCGA signature endpoints | `results/tables/unified_locked_endpoint_results.csv` |
| UGA mean, KME, tuned KME, XGBoost sensitivity, and Standard+KME subset | `results/tables/uga_kme_endpoint_results.csv` |
| Exploratory MAF event-gene/locus stack across the 16-endpoint panel | `results/tables/maf_event_gene_locus_endpoint_results.csv` |
| EXP024 shared-space and EXP025 payload-absence mechanism controls | `results/tables/mechanistic_representation_endpoint_results.csv` |
| Combined endpoint table | `results/tables/manuscript_benchmark_endpoint_table.csv` |
| Delta bar plot | `results/figures/manuscript_auroc_spearman_delta_bars.png` |
| Representation by learner heatmap | `results/figures/manuscript_representation_learner_heatmap.png` |
| Mechanistic summary panel | `results/figures/manuscript_mechanistic_summary.png` |

## Bundle Layout

```text
uga_maf_representation_paper/
  README.md
  environment/
  config/
  data/
  docs/
  src/
    run_all_experiments.py
    runners/
    utils/
    legacy_source/
  results/
```

`docs/experiment_manifest.yaml` inventories the experiments, source entrypoints, data dependencies, and normalized outputs. `docs/representation_specs/` documents each representation used in the manuscript narrative.

## QA Checklist

```bash
python -m compileall src
python src/run_all_experiments.py --config config/experiment_settings.cpu.yaml --paths config/paths.yaml --dry-run
rg "<local-machine-specific-path-patterns>" .
```

The path scan should return no hits in a clean bundle after regenerating logs with the current runners.
