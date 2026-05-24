# EXP023 — TCGA-BRCA HRD (paper workflow)

Single entrypoint: **`exp23_pipeline.py`**. Stages: `prepare`, `exposures`, `analyze`, `figure`, `all`.

**Environment:** needs **pandas**, **numpy**, **scipy**, **scikit-learn**, **matplotlib**, **optuna** (e.g. conda `cgr_gen` + `pip install optuna` if missing). Example:

```bash
conda run -n cgr_gen python cgr_validation_results/research/scripts/EXP023_tcga_brca_hrd/exp23_pipeline.py all
```

**Outputs**

- **Assets (machine-readable + figures):** `cgr_validation_results/research/assets/EXP023_tcga_brca_hrd/TCGA-BRCA/` — catalogs, exposures, cohort TSVs, `modeling_results/*.tsv`, `tables/*.tsv`, `figures/`, `metadata/*.json`.
- **Reports (human-readable markdown with pipe tables):** `cgr_validation_results/research/reports/EXP023_tcga_brca_hrd/TCGA-BRCA/` — `input_validation.md`, `data_preparation.md`, `exposure_inference.md`, per-endpoint `summary_<endpoint>.md`, `cross_endpoint_summary.md`, `burden_sensitivity_summary.md`, `hrd_binary_threshold_summary.md` (full-cohort **42 / 33 / 24**), `statistical_tests_summary.md`.

Optional CLI: `--repo-root`, `--optuna-storage sqlite:///optuna_exp023.db`, `--reg-trials`, `--clf-trials`.

Modules: `exp23_config`, `exp23_prepare`, `exp23_exposures`, `exp23_modeling`, `exp23_analyze`, `exp23_stats`, `exp23_utils`, `exp23_workflow` (re-exports), `figure_exp23_brca_hrd.py` (composite Δ figure: lollipop, binary dumbbells, burden heatmap). Tables in markdown use `scripts/_md_report_utils.py` (`df_to_md_table` in `exp23_utils`).

Replaces former **EXP027** / **EXP029** folders and the retired **PCAWG channel-reconstruction** track that previously used the EXP-023 experiment id.
