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

The strict command validates active source/result tables, uses large active artifacts directly from the readable sibling bundle when they are intentionally absent from Git, checks Bio MAF v4 resources, confirms top-20 and HRD binary endpoints, verifies that generated manuscript outputs are not tracked for GitHub, scans active configs/source/docs/manifests for disallowed external path dependencies, and writes repo plus dataset read audits under `results/logs/`.

To check the sibling dataset folder without copying files back into the checkout:

```bash
python scripts/reproduce_manuscript.py --check-datasets --datasets-dir ../github_exports_datasets
```

## Full Regeneration

The one-line reviewer command for full regeneration is:

```bash
python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets
```

It verifies required dataset assets, runs strict validation, runs the active manuscript experiments, and regenerates manuscript tables/figures into the ignored local `results/manuscript/` output directory.

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

## MuAt-compatible HRD comparator: frozen run identification

The manuscript's MuAt-compatible HRD headline numbers (HRD-score Spearman 0.507, HRD33 AUROC 0.746, fold means with t-based 95% CIs) come from the faithful fixed-architecture run (VIJAY, 2026-10-04; fixed architecture, constant LR, weight decay 0.001, all epochs, arm D token grammar).

- Fold-level metrics: `results/tables/muat_style_tcga_comparator_hrd_faithful_fixed_arch_fold_metrics.csv`. HRD-score Spearman per fold: 0.486, 0.459, 0.494, 0.484, 0.612 (mean 0.507). HRD33 per-fold AUROC recomputed from `muat_style_tcga_comparator_hrd_faithful_fixed_arch_oof_predictions.csv`: 0.718, 0.713, 0.714, 0.776, 0.807 (mean 0.746).
- Paired Sig+MAF vs MuAt-compatible tests: `results/tables/hrd_muat_paired_tests.json` (recomputed 2026-10-14; diff 0.270 [0.220, 0.319] for HRD score, 0.165 [0.094, 0.236] for HRD33).
- Ablation ledger: `results/tables/muat_ablation_summary.csv`.

Correction note (2026-10-14): an earlier version of this file tied the HRD headlines to the `ab_c_full` run (0.527/0.751), which used a non-faithful configuration (epoch fix + exact token dictionaries but not the MuAt-faithful motif grammar). The manuscript now reports the faithful-run numbers. The ab_c_full-based paired test is superseded; its values are preserved in git history.

---

## Reviewer-analyses tasks (Tasks 0–6): version table and reproduce commands

Added 2026-10-14. Each task below lists the exact environment that produced its
committed outputs, plus the minimal commands to reproduce on a fresh clone.

### Version table

| Task | Python | xgboost | sklearn | pandas | numpy | scipy | Lockfile |
|---|---|---|---|---|---|---|---|
| 0 (Table 1) | 3.12.3 | 1.6.2 | 1.7.2 | 2.3.1 | 1.26.4 | 1.15.3 | `environment/table1_lockfile.txt` |
| 1 (gene ablations) | 3.12.3 | 1.6.2 | 1.7.2 | 2.3.1 | 1.26.4 | 1.15.3 | `environment/reviewer_analyses_lockfile.txt` |
| 2 (paired bootstrap) | 3.12.3 | — (not used) | — | 2.3.1 | 1.26.4 | 1.15.3 | `environment/reviewer_analyses_lockfile.txt` |
| 3 (HRD attribution) | 3.12.3 | 1.6.2 | 1.7.2 | 2.3.1 | 1.26.4 | 1.15.3 | `environment/reviewer_analyses_lockfile.txt` |
| 4 (centre held-out) | 3.12.3 | 1.6.2 | 1.7.2 | 2.3.1 | 1.26.4 | 1.15.3 | `environment/reviewer_analyses_lockfile.txt` |
| 5 (PCAWG audit) | 3.12.3 | 1.6.2 | 1.7.2 | 2.3.1 | 1.26.4 | 1.15.3 | `environment/reviewer_analyses_lockfile.txt` |
| 6 (repro deposit) | — (docs only) | — | — | — | — | — | — |
| MuAt faithful | VIJAY's env (reconstructed) | — | — | — | — | — | `environment/muat_faithful_lockfile.txt` (reconstructed) |
| Frozen model (CPU regen) | 3.12.3 | 3.4.1 | 1.9.1 | 2.1.4 | 1.26.4 | 1.11.4 | `environment/frozen_model_lockfile.txt` |
| PCAWG transfer | 3.13.5 (reconstructed) | 3.4.1 | 1.9.0 | — | — | — | `environment/pcawg_transfer_lockfile.txt` (reconstructed) |

GPU note (Task 0): production Table 1 XGBoost cells used `gpu_hist` on a GPU
machine; CPU reruns use `hist` and differ by ~0.001 on XGBoost cells. Linear
cells reproduce bit-for-bit on CPU.

### MuAt faithful-protocol numbers

The manuscript's MuAt headline comparisons (cancer typing 0.528, HRD score
0.507, HRD33 0.746; paired differences vs Sig+MAF) come from VIJAY Vankadaru's
faithful-protocol rerun. VIJAY's configs, fidelity audit, run logs, and
divergence guard are now in this repository (merged from branch
`codex/coxnet-muat-manuscript-updates`):
- Fidelity audit: `docs/muat_fidelity_audit_2026-10.md`
- Faithful-protocol configs: `config/experiment_settings.muat_hrd_faithful_fixed_arch.yaml`,
  `config/experiment_settings.muat_top20_faithful_fixed_arch.yaml`,
  `config/experiment_settings.muat_top20_fold3_rerun.yaml`
- Divergence guard: `scripts/test_muat_divergence_guard.py`
- Run logs and manifests: `results/logs/muat_*.log`, `results/logs/*manifest*.json`
- Results summary: `docs/results_summary_2026-10.md` (also committed as `4da1160`)

This repo also contains a MuAt-compatible reimplementation
(`src/runners/run_muat_style_tcga_comparator.py`) which is explicitly not
official MuAt; see `docs/muat_replication_rigor.md`.

### Reproduce commands (fresh clone)

```bash
# 0. One-time setup (all tasks)
git clone https://github.com/aaronge-2020/uga-maf-representation-paper.git
cd uga-maf-representation-paper
python3.12 -m venv venv && venv/bin/pip install -r environment/reviewer_analyses_lockfile.txt
python scripts/fetch_datasets.py   # downloads the HF data bundle
```

```bash
# Task 1: gene-identity / knowledge ablations
venv/bin/python scripts/task1_gene_knowledge_ablations_v2.py
# → results/tables/task1_oof_*.csv + task1_gene_knowledge_ablations.csv
```

```bash
# Task 2: paired bootstrap inference
venv/bin/python scripts/task2_paired_bootstrap.py
# → results/tables/task2_inference_results.csv
```

```bash
# Task 3: HRD attribution baselines + POG570 metrics
venv/bin/python scripts/run_task3_hrd_attribution_baselines.py
venv/bin/python scripts/run_task3_pog570_metrics.py
# → results/tables/task3_*.csv
```

```bash
# Task 4: centre-held-out evaluation
venv/bin/python scripts/task4_centre_heldout.py
# → results/tables/task4_*.csv
```

```bash
# Task 5: PCAWG audit
venv/bin/python scripts/task5_pcawg_audit.py
venv/bin/python scripts/task5_column_alignment_check.py
# → results/tables/task5_*.csv
```

```bash
# Task 0: Table 1 reproduction (slow: 60+ CPU hours; GPU recommended)
venv/bin/python scripts/reproduce_table1_task0.py
# → results/tables/task0_rerun_outputs/
```

---

## Known gaps (documented 2026-10-14; not fabricated)

### Seven secondary contrasts not enumerated
The manuscript Methods states "We corrected seven secondary comparisons for multiple testing" and Results reports one example (HRD33 gain over mutation patterns alone: +0.037, raw p=0.058, BH q=0.068). The full list of seven contrasts and their raw p-values / q-values are not committed in a repo artifact. The Task 2 CSV (`results/tables/task2_inference_results.csv`) covers the primary paired-bootstrap contrasts (C1a/C1b/C4a) with q_bh values. The seven secondary contrasts need a committed source file; until then, treat the enumeration as a documentation gap.

### OncoKB version pin
The OncoKB cancer-gene table is fetched live from `https://public.api.oncokb.org/api/v1/utils/cancerGeneList` (unversioned API). OncoKB Terms of Use prohibit redistribution, so the table is not committed. Provenance:
- Table 1 production: downloaded 2026-09-23 (see `config/feature_resources/oncokb_cancer_genes.PROVENANCE.md`)
- Snapshot archived: `results/tables/task6_oncokb_snapshot/oncokb_cancer_genes_2026-10-13.tsv` (sha256 9f80168c8fc9206f707488a5ba06b779cade26aa5c0900c93fa2a168e8fcadb5, 1,247 genes)
- The 2026-09-23 production table is not archived; a fresh download drifts from it. The feature builder documents the mapping in the PROVENANCE file.

### Environment lockfiles
- `environment/table1_lockfile.txt`: production Table 1 (Python 3.12.3, xgboost 1.6.2, gpu_hist) — authoritative, captured from actual venv
- `environment/reviewer_analyses_lockfile.txt`: Tasks 1-6 (Python 3.12.3, xgboost 1.6.2)
- `environment/frozen_model_lockfile.txt`: CPU Table 1 regeneration (Python 3.12.3, xgboost 3.4.1, hist/CPU) — captured 2026-10-14 from ~/workspace/task6_venv
- `environment/pcawg_transfer_lockfile.txt`: RECONSTRUCTED (Python 3.13.5, sklearn 1.9.0 known; pandas/numpy unrecorded)
- `environment/muat_faithful_lockfile.txt`: RECONSTRUCTED (architecture and config known; Python/torch versions unrecorded)

### Zenodo bundle
The v2 bundle contains the superseded `hrd_muat_paired_tests.json` (ab_c_full numbers). A v3 bundle or version note is needed to publish the corrected faithful-run paired test.
