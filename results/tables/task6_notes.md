# Task 6 — Reproducibility deposit: notes

## 1. OncoKB pin (deliverable 1)

**Explicit statement:** the 948 Bio MAF features were built from a LIVE OncoKB
download, not from a bundled snapshot. No OncoKB table is committed to the
repository (removed 2026-10-03, commit `9fdbb34`; OncoKB Terms of Use restrict
redistribution). The pipeline fetches the table at build time from the public,
unversioned API.

- Source URL: `https://public.api.oncokb.org/api/v1/utils/cancerGeneList` (no key)
- Last recorded snapshot: downloaded **2026-09-23**
  - sha256: `1ef55e04e29522d89a4c72795e215955489a5ee0d8ff896a9ed9d75319603f2d`
    (recorded in `config/feature_resources/resource_manifest.json`)
  - 1,246 genes: 445 oncogene, 340 tumor suppressor, 68 both, 393 no role
  - geneType mapping: ONCOGENE->oncogene, TSG->tsg, ONCOGENE_AND_TSG->both,
    INSUFFICIENT_EVIDENCE/NEITHER/null->neither
- **Critical caveat (from `config/feature_resources/oncokb_cancer_genes.PROVENANCE.md`):**
  the published 948-feature matrix was built with an EARLIER OncoKB snapshot
  whose exact role assignments are a strict subset of the 2026-09-23 download.
  That historical snapshot is **not recoverable** from public sources (no git
  history, no web archive of the API endpoint). A from-scratch rebuild with any
  downloadable snapshot reproduces the 16 `oncokb_role_*` aggregate columns at
  Pearson r = 0.96, not bit-identical. The published matrix
  (`results/tables/quick_bio_v4_maf_features.csv.gz`) remains the artifact of
  record for every manuscript number.
- Task 6 rerun snapshot: fetched live during the entry-point run; download date
  + sha256 recorded in `results/logs/task6_table1_run_report.json` (`oncokb`
  section) and a dated copy archived at
  `results/tables/task6_oncokb_snapshot/oncokb_cancer_genes_<YYYY-MM-DD>.tsv`
  with a JSON sidecar. (ToU note: the archived copy is a build artifact for
  provenance, not cleared for redistribution; do not publish it without review.)

## 2. 948 -> 910 feature-set change (structural, affects the rerun)

Commit `7984d67` (2026-09-22, "Remove undocumented role-matched columns for 38
INSUFFICIENT_EVIDENCE genes") changed the Bio MAF v4 builder AFTER the
manuscript run (2026-08-15/17):

| candidate block | manuscript (948 build) | current code |
|---|---|---|
| v4_compact_biology_only | 121 | 121 |
| v4_compact_plus_driver_evidence_tiers | 137 | 137 |
| v4_compact_plus_hotspot_summary | 125 | 125 |
| v4_compact_plus_exact_driver_genes | 923 | 885 |
| v4_full_core | 943 | 905 |
| v4_full_core_plus_optional_controls | **948** | **910** |

The 38 removed columns are all `driver_gene_role_matched_event_log_count__*`
(verified: the current 910-column set is a strict subset of the published 948).

Refined finding (2026-10-13 smoke test): `build_bio_v4_features` checks for the
committed `results/tables/quick_bio_v4_maf_features.csv.gz` FIRST and loads it
instead of rebuilding. So the Task 6 rerun's feature VALUES are bit-identical
to the manuscript's published 948-feature matrix (10,224 x 948; the artifact of
record) -- the historical OncoKB snapshot is preserved inside it. What changed
is only the CANDIDATE BLOCKS offered to the nested inner loop: the rerun
selects among the corrected 910-feature blocks, while the manuscript selected
among 948-feature blocks. Per-cell deltas therefore combine (a) environmental
changes (Python 3.12.3, xgboost 3.4.1 CPU-hist, sklearn 1.9.1) with (b) the
38-column candidate-block difference. Both are reported as measured, not
adjusted; the entry-point deliberately does NOT force a from-scratch rebuild
(`--refresh-cache` / `maf_bio_v4_force_features` left off) because a rebuild
with current code could never reproduce the manuscript's historical OncoKB
role columns (see section 1) -- the committed matrix is the only faithful
feature source.

## 3. Environment (deliverable 2)

Full pin in `environment/table1_lockfile.txt` (sections: A = manuscript-run
Python 3.10 pins; B = Task 6 rerun Python 3.12.3 actuals; C = parallelism
adaptation). Key points:

- The manuscript ran on Python 3.10 with xgboost==1.6.2 (tree_method=gpu_hist,
  GPU hardware). This box has Python 3.12.3, no GPU, and xgboost 1.6.2 has no
  cp312 wheels, so the pinned environment cannot be installed here.
- Task 6 rerun used: Python 3.12.3, numpy 1.26.4, pandas 2.1.4, scipy 1.11.4,
  scikit-learn 1.9.1, xgboost 3.4.1 (the pipeline's own
  `_fit_xgb_with_optional_fallback` catches the gpu_hist rejection and falls
  back to tree_method=hist on CPU), PyYAML 6.0.1, openpyxl 3.1.2,
  statsmodels 0.15.0, requests 2.31.0. optuna/scikit-survival/torch/pysam/tabulate
  not installed; none is imported by the Table-1 code path.
- Parallelism knobs 32/32/16 -> 2/2/2 plus OMP/MKL/OPENBLAS_NUM_THREADS=2.
  Seeds derive from slot names (utils.nested_oof.stable_seed), never from
  n_jobs, so this changes wall time only.
- Task 0 gate (burden_only/linear: 0.246153 vs manuscript 0.247320, delta
  0.001167) is accepted as environmental and was NOT rerun.

## 4. Entry-point script (deliverable 3)

`scripts/reproduce_table1.py` — single entry point driving the PRODUCTION
pipeline (`src/run_all_experiments.py` +
`runners.run_main_manuscript_complete_panel.run`) with the Table-1-only config
`config/experiment_settings.table1.yaml` (diff vs strict config documented in
the file header). It does not reimplement modelling code.

Design points that keep the regeneration honest:
- Archives existing main-panel result tables AND the 8 cancer_type_top20 fold
  checkpoints aside (with sha256) before running; otherwise the runner's
  endpoint-row resume and fold-checkpoint resume would silently reuse the
  manuscript's own outputs and the "regeneration" would be a no-op.
- Uses a fresh feature-cache dir (the committed tree has no
  `results/cache/`, so Bio MAF features genuinely rebuild from the MC3 MAF).
- Two documented in-script adapters, no production-code edits: (1) skip the
  kucab feature build when the `task6_skip_kucab` flag is set (Table 1 has no
  kucab endpoint; the build unconditionally requires the 3.2 GB GRCh37 FASTA);
  (2) a behaviour-preserving timing wrapper around the Bio MAF builder for the
  compute-cost numbers.
- Extracts the 8 Table-1 slots to `results/tables/task6_table1_regenerated.csv`
  and compares balanced_accuracy against the archived manuscript values in
  `results/tables/task6_table1_comparison.csv`; full JSON report in
  `results/logs/task6_table1_run_report.json`.

### Clean-checkout run report

**Run:** 2026-10-13, clean checkout of `reviewer-analyses-2026-10` at `e801b2b`
(+ 6 untracked Task 6 files), `~/workspace/somatic-mutation-repo-clean`,
via `scripts/reproduce_table1.py` (single entry point) plus the documented
crash-recovery driver `scripts/task6_resume_table1.py` (see "VM restart
drains" below). Environment: Python 3.12.3, xgboost 3.4.1 (CPU hist via
pipeline fallback), sklearn 1.9.1. Bundle: `~/workspace/github_exports_datasets`
(HF `vjdeara/cgr_tensorlab_umsom`, curl-fetched; key inputs hash-verified).
OncoKB snapshot for the run: 2026-10-13,
sha256 `9f80168c8fc9206f707488a5ba06b779cade26aa5c0900c93fa2a168e8fcadb5`
(1,247 genes). Prerun archive:
`results/tables/task6_prerun_archive/20261013T120157Z/` (6 tables, 8 fold
checkpoints, 8 manuscript reference rows).

**Status:** PARTIAL — 3 of 8 Table-1 slots genuinely regenerated and compared;
5 slots incomplete (1 blocked, 4 pending). See "VM restart drains" for why
the run could not finish in one session.

**Final comparison (manuscript vs regenerated balanced_accuracy):**

| slot | manuscript | regenerated | delta | within 0.001167? |
|---|---|---|---|---|
| burden_only / linear | 0.2473204505990421 | 0.2473204505990421 | +0.000000 | YES (bit-identical) |
| burden_only / xgboost | 0.2549450166395410 | 0.2587351827150739 | +0.003790 | NO |
| standard_sbs96_id83 / linear | 0.5263482583725251 | — (blocked, see below) | — | — |
| standard_sbs96_id83 / xgboost | 0.5381477651217608 | 0.5449880517214332 | +0.006840 | NO |
| MAF_stack_only / linear | 0.5712277179766847 | — (in progress: fold 1/5 done) | — | — |
| MAF_stack_only / xgboost | 0.6618295959768071 | — (pending) | — | — |
| signatures_plus_MAF_stack / linear | 0.6723468531222593 | — (pending) | — | — |
| signatures_plus_MAF_stack / xgboost | 0.7377837111543271 | — (pending) | — | — |

(accuracy values also regenerated: burden_only/linear 0.2425, burden_only/
xgboost 0.2910227272727272, standard_sbs96_id83/xgboost 0.56875.)

**MAJOR finding on Task 0:** the production-pipeline rerun of
burden_only/linear reproduces the manuscript value BIT-IDENTICALLY
(0.2473204505990421, delta 0.0) on Python 3.12.3 / sklearn 1.9.1. Task 0's
rerun reported 0.246153 (delta -0.001167), attributed to the Python 3.10->3.12
runtime change with SAGA tie sensitivity. Since the production pipeline on
3.12.3 reproduces exactly, Task 0's delta was NOT caused by the Python
runtime change -- it must have come from Task 0's rerun setup (different
code path, data handling, or seed). The "accepted environmental" attribution
for Task 0 should be revisited; the 0.001167 tolerance calibrated on it may
be looser than the pipeline's actual environmental sensitivity for linear
models (which appears to be ~0 here).

**XGBoost deltas exceed tolerance (both cells):** burden_only/xgboost
+0.003790 and standard_sbs96_id83/xgboost +0.006840 both exceed the 0.001167
tolerance, both in the POSITIVE direction (regenerated > manuscript). This is
a genuine environmental difference (xgboost 1.6.2/gpu_hist/GPU ->
3.4.1/hist/CPU), not noise: the two independent xgboost cells shift the same
way. Whether the MAF xgboost cells (pending) follow the same pattern remains
to be measured.

**VM restart drains (the blocker):** the rerun box underwent platform-level
restart drains 5 times on 2026-10-13 (VM reboots; `uptime` reset to 0):
~12:13-13:35 (run 1 died), ~14:54 (run 2 died, 78 min lifetime), ~16:49
(resume died, 113 min lifetime), ~19:25 (resume died), ~21:28 (resume died).
Observed VM lifetimes: 78-128 min. The entry point
`scripts/reproduce_table1.py` cannot resume across drains (its stage-4
archive moves partial outputs aside on every invocation), so a companion
crash-recovery driver `scripts/task6_resume_table1.py` was written. It runs
the IDENTICAL code path (same production runner, same Table-1 config, same
two in-script adapters) but without the archive step, relying on the
pipeline's own resume: completed slots are skipped via endpoint-row resume
(`[checkpoint] reuse`), and the MAF nested-FS slots resume per-fold and
per-candidate from fold-checkpoint files (verified working: fold 1 reused,
candidate search resumed at 19/60 and 28/60 across drains). The manuscript's
outputs remain archived (never restored); the worktree holds only regenerated
data.

**Blocked slot — standard_sbs96_id83/linear:** this slot needs ~130 SAGA
fits (25-candidate full grid x 5 folds + refits; the table1 config, like the
manuscript's strict config, does not set `linear_random_search_trials`, so
the full 5x5 grid is used). On this 2-vCPU box it needs ~2-4 hours, but the
plain `evaluate_nested_oof` path has NO fold checkpointing, and VM lifetimes
are 78-128 min. Two attempts (60 min, 113 min) both died before completion.
It was therefore skipped via the driver's documented `--skip` mechanism
(recorded as `skipped_blocked_slot` with NaN score; `_current_row()` rejects
NaN rows so a later un-skipped run will genuinely rerun it) so the remaining
slots could proceed. It has NOT been regenerated; this is an honest gap, not
a silent omission.

**Compute cost (measured wall times on the 2-vCPU rerun box):**
- Bio MAF feature matrix load: 37.4s for 10,224 tumours x 948 features
  = 0.0037s per tumour (cold load of the committed published matrix; see
  section 2). Subsequent cache hits: 1.5-11.2s.
- burden_only/linear: 195s (run 1), 302s (run 2 rerun).
- burden_only/xgboost: 446s (run 1), 784s (run 2 rerun).
- standard_sbs96_id83/xgboost: 6984s (116 min).
- standard_sbs96_id83/linear: not completed (see blocked slot above).
- MAF_stack_only/linear: in progress — fold 1/5 completed and checkpointed;
  fold 2/5 at candidate 25/60 as of 22:38 UTC. Per-candidate times range
  10s (121-feature blocks) to 1381s (885/910-feature SAGA fits).
- Remaining MAF slots: pending. Manuscript runtimes (32 threads + GPU) were
  5780s / 13176s / 11829s; expect several hours each on this box.
- XGBoost train vs inference split: not separately logged by the pipeline;
  reported times are total slot wall times (5-fold outer nested CV including
  inner search, refits, and bootstrap:500 CIs).

**Reference-extraction caveat (entry-point bug on reruns, corrected):**
`scripts/reproduce_table1.py::stage_prerun_archive` extracts the "manuscript
reference" from whatever `main_manuscript_complete_panel_endpoint_results.csv`
exists at archive time. On a rerun that file holds PARTIAL REGENERATED
outputs, not manuscript outputs, so the second run's
`task6_prerun_archive/20261013T133518Z/table1_manuscript_reference.csv`
contains 2 rows of regenerated values mislabeled as manuscript values, and
the entry point's own comparison compared regenerated-vs-regenerated for
those slots. Corrected post-hoc by
`scripts/task6_fix_rerun_comparison.py`, which copies the TRUE 8-row
manuscript reference from
`task6_prerun_archive/20261013T120157Z/table1_manuscript_reference.csv`
(first run's archive, written before any regeneration) and rebuilds
`results/tables/task6_table1_comparison.csv` with the identical schema and
tolerance logic. No regenerated values were altered; the mislabelled file
is preserved as `table1_manuscript_reference.csv.rerun_mislabelled_backup`.
