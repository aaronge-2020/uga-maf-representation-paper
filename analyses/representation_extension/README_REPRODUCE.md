# Reproducing the representation-extension analysis and the manuscript numbers

```bash
bash analyses/representation_extension/reproduce.sh
```

That single command, run from a fresh clone of this repository, verifies
the findings of the UGA/MAF representation paper (main panel plus the
49-cell extension grid) without fitting any model. Two more modes exist:

```bash
bash analyses/representation_extension/reproduce.sh --spotcheck   # refit 4 banked folds, compare
bash analyses/representation_extension/reproduce.sh --full        # rerun all 49 grid cells (days)
```

Requirements: Python 3.12, git, and git-lfs. The main panel's
out-of-fold predictions file is stored in git-lfs; clone with git-lfs
installed (or run `git lfs pull` afterwards). The script creates its own
pinned virtual environment in `.venv-reproduce/` from
`requirements-reproduce.txt` (numpy 1.26.4, pandas 2.3.1, scipy 1.15.3,
scikit-learn 1.7.2, xgboost 3.0.5, the versions the grid ran on). The
repo-level `requirements.txt` pins xgboost 1.6.2 for older main-panel
code; do not use it here.

## What the default mode checks

1. **Artifact hashes.** Every file the numbers come from is pinned by
   SHA-256 in `metrics/artifact_hashes.csv` (the 49 grid predictions, the
   49 trial tables, and 27 manuscript artifacts) and re-hashed.
2. **Features.** The R2 to R4 feature matrices are rebuilt from the
   tracked input files by the original builder, run in a shadow tree so
   nothing in the clone is overwritten, and compared value by value with
   the committed matrices. R5 (TF-IDF) is a run-time transform of R3; its
   fold-1 vocabulary is checked against `metrics/integration_check.json`.
3. **Canonical splits.** Each patient sits in exactly one fold, and the
   splits are regenerated from the main panel OOF file and compared.
4. **Grid scores.** Every per-fold score, primary score, and A8 primary
   contrast (with bootstrap confidence intervals, 2,000 resamples, seed
   20261025) is recomputed from the saved predictions and checked against
   `results_ledger.csv`. Primary scores and per-fold differences must
   match at printed precision; bootstrap interval bounds allow a Monte
   Carlo tolerance of plus or minus 0.002 (the shipped code replicates
   the reference random stream, and the largest deviation seen is far
   inside the tolerance).
5. **Trial-selection audit.** For each cell and fold, the runner's own
   selection rule is replayed over the saved trial table and the FINAL
   record must match the trial it picks, in configuration and inner
   score. The tuning outcomes are internally consistent without any
   refit.
6. **Manuscript numbers.** Table 1 in full (all 44 representation cells
   as fold means from the fold-metrics file, plus the MuAt column from
   the faithful run's OOF file), the printed confidence intervals as
   t-intervals from the five fold scores, the paired differences against
   MuAt, the ablations, PCAWG and POG570 transfer, centre-held-out
   recalls, the survival covariate run, the HRD controls, and the
   Figure 3 and Figure 6 values are recomputed from saved repository
   artifacts and checked against the ledger.
7. **Known discrepancies.** The output ends with a KNOWN DISCREPANCIES
   section: expected findings, printed for the record, never counted as
   failures. The saved main OOF predictions file disagrees with the
   endpoint and fold tables for a subset of groups (56 of 96 groups by
   more than 0.002); the results summary document's SHAP table conflicts
   with its own SHAP CSV (the CSV matches the manuscript); a fold
   differences JSON is stale; and the plot-data manifest references two
   canonical files that do not exist. Details and worst cases are in
   `claim_audit.md`.

## How verification works, honestly

The findings are verified by recomputing every number from banked,
hash-pinned model outputs and rebuilt features, not by refitting. A full
refit of the 49 cells took about a week on the project's 2-core virtual
machine and kept dying; it is not the verification mechanism. The
`--spotcheck` mode proves the fitting pipeline reproduces its own banked
outputs on sampled folds: it refits four folds end to end (one per
task/learner combination) from the configurations recorded in the saved
trial tables. On those folds the refit reproduces every tuning inner
score exactly and the selected configurations exactly; final predictions
agree with the banked ones to within 1e-5 (largest observed 3.8e-06),
which is the pass bound stated in `scripts/spotcheck.py`. The
determinism evidence for the checkpointed runner itself is the October
kill-and-resume proof: twice, a cell killed with kill -9 mid-run resumed
and produced predictions byte-identical to an uninterrupted run (md5
2bce989bb717ddc487962868d306f7e3; deviations log entries 9 and 10).

Boundary: the main panel has no spot check. Its original pipeline kept
no persisted trial configurations or checkpoints, so its verification is
recompute-from-artifacts plus the discrepancy list above. Nothing in
this package refits a main-panel model.

## Modes and runtimes

| Mode | What runs | Honest runtime |
|---|---|---|
| verify (default) | hash, feature, split, score, trial, and manuscript checks | about 15 minutes on the 2-core, 7.9 GB project VM (longer under load); most of it is the shadow feature rebuild |
| --spotcheck | refits 4 banked folds | about 20 to 30 minutes on the same VM |
| --full | all 49 cells through `scripts/runner.py` with their saved trial budgets | days; the original grid took about a week on this VM. For a large machine. Not required to verify the findings |

## Data provenance

All inputs are tracked in this repository: the TCGA event tables and the
main panel result files under `results/` and `data/`, and the extension
inputs under `analyses/representation_extension/` (feature matrices,
canonical splits, predictions, trial tables). No input is downloaded at
verify time. PCAWG and POG570 are previously examined cohorts, not
untouched validation; the paper examined both before this extension was
run. The Kucab cohort's raw release carries no VEP annotation, so Bio MAF
does not exist for Kucab; the Kucab event stack in the endpoint file is a
separate representation and is never labelled Bio MAF (see
`feature_dictionary.md`).

## Files

| File | Contents |
|---|---|
| `reproduce.sh` | the entrypoint above |
| `results_ledger.csv` | one row per reported number: value, source file, status (reproduced, sourced_not_rerun, unsourced) |
| `claim_audit.md` | verdicts per claim, the unsourced numbers listed at the top, naming rules, known discrepancies |
| `feature_dictionary.md` | every representation R0 to R6 and combinations: construction, feature counts, files |
| `metrics/artifact_hashes.csv` | SHA-256 pins for all source artifacts |
| `scripts/repro_common.py` | shared recomputation library (fits no models) |
| `scripts/verify_reproduce.py` | the default verification |
| `scripts/spotcheck.py` | the bounded refit check |
| `scripts/build_ledger.py` | regenerates `results_ledger.csv` and the hash pins from the artifacts |
| `scripts/run_full.py` | the `--full` runner loop |

## Citation

Ge, A. et al. Benchmarking somatic mutation representations (UGA/MAF).
Manuscript in preparation, 2026. Repository:
github.com/aaronge-2020/uga-maf-representation-paper, branch
`analysis/representation-extension`. Extension protocol:
`reports/protocol_v1.0.md` (as amended to v1.2), with deviations in
`reports/deviations_log.md` and the literature crosswalk in
`reports/literature_crosswalk.md`.
