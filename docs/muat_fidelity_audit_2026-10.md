# MuAt-compatible comparator: fidelity audit against the official MuAt trainer

Date: 2 October 2026
Author: Vijay Vankadaru
Scope: the optimisation recipe and selection protocol of `src/runners/run_muat_style_tcga_comparator.py`,
compared line by line against the official MuAt software.

## Sources read

| Tag | What | Where |
|---|---|---|
| PAPER | Sanjaya et al. 2023, Genome Medicine 15:47, Methods | doi 10.1186/s13073-023-01204-4 |
| CODE | Official trainer and CLI, read 2 Oct 2026 | github.com/primasanjaya/muat, `muat/trainer.py`, `muat/core.py` |
| LEDGER | Official reproduction recipes (tags a1, d1 to d6) | `muat/pkg_reproduce/experiments.json`, `experiments.md` |

Caveat on CODE: this is the maintained 2026 software (v0.1.26, bioconda). The 2023 training
scripts are not published separately. The ledger itself notes that the published checkpoints'
`trainer_config.json` is empty, so the exact 2023 optimiser values cannot be read off the
artefacts (`a1_d1_hfmatch`, `_hyperparam_note`). Where PAPER and CODE disagree, PAPER wins for
the values the paper states; CODE fills the rest.

## Setting by setting

| Setting | Official MuAt | Ours (20 Aug run, arms A to D) | Status |
|---|---|---|---|
| Optimiser | SGD, momentum 0.9 (CODE `optim.SGD(..., momentum=0.9)`) | SGD, momentum 0.9 | match |
| Learning rate | 6e-4 (PAPER); ledger recipes use 1e-3 or 6e-4 | 6e-4 | match (PAPER) |
| Batch size | 1 (PAPER); ledger recipes use 2 or 4 | 1 | match (PAPER) |
| Epochs | 150 (PAPER); ledger recipes use 100 | 150 budget | match (PAPER) |
| Gradient clipping | none. `TrainerConfig.grad_norm_clip=None`, never set by `core.py` | none | match |
| Weight decay | 0.001 (CODE `TrainerConfig` default, not overridden). PAPER does not state it | 0.0 | **deviation** |
| LR schedule | constant. Scheduler exists only when `patience>0` (ReduceLROnPlateau) | cosine annealing over 150 epochs | **deviation** |
| Early stopping | none by default; all epochs run | patience 25 on inner validation | **deviation** (compute shortcut) |
| Checkpoint selection | best validation **accuracy** across all epochs (`trainer.py`, `if test_acc > self.global_acc`) | best inner-validation **balanced accuracy** | **deviation** (metric) |
| Selection split | the held-out test split itself (LEDGER d1 `_split_note`: "the test set informed epoch selection") | nested inner split, test untouched | deviation, deliberate, ours is stricter |
| Architecture | one fixed config per run; ledger: n_emb 128, n_layer 1, n_head 1 (TCGA hg38 demo: n_head 8) | per-fold search over 18 candidates (embed 128/256/512 x layers 1/2/4 x heads 1/2), ranked on 3 epochs | **deviation** |
| Mutation cap per tumour | 5000 (LEDGER `mutation_sampling_size`) | 5000 | match |
| Token dictionaries | exact one-hot (PAPER) | exact since Aug 2026 (arms C, D, cancer typing); hashed in arms A, B only | match (current) |
| Motif grammar | bounded three-symbol windows (PAPER) | arm D and faithful config: yes; 20 Aug cancer typing run: legacy allele-string grammar | partial |
| Strand attribute | four-class transcriptional (PAPER) | arm D and faithful config: yes; 20 Aug cancer typing run: legacy +/-/0/na | partial |
| Training data | PCAWG whole genome | TCGA exome, from scratch | unavoidable |
| SV / MEI modalities | present | absent (not callable from WES MAF) | unavoidable |
| Pretrained weights | published | not used (WGS to WES token mismatch) | deliberate |

## What this means for the fold 3 rerun

The 20 Aug cancer typing run had one fold (fold 3) whose searched architecture (256-dim, 4-layer)
diverged at full budget (refit loss 1.658; other folds 0.003 to 0.020). Two things were
considered to stabilise it:

1. **Gradient clipping.** Rejected. Official MuAt does not clip. Adding it would stabilise
   training by departing from the published recipe, which is the opposite of what the comparator
   is for.
2. **A guard in the architecture-search harness.** Adopted. Official MuAt has no search, so this
   guard has no counterpart to be unfaithful to. It excludes a candidate whose inner-pass loss
   explodes (non-finite, or more than 0.5 nats above its own minimum after a 3-epoch warm-up) and falls back
   to the next-ranked candidate. It cannot alter any run in which no candidate diverges, which
   is why folds 1, 2, 4, 5 are reused unchanged.

Config: `config/experiment_settings.muat_top20_fold3_rerun.yaml`. Everything else in that config
is identical to the 20 Aug run, including the four deviations above, so fold 3 remains comparable
to the other four folds.

## What this means for a "properly trained" MuAt

The honest answer to "train MuAt properly" is not a bigger search. It is the official protocol:
one fixed architecture, constant LR, weight decay 0.001, all 150 epochs, best epoch by validation
accuracy, faithful token grammar. That is
`config/experiment_settings.muat_top20_faithful_fixed_arch.yaml`. It removes every deviation in
the table above except the three marked unavoidable or deliberate. No existing checkpoint matches
it, so all five folds train (roughly 15 to 25 h on an RTX 3080).

## Results (4 October 2026, all runs complete)

### B, faithful fixed architecture: complete

Run `20261002T202425Z`, 18.5 h, all five folds trained under the official protocol
(128 / 1 layer / 1 head, constant LR 6e-4, weight decay 0.001, 150 epochs, best epoch by
validation accuracy, arm D token grammar). No fold diverged; refit losses 0.24 to 0.78.

| Protocol | Accuracy | Balanced accuracy | Top-5 | Macro F1 |
|---|---|---|---|---|
| 20 Aug search protocol (fold 3 diverged) | 0.5703 | 0.5502 | 0.8711 | 0.5525 |
| **B: faithful fixed architecture** | **0.5485** | **0.5282** | **0.8607** | **0.5269** |
| Published MuAt, TCGA WES, 7,352 tumours | 0.641 | not reported | 0.906 | not reported |
| Tabular: Signatures + Bio MAF, XGBoost | 0.744 | 0.738 | 0.955 | |

Per fold (accuracy, the selection metric): 0.570, 0.543, 0.544, 0.547, 0.538. Fold 3 is
unremarkable under this protocol (0.544), which confirms the 20 Aug fold-3 collapse was an
artefact of the searched 256-dim / 4-layer architecture, not of the data split.

Reading: the faithful protocol scores about 2 points below the search protocol and 9 points
below the published MuAt number, which was obtained on PCAWG-pretrained weights and more
tumours. Neither MuAt protocol approaches the tabular baseline. The sample-efficiency
interpretation in Aaron's email ("if MuAt still doesn't beat 74%, that's a meaningful result")
holds under both protocols.

### A, fold 3 rerun: first attempt failed, guard redesigned, second attempt succeeded

Run `20261002T202415Z`, 7.2 h. The v1 guard used a 3x ratio on training loss and failed in
both directions on fold 3:

- Candidate 5 (128 / 4 / 1) won the 3-epoch ranking and trained healthily to validation 0.576
  at epoch 110 with loss 0.0006. At epoch 118 one epoch read 0.0019, ratio 3.2, and the guard
  killed it. Absolute change: 0.001 nats.
- Fallback went to candidate 11 (256 / 4 / 1), the same architecture that diverged on 20 Aug.
  It diverged again: loss 1.55 (epoch 7) to 2.67 (epoch 11), validation at chance (0.05). The
  ratio rule never tripped because 20-class cross-entropy saturates near ln 20 = 3.0, below
  3 x 1.55.

Fold 3 therefore came out at 0.390 balanced accuracy (vs 0.358 on 20 Aug); pooled 0.5565.
This is a second independent replicate of the divergence, not a fix, and is kept on disk under
`*_cancer_type_top20_comparable_fold3_rerun_*` for the record.

The rule is now an additive margin: diverged when loss rises more than 0.5 nats above its own
minimum. On the two logged trajectories this gives 0.001 (no trip) vs 1.1 (trip at epoch 10).
`scripts/test_muat_divergence_guard.py` replays both as acceptance tests.

**v2 result (run 20261003T202933Z, 7.1 h).** Candidate 5 (128 / 4 / 1) won the ranking and this
time survived its full-length pass: best epoch 80, validation 0.556, 105 epochs run, refit loss
0.011. The guard never tripped. Fold 3: 0.558 balanced accuracy (was 0.358). Pooled search
protocol: **0.602 accuracy / 0.590 balanced / 0.886 top-5**, up from 0.570 / 0.550 / 0.871.
Folds 1, 2, 4, 5 are the 20 August checkpoints unchanged. Tables under `*_fold3_rerun_v2_*`.

### HRD, faithful fixed architecture: complete

Run `20261003T205844Z`, 3.0 h. HRD_Score **0.484** Spearman, hrd_binary_33 **0.736** AUROC. No
fold diverged. These sit inside the arm A to D range (0.477 to 0.519 / 0.703 to 0.758), so six
HRD runs spanning every fidelity dimension we can vary agree to within fold noise. The tabular
baseline is 0.767 / 0.901.

## Open decisions for the group

1. Which protocol the manuscript reports: search-based with fold 3 repaired (0.602 / 0.590, four
   recipe deviations, directly comparable to the HRD arms), faithful fixed-architecture
   (0.549 / 0.528, verified against the official trainer), or both with the six-point gap
   attributed to per-fold architecture tuning. Recommendation: faithful in the main text, search
   in the supplement.
2. Whether the HRD column moves to the faithful protocol (0.484 / 0.736) so the whole MuAt column
   is one protocol. The run is done and clean; this is now a one-line manuscript edit.
3. Whether to state weight decay as 0.001 "per the official software" or leave it unresolved,
   given the 2023 value is not recoverable.

## Code changes made 2 Oct 2026

- `src/runners/run_muat_style_tcga_comparator.py`
  - `_loss_diverged`, `_ranked_finite_candidates`: pure decision rules, unit-tested.
  - `divergence_loss_margin`, `divergence_warmup_epochs`, `architecture_fallback_attempts`
    settings; default off; added to the checkpoint fingerprint only when on, so no prior
    fingerprint changes.
  - `force_retrain_folds`: bypasses the cache for listed folds and archives the superseded
    checkpoint files as `*.superseded_<utc>` rather than deleting them.
  - Compatible-fingerprint reuse no longer compares a searched fold's selected architecture
    against the base settings (it could never match); split, endpoint and fold are still checked.
  - Fold rows now record `train_loss_min`, `refit_diverged`, `divergence_loss_margin`,
    `diverged_candidate_ids`, `fingerprint`, `fold_trained_utc`.
  - A comment at the optimiser step records why there is no gradient clipping.
- `scripts/test_muat_divergence_guard.py`: 19 checks, including a replay of a fold-3-like loss
  trajectory and a static assertion that `clip_grad_norm_` is not called anywhere.
- `config/experiment_settings.muat_top20_fold3_rerun.yaml` (A) and
  `config/experiment_settings.muat_top20_faithful_fixed_arch.yaml` (B).
