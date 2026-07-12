# MuAt Comparator — Change Log (Vijay)

**Date:** 11 July 2026
**Branch:** `codex/coxnet-muat-manuscript-updates` (base commit `38f507c`)
**Scope:** MuAt-compatible comparator fidelity + rerun enablement. No changes to the tabular panel, the CoxNet survival path, or the manuscript text.

Reference throughout: Sanjaya P, Maljanen K, Katainen R, et al. *Mutation-Attention (MuAt)*. Genome Medicine 2023;15:47 — Methods, "Preparing MuAt inputs from somatic variant callsets" and "MuAt hyperparameter search, model training and validation."

---

## 1. BLOCKING — final model was training for ≤ 3 epochs, not 150

**Severity: this invalidates any MuAt rerun launched from `38f507c`.**

### What was wrong

`architecture_search_epochs: 3` is set in all three manuscript configs. That value was doing double duty: it capped the architecture *ranking* budget (intended) **and** became the *final training* budget (not intended).

```python
# run_muat_style_tcga_comparator.py, before
architecture_search_epochs = max(1, min(int(local.epochs), 3))   # -> 3
...
selected_epoch = int(best_candidate_epoch)          # best epoch seen in the 3-epoch search: 1, 2 or 3
scheduler      = make_scheduler(optimizer, selected_local, selected_epoch)   # cosine anneals to 0 in <= 3 epochs
for epoch in range(start_epoch, selected_epoch + 1):                         # final model trains <= 3 epochs
```

`epochs: 150` was reachable **only** when the architecture search was disabled (`selected_epoch = int(local.epochs)`). With the search on — which it is in every manuscript config — the configured 150 was dead. MuAt's Methods are explicit: *"we set the learning rate to 6×10⁻⁴, momentum to 0.9, and minibatch size to one, training for 150 epochs."*

### Why the current numbers are still safe

The MuAt-compatible results already in `results/tables/` were produced **before** the architecture search existed, so they ran the full 150 epochs and are unaffected. The bug only bites on the *next* run — i.e. the full rerun that is the last item blocking submission.

### Fix

The two budgets are now separate. The 18 candidates are still ranked over `architecture_search_epochs`; the **winner is then retrained at full length on the same inner split** to select its true stopping epoch, and only then refit on the outer-training fold.

- Extracted the inner-training body into a single `run_inner_pass(model_settings, n_epochs, ...)` used by both stages.
- New setting `final_epoch_selection: true` (default on). Skipped automatically when `architecture_search_epochs >= epochs`, so behaviour is unchanged for configs that never had the bug.
- The LR schedule now uses `T_max = epochs` in **both** the selection pass and the outer refit. Previously the stopping epoch was chosen under one LR trajectory and the refit followed a different one.

### Residual issue — *not* fixed, needs a decision

Even with the refit repaired, the *architecture* is still ranked on 3-epoch proxies. MuAt trained **every** hyperparameter combination to completion inside each fold. Three epochs of batch-size-1 SGD measures early convergence speed, which is anti-correlated with model size — and MuAt's own selected model was embed 512 / 2 layers / 1 head, the kind of model that looks worst at epoch 3. See §5.

---

## 2. Token dictionaries — hashing replaced with exact vocabularies

### What was wrong

`dictionary_mode: fixed_hash` hashed each modality into **fewer buckets than the vocabulary contains**:

| Modality | MuAt vocabulary | Buckets used | Tokens colliding |
|---|---|---|---|
| Motif (SNV+MNV+indel) | 3,426 of MuAt's 3,692 | 1,024 | **~96.5%** (~3.5 motifs per embedding) |
| Position (1-Mb bins) | 2,915 | 2,048 | **~75.9%** |
| Annotation | 16 | 256 | 0% (240 unused embeddings) |

Distinct mutation motifs, and unrelated genomic regions, were sharing single embedding vectors. MuAt's Methods specify exact one-hot dictionaries at those sizes.

The stated justification (`docs/muat_replication_rigor.md`: *"learned token dictionaries could be influenced by held-out samples"*) does not apply: 1-Mb bins are a property of the GRCh37 assembly, and the annotation vocabulary is a property of the genic × exonic × strand grammar. Neither depends on which tumours are in a fold.

### Fix

New `dictionary_mode: exact` (now set in all three manuscript configs):

- **Position** — enumerated from the GRCh37 assembly (`enumerate_position_tokens`, 3,114 bins across chr1-22, X, Y, MT). Reference-derived; label-free; zero collisions.
- **Annotation** — enumerated from the 16-value grammar (`enumerate_annotation_tokens`). Exactly matches MuAt's 2×2×4.
- **Motif** — one slot per distinct observed token, no hashing.

On the motif vocabulary specifically: it is collected from the event table without reference to any label. A motif seen only in a held-out fold receives a randomly initialised embedding that no gradient ever reaches, which is behaviourally identical to `<UNK>`. No label information crosses a fold boundary.

`fixed_hash` is retained and still selectable, for a sensitivity arm if we want one.

**Known gap:** our motif *grammar* still differs from MuAt's. They build 3-symbol windows over a 25-symbol alphabet (4 bases + 6 substitutions + 4 deletions + 4 insertions + SV/MEI symbols) giving exactly 3,692 tokens; we emit `left[ref>alt]right` strings with allele text truncated to 12 characters, which is an open-ended vocabulary. Worth doing properly, but it is a bigger change than this pass.

---

## 3. Event truncation no longer deletes half the genome

Tumours above the 5,000-event cap were truncated with `sort_values(["chromosome", ...]).head(max_events)`. `chromosome` is a **string**, so the sort is lexicographic: 1, 10, 11, … 19, 2, 20, 21, 22, 3, 4, … 9, X, Y. Taking the head therefore kept chromosome 1 and the teens and systematically discarded chromosomes 3–9, X and Y.

This only affects tumours above the cap — which are the POLE-proofreading and MSI hypermutators, heavily enriched in COAD and UCEC, and exactly the samples whose regional mutation distribution carries the most tumour-type signal.

Replaced with a deterministic per-sample RNG (seeded from `seed` + sample barcode), so the subsample is reproducible but unbiased. `encode_events` now reports how many tumours were down-sampled — **we have never actually counted this**, and the number should go in the manuscript.

---

## 4. Dataset bundle now comes from Hugging Face

New: `scripts/fetch_datasets.py`.

```bash
python scripts/fetch_datasets.py --with-reference --with-event-bags
python scripts/reproduce_manuscript.py --datasets-dir ../hf_bundle/github_exports_datasets --strict
```

Replaces the hand-shared `../github_exports_datasets` Drive folder with `vjdeara/cgr_tensorlab_umsom`. Two things it has to fix up, because neither artefact sits where the code looks for it:

1. **`main_manuscript_complete_panel_oof_predictions.csv` is in `.gitignore`.** It is therefore absent from every clone — yet all three MuAt configs set `require_canonical_split: true` and point at `results/tables/…`. **The MuAt comparator cannot run from a fresh checkout.** The bundle carries it under `results/large_tables/`; the script stages it into place. This is very likely a reason the full rerun has never completed for anyone.
2. **The GRCh37 FASTA is published at the dataset root as `GCF_000001405.13_GRCh37_genomic-002.fna`** — the `-002` is a Google Drive duplicate-name artefact. The pipeline globs `*.fna` under `references/grch37`, so the script places it there under its canonical name with the `.fai` index alongside.

The bundle also carries the **precomputed MuAt event bags** (`--with-event-bags`), which skips the expensive MAF → event-table build.

---

## 4a. The attention-weighted pooling has never actually run on a GPU

`MuAtCompatibleModel.forward` (added in `38f507c`, the commit that introduced attention pooling):

```python
pool_logits = self.pool_score(z).squeeze(-1).masked_fill(~mask, torch.finfo(z.dtype).min)
```

Under `torch.autocast`, `LayerNorm` returns float32 while `nn.Linear` returns float16. So `z` is float32 but `self.pool_score(z)` is float16, and the mask fill value is taken from the wrong tensor's dtype: it attempts to write `-3.4e38` (float32 min) into a float16 tensor.

```
RuntimeError: value cannot be converted to type at::Half without overflow
```

Every manuscript config sets `amp: true`, and `use_amp` is true whenever the device is CUDA. **This line therefore raises on the first training step of any GPU run.** It can only execute with AMP disabled, which in this codebase means CPU only.

The implication for the manuscript is direct: the attention-weighted set pooling described in Methods, and reported in the change log as smoke-tested, has never completed a training step in the configured environment. Whatever smoke test was run must have been on CPU.

Fixed by computing the pooling logits in float32, which resolves the dtype mismatch and is also the right place numerically for a softmax over up to 5,000 masked events.

---

## 4b. Robustness and memory (second pass)

### Dataset-path resolution — a whole class of silent failure

`run_all_experiments.py` had **no `--datasets-dir` flag at all**. It depended entirely on `CGR_DATASETS_DIR`, and `validate_environment()` recorded missing paths as `"missing"` but only *aborted* under `--dry-run` (`if blocking and args.dry_run`). So a wrong bundle path was logged, ignored, and the run continued until a runner died inside pandas with `FileNotFoundError` minutes later.

- Added `--datasets-dir` to `run_all_experiments.py`. It is applied *before* `utils.config` is imported, because `DATASETS_ROOT` is computed at import time and anything argparse produced would arrive too late.
- The run now prints the resolved `datasets_root` up front and **aborts immediately** if the bundle or `mc3_source_dir` is absent. Endpoint-dependent paths (kucab, pcawg, grch37) remain advisory, so a MuAt-only run does not require the full bundle.
- `run_muat_ablation.py` resolves the bundle location and passes `CGR_DATASETS_DIR` into each arm's subprocess, and its preflight verifies the MC3 MAF and HRD cohort actually exist rather than only checking the split manifest.

### Stale-checkpoint reuse would have silently undone the epoch fix

The fold-checkpoint fingerprint (`_fit_predict_local`) included the model settings, the architecture grid, `architecture_search_epochs` and the dictionary digest, but **not `final_epoch_selection`**. Checkpoints are reused whenever the fingerprint matches and `resume_checkpoints` is on, which is the default and is set in all three manuscript configs.

So a checkpoint written by the pre-fix code (final model trained for <= 3 epochs, hashed dictionaries) hashed **identically** to a post-fix run with otherwise identical settings. The fixed rerun would have loaded the buggy fold results from disk, skipped training entirely, and reported the old undertrained numbers as if they were the fix. Silent, and it would have looked like the fix simply did nothing.

Added `final_epoch_selection`, a `training_protocol` version tag, and an `event_truncation` version tag to the fingerprint, so any pre-fix checkpoint is now incompatible by construction.

**Action for anyone rerunning:** delete `results/cache/features/muat_style_tcga_comparator/` before the first post-fix run, as belt and braces.

### Two further bugs in my own tooling

- `run_muat_ablation.py` resolved the dataset-bundle path **at import time**. On a machine where the bundle does not exist yet, it bound to the legacy sibling path, `fetch_datasets.py` then created `../hf_bundle/`, and the constant never updated -- so preflight failed *after* a successful download. Now resolved on every call.
- `cancer_type_top20` at n=8,800 and minibatch size 1 is ~7,000 optimiser steps per epoch. A strict 150-epoch selection sweep plus the refit is ~12M steps, which does not finish overnight on one consumer GPU. Added `early_stopping_patience: 25` to that config only. This is the same mechanism the tabular panel already uses ("XGBoost used seeded candidate search with inner-validation early stopping"), and it cannot change the selected epoch unless the validation score improves again after 25 consecutive epochs of no improvement. Set to 0 for a strictly faithful full sweep.

### Event-table cache was rebuilt unnecessarily

`_cache_key` keyed the cached event table on `dictionary_mode` and the hash-bucket sizes. The cached artefact is produced by `build_muat_event_table`, which emits motif/position/annotation as plain strings *before* any dictionary is applied. Changing only the dictionary therefore forced a full re-parse of the 718 MB MC3 MAF to produce a byte-identical table. Removed those keys, so the three ablation arms share one event-table build.

### Memory

| Change | Effect |
|---|---|
| Token tensor `int64` → `int32` | The single largest allocation. `cancer_type_top20` (8,800 × 5,000 × 3): **1.06 GB → 528 MB**, in host RAM and again in VRAM under `preload_tensors_to_device`. `nn.Embedding` accepts `IntTensor`. `.long()` casts in the runner were forcing int64 back regardless of the numpy dtype. |
| `build_muat_event_table` chunking | Was accumulating one Python dict per mutation (~3.4M dicts) before converting. Now materialises a DataFrame per chunk, bounding interpreter overhead to a single chunk. |
| `gc.collect()` before `torch.cuda.empty_cache()` | `empty_cache()` only frees blocks torch already considers unreferenced, so calling it while the model object is alive is a no-op. Added at both the architecture-candidate boundary (18 per fold) and the fold boundary, where the refit model was previously never released before the next fold's search began. |

---

## 5. Raised, deliberately NOT changed — needs the group's decision

| # | Issue | Why I didn't just fix it |
|---|---|---|
| 1 | **Architecture ranked on 3-epoch proxies.** MuAt trained every candidate to completion per fold. Ours ranks 18 candidates over 3 epochs. | The faithful fix (18 × 5 folds × 150 epochs) is expensive. Options: (a) do it properly; (b) run the full grid at full length on fold 1 only and fix the architecture, stating so in Methods; (c) adopt MuAt's own published pick (embed 512 / 2 layers / 1 head) and drop the search. All three are defensible; a 3-epoch search described as "architecture search" is not. |
| 2 | **Attention-weighted pooling is asserted, not sourced.** The MuAt paper never specifies pooling — only *"the third module combines the mutation features with fully connected layers."* The change log claims the source model used attention-weighted set pooling. | Needs checking against the official implementation (the paper cites `github.com/primasanjaya/mutation-attention`, commit `3f2d561`). Either cite it, or describe attention pooling as our design choice. As written the Methods sentence may simply be false. |
| 3 | **MuAt is reported only on HRD.** The tumour-typing result exists (accuracy 0.539 vs MuAt's published 0.641 vs our tabular 0.744) and is excluded from the manuscript's active claims. | Scientific call, not mine to make alone. See the separate audit memo, §7. |
| 4 | **CoxNet is fit one alpha at a time** (`alphas=[alpha]`, `nested_oof.py`). scikit-survival warns this is unstable; the recommended pattern is one fit per `l1_ratio` along the full regularisation path, selecting alpha on the inner split. | Cindy owns survival. Flagging only. Note the OS C-index moved a lot with the CoxNet swap (MAF-stack 0.491 → 0.632). |
| 5 | **`damage_class` metric drift.** `endpoint_results` reports `macro_auroc`; manuscript Table 1 calls the primary metric balanced accuracy. The cleanup was applied to `cancer_type_top20` but not Kucab. | One-line fix, but it changes a reported table — Aaron should decide. |

---

## 6. Repository hygiene (found while getting the rerun to work)

- **Windows clone failure explained.** The longest tracked path is 249 characters (`results/tables/main_manuscript_complete_panel_nested_fold_checkpoints/candidate_rows/cancer_type_top20__signatures_plus_MAF_stack__xgboost__…`). Plus a typical checkout prefix that exceeds Windows' 260-character `MAX_PATH`. This is why the June clone failed — not permissions. Workaround: `git config core.longpaths true`. Real fix: hash or shorten those checkpoint filenames.
- **LFS.** `git lfs pull` is required after clone or several tracked assets are 130-byte pointer files.

---

## 7. Discard inventory — proposed, pending Aaron's approval

Nothing has been moved or deleted. Reachability computed from the three entry points (`src/run_all_experiments.py`, `scripts/reproduce_manuscript.py`, `scripts/reviewer_workflow.py`), following the dynamic `RUNNERS` dispatch in `run_all_experiments.py`.

### Safe to discard — dead code, no importers, superseded

| File | Lines | Note |
|---|---|---|
| `src/utils/biological_maf_features.py` | 799 | **Nothing imports it.** Superseded by `bio_maf_base_features.py` + `quick_bio_v4_nested_feature_selection.py` (the "Bio MAF v4" stack that the manuscript actually uses). The only genuinely dead module in the tree. |
| `scripts/rerun_survival_cox.py` | 284 | One-off script from the CoxNet swap. The main panel runner now uses CoxNet directly, so this has served its purpose. Keep only if we want it as provenance for the survival rerun. |

### Do NOT discard — standalone tools, not dead

These are unreachable from the automated pipeline but are working CLIs invoked by hand. Listing them so nobody deletes them by mistake in a future cleanup:

| File | Lines | Why keep |
|---|---|---|
| `src/utils/build_submission_docx.py` | 784 | Builds `submission_manuscript.docx`. Actively used. |
| `src/utils/analyze_bio_maf_v4_interpretability.py` | 510 | Standalone interpretability CLI; produces manuscript supplement outputs. |
| `src/utils/check_benchmark_completion.py` | 339 | Completion-gate CLI, invoked by path from `run_benchmark_phases.py`. |
| `src/utils/run_benchmark_phases.py` | 181 | Phase orchestrator CLI. Possibly superseded by `run_all_experiments.py` — worth asking Aaron. |
| `src/utils/clean_generated_outputs.py` | 166 | Output cleanup CLI. |

**Proposal:** move only the two "safe to discard" files into `discard/`, keep them in git history, and delete in a later commit once we're confident. Everything else stays.

---

## 8. How to rerun

```bash
# 1. bundle
python scripts/fetch_datasets.py --with-reference --with-event-bags

# 2. cancer-type comparison (the one with a published MuAt number to compare against)
python src/run_all_experiments.py \
    --config config/experiment_settings.muat_top20_comparable.yaml \
    --paths config/paths.yaml

# 3. HRD endpoints (small; ~772 samples, good for a fast before/after check)
python src/run_all_experiments.py \
    --config config/experiment_settings.strict_no_leakage.yaml \
    --paths config/paths.yaml
```

**Suggested first run for the meeting:** HRD only, twice — once with `final_epoch_selection: false` (reproduces the ≤3-epoch bug) and once with it on. Same folds, same seed. That gives a hard number for what the bug costs, which is a far better artefact than a cancer-type run we cannot finish in time.

Expected cost at batch size 1 (MuAt's own setting): HRD (n=772) is minutes per fold. `cancer_type_top20` (n=8,800) is ~7,000 optimiser steps per epoch × 150 epochs × 5 folds — hours on a GPU, and now *more* than before, because the model finally trains for its full budget.
