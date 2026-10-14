# Task 2 notes: one inferential procedure for the headline comparisons

Question: which headline differences have valid intervals, including the HRD42
primary endpoint, which currently has none?

## Method as run

Paired bootstrap over tumours on SAVED out-of-fold predictions. Tumours
resampled with replacement, 10,000 replicates. Each contrast used its own
`numpy.random.default_rng(42)` stream (i.e. every contrast's replicate sequence
starts from seed 42). In each replicate the metric was recomputed from that
model's own out-of-fold predictions on the resampled tumours:

- Cancer typing (C1a, C1b): pooled-OOF balanced accuracy per model
  (mean per-class recall over the 20 classes; classes absent from a replicate
  are skipped via nan-mean — 0 of 20,000 replicates had an absent class),
  difference = BA(Sig+MAF) − BA(Bio MAF), same resample indices for both
  models (paired by tumour).
- HRD42 (C4a): fold-mean AUROC for Sig+MAF only — per-fold AUROC on the
  resampled tumours in each fold (folds with a single class in a replicate are
  skipped — 0 of 10,000 replicates had one), averaged over the 5 folds.

Percentile 95% intervals: 2.5th and 97.5th percentiles of the 10,000 replicate
statistics.

Two-sided bootstrap p-value (definition used): let d be the observed point
difference and B the 10,000 bootstrap differences. If d ≥ 0,
p = 2 × fraction of replicates with B ≤ 0; if d < 0,
p = 2 × fraction with B ≥ 0; capped at 1.0. In words: twice the fraction of
replicates at or beyond the null-consistent tail (the tail containing 0 on the
side opposite the observed sign).

Multiple testing: Benjamini-Hochberg over the raw p-values of exactly the
computable contrasts (C1a, C1b). The family was fixed in
results/tables/inference-family.md BEFORE any q-value was computed. Blocked
contrasts are listed with status and do not enter the correction.

## OOF files used per contrast

- C1a: `results/tables/task1_oof_a_sigmaf_full_archived.csv` (Sig+MAF,
  reproduces pooled BA 0.737784) vs
  `results/tables/task1_oof_a_maf_full.csv` (Bio MAF Task-1 control (a) refit,
  pooled BA 0.665743). Same 8,800 tumours, true labels agree; pairing by
  tumour (sample barcode). The two archived outer-split seeds differ (615371 vs
  750978) but pairing is by tumour, so the paired bootstrap is valid.
- C1b: same Sig+MAF file vs
  `results/tables/task1_oof_a_maf_full_archived.csv` (archived Bio MAF OOF,
  pooled BA 0.661830). Sensitivity line.
- C4a: HRD42 Sig+MAF OOF from the nested-fold checkpoint npz
  `results/tables/main_manuscript_complete_panel_nested_fold_checkpoints/hrd_binary_42__signatures_plus_MAF_stack__xgboost__main_complete_panel_mc3_signatures_plus_maf_stack_bio_v4_nested_fs_ac11587791bbbd40db3991bb.npz`
  (`pred` = (772,2) class probabilities; score = P(class 1); `sample_folds` =
  archived fold assignments 1–5). Sample order from
  `~/workspace/task3_inputs/order_hrd_binary_42.csv`; true labels
  (622 negative / 150 positive) from the cohort labels. Validation: per-fold
  AUROC recomputed from this OOF reproduces the archived
  `main_manuscript_complete_panel_fold_metrics.csv` values exactly
  (0.902400, 0.897733, 0.953763, 0.934409, 0.937634); fold-mean 0.925188.

## Results (match task2_inference_results.csv exactly)

| contrast | point | 95% CI | raw p | q (BH) |
|---|---|---|---|---|
| C1a cancer BA, Sig+MAF vs Bio MAF (refit) | +0.072041 | [0.063315, 0.080799] | 0.0 (0/10,000 ≤ 0; p < 0.0001) | 0.0 |
| C1b cancer BA, Sig+MAF vs Bio MAF (archived) | +0.075954 | [0.066996, 0.084998] | 0.0 (0/10,000 ≤ 0; p < 0.0001) | 0.0 |
| C4a HRD42 Sig+MAF fold-mean AUROC | 0.925188 | [0.902291, 0.945974] | — (not a contrast) | — |
| C2 cancer BA vs patterns alone | blocked | — | — | — |
| C3 HRD Spearman vs patterns alone | blocked | — | — | — |
| C4b HRD42 AUROC difference vs patterns alone | blocked | — | — | — |
| C5 HRD33 AUROC difference vs patterns alone | blocked | — | — | — |
| C6 HRD24 AUROC difference vs patterns alone | blocked | — | — | — |

BH over the two computable raw p-values (both 0.0) gives q = 0.0 for C1a and
C1b.

## The HRD42 line (as required)

- Fold-mean AUROC, Sig+MAF: point 0.925188 (reproduces the printed 0.925),
  95% interval [0.902291, 0.945974] (10,000 tumour-resample replicates, seed
  42; per-replicate fold-mean over the 5 archived outer folds).
- Paired difference vs patterns alone (printed +0.010 fold-means: 0.925188 vs
  0.915043): BLOCKED — the archived patterns-alone OOF is unavailable (see
  blockers). No interval and no q-value can be computed for this difference
  from saved OOF predictions.
- Archived per-fold AUROC (Sig+MAF): 0.902400, 0.897733, 0.953763, 0.934409,
  0.937634. Archived per-fold AUROC (patterns alone, fold_metrics only):
  0.899067, 0.890933, 0.922581, 0.922312, 0.940323; fold-mean 0.915043.

## Deviations and why

1. Each contrast used its own `default_rng(42)` stream rather than one shared
   stream across contrasts. Within-contrast reproducibility is exact (fixed
   seed, fixed replicate count); cross-contrast RNG independence is cleaner
   than chaining one stream, and the brief only fixes the seed and replicate
   count, not the streaming scheme.
2. Balanced accuracy in bootstrap replicates uses nan-mean over classes present
   in the replicate (0 replicates affected). AUROC fold-mean skips folds with a
   single class in a replicate (0 replicates affected). Both are the natural
   "defined folds/classes" rule; neither changed any number.
3. Raw p reported as 0.0 (the measured 0/10,000). Displayed as p < 0.0001 in
   text. BH on {0.0, 0.0} yields q = 0.0.
4. The C4a interval uses the npz checkpoint OOF rather than the LFS OOF table,
   because the npz is the archived source that reproduces the printed
   fold metrics exactly (see validation above); the LFS table does not (see
   blockers).
5. No HRD24/HRD33/Sig+MAF-vs-patterns bootstrap was run: blocked (below). Per
   the brief, blocked contrasts stop only their own contrast.

## Blockers (with evidence)

All five patterns-alone contrasts (C2, C3, C4b, C5, C6) are BLOCKED: no
archived out-of-fold predictions exist for the patterns-alone representation
(standard_sbs96_id83, XGBoost) from the archived analysis.

Evidence trail:
1. The only OOF table containing patterns-alone predictions,
   `results/tables/main_manuscript_complete_panel_oof_predictions.csv`, is a
   git-LFS pointer (content absent; git-lfs not installed).
2. ONE recovery attempt was made, via the GitHub LFS batch API over HTTPS
   (POST to the repo's `/info/lfs/objects/batch`), which returned HTTP 200
   with a signed S3 download URL; the 57,960,349-byte object downloaded
   cleanly and its SHA-256
   (84d7472c14383f6f7adac162aa4372e85f7823cdd686294fd03c8f50d6d0e99c) matches
   the LFS pointer exactly. So the fetch itself succeeded.
3. The recovered file is NOT the archived analysis's OOF: it does not
   reproduce any archived metric. Cancer typing: pooled BA from the file is
   0.539880 (patterns alone) and 0.742417 (Sig+MAF) vs archived 0.538148 and
   0.737784; per-sample predicted-label agreement between the file's Sig+MAF
   rows and the archived `task1_oof_a_sigmaf_full_archived.csv` is only
   87.9% (mean max-abs probability difference 0.10). HRD: per-fold Spearman /
   AUROC recomputed from the file's rows do not match
   `main_manuscript_complete_panel_fold_metrics.csv` for any endpoint or
   representation (e.g. HRD_Score Sig+MAF pooled 0.780986 vs archived
   0.766645; per-fold 0.8034/0.7733/0.7760/0.7850/0.7825 vs archived
   0.7679/0.7575/0.7541/0.7863/0.8167). Git history is consistent with this:
   the fold metrics and nested checkpoints come from the 2026-08-20 "NEW RUN"
   commit (fa1c20e); the LFS pointer was added later (2026-09-18, 567de05)
   but its object content predates that run.
4. No other archived source exists: the nested-fold checkpoints
   (`results/tables/main_manuscript_complete_panel_nested_fold_checkpoints/`)
   contain OOF only for MAF_stack_only and signatures_plus_MAF_stack (91
   files; no standard_sbs96_id83 / burden_only entries); no other CSV in
   results/tables/ holds per-tumour patterns-alone predictions.
5. Per the brief, fold-level scores were NOT substituted for the paired
   bootstrap, and Task 0 was NOT rerun.

Consequence: the paired bootstrap — and therefore a valid interval, p-value,
and q-value — cannot be constructed for any Sig+MAF-vs-patterns-alone
headline difference (C2, C3, C4b, C5, C6) from saved OOF predictions. The
printed point differences remain as reported from the archived fold metrics
(cancer BA +0.199636; HRD score Spearman fold-mean +0.082246
[0.776503 − 0.694257]; HRD42 fold-mean +0.010145; HRD33 fold-mean +0.036849;
HRD24 fold-mean +0.021303), but they have no paired-bootstrap interval from
this procedure.

## Reproducibility

- Scripts: `~/workspace/task2_scratch/run_task2_bootstrap.py` (C1a/C1b),
  `~/workspace/task2_scratch/run_task2_c4a.py` (C4a).
- Raw replicate draws: `~/workspace/task2_scratch/bootdiff_C1a.npy`,
  `bootdiff_C1b.npy`, `bootfm_C4a.npy` (10,000 floats each).
- No git operations were performed; no files outside `results/tables/`
  (the three deliverables) and `~/workspace/task2_scratch/` were written;
  the manuscript/Google Doc were not touched.
