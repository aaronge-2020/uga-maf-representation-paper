# Revision Analyses Added For Current Manuscript Critiques

This revision adds reproducible analyses for the reviewer concerns about WES sparsity, fixed hyperparameters, binary survival modeling, KME/signature interpretation, deep-learning comparators, and unstable Kucab multiclass metrics.

## Evaluation Protocol

All model-based manuscript endpoints now use five outer folds with one inner validation split inside each outer-training fold. Hyperparameters are selected on the inner validation set only, then a single final model is retrained on the full outer-training fold. Primary scores are computed once from pooled out-of-fold predictions across the cohort.

## Survival

The previous binary survival-event shortcut is replaced by TCGA CDR Cox survival endpoints (`OS` and `PFI`) in the strict manuscript configuration. Survival models report Harrell's C-index from pooled OOF scikit-survival CoxNet risk scores.

## Sparsity Control

The Kucab low-burden analysis now includes a 41-mutation exome-like budget, with additional 20, 100, and 200 mutation sensitivity budgets. It uses balanced accuracy as the multiclass primary metric and also reports macro-AUROC, micro-AUROC, macro-F1, and Cohen's kappa.

## MuAt-Style Comparator

The pipeline includes a TCGA-WES MuAt-compatible attention MIL comparator trained from bundled MC3 mutation bags. The active manuscript comparison is HRD-only and uses `HRD_Score` plus `hrd_binary_33` on the same canonical outer folds as the Bio MAF v4 `signatures_plus_MAF_stack` XGBoost result for each endpoint. The comparator uses fixed-hash motif, 1-Mb position, and annotation dictionaries so held-out samples do not change the token vocabulary; it now uses attention-weighted set pooling and limited inner-split architecture search over embedding size, layer count, and attention-head count.

The original MuAt paper reports TCGA whole-exome performance on 7,352 tumours across 20 tumour types, with 64.1% top-1 accuracy and 90.6% top-5 accuracy. Because the bundled comparator does not use the official MuAt package/checkpoints and our fixed matched-feature endpoint has a different patient set, manuscript text must call the local result `MuAt-compatible reimplementation`, not official `MuAt` replication. Accuracy and top-5 accuracy are reported for the paper-facing comparison; macro-AUROC remains available for the manuscript's cross-endpoint score table.

## Dataset Scope

The default manuscript run remains offline from bundled MC3, TCGA CDR, TCGA-BRCA HRD, and Kucab data. GDC curation helpers are outside the active source tree and are not part of the GitHub-ready reproduction path.
