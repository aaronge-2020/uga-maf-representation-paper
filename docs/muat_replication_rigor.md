# MuAt Reproduction And Comparator Rigor

## Current Position

This repository does not claim an official MuAt reproduction unless the official MuAt CLI and checkpoints are configured and run. The default manuscript claim is a MuAt-compatible TCGA-WES event-bag comparator evaluated on the same held-out folds as the corresponding tabular benchmark.

The strict manuscript profile currently emphasizes `HRD_Score` and `hrd_binary_33` because those MuAt-compatible rows are measured against canonical tabular folds. Dedicated configs still support the fixed `cancer_type_top20` comparison; that path uses 20 TCGA classes and balanced accuracy as the primary metric.

## Paper Reference

The MuAt paper's TCGA-WES reference point is 7,352 tumours across 20 tumour types, with 64.1% top-1 accuracy and 90.6% top-5 accuracy. It used MuAt's official mutation-attention training framework and paper-specific TCGA exome inclusion criteria.

## Fixes Now Enforced

| Issue | Why it matters | Current handling |
|---|---|---|
| Local model is not official MuAt | Reviewers can object to calling it a replication | Results are labelled `MuAt-compatible reimplementation` unless official MuAt CLI/checkpoints are available |
| Pooling mismatch | The source model used attention-weighted set pooling, while the old local code used masked mean pooling | `MuAtCompatibleModel` now uses learned attention-weighted set pooling and reports those weights in the attention summary |
| Fixed architecture | A single 128-dimension, one-layer, one-head architecture could understate architecture sensitivity | Full configs run a limited inner-split search over embedding sizes 128/256/512, layer counts 1/2/4, and heads 1/2 |
| Metric drift | Cancer-type text previously mixed accuracy, macro-AUROC, and balanced accuracy | `cancer_type_top20` uses balanced accuracy as the primary manuscript metric; accuracy/top-5 remain paper-facing comparability metrics |
| Vocabulary leakage risk | Learned token dictionaries could be influenced by held-out samples | Manuscript configs use fixed-hash motif, position, and annotation dictionaries |
| Biological feature mismatch | The paper is clear about sequence-context style encodings but less explicit about all auxiliary annotation tokens | The local event bag uses deterministic MAF/VEP-derived motif, 1-Mb position-bin, and genic/exonic/strand tokens; broader external-resource Bio MAF v4 annotations are tabular features, not MuAt inputs |
| Split comparability | Different folds can obscure whether model differences are real | Full manuscript comparator runs use canonical folds from `results/tables/main_manuscript_complete_panel_oof_predictions.csv` where those rows are available |

## Reviewer-Facing Framing

Use: "We implemented a MuAt-compatible TCGA-WES event-bag comparator with attention-weighted set pooling and limited architecture search, then evaluated it on the same held-out folds as the tabular benchmark where canonical splits were available."

Avoid: "We reproduced MuAt" unless the official package/checkpoints are used under a predeclared protocol.

## Acceptance Checks

- `pooling_mode` is `attention_weighted`.
- Architecture-search configs include embedding sizes `128, 256, 512`, layers `1, 2, 4`, and heads `1, 2`.
- `cancer_type_top20` reports 20 classes with balanced accuracy as the primary metric.
- Fixed-hash dictionaries are used for manuscript comparisons.
- The fidelity report states that the local model is not official MuAt unless the official CLI is available.
