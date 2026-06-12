# MuAt Replication And Comparator Rigor

## Current Answer

We have not replicated the official MuAt TCGA-WES result. The active local output is a same-fold MuAt-compatible comparator for the fixed 20-class TCGA endpoint, not the official MuAt paper pipeline.

The Bio MAF v4 top-20 manuscript benchmark is complete and strong: `signatures_plus_MAF_stack` with XGBoost reaches 0.7436 accuracy, 0.9091 top-3 accuracy, 0.9551 top-5 accuracy, and 0.9758 macro-AUROC on 8,800 patients across 20 fixed TCGA classes.

## Paper Reference

The MuAt paper's TCGA-WES reference point is 7,352 tumours across 20 tumour types, with 64.1% top-1 accuracy and 90.6% top-5 accuracy. It used MuAt's official mutation-attention training framework and paper-specific TCGA exome inclusion criteria.

## Why Our Prior MuAt Output Was Not Comparable

| Issue | Why it matters | Fix now enforced |
|---|---|---|
| Endpoint mismatch | A narrower historical cancer-type task is not comparable to the paper's 20-type exome result | Active output tag is `cancer_type_top20_comparable` and uses the fixed manuscript class list |
| Local model is not official MuAt | Reviewers can object to calling it a replication | Results are labeled `MuAt-compatible reimplementation` unless official MuAt CLI/checkpoints are configured |
| Different patient set | Paper used 7,352 TCGA exomes; manuscript endpoint uses 8,800 matched-feature patients | Report the difference explicitly and compare as same-fold local comparator |
| Vocabulary leakage risk | Learned token dictionaries could be influenced by held-out samples | Use fixed-hash motif, position, and annotation dictionaries |
| Metric mismatch | Paper headline is accuracy, while manuscript often uses macro-AUROC | Select MuAt epochs by inner-fold accuracy and report accuracy/top-5 alongside macro-AUROC |
| Split mismatch | Different folds can obscure whether model differences are real | Use canonical `cancer_type_top20` folds from the Bio MAF v4 main benchmark |

## Reviewer-Facing Position

This should not be framed as "we reproduced MuAt." The defensible framing is: "We implemented a MuAt-compatible TCGA-WES event-bag comparator and evaluated it on the same held-out folds as the tabular Bio MAF benchmark. This directly tests whether a mutation-attention event model dominates our feature representation under our endpoint definition, while avoiding claims about the official pretrained MuAt model."

The stronger claim available now is that Bio MAF v4 exceeds the MuAt paper's quoted TCGA-WES accuracy reference on our fixed 20-class matched-feature endpoint. That is encouraging, but it is not a head-to-head official MuAt comparison until the official MuAt pipeline is installed and run under a predeclared protocol.

## Acceptance Criteria For The Rigorous Local Comparator

- Endpoint is exactly `cancer_type_top20`.
- Result row has 8,800 samples and 20 classes.
- Outer folds come from `results/tables/main_manuscript_complete_panel_oof_predictions.csv` for `cancer_type_top20`, `signatures_plus_MAF_stack`, `xgboost`.
- Token dictionaries use `dictionary_mode: fixed_hash`.
- Epoch selection uses inner-fold accuracy.
- Final row reports accuracy, balanced accuracy, macro-F1, macro-AUROC, micro-AUROC, top-3 accuracy, top-5 accuracy, calibration, and paper-reference columns.
- The fidelity report states that this is not official MuAt replication unless the official CLI status is available.
