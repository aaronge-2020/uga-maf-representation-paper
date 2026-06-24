# CoxNet And MuAt/eMuAT Change Log

## Survival Modeling

- Removed the CoxPH/custom-style survival branch from `src/utils/nested_oof.py`.
- Survival fitting now uses `sksurv.linear_model.CoxnetSurvivalAnalysis` only.
- Survival candidates exclude `l1_ratio = 0`; the grid searches elastic-net CoxNet candidates with positive L1 mixing and an alpha floor of `1e-2` by default.
- CoxNet fits now fail loudly if coefficients or predictions are missing or non-finite.
- Reader-facing labels now describe the survival panel as CoxNet/scikit-survival, while the internal learner key remains `cox_ph` for table compatibility.

## MuAt-Compatible Reimplementation

- Replaced masked mean pooling with learned attention-weighted set pooling in `MuAtCompatibleModel`.
- Added stacked self-attention support through `num_layers`.
- Added limited architecture search over embedding sizes `128, 256, 512`, self-attention layers `1, 2, 4`, and heads `1, 2`.
- The architecture search uses the existing inner validation split and records per-candidate histories in fold metrics.
- The selected per-fold architecture is written to `model_settings_json`; endpoint summaries include candidate-grid and selected-architecture metadata.
- Optimizer settings remain SGD with momentum; config support for a cosine scheduler has been added.
- The 5,000 mutation cap, deterministic event ordering, fixed-hash dictionaries, and held-out-fold leakage prevention are preserved.

## Biological Inputs

- Exact MuAt auxiliary annotation inputs could not be fully reconstructed from the bundled sources.
- The local event-bag model uses deterministic MAF/VEP-derived motif, 1-Mb position-bin, and genic/exonic/strand annotation tokens.
- Broader external-resource annotations, including cancer-gene and hotspot resources, remain part of Bio MAF v4 tabular features rather than MuAt-compatible event inputs.

## Manuscript Alignment

- Manuscript-generation text now describes scikit-survival CoxNet rather than a custom optimizer or CoxPH/CoxNet mixed path.
- Multiclass primary metrics are described as balanced accuracy; macro-AUROC is retained only as a supplementary diagnostic metric.
- TCGA cancer type is described as the fixed top-20 endpoint.
- The single inner validation split is described as the current leakage-prevention/runtime compromise for expensive experiments, not as a final statistical ideal.
