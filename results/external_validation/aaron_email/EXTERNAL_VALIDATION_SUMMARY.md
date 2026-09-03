# External validation summary: POG570 HRD and PCAWG cancer type

Run date: 2026-08-22

Repository commit: `9615458441a64272368b5a62432b437209796231`

## POG570 primary HRD validation

The POG570 analysis used the frozen `HRD_Score__signatures_plus_MAF_stack` XGBoost regression model with 1,130 input features. All 570 samples were usable, with no sample-level exclusions. The prespecified primary analysis included 147 breast cancers, while the exploratory analysis included the remaining 423 non-breast cancers across 41 cancer types. SigProfiler successfully generated mutation-context features for all 570 samples and skipped only 97 of 8,141,138 mutation rows (0.0012%), which did not result in the exclusion of any sample.

### Continuous HRD score

| Cohort | N | Pearson r (95% CI) | Spearman rho (95% CI) | MAE (95% CI) | RMSE (95% CI) | R2 (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Breast primary | 147 | 0.670 (0.575, 0.751) | 0.667 (0.558, 0.754) | 15.91 (14.28, 17.54) | 18.88 (17.07, 20.63) | -0.225 (-0.583, 0.039) |
| All POG570 | 570 | 0.551 (0.487, 0.611) | 0.520 (0.451, 0.585) | 20.64 (19.70, 21.65) | 23.72 (22.77, 24.70) | -1.201 (-1.549, -0.917) |

In the primary breast cancer cohort, the model retained meaningful rank-order information, with a Pearson correlation of 0.670 and a Spearman correlation of 0.667. However, the predictions were shifted upward, with an observed mean HRD score of 22.30 compared with a predicted mean of 41.69. The negative R2 and relatively large MAE and RMSE indicate poor absolute calibration despite the preserved correlation. No cutoff or model parameter was changed after observing the POG570 results.

### Prespecified HRD cutoffs in breast cancers

| Cutoff | Accuracy (95% CI) | Balanced accuracy (95% CI) | Sensitivity | Specificity | AUROC (95% CI) |
|---:|---:|---:|---:|---:|---:|
| 24 | 0.619 (0.544, 0.694) | 0.517 (0.500, 0.545) | 1.000 | 0.034 | 0.849 (0.785, 0.908) |
| 33 | 0.605 (0.524, 0.680) | 0.643 (0.576, 0.705) | 0.905 | 0.381 | 0.805 (0.727, 0.874) |
| 42 | 0.626 (0.544, 0.701) | 0.714 (0.639, 0.783) | 0.892 | 0.536 | 0.871 (0.801, 0.932) |

At the prespecified cutoffs, sensitivity remained high but specificity was low, particularly at the cutoff of 24, indicating that the upward prediction shift caused many HRD-low tumors to be classified as HRD-high. Discrimination remained useful based on AUROC, but the existing cutoffs were not well calibrated for direct use in POG570. Full cutoff metrics and confidence intervals are available in `pog570/pog570_primary_and_all_metrics_with_95ci.csv`, and the exploratory non-breast results by cancer type are available in `pog570/pog570_exploratory_metrics_by_cancer_type.csv`.

## PCAWG ICGC-only cancer-type validation

The PCAWG ICGC-only analysis used the frozen `cancer_type_top20__standard_sbs96_id83` XGBoost model with 182 mutation-context features and retained the original 20-class output without modification. All 913 tumors across the nine prespecified TCGA mappings were usable, with no sample-level exclusions after ingestion. Breast-DCIS and Pancan or other combined files were excluded a priori and were never ingested, thereby avoiding both the prespecified excluded subtype and duplicate patients from combined files. SigProfiler skipped 1,047 of 16,630,677 mutation rows (0.0063%), but this did not result in the exclusion of any sample.

| Metric | Estimate | 95% CI |
|---|---:|---:|
| Overall accuracy | 0.479 | 0.462, 0.495 |
| Balanced accuracy | 0.365 | 0.341, 0.390 |
| Macro-F1 over the nine mapped classes | 0.362 | 0.335, 0.387 |
| Top-3 accuracy | 0.622 | 0.601, 0.642 |

| Cancer type | N | Recall | 95% CI |
|---|---:|---:|---:|
| BRCA | 117 | 0.077 | 0.034, 0.128 |
| ESCA | 97 | 0.856 | 0.784, 0.918 |
| HNSC | 13 | 0.000 | 0.000, 0.000 |
| KIRC | 74 | 0.000 | 0.000, 0.000 |
| LIHC | 261 | 0.973 | 0.950, 0.992 |
| OV | 69 | 0.507 | 0.391, 0.623 |
| PRAD | 180 | 0.006 | 0.000, 0.017 |
| SKCM | 70 | 0.714 | 0.600, 0.814 |
| STAD | 32 | 0.156 | 0.031, 0.313 |

The overall accuracy was 0.479, while balanced accuracy and macro-F1 were 0.365 and 0.362, respectively, indicating modest and heterogeneous external transfer across the nine mapped cancer types. Top-3 accuracy increased to 0.622, showing that the correct diagnosis was often assigned a relatively high probability even when it was not the top prediction. Performance varied substantially by cancer type: LIHC, ESCA, and SKCM transferred relatively well, whereas BRCA, PRAD, HNSC, KIRC, and STAD performed poorly. The 20-class confusion matrix, all 20 output probabilities, sample mapping, input checksums, and preprocessing and software records are saved under `pcawg/`.

## Cohort and model notes

The observed performance differences likely reflect substantial cohort and sequencing-platform shifts between the training and external validation datasets. The TCGA training cohort consists predominantly of primary tumors characterized using WES-based MC3 mutation calls, whereas POG570 and PCAWG were profiled using whole-genome sequencing. In addition, POG570 consists largely of advanced, metastatic, or recurrent cancers. These differences in disease stage, genomic coverage, mutation burden, variant-calling pipelines, consortium context, and cohort composition may have shifted the feature distributions relative to TCGA, contributing to the reduced calibration and heterogeneous cross-cancer performance observed in the external cohorts.

Vijay's newly pushed frozen files are deployment artifacts created by selecting hyperparameters using TCGA-only cross-validation and then refitting on all eligible TCGA samples. They are not exports of the discarded nested-CV fold models. No POG570 or PCAWG data were used for model fitting, parameter selection, cutoff selection, or post hoc adjustment.
