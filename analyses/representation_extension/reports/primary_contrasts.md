# Primary contrasts (workstream A8)

Representation extension analysis, UGA/MAF paper. All 49 grid cells complete (2026-11-01). Scores computed from the saved out-of-fold predictions on branch `analysis/representation-extension` (head 5f32e9b).

Method: each score is the protocol's primary estimator, the mean of the five within-fold scores (cancer: balanced accuracy, n=8,800; HRD: Spearman correlation against the continuous HRD score, n=772). Differences are paired against spectra+maf_stack (Sig+MAF): the per-fold mean difference, plus a patient-level paired bootstrap 95% confidence interval for the pooled difference (2,000 resamples, seed 20261025, the same patients resampled for both sides). The protocol's margins for practical similarity are 0.01 for cancer balanced accuracy and 0.02 for HRD Spearman.

## Cancer, XGBoost

| Representation | Balanced accuracy | Difference vs Sig+MAF (per-fold) | Bootstrap 95% CI |
|---|---|---|---|
| spectra+maf_stack (Sig+MAF) | 0.7565 | reference | |
| spectra+r3_counts_panel | 0.6711 | −0.0854 | [−0.0949, −0.0761] |
| spectra+r2_binary_panel | 0.6725 | −0.0840 | [−0.0927, −0.0747] |
| spectra+r4_consequence_panel | 0.6678 | −0.0887 | [−0.0980, −0.0787] |
| spectra+r5_tfidf_panel | 0.6742 | −0.0824 | [−0.0917, −0.0724] |

Sig+MAF wins clearly. Every simpler encoding trails by 0.08 or more, with intervals far from zero and far outside the 0.01 similarity margin.

## Cancer, elastic net

| Representation | Balanced accuracy | Difference vs Sig+MAF (per-fold) | Bootstrap 95% CI |
|---|---|---|---|
| spectra+maf_stack (Sig+MAF) | 0.6215 | reference | |
| spectra+r3_counts_panel | 0.6186 | −0.0029 | [−0.0139, +0.0081] |
| spectra+r2_binary_panel | 0.5847 | −0.0368 | [−0.0480, −0.0250] |
| spectra+r4_consequence_panel | 0.6129 | −0.0086 | [−0.0206, +0.0030] |
| spectra+r5_tfidf_panel | 0.6483 | +0.0268 | [+0.0152, +0.0379] |

TF-IDF leads Sig+MAF by 0.027, with an interval that excludes zero and the similarity margin. The counts panel is statistically indistinguishable from Sig+MAF (difference −0.003), though its interval is not fully inside the 0.01 margin, so practical similarity is compatible but not proven. Binary is clearly worse.

## HRD, XGBoost

| Representation | Spearman | Difference vs Sig+MAF (per-fold) | Bootstrap 95% CI |
|---|---|---|---|
| spectra+maf_stack (Sig+MAF) | 0.7610 | reference | |
| spectra+r3_counts_panel | 0.7599 | −0.0011 | [−0.0184, +0.0212] |
| spectra+r2_binary_panel | 0.7466 | −0.0144 | [−0.0338, +0.0110] |
| spectra+r4_consequence_panel | 0.7502 | −0.0108 | [−0.0320, +0.0097] |
| spectra+r5_tfidf_panel | 0.7651 | +0.0042 | [−0.0159, +0.0239] |

A dead heat at the top. TF-IDF, counts, and Sig+MAF sit within 0.005 of each other, inside the 0.02 similarity margin, and no interval separates any of them from Sig+MAF. Binary and consequence counts trail slightly, also without a separable difference.

## HRD, elastic net

| Representation | Spearman | Difference vs Sig+MAF (per-fold) | Bootstrap 95% CI |
|---|---|---|---|
| spectra+maf_stack (Sig+MAF) | 0.5952 | reference | |
| spectra+r3_counts_panel | 0.5166 | −0.0786 | [−0.1159, −0.0015] |
| spectra+r2_binary_panel | 0.6184 | +0.0232 | [−0.0003, +0.1105] |
| spectra+r4_consequence_panel | 0.5222 | −0.0729 | [−0.0881, +0.0273] |
| spectra+r5_tfidf_panel | 0.5608 | −0.0344 | [−0.0564, +0.0639] |

Weak across the board (0.52 to 0.62). Binary nominally leads Sig+MAF by 0.023 but its interval touches zero and is very wide, so the lead is not established. Counts is clearly worse; the rest are inseparable from Sig+MAF.

## Bottom line

The full Bio MAF stack earns its complexity on cancer typing with XGBoost, the strongest learner pairing, where it beats every simpler encoding by about 8 points. With elastic net on cancer, TF-IDF beats it by about 3 points and raw panel counts match it. On HRD with XGBoost, the three leading encodings tie, so the simpler counts and TF-IDF forms do as well as the full stack there. Under the protocol rule that a simpler representation matching or beating Bio MAF must be reported, the honest summary is: Bio MAF's advantage is real but specific to cancer typing with gradient boosting, not general across tasks and learners.
