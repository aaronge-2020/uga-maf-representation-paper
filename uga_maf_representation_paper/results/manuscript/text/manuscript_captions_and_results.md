# Manuscript Captions And Results Text

Author note: this text describes the generated manuscript outputs currently produced by the pipeline. Figure 3 uses One-hot sequence KME as the main geometry comparison, while UGA and channel-KME variants are handled in Supplementary Figure S3.

## Captions

### Figure 1. Conceptual overview of mutation-catalogue representations.

Each sample is represented as a catalogue of somatic mutation events, which can be transformed into complementary tabular feature families. Signature features summarize mutation spectra, geometry features encode sequence-context distributions from FASTA-derived windows or UGA/channel encodings, and MAF-stack features aggregate event-level biological annotations such as gene, locus, consequence, and burden summaries. Combined representations concatenate process-level spectra with event-level biology. These tabular representations are evaluated with elastic-net and XGBoost models across mechanistic, HRD, cancer-type, and clinical endpoints. MuAt and ATGC are shown as conceptually distinct end-to-end alternatives that operate directly on mutation event sets rather than on precomputed tabular summaries.

### Figure 2. Signature baselines compared with mutational burden.

Single 5-fold out-of-fold performance is shown for Mutational burden and Mutational signatures across the five main endpoints and two model families. Primary metrics are Spearman correlation for continuous HRD score, AUROC for binary endpoints, and macro-AUROC for multiclass endpoints. Mutational signatures improve over Mutational burden for XGBoost on Kucab damage class (0.651 vs 0.459), HRD score (0.694 vs 0.615), HRD33 high/low (0.884 vs 0.826), Cancer type (top 10) (0.926 vs 0.840), and Overall survival event (0.652 vs 0.623). Elastic net shows the same pattern for Kucab and cancer type, but Mutational burden remains stronger for Elastic net HRD33 high/low and Overall survival event.

### Figure 3. Geometry-only one-hot event KME compared with signatures.

Performance of FASTA-derived One-hot sequence KME is compared with Mutational signatures using the same canonical out-of-fold results. One-hot sequence KME is strongest for Kucab damage class, where it slightly exceeds Mutational signatures for XGBoost (0.656 vs 0.651) and Elastic net (0.623 vs 0.596). Across the remaining endpoints, One-hot sequence KME is generally competitive but does not consistently outperform Mutational signatures; it is lower than Mutational signatures for XGBoost Cancer type (top 10) (0.906 vs 0.926), HRD33 high/low (0.864 vs 0.884), and Overall survival event (0.644 vs 0.652). This supports the conclusion that sequence-context geometry can be useful for mechanistic process classification, but is not a general replacement for spectra.

### Figure 4. Event-level MAF-stack features and combined signature-plus-event representations.

This figure compares Mutational signatures, Event-level MAF stack, and Signatures + MAF stack for each endpoint and model family. XGBoost with Signatures + MAF stack gives the strongest results for HRD score (0.775), HRD33 high/low (0.902), Cancer type (top 10) (0.958), and Overall survival event (0.667). Event-level MAF stack alone improves over Mutational signatures for XGBoost Cancer type (top 10) (0.937 vs 0.926), but underperforms Mutational signatures for Kucab damage class (0.585 vs 0.651). The combined representation improves over Event-level MAF stack in 7 of 10 tested Figure 4 comparisons at q < 0.05, showing that process-level spectra and event-level biology are complementary.

### Figure 5. Cross-endpoint summary of representation tradeoffs.

A canonical heatmap summarizes all five main representations across the five main endpoints and two model families. Values exactly match the canonical rows used in Figures 2-4. XGBoost dominates the best overall results, with Signatures + MAF stack winning four of five endpoints: HRD score, HRD33 high/low, Cancer type (top 10), and Overall survival event. The exception is Kucab damage class, where One-hot sequence KME is best. No single representation wins everywhere, but the combined signature-plus-MAF representation is the most consistently strong practical default for tabular models.

### Table 1. Datasets, endpoints, and evaluation design.

This table summarizes the benchmark endpoints, sample counts, endpoint tiers, task types, data sources, label definitions, and splitting scheme. The main panel includes Kucab damage class (n = 259), HRD score (n = 772), HRD33 high/low (n = 772), MC3 Cancer type (top 10) (n = 5,462), and MC3 Overall survival event (n = 10,139). Supplementary endpoints include additional HRD metrics and thresholds, MC3 clinical and driver endpoints, LUAD KMT2C status, and other validation tasks. Model-based results use a single 5-fold cross-validation design with aggregated out-of-fold predictions.

### Table 2. Full performance metrics by endpoint, representation, and model.

This table is the numeric backbone for the manuscript figures and supplement, containing 964 benchmark rows. Rows report endpoint, representation, model family, primary metric, AUROC or macro-AUROC where applicable, AUPRC, accuracy/F1-style metrics where available, fold metadata, feature counts, runtime/provenance fields, and run identifiers. Main-figure values are drawn only from canonical measured rows with valid out-of-fold prediction provenance.

### Table 3. Hyperparameters and feature dimensionality.

This table summarizes representation dimensionality, model family, atlas status, folds/repeats, XGBoost estimator settings where applicable, linear model type, and tuning policy. Mutational burden features are compact with a median of 3 features; Mutational signatures have a median of 182 features; One-hot sequence KME has 68-132 features depending on model family; Event-level MAF stack features are substantially larger, with median dimensionality around 1,823 features; and Signatures + MAF stack reaches a median of about 2,005 features. These values make the performance/complexity tradeoff explicit.

### Table 4. Machine-readable to manuscript label mapping.

This table maps internal endpoint, representation, model, metric, task, and atlas-status identifiers to manuscript-facing display labels. Machine-readable identifiers are retained in technical CSVs for reproducibility, while display labels are used in figures, captions, text, and publication-friendly table copies.

### Supplementary Figure S1. Representation construction and reproducibility workflow.

This schematic details how raw mutation catalogues are transformed into spectra, FASTA-window one-hot KME features, UGA/channel-KME variants, MAF-stack aggregates, and combined representations. It also illustrates the cache/checkpoint workflow used to make feature generation reusable and restartable. Context-derived features use GRCh37 FASTA windows where appropriate, while atlas-based UGA/channel features are treated as supplementary geometry variants.

### Supplementary Figure S2. Calibration of selected main models.

Reliability curves are shown for classification endpoints using the selected Signatures + MAF stack XGBoost models. Calibration is evaluated from out-of-fold predictions for Kucab damage class, HRD33 high/low, Cancer type (top 10), and Overall survival event. These plots check whether the strongest models' predicted probabilities are broadly aligned with observed event frequencies rather than merely improving rank-based metrics.

### Supplementary Figure S3. Supplementary measured representation panels.

Measured supplementary results are shown for alternative geometry encodings, COSMIC/NNLS exposure checks, and mechanistic-control benchmarks. Visible marks are measured only and specify the model family or analysis family used. Unsupported or intentionally omitted combinations are excluded from the figure and documented separately in Supplementary Table S3.

### Supplementary Table S1. Class distribution and baseline rates.

This table reports endpoint-level sample counts, class distributions, prevalence, and naive baseline context. It provides the denominator and imbalance information needed to interpret AUROC, macro-AUROC, AUPRC, and accuracy-like metrics across binary, multiclass, and continuous tasks.

### Supplementary Table S2. Sensitivity analyses and supplementary endpoint results.

This table reports 914 supplementary benchmark rows across additional endpoints, representations, and analysis families. It extends the main conclusions to extra HRD metrics/thresholds, additional MC3 clinical or driver endpoints, geometry variants, COSMIC/NNLS exposure checks, and mechanistic-control analyses.

### Supplementary Table S3. Completeness and non-applicability registry.

This table records combinations that are unsupported, intentionally omitted, or not applicable for supplementary analyses. Main manuscript figures contain no N/A rows; missing or unsupported supplementary combinations are documented here rather than rendered as visual placeholders.

## Results Section Text

We first established the strength of conventional mutational spectra relative to a minimal burden baseline. In the canonical main panel, XGBoost models using Mutational signatures outperformed Mutational burden across all five main endpoints: Kucab damage class improved from 0.459 to 0.651 macro-AUROC, HRD score from 0.615 to 0.694 Spearman correlation, HRD33 high/low from 0.826 to 0.884 AUROC, Cancer type (top 10) from 0.840 to 0.926 macro-AUROC, and Overall survival event from 0.623 to 0.652 AUROC. These gains were statistically significant for 5 of 5 XGBoost comparisons after FDR correction. Elastic net models showed the same qualitative gain for Kucab and cancer-type prediction, but not for every clinical or HRD endpoint; in particular, Mutational burden exceeded Mutational signatures for Elastic net HRD33 high/low and Overall survival event. Thus, signatures are a strong baseline, but their advantage depends on both endpoint and model class.

We next tested whether geometry-only sequence-context encodings provide a general replacement for spectra. The main geometry comparison used FASTA-derived One-hot sequence KME. This representation performed best on Kucab damage class, reaching 0.656 macro-AUROC with XGBoost and 0.623 with Elastic net, slightly above the corresponding signature models. However, One-hot sequence KME did not consistently outperform Mutational signatures elsewhere. For XGBoost, it was lower than Mutational signatures on Cancer type (top 10) (0.906 vs 0.926), HRD33 high/low (0.864 vs 0.884), and Overall survival event (0.644 vs 0.652). Pairwise testing showed significant underperformance of One-hot sequence KME versus Mutational signatures in 5 of 10 main comparisons, with 0 significant positive comparisons. These results support a conditional role for geometry encodings, especially in mechanistic damage-class prediction, rather than a universal replacement for spectra.

Event-level MAF-stack features provided a complementary source of biological information. Event-level MAF stack alone was particularly useful for cancer-type prediction with XGBoost, improving over Mutational signatures from 0.926 to 0.937 macro-AUROC. However, it was not uniformly better than spectra: for Kucab damage class, Event-level MAF stack alone was lower than Mutational signatures with XGBoost (0.585 vs 0.651), consistent with the idea that mechanistic mutagen exposure is better captured by sequence-context or spectral information than by event-level gene/locus aggregates alone.

The strongest overall pattern emerged from combining spectra with event-level MAF features. Signatures + MAF stack was the best overall representation for four of five main endpoints with XGBoost: HRD score reached 0.775 Spearman correlation, HRD33 high/low reached 0.902 AUROC, Cancer type (top 10) reached 0.958 macro-AUROC, and Overall survival event reached 0.667 AUROC. The only main endpoint where it did not win was Kucab damage class, where One-hot sequence KME was slightly higher. In pairwise tests, Signatures + MAF stack significantly improved over Event-level MAF stack alone in 7 of 10 Figure 4 comparisons and significantly improved over Mutational signatures alone for XGBoost HRD score, Cancer type (top 10), Overall survival event.

Taken together, the cross-endpoint summary shows that there is no single magic representation. Geometry-only features are useful for some mechanistic settings, Mutational signatures remain a strong and efficient baseline, and Event-level MAF stack features capture endpoint-relevant biology that spectra alone can miss. Across the main panel, the most robust practical default for tabular models is Signatures + MAF stack, particularly when paired with XGBoost. The supplementary analyses document additional geometry variants, exposure checks, endpoint extensions, and completeness metadata without introducing N/A placeholders into the main manuscript figures.

## Statistical Notes

Primary metrics are Spearman correlation for HRD score, AUROC for binary endpoints, and macro-AUROC for multiclass endpoints. Statistical statements refer to canonical/main_panel_pairwise_tests.csv, using paired DeLong tests for binary AUROC and paired bootstrap tests for macro-AUROC or Spearman correlation.
