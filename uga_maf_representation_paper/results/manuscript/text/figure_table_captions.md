# Manuscript Captions And Results Text

Author note: this text describes the generated manuscript outputs currently produced by the pipeline. Figure 3 uses One-hot sequence KME v2 as the main geometry comparison, while UGA and channel-KME variants are handled in Supplementary Figure S3.

## Captions

### Figure 1. Conceptual overview of mutation-catalogue representations.

Each sample is represented as a catalogue of somatic mutation events, which can be transformed into complementary tabular feature families. Signature features summarize mutation spectra, geometry features encode sequence-context distributions from FASTA-derived windows or UGA/channel encodings, and MAF-stack features aggregate event-level biological annotations such as gene, locus, consequence, and burden summaries. Combined representations concatenate process-level spectra with event-level biology. These tabular representations are evaluated with elastic-net and XGBoost models across mechanistic, HRD, cancer-type, and clinical endpoints. MuAt and ATGC are shown as conceptually distinct end-to-end alternatives that operate directly on mutation event sets rather than on precomputed tabular summaries.

### Figure 2. Signature baselines compared with mutational burden.

Single 5-fold out-of-fold performance is shown for Mutational burden and Mutational signatures across the five main endpoints and two model families. Primary metrics are Spearman correlation for continuous HRD score, AUROC for binary endpoints, and macro-AUROC for multiclass endpoints. Mutational signatures improve over Mutational burden for XGBoost on Kucab damage class (0.651 vs 0.459), HRD score (0.694 vs 0.615), HRD33 high/low (0.884 vs 0.826), Cancer type (top 10) (0.926 vs 0.840), and Overall survival event (0.652 vs 0.623). Elastic net shows the same pattern for Kucab and cancer type, but Mutational burden remains stronger for Elastic net HRD33 high/low and Overall survival event.

### Figure 3. Geometry-only one-hot event KME compared with signatures.

Performance of FASTA-derived One-hot sequence KME v2 is compared with Mutational signatures using the same canonical out-of-fold results. One-hot sequence KME v2 numerically exceeds Mutational signatures in 5 of 10 endpoint/model comparisons, including Elastic net Kucab damage class (0.618 vs 0.596), Elastic net HRD score (0.579 vs 0.510), and XGBoost Cancer type (top 10) (0.935 vs 0.926). FDR-significant positive KME differences are observed for Elastic net HRD score, XGBoost Cancer type (top 10), while significant negative differences are observed for Elastic net Cancer type (top 10). For XGBoost Kucab damage class, One-hot sequence KME v2 is lower than Mutational signatures (0.630 vs 0.651). This supports the conclusion that sequence-context geometry can be useful in selected settings, but is not a general replacement for spectra.

### Figure 4. Event-level MAF-stack features and combined signature-plus-event representations.

This figure compares Mutational signatures, Event-level MAF stack, and Signatures + MAF stack for each endpoint and model family. XGBoost with Signatures + MAF stack gives the strongest results for HRD score (0.775), HRD33 high/low (0.902), Cancer type (top 10) (0.958), and Overall survival event (0.667). Event-level MAF stack alone improves over Mutational signatures for XGBoost Cancer type (top 10) (0.937 vs 0.926), but underperforms Mutational signatures for Kucab damage class (0.585 vs 0.651). The combined representation improves over Event-level MAF stack in 7 of 10 tested Figure 4 comparisons at q < 0.05, showing that process-level spectra and event-level biology are complementary.

### Figure 5. Cross-endpoint summary of representation tradeoffs.

A canonical heatmap summarizes all five main representations across the five main endpoints and two model families. Values exactly match the canonical rows used in Figures 2-4. For XGBoost, Signatures + MAF stack wins four of five endpoints (HRD score, HRD33 high/low, Cancer type (top 10), and Overall survival event), while Mutational signatures are highest for Kucab damage class. For Elastic net, the winners are Kucab damage class: One-hot sequence KME v2, HRD score: One-hot sequence KME v2, HRD33 high/low: Mutational burden, Cancer type (top 10): Signatures + MAF stack, Overall survival event: Mutational burden. No single representation wins everywhere, but the combined signature-plus-MAF representation is the strongest practical default for XGBoost tabular models.

### Table 1. Datasets, endpoints, and evaluation design.

This table summarizes the five main manuscript endpoints, sample counts, task types, data sources, primary metrics, and label definitions. The main panel includes Kucab damage class (n = 259), HRD score (n = 772), HRD33 high/low (n = 772), MC3 Cancer type (top 10) (n = 5,462), and MC3 Overall survival event (n = 10,139). Supplementary endpoints are listed separately in Supplementary Table S1. Model-based results use a single 5-fold cross-validation design with aggregated out-of-fold predictions.

### Table 2. Main-panel performance matrix.

This table is the compact numeric backbone for the main manuscript figures, containing 5 endpoint rows. Each representation column reports elastic-net and XGBoost scores as EN / XGB, using the endpoint-specific primary metric. Full provenance-heavy versions with run identifiers, cache keys, and source files are retained under `tables/technical/`.

### Table 3. Representation summary and dimensionality.

This table summarizes the five main representations, their input signal, feature dimensionality range, context or atlas status, evaluated models, and manuscript role. Mutational burden features are compact with a median of 3 features; Mutational signatures have a median of 182 features; One-hot sequence KME v2 has 68-132 features depending on model family; Event-level MAF stack features are substantially larger, with median dimensionality around 1,823 features; and Signatures + MAF stack reaches a median of about 2,005 features. These values make the performance/complexity tradeoff explicit.

### Table 4. Key terminology and abbreviations.

This short glossary defines the key abbreviations, metrics, representation names, and model shorthand needed to read the main tables. The full machine-readable label mapping is retained in `tables/technical/table_4_label_mapping_technical.csv`.

### Supplementary Figure S1. Representation construction and reproducibility workflow.

This schematic details how raw mutation catalogues are transformed into spectra, FASTA-window one-hot KME features, UGA/channel-KME variants, MAF-stack aggregates, and combined representations. It also illustrates the cache/checkpoint workflow used to make feature generation reusable and restartable. Context-derived features use GRCh37 FASTA windows where appropriate, while atlas-based UGA/channel features are treated as supplementary geometry variants.

### Supplementary Figure S2. Calibration of selected main models.

Reliability curves are shown for classification endpoints using the selected Signatures + MAF stack XGBoost models. Calibration is evaluated from out-of-fold predictions for Kucab damage class, HRD33 high/low, Cancer type (top 10), and Overall survival event. These plots check whether the strongest models' predicted probabilities are broadly aligned with observed event frequencies rather than merely improving rank-based metrics.

### Supplementary Figure S3. Supplementary measured representation panels.

Measured supplementary results are shown for alternative geometry encodings, COSMIC/NNLS exposure checks, and mechanistic-control benchmarks. Visible marks are measured only and specify the model family or analysis family used. Unsupported or intentionally omitted combinations are excluded from the figure and documented separately in Supplementary Table S3.

### Supplementary Table S1. Supplementary endpoint inventory.

This table lists supplementary endpoints, sample counts, task types, data sources, primary metrics, and representation families evaluated outside the main five-endpoint panel.

### Supplementary Table S2. Headline supplementary results.

This table reports 17 endpoint-level headline rows summarizing the best baseline/event-level result, best geometry or sensitivity result, and best overall supplementary result. The exhaustive supplementary result matrix is retained under `tables/technical/`.

### Supplementary Table S3. Non-applicability summary.

This compact table groups unsupported, intentionally omitted, or not-applicable supplementary combinations by representation or analysis family. Full endpoint-level details are retained under `tables/technical/`.
