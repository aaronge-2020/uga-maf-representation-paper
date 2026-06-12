# Manuscript Captions And Results Text

Author note: this text describes the generated manuscript outputs currently produced by the pipeline. Figure 3 uses the MuAt-compatible event-bag comparator as the direct neural-model comparison, while UGA and channel-KME variants are handled as supplementary geometry analyses.

## Captions

### Figure 1. Conceptual overview of mutation-catalogue representations.

Each sample is represented as a catalogue of somatic mutation events, which can be transformed into complementary tabular feature families. Signature features summarize mutation spectra, geometry features encode sequence-context distributions from FASTA-derived windows or UGA/channel encodings, and MAF-stack features aggregate event-level biological annotations such as gene, locus, consequence, and burden summaries. Combined representations concatenate process-level spectra with event-level biology. These tabular representations are evaluated with nested elastic-net/logistic, XGBoost, and Cox PH models across mechanistic, HRD, cancer-type, and survival endpoints. A MuAt-compatible event-bag comparator is measured directly on MC3 mutation events rather than presented only as a conceptual alternative.

### Figure 2. Signature baselines compared with mutational burden.

Nested five-fold out-of-fold performance is shown for Mutational burden and Mutational signatures across the main endpoint families. Primary metrics are pooled Spearman correlation for continuous HRD score, AUROC for binary endpoints, macro-AUROC for multiclass endpoints, and Harrell C-index for Cox survival endpoints. Mutational signatures improve over Mutational burden for XGBoost on Kucab damage class (0.636 vs 0.505), HRD score (0.704 vs 0.596), HRD33 high/low (0.845 vs 0.811), and Cancer type (top 20) (0.935 vs 0.808). Cox PH survival rows are reported separately with Overall survival C-index.

### Figure 3. MuAt-compatible event-bag comparator.

The MuAt-compatible reimplementation is evaluated on the same TCGA-WES samples and held-out folds used by the main tabular benchmark for HRD score, HRD24 high/low, HRD33 high/low, and HRD42 high/low, Cancer type (top 20), and Overall survival. Its pooled macro-AUROC is 0.920, compared with 0.976 for the strongest tabular signature-plus-MAF XGBoost baseline. This makes the neural event-bag comparison directly comparable to the manuscript endpoints rather than to the original MuAt paper's TCGA-20 task.

### Figure 4. Event-level MAF-stack features and combined signature-plus-event representations.

This figure compares Mutational signatures, Bio MAF v4, and Signatures + Bio MAF v4 for each endpoint and model family. XGBoost with Signatures + Bio MAF v4 gives the strongest results for HRD score (0.781), HRD33 high/low (0.867), Cancer type (top 20) (0.976), and Cox PH Overall survival (0.633). Bio MAF v4 alone improves over Mutational signatures for XGBoost Cancer type (top 20) (0.959 vs 0.935), but underperforms Mutational signatures for Kucab damage class (0.529 vs 0.636). The combined representation improves over Bio MAF v4 in 5 of 10 tested Figure 4 comparisons at q < 0.05, showing that process-level spectra and event-level biology are complementary.

### Figure 5. Cross-endpoint summary of representation tradeoffs.

A canonical heatmap summarizes all five main representations across the main endpoint families and two model families. Values exactly match the canonical rows used in Figures 2-4. For XGBoost, Signatures + Bio MAF v4 is evaluated across non-survival endpoints, while Cox PH rows report C-index for Overall survival and other CDR survival endpoints. For Elastic net, the winners are Kucab damage class: Signatures + Bio MAF v4, HRD score: Signatures + Bio MAF v4, HRD24 high/low: Signatures + Bio MAF v4, HRD33 high/low: Signatures + Bio MAF v4, HRD42 high/low: Signatures + Bio MAF v4, Cancer type (top 20): Bio MAF v4. No single representation wins everywhere, but the combined signature-plus-MAF representation is the strongest practical default for XGBoost tabular models.

### Table 1. Datasets, endpoints, and evaluation design.

This table summarizes the seven main manuscript endpoints, sample counts, task types, data sources, primary metrics, and label definitions. The main panel includes Kucab damage class, HRD score, HRD24 high/low, HRD33 high/low, and HRD42 high/low, MC3 Cancer type (top 20), and TCGA CDR Overall survival. Supplementary endpoints are listed separately in Supplementary Table S1. Model-based results use five outer folds with an inner validation split and pooled global out-of-fold metrics.

### Table 2. Main-panel performance matrix.

This table is the compact numeric backbone for the main manuscript figures, containing 7 endpoint rows. Each representation column reports elastic-net and XGBoost scores as EN / XGB, using the endpoint-specific primary metric. Full provenance-heavy versions with run identifiers, cache keys, and source files are retained under `tables/technical/`.

### Table 3. Representation summary and dimensionality.

This table summarizes the main representations, their input signal, feature dimensionality range, context or atlas status, evaluated models, and manuscript role. Mutational burden features are compact with a median of 3 features; Mutational signatures have a median of 182 features; Bio MAF v4 dimensionality reflects the selected Bio MAF v4 feature block within each outer fold, and Signatures + Bio MAF v4 adds the same nested-selected biology blocks to mutational spectra. The MuAt-compatible comparator is reported separately as an event-bag neural model with learned tumour-level features. These values make the performance/complexity tradeoff explicit.

### Table 4. Key terminology and abbreviations.

This short glossary defines the key abbreviations, metrics, representation names, and model shorthand needed to read the main tables. The full machine-readable label mapping is retained in `tables/technical/table_4_label_mapping_technical.csv`.

### Supplementary Figure S1. Representation construction and reproducibility workflow.

This schematic details how raw mutation catalogues are transformed into spectra, UGA/channel-KME variants, MAF-stack aggregates, combined tabular representations, and MuAt-compatible event bags. It also illustrates the cache/checkpoint workflow used to make feature generation reusable and restartable. Context-derived features use GRCh37 FASTA windows where appropriate, while atlas-based UGA/channel features are treated as supplementary geometry variants.

### Supplementary Figure S2. Calibration of selected main models.

Reliability curves are shown for classification endpoints using the selected Signatures + Bio MAF v4 XGBoost models. Calibration is evaluated from out-of-fold predictions for Kucab damage class, HRD24 high/low, HRD33 high/low, and HRD42 high/low, and Cancer type (top 20). These plots check whether the strongest models' predicted probabilities are broadly aligned with observed event frequencies rather than merely improving rank-based metrics.

### Supplementary Figure S3. Supplementary measured representation panels.

Measured supplementary results are shown for alternative geometry encodings, COSMIC/NNLS exposure checks, and supplementary checks. Visible marks are measured only and specify the model family or analysis family used. Unsupported or intentionally omitted combinations are excluded from the figure and documented separately in Supplementary Table S3.

### Supplementary Figure S4A. Bio MAF v4 feature groups, without the jargon.

This plain-language overview explains how Bio MAF v4 turns a patient's mutation file and fixed cancer reference resources into five groups of biological features. The figure separates the readable overview from the technical audit trail: VEP annotation features come from a fixed VEP label vocabulary, external evidence-confidence controls summarize reference support without treating source count as a tumor biology mechanism, exact-gene features come from a fixed 401-gene external rule, and hotspot features come from Cancer Hotspots. Feature values are per-sample aggregates: count-like features use log1p(count), fraction features divide by total annotated mutations, and allele-fraction features use summary statistics or maxima. The bottom jargon decoder defines VEP labels, allele fraction, log-counts, protein-affecting mutation counts, cancer-role-fit mutation counts, and fixed-before-training language; Supplementary Table S5 gives worked feature examples, exact feature definitions, and candidate-size derivations, and Supplementary Table S6 reports nested-CV-selected feature groups.

### Supplementary Figure S4B. Distribution of representative Bio MAF v4 features.

This figure shows empirical distributions for representative Bio MAF v4 features in the fixed 8,800-sample TCGA top-20 cancer-type cohort. Panels include compact VEP annotation features, allele-fraction summaries, evidence-confidence controls, exact TP53 gene features, Cancer Hotspots summaries, and optional mutation-burden controls. The annotations report nonzero percentage, median, and 95th percentile so readers can see which features are continuous, sparse, or strongly zero-inflated.

### Supplementary Figure S4C. Representative Bio MAF v4 feature nonzero rates by cancer type.

This heatmap summarizes the same feature-family logic by cancer type. Each cell is the percent of tumors in a TCGA top-20 cancer type with a nonzero value for the representative feature, highlighting that exact-gene and hotspot features are intentionally sparse while broad VEP and burden features are present in most tumors. The figure is descriptive only and is not used for feature selection.

### Supplementary Figure S5. Nested-CV Bio MAF v4 feature-block selections.

This heatmap reports which predeclared Bio MAF v4 feature block was selected by the inner-loop validation search within each outer fold. Cell values count outer folds across manuscript endpoints with Bio MAF v4 feature-block tuning. The figure separates Bio MAF-only models from signatures-plus-biology models so readers can see when the inner loop selected compact biology, exact driver-gene features, the full strict core, optional controls, or signatures alone.

### Supplementary Table S1. Supplementary endpoint inventory.

This table lists supplementary endpoints, sample counts, task types, data sources, primary metrics, and representation families evaluated outside the simplified main panel.

### Supplementary Table S2. Headline supplementary results.

This table reports 6 endpoint-level headline rows summarizing the best baseline/event-level result, best geometry or sensitivity result, and best overall supplementary result. The exhaustive supplementary result matrix is retained under `tables/technical/`.

### Supplementary Table S3. Non-applicability summary.

This compact table groups unsupported, intentionally omitted, or not-applicable supplementary combinations by representation or analysis family. Full endpoint-level details are retained under `tables/technical/`.

### Supplementary Table S4. TCGA top-20 cancer-type metrics.

This table reports complementary classification metrics for Cancer type (top 20), including macro-AUROC, micro-AUROC, accuracy, balanced accuracy, macro-F1, Cohen's kappa, and top-k accuracy where available. Macro-AUROC remains the primary ranking metric.

### Supplementary Table S5. Bio MAF v4 feature guide.

This 5-row guide explains what the Bio MAF v4 feature blocks contain, how each block should be interpreted, where the feature counts came from, how feature values are calculated, and why each block is scientifically grounded. For each feature block, it includes a representative real feature name, a toy MAF input, the exact arithmetic used to turn that toy input into a feature value, and a plain-English interpretation of the resulting number. It explicitly defines VEP labels, SIFT, PolyPhen, transcript biotype, canonical transcript, OncoKB roles, Vogelstein genes, fixed-before-modeling rules, and the external evidence-confidence control block. It also defines protein-affecting mutation counts and cancer-role-fit mutation counts, then lists every nested-CV Bio MAF candidate feature-set size derivation: compact only, compact plus evidence-confidence controls, compact plus hotspots, compact plus exact cancer-gene identity, full strict core, and full plus optional controls. The accompanying technical table gives the exact feature-level glossary for all 948 Bio MAF v4 features, including the 943 strict-core features and 5 optional controls. Exact consensus driver-gene features come from a fixed external rule before cross-validation; nested CV selects among predeclared blocks inside the inner loop rather than using held-out endpoint performance.

### Supplementary Table S6. Nested-CV Bio MAF v4 feature-block selections.

This 112-row table summarizes the feature-block candidate selected inside the inner loop for each endpoint, representation, and model. The corresponding technical table reports the exact by-fold selection decisions. Because selection occurs within the training partition of each outer fold, these rows document the tuning choices without leaking held-out test-fold information.
