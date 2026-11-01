# Caption — Figure 1 (as it stands in the manuscript)

Figure 1. Study design. One tumour's mutation list (MAF rows: locus, alleles,
gene, consequence, VAF) is converted into four feature tables: Burden
(3 features), Spectra SBS96+ID83 (182), Bio MAF (948), and Signatures + Bio MAF
(1,130). Paired elastic net and XGBoost models are tuned inside training folds
only and scored on five held-out test groups. Endpoints: cancer type, 20 types
(n=8,800); HRD score and HRD high/low at 24, 33, and 42 (n=772); and overall
survival (n=9,986). A MuAt-compatible rebuild that consumes mutation event
bags directly is benchmarked on the same five groups, outside the tabular path.
