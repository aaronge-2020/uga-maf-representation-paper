# Feature dictionary: mutation representations R0 to R6 and their combinations

Every representation below is defined for an unlabeled patient. Learned
statistics (vocabularies, IDF weights, scalers, feature selection) are
fitted inside training partitions only. The extension grid (49 cells) is
built from R0 to R6 and the combined forms marked "grid"; the main panel
used R0, R1, R6, and Sig+MAF only.

Conventions: "panel" is the 401-gene Bio MAF panel (the genes with at least
one curated feature in the Bio MAF matrix; the list is written by the
builder to `config/panel_genes_401.csv`). "All genes" is every gene with at
least one eligible event in the cohort: 19,672 genes for cancer typing and
15,867 for the HRD cohort. Event eligibility and counting follow
`scripts/build_gene_features.py`: MuAt-compatible gene-by-consequence event
tables derived from the TCGA MAF, with 12-character patient barcodes.

## Base representations

### R0, burden (3 features)
Total mutation burden summaries: log10 total mutation count and the
insertion/deletion and double-base-substitution fractions.
File: `data/task0_features/features_burden_only.csv` (repo tree).
Grid name: `burden`.

### R1, spectra (182 features)
Fitted single-base-substitution (SBS96, 96 channels) and indel (ID83, 83
channels) signature exposures plus the 3 burden features. Signature
exposures are fitted signature activities, not raw trinucleotide counts.
File: `data/task0_features/features_standard_sbs96_id83.csv.gz` (repo tree).
Grid name: `spectra`.

### R2, binary gene indicators
One column per gene: 1 if the patient has at least one eligible event in
the gene, else 0.
Panel form: 401 columns. All-genes form: 19,672 (cancer) / 15,867 (HRD)
columns.
Files: `features/cancer_r2_binary_panel.csv.gz`,
`features/cancer_r2_binary_allgenes.csv.gz`, and the `hrd_` equivalents.
Grid names: `r2_binary_panel`, `spectra+r2_binary_allgenes` (combined form
only for all-genes).

### R3, per-gene event counts (in the style of CPEM)
One column per gene: the number of eligible events in that gene. This is a
per-gene count encoding in the style of the gene-level count block of Lee
et al. 2019 (CPEM); it is not the CPEM model, which also uses copy-number
ratios, mutation rates, and a random-forest plus deep-network ensemble.
Panel form: 401 columns. All-genes form: 19,672 / 15,867 columns.
Files: `features/cancer_r3_counts_panel.csv.gz`,
`features/cancer_r3_counts_allgenes.csv.gz`, and the `hrd_` equivalents.
Grid names: `r3_counts_panel`, `r3_counts_allgenes`,
`spectra+r3_counts_panel`, `spectra+r3_counts_allgenes`, and the log1p
variant `spectra+r3_counts_panel_log1p` (counts transformed by log1p).

### R4, consequence-grouped per-gene counts (401 x 3 = 1,203 features)
The R3 panel counts split into three consequence groups per gene
(missense, truncating, other), 401 genes x 3 groups.
File: `features/cancer_r4_consequence_panel.csv.gz` and the `hrd_`
equivalent.
Grid names: `r4_consequence_panel`, `spectra+r4_consequence_panel`.

### R5, TF-IDF over per-gene counts (run-time transform of R3 panel)
For each training partition, genes observed in that partition form the
vocabulary; a gene's weight is its document frequency in the partition and
the transform multiplies counts by log inverse document frequency, exactly
as implemented in `scripts/runner.py` (`tfidf`). R5 is the inference-safe
substitute for the tf-rf family of Simsek et al. 2019: their winning
weight, rf(g, c), is defined per true class c, so no faithful scoring rule
exists for an unlabeled patient (see
`reports/literature_crosswalk.md`). R5 is never the Simsek winning method.
Vocabulary size depends on the partition: 391 genes for the cancer fold-1
training partition (integration smoke check).
No stored matrix: the transform is recomputed per fold from
`features/<task>_r3_counts_panel.csv.gz`.
Grid names: `r5_tfidf_panel`, `spectra+r5_tfidf_panel`.

### R6, Bio MAF (948 features)
The curated MAF feature stack: 802 per-gene columns, 121 compact biology
summaries (consequence, impact, functional-effect predictions, allele
fraction, annotation blocks), 16 driver-evidence tier columns, 4 hotspot
columns, and 5 control columns. Construction is documented by
`results/manuscript/tables/technical/table_s5_bio_maf_feature_guide_technical.csv`
(repo tree). A 910-column variant from an earlier curation round appears in
the repository history; the benchmarked matrix is the 948-column one.
File: `results/tables/quick_bio_maf_features.csv.gz` (repo tree; 10,224
patients, subset to each cohort by the runner).
Grid name: `maf_stack`.

## Combined forms

### Sig+MAF (1,130 features)
R1 spectra concatenated with R6 Bio MAF. Grid name:
`spectra+maf_stack`. This is the manuscript's headline representation.

### Spectra plus a panel encoding
R1 concatenated with R2, R3, R4, or R5 panel forms. Grid names:
`spectra+r2_binary_panel`, `spectra+r3_counts_panel`,
`spectra+r4_consequence_panel`, `spectra+r5_tfidf_panel`,
`spectra+r3_counts_panel_log1p`.

### Spectra plus an all-genes encoding
R1 concatenated with the all-genes R2 or R3 matrix. Grid names:
`spectra+r2_binary_allgenes`, `spectra+r3_counts_allgenes`.

## Cohort-specific matrices (not Bio MAF)

### Kucab event stack (2,980 features)
A bespoke event-token stack built for the Kucab damage-class task from the
raw de novo subclone tables (Sample, Chrom, Pos, Ref, Alt, Gene, context,
caller scores). Those tables carry no VEP annotation, so Bio MAF cannot be
constructed for Kucab; the event stack is not Bio MAF and is never labelled
as Bio MAF. It appears in the endpoint results file under the file
convention name `MAF_stack_only` for that endpoint only. Its spectra
companion matrix there (`standard_sbs96_dbs78_id83`, 261 features) is also
Kucab-specific.

## Files and builders

| Object | File | Built by |
|---|---|---|
| R2/R3/R4 matrices | `analyses/representation_extension/features/*.csv.gz` | `scripts/build_gene_features.py` |
| Canonical splits | `analyses/representation_extension/config/canonical_split_<task>.csv` | `scripts/make_canonical_splits.py` (from the main panel OOF file) |
| Panel gene list | `analyses/representation_extension/config/panel_genes_401.csv` | `scripts/build_gene_features.py` |
| R5 TF-IDF | none (per-fold transform) | `scripts/runner.py` |
| Grid predictions | `analyses/representation_extension/predictions/*.csv` | `scripts/runner.py` |
| Trial tables | `analyses/representation_extension/metrics/trials/*.csv` | `scripts/runner.py` |
