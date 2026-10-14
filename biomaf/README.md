# Bio MAF

**Version 1.0.0** — fixed, interpretable per-tumor summaries of somatic mutations.

Bio MAF converts a tumor's annotated mutation list (MAF format) into **948 numeric
features** (943 core + 5 optional burden controls) built only from fixed external
references. No endpoint labels are used at any point, so the same featurization
applies unchanged to cancer-type classification, HRD prediction, survival analysis,
or any other per-tumor endpoint.

This package is the standalone release of the featurization used in the manuscript
*Benchmarking Somatic Mutation Representations for Cancer Typing and HRD Prediction*.
The logic is a faithful, self-contained port of the manuscript pipeline's feature
builder; the pipeline itself is untouched.

## What it computes

| Block | Features | What it measures |
|---|---|---|
| VEP annotation spectra | 100 | Counts and fractions of VEP impact, consequence, canonical, biotype, SIFT, and PolyPhen classes |
| VAF summaries | 5 | Mean, SD, median, 90th percentile, max variant allele frequency |
| OncoKB role aggregates | 12 | High/moderate-impact event counts, unique genes, max VAF per OncoKB role (oncogene / tumor suppressor / both) |
| Residual (non-OncoKB) aggregates | 4 | Same summaries for genes outside OncoKB |
| Driver-evidence tiers | 16 | Event counts, unique genes, max VAF per tier of external driver evidence (IntOGen/CGC/Vogelstein/Hotspots) |
| Exact driver-gene identity | 802 | Per-gene functional and role-matched event counts for 401 externally supported cancer genes |
| Cancer hotspot summaries | 4 | Exact-variant and residue-level hotspot event counts, unique genes, max VAF |
| Optional burden controls | 5 | log10 SBS/DBS/ID burden, DBS and ID fractions |

Fixed reference resources shipped in `biomaf/resources/`:

- `driver_gene_panel.csv` — the 401-gene driver panel and role interpretations
- `driver_gene_evidence.csv` — per-gene external driver-evidence tiers
- `v3_multi_type_residue.txt`, `v3_multi_type_variant_file.txt` — cancer hotspot catalogues

The OncoKB oncogene / tumor-suppressor role table is not bundled: OncoKB's Terms of Use
restrict redistribution, so the package downloads it at first use from the OncoKB public
API (no key required) via `biomaf/fetch_oncokb.py` and caches it as
`biomaf/resources/oncokb_cancer_genes.tsv`.

## Installation

```bash
pip install .
# or, for development:
pip install -e .
```

Requires Python 3.10+, pandas, numpy.

## Usage

Command line (takes an annotated MAF, writes the feature matrix):

```bash
biomaf --maf tumor_mutations.maf.gz --out biomaf_features.csv.gz
```

Options:

- `--samples A,B,C` or `--sample-list samples.txt` — featurize a subset of samples
  (default: every `Tumor_Sample_Barcode` in the MAF, truncated to 12 characters)
- `--no-optional-controls` — emit the 943 core features only
- `--columns-out columns.txt` — also write the feature-name list
- `--check-input` — inspect the MAF and report which expected columns are
  present/missing and which feature groups will be active vs zero-filled.
  No features are computed in this mode.

Python API:

```python
from biomaf import featurize_maf, feature_names

matrix = featurize_maf("tumor_mutations.maf.gz")  # samples x 948 DataFrame
names = feature_names()                          # the 948 column names
```

## Input MAF columns

### Required

| Column | Used for |
|---|---|
| `Tumor_Sample_Barcode` | Sample identity (first 12 characters) |

### Strongly recommended

| Column | Used for | Without it |
|---|---|---|
| `Hugo_Symbol` (or `SYMBOL`) | All 802 per-gene identity columns, driver tiers, hotspot matching | Per-gene columns are all zeros; only the 100 VEP-spectra + 5 VAF + burden features carry signal |

### Optional (VEP / annotation columns)

All annotation columns are optional; missing columns are treated as "unknown"
and their features are zero-filled. VEP-annotated MAFs (e.g. the TCGA MC3
public MAF) work directly.

| Column | Feature groups affected |
|---|---|
| `Variant_Type` | VEP spectra (variant-type fractions) |
| `IMPACT` | `vep_impact_*` (5 count + 5 fraction columns) |
| `Consequence` | `vep_consequence_*` (26 count + 26 fraction columns) |
| `CANONICAL` | `vep_canonical_*` columns |
| `BIOTYPE` | `vep_biotype_*` columns |
| `SIFT` | `vep_sift_*` columns |
| `PolyPhen` | `vep_polyphen_*` columns |
| `Protein_position`, `Amino_acids`, `HGVSp_Short` | Cancer hotspot matching (4 hotspot columns) |
| `t_alt_count`, `t_depth` | VAF summaries (5 columns: mean, SD, median, p90, max) |

## Without VEP annotation

The package runs on a minimal MAF containing only `Tumor_Sample_Barcode` and
`Hugo_Symbol`, but **VEP annotation is strongly recommended**. All 948 columns
are still emitted; columns requiring annotation fall back to "unknown"
categories or zeros.

**What you lose without VEP columns:** Without `IMPACT` and `Consequence`,
the pipeline cannot determine which mutations are functional
(high/moderate impact or protein-affecting). This disables:
- The 802 per-gene functional-event columns (the manuscript ablation shows
  gene identity contributes 0.122 balanced accuracy points — this is the
  load-bearing signal, and it requires VEP to identify functional events)
- The 100 VEP-spectra features (specific impact/consequence categories;
  all events fall into "unknown")
- The 16 driver-tier columns and 12 OncoKB role columns (both require
  high/moderate-impact filtering)
- The 5 VAF summaries (require `t_alt_count`/`t_depth`)
- The 4 hotspot columns (require protein positions)

In practice, a MAF with only barcode + gene symbol yields just 12 non-zero
columns (all "unknown" categories) out of 948. **VEP annotation is not
optional for meaningful results.**

**What the paper's ablation says:** removing the 32 curated-knowledge columns
(OncoKB roles, driver tiers, hotspots — a subset of VEP-derived features)
changed balanced accuracy by only 0.003. But the 802 per-gene functional
columns, which require VEP impact/consequence to identify functional events,
contribute 0.122. The ablation does not imply VEP is dispensable.

Use `--check-input` to see exactly which columns your MAF has and which
feature groups will be active before running the full pipeline:

```bash
biomaf --maf tumor_mutations.maf.gz --check-input
```

## Testing

```bash
python -m pytest tests/ -v
```

The smoke test builds a tiny synthetic MAF, runs the full pipeline, and checks the
output has exactly 948 columns matching the manuscript's feature manifest.
A separate no-VEP test verifies the pipeline completes on a minimal MAF
(barcode + gene symbol only) with the expected VEP-dependent columns zero-filled.

## License

CC-BY-4.0, following the parent manuscript repository (see its LICENSE file).

## Reference data licensing

The OncoKB gene-role table is not bundled with this package. It is fetched at first use
from the OncoKB Cancer Gene List (public API, https://www.oncokb.org), which is free for
academic research use; commercial users should review OncoKB's Terms of Use
(https://www.oncokb.org/terms). COSMIC mutational signature definitions are referenced by
name (e.g., SBS1) rather than redistributed; users supply their own signature matrix. No
COSMIC or OncoKB proprietary data files are redistributed.

## Citation

If you use Bio MAF, please cite the manuscript. Details to be added on publication.
