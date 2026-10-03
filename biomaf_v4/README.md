# Bio MAF v4

**Version 4.0.0** — fixed, interpretable per-tumor summaries of somatic mutations.

Bio MAF v4 converts a tumor's annotated mutation list (MAF format) into **948 numeric
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

Fixed reference resources shipped in `biomaf_v4/resources/`:

- `driver_gene_panel.csv` — the 401-gene driver panel and role interpretations
- `driver_gene_evidence.csv` — per-gene external driver-evidence tiers
- `v3_multi_type_residue.txt`, `v3_multi_type_variant_file.txt` — cancer hotspot catalogues

The OncoKB oncogene / tumor-suppressor role table is not bundled: OncoKB's Terms of Use
restrict redistribution, so the package downloads it at first use from the OncoKB public
API (no key required) via `biomaf_v4/fetch_oncokb.py` and caches it as
`biomaf_v4/resources/oncokb_cancer_genes.tsv`.

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
biomaf-v4 --maf tumor_mutations.maf.gz --out biomaf_v4_features.csv.gz
```

Options:

- `--samples A,B,C` or `--sample-list samples.txt` — featurize a subset of samples
  (default: every `Tumor_Sample_Barcode` in the MAF, truncated to 12 characters)
- `--no-optional-controls` — emit the 943 core features only
- `--columns-out columns.txt` — also write the feature-name list

Python API:

```python
from biomaf_v4 import featurize_maf, feature_names

matrix = featurize_maf("tumor_mutations.maf.gz")  # samples x 948 DataFrame
names = feature_names()                          # the 948 column names
```

## Input MAF columns

`Tumor_Sample_Barcode` is required. All annotation columns are optional; missing
columns are treated as unknown: `Hugo_Symbol`, `SYMBOL`, `Variant_Type`, `IMPACT`,
`Consequence`, `CANONICAL`, `BIOTYPE`, `SIFT`, `PolyPhen`, `Protein_position`,
`Amino_acids`, `HGVSp_Short`, `t_alt_count`, `t_depth`. VEP-annotated MAFs (e.g. the
TCGA MC3 public MAF) work directly.

## Testing

```bash
python -m pytest tests/ -v
```

The smoke test builds a tiny synthetic MAF, runs the full pipeline, and checks the
output has exactly 948 columns matching the manuscript's feature manifest.

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

If you use Bio MAF v4, please cite the manuscript. Details to be added on publication.
