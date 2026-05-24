# EXP-024 UGA Shared-Space Benchmark

## Purpose

This experiment documents a precise version of the manuscript claim:

- SBS, DBS, and indels can be encoded in one Universal Genomic Address (UGA) feature space.
- A naive all-at-once exposure solve is not necessarily the right estimator and can be worse than separate per-modality solving.
- The shared coordinate space is still valuable because it enables cross-modality measurements that standard SBS96, DBS78, and ID83 catalogs cannot define on their own.

## Reproducible Command

Run from the repository root:

```powershell
python cgr_validation_results\research\scripts\EXP024_shared_space\run_shared_space_benchmark.py
```

Equivalent compatibility wrapper:

```powershell
python bench\run_shared_space_benchmark.py
```

Default parameters:

| Parameter | Value |
|---|---:|
| seed | 240513 |
| tumors | 520 |
| latent processes | 12 |
| signature events per process | 5500 |
| context noise | 0.055 |
| d_context | 10 |
| d_payload | 2 |
| UGA dimensions | 48 |

## Output Layout

| Path | Contents |
|---|---|
| `index.html` | Landing page linking the main artifacts. |
| `html/narrative_walkthrough.html` | Guided explanation of the methodology and results. |
| `html/manuscript_tables.html` | Publication-styled HTML tables. |
| `html/manuscript_tables_fragment.html` | Table-only HTML fragments for manuscript copy/paste. |
| `figures/shared_space_d3_figures.html` | Interactive D3 figure page. |
| `tables/table1_experiment_design.csv` | Machine-readable design summary. |
| `tables/table2_exposure_recovery_summary.csv` | Summary of separate and pooled exposure recovery. |
| `tables/table3_cross_modality_retrieval_summary.csv` | Cross-modality nearest-neighbor retrieval summary. |
| `tables/table4_heldout_imputation_summary.csv` | Held-out DBS/ID imputation summary. |
| `tables/*patient_metrics.csv` | Patient-level rows supporting the summary tables. |
| `data/shared_space_benchmark_data.json` | Embedded figure data and distance matrices. |
| `manifest.json` | Run parameters and artifact paths. |

## Experiment Design

The simulation creates 520 tumors from sparse mixtures of 12 latent mutational processes. For every process, SBS, DBS, and ID events share a clean flanking-context program but use modality-specific payloads and standard channel labels.

The standard side uses the SBS96, DBS78, and ID83 channel universes from `data/Signatures/COSMIC_v3.5_*_GRCh37.txt`. The UGA side encodes each synthetic event as:

```text
[Lx, Ly, Xref, Yref, Rx, Ry, Xalt, Yalt]
```

with atlas model `compact_event_legacy_d10`, yielding a 48-dimensional coordinate.

## Benchmarks

1. **Exposure recovery:** NNLS is fit separately within SBS, DBS, and ID using standard categorical signatures and UGA centroids. A deliberately naive pooled UGA comparator collapses all events into one burden-weighted 48D mean and fits SBS/DBS/ID centroids together.
2. **Cross-modality retrieval:** UGA centroids are matched across ordered modality pairs using nearest-neighbor distance in full 48D, clean-context 40D, and payload-only 8D coordinates. Standard catalogs receive a chance/tie baseline because their axes are not shared.
3. **Held-out imputation:** DBS and ID exposures are withheld. SBS exposures are transferred to DBS/ID using an unsupervised nearest-neighbor bridge in UGA clean-context coordinates.

## Main Results

- Separate UGA solving outperformed the naive pooled UGA solve for low-burden modalities. DBS MAE increased from 0.004 to 0.032; ID MAE increased from 0.004 to 0.030.
- UGA clean-context retrieval recovered cross-modality process partners with mean top-1 accuracy 1.000; the standard separate-catalog chance/tie expectation was 0.083.
- SBS-to-DBS imputation using the unsupervised UGA bridge had mean MAE 0.014, compared with 0.126 for the no-bridge standard cohort-mean baseline.

## Interpretation

This is the result the manuscript needs: UGA does not require claiming that a single pooled exposure solve is best. Instead, it demonstrates that a common coordinate system creates a geometry in which SBS, DBS, and indel processes can be compared, matched, visualized, and transferred across modalities. Standard mutational signature definitions do not provide this operation because each mutation class is defined on a separate categorical axis.

## Notes

- The D3 page uses a CDN import for D3 v7 and embeds the experiment data directly in the HTML.
- Re-running the script overwrites this package deterministically for the same seed and parameters.
- Patient-level rows are retained so summary values can be audited or re-aggregated.
