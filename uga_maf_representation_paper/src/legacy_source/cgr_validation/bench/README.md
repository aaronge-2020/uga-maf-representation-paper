# CGR Benchmark Scripts

This directory contains CGR / Universal-BiCGR benchmark utilities that belong to the mutational-signature validation project.

Key files:

- `run_pcawg_benchmark.py` - PCAWG / COSMIC signature benchmark engine.
- `run_payload_absence_benchmark.py` - payload absence / UGA payload ablation benchmark.
- `run_shared_space_benchmark.py` - EXP-024 shared-space compatibility wrapper.
- `build_atlas.py` and `build_dinucleotide_atlas.py` - atlas construction helpers.
- `prepare_visualization_data.py` and `generate_visualizer.py` - signature visualizer helpers.
- `encoding_efficiency.py`, `precision_sweep.py`, and `vdk_bi_simplex_audit.py` - cross-cutting method diagnostics.

The ML/WGS benchmark generator remains at the repository root in `bench/generate_benchmarks.py`.

