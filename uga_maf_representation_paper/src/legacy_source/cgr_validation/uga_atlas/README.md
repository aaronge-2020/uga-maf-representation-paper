# UGA Atlas

This folder is the shared construction layer for Universal Genomic Address representations used by CGR validation experiments.

The master specification is `docs/UGA_MASTER_SPECIFICATION.md`. The scripts in this folder turn that specification into reusable code: model definitions, genome-wide context atlas loading, SBS/DBS/ID channel encoders, and projection helpers for patient spectra and reference signatures.

## Model Registry

`models.py` defines named UGA operating points. Benchmarks should choose a registered component or composite model rather than redefining bit depth, payload layout, modality pairing, or ID proxy behavior inside an experiment script.

Common entries:

- `master_spec_sbs_dbs_d22`: SBS96/DBS78 channel projection with `d_context=22`, `d_payload=2`, and masked REF/ALT payloads.
- `compact_sbs_dbs_d10`: compact masked SBS96/DBS78 projection for rapid benchmark sweeps.
- `compact_event_legacy_d10`: legacy 48-feature event-level coordinate used by EXP-024 shared-space simulations.
- `islam2022_sbs1536_d10`, `islam2022_dbs78_d10`, `label_sbs1536_d22`, and `label_dbs78_d22`: label-projection models for Islam 2022 channel-level and fusion benchmarks.
- `id83_proxy_d10_dp5` and `id83_proxy_d22_dp5`: categorical ID83 proxy encoders with wider payload capacity.
- `id83_token_pair_*`, `id83_repeat_context_*`, and `id83_payload_only_*`: ID83 proxy variants retained for legacy sensitivity analyses.
- `observed_context_events_d10_dp10`: event-level SBS, DBS, and indel encoder using observed GRCh37 FASTA flanks and raw REF/ALT payloads.
- `bicgr52_context_d10` and `bicgr52_context_d13`: context-only ablation models.
- `gdsc_sbs_d10_dp10`: legacy EXP026 GDSC SBS96 operating point.

Composite entries:

- `locked_payload_sbs_dbs_id_d10_v1`: locked 2026-05-14 manuscript composite using `master_spec_sbs_dbs_d10_dp5` for SBS/DBS and `id83_payload_only_d10_dp5` for ID83.
- `compact_proxy_sbs_dbs_id_d10_v1`: best 2026-05-14 encoding-scout composite using `compact_sbs_dbs_d10` for SBS/DBS and `id83_proxy_d10_dp5` for ID83 with separate SBS/ID feature blocks.
- `kernel_density_compact_payload_sbs_dbs_id_d10_v1`: best 2026-05-15 scout candidate using `compact_sbs_dbs_d10` for SBS/DBS and `id83_payload_only_d10_dp5` for ID83, followed by a row-normalized RBF kernel-density transform over UGA channel addresses.

## Usage

List registered models:

```powershell
python projects\cgr_validation\uga_atlas\list_models.py
```

Build a basis matrix from a channel table:

```powershell
python projects\cgr_validation\uga_atlas\build_channel_basis.py `
  --channels projects\cgr_validation\data\Signatures\COSMIC_v3.5_SBS_GRCh37.txt `
  --column Type `
  --model master_spec_sbs_dbs_d22 `
  --modality SBS `
  --out basis_sbs96_d22.npy `
  --diagnostics basis_sbs96_d22_diagnostics.csv
```

Experiment scripts should import from `uga_atlas`:

```python
from uga_atlas import load_context_atlas, get_uga_model, get_composite_uga_model, build_channel_basis

model = get_uga_model("master_spec_sbs_dbs_d22")
atlas = load_context_atlas(d_context=model.d_context)
basis, valid = build_channel_basis(channels, model, atlas=atlas, modality="SBS")

composite = get_composite_uga_model("compact_proxy_sbs_dbs_id_d10_v1")
component_models = composite.component_names()
```

`vdkm.render` remains a low-level bit primitive module. Benchmark-facing model selection and channel-to-UGA construction live here.
