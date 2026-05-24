# EXP-025: UGA payload absence encoding benchmark

## Recommendation

Use an **absence-safe payload** schema for INDEL REF/ALT payloads. There are now two UGA-admissible options:

```text
masked payload      = [X_1..X_d, Y_1..Y_d, M_1..M_d]
length-coded payload = [X_1..X_d, Y_1..Y_d, L_1..L_k]
```

`M_i` marks whether slot `i` contains a real base. `L` encodes the payload length from 0..d and reserves one extra state for overflow/truncation. In both schemas, `(X,Y)=(0,0)` can safely remain the nucleotide `G` when the absence/length bits say that slot is occupied.

For the current `d_context=10` and `d_payload=2` operating point, both schemas produce **52D** vectors. The masked schema had the best simulated recovery at this depth. Length coding is the more information-efficient option when `d_payload` is increased because biological payloads are contiguous prefixes, not arbitrary sparse slot patterns.

## Why this is optimal under the UGA bit-identity rules

Per-slot masking is the minimal slot-local binary representation of `{A,C,G,T,absent}`. Length coding exploits the stronger biological constraint that payload bases form a compact prefix. At the selected payload depth, the length-coded block needs nucleotide bits plus length bits, matching or reducing the per-slot masked width while also giving one overflow/reserved length state. For larger `d_payload`, length coding uses `2*d + ceil(log2(d+2))` bits per block instead of `3*d`, so it captures the same absence information with fewer dimensions.

Continuous sentinels can also avoid exact collisions, but they abandon the discrete binary identity space. `out_of_range_pad` also leaves the bounded `[0,1]` cube.

## Reproducible command

```powershell
python bench\run_payload_absence_benchmark.py --out-dir ${BUNDLE_ROOT}\results\work\mechanistic\exp025_payload_absence
```

Parameters: seed=250513, simulated patients=480, events per patient=300, d_payload=2.

## Main empirical result

The legacy payload had **185** payload-pair collisions over the tested state space. Masked and length-coded payloads both had **0** collisions. In the indel-like exposure recovery task, mean MAE improved from **0.0792** with legacy zero padding to **0.0173** with masked payloads and **0.0211** with length-coded payloads; top-1 process recovery improved from **0.344** to **0.896** and **0.871**, respectively.

## Outputs

| Path | Contents |
|---|---|
| `tables/payload_candidate_summary.csv` | Collision, distance, dimensionality, and admissibility metrics. |
| `tables/payload_collision_detail.csv` | Exact state buckets for every collision. |
| `tables/mixture_recovery_summary.csv` | NNLS exposure recovery summary by encoding candidate. |
| `tables/mixture_patient_metrics.csv` | Patient-level recovery metrics. |
| `tables/signature_geometry.csv` | Signature rank and distance diagnostics. |
| `tables/payload_dimension_efficiency.csv` | Payload-depth dimensionality sweep. |
| `html/manuscript_tables.html` | Copy/paste-ready manuscript tables. |
| `figures/payload_absence_d3_figures.html` | D3 figure page. |
| `data/payload_absence_benchmark_data.json` | Source data used by the figure page. |
| `manifest.json` | Run parameters and output paths. |
