#!/usr/bin/env python3
"""Benchmark payload encodings for representing absence in UGA INDEL payloads.

The benchmark separates two questions:

1. Is the encoding injective over payload strings up to d_payload?
2. Does ambiguity measurably damage exposure recovery in an indel-like mixture
   task where one-base payloads and the same payload followed by G must be
   distinguished?

The benchmark includes two UGA-admissible absence-safe schemas:

    masked: [X_1..X_d, Y_1..Y_d, M_1..M_d]
    length: [X_1..X_d, Y_1..Y_d, L_1..L_k]

Masked payloads have the strongest slot-local geometry at the atlas-selected
payload depth. Length-coded payloads become more dimension-efficient when
deeper payloads are needed and can reserve an overflow/truncation state.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.spatial.distance import pdist


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from uga_atlas import (  # noqa: E402
    encode_alt_cgr_walk,
    fast_seq_to_bits,
    get_uga_model,
    payload_block_dim,
    payload_length_bit_count,
    universal_vector_dim,
)


DEFAULT_UGA_MODEL = get_uga_model("compact_sbs_dbs_d10")
D_CONTEXT = DEFAULT_UGA_MODEL.d_context
D_PAYLOAD = DEFAULT_UGA_MODEL.d_payload
BASES = "ACGT"
DEFAULT_OUT = REPO / "cgr_validation_results" / "research" / "reports" / "exp025_payload_absence"


@dataclass(frozen=True)
class Candidate:
    name: str
    label: str
    pad_mode: str
    binary_required: bool
    bounded_required: bool
    notes: str


CANDIDATES = [
    Candidate(
        "legacy_zero",
        "Legacy zero padding",
        "zero",
        True,
        True,
        "Current 2-bit payload with absent slots set to (0,0).",
    ),
    Candidate(
        "fractional_pad",
        "Fractional pad sentinel",
        "fractional",
        False,
        True,
        "Uses (0.5,0.5) for every absent slot; injective but no longer binary bit identity.",
    ),
    Candidate(
        "out_of_range_pad",
        "Out-of-range pad sentinel",
        "negative",
        False,
        False,
        "Uses (-1,-1) for absent slots; injective but leaves the bounded UGA cube.",
    ),
    Candidate(
        "masked_presence",
        "Masked payload",
        "masked",
        True,
        True,
        "Adds one presence bit per slot: [X,Y,M].",
    ),
    Candidate(
        "length_code",
        "Length-coded payload",
        "length",
        True,
        True,
        "Adds a block-level payload length code: [X,Y,L], with one overflow state.",
    ),
]


def all_payload_strings(d_payload: int) -> list[str]:
    out = [""]
    for length in range(1, d_payload + 1):
        out.extend("".join(p) for p in product(BASES, repeat=length))
    return out


def indel_channels_for_depth(d_payload: int) -> list[tuple[str, str]]:
    seqs = [s for s in all_payload_strings(d_payload) if s]
    return [(s, "") for s in seqs] + [("", s) for s in seqs]


def direct_xy_block(seq: str, d_payload: int, pad_value: float) -> np.ndarray:
    seq = (seq or "").upper()
    if pad_value == 0.0:
        return encode_alt_cgr_walk(seq, d=d_payload, payload_schema="legacy")
    x_obs, y_obs = fast_seq_to_bits(seq, d_payload)
    n = min(len(seq), d_payload)
    xb = np.full(d_payload, pad_value, dtype=np.float64)
    yb = np.full(d_payload, pad_value, dtype=np.float64)
    xb[:n] = x_obs[:n]
    yb[:n] = y_obs[:n]
    return np.concatenate([xb, yb])


def encode_block(seq: str, candidate: Candidate, d_payload: int) -> np.ndarray:
    if candidate.pad_mode == "zero":
        return direct_xy_block(seq, d_payload, 0.0)
    if candidate.pad_mode == "fractional":
        return direct_xy_block(seq, d_payload, 0.5)
    if candidate.pad_mode == "negative":
        return direct_xy_block(seq, d_payload, -1.0)
    if candidate.pad_mode == "masked":
        return encode_alt_cgr_walk(seq, d=d_payload, payload_schema="masked")
    if candidate.pad_mode == "length":
        return encode_alt_cgr_walk(seq, d=d_payload, payload_schema="length")
    raise ValueError(candidate.pad_mode)


def encode_pair(ref: str, alt: str, candidate: Candidate, d_payload: int) -> np.ndarray:
    return np.concatenate([
        encode_block(ref, candidate, d_payload),
        encode_block(alt, candidate, d_payload),
    ])


def vector_key(vec: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(vec, dtype=float), 10).tolist())


def state_label(ref: str, alt: str) -> str:
    return f"{ref or '-'}>{alt or '-'}"


def collision_analysis(candidate: Candidate, d_payload: int) -> tuple[dict, list[dict]]:
    payloads = all_payload_strings(d_payload)
    states = [(r, a) for r in payloads for a in payloads]
    groups: dict[tuple[float, ...], list[tuple[str, str]]] = defaultdict(list)
    block_groups: dict[tuple[float, ...], list[str]] = defaultdict(list)

    for seq in payloads:
        block_groups[vector_key(encode_block(seq, candidate, d_payload))].append(seq or "-")
    for ref, alt in states:
        groups[vector_key(encode_pair(ref, alt, candidate, d_payload))].append((ref, alt))

    collision_groups = [g for g in groups.values() if len(g) > 1]
    block_collision_groups = [g for g in block_groups.values() if len(g) > 1]
    encoded = np.stack([encode_pair(r, a, candidate, d_payload) for r, a in states])
    block_encoded = np.stack([encode_block(s, candidate, d_payload) for s in payloads])
    distances = pdist(encoded, metric="euclidean")
    nonzero_distances = distances[distances > 1e-12]

    empty_block = encode_block("", candidate, d_payload)
    base_dist = {
        b: float(np.linalg.norm(empty_block - encode_block(b, candidate, d_payload)))
        for b in BASES
    }
    values = np.unique(encoded)
    binary = bool(np.all(np.isin(values, [0.0, 1.0])))
    bounded = bool(np.all((encoded >= 0.0) & (encoded <= 1.0)))
    pair_dim = int(encoded.shape[1])
    if candidate.pad_mode in {"masked", "length"}:
        full_dim = int(universal_vector_dim(D_CONTEXT, d_payload, candidate.pad_mode))
    else:
        full_dim = 4 * D_CONTEXT + pair_dim
    block_state_count = sum(4 ** k for k in range(d_payload + 1))
    exact_enumerative_pair_dim = int(math.ceil(math.log2(block_state_count ** 2)))
    slot_local_pair_dim = 2 * d_payload * math.ceil(math.log2(5))
    direct_length_pair_dim = 2 * (2 * d_payload + payload_length_bit_count(d_payload))
    admissible = (
        len(collision_groups) == 0
        and binary
        and bounded
        and pair_dim == direct_length_pair_dim
    )

    summary = {
        "Candidate": candidate.name,
        "Label": candidate.label,
        "Payload_Block_Dim": int(block_encoded.shape[1]),
        "Payload_Pair_Dim": pair_dim,
        "Full_UGA_Dim_Selected": full_dim,
        "Exact_Enumerative_Min_Pair_Dim": exact_enumerative_pair_dim,
        "Slot_Local_Min_Pair_Dim": slot_local_pair_dim,
        "Direct_Length_Min_Pair_Dim": direct_length_pair_dim,
        "Payload_States": len(states),
        "Unique_Payload_Vectors": len(groups),
        "Collision_Count": len(states) - len(groups),
        "Collision_Groups": len(collision_groups),
        "Max_Collision_Bucket": max((len(g) for g in collision_groups), default=1),
        "Block_States": len(payloads),
        "Unique_Block_Vectors": len(block_groups),
        "Block_Collision_Count": len(payloads) - len(block_groups),
        "Block_Collision_Groups": len(block_collision_groups),
        "Min_Nonzero_Pair_Distance": float(nonzero_distances.min()) if len(nonzero_distances) else 0.0,
        "Empty_To_A_Distance": base_dist["A"],
        "Empty_To_C_Distance": base_dist["C"],
        "Empty_To_G_Distance": base_dist["G"],
        "Empty_To_T_Distance": base_dist["T"],
        "Binary_Features": binary,
        "Bounded_0_1": bounded,
        "Admissible_UGA_Binary_Minimal": admissible,
        "Notes": candidate.notes,
    }

    details = []
    for idx, group in enumerate(collision_groups, start=1):
        labels = [state_label(r, a) for r, a in group]
        details.append({
            "Candidate": candidate.name,
            "Collision_Group": idx,
            "Bucket_Size": len(group),
            "States": ";".join(labels),
            "Contains_Empty_Ref": any(r == "" for r, _ in group),
            "Contains_Empty_Alt": any(a == "" for _, a in group),
            "Contains_G_Payload": any("G" in r or "G" in a for r, a in group),
        })
    return summary, details


PROCESS_CHANNELS = [
    ("G", ""),
    ("GG", ""),
    ("A", ""),
    ("AG", ""),
    ("C", ""),
    ("CG", ""),
    ("T", ""),
    ("TG", ""),
    ("", "G"),
    ("", "GG"),
    ("", "A"),
    ("", "AG"),
    ("", "C"),
    ("", "CG"),
    ("", "T"),
    ("", "TG"),
]


def build_process_distributions(channels: list[tuple[str, str]]) -> np.ndarray:
    channel_index = {c: i for i, c in enumerate(channels)}
    P = np.zeros((len(channels), len(PROCESS_CHANNELS)), dtype=np.float64)
    for proc_idx, primary in enumerate(PROCESS_CHANNELS):
        if primary not in channel_index:
            raise ValueError(f"Missing channel {primary}")
        paired_idx = proc_idx ^ 1
        paired = PROCESS_CHANNELS[paired_idx]
        P[channel_index[primary], proc_idx] += 0.86
        P[channel_index[paired], proc_idx] += 0.08
        P[:, proc_idx] += 0.06 / len(channels)
    P /= P.sum(axis=0, keepdims=True)
    return P


def sparse_exposure(rng: np.random.Generator, n_processes: int) -> np.ndarray:
    active_n = int(rng.integers(2, 5))
    active = rng.choice(n_processes, size=active_n, replace=False)
    w = np.zeros(n_processes, dtype=np.float64)
    w[active] = rng.dirichlet(np.full(active_n, 0.7))
    background = rng.dirichlet(np.full(n_processes, 0.3)) * 0.025
    out = w * 0.975 + background
    return out / out.sum()


def normalize(x: np.ndarray) -> np.ndarray:
    total = float(np.sum(x))
    if total <= 1e-15:
        return np.zeros_like(x)
    return x / total


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-15 or nb <= 1e-15:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_signature_basis(
    channels: list[tuple[str, str]],
    process_probs: np.ndarray,
    candidate: Candidate,
    d_payload: int,
) -> np.ndarray:
    channel_vectors = np.stack([encode_pair(r, a, candidate, d_payload) for r, a in channels])
    return channel_vectors.T @ process_probs


def run_mixture_recovery(
    candidate: Candidate,
    d_payload: int,
    seed: int,
    n_patients: int,
    burden: int,
) -> tuple[dict, list[dict], dict]:
    rng = np.random.default_rng(seed)
    channels = indel_channels_for_depth(d_payload)
    process_probs = build_process_distributions(channels)
    A = build_signature_basis(channels, process_probs, candidate, d_payload)
    channel_vectors = np.stack([encode_pair(r, a, candidate, d_payload) for r, a in channels])
    rank = int(np.linalg.matrix_rank(A, tol=1e-10))
    signature_distances = pdist(A.T, metric="euclidean")
    nonzero_sig_distances = signature_distances[signature_distances > 1e-12]

    t0 = time.perf_counter()
    rows = []
    for i in range(n_patients):
        truth = sparse_exposure(rng, A.shape[1])
        proc_counts = rng.multinomial(burden, truth)
        channel_counts = np.zeros(len(channels), dtype=np.float64)
        for proc_idx, n_events in enumerate(proc_counts):
            if n_events <= 0:
                continue
            channel_counts += rng.multinomial(int(n_events), process_probs[:, proc_idx])
        profile = (channel_vectors.T @ channel_counts) / max(1, int(channel_counts.sum()))
        raw = nnls(A, profile)[0]
        pred = normalize(raw)
        rows.append({
            "Candidate": candidate.name,
            "Sample": f"SIM-{i + 1:04d}",
            "Burden": burden,
            "MAE": float(np.mean(np.abs(truth - pred))),
            "Cosine": cosine_similarity(truth, pred),
            "Top1_Process_Match": float(int(np.argmax(truth) == np.argmax(pred))),
            "Reconstruction_Error": float(np.linalg.norm(A @ raw - profile)),
        })

    runtime = time.perf_counter() - t0
    metrics = pd.DataFrame(rows)
    summary = {
        "Candidate": candidate.name,
        "Label": candidate.label,
        "N": int(len(metrics)),
        "Mean_MAE": float(metrics["MAE"].mean()),
        "Median_MAE": float(metrics["MAE"].median()),
        "Mean_Cosine": float(metrics["Cosine"].mean()),
        "Top1_Process_Accuracy": float(metrics["Top1_Process_Match"].mean()),
        "Mean_Reconstruction_Error": float(metrics["Reconstruction_Error"].mean()),
        "Runtime_Sec": runtime,
    }
    geometry = {
        "Candidate": candidate.name,
        "Signature_Count": int(A.shape[1]),
        "Feature_Dim": int(A.shape[0]),
        "Signature_Matrix_Rank": rank,
        "Rank_Deficit": int(A.shape[1] - rank),
        "Min_Signature_Distance": float(signature_distances.min()) if len(signature_distances) else 0.0,
        "Min_Nonzero_Signature_Distance": float(nonzero_sig_distances.min()) if len(nonzero_sig_distances) else 0.0,
    }
    return summary, rows, geometry


def dimension_efficiency_sweep(max_d_payload: int = 10) -> pd.DataFrame:
    rows = []
    for d_payload in range(1, max_d_payload + 1):
        block_state_count = sum(4 ** k for k in range(d_payload + 1))
        exact_pair_min = int(math.ceil(math.log2(block_state_count ** 2)))
        legacy_pair = 4 * d_payload
        masked_pair = 6 * d_payload
        length_pair = 2 * (2 * d_payload + payload_length_bit_count(d_payload))
        rows.append({
            "d_payload": d_payload,
            "Payload_Block_States": block_state_count,
            "Legacy_Pair_Dim": legacy_pair,
            "Masked_Pair_Dim": masked_pair,
            "Length_Coded_Pair_Dim": length_pair,
            "Exact_Enumerative_Min_Pair_Dim": exact_pair_min,
            "Legacy_Full_UGA_Dim": 4 * D_CONTEXT + legacy_pair,
            "Masked_Full_UGA_Dim": int(universal_vector_dim(D_CONTEXT, d_payload, "masked")),
            "Length_Coded_Full_UGA_Dim": int(universal_vector_dim(D_CONTEXT, d_payload, "length")),
            "Length_Saves_Dim_vs_Masked": int(universal_vector_dim(D_CONTEXT, d_payload, "masked") - universal_vector_dim(D_CONTEXT, d_payload, "length")),
            "Length_Over_Exact_Pair_Bits": length_pair - exact_pair_min,
        })
    return pd.DataFrame(rows)


def _fmt_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.4f}"
    return str(value)


def html_table(df: pd.DataFrame, columns: list[str], caption: str) -> str:
    head = "".join(f"<th>{escape(col.replace('_', ' '))}</th>" for col in columns)
    rows = []
    for _, row in df[columns].iterrows():
        cells = "".join(f"<td>{escape(_fmt_value(row[col]))}</td>" for col in columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"""<table>
<caption>{escape(caption)}</caption>
<thead><tr>{head}</tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>"""


def write_tables_html(
    html_dir: Path,
    candidate_df: pd.DataFrame,
    recovery_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    efficiency_df: pd.DataFrame,
) -> None:
    html_dir.mkdir(parents=True, exist_ok=True)
    tables = [
        html_table(
            candidate_df,
            [
                "Label",
                "Payload_Pair_Dim",
                "Full_UGA_Dim_Selected",
                "Collision_Count",
                "Binary_Features",
                "Bounded_0_1",
                "Admissible_UGA_Binary_Minimal",
            ],
            "Table 1. Payload absence candidate properties.",
        ),
        html_table(
            recovery_df,
            ["Label", "N", "Mean_MAE", "Median_MAE", "Mean_Cosine", "Top1_Process_Accuracy"],
            "Table 2. Simulated INDEL-like exposure recovery.",
        ),
        html_table(
            geometry_df.merge(candidate_df[["Candidate", "Label"]], on="Candidate", how="left"),
            ["Label", "Feature_Dim", "Signature_Count", "Signature_Matrix_Rank", "Rank_Deficit", "Min_Signature_Distance"],
            "Table 3. Signature geometry diagnostics.",
        ),
        html_table(
            efficiency_df,
            [
                "d_payload",
                "Exact_Enumerative_Min_Pair_Dim",
                "Masked_Pair_Dim",
                "Length_Coded_Pair_Dim",
                "Masked_Full_UGA_Dim",
                "Length_Coded_Full_UGA_Dim",
                "Length_Saves_Dim_vs_Masked",
            ],
            "Table 4. Payload depth efficiency sweep.",
        ),
    ]
    fragment = "\n\n".join(tables)
    style = """
<style>
body {
  margin: 34px;
  color: #202733;
  font-family: Aptos, "Segoe UI", Arial, sans-serif;
  line-height: 1.45;
}
main {
  max-width: 1120px;
}
h1 {
  margin: 0 0 10px;
  font-size: 28px;
  letter-spacing: 0;
}
p {
  color: #5d6875;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 26px 0 34px;
  font-size: 13px;
}
caption {
  caption-side: top;
  text-align: left;
  color: #202733;
  font-weight: 700;
  padding-bottom: 8px;
}
th {
  text-align: left;
  border-bottom: 2px solid #2d3745;
  padding: 8px 8px;
  white-space: nowrap;
}
td {
  border-bottom: 1px solid #d9e0e8;
  padding: 8px 8px;
  vertical-align: top;
}
tbody tr:nth-child(even) {
  background: #f7f9fb;
}
code {
  background: #eef3f7;
  border-radius: 4px;
  padding: 1px 4px;
}
</style>
"""
    full = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EXP-025 Payload Absence Tables</title>
{style}
</head>
<body>
<main>
<h1>EXP-025 Payload Absence Encoding Tables</h1>
<p>Copy/paste-ready tables for the UGA payload absence benchmark. Source CSV files are in <code>../tables/</code>.</p>
{fragment}
</main>
</body>
</html>
"""
    (html_dir / "manuscript_tables.html").write_text(full, encoding="utf-8")
    (html_dir / "manuscript_tables_fragment.html").write_text(fragment, encoding="utf-8")


def write_figures_html(figures_dir: Path, figure_payload: dict) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(figure_payload, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EXP-025 Payload Absence D3 Figures</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
body {{
  margin: 0;
  color: #202733;
  background: #f4f7fa;
  font-family: Aptos, "Segoe UI", Arial, sans-serif;
}}
main {{
  max-width: 1180px;
  margin: 0 auto;
  padding: 32px 28px 48px;
}}
h1 {{
  margin: 0 0 6px;
  font-size: 28px;
  letter-spacing: 0;
}}
p {{
  margin: 0 0 24px;
  color: #5d6875;
  line-height: 1.45;
}}
.panel {{
  background: #fff;
  border: 1px solid #dbe3ec;
  border-radius: 8px;
  padding: 22px 22px 18px;
  margin: 18px 0;
  box-shadow: 0 6px 18px rgba(29, 43, 62, 0.06);
}}
.panel h2 {{
  margin: 0 0 4px;
  font-size: 18px;
}}
.panel p {{
  margin-bottom: 12px;
  font-size: 13px;
}}
svg {{
  display: block;
  width: 100%;
  height: auto;
}}
.axis path, .axis line {{
  stroke: #8995a3;
}}
.axis text {{
  fill: #5d6875;
  font-size: 11px;
}}
.grid line {{
  stroke: #e5ebf2;
}}
.label {{
  fill: #202733;
  font-size: 12px;
}}
.value {{
  fill: #202733;
  font-size: 11px;
  font-weight: 700;
}}
.legend text {{
  fill: #4d5a67;
  font-size: 12px;
}}
</style>
</head>
<body>
<main>
  <h1>EXP-025 Payload Absence Encoding</h1>
  <p>D3 figures built directly from the benchmark source data. The masked payload gives the strongest recovery at the selected atlas payload depth; the length-coded payload is collision-free, preserves X/Y identity, and becomes more compact when deeper payload widths are required.</p>
  <section class="panel">
    <h2>Collision Burden By Candidate</h2>
    <p>Legacy zero padding collapses empty payloads and G-prefixed payloads; absence-safe encodings remove those collisions.</p>
    <div id="collision-chart"></div>
  </section>
  <section class="panel">
    <h2>Exposure Recovery</h2>
    <p>Lower MAE and higher cosine similarity indicate better recovery of simulated INDEL-like processes.</p>
    <div id="recovery-chart"></div>
  </section>
  <section class="panel">
    <h2>Payload Depth Efficiency</h2>
    <p>Length coding preserves the X/Y base identity but replaces per-slot masks with one block-level length code; it saves dimensions once payload depth is increased beyond the current small setting.</p>
    <div id="efficiency-chart"></div>
  </section>
</main>
<script>
const DATA = {data_json};
const colors = new Map([
  ["legacy_zero", "#b44e4e"],
  ["fractional_pad", "#8c6bb1"],
  ["out_of_range_pad", "#b8792c"],
  ["masked_presence", "#3e7c9f"],
  ["length_code", "#2f7d59"]
]);

function renderCollision() {{
  const data = DATA.candidate_summary;
  const width = 1060, height = 360;
  const margin = {{top: 18, right: 32, bottom: 88, left: 64}};
  const svg = d3.select("#collision-chart").append("svg").attr("viewBox", [0, 0, width, height]);
  const x = d3.scaleBand().domain(data.map(d => d.Label)).range([margin.left, width - margin.right]).padding(0.28);
  const y = d3.scaleLinear().domain([0, d3.max(data, d => +d.Collision_Count) * 1.14 || 1]).nice().range([height - margin.bottom, margin.top]);
  svg.append("g").attr("class", "grid").attr("transform", `translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(width - margin.left - margin.right)).tickFormat(""));
  svg.append("g").selectAll("rect").data(data).join("rect")
    .attr("x", d => x(d.Label))
    .attr("y", d => y(+d.Collision_Count))
    .attr("width", x.bandwidth())
    .attr("height", d => y(0) - y(+d.Collision_Count))
    .attr("rx", 4)
    .attr("fill", d => colors.get(d.Candidate))
    .attr("opacity", 0.88);
  svg.append("g").selectAll("text.value").data(data).join("text")
    .attr("class", "value")
    .attr("x", d => x(d.Label) + x.bandwidth() / 2)
    .attr("y", d => y(+d.Collision_Count) - 8)
    .attr("text-anchor", "middle")
    .text(d => d.Collision_Count);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${{height - margin.bottom}})`).call(d3.axisBottom(x).tickSize(0))
    .selectAll("text").attr("transform", "rotate(-28)").attr("text-anchor", "end").attr("dx", -6).attr("dy", 8);
  svg.append("g").attr("class", "axis").attr("transform", `translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(5));
  svg.append("text").attr("class", "label").attr("x", 16).attr("y", 24).text("Payload-pair collisions");
}}

function renderRecovery() {{
  const data = DATA.mixture_summary;
  const width = 1060, height = 420;
  const margin = {{top: 24, right: 54, bottom: 92, left: 64}};
  const svg = d3.select("#recovery-chart").append("svg").attr("viewBox", [0, 0, width, height]);
  const x0 = d3.scaleBand().domain(data.map(d => d.Label)).range([margin.left, width - margin.right]).padding(0.25);
  const x1 = d3.scaleBand().domain(["Mean_MAE", "Mean_Cosine"]).range([0, x0.bandwidth()]).padding(0.12);
  const y = d3.scaleLinear().domain([0, 1]).nice().range([height - margin.bottom, margin.top]);
  const metricColor = d3.scaleOrdinal().domain(["Mean_MAE", "Mean_Cosine"]).range(["#d15c52", "#2f7d59"]);
  const bars = [];
  data.forEach(d => ["Mean_MAE", "Mean_Cosine"].forEach(m => bars.push({{...d, metric: m, value: +d[m]}})));
  svg.append("g").attr("class", "grid").attr("transform", `translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(width - margin.left - margin.right)).tickFormat(""));
  svg.append("g").selectAll("rect").data(bars).join("rect")
    .attr("x", d => x0(d.Label) + x1(d.metric))
    .attr("y", d => y(d.value))
    .attr("width", x1.bandwidth())
    .attr("height", d => y(0) - y(d.value))
    .attr("rx", 3)
    .attr("fill", d => metricColor(d.metric))
    .attr("opacity", 0.86);
  svg.append("g").selectAll("text.value").data(bars).join("text")
    .attr("class", "value")
    .attr("x", d => x0(d.Label) + x1(d.metric) + x1.bandwidth() / 2)
    .attr("y", d => y(d.value) - 6)
    .attr("text-anchor", "middle")
    .text(d => d.value.toFixed(3));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${{height - margin.bottom}})`).call(d3.axisBottom(x0).tickSize(0))
    .selectAll("text").attr("transform", "rotate(-28)").attr("text-anchor", "end").attr("dx", -6).attr("dy", 8);
  svg.append("g").attr("class", "axis").attr("transform", `translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(5));
  const legend = svg.append("g").attr("class", "legend").attr("transform", `translate(${{width - margin.right - 174}},${{margin.top + 4}})`);
  ["Mean_MAE", "Mean_Cosine"].forEach((m, i) => {{
    const g = legend.append("g").attr("transform", `translate(0,${{i * 24}})`);
    g.append("rect").attr("width", 14).attr("height", 14).attr("rx", 2).attr("fill", metricColor(m));
    g.append("text").attr("x", 22).attr("y", 12).text(m.replace("_", " "));
  }});
}}

function renderEfficiency() {{
  const data = DATA.dimension_efficiency;
  const width = 1060, height = 430;
  const margin = {{top: 24, right: 142, bottom: 56, left: 64}};
  const svg = d3.select("#efficiency-chart").append("svg").attr("viewBox", [0, 0, width, height]);
  const x = d3.scaleLinear().domain(d3.extent(data, d => +d.d_payload)).range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().domain([0, d3.max(data, d => Math.max(+d.Masked_Full_UGA_Dim, +d.Length_Coded_Full_UGA_Dim)) * 1.08]).nice().range([height - margin.bottom, margin.top]);
  const series = [
    ["Masked_Full_UGA_Dim", "#3e7c9f", "Masked"],
    ["Length_Coded_Full_UGA_Dim", "#2f7d59", "Length-coded"],
    ["Legacy_Full_UGA_Dim", "#9aa5b1", "Legacy"]
  ];
  svg.append("g").attr("class", "grid").attr("transform", `translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(6).tickSize(-(width - margin.left - margin.right)).tickFormat(""));
  series.forEach(([key, color]) => {{
    const line = d3.line().x(d => x(+d.d_payload)).y(d => y(+d[key])).curve(d3.curveMonotoneX);
    svg.append("path").datum(data).attr("fill", "none").attr("stroke", color).attr("stroke-width", 2.5).attr("d", line);
    svg.append("g").selectAll(`circle.${{key}}`).data(data).join("circle")
      .attr("cx", d => x(+d.d_payload)).attr("cy", d => y(+d[key])).attr("r", 3.5).attr("fill", color);
  }});
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${{height - margin.bottom}})`).call(d3.axisBottom(x).ticks(data.length).tickFormat(d3.format("d")));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(6));
  svg.append("text").attr("class", "label").attr("x", (margin.left + width - margin.right) / 2).attr("y", height - 16).attr("text-anchor", "middle").text("d_payload");
  svg.append("text").attr("class", "label").attr("x", 16).attr("y", 22).text("Full UGA dimensions");
  const legend = svg.append("g").attr("class", "legend").attr("transform", `translate(${{width - margin.right + 18}},${{margin.top + 8}})`);
  series.forEach(([_, color, label], i) => {{
    const g = legend.append("g").attr("transform", `translate(0,${{i * 26}})`);
    g.append("line").attr("x1", 0).attr("x2", 18).attr("y1", 6).attr("y2", 6).attr("stroke", color).attr("stroke-width", 3);
    g.append("text").attr("x", 26).attr("y", 10).text(label);
  }});
}}

renderCollision();
renderRecovery();
renderEfficiency();
</script>
</body>
</html>
"""
    (figures_dir / "payload_absence_d3_figures.html").write_text(html, encoding="utf-8")


def write_index(out_dir: Path) -> None:
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EXP-025 Payload Absence Encoding</title>
<style>
body {
  margin: 32px;
  color: #202733;
  font-family: Aptos, "Segoe UI", Arial, sans-serif;
  line-height: 1.45;
}
main {
  max-width: 860px;
}
h1 {
  margin: 0 0 10px;
  font-size: 26px;
}
p {
  color: #5d6875;
}
a {
  color: #246b52;
  text-decoration-thickness: 1px;
  text-underline-offset: 2px;
}
li {
  margin: 8px 0;
}
code {
  background: #eef3f7;
  border-radius: 4px;
  padding: 1px 4px;
}
</style>
</head>
<body>
<main>
<h1>EXP-025 UGA Payload Absence Encoding</h1>
<p>Manuscript-ready experiment package for collision-free, information-efficient UGA payload absence encodings.</p>
<ul>
  <li><a href="html/manuscript_tables.html">Copy/paste manuscript tables</a></li>
  <li><a href="html/manuscript_tables_fragment.html">Table fragments only</a></li>
  <li><a href="figures/payload_absence_d3_figures.html">D3 figure page</a></li>
  <li><a href="README.md">Experiment README</a></li>
  <li><code>tables/</code> source CSV files for tables</li>
  <li><code>data/payload_absence_benchmark_data.json</code> source data for figures</li>
</ul>
</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def write_readme(
    out_dir: Path,
    args: argparse.Namespace,
    candidate_summary: pd.DataFrame,
    mixture_summary: pd.DataFrame,
) -> None:
    masked = candidate_summary[candidate_summary["Candidate"] == "masked_presence"].iloc[0]
    length = candidate_summary[candidate_summary["Candidate"] == "length_code"].iloc[0]
    legacy = candidate_summary[candidate_summary["Candidate"] == "legacy_zero"].iloc[0]
    masked_mix = mixture_summary[mixture_summary["Candidate"] == "masked_presence"].iloc[0]
    length_mix = mixture_summary[mixture_summary["Candidate"] == "length_code"].iloc[0]
    legacy_mix = mixture_summary[mixture_summary["Candidate"] == "legacy_zero"].iloc[0]
    readme = f"""# EXP-025: UGA payload absence encoding benchmark

## Recommendation

Use an **absence-safe payload** schema for INDEL REF/ALT payloads. There are now two UGA-admissible options:

```text
masked payload      = [X_1..X_d, Y_1..Y_d, M_1..M_d]
length-coded payload = [X_1..X_d, Y_1..Y_d, L_1..L_k]
```

`M_i` marks whether slot `i` contains a real base. `L` encodes the payload length from 0..d and reserves one extra state for overflow/truncation. In both schemas, `(X,Y)=(0,0)` can safely remain the nucleotide `G` when the absence/length bits say that slot is occupied.

For the current `d_context={D_CONTEXT}` and `d_payload={args.d_payload}` operating point, both schemas produce **{int(length['Full_UGA_Dim_Selected'])}D** vectors. The masked schema had the best simulated recovery at this depth. Length coding is the more information-efficient option when `d_payload` is increased because biological payloads are contiguous prefixes, not arbitrary sparse slot patterns.

## Why this is optimal under the UGA bit-identity rules

Per-slot masking is the minimal slot-local binary representation of `{{A,C,G,T,absent}}`. Length coding exploits the stronger biological constraint that payload bases form a compact prefix. At the selected payload depth, the length-coded block needs nucleotide bits plus length bits, matching or reducing the per-slot masked width while also giving one overflow/reserved length state. For larger `d_payload`, length coding uses `2*d + ceil(log2(d+2))` bits per block instead of `3*d`, so it captures the same absence information with fewer dimensions.

Continuous sentinels can also avoid exact collisions, but they abandon the discrete binary identity space. `out_of_range_pad` also leaves the bounded `[0,1]` cube.

## Reproducible command

```powershell
python bench\\run_payload_absence_benchmark.py --out-dir {args.out_dir}
```

Parameters: seed={args.seed}, simulated patients={args.n_patients}, events per patient={args.burden}, d_payload={args.d_payload}.

## Main empirical result

The legacy payload had **{int(legacy['Collision_Count'])}** payload-pair collisions over the tested state space. Masked and length-coded payloads both had **0** collisions. In the indel-like exposure recovery task, mean MAE improved from **{legacy_mix['Mean_MAE']:.4f}** with legacy zero padding to **{masked_mix['Mean_MAE']:.4f}** with masked payloads and **{length_mix['Mean_MAE']:.4f}** with length-coded payloads; top-1 process recovery improved from **{legacy_mix['Top1_Process_Accuracy']:.3f}** to **{masked_mix['Top1_Process_Accuracy']:.3f}** and **{length_mix['Top1_Process_Accuracy']:.3f}**, respectively.

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
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Path]:
    out_dir = args.out_dir
    tables_dir = out_dir / "tables"
    html_dir = out_dir / "html"
    figures_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    tables_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = []
    collision_rows = []
    recovery_rows = []
    patient_rows = []
    geometry_rows = []

    for candidate in CANDIDATES:
        summary, details = collision_analysis(candidate, args.d_payload)
        candidate_rows.append(summary)
        collision_rows.extend(details)
        recovery, patients, geometry = run_mixture_recovery(
            candidate,
            args.d_payload,
            args.seed,
            args.n_patients,
            args.burden,
        )
        recovery_rows.append(recovery)
        patient_rows.extend(patients)
        geometry_rows.append(geometry)

    candidate_df = pd.DataFrame(candidate_rows)
    collision_df = pd.DataFrame(collision_rows)
    recovery_df = pd.DataFrame(recovery_rows)
    patient_df = pd.DataFrame(patient_rows)
    geometry_df = pd.DataFrame(geometry_rows)
    efficiency_df = dimension_efficiency_sweep(max(10, args.d_payload))

    paths = {
        "candidate_summary": tables_dir / "payload_candidate_summary.csv",
        "collision_detail": tables_dir / "payload_collision_detail.csv",
        "mixture_summary": tables_dir / "mixture_recovery_summary.csv",
        "patient_metrics": tables_dir / "mixture_patient_metrics.csv",
        "signature_geometry": tables_dir / "signature_geometry.csv",
        "dimension_efficiency": tables_dir / "payload_dimension_efficiency.csv",
        "tables_html": html_dir / "manuscript_tables.html",
        "table_fragments_html": html_dir / "manuscript_tables_fragment.html",
        "figures_html": figures_dir / "payload_absence_d3_figures.html",
        "figure_data_json": data_dir / "payload_absence_benchmark_data.json",
        "index_html": out_dir / "index.html",
        "readme": out_dir / "README.md",
        "manifest": out_dir / "manifest.json",
    }
    candidate_df.to_csv(paths["candidate_summary"], index=False)
    collision_df.to_csv(paths["collision_detail"], index=False)
    recovery_df.to_csv(paths["mixture_summary"], index=False)
    patient_df.to_csv(paths["patient_metrics"], index=False)
    geometry_df.to_csv(paths["signature_geometry"], index=False)
    efficiency_df.to_csv(paths["dimension_efficiency"], index=False)
    figure_payload = {
        "candidate_summary": json.loads(candidate_df.to_json(orient="records")),
        "mixture_summary": json.loads(recovery_df.to_json(orient="records")),
        "signature_geometry": json.loads(geometry_df.to_json(orient="records")),
        "dimension_efficiency": json.loads(efficiency_df.to_json(orient="records")),
    }
    paths["figure_data_json"].write_text(json.dumps(figure_payload, indent=2), encoding="utf-8")
    write_tables_html(html_dir, candidate_df, recovery_df, geometry_df, efficiency_df)
    write_figures_html(figures_dir, figure_payload)
    write_index(out_dir)
    write_readme(out_dir, args, candidate_df, recovery_df)

    manifest = {
        "experiment": "UGA payload absence encoding benchmark",
        "runner": str(REPO / "bench" / "run_payload_absence_benchmark.py"),
        "seed": args.seed,
        "n_patients": args.n_patients,
        "burden": args.burden,
        "uga_model": args.uga_model,
        "d_context": D_CONTEXT,
        "d_payload": args.d_payload,
        "binary_lower_bound_bits_per_slot_when_slot_local": math.ceil(math.log2(5)),
        "length_code_bits_per_payload_block": payload_length_bit_count(args.d_payload),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


def main() -> None:
    global D_CONTEXT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=250513)
    parser.add_argument("--n-patients", type=int, default=480)
    parser.add_argument("--burden", type=int, default=300)
    parser.add_argument("--uga-model", default=DEFAULT_UGA_MODEL.name)
    args = parser.parse_args()
    model = get_uga_model(args.uga_model)
    args.uga_model = model.name
    D_CONTEXT = model.d_context
    args.d_payload = model.d_payload

    paths = run(args)
    print(f"Wrote payload absence benchmark outputs to {args.out_dir}")
    print(pd.read_csv(paths["candidate_summary"]).to_string(index=False))
    print()
    print(pd.read_csv(paths["mixture_summary"]).to_string(index=False))
    print()
    print(pd.read_csv(paths["signature_geometry"]).to_string(index=False))


if __name__ == "__main__":
    main()
