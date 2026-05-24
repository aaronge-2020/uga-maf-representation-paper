#!/usr/bin/env python3
"""Event-level fair representation benchmark for Islam et al. 2022 signatures.

This benchmark avoids projecting the Islam channel matrices directly into UGA.
Instead, it uses the Islam benchmark signatures and exposures as a generative
model, simulates concrete mutation events, and derives both representations from
the same sampled events:

* Standard_Channel: normalized categorical counts from sampled event labels.
* UGA_v1.4_masked: average masked UGA event vectors from those same events.

SBS, DBS, and ID are solved separately. The extraction algorithm, NMF rank,
initialization seeds, NNLS attribution solver, and exposure metrics are fixed
across representations.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))
sys.path.append(str(REPO / "bench"))

import run_islam2022_corrected_representation_benchmark as base  # noqa: E402
from uga_atlas import assemble_uga_vector, channel_kind, get_uga_model, universal_vector_dim  # noqa: E402
from uga_atlas.channels import parse_capped_int  # noqa: E402


DEFAULT_OUT_DIR = (
    REPO
    / "cgr_validation_results"
    / "research"
    / "results"
    / "EXP032_islam2022_event_level_fair_benchmark"
)

BASES = np.array(list("ACGT"))


@dataclass(frozen=True)
class EventPanel:
    name: str
    base_panel: base.Panel
    event_burden: int


def event_panel_from_model(name: str, model_name: str, note: str, event_burden: int = 1000) -> EventPanel:
    model = get_uga_model(model_name)
    return EventPanel(
        name,
        base.Panel(
            name=name,
            kinds=model.kinds,
            d_context=model.d_context,
            d_payload=model.d_payload,
            payload_schema=model.payload_schema,
            note=note,
            uga_model=model.name,
        ),
        event_burden,
    )


PANELS = {
    "sbs1536": event_panel_from_model(
        "sbs1536",
        "islam2022_sbs1536_d10",
        "Event-level SBS with sampled 10-base flanks and masked payload.",
    ),
    "dbs78": event_panel_from_model(
        "dbs78",
        "islam2022_dbs78_d10",
        "Event-level DBS with sampled 10-base flanks and masked payload.",
    ),
    "id83": event_panel_from_model(
        "id83",
        "id83_proxy_d10_dp5",
        "Event-level ID proxy with concrete ref/alt strings, sampled local context, and masked payload.",
    ),
}


def random_dna(rng: np.random.Generator, length: int) -> str:
    if length <= 0:
        return ""
    return "".join(rng.choice(BASES, size=int(length)).tolist())


def padded_left(core_near: str, rng: np.random.Generator, d_context: int) -> str:
    core = str(core_near or "")
    return (random_dna(rng, max(0, d_context - len(core))) + core)[-d_context:]


def padded_right(core_near: str, rng: np.random.Generator, d_context: int) -> str:
    core = str(core_near or "")
    return (core + random_dna(rng, max(0, d_context - len(core))))[:d_context]


def repeat_context(unit: str, copies: int, rng: np.random.Generator, d_context: int) -> tuple[str, str]:
    unit = unit or "C"
    copies = max(1, int(copies))
    repeated = unit * (copies + math.ceil(d_context / max(1, len(unit))) + 1)
    left_core = repeated[-d_context:]
    right_core = repeated[:d_context]
    return padded_left(left_core, rng, d_context), padded_right(right_core, rng, d_context)


def id_event_from_label(
    channel: str,
    rng: np.random.Generator,
    d_context: int,
    d_payload: int,
) -> tuple[str, str, str, str]:
    parts = str(channel).split(":")
    if len(parts) < 4:
        raise ValueError(f"Unsupported ID83 label: {channel}")
    event, subtype = parts[0], parts[1]
    length = max(1, parse_capped_int(parts[2], d_payload))
    aux = max(0, parse_capped_int(parts[3], d_context))

    if subtype in {"C", "T"}:
        payload = subtype * min(length, d_payload)
        left, right = repeat_context(subtype, aux + 1, rng, d_context)
    elif subtype == "repeats":
        unit = random_dna(rng, min(length, d_payload))
        payload = unit
        left, right = repeat_context(unit, aux + 1, rng, d_context)
    elif subtype == "MH":
        payload = random_dna(rng, min(length, d_payload))
        mh_len = max(1, min(aux, len(payload), d_context))
        mh = payload[:mh_len]
        left = padded_left(mh, rng, d_context)
        right = padded_right(mh, rng, d_context)
    else:
        payload = random_dna(rng, min(length, d_payload))
        left = random_dna(rng, d_context)
        right = random_dna(rng, d_context)

    if event == "DEL":
        return left, right, payload, ""
    if event == "INS":
        return left, right, "", payload
    raise ValueError(f"Unsupported ID83 event label: {channel}")


def event_uga_vector(channel: str, panel: base.Panel, rng: np.random.Generator) -> np.ndarray:
    kind = channel_kind(channel)
    dc = panel.d_context
    dp = panel.d_payload
    schema = panel.payload_schema
    if kind == "SBS1536":
        left = padded_left(channel[:2], rng, dc)
        right = padded_right(channel[3:5], rng, dc)
        return assemble_uga_vector(left, right, channel[2], channel[5], dc, dp, schema)
    if kind == "DBS78":
        left = random_dna(rng, dc)
        right = random_dna(rng, dc)
        return assemble_uga_vector(left, right, channel[:2], channel[2:], dc, dp, schema)
    if kind == "ID83":
        left, right, ref, alt = id_event_from_label(channel, rng, dc, dp)
        return assemble_uga_vector(left, right, ref, alt, dc, dp, schema)
    raise ValueError(f"Unsupported event channel: {channel}")


def build_event_bank(
    channels: list[str],
    panel: base.Panel,
    pool_size: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dim = universal_vector_dim(panel.d_context, panel.d_payload, panel.payload_schema)
    bank = np.zeros((len(channels), pool_size, dim), dtype=np.float64)
    rows = []
    for i, ch in enumerate(channels):
        for j in range(pool_size):
            bank[i, j] = event_uga_vector(ch, panel, rng)
        rows.append(
            {
                "Channel": ch,
                "Kind": channel_kind(ch),
                "UGA_Dim": dim,
                "Pool_Size": pool_size,
                "Channel_Event_Mean_Variance": float(np.var(bank[i].mean(axis=0))),
            }
        )
    rounded = np.round(bank.mean(axis=1), 10)
    unique_count = int(np.unique(rounded, axis=0).shape[0])
    diag = pd.DataFrame(rows)
    diag["Unique_Channel_Event_Centroids"] = unique_count
    diag["Centroid_Collision_Count"] = len(channels) - unique_count
    return bank, diag


def simulate_profiles(
    cat_norm: pd.DataFrame,
    sig_norm: pd.DataFrame,
    truth_props: pd.DataFrame,
    event_bank: np.ndarray,
    burden: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    channels = list(cat_norm.index)
    n_channels = len(channels)
    dim = event_bank.shape[2]
    pool_size = event_bank.shape[1]
    sig_probs = sig_norm.to_numpy(dtype=np.float64)
    truth = truth_props.T.to_numpy(dtype=np.float64)
    x_std = np.zeros((truth.shape[0], n_channels), dtype=np.float64)
    x_uga = np.zeros((truth.shape[0], dim), dtype=np.float64)
    for i, exposure in enumerate(truth):
        probs = sig_probs @ exposure
        probs = probs / probs.sum() if probs.sum() > 1e-15 else np.full(n_channels, 1 / n_channels)
        event_channels = rng.choice(n_channels, size=burden, replace=True, p=probs)
        x_std[i] = np.bincount(event_channels, minlength=n_channels) / burden
        pool_idx = rng.integers(0, pool_size, size=burden)
        x_uga[i] = event_bank[event_channels, pool_idx].mean(axis=0)
    return x_std, x_uga, truth


def expected_uga_truth(sig_norm: pd.DataFrame, event_bank: np.ndarray) -> np.ndarray:
    channel_centroids = event_bank.mean(axis=1)
    return sig_norm.T.to_numpy(dtype=np.float64) @ channel_centroids


def run_one_panel(
    scenario_dir: Path,
    panel: EventPanel,
    pool_size: int,
    base_seed: int,
    nmf_init: str,
    nmf_seeds: list[int],
    nmf_max_iter: int,
    nmf_tol: float,
    active_threshold: float,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[dict[str, object]]]:
    scenario = scenario_dir.name
    catalog, signatures, exposures = base.load_inputs(scenario_dir / "Input")
    cat_norm, sig_norm, truth_props = base.panel_inputs(catalog, signatures, exposures, panel.base_panel)
    channels = list(cat_norm.index)
    sig_names = list(sig_norm.columns)
    sample_names = list(cat_norm.columns)
    rank = len(sig_names)
    if rank < 1 or not channels:
        return [], [], [], []

    scenario_num = base.natural_scenario_key(scenario_dir)[0]
    panel_seed_offset = {"sbs1536": 101, "dbs78": 202, "id83": 303}[panel.name]
    seed = int(base_seed + 10_000 * scenario_num + panel_seed_offset)
    event_bank, diag = build_event_bank(channels, panel.base_panel, pool_size, seed)
    diag.insert(0, "Scenario", scenario)
    diag.insert(1, "Panel", panel.name)

    x_std, x_uga, truth = simulate_profiles(
        cat_norm,
        sig_norm,
        truth_props,
        event_bank,
        panel.event_burden,
        seed + 17,
    )
    h_truth_std = sig_norm.T.to_numpy(dtype=np.float64)
    h_truth_uga = expected_uga_truth(sig_norm, event_bank)

    metric_frames: list[pd.DataFrame] = []
    sig_frames: list[pd.DataFrame] = []
    run_infos: list[dict[str, object]] = []

    for representation, x, h_truth in [
        ("Standard_Channel", x_std, h_truth_std),
        ("UGA_v1.4_masked_event", x_uga, h_truth_uga),
    ]:
        start = time.perf_counter()
        w, h, fit_info = base.fit_nmf_best(x, rank, nmf_init, nmf_seeds, nmf_max_iter, nmf_tol)
        component_order, sig_match = base.match_components(h, h_truth, sig_names)
        sig_match.insert(0, "Scenario", scenario)
        sig_match.insert(1, "Panel", panel.name)
        sig_match.insert(2, "Representation", representation)
        sig_frames.append(sig_match)

        pred_nmf = w[:, component_order]
        pred_nnls = base.nnls_attribution(x, h)[:, component_order]
        pred_oracle = base.nnls_attribution(x, h_truth)
        for method, pred in [
            ("NMF_W_matched", pred_nmf),
            ("NNLS_extracted_basis", pred_nnls),
            ("Oracle_true_basis_control", pred_oracle),
        ]:
            metric_frames.append(
                pd.DataFrame(
                    base.exposure_metric_rows(
                        truth,
                        pred,
                        scenario,
                        panel.name,
                        representation,
                        method,
                        sample_names,
                        active_threshold,
                    )
                )
            )
        run_infos.append(
            {
                "Scenario": scenario,
                "Panel": panel.name,
                "Representation": representation,
                "Rank": rank,
                "Samples": int(x.shape[0]),
                "Features": int(x.shape[1]),
                "Event_Burden_Per_Sample": panel.event_burden,
                "Event_Pool_Size_Per_Channel": pool_size,
                "Seed": seed,
                "NMF_Init": nmf_init,
                "NMF_Seed": fit_info["seed"],
                "NMF_Iterations": fit_info["n_iter"],
                "NMF_Reconstruction_Error": fit_info["reconstruction_error"],
                "NMF_Runtime_Sec": fit_info["runtime_sec"],
                "Total_Runtime_Sec": time.perf_counter() - start,
                "UGA_DContext": panel.base_panel.d_context if representation.startswith("UGA") else np.nan,
                "UGA_DPayload": panel.base_panel.d_payload if representation.startswith("UGA") else np.nan,
                "UGA_Payload_Schema": panel.base_panel.payload_schema if representation.startswith("UGA") else "",
                "Panel_Note": panel.base_panel.note,
            }
        )

    return metric_frames, sig_frames, [diag], run_infos


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(index=False, floatfmt=".6f")
    except Exception:
        return df.to_string(index=False)


def write_report(
    out_dir: Path,
    metadata: dict[str, object],
    summary: pd.DataFrame,
    sig_summary: pd.DataFrame,
) -> None:
    main = summary[summary["Attribution_Method"] == "NNLS_extracted_basis"].copy()
    oracle = summary[summary["Attribution_Method"] == "Oracle_true_basis_control"].copy()
    lines = [
        "# EXP032 Islam 2022 Event-Level Fair Representation Benchmark",
        "",
        "## Design",
        "",
        "Islam signatures/exposures are used as the generative truth, but both profiles are derived from the same newly sampled concrete events.",
        "SBS, DBS, and ID are solved separately. The main comparison is Standard_Channel versus UGA_v1.4_masked_event with the same NMF/NNLS workflow.",
        "",
        f"- Input root: `{metadata['input_root']}`",
        f"- Scenarios: `{', '.join(metadata['scenarios'])}`",
        f"- Event burden per sample: `{metadata['event_burdens']}`",
        f"- Event pool size per channel: `{metadata['event_pool_size']}`",
        f"- NMF init: `{metadata['nmf_init']}`",
        f"- NMF seeds: `{metadata['nmf_seeds']}`",
        f"- Active exposure threshold: `{metadata['active_threshold']}`",
        "",
        "## Main Exposure Attribution",
        "",
        markdown_table(main.sort_values(["Panel", "Representation"])),
        "",
        "## Signature Recovery",
        "",
        markdown_table(sig_summary.sort_values(["Panel", "Representation"])),
        "",
        "## Oracle Control",
        "",
        markdown_table(oracle.sort_values(["Panel", "Representation"])),
        "",
        "## Notes",
        "",
        "- The standard and UGA profiles are generated from identical sampled event labels.",
        "- SBS/DBS events use sampled 10-base flanks so UGA is no longer just a categorical-label projection.",
        "- ID83 remains a proxy because Islam provides ID categories, not raw indel loci; this script simulates concrete ref/alt strings and local contexts from those categories.",
    ]
    (out_dir / "RUN_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=base.DEFAULT_INPUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--panels", nargs="+", choices=sorted(PANELS), default=["sbs1536", "dbs78", "id83"])
    parser.add_argument("--event-burden", type=int, default=1000)
    parser.add_argument("--event-pool-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=23032)
    parser.add_argument("--nmf-init", choices=["nndsvda", "nndsvdar", "random"], default="nndsvdar")
    parser.add_argument("--nmf-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--nmf-max-iter", type=int, default=2000)
    parser.add_argument("--nmf-tol", type=float, default=1e-5)
    parser.add_argument("--active-threshold", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = base.discover_scenarios(args.input_root, args.scenarios)
    panels = [
        EventPanel(p.name, p.base_panel, int(args.event_burden))
        for p in (PANELS[name] for name in args.panels)
    ]

    all_metrics: list[pd.DataFrame] = []
    all_sigs: list[pd.DataFrame] = []
    all_diags: list[pd.DataFrame] = []
    all_infos: list[dict[str, object]] = []

    for scenario_dir in scenarios:
        for panel in panels:
            print(f"Running {scenario_dir.name} {panel.name}", flush=True)
            metric_frames, sig_frames, diag_frames, infos = run_one_panel(
                scenario_dir,
                panel,
                args.event_pool_size,
                args.seed,
                args.nmf_init,
                args.nmf_seeds,
                args.nmf_max_iter,
                args.nmf_tol,
                args.active_threshold,
            )
            all_metrics.extend(metric_frames)
            all_sigs.extend(sig_frames)
            all_diags.extend(diag_frames)
            all_infos.extend(infos)

    if not all_metrics:
        raise RuntimeError("No event-level benchmark metrics were produced")

    metrics = pd.concat(all_metrics, ignore_index=True)
    signature_recovery = pd.concat(all_sigs, ignore_index=True)
    diagnostics = pd.concat(all_diags, ignore_index=True)
    run_info = pd.DataFrame(all_infos)
    summary = base.summarize_metrics(metrics)
    scenario_summary = base.summarize_scenarios(metrics)
    sig_summary = (
        signature_recovery.groupby(["Panel", "Representation"], dropna=False)
        .agg(
            N=("Truth_Signature", "count"),
            Mean_Signature_Cosine=("Signature_Cosine", "mean"),
            Median_Signature_Cosine=("Signature_Cosine", "median"),
            Min_Signature_Cosine=("Signature_Cosine", "min"),
        )
        .reset_index()
    )

    metrics.to_csv(out_dir / "patient_metrics.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "summary_metrics.tsv", sep="\t", index=False)
    scenario_summary.to_csv(out_dir / "scenario_summary_metrics.tsv", sep="\t", index=False)
    signature_recovery.to_csv(out_dir / "signature_recovery.tsv", sep="\t", index=False)
    sig_summary.to_csv(out_dir / "signature_recovery_summary.tsv", sep="\t", index=False)
    diagnostics.to_csv(out_dir / "event_bank_diagnostics.tsv", sep="\t", index=False)
    run_info.to_csv(out_dir / "nmf_run_info.tsv", sep="\t", index=False)

    metadata = {
        "input_root": str(args.input_root),
        "out_dir": str(out_dir),
        "scenarios": [p.name for p in scenarios],
        "panels": [p.name for p in panels],
        "event_burdens": {p.name: p.event_burden for p in panels},
        "uga_models": {p.name: p.base_panel.uga_model for p in panels},
        "event_pool_size": args.event_pool_size,
        "seed": args.seed,
        "nmf_init": args.nmf_init,
        "nmf_seeds": args.nmf_seeds,
        "nmf_max_iter": args.nmf_max_iter,
        "nmf_tol": args.nmf_tol,
        "active_threshold": args.active_threshold,
        "uga_payload_schema": "masked",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(out_dir, metadata, summary, sig_summary)
    print(f"Wrote {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
