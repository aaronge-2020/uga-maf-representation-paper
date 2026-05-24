#!/usr/bin/env python3
"""Representation-isolated signature extraction benchmark for Islam et al. 2022.

This script corrects the earlier oracle attribution sanity check. It compares
standard channel-space signatures against UGA-projected signatures while keeping
the extraction algorithm, rank, samples, attribution solver, and metrics fixed.

Main design:
* Split SBS, DBS, and ID panels before extraction.
* Extract signatures de novo with the same sklearn NMF configuration.
* Attribute exposures with the same NNLS solver against the extracted basis.
* Compare recovered exposures to the benchmark ground truth after component
  matching by signature cosine in each representation.

The Islam benchmark is channel-level. SBS1536 and DBS78 labels can be converted
to sequence-resolved UGA label vectors. ID83 labels are categorical summaries,
so the ID arm uses an absence-safe v1.4 proxy that encodes payload absence plus
repeat/microhomology tokens as synthetic clean context. It is not a substitute
for event-level indel loci.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, nnls
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))

from uga_atlas import (  # noqa: E402
    build_uga_basis,
    channel_kind,
    get_uga_model,
)


DEFAULT_INPUT_ROOT = (
    REPO
    / "cgr_validation_results"
    / "research"
    / "data"
    / "EXP030_islam2022_sigprofiler_benchmark"
    / "extracted"
    / "Benchmark"
    / "Extended_Scenarios_withOUT_Noise"
    / "SigProfilerExtractor"
)
DEFAULT_OUT_DIR = (
    REPO
    / "cgr_validation_results"
    / "research"
    / "results"
    / "EXP031_islam2022_corrected_representation_benchmark"
)

@dataclass(frozen=True)
class Panel:
    name: str
    kinds: tuple[str, ...]
    d_context: int
    d_payload: int
    payload_schema: str
    note: str
    uga_model: str = ""


def panel_from_model(name: str, model_name: str, note: str | None = None) -> Panel:
    model = get_uga_model(model_name)
    return Panel(
        name=name,
        kinds=model.kinds,
        d_context=model.d_context,
        d_payload=model.d_payload,
        payload_schema=model.payload_schema,
        note=note or model.note,
        uga_model=model.name,
    )


PANELS = {
    "sbs1536": panel_from_model(
        "sbs1536",
        "islam2022_sbs1536_d10",
        "SBS1536 label UGA with two observed flanks and masked payload slots.",
    ),
    "dbs78": panel_from_model(
        "dbs78",
        "islam2022_dbs78_d10",
        "DBS78 label UGA with sequence payload, zero flanking context, and masked payload slots.",
    ),
    "id83": panel_from_model(
        "id83",
        "id83_proxy_d10_dp5",
        "ID83 categorical proxy using v1.4 masked absence-safe payloads; not event-level indel UGA.",
    ),
}


def natural_scenario_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.rsplit("_", 1)[1]), path.name
    except Exception:
        return 10**9, path.name


def discover_scenarios(root: Path, requested: Iterable[str] | None = None) -> list[Path]:
    if requested:
        out = []
        for name in requested:
            p = root / name
            if not (p / "Input" / "ground.truth.syn.catalog.csv").is_file():
                raise FileNotFoundError(f"Scenario input not found: {p}")
            out.append(p)
        return out
    scenarios = [
        p
        for p in root.glob("Scenario_ext_*")
        if (p / "Input" / "ground.truth.syn.catalog.csv").is_file()
    ]
    return sorted(scenarios, key=natural_scenario_key)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    sums = df.sum(axis=0).replace(0, np.nan)
    return df.div(sums, axis=1).fillna(0.0)


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    sums = arr.sum(axis=1, keepdims=True)
    return np.divide(arr, sums, out=np.zeros_like(arr), where=sums > 1e-15)


def l2_normalize_rows(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1)
    out = np.divide(arr, norms[:, None], out=np.zeros_like(arr), where=norms[:, None] > 1e-15)
    return out, norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-15:
        return 1.0 if np.linalg.norm(a) <= 1e-15 and np.linalg.norm(b) <= 1e-15 else 0.0
    return float(np.dot(a, b) / denom)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denom = a_norm[:, None] * b_norm[None, :]
    out = np.divide(a @ b.T, denom, out=np.zeros((a.shape[0], b.shape[0])), where=denom > 1e-15)
    return out


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    catalog = pd.read_csv(input_dir / "ground.truth.syn.catalog.csv", index_col=0)
    signatures = pd.read_csv(input_dir / "ground.truth.syn.sigs.csv", index_col=0)
    exposures = pd.read_csv(input_dir / "ground.truth.syn.exposures.csv", index_col=0)
    catalog.index = catalog.index.astype(str)
    signatures.index = signatures.index.astype(str)
    exposures.index = exposures.index.astype(str)
    common_channels = [c for c in catalog.index if c in signatures.index]
    common_samples = [s for s in catalog.columns if s in exposures.columns]
    common_sigs = [s for s in signatures.columns if s in exposures.index]
    catalog = catalog.loc[common_channels, common_samples].astype(np.float64)
    signatures = signatures.loc[common_channels, common_sigs].astype(np.float64)
    exposures = exposures.loc[common_sigs, common_samples].astype(np.float64)
    return catalog, signatures, exposures


def panel_inputs(
    catalog: pd.DataFrame,
    signatures: pd.DataFrame,
    exposures: pd.DataFrame,
    panel: Panel,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keep_channels = [ch for ch in catalog.index if channel_kind(ch) in panel.kinds]
    cat = catalog.loc[keep_channels]
    sig = signatures.loc[keep_channels]
    nonzero_samples = cat.sum(axis=0) > 0
    nonzero_sigs = sig.sum(axis=0) > 0
    cat = cat.loc[:, nonzero_samples]
    sig = sig.loc[:, nonzero_sigs]
    exp = exposures.loc[sig.columns, cat.columns]
    return normalize_columns(cat), normalize_columns(sig), normalize_columns(exp)


def fit_nmf_best(
    x: np.ndarray,
    rank: int,
    init: str,
    seeds: list[int],
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    best: tuple[float, np.ndarray, np.ndarray, dict[str, object]] | None = None
    for seed in seeds:
        start = time.perf_counter()
        model = NMF(
            n_components=rank,
            init=init,
            random_state=seed,
            solver="cd",
            beta_loss="frobenius",
            max_iter=max_iter,
            tol=tol,
            alpha_W=0.0,
            alpha_H=0.0,
            l1_ratio=0.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            w = model.fit_transform(x)
        h = model.components_
        runtime = time.perf_counter() - start
        recon = float(np.linalg.norm(x - w @ h) / max(np.linalg.norm(x), 1e-15))
        info = {
            "seed": int(seed),
            "init": init,
            "n_iter": int(model.n_iter_),
            "reconstruction_error": recon,
            "runtime_sec": runtime,
        }
        if best is None or recon < best[0]:
            best = (recon, w, h, info)
    if best is None:
        raise RuntimeError("No NMF fits were attempted")
    return best[1], best[2], best[3]


def match_components(extracted_h: np.ndarray, truth_h: np.ndarray, sig_names: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    sims = cosine_matrix(extracted_h, truth_h)
    component_idx, truth_idx = linear_sum_assignment(-sims)
    order_by_truth = np.full(len(sig_names), -1, dtype=int)
    rows = []
    for comp, truth in zip(component_idx, truth_idx):
        order_by_truth[truth] = comp
        rows.append(
            {
                "Truth_Signature": sig_names[truth],
                "Component": int(comp),
                "Signature_Cosine": float(sims[comp, truth]),
            }
        )
    if np.any(order_by_truth < 0):
        raise RuntimeError("Component matching failed")
    return order_by_truth, pd.DataFrame(rows)


def nnls_attribution(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    basis = np.asarray(h, dtype=np.float64).T
    weights = np.zeros((x.shape[0], h.shape[0]), dtype=np.float64)
    maxiter = max(10_000, 100 * h.shape[0])
    for i in range(x.shape[0]):
        weights[i] = nnls(basis, x[i], maxiter=maxiter)[0]
    return weights


def exposure_metric_rows(
    truth: np.ndarray,
    pred: np.ndarray,
    scenario: str,
    panel: str,
    representation: str,
    attribution_method: str,
    sample_names: list[str],
    active_threshold: float,
) -> list[dict[str, object]]:
    truth_p = normalize_rows(truth)
    pred_p = normalize_rows(pred)
    rows = []
    for i, sample in enumerate(sample_names):
        t = truth_p[i]
        p = pred_p[i]
        t_active = t > active_threshold
        p_active = p > active_threshold
        tp = int(np.sum(t_active & p_active))
        fp = int(np.sum(~t_active & p_active))
        fn = int(np.sum(t_active & ~p_active))
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "Scenario": scenario,
                "Panel": panel,
                "Representation": representation,
                "Attribution_Method": attribution_method,
                "Sample": sample,
                "Cosine_Similarity": cosine_similarity(t, p),
                "MAE": float(np.mean(np.abs(t - p))),
                "L1": float(np.sum(np.abs(t - p))),
                "Precision": float(precision),
                "Recall": float(recall),
                "F1": float(f1),
                "Top1_Match": bool(int(np.argmax(t)) == int(np.argmax(p))),
                "Active_Truth": int(np.sum(t_active)),
                "Active_Pred": int(np.sum(p_active)),
            }
        )
    return rows


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["Panel", "Representation", "Attribution_Method"], dropna=False)
        .agg(
            N=("Sample", "count"),
            Mean_Cosine=("Cosine_Similarity", "mean"),
            Median_Cosine=("Cosine_Similarity", "median"),
            Mean_MAE=("MAE", "mean"),
            Mean_L1=("L1", "mean"),
            Mean_Precision=("Precision", "mean"),
            Mean_Recall=("Recall", "mean"),
            Mean_F1=("F1", "mean"),
            Top1_Accuracy=("Top1_Match", "mean"),
            Mean_Active_Pred=("Active_Pred", "mean"),
        )
        .reset_index()
    )


def summarize_scenarios(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["Scenario", "Panel", "Representation", "Attribution_Method"], dropna=False)
        .agg(
            N=("Sample", "count"),
            Mean_Cosine=("Cosine_Similarity", "mean"),
            Median_Cosine=("Cosine_Similarity", "median"),
            Mean_MAE=("MAE", "mean"),
            Mean_L1=("L1", "mean"),
            Mean_F1=("F1", "mean"),
            Top1_Accuracy=("Top1_Match", "mean"),
        )
        .reset_index()
    )


def write_sigprofiler_inputs(
    out_dir: Path,
    scenario: str,
    panel: str,
    catalog: pd.DataFrame,
    signatures: pd.DataFrame,
) -> tuple[Path, Path]:
    spa_input = out_dir / "sigprofiler_assignment" / scenario / panel / "input"
    spa_input.mkdir(parents=True, exist_ok=True)
    samples_path = spa_input / "samples.tsv"
    signatures_path = spa_input / "signature_database.tsv"
    catalog.to_csv(samples_path, sep="\t")
    signatures.to_csv(signatures_path, sep="\t")
    return samples_path, signatures_path


def run_sigprofiler_assignment(
    out_dir: Path,
    scenario: str,
    panel: str,
    raw_catalog: pd.DataFrame,
    raw_signatures: pd.DataFrame,
    exposures: pd.DataFrame,
    active_threshold: float,
) -> tuple[pd.DataFrame | None, dict[str, object]]:
    info: dict[str, object] = {
        "Scenario": scenario,
        "Panel": panel,
        "attempted": True,
        "ok": False,
    }
    samples_path, signatures_path = write_sigprofiler_inputs(
        out_dir,
        scenario,
        panel,
        raw_catalog,
        raw_signatures,
    )
    spa_dir = out_dir / "sigprofiler_assignment" / scenario / panel / "run"
    spa_dir.mkdir(parents=True, exist_ok=True)
    try:
        from SigProfilerAssignment import Analyzer as sig

        sig.cosmic_fit(
            samples=str(samples_path),
            output=str(spa_dir),
            signature_database=str(signatures_path),
            genome_build="GRCh37",
            cosmic_version=3.5,
            make_plots=False,
            collapse_to_SBS96=False,
            connected_sigs=False,
            verbose=False,
            input_type="matrix",
            context_type=str(raw_catalog.shape[0]),
            export_probabilities=False,
            sample_reconstruction_plots=False,
            cpu=1,
            add_background_signatures=False,
        )
        activities_path = spa_dir / "Assignment_Solution" / "Activities" / "Assignment_Solution_Activities.txt"
        acts = pd.read_csv(activities_path, sep="\t", index_col=0)
        sig_names = [s for s in exposures.index if s in acts.columns]
        pred_props = acts[sig_names].div(acts[sig_names].sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        common_samples = [s for s in pred_props.index if s in exposures.columns]
        pred = pred_props.loc[common_samples, sig_names].to_numpy(dtype=np.float64)
        truth = normalize_columns(exposures.loc[sig_names, common_samples]).T.to_numpy(dtype=np.float64)
        rows = exposure_metric_rows(
            truth,
            pred,
            scenario,
            panel,
            "SigProfilerAssignment",
            "SPA_custom_signature_database",
            common_samples,
            active_threshold,
        )
        info.update({"ok": True, "activities_path": str(activities_path), "n_samples": len(common_samples)})
        return pd.DataFrame(rows), info
    except Exception as exc:
        info.update(
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "samples_path": str(samples_path),
                "signatures_path": str(signatures_path),
            }
        )
        return None, info


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(index=False, floatfmt=".6f")
    except Exception:
        return df.to_string(index=False)


def run_one_panel(
    scenario_dir: Path,
    panel: Panel,
    init: str,
    seeds: list[int],
    max_iter: int,
    tol: float,
    active_threshold: float,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[dict[str, object]]]:
    scenario = scenario_dir.name
    catalog, signatures, exposures = load_inputs(scenario_dir / "Input")
    cat_norm, sig_norm, truth_props = panel_inputs(catalog, signatures, exposures, panel)
    sample_names = list(cat_norm.columns)
    sig_names = list(sig_norm.columns)
    channels = list(cat_norm.index)
    rank = len(sig_names)
    if rank < 1 or len(sample_names) < 1:
        return [], [], [], []

    truth = truth_props.T.to_numpy(dtype=np.float64)
    x_std = cat_norm.T.to_numpy(dtype=np.float64)
    h_truth_std = sig_norm.T.to_numpy(dtype=np.float64)
    uga_basis, channel_diag = build_uga_basis(channels, panel)
    x_uga = x_std @ uga_basis
    h_truth_uga = h_truth_std @ uga_basis
    channel_diag.insert(0, "Scenario", scenario)
    channel_diag.insert(1, "Panel", panel.name)

    metric_frames: list[pd.DataFrame] = []
    sig_frames: list[pd.DataFrame] = []
    diag_frames: list[pd.DataFrame] = [channel_diag]
    run_infos: list[dict[str, object]] = []

    for representation, x, h_truth in [
        ("Standard_Channel", x_std, h_truth_std),
        ("UGA_v1.4_masked", x_uga, h_truth_uga),
    ]:
        w, h, fit_info = fit_nmf_best(x, rank, init, seeds, max_iter, tol)
        component_order, sig_match = match_components(h, h_truth, sig_names)
        sig_match.insert(0, "Scenario", scenario)
        sig_match.insert(1, "Panel", panel.name)
        sig_match.insert(2, "Representation", representation)
        sig_frames.append(sig_match)

        pred_w = w[:, component_order]
        pred_nnls = nnls_attribution(x, h)[:, component_order]

        for method, pred in [
            ("NMF_W_matched", pred_w),
            ("NNLS_extracted_basis", pred_nnls),
        ]:
            metric_frames.append(
                pd.DataFrame(
                    exposure_metric_rows(
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

        oracle = nnls_attribution(x, h_truth)
        metric_frames.append(
            pd.DataFrame(
                exposure_metric_rows(
                    truth,
                    oracle,
                    scenario,
                    panel.name,
                    representation,
                    "Oracle_true_basis_control",
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
                "Features": int(x.shape[1]),
                "Samples": int(x.shape[0]),
                "NMF_Init": init,
                "NMF_Seed": fit_info["seed"],
                "NMF_Iterations": fit_info["n_iter"],
                "NMF_Reconstruction_Error": fit_info["reconstruction_error"],
                "NMF_Runtime_Sec": fit_info["runtime_sec"],
                "UGA_DContext": panel.d_context if representation.startswith("UGA") else np.nan,
                "UGA_DPayload": panel.d_payload if representation.startswith("UGA") else np.nan,
                "UGA_Payload_Schema": panel.payload_schema if representation.startswith("UGA") else "",
                "Panel_Note": panel.note,
            }
        )
    return metric_frames, sig_frames, diag_frames, run_infos


def raw_panel_inputs(
    catalog: pd.DataFrame,
    signatures: pd.DataFrame,
    exposures: pd.DataFrame,
    panel: Panel,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keep_channels = [ch for ch in catalog.index if channel_kind(ch) in panel.kinds]
    cat = catalog.loc[keep_channels]
    sig = signatures.loc[keep_channels]
    nonzero_samples = cat.sum(axis=0) > 0
    nonzero_sigs = sig.sum(axis=0) > 0
    cat = cat.loc[:, nonzero_samples]
    sig = sig.loc[:, nonzero_sigs]
    exp = exposures.loc[sig.columns, cat.columns]
    return cat, sig, exp


def write_report(out_dir: Path, metadata: dict[str, object], summary: pd.DataFrame, sig_summary: pd.DataFrame) -> None:
    main = summary[summary["Attribution_Method"] == "NNLS_extracted_basis"].copy()
    oracle = summary[summary["Attribution_Method"] == "Oracle_true_basis_control"].copy()
    lines = [
        "# EXP031 Corrected Islam 2022 Representation Benchmark",
        "",
        "## Design",
        "",
        "This run isolates representation by holding the synthetic samples, extraction algorithm, rank, attribution solver, and metrics fixed.",
        "SBS, DBS, and ID are solved separately. The main comparison is Standard_Channel versus UGA_v1.4_masked using NMF-extracted signatures and NNLS attribution against those extracted signatures.",
        "",
        f"- Input root: `{metadata['input_root']}`",
        f"- Scenarios: `{', '.join(metadata['scenarios'])}`",
        f"- NMF init: `{metadata['nmf_init']}`",
        f"- NMF seeds: `{metadata['nmf_seeds']}`",
        f"- NMF max_iter: `{metadata['nmf_max_iter']}`",
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
        "These rows use the ground-truth signature basis and are retained only as a sanity ceiling, not as the representation comparison.",
        "",
        markdown_table(oracle.sort_values(["Panel", "Representation"])),
        "",
        "## Caveats",
        "",
        "- The benchmark provides channel-level matrices rather than underlying variant loci.",
        "- SBS1536 and DBS78 UGA use sequence information present in the labels.",
        "- ID83 uses an absence-safe categorical proxy because the labels do not contain literal indel alleles plus genomic context.",
    ]
    (out_dir / "RUN_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--panels", nargs="+", choices=sorted(PANELS), default=["sbs1536", "dbs78", "id83"])
    parser.add_argument("--nmf-init", choices=["nndsvda", "nndsvdar", "random"], default="nndsvda")
    parser.add_argument("--nmf-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--nmf-max-iter", type=int, default=2000)
    parser.add_argument("--nmf-tol", type=float, default=1e-5)
    parser.add_argument("--active-threshold", type=float, default=0.01)
    parser.add_argument("--run-spa", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = discover_scenarios(args.input_root, args.scenarios)
    panels = [PANELS[p] for p in args.panels]

    all_metrics: list[pd.DataFrame] = []
    all_sigs: list[pd.DataFrame] = []
    all_diags: list[pd.DataFrame] = []
    run_infos: list[dict[str, object]] = []
    spa_infos: list[dict[str, object]] = []

    for scenario_dir in scenarios:
        for panel in panels:
            metric_frames, sig_frames, diag_frames, infos = run_one_panel(
                scenario_dir,
                panel,
                args.nmf_init,
                args.nmf_seeds,
                args.nmf_max_iter,
                args.nmf_tol,
                args.active_threshold,
            )
            all_metrics.extend(metric_frames)
            all_sigs.extend(sig_frames)
            all_diags.extend(diag_frames)
            run_infos.extend(infos)

            if args.run_spa:
                catalog, signatures, exposures = load_inputs(scenario_dir / "Input")
                raw_cat, raw_sig, raw_exp = raw_panel_inputs(catalog, signatures, exposures, panel)
                spa_metrics, spa_info = run_sigprofiler_assignment(
                    out_dir,
                    scenario_dir.name,
                    panel.name,
                    raw_cat,
                    raw_sig,
                    raw_exp,
                    args.active_threshold,
                )
                spa_infos.append(spa_info)
                if spa_metrics is not None and not spa_metrics.empty:
                    all_metrics.append(spa_metrics)

    if not all_metrics:
        raise RuntimeError("No benchmark metrics were produced")

    metrics = pd.concat(all_metrics, ignore_index=True)
    signature_recovery = pd.concat(all_sigs, ignore_index=True) if all_sigs else pd.DataFrame()
    channel_diagnostics = pd.concat(all_diags, ignore_index=True) if all_diags else pd.DataFrame()
    run_info_df = pd.DataFrame(run_infos)
    summary = summarize_metrics(metrics)
    scenario_summary = summarize_scenarios(metrics)
    sig_summary = (
        signature_recovery.groupby(["Panel", "Representation"], dropna=False)
        .agg(
            N=("Truth_Signature", "count"),
            Mean_Signature_Cosine=("Signature_Cosine", "mean"),
            Median_Signature_Cosine=("Signature_Cosine", "median"),
            Min_Signature_Cosine=("Signature_Cosine", "min"),
        )
        .reset_index()
        if not signature_recovery.empty
        else pd.DataFrame()
    )

    metrics.to_csv(out_dir / "patient_metrics.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "summary_metrics.tsv", sep="\t", index=False)
    scenario_summary.to_csv(out_dir / "scenario_summary_metrics.tsv", sep="\t", index=False)
    signature_recovery.to_csv(out_dir / "signature_recovery.tsv", sep="\t", index=False)
    sig_summary.to_csv(out_dir / "signature_recovery_summary.tsv", sep="\t", index=False)
    channel_diagnostics.to_csv(out_dir / "uga_channel_diagnostics.tsv", sep="\t", index=False)
    run_info_df.to_csv(out_dir / "nmf_run_info.tsv", sep="\t", index=False)
    pd.DataFrame(spa_infos).to_csv(out_dir / "sigprofiler_assignment_runs.tsv", sep="\t", index=False)

    metadata = {
        "input_root": str(args.input_root),
        "out_dir": str(out_dir),
        "scenarios": [p.name for p in scenarios],
        "panels": [p.name for p in panels],
        "nmf_init": args.nmf_init,
        "nmf_seeds": args.nmf_seeds,
        "nmf_max_iter": args.nmf_max_iter,
        "nmf_tol": args.nmf_tol,
        "active_threshold": args.active_threshold,
        "run_spa": bool(args.run_spa),
        "spa_runs": spa_infos,
        "uga_models": {p.name: p.uga_model for p in panels},
        "id83_caveat": PANELS["id83"].note,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(out_dir, metadata, summary, sig_summary)
    print(f"Wrote {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
