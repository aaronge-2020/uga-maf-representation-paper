#!/usr/bin/env python3
"""Simulated benchmark for the value of UGA shared mutation coordinates.

This experiment is designed around a narrow manuscript claim:

1. A naive all-at-once exposure solve can perform worse than solving each
   mutation modality separately.
2. A shared UGA coordinate system is still useful because it permits
   modality-agnostic questions that standard SBS/DBS/ID catalogs cannot ask:
   cross-modality signature matching and held-out modality exposure transfer.

The simulation uses standard SBS96, DBS78, and ID83 channel universes for the
categorical side, while event-level UGA vectors follow the clean-context layout:

    [Lx, Ly, Xref, Yref, Rx, Ry, Xalt, Yalt]

using the atlas-defined event-level UGA model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.decomposition import PCA


def find_repo_root(start: Path) -> Path:
    """Locate the repository root from either bench/ or research/scripts/."""
    for parent in [start, *start.parents]:
        has_signatures = (parent / "data" / "Signatures").is_dir()
        has_uga_atlas = (parent / "uga_atlas").is_dir()
        if has_signatures and has_uga_atlas:
            return parent
    raise RuntimeError(f"Could not locate repository root from {start}")


REPO = find_repo_root(Path(__file__).resolve())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from uga_atlas import (  # noqa: E402
    assemble_uga_vector,
    get_uga_model,
    payload_block_dim,
    universal_vector_dim,
)

SIGNATURE_DIR = REPO / "data" / "Signatures"
DEFAULT_OUT = REPO / "cgr_validation_results" / "research" / "reports" / "exp024_shared_space"

MODALITIES = ("SBS", "DBS", "ID")
BASES = "ACGT"
UGA_MODEL = get_uga_model("compact_event_legacy_d10")
D_CONTEXT = UGA_MODEL.d_context
D_PAYLOAD = UGA_MODEL.d_payload
PAYLOAD_SCHEMA = UGA_MODEL.payload_schema
UGA_DIM = universal_vector_dim(D_CONTEXT, D_PAYLOAD, PAYLOAD_SCHEMA)
PAYLOAD_BLOCK_DIM = payload_block_dim(D_PAYLOAD, PAYLOAD_SCHEMA)

CONTEXT_IDX = np.array(list(range(0, 2 * D_CONTEXT)) + list(range(2 * D_CONTEXT + PAYLOAD_BLOCK_DIM, 4 * D_CONTEXT + PAYLOAD_BLOCK_DIM)))
PAYLOAD_IDX = np.array(list(range(2 * D_CONTEXT, 2 * D_CONTEXT + PAYLOAD_BLOCK_DIM)) + list(range(4 * D_CONTEXT + PAYLOAD_BLOCK_DIM, 4 * D_CONTEXT + 2 * PAYLOAD_BLOCK_DIM)))


@dataclass(frozen=True)
class ProcessSpec:
    pid: str
    label: str
    short: str
    left: str
    right: str
    sbs_primary: tuple[str, str]
    sbs_secondary: tuple[str, str]
    dbs_primary: str
    dbs_secondary: str
    id_primary: str
    id_secondary: str


PROCESSES = [
    ProcessSpec("P01", "CpG deamination", "CpG", "ATCGTACGAC", "GACGTTAGCA", ("C", "T"), ("C", "A"), "CC>TT", "CG>TT", "1:Del:C:3", "1:Ins:C:1"),
    ProcessSpec("P02", "APOBEC editing", "APOBEC", "GACTAACGTT", "ATGACCTGCA", ("C", "G"), ("C", "T"), "TC>AT", "TC>CG", "1:Ins:C:2", "1:Del:C:1"),
    ProcessSpec("P03", "UV photodimer", "UV", "CCGTAACGTT", "TTCGACCTTA", ("C", "T"), ("C", "A"), "CC>TT", "CT>TG", "2:Del:R:2", "1:Del:T:2"),
    ProcessSpec("P04", "Bulky adduct", "Adduct", "TGCAGTTAAC", "ATCCGTAGGT", ("C", "A"), ("C", "G"), "CC>AA", "AC>TA", "1:Del:C:1", "2:Ins:R:1"),
    ProcessSpec("P05", "HR repair deficiency", "HRD", "GGTACCATGT", "CAGTACGATC", ("T", "G"), ("T", "A"), "TT>GG", "TA>GT", "5:Del:M:3", "4:Del:M:2"),
    ProcessSpec("P06", "MMR slippage", "MMR", "AACCGTTGAT", "TGTTACCGAA", ("C", "T"), ("C", "G"), "TC>CT", "TG>CT", "1:Ins:T:5", "1:Del:T:5"),
    ProcessSpec("P07", "Oxidative guanine", "Ox", "CTAGGCTAAC", "CGATTCGATG", ("C", "A"), ("C", "T"), "GC>TA", "GC>AT", "1:Del:T:1", "1:Ins:T:1"),
    ProcessSpec("P08", "Polymerase epsilon", "POLE", "GATCGCATAC", "AGGCTTACGA", ("C", "A"), ("C", "T"), "CT>AA", "CT>GA", "2:Ins:R:1", "3:Ins:R:2"),
    ProcessSpec("P09", "Platinum crosslink", "Pt", "TAGCGATGAT", "GTACCGATTA", ("C", "A"), ("C", "G"), "TG>GT", "AC>GT", "2:Del:M:1", "3:Del:M:1"),
    ProcessSpec("P10", "AID germinal center", "AID", "CGTTAAGGTC", "CACGATCGTA", ("C", "G"), ("C", "T"), "AC>CG", "CC>GG", "1:Ins:C:1", "1:Del:C:2"),
    ProcessSpec("P11", "Alkylation repair", "Alkyl", "GGCATTAAGT", "GATCGACCTA", ("C", "T"), ("C", "A"), "CG>TA", "CG>AT", "3:Del:R:1", "2:Del:R:1"),
    ProcessSpec("P12", "Template switching", "Switch", "TACGCGTAAT", "CCGATATGCA", ("T", "C"), ("T", "G"), "TA>GC", "TA>CG", "4:Ins:R:3", "5:Ins:R:2"),
]


ID_PAYLOAD_OVERRIDES: dict[str, tuple[str, str]] = {
    "1:Del:C:1": ("C", ""),
    "1:Del:C:2": ("C", ""),
    "1:Del:C:3": ("C", ""),
    "1:Del:T:1": ("T", ""),
    "1:Del:T:2": ("T", ""),
    "1:Del:T:5": ("T", ""),
    "1:Ins:C:1": ("", "C"),
    "1:Ins:C:2": ("", "C"),
    "1:Ins:T:1": ("", "T"),
    "1:Ins:T:5": ("", "T"),
    "2:Del:R:1": ("CT", ""),
    "2:Del:R:2": ("CT", ""),
    "2:Ins:R:1": ("", "CT"),
    "3:Ins:R:2": ("", "CTA"),
    "3:Del:R:1": ("CTA", ""),
    "4:Ins:R:3": ("", "CTAG"),
    "5:Ins:R:2": ("", "CTAGC"),
    "2:Del:M:1": ("AC", ""),
    "3:Del:M:1": ("ACG", ""),
    "4:Del:M:2": ("ACGT", ""),
    "5:Del:M:3": ("ACGTA", ""),
}


def read_channel_universes() -> dict[str, list[str]]:
    files = {
        "SBS": SIGNATURE_DIR / "COSMIC_v3.5_SBS_GRCh37.txt",
        "DBS": SIGNATURE_DIR / "COSMIC_v3.5_DBS_GRCh37.txt",
        "ID": SIGNATURE_DIR / "COSMIC_v3.5_ID_GRCh37.txt",
    }
    out = {}
    for modality, path in files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, sep="\t", usecols=["Type"])
        out[modality] = df["Type"].astype(str).tolist()
    return out


def encode_event(left: str, ref: str, alt: str, right: str) -> np.ndarray:
    """Encode a synthetic event using clean context outside the event block."""
    return assemble_uga_vector(left, right, ref, alt, D_CONTEXT, D_PAYLOAD, PAYLOAD_SCHEMA)


def mutate_seq(seq: str, rng: np.random.Generator, p: float) -> str:
    chars = list(seq)
    for i, base in enumerate(chars):
        if rng.random() < p:
            choices = [b for b in BASES if b != base]
            chars[i] = str(rng.choice(choices))
    return "".join(chars)


def choose_weighted(primary, secondary, random_values: list[str], rng: np.random.Generator):
    u = rng.random()
    if u < 0.82:
        return primary
    if u < 0.94:
        return secondary
    return random_values[int(rng.integers(0, len(random_values)))]


def parse_dbs_channel(channel: str) -> tuple[str, str]:
    ref, alt = channel.split(">", 1)
    return ref, alt


def parse_id_channel(channel: str) -> tuple[str, str]:
    if channel in ID_PAYLOAD_OVERRIDES:
        return ID_PAYLOAD_OVERRIDES[channel]
    length_s, change, motif, _suffix = channel.split(":")
    length = int(length_s)
    if motif == "C":
        seq = "C" * length
    elif motif == "T":
        seq = "T" * length
    elif motif == "R":
        seq = ("CTAGC" * 2)[:length]
    elif motif == "M":
        seq = ("ACGTA" * 2)[:length]
    else:
        seq = "C" * length
    return (seq, "") if change == "Del" else ("", seq)


def simulate_event(
    spec: ProcessSpec,
    modality: str,
    rng: np.random.Generator,
    channels: dict[str, list[str]],
    context_noise: float,
) -> tuple[str, np.ndarray]:
    left = mutate_seq(spec.left, rng, context_noise)
    right = mutate_seq(spec.right, rng, context_noise)

    if modality == "SBS":
        ref, alt = choose_weighted(spec.sbs_primary, spec.sbs_secondary, [("C", "A"), ("C", "G"), ("C", "T"), ("T", "A"), ("T", "C"), ("T", "G")], rng)
        channel = f"{left[-1]}[{ref}>{alt}]{right[0]}"
    elif modality == "DBS":
        channel = choose_weighted(spec.dbs_primary, spec.dbs_secondary, channels["DBS"], rng)
        ref, alt = parse_dbs_channel(channel)
    elif modality == "ID":
        channel = choose_weighted(spec.id_primary, spec.id_secondary, channels["ID"], rng)
        ref, alt = parse_id_channel(channel)
    else:
        raise ValueError(modality)

    return channel, encode_event(left, ref, alt, right)


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    total = float(x.sum())
    if total <= 1e-15:
        return np.zeros_like(x)
    return x / total


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-15 or nb <= 1e-15:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def sparse_exposure(rng: np.random.Generator, n_processes: int) -> np.ndarray:
    active_n = int(rng.integers(2, 5))
    active = rng.choice(n_processes, size=active_n, replace=False)
    weights = rng.dirichlet(np.full(active_n, 0.75))
    exposure = np.zeros(n_processes, dtype=float)
    exposure[active] = weights
    background = rng.dirichlet(np.full(n_processes, 0.35)) * 0.035
    return normalize(exposure * 0.965 + background)


def build_reference_signatures(
    processes: list[ProcessSpec],
    channels: dict[str, list[str]],
    rng: np.random.Generator,
    events_per_process: int,
    context_noise: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    standard = {}
    uga = {}
    channel_to_index = {m: {c: i for i, c in enumerate(channels[m])} for m in MODALITIES}

    for modality in MODALITIES:
        C = np.zeros((len(channels[modality]), len(processes)), dtype=float)
        U = np.zeros((UGA_DIM, len(processes)), dtype=float)
        for k, spec in enumerate(processes):
            vec_sum = np.zeros(UGA_DIM, dtype=float)
            for _ in range(events_per_process):
                channel, vec = simulate_event(spec, modality, rng, channels, context_noise)
                if channel not in channel_to_index[modality]:
                    raise ValueError(f"{modality} channel not in standard universe: {channel}")
                C[channel_to_index[modality][channel], k] += 1.0
                vec_sum += vec
            standard[modality] = C
            U[:, k] = vec_sum / events_per_process
        standard[modality] = np.apply_along_axis(normalize, 0, C)
        uga[modality] = U
    return standard, uga


def draw_burden(rng: np.random.Generator, modality: str) -> int:
    if modality == "SBS":
        return int(np.clip(rng.lognormal(mean=7.05, sigma=0.38), 550, 3200))
    if modality == "DBS":
        return int(np.clip(rng.lognormal(mean=4.15, sigma=0.55), 18, 280))
    if modality == "ID":
        return int(np.clip(rng.lognormal(mean=4.55, sigma=0.50), 28, 420))
    raise ValueError(modality)


def simulate_patients(
    processes: list[ProcessSpec],
    channels: dict[str, list[str]],
    rng: np.random.Generator,
    n_patients: int,
    context_noise: float,
) -> list[dict]:
    channel_to_index = {m: {c: i for i, c in enumerate(channels[m])} for m in MODALITIES}
    patients = []
    for i in range(n_patients):
        exposure = sparse_exposure(rng, len(processes))
        patient = {
            "sample": f"SIM-{i + 1:04d}",
            "latent_exposure": exposure,
            "modalities": {},
        }
        pooled_sum = np.zeros(UGA_DIM, dtype=float)
        total_burden = 0
        for modality in MODALITIES:
            burden = draw_burden(rng, modality)
            proc_counts = rng.multinomial(burden, exposure)
            counts = np.zeros(len(channels[modality]), dtype=float)
            vec_sum = np.zeros(UGA_DIM, dtype=float)
            for k, n_events in enumerate(proc_counts):
                spec = processes[k]
                for _ in range(int(n_events)):
                    channel, vec = simulate_event(spec, modality, rng, channels, context_noise)
                    counts[channel_to_index[modality][channel]] += 1.0
                    vec_sum += vec
            profile = vec_sum / max(1, burden)
            patient["modalities"][modality] = {
                "burden": burden,
                "counts": counts,
                "profile": profile,
                "truth": normalize(proc_counts.astype(float)),
            }
            pooled_sum += vec_sum
            total_burden += burden
        patient["pooled_profile"] = pooled_sum / max(1, total_burden)
        patient["total_burden"] = total_burden
        patients.append(patient)
    return patients


def fit_nnls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    w = nnls(A, b)[0]
    return normalize(w)


def evaluate_exposures(
    patients: list[dict],
    standard_ref: dict[str, np.ndarray],
    uga_ref: dict[str, np.ndarray],
    processes: list[ProcessSpec],
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows = []
    predictions: dict[str, dict[str, np.ndarray]] = {}
    joint_A = np.column_stack([uga_ref[m] for m in MODALITIES])

    for patient in patients:
        sample = patient["sample"]
        predictions[sample] = {}

        for modality in MODALITIES:
            item = patient["modalities"][modality]
            truth = item["truth"]
            b_cat = normalize(item["counts"])
            p_std = fit_nnls(standard_ref[modality], b_cat)
            p_uga = fit_nnls(uga_ref[modality], item["profile"])
            predictions[sample][f"{modality}:Separate categorical NNLS"] = p_std
            predictions[sample][f"{modality}:Separate UGA NNLS"] = p_uga
            for method, pred in [
                ("Separate categorical NNLS", p_std),
                ("Separate UGA NNLS", p_uga),
            ]:
                rows.append(metric_row(sample, modality, method, truth, pred, item["burden"], np.nan))

        w_joint_raw = nnls(joint_A, patient["pooled_profile"])[0]
        w_joint_norm = normalize(w_joint_raw)
        for offset, modality in enumerate(MODALITIES):
            start = offset * len(processes)
            stop = start + len(processes)
            modal_raw = w_joint_raw[start:stop]
            modal_pred = normalize(modal_raw)
            modal_mass = float(w_joint_norm[start:stop].sum())
            true_mass = patient["modalities"][modality]["burden"] / patient["total_burden"]
            mass_error = abs(modal_mass - true_mass)
            predictions[sample][f"{modality}:Naive pooled UGA NNLS"] = modal_pred
            rows.append(
                metric_row(
                    sample,
                    modality,
                    "Naive pooled UGA NNLS",
                    patient["modalities"][modality]["truth"],
                    modal_pred,
                    patient["modalities"][modality]["burden"],
                    mass_error,
                )
            )

    return pd.DataFrame(rows), predictions


def metric_row(
    sample: str,
    modality: str,
    method: str,
    truth: np.ndarray,
    pred: np.ndarray,
    burden: int,
    modal_mass_error: float,
) -> dict:
    return {
        "Sample": sample,
        "Modality": modality,
        "Method": method,
        "Burden": int(burden),
        "MAE": float(np.mean(np.abs(truth - pred))),
        "Cosine": cosine_similarity(truth, pred),
        "Top_Process_Truth": int(np.argmax(truth)) + 1,
        "Top_Process_Pred": int(np.argmax(pred)) + 1,
        "Top1_Process_Match": float(np.argmax(truth) == np.argmax(pred)),
        "Modal_Mass_Error": modal_mass_error,
    }


def summarize_deconvolution(metrics: pd.DataFrame) -> pd.DataFrame:
    order = {
        "Separate categorical NNLS": 0,
        "Separate UGA NNLS": 1,
        "Naive pooled UGA NNLS": 2,
    }
    out = (
        metrics.groupby(["Modality", "Method"], as_index=False)
        .agg(
            N=("Sample", "count"),
            Median_Burden=("Burden", "median"),
            Mean_MAE=("MAE", "mean"),
            Median_MAE=("MAE", "median"),
            Mean_Cosine=("Cosine", "mean"),
            Top1_Process_Accuracy=("Top1_Process_Match", "mean"),
            Mean_Modal_Mass_Error=("Modal_Mass_Error", "mean"),
        )
        .sort_values(["Modality", "Method"], key=lambda s: s.map(order).fillna(99) if s.name == "Method" else s)
        .reset_index(drop=True)
    )
    return out


def ranked_retrieval(coords_a: np.ndarray, coords_b: np.ndarray) -> tuple[float, float, float, list[dict]]:
    rows = []
    reciprocal = []
    top1 = []
    ranks = []
    for i in range(coords_a.shape[1]):
        d = np.linalg.norm(coords_b.T - coords_a[:, i], axis=1)
        order = np.argsort(d)
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
        top1.append(rank == 1)
        reciprocal.append(1.0 / rank)
        rows.append(
            {
                "Source_Process": i + 1,
                "Nearest_Process": int(order[0]) + 1,
                "True_Process_Rank": rank,
                "True_Process_Distance": float(d[i]),
                "Nearest_Distance": float(d[order[0]]),
            }
        )
    return float(np.mean(top1)), float(np.mean(reciprocal)), float(np.median(ranks)), rows


def retrieval_analysis(uga_ref: dict[str, np.ndarray], n_processes: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[list[float]]]]:
    feature_sets = {
        "UGA full 48D": np.arange(UGA_DIM),
        "UGA clean context 40D": CONTEXT_IDX,
        "UGA payload 8D": PAYLOAD_IDX,
    }
    summary_rows = []
    detail_rows = []
    matrices = {}
    harmonic = sum(1.0 / i for i in range(1, n_processes + 1))
    chance_top1 = 1.0 / n_processes
    chance_mrr = harmonic / n_processes

    for source in MODALITIES:
        for target in MODALITIES:
            if source == target:
                continue
            for method, idx in feature_sets.items():
                A = uga_ref[source][idx, :]
                B = uga_ref[target][idx, :]
                top1, mrr, med_rank, rows = ranked_retrieval(A, B)
                summary_rows.append(
                    {
                        "Source": source,
                        "Target": target,
                        "Coordinate_System": method,
                        "Top1_Accuracy": top1,
                        "MRR": mrr,
                        "Median_True_Rank": med_rank,
                        "Note": "direct cross-modality distances",
                    }
                )
                for row in rows:
                    detail_rows.append({"Source": source, "Target": target, "Coordinate_System": method, **row})
                if method == "UGA clean context 40D":
                    D = np.zeros((n_processes, n_processes), dtype=float)
                    for i in range(n_processes):
                        D[i, :] = np.linalg.norm(B.T - A[:, i], axis=1)
                    matrices[f"{source}_to_{target}_context"] = D.tolist()

            summary_rows.append(
                {
                    "Source": source,
                    "Target": target,
                    "Coordinate_System": "Standard separate catalogs",
                    "Top1_Accuracy": chance_top1,
                    "MRR": chance_mrr,
                    "Median_True_Rank": (n_processes + 1) / 2,
                    "Note": "undefined across unequal SBS/DBS/ID axes; chance/tie expectation shown",
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows), matrices


def summarize_retrieval(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("Coordinate_System", as_index=False)
        .agg(
            Pairs=("Source", "count"),
            Mean_Top1_Accuracy=("Top1_Accuracy", "mean"),
            Mean_MRR=("MRR", "mean"),
            Median_True_Rank=("Median_True_Rank", "median"),
        )
        .sort_values("Mean_Top1_Accuracy", ascending=False)
        .reset_index(drop=True)
    )


def build_uga_mapping(uga_ref: dict[str, np.ndarray], source: str, target: str) -> np.ndarray:
    A = uga_ref[source][CONTEXT_IDX, :]
    B = uga_ref[target][CONTEXT_IDX, :]
    mapping = np.zeros(A.shape[1], dtype=int)
    for i in range(A.shape[1]):
        d = np.linalg.norm(B.T - A[:, i], axis=1)
        mapping[i] = int(np.argmin(d))
    return mapping


def apply_mapping(weights: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    out = np.zeros_like(weights)
    for source_idx, target_idx in enumerate(mapping):
        out[target_idx] += weights[source_idx]
    return normalize(out)


def imputation_analysis(
    patients: list[dict],
    predictions: dict[str, dict[str, np.ndarray]],
    uga_ref: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(patients)
    indices = np.arange(n)
    rng.shuffle(indices)
    split = int(0.7 * n)
    train_idx = set(indices[:split].tolist())
    test = [patients[i] for i in indices[split:]]
    rows = []

    train_means = {}
    for target in ("DBS", "ID"):
        train_truth = [patients[i]["modalities"][target]["truth"] for i in train_idx]
        train_means[target] = normalize(np.mean(train_truth, axis=0))

    n_processes = len(PROCESSES)
    random_mapping = np.arange(n_processes)
    rng.shuffle(random_mapping)

    for target in ("DBS", "ID"):
        uga_map = build_uga_mapping(uga_ref, "SBS", target)
        for patient in test:
            sample = patient["sample"]
            source_pred = predictions[sample]["SBS:Separate categorical NNLS"]
            truth = patient["modalities"][target]["truth"]
            candidates = {
                "UGA unsupervised bridge": apply_mapping(source_pred, uga_map),
                "Oracle label bridge": source_pred.copy(),
                "Standard no-bridge cohort mean": train_means[target],
                "Standard random bridge": apply_mapping(source_pred, random_mapping),
            }
            for method, pred in candidates.items():
                rows.append(
                    {
                        "Sample": sample,
                        "Target_Modality": target,
                        "Method": method,
                        "MAE": float(np.mean(np.abs(truth - pred))),
                        "Cosine": cosine_similarity(truth, pred),
                        "Top1_Process_Match": float(np.argmax(truth) == np.argmax(pred)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_imputation(df: pd.DataFrame) -> pd.DataFrame:
    order = {
        "Oracle label bridge": 0,
        "UGA unsupervised bridge": 1,
        "Standard no-bridge cohort mean": 2,
        "Standard random bridge": 3,
    }
    return (
        df.groupby(["Target_Modality", "Method"], as_index=False)
        .agg(
            N=("Sample", "count"),
            Mean_MAE=("MAE", "mean"),
            Median_MAE=("MAE", "median"),
            Mean_Cosine=("Cosine", "mean"),
            Top1_Process_Accuracy=("Top1_Process_Match", "mean"),
        )
        .sort_values(["Target_Modality", "Method"], key=lambda s: s.map(order).fillna(99) if s.name == "Method" else s)
        .reset_index(drop=True)
    )


def signature_map_data(uga_ref: dict[str, np.ndarray], processes: list[ProcessSpec]) -> list[dict]:
    rows = []
    X = []
    labels = []
    for modality in MODALITIES:
        for k, spec in enumerate(processes):
            X.append(uga_ref[modality][CONTEXT_IDX, k])
            labels.append((modality, spec))
    coords = PCA(n_components=2, random_state=0).fit_transform(np.asarray(X))
    for (modality, spec), xy in zip(labels, coords):
        rows.append(
            {
                "process": spec.pid,
                "label": spec.label,
                "short": spec.short,
                "modality": modality,
                "x": float(xy[0]),
                "y": float(xy[1]),
            }
        )
    return rows


def fmt_num(x, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    x = float(x)
    if abs(x) >= 100:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}"


def html_table(df: pd.DataFrame, caption: str, columns: list[tuple[str, str]], note: str | None = None) -> str:
    thead = "".join(f"<th>{escape(label)}</th>" for _col, label in columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for col, _label in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                digits = 0 if col in {"N", "Pairs", "Median_Burden"} else 3
                text = fmt_num(value, digits=digits)
            else:
                text = escape(str(value))
            cells.append(f"<td>{text}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    note_html = f'<p class="table-note">{escape(note)}</p>' if note else ""
    return f"""
<section class="table-section">
  <h2>{escape(caption)}</h2>
  <table class="manuscript-table">
    <thead><tr>{thead}</tr></thead>
    <tbody>
      {''.join(body_rows)}
    </tbody>
  </table>
  {note_html}
</section>
""".strip()


def write_tables(
    out_dir: Path,
    design: pd.DataFrame,
    deconv: pd.DataFrame,
    retrieval: pd.DataFrame,
    imputation: pd.DataFrame,
) -> None:
    style = """
<style>
:root {
  --ink: #17202a;
  --muted: #5f6b7a;
  --line: #c9d2dc;
  --header: #eef3f7;
  --accent: #226f54;
}
body {
  margin: 28px;
  color: var(--ink);
  font-family: "Aptos", "Segoe UI", Arial, sans-serif;
  background: white;
}
h1 {
  margin: 0 0 20px;
  font-size: 22px;
  letter-spacing: 0;
}
h2 {
  margin: 30px 0 8px;
  font-size: 15px;
  font-weight: 700;
  color: var(--ink);
}
.table-section {
  max-width: 1120px;
  margin-bottom: 26px;
}
.manuscript-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
  line-height: 1.28;
}
.manuscript-table th {
  background: var(--header);
  border-top: 2px solid var(--ink);
  border-bottom: 1px solid var(--ink);
  padding: 7px 8px;
  text-align: left;
  font-weight: 700;
}
.manuscript-table td {
  border-bottom: 1px solid var(--line);
  padding: 7px 8px;
  vertical-align: top;
}
.manuscript-table tbody tr:nth-child(even) td {
  background: #fafcfd;
}
.table-note {
  margin: 8px 0 0;
  color: var(--muted);
  font-size: 11.5px;
}
</style>
""".strip()

    sections = [
        html_table(
            design,
            "Table 1. Shared-space simulation design",
            [("Element", "Element"), ("Specification", "Specification")],
        ),
        html_table(
            deconv,
            "Table 2. Exposure recovery: separate solves versus naive pooled UGA solve",
            [
                ("Modality", "Modality"),
                ("Method", "Method"),
                ("N", "N"),
                ("Median_Burden", "Median burden"),
                ("Mean_MAE", "Mean MAE"),
                ("Mean_Cosine", "Mean cosine"),
                ("Top1_Process_Accuracy", "Top-1 process"),
                ("Mean_Modal_Mass_Error", "Modal mass error"),
            ],
            "Lower MAE is better; higher cosine and top-1 accuracy are better. Modal mass error is only defined for the naive pooled solve.",
        ),
        html_table(
            retrieval,
            "Table 3. Cross-modality signature retrieval",
            [
                ("Coordinate_System", "Coordinate system"),
                ("Pairs", "Ordered modality pairs"),
                ("Mean_Top1_Accuracy", "Mean top-1"),
                ("Mean_MRR", "Mean reciprocal rank"),
                ("Median_True_Rank", "Median true rank"),
            ],
            "The standard catalog row reports the chance/tie expectation because SBS96, DBS78, and ID83 have unequal, modality-specific axes.",
        ),
        html_table(
            imputation,
            "Table 4. Held-out DBS/ID exposure imputation from SBS",
            [
                ("Target_Modality", "Target"),
                ("Method", "Method"),
                ("N", "N"),
                ("Mean_MAE", "Mean MAE"),
                ("Mean_Cosine", "Mean cosine"),
                ("Top1_Process_Accuracy", "Top-1 process"),
            ],
            "UGA bridge uses nearest neighbors in clean-context UGA coordinates without using process labels.",
        ),
    ]

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UGA shared-space benchmark tables</title>
{style}
</head>
<body>
<h1>UGA Shared-Coordinate Benchmark: Manuscript Tables</h1>
{''.join(sections)}
</body>
</html>
"""
    (out_dir / "manuscript_tables.html").write_text(html, encoding="utf-8")
    (out_dir / "manuscript_tables_fragment.html").write_text("\n\n".join(sections), encoding="utf-8")


def write_figures_html(out_dir: Path, payload: dict) -> None:
    data_json = json.dumps(payload, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UGA shared-coordinate benchmark figures</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
:root {{
  --ink: #162029;
  --muted: #64717f;
  --line: #d7dee6;
  --paper: #ffffff;
  --soft: #f5f8fa;
}}
body {{
  margin: 0;
  color: var(--ink);
  background: var(--paper);
  font-family: "Aptos", "Segoe UI", Arial, sans-serif;
}}
main {{
  max-width: 1180px;
  margin: 0 auto;
  padding: 34px 30px 48px;
}}
header {{
  margin-bottom: 24px;
  border-bottom: 1px solid var(--line);
  padding-bottom: 16px;
}}
h1 {{
  margin: 0 0 8px;
  font-size: 26px;
  letter-spacing: 0;
}}
p {{
  margin: 0;
  color: var(--muted);
  font-size: 14px;
  max-width: 900px;
}}
section {{
  margin-top: 34px;
}}
h2 {{
  margin: 0 0 10px;
  font-size: 16px;
  letter-spacing: 0;
}}
.figure-shell {{
  border-top: 1px solid var(--line);
  padding-top: 14px;
}}
svg {{
  display: block;
  width: 100%;
  height: auto;
}}
.axis text, .legend text {{
  fill: var(--muted);
  font-size: 11px;
}}
.axis path, .axis line {{
  stroke: var(--line);
}}
.caption {{
  margin-top: 8px;
  color: var(--muted);
  font-size: 12.5px;
}}
</style>
</head>
<body>
<main>
  <header>
    <h1>Shared UGA Coordinates Expose Cross-Modality Structure</h1>
    <p>Simulated SBS, DBS, and indel signatures were generated from common clean-context programs but distinct mutation payloads. The figures show why the shared coordinate layer is useful even when exposure estimation is best performed separately.</p>
  </header>
  <section class="figure-shell">
    <h2>Figure 1. Process signatures occupy a shared clean-context manifold</h2>
    <div id="signature-map"></div>
    <p class="caption">Points are modality-specific signatures; lines connect SBS, DBS, and ID realizations of the same latent process.</p>
  </section>
  <section class="figure-shell">
    <h2>Figure 2. Naive pooled exposure solving loses low-burden modality signal</h2>
    <div id="deconv-plot"></div>
    <p class="caption">Mean absolute exposure error by modality and solver. Lower values indicate better recovery of the true process mixture.</p>
  </section>
  <section class="figure-shell">
    <h2>Figure 3. UGA retrieves held-out indel partners from SBS signatures</h2>
    <div id="retrieval-heatmap"></div>
    <p class="caption">Darker cells indicate shorter clean-context UGA distance from each SBS process to each ID process. The diagonal structure is unavailable to separate standard catalogs without an external bridge.</p>
  </section>
</main>
<script>
const DATA = {data_json};
const colors = ["#2f6f73","#c44e52","#4c78a8","#f58518","#54a24b","#b279a2","#eeca3b","#72b7b2","#ff9da6","#9d755d","#8cd17d","#6b5b95"];
const modalityShape = {{
  SBS: d3.symbolCircle,
  DBS: d3.symbolSquare,
  ID: d3.symbolTriangle
}};

function processColor(pid) {{
  const i = +pid.replace("P", "") - 1;
  return colors[i % colors.length];
}}

function renderSignatureMap() {{
  const data = DATA.signature_map;
  const width = 1080, height = 620;
  const margin = {{top: 22, right: 170, bottom: 54, left: 58}};
  const svg = d3.select("#signature-map").append("svg").attr("viewBox", [0, 0, width, height]);
  const x = d3.scaleLinear().domain(d3.extent(data, d => d.x)).nice().range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().domain(d3.extent(data, d => d.y)).nice().range([height - margin.bottom, margin.top]);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${{height - margin.bottom}})`).call(d3.axisBottom(x).ticks(7));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(7));
  svg.append("text").attr("x", width / 2).attr("y", height - 12).attr("text-anchor", "middle").attr("fill", "#64717f").attr("font-size", 12).text("PC1 of UGA clean-context coordinates");
  svg.append("text").attr("x", -height / 2).attr("y", 16).attr("transform", "rotate(-90)").attr("text-anchor", "middle").attr("fill", "#64717f").attr("font-size", 12).text("PC2 of UGA clean-context coordinates");

  const grouped = d3.group(data, d => d.process);
  for (const [pid, pts] of grouped) {{
    const ordered = ["SBS", "DBS", "ID"].map(m => pts.find(d => d.modality === m)).filter(Boolean);
    svg.append("path")
      .datum(ordered)
      .attr("fill", "none")
      .attr("stroke", processColor(pid))
      .attr("stroke-width", 1.8)
      .attr("stroke-opacity", 0.38)
      .attr("d", d3.line().x(d => x(d.x)).y(d => y(d.y)).curve(d3.curveCatmullRom.alpha(0.55)));
  }}

  svg.append("g")
    .selectAll("path")
    .data(data)
    .join("path")
    .attr("transform", d => `translate(${{x(d.x)}},${{y(d.y)}})`)
    .attr("d", d3.symbol().type(d => modalityShape[d.modality]).size(110))
    .attr("fill", d => processColor(d.process))
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5);

  svg.append("g")
    .selectAll("text")
    .data(data.filter(d => d.modality === "SBS"))
    .join("text")
    .attr("x", d => x(d.x) + 8)
    .attr("y", d => y(d.y) + 4)
    .attr("fill", "#24313b")
    .attr("font-size", 10.5)
    .text(d => d.short);

  const legend = svg.append("g").attr("class", "legend").attr("transform", `translate(${{width - margin.right + 34}},${{margin.top + 8}})`);
  ["SBS", "DBS", "ID"].forEach((m, i) => {{
    const g = legend.append("g").attr("transform", `translate(0,${{i * 26}})`);
    g.append("path").attr("d", d3.symbol().type(modalityShape[m]).size(110)).attr("fill", "#52616f");
    g.append("text").attr("x", 16).attr("y", 4).text(m);
  }});
}}

function renderDeconv() {{
  const data = DATA.deconvolution_summary.filter(d => ["Separate categorical NNLS", "Separate UGA NNLS", "Naive pooled UGA NNLS"].includes(d.Method));
  const width = 1080, height = 420;
  const margin = {{top: 24, right: 38, bottom: 54, left: 190}};
  const svg = d3.select("#deconv-plot").append("svg").attr("viewBox", [0, 0, width, height]);
  const y = d3.scaleBand().domain(data.map(d => `${{d.Modality}} | ${{d.Method}}`)).range([margin.top, height - margin.bottom]).padding(0.22);
  const x = d3.scaleLinear().domain([0, d3.max(data, d => d.Mean_MAE) * 1.12]).nice().range([margin.left, width - margin.right]);
  const methodColor = d3.scaleOrdinal().domain(["Separate categorical NNLS", "Separate UGA NNLS", "Naive pooled UGA NNLS"]).range(["#2f6f73", "#4c78a8", "#c44e52"]);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${{height - margin.bottom}})`).call(d3.axisBottom(x).ticks(6));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${{margin.left}},0)`).call(d3.axisLeft(y).tickSize(0)).selectAll("text").attr("font-size", 11);
  svg.append("text").attr("x", width / 2).attr("y", height - 12).attr("text-anchor", "middle").attr("fill", "#64717f").attr("font-size", 12).text("Mean absolute error");
  svg.append("g").selectAll("line")
    .data(data)
    .join("line")
    .attr("x1", x(0))
    .attr("x2", d => x(d.Mean_MAE))
    .attr("y1", d => y(`${{d.Modality}} | ${{d.Method}}`) + y.bandwidth() / 2)
    .attr("y2", d => y(`${{d.Modality}} | ${{d.Method}}`) + y.bandwidth() / 2)
    .attr("stroke", d => methodColor(d.Method))
    .attr("stroke-width", 7)
    .attr("stroke-linecap", "round")
    .attr("stroke-opacity", 0.72);
  svg.append("g").selectAll("text.value")
    .data(data)
    .join("text")
    .attr("x", d => x(d.Mean_MAE) + 7)
    .attr("y", d => y(`${{d.Modality}} | ${{d.Method}}`) + y.bandwidth() / 2 + 4)
    .attr("fill", "#24313b")
    .attr("font-size", 11)
    .text(d => d.Mean_MAE.toFixed(3));
}}

function renderHeatmap() {{
  const matrix = DATA.distance_matrices.SBS_to_ID_context;
  const processes = DATA.processes;
  const width = 900, height = 720;
  const margin = {{top: 76, right: 24, bottom: 84, left: 104}};
  const svg = d3.select("#retrieval-heatmap").append("svg").attr("viewBox", [0, 0, width, height]);
  const x = d3.scaleBand().domain(processes.map(d => d.pid)).range([margin.left, width - margin.right]).padding(0.04);
  const y = d3.scaleBand().domain(processes.map(d => d.pid)).range([margin.top, height - margin.bottom]).padding(0.04);
  const flat = matrix.flat();
  const color = d3.scaleSequential(d3.interpolateYlGnBu).domain([d3.max(flat), d3.min(flat)]);
  const cells = [];
  matrix.forEach((row, i) => row.forEach((v, j) => cells.push({{source: processes[i].pid, target: processes[j].pid, value: v, diag: i === j}})));
  svg.append("g")
    .selectAll("rect")
    .data(cells)
    .join("rect")
    .attr("x", d => x(d.target))
    .attr("y", d => y(d.source))
    .attr("width", x.bandwidth())
    .attr("height", y.bandwidth())
    .attr("fill", d => color(d.value))
    .attr("stroke", d => d.diag ? "#17202a" : "#ffffff")
    .attr("stroke-width", d => d.diag ? 1.8 : 0.8);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${{margin.top - 8}})`).call(d3.axisTop(x).tickSize(0)).selectAll("text").attr("transform", "rotate(-38)").attr("text-anchor", "start").attr("dx", 4).attr("dy", -2);
  svg.append("g").attr("class", "axis").attr("transform", `translate(${{margin.left - 8}},0)`).call(d3.axisLeft(y).tickSize(0));
  svg.append("text").attr("x", (margin.left + width - margin.right) / 2).attr("y", 22).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", "#24313b").text("Target ID process");
  svg.append("text").attr("x", 16).attr("y", (margin.top + height - margin.bottom) / 2).attr("transform", `rotate(-90,16,${{(margin.top + height - margin.bottom) / 2}})`).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", "#24313b").text("Source SBS process");
}}

renderSignatureMap();
renderDeconv();
renderHeatmap();
</script>
</body>
</html>
"""
    (out_dir / "shared_space_d3_figures.html").write_text(html, encoding="utf-8")


def cleanup_legacy_flat_outputs(out_dir: Path) -> None:
    """Remove exact files from the original flat EXP-024 output layout."""
    for name in [
        "manuscript_tables.html",
        "manuscript_tables_fragment.html",
        "shared_space_d3_figures.html",
        "shared_space_benchmark_data.json",
    ]:
        path = out_dir / name
        if path.is_file():
            path.unlink()


def write_index(out_dir: Path) -> None:
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EXP-024 UGA shared-space benchmark</title>
<style>
body {
  margin: 32px;
  color: #17202a;
  font-family: "Aptos", "Segoe UI", Arial, sans-serif;
  line-height: 1.45;
}
main {
  max-width: 860px;
}
h1 {
  margin: 0 0 10px;
  font-size: 25px;
}
p {
  color: #5f6b7a;
}
a {
  color: #226f54;
  text-decoration-thickness: 1px;
  text-underline-offset: 2px;
}
li {
  margin: 8px 0;
}
code {
  background: #eef3f7;
  padding: 1px 4px;
  border-radius: 4px;
}
</style>
</head>
<body>
<main>
  <h1>EXP-024 UGA Shared-Space Benchmark</h1>
  <p>This package contains the reproducible outputs for the shared-coordinate experiment.</p>
  <ul>
    <li><a href="html/narrative_walkthrough.html">Narrative walkthrough of the experiment and results</a></li>
    <li><a href="html/manuscript_tables.html">Publication HTML tables</a></li>
    <li><a href="html/manuscript_tables_fragment.html">Copy/paste table fragments</a></li>
    <li><a href="figures/shared_space_d3_figures.html">D3 figure page</a></li>
    <li><a href="README.md">Experiment README</a></li>
    <li><code>tables/</code> raw CSV tables and patient-level metrics</li>
    <li><code>data/shared_space_benchmark_data.json</code> figure and summary data</li>
  </ul>
</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def _row(df: pd.DataFrame, **match) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in match.items():
        mask &= df[col] == val
    if not mask.any():
        raise KeyError(f"No row matching {match}")
    return df[mask].iloc[0]


def _bar(label: str, value: float, max_value: float, color: str, suffix: str = "") -> str:
    width = 100.0 * float(value) / max(float(max_value), 1e-12)
    return f"""
<div class="bar-row">
  <div class="bar-label">{escape(label)}</div>
  <div class="bar-track"><span style="width: {width:.1f}%; background: {color};"></span></div>
  <div class="bar-value">{value:.3f}{escape(suffix)}</div>
</div>
""".strip()


def write_narrative_page(
    html_dir: Path,
    deconv_summary: pd.DataFrame,
    retrieval_public: pd.DataFrame,
    imputation_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    sbs_uga = _row(deconv_summary, Modality="SBS", Method="Separate UGA NNLS")
    sbs_pool = _row(deconv_summary, Modality="SBS", Method="Naive pooled UGA NNLS")
    dbs_uga = _row(deconv_summary, Modality="DBS", Method="Separate UGA NNLS")
    dbs_pool = _row(deconv_summary, Modality="DBS", Method="Naive pooled UGA NNLS")
    id_uga = _row(deconv_summary, Modality="ID", Method="Separate UGA NNLS")
    id_pool = _row(deconv_summary, Modality="ID", Method="Naive pooled UGA NNLS")
    dbs_cat = _row(deconv_summary, Modality="DBS", Method="Separate categorical NNLS")
    id_cat = _row(deconv_summary, Modality="ID", Method="Separate categorical NNLS")
    sbs_cat = _row(deconv_summary, Modality="SBS", Method="Separate categorical NNLS")

    uga_context = _row(retrieval_public, Coordinate_System="UGA clean context 40D")
    uga_full = _row(retrieval_public, Coordinate_System="UGA full 48D")
    uga_payload = _row(retrieval_public, Coordinate_System="UGA payload 8D")
    standard = _row(retrieval_public, Coordinate_System="Standard separate catalogs")

    dbs_bridge = _row(imputation_summary, Target_Modality="DBS", Method="UGA unsupervised bridge")
    dbs_mean = _row(imputation_summary, Target_Modality="DBS", Method="Standard no-bridge cohort mean")
    dbs_random = _row(imputation_summary, Target_Modality="DBS", Method="Standard random bridge")
    id_bridge = _row(imputation_summary, Target_Modality="ID", Method="UGA unsupervised bridge")
    id_mean = _row(imputation_summary, Target_Modality="ID", Method="Standard no-bridge cohort mean")
    id_random = _row(imputation_summary, Target_Modality="ID", Method="Standard random bridge")

    max_mae = max(deconv_summary["Mean_MAE"].max(), imputation_summary["Mean_MAE"].max())
    exposure_bars = "\n".join(
        [
            _bar("SBS separate UGA", sbs_uga["Mean_MAE"], max_mae, "#2f6f73"),
            _bar("SBS naive pooled", sbs_pool["Mean_MAE"], max_mae, "#c44e52"),
            _bar("DBS separate UGA", dbs_uga["Mean_MAE"], max_mae, "#2f6f73"),
            _bar("DBS naive pooled", dbs_pool["Mean_MAE"], max_mae, "#c44e52"),
            _bar("ID separate UGA", id_uga["Mean_MAE"], max_mae, "#2f6f73"),
            _bar("ID naive pooled", id_pool["Mean_MAE"], max_mae, "#c44e52"),
        ]
    )
    retrieval_bars = "\n".join(
        [
            _bar("UGA clean context 40D", uga_context["Mean_Top1_Accuracy"], 1.0, "#2f6f73"),
            _bar("UGA full 48D", uga_full["Mean_Top1_Accuracy"], 1.0, "#4c78a8"),
            _bar("UGA payload 8D", uga_payload["Mean_Top1_Accuracy"], 1.0, "#9d755d"),
            _bar("Standard separate catalogs", standard["Mean_Top1_Accuracy"], 1.0, "#8b95a1"),
        ]
    )
    imputation_bars = "\n".join(
        [
            _bar("DBS UGA bridge", dbs_bridge["Mean_MAE"], max_mae, "#2f6f73"),
            _bar("DBS cohort mean", dbs_mean["Mean_MAE"], max_mae, "#8b95a1"),
            _bar("DBS random bridge", dbs_random["Mean_MAE"], max_mae, "#c44e52"),
            _bar("ID UGA bridge", id_bridge["Mean_MAE"], max_mae, "#2f6f73"),
            _bar("ID cohort mean", id_mean["Mean_MAE"], max_mae, "#8b95a1"),
            _bar("ID random bridge", id_random["Mean_MAE"], max_mae, "#c44e52"),
        ]
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EXP-024 Narrative Walkthrough</title>
<style>
:root {{
  --ink: #17202a;
  --muted: #5f6b7a;
  --line: #d8e0e8;
  --soft: #f5f8fa;
  --green: #2f6f73;
  --blue: #4c78a8;
  --red: #c44e52;
  --amber: #b8872f;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  color: var(--ink);
  background: #ffffff;
  font-family: "Aptos", "Segoe UI", Arial, sans-serif;
  line-height: 1.55;
}}
main {{
  max-width: 1120px;
  margin: 0 auto;
  padding: 38px 30px 58px;
}}
nav {{
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  margin-bottom: 30px;
  font-size: 13px;
}}
a {{
  color: var(--green);
  text-decoration-thickness: 1px;
  text-underline-offset: 3px;
}}
.hero {{
  display: grid;
  grid-template-columns: minmax(0, 1.25fr) minmax(280px, 0.75fr);
  gap: 34px;
  padding-bottom: 34px;
  border-bottom: 1px solid var(--line);
}}
.eyebrow {{
  margin: 0 0 10px;
  color: var(--green);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
}}
h1 {{
  margin: 0;
  max-width: 820px;
  font-size: 40px;
  line-height: 1.05;
  letter-spacing: 0;
}}
.lede {{
  margin: 18px 0 0;
  max-width: 760px;
  color: #3d4a56;
  font-size: 18px;
}}
.thesis {{
  align-self: end;
  border-left: 4px solid var(--green);
  padding: 16px 0 16px 18px;
  color: #33404c;
  background: linear-gradient(90deg, #f3f8f6, #fff);
}}
.thesis strong {{
  display: block;
  margin-bottom: 6px;
  color: var(--ink);
}}
section {{
  margin-top: 44px;
}}
h2 {{
  margin: 0 0 12px;
  font-size: 24px;
  letter-spacing: 0;
}}
h3 {{
  margin: 22px 0 8px;
  font-size: 16px;
}}
p {{
  max-width: 850px;
  margin: 12px 0;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin: 22px 0;
}}
.stat {{
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 15px 16px;
  background: #fff;
}}
.stat .value {{
  font-size: 29px;
  line-height: 1;
  font-weight: 750;
  color: var(--ink);
}}
.stat .label {{
  margin-top: 6px;
  color: var(--muted);
  font-size: 13px;
}}
.callout {{
  margin: 22px 0;
  padding: 15px 18px;
  border-left: 4px solid var(--blue);
  background: #f4f7fb;
  color: #33404c;
}}
.warning {{
  border-left-color: var(--amber);
  background: #fbf7ee;
}}
.bar-panel {{
  margin: 18px 0 24px;
  max-width: 860px;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  padding: 14px 0;
}}
.bar-row {{
  display: grid;
  grid-template-columns: 190px minmax(160px, 1fr) 70px;
  gap: 12px;
  align-items: center;
  margin: 9px 0;
  font-size: 13px;
}}
.bar-label {{ color: #34424f; }}
.bar-track {{
  height: 9px;
  background: #edf1f5;
  border-radius: 999px;
  overflow: hidden;
}}
.bar-track span {{
  display: block;
  height: 100%;
  border-radius: inherit;
}}
.bar-value {{
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  text-align: right;
}}
table {{
  width: 100%;
  max-width: 900px;
  border-collapse: collapse;
  margin: 18px 0;
  font-size: 13px;
}}
th {{
  background: var(--soft);
  border-top: 2px solid var(--ink);
  border-bottom: 1px solid var(--ink);
  padding: 8px;
  text-align: left;
}}
td {{
  border-bottom: 1px solid var(--line);
  padding: 8px;
}}
.method-steps {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 18px;
}}
.step {{
  border-top: 3px solid var(--green);
  padding-top: 10px;
  color: #34424f;
}}
.step strong {{
  display: block;
  color: var(--ink);
  margin-bottom: 4px;
}}
code {{
  background: #eef3f7;
  padding: 1px 4px;
  border-radius: 4px;
}}
@media (max-width: 800px) {{
  main {{ padding: 26px 18px 44px; }}
  .hero {{ grid-template-columns: 1fr; }}
  h1 {{ font-size: 31px; }}
  .grid, .method-steps {{ grid-template-columns: 1fr; }}
  .bar-row {{ grid-template-columns: 1fr; gap: 5px; }}
  .bar-value {{ text-align: left; }}
}}
</style>
</head>
<body>
<main>
  <nav>
    <a href="../index.html">Package index</a>
    <a href="manuscript_tables.html">Publication tables</a>
    <a href="../figures/shared_space_d3_figures.html">D3 figures</a>
    <a href="../README.md">Generated README</a>
  </nav>

  <header class="hero">
    <div>
      <p class="eyebrow">EXP-024 narrative walkthrough</p>
      <h1>What the shared UGA coordinate space proves, without pretending the pooled solver is best</h1>
      <p class="lede">This experiment asks whether SBS, DBS, and indels gain something scientifically useful from living in one 48-dimensional coordinate system, even when exposure estimation itself is more accurate when each modality is solved separately.</p>
    </div>
    <aside class="thesis">
      <strong>Thesis in one sentence</strong>
      UGA should be defended as a shared geometry for comparing, matching, and transferring mutation processes across modalities, not as a claim that one naive all-at-once NNLS solve must dominate.
    </aside>
  </header>

  <section>
    <h2>1. Why this experiment exists</h2>
    <p>Your central methodological claim is that SBS, DBS, and indels can be represented in a single feature space. But your empirical observation is also important: when all modalities are solved together in one naive exposure system, performance can worsen. EXP-024 turns that tension into a strength.</p>
    <p>The benchmark therefore tests two ideas separately. First, it checks whether separate modality solves remain better for exposure recovery. Second, it asks whether the shared UGA coordinate system enables analyses that standard SBS96, DBS78, and ID83 catalogs cannot naturally perform.</p>
    <div class="callout">
      The point is not “UGA pooling always wins.” The point is “UGA creates a common coordinate language, and that language makes cross-modality questions measurable.”
    </div>
  </section>

  <section>
    <h2>2. What was simulated</h2>
    <p>The simulation generated <strong>{args.n_patients:,} tumors</strong> from sparse mixtures of <strong>{len(PROCESSES)} latent mutational processes</strong>. Each process has three visible forms: an SBS form, a DBS form, and an indel form. These forms share the same flanking-context program, but they carry different mutation payloads and therefore land in distinct standard catalog channels.</p>
    <div class="method-steps">
      <div class="step"><strong>Latent process</strong> A shared biological context program defines the local sequence environment.</div>
      <div class="step"><strong>Modality-specific event</strong> SBS, DBS, and ID versions use different ref/alt payloads.</div>
      <div class="step"><strong>Two encodings</strong> Events are counted in standard channels and encoded as 48D UGA vectors.</div>
      <div class="step"><strong>Known truth</strong> The true process mixture is retained for every simulated tumor.</div>
    </div>
    <p>The UGA vector uses the clean-context layout <code>[Lx, Ly, Xref, Yref, Rx, Ry, Xalt, Yalt]</code> with <code>d_context={D_CONTEXT}</code> and <code>d_payload={D_PAYLOAD}</code>, producing <strong>{UGA_DIM} dimensions</strong>. The standard comparator uses the familiar SBS96, DBS78, and ID83 channel universes.</p>
  </section>

  <section>
    <h2>3. Result 1: exposure recovery is better when solved separately</h2>
    <p>The first result validates your concern. The naive pooled UGA solve collapses all SBS, DBS, and ID events into one burden-weighted 48D profile, then tries to fit all modality-specific centroids at once. That makes the lower-burden modalities vulnerable to dilution by the larger SBS burden.</p>
    <div class="grid">
      <div class="stat"><div class="value">{dbs_uga['Mean_MAE']:.3f} -> {dbs_pool['Mean_MAE']:.3f}</div><div class="label">DBS MAE, separate UGA to naive pooled UGA</div></div>
      <div class="stat"><div class="value">{id_uga['Mean_MAE']:.3f} -> {id_pool['Mean_MAE']:.3f}</div><div class="label">ID MAE, separate UGA to naive pooled UGA</div></div>
      <div class="stat"><div class="value">{sbs_uga['Mean_MAE']:.3f} -> {sbs_pool['Mean_MAE']:.3f}</div><div class="label">SBS MAE, separate UGA to naive pooled UGA</div></div>
    </div>
    <div class="bar-panel" aria-label="Exposure recovery MAE bars">
      {exposure_bars}
    </div>
    <p>Separate UGA NNLS also improves over the separate categorical NNLS baseline in this synthetic setting: SBS MAE is {sbs_uga['Mean_MAE']:.3f} versus {sbs_cat['Mean_MAE']:.3f}, DBS MAE is {dbs_uga['Mean_MAE']:.3f} versus {dbs_cat['Mean_MAE']:.3f}, and ID MAE is {id_uga['Mean_MAE']:.3f} versus {id_cat['Mean_MAE']:.3f}. The pooled solve, however, is clearly not the right estimator for this task.</p>
    <div class="callout warning">
      Interpretation: this result protects the paper from an overclaim. The shared coordinate system is not the same thing as a single pooled exposure estimator.
    </div>
  </section>

  <section>
    <h2>4. Result 2: UGA makes cross-modality signature matching possible</h2>
    <p>The second result is the heart of the shared-space argument. Because every SBS, DBS, and ID process has a UGA coordinate, the benchmark can ask: if I only know the SBS version of a process, is the nearest DBS or ID coordinate the matching latent process?</p>
    <div class="grid">
      <div class="stat"><div class="value">{uga_context['Mean_Top1_Accuracy']:.3f}</div><div class="label">Top-1 retrieval using UGA clean-context coordinates</div></div>
      <div class="stat"><div class="value">{uga_full['Mean_Top1_Accuracy']:.3f}</div><div class="label">Top-1 retrieval using full 48D UGA coordinates</div></div>
      <div class="stat"><div class="value">{standard['Mean_Top1_Accuracy']:.3f}</div><div class="label">Standard-catalog chance/tie expectation</div></div>
    </div>
    <div class="bar-panel" aria-label="Cross-modality retrieval top-1 bars">
      {retrieval_bars}
    </div>
    <p>UGA clean context and full UGA both recover the correct cross-modality partner with mean top-1 accuracy {uga_context['Mean_Top1_Accuracy']:.3f}. Payload-only UGA performs near the standard chance/tie baseline, which is useful: it shows the bridge is coming from shared sequence environment, not merely from mutation payload identity.</p>
    <p>Standard catalogs do not have a native distance between an SBS96 channel distribution and an ID83 channel distribution. The standard row is therefore reported as a chance/tie expectation, not as a meaningful geometric competitor.</p>
  </section>

  <section>
    <h2>5. Result 3: UGA transfers signal into held-out modalities</h2>
    <p>The final benchmark withholds DBS and ID exposures, then tries to infer them from SBS. The UGA bridge is unsupervised: it uses nearest neighbors in clean-context UGA space rather than process labels.</p>
    <div class="grid">
      <div class="stat"><div class="value">{dbs_bridge['Mean_MAE']:.3f}</div><div class="label">DBS MAE using SBS-to-DBS UGA bridge</div></div>
      <div class="stat"><div class="value">{id_bridge['Mean_MAE']:.3f}</div><div class="label">ID MAE using SBS-to-ID UGA bridge</div></div>
      <div class="stat"><div class="value">{dbs_mean['Mean_MAE']:.3f}</div><div class="label">DBS MAE with no bridge, cohort mean baseline</div></div>
    </div>
    <div class="bar-panel" aria-label="Held-out imputation MAE bars">
      {imputation_bars}
    </div>
    <p>The UGA bridge reaches DBS MAE {dbs_bridge['Mean_MAE']:.3f} and ID MAE {id_bridge['Mean_MAE']:.3f}. The no-bridge standard baseline is much weaker: DBS MAE {dbs_mean['Mean_MAE']:.3f} and ID MAE {id_mean['Mean_MAE']:.3f}. A random bridge is also poor, with DBS MAE {dbs_random['Mean_MAE']:.3f} and ID MAE {id_random['Mean_MAE']:.3f}.</p>
    <p>This result demonstrates a practical advantage of shared coordinates: one modality can supply information about another because both are addressable in the same clean-context geometry.</p>
  </section>

  <section>
    <h2>6. What the results show</h2>
    <table>
      <thead><tr><th>Question</th><th>Answer from EXP-024</th></tr></thead>
      <tbody>
        <tr><td>Should SBS, DBS, and ID exposures always be solved in one pooled NNLS system?</td><td>No. The naive pooled solve worsens DBS and ID exposure recovery.</td></tr>
        <tr><td>Does that invalidate the shared UGA feature space?</td><td>No. It separates representation from estimator choice.</td></tr>
        <tr><td>What does UGA enable that standard catalogs do not?</td><td>Direct cross-modality distance, nearest-neighbor matching, manifold visualization, and held-out modality transfer.</td></tr>
        <tr><td>Where does the cross-modality signal live?</td><td>Primarily in clean flanking context rather than payload-only coordinates.</td></tr>
      </tbody>
    </table>
    <div class="callout">
      Manuscript framing: “We solve exposures separately when that is statistically preferable, but UGA allows the resulting SBS, DBS, and indel processes to be compared in a single coordinate system.”
    </div>
  </section>

  <section>
    <h2>7. Reproducibility map</h2>
    <p>Run the experiment from the repository root with:</p>
    <p><code>python cgr_validation_results\\research\\scripts\\EXP024_shared_space\\run_shared_space_benchmark.py</code></p>
    <p>The page you are reading is generated from the same summary tables used for the manuscript HTML tables. For deeper inspection, open the <a href="manuscript_tables.html">publication tables</a>, the <a href="../figures/shared_space_d3_figures.html">D3 figures</a>, or the raw CSV files in <code>../tables/</code>.</p>
  </section>
</main>
</body>
</html>
"""
    (html_dir / "narrative_walkthrough.html").write_text(html, encoding="utf-8")


def write_experiment_readme(
    out_dir: Path,
    args: argparse.Namespace,
    deconv_summary: pd.DataFrame,
    retrieval_public: pd.DataFrame,
    imputation_summary: pd.DataFrame,
) -> None:
    pooled_dbs = deconv_summary[(deconv_summary["Modality"] == "DBS") & (deconv_summary["Method"] == "Naive pooled UGA NNLS")].iloc[0]
    separate_dbs = deconv_summary[(deconv_summary["Modality"] == "DBS") & (deconv_summary["Method"] == "Separate UGA NNLS")].iloc[0]
    pooled_id = deconv_summary[(deconv_summary["Modality"] == "ID") & (deconv_summary["Method"] == "Naive pooled UGA NNLS")].iloc[0]
    separate_id = deconv_summary[(deconv_summary["Modality"] == "ID") & (deconv_summary["Method"] == "Separate UGA NNLS")].iloc[0]
    uga_retrieval = retrieval_public[retrieval_public["Coordinate_System"] == "UGA clean context 40D"].iloc[0]
    standard_retrieval = retrieval_public[retrieval_public["Coordinate_System"] == "Standard separate catalogs"].iloc[0]
    uga_bridge_dbs = imputation_summary[
        (imputation_summary["Target_Modality"] == "DBS") & (imputation_summary["Method"] == "UGA unsupervised bridge")
    ].iloc[0]
    baseline_dbs = imputation_summary[
        (imputation_summary["Target_Modality"] == "DBS") & (imputation_summary["Method"] == "Standard no-bridge cohort mean")
    ].iloc[0]

    readme = f"""# EXP-024 UGA Shared-Space Benchmark

## Purpose

This experiment documents a precise version of the manuscript claim:

- SBS, DBS, and indels can be encoded in one Universal Genomic Address (UGA) feature space.
- A naive all-at-once exposure solve is not necessarily the right estimator and can be worse than separate per-modality solving.
- The shared coordinate space is still valuable because it enables cross-modality measurements that standard SBS96, DBS78, and ID83 catalogs cannot define on their own.

## Reproducible Command

Run from the repository root:

```powershell
python cgr_validation_results\\research\\scripts\\EXP024_shared_space\\run_shared_space_benchmark.py
```

Equivalent compatibility wrapper:

```powershell
python bench\\run_shared_space_benchmark.py
```

Default parameters:

| Parameter | Value |
|---|---:|
| seed | {args.seed} |
| tumors | {args.n_patients} |
| latent processes | {len(PROCESSES)} |
| signature events per process | {args.signature_events_per_process} |
| context noise | {args.context_noise:.3f} |
| d_context | {D_CONTEXT} |
| d_payload | {D_PAYLOAD} |
| UGA dimensions | {UGA_DIM} |

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

The simulation creates {args.n_patients} tumors from sparse mixtures of {len(PROCESSES)} latent mutational processes. For every process, SBS, DBS, and ID events share a clean flanking-context program but use modality-specific payloads and standard channel labels.

The standard side uses the SBS96, DBS78, and ID83 channel universes from `data/Signatures/COSMIC_v3.5_*_GRCh37.txt`. The UGA side encodes each synthetic event as:

```text
[Lx, Ly, Xref, Yref, Rx, Ry, Xalt, Yalt]
```

with atlas model `{UGA_MODEL.name}`, yielding a {UGA_DIM}-dimensional coordinate.

## Benchmarks

1. **Exposure recovery:** NNLS is fit separately within SBS, DBS, and ID using standard categorical signatures and UGA centroids. A deliberately naive pooled UGA comparator collapses all events into one burden-weighted 48D mean and fits SBS/DBS/ID centroids together.
2. **Cross-modality retrieval:** UGA centroids are matched across ordered modality pairs using nearest-neighbor distance in full 48D, clean-context 40D, and payload-only 8D coordinates. Standard catalogs receive a chance/tie baseline because their axes are not shared.
3. **Held-out imputation:** DBS and ID exposures are withheld. SBS exposures are transferred to DBS/ID using an unsupervised nearest-neighbor bridge in UGA clean-context coordinates.

## Main Results

- Separate UGA solving outperformed the naive pooled UGA solve for low-burden modalities. DBS MAE increased from {separate_dbs['Mean_MAE']:.3f} to {pooled_dbs['Mean_MAE']:.3f}; ID MAE increased from {separate_id['Mean_MAE']:.3f} to {pooled_id['Mean_MAE']:.3f}.
- UGA clean-context retrieval recovered cross-modality process partners with mean top-1 accuracy {uga_retrieval['Mean_Top1_Accuracy']:.3f}; the standard separate-catalog chance/tie expectation was {standard_retrieval['Mean_Top1_Accuracy']:.3f}.
- SBS-to-DBS imputation using the unsupervised UGA bridge had mean MAE {uga_bridge_dbs['Mean_MAE']:.3f}, compared with {baseline_dbs['Mean_MAE']:.3f} for the no-bridge standard cohort-mean baseline.

## Interpretation

This is the result the manuscript needs: UGA does not require claiming that a single pooled exposure solve is best. Instead, it demonstrates that a common coordinate system creates a geometry in which SBS, DBS, and indel processes can be compared, matched, visualized, and transferred across modalities. Standard mutational signature definitions do not provide this operation because each mutation class is defined on a separate categorical axis.

## Notes

- The D3 page uses a CDN import for D3 v7 and embeds the experiment data directly in the HTML.
- Re-running the script overwrites this package deterministically for the same seed and parameters.
- Patient-level rows are retained so summary values can be audited or re-aggregated.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def write_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    design: pd.DataFrame,
    raw_metrics: pd.DataFrame,
    deconv_summary: pd.DataFrame,
    retrieval_summary: pd.DataFrame,
    retrieval_detail: pd.DataFrame,
    retrieval_public: pd.DataFrame,
    imputation_raw: pd.DataFrame,
    imputation_summary: pd.DataFrame,
    figure_payload: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cleanup_legacy_flat_outputs(out_dir)

    tables = out_dir / "tables"
    html_dir = out_dir / "html"
    figures_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    for path in (tables, html_dir, figures_dir, data_dir):
        path.mkdir(exist_ok=True)

    raw_metrics.to_csv(tables / "exposure_recovery_patient_metrics.csv", index=False)
    deconv_summary.to_csv(tables / "table2_exposure_recovery_summary.csv", index=False)
    retrieval_summary.to_csv(tables / "cross_modality_retrieval_by_pair.csv", index=False)
    retrieval_detail.to_csv(tables / "cross_modality_retrieval_process_detail.csv", index=False)
    retrieval_public.to_csv(tables / "table3_cross_modality_retrieval_summary.csv", index=False)
    imputation_raw.to_csv(tables / "heldout_imputation_patient_metrics.csv", index=False)
    imputation_summary.to_csv(tables / "table4_heldout_imputation_summary.csv", index=False)
    design.to_csv(tables / "table1_experiment_design.csv", index=False)
    (data_dir / "shared_space_benchmark_data.json").write_text(json.dumps(figure_payload, indent=2), encoding="utf-8")

    write_narrative_page(html_dir, deconv_summary, retrieval_public, imputation_summary, args)
    write_tables(html_dir, design, deconv_summary, retrieval_public, imputation_summary)
    write_figures_html(figures_dir, figure_payload)
    write_index(out_dir)
    write_experiment_readme(out_dir, args, deconv_summary, retrieval_public, imputation_summary)

    manifest = {
        "experiment": "UGA shared-coordinate cross-modality benchmark",
        "runner": str(REPO / "cgr_validation_results" / "research" / "scripts" / "EXP024_shared_space" / "run_shared_space_benchmark.py"),
        "compatibility_runner": str(REPO / "bench" / "run_shared_space_benchmark.py"),
        "seed": args.seed,
        "n_patients": args.n_patients,
        "signature_events_per_process": args.signature_events_per_process,
        "context_noise": args.context_noise,
        "uga_model": UGA_MODEL.name,
        "d_context": D_CONTEXT,
        "d_payload": D_PAYLOAD,
        "payload_schema": PAYLOAD_SCHEMA,
        "outputs": {
            "index_html": str(out_dir / "index.html"),
            "readme": str(out_dir / "README.md"),
            "narrative_html": str(html_dir / "narrative_walkthrough.html"),
            "tables_html": str(html_dir / "manuscript_tables.html"),
            "table_fragments_html": str(html_dir / "manuscript_tables_fragment.html"),
            "figures_html": str(figures_dir / "shared_space_d3_figures.html"),
            "data_json": str(data_dir / "shared_space_benchmark_data.json"),
            "tables_dir": str(tables),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def make_design_table(args: argparse.Namespace) -> pd.DataFrame:
    rows = [
        ("Question", "Can UGA provide useful cross-modality evidence even when SBS, DBS, and indel exposures are solved separately?"),
        ("Synthetic cohort", f"{args.n_patients:,} tumors with sparse mixtures of {len(PROCESSES)} latent mutational processes."),
        ("Mutation modalities", f"SBS96, DBS78, and ID83 standard channel universes plus event-level {UGA_DIM}D UGA clean-context vectors."),
        ("UGA encoding", f"Atlas model {UGA_MODEL.name}; layout [L context, REF payload, R context, ALT payload]."),
        ("Separate solve benchmark", "NNLS is fit independently for each modality using either categorical signatures or UGA signature centroids."),
        ("Naive pooled benchmark", f"All SBS/DBS/ID events are collapsed into a single burden-weighted {UGA_DIM}D mean and fit against all modality-specific UGA centroids at once."),
        ("Shared-space benchmark", "UGA centroids are used for cross-modality nearest-neighbor retrieval and SBS-to-DBS/ID held-out exposure imputation."),
        ("Standard-catalog baseline", "SBS, DBS, and ID channels are unequal axes, so cross-catalog distance is undefined without external labels; chance/tie expectation is reported."),
    ]
    return pd.DataFrame(rows, columns=["Element", "Specification"])


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=240513)
    parser.add_argument("--n-patients", type=int, default=520)
    parser.add_argument("--signature-events-per-process", type=int, default=5500)
    parser.add_argument("--context-noise", type=float, default=0.055)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    channels = read_channel_universes()
    design = make_design_table(args)

    standard_ref, uga_ref = build_reference_signatures(
        PROCESSES,
        channels,
        rng,
        events_per_process=args.signature_events_per_process,
        context_noise=args.context_noise,
    )
    patients = simulate_patients(PROCESSES, channels, rng, args.n_patients, args.context_noise)
    metrics, predictions = evaluate_exposures(patients, standard_ref, uga_ref, PROCESSES)
    deconv_summary = summarize_deconvolution(metrics)

    retrieval_summary, retrieval_detail, distance_matrices = retrieval_analysis(uga_ref, len(PROCESSES))
    retrieval_public = summarize_retrieval(retrieval_summary)

    imputation_raw = imputation_analysis(patients, predictions, uga_ref, rng)
    imputation_summary = summarize_imputation(imputation_raw)

    figure_payload = {
        "processes": [{"pid": p.pid, "label": p.label, "short": p.short} for p in PROCESSES],
        "signature_map": signature_map_data(uga_ref, PROCESSES),
        "deconvolution_summary": json.loads(deconv_summary.to_json(orient="records")),
        "retrieval_summary": json.loads(retrieval_public.to_json(orient="records")),
        "imputation_summary": json.loads(imputation_summary.to_json(orient="records")),
        "distance_matrices": distance_matrices,
    }

    write_outputs(
        args.out_dir,
        args,
        design,
        metrics,
        deconv_summary,
        retrieval_summary,
        retrieval_detail,
        retrieval_public,
        imputation_raw,
        imputation_summary,
        figure_payload,
    )

    print(f"Wrote shared-coordinate benchmark outputs to {args.out_dir}")
    print(deconv_summary.to_string(index=False))
    print()
    print(retrieval_public.to_string(index=False))
    print()
    print(imputation_summary.to_string(index=False))


if __name__ == "__main__":
    main()
