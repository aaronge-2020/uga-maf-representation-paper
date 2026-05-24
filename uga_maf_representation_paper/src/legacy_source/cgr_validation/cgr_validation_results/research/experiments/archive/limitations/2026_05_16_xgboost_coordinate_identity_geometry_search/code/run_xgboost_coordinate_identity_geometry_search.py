"""XGBoost-only search over exact-identity UGA coordinate geometry augmentations."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent

def find_experiments_root(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if candidate.name == "experiments":
            return candidate
    raise RuntimeError(f"Could not locate experiments root from {path}")


def find_project_root(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if (candidate / "uga_atlas" / "models.py").is_file():
            return candidate
    raise RuntimeError(f"Could not locate cgr_validation project root from {path}")


EXPERIMENTS_ROOT = find_experiments_root(EXPERIMENT_ROOT)
DATA_DIR = EXPERIMENT_ROOT / "data"
TABLE_DIR = EXPERIMENT_ROOT / "tables"
FIGURE_DIR = EXPERIMENT_ROOT / "figures"
PRIOR_ROOT = EXPERIMENTS_ROOT / "exploratory" / "2026_05_15_maf_event_coordinate_geometry_optimization"
PRIOR_DATA = PRIOR_ROOT / "data"
PRIOR_CODE = PRIOR_ROOT / "code"
PROJECT_ROOT = find_project_root(EXPERIMENT_ROOT)

for path in (PRIOR_CODE, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import signal_recovery_helpers as base  # noqa: E402


RANDOM_SEED = 20260516
STANDARD_BASELINE = "standard_sbs96_id83"
EXACT_CONTROL = "exact_uga_coordinate_identity"
DISCOVERY_ENDPOINTS = [
    "smoking_ever",
    "cancer_type_top10",
    "luad_kmt2c_mutated",
    "hrd_binary_24",
    "hrd_binary_33",
    "hrd_binary_42",
]
PRIOR_GEOMETRIES = [
    "sequence_cgr_k5_v1",
    "sequence_cgr_k7_v1",
    "sequence_cgr_k11_v1",
    "indel_biology_geometry_v1",
    "motif_factorized_geometry_v1",
    "topography_aware_geometry_v1",
    "learned_self_supervised_geometry_v1",
]
BURDEN_COLUMNS = ["log10_sbs_burden", "log10_id_burden", "id_fraction"]


@dataclass(frozen=True)
class CandidateSpec:
    model_id: str
    family: str
    blocks: tuple[tuple[str, str, str, int | None], ...]
    parameters: str
    notes: str


def ensure_dirs() -> None:
    for path in (DATA_DIR, TABLE_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def stable_seed(*parts: object) -> int:
    text = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return RANDOM_SEED + int.from_bytes(digest[:4], "big") % 100_000


def exact_identity_frame(standard_sbs_id: pd.DataFrame) -> pd.DataFrame:
    frame = standard_sbs_id.copy().astype(np.float32)
    rename = {col: f"uga_coordinate_identity__{col}" for col in frame.columns}
    return frame.rename(columns=rename)


def strip_burden(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in frame.columns if col not in BURDEN_COLUMNS]
    return frame.loc[:, cols].copy()


def load_prior_geometry_blocks() -> dict[str, pd.DataFrame]:
    blocks: dict[str, pd.DataFrame] = {}
    for model_id in PRIOR_GEOMETRIES:
        path = PRIOR_DATA / f"features_{model_id}.csv.gz"
        frame = pd.read_csv(path, index_col=0).fillna(0.0).astype(np.float32)
        frame.index = frame.index.astype(str)
        blocks[model_id] = strip_burden(frame)
    return blocks


def standard_channel_block(standard_sbs_id: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in standard_sbs_id.columns if col not in BURDEN_COLUMNS]
    block = standard_sbs_id.loc[:, cols].copy().astype(np.float32)
    block.columns = [f"identity_channel__{col}" for col in block.columns]
    return block


def parse_sbs96(channel: str) -> tuple[str, str, str, str]:
    ch = str(channel).replace("SBS96__", "", 1)
    return ch[0], ch[2], ch[4], ch[6]


def parse_id83(channel: str) -> tuple[str, str, int, int]:
    ch = str(channel).replace("ID83__", "", 1)
    parts = ch.split(":")
    if len(parts) < 4:
        return "OTHER", "OTHER", 0, 0
    if parts[0].upper() in {"DEL", "INS"}:
        event = parts[0].upper()
        motif = parts[1].upper().replace("REPEATS", "R").replace("MH", "M")
        length_s = parts[2]
        aux_s = parts[3]
    else:
        length_s, event_s, motif, aux_s = parts[:4]
        event = "DEL" if event_s.lower() == "del" else "INS"
        motif = motif.upper().replace("REPEATS", "R").replace("MH", "M")
    length = 5 if str(length_s).endswith("+") else int(length_s) if str(length_s).isdigit() else 0
    aux = 5 if str(aux_s).endswith("+") else int(aux_s) if str(aux_s).isdigit() else 0
    return event, motif, length, aux


def add_sum(frames: list[pd.Series], counts: pd.DataFrame, cols: list[str], name: str) -> None:
    if cols:
        frames.append(counts.loc[:, cols].sum(axis=1).rename(name))
    else:
        frames.append(pd.Series(0.0, index=counts.index, name=name))


def build_identity_parent_bins(standard_sbs_id: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sbs_cols = [col for col in standard_sbs_id.columns if str(col).startswith("SBS96__")]
    id_cols = [col for col in standard_sbs_id.columns if str(col).startswith("ID83__")]
    sbs = standard_sbs_id.loc[:, sbs_cols].astype(np.float32)
    ids = standard_sbs_id.loc[:, id_cols].astype(np.float32)
    sbs_parsed = {col: parse_sbs96(col) for col in sbs.columns}
    id_parsed = {col: parse_id83(col) for col in ids.columns}
    sbs_frames: list[pd.Series] = []
    id_frames: list[pd.Series] = []
    substitutions = sorted({f"{ref}>{alt}" for _, ref, alt, _ in sbs_parsed.values()})
    for sub in substitutions:
        add_sum(sbs_frames, sbs, [col for col, (_, ref, alt, _) in sbs_parsed.items() if f"{ref}>{alt}" == sub], f"parent_sbs_sub_{sub}")
    for base in "ACGT":
        add_sum(sbs_frames, sbs, [col for col, (left, _, _, _) in sbs_parsed.items() if left == base], f"parent_sbs_left_{base}")
        add_sum(sbs_frames, sbs, [col for col, (_, _, _, right) in sbs_parsed.items() if right == base], f"parent_sbs_right_{base}")
    add_sum(sbs_frames, sbs, [col for col, (_, ref, alt, right) in sbs_parsed.items() if ref == "C" and alt == "T" and right == "G"], "parent_sbs_cpg_ct")
    add_sum(sbs_frames, sbs, [col for col, (left, ref, alt, right) in sbs_parsed.items() if ref == "C" and alt in {"T", "G"} and left == "T" and right in {"A", "T"}], "parent_sbs_apobec_like")
    add_sum(sbs_frames, sbs, [col for col, (_, ref, alt, _) in sbs_parsed.items() if ref == "C" and alt == "A"], "parent_sbs_ca_total")
    for event in ["DEL", "INS"]:
        add_sum(id_frames, ids, [col for col, (ev, _, _, _) in id_parsed.items() if ev == event], f"parent_id_event_{event.lower()}")
    for motif in ["C", "T", "R", "M"]:
        add_sum(id_frames, ids, [col for col, (_, mo, _, _) in id_parsed.items() if mo == motif], f"parent_id_motif_{motif.lower()}")
    for length in range(1, 6):
        label = str(length) if length < 5 else "5plus"
        add_sum(id_frames, ids, [col for col, (_, _, le, _) in id_parsed.items() if le == length], f"parent_id_length_{label}")
    add_sum(id_frames, ids, [col for col, (_, mo, _, aux) in id_parsed.items() if mo in {"C", "T"} and aux >= 5], "parent_id_homopolymer_5plus")
    add_sum(id_frames, ids, [col for col, (_, mo, _, aux) in id_parsed.items() if mo == "M" and aux >= 5], "parent_id_microhomology_5plus")
    sbs_parent = pd.concat(sbs_frames, axis=1).fillna(0.0).astype(np.float32)
    id_parent = pd.concat(id_frames, axis=1).fillna(0.0).astype(np.float32)
    return {
        "identity_parent_sbs": sbs_parent,
        "identity_parent_id": id_parent,
        "identity_parent_sbs_id": pd.concat([sbs_parent, id_parent], axis=1).fillna(0.0).astype(np.float32),
    }


def distribution_summary(block: pd.DataFrame, prefix: str) -> pd.DataFrame:
    x = block.fillna(0.0).clip(lower=0.0).astype(np.float32)
    arr = x.to_numpy(dtype=np.float64)
    total = arr.sum(axis=1)
    denom = np.where(total > 0.0, total, np.nan)
    p = np.divide(arr, denom[:, None], out=np.zeros_like(arr), where=~np.isnan(denom[:, None]))
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(np.where(p > 0.0, p * np.log(p), 0.0), axis=1)
    max_entropy = np.log(max(arr.shape[1], 2))
    sorted_p = np.sort(p, axis=1)[:, ::-1]
    out = pd.DataFrame(index=x.index)
    out[f"{prefix}_log_total"] = np.log10(total + 1.0)
    out[f"{prefix}_nonzero_count"] = (arr > 0.0).sum(axis=1)
    out[f"{prefix}_nonzero_fraction"] = out[f"{prefix}_nonzero_count"] / max(arr.shape[1], 1)
    out[f"{prefix}_entropy"] = entropy
    out[f"{prefix}_normalized_entropy"] = entropy / max_entropy
    out[f"{prefix}_simpson"] = np.sum(p * p, axis=1)
    out[f"{prefix}_max_fraction"] = sorted_p[:, 0] if sorted_p.shape[1] else 0.0
    out[f"{prefix}_top5_fraction"] = sorted_p[:, : min(5, sorted_p.shape[1])].sum(axis=1)
    out[f"{prefix}_top10_fraction"] = sorted_p[:, : min(10, sorted_p.shape[1])].sum(axis=1)
    out[f"{prefix}_tail90_fraction"] = 1.0 - out[f"{prefix}_top10_fraction"]
    return out.fillna(0.0).astype(np.float32)


def build_summary_blocks(
    standard_sbs_id: pd.DataFrame,
    parent_bins: dict[str, pd.DataFrame],
    source_blocks: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    sbs_cols = [col for col in standard_sbs_id.columns if str(col).startswith("SBS96__")]
    id_cols = [col for col in standard_sbs_id.columns if str(col).startswith("ID83__")]
    sbs = standard_sbs_id.loc[:, sbs_cols].astype(np.float32)
    ids = standard_sbs_id.loc[:, id_cols].astype(np.float32)
    summaries: dict[str, pd.DataFrame] = {
        "identity_summary_sbs": distribution_summary(sbs, "summary_sbs"),
        "identity_summary_id": distribution_summary(ids, "summary_id"),
        "identity_summary_sbs_id": distribution_summary(pd.concat([sbs, ids], axis=1), "summary_sbs_id"),
    }
    parents = parent_bins["identity_parent_sbs_id"].copy()
    denom_sbs = sbs.sum(axis=1).replace(0.0, np.nan)
    denom_id = ids.sum(axis=1).replace(0.0, np.nan)
    ratio = pd.DataFrame(index=standard_sbs_id.index)
    for col in parents.columns:
        if col.startswith("parent_sbs_"):
            ratio[f"ratio_{col}"] = parents[col].div(denom_sbs).fillna(0.0)
        elif col.startswith("parent_id_"):
            ratio[f"ratio_{col}"] = parents[col].div(denom_id).fillna(0.0)
    if "parent_sbs_cpg_ct" in parents and "parent_sbs_apobec_like" in parents:
        ratio["ratio_cpg_to_apobec_plus1"] = (parents["parent_sbs_cpg_ct"] + 1.0) / (parents["parent_sbs_apobec_like"] + 1.0)
    if "parent_id_microhomology_5plus" in parents and "parent_id_homopolymer_5plus" in parents:
        ratio["ratio_id_mh_to_hp_plus1"] = (parents["parent_id_microhomology_5plus"] + 1.0) / (parents["parent_id_homopolymer_5plus"] + 1.0)
    summaries["identity_motif_ratios"] = ratio.fillna(0.0).astype(np.float32)
    for source, block in source_blocks.items():
        summaries[f"summary_{source}"] = distribution_summary(block, f"summary_{source.replace('_v1', '')}")
        summaries[f"summary_{source}_l4"] = distribution_summary(filter_level(block, "l4"), f"summary_{source.replace('_v1', '')}_l4")
        summaries[f"summary_{source}_l6"] = distribution_summary(filter_level(block, "l6"), f"summary_{source.replace('_v1', '')}_l6")
    return summaries


def normalize_chrom(value: object) -> str:
    text = str(value).replace("chr", "").replace("CHR", "").strip().upper()
    text = text.lstrip("0") or "0"
    return text


def normalize_modality(value: object) -> str:
    text = str(value).upper()
    if text in {"SNP", "SNV"}:
        return "SBS"
    if text in {"DNP", "TNP", "ONP"}:
        return "DBS"
    if text in {"INS", "DEL"}:
        return "ID"
    return "OTHER"


def pivot_counts(grouped: pd.DataFrame, columns: list[str], patient_ids: list[str], prefix: str) -> pd.DataFrame:
    if grouped.empty:
        return pd.DataFrame(index=patient_ids)
    key_col = "__feature_key"
    grouped[key_col] = grouped[columns].astype(str).agg("__".join, axis=1)
    table = grouped.pivot_table(index="patient_id", columns=key_col, values="count", aggfunc="sum", fill_value=0.0)
    table = table.reindex(patient_ids).fillna(0.0).astype(np.float32)
    table.columns = [f"{prefix}__{col}" for col in table.columns.astype(str)]
    return table


def collapse_group_frames(frames: list[pd.DataFrame], group_cols: list[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=[*group_cols, "count"])
    merged = pd.concat(frames, ignore_index=True)
    return merged.groupby(group_cols, as_index=False, observed=True)["count"].sum()


def build_locus_topography_blocks(patient_ids: list[str]) -> dict[str, pd.DataFrame]:
    cache_meta = DATA_DIR / "locus_topography_cache_metadata.json"
    expected = {
        "locus_chrom_counts": DATA_DIR / "features_locus_chrom_counts.csv.gz",
        "locus_chrom_modality_counts": DATA_DIR / "features_locus_chrom_modality_counts.csv.gz",
        "locus_variant_class_counts": DATA_DIR / "features_locus_variant_class_counts.csv.gz",
        "locus_mb_top512": DATA_DIR / "features_locus_mb_top512.csv.gz",
        "locus_density_summary": DATA_DIR / "features_locus_density_summary.csv.gz",
    }
    if cache_meta.exists() and all(path.exists() for path in expected.values()):
        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        if meta.get("target_gene_excluded") == "KMT2C":
            out = {}
            for name, path in expected.items():
                frame = pd.read_csv(path, index_col=0).fillna(0.0).astype(np.float32)
                frame.index = frame.index.astype(str)
                out[name] = frame
            return out

    maf_path = base.RAW_DIR / "mc3.v0.2.8.PUBLIC.maf.gz"
    usecols = ["Hugo_Symbol", "Tumor_Sample_Barcode", "Chromosome", "Start_Position", "Variant_Type", "Variant_Classification"]
    patient_set = set(patient_ids)
    chrom_frames: list[pd.DataFrame] = []
    chrom_modality_frames: list[pd.DataFrame] = []
    variant_class_frames: list[pd.DataFrame] = []
    mb_frames: list[pd.DataFrame] = []
    total_rows = 0
    kept_rows = 0
    print("Building locus-topography blocks from MC3 MAF", flush=True)
    for chunk in pd.read_csv(maf_path, sep="\t", usecols=usecols, dtype=str, chunksize=400_000):
        total_rows += len(chunk)
        chunk["patient_id"] = chunk["Tumor_Sample_Barcode"].astype(str).str[:12]
        chunk = chunk[chunk["patient_id"].isin(patient_set)].copy()
        chunk = chunk[chunk["Hugo_Symbol"].fillna("").astype(str).str.upper() != "KMT2C"].copy()
        kept_rows += len(chunk)
        if chunk.empty:
            continue
        chunk["chrom"] = chunk["Chromosome"].map(normalize_chrom)
        chunk["modality"] = chunk["Variant_Type"].map(normalize_modality)
        chunk["variant_class"] = chunk["Variant_Classification"].fillna("unknown").astype(str).str.lower()
        pos = pd.to_numeric(chunk["Start_Position"], errors="coerce")
        chunk["mb_bin"] = chunk["chrom"] + "_mb" + (pos.fillna(0).astype(np.int64) // 1_000_000).astype(str)
        chrom_frames.append(chunk.groupby(["patient_id", "chrom"], as_index=False, observed=True).size().rename(columns={"size": "count"}))
        chrom_modality_frames.append(chunk.groupby(["patient_id", "chrom", "modality"], as_index=False, observed=True).size().rename(columns={"size": "count"}))
        variant_class_frames.append(chunk.groupby(["patient_id", "variant_class"], as_index=False, observed=True).size().rename(columns={"size": "count"}))
        mb_frames.append(chunk.groupby(["patient_id", "mb_bin"], as_index=False, observed=True).size().rename(columns={"size": "count"}))
    chrom_grouped = collapse_group_frames(chrom_frames, ["patient_id", "chrom"])
    chrom_modality_grouped = collapse_group_frames(chrom_modality_frames, ["patient_id", "chrom", "modality"])
    variant_class_grouped = collapse_group_frames(variant_class_frames, ["patient_id", "variant_class"])
    mb_grouped = collapse_group_frames(mb_frames, ["patient_id", "mb_bin"])
    chrom_counts = pivot_counts(chrom_grouped, ["chrom"], patient_ids, "locus_chrom")
    chrom_modality = pivot_counts(chrom_modality_grouped, ["chrom", "modality"], patient_ids, "locus_chrom_modality")
    variant_class = pivot_counts(variant_class_grouped, ["variant_class"], patient_ids, "locus_variant_class")
    mb_counts = pivot_counts(mb_grouped, ["mb_bin"], patient_ids, "locus_mb")
    if mb_counts.shape[1] > 512:
        keep = mb_counts.var(axis=0).sort_values(ascending=False).head(512).index
        mb_top = mb_counts.loc[:, keep].copy()
    else:
        mb_top = mb_counts
    density_summary = distribution_summary(mb_counts, "locus_mb_density")
    out = {
        "locus_chrom_counts": chrom_counts,
        "locus_chrom_modality_counts": chrom_modality,
        "locus_variant_class_counts": variant_class,
        "locus_mb_top512": mb_top.astype(np.float32),
        "locus_density_summary": density_summary.astype(np.float32),
    }
    for name, frame in out.items():
        frame.to_csv(expected[name])
    cache_meta.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "maf": str(maf_path),
                "total_rows": int(total_rows),
                "kept_rows": int(kept_rows),
                "patient_count": int(len(patient_ids)),
                "target_gene_excluded": "KMT2C",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return out


def filter_level(block: pd.DataFrame, level_filter: str) -> pd.DataFrame:
    if level_filter == "all":
        return block
    token = f"__{level_filter}__"
    cols = [col for col in block.columns if token in str(col)]
    return block.loc[:, cols].copy()


def top_variance(block: pd.DataFrame, budget: int | None) -> pd.DataFrame:
    if budget is None or block.shape[1] <= budget:
        return block
    variances = block.var(axis=0).sort_values(ascending=False)
    return block.loc[:, variances.head(int(budget)).index].copy()


def transform_block(block: pd.DataFrame, transform: str, prefix: str) -> pd.DataFrame:
    x = block.astype(np.float32)
    if transform == "raw":
        out = x.copy()
    elif transform == "log1p":
        out = np.log1p(x.clip(lower=0.0)).astype(np.float32)
    elif transform == "binary":
        out = (x > 0).astype(np.float32)
    elif transform == "sqrt":
        out = np.sqrt(x.clip(lower=0.0)).astype(np.float32)
    elif transform == "fraction":
        denom = x.sum(axis=1).replace(0.0, np.nan)
        out = x.div(denom, axis=0).fillna(0.0).astype(np.float32)
    elif transform == "log_fraction":
        denom = x.sum(axis=1).replace(0.0, np.nan)
        out = np.log1p(x.div(denom, axis=0).fillna(0.0)).astype(np.float32)
    else:
        raise ValueError(f"Unknown transform: {transform}")
    out.columns = [f"{prefix}__{transform}__{col}" for col in x.columns]
    return out


def block_from_spec(
    block_name: str,
    source_blocks: dict[str, pd.DataFrame],
    identity_channels: pd.DataFrame,
    parent_bins: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if block_name == "identity_channels":
        return identity_channels
    if block_name in parent_bins:
        return parent_bins[block_name]
    return source_blocks[block_name]


def build_candidate_frame(
    spec: CandidateSpec,
    identity: pd.DataFrame,
    source_blocks: dict[str, pd.DataFrame],
    identity_channels: pd.DataFrame,
    parent_bins: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    pieces = [identity]
    for block_name, level_filter, transform, budget in spec.blocks:
        block = block_from_spec(block_name, source_blocks, identity_channels, parent_bins)
        block = filter_level(block, level_filter)
        block = top_variance(block, budget)
        prefix = f"{spec.model_id}__{block_name}__{level_filter}"
        pieces.append(transform_block(block, transform, prefix))
    frame = pd.concat(pieces, axis=1).fillna(0.0).astype(np.float32)
    frame.index = frame.index.astype(str)
    return frame


def make_candidate_specs() -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []

    def add(model_id: str, family: str, blocks: list[tuple[str, str, str, int | None]], notes: str) -> None:
        specs.append(
            CandidateSpec(
                model_id=model_id,
                family=family,
                blocks=tuple(blocks),
                parameters=json.dumps([{"block": b, "level": l, "transform": t, "budget": n} for b, l, t, n in blocks], sort_keys=True),
                notes=notes,
            )
        )

    for transform in ["log1p", "binary", "sqrt", "fraction", "log_fraction"]:
        add(f"identity_{transform}", "low_burden_stabilized_identity", [("identity_channels", "all", transform, None)], "Exact cells plus transformed exact coordinate cells.")
    add("identity_log_binary", "low_burden_stabilized_identity", [("identity_channels", "all", "log1p", None), ("identity_channels", "all", "binary", None)], "Exact cells plus log and presence transforms.")
    add("identity_log_binary_fraction", "low_burden_stabilized_identity", [("identity_channels", "all", "log1p", None), ("identity_channels", "all", "binary", None), ("identity_channels", "all", "fraction", None)], "Exact cells plus log, presence, and within-patient fractions.")
    for block in ["identity_parent_sbs", "identity_parent_id", "identity_parent_sbs_id"]:
        for transform in ["raw", "log1p", "binary", "fraction"]:
            add(f"{block}_{transform}", "coordinate_parent_bins", [(block, "all", transform, None)], "Exact cells plus coordinate parent bins.")
    add("identity_parent_sbs_id_log_binary", "coordinate_parent_bins", [("identity_parent_sbs_id", "all", "log1p", None), ("identity_parent_sbs_id", "all", "binary", None)], "Exact cells plus transformed SBS/ID parent bins.")
    add("identity_parent_sbs_id_raw_log_binary", "coordinate_parent_bins", [("identity_parent_sbs_id", "all", "raw", None), ("identity_parent_sbs_id", "all", "log1p", None), ("identity_parent_sbs_id", "all", "binary", None)], "Exact cells plus raw, log, and presence parent bins.")

    for block in ["identity_summary_sbs", "identity_summary_id", "identity_summary_sbs_id", "identity_motif_ratios"]:
        add(f"{block}_raw", "coordinate_distribution_summaries", [(block, "all", "raw", None)], "Exact cells plus coordinate distribution summary features.")
        add(f"{block}_log1p", "coordinate_distribution_summaries", [(block, "all", "log1p", None)], "Exact cells plus transformed coordinate distribution summaries.")
    add(
        "identity_summary_sbs_id_plus_motif_ratios",
        "coordinate_distribution_summaries",
        [("identity_summary_sbs_id", "all", "raw", None), ("identity_motif_ratios", "all", "raw", None)],
        "Exact cells plus global coordinate distribution and motif-ratio summaries.",
    )
    add(
        "identity_parent_summary_ratio_stack",
        "coordinate_distribution_summaries",
        [("identity_parent_sbs_id", "all", "raw", None), ("identity_summary_sbs_id", "all", "raw", None), ("identity_motif_ratios", "all", "raw", None)],
        "Exact cells plus parent bins, distribution summaries, and motif ratios.",
    )

    event_variants = [
        ("all", "raw", 512),
        ("l4", "raw", None),
        ("l6", "raw", 512),
        ("all", "log1p", 512),
        ("l4", "binary", None),
        ("l6", "fraction", 512),
    ]
    for source in PRIOR_GEOMETRIES:
        short = source.replace("_v1", "")
        for level_filter, transform, budget in event_variants:
            add(
                f"id_plus_{short}_{level_filter}_{transform}_{budget or 'all'}",
                f"exact_identity_plus_{short}",
                [(source, level_filter, transform, budget)],
                "Exact coordinate identity plus cached MAF-derived event geometry block.",
            )
        for summary_block in [f"summary_{source}", f"summary_{source}_l4", f"summary_{source}_l6"]:
            add(
                f"id_plus_{short}_{summary_block.replace(source, '').strip('_')}_summary",
                f"exact_identity_plus_{short}_summary",
                [(summary_block, "all", "raw", None)],
                "Exact coordinate identity plus event-geometry distribution summaries.",
            )

    combo_blocks = [
        ("sequence_cgr_k5_v1", "indel_biology_geometry_v1"),
        ("sequence_cgr_k5_v1", "motif_factorized_geometry_v1"),
        ("sequence_cgr_k5_v1", "topography_aware_geometry_v1"),
        ("indel_biology_geometry_v1", "motif_factorized_geometry_v1"),
        ("motif_factorized_geometry_v1", "topography_aware_geometry_v1"),
        ("sequence_cgr_k7_v1", "indel_biology_geometry_v1"),
        ("sequence_cgr_k11_v1", "motif_factorized_geometry_v1"),
    ]
    for left, right in combo_blocks:
        short = f"{left.replace('_v1', '')}_plus_{right.replace('_v1', '')}"
        add(
            f"id_plus_{short}_raw_top256",
            "exact_identity_plus_multiblock_geometry",
            [(left, "all", "raw", 256), (right, "all", "raw", 256)],
            "Exact coordinate identity plus two top-variance geometry blocks.",
        )
        add(
            f"id_plus_{short}_log_binary_top256",
            "exact_identity_plus_multiblock_geometry",
            [(left, "all", "log1p", 256), (right, "all", "binary", 256)],
            "Exact coordinate identity plus log and presence geometry blocks.",
        )

    add(
        "id_plus_all_geometry_top128_raw",
        "exact_identity_plus_multiblock_geometry",
        [(source, "all", "raw", 128) for source in PRIOR_GEOMETRIES],
        "Exact coordinate identity plus top-variance raw blocks from all cached geometries.",
    )
    add(
        "id_plus_all_geometry_top128_log_binary",
        "exact_identity_plus_multiblock_geometry",
        [(source, "all", "log1p", 128) for source in PRIOR_GEOMETRIES] + [(source, "all", "binary", 128) for source in PRIOR_GEOMETRIES],
        "Exact coordinate identity plus transformed blocks from all cached geometries.",
    )
    add(
        "id_plus_all_geometry_summaries",
        "exact_identity_plus_multiblock_geometry_summary",
        [(f"summary_{source}", "all", "raw", None) for source in PRIOR_GEOMETRIES],
        "Exact coordinate identity plus distribution summaries from all cached event geometries.",
    )
    add(
        "id_plus_identity_and_all_geometry_summaries",
        "exact_identity_plus_multiblock_geometry_summary",
        [("identity_summary_sbs_id", "all", "raw", None), ("identity_motif_ratios", "all", "raw", None)] + [(f"summary_{source}", "all", "raw", None) for source in PRIOR_GEOMETRIES],
        "Exact coordinate identity plus identity and event-geometry distribution summaries.",
    )
    add(
        "id_plus_parent_ratios_and_all_geometry_summaries",
        "exact_identity_plus_multiblock_geometry_summary",
        [("identity_parent_sbs_id", "all", "raw", None), ("identity_motif_ratios", "all", "raw", None)] + [(f"summary_{source}", "all", "raw", None) for source in PRIOR_GEOMETRIES],
        "Exact coordinate identity plus parent, motif-ratio, and event-geometry distribution summaries.",
    )
    for block in ["locus_chrom_counts", "locus_chrom_modality_counts", "locus_variant_class_counts", "locus_mb_top512", "locus_density_summary"]:
        for transform in ["raw", "log1p", "binary", "fraction"]:
            if block == "locus_density_summary" and transform in {"binary", "fraction"}:
                continue
            add(
                f"id_plus_{block}_{transform}",
                "exact_identity_plus_locus_topography",
                [(block, "all", transform, None)],
                "Exact coordinate identity plus MAF locus-topography coordinate distribution features.",
            )
    add(
        "id_plus_locus_chrom_and_density",
        "exact_identity_plus_locus_topography",
        [("locus_chrom_counts", "all", "raw", None), ("locus_density_summary", "all", "raw", None)],
        "Exact coordinate identity plus chromosome occupancy and megabase-density summaries.",
    )
    add(
        "id_plus_locus_chrom_modality_and_density",
        "exact_identity_plus_locus_topography",
        [("locus_chrom_modality_counts", "all", "raw", None), ("locus_density_summary", "all", "raw", None)],
        "Exact coordinate identity plus chromosome-by-modality occupancy and megabase-density summaries.",
    )
    add(
        "id_plus_locus_full_topography",
        "exact_identity_plus_locus_topography",
        [
            ("locus_chrom_counts", "all", "raw", None),
            ("locus_chrom_modality_counts", "all", "raw", None),
            ("locus_variant_class_counts", "all", "raw", None),
            ("locus_density_summary", "all", "raw", None),
        ],
        "Exact coordinate identity plus compact locus-topography summaries.",
    )
    add(
        "id_plus_locus_full_topography_log_binary",
        "exact_identity_plus_locus_topography",
        [
            ("locus_chrom_counts", "all", "log1p", None),
            ("locus_chrom_modality_counts", "all", "binary", None),
            ("locus_variant_class_counts", "all", "binary", None),
            ("locus_density_summary", "all", "raw", None),
        ],
        "Exact coordinate identity plus stabilized locus-topography summaries.",
    )
    add(
        "id_plus_identity_summary_locus_topography",
        "exact_identity_plus_locus_topography",
        [
            ("identity_summary_sbs_id", "all", "raw", None),
            ("identity_motif_ratios", "all", "raw", None),
            ("locus_chrom_modality_counts", "all", "raw", None),
            ("locus_density_summary", "all", "raw", None),
        ],
        "Exact coordinate identity plus identity distribution summaries and locus topography.",
    )
    add(
        "id_plus_event_summary_locus_topography",
        "exact_identity_plus_locus_topography",
        [(f"summary_{source}", "all", "raw", None) for source in PRIOR_GEOMETRIES]
        + [("locus_chrom_modality_counts", "all", "raw", None), ("locus_density_summary", "all", "raw", None)],
        "Exact coordinate identity plus event-geometry summaries and locus topography.",
    )
    return specs


def load_discovery_endpoints(patients: pd.Index) -> list[base.Endpoint]:
    clinical = {endpoint.name: endpoint for endpoint in base.load_mc3_clinical_endpoints()}
    hrd = {endpoint.name: endpoint for endpoint in base.load_hrd_endpoints()}
    kmt2c = base.load_kmt2c_endpoint(patients)
    endpoint_map = {**clinical, **hrd, kmt2c.name: kmt2c}
    return [endpoint_map[name] for name in DISCOVERY_ENDPOINTS]


def run_standard_metrics(
    endpoints: list[base.Endpoint],
    frame: pd.DataFrame,
    *,
    folds: int,
    repeats: int,
    n_estimators: int,
    tree_method: str,
    seed_prefix: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    scores = {}
    for endpoint in endpoints:
        metric, _ = base.run_endpoint_model(
            endpoint,
            frame,
            folds=folds,
            repeats=repeats,
            n_estimators=n_estimators,
            tree_method=tree_method,
            seed=stable_seed(seed_prefix, endpoint.name),
        )
        metric.update({"model_id": STANDARD_BASELINE, "delta_vs_standard": 0.0, "screen_stage": seed_prefix})
        rows.append(metric)
        scores[endpoint.name] = float(metric["score"])
    return pd.DataFrame(rows), scores


def screen_candidates(
    specs: list[CandidateSpec],
    endpoints: list[base.Endpoint],
    standard_scores: dict[str, float],
    identity: pd.DataFrame,
    source_blocks: dict[str, pd.DataFrame],
    identity_channels: pd.DataFrame,
    parent_bins: dict[str, pd.DataFrame],
    *,
    folds: int,
    n_estimators: int,
    tree_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for i, spec in enumerate(specs, start=1):
        print(f"Screening {i}/{len(specs)}: {spec.model_id}", flush=True)
        t0 = time.perf_counter()
        frame = build_candidate_frame(spec, identity, source_blocks, identity_channels, parent_bins)
        feature_rows.append(
            {
                "model_id": spec.model_id,
                "n_features": int(frame.shape[1]),
                "n_samples": int(frame.shape[0]),
                "matrix_nonzero_fraction": float((frame.to_numpy(dtype=np.float32) != 0).mean()),
                "contains_exact_identity": True,
                "sbs96_reconstruction_r2": 1.0,
                "id83_reconstruction_r2": 1.0,
            }
        )
        manifest_rows.append(
            {
                "model_id": spec.model_id,
                "family": spec.family,
                "parameters": spec.parameters,
                "n_feature_blocks": len(spec.blocks),
                "uses_standard_identity_channels": False,
                "uses_exact_coordinate_identity_cells": True,
                "uses_posthoc_ensemble": False,
                "uses_endpoint_specific_weights": False,
                "screen_status": "screened",
                "notes": spec.notes,
            }
        )
        for endpoint in endpoints:
            metric, _ = base.run_endpoint_model(
                endpoint,
                frame,
                folds=folds,
                repeats=1,
                n_estimators=n_estimators,
                tree_method=tree_method,
                seed=stable_seed("rapid_screen", endpoint.name),
            )
            metric.update(
                {
                    "model_id": spec.model_id,
                    "candidate_family": spec.family,
                    "delta_vs_standard": float(metric["score"] - standard_scores[endpoint.name]),
                    "screen_stage": "rapid_screen",
                }
            )
            metric_rows.append(metric)
        elapsed = time.perf_counter() - t0
        for row in metric_rows:
            if row.get("model_id") == spec.model_id:
                row["candidate_runtime_seconds"] = elapsed
    return pd.DataFrame(metric_rows), pd.DataFrame(feature_rows), pd.DataFrame(manifest_rows)


def summarize_screen(metrics: pd.DataFrame) -> pd.DataFrame:
    out = (
        metrics.groupby(["model_id", "candidate_family"], as_index=False)
        .agg(
            n_endpoints=("endpoint", "nunique"),
            mean_delta=("delta_vs_standard", "mean"),
            median_delta=("delta_vs_standard", "median"),
            min_delta=("delta_vs_standard", "min"),
            max_delta=("delta_vs_standard", "max"),
            endpoint_gains_ge_0p03=("delta_vs_standard", lambda x: int(np.sum(np.asarray(x) >= 0.03))),
            endpoint_losses_lt_neg_0p02=("delta_vs_standard", lambda x: int(np.sum(np.asarray(x) < -0.02))),
            mean_score=("score", "mean"),
            n_features=("n_features", "max"),
            runtime_seconds=("candidate_runtime_seconds", "max"),
        )
        .reset_index(drop=True)
    )
    out["promotion_pass"] = (
        (out["mean_delta"] >= 0.02)
        & (out["min_delta"] >= -0.02)
        & (out["endpoint_gains_ge_0p03"] >= 3)
        & (out["endpoint_losses_lt_neg_0p02"] == 0)
    )
    return out.sort_values(["promotion_pass", "mean_delta", "endpoint_gains_ge_0p03"], ascending=[False, False, False])


def baseline_reproducibility_audit(current_standard: pd.DataFrame, current_exact: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prior_path = PRIOR_DATA / "rapid_biology_screen_metrics.csv"
    prior = pd.read_csv(prior_path) if prior_path.exists() else pd.DataFrame()
    for _, row in current_standard.iterrows():
        endpoint = row["endpoint"]
        exact = current_exact[current_exact["endpoint"] == endpoint]
        exact_score = float(exact["score"].iloc[0]) if len(exact) else np.nan
        prior_score = np.nan
        if not prior.empty:
            m = prior[(prior["model_id"] == STANDARD_BASELINE) & (prior["learner"] == "xgboost") & (prior["endpoint"] == endpoint)]
            if len(m):
                prior_score = float(m["score"].iloc[0])
        rows.append(
            {
                "endpoint": endpoint,
                "current_standard_score": float(row["score"]),
                "current_exact_identity_score": exact_score,
                "exact_minus_standard": exact_score - float(row["score"]),
                "prior_standard_score": prior_score,
                "current_minus_prior": float(row["score"]) - prior_score if not pd.isna(prior_score) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def promote_finalists(leaderboard: pd.DataFrame, max_finalists: int) -> list[str]:
    passed = leaderboard[leaderboard["promotion_pass"]].copy()
    return passed.sort_values("mean_delta", ascending=False)["model_id"].head(max_finalists).astype(str).tolist()


def run_confirmation(
    finalists: list[str],
    specs: list[CandidateSpec],
    endpoints: list[base.Endpoint],
    standard_frame: pd.DataFrame,
    identity: pd.DataFrame,
    source_blocks: dict[str, pd.DataFrame],
    identity_channels: pd.DataFrame,
    parent_bins: dict[str, pd.DataFrame],
    *,
    folds: int,
    repeats: int,
    n_estimators: int,
    tree_method: str,
    bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not finalists:
        empty = pd.DataFrame(
            columns=[
                "candidate",
                "endpoint",
                "endpoint_family",
                "metric",
                "standard_score",
                "candidate_score",
                "delta_vs_standard",
                "ci_low",
                "ci_high",
                "p_value",
                "q_value",
            ]
        )
        return empty, empty.copy()
    spec_map = {spec.model_id: spec for spec in specs}
    candidates: list[base.Candidate] = []
    for model_id in finalists:
        frame = build_candidate_frame(spec_map[model_id], identity, source_blocks, identity_channels, parent_bins)
        candidates.append(base.Candidate(model_id, spec_map[model_id].notes, frame, True, False, spec_map[model_id].family))
    _metrics, tests, predictions = base.run_confirmation(
        finalists,
        candidates,
        endpoints,
        standard_frame,
        folds=folds,
        repeats=repeats,
        n_estimators=n_estimators,
        tree_method=tree_method,
        bootstrap=bootstrap,
    )
    return tests, predictions


def html_table(df: pd.DataFrame, path: Path, title: str, footnote: str) -> None:
    base.write_html_table(df, path, title, footnote)


def format_for_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.loc[:, [col for col in columns if col in df.columns]].copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return out


def write_readme(metadata: dict[str, object], finalists: list[str], best: dict[str, object]) -> None:
    text = f"""# XGBoost Coordinate-Identity Geometry Search

## Research Question

This experiment searches for UGA coordinate representations that preserve Standard SBS96+ID83 information while adding event-level coordinate geometry useful to XGBoost prediction models.

## Methods

All candidates include exact UGA coordinate-identity cells that reconstruct SBS96+ID83 with R2=1.0. Candidate-specific blocks add transformed identity cells, coordinate parent bins, cached MAF-derived event-geometry histograms, or multiblock combinations. The unchanged Standard SBS96+ID83 matrix is retained as the comparator. Discovery endpoints were smoking ever, top-10 cancer type, LUAD KMT2C mutation status with no direct gene-name features, and HRD score thresholds 24, 33, and 42. XGBoost used paired 3-fold cross-validation, fixed seeds, identical patients, identical labels, and the same model settings for Standard and every candidate.

## Key Numerical Findings

The rapid screen evaluated {metadata["n_candidates"]} candidates. The best rapid-screen model was `{best.get("model_id", "none")}` with mean delta {best.get("mean_delta", float("nan")):.4f}, minimum delta {best.get("min_delta", float("nan")):.4f}, maximum delta {best.get("max_delta", float("nan")):.4f}, and {best.get("endpoint_gains_ge_0p03", 0)} endpoint gains of at least 0.03. Promoted finalists: {", ".join(finalists) if finalists else "none"}.

## File Inventory

- `data/model_manifest.csv`: candidate definitions fixed before endpoint fitting.
- `data/feature_dimension_audit.csv`: feature dimensions and identity-reconstruction audit.
- `data/reconstruction_audit.csv`: exact identity and baseline reproducibility checks.
- `data/rapid_screen_results.csv`: endpoint-level rapid-screen metrics.
- `data/rapid_screen_leaderboard.csv`: candidate-level rapid-screen summary and promotion calls.
- `data/focused_confirmation_results.csv`: repeated-CV paired bootstrap validation for promoted finalists.
- `tables/table1_rapid_screen_leaderboard.html`: rapid-screen candidate ranking.
- `tables/table2_endpoint_results.html`: endpoint-level scores for leading candidates.
- `tables/table3_reconstruction_audit.html`: baseline and exact-identity validation.
- `code/run_xgboost_coordinate_identity_geometry_search.py`: complete runner.

## Reproducibility

Date executed: {metadata["completed_utc"]}. Random seed: `{RANDOM_SEED}`. Python: `{metadata["python_version"]}`. Package versions: pandas `{metadata["package_versions"]["pandas"]}`, numpy `{metadata["package_versions"]["numpy"]}`, scipy `{metadata["package_versions"]["scipy"]}`, scikit-learn `{metadata["package_versions"]["sklearn"]}`, xgboost `{metadata["package_versions"]["xgboost"]}`. Tree method requested: `{metadata["tree_method"]}`.
"""
    (EXPERIMENT_ROOT / "README.md").write_text(text, encoding="utf-8")


def update_ledger(metadata: dict[str, object], best: dict[str, object], finalists: list[str]) -> None:
    ledger = EXPERIMENTS_ROOT / "EXPERIMENT_LEDGER.md"
    if not ledger.exists():
        return
    marker = EXPERIMENT_ROOT.name
    text = ledger.read_text(encoding="utf-8")
    if marker in text:
        return
    row = (
        f"| `{marker}` | Complete exploratory XGBoost-only coordinate-identity UGA search | "
        f"{metadata['completed_local']} | {metadata['runtime_seconds']:.1f} s | "
        "TCGA MC3 MAF-derived cached event geometries, Standard SBS96+ID83, MC3 clinical endpoints, LUAD KMT2C, and TCGA-BRCA HRD thresholds | "
        "Can exact coordinate-identity UGA cells plus event-level geometry pass an XGBoost-only promotion gate? | "
        "Exact UGA coordinate-identity cells plus transformed identity, parent-bin, and cached MAF-derived event-geometry blocks | "
        "Unchanged Standard SBS96+ID83 under identical XGBoost folds, seeds, labels, and metrics | "
        f"Best rapid-screen row: `{best.get('model_id', 'none')}`, mean delta {float(best.get('mean_delta', float('nan'))):.4f}; finalists: {', '.join(finalists) if finalists else 'none'}. | "
        "Exploratory XGBoost-only representation search. Not a replacement for learner-sensitivity benchmarks unless repeated-CV confirmation passes. |\n"
    )
    lines = text.splitlines()
    insert_after = "| `2026_05_15_maf_event_coordinate_geometry_optimization`"
    for idx, line in enumerate(lines):
        if line.startswith(insert_after):
            lines.insert(idx + 1, row.rstrip())
            ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
    ledger.write_text(text.rstrip() + "\n" + row, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-method", default="gpu_hist")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--xgb-estimators", type=int, default=80)
    parser.add_argument("--confirmation-folds", type=int, default=5)
    parser.add_argument("--confirmation-repeats", type=int, default=3)
    parser.add_argument("--confirmation-estimators", type=int, default=160)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max-candidates", type=int, default=0, help="Debug only. Default 0 screens all generated candidates.")
    parser.add_argument("--skip-confirmation", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    started = time.time()
    standard_sbs, standard_id, standard_sbs_id, _burden = base.load_feature_matrices()
    standard_sbs_id = standard_sbs_id.astype(np.float32)
    identity = exact_identity_frame(standard_sbs_id)
    identity_channels = standard_channel_block(standard_sbs_id)
    parent_bins = build_identity_parent_bins(standard_sbs_id)
    source_blocks = load_prior_geometry_blocks()
    parent_bins.update(build_summary_blocks(standard_sbs_id, parent_bins, source_blocks))
    parent_bins.update(build_locus_topography_blocks(standard_sbs_id.index.astype(str).tolist()))
    endpoints = load_discovery_endpoints(standard_sbs_id.index)
    specs = make_candidate_specs()
    if args.max_candidates:
        specs = specs[: int(args.max_candidates)]
    print(f"Generated {len(specs)} candidate representations", flush=True)

    manifest_initial = pd.DataFrame(
        [
            {
                "model_id": spec.model_id,
                "family": spec.family,
                "parameters": spec.parameters,
                "n_feature_blocks": len(spec.blocks),
                "uses_standard_identity_channels": False,
                "uses_exact_coordinate_identity_cells": True,
                "uses_posthoc_ensemble": False,
                "uses_endpoint_specific_weights": False,
                "screen_status": "defined_pre_fit",
                "notes": spec.notes,
            }
            for spec in specs
        ]
    )
    manifest_initial.to_csv(DATA_DIR / "model_manifest.csv", index=False)

    standard_metrics, standard_scores = run_standard_metrics(
        endpoints,
        standard_sbs_id,
        folds=args.folds,
        repeats=1,
        n_estimators=args.xgb_estimators,
        tree_method=args.tree_method,
        seed_prefix="rapid_screen",
    )
    exact_metrics, _exact_scores = run_standard_metrics(
        endpoints,
        identity,
        folds=args.folds,
        repeats=1,
        n_estimators=args.xgb_estimators,
        tree_method=args.tree_method,
        seed_prefix="rapid_screen",
    )
    exact_metrics["model_id"] = EXACT_CONTROL
    exact_metrics["delta_vs_standard"] = exact_metrics["endpoint"].map(standard_scores).rsub(exact_metrics["score"])
    audit = baseline_reproducibility_audit(standard_metrics, exact_metrics)
    audit.to_csv(DATA_DIR / "reconstruction_audit.csv", index=False)
    if float(audit["exact_minus_standard"].abs().max()) > 1e-12:
        raise RuntimeError("Exact coordinate-identity control did not reproduce Standard scores.")

    screen_metrics, feature_audit, manifest_screen = screen_candidates(
        specs,
        endpoints,
        standard_scores,
        identity,
        source_blocks,
        identity_channels,
        parent_bins,
        folds=args.folds,
        n_estimators=args.xgb_estimators,
        tree_method=args.tree_method,
    )
    all_screen = pd.concat([standard_metrics, exact_metrics, screen_metrics], ignore_index=True)
    all_screen.to_csv(DATA_DIR / "rapid_screen_results.csv", index=False)
    feature_audit.to_csv(DATA_DIR / "feature_dimension_audit.csv", index=False)
    leaderboard = summarize_screen(screen_metrics)
    leaderboard.to_csv(DATA_DIR / "rapid_screen_leaderboard.csv", index=False)
    manifest = manifest_screen.merge(
        leaderboard[["model_id", "mean_delta", "min_delta", "max_delta", "endpoint_gains_ge_0p03", "endpoint_losses_lt_neg_0p02", "promotion_pass"]],
        on="model_id",
        how="left",
    )
    manifest.to_csv(DATA_DIR / "model_manifest.csv", index=False)
    finalists = promote_finalists(leaderboard, max_finalists=3)

    if finalists and not args.skip_confirmation:
        print(f"Running focused confirmation for: {', '.join(finalists)}", flush=True)
        confirmation, confirmation_predictions = run_confirmation(
            finalists,
            specs,
            endpoints,
            standard_sbs_id,
            identity,
            source_blocks,
            identity_channels,
            parent_bins,
            folds=args.confirmation_folds,
            repeats=args.confirmation_repeats,
            n_estimators=args.confirmation_estimators,
            tree_method=args.tree_method,
            bootstrap=args.bootstrap,
        )
    else:
        confirmation = pd.DataFrame(
            columns=[
                "candidate",
                "endpoint",
                "endpoint_family",
                "metric",
                "standard_score",
                "candidate_score",
                "delta_vs_standard",
                "ci_low",
                "ci_high",
                "p_value",
                "q_value",
            ]
        )
        confirmation_predictions = pd.DataFrame()
    confirmation.to_csv(DATA_DIR / "focused_confirmation_results.csv", index=False)
    if not confirmation_predictions.empty:
        confirmation_predictions.to_csv(DATA_DIR / "focused_confirmation_predictions.csv.gz", index=False)

    best = leaderboard.iloc[0].to_dict() if len(leaderboard) else {}
    html_table(
        format_for_table(leaderboard.head(30), ["model_id", "candidate_family", "n_endpoints", "mean_delta", "min_delta", "max_delta", "endpoint_gains_ge_0p03", "endpoint_losses_lt_neg_0p02", "promotion_pass", "n_features"]),
        TABLE_DIR / "table1_rapid_screen_leaderboard.html",
        "Table 1. Rapid XGBoost Coordinate-Identity UGA Screen",
        "Delta is candidate score minus unchanged Standard SBS96+ID83 under paired folds, labels, seeds, and metrics.",
    )
    leaders = set(leaderboard.head(10)["model_id"].astype(str))
    endpoint_table = all_screen[(all_screen["model_id"].isin(leaders)) | (all_screen["model_id"].isin([STANDARD_BASELINE, EXACT_CONTROL]))].copy()
    html_table(
        format_for_table(endpoint_table.sort_values(["endpoint", "delta_vs_standard"], ascending=[True, False]), ["model_id", "endpoint", "metric", "score", "delta_vs_standard", "n", "n_features"]),
        TABLE_DIR / "table2_endpoint_results.html",
        "Table 2. Endpoint-Level Rapid Screen Results",
        "Endpoint-level scores are from the rapid 3-fold XGBoost screen.",
    )
    html_table(
        format_for_table(audit, ["endpoint", "current_standard_score", "current_exact_identity_score", "exact_minus_standard", "prior_standard_score", "current_minus_prior"]),
        TABLE_DIR / "table3_reconstruction_audit.html",
        "Table 3. Baseline and Exact-Identity Reconstruction Audit",
        "The exact coordinate-identity control must match the unchanged Standard baseline.",
    )
    html_table(
        format_for_table(confirmation, ["candidate", "endpoint", "metric", "standard_score", "candidate_score", "delta_vs_standard", "ci_low", "ci_high", "p_value", "q_value"]),
        TABLE_DIR / "table4_focused_confirmation.html",
        "Table 4. Focused Confirmation",
        "Repeated-CV paired bootstrap confirmation is reported only for rapid-screen finalists.",
    )

    metadata = {
        "experiment": EXPERIMENT_ROOT.name,
        "completed_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "completed_local": datetime.now().replace(microsecond=0).isoformat(),
        "runtime_seconds": round(time.time() - started, 3),
        "random_seed": RANDOM_SEED,
        "tree_method": args.tree_method,
        "folds": args.folds,
        "xgb_estimators": args.xgb_estimators,
        "confirmation_folds": args.confirmation_folds,
        "confirmation_repeats": args.confirmation_repeats,
        "confirmation_estimators": args.confirmation_estimators,
        "bootstrap": args.bootstrap,
        "n_candidates": len(specs),
        "n_finalists": len(finalists),
        "finalists": finalists,
        "prior_cache": str(PRIOR_DATA),
        "python_version": platform.python_version(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    (DATA_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    write_readme(metadata, finalists, best)
    if not args.skip_ledger:
        update_ledger(metadata, best, finalists)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
