"""XGBoost-only search over MAF event-gene coordinate representations."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from utils.checkpointing import merge_checkpoint_rows, read_completed_keys


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
PRIOR_CODE = PRIOR_ROOT / "code"
PROJECT_ROOT = find_project_root(EXPERIMENT_ROOT)
REFERENCE_ROOT = EXPERIMENTS_ROOT / "archive" / "limitations" / "2026_05_16_xgboost_coordinate_identity_geometry_search"
REFERENCE_DATA = REFERENCE_ROOT / "data"

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
BURDEN_COLUMNS = ["log10_sbs_burden", "log10_id_burden", "id_fraction"]
TARGET_GENE_EXCLUSION = "KMT2C"

PATHWAY_SETS: dict[str, set[str]] = {
    "ddr_hr": {
        "BRCA1",
        "BRCA2",
        "PALB2",
        "RAD51",
        "RAD51B",
        "RAD51C",
        "RAD51D",
        "BARD1",
        "BRIP1",
        "FANCA",
        "FANCC",
        "FANCD2",
        "FANCE",
        "FANCF",
        "FANCG",
        "FANCI",
        "FANCL",
    },
    "ddr_mmr": {"MLH1", "MSH2", "MSH6", "PMS2", "EPCAM"},
    "ddr_checkpoint": {"ATM", "ATR", "CHEK1", "CHEK2", "TP53", "MDM2"},
    "polymerase": {"POLE", "POLD1", "POLD2", "POLD3", "POLD4"},
    "chromatin": {
        "ARID1A",
        "ARID1B",
        "SMARCA4",
        "SMARCB1",
        "PBRM1",
        "SETD2",
        "CREBBP",
        "EP300",
        "KMT2A",
        "KMT2B",
        "KMT2D",
        "KDM6A",
        "EZH2",
    },
    "ras_rtk": {
        "KRAS",
        "NRAS",
        "HRAS",
        "BRAF",
        "EGFR",
        "ERBB2",
        "MET",
        "ALK",
        "RET",
        "ROS1",
        "FGFR1",
        "FGFR2",
        "FGFR3",
    },
    "pi3k": {"PIK3CA", "PIK3R1", "PTEN", "AKT1", "AKT2", "MTOR", "TSC1", "TSC2"},
    "cell_cycle": {"TP53", "RB1", "CDKN2A", "CDK4", "CDK6", "CCND1", "CCNE1", "MYC"},
    "lung_smoking_relevant": {"TP53", "KRAS", "KEAP1", "STK11", "NFE2L2", "SMARCA4", "RBM10", "BRAF"},
    "breast_hrd_relevant": {"BRCA1", "BRCA2", "PALB2", "ATM", "CHEK2", "RAD51C", "RAD51D", "BRIP1", "BARD1"},
}

DAMAGING_CLASSES = {
    "missense_mutation",
    "nonsense_mutation",
    "frame_shift_del",
    "frame_shift_ins",
    "splice_site",
    "translation_start_site",
    "nonstop_mutation",
    "in_frame_del",
    "in_frame_ins",
}


@dataclass(frozen=True)
class CandidateSpec:
    model_id: str
    family: str
    blocks: tuple[str, ...]
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
    return frame.rename(columns={col: f"uga_coordinate_identity__{col}" for col in frame.columns})


def transform_counts(block: pd.DataFrame, transform: str, prefix: str) -> pd.DataFrame:
    x = block.fillna(0.0).astype(np.float32)
    if transform == "raw":
        out = x.copy()
    elif transform == "binary":
        out = (x > 0).astype(np.float32)
    elif transform == "log1p":
        out = np.log1p(x.clip(lower=0.0)).astype(np.float32)
    elif transform == "fraction":
        denom = x.sum(axis=1).replace(0.0, np.nan)
        out = x.div(denom, axis=0).fillna(0.0).astype(np.float32)
    elif transform == "tfidf":
        binary = (x > 0).astype(np.float32)
        df = binary.sum(axis=0).replace(0.0, np.nan)
        idf = np.log((1.0 + x.shape[0]) / (1.0 + df)) + 1.0
        out = x.mul(idf, axis=1).fillna(0.0).astype(np.float32)
    elif transform == "rank":
        out = x.rank(axis=1, method="average", pct=True).fillna(0.0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported transform: {transform}")
    out.columns = [f"{prefix}__{transform}__{col}" for col in x.columns]
    return out


def top_variance(block: pd.DataFrame, n: int) -> pd.DataFrame:
    if block.shape[1] <= n:
        return block
    keep = block.var(axis=0).sort_values(ascending=False).head(n).index
    return block.loc[:, keep].copy()


def normalize_gene(value: object) -> str:
    text = str(value).upper().strip()
    if text in {"", "NAN", "NONE", "UNKNOWN"}:
        return "UNKNOWN"
    return text.replace(" ", "_")


def normalize_token(value: object, fallback: str = "unknown") -> str:
    text = str(value).lower().strip()
    if text in {"", "nan", "none"}:
        text = fallback
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or fallback


def pivot_counts(grouped: pd.DataFrame, index: list[str], patient_ids: list[str], prefix: str) -> pd.DataFrame:
    if grouped.empty:
        return pd.DataFrame(index=patient_ids)
    table = grouped.pivot_table(index="patient_id", columns=index, values="count", aggfunc="sum", fill_value=0.0)
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = [prefix + "__" + "__".join(str(part) for part in col) for col in table.columns]
    else:
        table.columns = [prefix + "__" + str(col) for col in table.columns]
    table = table.reindex(patient_ids).fillna(0.0).astype(np.float32)
    table.index = table.index.astype(str)
    return table


def collapse(frames: list[pd.DataFrame], group_cols: list[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=[*group_cols, "count"])
    merged = pd.concat(frames, ignore_index=True)
    return merged.groupby(group_cols, as_index=False, observed=True)["count"].sum()


def build_vaf_summary(vaf_rows: list[pd.DataFrame], patient_ids: list[str]) -> pd.DataFrame:
    if not vaf_rows:
        return pd.DataFrame(index=patient_ids)
    data = pd.concat(vaf_rows, ignore_index=True)
    grouped = data.groupby("patient_id")["vaf"]
    summary = pd.DataFrame(
        {
            "maf_vaf_mean": grouped.mean(),
            "maf_vaf_std": grouped.std().fillna(0.0),
            "maf_vaf_median": grouped.median(),
            "maf_vaf_q90": grouped.quantile(0.90),
            "maf_vaf_high_fraction": grouped.apply(lambda s: float(np.mean(s >= 0.35))),
            "maf_vaf_low_fraction": grouped.apply(lambda s: float(np.mean(s <= 0.10))),
        }
    )
    return summary.reindex(patient_ids).fillna(0.0).astype(np.float32)


def build_gene_coordinate_blocks(patient_ids: list[str]) -> dict[str, pd.DataFrame]:
    cache_meta = DATA_DIR / "gene_coordinate_cache_metadata.json"
    expected_names = [
        "gene_counts_top64",
        "gene_counts_top128",
        "gene_counts_top256",
        "gene_counts_top512",
        "gene_damaging_top64",
        "gene_damaging_top128",
        "gene_damaging_top256",
        "gene_damaging_top512",
        "pathway_counts",
        "pathway_damaging_counts",
        "pathway_impact_counts",
        "impact_counts",
        "variant_class_counts",
        "variant_type_counts",
        "consequence_counts_top128",
        "gene_pathway_detail",
        "vaf_summary",
    ]
    expected = {name: DATA_DIR / f"features_{name}.csv.gz" for name in expected_names}
    if cache_meta.exists() and all(path.exists() for path in expected.values()):
        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        if meta.get("target_gene_excluded") == TARGET_GENE_EXCLUSION:
            blocks = {}
            for name, path in expected.items():
                frame = pd.read_csv(path, index_col=0).fillna(0.0).astype(np.float32)
                frame.index = frame.index.astype(str)
                blocks[name] = frame
            return blocks

    maf_path = base.RAW_DIR / "mc3.v0.2.8.PUBLIC.maf.gz"
    usecols = [
        "Hugo_Symbol",
        "Tumor_Sample_Barcode",
        "Variant_Type",
        "Variant_Classification",
        "IMPACT",
        "Consequence",
        "t_alt_count",
        "t_depth",
    ]
    patient_set = set(patient_ids)
    gene_frames: list[pd.DataFrame] = []
    gene_damaging_frames: list[pd.DataFrame] = []
    pathway_frames: list[pd.DataFrame] = []
    pathway_damaging_frames: list[pd.DataFrame] = []
    pathway_impact_frames: list[pd.DataFrame] = []
    impact_frames: list[pd.DataFrame] = []
    variant_class_frames: list[pd.DataFrame] = []
    variant_type_frames: list[pd.DataFrame] = []
    consequence_frames: list[pd.DataFrame] = []
    pathway_detail_frames: list[pd.DataFrame] = []
    vaf_frames: list[pd.DataFrame] = []
    total_rows = 0
    kept_rows = 0

    print("Building MAF event-gene coordinate blocks", flush=True)
    for chunk in pd.read_csv(maf_path, sep="\t", usecols=usecols, dtype=str, chunksize=350_000):
        total_rows += len(chunk)
        chunk["patient_id"] = chunk["Tumor_Sample_Barcode"].astype(str).str[:12]
        chunk = chunk[chunk["patient_id"].isin(patient_set)].copy()
        chunk["gene"] = chunk["Hugo_Symbol"].map(normalize_gene)
        chunk = chunk[chunk["gene"] != TARGET_GENE_EXCLUSION].copy()
        kept_rows += len(chunk)
        if chunk.empty:
            continue
        chunk["variant_class"] = chunk["Variant_Classification"].map(normalize_token)
        chunk["variant_type"] = chunk["Variant_Type"].map(normalize_token)
        chunk["impact"] = chunk["IMPACT"].map(normalize_token)
        chunk["consequence"] = chunk["Consequence"].map(normalize_token)
        chunk["is_damaging"] = chunk["variant_class"].isin(DAMAGING_CLASSES) | chunk["impact"].isin({"high", "moderate"})
        gene_frames.append(chunk.groupby(["patient_id", "gene"], as_index=False, observed=True).size().rename(columns={"size": "count"}))
        damaging = chunk[chunk["is_damaging"]]
        if not damaging.empty:
            gene_damaging_frames.append(damaging.groupby(["patient_id", "gene"], as_index=False, observed=True).size().rename(columns={"size": "count"}))
        impact_frames.append(chunk.groupby(["patient_id", "impact"], as_index=False, observed=True).size().rename(columns={"size": "count"}))
        variant_class_frames.append(chunk.groupby(["patient_id", "variant_class"], as_index=False, observed=True).size().rename(columns={"size": "count"}))
        variant_type_frames.append(chunk.groupby(["patient_id", "variant_type"], as_index=False, observed=True).size().rename(columns={"size": "count"}))
        consequence_frames.append(chunk.groupby(["patient_id", "consequence"], as_index=False, observed=True).size().rename(columns={"size": "count"}))

        pathway_rows = []
        for pathway, genes in PATHWAY_SETS.items():
            mask = chunk["gene"].isin(genes)
            if mask.any():
                temp = chunk.loc[mask, ["patient_id", "gene", "impact", "is_damaging"]].copy()
                temp["pathway"] = pathway
                pathway_rows.append(temp)
        if pathway_rows:
            pathway_df = pd.concat(pathway_rows, ignore_index=True)
            pathway_frames.append(pathway_df.groupby(["patient_id", "pathway"], as_index=False, observed=True).size().rename(columns={"size": "count"}))
            pathway_impact_frames.append(pathway_df.groupby(["patient_id", "pathway", "impact"], as_index=False, observed=True).size().rename(columns={"size": "count"}))
            pathway_detail_frames.append(pathway_df.groupby(["patient_id", "pathway", "gene"], as_index=False, observed=True).size().rename(columns={"size": "count"}))
            pathway_damaging = pathway_df[pathway_df["is_damaging"]]
            if not pathway_damaging.empty:
                pathway_damaging_frames.append(pathway_damaging.groupby(["patient_id", "pathway"], as_index=False, observed=True).size().rename(columns={"size": "count"}))

        alt = pd.to_numeric(chunk["t_alt_count"], errors="coerce")
        depth = pd.to_numeric(chunk["t_depth"], errors="coerce")
        vaf = (alt / depth.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        vaf_df = pd.DataFrame({"patient_id": chunk["patient_id"].to_numpy(), "vaf": vaf.to_numpy(dtype=float)})
        vaf_df = vaf_df.dropna()
        if not vaf_df.empty:
            vaf_frames.append(vaf_df)

    gene_all = pivot_counts(collapse(gene_frames, ["patient_id", "gene"]), ["gene"], patient_ids, "maf_gene")
    gene_damaging_all = pivot_counts(collapse(gene_damaging_frames, ["patient_id", "gene"]), ["gene"], patient_ids, "maf_gene_damaging")
    consequence_all = pivot_counts(collapse(consequence_frames, ["patient_id", "consequence"]), ["consequence"], patient_ids, "maf_consequence")

    blocks: dict[str, pd.DataFrame] = {
        "pathway_counts": pivot_counts(collapse(pathway_frames, ["patient_id", "pathway"]), ["pathway"], patient_ids, "maf_pathway"),
        "pathway_damaging_counts": pivot_counts(collapse(pathway_damaging_frames, ["patient_id", "pathway"]), ["pathway"], patient_ids, "maf_pathway_damaging"),
        "pathway_impact_counts": pivot_counts(collapse(pathway_impact_frames, ["patient_id", "pathway", "impact"]), ["pathway", "impact"], patient_ids, "maf_pathway_impact"),
        "impact_counts": pivot_counts(collapse(impact_frames, ["patient_id", "impact"]), ["impact"], patient_ids, "maf_impact"),
        "variant_class_counts": pivot_counts(collapse(variant_class_frames, ["patient_id", "variant_class"]), ["variant_class"], patient_ids, "maf_variant_class"),
        "variant_type_counts": pivot_counts(collapse(variant_type_frames, ["patient_id", "variant_type"]), ["variant_type"], patient_ids, "maf_variant_type"),
        "gene_pathway_detail": pivot_counts(collapse(pathway_detail_frames, ["patient_id", "pathway", "gene"]), ["pathway", "gene"], patient_ids, "maf_pathway_gene"),
        "vaf_summary": build_vaf_summary(vaf_frames, patient_ids),
    }
    for budget in (64, 128, 256, 512):
        blocks[f"gene_counts_top{budget}"] = top_variance(gene_all, budget).astype(np.float32)
        blocks[f"gene_damaging_top{budget}"] = top_variance(gene_damaging_all, budget).astype(np.float32)
    blocks["consequence_counts_top128"] = top_variance(consequence_all, 128).astype(np.float32)

    for name, frame in blocks.items():
        frame.to_csv(expected[name])
    cache_meta.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "maf": str(maf_path),
                "total_rows": int(total_rows),
                "kept_rows": int(kept_rows),
                "patient_count": int(len(patient_ids)),
                "target_gene_excluded": TARGET_GENE_EXCLUSION,
                "pathway_sets": {name: sorted(genes) for name, genes in PATHWAY_SETS.items()},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return blocks


def load_reference_locus_blocks(patient_ids: list[str]) -> dict[str, pd.DataFrame]:
    names = [
        "locus_chrom_counts",
        "locus_chrom_modality_counts",
        "locus_variant_class_counts",
        "locus_mb_top512",
        "locus_density_summary",
    ]
    meta_path = REFERENCE_DATA / "locus_topography_cache_metadata.json"
    expected_paths = [REFERENCE_DATA / f"features_{name}.csv.gz" for name in names]
    if not meta_path.exists() or any(not path.exists() for path in expected_paths):
        from utils.maf_features import build_locus_topography_blocks

        build_locus_topography_blocks(
            patient_ids,
            base.RAW_DIR / "mc3.v0.2.8.PUBLIC.maf.gz",
            REFERENCE_DATA,
            target_gene=TARGET_GENE_EXCLUSION,
        )
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("target_gene_excluded") != TARGET_GENE_EXCLUSION:
            raise RuntimeError("Reference locus cache was not built with the required target-gene exclusion.")
    out: dict[str, pd.DataFrame] = {}
    for name in names:
        path = REFERENCE_DATA / f"features_{name}.csv.gz"
        if not path.exists():
            continue
        frame = pd.read_csv(path, index_col=0).fillna(0.0).astype(np.float32)
        frame.index = frame.index.astype(str)
        out[name] = frame.reindex(patient_ids).fillna(0.0).astype(np.float32)
    return out


def add_low_rank_blocks(blocks: dict[str, pd.DataFrame]) -> None:
    for source in ["gene_counts_top512_log1p", "gene_damaging_top512_log1p"]:
        if source not in blocks:
            continue
        x = blocks[source].clip(lower=0.0).to_numpy(dtype=np.float32)
        if x.shape[1] < 4:
            continue
        for n_components in (16, 32, 64):
            if n_components >= min(x.shape):
                continue
            svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
            arr = svd.fit_transform(x).astype(np.float32)
            frame = pd.DataFrame(
                arr,
                index=blocks[source].index,
                columns=[f"{source}_svd{n_components}__{i:02d}" for i in range(n_components)],
            )
            blocks[f"{source}_svd{n_components}"] = frame


def build_transformed_blocks(raw_blocks: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    blocks: dict[str, pd.DataFrame] = {}
    for name, frame in raw_blocks.items():
        blocks[f"{name}_raw"] = transform_counts(frame, "raw", name)
        blocks[f"{name}_binary"] = transform_counts(frame, "binary", name)
        blocks[f"{name}_log1p"] = transform_counts(frame, "log1p", name)
        if name not in {"vaf_summary"}:
            blocks[f"{name}_fraction"] = transform_counts(frame, "fraction", name)
            blocks[f"{name}_tfidf"] = transform_counts(frame, "tfidf", name)
    add_low_rank_blocks(blocks)
    if "pathway_counts_binary" in blocks and "impact_counts_binary" in blocks:
        left = blocks["pathway_counts_binary"]
        right = blocks["impact_counts_binary"]
        pieces = {}
        for lcol in left.columns:
            for rcol in right.columns:
                pieces[f"pathway_x_impact__{lcol}__{rcol}"] = left[lcol] * right[rcol]
        blocks["pathway_impact_presence_interactions"] = pd.DataFrame(pieces, index=left.index).astype(np.float32)
    if "gene_counts_top256_binary" in blocks and "vaf_summary_raw" in blocks:
        gene = blocks["gene_counts_top256_binary"]
        vaf = blocks["vaf_summary_raw"]
        pieces = {}
        for vcol in vaf.columns:
            if vcol.endswith(("mean", "high_fraction", "low_fraction")):
                pieces.update({f"gene_presence_x_{vcol}__{gcol}": gene[gcol] * vaf[vcol] for gcol in gene.columns})
        blocks["gene_presence_vaf_interactions_top256"] = pd.DataFrame(pieces, index=gene.index).astype(np.float32)
    return blocks


def make_candidate_specs() -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []

    def add(model_id: str, family: str, blocks: list[str], notes: str) -> None:
        specs.append(CandidateSpec(model_id, family, tuple(blocks), ";".join(blocks), notes))

    for budget in (64, 128, 256, 512):
        for transform in ("binary", "log1p", "fraction", "tfidf"):
            add(
                f"id_plus_gene_top{budget}_{transform}",
                "exact_identity_plus_gene_coordinate_cells",
                [f"gene_counts_top{budget}_{transform}"],
                "Exact coordinate identity plus gene coordinate occupancy cells.",
            )
            add(
                f"id_plus_damaging_gene_top{budget}_{transform}",
                "exact_identity_plus_gene_coordinate_cells",
                [f"gene_damaging_top{budget}_{transform}"],
                "Exact coordinate identity plus damaging-gene coordinate occupancy cells.",
            )
    for block in [
        "pathway_counts_binary",
        "pathway_counts_log1p",
        "pathway_damaging_counts_binary",
        "pathway_damaging_counts_log1p",
        "pathway_impact_counts_binary",
        "pathway_impact_counts_log1p",
        "impact_counts_binary",
        "variant_class_counts_binary",
        "variant_type_counts_binary",
        "consequence_counts_top128_binary",
        "gene_pathway_detail_binary",
        "gene_pathway_detail_log1p",
        "vaf_summary_raw",
        "pathway_impact_presence_interactions",
    ]:
        add(f"id_plus_{block}", "exact_identity_plus_event_annotation_cells", [block], "Exact coordinate identity plus MAF event annotation coordinate cells.")
    for source in ("gene_counts_top512", "gene_damaging_top512"):
        for n_components in (16, 32, 64):
            add(
                f"id_plus_{source}_svd{n_components}",
                "exact_identity_plus_self_supervised_gene_embedding",
                [f"{source}_log1p_svd{n_components}"],
                "Exact coordinate identity plus unsupervised gene-coordinate SVD features.",
            )

    add("id_plus_gene_top256_binary_and_pathway", "gene_pathway_combo", ["gene_counts_top256_binary", "pathway_counts_binary", "pathway_damaging_counts_binary"], "Gene occupancy plus pathway occupancy.")
    add("id_plus_gene_top512_binary_and_pathway", "gene_pathway_combo", ["gene_counts_top512_binary", "pathway_counts_binary", "pathway_damaging_counts_binary"], "Gene occupancy plus pathway occupancy.")
    add("id_plus_damaging_gene_top256_binary_and_pathway", "gene_pathway_combo", ["gene_damaging_top256_binary", "pathway_damaging_counts_binary", "pathway_impact_counts_binary"], "Damaging-gene occupancy plus pathway impact occupancy.")
    add("id_plus_damaging_gene_top512_binary_and_pathway", "gene_pathway_combo", ["gene_damaging_top512_binary", "pathway_damaging_counts_binary", "pathway_impact_counts_binary"], "Damaging-gene occupancy plus pathway impact occupancy.")
    add("id_plus_gene_top512_log_tfidf_svd", "gene_multimap_combo", ["gene_counts_top512_log1p", "gene_counts_top512_tfidf", "gene_counts_top512_log1p_svd32"], "Gene coordinate count, TF-IDF, and low-rank maps.")
    add("id_plus_damaging_gene_top512_log_tfidf_svd", "gene_multimap_combo", ["gene_damaging_top512_log1p", "gene_damaging_top512_tfidf", "gene_damaging_top512_log1p_svd32"], "Damaging-gene count, TF-IDF, and low-rank maps.")
    add("id_plus_pathway_impact_gene_detail", "pathway_impact_combo", ["pathway_counts_binary", "pathway_damaging_counts_binary", "pathway_impact_counts_binary", "gene_pathway_detail_binary"], "Pathway, damaging pathway, and pathway-gene detail cells.")
    add("id_plus_gene_vaf_interactions", "gene_vaf_combo", ["gene_counts_top256_binary", "vaf_summary_raw", "gene_presence_vaf_interactions_top256"], "Gene presence coordinate cells crossed with allele-fraction summaries.")
    add("id_plus_full_gene_annotation_compact", "full_compact_gene_event_stack", ["gene_counts_top256_binary", "gene_damaging_top256_binary", "pathway_impact_counts_binary", "variant_class_counts_binary", "impact_counts_binary", "vaf_summary_raw"], "Compact event-gene coordinate stack.")
    add("id_plus_full_gene_annotation_large", "full_large_gene_event_stack", ["gene_counts_top512_binary", "gene_damaging_top512_binary", "pathway_impact_counts_binary", "gene_pathway_detail_binary", "variant_class_counts_binary", "consequence_counts_top128_binary", "impact_counts_binary", "vaf_summary_raw", "gene_counts_top512_log1p_svd64"], "Large event-gene coordinate stack.")
    add("id_plus_damaging_gene_tfidf_svd_vaf", "best_signal_expansion_stack", ["gene_damaging_top512_log1p", "gene_damaging_top512_tfidf", "gene_damaging_top512_log1p_svd32", "vaf_summary_raw"], "Damaging-gene coordinate maps plus allele-fraction summaries.")
    add("id_plus_damaging_gene_tfidf_svd_pathway_vaf", "best_signal_expansion_stack", ["gene_damaging_top512_log1p", "gene_damaging_top512_tfidf", "gene_damaging_top512_log1p_svd32", "pathway_impact_counts_binary", "gene_pathway_detail_binary", "vaf_summary_raw"], "Damaging-gene maps plus pathway detail and allele-fraction summaries.")
    add("id_plus_full_gene_large_locus_mb", "gene_locus_coordinate_stack", ["gene_counts_top512_binary", "gene_damaging_top512_binary", "pathway_impact_counts_binary", "gene_pathway_detail_binary", "locus_mb_top512_binary", "locus_density_summary_raw", "vaf_summary_raw"], "Gene, pathway, megabase-locus, and allele-fraction coordinate cells.")
    add("id_plus_full_gene_large_locus_all", "gene_locus_coordinate_stack", ["gene_counts_top512_binary", "gene_damaging_top512_binary", "pathway_impact_counts_binary", "gene_pathway_detail_binary", "variant_class_counts_binary", "consequence_counts_top128_binary", "impact_counts_binary", "locus_chrom_counts_raw", "locus_chrom_modality_counts_raw", "locus_variant_class_counts_binary", "locus_mb_top512_binary", "locus_density_summary_raw", "vaf_summary_raw", "gene_counts_top512_log1p_svd64"], "Large event-gene stack plus locus-topography coordinate cells.")
    add("id_plus_best_gene_locus_vaf_stack", "best_signal_expansion_stack", ["gene_damaging_top512_log1p", "gene_damaging_top512_tfidf", "gene_damaging_top512_log1p_svd32", "gene_counts_top512_log1p_svd64", "pathway_impact_counts_binary", "gene_pathway_detail_binary", "locus_mb_top512_binary", "locus_density_summary_raw", "vaf_summary_raw"], "Combined best-performing gene, pathway, locus, and allele-fraction maps.")
    add("id_plus_best_gene_locus_multiscale_stack", "best_signal_expansion_stack", ["gene_counts_top512_binary", "gene_damaging_top512_binary", "gene_damaging_top512_log1p_svd16", "gene_damaging_top512_log1p_svd32", "pathway_counts_binary", "pathway_damaging_counts_binary", "pathway_impact_counts_binary", "gene_pathway_detail_binary", "locus_chrom_modality_counts_raw", "locus_mb_top512_binary", "locus_density_summary_raw", "vaf_summary_raw"], "Multiscale gene, pathway, locus, and allele-fraction coordinate stack.")
    add("maf_only_best_gene_locus_multiscale_stack", "maf_event_gene_locus_stack_only", ["gene_counts_top512_binary", "gene_damaging_top512_binary", "gene_damaging_top512_log1p_svd16", "gene_damaging_top512_log1p_svd32", "pathway_counts_binary", "pathway_damaging_counts_binary", "pathway_impact_counts_binary", "gene_pathway_detail_binary", "locus_chrom_modality_counts_raw", "locus_mb_top512_binary", "locus_density_summary_raw", "vaf_summary_raw"], "MAF-only multiscale gene, pathway, locus, and allele-fraction coordinate stack without exact SBS96+ID83 identity cells.")
    return specs


def load_discovery_endpoints(patients: pd.Index) -> list[base.Endpoint]:
    clinical = {endpoint.name: endpoint for endpoint in base.load_mc3_clinical_endpoints()}
    hrd = {endpoint.name: endpoint for endpoint in base.load_hrd_endpoints()}
    kmt2c = base.load_kmt2c_endpoint(patients)
    endpoint_map = {**clinical, **hrd, kmt2c.name: kmt2c}
    return [endpoint_map[name] for name in DISCOVERY_ENDPOINTS]


def build_candidate_frame(spec: CandidateSpec, identity: pd.DataFrame, blocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pieces = [] if spec.model_id.startswith("maf_only_") else [identity]
    for block in spec.blocks:
        pieces.append(blocks[block])
    return pd.concat(pieces, axis=1).fillna(0.0).astype(np.float32)


def run_standard_metrics(endpoints: list[base.Endpoint], frame: pd.DataFrame, *, folds: int, repeats: int, n_estimators: int, tree_method: str) -> tuple[pd.DataFrame, dict[str, float]]:
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
            seed=stable_seed("rapid_screen", endpoint.name),
        )
        metric.update({"model_id": STANDARD_BASELINE, "delta_vs_standard": 0.0})
        rows.append(metric)
        scores[endpoint.name] = float(metric["score"])
    return pd.DataFrame(rows), scores


def baseline_reproducibility_audit(current_standard: pd.DataFrame, exact_metrics: pd.DataFrame) -> pd.DataFrame:
    prior_path = REFERENCE_DATA / "reconstruction_audit.csv"
    prior = pd.read_csv(prior_path) if prior_path.exists() else pd.DataFrame()
    rows = []
    for _, row in current_standard.iterrows():
        endpoint = row["endpoint"]
        exact = exact_metrics[exact_metrics["endpoint"] == endpoint]
        prior_score = np.nan
        if not prior.empty:
            m = prior[prior["endpoint"] == endpoint]
            if len(m):
                prior_score = float(m["current_standard_score"].iloc[0])
        exact_score = float(exact["score"].iloc[0]) if len(exact) else np.nan
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


def screen_candidates(
    specs: list[CandidateSpec],
    endpoints: list[base.Endpoint],
    standard_scores: dict[str, float],
    identity: pd.DataFrame,
    blocks: dict[str, pd.DataFrame],
    *,
    folds: int,
    n_estimators: int,
    tree_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for i, spec in enumerate(specs, start=1):
        print(f"Screening {i}/{len(specs)}: {spec.model_id}", flush=True)
        t0 = time.perf_counter()
        frame = build_candidate_frame(spec, identity, blocks)
        feature_rows.append(
            {
                "model_id": spec.model_id,
                "n_features": int(frame.shape[1]),
                "n_samples": int(frame.shape[0]),
                "matrix_nonzero_fraction": float((frame.to_numpy(dtype=np.float32) != 0).mean()),
                "contains_exact_identity": not spec.model_id.startswith("maf_only_"),
                "sbs96_reconstruction_r2": 0.0 if spec.model_id.startswith("maf_only_") else 1.0,
                "id83_reconstruction_r2": 0.0 if spec.model_id.startswith("maf_only_") else 1.0,
            }
        )
        manifest_rows.append(
            {
                "model_id": spec.model_id,
                "family": spec.family,
                "feature_blocks": ";".join(spec.blocks),
                "uses_standard_identity_channels": False,
                "uses_exact_coordinate_identity_cells": not spec.model_id.startswith("maf_only_"),
                "uses_endpoint_labels_in_features": False,
                "uses_posthoc_ensemble": False,
                "uses_endpoint_specific_weights": False,
                "target_gene_excluded_from_features": TARGET_GENE_EXCLUSION,
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
                    "candidate_runtime_seconds": time.perf_counter() - t0,
                }
            )
            rows.append(metric)
    return pd.DataFrame(rows), pd.DataFrame(feature_rows), pd.DataFrame(manifest_rows)


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


def promote_finalists(leaderboard: pd.DataFrame, max_finalists: int = 3) -> list[str]:
    passed = leaderboard[leaderboard["promotion_pass"]].copy()
    return passed.sort_values("mean_delta", ascending=False)["model_id"].head(max_finalists).astype(str).tolist()


def paired_bootstrap_delta(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, task: str, classes: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            _, score_a, _ = base.score_predictions(y[idx], pred_a[idx], task, classes)
            _, score_b, _ = base.score_predictions(y[idx], pred_b[idx], task, classes)
            if np.isfinite(score_b - score_a):
                deltas.append(score_b - score_a)
        except Exception:
            continue
    if not deltas:
        return np.nan, np.nan, np.nan
    arr = np.asarray(deltas)
    p = 2.0 * min(float(np.mean(arr <= 0.0)), float(np.mean(arr >= 0.0)))
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), min(p, 1.0)


def bh_qvalues(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float).to_numpy()
    order = np.argsort(p)
    q = np.empty_like(p, dtype=float)
    prev = 1.0
    m = len(p)
    for rank, idx in enumerate(order[::-1], start=1):
        original_rank = m - rank + 1
        val = p[idx] * m / original_rank
        prev = min(prev, val)
        q[idx] = min(prev, 1.0)
    return pd.Series(q, index=p_values.index)


def run_confirmation(
    finalists: list[str],
    specs: list[CandidateSpec],
    endpoints: list[base.Endpoint],
    standard_frame: pd.DataFrame,
    identity: pd.DataFrame,
    blocks: dict[str, pd.DataFrame],
    *,
    folds: int,
    repeats: int,
    n_estimators: int,
    tree_method: str,
    bootstrap: int,
) -> pd.DataFrame:
    if not finalists:
        return pd.DataFrame()
    spec_map = {spec.model_id: spec for spec in specs}
    checkpoint_name = "confirmation_checkpoint_" + "_".join(finalists)[:120].replace("/", "_") + ".csv"
    checkpoint_path = DATA_DIR / checkpoint_name
    completed = read_completed_keys(checkpoint_path, ["model_id", "endpoint"])
    rows = []
    for endpoint in endpoints:
        standard_metric, standard_pred = base.run_endpoint_model(
            endpoint,
            standard_frame,
            folds=folds,
            repeats=repeats,
            n_estimators=n_estimators,
            tree_method=tree_method,
            seed=stable_seed("confirmation", endpoint.name),
        )
        for finalist in finalists:
            key = (str(finalist), str(endpoint.name))
            if key in completed:
                print(f"[checkpoint] skip {finalist}/{endpoint.name}", flush=True)
                continue
            frame = build_candidate_frame(spec_map[finalist], identity, blocks)
            metric, pred = base.run_endpoint_model(
                endpoint,
                frame,
                folds=folds,
                repeats=repeats,
                n_estimators=n_estimators,
                tree_method=tree_method,
                seed=stable_seed("confirmation", endpoint.name),
            )
            common = endpoint.y.index.intersection(standard_pred["patient_id"]).intersection(pred["patient_id"])
            std_aligned = standard_pred.set_index("patient_id").loc[common]
            pred_aligned = pred.set_index("patient_id").loc[common]
            y, classes = base.encode_target(endpoint.y.loc[common], endpoint.task)
            if endpoint.task == "regression":
                std_arr = std_aligned["pred_value"].to_numpy(dtype=float)
                cand_arr = pred_aligned["pred_value"].to_numpy(dtype=float)
            else:
                pred_cols = [col for col in pred_aligned.columns if col.startswith("pred_class_")]
                std_arr = std_aligned[pred_cols].to_numpy(dtype=float)
                cand_arr = pred_aligned[pred_cols].to_numpy(dtype=float)
            ci_low, ci_high, p_value = paired_bootstrap_delta(y, std_arr, cand_arr, endpoint.task, classes, n_boot=bootstrap, seed=stable_seed("bootstrap", finalist, endpoint.name))
            row = {
                "model_id": finalist,
                "endpoint": endpoint.name,
                "endpoint_family": endpoint.family,
                "metric": metric["metric"],
                "standard_score": standard_metric["score"],
                "candidate_score": metric["score"],
                "delta_vs_standard": float(metric["score"] - standard_metric["score"]),
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "p_value": p_value,
                "n": metric["n"],
                "folds": folds,
                "repeats": repeats,
                "n_estimators": n_estimators,
            }
            rows.append(row)
            merge_checkpoint_rows(
                checkpoint_path,
                [row],
                key_columns=["model_id", "endpoint"],
                sort_columns=["model_id", "endpoint"],
            )
            completed.add(key)
            print(f"[checkpoint] wrote {finalist}/{endpoint.name}", flush=True)
    out = pd.read_csv(checkpoint_path) if checkpoint_path.exists() else pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = out.groupby("endpoint_family", group_keys=False)["p_value"].apply(bh_qvalues)
    return out


def html_table(df: pd.DataFrame, title: str, footnote: str) -> str:
    style = """
    <style>
    body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#111}
    table{border-collapse:collapse;font-size:13px;line-height:1.35}
    caption{caption-side:top;text-align:left;font-weight:700;margin-bottom:8px}
    th,td{border-bottom:1px solid #d0d0d0;padding:6px 8px;text-align:right;vertical-align:top}
    th:first-child,td:first-child{text-align:left}
    th{border-top:1.5px solid #111;border-bottom:1px solid #111;font-weight:700}
    tfoot td{border-bottom:0;text-align:left;font-size:12px;color:#333;padding-top:10px}
    </style>
    """
    header = "".join(f"<th>{escape(str(col))}</th>" for col in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{escape(str(value))}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    return f"<!doctype html><html><head><meta charset='utf-8'>{style}</head><body><table><caption>{escape(title)}</caption><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody><tfoot><tr><td colspan='{len(df.columns)}'>{escape(footnote)}</td></tr></tfoot></table></body></html>"


def format_for_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.loc[:, columns].copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    return out


def write_tables(leaderboard: pd.DataFrame, results: pd.DataFrame, audit: pd.DataFrame, confirmation: pd.DataFrame) -> None:
    (TABLE_DIR / "table1_rapid_screen_leaderboard.html").write_text(
        html_table(
            format_for_table(leaderboard.head(40), ["model_id", "candidate_family", "n_endpoints", "mean_delta", "min_delta", "max_delta", "endpoint_gains_ge_0p03", "endpoint_losses_lt_neg_0p02", "promotion_pass", "n_features"]),
            "Rapid-screen candidate leaderboard",
            "Promotion required mean delta >=0.02, at least three endpoint gains >=0.03, and no endpoint loss below -0.02.",
        ),
        encoding="utf-8",
    )
    endpoint_table = results[~results["model_id"].isin([STANDARD_BASELINE, EXACT_CONTROL])].copy()
    keep_models = leaderboard.head(8)["model_id"].astype(str).tolist()
    endpoint_table = endpoint_table[endpoint_table["model_id"].isin(keep_models)]
    (TABLE_DIR / "table2_endpoint_results.html").write_text(
        html_table(
            format_for_table(endpoint_table.sort_values(["endpoint", "delta_vs_standard"], ascending=[True, False]), ["model_id", "endpoint", "metric", "score", "delta_vs_standard", "n", "n_features"]),
            "Endpoint-level rapid-screen results",
            "Scores are out-of-fold metrics from paired three-fold XGBoost screening.",
        ),
        encoding="utf-8",
    )
    (TABLE_DIR / "table3_reconstruction_audit.html").write_text(
        html_table(
            format_for_table(audit, ["endpoint", "current_standard_score", "current_exact_identity_score", "exact_minus_standard", "prior_standard_score", "current_minus_prior"]),
            "Baseline and exact-identity audit",
            "Exact coordinate identity must reproduce Standard SBS96+ID83 under identical folds and model settings.",
        ),
        encoding="utf-8",
    )
    if confirmation.empty:
        confirmation_display = pd.DataFrame({"result": ["No candidate passed the rapid-screen promotion gate."]})
    else:
        confirmation_display = format_for_table(confirmation, ["model_id", "endpoint", "metric", "standard_score", "candidate_score", "delta_vs_standard", "delta_ci_low", "delta_ci_high", "p_value", "q_value", "n"])
    (TABLE_DIR / "table4_focused_confirmation.html").write_text(
        html_table(
            confirmation_display,
            "Focused confirmation results",
            "Confirmation uses repeated paired cross-validation and paired bootstrap inference for promoted finalists.",
        ),
        encoding="utf-8",
    )


def write_readme(metadata: dict[str, object], best: dict[str, object], finalists: list[str]) -> None:
    text = f"""# XGBoost MAF Event-Gene Coordinate Search

## Research Question

This experiment tests whether MAF event-level coordinate cells for gene, pathway, predicted impact, consequence, and allele-fraction summaries add biological prediction signal beyond an exact UGA coordinate-identity reconstruction of SBS96+ID83.

## Methods

All candidates include exact UGA coordinate-identity cells that reconstruct SBS96+ID83 with R2=1.0. Candidate-specific blocks are built from the MC3 MAF by assigning events to coordinate cells defined by gene symbol, damaging-gene status, pathway membership, predicted impact, variant consequence, variant type, and tumor allele-fraction summaries. KMT2C events are excluded before any event-gene feature construction. The unchanged Standard SBS96+ID83 matrix is retained as the comparator. Discovery endpoints were smoking ever, top-10 cancer type, LUAD KMT2C mutation status, and HRD score thresholds 24, 33, and 42. XGBoost used paired 3-fold cross-validation, fixed seeds, identical patients, identical labels, and the same model settings for Standard and every candidate.

## Key Numerical Findings

The rapid screen evaluated {metadata["n_candidates"]} candidates. The best rapid-screen model was `{best.get("model_id", "none")}` with mean delta {best.get("mean_delta", float("nan")):.4f}, minimum delta {best.get("min_delta", float("nan")):.4f}, maximum delta {best.get("max_delta", float("nan")):.4f}, and {int(best.get("endpoint_gains_ge_0p03", 0))} endpoint gains of at least 0.03. Promoted finalists: {", ".join(finalists) if finalists else "none"}.

## File Inventory

- `data/model_manifest.csv`: feature definitions fixed before endpoint fitting.
- `data/feature_dimension_audit.csv`: feature dimensions and identity-reconstruction audit.
- `data/reconstruction_audit.csv`: exact identity and baseline reproducibility checks.
- `data/rapid_screen_results.csv`: endpoint-level rapid-screen metrics.
- `data/rapid_screen_leaderboard.csv`: candidate-level rapid-screen summary and promotion calls.
- `data/focused_confirmation_results.csv`: repeated-CV paired bootstrap validation for promoted finalists.
- `tables/table1_rapid_screen_leaderboard.html`: rapid-screen candidate ranking.
- `tables/table2_endpoint_results.html`: endpoint-level scores for leading candidates.
- `tables/table3_reconstruction_audit.html`: baseline and exact-identity validation.
- `tables/table4_focused_confirmation.html`: repeated-CV confirmation for promoted finalists.
- `code/run_xgboost_maf_event_gene_coordinate_search.py`: complete runner.

## Reproducibility

Date executed: {metadata["completed_utc"]}. Random seed: `{metadata["random_seed"]}`. Python: `{metadata["python_version"]}`. Package versions: pandas `{metadata["package_versions"]["pandas"]}`, numpy `{metadata["package_versions"]["numpy"]}`, scipy `{metadata["package_versions"]["scipy"]}`, scikit-learn `{metadata["package_versions"]["sklearn"]}`, xgboost `{metadata["package_versions"]["xgboost"]}`. Tree method requested: `{metadata["tree_method"]}`.
"""
    (EXPERIMENT_ROOT / "README.md").write_text(text, encoding="utf-8")


def update_ledger(metadata: dict[str, object], best: dict[str, object], finalists: list[str]) -> None:
    ledger = EXPERIMENTS_ROOT / "EXPERIMENT_LEDGER.md"
    if not ledger.exists():
        return
    text = ledger.read_text(encoding="utf-8")
    exp_name = EXPERIMENT_ROOT.name
    if exp_name in text:
        return
    row = (
        f"| `{exp_name}` | {metadata['completed_utc']} | {metadata['runtime_seconds']:.1f} s | Yes | "
        "XGBoost-only MAF event-gene coordinate search | MC3 MAF event-gene coordinate cells plus exact SBS96+ID83 coordinate identity | "
        "Discovery endpoints: smoking ever, top-10 cancer type, LUAD KMT2C, HRD >=24, >=33, >=42 | "
        f"Best rapid-screen model `{best.get('model_id', 'none')}` mean delta {best.get('mean_delta', float('nan')):.4f}; "
        f"max delta {best.get('max_delta', float('nan')):.4f}; finalists {', '.join(finalists) if finalists else 'none'} | "
        "Exploratory XGBoost-only search; KMT2C events excluded from feature construction; candidates retain exact identity reconstruction. |\n"
    )
    insert_after = "| `2026_05_16_xgboost_coordinate_feature_map_search`"
    lines = text.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if line.startswith(insert_after):
            lines.insert(idx + 1, row)
            ledger.write_text("".join(lines), encoding="utf-8")
            return
    lines.append(row)
    ledger.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree-method", default="gpu_hist")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--xgb-estimators", type=int, default=80)
    parser.add_argument("--confirmation-folds", type=int, default=5)
    parser.add_argument("--confirmation-repeats", type=int, default=3)
    parser.add_argument("--confirmation-estimators", type=int, default=160)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--skip-confirmation", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    started = time.perf_counter()
    standard_sbs, standard_id, standard_sbs_id, _burden = base.load_feature_matrices()
    patient_ids = standard_sbs_id.index.astype(str).tolist()
    endpoints = load_discovery_endpoints(standard_sbs_id.index)
    identity = exact_identity_frame(standard_sbs_id)
    raw_blocks = build_gene_coordinate_blocks(patient_ids)
    raw_blocks.update(load_reference_locus_blocks(patient_ids))
    blocks = build_transformed_blocks(raw_blocks)
    specs = make_candidate_specs()
    if args.max_candidates:
        specs = specs[: int(args.max_candidates)]
    print(f"Generated {len(specs)} candidate representations", flush=True)

    standard_metrics, standard_scores = run_standard_metrics(
        endpoints,
        standard_sbs_id.astype(np.float32),
        folds=args.folds,
        repeats=1,
        n_estimators=args.xgb_estimators,
        tree_method=args.tree_method,
    )
    exact_rows = []
    for endpoint in endpoints:
        metric, _ = base.run_endpoint_model(
            endpoint,
            identity,
            folds=args.folds,
            repeats=1,
            n_estimators=args.xgb_estimators,
            tree_method=args.tree_method,
            seed=stable_seed("rapid_screen", endpoint.name),
        )
        metric.update({"model_id": EXACT_CONTROL, "delta_vs_standard": float(metric["score"] - standard_scores[endpoint.name])})
        exact_rows.append(metric)
    exact_metrics = pd.DataFrame(exact_rows)
    audit = baseline_reproducibility_audit(standard_metrics, exact_metrics)

    screen_metrics, feature_audit, manifest = screen_candidates(
        specs,
        endpoints,
        standard_scores,
        identity,
        blocks,
        folds=args.folds,
        n_estimators=args.xgb_estimators,
        tree_method=args.tree_method,
    )
    all_metrics = pd.concat([standard_metrics, exact_metrics, screen_metrics], ignore_index=True)
    leaderboard = summarize_screen(screen_metrics)
    finalists = promote_finalists(leaderboard)
    if args.skip_confirmation:
        confirmation = pd.DataFrame()
    else:
        confirmation = run_confirmation(
            finalists,
            specs,
            endpoints,
            standard_sbs_id.astype(np.float32),
            identity,
            blocks,
            folds=args.confirmation_folds,
            repeats=args.confirmation_repeats,
            n_estimators=args.confirmation_estimators,
            tree_method=args.tree_method,
            bootstrap=args.bootstrap,
        )

    feature_audit.to_csv(DATA_DIR / "feature_dimension_audit.csv", index=False)
    manifest["screen_status"] = manifest["model_id"].map(leaderboard.set_index("model_id")["promotion_pass"]).map({True: "promoted_to_confirmation", False: "rejected_rapid_screen"})
    manifest.to_csv(DATA_DIR / "model_manifest.csv", index=False)
    audit.to_csv(DATA_DIR / "reconstruction_audit.csv", index=False)
    all_metrics.to_csv(DATA_DIR / "rapid_screen_results.csv", index=False)
    leaderboard.to_csv(DATA_DIR / "rapid_screen_leaderboard.csv", index=False)
    confirmation.to_csv(DATA_DIR / "focused_confirmation_results.csv", index=False)
    write_tables(leaderboard, all_metrics, audit, confirmation)

    completed_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    metadata = {
        "experiment": EXPERIMENT_ROOT.name,
        "completed_local": datetime.now().replace(microsecond=0).isoformat(),
        "completed_utc": completed_utc,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "random_seed": RANDOM_SEED,
        "tree_method": args.tree_method,
        "folds": args.folds,
        "xgb_estimators": args.xgb_estimators,
        "confirmation_folds": args.confirmation_folds,
        "confirmation_repeats": args.confirmation_repeats,
        "confirmation_estimators": args.confirmation_estimators,
        "bootstrap": args.bootstrap,
        "n_candidates": len(specs),
        "finalists": finalists,
        "n_finalists": len(finalists),
        "python_version": platform.python_version(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    (DATA_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    write_readme(metadata, best, finalists)
    if not args.skip_ledger:
        update_ledger(metadata, best, finalists)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
