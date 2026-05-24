"""MAF-derived feature helpers shared by the bundle and legacy wrappers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_chrom(value: object) -> str:
    text = str(value).replace("chr", "").replace("CHR", "").strip().upper()
    if not text or text.lower() == "nan":
        return "0"
    return text.lstrip("0") or "0"


def normalize_modality(value: object) -> str:
    text = str(value).strip().upper()
    if text in {"SNP", "SNV"}:
        return "SBS"
    if text in {"DNP", "TNP", "ONP"}:
        return "DBS"
    if text in {"INS", "DEL"}:
        return "ID"
    return "OTHER"


def pivot_counts(grouped: pd.DataFrame, columns: list[str], patient_ids: list[str], prefix: str) -> pd.DataFrame:
    if grouped.empty:
        return pd.DataFrame(index=pd.Index(patient_ids, name="patient_id"), dtype=np.float32)
    frame = grouped.copy()
    frame["feature"] = frame[columns].astype(str).agg("__".join, axis=1)
    wide = frame.pivot_table(index="patient_id", columns="feature", values="count", aggfunc="sum", fill_value=0.0)
    wide = wide.reindex(patient_ids).fillna(0.0).astype(np.float32)
    wide.columns = [f"{prefix}__{col}" for col in wide.columns]
    return wide


def distribution_summary(counts: pd.DataFrame, prefix: str) -> pd.DataFrame:
    values = counts.to_numpy(dtype=np.float64)
    total = values.sum(axis=1)
    denom = np.where(total > 0, total, 1.0)
    p = values / denom[:, None]
    nonzero = values > 0
    entropy = -(np.where(p > 0, p * np.log(p), 0.0)).sum(axis=1)
    max_entropy = np.log(np.maximum(1, values.shape[1]))
    normalized_entropy = np.divide(entropy, max_entropy, out=np.zeros_like(entropy), where=max_entropy > 0)
    sorted_p = -np.sort(-p, axis=1) if p.size else np.zeros((len(counts), 0))
    top5 = sorted_p[:, :5].sum(axis=1) if sorted_p.size else np.zeros(len(counts))
    top10 = sorted_p[:, :10].sum(axis=1) if sorted_p.size else np.zeros(len(counts))
    max_fraction = sorted_p[:, 0] if sorted_p.size else np.zeros(len(counts))
    out = pd.DataFrame(
        {
            f"{prefix}__log_total": np.log1p(total),
            f"{prefix}__nonzero_count": nonzero.sum(axis=1),
            f"{prefix}__nonzero_fraction": nonzero.mean(axis=1) if values.shape[1] else 0.0,
            f"{prefix}__entropy": entropy,
            f"{prefix}__normalized_entropy": normalized_entropy,
            f"{prefix}__simpson": (p * p).sum(axis=1),
            f"{prefix}__max_fraction": max_fraction,
            f"{prefix}__top5_fraction": top5,
            f"{prefix}__top10_fraction": top10,
            f"{prefix}__tail90_fraction": 1.0 - top10,
        },
        index=counts.index,
    )
    return out.astype(np.float32)


def _collapse_group_frames(frames: list[pd.DataFrame], group_cols: list[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=[*group_cols, "count"])
    merged = pd.concat(frames, ignore_index=True)
    return merged.groupby(group_cols, as_index=False, observed=True)["count"].sum()


def build_locus_topography_blocks(
    patient_ids: list[str],
    maf_path: str | Path,
    cache_dir: str | Path,
    *,
    target_gene: str = "KMT2C",
) -> dict[str, pd.DataFrame]:
    """Build KMT2C-excluded locus-topography feature blocks from an MC3-style MAF."""
    maf_path = Path(maf_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_meta = cache_dir / "locus_topography_cache_metadata.json"
    expected = {
        "locus_chrom_counts": cache_dir / "features_locus_chrom_counts.csv.gz",
        "locus_chrom_modality_counts": cache_dir / "features_locus_chrom_modality_counts.csv.gz",
        "locus_variant_class_counts": cache_dir / "features_locus_variant_class_counts.csv.gz",
        "locus_mb_top512": cache_dir / "features_locus_mb_top512.csv.gz",
        "locus_density_summary": cache_dir / "features_locus_density_summary.csv.gz",
    }
    if cache_meta.exists() and all(path.exists() for path in expected.values()):
        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        if meta.get("target_gene_excluded") == target_gene:
            return {name: pd.read_csv(path, index_col=0).fillna(0.0).astype(np.float32) for name, path in expected.items()}

    usecols = ["Hugo_Symbol", "Tumor_Sample_Barcode", "Chromosome", "Start_Position", "Variant_Type", "Variant_Classification"]
    patient_set = set(patient_ids)
    chrom_frames: list[pd.DataFrame] = []
    chrom_modality_frames: list[pd.DataFrame] = []
    variant_class_frames: list[pd.DataFrame] = []
    mb_frames: list[pd.DataFrame] = []
    total_rows = 0
    kept_rows = 0
    for chunk in pd.read_csv(maf_path, sep="\t", usecols=usecols, dtype=str, chunksize=400_000):
        total_rows += len(chunk)
        chunk["patient_id"] = chunk["Tumor_Sample_Barcode"].astype(str).str[:12]
        chunk = chunk[chunk["patient_id"].isin(patient_set)].copy()
        chunk = chunk[chunk["Hugo_Symbol"].fillna("").astype(str).str.upper() != target_gene.upper()].copy()
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

    chrom_counts = pivot_counts(_collapse_group_frames(chrom_frames, ["patient_id", "chrom"]), ["chrom"], patient_ids, "locus_chrom")
    chrom_modality = pivot_counts(_collapse_group_frames(chrom_modality_frames, ["patient_id", "chrom", "modality"]), ["chrom", "modality"], patient_ids, "locus_chrom_modality")
    variant_class = pivot_counts(_collapse_group_frames(variant_class_frames, ["patient_id", "variant_class"]), ["variant_class"], patient_ids, "locus_variant_class")
    mb_counts = pivot_counts(_collapse_group_frames(mb_frames, ["patient_id", "mb_bin"]), ["mb_bin"], patient_ids, "locus_mb")
    mb_top = mb_counts.loc[:, mb_counts.var(axis=0).sort_values(ascending=False).head(512).index] if mb_counts.shape[1] > 512 else mb_counts
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
                "target_gene_excluded": target_gene,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return out

