"""Bio MAF v4 featurization: fixed, interpretable per-tumor summaries of somatic mutations.

This module is a self-contained port of the featurization logic used in the
manuscript pipeline (``src/utils/bio_maf_base_features.py`` and the v4 builder in
``src/utils/quick_bio_v4_nested_feature_selection.py``). It computes exactly the
948 Bio MAF v4 features (943 core + 5 optional burden controls) from an annotated
MAF, using only fixed external references and no endpoint labels.

Input MAF columns used (all optional except Tumor_Sample_Barcode; missing
columns are treated as unknown):
    Tumor_Sample_Barcode, Hugo_Symbol, SYMBOL, Variant_Type, IMPACT, Consequence,
    CANONICAL, BIOTYPE, SIFT, PolyPhen, Protein_position, Amino_acids,
    HGVSp_Short, t_alt_count, t_depth
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .fetch_oncokb import ensure_oncokb_tsv

# ---------------------------------------------------------------------------
# Fixed vocabularies (identical to the manuscript pipeline)
# ---------------------------------------------------------------------------

IMPACTS = ["high", "moderate", "low", "modifier", "unknown"]
CONSEQUENCES = [
    "missense_variant",
    "synonymous_variant",
    "stop_gained",
    "frameshift_variant",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "splice_region_variant",
    "inframe_deletion",
    "inframe_insertion",
    "start_lost",
    "stop_lost",
    "stop_retained_variant",
    "protein_altering_variant",
    "coding_sequence_variant",
    "transcript_ablation",
    "intron_variant",
    "3_prime_utr_variant",
    "5_prime_utr_variant",
    "upstream_gene_variant",
    "downstream_gene_variant",
    "non_coding_transcript_exon_variant",
    "non_coding_transcript_variant",
    "mature_mirna_variant",
    "incomplete_terminal_codon_variant",
    "other",
    "unknown",
]
CANONICAL = ["yes", "no", "unknown"]
BIOTYPES = ["protein_coding", "lncrna", "nmd", "retained_intron", "processed_transcript", "other", "unknown"]
SIFT = ["deleterious", "deleterious_low_confidence", "tolerated", "tolerated_low_confidence", "unknown"]
POLYPHEN = ["probably_damaging", "possibly_damaging", "benign", "unknown"]

ROLE_LABELS = ["oncogene", "tumor_suppressor", "oncogene_and_tumor_suppressor"]
ROLE_METRICS = [
    "vep_high_or_moderate_impact_log_count",
    "vep_high_impact_log_count",
    "unique_high_or_moderate_impact_genes_log_count",
    "max_vaf_high_or_moderate_impact",
]
OPTIONAL_CONTROLS = ["log10_sbs_burden", "log10_dbs_burden", "log10_id_burden", "dbs_fraction", "id_fraction"]

ROLE_MATCH_LOF = {
    "stop_gained",
    "frameshift_variant",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "start_lost",
    "transcript_ablation",
}
ROLE_MATCH_ACT = {
    "missense_variant",
    "inframe_deletion",
    "inframe_insertion",
    "protein_altering_variant",
}
PROTEIN_AFFECTING = {
    *ROLE_MATCH_LOF,
    *ROLE_MATCH_ACT,
    "stop_lost",
    "coding_sequence_variant",
}
DRIVER_TIER_METRICS = [
    "high_or_moderate_impact_log_count",
    "high_impact_log_count",
    "unique_high_or_moderate_genes_log_count",
    "max_vaf_high_or_moderate_impact",
]
HOTSPOT_FEATURES = [
    "cancer_hotspot_exact_variant_log_count",
    "cancer_hotspot_residue_log_count",
    "cancer_hotspot_unique_gene_log_count",
    "cancer_hotspot_max_vaf",
]


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------

def normalize_gene(value: object) -> str:
    text = "" if value is None else str(value).strip().upper()
    if text in {"", "-", ".", "NAN", "NONE", "NA", "N/A", "UNKNOWN"}:
        return "UNKNOWN"
    return re.sub(r"[^A-Z0-9]+", "_", text).strip("_") or "UNKNOWN"


def first_12(value: object) -> str:
    return str(value).strip()[:12]


def _token(value: object, valid: set[str], fallback: str = "unknown") -> str:
    text = "" if value is None else str(value).strip().lower()
    return text if text in valid else fallback


def _impact(value: object) -> str:
    return _token(value, {"high", "moderate", "low", "modifier"})


def _consequence(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    if text in set(CONSEQUENCES):
        return text
    return "unknown"


def _canonical(value: object) -> str:
    return _token(value, {"yes", "no"})


def _biotype(value: object) -> str:
    return _token(value, {"protein_coding", "lncrna", "nmd", "retained_intron", "processed_transcript"})


def _sift(value: object) -> str:
    return _token(value, {"deleterious", "deleterious_low_confidence", "tolerated", "tolerated_low_confidence"})


def _polyphen(value: object) -> str:
    return _token(value, {"probably_damaging", "possibly_damaging", "benign"})


def _variant_event_class(value: object) -> str:
    token = "" if value is None else str(value).strip().lower()
    if token in {"snp", "snv"}:
        return "sbs"
    if token in {"dnp", "dnv"}:
        return "dbs"
    if token in {"ins", "del", "insertion", "deletion"}:
        return "id"
    return "other"


def parse_protein_position(value: object) -> int | None:
    text = "" if value is None else str(value)
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None


def parse_amino_acids(row: pd.Series) -> tuple[str, str]:
    aa = "" if row.get("Amino_acids") is None else str(row.get("Amino_acids"))
    if "/" in aa:
        ref, alt = aa.split("/", 1)
        return ref.strip().upper(), alt.strip().upper()
    hgvsp = "" if row.get("HGVSp_Short") is None else str(row.get("HGVSp_Short"))
    match = re.search(r"p\.?([A-Za-z\*]+)(\d+)([A-Za-z\*]+)", hgvsp)
    if match:
        return match.group(1).upper(), match.group(3).upper()
    return "", ""


# ---------------------------------------------------------------------------
# Feature layout
# ---------------------------------------------------------------------------

def base_feature_columns() -> tuple[list[str], list[str], dict[str, list[str]]]:
    vep: list[str] = []
    for prefix, labels in [
        ("vep_impact", IMPACTS),
        ("vep_consequence", CONSEQUENCES),
        ("vep_canonical", CANONICAL),
        ("vep_biotype", BIOTYPES),
        ("vep_sift", SIFT),
        ("vep_polyphen", POLYPHEN),
    ]:
        for label in labels:
            vep.extend([f"{prefix}_count__{label}", f"{prefix}_fraction__{label}"])
    vaf = ["vaf_mean", "vaf_std", "vaf_median", "vaf_q90", "vaf_max"]
    oncokb = [f"oncokb_role_{metric}__{role}" for role in ROLE_LABELS for metric in ROLE_METRICS]
    residual = [f"residual_non_oncokb_positive_role_{metric}" for metric in ROLE_METRICS]
    core = [*vep, *vaf, *oncokb, *residual]
    all_cols = [*core, *OPTIONAL_CONTROLS]
    blocks = {
        "vep_annotations": vep,
        "vep_plus_vaf": [*vep, *vaf],
        "oncokb_roles": oncokb,
        "oncokb_plus_residual": [*oncokb, *residual],
        "core_bio_maf": core,
        "core_bio_maf_plus_burden_controls": all_cols,
    }
    return core, all_cols, blocks


def v4_feature_columns(panel_genes: list[str]) -> tuple[list[str], list[str], dict[str, list[str]]]:
    base_core, _base_all, _base_blocks = base_feature_columns()
    tier_cols = [
        f"driver_evidence_{metric}__source_count_{tier}"
        for tier in range(1, 5)
        for metric in DRIVER_TIER_METRICS
    ]
    # Bio MAF v4 (948 features, as used in the manuscript) emits both the
    # functional-event and the role-matched-event column for every one of the
    # 401 panel genes.
    exact_cols: list[str] = []
    for gene in panel_genes:
        exact_cols.append(f"driver_gene_functional_event_log_count__{gene}")
        exact_cols.append(f"driver_gene_role_matched_event_log_count__{gene}")
    hotspot_cols = list(HOTSPOT_FEATURES)
    core = [*base_core, *tier_cols, *exact_cols, *hotspot_cols]
    all_cols = [*core, *OPTIONAL_CONTROLS]
    blocks = {
        "v4_compact_biology_only": base_core,
        "v4_compact_plus_driver_evidence_tiers": [*base_core, *tier_cols],
        "v4_compact_plus_hotspot_summary": [*base_core, *hotspot_cols],
        "v4_compact_plus_exact_driver_genes": [*base_core, *exact_cols],
        "v4_full_core": core,
        "v4_full_core_plus_optional_controls": all_cols,
    }
    return core, all_cols, blocks


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

def resources_dir() -> Path:
    return Path(__file__).resolve().parent / "resources"


def load_oncokb_roles(path: Path) -> dict[str, str]:
    table = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    roles: dict[str, str] = {}
    for row in table.to_dict(orient="records"):
        symbol = normalize_gene(row.get("hugo_symbol") or row.get("Hugo Symbol"))
        if symbol == "UNKNOWN":
            continue
        oncogene = str(row.get("oncogene", "")).strip().lower() == "true"
        tsg = str(row.get("tsg", "")).strip().lower() == "true"
        gene_type = str(row.get("Gene Type", "")).strip().upper()
        if gene_type:
            oncogene = oncogene or gene_type in {"ONCOGENE", "ONCOGENE_AND_TSG"}
            tsg = tsg or gene_type in {"TSG", "ONCOGENE_AND_TSG"}
        if oncogene and tsg:
            roles[symbol] = "oncogene_and_tumor_suppressor"
        elif oncogene:
            roles[symbol] = "oncogene"
        elif tsg:
            roles[symbol] = "tumor_suppressor"
    return roles


def load_v4_panel(res_dir: Path) -> tuple[pd.DataFrame, dict[str, int], dict[str, str], set, set]:
    """Load the 401-gene driver panel, evidence tiers, and hotspot sets."""
    tables = res_dir
    evidence = pd.read_csv(tables / "driver_gene_evidence.csv")
    evidence["gene"] = evidence["gene"].map(normalize_gene)
    tiers = evidence.set_index("gene")["primary_driver_evidence_count"].fillna(0).astype(int).to_dict()
    panel = pd.read_csv(tables / "driver_gene_panel.csv")
    panel["gene"] = panel["gene"].map(normalize_gene)
    role_lookup = panel.set_index("gene")["role_interpretation"].fillna("").astype(str).to_dict()

    residues = pd.read_csv(tables / "v3_multi_type_residue.txt", sep="\t")
    residues["gene"] = residues["Hugo_Symbol"].map(normalize_gene)
    residue_set = {
        (str(row.gene), int(pos))
        for row in residues.itertuples(index=False)
        for pos in [parse_protein_position(getattr(row, "Amino_Acid_Position", None))]
        if pos is not None and str(row.gene) != "UNKNOWN"
    }
    variants = pd.read_csv(tables / "v3_multi_type_variant_file.txt", sep="\t")
    variants["gene"] = variants["Hugo_Symbol"].map(normalize_gene)
    variant_set = set()
    for row in variants.itertuples(index=False):
        pos = parse_protein_position(getattr(row, "Amino_Acid_Position", None))
        ref = str(getattr(row, "Reference_Amino_Acid", "")).strip().upper()
        alt = str(getattr(row, "Variant_Amino_Acid", "")).strip().upper()
        gene = str(row.gene)
        if pos is not None and gene != "UNKNOWN" and ref and alt:
            variant_set.add((gene, pos, ref, alt))
    return panel, tiers, role_lookup, residue_set, variant_set


def is_role_matched(row: pd.Series, role_lookup: Mapping[str, str]) -> bool:
    role = role_lookup.get(str(row["_gene"]), "")
    consequence = str(row["_consequence"])
    impact = str(row["_impact"])
    hotspot = bool(row["_hotspot_residue"] or row["_hotspot_exact"])
    lof = impact == "high" or consequence in ROLE_MATCH_LOF
    activating = hotspot or consequence in ROLE_MATCH_ACT
    if "tumor-suppressor" in role:
        return lof
    if "activating/oncogene" in role:
        return activating
    if "both-role" in role or "ambiguous" in role:
        return lof or activating
    return impact in {"high", "moderate"} or hotspot


# ---------------------------------------------------------------------------
# Main featurization
# ---------------------------------------------------------------------------

def _add_group_counts(matrix, frame, sample_col, token_col, prefix, valid_tokens) -> None:
    counts = frame.groupby([sample_col, token_col], observed=True).size()
    for (sample, token), count in counts.items():
        if token in valid_tokens:
            matrix.at[sample, f"{prefix}_count__{token}"] += float(count)


def featurize_maf(
    maf_path: str | Path,
    sample_ids: Iterable[str] | None = None,
    *,
    chunksize: int = 350_000,
    include_optional_controls: bool = True,
) -> pd.DataFrame:
    """Build the Bio MAF v4 feature matrix from an annotated MAF.

    Parameters
    ----------
    maf_path : path to a MAF file (plain or .gz), tab-separated.
    sample_ids : optional explicit sample list. Defaults to all
        Tumor_Sample_Barcode values in the MAF (first 12 characters).
    chunksize : rows per chunk when streaming the MAF.
    include_optional_controls : if False, drop the 5 optional burden
        controls (943 core features only).

    Returns
    -------
    DataFrame indexed by sample with 948 (or 943) float32 feature columns.
    """
    res_dir = resources_dir()
    role_lookup_oncokb = load_oncokb_roles(ensure_oncokb_tsv(res_dir / "oncokb_cancer_genes.tsv"))
    panel, tier_lookup, role_lookup, residue_set, variant_set = load_v4_panel(res_dir)

    panel_genes = panel["gene"].astype(str).tolist()
    core_cols, all_cols, _blocks = v4_feature_columns(panel_genes)
    out_cols = all_cols if include_optional_controls else core_cols

    # Discover samples from the MAF unless given explicitly.
    if sample_ids is None:
        seen: list[str] = []
        seen_set: set[str] = set()
        for chunk in pd.read_csv(maf_path, sep="\t", usecols=["Tumor_Sample_Barcode"],
                                 dtype=str, chunksize=chunksize):
            for barcode in chunk["Tumor_Sample_Barcode"].map(first_12):
                if barcode not in seen_set:
                    seen_set.add(barcode)
                    seen.append(barcode)
        sample_ids = seen
    samples = [str(s) for s in sample_ids]
    sample_set = set(samples)

    base_core, base_all, _ = base_feature_columns()
    raw = pd.DataFrame(0.0, index=pd.Index(samples, name="sample"), columns=base_all, dtype=np.float64)
    total_events = pd.Series(0.0, index=raw.index)
    broad_counts = {kind: pd.Series(0.0, index=raw.index) for kind in ["sbs", "dbs", "id"]}
    vaf_values: dict[str, list[np.ndarray]] = defaultdict(list)
    unique_gene_sets: dict[str, dict[str, set[str]]] = {role: defaultdict(set) for role in [*ROLE_LABELS, "residual"]}

    # ---- Pass 1: base biology features ----
    usecols_base = [
        "Tumor_Sample_Barcode", "Hugo_Symbol", "SYMBOL", "Variant_Type",
        "IMPACT", "Consequence", "CANONICAL", "BIOTYPE", "SIFT", "PolyPhen",
        "t_alt_count", "t_depth",
    ]
    for chunk in pd.read_csv(maf_path, sep="\t", usecols=lambda c: c in usecols_base,
                             dtype=str, chunksize=chunksize):
        chunk["sample"] = chunk["Tumor_Sample_Barcode"].map(first_12)
        chunk = chunk[chunk["sample"].isin(sample_set)].copy()
        if chunk.empty:
            continue
        total_events = total_events.add(chunk.groupby("sample", observed=True).size(), fill_value=0.0)

        chunk["_impact"] = chunk["IMPACT"].map(_impact) if "IMPACT" in chunk else "unknown"
        chunk["_consequence"] = chunk["Consequence"].map(_consequence) if "Consequence" in chunk else "unknown"
        chunk["_canonical"] = chunk["CANONICAL"].map(_canonical) if "CANONICAL" in chunk else "unknown"
        chunk["_biotype"] = chunk["BIOTYPE"].map(_biotype) if "BIOTYPE" in chunk else "unknown"
        chunk["_sift"] = chunk["SIFT"].map(_sift) if "SIFT" in chunk else "unknown"
        chunk["_polyphen"] = chunk["PolyPhen"].map(_polyphen) if "PolyPhen" in chunk else "unknown"
        for token_col, prefix, labels in [
            ("_impact", "vep_impact", IMPACTS),
            ("_consequence", "vep_consequence", CONSEQUENCES),
            ("_canonical", "vep_canonical", CANONICAL),
            ("_biotype", "vep_biotype", BIOTYPES),
            ("_sift", "vep_sift", SIFT),
            ("_polyphen", "vep_polyphen", POLYPHEN),
        ]:
            _add_group_counts(raw, chunk, "sample", token_col, prefix, labels)

        alt = pd.to_numeric(chunk["t_alt_count"], errors="coerce") if "t_alt_count" in chunk else pd.Series(np.nan, index=chunk.index)
        depth = pd.to_numeric(chunk["t_depth"], errors="coerce") if "t_depth" in chunk else pd.Series(np.nan, index=chunk.index)
        vaf = alt / depth.replace(0, np.nan)
        chunk["_vaf"] = vaf.where(np.isfinite(vaf), np.nan)
        valid_vaf = chunk[chunk["_vaf"].notna()]
        for sample, values in valid_vaf.groupby("sample", observed=True)["_vaf"]:
            vaf_values[sample].append(values.to_numpy(dtype=np.float64))

        hugo = chunk["Hugo_Symbol"] if "Hugo_Symbol" in chunk else pd.Series(np.nan, index=chunk.index)
        symb = chunk["SYMBOL"] if "SYMBOL" in chunk else pd.Series(np.nan, index=chunk.index)
        chunk["_gene"] = hugo.where(hugo.notna(), symb).map(normalize_gene)
        chunk["_role"] = chunk["_gene"].map(role_lookup_oncokb).fillna("residual")
        for role in [*ROLE_LABELS, "residual"]:
            role_frame = chunk[chunk["_role"] == role]
            if role_frame.empty:
                continue
            hm = role_frame[role_frame["_impact"].isin(["high", "moderate"])]
            hi = role_frame[role_frame["_impact"] == "high"]
            for sample, count in hm.groupby("sample", observed=True).size().items():
                col = (f"oncokb_role_vep_high_or_moderate_impact_log_count__{role}"
                       if role != "residual"
                       else "residual_non_oncokb_positive_role_vep_high_or_moderate_impact_log_count")
                raw.at[sample, col] += float(count)
            for sample, count in hi.groupby("sample", observed=True).size().items():
                col = (f"oncokb_role_vep_high_impact_log_count__{role}"
                       if role != "residual"
                       else "residual_non_oncokb_positive_role_vep_high_impact_log_count")
                raw.at[sample, col] += float(count)
            for sample, genes in hm.groupby("sample", observed=True)["_gene"]:
                unique_gene_sets[role][sample].update(g for g in genes if g != "UNKNOWN")
            for sample, max_vaf in hm.groupby("sample", observed=True)["_vaf"].max().dropna().items():
                col = (f"oncokb_role_max_vaf_high_or_moderate_impact__{role}"
                       if role != "residual"
                       else "residual_non_oncokb_positive_role_max_vaf_high_or_moderate_impact")
                raw.at[sample, col] = max(raw.at[sample, col], float(max_vaf))

        chunk["_event_class"] = chunk["Variant_Type"].map(_variant_event_class) if "Variant_Type" in chunk else "other"
        for kind in ["sbs", "dbs", "id"]:
            counts = chunk[chunk["_event_class"] == kind].groupby("sample", observed=True).size()
            broad_counts[kind] = broad_counts[kind].add(counts, fill_value=0.0)

    for role in [*ROLE_LABELS, "residual"]:
        for sample, genes in unique_gene_sets[role].items():
            col = (f"oncokb_role_unique_high_or_moderate_impact_genes_log_count__{role}"
                   if role != "residual"
                   else "residual_non_oncokb_positive_role_unique_high_or_moderate_impact_genes_log_count")
            raw.at[sample, col] = float(len(genes))

    features = raw.copy()
    for col in [c for c in features.columns if "_count__" in c or c.endswith("_log_count") or "unique_high_or_moderate" in c]:
        if col.startswith("log10_"):
            continue
        features[col] = np.log1p(raw[col].to_numpy(dtype=np.float64))
    for prefix, labels in [
        ("vep_impact", IMPACTS),
        ("vep_consequence", CONSEQUENCES),
        ("vep_canonical", CANONICAL),
        ("vep_biotype", BIOTYPES),
        ("vep_sift", SIFT),
        ("vep_polyphen", POLYPHEN),
    ]:
        for label in labels:
            count_col = f"{prefix}_count__{label}"
            frac_col = f"{prefix}_fraction__{label}"
            features[frac_col] = np.divide(
                raw[count_col].to_numpy(dtype=np.float64),
                total_events.to_numpy(dtype=np.float64),
                out=np.zeros(len(total_events), dtype=np.float64),
                where=total_events.to_numpy(dtype=np.float64) > 0,
            )

    for sample in features.index:
        arrays = vaf_values.get(sample, [])
        if not arrays:
            continue
        vals = np.concatenate(arrays)
        if vals.size:
            features.at[sample, "vaf_mean"] = float(np.mean(vals))
            features.at[sample, "vaf_std"] = float(np.std(vals))
            features.at[sample, "vaf_median"] = float(np.median(vals))
            features.at[sample, "vaf_q90"] = float(np.quantile(vals, 0.9))
            features.at[sample, "vaf_max"] = float(np.max(vals))

    broad_total = broad_counts["sbs"] + broad_counts["dbs"] + broad_counts["id"]
    features["log10_sbs_burden"] = np.log10(1.0 + broad_counts["sbs"])
    features["log10_dbs_burden"] = np.log10(1.0 + broad_counts["dbs"])
    features["log10_id_burden"] = np.log10(1.0 + broad_counts["id"])
    broad_total_arr = broad_total.to_numpy(dtype=np.float64)
    features["dbs_fraction"] = np.divide(
        broad_counts["dbs"].to_numpy(dtype=np.float64), broad_total_arr,
        out=np.zeros(len(broad_total_arr), dtype=np.float64), where=broad_total_arr > 0)
    features["id_fraction"] = np.divide(
        broad_counts["id"].to_numpy(dtype=np.float64), broad_total_arr,
        out=np.zeros(len(broad_total_arr), dtype=np.float64), where=broad_total_arr > 0)

    # ---- Pass 2: v4 driver-gene / hotspot columns ----
    matrix = pd.DataFrame(0.0, index=pd.Index(samples, name="sample"), columns=all_cols, dtype=np.float64)
    matrix.loc[:, features.columns] = features.reindex(matrix.index).astype(np.float64)

    selected_genes = set(panel_genes)
    tier_unique_genes: dict[int, dict[str, set[str]]] = {tier: defaultdict(set) for tier in range(1, 5)}
    hotspot_unique_genes: dict[str, set[str]] = defaultdict(set)

    usecols_v4 = [
        "Tumor_Sample_Barcode", "Hugo_Symbol", "SYMBOL", "IMPACT", "Consequence",
        "Protein_position", "Amino_acids", "HGVSp_Short", "t_alt_count", "t_depth",
    ]
    for chunk in pd.read_csv(maf_path, sep="\t", usecols=lambda c: c in usecols_v4,
                             dtype=str, chunksize=chunksize):
        chunk["sample"] = chunk["Tumor_Sample_Barcode"].map(first_12)
        chunk = chunk[chunk["sample"].isin(sample_set)].copy()
        if chunk.empty:
            continue
        hugo = chunk["Hugo_Symbol"] if "Hugo_Symbol" in chunk else pd.Series(np.nan, index=chunk.index)
        symb = chunk["SYMBOL"] if "SYMBOL" in chunk else pd.Series(np.nan, index=chunk.index)
        chunk["_gene"] = hugo.where(hugo.notna(), symb).map(normalize_gene)
        chunk["_impact"] = chunk["IMPACT"].map(_impact) if "IMPACT" in chunk else "unknown"
        chunk["_consequence"] = chunk["Consequence"].map(_consequence) if "Consequence" in chunk else "unknown"
        alt = pd.to_numeric(chunk["t_alt_count"], errors="coerce") if "t_alt_count" in chunk else pd.Series(np.nan, index=chunk.index)
        depth = pd.to_numeric(chunk["t_depth"], errors="coerce") if "t_depth" in chunk else pd.Series(np.nan, index=chunk.index)
        vaf = alt / depth.replace(0, np.nan)
        chunk["_vaf"] = vaf.where(np.isfinite(vaf), np.nan)
        chunk["_protein_pos"] = chunk["Protein_position"].map(parse_protein_position) if "Protein_position" in chunk else None
        aa = chunk.apply(parse_amino_acids, axis=1, result_type="expand")
        chunk["_ref_aa"] = aa[0]
        chunk["_alt_aa"] = aa[1]
        chunk["_tier"] = chunk["_gene"].map(tier_lookup).fillna(0).astype(int).clip(0, 4)
        chunk["_hotspot_residue"] = [
            (gene, pos) in residue_set if pos is not None else False
            for gene, pos in zip(chunk["_gene"], chunk["_protein_pos"])
        ]
        chunk["_hotspot_exact"] = [
            (gene, pos, ref, alt_) in variant_set if pos is not None else False
            for gene, pos, ref, alt_ in zip(chunk["_gene"], chunk["_protein_pos"], chunk["_ref_aa"], chunk["_alt_aa"])
        ]
        chunk["_functional"] = chunk["_impact"].isin(["high", "moderate"]) | chunk["_consequence"].isin(PROTEIN_AFFECTING)
        chunk["_high_mod"] = chunk["_impact"].isin(["high", "moderate"])
        chunk["_high"] = chunk["_impact"].eq("high")
        chunk["_role_matched"] = chunk.apply(is_role_matched, axis=1, role_lookup=role_lookup)

        for tier in range(1, 5):
            tier_frame = chunk[chunk["_tier"].eq(tier)]
            hm = tier_frame[tier_frame["_high_mod"]]
            hi = tier_frame[tier_frame["_high"]]
            for sample, count in hm.groupby("sample", observed=True).size().items():
                matrix.at[sample, f"driver_evidence_high_or_moderate_impact_log_count__source_count_{tier}"] += float(count)
            for sample, count in hi.groupby("sample", observed=True).size().items():
                matrix.at[sample, f"driver_evidence_high_impact_log_count__source_count_{tier}"] += float(count)
            for sample, genes in hm.groupby("sample", observed=True)["_gene"]:
                tier_unique_genes[tier][sample].update(g for g in genes if g != "UNKNOWN")
            for sample, max_vaf in hm.groupby("sample", observed=True)["_vaf"].max().dropna().items():
                col = f"driver_evidence_max_vaf_high_or_moderate_impact__source_count_{tier}"
                matrix.at[sample, col] = max(float(matrix.at[sample, col]), float(max_vaf))

        gene_frame = chunk[chunk["_gene"].isin(selected_genes)]
        functional = gene_frame[gene_frame["_functional"]]
        role_matched = gene_frame[gene_frame["_role_matched"]]
        for (sample, gene), count in functional.groupby(["sample", "_gene"], observed=True).size().items():
            matrix.at[sample, f"driver_gene_functional_event_log_count__{gene}"] += float(count)
        for (sample, gene), count in role_matched.groupby(["sample", "_gene"], observed=True).size().items():
            col = f"driver_gene_role_matched_event_log_count__{gene}"
            if col in matrix.columns:
                matrix.at[sample, col] += float(count)

        hs_residue = chunk[chunk["_hotspot_residue"]]
        hs_exact = chunk[chunk["_hotspot_exact"]]
        for sample, count in hs_exact.groupby("sample", observed=True).size().items():
            matrix.at[sample, "cancer_hotspot_exact_variant_log_count"] += float(count)
        for sample, count in hs_residue.groupby("sample", observed=True).size().items():
            matrix.at[sample, "cancer_hotspot_residue_log_count"] += float(count)
        for sample, genes in hs_residue.groupby("sample", observed=True)["_gene"]:
            hotspot_unique_genes[sample].update(g for g in genes if g != "UNKNOWN")
        for sample, max_vaf in hs_residue.groupby("sample", observed=True)["_vaf"].max().dropna().items():
            matrix.at[sample, "cancer_hotspot_max_vaf"] = max(float(matrix.at[sample, "cancer_hotspot_max_vaf"]), float(max_vaf))

    for tier in range(1, 5):
        for sample, genes in tier_unique_genes[tier].items():
            matrix.at[sample, f"driver_evidence_unique_high_or_moderate_genes_log_count__source_count_{tier}"] = float(len(genes))
    for sample, genes in hotspot_unique_genes.items():
        matrix.at[sample, "cancer_hotspot_unique_gene_log_count"] = float(len(genes))

    log_count_cols = [col for col in matrix.columns if col.endswith("_log_count") or "_log_count__" in col]
    for col in log_count_cols:
        matrix[col] = np.log1p(matrix[col].to_numpy(dtype=np.float64))
    matrix = matrix.loc[:, out_cols].fillna(0.0).astype(np.float32)
    return matrix


def feature_names(include_optional_controls: bool = True) -> list[str]:
    """Return the Bio MAF v4 column names without reading a MAF."""
    res_dir = resources_dir()
    panel = pd.read_csv(res_dir / "driver_gene_panel.csv")
    panel["gene"] = panel["gene"].map(normalize_gene)
    core_cols, all_cols, _blocks = v4_feature_columns(panel["gene"].astype(str).tolist())
    return all_cols if include_optional_controls else core_cols
