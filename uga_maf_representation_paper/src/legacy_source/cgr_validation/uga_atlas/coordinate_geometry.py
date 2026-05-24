"""Event-level coordinate geometries for MAF-derived mutation features."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .encoding import X_MAP, Y_MAP, trim_shared_alleles
from .events import clean_observed_allele, normalize_observed_chrom


BASES = set("ACGT")
COMP = str.maketrans("ACGTN", "TGCAN")


@dataclass(frozen=True)
class CoordinateGeometrySpec:
    """Specification for one event-level coordinate geometry."""

    name: str
    version: str
    family: str
    required_inputs: tuple[str, ...]
    coordinate_dimensions: int = 2
    aggregation_method: str = "multiresolution_histogram"
    leakage_rules: str = "No endpoint labels, no sample labels, and no clinical covariates are used to define coordinates."
    parameters: tuple[tuple[str, str], ...] = ()
    preserves_standard_identity: bool = False
    note: str = ""


GEOMETRY_REGISTRY: dict[str, CoordinateGeometrySpec] = {
    "exact_token_geometry_v1": CoordinateGeometrySpec(
        name="exact_token_geometry_v1",
        version="v1",
        family="exact_token_geometry",
        required_inputs=("standard_sbs96_id83_counts",),
        aggregation_method="exact_channel_identity",
        preserves_standard_identity=True,
        note="Exact recoverable coordinate identity control for Standard SBS96+ID83.",
    ),
    "sequence_cgr_k5_v1": CoordinateGeometrySpec(
        name="sequence_cgr_k5_v1",
        version="v1",
        family="sequence_cgr_geometry",
        required_inputs=("MAF", "GRCh37_FASTA", "5mer_context"),
        parameters=(("k", "5"),),
        note="Observed 5-mer context mapped into a CGR-style 2D coordinate and aggregated by quadtree bins.",
    ),
    "sequence_cgr_k7_v1": CoordinateGeometrySpec(
        name="sequence_cgr_k7_v1",
        version="v1",
        family="sequence_cgr_geometry",
        required_inputs=("MAF", "GRCh37_FASTA", "7mer_context"),
        parameters=(("k", "7"),),
        note="Observed 7-mer context mapped into a CGR-style 2D coordinate and aggregated by quadtree bins.",
    ),
    "sequence_cgr_k11_v1": CoordinateGeometrySpec(
        name="sequence_cgr_k11_v1",
        version="v1",
        family="sequence_cgr_geometry",
        required_inputs=("MAF", "GRCh37_FASTA", "11mer_context"),
        parameters=(("k", "11"),),
        note="Observed 11-mer context mapped into a CGR-style 2D coordinate and aggregated by quadtree bins.",
    ),
    "indel_biology_geometry_v1": CoordinateGeometrySpec(
        name="indel_biology_geometry_v1",
        version="v1",
        family="indel_biology_geometry",
        required_inputs=("MAF", "GRCh37_FASTA", "REF_ALT_payload", "local_flanks"),
        parameters=(("length_bins", "1,2,3,4,5,6_10,11p"), ("repeat_bins", "0,1,2,3,4,5,6p")),
        note="Indel coordinates ordered by insertion/deletion class, length, homopolymer span, microhomology, and base composition.",
    ),
    "motif_factorized_geometry_v1": CoordinateGeometrySpec(
        name="motif_factorized_geometry_v1",
        version="v1",
        family="motif_factorized_geometry",
        required_inputs=("MAF", "GRCh37_FASTA", "trinucleotide_context"),
        note="Coordinates arranged by SBS/ID motif class, mutation class, CpG status, APOBEC-like context, and payload class.",
    ),
    "topography_aware_geometry_v1": CoordinateGeometrySpec(
        name="topography_aware_geometry_v1",
        version="v1",
        family="topography_aware_geometry",
        required_inputs=("MAF", "GRCh37_FASTA", "variant_classification", "strand", "local_density"),
        note="Coordinates combine context with coding status, strand, local GC, and clustered-bin status.",
    ),
    "learned_self_supervised_geometry_v1": CoordinateGeometrySpec(
        name="learned_self_supervised_geometry_v1",
        version="v1",
        family="learned_self_supervised_geometry",
        required_inputs=("MAF", "GRCh37_FASTA", "event_attributes"),
        aggregation_method="frozen_hash_projection_histogram",
        note="Frozen label-free hash projection of event attributes into 2D; endpoint labels are not used.",
    ),
}


def list_coordinate_geometries() -> list[CoordinateGeometrySpec]:
    return [GEOMETRY_REGISTRY[key] for key in sorted(GEOMETRY_REGISTRY)]


def get_coordinate_geometry(name: str | CoordinateGeometrySpec) -> CoordinateGeometrySpec:
    if isinstance(name, CoordinateGeometrySpec):
        return name
    key = str(name)
    if key not in GEOMETRY_REGISTRY:
        known = ", ".join(sorted(GEOMETRY_REGISTRY))
        raise KeyError(f"Unknown coordinate geometry '{key}'. Available geometries: {known}")
    return GEOMETRY_REGISTRY[key]


def clean_maf_allele(value: object) -> str:
    return clean_observed_allele(value)


def choose_tumor_alt(row: Mapping[str, object]) -> str:
    ref = clean_maf_allele(row.get("Reference_Allele", ""))
    allele2 = clean_maf_allele(row.get("Tumor_Seq_Allele2", ""))
    allele1 = clean_maf_allele(row.get("Tumor_Seq_Allele1", ""))
    if allele2 and allele2 != ref:
        return allele2
    if allele1 and allele1 != ref:
        return allele1
    return allele2


def infer_event_modality(ref: str, alt: str, variant_type: object = "") -> str:
    vt = str(variant_type or "").upper()
    if vt in {"INS", "DEL"} or len(ref) != len(alt):
        return "ID"
    if len(ref) == 1 and len(alt) == 1:
        return "SBS"
    if len(ref) == 2 and len(alt) == 2:
        return "DBS"
    return "OTHER"


def reverse_complement(seq: str) -> str:
    return str(seq or "").upper().translate(COMP)[::-1]


def canonical_sbs96(ref: str, alt: str, left_base: str, right_base: str) -> str | None:
    if len(ref) != 1 or len(alt) != 1 or len(left_base) != 1 or len(right_base) != 1 or ref == alt:
        return None
    ref_s = ref
    alt_s = alt
    left = left_base
    right = right_base
    if ref_s in {"A", "G"}:
        ref_s = reverse_complement(ref_s)
        alt_s = reverse_complement(alt_s)
        left, right = reverse_complement(right), reverse_complement(left)
    if ref_s not in {"C", "T"} or alt_s not in BASES:
        return None
    return f"{left}[{ref_s}>{alt_s}]{right}"


def bits_to_unit_interval(bits: list[float] | np.ndarray) -> float:
    arr = np.asarray(bits, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    weights = 2.0 ** -np.arange(1, arr.size + 1, dtype=np.float64)
    denom = float(weights.sum())
    return float(np.dot(arr, weights) / denom)


def cgr_xy(sequence: str) -> tuple[float, float]:
    seq = "".join(ch for ch in str(sequence or "").upper() if ch in BASES)
    x_bits = [X_MAP.get(ch, 0.0) for ch in seq]
    y_bits = [Y_MAP.get(ch, 0.0) for ch in seq]
    return bits_to_unit_interval(x_bits), bits_to_unit_interval(y_bits)


def stable_unit_pair(text: str) -> tuple[float, float]:
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    x = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    y = int.from_bytes(digest[8:16], "big") / float(2**64 - 1)
    return float(x), float(y)


def suffix_run(seq: str, base: str) -> int:
    out = 0
    for ch in reversed(str(seq or "").upper()):
        if ch == base:
            out += 1
        else:
            break
    return out


def prefix_run(seq: str, base: str) -> int:
    out = 0
    for ch in str(seq or "").upper():
        if ch == base:
            out += 1
        else:
            break
    return out


def homopolymer_span(payload: str, left_context: str, right_context: str) -> int:
    payload = "".join(ch for ch in str(payload or "").upper() if ch in BASES)
    if not payload or len(set(payload)) != 1:
        return 0
    base = payload[0]
    return len(payload) + suffix_run(left_context, base) + prefix_run(right_context, base)


def boundary_microhomology(left_context: str, right_context: str, max_len: int = 20) -> int:
    left = str(left_context or "").upper()
    right = str(right_context or "").upper()
    best = 0
    for size in range(1, min(max_len, len(left), len(right)) + 1):
        if left[-size:] == right[:size]:
            best = size
    return best


def bin_numeric(value: int | float, edges: tuple[int, ...]) -> int:
    v = float(value)
    for idx, edge in enumerate(edges):
        if v <= edge:
            return idx
    return len(edges)


def motif_class(attrs: Mapping[str, object]) -> str:
    modality = str(attrs.get("modality", "OTHER"))
    if modality == "ID":
        return str(attrs.get("indel_class", "ID_other"))
    if modality == "DBS":
        ref = str(attrs.get("ref", ""))
        alt = str(attrs.get("alt", ""))
        return f"DBS_{ref}>{alt}"
    ref = str(attrs.get("ref", ""))
    alt = str(attrs.get("alt", ""))
    context = str(attrs.get("context_3", "NNN"))
    if len(context) == 3 and context[1] == "C" and alt == "T" and context[2] == "G":
        return "CpG_CtoT"
    if len(context) == 3 and context[1] == "C" and alt in {"G", "T"} and context[0] == "T":
        return "APOBEC_like"
    if ref and alt:
        return f"SBS_{ref}>{alt}"
    return "OTHER"


def build_event_attributes(
    row: Mapping[str, object],
    *,
    left_context: str,
    right_context: str,
    local_density_count: int = 0,
    reference_match: bool | None = None,
) -> dict[str, object]:
    ref = clean_maf_allele(row.get("Reference_Allele", ""))
    alt = choose_tumor_alt(row)
    modality = infer_event_modality(ref, alt, row.get("Variant_Type", ""))
    left = str(left_context or "").upper()
    right = str(right_context or "").upper()
    center = ref[:1] if ref else "N"
    context_11 = (left[-5:] + center + right[:5]).upper()
    context_7 = (left[-3:] + center + right[:3]).upper()
    context_5 = (left[-2:] + center + right[:2]).upper()
    ref_payload, alt_payload = trim_shared_alleles(ref, alt)
    payload = ref_payload if len(ref_payload) >= len(alt_payload) else alt_payload
    payload = "".join(ch for ch in str(payload or "").upper() if ch in BASES)
    event_class = "other"
    if modality == "ID":
        event_class = "del" if len(ref_payload) >= len(alt_payload) else "ins"
    hp = homopolymer_span(payload, left, right)
    mh = boundary_microhomology(left, right) if event_class == "del" else 0
    gc_context = left[-10:] + right[:10]
    gc = (gc_context.count("G") + gc_context.count("C")) / max(len(gc_context), 1)
    coding = str(row.get("Variant_Classification", "")).lower()
    gene_region = "coding" if any(term in coding for term in ["missense", "nonsense", "splice", "frame", "in_frame"]) else "noncoding_or_silent"
    strand = str(row.get("STRAND", row.get("Strand", "")) or "")
    attrs = {
        "patient_id": str(row.get("patient_id_12", row.get("Tumor_Sample_Barcode", "")))[:12],
        "chrom": normalize_observed_chrom(row.get("Chromosome", "")),
        "pos": int(float(row.get("Start_Position", 0) or 0)),
        "gene": str(row.get("Hugo_Symbol", "") or "").upper(),
        "ref": ref,
        "alt": alt,
        "modality": modality,
        "variant_classification": str(row.get("Variant_Classification", "") or ""),
        "context_5": context_5,
        "context_7": context_7,
        "context_11": context_11,
        "context_3": context_5[1:4] if len(context_5) >= 5 else "NNN",
        "ref_payload": ref_payload,
        "alt_payload": alt_payload,
        "payload": payload,
        "payload_len": len(payload),
        "event_class": event_class,
        "homopolymer_len": hp,
        "microhomology_len": mh,
        "local_gc": gc,
        "gene_region": gene_region,
        "strand": strand,
        "local_density_count": int(local_density_count),
        "clustered": int(local_density_count >= 3),
        "reference_match": bool(reference_match) if reference_match is not None else False,
    }
    attrs["sbs96"] = canonical_sbs96(ref, alt, context_5[1:2], context_5[3:4]) if modality == "SBS" else None
    attrs["indel_class"] = (
        f"{event_class}_len{bin_numeric(len(payload), (1, 2, 3, 4, 5, 10))}_hp{bin_numeric(hp, (0, 1, 2, 3, 4, 5))}_mh{bin_numeric(mh, (0, 1, 2, 3, 4, 5))}"
        if modality == "ID"
        else ""
    )
    attrs["motif_class"] = motif_class(attrs)
    return attrs


def encode_maf_event(attrs: Mapping[str, object], geometry: str | CoordinateGeometrySpec) -> tuple[float, float, str]:
    spec = get_coordinate_geometry(geometry)
    name = spec.name
    modality = str(attrs.get("modality", "OTHER"))
    modality_offset = {"SBS": 0.0, "DBS": 1.0 / 3.0, "ID": 2.0 / 3.0}.get(modality, 0.0)
    modality_width = 1.0 / 3.0

    if name.startswith("sequence_cgr_k"):
        k = int(dict(spec.parameters).get("k", "5"))
        seq = str(attrs.get(f"context_{k}", ""))
        x, y = cgr_xy(seq)
        return modality_offset + modality_width * x, y, f"{modality}:sequence_k{k}"

    if name == "indel_biology_geometry_v1":
        if modality != "ID":
            x, y = stable_unit_pair(f"{modality}|{attrs.get('motif_class', '')}")
            return modality_offset + modality_width * x, y, f"{modality}:non_id"
        event_class = str(attrs.get("event_class", "other"))
        event_x = 0.15 if event_class == "del" else 0.85 if event_class == "ins" else 0.5
        length_component = min(float(attrs.get("payload_len", 0)) / 12.0, 1.0)
        hp_component = min(float(attrs.get("homopolymer_len", 0)) / 12.0, 1.0)
        mh_component = min(float(attrs.get("microhomology_len", 0)) / 12.0, 1.0)
        x = 0.65 * event_x + 0.35 * length_component
        y = 0.55 * hp_component + 0.45 * mh_component
        return modality_offset + modality_width * x, y, str(attrs.get("indel_class", "ID_other"))

    if name == "motif_factorized_geometry_v1":
        motif = str(attrs.get("motif_class", "OTHER"))
        x, _ = stable_unit_pair(f"motif-x|{motif}")
        _, y = stable_unit_pair(f"motif-y|{modality}|{attrs.get('context_3', '')}|{attrs.get('payload_len', 0)}")
        return modality_offset + modality_width * x, y, motif

    if name == "topography_aware_geometry_v1":
        motif = str(attrs.get("motif_class", "OTHER"))
        x0, _ = stable_unit_pair(f"topo|{modality}|{motif}|{attrs.get('context_5', '')}")
        gc = float(attrs.get("local_gc", 0.0) or 0.0)
        clustered = float(attrs.get("clustered", 0) or 0)
        coding = 1.0 if str(attrs.get("gene_region", "")) == "coding" else 0.0
        strand_token = str(attrs.get("strand", ""))
        _, strand_y = stable_unit_pair(f"strand|{strand_token}")
        y = min(max(0.45 * gc + 0.25 * clustered + 0.20 * coding + 0.10 * strand_y, 0.0), 1.0)
        return modality_offset + modality_width * x0, y, f"{modality}:topography"

    if name == "learned_self_supervised_geometry_v1":
        token = "|".join(
            [
                modality,
                str(attrs.get("motif_class", "")),
                str(attrs.get("context_5", "")),
                str(attrs.get("indel_class", "")),
                str(attrs.get("gene_region", "")),
                str(attrs.get("clustered", "")),
            ]
        )
        x, y = stable_unit_pair(f"selfsup|{token}")
        return modality_offset + modality_width * x, y, token

    token = str(attrs.get("sbs96") or attrs.get("indel_class") or attrs.get("motif_class") or modality)
    x, y = stable_unit_pair(token)
    return modality_offset + modality_width * x, y, token


def coordinate_bin(x: float, y: float, level: int) -> tuple[int, int]:
    size = 2**int(level)
    ix = min(max(int(math.floor(float(x) * size)), 0), size - 1)
    iy = min(max(int(math.floor(float(y) * size)), 0), size - 1)
    return ix, iy


def morton_index(ix: int, iy: int, level: int) -> int:
    out = 0
    for bit in range(int(level)):
        shift = int(level) - 1 - bit
        out = (out << 1) | ((int(ix) >> shift) & 1)
        out = (out << 1) | ((int(iy) >> shift) & 1)
    return int(out)


def aggregate_event_coordinates(
    events: pd.DataFrame,
    geometry: str | CoordinateGeometrySpec,
    patient_ids: list[str],
    *,
    levels: tuple[int, ...] = (4, 6),
    use_morton: bool = True,
) -> pd.DataFrame:
    spec = get_coordinate_geometry(geometry)
    rows = {str(patient): {} for patient in patient_ids}
    for event in events.to_dict(orient="records"):
        patient = str(event.get("patient_id", ""))
        if patient not in rows:
            continue
        x, y, _token = encode_maf_event(event, spec)
        for level in levels:
            ix, iy = coordinate_bin(x, y, level)
            if use_morton:
                feature = f"{spec.name}__l{level}__m{morton_index(ix, iy, level)}"
            else:
                feature = f"{spec.name}__l{level}__x{ix}_y{iy}"
            rows[patient][feature] = rows[patient].get(feature, 0.0) + 1.0
    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).astype(np.float32)
