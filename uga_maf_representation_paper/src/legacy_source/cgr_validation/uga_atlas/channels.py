"""Channel-to-UGA basis construction."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .context import DEFAULT_DINUCLEOTIDE_ATLAS, load_dinucleotide_context_atlas
from .encoding import (
    PAYLOAD_SCHEMA_MASKED,
    assemble_uga_vector,
    encode_alt_cgr_walk,
    payload_block_dim,
    revcomp,
    trim_shared_alleles,
    universal_context_slice,
    universal_vector_dim,
)
from .models import UGAModelSpec, get_uga_model


BASES = "ACGT"
BASE_SET = set(BASES)
MOTIF = "ACGTA"
SBS96_RE = re.compile(r"^([ACGT])\[([ACGT])>([ACGT])\]([ACGT])$")


def channel_kind(channel: str) -> str:
    ch = str(channel)
    if SBS96_RE.match(ch):
        return "SBS96"
    if len(ch) == 6 and set(ch) <= BASE_SET and ch[2] in "CT" and ch[5] in BASE_SET and ch[5] != ch[2]:
        return "SBS1536"
    if ">" in ch:
        left, right = ch.split(">", 1)
        if len(left) == 2 and len(right) == 2 and set(left + right) <= BASE_SET and left != right:
            return "DBS78"
    if len(ch) == 4 and set(ch) <= BASE_SET and ch[:2] != ch[2:]:
        return "DBS78"
    parts = ch.split(":")
    if len(parts) >= 4 and (parts[0].upper() in {"DEL", "INS"} or parts[1].lower() in {"del", "ins"}):
        return "ID83"
    return "other"


def sbs_ctx_alt(channel: str) -> tuple[str, str]:
    match = SBS96_RE.match(str(channel))
    if not match:
        raise ValueError(f"Unsupported SBS96 channel: {channel}")
    left, ref, alt, right = match.groups()
    return left + ref + right, alt


def _dbs_canonical_ref(ref: str) -> str:
    ref = str(ref).upper()
    return ref if ref and ref[0] in "CT" else revcomp(ref)


def _split_lr_context(context: np.ndarray, d_context: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(context, dtype=np.float64)
    max_d = len(arr) // 4
    dc = min(int(d_context), max_d)
    lx = arr[0 * max_d : 0 * max_d + dc]
    ly = arr[1 * max_d : 1 * max_d + dc]
    rx = arr[2 * max_d : 2 * max_d + dc]
    ry = arr[3 * max_d : 3 * max_d + dc]
    return np.concatenate([lx, ly]), np.concatenate([rx, ry])


def _assemble_from_context(
    left_block: np.ndarray,
    right_block: np.ndarray,
    ref: str,
    alt: str,
    d_payload: int,
    payload_schema: str,
) -> np.ndarray:
    ref_payload, alt_payload = trim_shared_alleles(ref, alt)
    return np.concatenate(
        [
            np.asarray(left_block, dtype=np.float64),
            encode_alt_cgr_walk(ref_payload, d=d_payload, payload_schema=payload_schema).astype(np.float64),
            np.asarray(right_block, dtype=np.float64),
            encode_alt_cgr_walk(alt_payload, d=d_payload, payload_schema=payload_schema).astype(np.float64),
        ]
    )


def dbs_ctx_alt(
    channel: str,
    atlas: dict[str, np.ndarray],
    d_context: int,
    dino_atlas_path: Path | str = DEFAULT_DINUCLEOTIDE_ATLAS,
) -> tuple[np.ndarray, str] | None:
    ch = str(channel)
    ref, alt = ch.split(">", 1) if ">" in ch else (ch[:2], ch[2:])
    canon_ref = _dbs_canonical_ref(ref)
    dino = load_dinucleotide_context_atlas(dino_atlas_path, d_context)
    for key in (canon_ref, ref):
        if key in dino:
            return dino[key], alt
    if canon_ref in atlas:
        return atlas[canon_ref], alt
    vecs = [atlas[b + canon_ref] for b in BASES if b + canon_ref in atlas]
    return (np.mean(vecs, axis=0), alt) if vecs else None


def variant_vec(
    channel: str,
    atlas: dict[str, np.ndarray],
    d_context: int,
    modality: str,
    d_payload: int | None = None,
    payload_schema: str = PAYLOAD_SCHEMA_MASKED,
) -> np.ndarray | None:
    """Embed an SBS96 or DBS78 channel using a genome-wide context atlas."""
    dp = int(d_payload if d_payload is not None else d_context)
    modality = str(modality).upper()
    if modality == "SBS":
        ctx, alt_one = sbs_ctx_alt(channel)
        if ctx not in atlas:
            return None
        left_block, right_block = _split_lr_context(atlas[ctx], d_context)
        return _assemble_from_context(left_block, right_block, channel[2], alt_one, dp, payload_schema)
    if modality == "DBS":
        got = dbs_ctx_alt(channel, atlas, d_context)
        if got is None:
            return None
        context, _ = got
        ref, alt = str(channel).split(">", 1) if ">" in str(channel) else (str(channel)[:2], str(channel)[2:])
        left_block, right_block = _split_lr_context(context, d_context)
        return _assemble_from_context(left_block, right_block, ref, alt, dp, payload_schema)
    if modality == "ID":
        left, right, ref, alt = id83_proxy_event(channel, int(d_context), dp)
        return assemble_uga_vector(left, right, ref, alt, int(d_context), dp, payload_schema)
    raise ValueError(f"Unsupported modality: {modality}")


def variant_vec_bicgr52_context_only(
    channel: str,
    atlas: dict[str, np.ndarray],
    d_context: int,
    modality: str,
    d_payload: int | None = None,
    payload_schema: str = PAYLOAD_SCHEMA_MASKED,
) -> np.ndarray | None:
    dp = int(d_payload if d_payload is not None else d_context)
    vec = variant_vec(channel, atlas, d_context, modality, dp, payload_schema)
    return None if vec is None else universal_context_slice(vec, d_context, dp, payload_schema).copy()


def parse_capped_int(token: str, cap: int) -> int:
    token = str(token)
    if token.endswith("+"):
        return int(cap)
    try:
        return max(0, min(int(token), int(cap)))
    except ValueError:
        return int(cap)


def _int_to_base4_sequence(value: int, width: int) -> str:
    chars = []
    value = int(value)
    for _ in range(int(width)):
        chars.append(BASES[value & 3])
        value >>= 2
    return "".join(reversed(chars))


def _id83_token_code(parts: list[str], d_context: int) -> str:
    event_code = {"DEL": 0, "INS": 1}.get(parts[0].upper(), 0)
    subtype_code = {"C": 0, "T": 1, "R": 2, "REPEATS": 2, "M": 3, "MH": 3}.get(parts[1].upper(), 0)
    length_code = parse_capped_int(parts[2], 5) if len(parts) > 2 else 0
    aux_code = parse_capped_int(parts[3], 5) if len(parts) > 3 else 0
    value = (((event_code * 4 + subtype_code) * 6 + length_code) * 6 + aux_code)
    seq = _int_to_base4_sequence(value, max(1, int(d_context)))
    if len(seq) < d_context:
        digest = hashlib.sha1(":".join(parts).encode("utf-8")).digest()
        filler_value = int.from_bytes(digest[:4], "big")
        seq = (seq + _int_to_base4_sequence(filler_value, d_context))[:d_context]
    return seq[-int(d_context) :]


def _hash_token_code(parts: list[str], d_context: int) -> str:
    digest = hashlib.sha1(":".join(str(p) for p in parts).encode("utf-8")).digest()
    value = int.from_bytes(digest, "big")
    return _int_to_base4_sequence(value, max(1, int(d_context)))[-int(d_context) :]


def repeated_context(unit: str, repeats: int, d_context: int) -> tuple[str, str]:
    unit = unit or "C"
    repeats = max(1, int(repeats))
    needed = max(int(d_context), len(unit) * (repeats + 2))
    seq = (unit * (math.ceil(needed / len(unit)) + 1))[:needed]
    return seq[-int(d_context) :], seq[: int(d_context)]


def _normalize_id83_parts(channel: str) -> list[str]:
    parts = str(channel).split(":")
    if len(parts) < 4:
        raise ValueError(f"Unsupported ID83 label: {channel}")
    if parts[0].upper() in {"DEL", "INS"}:
        event = parts[0].upper()
        subtype = parts[1].upper()
        length = parts[2]
        aux = parts[3]
        if subtype == "REPEATS":
            subtype = "R"
        if subtype == "MH":
            subtype = "M"
        return [event, subtype, length, aux]
    if parts[1].lower() in {"del", "ins"}:
        length, event, motif, aux = parts[:4]
        event = "DEL" if event.lower() == "del" else "INS"
        motif = motif.upper()
        return [event, motif, length, aux]
    raise ValueError(f"Unsupported ID83 label: {channel}")


def id83_proxy_event(channel: str, d_context: int, d_payload: int) -> tuple[str, str, str, str]:
    """Return left context, right context, REF, ALT for a categorical ID83 proxy."""
    return id83_proxy_event_for_source(channel, d_context, d_payload, "id83_proxy")


def id83_proxy_event_for_source(
    channel: str,
    d_context: int,
    d_payload: int,
    context_source: str = "id83_proxy",
) -> tuple[str, str, str, str]:
    """Return left context, right context, REF, ALT for a registered ID83 proxy."""
    event, motif, length_s, aux_s = _normalize_id83_parts(channel)
    length = max(1, parse_capped_int(length_s, d_payload))
    aux = parse_capped_int(aux_s, d_context)
    if motif == "C":
        unit = "C" * length
    elif motif == "T":
        unit = "T" * length
    elif motif in {"R", "REPEATS"}:
        unit = (MOTIF * 2)[:length]
    elif motif in {"M", "MH"}:
        unit = (MOTIF * 2)[:length]
    else:
        unit = "C" * length
    unit = unit[: int(d_payload)]
    source = str(context_source or "id83_proxy")
    if source == "id83_token_pair":
        left = _hash_token_code(["left", event, motif, str(length), str(aux)], d_context)
        right = _hash_token_code(["right", event, motif, str(length), str(aux)], d_context)
    elif source == "id83_repeat_context":
        left, right = repeated_context(unit, aux, d_context)
    elif source == "id83_payload_only":
        left, right = "", ""
    else:
        left, right = repeated_context(unit, aux, d_context)
        right = _id83_token_code([event, motif, str(length), str(aux)], d_context)
    if event == "DEL":
        return left, right, unit, ""
    if event == "INS":
        return left, right, "", unit
    raise ValueError(f"Unsupported ID83 event: {channel}")


def encode_channel_uga(
    channel: str,
    model: str | UGAModelSpec,
    atlas: dict[str, np.ndarray] | None = None,
    modality: str | None = None,
) -> np.ndarray:
    spec = get_uga_model(model)
    kind = channel_kind(channel)
    if kind == "SBS96":
        if atlas is None:
            ctx, alt = sbs_ctx_alt(channel)
            return assemble_uga_vector(ctx[0], ctx[2], ctx[1], alt, spec.d_context, spec.d_payload, spec.payload_schema)
        return variant_vec(channel, atlas, spec.d_context, "SBS", spec.d_payload, spec.payload_schema)
    if kind == "SBS1536":
        ch = str(channel)
        return assemble_uga_vector(
            ch[:2],
            ch[3:5],
            ch[2],
            ch[5],
            spec.d_context,
            spec.d_payload,
            spec.payload_schema,
        )
    if kind == "DBS78":
        ch = str(channel)
        if atlas is not None and (modality or "").upper() == "DBS":
            out = variant_vec(ch, atlas, spec.d_context, "DBS", spec.d_payload, spec.payload_schema)
            if out is not None:
                return out
        ref, alt = ch.split(">", 1) if ">" in ch else (ch[:2], ch[2:])
        return assemble_uga_vector("", "", ref, alt, spec.d_context, spec.d_payload, spec.payload_schema)
    if kind == "ID83":
        left, right, ref, alt = id83_proxy_event_for_source(
            channel,
            spec.d_context,
            spec.d_payload,
            spec.context_source,
        )
        return assemble_uga_vector(left, right, ref, alt, spec.d_context, spec.d_payload, spec.payload_schema)
    raise ValueError(f"Unsupported channel for UGA encoding: {channel}")


def build_uga_basis(
    channels: list[str],
    model: str | UGAModelSpec,
    atlas: dict[str, np.ndarray] | None = None,
    modality: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    spec = get_uga_model(model)
    expected_dim = universal_vector_dim(spec.d_context, spec.d_payload, spec.payload_schema)
    rows: list[np.ndarray] = []
    diagnostics: list[dict[str, object]] = []
    for channel in channels:
        try:
            vec = encode_channel_uga(channel, spec, atlas=atlas, modality=modality)
            if vec is None:
                raise ValueError("channel is not represented in the selected atlas")
            if len(vec) != expected_dim:
                raise ValueError(f"expected {expected_dim} dims, got {len(vec)}")
            rows.append(np.asarray(vec, dtype=np.float64))
            diagnostics.append({"Channel": channel, "UGA_Encoded": True, "Reason": ""})
        except Exception as exc:
            rows.append(np.zeros(expected_dim, dtype=np.float64))
            diagnostics.append({"Channel": channel, "UGA_Encoded": False, "Reason": str(exc)})
    basis = np.vstack(rows) if rows else np.zeros((0, expected_dim), dtype=np.float64)
    rounded = np.round(basis, 10)
    unique_count = int(np.unique(rounded, axis=0).shape[0]) if len(rounded) else 0
    diag = pd.DataFrame(diagnostics)
    diag["UGA_Model"] = spec.name
    diag["UGA_Dim"] = expected_dim
    diag["Unique_UGA_Vectors_In_Panel"] = unique_count
    diag["Collision_Count_In_Panel"] = len(channels) - unique_count
    return basis, diag


def build_channel_basis(
    channels: list[str],
    model: str | UGAModelSpec,
    atlas: dict[str, np.ndarray] | None = None,
    modality: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    basis, diag = build_uga_basis(channels, model, atlas=atlas, modality=modality)
    return basis, diag["UGA_Encoded"].to_numpy(dtype=bool)


def id_channel_uga_matrix(
    channels: list[str],
    d_context: int,
    d_payload: int,
    payload_schema: str = PAYLOAD_SCHEMA_MASKED,
) -> np.ndarray:
    spec = UGAModelSpec(
        name=f"id83_proxy_d{int(d_context)}_dp{int(d_payload)}",
        kinds=("ID83",),
        d_context=int(d_context),
        d_payload=int(d_payload),
        payload_schema=payload_schema,
        context_source="id83_proxy",
    )
    basis, diag = build_uga_basis([str(ch) for ch in channels], spec)
    if not bool(diag["UGA_Encoded"].all()):
        missing = diag.loc[~diag["UGA_Encoded"], "Channel"].head(8).tolist()
        raise ValueError(f"Could not encode {int((~diag['UGA_Encoded']).sum())} ID83 channels: {missing}")
    return basis.astype(np.float64)
