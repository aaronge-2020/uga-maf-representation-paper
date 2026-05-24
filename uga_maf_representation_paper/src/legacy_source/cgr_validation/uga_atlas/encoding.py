"""Master-spec UGA bit and event-vector encoders."""

from __future__ import annotations

import numpy as np

from vdkm.render import (
    CHROM_TO_ACCESSION,
    FastaReader,
    PAYLOAD_SCHEMA_LEGACY,
    PAYLOAD_SCHEMA_LENGTH,
    PAYLOAD_SCHEMA_MASKED,
    X_MAP,
    Y_MAP,
    encode_alt_cgr_walk,
    encode_bicgr_context,
    fast_seq_to_bits,
    payload_block_dim,
    payload_length_bit_count,
    revcomp,
    trim_shared_alleles,
    universal_vector_dim,
)
from vdkm.render import encode_variant_universal as _encode_variant_universal


def assemble_uga_vector(
    left_seq: str,
    right_seq: str,
    ref: str,
    alt: str,
    d_context: int,
    d_payload: int,
    payload_schema: str = PAYLOAD_SCHEMA_MASKED,
) -> np.ndarray:
    """Assemble [upstream context, REF payload, downstream context, ALT payload]."""
    d_context = int(d_context)
    d_payload = int(d_payload)
    context = encode_bicgr_context(left_seq, right_seq, d=d_context).astype(np.float64)
    left_block = context[: 2 * d_context]
    right_block = context[2 * d_context :]
    ref_payload, alt_payload = trim_shared_alleles(ref, alt)
    return np.concatenate(
        [
            left_block,
            encode_alt_cgr_walk(ref_payload, d=d_payload, payload_schema=payload_schema).astype(np.float64),
            right_block,
            encode_alt_cgr_walk(alt_payload, d=d_payload, payload_schema=payload_schema).astype(np.float64),
        ]
    )


def encode_variant_universal(
    ref_context_45mer: str,
    alt_sequence: str,
    d: int = 13,
    *,
    d_context: int | None = None,
    d_payload: int | None = None,
    ref_allele: str | None = None,
    canonicalize: bool = True,
    payload_schema: str = PAYLOAD_SCHEMA_MASKED,
) -> np.ndarray:
    """Encode a locus-level variant using the selected UGA payload schema."""
    return _encode_variant_universal(
        ref_context_45mer,
        alt_sequence,
        d=d,
        d_context=d_context,
        d_payload=d_payload,
        ref_allele=ref_allele,
        canonicalize=canonicalize,
        payload_schema=payload_schema,
    )


def universal_context_slice(
    arr: np.ndarray,
    d_context: int,
    d_payload: int,
    payload_schema: str = PAYLOAD_SCHEMA_MASKED,
) -> np.ndarray:
    """Extract [XL, YL, XR, YR] from a UGA vector or matrix."""
    d_context = int(d_context)
    payload_width = payload_block_dim(d_payload, payload_schema)
    left_end = 2 * d_context
    right_start = left_end + payload_width
    right_end = right_start + 2 * d_context
    x = np.asarray(arr)
    if x.ndim == 1:
        return np.concatenate([x[:left_end], x[right_start:right_end]])
    return np.column_stack([x[:, :left_end], x[:, right_start:right_end]])
