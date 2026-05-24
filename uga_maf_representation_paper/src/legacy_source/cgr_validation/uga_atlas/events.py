"""Observed genomic-context UGA encoders for event-level mutations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .context import GRCH37_FASTA
from .encoding import FastaReader, assemble_uga_vector, trim_shared_alleles, universal_vector_dim
from .models import UGAModelSpec, get_uga_model


BASES = set("ACGT")


@dataclass(frozen=True)
class ObservedUGAEvent:
    vector: np.ndarray
    encoded: bool
    reason: str
    chrom: str
    pos: int
    ref: str
    alt: str
    ref_payload: str
    alt_payload: str
    left_context: str
    right_context: str
    reference_match: bool


def clean_observed_allele(value: object) -> str:
    s = str(value or "").upper()
    if s in {"", "-", ".", "NAN", "NONE"}:
        return ""
    return "".join(ch for ch in s if ch in BASES)


def normalize_observed_chrom(value: object) -> str:
    chrom = str(value).upper().replace("CHR", "")
    aliases = {"23": "X", "24": "Y", "25": "MT", "M": "MT"}
    return aliases.get(chrom, chrom)


def _open_fasta(fasta: FastaReader | Path | str | None) -> tuple[FastaReader, bool]:
    if fasta is None:
        return FastaReader(GRCH37_FASTA), True
    if isinstance(fasta, FastaReader):
        return fasta, False
    return FastaReader(Path(fasta)), True


def encode_observed_variant_uga(
    chrom: object,
    pos: object,
    ref: object,
    alt: object,
    model: str | UGAModelSpec,
    *,
    fasta: FastaReader | Path | str | None = None,
) -> ObservedUGAEvent:
    """Encode a single observed event from its FASTA flanks and REF/ALT alleles."""
    spec = get_uga_model(model)
    if spec.context_source != "observed_context":
        raise ValueError(f"Model {spec.name} is not an observed-context event model")

    expected_dim = universal_vector_dim(spec.d_context, spec.d_payload, spec.payload_schema)
    chrom_s = normalize_observed_chrom(chrom)
    ref_s = clean_observed_allele(ref)
    alt_s = clean_observed_allele(alt)
    try:
        pos_i = int(float(pos))
    except (TypeError, ValueError):
        return ObservedUGAEvent(
            vector=np.zeros(expected_dim, dtype=np.float64),
            encoded=False,
            reason="invalid position",
            chrom=chrom_s,
            pos=0,
            ref=ref_s,
            alt=alt_s,
            ref_payload="",
            alt_payload="",
            left_context="",
            right_context="",
            reference_match=False,
        )

    if not ref_s and not alt_s:
        return ObservedUGAEvent(
            vector=np.zeros(expected_dim, dtype=np.float64),
            encoded=False,
            reason="missing REF and ALT",
            chrom=chrom_s,
            pos=pos_i,
            ref=ref_s,
            alt=alt_s,
            ref_payload="",
            alt_payload="",
            left_context="",
            right_context="",
            reference_match=False,
        )

    fasta_reader, should_close = _open_fasta(fasta)
    try:
        ref_len = max(1, len(ref_s))
        left = fasta_reader.fetch_range(chrom_s, pos_i - spec.d_context, pos_i - 1)
        right = fasta_reader.fetch_range(chrom_s, pos_i + ref_len, pos_i + ref_len + spec.d_context - 1)
        observed_ref = fasta_reader.fetch_range(chrom_s, pos_i, pos_i + ref_len - 1)
    finally:
        if should_close:
            fasta_reader.close()

    reference_match = bool(ref_s) and observed_ref[: len(ref_s)].upper() == ref_s
    ref_payload, alt_payload = trim_shared_alleles(ref_s, alt_s)
    vector = assemble_uga_vector(
        left,
        right,
        ref_s,
        alt_s,
        spec.d_context,
        spec.d_payload,
        spec.payload_schema,
    )
    reason = "" if reference_match else f"FASTA REF mismatch: observed {observed_ref}, event {ref_s}"
    return ObservedUGAEvent(
        vector=vector.astype(np.float64),
        encoded=True,
        reason=reason,
        chrom=chrom_s,
        pos=pos_i,
        ref=ref_s,
        alt=alt_s,
        ref_payload=ref_payload,
        alt_payload=alt_payload,
        left_context=left,
        right_context=right,
        reference_match=reference_match,
    )
