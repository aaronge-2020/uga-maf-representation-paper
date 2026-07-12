"""MuAt-compatible preprocessing, model, and official CLI helpers.

This module intentionally separates two concepts:

* "MuAt" means the official package/checkpoints are configured and used.
* "MuAt-compatible" means local code follows the paper's data modalities and
  supplement architecture closely enough for manuscript-side smoke testing.
"""

from __future__ import annotations

import gc
import gzip
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
DNA = set("ACGT")
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")

MAF_COMPATIBLE_USECOLS = [
    "Hugo_Symbol",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Variant_Type",
    "Variant_Classification",
    "Reference_Allele",
    "Tumor_Seq_Allele1",
    "Tumor_Seq_Allele2",
    "Tumor_Sample_Barcode",
    "CONTEXT",
    "STRAND",
    "Strand",
    "EXON",
    "Exon_Number",
    "Feature_type",
    "BIOTYPE",
    "Consequence",
]


def stable_sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def reverse_complement(seq: str) -> str:
    return str(seq or "").upper().translate(COMPLEMENT)[::-1]


def clean_allele(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"", ".", "-", "NAN", "NONE", "NULL"}:
        return ""
    return text


def choose_alt(ref: object, allele1: object, allele2: object) -> str:
    ref_text = clean_allele(ref)
    alleles = [clean_allele(allele1), clean_allele(allele2)]
    for allele in alleles:
        if allele and allele != ref_text:
            return allele
    return alleles[-1] if alleles else ""


def normalize_chromosome(value: object) -> str:
    text = str(value or "NA").strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    return text.upper() if text.upper() in {"X", "Y", "M", "MT"} else text


def position_bin_token(chromosome: object, position: object, *, bin_size: int = 1_000_000) -> str:
    chrom = normalize_chromosome(chromosome)
    try:
        pos = max(0, int(float(position)))
    except (TypeError, ValueError):
        pos = 0
    return f"chr{chrom}_{pos // int(bin_size)}"


def _context_flanks(context: object) -> tuple[str, str, str]:
    seq = str(context or "").strip().upper()
    if not seq or any(base not in DNA for base in seq):
        return "N", "N", "N"
    center = len(seq) // 2
    left = seq[center - 1] if center > 0 else "N"
    mid = seq[center] if center < len(seq) else "N"
    right = seq[center + 1] if center + 1 < len(seq) else "N"
    return left, mid, right


def mutation_motif_token(row: Mapping[str, object]) -> str:
    """Return a paper-style local mutation motif token from an MC3/MAF row.

    SNVs are canonicalized to pyrimidine reference alleles where possible. MNV
    and indel tokens use the same three-position local window but keep explicit
    alleles because TCGA MC3 does not contain all MuAt WGS event classes.
    """

    ref = clean_allele(row.get("Reference_Allele"))
    alt = choose_alt(ref, row.get("Tumor_Seq_Allele1"), row.get("Tumor_Seq_Allele2"))
    variant_type = str(row.get("Variant_Type") or "").upper()
    left, mid, right = _context_flanks(row.get("CONTEXT"))

    if len(ref) == 1 and len(alt) == 1 and set(ref + alt).issubset(DNA):
        if ref in {"A", "G"}:
            ref = reverse_complement(ref)
            alt = reverse_complement(alt)
            tri = reverse_complement(left + mid + right)
            left, mid, right = tri[0], tri[1], tri[2]
        return f"{left}[{ref}>{alt}]{right}"

    if "INS" in variant_type or (len(alt) > len(ref) and alt):
        inserted = alt if not ref or not alt.startswith(ref) else alt[len(ref) :]
        inserted = inserted or alt or "N"
        return f"{left}[ins{inserted[:12]}]{right}"

    if "DEL" in variant_type or (len(ref) > len(alt) and ref):
        deleted = ref if not alt or not ref.startswith(alt) else ref[len(alt) :]
        deleted = deleted or ref or "N"
        return f"{left}[del{deleted[:12]}]{right}"

    if len(ref) > 1 or len(alt) > 1:
        return f"{left}[{(ref or 'N')[:12]}>{(alt or 'N')[:12]}]{right}"

    return f"{left}[{ref or 'N'}>{alt or 'N'}]{right}"


def muat_symbolic_motif_token(row: Mapping[str, object]) -> str:
    """Return a bounded, MuAt-style three-symbol mutation window.

    MuAt encodes every variant as a sequence s in Sigma^3, where Sigma is {A,C,G,T} plus a set of
    mutation symbols M: six pyrimidine-referenced substitutions, single-base deletions of A/C/G/T,
    single-base insertions of A/C/G/T, plus SV breakpoint and MEI symbols that whole-exome data
    cannot supply. Their examples: ApCpG>ApTpG is ``A[C>T]G``; a diadenine deletion preceded by a
    cytosine is ``C[del A][del A]``.

    ``mutation_motif_token`` (the legacy grammar) instead emits one token per distinct allele
    string of up to 12 characters, so a 12-base substitution becomes a single motif and the
    vocabulary is unbounded and data-dependent (1,464 tokens on 772 HRD samples).

    This function bounds the vocabulary by construction, as MuAt does. Deviation to note: our WES
    alphabet excludes SV/MEI symbols, and we take the first two mutated positions of an MNV or
    indel rather than enumerating every window MuAt's grammar admits, so the reachable vocabulary
    is smaller than their 3,692 tokens.
    """

    ref = clean_allele(row.get("Reference_Allele"))
    alt = choose_alt(ref, row.get("Tumor_Seq_Allele1"), row.get("Tumor_Seq_Allele2"))
    variant_type = str(row.get("Variant_Type") or "").upper()
    left, mid, right = _context_flanks(row.get("CONTEXT"))

    def substitution_symbol(ref_base: str, alt_base: str) -> str:
        """Pyrimidine-referenced substitution symbol, or the base itself if unchanged."""
        if ref_base not in DNA or alt_base not in DNA:
            return "N"
        if ref_base == alt_base:
            return ref_base
        if ref_base in {"A", "G"}:
            ref_base = reverse_complement(ref_base)
            alt_base = reverse_complement(alt_base)
        return f"{ref_base}>{alt_base}"

    # Single-base substitution: canonicalise the whole trinucleotide to a pyrimidine reference.
    if len(ref) == 1 and len(alt) == 1 and set(ref + alt).issubset(DNA):
        if ref in {"A", "G"}:
            tri = reverse_complement(left + mid + right)
            left, right = tri[0], tri[2]
            ref, alt = reverse_complement(ref), reverse_complement(alt)
        return f"{left}[{ref}>{alt}]{right}"

    is_insertion = "INS" in variant_type or (bool(alt) and len(alt) > len(ref))
    is_deletion = "DEL" in variant_type or (bool(ref) and len(ref) > len(alt))

    if is_insertion:
        inserted = alt[len(ref):] if ref and alt.startswith(ref) else alt
        inserted = "".join(base for base in inserted if base in DNA) or "N"
        first = inserted[0]
        if len(inserted) >= 2:
            return f"{left}[ins{first}][ins{inserted[1]}]"
        return f"{left}[ins{first}]{right}"

    if is_deletion:
        deleted = ref[len(alt):] if alt and ref.startswith(alt) else ref
        deleted = "".join(base for base in deleted if base in DNA) or "N"
        first = deleted[0]
        if len(deleted) >= 2:
            return f"{left}[del{first}][del{deleted[1]}]"
        return f"{left}[del{first}]{right}"

    # Multi-nucleotide substitution: two adjacent substitution symbols, as in MuAt's Sigma^3.
    if len(ref) > 1 and len(ref) == len(alt):
        first = substitution_symbol(ref[0], alt[0])
        second = substitution_symbol(ref[1], alt[1])
        return f"{left}[{first}][{second}]"

    return f"{left}[{(ref or 'N')[0]}>{(alt or 'N')[0]}]{right}"


def transcriptional_strand_class(row: Mapping[str, object]) -> str:
    """Return MuAt's four-class transcriptional-strand attribute.

    MuAt: "we categorize each mutation into one of four mutually exclusive classes (strand):
    mutation's pyrimidine reference base is on the (1) same or (2) opposite strand as a gene, or
    (3) mutation overlaps two genes on opposite strands, or (4) mutation is intergenic."

    The legacy ``annotation_token`` instead read the MAF ``STRAND`` field straight through as
    +/-/0/na, which has the same cardinality but an entirely different meaning: it records the
    gene's orientation, not the orientation of the pyrimidine reference base relative to the gene.
    That destroys the transcriptional strand asymmetry the attribute exists to capture.

    MC3 gives one row per variant with a single Hugo_Symbol, so class (3) is not recoverable; we
    emit ``unknown`` in its place and keep the cardinality at four.
    """

    classification = str(row.get("Variant_Classification") or row.get("Consequence") or "").upper()
    gene = str(row.get("Hugo_Symbol") or "").strip()
    if not gene or gene in {".", "NA", "UNKNOWN", "NONE"} or "IGR" in classification or "INTERGENIC" in classification:
        return "intergenic"

    raw_strand = row.get("STRAND") if row.get("STRAND") not in (None, "") else row.get("Strand")
    text = str(raw_strand or "").strip()
    if text in {"1", "+1", "+"}:
        gene_strand = 1
    elif text in {"-1", "-"}:
        gene_strand = -1
    else:
        return "unknown"

    ref = clean_allele(row.get("Reference_Allele"))
    base = ref[0] if ref else ""
    if base not in DNA:
        return "unknown"

    # MAF reference alleles are reported on the plus strand. If the reference base is a pyrimidine
    # (C or T) the pyrimidine sits on the plus strand; otherwise it sits on the minus strand.
    pyrimidine_strand = 1 if base in {"C", "T"} else -1
    return "same" if pyrimidine_strand == gene_strand else "opposite"


def annotation_token(row: Mapping[str, object]) -> str:
    """Return the MuAt genic/exonic/strand annotation token."""

    gene = str(row.get("Hugo_Symbol") or "").strip()
    biotype = str(row.get("BIOTYPE") or "").strip().lower()
    genic = "yes" if gene and gene not in {".", "NA", "Unknown", "UNKNOWN"} else "no"
    if biotype in {"intergenic", "antisense", "unknown"}:
        genic = "no"

    exon_values = [row.get("EXON"), row.get("Exon_Number")]
    classification = str(row.get("Variant_Classification") or row.get("Consequence") or "").lower()
    exonic = "yes" if any(str(value or "").strip() not in {"", ".", "nan", "None"} for value in exon_values) else "no"
    if any(word in classification for word in ["intron", "intergenic", "upstream", "downstream"]):
        exonic = "no"

    raw_strand = str(row.get("STRAND") if row.get("STRAND") not in [None, ""] else row.get("Strand") or "").strip()
    strand_map = {"1": "+", "+1": "+", "+": "+", "-1": "-", "-": "-", "0": "0", ".": "na", "": "na"}
    strand = strand_map.get(raw_strand, "na")
    return f"{genic}_{exonic}_{strand}"


LEGACY_STRAND_CLASSES = ("+", "-", "0", "na")
MUAT_STRAND_CLASSES = ("same", "opposite", "intergenic", "unknown")


def muat_annotation_token(row: Mapping[str, object]) -> str:
    """Genic x exonic x transcriptional-strand annotation token (MuAt's 2 x 2 x 4)."""

    legacy = annotation_token(row)
    genic, exonic, _legacy_strand = legacy.split("_", 2)
    return f"{genic}_{exonic}_{transcriptional_strand_class(row)}"


def motif_token_for(row: Mapping[str, object], grammar: str = "legacy") -> str:
    grammar = str(grammar or "legacy").strip().lower()
    if grammar in {"muat", "muat_symbolic", "symbolic"}:
        return muat_symbolic_motif_token(row)
    return mutation_motif_token(row)


def annotation_token_for(row: Mapping[str, object], strand_mode: str = "legacy") -> str:
    strand_mode = str(strand_mode or "legacy").strip().lower()
    if strand_mode in {"transcriptional", "muat"}:
        return muat_annotation_token(row)
    return annotation_token(row)


def maf_header_columns(maf_path: str | Path) -> list[str]:
    path = Path(maf_path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            return line.rstrip("\n").split("\t")
    return []


def collect_maf_patients(maf_path: str | Path, *, chunksize: int = 500_000) -> set[str]:
    patients: set[str] = set()
    for chunk in pd.read_csv(maf_path, sep="\t", usecols=["Tumor_Sample_Barcode"], dtype=str, chunksize=chunksize):
        patients.update(chunk["Tumor_Sample_Barcode"].astype(str).str[:12].dropna().unique())
    return patients


def build_muat_event_table(
    maf_path: str | Path,
    patients: Sequence[str],
    *,
    chunksize: int = 250_000,
    motif_grammar: str = "legacy",
    annotation_strand: str = "legacy",
) -> pd.DataFrame:
    """Convert MC3/MAF rows into MuAt-compatible mutation-event rows.

    ``motif_grammar`` and ``annotation_strand`` select the input representation:

    * ``legacy`` reproduces the tokens the manuscript results were produced with: one motif token
      per distinct allele string of up to 12 characters (unbounded vocabulary), and the MAF
      ``STRAND`` field passed through as +/-/0/na.
    * ``muat_symbolic`` / ``transcriptional`` follow the paper: a bounded three-symbol mutation
      window, and the four-class transcriptional-strand attribute defined relative to the
      pyrimidine reference base.

    Both are held as options rather than one replacing the other, because the tokenisation is
    itself a representation choice and the difference between them is a measurable result.
    """

    patient_set = set(map(str, patients))
    available = set(maf_header_columns(maf_path))
    usecols = [column for column in MAF_COMPATIBLE_USECOLS if column in available]
    if "Tumor_Sample_Barcode" not in usecols:
        raise ValueError("MAF input is missing Tumor_Sample_Barcode")

    # One DataFrame per chunk rather than one giant list of dicts. MC3 yields ~3.4M events, and a
    # list of 3.4M Python dicts costs several GB of pure interpreter overhead before it is ever
    # converted. Materialising each chunk immediately bounds that cost to one chunk at a time.
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(maf_path, sep="\t", usecols=usecols, dtype=str, chunksize=chunksize):
        chunk["sample"] = chunk["Tumor_Sample_Barcode"].astype(str).str[:12]
        chunk = chunk[chunk["sample"].isin(patient_set)].copy()
        if chunk.empty:
            continue
        chunk["position"] = pd.to_numeric(chunk.get("Start_Position"), errors="coerce").fillna(0).astype(np.int64)
        chunk = chunk.sort_values(["sample", "Chromosome", "position"], kind="mergesort")
        chunk_rows: list[dict[str, object]] = []
        for record in chunk.to_dict(orient="records"):
            ref = clean_allele(record.get("Reference_Allele"))
            alt = choose_alt(ref, record.get("Tumor_Seq_Allele1"), record.get("Tumor_Seq_Allele2"))
            chunk_rows.append(
                {
                    "sample": str(record.get("sample")),
                    "chromosome": normalize_chromosome(record.get("Chromosome")),
                    "position": int(record.get("position") or 0),
                    "motif": motif_token_for(record, motif_grammar),
                    "position_token": position_bin_token(record.get("Chromosome"), record.get("position")),
                    "annotation": annotation_token_for(record, annotation_strand),
                    "variant_type": str(record.get("Variant_Type") or ""),
                    "variant_classification": str(record.get("Variant_Classification") or ""),
                    "gene": str(record.get("Hugo_Symbol") or ""),
                    "ref": ref,
                    "alt": alt,
                    "context": str(record.get("CONTEXT") or ""),
                }
            )
        if chunk_rows:
            frames.append(pd.DataFrame(chunk_rows))
        del chunk, chunk_rows
        gc.collect()
    if not frames:
        return pd.DataFrame(columns=["sample", "chromosome", "position", "motif", "position_token", "annotation"])
    events = pd.concat(frames, ignore_index=True, copy=False)
    del frames
    gc.collect()
    return events.sort_values(["sample", "chromosome", "position", "motif"], kind="mergesort").reset_index(drop=True)


# MuAt's three modalities. The architecture was hardcoded to exactly these, which is why an
# extended event representation had nowhere to go: Bio MAF v4's driver, hotspot and pathway
# annotations could only ever enter the benchmark as tabular features, never as event tokens.
# Note that gene identity is not among them -- MuAt does not use it.
CORE_MODALITIES = ("motif", "position", "annotation")

# Column in the event table that each modality reads from.
MODALITY_COLUMNS = {
    "motif": "motif",
    "position": "position_token",
    "annotation": "annotation",
    "gene": "gene_token",
    "consequence": "consequence_token",
    "impact": "impact_token",
}


@dataclass(frozen=True)
class MuAtDictionaries:
    """Token dictionaries, one per categorical event modality.

    ``extra`` holds any modality beyond MuAt's three. It is ordered, and the order defines the
    column order of the encoded event tensor.
    """

    motif: dict[str, int]
    position: dict[str, int]
    annotation: dict[str, int]
    extra: dict[str, dict[str, int]] = field(default_factory=dict)

    def ordered(self) -> list[tuple[str, dict[str, int]]]:
        core = [("motif", self.motif), ("position", self.position), ("annotation", self.annotation)]
        return core + [(name, mapping) for name, mapping in self.extra.items()]

    def names(self) -> list[str]:
        return [name for name, _ in self.ordered()]

    def sizes(self) -> dict[str, int]:
        return {name: len(mapping) for name, mapping in self.ordered()}

    def to_jsonable(self) -> dict[str, dict[str, int]]:
        return {name: mapping for name, mapping in self.ordered()}

    @classmethod
    def from_jsonable(cls, payload: Mapping[str, Mapping[str, int]]) -> "MuAtDictionaries":
        def coerce(mapping: Mapping[str, int]) -> dict[str, int]:
            return {str(k): int(v) for k, v in mapping.items()}

        extra = {name: coerce(mapping) for name, mapping in payload.items() if name not in CORE_MODALITIES}
        return cls(
            motif=coerce(payload["motif"]),
            position=coerce(payload["position"]),
            annotation=coerce(payload["annotation"]),
            extra=extra,
        )


def _dictionary(values: Iterable[object]) -> dict[str, int]:
    tokens = [PAD_TOKEN, UNK_TOKEN]
    tokens.extend(sorted({str(value) for value in values if str(value) not in {PAD_TOKEN, UNK_TOKEN}}))
    return {token: i for i, token in enumerate(tokens)}


def hash_token(namespace: str, token: object, n_buckets: int) -> str:
    digest = hashlib.sha256(f"{namespace}::{token}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % int(n_buckets)
    return f"{namespace}_hash_{bucket:04d}"


def fixed_hash_dictionary(namespace: str, n_buckets: int) -> dict[str, int]:
    tokens = [PAD_TOKEN, UNK_TOKEN]
    tokens.extend(f"{namespace}_hash_{idx:04d}" for idx in range(int(n_buckets)))
    return {token: i for i, token in enumerate(tokens)}


def build_fixed_hash_dictionaries(*, motif_buckets: int = 1024, position_buckets: int = 2048, annotation_buckets: int = 256) -> MuAtDictionaries:
    return MuAtDictionaries(
        motif=fixed_hash_dictionary("motif", int(motif_buckets)),
        position=fixed_hash_dictionary("position", int(position_buckets)),
        annotation=fixed_hash_dictionary("annotation", int(annotation_buckets)),
    )


# GRCh37/hg19 primary-assembly sequence lengths. These are a property of the reference build,
# not of any cohort, so enumerating position bins from them introduces no dependence on which
# tumours are in a training or held-out fold.
GRCH37_CHROMOSOME_LENGTHS: dict[str, int] = {
    "1": 249250621, "2": 243199373, "3": 198022430, "4": 191154276, "5": 180915260,
    "6": 171115067, "7": 159138663, "8": 146364022, "9": 141213431, "10": 135534747,
    "11": 135006516, "12": 133851895, "13": 115169878, "14": 107349540, "15": 102531392,
    "16": 90354753, "17": 81195210, "18": 78077248, "19": 59128983, "20": 63025520,
    "21": 48129895, "22": 51304566, "X": 155270560, "Y": 59373566, "MT": 16569,
}


def enumerate_position_tokens(*, bin_size: int = 1_000_000) -> list[str]:
    """Enumerate every 1-Mb position-bin token in GRCh37.

    MuAt uses an exact one-hot dictionary of 2,915 position tokens covering all 1-Mb genomic
    bins (Sanjaya et al. 2023, "Preparing MuAt inputs from somatic variant callsets"). The bins
    are a function of the reference assembly alone, so they can be enumerated up front.
    """

    tokens: list[str] = []
    for chromosome, length in GRCH37_CHROMOSOME_LENGTHS.items():
        n_bins = int(math.ceil(int(length) / int(bin_size)))
        tokens.extend(f"chr{chromosome}_{index}" for index in range(n_bins))
    return tokens


def enumerate_annotation_tokens(*, strand_mode: str = "legacy") -> list[str]:
    """Enumerate the 16 genic x exonic x strand annotation values.

    MuAt's annotation dictionary contains 2 x 2 x 4 = 16 values. Both strand vocabularies here
    have cardinality four, so the product is 16 either way.
    """

    strand_mode = str(strand_mode or "legacy").strip().lower()
    strands = MUAT_STRAND_CLASSES if strand_mode in {"transcriptional", "muat"} else LEGACY_STRAND_CLASSES
    return [
        f"{genic}_{exonic}_{strand}"
        for genic in ("yes", "no")
        for exonic in ("yes", "no")
        for strand in strands
    ]


def build_exact_dictionaries(events: pd.DataFrame, *, bin_size: int = 1_000_000, strand_mode: str = "legacy") -> MuAtDictionaries:
    """Build collision-free token dictionaries.

    This replaces ``build_fixed_hash_dictionaries``, which hashed tokens into fewer buckets than
    the vocabularies actually contain (1,024 motif buckets against a TCGA-WES motif space of
    ~3,400; 2,048 position buckets against ~3,100 genomic bins). That guaranteed collisions:
    unrelated mutation motifs, and unrelated genomic regions, were forced to share a single
    embedding vector.

    Hashing was introduced to prevent held-out samples from influencing the token dictionary.
    That risk does not apply here:

    * position tokens are enumerated from the GRCh37 assembly;
    * annotation tokens are enumerated from the genic/exonic/strand grammar;
    * motif tokens are collected from the event table without reference to any label.

    Motifs observed only in a held-out fold receive a randomly initialised embedding that no
    gradient ever reaches, which is behaviourally identical to mapping them to ``<UNK>``. No
    label information crosses the fold boundary.
    """

    return MuAtDictionaries(
        motif=_dictionary(events.get("motif", pd.Series(dtype=str)).astype(str)),
        position=_dictionary(
            list(enumerate_position_tokens(bin_size=int(bin_size)))
            + list(events.get("position_token", pd.Series(dtype=str)).astype(str))
        ),
        annotation=_dictionary(
            list(enumerate_annotation_tokens(strand_mode=strand_mode))
            + list(events.get("annotation", pd.Series(dtype=str)).astype(str))
        ),
    )


def hash_muat_event_tokens(
    events: pd.DataFrame,
    *,
    motif_buckets: int = 1024,
    position_buckets: int = 2048,
    annotation_buckets: int = 256,
) -> pd.DataFrame:
    out = events.copy()
    out["motif"] = out.get("motif", pd.Series(dtype=str)).astype(str).map(lambda value: hash_token("motif", value, int(motif_buckets)))
    out["position_token"] = out.get("position_token", pd.Series(dtype=str)).astype(str).map(lambda value: hash_token("position", value, int(position_buckets)))
    out["annotation"] = out.get("annotation", pd.Series(dtype=str)).astype(str).map(lambda value: hash_token("annotation", value, int(annotation_buckets)))
    return out


def build_dictionaries(events: pd.DataFrame) -> MuAtDictionaries:
    return MuAtDictionaries(
        motif=_dictionary(events.get("motif", pd.Series(dtype=str)).astype(str)),
        position=_dictionary(events.get("position_token", pd.Series(dtype=str)).astype(str)),
        annotation=_dictionary(events.get("annotation", pd.Series(dtype=str)).astype(str)),
    )


def dictionary_summary(dictionaries: MuAtDictionaries) -> pd.DataFrame:
    rows = []
    for modality, mapping in dictionaries.to_jsonable().items():
        rows.append(
            {
                "modality": modality,
                "n_tokens": len(mapping),
                "pad_index": mapping[PAD_TOKEN],
                "unk_index": mapping[UNK_TOKEN],
                "example_tokens": ";".join(list(mapping)[:8]),
            }
        )
    return pd.DataFrame(rows)


def encode_events(
    events: pd.DataFrame,
    patients: Sequence[str],
    dictionaries: MuAtDictionaries,
    *,
    max_events: int,
    seed: int = 20260524,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode event rows as integer modality tensors plus a valid-event mask.

    Tumours carrying more than ``max_events`` mutations are down-sampled with a deterministic,
    per-sample RNG. The previous behaviour took ``head(max_events)`` after sorting by
    ``chromosome``; because that column holds strings, the sort was lexicographic
    (1, 10, 11, ... 19, 2, 20, 21, 22, 3, 4, ... 9, X, Y), so truncation systematically discarded
    chromosomes 3-9, X and Y rather than sampling uniformly. This only affects tumours above the
    cap -- which are precisely the POLE-proofreading and MSI hypermutators, whose regional
    mutation distribution carries the most tumour-type signal.
    """

    patient_list = [str(patient) for patient in patients]
    patient_to_row = {patient: i for i, patient in enumerate(patient_list)}
    # int32 rather than int64: every vocabulary here is far below 2^31, and this tensor is the
    # single largest allocation in the run. For cancer_type_top20 (8,800 x 5,000 x 3) it is
    # 528 MB instead of 1.06 GB, on host and again on device when preload_tensors_to_device is on.
    # torch.nn.Embedding accepts IntTensor indices.
    x = np.zeros((len(patient_list), int(max_events), 3), dtype=np.int32)
    mask = np.zeros((len(patient_list), int(max_events)), dtype=bool)
    counts = np.zeros(len(patient_list), dtype=np.int32)
    if events.empty:
        return x, mask, counts
    dicts = dictionaries.to_jsonable()
    n_truncated = 0
    for sample, sample_events in events.groupby("sample", sort=False):
        sample = str(sample)
        if sample not in patient_to_row:
            continue
        row_idx = patient_to_row[sample]
        ordered = sample_events.sort_values(["chromosome", "position", "motif"], kind="mergesort")
        if len(ordered) > int(max_events):
            digest = hashlib.sha256(f"{int(seed)}::{sample}".encode("utf-8")).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            keep = np.sort(rng.choice(len(ordered), size=int(max_events), replace=False))
            limited = ordered.iloc[keep]
            n_truncated += 1
        else:
            limited = ordered
        for event_idx, record in enumerate(limited.to_dict(orient="records")):
            x[row_idx, event_idx, 0] = dicts["motif"].get(str(record.get("motif")), dicts["motif"][UNK_TOKEN])
            x[row_idx, event_idx, 1] = dicts["position"].get(str(record.get("position_token")), dicts["position"][UNK_TOKEN])
            x[row_idx, event_idx, 2] = dicts["annotation"].get(str(record.get("annotation")), dicts["annotation"][UNK_TOKEN])
            mask[row_idx, event_idx] = True
            counts[row_idx] += 1
    if n_truncated:
        print(
            f"[muat_compatible] {n_truncated} tumour(s) exceeded max_events={int(max_events)} "
            f"and were down-sampled with a seeded per-sample RNG (seed={int(seed)})",
            flush=True,
        )
    return x, mask, counts


def load_tcga_20_type_labels(
    mc3_dir: str | Path,
    *,
    maf_patients: set[str] | None = None,
    min_count: int = 100,
    max_types: int = 20,
) -> tuple[pd.Series, pd.DataFrame]:
    """Build the paper-facing TCGA WES tumour-type task from bundled MC3 data."""

    clinical_path = Path(mc3_dir) / "raw" / "clinical_PANCAN_patient_with_followup.tsv"
    clinical = pd.read_csv(
        clinical_path,
        sep="\t",
        usecols=["bcr_patient_barcode", "acronym"],
        dtype=str,
        encoding="latin1",
    ).dropna()
    clinical = clinical.drop_duplicates("bcr_patient_barcode")
    if maf_patients is not None:
        clinical = clinical[clinical["bcr_patient_barcode"].isin(maf_patients)].copy()
    counts = clinical["acronym"].value_counts()
    eligible = counts[counts > int(min_count)]
    selected_types = list(eligible.head(int(max_types)).index)
    labels = clinical[clinical["acronym"].isin(selected_types)].set_index("bcr_patient_barcode")["acronym"].sort_index()
    audit_rows = []
    for tumour_type, n in counts.items():
        if tumour_type in selected_types:
            status = "included"
            reason = ""
        elif n > int(min_count):
            status = "excluded"
            reason = f"eligible_but_outside_top_{max_types}"
        else:
            status = "excluded"
            reason = f"n_le_{min_count}"
        audit_rows.append({"tumour_type": tumour_type, "n_patients": int(n), "status": status, "reason": reason})
    audit = pd.DataFrame(audit_rows)
    return labels, audit


def sample_smoke_labels(labels: pd.Series, *, per_class: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for _label, group in labels.groupby(labels, sort=True):
        values = np.array(group.index.astype(str))
        if len(values) <= int(per_class):
            chosen = values
        else:
            chosen = rng.choice(values, size=int(per_class), replace=False)
        selected.extend(map(str, chosen))
    return labels.loc[sorted(selected)]


def topk_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    if proba.size == 0:
        return float("nan")
    top = np.argsort(-proba, axis=1)[:, : min(int(k), proba.shape[1])]
    return float(np.mean([int(y_true[i]) in top[i] for i in range(len(y_true))]))


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, *, bins: int = 10) -> dict[str, float]:
    pred = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    used = 0
    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (conf >= left) & (conf < right if right < 1.0 else conf <= right)
        if not np.any(in_bin):
            continue
        used += 1
        ece += float(np.mean(in_bin)) * abs(float(np.mean(correct[in_bin])) - float(np.mean(conf[in_bin])))
    return {
        "ece": float(ece),
        "mean_confidence": float(np.mean(conf)) if len(conf) else float("nan"),
        "accuracy": float(np.mean(correct)) if len(correct) else float("nan"),
        "n_bins_used": int(used),
    }


class OfficialMuAtCLI:
    """Small wrapper for the official ``muat`` command-line interface."""

    def __init__(self, executable: str | None = None):
        self.executable = executable or shutil.which("muat")

    @property
    def available(self) -> bool:
        return bool(self.executable)

    def version(self) -> dict[str, object]:
        if not self.available:
            return {"available": False, "executable": None, "version": None}
        assert self.executable is not None
        for args in ([self.executable, "--version"], [self.executable, "-h"]):
            try:
                completed = subprocess.run(args, capture_output=True, text=True, timeout=20, check=False)
                text = (completed.stdout or completed.stderr or "").strip().splitlines()
                if text:
                    return {"available": True, "executable": self.executable, "version": text[0], "returncode": completed.returncode}
            except Exception as exc:  # pragma: no cover - defensive environment reporting
                return {"available": True, "executable": self.executable, "version": None, "error": str(exc)}
        return {"available": True, "executable": self.executable, "version": None}

    def preprocess_command(self, *, input_filepath: str | Path, result_dir: str | Path, reference_build: str, reference_fasta: str | Path) -> list[str]:
        exe = self.executable or "muat"
        return [
            exe,
            "preprocess",
            f"--{reference_build.lower()}",
            str(reference_fasta),
            "--input-filepath",
            str(input_filepath),
            "--result-dir",
            str(result_dir),
        ]

    def predict_pretrained_command(
        self,
        *,
        assay: str,
        mutation_type: str,
        input_filepath: str | Path,
        result_dir: str | Path,
        reference_build: str | None = None,
        reference_fasta: str | Path | None = None,
        ensemble: bool = False,
    ) -> list[str]:
        exe = self.executable or "muat"
        command = [exe, "predict-ensemble" if ensemble else "predict", "pretrained", assay, "--mutation-type", mutation_type]
        if reference_build and reference_fasta:
            command.extend([f"--{reference_build.lower()}", str(reference_fasta)])
        command.extend(["--input-filepath", str(input_filepath), "--result-dir", str(result_dir)])
        return command

    def train_command(self, *, train_filepath: str | Path, result_dir: str | Path, config_filepath: str | Path | None = None, extra_args: Sequence[str] = ()) -> list[str]:
        exe = self.executable or "muat"
        command = [exe, "train", "--input-filepath", str(train_filepath), "--result-dir", str(result_dir)]
        if config_filepath is not None:
            command.extend(["--config", str(config_filepath)])
        command.extend(map(str, extra_args))
        return command

    @staticmethod
    def command_record(command: Sequence[str], *, input_paths: Sequence[str | Path] = ()) -> dict[str, object]:
        fingerprints = []
        for path_like in input_paths:
            path = Path(path_like)
            if path.exists():
                stat = path.stat()
                fingerprints.append({"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
            else:
                fingerprints.append({"path": str(path), "missing": True})
        return {"command": list(map(str, command)), "fingerprint": stable_sha256({"command": list(command), "inputs": fingerprints}), "inputs": fingerprints}


try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - import environment dependent
    torch = None
    nn = None


if nn is not None:

    class MuAtCompatibleModel(nn.Module):
        """Supplement-faithful local MuAt-compatible architecture."""

        def __init__(
            self,
            *,
            n_outputs: int,
            motif_vocab_size: int | None = None,
            position_vocab_size: int | None = None,
            annotation_vocab_size: int | None = None,
            modality_vocab_sizes: Sequence[int] | None = None,
            numeric_features: int = 0,
            embed_dim: int = 128,
            attention_heads: int = 1,
            num_layers: int = 1,
            feature_dim: int = 24,
            dropout: float = 0.1,
            count_feature_mode: str = "none",
            pooling_mode: str = "attention_weighted",
        ) -> None:
            """Mutation-attention over a bag of events.

            MuAt uses exactly three modalities (motif, 1-Mb position, genic/exonic/strand), and
            this class was hardcoded to them: three named embeddings and a fixed width of
            3 * embed_dim. An extended event representation therefore had nowhere to go.

            ``modality_vocab_sizes`` generalises that to any number of categorical token streams,
            and ``numeric_features`` adds continuous per-event channels (variant allele fraction,
            read depth) through a learned projection into one further embed_dim-wide block. Passing
            the three named vocab sizes reproduces the original architecture exactly.
            """

            super().__init__()
            if modality_vocab_sizes is None:
                if motif_vocab_size is None or position_vocab_size is None or annotation_vocab_size is None:
                    raise ValueError("Provide modality_vocab_sizes, or all three of the named MuAt vocab sizes")
                modality_vocab_sizes = [int(motif_vocab_size), int(position_vocab_size), int(annotation_vocab_size)]
            modality_vocab_sizes = [int(size) for size in modality_vocab_sizes]
            if not modality_vocab_sizes:
                raise ValueError("At least one categorical modality is required")
            numeric_features = max(0, int(numeric_features))

            n_blocks = len(modality_vocab_sizes) + (1 if numeric_features else 0)
            width = int(embed_dim) * n_blocks
            if width % int(attention_heads) != 0:
                raise ValueError(
                    f"{n_blocks} * embed_dim ({width}) must be divisible by attention_heads ({attention_heads})"
                )
            self.modality_vocab_sizes = list(modality_vocab_sizes)
            self.numeric_features = numeric_features
            num_layers = max(1, int(num_layers))
            count_feature_mode = str(count_feature_mode or "none").strip().lower()
            if count_feature_mode not in {"none", "log_count", "log_count_density"}:
                raise ValueError(f"Unsupported count_feature_mode: {count_feature_mode}")
            pooling_mode = str(pooling_mode or "attention_weighted").strip().lower()
            if pooling_mode != "attention_weighted":
                raise ValueError("MuAtCompatibleModel now supports only attention_weighted pooling")
            self.count_feature_mode = count_feature_mode
            self.pooling_mode = pooling_mode
            self.num_layers = num_layers
            self.modality_embeddings = nn.ModuleList(
                [nn.Embedding(size, int(embed_dim), padding_idx=0) for size in self.modality_vocab_sizes]
            )
            # Continuous per-event channels (e.g. variant allele fraction) get their own
            # embed_dim-wide block so they sit alongside the token embeddings rather than being
            # forced through a vocabulary.
            self.numeric_projection = (
                nn.Linear(self.numeric_features, int(embed_dim)) if self.numeric_features else None
            )
            self.dropout = nn.Dropout(float(dropout))
            self.attention_layers = nn.ModuleList(
                [nn.MultiheadAttention(width, int(attention_heads), dropout=float(dropout), batch_first=True) for _ in range(num_layers)]
            )
            self.norm1_layers = nn.ModuleList([nn.LayerNorm(width) for _ in range(num_layers)])
            self.event_layers = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(float(dropout)), nn.Linear(width, width))
                    for _ in range(num_layers)
                ]
            )
            self.norm2_layers = nn.ModuleList([nn.LayerNorm(width) for _ in range(num_layers)])
            self.pool_score = nn.Linear(width, 1)
            count_width = 0 if count_feature_mode == "none" else (2 if count_feature_mode == "log_count_density" else 1)
            self.feature_layer = nn.Linear(width + count_width, int(feature_dim))
            self.classifier = nn.Linear(int(feature_dim), int(n_outputs))

        def forward(
            self,
            x: "torch.Tensor",
            mask: "torch.Tensor",
            numeric: "torch.Tensor | None" = None,
            *,
            return_attention: bool = False,
        ):
            mask = mask.bool()
            empty = ~mask.any(dim=1)
            if torch.any(empty):
                mask = mask.clone()
                mask[empty, 0] = True
            if x.shape[-1] != len(self.modality_embeddings):
                raise ValueError(
                    f"event tensor has {x.shape[-1]} modality columns but the model was built for "
                    f"{len(self.modality_embeddings)}"
                )
            parts = [embedding(x[:, :, i]) for i, embedding in enumerate(self.modality_embeddings)]
            if self.numeric_projection is not None:
                if numeric is None:
                    raise ValueError("model expects numeric event features but none were supplied")
                parts.append(self.numeric_projection(numeric.to(parts[0].dtype)))
            z = torch.cat(parts, dim=-1)
            z = self.dropout(z)
            for attention, norm1, event_layer, norm2 in zip(
                self.attention_layers,
                self.norm1_layers,
                self.event_layers,
                self.norm2_layers,
            ):
                attended, _weights = attention(
                    z,
                    z,
                    z,
                    key_padding_mask=~mask,
                    need_weights=False,
                    average_attn_weights=False,
                )
                z = norm1(z + attended)
                z = norm2(z + event_layer(z))
            valid = mask.unsqueeze(-1).to(z.dtype)
            event_count = valid.sum(dim=1).clamp_min(1.0)
            # Attention-weighted set pooling, computed in float32.
            #
            # Under autocast, LayerNorm returns float32 but nn.Linear returns float16, so `z` and
            # `self.pool_score(z)` have different dtypes. Taking the mask fill value from
            # torch.finfo(z.dtype) therefore tried to write -3.4e38 (float32 min) into a float16
            # tensor and raised "value cannot be converted to type at::Half without overflow".
            # This line could only ever run with AMP disabled, i.e. on CPU.
            #
            # Promoting the pooling logits to float32 fixes the dtype mismatch and is also the
            # numerically correct place to do a softmax over up to 5,000 masked events.
            pool_logits = self.pool_score(z).squeeze(-1).float()
            pool_logits = pool_logits.masked_fill(~mask, torch.finfo(pool_logits.dtype).min)
            pool_weights = torch.softmax(pool_logits, dim=1).unsqueeze(-1) * valid.float()
            weight_sum = pool_weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(pool_weights.dtype).eps)
            pool_weights = pool_weights / weight_sum
            pooled = (z.float() * pool_weights).sum(dim=1).to(z.dtype)
            if self.count_feature_mode != "none":
                log_count = torch.log1p(event_count)
                if self.count_feature_mode == "log_count_density":
                    density = event_count / float(max(1, mask.shape[1]))
                    pooled = torch.cat([pooled, log_count, density], dim=1)
                else:
                    pooled = torch.cat([pooled, log_count], dim=1)
            features = self.feature_layer(pooled)
            logits = self.classifier(features)
            if return_attention:
                return logits, features, pool_weights.squeeze(-1).unsqueeze(1).unsqueeze(1)
            return logits, features

else:

    class MuAtCompatibleModel:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("torch is required for MuAtCompatibleModel")
