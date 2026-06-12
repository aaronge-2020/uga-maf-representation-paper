"""Shared Kucab event-inventory and COSMIC-count feature helpers."""

from __future__ import annotations

import gzip
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils.checkpointing import atomic_write_csv, atomic_write_json
from utils.config import BUNDLE_ROOT
from utils.endpoint_data import EndpointData
from utils.feature_cache import fasta_fingerprint, file_fingerprint, make_cache_key, stable_json_hash
from utils.runner_support import RunnerContext


INVENTORY_NAMESPACE = "kucab_event_inventory"
BASES = "ACGT"
BASE_TO_INDEX = {base: i for i, base in enumerate(BASES)}
COMP = str.maketrans("ACGTN-", "TGCAN-")
DAMAGE_CLASS_BY_AGENT = {
    "Acetaldehyde": "Alkylating and aldehyde adduct stress",
    "Acrolein": "Alkylating and aldehyde adduct stress",
    "Cyclophosphamide": "Alkylating and aldehyde adduct stress",
    "DES": "Alkylating and aldehyde adduct stress",
    "DMH": "Alkylating and aldehyde adduct stress",
    "DMS": "Alkylating and aldehyde adduct stress",
    "ENU": "Alkylating and aldehyde adduct stress",
    "Formaldehyde": "Alkylating and aldehyde adduct stress",
    "Glycidamide": "Alkylating and aldehyde adduct stress",
    "MMS": "Alkylating and aldehyde adduct stress",
    "MNNG": "Alkylating and aldehyde adduct stress",
    "MNU": "Alkylating and aldehyde adduct stress",
    "Mechlorethamine": "Alkylating and aldehyde adduct stress",
    "Melphalan": "Alkylating and aldehyde adduct stress",
    "Propylene oxide": "Alkylating and aldehyde adduct stress",
    "Semustine": "Alkylating and aldehyde adduct stress",
    "Styrene oxide": "Alkylating and aldehyde adduct stress",
    "Temozolomide": "Alkylating and aldehyde adduct stress",
    "2-Naphthylamine": "Aromatic and heterocyclic amines",
    "2,6-Dimethylaniline": "Aromatic and heterocyclic amines",
    "4-ABP": "Aromatic and heterocyclic amines",
    "Benzidine": "Aromatic and heterocyclic amines",
    "IQ": "Aromatic and heterocyclic amines",
    "MOCA": "Aromatic and heterocyclic amines",
    "MeAaC": "Aromatic and heterocyclic amines",
    "MeIQX": "Aromatic and heterocyclic amines",
    "PhIP": "Aromatic and heterocyclic amines",
    "o-Anisidine": "Aromatic and heterocyclic amines",
    "o-Toluidine": "Aromatic and heterocyclic amines",
    "5-Methylchrysene": "PAH bulky adducts",
    "BaP": "PAH bulky adducts",
    "BPDE": "PAH bulky adducts",
    "DBA": "PAH bulky adducts",
    "DBAC": "PAH bulky adducts",
    "DBADE": "PAH bulky adducts",
    "DBC": "PAH bulky adducts",
    "DBP": "PAH bulky adducts",
    "DBPDE": "PAH bulky adducts",
    "1,6-DNP": "Nitroaromatic and nitro-PAH adducts",
    "1,8-DNP": "Nitroaromatic and nitro-PAH adducts",
    "1-Nitropyrene": "Nitroaromatic and nitro-PAH adducts",
    "2-Nitrofluorene": "Nitroaromatic and nitro-PAH adducts",
    "2-Nitrotoluene": "Nitroaromatic and nitro-PAH adducts",
    "3-NBA": "Nitroaromatic and nitro-PAH adducts",
    "6-Nitrochrysene": "Nitroaromatic and nitro-PAH adducts",
    "AZ20": "Replication stress and DNA repair inhibition",
    "AZD7762": "Replication stress and DNA repair inhibition",
    "Bleomycin": "Replication stress and DNA repair inhibition",
    "Camptothecin": "Replication stress and DNA repair inhibition",
    "Etoposide": "Replication stress and DNA repair inhibition",
    "Olaparib": "Replication stress and DNA repair inhibition",
    "Carboplatin": "Platinum and crosslinking therapy",
    "Cisplatin": "Platinum and crosslinking therapy",
    "Mitomycin C": "Platinum and crosslinking therapy",
    "1,4-Benzoquinone": "Oxidative and radiation-like stress",
    "Catechol": "Oxidative and radiation-like stress",
    "Gamma irradiation": "Oxidative and radiation-like stress",
    "H2O2": "Oxidative and radiation-like stress",
    "Peroxynitrite": "Oxidative and radiation-like stress",
    "Potassium bromate": "Oxidative and radiation-like stress",
    "SSR": "Oxidative and radiation-like stress",
    "AAI": "Environmental organ-specific DNA adducts",
    "AAII": "Environmental organ-specific DNA adducts",
    "AFB1": "Environmental organ-specific DNA adducts",
    "Acrylamide": "Environmental organ-specific DNA adducts",
    "Furan": "Environmental organ-specific DNA adducts",
    "MX": "Environmental organ-specific DNA adducts",
    "Methyleugenol": "Environmental organ-specific DNA adducts",
    "N-Nitrosopyrrolidine": "Environmental organ-specific DNA adducts",
    "OTA": "Environmental organ-specific DNA adducts",
}
CHROM_TO_ACCESSION = {
    "1": "NC_000001.10",
    "2": "NC_000002.11",
    "3": "NC_000003.11",
    "4": "NC_000004.11",
    "5": "NC_000005.9",
    "6": "NC_000006.11",
    "7": "NC_000007.13",
    "8": "NC_000008.10",
    "9": "NC_000009.11",
    "10": "NC_000010.10",
    "11": "NC_000011.9",
    "12": "NC_000012.11",
    "13": "NC_000013.10",
    "14": "NC_000014.8",
    "15": "NC_000015.9",
    "16": "NC_000016.9",
    "17": "NC_000017.10",
    "18": "NC_000018.9",
    "19": "NC_000019.9",
    "20": "NC_000020.10",
    "21": "NC_000021.8",
    "22": "NC_000022.10",
    "X": "NC_000023.10",
    "Y": "NC_000024.9",
    "MT": "NC_012920.1",
}


@dataclass
class EventInventory:
    name: str
    event_vectors: np.ndarray
    event_patients: np.ndarray
    patient_ids: list[str]
    standard_counts: pd.DataFrame
    covariates: pd.DataFrame
    qc: dict[str, object]
    event_modalities: np.ndarray | None = None
    aggregation_key: str | None = None


class FastaReader:
    """Minimal indexed FASTA reader for GRCh37 accessions."""

    def __init__(self, fasta_path: Path, *, cache_sequences: bool = False):
        self.fasta_path = Path(fasta_path)
        self.fai_path = self.fasta_path.parent / f"{self.fasta_path.name}.fai"
        self.index: dict[str, tuple[int, int, int, int]] = {}
        self._handle = None
        self.cache_sequences = bool(cache_sequences)
        self._sequence_cache: dict[str, str] = {}
        with self.fai_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if parts:
                    self.index[parts[0]] = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))

    def open(self) -> None:
        if self._handle is None:
            self._handle = self.fasta_path.open("rb")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._sequence_cache.clear()

    def _record_sequence(self, accession: str) -> str:
        if accession in self._sequence_cache:
            return self._sequence_cache[accession]
        self.open()
        assert self._handle is not None
        ref_len, offset, line_bases, line_bytes = self.index[accession]
        n_lines = int(math.ceil(ref_len / line_bases))
        bytes_to_read = ref_len + n_lines * max(0, line_bytes - line_bases)
        self._handle.seek(offset)
        raw = self._handle.read(bytes_to_read)
        sequence = raw.decode("ascii", errors="ignore").replace("\n", "").replace("\r", "")[:ref_len].upper()
        self._sequence_cache[accession] = sequence
        return sequence

    def fetch_range(self, chrom: object, start1: int, end1: int) -> str:
        if end1 < start1:
            return ""
        width = int(end1) - int(start1) + 1
        key = normalize_chrom(chrom)
        accession = CHROM_TO_ACCESSION.get(key, str(chrom))
        if accession not in self.index:
            return "N" * width
        ref_len, offset, line_bases, line_bytes = self.index[accession]
        start0 = int(start1) - 1
        end0 = int(end1)
        prefix_ns = max(0, -start0)
        suffix_ns = max(0, end0 - ref_len)
        start0 = max(0, start0)
        end0 = min(ref_len, end0)
        if start0 >= end0:
            return "N" * width
        if self.cache_sequences:
            seq = self._record_sequence(accession)[start0:end0]
            return ("N" * prefix_ns) + seq + ("N" * suffix_ns)
        self.open()
        assert self._handle is not None
        bases_to_read = end0 - start0
        lines_before = start0 // line_bases
        bytes_before = lines_before * line_bytes + (start0 % line_bases)
        byte_start = offset + bytes_before
        bytes_to_read = bases_to_read + (bases_to_read // line_bases + 2) * (line_bytes - line_bases)
        self._handle.seek(byte_start)
        raw = self._handle.read(bytes_to_read)
        seq = raw.decode("ascii", errors="ignore").replace("\n", "").replace("\r", "")[:bases_to_read].upper()
        return ("N" * prefix_ns) + seq + ("N" * suffix_ns)


def normalize_chrom(value: object) -> str:
    text = str(value).replace("chr", "").replace("CHR", "").strip().upper()
    if not text or text.lower() == "nan":
        return "0"
    return text.lstrip("0") or "0"


def clean_allele(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"", "-", ".", "NAN", "NONE"}:
        return ""
    return "".join(base for base in text if base in BASES)


def revcomp(seq: str) -> str:
    return str(seq).upper().translate(COMP)[::-1]


def trim_shared_alleles(ref: str, alt: str) -> tuple[str, str]:
    r, a = clean_allele(ref), clean_allele(alt)
    while r and a and r[0] == a[0]:
        r, a = r[1:], a[1:]
    while r and a and r[-1] == a[-1]:
        r, a = r[:-1], a[:-1]
    return r, a


def infer_modality(ref: str, alt: str, variant_type: object = "") -> str:
    vt = str(variant_type or "").upper()
    if vt in {"SNP", "SNV"} or (len(ref) == 1 and len(alt) == 1):
        return "SBS"
    if vt == "DNP" or (len(ref) == 2 and len(alt) == 2):
        return "DBS"
    if vt in {"INS", "DEL"} or len(ref) != len(alt):
        return "ID"
    return "OTHER"


def fetch_context(reader: FastaReader, chrom: object, pos: int, ref: str, d_context: int, modality: str) -> tuple[str, str, bool, str]:
    if modality == "ID" and not ref:
        window = reader.fetch_range(chrom, pos - d_context + 1, pos + d_context)
        left = window[:d_context]
        right = window[d_context:]
        return left, right, True, ""
    ref_len = max(1, len(ref))
    window = reader.fetch_range(chrom, pos - d_context, pos + ref_len + d_context - 1)
    left = window[:d_context]
    observed = window[d_context : d_context + ref_len]
    right = window[d_context + ref_len : d_context + ref_len + d_context]
    return left, right, bool(ref) and observed[: len(ref)].upper() == ref, observed


def canonicalize_for_context(left: str, right: str, ref: str, alt: str, modality: str) -> tuple[str, str, str, str]:
    if modality == "SBS" and ref in {"A", "G"}:
        return revcomp(right), revcomp(left), revcomp(ref), revcomp(alt)
    if modality == "DBS" and ref[:1] in {"A", "G"}:
        return revcomp(right), revcomp(left), revcomp(ref), revcomp(alt)
    return left, right, ref, alt


def base_indicator_sequence(seq: str, width: int) -> np.ndarray:
    out = np.zeros((int(width), 4), dtype=np.uint8)
    seq = str(seq or "").upper()[: int(width)]
    for i, base in enumerate(seq):
        j = BASE_TO_INDEX.get(base)
        if j is not None:
            out[i, j] = 1.0
    return out.ravel()


def allele_indicator_payload(seq: str, width: int) -> np.ndarray:
    width = int(width)
    out = np.zeros((width, 5), dtype=np.uint8)
    seq = clean_allele(seq)[:width]
    for i, base in enumerate(seq):
        j = BASE_TO_INDEX.get(base)
        if j is not None:
            out[i, j] = 1.0
            out[i, 4] = 1.0
    return out.ravel()


def encode_event_indicator(left: str, right: str, ref: str, alt: str, d_context: int, d_payload: int) -> np.ndarray:
    return np.concatenate(
        [
            base_indicator_sequence(left[-d_context:], d_context),
            base_indicator_sequence(right[:d_context], d_context),
            allele_indicator_payload(ref, d_payload),
            allele_indicator_payload(alt, d_payload),
            np.array([int(len(clean_allele(ref)) > d_payload), int(len(clean_allele(alt)) > d_payload)], dtype=np.uint8),
        ]
    )


def sbs_channel(left: str, ref: str, alt: str, right: str) -> str | None:
    if len(ref) != 1 or len(alt) != 1 or not left or not right:
        return None
    return f"{left[-1]}[{ref}>{alt}]{right[0]}"


def dbs_channel(ref: str, alt: str, valid: set[str]) -> str | None:
    if len(ref) != 2 or len(alt) != 2:
        return None
    channel = f"{ref}>{alt}"
    return channel if channel in valid else None


def id83_channel(ref: str, alt: str, valid: set[str]) -> str | None:
    r, a = trim_shared_alleles(ref, alt)
    if len(r) == len(a):
        return None
    event = "Del" if len(r) > len(a) else "Ins"
    payload = r if event == "Del" else a
    length = min(max(1, len(payload)), 5)
    repeat = 0
    if length == 1:
        base = payload[:1]
        if base not in {"C", "T"}:
            base = revcomp(base)[:1] if base else "C"
        if base not in {"C", "T"}:
            base = "C"
        channel = f"1:{event}:{base}:{repeat}"
    else:
        channel = f"{length}:{event}:R:{repeat}"
    return channel if channel in valid else None


def find_fasta(grch37_dir: Path) -> Path:
    if grch37_dir.is_file():
        fasta = grch37_dir
    else:
        fasta = grch37_dir / "GCF_000001405.13" / "GCF_000001405.13_GRCh37_genomic.fna"
    if not fasta.exists() and fasta.with_suffix(f"{fasta.suffix}.gz").exists():
        decompress_fasta_gzip(fasta.with_suffix(f"{fasta.suffix}.gz"), fasta)
    if not fasta.exists():
        matches = sorted(grch37_dir.rglob("*.fna")) + sorted(grch37_dir.rglob("*.fa")) + sorted(grch37_dir.rglob("*.fasta"))
        if matches:
            fasta = matches[0]
    if not fasta.exists():
        gz_matches = sorted(grch37_dir.rglob("*.fna.gz")) + sorted(grch37_dir.rglob("*.fa.gz")) + sorted(grch37_dir.rglob("*.fasta.gz"))
        if gz_matches:
            fasta = gz_matches[0].with_suffix("")
            decompress_fasta_gzip(gz_matches[0], fasta)
    if not fasta.exists() or not (fasta.parent / f"{fasta.name}.fai").exists():
        raise FileNotFoundError(f"Could not find indexed GRCh37 FASTA under {grch37_dir}")
    return fasta


def decompress_fasta_gzip(gzip_path: Path, fasta_path: Path) -> None:
    """Create the uncompressed FASTA beside a curated .gz bundle copy."""
    gzip_path = Path(gzip_path)
    fasta_path = Path(fasta_path)
    if fasta_path.exists():
        return
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = fasta_path.with_name(f".{fasta_path.name}.tmp")
    with gzip.open(gzip_path, "rb") as src, tmp_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    tmp_path.replace(fasta_path)


def display_fasta_path(fasta_path: Path, grch37_dir: Path) -> str:
    try:
        suffix = fasta_path.relative_to(grch37_dir)
        return str(Path("${raw_data.grch37_dir}") / suffix)
    except ValueError:
        return "${raw_data.grch37_dir}/GCF_000001405.13/GCF_000001405.13_GRCh37_genomic.fna"


def read_cosmic_channels(grch37_dir: Path | None = None) -> tuple[list[str], list[str], list[str]]:
    if grch37_dir is not None:
        root = Path(grch37_dir) / "cgr_validation" / "data" / "Signatures"
    else:
        root = BUNDLE_ROOT.parent / "github_exports_datasets" / "references" / "grch37" / "cgr_validation" / "data" / "Signatures"
    sbs = pd.read_csv(root / "COSMIC_v3.5_SBS_GRCh37.txt", sep="\t", usecols=["Type"])["Type"].astype(str).tolist()
    dbs = pd.read_csv(root / "COSMIC_v3.5_DBS_GRCh37.txt", sep="\t", usecols=["Type"])["Type"].astype(str).tolist()
    ids = pd.read_csv(root / "COSMIC_v3.5_ID_GRCh37.txt", sep="\t", usecols=["Type"])["Type"].astype(str).tolist()
    return sbs, dbs, ids


def empty_counts(patient_ids: list[str], sbs: list[str], dbs: list[str], ids: list[str]) -> pd.DataFrame:
    columns = [f"SBS96:{c}" for c in sbs] + [f"DBS78:{c}" for c in dbs] + [f"ID83:{c}" for c in ids]
    return pd.DataFrame(0.0, index=pd.Index(patient_ids, name="sample"), columns=columns, dtype=np.float32)


def standard_features_from_counts(counts: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    arr = counts.to_numpy(dtype=np.float64)
    total = arr.sum(axis=1, keepdims=True)
    freq = np.divide(arr, total, out=np.zeros_like(arr), where=total > 0)
    return pd.concat([pd.DataFrame(freq, index=counts.index, columns=counts.columns), covariates.reindex(counts.index).fillna(0.0)], axis=1).astype(np.float32)


def burden_features_from_covariates(covariates: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in ["log_total_burden"] if col in covariates.columns]
    if not cols:
        return pd.DataFrame({"log_total_burden": np.zeros(len(covariates), dtype=np.float32)}, index=covariates.index)
    return covariates.loc[:, cols].fillna(0.0).astype(np.float32)


def covariates_from_modalities(patient_ids: list[str], modality_counts: dict[str, Counter[str]]) -> pd.DataFrame:
    rows = []
    for patient in patient_ids:
        counts = modality_counts.get(patient, Counter())
        sbs = float(counts.get("SBS", 0))
        dbs = float(counts.get("DBS", 0))
        ids = float(counts.get("ID", 0))
        total = sbs + dbs + ids
        denom = total if total > 0 else 1.0
        rows.append(
            {
                "log_total_burden": math.log1p(total),
                "fraction_sbs": sbs / denom,
                "fraction_dbs": dbs / denom,
                "fraction_id": ids / denom,
            }
        )
    return pd.DataFrame(rows, index=pd.Index(patient_ids, name="sample"), dtype=np.float32)


def _inventory_cache_inputs(ctx: RunnerContext, *, fasta_path: Path, sample_ids: list[str], extra_paths: Iterable[Path]) -> dict[str, object]:
    return {
        "sample_ids_sha256": stable_json_hash(sorted(map(str, sample_ids)), length=32),
        "n_sample_ids": len(sample_ids),
        "fasta": fasta_fingerprint(fasta_path),
        "inputs": {Path(path).name: file_fingerprint(path) for path in extra_paths},
        "cache_schema": {
            "inventory_schema_version": 3,
            "context_fetch": "single_window_indexed_fasta",
            "event_vector_encoding": "base_context_payload_indicator_v2",
        },
    }


def _save_inventory(ctx: RunnerContext, key: str, inventory: EventInventory, *, metadata: dict[str, object]) -> str:
    cache = ctx.feature_cache
    entry = cache.entry(key)
    entry.dir.mkdir(parents=True, exist_ok=True)
    metadata_payload = {
        **metadata,
        "namespace": INVENTORY_NAMESPACE,
        "cache_key": key,
        "feature_count": int(inventory.event_vectors.shape[1]) if inventory.event_vectors.ndim == 2 else 0,
        "sample_count": len(inventory.patient_ids),
        "event_count": int(inventory.event_vectors.shape[0]),
        "qc": inventory.qc,
    }
    cache.save_npz(
        key,
        "event_inventory.npz",
        metadata=metadata_payload,
        event_vectors=inventory.event_vectors,
        event_patients=np.asarray(inventory.event_patients, dtype=object),
        patient_ids=np.asarray(inventory.patient_ids, dtype=object),
        event_modalities=np.asarray(
            inventory.event_modalities
            if inventory.event_modalities is not None
            else np.full(len(inventory.event_patients), "UNK", dtype=object),
            dtype=object,
        ),
    )
    atomic_write_csv(inventory.standard_counts, entry.path("standard_counts.csv.gz"), index=True)
    atomic_write_csv(inventory.covariates, entry.path("covariates.csv.gz"), index=True)
    atomic_write_json(entry.path("qc.json"), dict(inventory.qc))
    cache.write_metadata(key, {**metadata_payload, "artifacts": ["covariates.csv.gz", "event_inventory.npz", "qc.json", "standard_counts.csv.gz"]})
    return key


def _load_inventory(ctx: RunnerContext, key: str, name: str) -> EventInventory | None:
    cache = ctx.feature_cache
    arrays = cache.load_npz(key, "event_inventory.npz")
    if arrays is None:
        return None
    entry = cache.entry(key)
    counts_path = entry.path("standard_counts.csv.gz")
    cov_path = entry.path("covariates.csv.gz")
    if not counts_path.exists() or not cov_path.exists():
        return None
    counts = pd.read_csv(counts_path, index_col=0)
    covariates = pd.read_csv(cov_path, index_col=0)
    counts.index = counts.index.astype(str)
    covariates.index = covariates.index.astype(str)
    qc_path = entry.path("qc.json")
    qc = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {}
    qc["cache_status"] = "hit"
    event_modalities = arrays["event_modalities"] if "event_modalities" in arrays else np.full(len(arrays["event_patients"]), "UNK", dtype=object)
    return EventInventory(
        name,
        np.asarray(arrays["event_vectors"]),
        np.asarray(arrays["event_patients"], dtype=object),
        [str(value) for value in arrays["patient_ids"].tolist()],
        counts.astype(np.float32),
        covariates.astype(np.float32),
        qc,
        np.asarray(event_modalities, dtype=object),
    )


def event_dim(settings: dict) -> int:
    return 8 * int(settings["d_context"]) + 10 * int(settings["d_payload"]) + 2


def agent_condition_label(treatment: str) -> str:
    label = re.sub(r"\s*\([^)]*\)", "", str(treatment)).strip()
    return re.sub(r"\s+", " ", label) or "Unknown"


def agent_core_label(treatment: str) -> str:
    raw = str(treatment)
    if "control" in raw.lower():
        return "Control"
    label = agent_condition_label(raw)
    return re.sub(r"\s*\+\s*H?S9\b", "", label, flags=re.IGNORECASE).strip() or "Unknown"


def damage_class_for_agent(agent_core: str) -> str:
    if agent_core == "Control":
        return "Control/background"
    return DAMAGE_CLASS_BY_AGENT.get(str(agent_core), "Other or mechanism-ambiguous")


def parse_treatment_map(readme_path: Path) -> pd.DataFrame:
    rows = []
    for line in readme_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.match(r"^MSM0\.\d+\t", line):
            sample_name, treatment = line.split("\t", 1)
            rows.append({"sample_name": sample_name.strip(), "treatment": treatment.strip()})
    out = pd.DataFrame(rows)
    out["agent_core"] = out["treatment"].map(agent_core_label)
    out["agent_condition"] = out["treatment"].map(agent_condition_label)
    out["damage_class"] = out["agent_core"].map(damage_class_for_agent)
    return out


def build_kucab_inventory(ctx: RunnerContext, settings: dict, fasta: FastaReader, fasta_path: Path, sbs: list[str], dbs: list[str], ids: list[str]) -> tuple[EventInventory, EndpointData, str]:
    raw_dir = Path(ctx.paths["raw_data"]["kucab_raw_dir"])
    input_paths = [
        raw_dir / "README.txt",
        raw_dir / "denovo_subclone_subs_final.txt",
        raw_dir / "denovo_subclone_doublesub_final.txt",
        raw_dir / "denovo_subclone_indels.final.txt",
    ]
    cache_inputs = _inventory_cache_inputs(ctx, fasta_path=fasta_path, sample_ids=["kucab_all_samples"], extra_paths=input_paths)
    cache_key = make_cache_key(
        "kucab_event_inventory_damage_class",
        params={"d_context": int(settings["d_context"]), "d_payload": int(settings["d_payload"]), "min_total_events": int(settings.get("kucab_min_total_events", 20))},
        inputs=cache_inputs,
    )
    cached = _load_inventory(ctx, cache_key, "kucab_damage_class")
    treatments = parse_treatment_map(raw_dir / "README.txt")
    treatment_lookup = treatments.set_index("sample_name").to_dict("index")
    if cached is not None:
        metadata_path = ctx.feature_cache.entry(cache_key).path("sample_metadata.csv.gz")
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path, index_col=0)
            metadata.index = metadata.index.astype(str)
            endpoint = EndpointData("damage_class", "kucab_damage_class", "multiclass_grouped", metadata["damage_class"].astype(str), metadata["agent_core"].astype(str))
            return cached, endpoint, cache_key
    valid_dbs = set(dbs)
    valid_ids = set(ids)
    event_vectors: list[np.ndarray] = []
    event_patients: list[str] = []
    event_modalities: list[str] = []
    sample_to_name: dict[str, str] = {}
    modality_counts: dict[str, Counter[str]] = defaultdict(Counter)
    raw_counts: dict[str, Counter[str]] = defaultdict(Counter)
    qc = Counter()

    def process_event(sample: str, sample_name: str, chrom: object, pos_value: object, ref_value: object, alt_value: object, variant_type: str) -> None:
        sample_to_name[sample] = sample_name
        qc["rows_scanned"] += 1
        ref = clean_allele(ref_value)
        alt = clean_allele(alt_value)
        modality = infer_modality(ref, alt, variant_type)
        if modality == "OTHER" or (not ref and not alt):
            qc["invalid_or_other"] += 1
            return
        try:
            pos = int(float(pos_value or 0))
        except (TypeError, ValueError):
            qc["invalid_position"] += 1
            return
        left, right, ref_ok, _observed = fetch_context(fasta, chrom, pos, ref, int(settings["d_context"]), modality)
        if not ref_ok:
            qc["fasta_ref_mismatch"] += 1
            return
        if len(left) < int(settings["d_context"]) or len(right) < int(settings["d_context"]) or "N" in left.upper() or "N" in right.upper():
            qc["unresolved_context"] += 1
            return
        left, right, cref, calt = canonicalize_for_context(left, right, ref, alt, modality)
        if not set(cref + calt) <= set(BASES):
            qc["invalid_allele_after_canonicalization"] += 1
            return
        vector = encode_event_indicator(left, right, cref, calt, int(settings["d_context"]), int(settings["d_payload"]))
        channel = None
        if modality == "SBS":
            channel = sbs_channel(left, cref, calt, right)
        elif modality == "DBS":
            channel = dbs_channel(cref, calt, valid_dbs)
        elif modality == "ID":
            channel = id83_channel(cref, calt, valid_ids)
        event_vectors.append(vector.astype(np.uint8))
        event_patients.append(sample)
        event_modalities.append(modality)
        modality_counts[sample][modality] += 1
        if channel is not None:
            raw_counts[sample][f"{ {'SBS':'SBS96','DBS':'DBS78','ID':'ID83'}[modality]}:{channel}"] += 1
        qc["encoded_events"] += 1
        qc[f"encoded_{modality.lower()}"] += 1

    subs = pd.read_csv(raw_dir / "denovo_subclone_subs_final.txt", sep="\t").rename(columns={"Sample.Name": "sample_name"})
    for row in subs.itertuples(index=False):
        process_event(str(row.Sample), str(row.sample_name), row.Chrom, row.Pos, row.Ref, row.Alt, "SNP")

    dbs_frame = pd.read_csv(raw_dir / "denovo_subclone_doublesub_final.txt", sep="\t").rename(columns={"Sample.Name": "sample_name"})
    for row in dbs_frame.itertuples(index=False):
        rd = row._asdict()
        process_event(str(rd["Sample"]), str(rd["sample_name"]), rd["Chrom"], rd["Pos"], rd["dinuc_Ref"], rd["dinuc_Alt"], "DNP")

    indels = pd.read_csv(raw_dir / "denovo_subclone_indels.final.txt", sep="\t").rename(columns={"Sample.Name": "sample_name"})
    for row in indels.itertuples(index=False):
        process_event(str(row.Sample), str(row.sample_name), row.Chrom, row.Pos, row.Ref, row.Alt, str(row.Type))

    samples = sorted(sample for sample, sample_name in sample_to_name.items() if sample_name in treatment_lookup)
    rows = []
    for sample in samples:
        info = treatment_lookup[sample_to_name[sample]]
        total_events = int(sum(modality_counts.get(sample, Counter()).values()))
        rows.append(
            {
                "sample": sample,
                "sample_name": sample_to_name[sample],
                "agent_core": info["agent_core"],
                "damage_class": info["damage_class"],
                "total_events": total_events,
                "eligible_event_burden": total_events >= int(settings.get("kucab_min_total_events", 20)),
            }
        )
    metadata = pd.DataFrame(rows).set_index("sample")
    use = metadata[
        metadata["eligible_event_burden"]
        & ~metadata["damage_class"].isin(["Control/background", "Other or mechanism-ambiguous"])
    ].copy()
    class_agent_counts = use.groupby("damage_class")["agent_core"].nunique()
    use = use[use["damage_class"].isin(class_agent_counts[class_agent_counts >= 2].index)].copy()
    class_counts = use["damage_class"].value_counts()
    use = use[use["damage_class"].isin(class_counts[class_counts >= 10].index)].sort_index()

    patient_ids = use.index.astype(str).tolist()
    counts = empty_counts(patient_ids, sbs, dbs, ids)
    for sample in patient_ids:
        for col, value in raw_counts.get(sample, Counter()).items():
            if col in counts.columns:
                counts.at[sample, col] = float(value)
    covariates = covariates_from_modalities(patient_ids, modality_counts)
    keep_mask = np.isin(np.asarray(event_patients, dtype=object), np.asarray(patient_ids, dtype=object))
    vectors = np.vstack(event_vectors).astype(np.uint8) if event_vectors else np.zeros((0, event_dim(settings)), dtype=np.uint8)
    qc_out = dict(qc)
    qc_out["eligible_samples"] = len(patient_ids)
    inventory = EventInventory(
        "kucab_damage_class",
        vectors[keep_mask],
        np.asarray(event_patients, dtype=object)[keep_mask],
        patient_ids,
        counts,
        covariates.reindex(patient_ids).fillna(0.0),
        qc_out,
        np.asarray(event_modalities, dtype=object)[keep_mask],
    )
    endpoint = EndpointData("damage_class", "kucab_damage_class", "multiclass_grouped", use["damage_class"].astype(str), use["agent_core"].astype(str))
    _save_inventory(ctx, cache_key, inventory, metadata={"inputs": cache_inputs, "representation": "kucab_event_inventory", "benchmark": "kucab_damage_class"})
    atomic_write_csv(use.loc[:, ["sample_name", "agent_core", "damage_class", "total_events", "eligible_event_burden"]], ctx.feature_cache.entry(cache_key).path("sample_metadata.csv.gz"), index=True)
    meta = ctx.feature_cache.read_metadata(cache_key) or {}
    meta["artifacts"] = sorted(set([*(meta.get("artifacts") or []), "sample_metadata.csv.gz"]))
    if "namespace" not in meta:
        meta.update({"namespace": INVENTORY_NAMESPACE, "representation": "kucab_event_inventory", "benchmark": "kucab_damage_class", "cache_key": cache_key})
    ctx.feature_cache.write_metadata(cache_key, meta)
    return inventory, endpoint, cache_key
