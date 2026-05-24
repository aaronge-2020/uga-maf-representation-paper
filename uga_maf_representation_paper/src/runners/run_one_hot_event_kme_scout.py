"""Atlas-free FASTA one-hot event KME scout."""

from __future__ import annotations

import hashlib
import gzip
import json
import math
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, SGDClassifier
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.config import BUNDLE_ROOT
from utils.checkpointing import atomic_write_csv, atomic_write_json
from utils.feature_cache import code_fingerprint, fasta_fingerprint, file_fingerprint, make_cache_key, stable_json_hash
from utils.runner_support import RunnerContext, sanitize_text, write_summary_csv


EXPERIMENT_ID = "one_hot_event_kme_scout"
RANDOM_SEED = 20260518
N_SPLITS = 5
OPTUNA_STUDY_SCHEMA = "kme_v2_no_queued_trials_v1"
BASES = "ACGT"
BASE_TO_INDEX = {base: i for i, base in enumerate(BASES)}
COMP = str.maketrans("ACGTN-", "TGCAN-")
_KME_AGGREGATION_CACHE: dict[tuple[str, tuple[int, int], int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
MC3_MAF_USECOLS = [
    "Hugo_Symbol",
    "Chromosome",
    "Start_Position",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele1",
    "Tumor_Seq_Allele2",
    "Tumor_Sample_Barcode",
]
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


@dataclass(frozen=True)
class EndpointData:
    name: str
    benchmark: str
    task: str
    labels: pd.Series
    groups: pd.Series | None = None


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


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("||".join(str(p) for p in parts).encode("utf-8")).digest()
    return RANDOM_SEED + int.from_bytes(digest[:4], "big") % 100_000


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


def choose_alt(ref: str, allele1: object, allele2: object) -> str:
    ref = clean_allele(ref)
    for value in (allele2, allele1):
        alt = clean_allele(value)
        if alt != ref:
            return alt
    return clean_allele(allele2)


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


def one_hot_sequence(seq: str, width: int) -> np.ndarray:
    out = np.zeros((int(width), 4), dtype=np.uint8)
    seq = str(seq or "").upper()[: int(width)]
    for i, base in enumerate(seq):
        j = BASE_TO_INDEX.get(base)
        if j is not None:
            out[i, j] = 1.0
    return out.ravel()


def one_hot_payload(seq: str, width: int) -> np.ndarray:
    width = int(width)
    out = np.zeros((width, 5), dtype=np.uint8)
    seq = clean_allele(seq)[:width]
    for i, base in enumerate(seq):
        j = BASE_TO_INDEX.get(base)
        if j is not None:
            out[i, j] = 1.0
            out[i, 4] = 1.0
    return out.ravel()


def encode_one_hot_event(left: str, right: str, ref: str, alt: str, d_context: int, d_payload: int) -> np.ndarray:
    return np.concatenate(
        [
            one_hot_sequence(left[-d_context:], d_context),
            one_hot_sequence(right[:d_context], d_context),
            one_hot_payload(ref, d_payload),
            one_hot_payload(alt, d_payload),
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


def read_cosmic_channels() -> tuple[list[str], list[str], list[str]]:
    root = BUNDLE_ROOT / "src" / "legacy_source" / "cgr_validation" / "data" / "Signatures"
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


def load_mc3_endpoints(ctx: RunnerContext, settings: dict) -> list[EndpointData]:
    requested = list(settings.get("main_mc3_endpoints") or [settings.get("hrd_endpoint", "hrd_binary_33")])
    hrd_path = Path(ctx.paths["raw_data"]["hrd_assets_dir"]) / "cohort" / "final_analysis_cohort.tsv"
    cohort = pd.read_csv(hrd_path, sep="\t")
    cohort["patient_id_12"] = cohort["patient_id_12"].astype(str)
    labels_path = Path(ctx.paths["raw_data"]["mc3_source_dir"]) / "biology_labels.csv"
    mc3_labels = pd.read_csv(labels_path, index_col=0)
    mc3_labels.index = mc3_labels.index.astype(str)
    endpoints: list[EndpointData] = []
    for endpoint in requested:
        endpoint = str(endpoint)
        if endpoint in {"HRD_Score", "eCARD", "HRD_TAI", "HRD_LST", "HRD_LOH", "PARPi7"} and endpoint in cohort.columns:
            data = cohort.dropna(subset=[endpoint]).copy()
            y = pd.Series(data[endpoint].astype(float).to_numpy(), index=data["patient_id_12"].astype(str), name=endpoint)
            endpoints.append(EndpointData(endpoint, "mc3_main", "regression", y))
        elif endpoint in {"hrd_binary_24", "hrd_binary_33", "hrd_binary_42", "parpi7_binary"} and endpoint in cohort.columns:
            allowed = ["PARPi-high", "PARPi-low"] if endpoint == "parpi7_binary" else ["HRD-high", "HRD-low"]
            positive = "PARPi-high" if endpoint == "parpi7_binary" else "HRD-high"
            data = cohort[cohort[endpoint].isin(allowed)].copy()
            y = pd.Series((data[endpoint] == positive).astype(int).to_numpy(), index=data["patient_id_12"].astype(str), name=endpoint)
            endpoints.append(EndpointData(endpoint, "mc3_main", "binary", y))
        elif endpoint in mc3_labels.columns:
            y = mc3_labels[endpoint].dropna()
            if endpoint == "cancer_type_top10":
                counts = y.astype(str).value_counts()
                y = y.astype(str)
                y = y[y.isin(counts[counts >= 50].index)]
                task = "multiclass"
            else:
                y = y.astype(int)
                task = "binary"
            endpoints.append(EndpointData(endpoint, "mc3_main", task, y))
    return endpoints


def _inventory_cache_inputs(ctx: RunnerContext, *, fasta_path: Path, sample_ids: list[str], extra_paths: Iterable[Path]) -> dict[str, object]:
    return {
        "sample_ids_sha256": stable_json_hash(sorted(map(str, sample_ids)), length=32),
        "n_sample_ids": len(sample_ids),
        "fasta": fasta_fingerprint(fasta_path),
        "inputs": {Path(path).name: file_fingerprint(path) for path in extra_paths},
        "cache_schema": {
            "inventory_schema_version": 3,
            "context_fetch": "single_window_indexed_fasta",
            "event_vector_encoding": "one_hot_context_payload_with_modality_manifest_v2",
        },
    }


def _save_inventory(ctx: RunnerContext, key: str, inventory: EventInventory, *, metadata: dict[str, object]) -> str:
    cache = ctx.feature_cache
    entry = cache.entry(key)
    entry.dir.mkdir(parents=True, exist_ok=True)
    metadata_payload = {
        **metadata,
        "namespace": EXPERIMENT_ID,
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


def append_event(
    *,
    event_vectors: list[np.ndarray],
    event_patients: list[str],
    event_modalities: list[str],
    counts: pd.DataFrame,
    modality_counts: dict[str, Counter[str]],
    patient: str,
    vector: np.ndarray,
    modality: str,
    channel: str | None,
) -> None:
    event_vectors.append(vector.astype(np.uint8))
    event_patients.append(patient)
    event_modalities.append(modality)
    modality_counts[patient][modality] += 1
    if channel is not None:
        prefix = {"SBS": "SBS96", "DBS": "DBS78", "ID": "ID83"}[modality]
        col = f"{prefix}:{channel}"
        if col in counts.columns:
            counts.at[patient, col] += 1.0


def build_mc3_inventory(ctx: RunnerContext, settings: dict, fasta: FastaReader, fasta_path: Path, sbs: list[str], dbs: list[str], ids: list[str]) -> tuple[EventInventory, list[EndpointData], str]:
    endpoints = load_mc3_endpoints(ctx, settings)
    patient_ids = sorted({str(patient) for endpoint in endpoints for patient in endpoint.labels.index.astype(str)})
    maf_path = Path(ctx.paths["raw_data"]["mc3_source_dir"]) / "raw" / "mc3.v0.2.8.PUBLIC.maf.gz"
    hrd_path = Path(ctx.paths["raw_data"]["hrd_assets_dir"]) / "cohort" / "final_analysis_cohort.tsv"
    labels_path = Path(ctx.paths["raw_data"]["mc3_source_dir"]) / "biology_labels.csv"
    cache_inputs = _inventory_cache_inputs(ctx, fasta_path=fasta_path, sample_ids=patient_ids, extra_paths=[maf_path, hrd_path, labels_path])
    cache_key = make_cache_key(
        "one_hot_event_inventory_mc3_main",
        params={"d_context": int(settings["d_context"]), "d_payload": int(settings["d_payload"]), "endpoints": [endpoint.name for endpoint in endpoints]},
        inputs=cache_inputs,
    )
    cached = _load_inventory(ctx, cache_key, "mc3_main")
    if cached is not None:
        return cached, endpoints, cache_key
    patient_set = set(patient_ids)
    counts = empty_counts(patient_ids, sbs, dbs, ids)
    modality_counts = {patient: Counter() for patient in patient_ids}
    event_vectors: list[np.ndarray] = []
    event_patients: list[str] = []
    event_modalities: list[str] = []
    valid_dbs = set(dbs)
    valid_ids = set(ids)
    qc = Counter()
    max_rows = int(settings.get("max_mc3_maf_rows", 0) or 0)
    total = 0
    for chunk in pd.read_csv(maf_path, sep="\t", usecols=MC3_MAF_USECOLS, dtype=str, chunksize=int(settings.get("maf_chunksize", 200_000))):
        if max_rows:
            remaining = max_rows - total
            if remaining <= 0:
                break
            chunk = chunk.head(remaining).copy()
        total += len(chunk)
        qc["rows_scanned"] += len(chunk)
        chunk["patient"] = chunk["Tumor_Sample_Barcode"].astype(str).str[:12]
        chunk = chunk[chunk["patient"].isin(patient_set)].copy()
        qc["rows_in_endpoint_patients"] += len(chunk)
        if not chunk.empty:
            chunk["_chrom_sort"] = chunk["Chromosome"].map(normalize_chrom)
            chunk["_pos_sort"] = pd.to_numeric(chunk["Start_Position"], errors="coerce").fillna(0).astype(np.int64)
            chunk = chunk.sort_values(["_chrom_sort", "_pos_sort"], kind="mergesort")
        for row in chunk.to_dict(orient="records"):
            patient = str(row["patient"])
            ref = clean_allele(row.get("Reference_Allele"))
            alt = choose_alt(ref, row.get("Tumor_Seq_Allele1"), row.get("Tumor_Seq_Allele2"))
            modality = infer_modality(ref, alt, row.get("Variant_Type"))
            if modality == "OTHER" or not alt:
                qc["invalid_or_other"] += 1
                continue
            try:
                pos = int(float(row.get("Start_Position") or 0))
            except (TypeError, ValueError):
                qc["invalid_position"] += 1
                continue
            left, right, ref_ok, _observed = fetch_context(fasta, row.get("Chromosome"), pos, ref, int(settings["d_context"]), modality)
            if not ref_ok:
                qc["fasta_ref_mismatch"] += 1
                continue
            if len(left) < int(settings["d_context"]) or len(right) < int(settings["d_context"]) or "N" in left.upper() or "N" in right.upper():
                qc["unresolved_context"] += 1
                continue
            left, right, cref, calt = canonicalize_for_context(left, right, ref, alt, modality)
            if not set(cref + calt) <= set(BASES):
                qc["invalid_allele_after_canonicalization"] += 1
                continue
            vector = encode_one_hot_event(left, right, cref, calt, int(settings["d_context"]), int(settings["d_payload"]))
            channel = None
            if modality == "SBS":
                channel = sbs_channel(left, cref, calt, right)
            elif modality == "DBS":
                channel = dbs_channel(cref, calt, valid_dbs)
            elif modality == "ID":
                channel = id83_channel(cref, calt, valid_ids)
            append_event(
                event_vectors=event_vectors,
                event_patients=event_patients,
                event_modalities=event_modalities,
                counts=counts,
                modality_counts=modality_counts,
                patient=patient,
                vector=vector,
                modality=modality,
                channel=channel,
            )
            qc["encoded_events"] += 1
            qc[f"encoded_{modality.lower()}"] += 1
    covariates = covariates_from_modalities(patient_ids, modality_counts)
    qc["cache_status"] = "miss_created"
    inventory = EventInventory(
        "mc3_main",
        np.vstack(event_vectors) if event_vectors else np.zeros((0, event_dim(settings)), dtype=np.uint8),
        np.asarray(event_patients, dtype=object),
        patient_ids,
        counts,
        covariates,
        dict(qc),
        np.asarray(event_modalities, dtype=object),
    )
    _save_inventory(ctx, cache_key, inventory, metadata={"inputs": cache_inputs, "representation": "one_hot_event_inventory", "benchmark": "mc3_main"})
    return inventory, endpoints, cache_key


def kucab_module():
    import sys

    project_root = BUNDLE_ROOT / "src" / "legacy_source" / "cgr_validation"
    code_root = project_root / "cgr_validation_results" / "research" / "experiments" / "supporting" / "2026_05_14_unified_uga_locked_manuscript_benchmark" / "code"
    for path in (project_root, code_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import run_locked_kucab_low_burden_benchmark as kucab

    return kucab


def build_kucab_inventory(ctx: RunnerContext, settings: dict, fasta: FastaReader, fasta_path: Path, sbs: list[str], dbs: list[str], ids: list[str]) -> tuple[EventInventory, EndpointData, str]:
    kucab = kucab_module()
    raw_dir = Path(ctx.paths["raw_data"]["kucab_raw_dir"])
    input_paths = [
        raw_dir / "README.txt",
        raw_dir / "denovo_subclone_subs_final.txt",
        raw_dir / "denovo_subclone_doublesub_final.txt",
        raw_dir / "denovo_subclone_indels.final.txt",
    ]
    cache_inputs = _inventory_cache_inputs(ctx, fasta_path=fasta_path, sample_ids=["kucab_all_samples"], extra_paths=input_paths)
    cache_key = make_cache_key(
        "one_hot_event_inventory_kucab_damage_class",
        params={"d_context": int(settings["d_context"]), "d_payload": int(settings["d_payload"]), "min_total_events": int(settings.get("kucab_min_total_events", 20))},
        inputs=cache_inputs,
    )
    cached = _load_inventory(ctx, cache_key, "kucab_damage_class")
    treatments = kucab.parse_treatment_map(raw_dir / "README.txt")
    treatment_lookup = treatments.set_index("sample_name").to_dict("index")
    if cached is not None:
        labels = []
        groups = []
        for sample in cached.patient_ids:
            sample_name = str(sample)
            # Cached Kucab sample IDs are internal sample numbers; recover metadata from treatment rows below if possible.
            labels.append("")
            groups.append("")
        # Rebuild the lightweight endpoint metadata even when the expensive FASTA inventory is cached.
        sample_to_treatment = {}
        for sample_name, row in treatment_lookup.items():
            sample_to_treatment[str(sample_name)] = row
        # The cached inventory stores only eligible samples, so use the original construction path below when labels are not recoverable.
        # Falling through would rebuild expensive event vectors; instead load labels from the saved metadata if present.
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
        vector = encode_one_hot_event(left, right, cref, calt, int(settings["d_context"]), int(settings["d_payload"]))
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
    _save_inventory(ctx, cache_key, inventory, metadata={"inputs": cache_inputs, "representation": "one_hot_event_inventory", "benchmark": "kucab_damage_class"})
    atomic_write_csv(use.loc[:, ["sample_name", "agent_core", "damage_class", "total_events", "eligible_event_burden"]], ctx.feature_cache.entry(cache_key).path("sample_metadata.csv.gz"), index=True)
    meta = ctx.feature_cache.read_metadata(cache_key) or {}
    meta["artifacts"] = sorted(set([*(meta.get("artifacts") or []), "sample_metadata.csv.gz"]))
    if "namespace" not in meta:
        meta.update({"namespace": EXPERIMENT_ID, "representation": "one_hot_event_inventory", "benchmark": "kucab_damage_class", "cache_key": cache_key})
    ctx.feature_cache.write_metadata(cache_key, meta)
    return inventory, endpoint, cache_key


def event_dim(settings: dict) -> int:
    return 8 * int(settings["d_context"]) + 10 * int(settings["d_payload"]) + 2


def median_pairwise_scale(points: np.ndarray, seed: int, max_points: int = 2000) -> float:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        return 1.0
    if points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        points = points[np.sort(rng.choice(points.shape[0], size=max_points, replace=False))]
    diffs = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diffs * diffs, axis=2))
    vals = dist[np.triu_indices(points.shape[0], k=1)]
    vals = vals[np.isfinite(vals) & (vals > 1e-12)]
    return float(np.median(vals)) if vals.size else 1.0


def select_landmarks(vectors: np.ndarray, mode: str, seed: int) -> np.ndarray:
    arr = np.asarray(vectors)
    if arr.size and np.nanmin(arr) >= 0.0 and np.nanmax(arr) <= 1.0:
        packed = np.packbits(arr.astype(np.uint8), axis=1)
        unique_packed = np.unique(packed, axis=0)
        unique = np.unpackbits(unique_packed, axis=1)[:, : arr.shape[1]].astype(np.float32)
    else:
        unique = np.unique(arr, axis=0)
    if str(mode) == "all" or unique.shape[0] <= int(mode):
        return unique
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(unique.shape[0], size=int(mode), replace=False))
    return unique[idx]


def _unique_binary_vectors_with_counts(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(vectors)
    if arr.size == 0:
        width = arr.shape[1] if arr.ndim == 2 else 0
        return np.zeros((0, width), dtype=np.float32), np.zeros(0, dtype=np.int64)
    if np.nanmin(arr) >= 0.0 and np.nanmax(arr) <= 1.0:
        packed = np.packbits(arr.astype(np.uint8), axis=1)
        unique_packed, counts = np.unique(packed, axis=0, return_counts=True)
        unique = np.unpackbits(unique_packed, axis=1)[:, : arr.shape[1]].astype(np.float32)
        return unique, counts.astype(np.int64)
    unique, counts = np.unique(arr, axis=0, return_counts=True)
    return unique.astype(np.float32), counts.astype(np.int64)


def select_landmarks_frequency_aware(vectors: np.ndarray, mode: str, seed: int, *, top_fraction: float = 0.5) -> np.ndarray:
    unique, counts = _unique_binary_vectors_with_counts(vectors)
    if str(mode) == "all" or unique.shape[0] <= int(mode):
        return unique
    n_landmarks = int(mode)
    n_top = int(max(1, min(n_landmarks, round(n_landmarks * float(top_fraction)))))
    order = np.argsort(-counts, kind="mergesort")
    top_idx = order[:n_top]
    remaining = np.setdiff1d(np.arange(unique.shape[0]), top_idx, assume_unique=False)
    n_tail = n_landmarks - len(top_idx)
    if n_tail > 0 and remaining.size:
        rng = np.random.default_rng(seed)
        tail_idx = rng.choice(remaining, size=min(n_tail, remaining.size), replace=False)
        idx = np.r_[top_idx, tail_idx]
    else:
        idx = top_idx
    idx = np.asarray(idx, dtype=np.int64)
    return unique[np.sort(idx)]


def _capped_event_indices(inventory: EventInventory, patient_index: dict[str, int], max_events_per_sample: int) -> np.ndarray:
    event_patients = np.asarray(inventory.event_patients, dtype=object)
    codes = np.asarray([patient_index.get(str(patient), -1) for patient in event_patients], dtype=np.int64)
    valid = np.flatnonzero(codes >= 0)
    if max_events_per_sample <= 0:
        return valid
    keep: list[np.ndarray] = []
    valid_codes = codes[valid]
    for sample_code in np.unique(valid_codes):
        sample_indices = valid[valid_codes == sample_code]
        if sample_indices.size > max_events_per_sample:
            patient = inventory.patient_ids[int(sample_code)]
            rng = np.random.default_rng(stable_seed(inventory.name, patient, "kme_event_cap", max_events_per_sample))
            sample_indices = np.sort(rng.choice(sample_indices, size=max_events_per_sample, replace=False))
        keep.append(sample_indices)
    return np.sort(np.concatenate(keep)) if keep else np.asarray([], dtype=np.int64)


def _kme_aggregation(inventory: EventInventory, max_events_per_sample: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_key = (inventory.name, tuple(inventory.event_vectors.shape), int(max_events_per_sample))
    if cache_key in _KME_AGGREGATION_CACHE:
        return _KME_AGGREGATION_CACHE[cache_key]
    patient_index = {patient: i for i, patient in enumerate(inventory.patient_ids)}
    keep = _capped_event_indices(inventory, patient_index, int(max_events_per_sample))
    vectors = np.asarray(inventory.event_vectors[keep], dtype=np.uint8)
    codes = np.asarray([patient_index[str(inventory.event_patients[idx])] for idx in keep], dtype=np.int64)
    sample_counts = np.bincount(codes, minlength=len(inventory.patient_ids)).astype(np.float32)
    packed = np.packbits(vectors.astype(np.uint8), axis=1)
    unique_packed, inverse = np.unique(packed, axis=0, return_inverse=True)
    unique_vectors = np.unpackbits(unique_packed, axis=1)[:, : vectors.shape[1]].astype(np.float32)
    order = np.lexsort((codes, inverse))
    if len(order):
        u_sorted = inverse[order]
        p_sorted = codes[order]
        start_mask = np.r_[True, (u_sorted[1:] != u_sorted[:-1]) | (p_sorted[1:] != p_sorted[:-1])]
        starts = np.flatnonzero(start_mask)
        pair_counts = np.diff(np.r_[starts, len(order)]).astype(np.float32)
        pair_u = u_sorted[starts].astype(np.int64)
        pair_p = p_sorted[starts].astype(np.int64)
    else:
        pair_u = np.asarray([], dtype=np.int64)
        pair_p = np.asarray([], dtype=np.int64)
        pair_counts = np.asarray([], dtype=np.float32)
    out = (unique_vectors, pair_u, pair_p, pair_counts, sample_counts)
    _KME_AGGREGATION_CACHE[cache_key] = out
    return out


def feature_weights(settings: dict, mode: str) -> np.ndarray | None:
    if str(mode or "uniform").lower() in {"", "uniform", "none"}:
        return None
    d_context = int(settings["d_context"])
    d_payload = int(settings["d_payload"])
    weights: list[float] = []
    for i in range(d_context):
        weights.extend([0.35 + 0.65 * ((i + 1) / max(d_context, 1))] * 4)
    for i in range(d_context):
        weights.extend([0.35 + 0.65 * ((d_context - i) / max(d_context, 1))] * 4)
    weights.extend([1.35] * (5 * d_payload))
    weights.extend([1.35] * (5 * d_payload))
    weights.extend([0.75, 0.75])
    return np.asarray(weights, dtype=np.float32)


def modality_summary_covariates(inventory: EventInventory) -> pd.DataFrame:
    modalities = np.asarray(inventory.event_modalities if inventory.event_modalities is not None else [], dtype=object)
    patients = np.asarray(inventory.event_patients, dtype=object)
    index = pd.Index(inventory.patient_ids, name="sample")
    out = pd.DataFrame(0.0, index=index, columns=["kme_log_sbs_burden", "kme_log_dbs_burden", "kme_log_id_burden"], dtype=np.float32)
    if len(modalities) != len(patients) or not len(patients):
        return out
    counts = (
        pd.DataFrame({"sample": patients.astype(str), "modality": modalities.astype(str)})
        .value_counts(["sample", "modality"])
        .rename("n")
        .reset_index()
    )
    for modality, column in {"SBS": "kme_log_sbs_burden", "DBS": "kme_log_dbs_burden", "ID": "kme_log_id_burden"}.items():
        sub = counts[counts["modality"].eq(modality)]
        if not sub.empty:
            values = pd.Series(np.log1p(sub["n"].astype(float).to_numpy()), index=sub["sample"].astype(str))
            out.loc[:, column] = values.reindex(out.index).fillna(0.0).to_numpy(dtype=np.float32)
    return out.astype(np.float32)


def kme_features(
    inventory: EventInventory,
    landmarks: np.ndarray,
    sigma: float,
    chunk_size: int = 512,
    max_events_per_sample: int = 0,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    unique_vectors, pair_u, pair_p, pair_counts, counts = _kme_aggregation(inventory, int(max_events_per_sample))
    sums = np.zeros((len(inventory.patient_ids), landmarks.shape[0]), dtype=np.float32)
    denom = max(float(sigma) * float(sigma), 1e-12)
    weighted_landmarks = landmarks.astype(np.float32, copy=False)
    if weights is not None:
        weighted_landmarks = weighted_landmarks * weights[None, :]
    landmark_norm = np.sum(weighted_landmarks * weighted_landmarks, axis=1, dtype=np.float32)[None, :]
    landmarks_t = weighted_landmarks.T.astype(np.float32, copy=False)
    unique_chunk_size = max(int(chunk_size), 4096)
    for start in range(0, unique_vectors.shape[0], unique_chunk_size):
        stop = min(start + unique_chunk_size, unique_vectors.shape[0])
        block = unique_vectors[start:stop]
        if weights is not None:
            block = block * weights[None, :]
        block_norm = np.sum(block * block, axis=1, dtype=np.float32)[:, None]
        dist2 = block_norm + landmark_norm - 2.0 * (block @ landmarks_t)
        np.maximum(dist2, 0.0, out=dist2)
        kernel = np.exp(-0.5 * dist2 / denom).astype(np.float32)
        lo = np.searchsorted(pair_u, start, side="left")
        hi = np.searchsorted(pair_u, stop, side="left")
        if hi > lo:
            local = pair_u[lo:hi] - start
            weighted = kernel[local] * pair_counts[lo:hi, None]
            np.add.at(sums, pair_p[lo:hi], weighted)
    features = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
    columns = [f"one_hot_kme__lm{i:05d}" for i in range(landmarks.shape[0])]
    frame = pd.DataFrame(features, index=pd.Index(inventory.patient_ids, name="sample"), columns=columns)
    return pd.concat([frame, inventory.covariates.reindex(frame.index).fillna(0.0)], axis=1).astype(np.float32)


def cached_landmarks(ctx: RunnerContext, inventory: EventInventory, inventory_cache_key: str, landmark_mode: object, settings: dict) -> tuple[np.ndarray, float, str]:
    key = make_cache_key(
        "one_hot_event_landmarks",
        params={"inventory": inventory.name, "inventory_cache_key": inventory_cache_key, "landmark_mode": str(landmark_mode), "d_context": int(settings["d_context"]), "d_payload": int(settings["d_payload"])},
        inputs={"event_count": int(inventory.event_vectors.shape[0]), "event_dim": int(inventory.event_vectors.shape[1]) if inventory.event_vectors.ndim == 2 else 0},
    )
    arrays = ctx.feature_cache.load_npz(key, "landmarks.npz")
    if arrays is not None:
        return np.asarray(arrays["landmarks"], dtype=np.float32), float(arrays["base_sigma"][0]), key
    landmarks = select_landmarks(inventory.event_vectors, str(landmark_mode), stable_seed(inventory.name, landmark_mode))
    base_sigma = median_pairwise_scale(landmarks, stable_seed(inventory.name, "sigma", landmark_mode))
    ctx.feature_cache.save_npz(
        key,
        "landmarks.npz",
        metadata={
            "namespace": EXPERIMENT_ID,
            "representation": "one_hot_event_kme_landmarks",
            "benchmark": inventory.name,
            "inventory_cache_key": inventory_cache_key,
            "landmark_mode": str(landmark_mode),
            "n_landmarks": int(landmarks.shape[0]),
            "base_sigma": float(base_sigma),
        },
        landmarks=landmarks.astype(np.float32),
        base_sigma=np.asarray([base_sigma], dtype=np.float64),
    )
    return landmarks, base_sigma, key


def cached_kme_features(ctx: RunnerContext, inventory: EventInventory, inventory_cache_key: str, landmarks: np.ndarray, landmark_cache_key: str, sigma: float, sigma_multiplier: object, settings: dict) -> tuple[pd.DataFrame, str, str]:
    key = make_cache_key(
        "one_hot_event_kme_features",
        params={
            "inventory": inventory.name,
            "inventory_cache_key": inventory_cache_key,
            "landmark_cache_key": landmark_cache_key,
            "sigma": float(sigma),
            "sigma_multiplier": float(sigma_multiplier),
            "kernel_chunk_size": int(settings.get("kernel_chunk_size", 512)),
            "kme_max_events_per_sample": int(settings.get("kme_max_events_per_sample", 0) or 0),
        },
    )
    frame = ctx.feature_cache.load_frame(key, "features.csv.gz")
    if frame is not None:
        frame.index = frame.index.astype(str)
        return frame.astype(np.float32), key, "hit"
    frame = kme_features(
        inventory,
        landmarks,
        sigma,
        chunk_size=int(settings.get("kernel_chunk_size", 512)),
        max_events_per_sample=int(settings.get("kme_max_events_per_sample", 0) or 0),
    )
    ctx.feature_cache.save_frame(
        key,
        frame,
        "features.csv.gz",
        metadata={
            "namespace": EXPERIMENT_ID,
            "representation": "one_hot_event_KME",
            "benchmark": inventory.name,
            "inventory_cache_key": inventory_cache_key,
            "landmark_cache_key": landmark_cache_key,
            "sigma": float(sigma),
            "sigma_multiplier": float(sigma_multiplier),
            "kme_max_events_per_sample": int(settings.get("kme_max_events_per_sample", 0) or 0),
            "sample_count": int(frame.shape[0]),
            "feature_count": int(frame.shape[1]),
        },
    )
    return frame, key, "miss_created"


def kme_config_payload(
    *,
    version: str,
    inventory: EventInventory,
    inventory_cache_key: str,
    landmark_mode: object,
    sigma_multiplier: object,
    sigma_strategy: str,
    modality_strategy: str,
    landmark_sampling: str,
    kernel_weighting: str,
    settings: dict,
) -> dict[str, object]:
    if str(sigma_strategy) == "multiscale_0.5_1_2":
        sigma_multipliers = [0.5, 1.0, 2.0]
    else:
        sigma_multipliers = [float(sigma_multiplier)]
    return {
        "version": str(version),
        "inventory": inventory.name,
        "inventory_cache_key": inventory_cache_key,
        "d_context": int(settings["d_context"]),
        "d_payload": int(settings["d_payload"]),
        "landmark_mode": str(landmark_mode),
        "sigma_multiplier": float(sigma_multiplier),
        "sigma_strategy": str(sigma_strategy),
        "sigma_multipliers_selected": sigma_multipliers,
        "modality_strategy": str(modality_strategy),
        "landmark_sampling": str(landmark_sampling),
        "kernel_weighting": str(kernel_weighting),
        "kernel": "weighted_rbf" if str(kernel_weighting) != "uniform" else "rbf",
        "event_cap": int(settings.get("kme_max_events_per_sample", 0) or 0),
        "cache_schema": "one_hot_event_kme_v2_config_2026_05_24",
    }


def kme_config_id(payload: dict[str, object]) -> str:
    return "kme_" + stable_json_hash(payload, length=24)


def _sigma_multipliers_for_strategy(sigma_strategy: str, sigma_multiplier: object) -> list[float]:
    if str(sigma_strategy) == "multiscale_0.5_1_2":
        return [0.5, 1.0, 2.0]
    return [float(sigma_multiplier)]


def _modality_inventory(inventory: EventInventory, modality: str, config_id: str) -> EventInventory:
    modalities = np.asarray(inventory.event_modalities if inventory.event_modalities is not None else [], dtype=object)
    if len(modalities) != len(inventory.event_patients):
        mask = np.ones(len(inventory.event_patients), dtype=bool)
    else:
        mask = modalities.astype(str) == str(modality)
    return EventInventory(
        name=f"{inventory.name}_{config_id}_{modality}",
        event_vectors=np.asarray(inventory.event_vectors[mask]),
        event_patients=np.asarray(inventory.event_patients, dtype=object)[mask],
        patient_ids=inventory.patient_ids,
        standard_counts=inventory.standard_counts,
        covariates=pd.DataFrame(index=pd.Index(inventory.patient_ids, name="sample")),
        qc=inventory.qc,
        event_modalities=modalities[mask] if len(modalities) == len(inventory.event_patients) else np.full(int(mask.sum()), modality, dtype=object),
    )


def cached_landmarks_v2(
    ctx: RunnerContext,
    inventory: EventInventory,
    inventory_cache_key: str,
    landmark_mode: object,
    settings: dict,
    *,
    modality: str,
    landmark_sampling: str,
    config_id: str,
) -> tuple[np.ndarray, float, str]:
    key = make_cache_key(
        "one_hot_event_landmarks_v2",
        params={
            "inventory": inventory.name,
            "inventory_cache_key": inventory_cache_key,
            "landmark_mode": str(landmark_mode),
            "d_context": int(settings["d_context"]),
            "d_payload": int(settings["d_payload"]),
            "modality": modality,
            "landmark_sampling": str(landmark_sampling),
            "config_id": config_id,
        },
        inputs={"event_count": int(inventory.event_vectors.shape[0]), "event_dim": int(inventory.event_vectors.shape[1]) if inventory.event_vectors.ndim == 2 else 0},
    )
    arrays = ctx.feature_cache.load_npz(key, "landmarks.npz")
    if arrays is not None:
        return np.asarray(arrays["landmarks"], dtype=np.float32), float(arrays["base_sigma"][0]), key
    if str(landmark_sampling) == "frequency_tail":
        landmarks = select_landmarks_frequency_aware(inventory.event_vectors, str(landmark_mode), stable_seed(inventory.name, modality, landmark_mode, landmark_sampling))
    else:
        landmarks = select_landmarks(inventory.event_vectors, str(landmark_mode), stable_seed(inventory.name, modality, landmark_mode))
    base_sigma = median_pairwise_scale(landmarks, stable_seed(inventory.name, modality, "sigma", landmark_mode, landmark_sampling))
    ctx.feature_cache.save_npz(
        key,
        "landmarks.npz",
        metadata={
            "namespace": EXPERIMENT_ID,
            "representation": "one_hot_event_kme_v2_landmarks",
            "benchmark": inventory.name,
            "inventory_cache_key": inventory_cache_key,
            "landmark_mode": str(landmark_mode),
            "modality": modality,
            "landmark_sampling": str(landmark_sampling),
            "n_landmarks": int(landmarks.shape[0]),
            "base_sigma": float(base_sigma),
            "config_id": config_id,
        },
        landmarks=landmarks.astype(np.float32),
        base_sigma=np.asarray([base_sigma], dtype=np.float64),
    )
    return landmarks, base_sigma, key


def cached_kme_features_v2(
    ctx: RunnerContext,
    inventory: EventInventory,
    inventory_cache_key: str,
    landmark_mode: object,
    sigma_multiplier: object,
    sigma_strategy: str,
    modality_strategy: str,
    landmark_sampling: str,
    kernel_weighting: str,
    settings: dict,
) -> tuple[pd.DataFrame, str, str, dict[str, object]]:
    payload = kme_config_payload(
        version="v2",
        inventory=inventory,
        inventory_cache_key=inventory_cache_key,
        landmark_mode=landmark_mode,
        sigma_multiplier=sigma_multiplier,
        sigma_strategy=sigma_strategy,
        modality_strategy=modality_strategy,
        landmark_sampling=landmark_sampling,
        kernel_weighting=kernel_weighting,
        settings=settings,
    )
    config_id = kme_config_id(payload)
    key = make_cache_key("one_hot_event_kme_features_v2", params=payload)
    frame = ctx.feature_cache.load_frame(key, "features.csv.gz")
    if frame is not None:
        frame.index = frame.index.astype(str)
        return frame.astype(np.float32), key, "hit", {**payload, "kme_config_id": config_id}

    weights = feature_weights(settings, str(kernel_weighting))
    if str(modality_strategy) == "stratified":
        feature_blocks: list[pd.DataFrame] = []
        base_sigmas: dict[str, float] = {}
        landmark_counts: dict[str, int] = {}
        modalities = ["SBS", "DBS", "ID"]
        for modality in modalities:
            sub_inventory = _modality_inventory(inventory, modality, config_id)
            if sub_inventory.event_vectors.shape[0] == 0:
                continue
            landmarks, base_sigma, landmark_key = cached_landmarks_v2(
                ctx,
                sub_inventory,
                inventory_cache_key,
                landmark_mode,
                settings,
                modality=modality,
                landmark_sampling=landmark_sampling,
                config_id=config_id,
            )
            base_sigmas[modality] = float(base_sigma)
            landmark_counts[modality] = int(landmarks.shape[0])
            for multiplier in _sigma_multipliers_for_strategy(sigma_strategy, sigma_multiplier):
                block = kme_features(
                    sub_inventory,
                    landmarks,
                    base_sigma * float(multiplier),
                    chunk_size=int(settings.get("kernel_chunk_size", 512)),
                    max_events_per_sample=int(settings.get("kme_max_events_per_sample", 0) or 0),
                    weights=weights,
                )
                block = block.loc[:, [col for col in block.columns if col.startswith("one_hot_kme__")]]
                block = block.add_prefix(f"{modality.lower()}__sigma{str(multiplier).replace('.', 'p')}__")
                feature_blocks.append(block)
        if feature_blocks:
            frame = pd.concat(feature_blocks, axis=1).reindex(pd.Index(inventory.patient_ids, name="sample")).fillna(0.0)
        else:
            frame = pd.DataFrame(index=pd.Index(inventory.patient_ids, name="sample"))
        cov = pd.concat(
            [
                inventory.covariates.reindex(frame.index).fillna(0.0),
                modality_summary_covariates(inventory).reindex(frame.index).fillna(0.0),
            ],
            axis=1,
        )
        frame = pd.concat([frame, cov], axis=1).astype(np.float32)
    else:
        landmarks, base_sigma, landmark_key = cached_landmarks_v2(
            ctx,
            inventory,
            inventory_cache_key,
            landmark_mode,
            settings,
            modality="mixed",
            landmark_sampling=landmark_sampling,
            config_id=config_id,
        )
        base_sigmas = {"mixed": float(base_sigma)}
        landmark_counts = {"mixed": int(landmarks.shape[0])}
        blocks = []
        for multiplier in _sigma_multipliers_for_strategy(sigma_strategy, sigma_multiplier):
            block = kme_features(
                inventory,
                landmarks,
                base_sigma * float(multiplier),
                chunk_size=int(settings.get("kernel_chunk_size", 512)),
                max_events_per_sample=int(settings.get("kme_max_events_per_sample", 0) or 0),
                weights=weights,
            )
            block = block.add_prefix(f"mixed__sigma{str(multiplier).replace('.', 'p')}__")
            blocks.append(block)
        frame = pd.concat(blocks, axis=1).astype(np.float32)

    metadata = {
        "namespace": EXPERIMENT_ID,
        "representation": "one_hot_event_KME_v2",
        "benchmark": inventory.name,
        "inventory_cache_key": inventory_cache_key,
        "kme_config_id": config_id,
        "config": payload,
        "base_sigmas": base_sigmas,
        "landmark_counts": landmark_counts,
        "sample_count": int(frame.shape[0]),
        "feature_count": int(frame.shape[1]),
    }
    ctx.feature_cache.save_frame(key, frame, "features.csv.gz", metadata=metadata)
    return frame, key, "miss_created", {**payload, "kme_config_id": config_id, "base_sigmas": base_sigmas, "landmark_counts": landmark_counts}


def make_linear_classifier(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                SGDClassifier(
                    loss="log_loss",
                    penalty="elasticnet",
                    alpha=0.0005,
                    l1_ratio=0.5,
                    class_weight="balanced",
                    max_iter=1000,
                    tol=1e-3,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )


def make_linear_regressor(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=seed)),
        ]
    )


def _xgb_base_params(seed: int, ctx: RunnerContext, settings: dict) -> dict[str, object]:
    params: dict[str, object] = {
        "n_estimators": int(settings.get("xgb_estimators", 100)),
        "max_depth": int(settings.get("xgb_max_depth", 2)),
        "learning_rate": float(settings.get("xgb_learning_rate", 0.05)),
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "reg_lambda": 2.0,
        "reg_alpha": 0.05,
        "random_state": int(seed),
        "n_jobs": int(ctx.xgb_n_jobs),
        "tree_method": str(ctx.tree_method),
        "verbosity": 0,
    }
    if str(ctx.tree_method) == "gpu_hist":
        params["predictor"] = "gpu_predictor"
    return params


def make_xgb_classifier(y_train: np.ndarray, task: str, n_classes: int, seed: int, ctx: RunnerContext, settings: dict):
    from xgboost import XGBClassifier

    params = _xgb_base_params(seed, ctx, settings)
    if task == "binary":
        positives = float(np.sum(y_train == 1))
        negatives = float(np.sum(y_train == 0))
        params.update({"objective": "binary:logistic", "eval_metric": "auc", "scale_pos_weight": negatives / max(positives, 1.0)})
    else:
        params.update({"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": int(n_classes)})
    return XGBClassifier(**params)


def make_xgb_regressor(seed: int, ctx: RunnerContext, settings: dict):
    from xgboost import XGBRegressor

    params = _xgb_base_params(seed, ctx, settings)
    params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})
    return XGBRegressor(**params)


def encode_labels(labels: pd.Series, task: str) -> tuple[np.ndarray, np.ndarray]:
    if task == "regression":
        return labels.astype(float).to_numpy(dtype=np.float64), np.array([], dtype=object)
    if task == "binary":
        return labels.astype(int).to_numpy(dtype=np.int32), np.array([0, 1], dtype=object)
    classes = np.array(sorted(labels.astype(str).unique()), dtype=object)
    mapping = {value: i for i, value in enumerate(classes)}
    return labels.astype(str).map(mapping).astype(int).to_numpy(dtype=np.int32), classes


def safe_macro_auroc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    scores: list[float] = []
    y_true = np.asarray(y_true, dtype=np.int32)
    proba = np.asarray(proba, dtype=np.float64)
    for class_id in range(int(n_classes)):
        binary = (y_true == class_id).astype(np.int32)
        if binary.min() == binary.max():
            continue
        try:
            score = float(roc_auc_score(binary, proba[:, class_id]))
        except ValueError:
            continue
        if np.isfinite(score):
            scores.append(score)
    return float(np.mean(scores)) if scores else float("nan")


def splits_for(endpoint: EndpointData, y: np.ndarray, seed: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    if endpoint.task == "regression":
        splitter = KFold(n_splits=min(N_SPLITS, len(y)), shuffle=True, random_state=seed)
        return [(fold, tr, te) for fold, (tr, te) in enumerate(splitter.split(np.zeros(len(y)), y), start=1)]
    if endpoint.task == "multiclass_grouped":
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        groups = endpoint.groups.loc[endpoint.labels.index].to_numpy()
        return [(fold, tr, te) for fold, (tr, te) in enumerate(splitter.split(np.zeros(len(y)), y, groups), start=1)]
    min_class = int(pd.Series(y).value_counts().min())
    splitter = StratifiedKFold(n_splits=min(N_SPLITS, min_class), shuffle=True, random_state=seed)
    return [(fold, tr, te) for fold, (tr, te) in enumerate(splitter.split(np.zeros(len(y)), y), start=1)]


def _fit_predict(learner: str, endpoint: EndpointData, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, n_classes: int, seed: int, ctx: RunnerContext, settings: dict) -> np.ndarray:
    if endpoint.task == "regression":
        model = make_linear_regressor(seed) if learner == "linear" else make_xgb_regressor(seed, ctx, settings)
        try:
            model.fit(x_train, y_train)
        except Exception:
            if learner == "xgboost" and str(ctx.tree_method) == "gpu_hist":
                fallback_ctx = RunnerContext(settings={**ctx.settings, "tree_method": "hist"}, paths=ctx.paths, dry_run=ctx.dry_run, refresh_cache=ctx.refresh_cache)
                model = make_xgb_regressor(seed, fallback_ctx, settings)
                model.fit(x_train, y_train)
            else:
                raise
        return np.asarray(model.predict(x_test), dtype=np.float64)
    if learner == "linear":
        model = make_linear_classifier(seed)
    else:
        model = make_xgb_classifier(y_train, "binary" if endpoint.task == "binary" else "multiclass", n_classes, seed, ctx, settings)
    try:
        model.fit(x_train, y_train)
    except Exception:
        if learner == "xgboost" and str(ctx.tree_method) == "gpu_hist":
            fallback_ctx = RunnerContext(settings={**ctx.settings, "tree_method": "hist"}, paths=ctx.paths, dry_run=ctx.dry_run, refresh_cache=ctx.refresh_cache)
            model = make_xgb_classifier(y_train, "binary" if endpoint.task == "binary" else "multiclass", n_classes, seed, fallback_ctx, settings)
            model.fit(x_train, y_train)
        else:
            raise
    proba = np.asarray(model.predict_proba(x_test), dtype=np.float64)
    proba = np.nan_to_num(proba, nan=1.0 / max(n_classes, 1), posinf=1.0, neginf=0.0)
    row_sums = proba.sum(axis=1, keepdims=True)
    proba = np.divide(proba, row_sums, out=np.full_like(proba, 1.0 / max(proba.shape[1], 1)), where=row_sums > 0)
    out = np.zeros((x_test.shape[0], n_classes), dtype=np.float64)
    model_classes = model.named_steps["model"].classes_ if learner == "linear" else model.classes_
    for local_col, class_id in enumerate(model_classes):
        out[:, int(class_id)] = proba[:, local_col]
    return out


def evaluate_features(endpoint: EndpointData, features: pd.DataFrame, representation: str, learner: str, ctx: RunnerContext, settings: dict) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    common = endpoint.labels.index.astype(str).intersection(features.index.astype(str))
    labels = endpoint.labels.loc[common]
    x = features.loc[common].fillna(0.0).to_numpy(dtype=np.float32)
    y, classes = encode_labels(labels, endpoint.task)
    seed = stable_seed(endpoint.benchmark, endpoint.name, learner)
    splits = splits_for(endpoint, y, seed)
    n_classes = 2 if endpoint.task == "binary" else len(classes)
    pred = np.zeros(len(y), dtype=np.float64) if endpoint.task == "regression" else np.zeros((len(y), n_classes), dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    for fold, train_idx, test_idx in splits:
        fold_pred = _fit_predict(learner, endpoint, x[train_idx], y[train_idx], x[test_idx], n_classes, seed + fold, ctx, settings)
        if endpoint.task == "regression":
            pred[test_idx] = fold_pred
            rho = spearmanr(y[test_idx], fold_pred)[0]
            fold_rows.append({"repeat": 1, "fold": fold, "metric": "spearman", "score": float(rho), "n_test": int(len(test_idx))})
        else:
            pred[test_idx] = fold_pred
            if endpoint.task == "binary":
                fold_score = roc_auc_score(y[test_idx], fold_pred[:, 1]) if len(np.unique(y[test_idx])) == 2 else float("nan")
                fold_rows.append({"repeat": 1, "fold": fold, "metric": "auroc", "score": float(fold_score), "n_test": int(len(test_idx))})
            else:
                fold_score = safe_macro_auroc(y[test_idx], fold_pred, n_classes)
                fold_rows.append({"repeat": 1, "fold": fold, "metric": "macro_auroc", "score": float(fold_score), "n_test": int(len(test_idx))})
    if endpoint.task == "regression":
        score = float(spearmanr(y, pred)[0])
        metric = "spearman"
        balanced = float("nan")
        accuracy = float("nan")
        macro_f1 = float("nan")
        auprc = float("nan")
        mae = float(mean_absolute_error(y, pred))
        r2 = float(r2_score(y, pred))
        pred_frame = pd.DataFrame({"sample": common.astype(str), "true_value": y.astype(float), "pred_value": pred.astype(float), "repeat": 1})
    elif endpoint.task == "binary":
        score = float(roc_auc_score(y, pred[:, 1]))
        metric = "auroc"
        balanced = float(balanced_accuracy_score(y, (pred[:, 1] >= 0.5).astype(int)))
        accuracy = float(accuracy_score(y, (pred[:, 1] >= 0.5).astype(int)))
        macro_f1 = float(f1_score(y, (pred[:, 1] >= 0.5).astype(int), average="macro", zero_division=0))
        auprc = float(average_precision_score(y, pred[:, 1]))
        mae = float("nan")
        r2 = float("nan")
        pred_frame = pd.DataFrame({"sample": common.astype(str), "true_value": y.astype(int), "pred_class_1": pred[:, 1], "repeat": 1})
        pred_frame["pred_class_0"] = pred[:, 0]
    else:
        score = safe_macro_auroc(y, pred, n_classes)
        metric = "macro_auroc"
        predicted = np.argmax(pred, axis=1)
        balanced = float(balanced_accuracy_score(y, predicted))
        accuracy = float(accuracy_score(y, predicted))
        macro_f1 = float(f1_score(y, predicted, average="macro", zero_division=0))
        auprc = float("nan")
        mae = float("nan")
        r2 = float("nan")
        pred_frame = pd.DataFrame({"sample": common.astype(str), "true_value": y.astype(int), "predicted_class": predicted.astype(int), "repeat": 1})
        for i in range(pred.shape[1]):
            pred_frame[f"pred_class_{i}"] = pred[:, i]
    row = {
        "benchmark": endpoint.benchmark,
        "endpoint": endpoint.name,
        "task": endpoint.task,
        "representation": representation,
        "learner": learner,
        "metric": metric,
        "score": score,
        "auroc": score if metric in {"auroc", "macro_auroc"} else float("nan"),
        "auprc": auprc,
        "balanced_accuracy": balanced,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mae": mae,
        "r2": r2,
        "n_samples": int(len(y)),
        "n_classes": int(pd.Series(y).nunique()),
        "n_features": int(features.shape[1]),
        "n_folds": int(len(splits)),
        "folds": int(len(splits)),
        "repeats": 1,
        "oof_aggregate": True,
        "split_strategy": "5-fold KFold" if endpoint.task == "regression" else "5-fold StratifiedGroupKFold grouped by agent_core" if endpoint.task == "multiclass_grouped" else "5-fold StratifiedKFold",
        "tuning": "endpoint_oracle" if representation.startswith("one_hot") else "none",
        "linear_solver": "sgd_log_loss_elasticnet_v1" if learner == "linear" and endpoint.task != "regression" else "elastic_net_coordinate_descent_v1" if learner == "linear" else "",
        "xgb_tree_method": str(ctx.tree_method) if learner == "xgboost" else "",
    }
    pred_frame.insert(0, "learner", learner)
    pred_frame.insert(0, "representation", representation)
    pred_frame.insert(0, "endpoint", endpoint.name)
    pred_frame.insert(0, "benchmark", endpoint.benchmark)
    fold_frame = pd.DataFrame(fold_rows)
    if not fold_frame.empty:
        fold_frame.insert(0, "learner", learner)
        fold_frame.insert(0, "representation", representation)
        fold_frame.insert(0, "endpoint", endpoint.name)
        fold_frame.insert(0, "benchmark", endpoint.benchmark)
    return row, pred_frame, fold_frame


def write_figures(endpoint_results: pd.DataFrame, grid_results: pd.DataFrame, ctx: RunnerContext) -> None:
    fig_path = ctx.figures_dir / f"{EXPERIMENT_ID}_delta_bars.png"
    plot = endpoint_results[endpoint_results["representation"].astype(str).str.contains("one_hot_event_kme", case=False, na=False) & endpoint_results["representation"].astype(str).str.contains("oracle", case=False, na=False)].copy()
    if plot.empty:
        return
    labels = plot["benchmark"].astype(str) + " | " + plot["learner"].astype(str)
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.55 * len(plot) + 1.5)))
    colors = np.where(plot["delta_vs_standard"].to_numpy(dtype=float) >= 0, "#2a9d8f", "#b23a48")
    ax.barh(labels, plot["delta_vs_standard"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Delta vs Standard")
    ax.set_title("One-Hot Event KME Endpoint-Oracle Delta")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    heat_path = ctx.figures_dir / f"{EXPERIMENT_ID}_grid_heatmap.png"
    heat = grid_results.copy()
    heat["grid"] = heat["landmark_mode"].astype(str) + " / " + heat["sigma_multiplier"].astype(str)
    for col in ["sigma_strategy", "modality_strategy", "kernel_weighting"]:
        if col in heat.columns:
            heat["grid"] = heat["grid"] + " / " + heat[col].astype(str)
    pivot = heat.pivot_table(index=["benchmark", "learner"], columns="grid", values="delta_vs_standard", aggfunc="max")
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.6 * max(1, len(pivot)) + 1.5)))
    arr = pivot.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 1.0
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]), labels=pivot.columns.astype(str), rotation=35, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]), labels=[f"{idx[0]} | {idx[1]}" for idx in pivot.index])
    ax.set_title("One-Hot Event KME Grid Delta vs Standard")
    fig.colorbar(im, ax=ax, label="Delta vs Standard")
    fig.tight_layout()
    fig.savefig(heat_path, dpi=220)
    plt.close(fig)


def run(ctx: RunnerContext) -> None:
    settings = dict(((ctx.settings.get("experiments") or {}).get(EXPERIMENT_ID) or {}))
    settings.setdefault("d_context", 6)
    settings.setdefault("d_payload", 6)
    settings.setdefault("sigma_multipliers", [0.5, 1.0, 2.0])
    settings.setdefault("landmark_modes", [128, "all"])
    settings.setdefault("kme_version", "v2")
    settings.setdefault("sigma_strategies", ["single"])
    settings.setdefault("modality_strategies", ["mixed"])
    settings.setdefault("landmark_sampling_modes", ["random_unique"])
    settings.setdefault("kernel_weighting_modes", ["uniform"])
    selected_representation = str(settings.get("selected_representation") or f"one_hot_event_kme_{settings['kme_version']}_oracle")
    raw_kme_representation = str(settings.get("raw_kme_representation") or f"one_hot_event_kme_{settings['kme_version']}")
    ctx.tables_dir.mkdir(parents=True, exist_ok=True)
    ctx.figures_dir.mkdir(parents=True, exist_ok=True)
    ctx.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = ctx.logs_dir / f"{EXPERIMENT_ID}.log"
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"one_hot_event_kme_scout started {datetime.now(timezone.utc).isoformat()}\n")
        log.write(json.dumps({k: str(v) for k, v in settings.items()}, indent=2) + "\n")
    if ctx.dry_run:
        write_summary_csv([{"status": "planned", "experiment_id": EXPERIMENT_ID}], ctx.tables_dir / f"{EXPERIMENT_ID}_summary.csv")
        return

    grch37_dir = Path(ctx.paths["raw_data"]["grch37_dir"])
    fasta_path = find_fasta(grch37_dir)
    fasta_display_path = display_fasta_path(fasta_path, grch37_dir)
    fasta = FastaReader(fasta_path, cache_sequences=bool(settings.get("cache_fasta_sequences", True)))
    sbs, dbs, ids = read_cosmic_channels()
    try:
        mc3_inventory, mc3_endpoints, mc3_cache_key = build_mc3_inventory(ctx, settings, fasta, fasta_path, sbs, dbs, ids)
        kucab_inventory, kucab_endpoint, kucab_cache_key = build_kucab_inventory(ctx, settings, fasta, fasta_path, sbs, dbs, ids)
    finally:
        fasta.close()

    inventories = {mc3_inventory.name: mc3_inventory, kucab_inventory.name: kucab_inventory}
    inventory_cache_keys = {mc3_inventory.name: mc3_cache_key, kucab_inventory.name: kucab_cache_key}
    endpoints = [*mc3_endpoints, kucab_endpoint]
    learners = ["linear", "xgboost"]
    endpoint_results_path = ctx.tables_dir / f"{EXPERIMENT_ID}_endpoint_results.csv"
    grid_results_path = ctx.tables_dir / f"{EXPERIMENT_ID}_grid_results.csv"
    selected_params_path = ctx.tables_dir / f"{EXPERIMENT_ID}_selected_params.csv"
    feature_manifest_path = ctx.tables_dir / f"{EXPERIMENT_ID}_feature_manifest.csv"
    event_qc_path = ctx.tables_dir / f"{EXPERIMENT_ID}_event_qc.csv"
    oof_predictions_path = ctx.tables_dir / f"{EXPERIMENT_ID}_oof_predictions.csv"
    fold_metrics_path = ctx.tables_dir / f"{EXPERIMENT_ID}_fold_metrics.csv"
    existing_endpoint_rows = pd.read_csv(endpoint_results_path).to_dict("records") if endpoint_results_path.exists() else []
    standard_rows: list[dict[str, object]] = [row for row in existing_endpoint_rows if row.get("representation") == "standard_sbs96_dbs78_id83"]
    burden_rows: list[dict[str, object]] = [row for row in existing_endpoint_rows if row.get("representation") == "burden_only"]
    selected_rows: list[dict[str, object]] = [row for row in existing_endpoint_rows if row.get("representation") == selected_representation]
    grid_rows: list[dict[str, object]] = pd.read_csv(grid_results_path).to_dict("records") if grid_results_path.exists() else []
    prediction_rows: list[pd.DataFrame] = [pd.read_csv(oof_predictions_path)] if oof_predictions_path.exists() else []
    fold_metric_rows: list[pd.DataFrame] = [pd.read_csv(fold_metrics_path)] if fold_metrics_path.exists() else []
    feature_rows: list[dict[str, object]] = []
    landmark_cache: dict[tuple[str, str], tuple[np.ndarray, float, str]] = {}
    kme_feature_cache: dict[tuple[str, str, float], tuple[pd.DataFrame, str, str]] = {}
    kme_feature_cache_v2: dict[str, tuple[pd.DataFrame, str, str, dict[str, object]]] = {}

    standard_features = {}
    burden_features = {}
    for inventory in inventories.values():
        standard = standard_features_from_counts(inventory.standard_counts, inventory.covariates)
        standard_features[inventory.name] = standard
        burden = burden_features_from_covariates(inventory.covariates)
        burden_features[inventory.name] = burden
        feature_rows.append({"benchmark": inventory.name, "representation": "standard_sbs96_dbs78_id83", "n_features": standard.shape[1], "n_samples": standard.shape[0], "cache_key": inventory_cache_keys[inventory.name], "construction": "same-inventory SBS96+DBS78+ID83 frequencies plus burden/modality covariates"})
        feature_rows.append({"benchmark": inventory.name, "representation": "burden_only", "n_features": burden.shape[1], "n_samples": burden.shape[0], "cache_key": inventory_cache_keys[inventory.name], "construction": "log total eligible event burden"})

    def save_current_results() -> None:
        if standard_rows or burden_rows or selected_rows:
            frame = pd.DataFrame([*burden_rows, *standard_rows, *selected_rows])
            if "representation" in frame.columns:
                selected_mask = frame["representation"].astype(str).eq(selected_representation)
                regular = frame[~selected_mask].drop_duplicates(["benchmark", "endpoint", "learner", "representation"], keep="last")
                selected = frame[selected_mask].copy()
                if not selected.empty:
                    selected["_trial_priority"] = pd.to_numeric(selected.get("optuna_trials_completed"), errors="coerce").fillna(-1)
                    selected["_score_priority"] = pd.to_numeric(selected.get("score"), errors="coerce").fillna(-np.inf)
                    selected = selected.sort_values(["benchmark", "endpoint", "learner", "representation", "_trial_priority", "_score_priority"]).drop_duplicates(["benchmark", "endpoint", "learner", "representation"], keep="last")
                    selected = selected.drop(columns=["_trial_priority"])
                    selected = selected.drop(columns=["_score_priority"])
                frame = pd.concat([regular, selected], ignore_index=True, sort=False)
            else:
                frame = frame.drop_duplicates(["benchmark", "endpoint", "learner", "representation"], keep="last")
            atomic_write_csv(frame, endpoint_results_path, index=False)
        if grid_rows:
            frame = pd.DataFrame(grid_rows)
            grid_subset = ["benchmark", "endpoint", "learner", "representation", "landmark_mode", "sigma_multiplier"]
            for col in ["kme_config_id", "sigma_strategy", "modality_strategy", "landmark_sampling", "kernel_weighting"]:
                if col in frame.columns:
                    grid_subset.append(col)
            grid_subset = [col for col in dict.fromkeys(grid_subset) if col in frame.columns]
            frame = frame.drop_duplicates(grid_subset, keep="last")
            atomic_write_csv(frame, grid_results_path, index=False)
        if selected_rows:
            selected_frame = pd.DataFrame(selected_rows).drop_duplicates(["benchmark", "endpoint", "learner", "representation"], keep="last")
            selected_cols = [
                "benchmark", "endpoint", "learner", "score", "delta_vs_standard", "kme_version", "kme_config_id",
                "landmark_mode", "n_landmarks", "base_sigma", "sigma_multiplier", "sigma_strategy",
                "sigma_multipliers_selected", "kernel_sigma", "modality_strategy", "landmark_sampling", "kernel_weighting",
                "optuna_study_name", "optuna_trials_completed",
            ]
            atomic_write_csv(selected_frame[[col for col in selected_cols if col in selected_frame.columns]], selected_params_path, index=False)
        if feature_rows:
            frame = pd.DataFrame(feature_rows).drop_duplicates(["benchmark", "representation"], keep="last")
            atomic_write_csv(frame, feature_manifest_path, index=False)
        if prediction_rows:
            pred_frame = pd.concat(prediction_rows, ignore_index=True, sort=False)
            pred_subset = ["benchmark", "endpoint", "representation", "learner", "sample"]
            if "kme_config_id" in pred_frame.columns:
                pred_subset.append("kme_config_id")
            pred_frame = pred_frame.drop_duplicates(pred_subset, keep="last")
            atomic_write_csv(pred_frame, oof_predictions_path, index=False)
        if fold_metric_rows:
            fold_frame = pd.concat(fold_metric_rows, ignore_index=True, sort=False)
            fold_subset = ["benchmark", "endpoint", "representation", "learner", "repeat", "fold"]
            if "kme_config_id" in fold_frame.columns:
                fold_subset.append("kme_config_id")
            fold_frame = fold_frame.drop_duplicates(fold_subset, keep="last")
            atomic_write_csv(fold_frame, fold_metrics_path, index=False)
        qc_list = []
        for inv in inventories.values():
            qc_list.append({"benchmark": inv.name, **inv.qc, "event_vector_dim": event_dim(settings), "d_context": int(settings["d_context"]), "d_payload": int(settings["d_payload"]), "fasta": fasta_display_path, "cache_key": inventory_cache_keys[inv.name]})
        atomic_write_csv(pd.DataFrame(qc_list), event_qc_path, index=False)
        if selected_rows and grid_rows:
            write_figures(pd.DataFrame([*burden_rows, *standard_rows, *selected_rows]), pd.DataFrame(grid_rows), ctx)

    def row_is_current(row: dict[str, object], *, tuned: bool) -> bool:
        folds = pd.to_numeric(pd.Series([row.get("n_folds", row.get("folds"))]), errors="coerce").iloc[0]
        repeats = pd.to_numeric(pd.Series([row.get("repeats", 1)]), errors="coerce").iloc[0]
        if not pd.notna(folds) or int(folds) != int(ctx.cv_folds) or (pd.notna(repeats) and int(repeats) != int(ctx.cv_repeats)):
            return False
        if tuned:
            trials = pd.to_numeric(pd.Series([row.get("optuna_trials_completed")]), errors="coerce").iloc[0]
            if not pd.notna(trials) or int(trials) != int(settings.get("optuna_trials", ctx.optuna_trials)):
                return False
            if str(row.get("kme_version", settings["kme_version"])) != str(settings["kme_version"]):
                return False
            if str(settings.get("kme_version")) == "v2" and not str(row.get("kme_config_id", "")).strip():
                return False
        return True

    for endpoint in endpoints:
        inventory = inventories[endpoint.benchmark]
        inventory_cache_key = inventory_cache_keys[endpoint.benchmark]
        standard = standard_features[endpoint.benchmark]
        burden = burden_features[endpoint.benchmark]
        for learner in learners:
            existing_selected = [
                row for row in selected_rows
                if str(row.get("benchmark")) == endpoint.benchmark and str(row.get("endpoint")) == endpoint.name and str(row.get("learner")) == learner
                and row_is_current(row, tuned=True)
            ]
            selected_already_done = bool(existing_selected)
            existing_burden = [
                row for row in burden_rows
                if str(row.get("benchmark")) == endpoint.benchmark and str(row.get("endpoint")) == endpoint.name and str(row.get("learner")) == learner
                and row_is_current(row, tuned=False)
            ]
            if existing_burden:
                print(f"[{EXPERIMENT_ID}] [checkpoint] reuse {endpoint.benchmark} {endpoint.name} {learner} burden", flush=True)
            else:
                print(f"[{EXPERIMENT_ID}] evaluating {endpoint.benchmark} {endpoint.name} {learner} burden", flush=True)
                burden_row, burden_pred, burden_folds = evaluate_features(endpoint, burden, "burden_only", learner, ctx, settings)
                burden_row.update({"delta_vs_standard": float("nan"), "cache_key": inventory_cache_key, "oof_prediction_file": oof_predictions_path.name, "fold_metrics_file": fold_metrics_path.name, "optuna_trials_completed": 0})
                burden_rows.append(burden_row)
                prediction_rows.append(burden_pred)
                if not burden_folds.empty:
                    fold_metric_rows.append(burden_folds)
                save_current_results()
            existing_standard = [
                row for row in standard_rows
                if str(row.get("benchmark")) == endpoint.benchmark and str(row.get("endpoint")) == endpoint.name and str(row.get("learner")) == learner
                and row_is_current(row, tuned=False)
            ]
            if existing_standard:
                std_row = dict(existing_standard[-1])
                print(f"[{EXPERIMENT_ID}] [checkpoint] reuse {endpoint.benchmark} {endpoint.name} {learner} standard", flush=True)
            else:
                print(f"[{EXPERIMENT_ID}] evaluating {endpoint.benchmark} {endpoint.name} {learner} standard", flush=True)
                std_row, std_pred, std_folds = evaluate_features(endpoint, standard, "standard_sbs96_dbs78_id83", learner, ctx, settings)
                std_row.update({"delta_vs_standard": 0.0, "cache_key": inventory_cache_key, "oof_prediction_file": oof_predictions_path.name, "fold_metrics_file": fold_metrics_path.name, "optuna_trials_completed": 0})
                standard_rows.append(std_row)
                prediction_rows.append(std_pred)
                if not std_folds.empty:
                    fold_metric_rows.append(std_folds)
                save_current_results()
            if selected_already_done:
                print(f"[{EXPERIMENT_ID}] [checkpoint] skip selected {endpoint.benchmark} {endpoint.name} {learner}", flush=True)
                continue
            best_row: dict[str, object] | None = None

            def evaluate_kme_combo(
                landmark_mode: object,
                sigma_multiplier: object,
                *,
                sigma_strategy: str = "single",
                modality_strategy: str = "mixed",
                landmark_sampling: str = "random_unique",
                kernel_weighting: str = "uniform",
            ) -> dict[str, object]:
                config_payload: dict[str, object] | None = None
                config_id = ""
                if str(settings.get("kme_version")) == "v2":
                    config_payload = kme_config_payload(
                        version="v2",
                        inventory=inventory,
                        inventory_cache_key=inventory_cache_key,
                        landmark_mode=landmark_mode,
                        sigma_multiplier=sigma_multiplier,
                        sigma_strategy=sigma_strategy,
                        modality_strategy=modality_strategy,
                        landmark_sampling=landmark_sampling,
                        kernel_weighting=kernel_weighting,
                        settings=settings,
                    )
                    config_id = kme_config_id(config_payload)
                existing_grid = [
                    row for row in grid_rows
                    if str(row.get("benchmark")) == endpoint.benchmark
                    and str(row.get("endpoint")) == endpoint.name
                    and str(row.get("learner")) == learner
                    and str(row.get("representation", raw_kme_representation)) == raw_kme_representation
                    and str(row.get("landmark_mode")) == str(landmark_mode)
                    and float(row.get("sigma_multiplier")) == float(sigma_multiplier)
                    and (not config_id or str(row.get("kme_config_id", "")) == config_id)
                    and row_is_current(row, tuned=False)
                ]
                if existing_grid:
                    print(f"[{EXPERIMENT_ID}] [checkpoint] reuse {endpoint.benchmark} {learner} KME {config_id or ''} landmarks={landmark_mode} sigma_multiplier={sigma_multiplier}", flush=True)
                    return dict(existing_grid[-1])
                print(
                    f"[{EXPERIMENT_ID}] evaluating {endpoint.benchmark} {learner} KME {config_id or ''} landmarks={landmark_mode} sigma_multiplier={sigma_multiplier} sigma_strategy={sigma_strategy} modality={modality_strategy} sampling={landmark_sampling} weighting={kernel_weighting}",
                    flush=True,
                )
                if str(settings.get("kme_version")) == "v2":
                    assert config_payload is not None
                    if config_id not in kme_feature_cache_v2:
                        kme_feature_cache_v2[config_id] = cached_kme_features_v2(
                            ctx,
                            inventory,
                            inventory_cache_key,
                            landmark_mode,
                            sigma_multiplier,
                            sigma_strategy,
                            modality_strategy,
                            landmark_sampling,
                            kernel_weighting,
                            settings,
                        )
                    features, kme_cache_key, cache_status, config_meta = kme_feature_cache_v2[config_id]
                    base_sigmas = config_meta.get("base_sigmas") or {}
                    landmark_counts = config_meta.get("landmark_counts") or {}
                    n_landmarks = int(sum(int(value) for value in dict(landmark_counts).values())) if landmark_counts else int(landmark_mode)
                    base_sigma = float(np.nanmean(list(dict(base_sigmas).values()))) if base_sigmas else float("nan")
                    sigma = base_sigma * float(sigma_multiplier) if np.isfinite(base_sigma) else float("nan")
                else:
                    landmark_key = (endpoint.benchmark, str(landmark_mode))
                    if landmark_key not in landmark_cache:
                        landmarks, base_sigma, landmark_cache_key = cached_landmarks(ctx, inventory, inventory_cache_key, landmark_mode, settings)
                        landmark_cache[landmark_key] = (landmarks, base_sigma, landmark_cache_key)
                    else:
                        landmarks, base_sigma, landmark_cache_key = landmark_cache[landmark_key]
                    sigma = base_sigma * float(sigma_multiplier)
                    feature_key = (endpoint.benchmark, str(landmark_mode), float(sigma_multiplier))
                    if feature_key not in kme_feature_cache:
                        kme_feature_cache[feature_key] = cached_kme_features(ctx, inventory, inventory_cache_key, landmarks, landmark_cache_key, sigma, sigma_multiplier, settings)
                    features, kme_cache_key, cache_status = kme_feature_cache[feature_key]
                    config_id = "kme_" + stable_json_hash(
                        {
                            "version": "v1",
                            "inventory": inventory.name,
                            "inventory_cache_key": inventory_cache_key,
                            "landmark_mode": str(landmark_mode),
                            "sigma_multiplier": float(sigma_multiplier),
                            "event_cap": int(settings.get("kme_max_events_per_sample", 0) or 0),
                        },
                        length=24,
                    )
                    n_landmarks = int(landmarks.shape[0])
                    sigma_strategy = "single"
                    modality_strategy = "mixed"
                    landmark_sampling = "random_unique"
                    kernel_weighting = "uniform"
                row, pred_frame, fold_frame = evaluate_features(endpoint, features, raw_kme_representation, learner, ctx, settings)
                for frame in (pred_frame, fold_frame):
                    if frame is not None and not frame.empty:
                        frame["kme_config_id"] = config_id
                        frame["kme_version"] = str(settings.get("kme_version"))
                        frame["landmark_mode"] = str(landmark_mode)
                        frame["sigma_multiplier"] = float(sigma_multiplier)
                        frame["sigma_strategy"] = str(sigma_strategy)
                        frame["modality_strategy"] = str(modality_strategy)
                        frame["landmark_sampling"] = str(landmark_sampling)
                        frame["kernel_weighting"] = str(kernel_weighting)
                row.update(
                    {
                        "kme_version": str(settings.get("kme_version")),
                        "kme_config_id": config_id,
                        "landmark_mode": str(landmark_mode),
                        "n_landmarks": int(n_landmarks),
                        "base_sigma": float(base_sigma),
                        "sigma_multiplier": float(sigma_multiplier),
                        "sigma_strategy": str(sigma_strategy),
                        "sigma_multipliers_selected": ",".join(map(str, _sigma_multipliers_for_strategy(sigma_strategy, sigma_multiplier))),
                        "kernel_sigma": float(sigma),
                        "modality_strategy": str(modality_strategy),
                        "landmark_sampling": str(landmark_sampling),
                        "kernel_weighting": str(kernel_weighting),
                        "delta_vs_standard": float(row["score"] - std_row["score"]),
                        "tuning": str(settings.get("tuning_strategy", "grid")),
                        "cache_key": kme_cache_key,
                        "cache_status": cache_status,
                        "inventory_cache_key": inventory_cache_key,
                        "oof_prediction_file": oof_predictions_path.name,
                        "fold_metrics_file": fold_metrics_path.name,
                    }
                )
                grid_rows.append(row)
                prediction_rows.append(pred_frame)
                if not fold_frame.empty:
                    fold_metric_rows.append(fold_frame)
                save_current_results()
                return row

            if str(settings.get("tuning_strategy", "grid")).lower() == "optuna":
                import optuna

                ctx.optuna_storage_dir.mkdir(parents=True, exist_ok=True)
                sampler = optuna.samplers.TPESampler(seed=stable_seed(endpoint.benchmark, endpoint.name, learner, "optuna"))
                storage = f"sqlite:///{(ctx.optuna_storage_dir / 'one_hot_event_kme_scout.sqlite3').as_posix()}"
                study_name = "one_hot_kme__" + stable_json_hash(
                    {
                        "study_schema": OPTUNA_STUDY_SCHEMA,
                        "kme_version": str(settings.get("kme_version")),
                        "benchmark": endpoint.benchmark,
                        "endpoint": endpoint.name,
                        "learner": learner,
                        "landmark_modes": [str(value) for value in settings["landmark_modes"]],
                        "sigma_multipliers": [float(value) for value in settings["sigma_multipliers"]],
                        "sigma_strategies": list(map(str, settings.get("sigma_strategies", ["single"]))),
                        "modality_strategies": list(map(str, settings.get("modality_strategies", ["mixed"]))),
                        "landmark_sampling_modes": list(map(str, settings.get("landmark_sampling_modes", ["random_unique"]))),
                        "kernel_weighting_modes": list(map(str, settings.get("kernel_weighting_modes", ["uniform"]))),
                        "trials": int(settings.get("optuna_trials", 10)),
                        "inventory_cache_key": inventory_cache_key,
                    },
                    length=24,
                )
                storage_backend = "sqlite"
                try:
                    study = optuna.create_study(direction="maximize", sampler=sampler, storage=storage, study_name=study_name, load_if_exists=True)
                except Exception as exc:
                    storage_backend = f"in_memory_after_sqlite_create_error:{type(exc).__name__}"
                    log.write(f"[optuna] sqlite create failed for {study_name}; using in-memory study ({type(exc).__name__}: {exc})\n")
                    study = optuna.create_study(direction="maximize", sampler=sampler)

                def objective(trial):
                    landmark_mode = trial.suggest_categorical("landmark_mode", [str(value) for value in settings["landmark_modes"]])
                    sigma_multiplier = trial.suggest_categorical("sigma_multiplier", [float(value) for value in settings["sigma_multipliers"]])
                    sigma_strategy = trial.suggest_categorical("sigma_strategy", [str(value) for value in settings.get("sigma_strategies", ["single"])])
                    modality_strategy = trial.suggest_categorical("modality_strategy", [str(value) for value in settings.get("modality_strategies", ["mixed"])])
                    landmark_sampling = trial.suggest_categorical("landmark_sampling", [str(value) for value in settings.get("landmark_sampling_modes", ["random_unique"])])
                    kernel_weighting = trial.suggest_categorical("kernel_weighting", [str(value) for value in settings.get("kernel_weighting_modes", ["uniform"])])
                    row = evaluate_kme_combo(
                        landmark_mode,
                        sigma_multiplier,
                        sigma_strategy=sigma_strategy,
                        modality_strategy=modality_strategy,
                        landmark_sampling=landmark_sampling,
                        kernel_weighting=kernel_weighting,
                    )
                    trial.set_user_attr("row", row)
                    return float(row["score"])

                target_trials = int(settings.get("optuna_trials", ctx.optuna_trials))
                completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
                if len(completed) > target_trials:
                    raise RuntimeError(f"Optuna study {study_name} has {len(completed)} completed trials; expected exactly {target_trials}. Use --refresh-cache or remove the study to rebuild.")
                remaining = target_trials - len(completed)
                try:
                    if remaining:
                        study.optimize(objective, n_trials=remaining, n_jobs=1, show_progress_bar=False)
                except Exception as exc:
                    if storage_backend != "sqlite":
                        raise
                    storage_backend = f"in_memory_after_sqlite_optimize_error:{type(exc).__name__}"
                    log.write(f"[optuna] sqlite optimize failed for {study_name}; retrying {target_trials} in-memory trials ({type(exc).__name__}: {exc})\n")
                    sampler = optuna.samplers.TPESampler(seed=stable_seed(endpoint.benchmark, endpoint.name, learner, "optuna", "fallback"))
                    study = optuna.create_study(direction="maximize", sampler=sampler)
                    study.optimize(objective, n_trials=target_trials, n_jobs=1, show_progress_bar=False)
                completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
                if len(completed) != target_trials:
                    raise RuntimeError(f"Optuna study {study_name} completed {len(completed)} trials; expected exactly {target_trials}.")
                best_trial = max(completed, key=lambda trial: float(trial.value if trial.value is not None else -np.inf))
                best_row = dict(best_trial.user_attrs["row"])
                best_row["optuna_study_name"] = study_name
                best_row["optuna_trials_completed"] = int(len(completed))
                best_row["optuna_storage_backend"] = storage_backend
            else:
                for landmark_mode in settings["landmark_modes"]:
                    for sigma_multiplier in settings["sigma_multipliers"]:
                        for sigma_strategy in settings.get("sigma_strategies", ["single"]):
                            for modality_strategy in settings.get("modality_strategies", ["mixed"]):
                                for landmark_sampling in settings.get("landmark_sampling_modes", ["random_unique"]):
                                    for kernel_weighting in settings.get("kernel_weighting_modes", ["uniform"]):
                                        row = evaluate_kme_combo(
                                            landmark_mode,
                                            sigma_multiplier,
                                            sigma_strategy=str(sigma_strategy),
                                            modality_strategy=str(modality_strategy),
                                            landmark_sampling=str(landmark_sampling),
                                            kernel_weighting=str(kernel_weighting),
                                        )
                                        if best_row is None or float(row["score"]) > float(best_row["score"]):
                                            best_row = dict(row)
                best_row["optuna_trials_completed"] = 0
            assert best_row is not None
            selected = dict(best_row)
            selected["representation"] = selected_representation
            selected["oof_prediction_file"] = oof_predictions_path.name
            selected["fold_metrics_file"] = fold_metrics_path.name
            selected_rows.append(selected)

            def _copy_selected_frames() -> None:
                selected_config = str(selected.get("kme_config_id", "")).strip()
                if not selected_config:
                    raise RuntimeError(f"Selected KME row for {endpoint.benchmark}/{endpoint.name}/{learner} lacks kme_config_id")
                all_preds = pd.concat(prediction_rows, ignore_index=True, sort=False) if prediction_rows else pd.DataFrame()
                pred_match = all_preds[
                    all_preds.get("benchmark", pd.Series(dtype=object)).astype(str).eq(endpoint.benchmark)
                    & all_preds.get("endpoint", pd.Series(dtype=object)).astype(str).eq(endpoint.name)
                    & all_preds.get("learner", pd.Series(dtype=object)).astype(str).eq(learner)
                    & all_preds.get("representation", pd.Series(dtype=object)).astype(str).eq(raw_kme_representation)
                    & all_preds.get("kme_config_id", pd.Series(dtype=object)).astype(str).eq(selected_config)
                ].copy()
                if pred_match.empty:
                    raise RuntimeError(f"Could not find OOF predictions for selected KME config {selected_config} ({endpoint.benchmark}/{endpoint.name}/{learner})")
                pred_match["representation"] = selected_representation
                prediction_rows.append(pred_match)
                all_folds = pd.concat(fold_metric_rows, ignore_index=True, sort=False) if fold_metric_rows else pd.DataFrame()
                if not all_folds.empty:
                    fold_match = all_folds[
                        all_folds.get("benchmark", pd.Series(dtype=object)).astype(str).eq(endpoint.benchmark)
                        & all_folds.get("endpoint", pd.Series(dtype=object)).astype(str).eq(endpoint.name)
                        & all_folds.get("learner", pd.Series(dtype=object)).astype(str).eq(learner)
                        & all_folds.get("representation", pd.Series(dtype=object)).astype(str).eq(raw_kme_representation)
                        & all_folds.get("kme_config_id", pd.Series(dtype=object)).astype(str).eq(selected_config)
                    ].copy()
                    if not fold_match.empty:
                        fold_match["representation"] = selected_representation
                        fold_metric_rows.append(fold_match)

            _copy_selected_frames()
            feature_rows.append(
                {
                    "benchmark": endpoint.benchmark,
                    "representation": f"{selected_representation}_{learner}",
                    "n_features": int(selected["n_features"]),
                    "n_samples": int(selected["n_samples"]),
                    "n_landmarks": int(selected["n_landmarks"]),
                    "kme_version": str(selected.get("kme_version", settings["kme_version"])),
                    "kme_config_id": str(selected.get("kme_config_id", "")),
                    "modality_strategy": str(selected.get("modality_strategy", "")),
                    "sigma_strategy": str(selected.get("sigma_strategy", "")),
                    "landmark_sampling": str(selected.get("landmark_sampling", "")),
                    "kernel_weighting": str(selected.get("kernel_weighting", "")),
                    "kernel_sigma": float(selected["kernel_sigma"]),
                    "construction": f"FASTA one-hot {settings['kme_version']} d_context={settings['d_context']} d_payload={settings['d_payload']} endpoint-selected KME",
                }
            )
            save_current_results()

    endpoint_results = pd.DataFrame([*burden_rows, *standard_rows, *selected_rows])
    grid_results = pd.DataFrame(grid_rows)
    if not grid_results.empty:
        grid_subset = ["benchmark", "endpoint", "learner", "representation", "landmark_mode", "sigma_multiplier"]
        for col in ["kme_config_id", "sigma_strategy", "modality_strategy", "landmark_sampling", "kernel_weighting"]:
            if col in grid_results.columns:
                grid_subset.append(col)
        grid_subset = [col for col in dict.fromkeys(grid_subset) if col in grid_results.columns]
        grid_results = grid_results.drop_duplicates(grid_subset, keep="last")
    selected_params_frame = pd.DataFrame(selected_rows)
    selected_param_cols = [
        "benchmark", "endpoint", "learner", "score", "delta_vs_standard", "kme_version", "kme_config_id",
        "landmark_mode", "n_landmarks", "base_sigma", "sigma_multiplier", "sigma_strategy",
        "sigma_multipliers_selected", "kernel_sigma", "modality_strategy", "landmark_sampling", "kernel_weighting",
        "optuna_study_name", "optuna_trials_completed",
    ]
    selected_params = selected_params_frame[[col for col in selected_param_cols if col in selected_params_frame.columns]]
    qc_rows = []
    for inventory in inventories.values():
        row = {"benchmark": inventory.name, **inventory.qc, "event_vector_dim": event_dim(settings), "d_context": int(settings["d_context"]), "d_payload": int(settings["d_payload"]), "fasta": fasta_display_path, "cache_key": inventory_cache_keys[inventory.name]}
        qc_rows.append(row)
    qc = pd.DataFrame(qc_rows)
    feature_manifest = pd.DataFrame(feature_rows)

    atomic_write_csv(endpoint_results, endpoint_results_path, index=False)
    atomic_write_csv(grid_results, grid_results_path, index=False)
    atomic_write_csv(selected_params, selected_params_path, index=False)
    atomic_write_csv(qc, event_qc_path, index=False)
    atomic_write_csv(feature_manifest, feature_manifest_path, index=False)
    if prediction_rows:
        pred_frame = pd.concat(prediction_rows, ignore_index=True, sort=False)
        pred_subset = ["benchmark", "endpoint", "representation", "learner", "sample"]
        if "kme_config_id" in pred_frame.columns:
            pred_subset.append("kme_config_id")
        atomic_write_csv(pred_frame.drop_duplicates(pred_subset, keep="last"), oof_predictions_path, index=False)
    if fold_metric_rows:
        fold_frame = pd.concat(fold_metric_rows, ignore_index=True, sort=False)
        fold_subset = ["benchmark", "endpoint", "representation", "learner", "repeat", "fold"]
        if "kme_config_id" in fold_frame.columns:
            fold_subset.append("kme_config_id")
        atomic_write_csv(fold_frame.drop_duplicates(fold_subset, keep="last"), fold_metrics_path, index=False)
    write_figures(endpoint_results, grid_results, ctx)
    write_summary_csv(
        [
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "completed",
                "elapsed_seconds": round(time.time() - start, 3),
                "d_context": int(settings["d_context"]),
                "d_payload": int(settings["d_payload"]),
                "tree_method": ctx.tree_method,
            }
        ],
        ctx.tables_dir / f"{EXPERIMENT_ID}_summary.csv",
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"completed": True, "elapsed_seconds": round(time.time() - start, 3), "outputs": "results/tables and results/figures"}, indent=2) + "\n")
    for path in [ctx.tables_dir / f"{EXPERIMENT_ID}_event_qc.csv", log_path]:
        text = path.read_text(encoding="utf-8")
        path.write_text(sanitize_text(text), encoding="utf-8")
