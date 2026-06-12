"""MuAt-compatible preprocessing, model, and official CLI helpers.

This module intentionally separates two concepts:

* "MuAt" means the official package/checkpoints are configured and used.
* "MuAt-compatible" means local code follows the paper's data modalities and
  supplement architecture closely enough for manuscript-side smoke testing.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
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
) -> pd.DataFrame:
    """Convert MC3/MAF rows into MuAt-compatible mutation-event rows."""

    patient_set = set(map(str, patients))
    available = set(maf_header_columns(maf_path))
    usecols = [column for column in MAF_COMPATIBLE_USECOLS if column in available]
    if "Tumor_Sample_Barcode" not in usecols:
        raise ValueError("MAF input is missing Tumor_Sample_Barcode")
    rows: list[dict[str, object]] = []
    for chunk in pd.read_csv(maf_path, sep="\t", usecols=usecols, dtype=str, chunksize=chunksize):
        chunk["sample"] = chunk["Tumor_Sample_Barcode"].astype(str).str[:12]
        chunk = chunk[chunk["sample"].isin(patient_set)].copy()
        if chunk.empty:
            continue
        chunk["position"] = pd.to_numeric(chunk.get("Start_Position"), errors="coerce").fillna(0).astype(np.int64)
        chunk = chunk.sort_values(["sample", "Chromosome", "position"], kind="mergesort")
        for record in chunk.to_dict(orient="records"):
            ref = clean_allele(record.get("Reference_Allele"))
            alt = choose_alt(ref, record.get("Tumor_Seq_Allele1"), record.get("Tumor_Seq_Allele2"))
            rows.append(
                {
                    "sample": str(record.get("sample")),
                    "chromosome": normalize_chromosome(record.get("Chromosome")),
                    "position": int(record.get("position") or 0),
                    "motif": mutation_motif_token(record),
                    "position_token": position_bin_token(record.get("Chromosome"), record.get("position")),
                    "annotation": annotation_token(record),
                    "variant_type": str(record.get("Variant_Type") or ""),
                    "variant_classification": str(record.get("Variant_Classification") or ""),
                    "gene": str(record.get("Hugo_Symbol") or ""),
                    "ref": ref,
                    "alt": alt,
                    "context": str(record.get("CONTEXT") or ""),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["sample", "chromosome", "position", "motif", "position_token", "annotation"])
    return pd.DataFrame(rows).sort_values(["sample", "chromosome", "position", "motif"], kind="mergesort").reset_index(drop=True)


@dataclass(frozen=True)
class MuAtDictionaries:
    motif: dict[str, int]
    position: dict[str, int]
    annotation: dict[str, int]

    def sizes(self) -> dict[str, int]:
        return {"motif": len(self.motif), "position": len(self.position), "annotation": len(self.annotation)}

    def to_jsonable(self) -> dict[str, dict[str, int]]:
        return {"motif": self.motif, "position": self.position, "annotation": self.annotation}

    @classmethod
    def from_jsonable(cls, payload: Mapping[str, Mapping[str, int]]) -> "MuAtDictionaries":
        return cls(
            motif={str(k): int(v) for k, v in payload["motif"].items()},
            position={str(k): int(v) for k, v in payload["position"].items()},
            annotation={str(k): int(v) for k, v in payload["annotation"].items()},
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode event rows as integer modality tensors plus a valid-event mask."""

    patient_list = [str(patient) for patient in patients]
    patient_to_row = {patient: i for i, patient in enumerate(patient_list)}
    x = np.zeros((len(patient_list), int(max_events), 3), dtype=np.int64)
    mask = np.zeros((len(patient_list), int(max_events)), dtype=bool)
    counts = np.zeros(len(patient_list), dtype=np.int32)
    if events.empty:
        return x, mask, counts
    dicts = dictionaries.to_jsonable()
    for sample, sample_events in events.groupby("sample", sort=False):
        sample = str(sample)
        if sample not in patient_to_row:
            continue
        row_idx = patient_to_row[sample]
        limited = sample_events.sort_values(["chromosome", "position", "motif"], kind="mergesort").head(int(max_events))
        for event_idx, record in enumerate(limited.to_dict(orient="records")):
            x[row_idx, event_idx, 0] = dicts["motif"].get(str(record.get("motif")), dicts["motif"][UNK_TOKEN])
            x[row_idx, event_idx, 1] = dicts["position"].get(str(record.get("position_token")), dicts["position"][UNK_TOKEN])
            x[row_idx, event_idx, 2] = dicts["annotation"].get(str(record.get("annotation")), dicts["annotation"][UNK_TOKEN])
            mask[row_idx, event_idx] = True
            counts[row_idx] += 1
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
            motif_vocab_size: int,
            position_vocab_size: int,
            annotation_vocab_size: int,
            n_outputs: int,
            embed_dim: int = 128,
            attention_heads: int = 1,
            feature_dim: int = 24,
            dropout: float = 0.1,
            count_feature_mode: str = "none",
        ) -> None:
            super().__init__()
            width = int(embed_dim) * 3
            if width % int(attention_heads) != 0:
                raise ValueError(f"3 * embed_dim ({width}) must be divisible by attention_heads ({attention_heads})")
            count_feature_mode = str(count_feature_mode or "none").strip().lower()
            if count_feature_mode not in {"none", "log_count", "log_count_density"}:
                raise ValueError(f"Unsupported count_feature_mode: {count_feature_mode}")
            self.count_feature_mode = count_feature_mode
            self.motif_embedding = nn.Embedding(int(motif_vocab_size), int(embed_dim), padding_idx=0)
            self.position_embedding = nn.Embedding(int(position_vocab_size), int(embed_dim), padding_idx=0)
            self.annotation_embedding = nn.Embedding(int(annotation_vocab_size), int(embed_dim), padding_idx=0)
            self.dropout = nn.Dropout(float(dropout))
            self.attention = nn.MultiheadAttention(width, int(attention_heads), dropout=float(dropout), batch_first=True)
            self.norm1 = nn.LayerNorm(width)
            self.fc_event = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(float(dropout)), nn.Linear(width, width))
            self.norm2 = nn.LayerNorm(width)
            count_width = 0 if count_feature_mode == "none" else (2 if count_feature_mode == "log_count_density" else 1)
            self.feature_layer = nn.Linear(width + count_width, int(feature_dim))
            self.classifier = nn.Linear(int(feature_dim), int(n_outputs))

        def forward(self, x: "torch.Tensor", mask: "torch.Tensor", *, return_attention: bool = False):
            mask = mask.bool()
            empty = ~mask.any(dim=1)
            if torch.any(empty):
                mask = mask.clone()
                mask[empty, 0] = True
            z = torch.cat(
                [
                    self.motif_embedding(x[:, :, 0]),
                    self.position_embedding(x[:, :, 1]),
                    self.annotation_embedding(x[:, :, 2]),
                ],
                dim=-1,
            )
            z = self.dropout(z)
            attended, weights = self.attention(
                z,
                z,
                z,
                key_padding_mask=~mask,
                need_weights=return_attention,
                average_attn_weights=False,
            )
            z = self.norm1(z + attended)
            z = self.norm2(z + self.fc_event(z))
            valid = mask.unsqueeze(-1).to(z.dtype)
            event_count = valid.sum(dim=1).clamp_min(1.0)
            pooled = (z * valid).sum(dim=1) / event_count
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
                return logits, features, weights
            return logits, features

else:

    class MuAtCompatibleModel:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("torch is required for MuAtCompatibleModel")
