"""Context-atlas loading and truncation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ATLAS52 = REPO / "cgr_validation_results" / "research" / "data" / "EXP022_atlas_genome_wide_45mer_universal.json"
ATLAS_D22 = REPO / "cgr_validation_results" / "research" / "data" / "EXP022_atlas_genome_wide_45mer_universal_d22.json"
DEFAULT_DINUCLEOTIDE_ATLAS = REPO / "cgr_validation_results" / "research" / "data" / "dinucleotide_atlas_d45.json"
GRCH37_FASTA = REPO / "data" / "GRCH37" / "GCF_000001405.13" / "GCF_000001405.13_GRCh37_genomic.fna"


def truncate_context(vec: np.ndarray | list[float], d_context: int) -> np.ndarray:
    """Truncate a [Lx, Ly, Rx, Ry] context centroid to d_context bits per block."""
    arr = np.asarray(vec, dtype=np.float64)
    d_context = int(d_context)
    if arr.ndim != 1 or len(arr) % 4 != 0:
        raise ValueError(f"Context vector must be one-dimensional with four equal blocks, got {arr.shape}")
    max_d = len(arr) // 4
    if d_context > max_d:
        raise ValueError(f"Requested d_context={d_context}, but atlas supports only {max_d}")
    return np.concatenate([arr[i * max_d : i * max_d + d_context] for i in range(4)]).astype(np.float64)


def load_context_atlas(path: Path | str = ATLAS52, d_context: int = 10) -> dict[str, np.ndarray]:
    """Load an SBS context atlas and return d_context-truncated centroids."""
    atlas_path = Path(path)
    raw = json.loads(atlas_path.read_text(encoding="utf-8"))
    return {str(k): truncate_context(v, d_context) for k, v in raw.items()}


_DINO_CACHE: dict[tuple[str, int | None], dict[str, np.ndarray]] = {}


def load_dinucleotide_context_atlas(
    path: Path | str = DEFAULT_DINUCLEOTIDE_ATLAS,
    d_context: int | None = None,
) -> dict[str, np.ndarray]:
    """Load a DBS context atlas, optionally truncating to d_context."""
    atlas_path = Path(path)
    key = (str(atlas_path.resolve()), None if d_context is None else int(d_context))
    if key in _DINO_CACHE:
        return _DINO_CACHE[key]
    if not atlas_path.is_file():
        _DINO_CACHE[key] = {}
        return {}
    raw = json.loads(atlas_path.read_text(encoding="utf-8"))
    if d_context is None:
        out = {str(k): np.asarray(v, dtype=np.float64) for k, v in raw.items()}
    else:
        out = {str(k): truncate_context(v, int(d_context)) for k, v in raw.items()}
    _DINO_CACHE[key] = out
    return out
