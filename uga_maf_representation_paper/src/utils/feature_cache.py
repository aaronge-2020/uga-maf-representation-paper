"""Persistent feature cache utilities for resumable manuscript benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from utils.checkpointing import atomic_write_csv, atomic_write_json, atomic_write_text
from utils.config import BUNDLE_ROOT, REPRO_ROOT


CACHE_SCHEMA_VERSION = 1
FINGERPRINT_BYTES = 1024 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_fingerprint(path: str | Path | None) -> dict[str, object]:
    """Return a cheap but useful fingerprint for one input file."""
    if path is None:
        return {"path": None, "exists": False}
    p = Path(path)
    if not p.exists():
        return {"path": _portable_path(p), "exists": False}
    stat = p.stat()
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        first = handle.read(FINGERPRINT_BYTES)
        digest.update(first)
        if stat.st_size > FINGERPRINT_BYTES:
            handle.seek(max(0, stat.st_size - FINGERPRINT_BYTES))
            digest.update(handle.read(FINGERPRINT_BYTES))
    return {
        "path": _portable_path(p),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256_head_tail": digest.hexdigest(),
    }


def directory_fingerprint(path: str | Path | None, *, patterns: Iterable[str] = ("*",)) -> dict[str, object]:
    """Fingerprint a directory by listing matching file fingerprints."""
    if path is None:
        return {"path": None, "exists": False, "files": []}
    p = Path(path)
    if not p.exists():
        return {"path": _portable_path(p), "exists": False, "files": []}
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(q for q in p.rglob(pattern) if q.is_file()))
    rows = []
    for file in sorted(set(files)):
        try:
            rel = str(file.relative_to(p))
        except ValueError:
            rel = file.name
        fp = file_fingerprint(file)
        fp["relative_path"] = rel
        rows.append(fp)
    return {"path": _portable_path(p), "exists": True, "files": rows}


def fasta_fingerprint(fasta_path: str | Path | None) -> dict[str, object]:
    """Fingerprint FASTA plus its required fai index."""
    if fasta_path is None:
        return {"fasta": file_fingerprint(None), "fai": file_fingerprint(None)}
    fasta = Path(fasta_path)
    return {
        "fasta": file_fingerprint(fasta),
        "fai": file_fingerprint(fasta.parent / f"{fasta.name}.fai"),
    }


def code_fingerprint(paths: Iterable[str | Path]) -> dict[str, object]:
    return {str(Path(path).name): file_fingerprint(path) for path in paths}


def stable_json_hash(payload: Any, *, length: int = 20) -> str:
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def make_cache_key(namespace: str, *, params: dict[str, Any], inputs: dict[str, Any] | None = None) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in namespace).strip("_")[:80]
    digest = stable_json_hash({"namespace": namespace, "params": params, "inputs": inputs or {}}, length=24)
    return f"{safe}_{digest}"


def _portable_path(path: Path) -> str:
    resolved = Path(path).resolve()
    for root, token in ((BUNDLE_ROOT, "${bundle_root}"), (REPRO_ROOT, "${repro_root}")):
        try:
            return str(Path(token) / resolved.relative_to(root))
        except ValueError:
            continue
    return str(path)


@dataclass(frozen=True)
class CacheEntry:
    key: str
    root: Path

    @property
    def dir(self) -> Path:
        return self.root / self.key

    @property
    def metadata_path(self) -> Path:
        return self.dir / "metadata.json"

    def path(self, name: str) -> Path:
        return self.dir / name


class FeatureCache:
    """Content-addressed cache with explicit metadata validation."""

    def __init__(self, root: str | Path, *, enabled: bool = True, refresh: bool = False):
        self.root = Path(root)
        self.enabled = bool(enabled)
        self.refresh = bool(refresh)
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def entry(self, key: str) -> CacheEntry:
        return CacheEntry(key=key, root=self.root)

    def has(self, key: str) -> bool:
        entry = self.entry(key)
        return self.enabled and not self.refresh and entry.metadata_path.exists()

    def read_metadata(self, key: str) -> dict[str, Any] | None:
        entry = self.entry(key)
        if not self.has(key):
            return None
        return json.loads(entry.metadata_path.read_text(encoding="utf-8"))

    def write_metadata(self, key: str, metadata: dict[str, Any]) -> None:
        if not self.enabled:
            return
        entry = self.entry(key)
        entry.dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "created_utc": _utc_now(),
            **metadata,
        }
        atomic_write_json(entry.metadata_path, payload)

    def load_frame(self, key: str, name: str = "frame.csv.gz", **kwargs: Any) -> pd.DataFrame | None:
        entry = self.entry(key)
        path = entry.path(name)
        if not self.has(key) or not path.exists():
            return None
        return pd.read_csv(path, index_col=0 if kwargs.pop("index_col_default", True) else None, **kwargs)

    def save_frame(self, key: str, frame: pd.DataFrame, name: str = "frame.csv.gz", *, metadata: dict[str, Any]) -> Path:
        entry = self.entry(key)
        entry.dir.mkdir(parents=True, exist_ok=True)
        path = entry.path(name)
        atomic_write_csv(frame, path, index=True)
        self.write_metadata(key, {**metadata, "artifacts": sorted(set([*(metadata.get("artifacts") or []), name]))})
        return path

    def load_npz(self, key: str, name: str = "matrix.npz") -> dict[str, np.ndarray] | None:
        entry = self.entry(key)
        path = entry.path(name)
        if not self.has(key) or not path.exists():
            return None
        loaded = np.load(path, allow_pickle=True)
        return {item: loaded[item] for item in loaded.files}

    def save_npz(self, key: str, name: str = "matrix.npz", *, metadata: dict[str, Any], **arrays: np.ndarray) -> Path:
        entry = self.entry(key)
        entry.dir.mkdir(parents=True, exist_ok=True)
        path = entry.path(name)
        tmp = path.with_name(f".{path.name}.tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp, path)
        self.write_metadata(key, {**metadata, "artifacts": sorted(set([*(metadata.get("artifacts") or []), name]))})
        return path

    def save_text(self, key: str, text: str, name: str, *, metadata: dict[str, Any]) -> Path:
        entry = self.entry(key)
        entry.dir.mkdir(parents=True, exist_ok=True)
        path = entry.path(name)
        atomic_write_text(path, text)
        self.write_metadata(key, {**metadata, "artifacts": sorted(set([*(metadata.get("artifacts") or []), name]))})
        return path


def snapshot_feature_artifacts(cache: FeatureCache, namespace: str, source_root: Path, patterns: Iterable[str]) -> dict[str, object]:
    """Copy generated feature/cache artifacts into a durable cache snapshot."""
    if not cache.enabled or cache.refresh:
        return {"namespace": namespace, "status": "disabled_or_refresh"}
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(path for path in Path(source_root).glob(pattern) if path.is_file()))
    files = sorted(set(files))
    key = make_cache_key(namespace, params={"kind": "legacy_artifact_snapshot"}, inputs={"source_root": _portable_path(Path(source_root))})
    entry = cache.entry(key)
    manifest_rows: list[dict[str, object]] = []
    for src in files:
        rel = src.relative_to(source_root)
        dst = entry.path(str(Path("files") / rel))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest_rows.append({"relative_path": str(rel), "bytes": int(src.stat().st_size)})
    if manifest_rows:
        atomic_write_csv(pd.DataFrame(manifest_rows), entry.path("artifact_manifest.csv"), index=False)
        cache.write_metadata(
            key,
            {
                "namespace": namespace,
                "status": "complete",
                "source_root": _portable_path(Path(source_root)),
                "file_count": len(manifest_rows),
                "artifacts": ["artifact_manifest.csv", "files/"],
            },
        )
    return {"namespace": namespace, "status": "snapshotted" if manifest_rows else "no_artifacts", "cache_key": key, "file_count": len(manifest_rows)}


def restore_feature_artifacts(cache: FeatureCache, namespace: str, source_root: Path) -> dict[str, object]:
    """Restore previously snapshotted legacy feature artifacts before running a legacy script."""
    if not cache.enabled or cache.refresh:
        return {"namespace": namespace, "status": "disabled_or_refresh"}
    key = make_cache_key(namespace, params={"kind": "legacy_artifact_snapshot"}, inputs={"source_root": _portable_path(Path(source_root))})
    entry = cache.entry(key)
    manifest_path = entry.path("artifact_manifest.csv")
    if not entry.metadata_path.exists() or not manifest_path.exists():
        return {"namespace": namespace, "status": "missing", "cache_key": key}
    manifest = pd.read_csv(manifest_path)
    restored = 0
    for rel in manifest["relative_path"].astype(str):
        src = entry.path(str(Path("files") / rel))
        dst = Path(source_root) / rel
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored += 1
    return {"namespace": namespace, "status": "restored", "cache_key": key, "restored_files": restored}
