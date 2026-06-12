"""Small durable checkpoint helpers for long-running manuscript scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def atomic_write_csv(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        tmp = path.with_name(f".{path.stem}.tmp.gz")
        kwargs.setdefault("compression", "gzip")
    else:
        tmp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(tmp, **kwargs)
    os.replace(tmp, path)


def read_completed_keys(path: Path, key_columns: Sequence[str]) -> set[tuple[str, ...]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path, usecols=lambda col: col in set(key_columns), low_memory=False)
    if frame.empty:
        return set()
    missing = [col for col in key_columns if col not in frame.columns]
    if missing:
        return set()
    return {tuple(str(value) for value in row) for row in frame.loc[:, list(key_columns)].itertuples(index=False, name=None)}


def merge_checkpoint_rows(
    path: Path,
    rows: Iterable[dict],
    *,
    key_columns: Sequence[str],
    sort_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Merge rows into a checkpoint CSV, keeping the latest row for each key."""
    path = Path(path)
    new_frame = pd.DataFrame(list(rows))
    if new_frame.empty:
        if path.exists():
            return pd.read_csv(path, low_memory=False)
        return new_frame
    if path.exists() and path.stat().st_size > 0:
        old_frame = pd.read_csv(path, low_memory=False)
        frame = pd.concat([old_frame, new_frame], ignore_index=True, sort=False)
    else:
        frame = new_frame
    existing_keys = [col for col in key_columns if col in frame.columns]
    if existing_keys:
        frame = frame.drop_duplicates(subset=existing_keys, keep="last")
    sort_keys = [col for col in (sort_columns or key_columns) if col in frame.columns]
    if sort_keys:
        frame = frame.sort_values(sort_keys).reset_index(drop=True)
    atomic_write_csv(frame, path, index=False)
    return frame
