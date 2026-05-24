from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def df_to_md_table(df: pd.DataFrame | None) -> str:
    """Render DataFrame as GitHub-flavored markdown pipe table via `scripts/_md_report_utils.py`."""
    if df is None or df.empty:
        return "_No rows._\n"
    util = Path(__file__).resolve().parent.parent / "_md_report_utils.py"
    if not util.is_file():
        return "_Table renderer missing._\n"
    spec = importlib.util.spec_from_file_location("_md_report_utils", util)
    if spec is None or spec.loader is None:
        return "_Table renderer failed._\n"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.md_table(df)

COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def ensure_stage_dirs(cfg) -> None:
    for path in [
        cfg.assets_dir,
        cfg.reports_dir,
        cfg.metadata_dir,
        cfg.catalogs_dir,
        cfg.labels_dir,
        cfg.cohort_dir,
        cfg.exposures_dir,
        cfg.modeling_dir,
        cfg.figures_dir,
        cfg.tables_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def clean_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        try:
            out[col] = out[col].astype(str).str.strip('"')
        except Exception:
            pass
    return out


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def choose_binary_n_splits(y: np.ndarray, desired_splits: int) -> int:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return 0
    return max(2, min(desired_splits, int(counts.min())))


def choose_regression_n_splits(n_samples: int, desired_splits: int) -> int:
    if n_samples < 2:
        return 0
    return max(2, min(desired_splits, n_samples))


def canonical_96_channels() -> list[str]:
    bases = "ACGT"
    muts = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
    channels = []
    for mut in muts:
        for left in bases:
            for right in bases:
                channels.append(f"{left}[{mut}]{right}")
    return channels


def sbs96_channel(ref: str, alt: str, context_11: str) -> str | None:
    if len(context_11) != 11:
        return None
    left = context_11[4].upper()
    right = context_11[6].upper()
    ref = ref.upper()
    alt = alt.upper()
    if ref not in "ACGT" or alt not in "ACGT" or left not in "ACGT" or right not in "ACGT":
        return None
    if ref in ("C", "T"):
        return f"{left}[{ref}>{alt}]{right}"
    ref_c = COMP[ref]
    alt_c = COMP[alt]
    left_c = COMP[right]
    right_c = COMP[left]
    return f"{left_c}[{ref_c}>{alt_c}]{right_c}"


def canonical_78_channels() -> list[str]:
    # Simplified COSMIC DBS78 channels
    # In practice, usually loaded from COSMIC file, but we can generate labels.
    # Ref: AC, AG, AT, CA, CG, CT, GA, GT, TA, TG (10 base pairs)
    # Each has various Alts.
    # For now, let's just use the COSMIC column names from the file later.
    return []


def dbs78_channel(ref: str, alt: str) -> str | None:
    # Basic DNP mapper
    ref, alt = ref.upper(), alt.upper()
    if len(ref) != 2 or len(alt) != 2:
        return None
    # This is a placeholder; real DBS78 requires canonicalization
    # Since we use Universal CGR, we might not even need the categorical label 
    # unless we run the 'Standard' model.
    return f"{ref}>{alt}"


def make_subtype_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Assign ER+ / TNBC / Other for QC tables (no subtype-specific HRD binary label)."""
    out = df.copy()
    out["subtype"] = "Other"
    if "ER_status" in out.columns:
        out.loc[out["ER_status"] == "Positive", "subtype"] = "ER+"
    tnbc_mask = pd.Series(False, index=out.index)
    if {"ER_status", "PR_status", "HER2_IHC_status"}.issubset(out.columns):
        tnbc_mask = (
            (out["ER_status"] == "Negative")
            & (out["PR_status"] == "Negative")
            & (out["HER2_IHC_status"] == "Negative")
        )
        out.loc[tnbc_mask, "subtype"] = "TNBC"
    return out


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None
