#!/usr/bin/env python3
"""Shared reference-signature exposure helpers for the locked UGA manuscript model."""

from __future__ import annotations

import math
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls


LOCKED_SBSDBS_MODEL = "master_spec_sbs_dbs_d10_dp5"
LOCKED_ID_MODEL = "id83_payload_only_d10_dp5"
RANDOM_SEED = 20260514


def find_cgr_root() -> Path:
    path = Path(__file__).resolve()
    for candidate in [path.parent, *path.parents]:
        if (candidate / "uga_atlas" / "models.py").is_file() and (candidate / "data" / "Signatures").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate cgr_validation root from {path}")


CGR_ROOT = find_cgr_root()
if str(CGR_ROOT) not in sys.path:
    sys.path.insert(0, str(CGR_ROOT))

from uga_atlas import build_uga_basis, get_uga_model, load_context_atlas, signature_projection  # noqa: E402


RESEARCH_ROOT = CGR_ROOT / "cgr_validation_results" / "research"
SIGNATURE_DIR = CGR_ROOT / "data" / "Signatures"
CONTEXT_ATLAS = RESEARCH_ROOT / "data" / "EXP022_atlas_genome_wide_45mer_universal_d22.json"


def modality_prefix(modality: str) -> str:
    value = modality.upper()
    if value not in {"SBS", "DBS", "ID"}:
        raise ValueError(f"Unsupported modality: {modality}")
    return value


def signature_path(modality: str) -> Path:
    prefix = modality_prefix(modality)
    return SIGNATURE_DIR / f"COSMIC_v3.5_{prefix}_GRCh37.txt"


def load_cosmic_reference(modality: str) -> pd.DataFrame:
    path = signature_path(modality)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t")
    if "Type" not in df.columns:
        raise ValueError(f"{path} does not contain a Type column")
    return df


def signature_columns(df: pd.DataFrame, modality: str) -> list[str]:
    prefix = modality_prefix(modality)
    columns = []
    for col in df.columns:
        if not str(col).startswith(prefix):
            continue
        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        if float(values.sum()) > 0.0:
            columns.append(str(col))
    return columns


def strip_feature_prefix(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [col for col in frame.columns if str(col).startswith(prefix)]
    out = frame[cols].copy()
    out.columns = [str(col).replace(prefix, "", 1) for col in cols]
    return out


def normalize_rows(frame: pd.DataFrame) -> pd.DataFrame:
    arr = frame.to_numpy(dtype=np.float64)
    sums = arr.sum(axis=1, keepdims=True)
    normed = np.divide(arr, sums, out=np.zeros_like(arr), where=sums > 1e-15)
    return pd.DataFrame(normed, index=frame.index, columns=frame.columns)


def prepare_reference(
    channels: list[str],
    modality: str,
    representation: str,
) -> tuple[np.ndarray, np.ndarray | None, list[str], list[str], dict[str, object]]:
    cosmic = load_cosmic_reference(modality).set_index("Type")
    sigs = signature_columns(cosmic.reset_index(), modality)
    common = [str(channel) for channel in channels if str(channel) in cosmic.index]
    if not common:
        raise ValueError(f"No channels from the input matrix matched COSMIC {modality_prefix(modality)}")
    ref = cosmic.loc[common, sigs].fillna(0.0).astype(float)
    ref_arr = ref.to_numpy(dtype=np.float64)
    col_sums = ref_arr.sum(axis=0, keepdims=True)
    ref_arr = np.divide(ref_arr, col_sums, out=np.zeros_like(ref_arr), where=col_sums > 1e-15)

    if representation == "standard":
        metadata = {
            "n_input_channels": len(channels),
            "n_matched_channels": len(common),
            "n_encoded_channels": len(common),
            "feature_dimension": len(common),
        }
        return ref_arr, None, common, sigs, metadata

    if representation != "locked_uga":
        raise ValueError(f"Unsupported representation: {representation}")

    if modality_prefix(modality) == "ID":
        basis, diag = build_uga_basis(common, LOCKED_ID_MODEL)
        model_name = LOCKED_ID_MODEL
    else:
        model = get_uga_model(LOCKED_SBSDBS_MODEL)
        atlas = load_context_atlas(CONTEXT_ATLAS, model.d_context)
        basis, diag = build_uga_basis(common, LOCKED_SBSDBS_MODEL, atlas=atlas, modality=modality_prefix(modality))
        model_name = LOCKED_SBSDBS_MODEL
    valid = diag["UGA_Encoded"].to_numpy(dtype=bool)
    projected_ref = signature_projection(basis, ref_arr)
    metadata = {
        "n_input_channels": len(channels),
        "n_matched_channels": len(common),
        "n_encoded_channels": int(valid.sum()),
        "feature_dimension": int(projected_ref.shape[0]),
        "uga_model": model_name,
    }
    return projected_ref, basis * valid[:, None], common, sigs, metadata


def patient_profiles(
    counts: pd.DataFrame,
    channels: list[str],
    basis: np.ndarray | None,
) -> pd.DataFrame:
    aligned = counts.loc[:, channels].fillna(0.0).astype(float)
    arr = aligned.to_numpy(dtype=np.float64)
    if basis is None:
        sums = arr.sum(axis=1, keepdims=True)
        profiles = np.divide(arr, sums, out=np.zeros_like(arr), where=sums > 1e-15)
        cols = [f"channel_{i + 1:03d}" for i in range(profiles.shape[1])]
        return pd.DataFrame(profiles, index=aligned.index, columns=cols)
    projected = arr @ basis
    valid = np.abs(basis).sum(axis=1) > 1e-15
    denom = (arr * valid[None, :]).sum(axis=1, keepdims=True)
    profiles = np.divide(projected, denom, out=np.zeros_like(projected), where=denom > 1e-15)
    cols = [f"uga_{i + 1:03d}" for i in range(profiles.shape[1])]
    return pd.DataFrame(profiles, index=aligned.index, columns=cols)


def fit_nnls_exposures(
    profiles: pd.DataFrame,
    reference: np.ndarray,
    sigs: list[str],
) -> pd.DataFrame:
    a = np.asarray(reference, dtype=np.float64)
    exposure = np.zeros((profiles.shape[0], len(sigs)), dtype=np.float64)
    for i, (_, row) in enumerate(profiles.iterrows()):
        b = row.to_numpy(dtype=np.float64)
        if not np.isfinite(b).all() or float(b.sum()) <= 1e-15:
            continue
        weights, _ = nnls(a, b)
        total = float(weights.sum())
        if total > 1e-15:
            exposure[i, :] = weights / total
    return pd.DataFrame(exposure, index=profiles.index, columns=sigs)


def extract_signature_exposures(
    counts: pd.DataFrame,
    modality: str,
    representation: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    reference, basis, channels, sigs, metadata = prepare_reference(counts.columns.astype(str).tolist(), modality, representation)
    aligned_counts = counts.copy()
    aligned_counts.columns = aligned_counts.columns.astype(str)
    profiles = patient_profiles(aligned_counts, channels, basis)
    exposures = fit_nnls_exposures(profiles, reference, sigs)
    metadata = {
        **metadata,
        "representation": representation,
        "modality": modality_prefix(modality),
        "n_patients": int(exposures.shape[0]),
        "n_signatures": int(exposures.shape[1]),
        "extraction_algorithm": "NNLS with L1-normalized nonnegative exposures",
    }
    return exposures, profiles, metadata


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    denom = math.sqrt(float(np.dot(aa, aa))) * math.sqrt(float(np.dot(bb, bb)))
    if denom <= 1e-15:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def bh_q_values(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    q = np.full(len(p), np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return q
    order = valid[np.argsort(p[valid])]
    ranked = p[order]
    adjusted = np.empty_like(ranked)
    running = 1.0
    m = len(ranked)
    for i in range(m - 1, -1, -1):
        running = min(running, ranked[i] * m / float(i + 1))
        adjusted[i] = running
    q[order] = np.minimum(adjusted, 1.0)
    return q


def format_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def write_html_table(df: pd.DataFrame, path: Path, title: str, footnote: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "".join(f"<th>{escape(str(col))}</th>" for col in df.columns)
    rows = []
    for _, row in df.iterrows():
        rows.append("".join(f"<td>{escape(format_value(row[col]))}</td>" for col in df.columns))
    html_rows = "".join(f"<tr>{row}</tr>" for row in rows)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<style>
body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
caption {{ caption-side: top; text-align: left; font-weight: 700; margin-bottom: 8px; }}
thead th {{ border-top: 1px solid #111; border-bottom: 1px solid #111; padding: 6px 8px; text-align: left; }}
tbody td {{ border-bottom: 0.5px solid #bbb; padding: 5px 8px; text-align: left; }}
tfoot td {{ border-top: 1px solid #111; padding: 6px 8px; font-size: 11px; line-height: 1.35; }}
</style>
</head>
<body>
<table>
<caption>{escape(title)}</caption>
<thead><tr>{header}</tr></thead>
<tbody>{html_rows}</tbody>
<tfoot><tr><td colspan="{len(df.columns)}">{escape(footnote)}</td></tr></tfoot>
</table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
