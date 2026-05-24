"""Projection helpers for channel-count and signature-matrix benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .channels import variant_vec, variant_vec_bicgr52_context_only
from .encoding import PAYLOAD_SCHEMA_MASKED, universal_vector_dim


def universal_fdim(d_context: int, d_payload: int, payload_schema: str = PAYLOAD_SCHEMA_MASKED) -> int:
    return int(universal_vector_dim(d_context, d_payload, payload_schema))


def stack_channel_matrix(chan_vecs: list[np.ndarray | None], fdim: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(chan_vecs)
    basis = np.zeros((n, int(fdim)), dtype=np.float64)
    valid = np.zeros(n, dtype=np.uint8)
    for i, vec in enumerate(chan_vecs):
        if vec is not None:
            valid[i] = 1
            basis[i, :] = np.asarray(vec, dtype=np.float64)
    return basis, valid


def weighted_patient_profiles(basis: np.ndarray, valid: np.ndarray, counts: np.ndarray) -> np.ndarray:
    weights = np.asarray(counts, dtype=np.float64) * np.asarray(valid, dtype=np.float64)[:, np.newaxis]
    denom = weights.sum(axis=0)
    numerator = np.asarray(basis, dtype=np.float64).T @ weights
    out = np.zeros((counts.shape[1], basis.shape[1]), dtype=np.float64)
    mask = denom > 1e-15
    out[mask] = (numerator[:, mask] / denom[mask]).T
    return out


def normalize_rows_l1(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    sums = x.sum(axis=1, keepdims=True)
    return np.divide(x, sums, out=np.zeros_like(x), where=sums > 1e-15)


def sig_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if str(c).startswith(prefix) and pd.to_numeric(df[c], errors="coerce").fillna(0).sum() > 0]


def build_standard_ref(df: pd.DataFrame, sigs: list[str]) -> np.ndarray:
    return np.column_stack([df[s].fillna(0.0).to_numpy(dtype=np.float64) for s in sigs])


def build_universal_ref(
    df: pd.DataFrame,
    atlas: dict[str, np.ndarray],
    d_context: int,
    sigs: list[str],
    modality: str,
    d_payload: int | None = None,
    payload_schema: str = PAYLOAD_SCHEMA_MASKED,
) -> np.ndarray:
    dp = int(d_payload if d_payload is not None else d_context)
    fdim = universal_fdim(d_context, dp, payload_schema)
    indexed = df.set_index("Type")
    cols = []
    for sig in sigs:
        total = np.zeros(fdim, dtype=np.float64)
        weight_sum = 0.0
        for channel, prob in indexed[sig].items():
            if pd.isna(prob) or prob <= 0:
                continue
            vec = variant_vec(str(channel), atlas, d_context, modality, dp, payload_schema)
            if vec is not None:
                total += np.asarray(vec, dtype=np.float64) * float(prob)
                weight_sum += float(prob)
        cols.append(total / weight_sum if weight_sum else total)
    return np.column_stack(cols)


def build_bicgr52_ref(
    df: pd.DataFrame,
    atlas: dict[str, np.ndarray],
    d_context: int,
    sigs: list[str],
    modality: str,
    payload_schema: str = PAYLOAD_SCHEMA_MASKED,
) -> np.ndarray:
    indexed = df.set_index("Type")
    cols = []
    for sig in sigs:
        total = np.zeros(4 * int(d_context), dtype=np.float64)
        weight_sum = 0.0
        for channel, prob in indexed[sig].items():
            if pd.isna(prob) or prob <= 0:
                continue
            vec = variant_vec_bicgr52_context_only(str(channel), atlas, d_context, modality, 2, payload_schema)
            if vec is not None:
                total += np.asarray(vec, dtype=np.float64) * float(prob)
                weight_sum += float(prob)
        cols.append(total / weight_sum if weight_sum else total)
    return np.column_stack(cols)


def project_counts_to_uga(counts: pd.DataFrame, basis: np.ndarray, valid: np.ndarray, prefix: str) -> pd.DataFrame:
    arr = counts.to_numpy(dtype=np.float64)
    weighted = arr * np.asarray(valid, dtype=np.float64)[np.newaxis, :]
    denom = weighted.sum(axis=1, keepdims=True)
    projected = weighted @ basis
    projected = np.divide(projected, denom, out=np.zeros_like(projected), where=denom > 1e-15)
    return pd.DataFrame(projected, index=counts.index, columns=[f"{prefix}_{i + 1:03d}" for i in range(projected.shape[1])])


def signature_projection(channel_basis: np.ndarray, signature_probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(signature_probabilities, dtype=np.float64)
    projected = np.asarray(channel_basis, dtype=np.float64).T @ probs
    col_sums = probs.sum(axis=0)
    return np.divide(projected, col_sums[None, :], out=np.zeros_like(projected), where=col_sums[None, :] > 1e-15)
