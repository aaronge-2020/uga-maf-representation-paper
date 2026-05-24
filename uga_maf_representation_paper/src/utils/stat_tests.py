"""Paired statistical tests for manuscript out-of-fold predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class PairedTestResult:
    delta: float
    p_value: float
    ci_low: float
    ci_high: float
    test_name: str
    n_resamples: int


def bh_qvalues(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values, preserving NaNs."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return q
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = float(len(ranked))
    adjusted = ranked * m / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.minimum(adjusted, 1.0)
    return q


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    midranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(len(x), dtype=float)
    out[order] = midranks
    return out


def _fast_delong(predictions_sorted: np.ndarray, n_positive: int) -> tuple[np.ndarray, np.ndarray]:
    n_classifiers = predictions_sorted.shape[0]
    n_negative = predictions_sorted.shape[1] - n_positive
    positive = predictions_sorted[:, :n_positive]
    negative = predictions_sorted[:, n_positive:]
    tx = np.empty((n_classifiers, n_positive), dtype=float)
    ty = np.empty((n_classifiers, n_negative), dtype=float)
    tz = np.empty((n_classifiers, n_positive + n_negative), dtype=float)
    for classifier_idx in range(n_classifiers):
        tx[classifier_idx, :] = _compute_midrank(positive[classifier_idx, :])
        ty[classifier_idx, :] = _compute_midrank(negative[classifier_idx, :])
        tz[classifier_idx, :] = _compute_midrank(predictions_sorted[classifier_idx, :])
    aucs = tz[:, :n_positive].sum(axis=1) / n_positive / n_negative - (n_positive + 1.0) / (2.0 * n_negative)
    v01 = (tz[:, :n_positive] - tx) / n_negative
    v10 = 1.0 - (tz[:, n_positive:] - ty) / n_positive
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / n_positive + sy / n_negative


def paired_delong_auc(y_true: np.ndarray, candidate_score: np.ndarray, baseline_score: np.ndarray) -> PairedTestResult:
    """Paired DeLong test for two correlated binary AUROCs."""
    y = np.asarray(y_true)
    if len(np.unique(y)) != 2:
        raise ValueError("paired_delong_auc requires exactly two classes")
    classes = np.sort(np.unique(y))
    positive = classes[-1]
    y_binary = (y == positive).astype(int)
    order = np.argsort(-y_binary)
    predictions = np.vstack([candidate_score, baseline_score])[:, order].astype(float)
    aucs, covariance = _fast_delong(predictions, int(y_binary.sum()))
    delta = float(aucs[0] - aucs[1])
    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ covariance @ contrast.T).item())
    if not math.isfinite(variance) or variance <= 0.0:
        p_value = 1.0
    else:
        z = abs(delta) / math.sqrt(variance)
        p_value = math.erfc(z / math.sqrt(2.0))
    return PairedTestResult(delta=delta, p_value=float(min(max(p_value, 0.0), 1.0)), ci_low=np.nan, ci_high=np.nan, test_name="paired_delong_auroc", n_resamples=0)


def _score_metric(y: np.ndarray, pred: np.ndarray, metric: str) -> float:
    metric = str(metric).lower()
    if metric in {"spearman", "spearman_r"}:
        return float(pd.Series(pred).corr(pd.Series(y), method="spearman"))
    if metric in {"macro_auroc", "macro_auc"}:
        return float(roc_auc_score(y, pred, average="macro", multi_class="ovr", labels=np.unique(y)))
    if metric in {"auroc", "auc"}:
        return float(roc_auc_score(y, pred))
    raise ValueError(f"Unsupported paired bootstrap metric: {metric}")


def paired_bootstrap_delta(
    y_true: np.ndarray,
    candidate_pred: np.ndarray,
    baseline_pred: np.ndarray,
    metric: str,
    *,
    n_bootstrap: int,
    seed: int,
    stratify: bool,
) -> PairedTestResult:
    """Paired bootstrap test for a metric difference from OOF predictions."""
    y = np.asarray(y_true)
    cand = np.asarray(candidate_pred)
    base = np.asarray(baseline_pred)
    rng = np.random.default_rng(seed)
    observed = _score_metric(y, cand, metric) - _score_metric(y, base, metric)
    deltas = np.full(int(n_bootstrap), np.nan, dtype=float)
    if stratify:
        groups = [np.where(y == cls)[0] for cls in np.unique(y)]
    for i in range(int(n_bootstrap)):
        if stratify:
            idx = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups if len(group)])
        else:
            idx = rng.choice(np.arange(len(y)), size=len(y), replace=True)
        try:
            deltas[i] = _score_metric(y[idx], cand[idx], metric) - _score_metric(y[idx], base[idx], metric)
        except Exception:
            deltas[i] = np.nan
    valid = deltas[np.isfinite(deltas)]
    if len(valid) == 0:
        return PairedTestResult(delta=float(observed), p_value=np.nan, ci_low=np.nan, ci_high=np.nan, test_name=f"paired_bootstrap_{metric}", n_resamples=int(n_bootstrap))
    p_lower = (float(np.sum(valid <= 0.0)) + 1.0) / (float(len(valid)) + 1.0)
    p_upper = (float(np.sum(valid >= 0.0)) + 1.0) / (float(len(valid)) + 1.0)
    p_value = min(1.0, 2.0 * min(p_lower, p_upper))
    return PairedTestResult(
        delta=float(observed),
        p_value=float(p_value),
        ci_low=float(np.quantile(valid, 0.025)),
        ci_high=float(np.quantile(valid, 0.975)),
        test_name=f"paired_bootstrap_{metric}",
        n_resamples=int(n_bootstrap),
    )
