from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

from exp23_config import WorkflowConfig



def bh_fdr(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    out = np.empty(len(p))
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        q = min(prev, p[idx] * len(p) / (len(p) - rank + 1))
        out[idx] = q
        prev = q
    return out.tolist()


def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, task: str, metric: str) -> float:
    if task == "classification":
        if np.unique(y_true).size < 2:
            return np.nan
        if metric == "auroc":
            return float(roc_auc_score(y_true, y_pred))
        if metric == "auprc":
            return float(average_precision_score(y_true, y_pred))
        raise ValueError(f"Unsupported classification metric: {metric}")
    if metric == "spearman":
        return float(spearmanr(y_true, y_pred)[0])
    raise ValueError(f"Unsupported regression metric: {metric}")


def bootstrap_delta(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    task: str,
    metric: str,
    n_boot: int,
    random_state: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    deltas = []
    indices = np.arange(n)
    for _ in range(n_boot):
        sample = rng.choice(indices, size=n, replace=True)
        ys = y_true[sample]
        pa = pred_a[sample]
        pb = pred_b[sample]
        try:
            a = compute_metric(ys, pa, task, metric)
            b = compute_metric(ys, pb, task, metric)
        except Exception:
            continue
        if pd.notna(a) and pd.notna(b):
            deltas.append(b - a)
    if not deltas:
        return np.nan, np.nan
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def bh_fdr(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    out = np.empty(len(p))
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        q = min(prev, p[idx] * len(p) / (len(p) - rank + 1))
        out[idx] = q
        prev = q
    return out.tolist()


def paired_wilcoxon_tests(fold_metrics: pd.DataFrame, comparisons: list[tuple[str, str]]) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()
    rows = []
    metric_map = {
        "regression": ["spearman", "pearson", "r2", "mae"],
        "classification": ["auroc", "auprc", "balanced_acc", "brier"],
    }
    group_cols = ["suite", "subset", "endpoint", "task"]
    for group_key, group_df in fold_metrics.groupby(group_cols):
        suite, subset, endpoint, task = group_key
        metrics = metric_map[task]
        for model_a, model_b in comparisons:
            a_df = group_df[group_df["model"] == model_a].sort_values("fold")
            b_df = group_df[group_df["model"] == model_b].sort_values("fold")
            if len(a_df) != len(b_df) or a_df.empty:
                continue
            for metric in metrics:
                a_vals = a_df[metric].to_numpy(dtype=float)
                b_vals = b_df[metric].to_numpy(dtype=float)
                mask = ~(np.isnan(a_vals) | np.isnan(b_vals))
                if mask.sum() < 2:
                    continue
                stat, p_value = wilcoxon(a_vals[mask], b_vals[mask], alternative="two-sided")
                rows.append(
                    {
                        "suite": suite,
                        "subset": subset,
                        "endpoint": endpoint,
                        "task": task,
                        "test": "wilcoxon",
                        "metric": metric,
                        "model_a": model_a,
                        "model_b": model_b,
                        "estimate_a": float(np.mean(a_vals[mask])),
                        "estimate_b": float(np.mean(b_vals[mask])),
                        "delta": float(np.mean(b_vals[mask] - a_vals[mask])),
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["q_value"] = bh_fdr(df["p_value"].tolist())
    return df


def get_structural_components(y_true: np.ndarray, y_scores: np.ndarray):
    y_true = np.asarray(y_true, dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)
    idx_neg = np.where(y_true == 0)[0]
    idx_pos = np.where(y_true == 1)[0]
    n_neg, n_pos = len(idx_neg), len(idx_pos)
    v_neg = np.array([
        np.sum(y_scores[idx_pos] > y_scores[i]) + 0.5 * np.sum(y_scores[idx_pos] == y_scores[i])
        for i in idx_neg
    ], dtype=float) / max(n_pos, 1)
    v_pos = np.array([
        np.sum(y_scores[idx_neg] < y_scores[i]) + 0.5 * np.sum(y_scores[idx_neg] == y_scores[i])
        for i in idx_pos
    ], dtype=float) / max(n_neg, 1)
    return v_neg, v_pos


def delong_roc_test(y_true: np.ndarray, y_scores1: np.ndarray, y_scores2: np.ndarray):
    v_a1, v_n1 = get_structural_components(y_true, y_scores1)
    v_a2, v_n2 = get_structural_components(y_true, y_scores2)
    n_a, n_n = len(v_a1), len(v_n1)
    auc1, auc2 = float(np.mean(v_n1)), float(np.mean(v_n2))
    diff = auc2 - auc1
    s_a12 = np.cov(v_a1, v_a2)[0, 1]
    s_n12 = np.cov(v_n1, v_n2)[0, 1]
    var_diff = (
        (np.var(v_a1, ddof=1) + np.var(v_a2, ddof=1) - 2 * s_a12) / max(n_a, 1)
        + (np.var(v_n1, ddof=1) + np.var(v_n2, ddof=1) - 2 * s_n12) / max(n_n, 1)
    )
    z_score = diff / np.sqrt(max(var_diff, 1e-12))
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    return auc1, auc2, diff, float(z_score), float(p_value)


def steiger_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray):
    n = len(y_true)
    r12, _ = spearmanr(y_true, y_pred1)
    r13, _ = spearmanr(y_true, y_pred2)
    r23, _ = spearmanr(y_pred1, y_pred2)
    z12 = 0.5 * np.log((1 + r12) / (1 - r12))
    z13 = 0.5 * np.log((1 + r13) / (1 - r13))
    rho = r23 * (1 - r12**2 - r13**2) + 0.5 * r12 * r13 * (r12**2 + r13**2 + r23**2 - 1)
    rho /= (1 - r12**2) * (1 - r13**2)
    diff_z = z13 - z12
    var_z = max((2 - 2 * rho) / max(n - 3, 1), 1e-12)
    z_score = diff_z / np.sqrt(var_z)
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    return float(r12), float(r13), float(z_score), float(p_value)


def prediction_tests(predictions: pd.DataFrame, cfg: WorkflowConfig, comparisons: list[tuple[str, str]]) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["suite", "subset", "endpoint", "task"]
    for group_key, group_df in predictions.groupby(group_cols):
        suite, subset, endpoint, task = group_key
        for model_a, model_b in comparisons:
            left = group_df[group_df["model"] == model_a][["patient_id", "true_value", "pred_value"]].rename(columns={"pred_value": "pred_a"})
            right = group_df[group_df["model"] == model_b][["patient_id", "pred_value"]].rename(columns={"pred_value": "pred_b"})
            merged = left.merge(right, on="patient_id", how="inner")
            if merged.empty:
                continue
            y_true = merged["true_value"].to_numpy(dtype=float)
            pred_a = merged["pred_a"].to_numpy(dtype=float)
            pred_b = merged["pred_b"].to_numpy(dtype=float)

            if task == "classification":
                if np.unique(y_true).size < 2:
                    continue
                auc_a, auc_b, diff, z_score, p_value = delong_roc_test(y_true, pred_a, pred_b)
                ci_low, ci_high = bootstrap_delta(
                    y_true,
                    pred_a,
                    pred_b,
                    task=task,
                    metric="auroc",
                    n_boot=cfg.bootstrap_iterations,
                    random_state=cfg.random_state,
                )
                rows.append(
                    {
                        "suite": suite,
                        "subset": subset,
                        "endpoint": endpoint,
                        "task": task,
                        "test": "delong",
                        "metric": "auroc",
                        "model_a": model_a,
                        "model_b": model_b,
                        "estimate_a": float(auc_a),
                        "estimate_b": float(auc_b),
                        "delta": float(diff),
                        "statistic": float(z_score),
                        "p_value": float(p_value),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
            else:
                rho_a, rho_b, z_score, p_value = steiger_test(y_true, pred_a, pred_b)
                ci_low, ci_high = bootstrap_delta(
                    y_true,
                    pred_a,
                    pred_b,
                    task=task,
                    metric="spearman",
                    n_boot=cfg.bootstrap_iterations,
                    random_state=cfg.random_state,
                )
                rows.append(
                    {
                        "suite": suite,
                        "subset": subset,
                        "endpoint": endpoint,
                        "task": task,
                        "test": "steiger",
                        "metric": "spearman",
                        "model_a": model_a,
                        "model_b": model_b,
                        "estimate_a": float(rho_a),
                        "estimate_b": float(rho_b),
                        "delta": float(rho_b - rho_a),
                        "statistic": float(z_score),
                        "p_value": float(p_value),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["q_value"] = bh_fdr(df["p_value"].tolist())
    return df
