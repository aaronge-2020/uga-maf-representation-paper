"""Nested out-of-fold evaluation with pooled global metrics.

The protocol is intentionally simple and auditable: five outer folds, one
inner validation split inside each outer-training fold, deterministic
candidate ordering, final refit on the full outer-training fold, and one
pooled metric computed from all OOF predictions.
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, ShuffleSplit, StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    from joblib import Parallel, delayed, parallel_config
except ImportError:  # pragma: no cover - joblib is a scikit-learn dependency in normal environments.
    Parallel = None
    delayed = None
    parallel_config = None

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - threadpoolctl is a scikit-learn dependency in normal environments.
    threadpool_limits = None


LINEAR_L1_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
LOGISTIC_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
ELASTIC_ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
COX_PENALIZER_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


@dataclass
class NestedOOFResult:
    summary: dict[str, Any]
    predictions: pd.DataFrame
    fold_diagnostics: pd.DataFrame


def stable_seed(*parts: object) -> int:
    import hashlib

    digest = hashlib.sha256("||".join(str(p) for p in parts).encode("utf-8")).digest()
    return 1729 + int.from_bytes(digest[:4], "big") % 1_000_000


def _json_safe(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _safe_json(payload: object) -> str:
    return json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))


def encode_labels(labels: pd.Series | pd.DataFrame, task: str) -> tuple[Any, np.ndarray]:
    if task == "survival":
        frame = pd.DataFrame(labels).copy()
        required = {"time", "event"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Survival labels require columns {sorted(required)}; missing {sorted(missing)}")
        frame = frame.loc[:, ["time", "event"]].copy()
        frame["time"] = pd.to_numeric(frame["time"], errors="coerce").astype(float)
        frame["event"] = pd.to_numeric(frame["event"], errors="coerce").fillna(0).astype(int)
        return frame, np.array([], dtype=object)
    series = pd.Series(labels)
    if task == "regression":
        return series.astype(float).to_numpy(dtype=np.float64), np.array([], dtype=object)
    if task == "binary":
        return series.astype(int).to_numpy(dtype=np.int32), np.array([0, 1], dtype=object)
    classes = np.array(sorted(series.astype(str).unique()), dtype=object)
    mapping = {value: i for i, value in enumerate(classes)}
    return series.astype(str).map(mapping).astype(int).to_numpy(dtype=np.int32), classes


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


def safe_micro_auroc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    proba = np.asarray(proba, dtype=np.float64)
    indicator = np.zeros((len(y_true), int(n_classes)), dtype=np.int32)
    indicator[np.arange(len(y_true)), y_true] = 1
    try:
        return float(roc_auc_score(indicator, proba, average="micro"))
    except ValueError:
        return float("nan")


def topk_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    proba = np.asarray(proba, dtype=np.float64)
    if proba.size == 0 or len(y_true) == 0:
        return float("nan")
    top = np.argsort(-proba, axis=1)[:, : min(int(k), proba.shape[1])]
    return float(np.mean([int(y_true[i]) in top[i] for i in range(len(y_true))]))


def harrell_c_index(time: Iterable[float], event: Iterable[int], risk: Iterable[float]) -> float:
    """Harrell C-index where higher risk means shorter survival."""
    try:
        from lifelines.utils import concordance_index

        return float(concordance_index(np.asarray(time, dtype=float), -np.asarray(risk, dtype=float), np.asarray(event, dtype=int)))
    except Exception:
        pass

    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    r = np.asarray(risk, dtype=float)
    concordant = 0.0
    permissible = 0.0
    n = len(t)
    for i in range(n):
        if e[i] != 1 or not np.isfinite(t[i]) or not np.isfinite(r[i]):
            continue
        later = np.where(t[i] < t)[0]
        for j in later:
            if not np.isfinite(r[j]):
                continue
            permissible += 1.0
            if r[i] > r[j]:
                concordant += 1.0
            elif r[i] == r[j]:
                concordant += 0.5
    return float(concordant / permissible) if permissible > 0 else float("nan")


def _split_y(task: str, y: Any) -> np.ndarray:
    if task == "survival":
        return pd.DataFrame(y)["event"].astype(int).to_numpy()
    if task == "regression":
        return np.zeros(len(y), dtype=int)
    return np.asarray(y)


def outer_splits(task: str, y: Any, *, groups: np.ndarray | None, seed: int, n_splits: int = 5) -> list[tuple[int, np.ndarray, np.ndarray]]:
    n = len(y)
    n_splits = max(2, min(int(n_splits), n))
    split_y = _split_y(task, y)
    zeros = np.zeros(n)
    if groups is not None:
        group_arr = np.asarray(groups)
        unique_groups = np.unique(group_arr)
        if len(unique_groups) < 2:
            raise ValueError("Grouped outer OOF evaluation requires at least two unique groups")
        group_splits = max(2, min(n_splits, len(unique_groups)))
        try:
            splitter = StratifiedGroupKFold(n_splits=group_splits, shuffle=True, random_state=seed)
            return [(fold, tr, te) for fold, (tr, te) in enumerate(splitter.split(zeros, split_y, group_arr), start=1)]
        except ValueError:
            splitter = GroupKFold(n_splits=group_splits)
            return [(fold, tr, te) for fold, (tr, te) in enumerate(splitter.split(zeros, split_y, group_arr), start=1)]
    if task in {"binary", "multiclass", "multiclass_grouped", "survival"}:
        counts = pd.Series(split_y).value_counts()
        if len(counts) > 1 and int(counts.min()) >= n_splits:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return [(fold, tr, te) for fold, (tr, te) in enumerate(splitter.split(zeros, split_y), start=1)]
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(fold, tr, te) for fold, (tr, te) in enumerate(splitter.split(zeros), start=1)]


def inner_train_val_split(task: str, y: Any, outer_train_idx: np.ndarray, *, groups: np.ndarray | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray(outer_train_idx, dtype=int)
    split_y = _split_y(task, y)[train_idx]
    zeros = np.zeros(len(train_idx))
    if groups is not None:
        sub_groups = np.asarray(groups)[train_idx]
        if len(np.unique(sub_groups)) >= 4:
            splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
            inner_tr, inner_val = next(splitter.split(zeros, split_y, sub_groups))
            return train_idx[inner_tr], train_idx[inner_val]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        inner_tr, inner_val = next(splitter.split(zeros, split_y, sub_groups))
        return train_idx[inner_tr], train_idx[inner_val]
    if task in {"binary", "multiclass", "multiclass_grouped", "survival"}:
        counts = pd.Series(split_y).value_counts()
        if len(counts) > 1 and int(counts.min()) >= 2:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
            inner_tr, inner_val = next(splitter.split(zeros, split_y))
            return train_idx[inner_tr], train_idx[inner_val]
    splitter = ShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    inner_tr, inner_val = next(splitter.split(zeros))
    return train_idx[inner_tr], train_idx[inner_val]


def assert_split_integrity(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    inner_train_idx: np.ndarray | None = None,
    inner_val_idx: np.ndarray | None = None,
    *,
    groups: np.ndarray | None = None,
) -> dict[str, int]:
    """Validate that outer and inner validation splits are isolated."""

    train_set = set(map(int, np.asarray(train_idx, dtype=int)))
    test_set = set(map(int, np.asarray(test_idx, dtype=int)))
    outer_overlap = len(train_set.intersection(test_set))
    if outer_overlap:
        raise ValueError(f"Outer train/test split overlap contains {outer_overlap} rows")
    diagnostics = {"outer_overlap_n": int(outer_overlap), "inner_overlap_n": 0, "inner_test_overlap_n": 0, "group_overlap_n": 0, "inner_group_overlap_n": 0}
    if inner_train_idx is not None and inner_val_idx is not None:
        inner_train_set = set(map(int, np.asarray(inner_train_idx, dtype=int)))
        inner_val_set = set(map(int, np.asarray(inner_val_idx, dtype=int)))
        inner_overlap = len(inner_train_set.intersection(inner_val_set))
        inner_test_overlap = len((inner_train_set | inner_val_set).intersection(test_set))
        outside_outer_train = len((inner_train_set | inner_val_set).difference(train_set))
        if inner_overlap:
            raise ValueError(f"Inner train/validation split overlap contains {inner_overlap} rows")
        if inner_test_overlap:
            raise ValueError(f"Inner split overlaps outer test fold in {inner_test_overlap} rows")
        if outside_outer_train:
            raise ValueError(f"Inner split contains {outside_outer_train} rows outside the outer-training fold")
        diagnostics["inner_overlap_n"] = int(inner_overlap)
        diagnostics["inner_test_overlap_n"] = int(inner_test_overlap)
    if groups is not None:
        group_arr = np.asarray(groups)
        train_groups = set(map(str, group_arr[np.asarray(train_idx, dtype=int)]))
        test_groups = set(map(str, group_arr[np.asarray(test_idx, dtype=int)]))
        group_overlap = len(train_groups.intersection(test_groups))
        if group_overlap:
            raise ValueError(f"Grouped outer split leaks {group_overlap} groups across train/test")
        diagnostics["group_overlap_n"] = int(group_overlap)
        if inner_train_idx is not None and inner_val_idx is not None:
            inner_train_groups = set(map(str, group_arr[np.asarray(inner_train_idx, dtype=int)]))
            inner_val_groups = set(map(str, group_arr[np.asarray(inner_val_idx, dtype=int)]))
            inner_group_overlap = len(inner_train_groups.intersection(inner_val_groups))
            if inner_group_overlap:
                raise ValueError(f"Grouped inner split leaks {inner_group_overlap} groups across train/validation")
            diagnostics["inner_group_overlap_n"] = int(inner_group_overlap)
    return diagnostics


def assign_oof_fold(sample_folds: np.ndarray, test_idx: np.ndarray, fold: int) -> None:
    already_assigned = np.asarray(test_idx, dtype=int)[sample_folds[np.asarray(test_idx, dtype=int)] != -1]
    if len(already_assigned):
        raise ValueError(f"OOF split assigns {len(already_assigned)} sample(s) to more than one outer fold")
    sample_folds[np.asarray(test_idx, dtype=int)] = int(fold)


def assert_all_oof_assigned(sample_folds: np.ndarray, samples: pd.Index) -> None:
    missing = np.asarray(samples)[np.asarray(sample_folds, dtype=int) < 0]
    if len(missing):
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(f"OOF split left {len(missing)} sample(s) without an outer-fold prediction: {preview}")


def _ranked_candidates(candidates: list[dict[str, Any]], seed: int, limit: int | None = None) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    shuffled = [dict(candidates[i]) for i in order]
    return shuffled[: int(limit)] if limit is not None and limit > 0 else shuffled


def _linear_candidates(task: str, seed: int, limit: int | None = None) -> list[dict[str, Any]]:
    if task == "regression":
        grid = [{"alpha": a, "l1_ratio": l1} for a, l1 in product(ELASTIC_ALPHA_GRID, LINEAR_L1_GRID)]
    elif task == "survival":
        grid = [{"penalizer": p, "l1_ratio": l1} for p, l1 in product(COX_PENALIZER_GRID, LINEAR_L1_GRID)]
    else:
        grid = [{"C": c, "l1_ratio": l1} for c, l1 in product(LOGISTIC_C_GRID, LINEAR_L1_GRID)]
    return _ranked_candidates(grid, seed, limit)


def _linear_candidate_limit(task: str, n_features: int, settings: dict[str, Any]) -> int | None:
    high_dim_threshold = int(settings.get("linear_high_dim_classification_solver_min_features", settings.get("linear_process_backend_min_features", 500)))
    if task in {"binary", "multiclass"} and int(n_features) >= high_dim_threshold:
        value = settings.get("linear_high_dim_random_search_trials", settings.get("linear_random_search_trials", settings.get("random_search_trials", settings.get("optuna_trials", None))))
    else:
        value = settings.get("linear_random_search_trials")
    if value is None:
        return None
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return None
    return max(1, limit) if limit > 0 else None


def _is_high_dim_linear_classification(task: str, learner: str, n_features: int, settings: dict[str, Any]) -> bool:
    threshold = int(settings.get("linear_high_dim_classification_solver_min_features", settings.get("linear_process_backend_min_features", 500)))
    return learner == "linear" and task in {"binary", "multiclass"} and int(n_features) >= threshold


def _xgb_candidates(seed: int, limit: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    candidates = []
    for _ in range(int(limit)):
        candidates.append(
            {
                "max_depth": int(rng.choice([2, 3, 4, 5])),
                "learning_rate": float(rng.choice([0.01, 0.03, 0.05, 0.08, 0.1])),
                "subsample": float(rng.choice([0.70, 0.85, 1.0])),
                "colsample_bytree": float(rng.choice([0.70, 0.85, 1.0])),
                "min_child_weight": float(rng.choice([1, 3, 5, 10])),
                "reg_lambda": float(rng.choice([0.5, 1.0, 2.0, 5.0, 10.0])),
                "reg_alpha": float(rng.choice([0.0, 0.01, 0.05, 0.1, 0.5])),
            }
        )
    return candidates


def _coerce_proba(proba: np.ndarray, n_classes: int) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    proba = np.nan_to_num(proba, nan=1.0 / max(n_classes, 1), posinf=1.0, neginf=0.0)
    row_sums = proba.sum(axis=1, keepdims=True)
    return np.divide(proba, row_sums, out=np.full_like(proba, 1.0 / max(proba.shape[1], 1)), where=row_sums > 0)


def _threadpool_context(limits: int = 1):
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=limits)


def _scale_pair(x_train: np.ndarray, x_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_pred_scaled = scaler.transform(x_pred)
    return np.asarray(x_train_scaled, dtype=np.float64), np.asarray(x_pred_scaled, dtype=np.float64)


def _logistic_n_jobs(candidate_parallel: bool, settings: dict[str, Any] | None = None) -> int:
    if candidate_parallel:
        return 1
    if settings and "linear_estimator_n_jobs" in settings:
        return int(settings["linear_estimator_n_jobs"])
    return -1


def _use_sgd_log_loss(task: str, n_features: int, settings: dict[str, Any] | None) -> bool:
    settings = settings or {}
    solver = str(settings.get("linear_high_dim_classification_solver", settings.get("linear_high_dim_multiclass_solver", ""))).strip().lower()
    threshold = int(settings.get("linear_high_dim_classification_solver_min_features", settings.get("linear_high_dim_multiclass_solver_min_features", settings.get("linear_process_backend_min_features", 500))))
    if task not in {"binary", "multiclass"} or solver not in {"sgd", "sgd_log_loss", "sgd_log_loss_elasticnet"}:
        return False
    if bool(settings.get("linear_force_sgd_log_loss", False)):
        return True
    return int(n_features) >= threshold


def _candidate_parallel_plan(settings: dict[str, Any], candidate_count: int, learner: str, task: str, n_features: int) -> tuple[int, str, str | int | None]:
    if learner == "xgboost" or candidate_count <= 1 or Parallel is None or delayed is None:
        return 1, "threading", None
    if task == "survival" or learner == "cox_ph":
        if int(n_features) >= int(settings.get("cox_high_dim_feature_threshold", 500)):
            configured = settings.get("cox_high_dim_candidate_n_jobs", settings.get("cox_candidate_n_jobs", 2))
        else:
            configured = settings.get("cox_candidate_n_jobs", 2)
    elif int(n_features) >= int(settings.get("linear_process_backend_min_features", 500)):
        configured = settings.get("linear_high_dim_candidate_n_jobs", 16)
    else:
        configured = settings.get("linear_candidate_n_jobs", settings.get("candidate_n_jobs", os.cpu_count() or 1))
    try:
        jobs = int(configured)
    except (TypeError, ValueError):
        jobs = os.cpu_count() or 1
    if jobs < 0:
        jobs = os.cpu_count() or 1
    if task == "survival" or learner == "cox_ph":
        if int(n_features) >= int(settings.get("cox_high_dim_feature_threshold", 500)):
            jobs = min(jobs, max(1, int(settings.get("cox_high_dim_max_candidate_n_jobs", settings.get("cox_max_candidate_n_jobs", 8)))))
        else:
            jobs = min(jobs, max(1, int(settings.get("cox_max_candidate_n_jobs", 8))))
    jobs = max(1, min(jobs, int(candidate_count)))
    high_dim_linear = task != "survival" and learner != "cox_ph" and int(n_features) >= int(settings.get("linear_process_backend_min_features", 500))
    multiclass_process_backend = bool(settings.get("linear_multiclass_process_backend", False))
    use_process_backend = high_dim_linear and (task != "multiclass" or multiclass_process_backend)
    backend = "loky" if use_process_backend else "threading"
    max_nbytes = settings.get("candidate_memmap_max_nbytes", "10M") if backend == "loky" else None
    return jobs, backend, max_nbytes


def _candidate_thread_limit(settings: dict[str, Any], learner: str, task: str, n_features: int) -> int:
    if task == "survival" or learner == "cox_ph":
        if int(n_features) >= int(settings.get("cox_high_dim_feature_threshold", 500)):
            return max(1, int(settings.get("cox_high_dim_blas_threads", 1)))
        return max(1, int(settings.get("cox_blas_threads", 16)))
    return max(1, int(settings.get("candidate_blas_threads", 1)))


def _linear_fit_predict_scaled(
    task: str,
    x_train_scaled: np.ndarray,
    y_train: Any,
    x_pred_scaled: np.ndarray,
    params: dict[str, Any],
    seed: int,
    n_classes: int,
    *,
    n_jobs: int = 1,
    settings: dict[str, Any] | None = None,
) -> np.ndarray:
    if task == "regression":
        model = ElasticNet(alpha=float(params["alpha"]), l1_ratio=float(params["l1_ratio"]), max_iter=10000, random_state=seed)
        model.fit(x_train_scaled, y_train)
        return np.asarray(model.predict(x_pred_scaled), dtype=np.float64)
    if _use_sgd_log_loss(task, x_train_scaled.shape[1], settings):
        c_value = max(float(params["C"]), 1e-12)
        alpha = float(settings.get("linear_sgd_alpha_scale", 1.0)) / max(c_value * float(len(y_train)), 1.0)
        model = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=float(params["l1_ratio"]),
            class_weight="balanced",
            max_iter=int(settings.get("linear_sgd_max_iter", 2000)) if settings else 2000,
            tol=float(settings.get("linear_sgd_tol", 1e-4)) if settings else 1e-4,
            random_state=seed,
            average=bool(settings.get("linear_sgd_average", True)) if settings else True,
            n_iter_no_change=int(settings.get("linear_sgd_n_iter_no_change", 5)) if settings else 5,
            early_stopping=bool(settings.get("linear_sgd_early_stopping", False)) if settings else False,
            validation_fraction=float(settings.get("linear_sgd_validation_fraction", 0.1)) if settings else 0.1,
        )
    else:
        model = LogisticRegression(
            C=float(params["C"]),
            l1_ratio=float(params["l1_ratio"]),
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=5000,
            n_jobs=int(n_jobs),
            random_state=seed,
        )
    model.fit(x_train_scaled, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        proba = _coerce_proba(model.predict_proba(x_pred_scaled), n_classes)
    out = np.zeros((x_pred_scaled.shape[0], n_classes), dtype=np.float64)
    for local_col, class_id in enumerate(model.classes_):
        out[:, int(class_id)] = proba[:, local_col]
    return out


def _linear_fit_predict(
    task: str,
    x_train: np.ndarray,
    y_train: Any,
    x_pred: np.ndarray,
    params: dict[str, Any],
    seed: int,
    n_classes: int,
    *,
    settings: dict[str, Any] | None = None,
) -> np.ndarray:
    x_train_scaled, x_pred_scaled = _scale_pair(x_train, x_pred)
    return _linear_fit_predict_scaled(
        task,
        x_train_scaled,
        y_train,
        x_pred_scaled,
        params,
        seed,
        n_classes,
        n_jobs=_logistic_n_jobs(candidate_parallel=False, settings=settings),
        settings=settings,
    )


def _cox_design_frames(
    x_train_scaled: np.ndarray,
    y_train: pd.DataFrame,
    x_pred_scaled: np.ndarray,
    *,
    min_std: float = 1e-8,
    drop_conditioned_low_variance: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_matrix = np.asarray(x_train_scaled, dtype=np.float64)
    pred_matrix = np.asarray(x_pred_scaled, dtype=np.float64)
    raw_feature_count = int(train_matrix.shape[1])
    train_std = np.nanstd(train_matrix, axis=0)
    keep = np.isfinite(train_std) & (train_std > float(min_std))
    if drop_conditioned_low_variance:
        events = y_train["event"].to_numpy(dtype=bool)
        if bool(np.any(events)) and bool(np.any(~events)):
            event_std = np.nanstd(train_matrix[events], axis=0)
            censor_std = np.nanstd(train_matrix[~events], axis=0)
            keep &= np.isfinite(event_std) & np.isfinite(censor_std)
            keep &= (event_std > float(min_std)) & (censor_std > float(min_std))
    if not np.any(keep):
        raise RuntimeError("No non-constant Cox covariates after fold-local variance filtering")
    train_matrix = train_matrix[:, keep]
    pred_matrix = pred_matrix[:, keep]
    columns = [f"x{i}" for i in np.flatnonzero(keep)]
    train_df = pd.DataFrame(train_matrix, columns=columns)
    train_df["time"] = pd.Series(y_train["time"].to_numpy(dtype=float))
    train_df["event"] = pd.Series(y_train["event"].to_numpy(dtype=int))
    pred_df = pd.DataFrame(pred_matrix, columns=columns)
    train_df.attrs["raw_feature_count"] = raw_feature_count
    pred_df.attrs["raw_feature_count"] = raw_feature_count
    return train_df, pred_df


def _cox_fit_predict_frames(train_df: pd.DataFrame, pred_df: pd.DataFrame, params: dict[str, Any], settings: dict[str, Any] | None = None) -> np.ndarray:
    settings = settings or {}
    feature_count = max(0, int(train_df.attrs.get("raw_feature_count", int(train_df.shape[1]) - 2)))
    high_dim = feature_count >= int(settings.get("cox_high_dim_feature_threshold", 500))
    if high_dim and str(settings.get("cox_high_dim_backend", "fast_breslow")) == "fast_breslow":
        return _fast_breslow_cox_fit_predict_frames(train_df, pred_df, params, settings)

    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise RuntimeError("lifelines is required for Cox survival endpoints") from exc

    batch_mode = settings.get("cox_batch_mode", True)
    fit_options: dict[str, Any] = {}
    max_steps = settings.get("cox_high_dim_max_steps") if high_dim and settings.get("cox_high_dim_max_steps") is not None else settings.get("cox_max_steps")
    precision = settings.get("cox_high_dim_precision") if high_dim and settings.get("cox_high_dim_precision") is not None else settings.get("cox_precision")
    if max_steps is not None:
        fit_options["max_steps"] = int(max_steps)
    if precision is not None:
        fit_options["precision"] = float(precision)
    if settings.get("cox_r_precision") is not None:
        fit_options["r_precision"] = float(settings["cox_r_precision"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph = CoxPHFitter(penalizer=float(params["penalizer"]), l1_ratio=float(params["l1_ratio"]))
        cph.fit(
            train_df,
            duration_col="time",
            event_col="event",
            show_progress=False,
            batch_mode=bool(batch_mode),
            fit_options=fit_options or None,
        )
    return cph.predict_partial_hazard(pred_df).to_numpy(dtype=np.float64).reshape(-1)


def _breslow_cox_gradient(x: np.ndarray, time: np.ndarray, event: np.ndarray, beta: np.ndarray) -> np.ndarray:
    order = np.argsort(-time, kind="mergesort")
    x_sorted = x[order]
    time_sorted = time[order]
    event_sorted = event[order].astype(bool)
    eta = np.clip(x_sorted @ beta, -50.0, 50.0)
    exp_eta = np.exp(eta)
    weighted_x = x_sorted * exp_eta[:, None]
    cum_exp = np.cumsum(exp_eta)
    cum_x = np.cumsum(weighted_x, axis=0)

    new_group = np.r_[True, time_sorted[1:] != time_sorted[:-1]]
    group_id = np.cumsum(new_group) - 1
    group_starts = np.flatnonzero(new_group)
    group_ends = np.r_[group_starts[1:] - 1, len(time_sorted) - 1]
    risk_exp = cum_exp[group_ends][group_id]
    risk_x = cum_x[group_ends][group_id]

    if not np.any(event_sorted):
        return np.zeros(x.shape[1], dtype=np.float64)
    expected = risk_x[event_sorted] / np.maximum(risk_exp[event_sorted], 1e-12)[:, None]
    gradient = -(x_sorted[event_sorted] - expected).sum(axis=0)
    return gradient / max(1, int(event_sorted.sum()))


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - float(threshold), 0.0)


def _fast_breslow_cox_fit_predict_frames(train_df: pd.DataFrame, pred_df: pd.DataFrame, params: dict[str, Any], settings: dict[str, Any]) -> np.ndarray:
    feature_cols = [column for column in train_df.columns if column not in {"time", "event"}]
    if not feature_cols:
        raise RuntimeError("No Cox covariates available for fast Breslow optimizer")
    x_train = train_df[feature_cols].to_numpy(dtype=np.float64, copy=False)
    x_pred = pred_df[feature_cols].to_numpy(dtype=np.float64, copy=False)
    time = train_df["time"].to_numpy(dtype=np.float64)
    event = train_df["event"].to_numpy(dtype=np.int8)
    beta = np.zeros(x_train.shape[1], dtype=np.float64)
    penalizer = float(params["penalizer"])
    l1_ratio = float(params["l1_ratio"])
    max_iter = max(1, int(settings.get("cox_fast_max_iter", settings.get("cox_high_dim_max_steps", 5))))
    learning_rate = float(settings.get("cox_fast_learning_rate", 0.05)) / (1.0 + penalizer)
    l2_weight = penalizer * max(0.0, 1.0 - l1_ratio)
    l1_weight = penalizer * max(0.0, l1_ratio)
    for _ in range(max_iter):
        grad = _breslow_cox_gradient(x_train, time, event, beta)
        if l2_weight:
            grad = grad + l2_weight * beta
        beta = beta - learning_rate * grad
        if l1_weight:
            beta = _soft_threshold(beta, learning_rate * l1_weight)
        beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(x_pred @ beta, dtype=np.float64).reshape(-1)


def _cox_fit_predict_scaled(x_train_scaled: np.ndarray, y_train: pd.DataFrame, x_pred_scaled: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    train_df, pred_df = _cox_design_frames(x_train_scaled, y_train, x_pred_scaled)
    return _cox_fit_predict_frames(train_df, pred_df, params)


def _cox_fit_predict(x_train: np.ndarray, y_train: pd.DataFrame, x_pred: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    x_train_scaled, x_pred_scaled = _scale_pair(x_train, x_pred)
    return _cox_fit_predict_scaled(x_train_scaled, y_train, x_pred_scaled, params)


def _xgb_model(task: str, y_train: np.ndarray, n_classes: int, params: dict[str, Any], seed: int, settings: dict[str, Any], *, n_estimators: int | None = None):
    base = {
        "n_estimators": int(n_estimators or settings.get("xgb_max_rounds", settings.get("xgb_estimators", 160))),
        "max_depth": int(params.get("max_depth", settings.get("xgb_max_depth", 2))),
        "learning_rate": float(params.get("learning_rate", settings.get("xgb_learning_rate", 0.05))),
        "subsample": float(params.get("subsample", 0.85)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.85)),
        "min_child_weight": float(params.get("min_child_weight", 5)),
        "reg_lambda": float(params.get("reg_lambda", 2.0)),
        "reg_alpha": float(params.get("reg_alpha", 0.05)),
        "random_state": int(seed),
        "n_jobs": int(settings.get("xgb_n_jobs", settings.get("n_jobs", 4))),
        "tree_method": str(settings.get("tree_method", "hist")),
        "verbosity": 0,
    }
    if base["tree_method"] == "gpu_hist":
        base["predictor"] = "gpu_predictor"
    if task == "regression":
        from xgboost import XGBRegressor

        return XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **base)
    from xgboost import XGBClassifier

    if task == "binary":
        positives = float(np.sum(y_train == 1))
        negatives = float(np.sum(y_train == 0))
        base.update({"objective": "binary:logistic", "eval_metric": "auc", "scale_pos_weight": negatives / max(positives, 1.0)})
    else:
        base.update({"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": int(n_classes)})
    return XGBClassifier(**base)


def _constant_class_proba(class_id: int, n_rows: int, n_classes: int) -> np.ndarray:
    if class_id < 0 or class_id >= int(n_classes):
        raise ValueError(f"Class id {class_id} is outside global class range 0..{int(n_classes) - 1}")
    out = np.zeros((int(n_rows), int(n_classes)), dtype=np.float64)
    out[:, int(class_id)] = 1.0
    return out


def _local_multiclass_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    global_classes = np.array(sorted(np.unique(np.asarray(y, dtype=np.int32))), dtype=np.int32)
    mapping = {int(class_id): local_id for local_id, class_id in enumerate(global_classes)}
    local = np.array([mapping[int(class_id)] for class_id in np.asarray(y, dtype=np.int32)], dtype=np.int32)
    return local, global_classes, mapping


def _fit_xgb_with_optional_fallback(model: Any, x_train: np.ndarray, y_train: np.ndarray, **fit_kwargs: Any) -> Any:
    def _fit_once(**kwargs: Any) -> Any:
        try:
            return model.fit(x_train, y_train, **kwargs)
        except Exception:
            if model.get_params().get("tree_method") == "gpu_hist":
                model.set_params(tree_method="hist", predictor="auto")
                return model.fit(x_train, y_train, **kwargs)
            raise

    fit_kwargs = dict(fit_kwargs)
    try:
        return _fit_once(**fit_kwargs)
    except TypeError as exc:
        if "early_stopping_rounds" in fit_kwargs:
            early = fit_kwargs.pop("early_stopping_rounds")
            verbose = fit_kwargs.pop("verbose", False)
            try:
                model.set_params(early_stopping_rounds=early)
                return _fit_once(verbose=verbose, **fit_kwargs)
            except TypeError:
                return _fit_once(**fit_kwargs)
        raise exc


def _xgb_fit_predict(
    task: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    params: dict[str, Any],
    seed: int,
    settings: dict[str, Any],
    n_classes: int,
    *,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    final_rounds: int | None = None,
) -> tuple[np.ndarray, int]:
    y_fit = np.asarray(y_train)
    local_classes: np.ndarray | None = None
    local_mapping: dict[int, int] | None = None
    model_n_classes = int(n_classes)
    if task in {"binary", "multiclass", "multiclass_grouped"}:
        train_classes = np.array(sorted(np.unique(np.asarray(y_train, dtype=np.int32))), dtype=np.int32)
        if len(train_classes) == 0:
            raise ValueError("XGBoost received an empty training label vector")
        if len(train_classes) == 1:
            return _constant_class_proba(int(train_classes[0]), x_pred.shape[0], n_classes), 1
        if task in {"multiclass", "multiclass_grouped"}:
            y_fit, local_classes, local_mapping = _local_multiclass_labels(np.asarray(y_train, dtype=np.int32))
            model_n_classes = int(len(local_classes))

    model = _xgb_model(task, y_fit, model_n_classes, params, seed, settings, n_estimators=final_rounds)
    fit_kwargs: dict[str, Any] = {}
    if x_val is not None and y_val is not None and final_rounds is None:
        eval_x = x_val
        eval_y = np.asarray(y_val)
        if task == "binary":
            if len(np.unique(eval_y)) >= 2:
                fit_kwargs["eval_set"] = [(eval_x, eval_y)]
        elif task in {"multiclass", "multiclass_grouped"} and local_classes is not None and local_mapping is not None:
            eval_y_global = np.asarray(y_val, dtype=np.int32)
            eval_mask = np.isin(eval_y_global, local_classes)
            if bool(np.any(eval_mask)):
                eval_x = x_val[eval_mask]
                eval_y = np.array([local_mapping[int(class_id)] for class_id in eval_y_global[eval_mask]], dtype=np.int32)
                fit_kwargs["eval_set"] = [(eval_x, eval_y)]
        else:
            fit_kwargs["eval_set"] = [(eval_x, eval_y)]
        if "eval_set" in fit_kwargs:
            fit_kwargs["verbose"] = False
            fit_kwargs["early_stopping_rounds"] = int(settings.get("xgb_early_stopping_rounds", 20))
    _fit_xgb_with_optional_fallback(model, x_train, y_fit, **fit_kwargs)
    best_round = int(getattr(model, "best_iteration", -1) + 1) if getattr(model, "best_iteration", None) is not None else int(model.get_params().get("n_estimators", 1))
    best_round = max(1, best_round)
    if task == "regression":
        return np.asarray(model.predict(x_pred), dtype=np.float64), best_round
    proba = _coerce_proba(model.predict_proba(x_pred), model_n_classes)
    out = np.zeros((x_pred.shape[0], n_classes), dtype=np.float64)
    if local_classes is not None:
        for local_col, class_id in enumerate(local_classes):
            if local_col < proba.shape[1]:
                out[:, int(class_id)] = proba[:, local_col]
    else:
        for local_col, class_id in enumerate(model.classes_):
            out[:, int(class_id)] = proba[:, local_col]
    return out, best_round


def _score_predictions(task: str, y_true: Any, pred: np.ndarray, n_classes: int, metric: str) -> float:
    metric = str(metric or "").lower()
    if task == "regression":
        score = spearmanr(np.asarray(y_true, dtype=float), np.asarray(pred, dtype=float))[0]
        return float(score) if np.isfinite(score) else float("nan")
    if task == "binary":
        if len(np.unique(y_true)) < 2:
            return float("nan")
        predicted = (pred[:, 1] >= 0.5).astype(int)
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, predicted))
        if metric == "accuracy":
            return float(accuracy_score(y_true, predicted))
        if metric == "macro_f1":
            return float(f1_score(y_true, predicted, average="macro", zero_division=0))
        return float(roc_auc_score(y_true, pred[:, 1]))
    if task in {"multiclass", "multiclass_grouped"}:
        predicted = np.argmax(pred, axis=1)
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, predicted))
        if metric == "accuracy":
            return float(accuracy_score(y_true, predicted))
        if metric == "macro_f1":
            return float(f1_score(y_true, predicted, average="macro", zero_division=0))
        if metric == "micro_auroc":
            return safe_micro_auroc(y_true, pred, n_classes)
        return safe_macro_auroc(y_true, pred, n_classes)
    y_df = pd.DataFrame(y_true)
    score = harrell_c_index(y_df["time"], y_df["event"], pred)
    return float(score) if np.isfinite(score) else float("nan")


def _candidate_score(task: str, y_true: Any, pred: np.ndarray, n_classes: int, metric: str) -> float:
    score = _score_predictions(task, y_true, pred, n_classes, metric)
    return float(score) if np.isfinite(score) else float("-inf")


def _slice_y(task: str, y: Any, idx: np.ndarray) -> Any:
    return y.iloc[idx].copy() if task == "survival" else np.asarray(y)[idx]


def _search_and_predict_fold(
    *,
    task: str,
    learner: str,
    x: np.ndarray,
    y: Any,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: np.ndarray | None,
    seed: int,
    n_classes: int,
    primary_metric: str,
    settings: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    inner_train_idx, inner_val_idx = inner_train_val_split(task, y, train_idx, groups=groups, seed=seed)
    split_integrity = assert_split_integrity(train_idx, test_idx, inner_train_idx, inner_val_idx, groups=groups)
    x_inner = x[inner_train_idx]
    y_inner = _slice_y(task, y, inner_train_idx)
    x_val = x[inner_val_idx]
    y_val = _slice_y(task, y, inner_val_idx)
    x_outer = x[train_idx]
    y_outer = _slice_y(task, y, train_idx)
    x_test = x[test_idx]

    if learner == "xgboost" and task == "survival":
        raise ValueError("Use learner='cox_ph' for survival endpoints")
    if learner == "xgboost":
        limit = int(settings.get("random_search_trials", settings.get("optuna_trials", 10)))
        candidates = _xgb_candidates(seed, max(1, limit))
    else:
        candidates = _linear_candidates(task, seed, _linear_candidate_limit(task, int(x.shape[1]), settings))

    if learner != "xgboost":
        x_inner_scaled, x_val_scaled = _scale_pair(x_inner, x_val)
        x_outer_scaled, x_test_scaled = _scale_pair(x_outer, x_test)
        if task == "survival" or learner == "cox_ph":
            cox_min_std = float(settings.get("cox_min_feature_std", 1e-8))
            drop_conditioned = bool(settings.get("cox_drop_conditioned_low_variance", True))
            cox_inner_train_df, cox_val_df = _cox_design_frames(
                x_inner_scaled,
                pd.DataFrame(y_inner),
                x_val_scaled,
                min_std=cox_min_std,
                drop_conditioned_low_variance=drop_conditioned,
            )
            cox_outer_train_df, cox_test_df = _cox_design_frames(
                x_outer_scaled,
                pd.DataFrame(y_outer),
                x_test_scaled,
                min_std=cox_min_std,
                drop_conditioned_low_variance=drop_conditioned,
            )
        else:
            cox_inner_train_df = cox_val_df = cox_outer_train_df = cox_test_df = None
    else:
        x_inner_scaled = x_val_scaled = x_outer_scaled = x_test_scaled = None
        cox_inner_train_df = cox_val_df = cox_outer_train_df = cox_test_df = None

    def _evaluate_validation_candidate(candidate_id: int, params: dict[str, Any]) -> dict[str, Any]:
        try:
            if learner == "xgboost":
                pred_val, rounds = _xgb_fit_predict(task, x_inner, y_inner, x_val, params, seed + candidate_id, settings, n_classes, x_val=x_val, y_val=y_val)
            elif task == "survival" or learner == "cox_ph":
                pred_val = _cox_fit_predict_frames(cox_inner_train_df, cox_val_df, params, settings)  # type: ignore[arg-type]
                rounds = None
            else:
                pred_val = _linear_fit_predict_scaled(
                    task,
                    x_inner_scaled,  # type: ignore[arg-type]
                    y_inner,
                    x_val_scaled,  # type: ignore[arg-type]
                    params,
                    seed + candidate_id,
                    n_classes,
                    n_jobs=_logistic_n_jobs(candidate_parallel=True, settings=settings),
                    settings=settings,
                )
                rounds = None
            score = _candidate_score(task, y_val, pred_val, n_classes, primary_metric)
            status = "ok"
        except Exception as exc:
            score = float("-inf")
            rounds = None
            message = str(exc).replace("\n", " ").strip()
            status = f"failed:{type(exc).__name__}:{message[:240]}"
        return {"candidate_id": candidate_id, "score": score, "params": params, "m_star": rounds, "status": status}

    candidate_jobs, candidate_backend, candidate_max_nbytes = _candidate_parallel_plan(settings, len(candidates), learner, task, x.shape[1])
    if candidate_jobs > 1:
        parallel_kwargs: dict[str, Any] = {"n_jobs": candidate_jobs, "backend": candidate_backend}
        if candidate_backend == "threading":
            parallel_kwargs.update({"prefer": "threads", "require": "sharedmem"})
        else:
            parallel_kwargs.update({"prefer": "processes", "max_nbytes": candidate_max_nbytes})
        config_ctx = parallel_config(backend="loky", inner_max_num_threads=1) if parallel_config is not None and candidate_backend == "loky" else nullcontext()
        with _threadpool_context(_candidate_thread_limit(settings, learner, task, x.shape[1])), config_ctx:
            candidate_rows = Parallel(**parallel_kwargs)(
                delayed(_evaluate_validation_candidate)(candidate_id, params)
                for candidate_id, params in enumerate(candidates, start=1)
            )
    else:
        with _threadpool_context(_candidate_thread_limit(settings, learner, task, x.shape[1])):
            candidate_rows = [_evaluate_validation_candidate(candidate_id, params) for candidate_id, params in enumerate(candidates, start=1)]

    best_params: dict[str, Any] | None = None
    best_score = float("-inf")
    best_rounds: int | None = None
    for row in sorted(candidate_rows, key=lambda item: int(item["candidate_id"])):
        score = float(row["score"])
        if score > best_score:
            best_score = score
            best_params = dict(row["params"])
            best_rounds = row["m_star"]
    if best_params is None or not np.isfinite(best_score):
        raise RuntimeError(f"No valid nested-search candidate for task={task} learner={learner}")

    if learner == "xgboost":
        pred_test, final_rounds = _xgb_fit_predict(task, x_outer, y_outer, x_test, best_params, seed + 99_991, settings, n_classes, final_rounds=best_rounds)
        best_rounds = int(final_rounds)
    elif task == "survival" or learner == "cox_ph":
        with _threadpool_context(max(1, int(settings.get("cox_final_blas_threads", os.cpu_count() or 1)))):
            pred_test = _cox_fit_predict_frames(cox_outer_train_df, cox_test_df, best_params, settings)  # type: ignore[arg-type]
    else:
        pred_test = _linear_fit_predict_scaled(
            task,
            x_outer_scaled,  # type: ignore[arg-type]
            y_outer,
            x_test_scaled,  # type: ignore[arg-type]
            best_params,
            seed + 99_991,
            n_classes,
            n_jobs=_logistic_n_jobs(candidate_parallel=False, settings=settings),
            settings=settings,
        )

    diagnostics = {
        "inner_train_n": int(len(inner_train_idx)),
        "inner_val_n": int(len(inner_val_idx)),
        "outer_train_n": int(len(train_idx)),
        "outer_test_n": int(len(test_idx)),
        "selected_score": float(best_score),
        "selected_params": best_params,
        "selected_m_star": int(best_rounds) if best_rounds is not None else None,
        "candidate_count": int(len(candidate_rows)),
        "candidate_results": candidate_rows,
        **split_integrity,
    }
    return pred_test, diagnostics


def _search_and_predict_feature_set_fold(
    *,
    task: str,
    learner: str,
    x_by_feature_set: dict[str, np.ndarray],
    y: Any,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: np.ndarray | None,
    seed: int,
    n_classes: int,
    primary_metric: str,
    settings: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    inner_train_idx, inner_val_idx = inner_train_val_split(task, y, train_idx, groups=groups, seed=seed)
    split_integrity = assert_split_integrity(train_idx, test_idx, inner_train_idx, inner_val_idx, groups=groups)
    y_inner = _slice_y(task, y, inner_train_idx)
    y_val = _slice_y(task, y, inner_val_idx)
    y_outer = _slice_y(task, y, train_idx)

    if learner == "xgboost" and task == "survival":
        raise ValueError("Use learner='cox_ph' for survival endpoints")
    if learner == "xgboost":
        limit = int(settings.get("xgb_feature_set_random_search_trials", settings.get("random_search_trials", settings.get("optuna_trials", 10))))
        model_candidates = _xgb_candidates(seed, max(1, limit))
    else:
        max_input_features = max(int(np.asarray(matrix).shape[1]) for matrix in x_by_feature_set.values())
        model_candidates = _linear_candidates(task, seed, _linear_candidate_limit(task, max_input_features, settings))
    fit_settings = settings
    if _is_high_dim_linear_classification(task, learner, max_input_features if learner != "xgboost" else 0, settings) and bool(
        settings.get("linear_force_sgd_in_feature_set_search", True)
    ):
        fit_settings = dict(settings)
        fit_settings["linear_force_sgd_log_loss"] = True

    prepared: dict[str, dict[str, Any]] = {}
    max_features = 0
    for feature_set_name, x in x_by_feature_set.items():
        arr = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        max_features = max(max_features, int(arr.shape[1]))
        prep: dict[str, Any] = {
            "x_inner": arr[inner_train_idx],
            "x_val": arr[inner_val_idx],
            "x_outer": arr[train_idx],
            "x_test": arr[test_idx],
            "feature_count": int(arr.shape[1]),
        }
        if learner != "xgboost":
            prep["x_inner_scaled"], prep["x_val_scaled"] = _scale_pair(prep["x_inner"], prep["x_val"])
            prep["x_outer_scaled"], prep["x_test_scaled"] = _scale_pair(prep["x_outer"], prep["x_test"])
            if task == "survival" or learner == "cox_ph":
                cox_min_std = float(settings.get("cox_min_feature_std", 1e-8))
                drop_conditioned = bool(settings.get("cox_drop_conditioned_low_variance", True))
                prep["cox_inner_train_df"], prep["cox_val_df"] = _cox_design_frames(
                    prep["x_inner_scaled"],
                    pd.DataFrame(y_inner),
                    prep["x_val_scaled"],
                    min_std=cox_min_std,
                    drop_conditioned_low_variance=drop_conditioned,
                )
                prep["cox_outer_train_df"], prep["cox_test_df"] = _cox_design_frames(
                    prep["x_outer_scaled"],
                    pd.DataFrame(y_outer),
                    prep["x_test_scaled"],
                    min_std=cox_min_std,
                    drop_conditioned_low_variance=drop_conditioned,
                )
        prepared[str(feature_set_name)] = prep

    flat_candidates: list[dict[str, Any]] = []
    candidate_id = 1
    for feature_set_name in sorted(prepared):
        for params in model_candidates:
            flat_candidates.append({"candidate_id": candidate_id, "feature_set": feature_set_name, "params": dict(params)})
            candidate_id += 1

    def _evaluate_validation_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        candidate_id = int(candidate["candidate_id"])
        feature_set_name = str(candidate["feature_set"])
        params = dict(candidate["params"])
        prep = prepared[feature_set_name]
        try:
            if learner == "xgboost":
                pred_val, rounds = _xgb_fit_predict(
                    task,
                    prep["x_inner"],
                    y_inner,
                    prep["x_val"],
                    params,
                    seed + candidate_id,
                    settings,
                    n_classes,
                    x_val=prep["x_val"],
                    y_val=y_val,
                )
            elif task == "survival" or learner == "cox_ph":
                pred_val = _cox_fit_predict_frames(prep["cox_inner_train_df"], prep["cox_val_df"], params, settings)
                rounds = None
            else:
                pred_val = _linear_fit_predict_scaled(
                    task,
                    prep["x_inner_scaled"],
                    y_inner,
                    prep["x_val_scaled"],
                    params,
                    seed + candidate_id,
                    n_classes,
                    n_jobs=_logistic_n_jobs(candidate_parallel=True, settings=fit_settings),
                    settings=fit_settings,
                )
                rounds = None
            score = _candidate_score(task, y_val, pred_val, n_classes, primary_metric)
            status = "ok"
        except Exception as exc:
            score = float("-inf")
            rounds = None
            message = str(exc).replace("\n", " ").strip()
            status = f"failed:{type(exc).__name__}:{message[:240]}"
        return {
            "candidate_id": candidate_id,
            "feature_set": feature_set_name,
            "feature_count": int(prep["feature_count"]),
            "score": score,
            "params": params,
            "m_star": rounds,
            "status": status,
        }

    candidate_checkpoint_path = _candidate_checkpoint_path(settings, seed)
    use_candidate_checkpoint = (
        candidate_checkpoint_path is not None
        and bool(settings.get("candidate_checkpoint_enabled", True))
        and (learner == "xgboost" or _is_high_dim_linear_classification(task, learner, max_features, settings))
    )
    if use_candidate_checkpoint:
        candidate_manifest = _candidate_checkpoint_manifest(
            task=task,
            learner=learner,
            seed=seed,
            primary_metric=primary_metric,
            n_classes=n_classes,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            inner_train_idx=inner_train_idx,
            inner_val_idx=inner_val_idx,
            flat_candidates=flat_candidates,
            feature_counts={name: int(prep["feature_count"]) for name, prep in prepared.items()},
            feature_digests={name: _matrix_digest(prep["x_inner"]) + ":" + _matrix_digest(prep["x_val"]) for name, prep in prepared.items()},
            settings=fit_settings,
        )
        candidate_rows_by_id = _load_candidate_checkpoint(candidate_checkpoint_path, candidate_manifest)
        progress_label = str(settings.get("fold_progress_label") or "nested feature-set search")
        if candidate_rows_by_id:
            print(
                f"[nested_oof] resume candidate search {progress_label}; "
                f"completed {len(candidate_rows_by_id)}/{len(flat_candidates)}",
                flush=True,
            )
        candidate_rows = []
        with _threadpool_context(_candidate_thread_limit(fit_settings, learner, task, max_features)):
            for candidate in flat_candidates:
                candidate_id = int(candidate["candidate_id"])
                if candidate_id in candidate_rows_by_id:
                    row = dict(candidate_rows_by_id[candidate_id])
                else:
                    start = time.time()
                    row = _evaluate_validation_candidate(candidate)
                    candidate_rows_by_id[candidate_id] = dict(row)
                    _write_candidate_checkpoint(candidate_checkpoint_path, candidate_manifest, candidate_rows_by_id, status="running")
                    progress_every = max(1, int(settings.get("candidate_progress_every", 1)))
                    if len(candidate_rows_by_id) % progress_every == 0 or len(candidate_rows_by_id) == len(flat_candidates):
                        score = row.get("score")
                        score_text = f"{float(score):.6f}" if score is not None and np.isfinite(float(score)) else "nan"
                        print(
                            f"[nested_oof] candidate {len(candidate_rows_by_id)}/{len(flat_candidates)} "
                            f"{progress_label}; feature_set={row.get('feature_set')} score={score_text} "
                            f"seconds={time.time() - start:.1f}",
                            flush=True,
                        )
                candidate_rows.append(row)
        _write_candidate_checkpoint(candidate_checkpoint_path, candidate_manifest, candidate_rows_by_id, status="completed")
    else:
        candidate_jobs, candidate_backend, candidate_max_nbytes = _candidate_parallel_plan(fit_settings, len(flat_candidates), learner, task, max_features)
        if candidate_jobs > 1:
            parallel_kwargs: dict[str, Any] = {"n_jobs": candidate_jobs, "backend": candidate_backend}
            if candidate_backend == "threading":
                parallel_kwargs.update({"prefer": "threads", "require": "sharedmem"})
            else:
                parallel_kwargs.update({"prefer": "processes", "max_nbytes": candidate_max_nbytes})
            config_ctx = parallel_config(backend="loky", inner_max_num_threads=1) if parallel_config is not None and candidate_backend == "loky" else nullcontext()
            with _threadpool_context(_candidate_thread_limit(fit_settings, learner, task, max_features)), config_ctx:
                candidate_rows = Parallel(**parallel_kwargs)(delayed(_evaluate_validation_candidate)(candidate) for candidate in flat_candidates)
        else:
            with _threadpool_context(_candidate_thread_limit(fit_settings, learner, task, max_features)):
                candidate_rows = [_evaluate_validation_candidate(candidate) for candidate in flat_candidates]

    best_row: dict[str, Any] | None = None
    best_score = float("-inf")
    for row in sorted(candidate_rows, key=lambda item: int(item["candidate_id"])):
        score = float(row["score"])
        if score > best_score:
            best_score = score
            best_row = row
    if best_row is None or not np.isfinite(best_score):
        raise RuntimeError(f"No valid nested feature-set candidate for task={task} learner={learner}")

    best_feature_set = str(best_row["feature_set"])
    best_params = dict(best_row["params"])
    best_rounds = best_row["m_star"]
    prep = prepared[best_feature_set]
    if learner == "xgboost":
        pred_test, final_rounds = _xgb_fit_predict(task, prep["x_outer"], y_outer, prep["x_test"], best_params, seed + 99_991, settings, n_classes, final_rounds=best_rounds)
        best_rounds = int(final_rounds)
    elif task == "survival" or learner == "cox_ph":
        with _threadpool_context(max(1, int(settings.get("cox_final_blas_threads", os.cpu_count() or 1)))):
            pred_test = _cox_fit_predict_frames(prep["cox_outer_train_df"], prep["cox_test_df"], best_params, settings)
    else:
        pred_test = _linear_fit_predict_scaled(
            task,
            prep["x_outer_scaled"],
            y_outer,
            prep["x_test_scaled"],
            best_params,
            seed + 99_991,
            n_classes,
            n_jobs=_logistic_n_jobs(candidate_parallel=False, settings=fit_settings),
            settings=fit_settings,
        )

    diagnostics = {
        "inner_train_n": int(len(inner_train_idx)),
        "inner_val_n": int(len(inner_val_idx)),
        "outer_train_n": int(len(train_idx)),
        "outer_test_n": int(len(test_idx)),
        "selected_score": float(best_score),
        "selected_params": best_params,
        "selected_m_star": int(best_rounds) if best_rounds is not None else None,
        "selected_feature_set": best_feature_set,
        "selected_feature_count": int(prep["feature_count"]),
        "candidate_feature_set_count": int(len(prepared)),
        "candidate_feature_sets": sorted(prepared),
        "candidate_count": int(len(candidate_rows)),
        "candidate_results": candidate_rows,
        **split_integrity,
    }
    return pred_test, diagnostics


def evaluate_nested_oof(
    *,
    samples: Iterable[str],
    x: np.ndarray,
    labels: pd.Series | pd.DataFrame,
    task: str,
    benchmark: str,
    endpoint: str,
    representation: str,
    learner: str,
    groups: pd.Series | np.ndarray | None = None,
    settings: dict[str, Any] | None = None,
    n_splits: int = 5,
    seed: int | None = None,
) -> NestedOOFResult:
    settings = dict(settings or {})
    grouped_task = task == "multiclass_grouped"
    task = "multiclass" if grouped_task else task
    samples = pd.Index([str(value) for value in samples], name="sample")
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y, classes = encode_labels(labels, task)
    group_arr = None
    if groups is not None:
        group_arr = pd.Series(groups).loc[samples].to_numpy() if isinstance(groups, pd.Series) else np.asarray(groups)
    seed = int(seed if seed is not None else stable_seed(benchmark, endpoint, representation, learner))
    n_classes = 2 if task == "binary" else len(classes)
    primary_metric = _primary_metric(task, endpoint, settings)
    splits = outer_splits(task, y, groups=group_arr, seed=seed, n_splits=n_splits)

    pred = np.zeros(len(samples), dtype=np.float64) if task in {"regression", "survival"} else np.zeros((len(samples), n_classes), dtype=np.float64)
    sample_folds = np.full(len(samples), -1, dtype=int)
    fold_rows: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for fold, train_idx, test_idx in splits:
            fold_pred, diag = _search_and_predict_fold(
                task=task,
                learner=learner,
                x=x,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                groups=group_arr,
                seed=seed + fold * 1009,
                n_classes=n_classes,
                primary_metric=primary_metric,
                settings=settings,
            )
            pred[test_idx] = fold_pred
            assign_oof_fold(sample_folds, test_idx, int(fold))
            fold_score = _candidate_score(task, _slice_y(task, y, test_idx), fold_pred, n_classes, primary_metric)
            fold_rows.append(
                {
                    "repeat": 1,
                    "fold": int(fold),
                    "metric": primary_metric,
                    "score": float(fold_score) if np.isfinite(fold_score) else float("nan"),
                    "n_test": int(len(test_idx)),
                    "inner_train_n": diag["inner_train_n"],
                    "inner_val_n": diag["inner_val_n"],
                    "outer_train_n": diag["outer_train_n"],
                    "outer_test_n": diag["outer_test_n"],
                    "selected_inner_score": diag["selected_score"],
                    "selected_params_json": _safe_json(diag["selected_params"]),
                    "selected_m_star": diag["selected_m_star"],
                    "candidate_count": diag["candidate_count"],
                    "candidate_results_json": _safe_json(diag["candidate_results"]),
                    "split_strategy": _split_strategy_label(task, group_arr is not None, len(splits)),
                    "outer_overlap_n": diag["outer_overlap_n"],
                    "inner_overlap_n": diag["inner_overlap_n"],
                    "inner_test_overlap_n": diag["inner_test_overlap_n"],
                    "group_overlap_n": diag["group_overlap_n"],
                    "inner_group_overlap_n": diag["inner_group_overlap_n"],
                }
            )

    assert_all_oof_assigned(sample_folds, samples)
    summary, pred_frame = _summarize_predictions(
        task=task,
        grouped_task=grouped_task,
        y=y,
        pred=pred,
        samples=samples,
        benchmark=benchmark,
        endpoint=endpoint,
        representation=representation,
        learner=learner,
        classes=classes,
        n_features=int(x.shape[1]),
        n_folds=len(splits),
        split_strategy=_split_strategy_label(task, group_arr is not None, len(splits)),
        linear_solver=_solver_label(learner, task, int(x.shape[1]), settings),
        primary_metric=primary_metric,
    )
    pred_frame["fold"] = sample_folds
    fold_frame = pd.DataFrame(fold_rows)
    if not fold_frame.empty:
        fold_frame.insert(0, "learner", learner)
        fold_frame.insert(0, "representation", representation)
        fold_frame.insert(0, "endpoint", endpoint)
        fold_frame.insert(0, "benchmark", benchmark)
    return NestedOOFResult(summary, pred_frame, fold_frame)


def _safe_checkpoint_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))
    return safe[:180] or "nested_oof"


def _feature_set_checkpoint_paths(settings: dict[str, Any]) -> tuple[Path, Path] | None:
    if not bool(settings.get("fold_checkpoint_enabled", False)):
        return None
    directory = settings.get("fold_checkpoint_dir")
    key = settings.get("fold_checkpoint_key")
    if not directory or not key:
        return None
    root = Path(str(directory))
    root.mkdir(parents=True, exist_ok=True)
    stem = _safe_checkpoint_name(str(key))
    return root / f"{stem}.npz", root / f"{stem}.json"


def _candidate_checkpoint_path(settings: dict[str, Any], seed: int) -> Path | None:
    if not bool(settings.get("fold_checkpoint_enabled", False)):
        return None
    directory = settings.get("fold_checkpoint_dir")
    key = settings.get("fold_checkpoint_key")
    if not directory or not key:
        return None
    root = Path(str(directory)) / "candidate_rows"
    root.mkdir(parents=True, exist_ok=True)
    stem = _safe_checkpoint_name(f"{key}__seed_{int(seed)}")
    return root / f"{stem}.json"


def _array_digest(*arrays: np.ndarray) -> str:
    import hashlib

    digest = hashlib.sha256()
    for arr in arrays:
        values = np.asarray(arr, dtype=np.int64)
        digest.update(np.asarray([len(values)], dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()[:24]


def _sample_digest(samples: pd.Index) -> str:
    import hashlib

    digest = hashlib.sha256()
    for sample in samples.astype(str):
        encoded = str(sample).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()[:24]


def _matrix_digest(matrix: np.ndarray) -> str:
    import hashlib

    arr = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()[:24]


def _label_digest(task: str, y: Any) -> str:
    import hashlib

    digest = hashlib.sha256()
    if task == "survival":
        frame = pd.DataFrame(y).loc[:, ["time", "event"]].copy()
        for column in ["time", "event"]:
            values = np.ascontiguousarray(pd.to_numeric(frame[column], errors="coerce").fillna(-1).to_numpy(dtype=np.float64))
            digest.update(column.encode("utf-8"))
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(values.tobytes())
    else:
        arr = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        digest.update(str(task).encode("utf-8"))
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(arr.tobytes())
    return digest.hexdigest()[:24]


def _candidate_checkpoint_manifest(
    *,
    task: str,
    learner: str,
    seed: int,
    primary_metric: str,
    n_classes: int,
    y: Any,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    inner_train_idx: np.ndarray,
    inner_val_idx: np.ndarray,
    flat_candidates: list[dict[str, Any]],
    feature_counts: dict[str, int],
    feature_digests: dict[str, str],
    settings: dict[str, Any],
) -> dict[str, Any]:
    linear_keys = [
        "linear_high_dim_classification_solver",
        "linear_high_dim_classification_solver_min_features",
        "linear_force_sgd_in_feature_set_search",
        "linear_force_sgd_log_loss",
        "linear_sgd_alpha_scale",
        "linear_sgd_average",
        "linear_sgd_early_stopping",
        "linear_sgd_max_iter",
        "linear_sgd_n_iter_no_change",
        "linear_sgd_tol",
        "linear_sgd_validation_fraction",
    ]
    xgb_keys = [
        "tree_method",
        "xgb_early_stopping_rounds",
        "xgb_estimators",
        "xgb_feature_set_random_search_trials",
        "xgb_learning_rate",
        "xgb_max_depth",
        "xgb_max_rounds",
        "xgb_n_jobs",
    ]
    return {
        "version": 2,
        "task": task,
        "learner": learner,
        "seed": int(seed),
        "primary_metric": str(primary_metric),
        "n_classes": int(n_classes),
        "labels_digest": _label_digest(task, y),
        "split_digest": _array_digest(train_idx, test_idx, inner_train_idx, inner_val_idx),
        "feature_counts": {str(name): int(count) for name, count in sorted(feature_counts.items())},
        "feature_digests": {str(name): str(value) for name, value in sorted(feature_digests.items())},
        "candidates": _json_safe(flat_candidates),
        "linear_settings": {key: _json_safe(settings.get(key)) for key in linear_keys if key in settings},
        "xgb_settings": {key: _json_safe(settings.get(key)) for key in xgb_keys if key in settings},
    }


def _load_candidate_checkpoint(path: Path | None, manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("manifest") != manifest:
            return {}
        rows = payload.get("candidate_rows") or []
        out: dict[int, dict[str, Any]] = {}
        for row in rows:
            candidate_id = int(row["candidate_id"])
            normalized = dict(row)
            if normalized.get("score") is None:
                normalized["score"] = float("-inf")
            out[candidate_id] = normalized
        return out
    except Exception:
        return {}


def _write_candidate_checkpoint(
    path: Path | None,
    manifest: dict[str, Any],
    candidate_rows_by_id: dict[int, dict[str, Any]],
    *,
    status: str,
) -> None:
    if path is None:
        return
    payload = {
        "manifest": manifest,
        "status": status,
        "completed_candidates": sorted(int(candidate_id) for candidate_id in candidate_rows_by_id),
        "updated_unix": round(time.time(), 3),
        "candidate_rows": [candidate_rows_by_id[candidate_id] for candidate_id in sorted(candidate_rows_by_id)],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _checkpoint_manifest(
    *,
    task: str,
    benchmark: str,
    endpoint: str,
    representation: str,
    learner: str,
    seed: int,
    primary_metric: str,
    n_splits: int,
    samples: pd.Index,
    n_classes: int,
    y: Any,
    feature_arrays: dict[str, np.ndarray],
    model_candidate_count: int,
) -> dict[str, Any]:
    return {
        "version": 2,
        "task": task,
        "benchmark": benchmark,
        "endpoint": endpoint,
        "representation": representation,
        "learner": learner,
        "seed": int(seed),
        "primary_metric": str(primary_metric),
        "n_splits": int(n_splits),
        "n_samples": int(len(samples)),
        "n_classes": int(n_classes),
        "samples_digest": _sample_digest(samples),
        "labels_digest": _label_digest(task, y),
        "feature_sets": {name: int(arr.shape[1]) for name, arr in sorted(feature_arrays.items())},
        "feature_digests": {name: _matrix_digest(arr) for name, arr in sorted(feature_arrays.items())},
        "model_candidate_count": int(model_candidate_count),
    }


def _load_feature_set_checkpoint(
    paths: tuple[Path, Path] | None,
    manifest: dict[str, Any],
    pred: np.ndarray,
    sample_folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], set[int]]:
    if paths is None:
        return pred, sample_folds, [], set()
    npz_path, json_path = paths
    if not npz_path.exists() or not json_path.exists():
        return pred, sample_folds, [], set()
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload.get("manifest") != manifest:
            return pred, sample_folds, [], set()
        arrays = np.load(npz_path, allow_pickle=False)
        loaded_pred = np.asarray(arrays["pred"], dtype=np.float64)
        loaded_folds = np.asarray(arrays["sample_folds"], dtype=int)
        if loaded_pred.shape != pred.shape or loaded_folds.shape != sample_folds.shape:
            return pred, sample_folds, [], set()
        rows = list(payload.get("fold_rows") or [])
        completed = {int(row["fold"]) for row in rows if "fold" in row}
        return loaded_pred, loaded_folds, rows, completed
    except Exception:
        return pred, sample_folds, [], set()


def _write_feature_set_checkpoint(
    paths: tuple[Path, Path] | None,
    manifest: dict[str, Any],
    pred: np.ndarray,
    sample_folds: np.ndarray,
    fold_rows: list[dict[str, Any]],
    *,
    status: str,
) -> None:
    if paths is None:
        return
    npz_path, json_path = paths
    tmp_npz = npz_path.with_suffix(".npz.tmp")
    tmp_json = json_path.with_suffix(".json.tmp")
    with tmp_npz.open("wb") as handle:
        np.savez_compressed(handle, pred=pred, sample_folds=sample_folds)
    tmp_npz.replace(npz_path)
    payload = {
        "manifest": manifest,
        "status": status,
        "completed_folds": sorted({int(row["fold"]) for row in fold_rows if "fold" in row}),
        "updated_unix": round(time.time(), 3),
        "fold_rows": fold_rows,
    }
    tmp_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_json.replace(json_path)


def evaluate_nested_feature_set_oof(
    *,
    samples: Iterable[str],
    feature_sets: dict[str, np.ndarray],
    labels: pd.Series | pd.DataFrame,
    task: str,
    benchmark: str,
    endpoint: str,
    representation: str,
    learner: str,
    groups: pd.Series | np.ndarray | None = None,
    settings: dict[str, Any] | None = None,
    n_splits: int = 5,
    seed: int | None = None,
) -> NestedOOFResult:
    settings = dict(settings or {})
    grouped_task = task == "multiclass_grouped"
    task = "multiclass" if grouped_task else task
    samples = pd.Index([str(value) for value in samples], name="sample")
    if not feature_sets:
        raise ValueError("At least one candidate feature set is required")
    feature_arrays = {
        str(name): np.nan_to_num(np.asarray(matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        for name, matrix in feature_sets.items()
    }
    row_counts = {name: arr.shape[0] for name, arr in feature_arrays.items()}
    if len(set(row_counts.values())) != 1 or next(iter(row_counts.values())) != len(samples):
        raise ValueError(f"Candidate feature-set row counts do not match samples: {row_counts}, samples={len(samples)}")
    y, classes = encode_labels(labels, task)
    group_arr = None
    if groups is not None:
        group_arr = pd.Series(groups).loc[samples].to_numpy() if isinstance(groups, pd.Series) else np.asarray(groups)
    seed = int(seed if seed is not None else stable_seed(benchmark, endpoint, representation, learner, "feature_sets"))
    n_classes = 2 if task == "binary" else len(classes)
    primary_metric = _primary_metric(task, endpoint, settings)
    splits = outer_splits(task, y, groups=group_arr, seed=seed, n_splits=n_splits)

    pred = np.zeros(len(samples), dtype=np.float64) if task in {"regression", "survival"} else np.zeros((len(samples), n_classes), dtype=np.float64)
    sample_folds = np.full(len(samples), -1, dtype=int)
    fold_rows: list[dict[str, Any]] = []
    selected_counts: list[int] = []
    selected_sets: list[str] = []
    max_feature_count = max(int(arr.shape[1]) for arr in feature_arrays.values())
    if learner == "xgboost":
        model_candidate_count = max(1, int(settings.get("xgb_feature_set_random_search_trials", settings.get("random_search_trials", settings.get("optuna_trials", 10)))))
    else:
        model_candidate_count = len(_linear_candidates(task, seed, _linear_candidate_limit(task, max_feature_count, settings)))
    checkpoint_manifest = _checkpoint_manifest(
        task=task,
        benchmark=benchmark,
        endpoint=endpoint,
        representation=representation,
        learner=learner,
        seed=seed,
        primary_metric=primary_metric,
        n_splits=n_splits,
        samples=samples,
        n_classes=n_classes,
        y=y,
        feature_arrays=feature_arrays,
        model_candidate_count=model_candidate_count,
    )
    checkpoint_paths = _feature_set_checkpoint_paths(settings)
    if bool(settings.get("fold_checkpoint_resume", True)):
        pred, sample_folds, fold_rows, completed_folds = _load_feature_set_checkpoint(checkpoint_paths, checkpoint_manifest, pred, sample_folds)
        selected_counts = [int(row["selected_feature_count"]) for row in fold_rows if pd.notna(row.get("selected_feature_count"))]
        selected_sets = [str(row["selected_feature_set"]) for row in fold_rows if row.get("selected_feature_set")]
    else:
        completed_folds = set()
    progress_label = str(settings.get("fold_progress_label") or f"{endpoint} {representation} {learner}")
    if completed_folds:
        print(f"[nested_oof] resume {progress_label}: completed folds {sorted(completed_folds)}", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for fold, train_idx, test_idx in splits:
            if int(fold) in completed_folds and np.all(sample_folds[test_idx] == int(fold)):
                print(f"[nested_oof] reuse fold {fold}/{len(splits)} {progress_label}", flush=True)
                continue
            fold_start = time.time()
            print(
                f"[nested_oof] start fold {fold}/{len(splits)} {progress_label}; "
                f"feature_sets={len(feature_arrays)} model_candidates={model_candidate_count}",
                flush=True,
            )
            fold_pred, diag = _search_and_predict_feature_set_fold(
                task=task,
                learner=learner,
                x_by_feature_set=feature_arrays,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                groups=group_arr,
                seed=seed + fold * 1009,
                n_classes=n_classes,
                primary_metric=primary_metric,
                settings=settings,
            )
            pred[test_idx] = fold_pred
            assign_oof_fold(sample_folds, test_idx, int(fold))
            selected_counts.append(int(diag["selected_feature_count"]))
            selected_sets.append(str(diag["selected_feature_set"]))
            fold_score = _candidate_score(task, _slice_y(task, y, test_idx), fold_pred, n_classes, primary_metric)
            fold_row = {
                "repeat": 1,
                "fold": int(fold),
                "metric": primary_metric,
                "score": float(fold_score) if np.isfinite(fold_score) else float("nan"),
                "n_test": int(len(test_idx)),
                "inner_train_n": diag["inner_train_n"],
                "inner_val_n": diag["inner_val_n"],
                "outer_train_n": diag["outer_train_n"],
                "outer_test_n": diag["outer_test_n"],
                "selected_inner_score": diag["selected_score"],
                "selected_params_json": _safe_json(diag["selected_params"]),
                "selected_m_star": diag["selected_m_star"],
                "selected_feature_set": diag["selected_feature_set"],
                "selected_feature_count": diag["selected_feature_count"],
                "candidate_feature_set_count": diag["candidate_feature_set_count"],
                "candidate_feature_sets_json": _safe_json(diag["candidate_feature_sets"]),
                "candidate_count": diag["candidate_count"],
                "candidate_results_json": _safe_json(diag["candidate_results"]),
                "split_strategy": _split_strategy_label(task, group_arr is not None, len(splits)),
                "outer_overlap_n": diag["outer_overlap_n"],
                "inner_overlap_n": diag["inner_overlap_n"],
                "inner_test_overlap_n": diag["inner_test_overlap_n"],
                "group_overlap_n": diag["group_overlap_n"],
                "inner_group_overlap_n": diag["inner_group_overlap_n"],
            }
            fold_rows.append(fold_row)
            _write_feature_set_checkpoint(checkpoint_paths, checkpoint_manifest, pred, sample_folds, fold_rows, status="running")
            print(
                f"[nested_oof] done fold {fold}/{len(splits)} {progress_label}; "
                f"score={fold_row['score']:.6f} selected={diag['selected_feature_set']} "
                f"features={diag['selected_feature_count']} seconds={time.time() - fold_start:.1f}",
                flush=True,
            )
    assert_all_oof_assigned(sample_folds, samples)
    _write_feature_set_checkpoint(checkpoint_paths, checkpoint_manifest, pred, sample_folds, fold_rows, status="completed")

    summary, pred_frame = _summarize_predictions(
        task=task,
        grouped_task=grouped_task,
        y=y,
        pred=pred,
        samples=samples,
        benchmark=benchmark,
        endpoint=endpoint,
        representation=representation,
        learner=learner,
        classes=classes,
        n_features=max_feature_count,
        n_folds=len(splits),
        split_strategy=_split_strategy_label(task, group_arr is not None, len(splits)),
        linear_solver=_solver_label(learner, task, max_feature_count, settings),
        primary_metric=primary_metric,
    )
    pred_frame["fold"] = sample_folds
    if selected_counts:
        summary["selected_feature_count_min"] = int(min(selected_counts))
        summary["selected_feature_count_median"] = float(np.median(selected_counts))
        summary["selected_feature_count_max"] = int(max(selected_counts))
        summary["selected_feature_sets"] = ";".join(sorted(set(selected_sets)))
    summary["candidate_feature_set_count"] = int(len(feature_arrays))
    summary["candidate_feature_sets"] = ";".join(sorted(feature_arrays))
    summary["feature_selection"] = "inner_validation_selects_predeclared_bio_v4_feature_block_with_model_hyperparameters"
    fold_frame = pd.DataFrame(fold_rows)
    if not fold_frame.empty:
        fold_frame.insert(0, "learner", learner)
        fold_frame.insert(0, "representation", representation)
        fold_frame.insert(0, "endpoint", endpoint)
        fold_frame.insert(0, "benchmark", benchmark)
    return NestedOOFResult(summary, pred_frame, fold_frame)


def _primary_metric(task: str, endpoint: str | None = None, settings: dict[str, Any] | None = None) -> str:
    endpoint_name = str(endpoint or "")
    settings = settings or {}
    endpoint_metrics = settings.get("primary_metric_by_endpoint") or {}
    configured = str(endpoint_metrics.get(endpoint_name) or settings.get("primary_metric") or "").strip().lower()
    if configured:
        return configured
    if task == "regression":
        return "spearman"
    if task == "binary":
        return "auroc"
    if task == "survival":
        return "c_index"
    if endpoint_name == "cancer_type_top20":
        return "balanced_accuracy"
    return "macro_auroc"


def _split_strategy_label(task: str, grouped: bool, n_folds: int) -> str:
    if grouped:
        return f"{int(n_folds)} outer StratifiedGroupKFold; inner grouped validation split"
    if task == "regression":
        return f"{int(n_folds)} outer KFold; inner validation split"
    if task == "survival":
        return f"{int(n_folds)} outer event-stratified folds; inner event-stratified validation split"
    return f"{int(n_folds)} outer stratified folds; inner stratified validation split"


def _solver_label(learner: str, task: str, n_features: int, settings: dict[str, Any]) -> str:
    if learner == "linear" and task not in {"regression", "survival"}:
        if _use_sgd_log_loss(task, n_features, settings):
            return "sgd_log_loss_elasticnet_v1"
        return "logistic_saga_elasticnet_nested_v1"
    if learner == "linear" and task == "regression":
        return "elastic_net_coordinate_descent_nested_v1"
    if learner == "cox_ph" or task == "survival":
        high_dim = int(n_features) >= int(settings.get("cox_high_dim_feature_threshold", 500))
        if high_dim and str(settings.get("cox_high_dim_backend", "fast_breslow")) == "fast_breslow":
            return "fast_breslow_elasticnet_cox_nested_v1"
        if high_dim:
            return "lifelines_penalized_cox_ph_nested_v4_conditioned_variance_highdim_stepcap"
        return "lifelines_penalized_cox_ph_nested_v3_conditioned_variance_stepcap"
    return ""


def _summarize_predictions(
    *,
    task: str,
    grouped_task: bool,
    y: Any,
    pred: np.ndarray,
    samples: pd.Index,
    benchmark: str,
    endpoint: str,
    representation: str,
    learner: str,
    classes: np.ndarray,
    n_features: int,
    n_folds: int,
    split_strategy: str,
    linear_solver: str,
    primary_metric: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    metric = str(primary_metric or _primary_metric(task, endpoint)).lower()
    balanced = accuracy = macro_f1 = auroc = macro_auroc = micro_auroc = kappa = top3 = top5 = auprc = mae = r2 = float("nan")
    if task == "regression":
        score = float(spearmanr(y, pred)[0])
        mae = float(mean_absolute_error(y, pred))
        r2 = float(r2_score(y, pred))
        pred_frame = pd.DataFrame({"sample": samples.astype(str), "true_value": np.asarray(y, dtype=float), "pred_value": pred.astype(float), "repeat": 1})
    elif task == "survival":
        y_df = pd.DataFrame(y)
        score = harrell_c_index(y_df["time"], y_df["event"], pred)
        pred_frame = pd.DataFrame({"sample": samples.astype(str), "time": y_df["time"].to_numpy(dtype=float), "event": y_df["event"].to_numpy(dtype=int), "risk_score": pred.astype(float), "repeat": 1})
    elif task == "binary":
        predicted = (pred[:, 1] >= 0.5).astype(int)
        auroc = float(roc_auc_score(y, pred[:, 1]))
        balanced = float(balanced_accuracy_score(y, predicted))
        accuracy = float(accuracy_score(y, predicted))
        macro_f1 = float(f1_score(y, predicted, average="macro", zero_division=0))
        auprc = float(average_precision_score(y, pred[:, 1]))
        kappa = float(cohen_kappa_score(y, predicted))
        score = _score_predictions(task, y, pred, 2, metric)
        pred_frame = pd.DataFrame({"sample": samples.astype(str), "true_value": np.asarray(y, dtype=int), "pred_class_1": pred[:, 1], "repeat": 1})
        pred_frame["pred_class_0"] = pred[:, 0]
    else:
        n_classes = len(classes)
        macro_auroc = safe_macro_auroc(y, pred, n_classes)
        micro_auroc = safe_micro_auroc(y, pred, n_classes)
        predicted = np.argmax(pred, axis=1)
        balanced = float(balanced_accuracy_score(y, predicted))
        accuracy = float(accuracy_score(y, predicted))
        macro_f1 = float(f1_score(y, predicted, average="macro", zero_division=0))
        kappa = float(cohen_kappa_score(y, predicted))
        top3 = topk_accuracy(y, pred, 3)
        top5 = topk_accuracy(y, pred, 5)
        score = _score_predictions(task, y, pred, n_classes, metric)
        pred_frame = pd.DataFrame({"sample": samples.astype(str), "true_value": np.asarray(y, dtype=int), "predicted_class": predicted.astype(int), "repeat": 1})
        for i, class_label in enumerate(classes):
            pred_frame[f"pred_class_{i}"] = pred[:, i]
            pred_frame[f"class_label_{i}"] = str(class_label)
    summary = {
        "benchmark": benchmark,
        "endpoint": endpoint,
        "task": "multiclass_grouped" if grouped_task else task,
        "representation": representation,
        "learner": learner,
        "metric": metric,
        "score": float(score),
        "auroc": auroc if task == "binary" else macro_auroc,
        "macro_auroc": macro_auroc,
        "micro_auroc": micro_auroc,
        "c_index": float(score) if metric == "c_index" else float("nan"),
        "auprc": auprc,
        "balanced_accuracy": balanced,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "cohen_kappa": kappa,
        "top3_accuracy": top3,
        "top5_accuracy": top5,
        "mae": mae,
        "r2": r2,
        "n_samples": int(len(samples)),
        "n_classes": int(pd.Series(y["event"] if task == "survival" else y).nunique()),
        "n_features": int(n_features),
        "n_folds": int(n_folds),
        "folds": int(n_folds),
        "repeats": 1,
        "oof_aggregate": True,
        "primary_metric_scope": "pooled_global_oof",
        "split_strategy": split_strategy,
        "tuning": "outer_train_inner_validation_randomized_search",
        "linear_solver": linear_solver,
    }
    pred_frame.insert(0, "learner", learner)
    pred_frame.insert(0, "representation", representation)
    pred_frame.insert(0, "endpoint", endpoint)
    pred_frame.insert(0, "benchmark", benchmark)
    return summary, pred_frame
