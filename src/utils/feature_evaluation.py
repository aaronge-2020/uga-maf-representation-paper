"""Shared nested out-of-fold evaluation for manuscript feature matrices."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from utils.endpoint_data import EndpointData
from utils.nested_oof import evaluate_nested_oof
from utils.runner_support import RunnerContext


RANDOM_SEED = 20260518


def stable_seed(*parts: object) -> int:
    """Preserve the historical manuscript seed schedule."""
    digest = hashlib.sha256("||".join(str(p) for p in parts).encode("utf-8")).digest()
    return RANDOM_SEED + int.from_bytes(digest[:4], "big") % 100_000


def evaluate_features(
    endpoint: EndpointData,
    features: pd.DataFrame,
    representation: str,
    learner: str,
    ctx: RunnerContext,
    settings: dict,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    common = endpoint.labels.index.astype(str).intersection(features.index.astype(str))
    labels = endpoint.labels.loc[common]
    x = features.loc[common].fillna(0.0).to_numpy(dtype=np.float32)
    groups = endpoint.groups.loc[common] if endpoint.groups is not None else None
    nested_settings = {
        **settings,
        "tree_method": str(ctx.tree_method),
        "xgb_n_jobs": int(ctx.xgb_n_jobs),
        "xgb_estimators": int(settings.get("xgb_estimators", ctx.xgb_estimators)),
        "optuna_trials": int(settings.get("random_search_trials", settings.get("optuna_trials", ctx.optuna_trials))),
    }
    result = evaluate_nested_oof(
        samples=common.astype(str),
        x=x,
        labels=labels,
        task=endpoint.task,
        benchmark=endpoint.benchmark,
        endpoint=endpoint.name,
        representation=representation,
        learner=learner,
        groups=groups,
        settings=nested_settings,
        n_splits=int(ctx.cv_folds),
        seed=stable_seed(endpoint.benchmark, endpoint.name, learner, representation),
    )
    row = dict(result.summary)
    row["xgb_tree_method"] = str(ctx.tree_method) if learner == "xgboost" else ""
    return row, result.predictions, result.fold_diagnostics
