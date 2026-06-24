"""Rerun survival CoxNet rows with the active scikit-survival workflow.

This script is intentionally scoped to survival endpoints so a CoxNet-only refresh
does not overwrite unrelated non-survival benchmark rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runners.run_main_manuscript_complete_panel import (  # noqa: E402
    BIO_MAF_V4_MODEL_IDS,
    EXPERIMENT_ID,
    MAF_REPRESENTATION,
    _load_mc3_endpoints,
    _load_mc3_maf_features,
    _load_mc3_standard_features,
    _main_model_family,
    _settings,
)
from utils.checkpointing import atomic_write_csv, atomic_write_json  # noqa: E402
from utils.config import load_yaml, resolve_paths_map  # noqa: E402
from utils.feature_evaluation import evaluate_features  # noqa: E402
from utils.nested_oof import evaluate_nested_feature_set_oof, stable_seed  # noqa: E402
from utils.runner_support import RunnerContext, ensure_output_dirs, sanitize_frame  # noqa: E402


REPRESENTATION_FAMILY = {
    "burden_only": "burden_only",
    "standard_sbs96_id83": "signatures_only",
    "standard_sbs96_dbs78_id83": "signatures_only",
    "MAF_stack_only": "MAF_stack_only",
    "signatures_plus_MAF_stack": "signatures_plus_MAF_stack",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment_settings.strict_no_leakage.yaml")
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--endpoint", default="OS")
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def _replace_rows(frame: pd.DataFrame, rows: Iterable[pd.DataFrame | dict], key_columns: list[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for row in rows:
        pieces.append(row if isinstance(row, pd.DataFrame) else pd.DataFrame([row]))
    new_frame = pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()
    if frame.empty:
        return new_frame
    if new_frame.empty:
        return frame
    for column in key_columns:
        if column not in frame.columns:
            frame[column] = ""
        if column not in new_frame.columns:
            new_frame[column] = ""
    old_keys = frame.loc[:, key_columns].astype(str).agg("\0".join, axis=1)
    new_keys = set(new_frame.loc[:, key_columns].astype(str).agg("\0".join, axis=1))
    kept = frame.loc[~old_keys.isin(new_keys)].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return pd.concat([kept, new_frame], ignore_index=True, sort=False)


def _plot_data_from_endpoint_results(endpoint_results: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    rows = endpoint_results[
        endpoint_results.get("endpoint", pd.Series(dtype=str)).astype(str).eq(str(endpoint))
        & endpoint_results.get("learner", pd.Series(dtype=str)).astype(str).eq("cox_ph")
        & endpoint_results.get("status", pd.Series("measured", index=endpoint_results.index)).astype(str).eq("measured")
    ].copy()
    if rows.empty:
        raise RuntimeError(f"No measured Cox rows found for endpoint={endpoint}")
    rows["representation_family"] = rows["representation"].astype(str).map(REPRESENTATION_FAMILY).fillna(rows["representation"].astype(str))
    rows["model_family"] = "cox_ph"
    rows["primary_score"] = pd.to_numeric(rows.get("score"), errors="coerce")
    if "c_index" in rows.columns:
        rows["primary_score"] = rows["primary_score"].fillna(pd.to_numeric(rows["c_index"], errors="coerce"))
    rows["metric"] = "c_index"
    order = {name: idx for idx, name in enumerate(["burden_only", "signatures_only", "MAF_stack_only", "signatures_plus_MAF_stack"])}
    rows["_order"] = rows["representation_family"].map(order).fillna(99)
    keep = [
        "endpoint",
        "task",
        "representation",
        "representation_family",
        "learner",
        "model_family",
        "metric",
        "primary_score",
        "score",
        "c_index",
        "n_samples",
        "n_features",
        "linear_solver",
        "selected_feature_sets",
        "selected_feature_count_median",
        "status",
        "atlas_status_note",
    ]
    return rows.sort_values(["_order", "representation"]).loc[:, [col for col in keep if col in rows.columns]].reset_index(drop=True)


def main() -> None:
    warnings.filterwarnings("ignore", message="all coefficients are zero.*", category=UserWarning)
    args = parse_args()
    settings = load_yaml(args.config)
    settings["_run_id"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = resolve_paths_map(load_yaml(args.paths))
    ctx = RunnerContext(settings=settings, paths=paths)
    ensure_output_dirs(ctx)
    runner_settings = _settings(ctx)
    endpoint_name = str(args.endpoint)
    started = time.time()

    endpoints, audit = _load_mc3_endpoints(ctx, [], [endpoint_name], runner_settings)
    if audit.empty:
        raise RuntimeError("Survival endpoint audit is empty")
    endpoint_matches = [endpoint for endpoint in endpoints if endpoint.name == endpoint_name and endpoint.task == "survival"]
    if not endpoint_matches:
        raise RuntimeError(f"Survival endpoint {endpoint_name!r} was not available: {audit.to_dict('records')}")
    endpoint = endpoint_matches[0]

    burden, standard, standard_cache_key = _load_mc3_standard_features(ctx)
    feature_specs: list[tuple[str, pd.DataFrame, str, str]] = [
        ("burden_only", burden, standard_cache_key, "MC3 cached burden-only matrix"),
        ("standard_sbs96_id83", standard, standard_cache_key, "MC3 cached SBS96+ID83 signature matrix"),
    ]
    for model_id in BIO_MAF_V4_MODEL_IDS:
        frame, cache_key = _load_mc3_maf_features(ctx, model_id, runner_settings)
        representation = MAF_REPRESENTATION[model_id]
        counts = frame.attrs.get("candidate_feature_set_counts") or {}
        construction = (
            f"MC3/HRD {model_id}; Bio MAF v4 with fixed external resources and nested inner-loop "
            f"feature-block selection ({json.dumps(counts, sort_keys=True)})"
        )
        feature_specs.append((representation, frame, cache_key, construction))

    new_endpoint_rows: list[dict[str, object]] = []
    new_prediction_frames: list[pd.DataFrame] = []
    new_fold_frames: list[pd.DataFrame] = []
    for representation, features, cache_key, construction in feature_specs:
        slot_start = time.time()
        candidate_feature_sets = features.attrs.get("candidate_feature_sets")
        print(f"[survival_coxnet] evaluating {endpoint_name} {representation} cox_ph", flush=True)
        if candidate_feature_sets:
            common = endpoint.labels.index.astype(str).intersection(features.index.astype(str))
            labels = endpoint.labels.loc[common]
            groups = endpoint.groups.loc[common] if endpoint.groups is not None else None
            nested_settings = {
                **runner_settings,
                "tree_method": str(ctx.tree_method),
                "xgb_n_jobs": int(ctx.xgb_n_jobs),
                "xgb_estimators": int(runner_settings.get("xgb_estimators", ctx.xgb_estimators)),
                "optuna_trials": int(runner_settings.get("random_search_trials", runner_settings.get("optuna_trials", ctx.optuna_trials))),
                "fold_checkpoint_enabled": bool(runner_settings.get("fold_checkpoint_enabled", True)),
                "fold_checkpoint_resume": bool(runner_settings.get("fold_checkpoint_resume", True)),
                "fold_checkpoint_dir": str(ctx.tables_dir / f"{EXPERIMENT_ID}_nested_fold_checkpoints"),
                "fold_checkpoint_key": f"{endpoint.name}__{representation}__cox_ph__{cache_key}",
                "fold_progress_label": f"{EXPERIMENT_ID} {endpoint.name} {representation} CoxNet",
            }
            matrices = {
                str(name): features.loc[common, list(cols)].fillna(0.0).to_numpy(dtype=np.float32)
                for name, cols in dict(candidate_feature_sets).items()
            }
            result = evaluate_nested_feature_set_oof(
                samples=common.astype(str),
                feature_sets=matrices,
                labels=labels,
                task=endpoint.task,
                benchmark=endpoint.benchmark,
                endpoint=endpoint.name,
                representation=representation,
                learner="cox_ph",
                groups=groups,
                settings=nested_settings,
                n_splits=int(ctx.cv_folds),
                seed=stable_seed(endpoint.benchmark, endpoint.name, "cox_ph", representation, cache_key, "bio_v4_feature_sets"),
            )
            row = dict(result.summary)
            pred_frame = result.predictions
            fold_frame = result.fold_diagnostics
        else:
            row, pred_frame, fold_frame = evaluate_features(endpoint, features, representation, "cox_ph", ctx, runner_settings)

        row.update(
            {
                "experiment_id": EXPERIMENT_ID,
                "model_family": _main_model_family("cox_ph"),
                "cache_key": cache_key,
                "oof_prediction_file": f"{EXPERIMENT_ID}_oof_predictions.csv",
                "fold_metrics_file": f"{EXPERIMENT_ID}_fold_metrics.csv",
                "optuna_trials_completed": 0,
                "run_id": settings["_run_id"],
                "status": "measured",
                "atlas_status_note": construction,
                "runtime_seconds": round(time.time() - slot_start, 3),
            }
        )
        new_endpoint_rows.append(row)
        new_prediction_frames.append(pred_frame)
        if not fold_frame.empty:
            new_fold_frames.append(fold_frame)
        print(
            f"[survival_coxnet] completed {endpoint_name} {representation} "
            f"c_index={float(row.get('score', np.nan)):.6g} seconds={time.time() - slot_start:.1f}",
            flush=True,
        )

    endpoint_results_path = ctx.tables_dir / f"{EXPERIMENT_ID}_endpoint_results.csv"
    fold_metrics_path = ctx.tables_dir / f"{EXPERIMENT_ID}_fold_metrics.csv"
    oof_predictions_path = ctx.tables_dir / f"{EXPERIMENT_ID}_oof_predictions.csv"
    endpoint_results = _replace_rows(_read_csv(endpoint_results_path), new_endpoint_rows, ["endpoint", "representation", "learner"])
    endpoint_results = endpoint_results.drop_duplicates(["endpoint", "representation", "learner"], keep="last")
    atomic_write_csv(sanitize_frame(endpoint_results), endpoint_results_path, index=False)

    if new_fold_frames:
        fold_metrics = _replace_rows(
            _read_csv(fold_metrics_path),
            new_fold_frames,
            ["benchmark", "endpoint", "representation", "learner", "repeat", "fold"],
        )
        atomic_write_csv(sanitize_frame(fold_metrics), fold_metrics_path, index=False)

    if new_prediction_frames:
        oof_predictions = _replace_rows(
            _read_csv(oof_predictions_path),
            new_prediction_frames,
            ["benchmark", "endpoint", "representation", "learner", "sample"],
        )
        atomic_write_csv(sanitize_frame(oof_predictions), oof_predictions_path, index=False)

    plot_dir = ROOT / "results" / "manuscript" / "plot_data"
    plot_dir.mkdir(parents=True, exist_ok=True)
    survival_plot = _plot_data_from_endpoint_results(endpoint_results, endpoint_name)
    atomic_write_csv(sanitize_frame(survival_plot), plot_dir / "figure_5_overall_survival_cox.csv", index=False)
    summary_path = ctx.tables_dir / f"{EXPERIMENT_ID}_survival_cox_summary.csv"
    atomic_write_csv(
        sanitize_frame(
            survival_plot.loc[
                :,
                [col for col in ["endpoint", "representation", "representation_family", "primary_score", "n_samples", "n_features", "linear_solver", "selected_feature_sets"] if col in survival_plot.columns],
            ]
        ),
        summary_path,
        index=False,
    )
    manifest = {
        "completed_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "endpoint": endpoint_name,
        "elapsed_seconds": round(time.time() - started, 3),
        "rows": new_endpoint_rows,
        "endpoint_results": str(endpoint_results_path),
        "fold_metrics": str(fold_metrics_path),
        "oof_predictions": str(oof_predictions_path),
        "plot_data": str(plot_dir / "figure_5_overall_survival_cox.csv"),
        "summary": str(summary_path),
    }
    atomic_write_json(ctx.logs_dir / f"{EXPERIMENT_ID}_survival_cox_rerun_manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
