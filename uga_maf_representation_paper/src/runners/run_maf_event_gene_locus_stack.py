"""Runner for the exploratory MAF event-gene/locus feature stack."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.runner_support import (
    RunnerContext,
    add_source_file_column,
    copy_artifacts,
    glob_many,
    prepare_legacy_workspace,
    restore_legacy_feature_snapshot,
    run_python,
    snapshot_legacy_feature_outputs,
    write_endpoint_results,
    write_summary_csv,
)


EXPERIMENT_ID = "maf_event_gene_locus"
EXPERIMENT_REL = Path("cgr_validation_results/research/experiments/exploratory/2026_05_16_xgboost_maf_event_gene_coordinate_search")


def _read_frame(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def run(ctx: RunnerContext) -> None:
    workspace = prepare_legacy_workspace(ctx, require_inputs=not ctx.dry_run)
    exp_root = workspace / EXPERIMENT_REL
    restore_status = restore_legacy_feature_snapshot(ctx, EXPERIMENT_ID, exp_root)
    code_dir = exp_root / "code"
    settings = (ctx.settings.get("experiments") or {}).get(EXPERIMENT_ID) or {}
    log_path = ctx.logs_dir / f"{EXPERIMENT_ID}.log"

    discovery_args = [
        "--tree-method",
        ctx.tree_method,
        "--folds",
        str(settings.get("discovery_folds", 3)),
        "--xgb-estimators",
        str(settings.get("discovery_estimators", 80)),
        "--confirmation-folds",
        str(settings.get("confirmation_folds", 5)),
        "--confirmation-repeats",
        str(settings.get("confirmation_repeats", 3)),
        "--confirmation-estimators",
        str(settings.get("confirmation_estimators", 160)),
        "--bootstrap",
        str(settings.get("bootstrap", 1000)),
        "--max-candidates",
        str(settings.get("max_candidates", 0)),
        "--skip-ledger",
    ]
    full_panel_args = [
        "--model-id",
        str(settings.get("model_id", "id_plus_best_gene_locus_multiscale_stack")),
        "--tree-method",
        ctx.tree_method,
        "--folds",
        str(settings.get("confirmation_folds", 5)),
        "--repeats",
        str(settings.get("confirmation_repeats", 3)),
        "--xgb-estimators",
        str(settings.get("confirmation_estimators", 160)),
        "--bootstrap",
        str(settings.get("bootstrap", 1000)),
    ]

    model_ids = list(settings.get("model_ids") or [settings.get("model_id", "id_plus_best_gene_locus_multiscale_stack")])
    summary_rows = [restore_status]
    if settings.get("run_discovery", True):
        summary_rows.append({"kind": "script", "name": "run_xgboost_maf_event_gene_coordinate_search.py", "status": "planned" if ctx.dry_run else "executed"})
        run_python(code_dir / "run_xgboost_maf_event_gene_coordinate_search.py", cwd=exp_root, log_path=log_path, args=discovery_args, dry_run=ctx.dry_run)
    else:
        summary_rows.append({"kind": "script", "name": "run_xgboost_maf_event_gene_coordinate_search.py", "status": "skipped", "reason": "fresh manuscript config uses frozen candidates"})
    for model_id in model_ids:
        model_args = list(full_panel_args)
        model_args[1] = str(model_id)
        summary_rows.append({"kind": "script", "name": "run_full_panel_confirmation.py", "model_id": str(model_id), "status": "planned" if ctx.dry_run else "executed"})
        run_python(code_dir / "run_full_panel_confirmation.py", cwd=exp_root, log_path=log_path, args=model_args, dry_run=ctx.dry_run)
    if ctx.dry_run:
        write_summary_csv(summary_rows, ctx.tables_dir / f"{EXPERIMENT_ID}_summary.csv")
        return

    table_sources = glob_many(
        exp_root,
        [
            "tables/**/*.html",
            "tables/**/*.csv",
            "data/**/*results*.csv",
            "data/**/*summary*.csv",
            "data/**/*leaderboard*.csv",
            "data/**/*audit*.csv",
            "data/**/*manifest*.csv",
            "data/**/*metadata*.json",
        ],
    )
    figure_sources = glob_many(exp_root, ["figures/**/*", "data/**/*figure*.csv"])

    summary_rows.extend(copy_artifacts(table_sources, ctx.tables_dir, prefix=EXPERIMENT_ID, source_root=exp_root, dry_run=ctx.dry_run))
    summary_rows.extend(copy_artifacts(figure_sources, ctx.figures_dir, prefix=EXPERIMENT_ID, source_root=exp_root, dry_run=ctx.dry_run))
    if not ctx.dry_run:
        summary_rows.append(snapshot_legacy_feature_outputs(ctx, EXPERIMENT_ID, exp_root))

    frames: list[pd.DataFrame] = []
    for path in table_sources:
        if path.suffix not in {".csv", ".gz"}:
            continue
        frame = _read_frame(path)
        if frame is not None:
            frames.append(add_source_file_column(frame, str(path.relative_to(exp_root)), EXPERIMENT_ID))

    write_endpoint_results(frames, ctx.tables_dir / f"{EXPERIMENT_ID}_endpoint_results.csv")
    write_summary_csv(summary_rows, ctx.tables_dir / f"{EXPERIMENT_ID}_summary.csv")
