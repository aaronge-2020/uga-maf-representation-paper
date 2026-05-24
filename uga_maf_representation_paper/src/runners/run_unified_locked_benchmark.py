"""Runner for the locked unified UGA benchmark."""

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


EXPERIMENT_ID = "unified_locked"
EXPERIMENT_REL = Path("cgr_validation_results/research/experiments/supporting/2026_05_14_unified_uga_locked_manuscript_benchmark")
DEFAULT_SCRIPTS = [
    "run_locked_hrd_cross_validation.py",
    "run_locked_kucab_low_burden_benchmark.py",
    "run_locked_mc3_luad_kmt2c_validation.py",
    "run_locked_mc3_clinical_endpoints.py",
    "run_locked_pcawg_signature_attribution.py",
    "run_locked_tcga_signature_endpoint_prediction.py",
]


def _read_frame(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        if path.suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        return pd.read_csv(path)
    except Exception:
        return None


def run(ctx: RunnerContext) -> None:
    workspace = prepare_legacy_workspace(ctx, require_inputs=not ctx.dry_run)
    exp_root = workspace / EXPERIMENT_REL
    restore_status = restore_legacy_feature_snapshot(ctx, EXPERIMENT_ID, exp_root)
    code_dir = exp_root / "code"
    settings = (ctx.settings.get("experiments") or {}).get(EXPERIMENT_ID) or {}
    scripts = list(settings.get("scripts") or DEFAULT_SCRIPTS)
    log_path = ctx.logs_dir / f"{EXPERIMENT_ID}.log"

    summary_rows: list[dict] = [restore_status]
    for script_name in scripts:
        script = code_dir / script_name
        summary_rows.append({"kind": "script", "name": script_name, "status": "planned" if ctx.dry_run else "executed"})
        run_python(script, cwd=exp_root, log_path=log_path, dry_run=ctx.dry_run)
    if ctx.dry_run:
        write_summary_csv(summary_rows, ctx.tables_dir / f"{EXPERIMENT_ID}_summary.csv")
        return

    table_sources = glob_many(
        exp_root,
        [
            "tables/**/*.csv",
            "tables/**/*.html",
            "data/**/*metrics*.csv",
            "data/**/*metrics*.tsv",
            "data/**/*summary*.csv",
            "data/**/*endpoint*.csv",
            "data/**/*tests*.csv",
            "data/**/*tests*.tsv",
        ],
    )
    figure_sources = glob_many(exp_root, ["figures/**/*", "data/**/*figure*.csv"])
    oof_sources = glob_many(exp_root, ["data/**/*oof*predictions*.csv*", "data/**/*predictions*.csv*"])

    summary_rows.extend(
        copy_artifacts(table_sources + oof_sources, ctx.tables_dir, prefix=EXPERIMENT_ID, source_root=exp_root, dry_run=ctx.dry_run)
    )
    summary_rows.extend(copy_artifacts(figure_sources, ctx.figures_dir, prefix=EXPERIMENT_ID, source_root=exp_root, dry_run=ctx.dry_run))
    if not ctx.dry_run:
        summary_rows.append(snapshot_legacy_feature_outputs(ctx, EXPERIMENT_ID, exp_root))

    frames: list[pd.DataFrame] = []
    for path in table_sources:
        if "predictions" in path.name or path.suffix == ".html":
            continue
        frame = _read_frame(path)
        if frame is not None:
            frames.append(add_source_file_column(frame, str(path.relative_to(exp_root)), EXPERIMENT_ID))

    write_endpoint_results(frames, ctx.tables_dir / f"{EXPERIMENT_ID}_endpoint_results.csv")
    write_summary_csv(summary_rows, ctx.tables_dir / f"{EXPERIMENT_ID}_summary.csv")
