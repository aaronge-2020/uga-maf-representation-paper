"""Runner for EXP024 shared-space and EXP025 payload-absence mechanics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.config import BUNDLE_ROOT
from utils.runner_support import (
    RunnerContext,
    add_source_file_column,
    copy_artifacts,
    glob_many,
    prepare_legacy_workspace,
    run_python,
    write_endpoint_results,
    write_summary_csv,
)


EXPERIMENT_ID = "mechanistic_representation"
EXP024_SCRIPT = Path("cgr_validation_results/research/scripts/EXP024_shared_space/run_shared_space_benchmark.py")
EXP025_SCRIPT = Path("bench/run_payload_absence_benchmark.py")


def _read_frame(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def run(ctx: RunnerContext) -> None:
    workspace = prepare_legacy_workspace(ctx, require_inputs=False)
    settings = (ctx.settings.get("experiments") or {}).get(EXPERIMENT_ID) or {}
    log_path = ctx.logs_dir / f"{EXPERIMENT_ID}.log"
    work_dir = BUNDLE_ROOT / "results" / "work" / "mechanistic"
    exp024_out = work_dir / "exp024_shared_space"
    exp025_out = work_dir / "exp025_payload_absence"
    summary_rows: list[dict] = []

    if settings.get("run_exp024", True):
        summary_rows.append({"kind": "script", "name": "EXP024_shared_space", "status": "planned" if ctx.dry_run else "executed"})
        run_python(
            workspace / EXP024_SCRIPT,
            cwd=workspace,
            log_path=log_path,
            args=["--out-dir", str(exp024_out)],
            dry_run=ctx.dry_run,
        )
    if settings.get("run_exp025", True):
        summary_rows.append({"kind": "script", "name": "EXP025_payload_absence", "status": "planned" if ctx.dry_run else "executed"})
        run_python(
            workspace / EXP025_SCRIPT,
            cwd=workspace,
            log_path=log_path,
            args=["--out-dir", str(exp025_out)],
            dry_run=ctx.dry_run,
        )
    if ctx.dry_run:
        write_summary_csv(summary_rows, ctx.tables_dir / f"{EXPERIMENT_ID}_summary.csv")
        return

    roots = [root for root in (exp024_out, exp025_out) if ctx.dry_run or root.exists()]
    table_sources: list[Path] = []
    figure_sources: list[Path] = []
    for root in roots:
        table_sources.extend(glob_many(root, ["tables/**/*.csv", "html/**/*.html", "manifest.json", "README.md"]))
        figure_sources.extend(glob_many(root, ["figures/**/*", "data/**/*.json"]))

    for root in roots:
        label = f"{EXPERIMENT_ID}_{root.name}"
        summary_rows.extend(copy_artifacts([p for p in table_sources if root in p.parents or p.parent == root], ctx.tables_dir, prefix=label, source_root=root, dry_run=ctx.dry_run))
        summary_rows.extend(copy_artifacts([p for p in figure_sources if root in p.parents or p.parent == root], ctx.figures_dir, prefix=label, source_root=root, dry_run=ctx.dry_run))

    frames: list[pd.DataFrame] = []
    for path in table_sources:
        if path.suffix != ".csv":
            continue
        frame = _read_frame(path)
        if frame is not None:
            root = next((source_root for source_root in roots if source_root in path.parents or path.parent == source_root), path.parent)
            frames.append(add_source_file_column(frame, str(path.relative_to(root)), EXPERIMENT_ID))

    write_endpoint_results(frames, ctx.tables_dir / f"{EXPERIMENT_ID}_endpoint_results.csv")
    write_summary_csv(summary_rows, ctx.tables_dir / f"{EXPERIMENT_ID}_summary.csv")
