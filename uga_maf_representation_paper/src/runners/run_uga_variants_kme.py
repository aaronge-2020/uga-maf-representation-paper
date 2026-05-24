"""Runner for UGA variants and RBF kernel mean embedding experiments."""

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


EXPERIMENT_ID = "uga_kme"
KME_REL = Path("cgr_validation_results/research/experiments/exploratory/2026_05_17_uga_kernel_mean_linear_benchmark")
SIGNAL_REL = Path("cgr_validation_results/research/experiments/supporting/2026_05_15_fast_defensible_uga_signal_recovery")


def _read_frame(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _script_plan(settings: dict) -> list[tuple[Path, str, list[str]]]:
    scripts: list[tuple[Path, str, list[str]]] = []
    if settings.get("run_linear_kme", True):
        scripts.append((KME_REL, "run_uga_kernel_mean_linear_benchmark.py", []))
    if settings.get("run_xgboost_sensitivity", True):
        scripts.append((KME_REL, "run_uga_kernel_mean_xgboost_sensitivity.py", []))
    if settings.get("run_tuned_kme", True):
        scripts.append((KME_REL, "run_hrd33_kme_grid_tuned_panel.py", []))
    if settings.get("run_standard_plus_tuned_kme_subset", True):
        scripts.append((KME_REL, "run_standard_plus_tuned_kme_subset.py", []))
    if settings.get("run_signal_recovery_linear", True):
        scripts.append(
            (
                SIGNAL_REL,
                "run_third_pass_linear_learner_screen.py",
                [
                    "--folds",
                    str(settings.get("folds", 5)),
                    "--confirm-repeats",
                    str(settings.get("confirm_repeats", 1)),
                    "--bootstrap",
                    str(settings.get("bootstrap", 200)),
                    "--max-finalists",
                    str(settings.get("max_finalists", 2)),
                ],
            )
        )
    return scripts


def run(ctx: RunnerContext) -> None:
    workspace = prepare_legacy_workspace(ctx, require_inputs=not ctx.dry_run)
    settings = (ctx.settings.get("experiments") or {}).get("uga_variants_kme") or {}
    log_path = ctx.logs_dir / f"{EXPERIMENT_ID}.log"
    summary_rows: list[dict] = []

    for rel_root, script_name, args in _script_plan(settings):
        exp_root = workspace / rel_root
        summary_rows.append(restore_legacy_feature_snapshot(ctx, f"{EXPERIMENT_ID}_{rel_root.name}", exp_root))
        script = exp_root / "code" / script_name
        summary_rows.append({"kind": "script", "name": script_name, "source_root": str(rel_root), "status": "planned" if ctx.dry_run else "executed"})
        run_python(script, cwd=exp_root, log_path=log_path, args=args, dry_run=ctx.dry_run)
    if ctx.dry_run:
        write_summary_csv(summary_rows, ctx.tables_dir / f"{EXPERIMENT_ID}_summary.csv")
        return

    source_roots = [workspace / KME_REL, workspace / SIGNAL_REL]
    table_sources: list[Path] = []
    oof_sources: list[Path] = []
    for root in source_roots:
        table_sources.extend(
            glob_many(
                root,
                [
                    "tables/**/*.csv",
                    "tables/**/*.html",
                    "data/**/*results*.csv",
                    "data/**/*summary*.csv",
                    "data/**/*metrics*.csv",
                    "data/**/*leaderboard*.csv",
                    "data/**/*selected_params*.csv",
                    "data/**/*manifest*.csv",
                    "data/**/*diagnostics*.csv",
                    "data/**/*tests*.csv",
                ],
            )
        )
        oof_sources.extend(glob_many(root, ["data/**/*oof*predictions*.csv*", "data/**/*probabilities*.csv*"]))

    for root in source_roots:
        summary_rows.extend(copy_artifacts([p for p in table_sources + oof_sources if root in p.parents], ctx.tables_dir, prefix=EXPERIMENT_ID, source_root=root, dry_run=ctx.dry_run))
        if not ctx.dry_run and root.exists():
            summary_rows.append(snapshot_legacy_feature_outputs(ctx, f"{EXPERIMENT_ID}_{root.name}", root))

    frames: list[pd.DataFrame] = []
    for path in table_sources:
        if path.suffix == ".html":
            continue
        frame = _read_frame(path)
        if frame is not None:
            root = next((source_root for source_root in source_roots if source_root in path.parents), path.parent)
            frames.append(add_source_file_column(frame, str(path.relative_to(root)), EXPERIMENT_ID))

    write_endpoint_results(frames, ctx.tables_dir / f"{EXPERIMENT_ID}_endpoint_results.csv")
    write_summary_csv(summary_rows, ctx.tables_dir / f"{EXPERIMENT_ID}_summary.csv")
