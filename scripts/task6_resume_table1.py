#!/usr/bin/env python3
"""Task 6 crash-recovery driver: resume the Table-1 regeneration across VM restarts.

WHY THIS EXISTS (2026-10-13): the canonical entry point
scripts/reproduce_table1.py is designed for ONE honest run: its stage 4
(stage_prerun_archive) moves all existing main-panel outputs aside before
running, so it cannot resume partial progress. The rerun box undergoes
runtime-level restart drains roughly every ~80 minutes (run 1 died at ~75
min, run 2 at ~80 min), while the 8 Table-1 slots need several hours on 2
vCPUs. A single entry-point invocation therefore cannot finish.

This driver runs the IDENTICAL code path as the entry point's stage 5
(stage_run) -- same production runner (src/run_all_experiments.py +
runners.run_main_manuscript_complete_panel.run), same Table-1 config
(config/experiment_settings.table1.yaml), same two documented in-script
adapters (kucab skip, Bio MAF timing wrapper) -- but WITHOUT the prerun
archive step. That is safe and honest here because:

  * The manuscript's own outputs were already archived aside by the two
    entry-point runs (results/tables/task6_prerun_archive/20261013T120157Z/
    holds the true manuscript tables, checkpoints, and the 8-row manuscript
    reference; 20261013T133518Z/ holds run 1's partial regenerated outputs).
  * The worktree's results/tables/ now contains ONLY regenerated data
    (run 2's 2 completed burden_only slots). There is nothing left for the
    runner to "silently reuse" that is not itself regenerated output.
  * The production runner resumes completed (endpoint, representation,
    learner) slots from main_manuscript_complete_panel_endpoint_results.csv
    and resumes nested folds from the fold-checkpoint files, so each
    invocation continues exactly where the previous one died.

Usage: run repeatedly (e.g. after each restart drain) until all 8 Table-1
slots are present in results/tables/task6_table1_regenerated.csv. Then run
scripts/task6_fix_rerun_comparison.py to rebuild the comparison against the
true manuscript reference.

The canonical single entry point remains scripts/reproduce_table1.py; this
driver is a documented crash-recovery companion, not a replacement.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Thread caps BEFORE any numeric library is imported (same as entry point).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE1_CONFIG = REPO_ROOT / "config" / "experiment_settings.table1.yaml"
PATHS_CONFIG = REPO_ROOT / "config" / "paths.yaml"

TABLE1_SLOTS = [
    ("cancer_type_top20", "burden_only", "linear"),
    ("cancer_type_top20", "burden_only", "xgboost"),
    ("cancer_type_top20", "standard_sbs96_id83", "linear"),
    ("cancer_type_top20", "standard_sbs96_id83", "xgboost"),
    ("cancer_type_top20", "MAF_stack_only", "linear"),
    ("cancer_type_top20", "MAF_stack_only", "xgboost"),
    ("cancer_type_top20", "signatures_plus_MAF_stack", "linear"),
    ("cancer_type_top20", "signatures_plus_MAF_stack", "xgboost"),
]


def log(msg: str) -> None:
    print(f"[task6-resume] {msg}", flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-dir", default=None)
    parser.add_argument(
        "--skip",
        default="",
        help=("Comma-separated representation/learner slots to skip this invocation, "
              "e.g. 'standard_sbs96_id83/linear'. Only applies to slots on the plain "
              "evaluate_features path (burden_only, standard_sbs96_id83); MAF slots "
              "use the nested feature-set path with fold checkpointing and do not "
              "need skipping. Documents a blocked slot so the remaining slots can "
              "proceed; the skipped slot is retried separately. "
              "Skips are recorded in the console log."),
    )
    args = parser.parse_args()
    skip_slots = {s.strip() for s in args.skip.split(",") if s.strip()}
    if skip_slots:
        log(f"SKIP requested for: {sorted(skip_slots)}")
    datasets_dir = Path(args.datasets_dir).expanduser().resolve() if args.datasets_dir else (REPO_ROOT.parent / "github_exports_datasets").resolve()
    if not datasets_dir.exists():
        raise SystemExit(f"dataset bundle not found: {datasets_dir}")
    log(f"datasets_dir={datasets_dir}")

    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.argv = [
        "run_all_experiments.py",
        "--config", str(TABLE1_CONFIG),
        "--paths", str(PATHS_CONFIG),
        "--datasets-dir", str(datasets_dir),
        "--only", "main_manuscript_complete_panel",
        "--skip-figures",
    ]
    import run_all_experiments  # noqa: E402
    import runners.run_main_manuscript_complete_panel as panel  # noqa: E402

    # Adapter 1 (same as entry point): skip the kucab feature build when the
    # task6_skip_kucab flag is set (Table 1 has no kucab endpoint).
    _orig_kucab = panel._load_kucab_features

    def _task6_kucab(ctx, settings):
        if settings.get("task6_skip_kucab"):
            log("task6_skip_kucab is set -> skipping kucab feature build")
            return None, {}
        return _orig_kucab(ctx, settings)

    panel._load_kucab_features = _task6_kucab

    # Adapter 2 (same as entry point): behaviour-preserving timing wrapper
    # around the Bio MAF builder for the compute-cost deliverable.
    _orig_maf = panel._load_mc3_maf_features

    def _timed_maf(ctx, model_id, settings=None):
        t0 = time.time()
        frame, cache_key = _orig_maf(ctx, model_id, settings)
        wall = time.time() - t0
        log(f"Bio MAF build {model_id}: {wall:.3f}s for {frame.shape[0]} tumours "
            f"x {frame.shape[1]} features ({wall / max(int(frame.shape[0]), 1):.6f}s per tumour)")
        return frame, cache_key

    panel._load_mc3_maf_features = _timed_maf

    # Optional documented skip for blocked slots (e.g. a slot that cannot
    # finish within one VM lifetime and has no fold checkpointing). The
    # skipped slot is recorded with a NaN score and status
    # "skipped_blocked_slot"; _current_row() rejects NaN-score rows, so a
    # later invocation WITHOUT --skip will genuinely (re)run the slot. The
    # placeholder row is replaced via keep="last" dedup when the real row
    # lands. This changes no modelling code and no completed results.
    if skip_slots:
        import pandas as pd  # noqa: E402

        _orig_evaluate_features = panel.evaluate_features

        def _skipping_evaluate_features(endpoint, features, representation, learner, ctx, settings):
            if f"{representation}/{learner}" in skip_slots:
                log(f"SKIP {endpoint.name} {representation}/{learner}: documented blocked slot; "
                    f"recorded as skipped_blocked_slot with NaN score (will be genuinely rerun later)")
                row = {
                    "benchmark": getattr(endpoint, "benchmark", ""),
                    "endpoint": endpoint.name,
                    "representation": representation,
                    "learner": learner,
                    "metric": "balanced_accuracy",
                    "score": float("nan"),
                    "status": "skipped_blocked_slot",
                    "tuning": "outer_train_inner_validation_randomized_search",
                    "primary_metric_scope": "pooled_global_oof",
                }
                return row, pd.DataFrame(), pd.DataFrame()
            return _orig_evaluate_features(endpoint, features, representation, learner, ctx, settings)

        panel.evaluate_features = _skipping_evaluate_features

    log("launching production runner (resume mode; no prerun archive)")
    t0 = time.time()
    run_all_experiments.main()
    log(f"production runner returned after {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
