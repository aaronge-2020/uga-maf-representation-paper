#!/usr/bin/env python3
"""EXP023: TCGA-BRCA multi-endpoint HRD paper workflow (nested CV, no BiCGR-52 in mains)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_EXP023 = Path(__file__).resolve().parent
if str(_EXP023) not in sys.path:
    sys.path.insert(0, str(_EXP023))

from exp23_workflow import build_config, run_analysis, run_fit_exposures, run_prepare  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP023 TCGA-BRCA HRD paper pipeline")
    p.add_argument("stage", choices=["prepare", "exposures", "analyze", "figure", "all"], nargs="?", default="all")
    p.add_argument("--repo-root", default=None)
    p.add_argument("--optuna-storage", default=None, help="Optional Optuna storage URL, e.g. sqlite:///optuna_exp023.db")
    p.add_argument("--reg-trials", type=int, default=None)
    p.add_argument("--clf-trials", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root).resolve() if args.repo_root else None
    cfg = build_config(
        repo_root=root,
        optuna_storage=args.optuna_storage,
        regression_trials=args.reg_trials,
        classification_trials=args.clf_trials,
    )
    if args.stage in {"prepare", "all"}:
        print("[EXP023] prepare")
        run_prepare(cfg)
    if args.stage in {"exposures", "all"}:
        print("[EXP023] exposures")
        run_fit_exposures(cfg)
    if args.stage in {"analyze", "all"}:
        print("[EXP023] analyze")
        run_analysis(cfg)
    if args.stage in {"figure", "all"}:
        print("[EXP023] figure + conclusion")
        from figure_exp23_brca_hrd import run as run_fig  # noqa: E402

        run_fig(cfg.assets_dir)


if __name__ == "__main__":
    main()
