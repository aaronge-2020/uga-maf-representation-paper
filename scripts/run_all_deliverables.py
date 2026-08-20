#!/usr/bin/env python3
"""Run every deliverable from the 1 Aug 2026 meeting, in dependency order.

    python scripts/run_all_deliverables.py --preflight     # check the machine, run nothing
    python scripts/run_all_deliverables.py                 # everything
    python scripts/run_all_deliverables.py --stages survival featsel
    python scripts/run_all_deliverables.py --report-only

Stages
------
    survival   Cancer-type-only survival baseline (Jeya). Does Bio MAF v4 predict survival
               beyond cancer type alone? Cheap, and the highest scientific stakes.
    featsel    Individual-feature selection inside Bio MAF v4 (Jeya), plus SHAP attribution
               (Vijay). Replaces the current block-level selection.
    muat_d     MuAt-faithful token representation on the HRD endpoints. The calibration
               baseline for any claim about the architecture.
    muat_top20 cancer_type_top20 against MuAt's published 0.641 / 0.906. Long: hours.

A failing stage logs and does not stop the stages after it, so an overnight run still produces
everything that can be produced. Each stage tees to results/logs/deliverable_<stage>.log.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "results/logs"

STAGES = {
    "survival": {
        "label": "Cancer-type survival baseline (Jeya)",
        "cmd": ["scripts/run_survival_cancer_type_baseline.py", "--endpoints", "OS"],
        "outputs": ["results/tables/survival_cancer_type_baseline_summary.csv"],
    },
    "featsel": {
        "label": "Individual-feature selection + SHAP (Jeya / Vijay)",
        "cmd": ["scripts/run_individual_feature_selection.py", "--endpoint", "cancer_type_top20"],
        "outputs": ["results/tables/feature_selection_cancer_type_top20_summary.json"],
    },
    "muat_d": {
        "label": "MuAt-faithful tokens, HRD endpoints (arm D)",
        "cmd": ["scripts/run_muat_ablation.py", "--stages", "d", "--skip-fetch"],
        "outputs": ["results/tables/muat_style_tcga_comparator_ab_d_faithful_endpoint_results.csv"],
    },
    "muat_top20": {
        "label": "MuAt cancer_type_top20 vs published 0.641",
        "cmd": ["scripts/run_muat_ablation.py", "--stages", "top20", "--skip-fetch"],
        "outputs": ["results/tables/muat_style_tcga_comparator_cancer_type_top20_comparable_endpoint_results.csv"],
    },
    "figures": {
        "label": "Figures for the new analyses",
        "cmd": ["scripts/make_new_analysis_figures.py"],
        "outputs": ["results/figures/figure_S_survival_cancer_type_baseline.png"],
    },
}

# `figures` runs last: it reads the tables the earlier stages write.
ORDER = ["survival", "featsel", "muat_d", "muat_top20", "figures"]


def resolve_datasets_dir() -> Path:
    env = os.environ.get("CGR_DATASETS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    fetched = REPO_ROOT.parent / "hf_bundle" / "github_exports_datasets"
    if fetched.exists():
        return fetched
    return REPO_ROOT.parent / "github_exports_datasets"


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["CGR_DATASETS_DIR"] = str(resolve_datasets_dir())
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _hr(title: str = "") -> None:
    print("\n" + "=" * 78, flush=True)
    if title:
        print(title, flush=True)
        print("=" * 78, flush=True)


def preflight(stages: list[str]) -> list[str]:
    _hr("PREFLIGHT")
    problems: list[str] = []
    datasets_dir = resolve_datasets_dir()
    print(f"python        : {sys.version.split()[0]}")
    print(f"datasets dir  : {datasets_dir}")

    cdr = datasets_dir / "datasets/tcga_mc3/raw/TCGA-CDR-SupplementalTableS1.xlsx"
    maf = datasets_dir / "datasets/tcga_mc3/raw/mc3.v0.2.8.PUBLIC.maf.gz"
    for label, path, needed_by in (
        ("CDR table", cdr, "survival, featsel, muat_top20"),
        ("MC3 MAF", maf, "muat_d, muat_top20"),
    ):
        status = "ok" if path.exists() else "MISSING"
        print(f"  {status:8s} {label:12s} ({needed_by})")
        if not path.exists():
            problems.append(f"{label} not found at {path}")

    bio = REPO_ROOT / "results/tables/quick_bio_v4_maf_features.csv.gz"
    print(f"  {'ok' if bio.exists() else 'MISSING':8s} Bio MAF v4 feature table (survival, featsel)")
    if not bio.exists():
        problems.append(f"Bio MAF v4 feature cache not found at {bio}")

    for module, needed_by, fatal in (
        ("xgboost", "featsel", True),
        ("sksurv", "survival, featsel(OS)", True),
        ("torch", "muat_d, muat_top20", True),
        ("shap", "featsel SHAP plots", False),
        ("matplotlib", "featsel SHAP plots", False),
    ):
        try:
            __import__(module)
            print(f"  {'ok':8s} {module:12s} ({needed_by})")
        except ImportError:
            print(f"  {'MISSING' if fatal else 'absent':8s} {module:12s} ({needed_by})")
            if fatal:
                problems.append(f"{module} is not installed (needed by {needed_by})")
            else:
                print(f"           -> pip install {module}   [optional; stage still runs without it]")

    try:
        import torch

        if any(key.startswith("muat") for key in stages):
            if torch.cuda.is_available():
                print(f"  {'ok':8s} CUDA         ({torch.cuda.get_device_name(0)})")
            else:
                print(f"  {'WARN':8s} CUDA         (CPU-only torch: MuAt stages will be very slow)")
                problems.append("torch cannot see a GPU; the MuAt stages need one to finish in reasonable time")
    except ImportError:
        pass

    print(f"\nstages        : {', '.join(stages)}")
    if problems:
        print("\nBLOCKING:")
        for item in problems:
            print(f"  - {item}")
    else:
        print("\nall clear.")
    return problems


def run_stage(key: str) -> tuple[int, float]:
    stage = STAGES[key]
    _hr(f"STAGE {key.upper()} - {stage['label']}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"deliverable_{key}.log"
    print(f"log: {log_path}", flush=True)

    started = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            [sys.executable, *stage["cmd"]],
            cwd=str(REPO_ROOT),
            env=child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        code = process.wait()

    elapsed = time.time() - started
    print(f"\n[{key}] exit={code} elapsed={elapsed / 60:.1f} min", flush=True)
    return code, elapsed


def report(stages: list[str]) -> int:
    import pandas as pd

    _hr("SUMMARY OF DELIVERABLES")

    surv = REPO_ROOT / "results/tables/survival_cancer_type_baseline_summary.csv"
    comp = REPO_ROOT / "results/tables/survival_cancer_type_baseline_comparisons.csv"
    if surv.exists():
        frame = pd.read_csv(surv)
        print("\nSurvival: does Bio MAF v4 beat cancer type alone?")
        print("-" * 70)
        for _, row in frame.iterrows():
            print(f"  {row['arm']:26s} C-index {float(row['score']):.4f}")
        if comp.exists():
            for _, row in pd.read_csv(comp).iterrows():
                print(
                    f"  {row['candidate']} vs {row['reference']}: "
                    f"{row['delta']:+.4f}  95% CI [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  p={row['p_two_sided']:.3f}"
                )
    else:
        print("\nSurvival baseline: no results")

    import json

    for endpoint in ("cancer_type_top20", "OS"):
        path = REPO_ROOT / f"results/tables/feature_selection_{endpoint}_summary.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            print(f"\nFeature selection ({endpoint})")
            print("-" * 70)
            print(f"  selected {data['n_features_selected']} / {data['n_features_available']} features "
                  f"({100 * data['selection_fraction']:.1f}%)")
            print(f"  inner-CV {data['winner_inner_score']:.4f} (best {data['best_inner_score']:.4f}), "
                  f"held-out {data['heldout_score']:.4f}")

    for key, label, published in (
        ("ab_d_faithful", "MuAt arm D (faithful tokens, HRD)", None),
        ("cancer_type_top20_comparable", "MuAt cancer_type_top20", "0.641 acc / 0.906 top-5"),
    ):
        path = REPO_ROOT / f"results/tables/muat_style_tcga_comparator_{key}_endpoint_results.csv"
        if path.exists():
            frame = pd.read_csv(path)
            print(f"\n{label}")
            print("-" * 70)
            for _, row in frame.iterrows():
                extra = ""
                if published and "accuracy" in row and pd.notna(row.get("accuracy")):
                    extra = f"   accuracy {row['accuracy']:.4f} top-5 {row.get('top5_accuracy', float('nan')):.4f}"
                print(f"  {row['endpoint']:18s} {row['metric']:10s} {float(row['score']):.4f}{extra}")
            if published:
                print(f"  {'MuAt published':18s} {published}")
        else:
            print(f"\n{label}: no results")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=None)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stages = args.stages or list(ORDER)
    stages = [key for key in ORDER if key in stages]

    if args.report_only:
        return report(stages)

    problems = preflight(stages)
    if args.preflight:
        return 1 if problems else 0
    if problems and not args.force:
        print("\n[fatal] preflight failed. Fix the above, or rerun with --force.")
        return 1

    results: dict[str, int] = {}
    total = 0.0
    for key in stages:
        code, elapsed = run_stage(key)
        results[key] = code
        total += elapsed
        if code != 0:
            print(f"[warn] stage {key} failed (exit {code}); continuing.", flush=True)

    _hr("STAGE STATUS")
    for key in stages:
        code = results.get(key)
        print(f"  {key:12s} {'ok' if code == 0 else f'FAILED (exit {code})':<20s} {STAGES[key]['label']}")
    print(f"\ntotal runtime: {total / 60:.1f} min")

    report(stages)
    return 0 if all(code == 0 for code in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
