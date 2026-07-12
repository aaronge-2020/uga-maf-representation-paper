#!/usr/bin/env python3
"""MuAt ablation and rerun driver: preflight -> data -> stages -> comparison tables.

    python scripts/run_muat_ablation.py --preflight     # check the machine, run nothing
    python scripts/run_muat_ablation.py                 # HRD ablation only (~1h)
    python scripts/run_muat_ablation.py --overnight     # HRD ablation + cancer_type_top20
    python scripts/run_muat_ablation.py --report-only   # rebuild tables from existing results

Stages
------
The three HRD arms isolate what each fix is worth. Same folds, same seed, same everything else:

    a   final_epoch_selection=off, fixed_hash    commit 38f507c exactly, as it would have run
    b   final_epoch_selection=ON,  fixed_hash    a->b = cost of the <=3-epoch training bug
    c   final_epoch_selection=on,  EXACT dicts   b->c = cost of the token hash collisions

    top20   cancer_type_top20, full fix. MuAt's own benchmark, and the only endpoint with a
            published MuAt number (0.641 accuracy / 0.906 top-5) to compare against. Long.

Everything is logged to results/logs/ablation_<stage>.log, so a failure overnight is diagnosable
in the morning. With --overnight a failing stage does not abort the ones after it.
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

STAGES: dict[str, dict[str, str]] = {
    "a": {
        "config": "config/experiment_settings.muat_ab_a_before.yaml",
        "tag": "ab_a_before",
        "label": "A  before (bug + hashed dicts)",
        "group": "hrd",
    },
    "b": {
        "config": "config/experiment_settings.muat_ab_b_epochfix.yaml",
        "tag": "ab_b_epochfix",
        "label": "B  epoch fix only",
        "group": "hrd",
    },
    "c": {
        "config": "config/experiment_settings.muat_ab_c_full.yaml",
        "tag": "ab_c_full",
        "label": "C  epoch fix + exact dicts",
        "group": "hrd",
    },
    "d": {
        "config": "config/experiment_settings.muat_ab_d_faithful.yaml",
        "tag": "ab_d_faithful",
        "label": "D  + MuAt-faithful tokens",
        "group": "hrd",
    },
    "top20": {
        "config": "config/experiment_settings.muat_top20_comparable.yaml",
        "tag": "cancer_type_top20_comparable",
        "label": "cancer_type_top20 (full fix)",
        "group": "top20",
    },
}

HRD_STAGES = [key for key, stage in STAGES.items() if stage["group"] == "hrd"]

CANONICAL_SPLIT = REPO_ROOT / "results/tables/main_manuscript_complete_panel_oof_predictions.csv"

# Stale fold checkpoints written by the pre-fix code are now rejected by the fingerprint, but the
# cache directory is still worth clearing before a first post-fix run.
MUAT_CACHE = REPO_ROOT / "results/cache/features/muat_style_tcga_comparator"


def resolve_datasets_dir() -> Path:
    """Locate the dataset bundle the way src/utils/config.py does.

    Resolved on every call, never cached at import: on a fresh machine the bundle does not exist
    until fetch_datasets.py has run, and a module-level constant would still be pointing at the
    legacy sibling path afterwards.
    """

    env = os.environ.get("CGR_DATASETS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    fetched = REPO_ROOT.parent / "hf_bundle" / "github_exports_datasets"
    if fetched.exists():
        return fetched
    return REPO_ROOT.parent / "github_exports_datasets"


def bundle_inputs(datasets_dir: Path) -> dict[str, Path]:
    return {
        "mc3 maf": datasets_dir / "datasets/tcga_mc3/raw/mc3.v0.2.8.PUBLIC.maf.gz",
        "mc3 clinical": datasets_dir / "datasets/tcga_mc3/raw/clinical_PANCAN_patient_with_followup.tsv",
        "hrd cohort": datasets_dir / "datasets/tcga_brca_hrd/cohort/final_analysis_cohort.tsv",
    }


# endpoint_registry.build_cdr_cancer_type_labels reads this for cancer_type_top20, and the CDR
# survival endpoints read it too. It is NOT in the published Hugging Face bundle -- there is no
# .xlsx in it at all -- so cancer_type_top20 fails after loading the 718 MB MAF unless it is
# staged by hand. Checked in preflight so a run is never wasted on it again.
def top20_inputs(datasets_dir: Path) -> dict[str, Path]:
    return {
        "tcga cdr table": datasets_dir / "datasets/tcga_mc3/raw/TCGA-CDR-SupplementalTableS1.xlsx",
    }


def child_env(datasets_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["CGR_DATASETS_DIR"] = str(datasets_dir)
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Event bags vary in length from a handful of mutations to the 5,000 cap, so allocation sizes
    # change on almost every batch-size-1 step. Expandable segments stop that fragmenting the
    # caching allocator over a long run, which is the usual cause of an OOM many hours in on a job
    # whose peak memory is well within the card. Not supported on Windows, where setting it only
    # produces a UserWarning on every process start.
    if sys.platform != "win32":
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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

    try:
        import torch

        print(f"torch         : {torch.__version__}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"cuda          : YES  {props.name}  ({props.total_memory / 1024**3:.1f} GB)")
        else:
            print("cuda          : NO  <-- CPU-only torch")
            problems.append(
                "torch cannot see a GPU. On Windows the default PyPI wheel is CPU-only.\n"
                "        pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall"
            )
    except ImportError:
        problems.append("torch is not installed (pip install -r requirements.txt)")

    if CANONICAL_SPLIT.exists():
        print(f"split manifest: ok ({CANONICAL_SPLIT.stat().st_size / 1024**2:.0f} MB)")
    else:
        print("split manifest: MISSING")
        problems.append(
            "results/tables/main_manuscript_complete_panel_oof_predictions.csv is absent.\n"
            "        It is gitignored, so no clone has it. Fetch it:\n"
            "        python scripts/fetch_datasets.py --minimal"
        )

    print(f"datasets dir  : {datasets_dir}")
    missing = {name: path for name, path in bundle_inputs(datasets_dir).items() if not path.exists()}
    if missing:
        for name, path in missing.items():
            print(f"                MISSING {name}: {path}")
        problems.append(
            "the dataset bundle does not contain the expected inputs at the path above.\n"
            "        python scripts/fetch_datasets.py --minimal\n"
            "        or point at an existing bundle:  $env:CGR_DATASETS_DIR = 'D:\\...\\github_exports_datasets'"
        )
    else:
        size = bundle_inputs(datasets_dir)["mc3 maf"].stat().st_size / 1024**2
        print(f"                mc3 maf ok ({size:.0f} MB), mc3 clinical ok, hrd cohort ok")

    for key in stages:
        config = REPO_ROOT / STAGES[key]["config"]
        if not config.exists():
            problems.append(f"missing config: {STAGES[key]['config']}")
    print(f"stages        : {', '.join(stages)}")

    if "top20" in stages:
        missing_top20 = {name: path for name, path in top20_inputs(datasets_dir).items() if not path.exists()}
        if missing_top20:
            for name, path in missing_top20.items():
                print(f"top20 inputs  : MISSING {name}")
                print(f"                {path}")
            problems.append(
                "cancer_type_top20 needs TCGA-CDR-SupplementalTableS1.xlsx, which is NOT in the\n"
                "        published Hugging Face bundle (it contains no .xlsx at all). Copy it from\n"
                "        Aaron's Google Drive github_exports_datasets export, where the manifest lists it at\n"
                "        .../mc3_source/raw/TCGA-CDR-SupplementalTableS1.xlsx, into the path above.\n"
                "        It is also needed by the main panel for cancer type and overall survival, so the\n"
                "        bundle should be re-uploaded with it included."
            )
        else:
            print("top20 inputs  : tcga cdr table ok")

    if MUAT_CACHE.exists():
        print(f"muat cache    : present ({MUAT_CACHE}) -- pass --clear-cache if these predate the fixes")

    if problems:
        print("\nBLOCKING:")
        for item in problems:
            print(f"  - {item}")
    else:
        print("\nall clear.")
    return problems


def fetch_data() -> int:
    _hr("FETCHING DATA BUNDLE")
    return subprocess.call(
        [sys.executable, str(REPO_ROOT / "scripts/fetch_datasets.py"), "--minimal"],
        cwd=str(REPO_ROOT),
    )


def archive_stale_outputs(key: str) -> None:
    """Move any pre-existing outputs for this stage out of results/tables.

    cancer_type_top20 already has results on disk from the pre-fix run (accuracy 0.539). Result
    files are keyed only by output_tag, so if a stage fails, the previous run's files are still
    sitting there and report() would read them and present them as this run's numbers. Archiving
    first means a failed stage reports as missing rather than as a stale success.
    """

    tag = STAGES[key]["tag"]
    tables = REPO_ROOT / "results/tables"
    stale = [path for path in tables.glob(f"muat_style_tcga_comparator_{tag}_*") if path.is_file()]
    if not stale:
        return
    archive = REPO_ROOT / "results/archive" / f"{tag}_{time.strftime('%Y%m%dT%H%M%S')}"
    archive.mkdir(parents=True, exist_ok=True)
    for path in stale:
        path.rename(archive / path.name)
    print(f"[stage {key}] archived {len(stale)} pre-existing output file(s) -> {archive.relative_to(REPO_ROOT)}", flush=True)


def run_stage(key: str, datasets_dir: Path) -> tuple[int, float]:
    stage = STAGES[key]
    _hr(f"STAGE {key.upper()} — {stage['label']}")
    archive_stale_outputs(key)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"ablation_{key}.log"
    print(f"log: {log_path}", flush=True)

    started = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(REPO_ROOT / "src/run_all_experiments.py"),
                "--config", stage["config"],
                "--paths", "config/paths.yaml",
                "--datasets-dir", str(datasets_dir),
            ],
            cwd=str(REPO_ROOT),
            env=child_env(datasets_dir),
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
    print(f"\n[stage {key}] exit={code} elapsed={elapsed / 60:.1f} min", flush=True)
    return code, elapsed


def _read_stage(key: str):
    import pandas as pd

    path = REPO_ROOT / f"results/tables/muat_style_tcga_comparator_{STAGES[key]['tag']}_endpoint_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def report(stages: list[str]) -> int:
    import pandas as pd

    _hr("RESULTS")
    rows = []
    for key in stages:
        frame = _read_stage(key)
        if frame is None:
            print(f"[report] no results for stage {key}")
            continue
        for _, row in frame.iterrows():
            rows.append(
                {
                    "stage": key,
                    "description": STAGES[key]["label"],
                    "endpoint": row.get("endpoint"),
                    "metric": row.get("metric"),
                    "score": row.get("score"),
                    "accuracy": row.get("accuracy"),
                    "balanced_accuracy": row.get("balanced_accuracy"),
                    "top5_accuracy": row.get("top5_accuracy"),
                    "dictionary": row.get("dictionary_mode"),
                }
            )
    if not rows:
        print("no results found. run the stages first.")
        return 1

    table = pd.DataFrame(rows)
    out = REPO_ROOT / "results/tables/muat_ablation_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    hrd = table[table["stage"].isin(HRD_STAGES)]
    for endpoint, group in hrd.groupby("endpoint", sort=False):
        print(f"\n{endpoint}  ({group['metric'].iloc[0]})")
        print("-" * 66)
        # Baseline is arm A specifically, never "whichever arm ran first". If A failed, the deltas
        # would otherwise be computed against B while still being labelled "vs A".
        arm_a = group[group["stage"] == "a"]
        baseline = float(arm_a["score"].iloc[0]) if not arm_a.empty else None
        for key in HRD_STAGES:
            match = group[group["stage"] == key]
            if match.empty:
                print(f"  {key.upper()}  {'(no result)':>10}          {STAGES[key]['label']}")
                continue
            score = float(match["score"].iloc[0])
            delta = "" if (baseline is None or key == "a") else f"  ({score - baseline:+.4f} vs A)"
            print(f"  {key.upper()}  {score:.4f}{delta}   {STAGES[key]['label']}")
        if baseline is None:
            print("  (arm A produced no result, so no deltas are shown)")

    top20 = table[table["stage"] == "top20"]
    if not top20.empty:
        row = top20.iloc[0]
        print("\ncancer_type_top20  (MuAt's own benchmark)")
        print("-" * 66)
        print(f"  MuAt-compatible, fixed     accuracy {row.get('accuracy')}   top-5 {row.get('top5_accuracy')}")
        print( "  MuAt, published            accuracy 0.641          top-5 0.906")
        print( "  previous MuAt-compatible   accuracy 0.539          top-5 0.861")
        print( "  Signatures+MAF / XGBoost   accuracy 0.744          top-5 0.955")

    print(f"\n[report] written -> {out.relative_to(REPO_ROOT)}")
    print("[report] tabular baselines for HRD: HRD_Score 0.781 (Spearman), hrd_binary_33 0.867 (AUROC)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=None)
    parser.add_argument("--overnight", action="store_true", help="HRD ablation followed by cancer_type_top20.")
    parser.add_argument("--preflight", action="store_true", help="Check the machine and exit.")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--clear-cache", action="store_true", help="Delete the MuAt feature cache before running.")
    parser.add_argument("--force", action="store_true", help="Run even if preflight reports problems.")
    args = parser.parse_args()

    if args.stages:
        stages = list(args.stages)
    elif args.overnight:
        stages = HRD_STAGES + ["top20"]
    else:
        stages = list(HRD_STAGES)

    if args.report_only:
        # Report everything that exists unless the caller narrowed it explicitly. Defaulting to the
        # HRD arms here would silently omit top20 when reading results back the next morning.
        return report(list(args.stages) if args.stages else list(STAGES))

    problems = preflight(stages)
    if args.preflight:
        return 1 if problems else 0

    if any("fetch_datasets" in item for item in problems) and not args.skip_fetch:
        if fetch_data() != 0:
            print("\n[fatal] data fetch failed.")
            return 1
        problems = preflight(stages)

    if problems and not args.force:
        print("\n[fatal] preflight failed. Fix the above, or rerun with --force.")
        return 1

    if args.clear_cache and MUAT_CACHE.exists():
        import shutil

        shutil.rmtree(MUAT_CACHE)
        print(f"[cache] removed {MUAT_CACHE}")

    datasets_dir = resolve_datasets_dir()
    results: dict[str, int] = {}
    total = 0.0
    for key in stages:
        code, elapsed = run_stage(key, datasets_dir)
        results[key] = code
        total += elapsed
        if code != 0:
            if args.overnight:
                # Do not lose the rest of the night to one bad stage.
                print(f"[warn] stage {key.upper()} exited {code}. Continuing to the next stage.", flush=True)
                continue
            print(f"\n[fatal] stage {key.upper()} exited {code}. Stopping.")
            report(stages)
            return code

    _hr("SUMMARY")
    for key in stages:
        code = results.get(key)
        status = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"  {key.upper():<6} {status:<18} {STAGES[key]['label']}")
    print(f"\ntotal runtime: {total / 60:.1f} min")
    if any(code != 0 for code in results.values()):
        print("see results/logs/ablation_<stage>.log for the failing stage")

    report(stages)
    return 0 if all(code == 0 for code in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
