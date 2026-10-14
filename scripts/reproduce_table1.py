#!/usr/bin/env python3
"""Regenerate manuscript Table 1 end to end from the deposited bundle. (Task 6)

Table 1 = the cancer_type_top20 multiclass benchmark:
    representations : burden_only, standard_sbs96_id83, MAF_stack_only,
                      signatures_plus_MAF_stack
    learners        : linear, xgboost
    metric          : balanced_accuracy            (8 slots total)

This is a SINGLE entry point. It drives the PRODUCTION pipeline
(src/run_all_experiments.py -> runners.run_main_manuscript_complete_panel.run)
with the Table-1-only config (config/experiment_settings.table1.yaml); it does
not reimplement any modelling code.

Stages:
  1. environment audit      -> results/logs/task6_table1_environment.json
  2. dataset bundle         -> download the Table-1 subset from the public
                               Hugging Face dataset if --datasets-dir is absent;
                               verify sha256 of key inputs against
                               manifests/dataset_assets_manifest.csv
  3. OncoKB role table      -> fetch live if missing; record download date +
                               sha256; archive a dated snapshot copy
  4. prerun archive         -> move existing main-panel result tables and the
                               cancer_type_top20 fold checkpoints aside so the
                               run cannot silently reuse committed outputs
                               (endpoint-row resume and fold-checkpoint resume
                               would otherwise make regeneration a no-op)
  5. run                    -> production runner, Table-1 slice only
  6. extract                -> results/tables/task6_table1_regenerated.csv
  7. compare                -> results/tables/task6_table1_comparison.csv and
                               results/logs/task6_table1_run_report.json

Deviations from the manuscript run (all documented, none hidden):
  * Python 3.12.3 instead of 3.10; see environment/table1_lockfile.txt.
  * xgboost 3.4.1 (CPU hist via the pipeline's own gpu_hist fallback) instead
    of 1.6.2/gpu_hist on GPU.
  * OncoKB role table fetched live at build time (unversioned API); the exact
    historical snapshot that built the published matrix is not recoverable
    (see config/feature_resources/oncokb_cancer_genes.PROVENANCE.md).
  * Parallelism knobs 32/32/16 -> 2/2/2 (wall time only; seeds do not depend
    on n_jobs).
  * kucab damage_class endpoint skipped via a documented in-script adapter
    (task6_skip_kucab flag; Table 1 has no kucab endpoint; skipping avoids
    the 3.2 GB GRCh37 FASTA).

Usage:
    python scripts/reproduce_table1.py [--datasets-dir DIR] [--dry-run]
                                       [--skip-bundle-fetch] [--skip-oncokb-fetch]

    --dry-run runs the production dry-run path (environment + planned slots)
    without training anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Thread caps BEFORE any numeric library is imported.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE1_CONFIG = REPO_ROOT / "config" / "experiment_settings.table1.yaml"
PATHS_CONFIG = REPO_ROOT / "config" / "paths.yaml"
HF_REPO_ID = "vjdeara/cgr_tensorlab_umsom"
HF_BUNDLE_PREFIX = "github_exports_datasets"

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
TABLE1_METRIC = "balanced_accuracy"
ENV_TOLERANCE = 0.001167  # accepted Task 0 environmental tolerance

MAIN_PANEL_FILES = [
    "main_manuscript_complete_panel_endpoint_results.csv",
    "main_manuscript_complete_panel_oof_predictions.csv",
    "main_manuscript_complete_panel_fold_metrics.csv",
    "main_manuscript_complete_panel_feature_manifest.csv",
    "main_manuscript_complete_panel_endpoint_audit.csv",
    "main_manuscript_complete_panel_summary.csv",
]

# Bundle-relative inputs the Table 1 run reads, with the manifest hash source.
# (hashes come from manifests/dataset_assets_manifest.csv where listed there)
KEY_INPUTS = [
    "datasets/tcga_mc3/features/features_burden_only.csv",
    "datasets/tcga_mc3/features/features_standard_sbs96_id83.csv.gz",
    "datasets/tcga_mc3/raw/mc3.v0.2.8.PUBLIC.maf.gz",
    "datasets/tcga_mc3/raw/TCGA-CDR-SupplementalTableS1.xlsx",
    "datasets/tcga_mc3/biology_labels.csv",
    "datasets/tcga_brca_hrd/cohort/final_analysis_cohort.tsv",
]


def log(msg: str) -> None:
    print(f"[task6] {msg}", flush=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_environment(report: dict) -> dict:
    """Record the actual runtime environment (never assumed)."""
    import importlib

    env = {
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "packages": {},
        "required_missing": [],
    }
    for module in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "yaml"):
        try:
            mod = importlib.import_module(module)
            env["packages"][module] = getattr(mod, "__version__", "installed")
        except Exception:
            env["packages"][module] = None
            env["required_missing"].append(module)
    for module in ("optuna", "sksurv", "torch", "pysam", "tabulate"):
        try:
            mod = importlib.import_module(module)
            env["packages"][module] = getattr(mod, "__version__", "installed")
        except Exception:
            env["packages"][module] = None
    env["known_deviation"] = (
        "Rerun environment is Python 3.12.3, not the manuscript run's Python 3.10 "
        "(see environment/table1_lockfile.txt sections A/B). xgboost 1.6.2 has no "
        "cp312 wheels, so the pinned environment cannot be installed here."
    )
    out = REPO_ROOT / "results" / "logs" / "task6_table1_environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=2), encoding="utf-8")
    log(f"environment audit -> {out.relative_to(REPO_ROOT)}")
    log(f"  python={env['python']} xgboost={env['packages'].get('xgboost')} "
        f"sklearn={env['packages'].get('sklearn')} pandas={env['packages'].get('pandas')}")
    if env["required_missing"]:
        raise SystemExit(f"missing required packages: {env['required_missing']}")
    report["environment"] = env
    return env


def stage_bundle(args: argparse.Namespace, report: dict) -> Path:
    """Resolve the dataset bundle, downloading the Table-1 subset if needed."""
    datasets_dir = Path(args.datasets_dir).expanduser().resolve() if args.datasets_dir else (REPO_ROOT.parent / "github_exports_datasets").resolve()
    if not datasets_dir.exists():
        if args.skip_bundle_fetch:
            raise SystemExit(f"dataset bundle not found: {datasets_dir} (--skip-bundle-fetch given)")
        log(f"bundle not found at {datasets_dir}; downloading Table-1 subset from Hugging Face")
        # Environment quirk (documented): this runtime exports NO_PROXY entries
        # like "[::1]" that huggingface_hub's vendored httpx2 cannot parse
        # ("Invalid port: ':1]'"). The egress proxy handles all traffic, so
        # sanitise NO_PROXY before importing huggingface_hub.
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
        os.environ["no_proxy"] = "localhost,127.0.0.1"
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise SystemExit("huggingface_hub is not installed; pip install huggingface_hub (or place the bundle manually)")
        download_root = datasets_dir.parent / "hf_bundle_task6"
        allow = [
            f"{HF_BUNDLE_PREFIX}/datasets/tcga_mc3/**",
            f"{HF_BUNDLE_PREFIX}/datasets/tcga_brca_hrd/**",
            f"{HF_BUNDLE_PREFIX}/resources/**",
            f"{HF_BUNDLE_PREFIX}/*.csv",
            f"{HF_BUNDLE_PREFIX}/*.json",
            f"{HF_BUNDLE_PREFIX}/*.md",
        ]
        t0 = time.time()
        snapshot_download(
            repo_id=args.hf_repo_id,
            repo_type="dataset",
            revision=args.hf_revision,
            local_dir=str(download_root),
            allow_patterns=allow,
            max_workers=4,
        )
        fetched = download_root / HF_BUNDLE_PREFIX
        if not fetched.exists():
            raise SystemExit(f"download finished but {fetched} is missing")
        datasets_dir.parent.mkdir(parents=True, exist_ok=True)
        if datasets_dir.exists():
            raise SystemExit(f"refusing to overwrite existing {datasets_dir}")
        shutil.move(str(fetched), str(datasets_dir))
        log(f"bundle download took {time.time() - t0:.1f}s -> {datasets_dir}")
        report["bundle_download_seconds"] = round(time.time() - t0, 1)
    else:
        log(f"using existing bundle: {datasets_dir}")
    report["datasets_dir"] = str(datasets_dir)
    report["hf_repo_id"] = args.hf_repo_id
    report["hf_revision"] = args.hf_revision or "main"

    # Verify key input hashes against the archived manifest where listed.
    manifest_path = REPO_ROOT / "manifests" / "dataset_assets_manifest.csv"
    manifest_hashes: dict[str, str] = {}
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                rel = (row.get("dataset_relative_path") or "").strip()
                h = (row.get("sha256") or "").strip()
                if rel and h:
                    manifest_hashes[rel] = h
    verification = []
    for rel in KEY_INPUTS:
        local = datasets_dir / rel
        entry: dict[str, object] = {"path": rel, "exists": local.exists(),
                                    "bytes": local.stat().st_size if local.exists() else 0}
        if local.exists():
            entry["sha256"] = sha256_of(local)
            # manifest rows are keyed by dataset_relative_path; match by suffix
            match = next((h for key, h in manifest_hashes.items() if key.endswith(rel) or rel.endswith(key)), None)
            entry["manifest_sha256"] = match
            entry["manifest_match"] = (match == entry["sha256"]) if match else None
        verification.append(entry)
        status = "ok" if local.exists() else "MISSING"
        extra = ""
        if local.exists() and entry.get("manifest_match") is False:
            extra = "  <-- HASH MISMATCH vs manifest (reported, not fatal)"
        log(f"  [bundle] {status} {rel} ({entry['bytes']} bytes){extra}")
    report["bundle_verification"] = verification
    missing = [v["path"] for v in verification if not v["exists"]]
    if missing:
        raise SystemExit(f"bundle is missing required Table-1 inputs: {missing}")
    return datasets_dir


def stage_oncokb(report: dict, skip_fetch: bool) -> dict:
    """Ensure the OncoKB role table exists; pin version/date/hash."""
    target = REPO_ROOT / "config" / "feature_resources" / "oncokb_cancer_genes.tsv"
    info: dict[str, object] = {
        "url": "https://public.api.oncokb.org/api/v1/utils/cancerGeneList",
        "target": str(target.relative_to(REPO_ROOT)),
        "note": ("OncoKB Terms of Use restrict redistribution, so the table is not "
                 "committed; it is fetched live at build time. The API is unversioned, "
                 "so a fresh download drifts from the snapshot that built the "
                 "published matrix (see oncokb_cancer_genes.PROVENANCE.md)."),
    }
    if not target.exists():
        if skip_fetch:
            raise SystemExit(f"OncoKB table missing: {target} (--skip-oncokb-fetch given)")
        log("OncoKB table missing; fetching live via config/feature_resources/fetch_oncokb_cancer_genes.py")
        subprocess.run([sys.executable, str(REPO_ROOT / "config" / "feature_resources" / "fetch_oncokb_cancer_genes.py"),
                        "--output", str(target)], check=True, cwd=str(REPO_ROOT))
        info["fetched_this_run"] = True
    else:
        info["fetched_this_run"] = False
    info["sha256"] = sha256_of(target)
    info["bytes"] = target.stat().st_size
    import pandas as pd
    frame = pd.read_csv(target, sep="\t")
    info["n_genes"] = int(frame.shape[0])
    info["download_date_utc"] = datetime.now(timezone.utc).date().isoformat()
    # Archive a dated snapshot copy with a sidecar (kept out of the way of the
    # pipeline, which reads only config/feature_resources/oncokb_cancer_genes.tsv).
    snap_dir = REPO_ROOT / "results" / "tables" / "task6_oncokb_snapshot"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_name = f"oncokb_cancer_genes_{info['download_date_utc']}.tsv"
    shutil.copy2(target, snap_dir / snap_name)
    (snap_dir / f"{snap_name}.json").write_text(json.dumps(
        {k: v for k, v in info.items() if k != "note"}, indent=2), encoding="utf-8")
    log(f"OncoKB pin: download_date={info['download_date_utc']} sha256={info['sha256']} n_genes={info['n_genes']}")
    log(f"  snapshot archived -> {(snap_dir / snap_name).relative_to(REPO_ROOT)}")
    report["oncokb"] = info
    return info


def stage_prerun_archive(report: dict) -> Path:
    """Move existing main-panel outputs aside so regeneration cannot reuse them.

    The production runner resumes completed (endpoint, representation, learner)
    slots from main_manuscript_complete_panel_endpoint_results.csv and resumes
    nested folds from the committed checkpoint files. Without this archive step
    a 'regeneration' in a checkout carrying those files would silently reuse the
    manuscript's own outputs. Archiving (not deleting) keeps everything
    recoverable and records sha256 of the manuscript artifacts.
    """
    tables_dir = REPO_ROOT / "results" / "tables"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = tables_dir / "task6_prerun_archive" / stamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = []
    for name in MAIN_PANEL_FILES:
        src = tables_dir / name
        if src.exists():
            entry = {"file": name, "sha256": sha256_of(src), "bytes": src.stat().st_size}
            shutil.move(str(src), str(archive_dir / name))
            archived.append(entry)
    # Fold checkpoints for the Table 1 slots only (other endpoints untouched).
    ckpt_dir = tables_dir / "main_manuscript_complete_panel_nested_fold_checkpoints"
    ckpt_arch = archive_dir / "nested_fold_checkpoints"
    n_ckpt = 0
    if ckpt_dir.exists():
        for child in sorted(ckpt_dir.glob("cancer_type_top20__*")):
            ckpt_arch.mkdir(parents=True, exist_ok=True)
            shutil.move(str(child), str(ckpt_arch / child.name))
            n_ckpt += 1
    # Manuscript reference: the 8 Table 1 rows, extracted BEFORE the move above
    # removed them (read from the archive copy).
    ref_rows = []
    ep_path = archive_dir / "main_manuscript_complete_panel_endpoint_results.csv"
    if ep_path.exists():
        with ep_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (row.get("endpoint"), row.get("representation"), row.get("learner")) in TABLE1_SLOTS:
                    ref_rows.append(row)
    ref_path = archive_dir / "table1_manuscript_reference.csv"
    if ref_rows:
        with ref_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ref_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ref_rows)
    log(f"prerun archive -> {archive_dir.relative_to(REPO_ROOT)} "
        f"({len(archived)} tables, {n_ckpt} checkpoints, {len(ref_rows)} reference rows)")
    report["prerun_archive"] = str(archive_dir.relative_to(REPO_ROOT))
    report["prerun_archived_tables"] = archived
    report["prerun_archived_checkpoints"] = n_ckpt
    report["manuscript_reference_rows"] = len(ref_rows)
    return archive_dir


def stage_run(args: argparse.Namespace, datasets_dir: Path, report: dict) -> dict:
    """Drive the production runner in-process on the Table-1 slice."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    # --datasets-dir must be on argv BEFORE run_all_experiments is imported
    # (its _preset_datasets_dir reads argv at import time).
    argv = [
        "run_all_experiments.py",
        "--config", str(TABLE1_CONFIG),
        "--paths", str(PATHS_CONFIG),
        "--datasets-dir", str(datasets_dir),
        "--only", "main_manuscript_complete_panel",
        "--skip-figures",
    ]
    if args.dry_run:
        argv.append("--dry-run")
    sys.argv = argv
    import run_all_experiments  # noqa: E402
    import runners.run_main_manuscript_complete_panel as panel  # noqa: E402

    # Documented adapter 1: skip the kucab feature build when the Table-1
    # config sets task6_skip_kucab. The production _load_kucab_features
    # unconditionally requires the 3.2 GB GRCh37 FASTA; Table 1 has no kucab
    # endpoint, so building it would be pure waste. (kucab_endpoints: [] can
    # not be used: the runner's _settings() treats [] as falsy and restores
    # ["damage_class"].)
    _orig_kucab = panel._load_kucab_features

    def _task6_kucab(ctx, settings):
        if settings.get("task6_skip_kucab"):
            log("task6_skip_kucab is set -> skipping kucab feature build (Table 1 has no kucab endpoint)")
            return None, {}
        return _orig_kucab(ctx, settings)

    panel._load_kucab_features = _task6_kucab

    # Timing wrapper 2 (behaviour-preserving): measure per-tumour Bio MAF
    # feature-building wall time for the compute-cost deliverable.
    _orig_maf = panel._load_mc3_maf_features
    build_times: dict[str, object] = {}

    def _timed_maf(ctx, model_id, settings=None):
        t0 = time.time()
        frame, cache_key = _orig_maf(ctx, model_id, settings)
        build_times[model_id] = {
            "wall_seconds": round(time.time() - t0, 3),
            "n_samples": int(frame.shape[0]),
            "n_features": int(frame.shape[1]),
            "per_tumour_seconds": round((time.time() - t0) / max(int(frame.shape[0]), 1), 6),
        }
        log(f"Bio MAF build {model_id}: {build_times[model_id]['wall_seconds']}s "
            f"for {frame.shape[0]} tumours x {frame.shape[1]} features "
            f"({build_times[model_id]['per_tumour_seconds']}s per tumour)")
        return frame, cache_key

    panel._load_mc3_maf_features = _timed_maf

    log(f"launching production runner: {' '.join(argv[1:])}")
    t0 = time.time()
    run_all_experiments.main()
    elapsed = round(time.time() - t0, 3)
    log(f"production runner finished in {elapsed}s")
    report["run_wall_seconds"] = elapsed
    report["bio_maf_build_times"] = build_times
    report["adapters"] = [
        "kucab feature build skipped when task6_skip_kucab is set (avoids GRCh37 FASTA requirement)",
        "timing wrapper around _load_mc3_maf_features (no behaviour change)",
    ]
    return report


def extract_table1() -> tuple[Path, list[dict]]:
    tables_dir = REPO_ROOT / "results" / "tables"
    src = tables_dir / "main_manuscript_complete_panel_endpoint_results.csv"
    if not src.exists():
        raise SystemExit(f"runner did not produce {src}")
    rows = []
    with src.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for row in reader:
            if (row.get("endpoint"), row.get("representation"), row.get("learner")) in TABLE1_SLOTS:
                rows.append(row)
    # Deterministic slot order.
    order = {slot: i for i, slot in enumerate(TABLE1_SLOTS)}
    rows.sort(key=lambda r: order[(r.get("endpoint"), r.get("representation"), r.get("learner"))])
    out = tables_dir / "task6_table1_regenerated.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log(f"extracted {len(rows)} Table-1 rows -> {out.relative_to(REPO_ROOT)}")
    return out, rows


def compare_table1(archive_dir: Path, regen_rows: list[dict], report: dict) -> Path:
    ref_path = archive_dir / "table1_manuscript_reference.csv"
    ref: dict[tuple[str, str, str], dict] = {}
    if ref_path.exists():
        with ref_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                ref[(row.get("endpoint"), row.get("representation"), row.get("learner"))] = row
    comparison = []
    for row in regen_rows:
        slot = (row.get("endpoint"), row.get("representation"), row.get("learner"))
        man = ref.get(slot, {})
        try:
            man_score = float(man.get(TABLE1_METRIC, "nan"))
        except (TypeError, ValueError):
            man_score = float("nan")
        try:
            new_score = float(row.get(TABLE1_METRIC, "nan"))
        except (TypeError, ValueError):
            new_score = float("nan")
        delta = new_score - man_score
        comparison.append({
            "endpoint": slot[0], "representation": slot[1], "learner": slot[2],
            "metric": TABLE1_METRIC,
            "manuscript_balanced_accuracy": man_score,
            "regenerated_balanced_accuracy": new_score,
            "delta": delta,
            "abs_delta": abs(delta),
            "within_tolerance": abs(delta) <= ENV_TOLERANCE,
            "manuscript_runtime_seconds": man.get("runtime_seconds", ""),
            "regenerated_runtime_seconds": row.get("runtime_seconds", ""),
        })
    out = REPO_ROOT / "results" / "tables" / "task6_table1_comparison.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)
    max_abs = max((c["abs_delta"] for c in comparison if c["abs_delta"] == c["abs_delta"]), default=float("nan"))
    n_within = sum(1 for c in comparison if c["within_tolerance"])
    log(f"comparison -> {out.relative_to(REPO_ROOT)}: max|delta|={max_abs:.6f} "
        f"tolerance={ENV_TOLERANCE} slots_within={n_within}/{len(comparison)}")
    report["comparison"] = {
        "max_abs_delta": max_abs,
        "tolerance": ENV_TOLERANCE,
        "slots_within_tolerance": n_within,
        "slots_total": len(comparison),
        "match": bool(n_within == len(comparison) == len(TABLE1_SLOTS)),
        "rows": comparison,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-dir", default=None,
                        help="Dataset bundle root (default: ../github_exports_datasets). Downloaded from Hugging Face if absent.")
    parser.add_argument("--hf-repo-id", default=HF_REPO_ID)
    parser.add_argument("--hf-revision", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Production dry-run only; no training.")
    parser.add_argument("--skip-bundle-fetch", action="store_true")
    parser.add_argument("--skip-oncokb-fetch", action="store_true")
    args = parser.parse_args()

    t_start = time.time()
    report: dict[str, object] = {
        "script": "scripts/reproduce_table1.py",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(TABLE1_CONFIG.relative_to(REPO_ROOT)),
        "table1_slots": ["/".join(s) for s in TABLE1_SLOTS],
        "tolerance": ENV_TOLERANCE,
    }
    log("stage 1/7: environment audit")
    stage_environment(report)
    log("stage 2/7: dataset bundle")
    datasets_dir = stage_bundle(args, report)
    log("stage 3/7: OncoKB pin")
    stage_oncokb(report, skip_fetch=args.skip_oncokb_fetch)
    if args.dry_run:
        log("stage 4/7: dry-run of the production runner (no archive, no training)")
        stage_run(args, datasets_dir, report)
    else:
        log("stage 4/7: prerun archive (force genuine regeneration)")
        archive_dir = stage_prerun_archive(report)
        log("stage 5/7: production run (Table-1 slice)")
        stage_run(args, datasets_dir, report)
        log("stage 6/7: extract Table 1")
        _, regen_rows = extract_table1()
        log("stage 7/7: compare against manuscript values")
        compare_table1(archive_dir, regen_rows, report)
    report["total_wall_seconds"] = round(time.time() - t_start, 3)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out = REPO_ROOT / "results" / "logs" / "task6_table1_run_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log(f"run report -> {out.relative_to(REPO_ROOT)} (total {report['total_wall_seconds']}s)")
    print(json.dumps({
        "completed": True,
        "dry_run": args.dry_run,
        "total_wall_seconds": report["total_wall_seconds"],
        "comparison": report.get("comparison", "dry-run: no comparison"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
