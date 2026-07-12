"""Master driver for the UGA/MAF representation paper bundle."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _preset_datasets_dir(argv: list[str]) -> None:
    """Honour --datasets-dir before utils.config is imported.

    utils.config computes DATASETS_ROOT at import time from CGR_DATASETS_DIR, falling back to
    ../github_exports_datasets. Anything parsed by argparse would therefore arrive too late. If we
    do not set the environment variable here, an incorrect bundle location is not reported at all:
    validate_environment() records the paths as "missing" and the run proceeds until a runner dies
    inside pandas with a FileNotFoundError several minutes later.
    """

    for index, token in enumerate(argv):
        if token == "--datasets-dir" and index + 1 < len(argv):
            os.environ["CGR_DATASETS_DIR"] = str(Path(argv[index + 1]).expanduser().resolve())
            return
        if token.startswith("--datasets-dir="):
            os.environ["CGR_DATASETS_DIR"] = str(Path(token.split("=", 1)[1]).expanduser().resolve())
            return


_preset_datasets_dir(sys.argv[1:])

from utils.config import BUNDLE_ROOT, DATASETS_ROOT, REPRO_ROOT, enabled, load_yaml, resolve_paths_map  # noqa: E402
from utils.runner_support import RunnerContext, ensure_output_dirs  # noqa: E402


RUNNERS = {
    "main_manuscript_complete_panel": "runners.run_main_manuscript_complete_panel",
    "muat_style_tcga_comparator": "runners.run_muat_style_tcga_comparator",
}


def load_runner(experiment_id: str):
    module = importlib.import_module(RUNNERS[experiment_id])
    return module.run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment settings YAML.")
    parser.add_argument("--paths", default="config/paths.yaml", help="Raw-data path YAML.")
    parser.add_argument(
        "--datasets-dir",
        default=None,
        help="Dataset bundle root (the folder containing datasets/, references/, resources/). "
             "Equivalent to setting CGR_DATASETS_DIR. Defaults to ../github_exports_datasets.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths and log planned commands without running expensive jobs.")
    parser.add_argument("--only", choices=sorted(RUNNERS), nargs="*", help="Optional subset of experiment ids to run.")
    parser.add_argument("--skip-figures", action="store_true", help="Do not regenerate manuscript tables/figures after experiment runners.")
    parser.add_argument("--refresh-cache", action="store_true", help="Rebuild persistent feature caches instead of reusing them.")
    return parser.parse_args()


def _path_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _display_path(path: str | Path | None) -> str:
    if path is None:
        return ""
    resolved = Path(path).resolve()
    for label, root in (("${BUNDLE_ROOT}", BUNDLE_ROOT), ("${REPRO_ROOT}", REPRO_ROOT), ("${DATASETS_ROOT}", DATASETS_ROOT)):
        try:
            relative = resolved.relative_to(root.resolve()).as_posix()
            return label if relative in {"", "."} else f"{label}/{relative}"
        except ValueError:
            continue
    return f"<external>/{resolved.name}"


def validate_environment(settings: dict, paths: dict, ctx: RunnerContext) -> list[dict[str, object]]:
    """Run cheap reproducibility checks before expensive work."""
    rows: list[dict[str, object]] = []
    required_modules = ["yaml", "numpy", "pandas", "sklearn", "xgboost", "optuna", "sksurv", "torch"]
    for module in required_modules:
        rows.append({"kind": "python_module", "name": module, "status": "ok" if importlib.util.find_spec(module) else "missing"})
    rows.append(
        {
            "kind": "optional_python_module",
            "name": "pysam",
            "status": "ok" if importlib.util.find_spec("pysam") else "optional_missing",
            "note": "Optional; active strict manuscript runners do not require pysam.",
        }
    )
    rows.append({"kind": "python_executable", "name": _display_path(sys.executable), "status": "ok"})
    rows.append({"kind": "cpu_count", "name": "os.cpu_count", "status": "ok", "value": os.cpu_count() or 1})
    for section_name in ("raw_data", "processed_helpers", "workspace"):
        for key, value in (paths.get(section_name) or {}).items():
            if value is None:
                rows.append({"kind": "path", "name": f"{section_name}.{key}", "status": "optional_null"})
                continue
            path = Path(value)
            status = "ok" if path.exists() else "missing"
            inside = _path_inside(path, REPRO_ROOT)
            rows.append({"kind": "path", "name": f"{section_name}.{key}", "status": status, "inside_repro_root": inside, "path": _display_path(path)})
    fasta_dir = Path((paths.get("raw_data") or {}).get("grch37_dir") or "")
    fasta_candidates = []
    compressed_fasta_candidates = []
    if fasta_dir.exists():
        fasta_candidates = sorted(fasta_dir.rglob("*.fna")) + sorted(fasta_dir.rglob("*.fa")) + sorted(fasta_dir.rglob("*.fasta"))
        compressed_fasta_candidates = sorted(fasta_dir.rglob("*.fna.gz")) + sorted(fasta_dir.rglob("*.fa.gz")) + sorted(fasta_dir.rglob("*.fasta.gz"))
    fasta = fasta_candidates[0] if fasta_candidates else None
    compressed_fasta = compressed_fasta_candidates[0] if compressed_fasta_candidates else None
    if fasta:
        rows.append({"kind": "fasta", "name": "GRCh37", "status": "ok", "path": _display_path(fasta)})
    elif compressed_fasta:
        fasta = compressed_fasta.with_suffix("")
        rows.append(
            {
                "kind": "fasta",
                "name": "GRCh37",
                "status": "ok_compressed",
                "path": _display_path(fasta),
                "compressed_path": _display_path(compressed_fasta),
                "note": "Active strict manuscript runners use the configured FASTA resources when needed.",
            }
        )
    else:
        rows.append({"kind": "fasta", "name": "GRCh37", "status": "missing", "path": ""})
    if fasta:
        fai = fasta.parent / f"{fasta.name}.fai"
        rows.append({"kind": "fasta_index", "name": ".fai", "status": "ok" if fai.exists() else "missing", "path": _display_path(fai)})
    rows.append({"kind": "feature_cache", "name": "dir", "status": "ok", "path": _display_path(ctx.feature_cache_dir), "refresh_cache": bool(ctx.refresh_cache)})
    node = shutil.which("node")
    if node is None:
        bundled_node = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "bin" / ("node.exe" if sys.platform.startswith("win") else "node")
        node = str(bundled_node) if bundled_node.exists() else None
    rows.append({"kind": "renderer", "name": "node", "status": "ok" if node else "missing", "path": _display_path(node) if node else ""})
    renderer_script = BUNDLE_ROOT / "src" / "visualization" / "render_manuscript_figures.mjs"
    rows.append({"kind": "renderer", "name": "manuscript_renderer", "status": "ok" if renderer_script.exists() else "missing", "path": _display_path(renderer_script)})
    return rows


def main() -> None:
    args = parse_args()
    settings = load_yaml(args.config)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    settings["_run_id"] = run_id
    paths = resolve_paths_map(load_yaml(args.paths))
    ctx = RunnerContext(settings=settings, paths=paths, dry_run=bool(args.dry_run), refresh_cache=bool(args.refresh_cache))
    ensure_output_dirs(ctx)
    env_rows = validate_environment(settings, paths, ctx)
    env_path = ctx.logs_dir / ("dry_run_environment_checks.csv" if args.dry_run else "environment_checks.csv")
    env_fields = sorted({key for row in env_rows for key in row})
    with env_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=env_fields)
        writer.writeheader()
        writer.writerows(env_rows)
    missing_modules = [row for row in env_rows if row.get("status") == "missing" and row.get("kind") == "python_module"]
    blocking = [row for row in env_rows if row.get("status") == "missing" and row.get("kind") in {"path", "fasta", "fasta_index"}]
    if args.dry_run and missing_modules:
        print(json.dumps({"environment_check": "module_advisory", "missing_python_modules": missing_modules}, indent=2, default=str), flush=True)
    if blocking and args.dry_run:
        print(json.dumps({"environment_check": "failed", "missing": blocking}, indent=2, default=str), flush=True)
        raise SystemExit("dry-run environment check failed: missing required paths/assets")

    # Fail fast on a mis-resolved dataset bundle. This check previously ran only under --dry-run,
    # so a wrong --datasets-dir / CGR_DATASETS_DIR was recorded as "missing" in environment_checks
    # and the run continued regardless, until a runner raised FileNotFoundError inside pandas
    # minutes later. Both runners read mc3_source_dir unconditionally, so its absence is fatal.
    # Everything else (kucab, pcawg, grch37) is endpoint-dependent and stays advisory.
    print(f"[env] datasets_root = {DATASETS_ROOT}", flush=True)
    if not args.dry_run:
        fatal: list[str] = []
        if not DATASETS_ROOT.exists():
            fatal.append(f"dataset bundle directory does not exist: {DATASETS_ROOT}")
        mc3_dir = (paths.get("raw_data") or {}).get("mc3_source_dir")
        if mc3_dir is None or not Path(mc3_dir).exists():
            fatal.append(f"raw_data.mc3_source_dir does not exist: {mc3_dir}")
        if fatal:
            for item in fatal:
                print(f"[env] FATAL  {item}", flush=True)
            raise SystemExit(
                "\nThe dataset bundle could not be resolved.\n"
                "  Point at it explicitly :  --datasets-dir <bundle>   (or set CGR_DATASETS_DIR)\n"
                "  Fetch it               :  python scripts/fetch_datasets.py --minimal\n"
                f"  Currently resolved to  :  {DATASETS_ROOT}\n"
            )
        if blocking:
            names = sorted({str(row.get("name")) for row in blocking})
            print(f"[env] advisory: paths not present (fine unless an enabled endpoint needs them): {names}", flush=True)

    started = time.time()
    selected = set(args.only or RUNNERS)
    rows = []
    for experiment_id in RUNNERS:
        if experiment_id not in selected or not enabled(settings, experiment_id):
            rows.append({"experiment_id": experiment_id, "status": "skipped"})
            continue
        t0 = time.time()
        print(f"[{experiment_id}] starting", flush=True)
        if args.dry_run:
            rows.append({"experiment_id": experiment_id, "status": "planned", "elapsed_seconds": round(time.time() - t0, 3)})
            print(f"[{experiment_id}] planned", flush=True)
            continue
        runner = load_runner(experiment_id)
        runner(ctx)
        rows.append({"experiment_id": experiment_id, "status": "planned" if args.dry_run else "completed", "elapsed_seconds": round(time.time() - t0, 3)})
        print(f"[{experiment_id}] done", flush=True)

    if args.dry_run:
        rows.append({"experiment_id": "make_all_figures", "status": "skipped_dry_run"})
    elif not args.skip_figures and (settings.get("outputs") or {}).get("make_figures_after_run", True):
        from utils.make_all_figures import make_all_figures

        make_all_figures(settings=settings, paths=paths)
        rows.append({"experiment_id": "make_all_figures", "status": "completed"})

    if not args.dry_run and (settings.get("outputs") or {}).get("cleanup_work_dir_after_run", True):
        work_dir = Path(paths["workspace"]["work_dir"]).resolve()
        if not _path_inside(work_dir, BUNDLE_ROOT):
            raise RuntimeError(f"Refusing to cleanup work_dir outside export root: {work_dir}")
        shutil.rmtree(work_dir, ignore_errors=True)

    run_manifest = {
        "completed_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dry_run": bool(args.dry_run),
        "config": args.config,
        "paths": args.paths,
        "runtime_profile": settings.get("runtime_profile"),
        "run_id": run_id,
        "repro_root": _display_path(REPRO_ROOT),
        "bundle_root": _display_path(BUNDLE_ROOT),
        "refresh_cache": bool(args.refresh_cache),
        "feature_cache_dir": _display_path(ctx.feature_cache_dir),
        "tree_method": settings.get("tree_method"),
        "elapsed_seconds": round(time.time() - started, 3),
        "experiments": rows,
        "active_runners": sorted(RUNNERS),
    }
    manifest_name = "dry_run_all_experiments_manifest.json" if args.dry_run else "run_all_experiments_manifest.json"
    manifest_path = ctx.logs_dir / manifest_name
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
