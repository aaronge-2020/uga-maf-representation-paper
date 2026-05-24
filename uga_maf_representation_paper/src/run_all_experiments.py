"""Master driver for the UGA/MAF representation paper bundle."""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from runners import (
    run_main_manuscript_complete_panel,
    run_maf_event_gene_locus_stack,
    run_mechanistic_representation,
    run_one_hot_event_kme_scout,
    run_uga_variants_kme,
    run_unified_locked_benchmark,
)
from utils.config import BUNDLE_ROOT, REPRO_ROOT, enabled, load_yaml, resolve_paths_map
from utils.runner_support import RunnerContext, ensure_output_dirs


RUNNERS = {
    "unified_locked": run_unified_locked_benchmark.run,
    "uga_variants_kme": run_uga_variants_kme.run,
    "maf_event_gene_locus": run_maf_event_gene_locus_stack.run,
    "main_manuscript_complete_panel": run_main_manuscript_complete_panel.run,
    "mechanistic_representation": run_mechanistic_representation.run,
    "one_hot_event_kme_scout": run_one_hot_event_kme_scout.run,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment settings YAML.")
    parser.add_argument("--paths", default="config/paths.yaml", help="Raw-data path YAML.")
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


def _find_uncompressed_fasta(fasta_dir: Path) -> Path | None:
    if not fasta_dir.exists():
        return None
    candidates = sorted(fasta_dir.rglob("*.fna")) + sorted(fasta_dir.rglob("*.fa")) + sorted(fasta_dir.rglob("*.fasta"))
    return candidates[0] if candidates else None


def _find_compressed_fasta(fasta_dir: Path) -> Path | None:
    if not fasta_dir.exists():
        return None
    candidates = sorted(fasta_dir.rglob("*.fna.gz")) + sorted(fasta_dir.rglob("*.fa.gz")) + sorted(fasta_dir.rglob("*.fasta.gz"))
    return candidates[0] if candidates else None


def materialize_reference_fasta(paths: dict) -> Path | None:
    """Expand a bundled compressed FASTA when the indexed FASTA is not present."""
    fasta_dir = Path((paths.get("raw_data") or {}).get("grch37_dir") or "")
    fasta = _find_uncompressed_fasta(fasta_dir)
    if fasta is not None:
        return fasta
    archive = _find_compressed_fasta(fasta_dir)
    if archive is None:
        return None
    target = archive.with_suffix("")
    print(f"[reference] materializing {target.name} from bundled {archive.name}", flush=True)
    with gzip.open(archive, "rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    return target


def validate_environment(settings: dict, paths: dict, ctx: RunnerContext) -> list[dict[str, object]]:
    """Run cheap reproducibility checks before expensive work."""
    rows: list[dict[str, object]] = []
    required_modules = ["yaml", "numpy", "pandas", "sklearn", "xgboost", "optuna", "pysam"]
    for module in required_modules:
        rows.append({"kind": "python_module", "name": module, "status": "ok" if importlib.util.find_spec(module) else "missing"})
    rows.append({"kind": "python_executable", "name": sys.executable, "status": "ok"})
    rows.append({"kind": "cpu_count", "name": "os.cpu_count", "status": "ok", "value": os.cpu_count() or 1})
    for section_name in ("raw_data", "processed_helpers", "workspace"):
        for key, value in (paths.get(section_name) or {}).items():
            if value is None:
                rows.append({"kind": "path", "name": f"{section_name}.{key}", "status": "optional_null"})
                continue
            path = Path(value)
            status = "ok" if path.exists() else "missing"
            inside = _path_inside(path, REPRO_ROOT)
            rows.append({"kind": "path", "name": f"{section_name}.{key}", "status": status, "inside_repro_root": inside, "path": str(path)})
    fasta_dir = Path((paths.get("raw_data") or {}).get("grch37_dir") or "")
    fasta = _find_uncompressed_fasta(fasta_dir)
    archive = _find_compressed_fasta(fasta_dir) if fasta is None else None
    if fasta is None and archive is not None:
        fasta = archive.with_suffix("")
        rows.append({"kind": "fasta_archive", "name": "GRCh37 compressed FASTA", "status": "ok", "path": str(archive), "materializes_to": str(fasta)})
    rows.append({"kind": "fasta", "name": "GRCh37", "status": "ok" if fasta and fasta.exists() else ("compressed_available" if archive else "missing"), "path": str(fasta) if fasta else ""})
    if fasta:
        fai = fasta.parent / f"{fasta.name}.fai"
        rows.append({"kind": "fasta_index", "name": ".fai", "status": "ok" if fai.exists() else "missing", "path": str(fai)})
    rows.append({"kind": "feature_cache", "name": "dir", "status": "ok", "path": str(ctx.feature_cache_dir), "refresh_cache": bool(ctx.refresh_cache)})
    node = shutil.which("node")
    if node is None:
        bundled_node = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "bin" / ("node.exe" if sys.platform.startswith("win") else "node")
        node = str(bundled_node) if bundled_node.exists() else None
    rows.append({"kind": "renderer", "name": "node", "status": "ok" if node else "missing", "path": node or ""})
    rows.append({"kind": "renderer", "name": "d3_vendor", "status": "ok" if (BUNDLE_ROOT / "src" / "legacy_source" / "cgr_validation" / "cgr_validation_results" / "research" / "experiments" / "supporting" / "2026_05_14_unified_uga_locked_manuscript_benchmark" / "code" / "d3.v7.min.js").exists() else "missing"})
    return rows


def main() -> None:
    args = parse_args()
    settings = load_yaml(args.config)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    settings["_run_id"] = run_id
    paths = resolve_paths_map(load_yaml(args.paths))
    ctx = RunnerContext(settings=settings, paths=paths, dry_run=bool(args.dry_run), refresh_cache=bool(args.refresh_cache))
    if not args.dry_run:
        materialize_reference_fasta(paths)
    ensure_output_dirs(ctx)
    env_rows = validate_environment(settings, paths, ctx)
    env_path = ctx.logs_dir / ("dry_run_environment_checks.csv" if args.dry_run else "environment_checks.csv")
    env_fields = sorted({key for row in env_rows for key in row})
    with env_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=env_fields)
        writer.writeheader()
        writer.writerows(env_rows)
    blocking = [row for row in env_rows if row.get("status") == "missing" and row.get("kind") in {"python_module", "path", "fasta", "fasta_index"}]
    if blocking and args.dry_run:
        print(json.dumps({"environment_check": "failed", "missing": blocking}, indent=2, default=str), flush=True)

    started = time.time()
    selected = set(args.only or RUNNERS)
    rows = []
    for experiment_id, runner in RUNNERS.items():
        if experiment_id not in selected or not enabled(settings, experiment_id):
            rows.append({"experiment_id": experiment_id, "status": "skipped"})
            continue
        t0 = time.time()
        print(f"[{experiment_id}] starting", flush=True)
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
        shutil.rmtree(Path(paths["workspace"]["work_dir"]), ignore_errors=True)

    run_manifest = {
        "completed_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dry_run": bool(args.dry_run),
        "config": args.config,
        "paths": args.paths,
        "runtime_profile": settings.get("runtime_profile"),
        "run_id": run_id,
        "repro_root": str(REPRO_ROOT),
        "bundle_root": str(BUNDLE_ROOT),
        "refresh_cache": bool(args.refresh_cache),
        "feature_cache_dir": str(ctx.feature_cache_dir),
        "tree_method": settings.get("tree_method"),
        "elapsed_seconds": round(time.time() - started, 3),
        "experiments": rows,
    }
    manifest_path = ctx.logs_dir / "run_all_experiments_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
