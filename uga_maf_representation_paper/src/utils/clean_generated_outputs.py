"""Safely remove generated result artifacts before a fresh manuscript run."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import BUNDLE_ROOT, REPRO_ROOT, load_yaml, resolve_paths_map


KEEP_NAMES = {".gitignore", ".gitkeep"}
DEFAULT_REL_TARGETS = [
    "results/tables",
    "results/figures",
    "results/logs",
    "results/work",
    "results/manuscript",
]
LEGACY_REL_TARGETS = [
    "cgr_validation_results/research/reports",
    "cgr_validation_results/research/results",
    "cgr_validation_results/research/scratch",
    "docs/poster/generated_figures",
]


def _resolve_targets(paths: dict, *, include_legacy: bool) -> list[Path]:
    workspace = paths.get("workspace", {})
    configured = [
        workspace.get("results_tables_dir"),
        workspace.get("results_figures_dir"),
        workspace.get("results_logs_dir"),
        workspace.get("work_dir"),
        BUNDLE_ROOT / "results" / "manuscript",
        REPRO_ROOT.parent.parent / "outputs",
    ]
    targets = [Path(path).resolve() for path in configured if path is not None]
    if include_legacy:
        targets.extend((REPRO_ROOT / rel).resolve() for rel in LEGACY_REL_TARGETS)
    return sorted(set(targets))


def _allowed(path: Path, targets: list[Path]) -> bool:
    resolved = path.resolve()
    for target in targets:
        if resolved == target or target in resolved.parents:
            return True
    return False


def _inventory(target: Path) -> list[dict[str, object]]:
    if not target.exists():
        return [{"path": str(target), "kind": "missing", "bytes": 0}]
    rows: list[dict[str, object]] = []
    if target.is_file():
        if target.name not in KEEP_NAMES:
            rows.append({"path": str(target), "kind": "file", "bytes": target.stat().st_size})
        return rows
    if target.resolve() == (BUNDLE_ROOT / "results" / "work").resolve():
        for item in sorted(target.iterdir()):
            if item.name in KEEP_NAMES:
                continue
            if item.is_dir() and not item.is_symlink():
                size = sum(int(p.stat().st_size) for p in item.rglob("*") if p.is_file() and not p.is_symlink())
                rows.append({"path": str(item), "kind": "tree", "bytes": size})
            elif item.exists():
                rows.append({"path": str(item), "kind": "file", "bytes": item.stat().st_size if item.is_file() else 0})
        return rows
    for item in sorted(target.rglob("*")):
        if item.name in KEEP_NAMES:
            continue
        if item.is_file():
            rows.append({"path": str(item), "kind": "file", "bytes": item.stat().st_size})
    for item in sorted([p for p in target.rglob("*") if p.is_dir()], reverse=True):
        if any(item.iterdir()):
            continue
        rows.append({"path": str(item), "kind": "empty_dir", "bytes": 0})
    return rows


def clean(paths: dict, *, execute: bool, include_legacy: bool) -> dict[str, object]:
    targets = _resolve_targets(paths, include_legacy=include_legacy)
    rows: list[dict[str, object]] = []
    for target in targets:
        rows.extend(_inventory(target))
    delete_rows = [row for row in rows if row["kind"] in {"file", "empty_dir", "tree"}]
    if execute:
        for row in sorted(delete_rows, key=lambda r: str(r["path"]), reverse=True):
            path = Path(str(row["path"]))
            if not _allowed(path, targets):
                raise RuntimeError(f"Refusing to delete outside allowlist: {path}")
            if path.is_dir():
                if row["kind"] == "tree":
                    shutil.rmtree(path)
                    continue
                try:
                    path.rmdir()
                except OSError:
                    pass
            elif path.exists():
                path.unlink()
        for target in targets:
            if target.suffix:
                continue
            target.mkdir(parents=True, exist_ok=True)
        for rel in DEFAULT_REL_TARGETS[:3]:
            directory = (BUNDLE_ROOT / rel).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            for keep in KEEP_NAMES:
                keep_path = directory / keep
                if not keep_path.exists():
                    keep_path.write_text("*\n!.gitignore\n!.gitkeep\n" if keep == ".gitignore" else "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "mode": "execute" if execute else "dry_run",
        "include_legacy": include_legacy,
        "target_count": len(targets),
        "artifact_count": len(delete_rows),
        "total_bytes": int(sum(int(row.get("bytes") or 0) for row in delete_rows)),
        "targets": [str(target) for target in targets],
        "artifacts": delete_rows,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--execute", action="store_true", help="Actually delete files. Omit for dry-run.")
    parser.add_argument("--include-legacy", action="store_true", help="Also clear approved legacy generated report/result/scratch folders.")
    parser.add_argument("--obsolete", action="store_true", help="Use the obsolete-cleanup manifest name for repo-consolidation review.")
    parser.add_argument("--manifest", default=None, help="Optional manifest JSON path.")
    args = parser.parse_args()
    paths = resolve_paths_map(load_yaml(args.paths))
    manifest = clean(paths, execute=bool(args.execute), include_legacy=bool(args.include_legacy))
    if args.manifest:
        manifest_path = Path(args.manifest)
    elif args.obsolete and not args.execute:
        manifest_path = BUNDLE_ROOT / "results" / "logs" / "obsolete_cleanup_dry_run_manifest.json"
    elif args.obsolete:
        manifest_path = BUNDLE_ROOT / "results" / "logs" / "obsolete_cleanup_execute_manifest.json"
    else:
        manifest_path = BUNDLE_ROOT / "results" / "logs" / ("cleanup_execute_manifest.json" if args.execute else "cleanup_dry_run_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["mode", "include_legacy", "target_count", "artifact_count", "total_bytes"]}, indent=2), flush=True)
    if not args.execute:
        print(f"Dry-run only. Re-run with --execute after reviewing {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
