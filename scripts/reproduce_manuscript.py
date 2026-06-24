#!/usr/bin/env python
"""Strictly validate the standalone manuscript export."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "manifests"
DEFAULT_DATASETS_DIR = ROOT.parent / "github_exports_datasets"
READ_LOG: list[str] = []
DATASET_READ_LOG: list[str] = []
GITHUB_SIZE_THRESHOLD_BYTES = 30 * 1024 * 1024
MANUSCRIPT_OUTPUT_PREFIX = "results/manuscript/"
MANUSCRIPT_POLICY_FILES = {
    "results/manuscript/.gitignore",
    "results/manuscript/.gitkeep",
}
IGNORED_GENERATED_FILES = {
    "manifests/manuscript_artifact_manifest.csv",
}
MAIN_ENDPOINT_RESULTS_PATH = ROOT / "results" / "tables" / "main_manuscript_complete_panel_endpoint_results.csv"
DIRECT_STRICT_COMMAND = "python scripts/reproduce_manuscript.py --datasets-dir ../github_exports_datasets --strict"
LOCAL_STRICT_COMMAND = "python scripts/reproduce_manuscript.py --strict"
REVIEWER_SETUP_COMMAND = "python scripts/reviewer_workflow.py check --datasets-dir ../github_exports_datasets"
REVIEWER_FAST_VALIDATION_COMMAND = "python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets --dry-run"
REVIEWER_FULL_REPRODUCTION_COMMAND = "python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets"

TEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}

EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "w",
}

NOISE_FILENAMES = {
    ".DS_Store",
    "desktop.ini",
    "Thumbs.db",
}

RESTORABLE_CACHE_GLOBS = [
]
REMOVED_FEATURE_OPTIMIZER_TOKEN = "sc" + "out"
REMOVED_EVENT_RUNNER_TOKEN = "run_" + "one_" + "hot_" + "event"
REMOVED_EVENT_KME_TOKEN = "one_" + "hot_" + "event_" + "kme_"

FORBIDDEN_PATTERNS = {
    "external_projects_cgr_validation": re.compile(r"projects[\\/]+cgr_validation", re.IGNORECASE),
    "absolute_drive_or_my_drive": re.compile(r"(?<![A-Za-z])[A-Za-z]:\\|My Drive", re.IGNORECASE),
    "nested_old_export_root": re.compile(r"uga-maf-representation-paper", re.IGNORECASE),
    "retired_top10_endpoint": re.compile(r"cancer_type_top10|top-10|\btop10\b", re.IGNORECASE),
    "retired_hash_maf": re.compile(
        r"hash[_ -]?MAF|hash_maf|maf_bio_hash|quick_bio_v3_vs_hash|hash_v1|frozen_biological_v2|maf_hash",
        re.IGNORECASE,
    ),
    "removed_feature_optimization": re.compile(
        REMOVED_EVENT_RUNNER_TOKEN
        + r"|"
        + REMOVED_EVENT_KME_TOKEN
        + REMOVED_FEATURE_OPTIMIZER_TOKEN
        + r"|feature[-_ ]?optimization "
        + REMOVED_FEATURE_OPTIMIZER_TOKEN
        + r"|\b"
        + REMOVED_FEATURE_OPTIMIZER_TOKEN
        + r"\b",
        re.IGNORECASE,
    ),
}

TEXT_SCAN_EXCLUDED_PREFIXES = (
    "results/cache/",
    "results/logs/",
    MANUSCRIPT_OUTPUT_PREFIX,
)

FORBIDDEN_PATTERN_PATH_EXEMPTIONS = {
    "removed_feature_optimization": {
        "manifests/dataset_assets_manifest.csv",
        "manifests/huggingface_asset_manifest.csv",
    },
}


def rel(path: Path) -> str:
    resolved = path.resolve()
    base = ROOT.resolve()
    ensure_within(resolved, base, f"path escapes repository root: {resolved}")
    return Path(os.path.relpath(comparable_path(resolved), comparable_path(base))).as_posix()


def display_datasets_dir(path: Path) -> str:
    resolved = path.resolve()
    try:
        return "../" + resolved.relative_to(ROOT.parent.resolve()).as_posix()
    except ValueError:
        return f"<external-datasets-dir>/{resolved.name}"


def comparable_path(path: Path) -> str:
    text = str(path.resolve())
    if text.startswith("\\\\?\\UNC\\"):
        text = "\\\\" + text[8:]
    elif text.startswith("\\\\?\\"):
        text = text[4:]
    return os.path.normcase(os.path.normpath(text))


def ensure_within(path: Path, base: Path, message: str) -> None:
    path_text = comparable_path(path)
    base_text = comparable_path(base)
    try:
        common = os.path.commonpath([path_text, base_text])
    except ValueError as exc:
        raise RuntimeError(message) from exc
    if common != base_text:
        raise RuntimeError(message)


def dataset_rel(datasets_dir: Path, path: Path) -> str:
    resolved = path.resolve()
    base = datasets_dir.resolve()
    ensure_within(resolved, base, f"dataset read escaped datasets directory: {resolved}")
    return Path(os.path.relpath(comparable_path(resolved), comparable_path(base))).as_posix()


def repo_manifest_path(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    ensure_within(path, ROOT.resolve(), f"manifest path escapes repository root: {relative_path}")
    return path


def dataset_manifest_path(datasets_dir: Path, relative_path: str) -> Path:
    base = datasets_dir.resolve()
    path = (base / relative_path).resolve()
    ensure_within(path, base, f"dataset manifest path escapes datasets directory: {relative_path}")
    return path


def manuscript_asset_path(relative_path: str) -> Path:
    base = (ROOT / "results" / "manuscript").resolve()
    path = (base / relative_path).resolve()
    ensure_within(path, base, f"manuscript asset path escapes manuscript directory: {relative_path}")
    return path


def read_bytes(path: Path) -> bytes:
    path = path.resolve()
    READ_LOG.append(rel(path))
    return path.read_bytes()


def read_text(path: Path) -> str:
    return read_bytes(path).decode("utf-8-sig", errors="replace")


def sha256_file(path: Path) -> str:
    path = path.resolve()
    READ_LOG.append(rel(path))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_external_file(path: Path, *, audit_root: Path | None = None) -> str:
    if audit_root is not None:
        DATASET_READ_LOG.append(dataset_rel(audit_root, path))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.name in NOISE_FILENAMES:
            continue
        parts = set(path.relative_to(ROOT).parts)
        if parts & EXCLUDED_PARTS:
            continue
        relative = rel(path)
        if relative in IGNORED_GENERATED_FILES:
            continue
        if relative.startswith(MANUSCRIPT_OUTPUT_PREFIX) and relative not in MANUSCRIPT_POLICY_FILES:
            continue
        files.append(path)
    return sorted(files, key=lambda p: rel(p))


def text_files() -> list[Path]:
    return [
        path
        for path in active_files()
        if path.suffix.lower() in TEXT_EXTENSIONS
        and not rel(path).startswith(TEXT_SCAN_EXCLUDED_PREFIXES)
    ]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        READ_LOG.append(rel(path))
        return list(csv.DictReader(handle))


def read_dataset_csv(path: Path, datasets_dir: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        DATASET_READ_LOG.append(dataset_rel(datasets_dir, path))
        return list(csv.DictReader(handle))


def read_json(path: Path) -> object:
    return json.loads(read_text(path))


def file_role(path: Path) -> str:
    relative = rel(path)
    if relative.startswith("results/manuscript/canonical/"):
        return "canonical manuscript result"
    if relative.startswith("results/manuscript/tables/"):
        return "manuscript table"
    if relative.startswith("results/manuscript/supplement/"):
        return "supplement artifact"
    if relative.startswith("results/manuscript/figures/"):
        return "manuscript figure"
    if relative.startswith("results/manuscript/interpretability/"):
        return "Bio MAF v4 interpretability artifact"
    if relative.startswith("results/manuscript/text/"):
        return "manuscript text"
    return "manuscript artifact"


def generate_manuscript_manifest() -> None:
    rows = []
    for path in sorted((ROOT / "results" / "manuscript").rglob("*")):
        if not path.is_file():
            continue
        if path.name in NOISE_FILENAMES:
            continue
        rows.append(
            {
                "path": rel(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": file_role(path),
            }
        )
    write_csv(MANIFEST_DIR / "manuscript_artifact_manifest.csv", rows, ["path", "bytes", "sha256", "role"])


def source_category(path: Path) -> str:
    name = path.name
    if name.startswith("main_manuscript_complete_panel"):
        return "Main complete panel"
    if name.startswith("muat_style_tcga_comparator"):
        return "MuAt-compatible comparator"
    if name.startswith("proposed_clinical_bio_v4") or name.startswith("quick_bio_v4"):
        return "Bio MAF v4 source artifacts"
    return "Other regenerated source artifacts"


def table_shape(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".tsv"}:
        return (0, 0)
    delimiter = "\t" if suffix == ".tsv" else ","
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
            return rows, len(header)
    except UnicodeDecodeError:
        return (0, 0)


def html_table(title: str, rows: list[dict[str, object]], fields: list[str]) -> str:
    body = "\n".join(
        "<tr>" + "".join(f"<td>{row.get(field, '')}</td>" for field in fields) + "</tr>"
        for row in rows
    )
    head = "".join(f"<th>{field}</th>" for field in fields)
    return (
        "<!doctype html>\n<meta charset=\"utf-8\">\n"
        f"<title>{title}</title>\n"
        f"<h1>{title}</h1>\n<table>\n<thead><tr>{head}</tr></thead>\n<tbody>\n{body}\n</tbody>\n</table>\n"
    )


def refresh_source_inventory() -> None:
    files = [
        path
        for path in sorted((ROOT / "results" / "tables").iterdir())
        if path.is_file() and path.name not in {".gitignore", ".gitkeep"}
    ]
    detail_rows = []
    summary: dict[str, dict[str, object]] = {}
    for path in files:
        rows, cols = table_shape(path)
        group = source_category(path)
        detail_rows.append({"Source file": path.name, "Rows": rows, "Columns": cols, "Source group": group})
        item = summary.setdefault(group, {"Source group": group, "Files": 0, "Total source rows": 0, "Max columns": 0})
        item["Files"] = int(item["Files"]) + 1
        item["Total source rows"] = int(item["Total source rows"]) + rows
        item["Max columns"] = max(int(item["Max columns"]), cols)
    summary_rows = sorted(summary.values(), key=lambda row: str(row["Source group"]))
    summary_fields = ["Source group", "Files", "Total source rows", "Max columns"]
    detail_fields = ["Source file", "Rows", "Columns", "Source group"]

    targets = [
        ROOT / "results/manuscript/tables/table_s0_source_inventory.csv",
        ROOT / "results/manuscript/tables/publication/table_s0_source_inventory.csv",
        ROOT / "results/manuscript/supplement/table_s0_source_inventory.csv",
    ]
    for target in targets:
        write_csv(target, summary_rows, summary_fields)
        target.with_suffix(".html").write_text(html_table("Supplementary Table S0. Regenerated source inventory", summary_rows, summary_fields), encoding="utf-8")
    tech = ROOT / "results/manuscript/tables/technical/table_s0_source_inventory_technical.csv"
    write_csv(tech, detail_rows, detail_fields)
    tech.with_suffix(".html").write_text(html_table("Technical Source Inventory", detail_rows, detail_fields), encoding="utf-8")


def generate_data_manifest() -> None:
    roots = [
        ROOT / "data",
        ROOT / "config" / "feature_resources",
    ]
    rows = []
    for base in roots:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rows.append(
                {
                    "path": rel(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "category": "reference/resource/input",
                    "purpose": purpose_for(path),
                    "source_or_url": source_for(path),
                    "restore_instructions": f"Place the file at {rel(path)} inside the repository root.",
                }
            )
    write_csv(
        MANIFEST_DIR / "data_manifest.csv",
        rows,
        ["path", "bytes", "sha256", "category", "purpose", "source_or_url", "restore_instructions"],
    )
    write_csv(
        ROOT / "DATA_MANIFEST.csv",
        rows,
        ["path", "bytes", "sha256", "category", "purpose", "source_or_url", "restore_instructions"],
    )


def purpose_for(path: Path) -> str:
    relative = rel(path)
    if "muat_style_tcga_comparator/fold_checkpoints/full/HRD_Score" in relative:
        return "MuAt-compatible HRD_Score fold checkpoint cache"
    if "muat_style_tcga_comparator/fold_checkpoints/full/hrd_binary_33" in relative:
        return "MuAt-compatible hrd_binary_33 fold checkpoint cache"
    if "muat_style_tcga_comparator/muat_compatible_events" in relative:
        return "MuAt-compatible event-token cache"
    if "GRCh37" in relative:
        return "GRCh37 reference FASTA and metadata"
    if "mc3.v0.2.8.PUBLIC.maf" in relative:
        return "TCGA MC3 public mutation calls"
    if "clinical_bio_v4_sources" in relative or "feature_resources" in relative:
        return "Bio MAF v4 fixed external feature resource"
    if "Signatures" in relative:
        return "COSMIC signature reference kept separate from Bio MAF features"
    return "Bundled input/resource"


def source_for(path: Path) -> str:
    relative = rel(path)
    if "muat_style_tcga_comparator/fold_checkpoints/full/HRD_Score" in relative:
        return "Bundled local export; regenerable by full MuAt-compatible HRD_Score comparator"
    if "muat_style_tcga_comparator/fold_checkpoints/full/hrd_binary_33" in relative:
        return "Bundled local export; regenerable by full MuAt-compatible hrd_binary_33 comparator"
    if "muat_style_tcga_comparator/fold_checkpoints/full/" in relative:
        return "Bundled local export; regenerable by full MuAt-compatible comparator"
    if "muat_style_tcga_comparator/muat_compatible_events" in relative:
        return "Bundled local export; regenerable from MC3 mutation calls"
    if "GRCh37" in relative:
        return "NCBI Datasets GCF_000001405.13 GRCh37"
    if "mc3.v0.2.8.PUBLIC.maf" in relative:
        return "GDC/TCGA MC3 public MAF"
    if "clinical_bio_v4_sources" in relative:
        return "Bundled OncoKB/IntOGen/Cancer Hotspots resources; see resource manifest"
    return "Bundled local export"


def generate_large_assets_manifest() -> None:
    lookup = dataset_path_lookup()
    explicit_paths: set[Path] = set()
    for pattern in RESTORABLE_CACHE_GLOBS:
        explicit_paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    rows = []
    seen: set[str] = set()
    for path in active_files():
        if path.stat().st_size < GITHUB_SIZE_THRESHOLD_BYTES and path not in explicit_paths:
            continue
        relative = rel(path)
        digest = sha256_file(path)
        dataset_relative = lookup.get(("restore_destination", relative)) or lookup.get(("sha256", digest.lower())) or relative
        seen.add(relative)
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "purpose": purpose_for(path),
                "source_or_url": source_for(path),
                "sha256": digest,
                "destination": "Readable path in sibling datasets folder or Hugging Face dataset",
                "dataset_relative_path": dataset_relative,
                "restore_instructions": f"Use directly from --datasets-dir at {dataset_relative}; validate with: {DIRECT_STRICT_COMMAND}",
            }
        )
    for item in load_dataset_restore_rows(required_only=True):
        destination = item.get("restore_destination")
        if not destination or destination in seen:
            continue
        repo_path = repo_manifest_path(destination)
        if repo_path.exists():
            size_text = str(repo_path.stat().st_size)
            digest = sha256_file(repo_path)
        else:
            size_text = str(item.get("bytes", "0"))
            digest = item.get("sha256", "")
        try:
            size = int(size_text)
        except ValueError:
            size = 0
        if size < GITHUB_SIZE_THRESHOLD_BYTES:
            continue
        dataset_relative = item.get("dataset_relative_path") or destination
        rows.append(
            {
                "path": destination,
                "bytes": size_text,
                "purpose": purpose_for(repo_path),
                "source_or_url": source_for(repo_path),
                "sha256": digest,
                "destination": "Readable path in sibling datasets folder or Hugging Face dataset",
                "dataset_relative_path": dataset_relative,
                "restore_instructions": f"Use directly from --datasets-dir at {dataset_relative}; validate with: {DIRECT_STRICT_COMMAND}",
            }
        )
        seen.add(destination)
    fields = [
        "path",
        "bytes",
        "purpose",
        "source_or_url",
        "sha256",
        "destination",
        "dataset_relative_path",
        "restore_instructions",
    ]
    write_csv(MANIFEST_DIR / "large_assets_manifest.csv", rows, fields)
    write_csv(ROOT / "LFS_MANIFEST.csv", rows, fields)


def generate_repro_manifest() -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": ".",
        "strict_command": DIRECT_STRICT_COMMAND,
        "local_repo_only_strict_command": LOCAL_STRICT_COMMAND,
        "dataset_backed_strict_command": DIRECT_STRICT_COMMAND,
        "reviewer_setup_check_command": REVIEWER_SETUP_COMMAND,
        "reviewer_fast_validation_command": REVIEWER_FAST_VALIDATION_COMMAND,
        "reviewer_full_reproduction_command": REVIEWER_FULL_REPRODUCTION_COMMAND,
        "github_upload_cleanup_command": "python scripts/reproduce_manuscript.py --prepare-github-upload --datasets-dir ../github_exports_datasets",
        "default_external_datasets_dir": "../github_exports_datasets",
        "full_regeneration_command": "python src/run_all_experiments.py --config config/experiment_settings.strict_no_leakage.yaml --paths config/paths.yaml",
        "full_regeneration_dry_run_command": "python src/run_all_experiments.py --config config/experiment_settings.strict_no_leakage.yaml --paths config/paths.yaml --dry-run",
        "active_directories": ["src", "config", "data", "results/tables", "docs", "manifests"],
        "generated_output_directories": {
            "results/manuscript": "ignored local outputs regenerated by the production pipeline"
        },
        "science_guards": [
            "Bio MAF v4 only for active MAF-stack manuscript outputs",
            "COSMIC signatures remain separate from Bio MAF feature resources",
            "Bio MAF feature-block selection is nested inside inner-loop tuning",
            "Fixed TCGA cancer-type endpoint is cancer_type_top20",
        ],
    }
    write_json(MANIFEST_DIR / "reproducibility_manifest.json", payload)


def refresh_manifests() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    refresh_source_inventory()
    generate_data_manifest()
    generate_large_assets_manifest()
    generate_repro_manifest()
    generate_manuscript_manifest()


def load_manifest_csv(name: str) -> list[dict[str, str]]:
    path = MANIFEST_DIR / name
    if not path.exists():
        raise RuntimeError(f"missing manifest: {rel(path)}")
    return read_csv(path)


def dataset_path_lookup() -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    path = MANIFEST_DIR / "dataset_assets_manifest.csv"
    if not path.exists():
        return lookup
    for row in read_csv(path):
        dataset_relative = row.get("dataset_relative_path", "")
        if not dataset_relative:
            continue
        restore_destination = row.get("restore_destination", "")
        if restore_destination:
            lookup[("restore_destination", restore_destination)] = dataset_relative
        digest = row.get("sha256", "").lower()
        if digest:
            lookup[("sha256", digest)] = dataset_relative
    return lookup


def dataset_backing_lookup() -> dict[str, dict[str, str]]:
    return {
        row["restore_destination"]: row
        for row in load_dataset_restore_rows(required_only=True)
        if row.get("restore_destination")
    }


def validate_dataset_backing(relative_path: str, row: dict[str, str], datasets_dir: Path) -> str:
    dataset_relative = row.get("dataset_relative_path") or row.get("path")
    if not dataset_relative:
        raise RuntimeError(f"dataset manifest row missing dataset_relative_path/path for {relative_path}")
    path = dataset_manifest_path(datasets_dir, dataset_relative)
    if not path.exists():
        raise RuntimeError(f"dataset-backed file is missing: {dataset_relative} for {relative_path}")
    size = path.stat().st_size
    if str(size) != str(row["bytes"]):
        raise RuntimeError(f"dataset-backed size mismatch for {relative_path}: manifest={row['bytes']} actual={size}")
    digest = sha256_external_file(path, audit_root=datasets_dir)
    if digest.lower() != str(row["sha256"]).lower():
        raise RuntimeError(f"dataset-backed sha256 mismatch for {relative_path}: {dataset_relative}")
    return dataset_relative


def validate_repo_file_row(row: dict[str, str]) -> None:
    path = repo_manifest_path(row["path"])
    size = path.stat().st_size
    if str(size) != str(row["bytes"]):
        raise RuntimeError(f"size mismatch for {row['path']}: manifest={row['bytes']} actual={size}")
    digest = sha256_file(path)
    if digest.lower() != str(row["sha256"]).lower():
        raise RuntimeError(f"sha256 mismatch for {row['path']}")


def validate_manifest_rows(
    rows: list[dict[str, str]],
    *,
    require_exact: bool = False,
    base: Path | None = None,
    datasets_dir: Path | None = None,
    allow_dataset_backed: bool = False,
) -> dict[str, object]:
    expected_paths = {row["path"] for row in rows}
    backing = dataset_backing_lookup() if allow_dataset_backed else {}
    if require_exact and base is not None:
        actual_paths = {
            rel(path)
            for path in sorted(base.rglob("*"))
            if path.is_file() and path.name not in NOISE_FILENAMES
        }
        missing_from_manifest = sorted(actual_paths - expected_paths)
        stale_in_manifest = sorted(path for path in expected_paths - actual_paths if path not in backing)
        if missing_from_manifest or stale_in_manifest:
            raise RuntimeError(
                "manifest path mismatch: "
                f"missing_from_manifest={missing_from_manifest[:10]}, stale_in_manifest={stale_in_manifest[:10]}"
            )
    repo_checked = 0
    dataset_checked = 0
    dataset_relative_paths: list[str] = []
    for row in rows:
        path = repo_manifest_path(row["path"])
        if path.exists():
            validate_repo_file_row(row)
            repo_checked += 1
            continue
        backing_row = backing.get(row["path"])
        if backing_row and datasets_dir is not None:
            dataset_relative_paths.append(validate_dataset_backing(row["path"], backing_row, datasets_dir))
            dataset_checked += 1
            continue
        raise RuntimeError(f"manifested file is missing from Git checkout and dataset bundle: {row['path']}")
    return {
        "manifest_rows": len(rows),
        "repo_files_checked": repo_checked,
        "dataset_backed_files_checked": dataset_checked,
        "dataset_backed_examples": dataset_relative_paths[:20],
    }


def load_dataset_restore_rows(*, required_only: bool) -> list[dict[str, str]]:
    rows = load_manifest_csv("dataset_assets_manifest.csv")
    if required_only:
        rows = [row for row in rows if row.get("restore_destination")]
    return rows


def load_dataset_asset_rows(datasets_dir: Path, *, required_only: bool) -> list[dict[str, str]]:
    resolved = datasets_dir.resolve()
    sibling_manifest = resolved / "dataset_assets_manifest.csv"
    if sibling_manifest.exists():
        rows = read_dataset_csv(sibling_manifest, resolved)
    else:
        rows = load_dataset_restore_rows(required_only=False)
    if required_only:
        rows = [row for row in rows if row.get("restore_destination")]
    return rows


def validate_dataset_assets(
    datasets_dir: Path,
    rows: list[dict[str, str]] | None = None,
    *,
    required_only: bool = False,
) -> dict[str, object]:
    datasets_dir = datasets_dir.resolve()
    rows = rows if rows is not None else load_dataset_asset_rows(datasets_dir, required_only=required_only)
    missing: list[str] = []
    mismatched: list[str] = []
    checked = 0
    for row in rows:
        dataset_relative = row.get("dataset_relative_path") or row.get("path")
        if not dataset_relative:
            raise RuntimeError(f"dataset manifest row missing dataset_relative_path/path: {row}")
        path = dataset_manifest_path(datasets_dir, dataset_relative)
        if not path.exists():
            missing.append(dataset_relative)
            continue
        size = path.stat().st_size
        if str(size) != str(row["bytes"]):
            mismatched.append(dataset_relative)
            continue
        digest = sha256_external_file(path, audit_root=datasets_dir)
        if digest.lower() != str(row["sha256"]).lower():
            mismatched.append(dataset_relative)
            continue
        checked += 1
    if missing or mismatched:
        raise RuntimeError(
            "external dataset validation failed: "
            f"missing={missing[:10]}, mismatched={mismatched[:10]}, datasets_dir={datasets_dir}"
        )
    return {
        "datasets_dir": display_datasets_dir(datasets_dir),
        "checked_dataset_assets": checked,
        "required_only": required_only,
    }


def prepare_github_upload(datasets_dir: Path) -> dict[str, object]:
    all_rows = load_dataset_restore_rows(required_only=False)
    validate_dataset_assets(datasets_dir, all_rows, required_only=False)
    rows = [row for row in all_rows if row.get("restore_destination")]
    removed: list[str] = []
    kept_small: list[str] = []
    already_absent: list[str] = []
    skipped_mismatch: list[str] = []
    for row in rows:
        restore_destination = row.get("restore_destination")
        if not restore_destination:
            continue
        repo_path = repo_manifest_path(restore_destination)
        if not repo_path.exists():
            already_absent.append(restore_destination)
            continue
        if repo_path.stat().st_size < GITHUB_SIZE_THRESHOLD_BYTES:
            kept_small.append(restore_destination)
            continue
        if str(repo_path.stat().st_size) != str(row["bytes"]) or sha256_file(repo_path).lower() != str(row["sha256"]).lower():
            skipped_mismatch.append(restore_destination)
            continue
        repo_path.unlink()
        removed.append(restore_destination)
    if skipped_mismatch:
        raise RuntimeError(f"refusing to remove large files with checksum mismatch: {skipped_mismatch[:20]}")
    return {
        "datasets_dir": display_datasets_dir(datasets_dir),
        "removed_large_assets": len(removed),
        "kept_small_assets": len(kept_small),
        "already_absent": len(already_absent),
        "removed_paths": removed,
    }


def validate_active_scan() -> None:
    hits: list[dict[str, object]] = []
    for path in text_files():
        if rel(path) == "scripts/reproduce_manuscript.py":
            continue
        text = read_text(path)
        relative = rel(path)
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if relative in FORBIDDEN_PATTERN_PATH_EXEMPTIONS.get(name, set()):
                continue
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append({"pattern": name, "path": relative, "line": line, "match": match.group(0)})
                break
    if hits:
        raise RuntimeError("forbidden active-tree references found: " + json.dumps(hits[:25], indent=2))


def validate_manuscript_output_policy() -> dict[str, object]:
    policy_file = ROOT / "results" / "manuscript" / ".gitignore"
    if not policy_file.exists():
        raise RuntimeError("missing results/manuscript/.gitignore output policy")
    policy_text = read_text(policy_file)
    required_rules = {"*", "!.gitignore", "!.gitkeep"}
    missing_rules = [rule for rule in sorted(required_rules) if rule not in policy_text.splitlines()]
    if missing_rules:
        raise RuntimeError(f"results/manuscript/.gitignore missing rules: {missing_rules}")

    tracked: list[str] = []
    git_index_checked = False
    if (ROOT / ".git").exists():
        try:
            proc = subprocess.run(
                ["git", "ls-files", "results/manuscript"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("git is required to verify manuscript-output tracking policy") from exc
        tracked = [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]
        git_index_checked = True
    exposed = [path for path in tracked if path not in MANUSCRIPT_POLICY_FILES]
    if exposed:
        raise RuntimeError(f"generated manuscript outputs are tracked in Git: {exposed[:25]}")

    output_dir = ROOT / "results" / "manuscript"
    local_generated = 0
    if output_dir.exists():
        local_generated = sum(
            1
            for path in output_dir.rglob("*")
            if path.is_file() and path.name not in NOISE_FILENAMES and rel(path) not in MANUSCRIPT_POLICY_FILES
        )
    return {
        "output_dir": "results/manuscript",
        "git_index_checked": git_index_checked,
        "tracked_policy_files": len(tracked),
        "generated_outputs_tracked": 0,
        "local_generated_files_present": local_generated,
    }


def validate_main_results() -> dict[str, object]:
    rows = read_csv(MAIN_ENDPOINT_RESULTS_PATH)
    endpoints = {row.get("endpoint", "") for row in rows}
    needed = {"cancer_type_top20", "hrd_binary_24", "hrd_binary_33", "hrd_binary_42", "HRD_Score", "damage_class", "OS"}
    if not needed.issubset(endpoints):
        raise RuntimeError(f"main endpoint results missing endpoints: {sorted(needed - endpoints)}")
    top20 = [row for row in rows if row.get("endpoint") == "cancer_type_top20"]
    if not top20:
        raise RuntimeError("main endpoint results missing cancer_type_top20 rows")
    if not any(str(row.get("n_samples", "")).startswith("8800") for row in top20):
        raise RuntimeError("cancer_type_top20 rows do not report n_samples=8800")
    missing_balanced_accuracy = [row for row in top20 if not str(row.get("balanced_accuracy", "")).strip()]
    if missing_balanced_accuracy:
        raise RuntimeError("cancer_type_top20 rows must include balanced_accuracy values")
    bad_metric = [
        row
        for row in top20
        if str(row.get("metric", "")).strip() not in {"macro_auroc", "balanced_accuracy"}
    ]
    if bad_metric:
        raise RuntimeError("cancer_type_top20 rows must use macro_auroc or balanced_accuracy metrics")
    return {"source_rows": len(rows), "endpoints": sorted(endpoints)}


def validate_bio_maf_outputs() -> dict[str, object]:
    required = [
        "results/tables/proposed_clinical_bio_v4_manifest.json",
        "results/tables/proposed_clinical_bio_v4_feature_summary.csv",
        "results/tables/proposed_clinical_bio_v4_feature_table_exact.csv",
        "results/tables/proposed_clinical_bio_v4_core_feature_table_exact.csv",
        "results/tables/proposed_clinical_bio_v4_driver_gene_panel.csv",
        "results/tables/proposed_clinical_bio_v4_driver_gene_evidence_all_sources.csv",
        "results/tables/main_manuscript_complete_panel_endpoint_results.csv",
        "results/tables/main_manuscript_complete_panel_feature_manifest.csv",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    if missing:
        raise RuntimeError(f"missing Bio MAF v4 source outputs: {missing}")
    endpoint_rows = read_csv(MAIN_ENDPOINT_RESULTS_PATH)
    maf_rows = [row for row in endpoint_rows if "MAF_stack" in row.get("representation", "")]
    if not maf_rows:
        raise RuntimeError("main endpoint results missing Bio MAF v4 rows")
    nested_policy_rows = [
        row
        for row in maf_rows
        if "inner_validation_selects_predeclared_bio_v4_feature_block" in row.get("feature_selection", "")
    ]
    required_nested_endpoints = {
        "HRD_Score",
        "OS",
        "cancer_type_top20",
        "hrd_binary_24",
        "hrd_binary_33",
        "hrd_binary_42",
    }
    nested_endpoints = {row.get("endpoint", "") for row in nested_policy_rows}
    if not required_nested_endpoints.issubset(nested_endpoints):
        missing = sorted(required_nested_endpoints - nested_endpoints)
        raise RuntimeError(f"Bio MAF v4 rows missing inner-loop feature-block selection for: {missing}")
    proposed = read_json(ROOT / "results/tables/proposed_clinical_bio_v4_manifest.json")
    if "No COSMIC" not in str(proposed):
        raise RuntimeError("Bio MAF v4 manifest does not state COSMIC signature separation")
    v4_resource_dir = ROOT / "config/feature_resources/clinical_bio_v4_sources"
    cosmic_in_v4 = [rel(path) for path in v4_resource_dir.rglob("*") if path.is_file() and "cosmic" in path.name.lower()]
    if cosmic_in_v4:
        raise RuntimeError(f"COSMIC files found inside Bio MAF v4 resources: {cosmic_in_v4}")
    dataset_rows = load_manifest_csv("dataset_assets_manifest.csv")
    cosmic_rows = [
        row
        for row in dataset_rows
        if row.get("filename") == "COSMIC_v3.5_SBS_GRCh37.txt"
        and not row.get("restore_destination")
        and "clinical_bio_v4_sources" not in row.get("dataset_relative_path", "")
    ]
    if not cosmic_rows:
        raise RuntimeError("COSMIC signature reference is missing from its separate dataset/resource manifest location")
    return {
        "bio_maf_v4_required_source_outputs": len(required),
        "bio_maf_v4_endpoint_rows": len(maf_rows),
        "bio_maf_v4_nested_policy_endpoints": sorted(nested_endpoints),
        "bio_maf_v4_feature_count": proposed.get("full_feature_count_including_optional_controls"),
    }


def validate_strict(dataset_validation: dict[str, object] | None = None, datasets_dir: Path | None = None) -> dict[str, object]:
    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError(f"strict validation must be run from repository root: {ROOT}")
    datasets_dir = (datasets_dir or DEFAULT_DATASETS_DIR).resolve()
    validate_active_scan()
    if dataset_validation is None:
        dataset_validation = validate_dataset_assets(datasets_dir, required_only=False)
    data_manifest = validate_manifest_rows(
        load_manifest_csv("data_manifest.csv"),
        datasets_dir=datasets_dir,
        allow_dataset_backed=True,
    )
    large_assets_manifest = validate_manifest_rows(
        load_manifest_csv("large_assets_manifest.csv"),
        datasets_dir=datasets_dir,
        allow_dataset_backed=True,
    )
    repro_manifest = read_json(MANIFEST_DIR / "reproducibility_manifest.json")
    if not isinstance(repro_manifest, dict) or repro_manifest.get("strict_command") != DIRECT_STRICT_COMMAND:
        raise RuntimeError("reproducibility manifest has an unexpected strict command")
    manuscript_outputs = validate_manuscript_output_policy()
    main_results = validate_main_results()
    bio_maf = validate_bio_maf_outputs()
    unique_reads = sorted(set(READ_LOG))
    unique_dataset_reads = sorted(set(DATASET_READ_LOG))
    payload = {
        "status": "passed",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "read_audit": {
            "read_count": len(READ_LOG),
            "unique_read_count": len(unique_reads),
            "outside_root_reads": 0,
            "allowed_roots": [".", display_datasets_dir(datasets_dir)],
            "sample_relative_paths": unique_reads[:50],
        },
        "dataset_read_audit": {
            "datasets_dir": display_datasets_dir(datasets_dir),
            "read_count": len(DATASET_READ_LOG),
            "unique_read_count": len(unique_dataset_reads),
            "outside_datasets_dir_reads": 0,
            "sample_relative_paths": unique_dataset_reads[:50],
        },
        "manuscript_outputs": manuscript_outputs,
        "main_results": main_results,
        "bio_maf_v4": bio_maf,
        "manifests": {
            "data_manifest": data_manifest,
            "large_assets_manifest": large_assets_manifest,
            "reproducibility_manifest": "passed",
        },
    }
    if dataset_validation is not None:
        payload["dataset_validation"] = dataset_validation
    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(log_dir / "strict_validation_latest.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Strictly validate the current manuscript export.")
    parser.add_argument("--refresh-manifests", action="store_true", help="Regenerate manifests and source inventory from the current filesystem.")
    parser.add_argument(
        "--datasets-dir",
        default=str(DEFAULT_DATASETS_DIR),
        help="Sibling/external datasets directory containing large assets with manifest-relative paths.",
    )
    parser.add_argument(
        "--check-datasets",
        action="store_true",
        help="Verify all dataset assets in --datasets-dir against manifests without copying them into the repository.",
    )
    parser.add_argument(
        "--prepare-github-upload",
        action="store_true",
        help="Remove dataset-backed active files >=30 MB from the repo after verifying they exist in --datasets-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets_dir = Path(args.datasets_dir)
    did_work = False
    if args.check_datasets:
        report = validate_dataset_assets(datasets_dir, required_only=False)
        print(json.dumps({"status": "datasets_passed", **report}, indent=2))
        did_work = True
    if args.prepare_github_upload:
        report = prepare_github_upload(datasets_dir)
        print(json.dumps({"status": "github_upload_ready", **report}, indent=2))
        did_work = True
    if args.refresh_manifests:
        refresh_manifests()
        print(json.dumps({"status": "refreshed", "manifest_dir": "manifests"}, indent=2))
        did_work = True
    if args.strict:
        result = validate_strict(datasets_dir=datasets_dir)
        print(json.dumps(result, indent=2))
        did_work = True
    if not did_work:
        raise SystemExit(
            "choose --strict, --refresh-manifests, --check-datasets, "
            "and/or --prepare-github-upload"
        )


if __name__ == "__main__":
    main()
