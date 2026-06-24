#!/usr/bin/env python
"""Reviewer-facing setup check and one-command manuscript reproduction."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import reproduce_manuscript as manuscript


ROOT = manuscript.ROOT
LOG_DIR = ROOT / "results" / "logs"
DEFAULT_DATASETS_DIR = manuscript.DEFAULT_DATASETS_DIR

REQUIRED_MODULES = {
    "yaml": "PyYAML==6.0.2",
    "numpy": "numpy==1.26.4",
    "pandas": "pandas==2.3.1",
    "scipy": "scipy==1.15.3",
    "sklearn": "scikit-learn==1.7.2",
    "xgboost": "xgboost==1.6.2",
    "matplotlib": "matplotlib==3.9.2",
    "optuna": "optuna==4.1.0",
    "sksurv": "scikit-survival==0.27.0",
    "torch": "torch==2.5.1",
    "requests": "requests==2.32.3",
    "statsmodels": "statsmodels==0.14.4",
    "openpyxl": "openpyxl==3.1.5",
    "tabulate": "tabulate==0.9.0",
}
MINIMUM_DRIVER_MODULES = {"yaml", "numpy", "pandas"}

REQUIRED_REPO_FILES = [
    "README.md",
    "REPRODUCIBILITY.md",
    "requirements.txt",
    "config/experiment_settings.strict_no_leakage.yaml",
    "config/paths.yaml",
    "manifests/dataset_assets_manifest.csv",
    "manifests/huggingface_asset_manifest.csv",
    "manifests/manuscript_artifact_manifest.csv",
    "manifests/reproducibility_manifest.json",
    "scripts/reproduce_manuscript.py",
    "scripts/reviewer_workflow.py",
    "src/run_all_experiments.py",
]

KEY_DATASET_LOCATIONS = [
    ("TCGA/MC3 mutation calls, labels, and feature tables", "datasets/tcga_mc3/"),
    ("TCGA-BRCA HRD labels and cohort inputs", "datasets/tcga_brca_hrd/"),
    ("KUCAB/Mendeley mutagenesis inputs", "datasets/kucab_mendeley/"),
    ("KUCAB raw mutation tables used by active runners", "datasets/kucab_mendeley/raw_mutation_tables/"),
    ("PCAWG/PanCancer signature data", "datasets/pcawg_pancan/"),
    ("GRCh37 reference genome", "references/grch37/"),
    ("Bio MAF v4 feature resources", "resources/bio_maf_v4/"),
    ("COSMIC and other signature resources", "resources/signatures_cosmic/"),
    ("Dataset-backed MuAt/cache artifacts", "caches/"),
    ("Large manuscript/result artifacts", "results/"),
]


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        try:
            return "../" + path.resolve().relative_to(ROOT.parent.resolve()).as_posix()
        except ValueError:
            return f"<external>/{path.name}"


def status_row(kind: str, name: str, status: str, fix: str = "", **extra: object) -> dict[str, object]:
    row: dict[str, object] = {"kind": kind, "name": name, "status": status}
    if fix:
        row["fix"] = fix
    row.update(extra)
    return row


def check_python_modules() -> list[dict[str, object]]:
    rows = []
    missing_packages = []
    for module, package in REQUIRED_MODULES.items():
        ok = importlib.util.find_spec(module) is not None
        rows.append(
            status_row(
                "python_module",
                module,
                "ok" if ok else "missing",
                "" if ok else "Install the full reviewer environment with: python -m pip install -r requirements.txt",
                package=package,
            )
        )
        if not ok:
            missing_packages.append(package)
    if missing_packages:
        rows.append(
            status_row(
                "python_install_command",
                "requirements.txt",
                "needs_action",
                "Run: python -m pip install -r requirements.txt",
                missing_packages=missing_packages,
            )
        )
    return rows


def check_node() -> list[dict[str, object]]:
    node = shutil.which("node")
    if node is None:
        bundled = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "bin" / ("node.exe" if sys.platform.startswith("win") else "node")
        node = str(bundled) if bundled.exists() else None
    if node:
        return [status_row("renderer", "node", "ok", path="<node-on-PATH>")]
    return [
        status_row(
            "renderer",
            "node",
            "missing",
            "Install Node.js from https://nodejs.org/ or provide a node executable on PATH before full figure rendering.",
        )
    ]


def check_repo_files() -> list[dict[str, object]]:
    rows = []
    cwd_ok = Path.cwd().resolve() == ROOT.resolve()
    rows.append(
        status_row(
            "working_directory",
            "repository root",
            "ok" if cwd_ok else "wrong_directory",
            "" if cwd_ok else "Change directory to the repository root before running commands.",
            current=display_path(Path.cwd()),
        )
    )
    for relative in REQUIRED_REPO_FILES:
        path = ROOT / relative
        rows.append(
            status_row(
                "repo_file",
                relative,
                "ok" if path.exists() else "missing",
                "" if path.exists() else "Restore this file from the GitHub repository checkout.",
            )
        )
    return rows


def read_manifest_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def dataset_rows(required_only: bool, datasets_dir: Path | None = None) -> list[dict[str, str]]:
    if required_only:
        return manuscript.load_dataset_restore_rows(required_only=True)
    if datasets_dir is not None:
        sibling_manifest = datasets_dir.resolve() / "dataset_assets_manifest.csv"
        if sibling_manifest.exists():
            return read_manifest_csv(sibling_manifest)
    return manuscript.load_dataset_restore_rows(required_only=False)


def dataset_location_summary(datasets_dir: Path) -> list[dict[str, object]]:
    resolved = datasets_dir.resolve()
    rows = dataset_rows(required_only=False, datasets_dir=resolved) if resolved.exists() else []
    relatives = [str(row.get("dataset_relative_path") or row.get("path") or "") for row in rows]
    family_counts = Counter(str(row.get("dataset_family") or "unclassified") for row in rows)
    out: list[dict[str, object]] = []
    for label, prefix in KEY_DATASET_LOCATIONS:
        count = sum(1 for item in relatives if item.startswith(prefix))
        out.append(
            {
                "label": label,
                "path": f"{manuscript.display_datasets_dir(resolved)}/{prefix.rstrip('/')}",
                "manifest_rows": count,
                "status": "present" if count else "missing",
            }
        )
    return out + [
        {
            "label": "Manifest rows by dataset family",
            "path": "manifests/dataset_assets_manifest.csv",
            "manifest_rows": dict(sorted(family_counts.items())),
            "status": "present" if rows else "missing",
        }
    ]


def check_dataset_assets(datasets_dir: Path, *, full: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    resolved = datasets_dir.resolve()
    if not resolved.exists():
        rows.append(
            status_row(
                "dataset_bundle",
                display_path(resolved),
                "missing",
                "Download the companion Hugging Face dataset bundle and place it as ../github_exports_datasets, or pass --datasets-dir PATH.",
            )
        )
        return rows
    rows.append(status_row("dataset_bundle", display_path(resolved), "ok"))

    manifest_rows = dataset_rows(required_only=not full, datasets_dir=resolved)
    missing: list[str] = []
    mismatched: list[str] = []
    checked = 0
    for item in manifest_rows:
        dataset_relative = item.get("dataset_relative_path") or item.get("path")
        if not dataset_relative:
            mismatched.append("<manifest row missing dataset_relative_path>")
            continue
        path = manuscript.dataset_manifest_path(resolved, dataset_relative)
        if not path.exists():
            missing.append(dataset_relative)
            continue
        if str(path.stat().st_size) != str(item.get("bytes")):
            mismatched.append(dataset_relative)
            continue
        digest = manuscript.sha256_external_file(path).lower()
        if digest != str(item.get("sha256", "")).lower():
            mismatched.append(dataset_relative)
            continue
        checked += 1

    scope = "full_dataset_bundle" if full else "required_dataset_backed_assets"
    status = "ok" if not missing and not mismatched else "needs_action"
    rows.append(
        status_row(
            "dataset_checksum",
            scope,
            status,
            ""
            if status == "ok"
            else "Re-download the listed paths from the companion Hugging Face dataset, then rerun this check.",
            checked=checked,
            expected=len(manifest_rows),
            missing_count=len(missing),
            mismatched_count=len(mismatched),
            missing_examples=missing[:25],
            mismatched_examples=mismatched[:25],
        )
    )
    return rows


def check_dataset_backed_state(datasets_dir: Path) -> list[dict[str, object]]:
    rows = []
    missing = []
    mismatched = []
    stale_repo_duplicates = []
    repo_current = 0
    dataset_backed = 0
    datasets_dir = datasets_dir.resolve()
    for item in dataset_rows(required_only=True):
        destination = item.get("restore_destination")
        if not destination:
            continue
        dataset_relative = item.get("dataset_relative_path") or destination
        dataset_path = manuscript.dataset_manifest_path(datasets_dir, dataset_relative)
        if not dataset_path.exists():
            missing.append(dataset_relative)
            continue
        if str(dataset_path.stat().st_size) != str(item.get("bytes")):
            mismatched.append(dataset_relative)
            continue
        if manuscript.sha256_external_file(dataset_path).lower() != str(item.get("sha256", "")).lower():
            mismatched.append(dataset_relative)
            continue
        repo_path = manuscript.repo_manifest_path(destination)
        if not repo_path.exists():
            dataset_backed += 1
            continue
        if str(repo_path.stat().st_size) == str(item.get("bytes")) and manuscript.sha256_file(repo_path).lower() == str(item.get("sha256", "")).lower():
            repo_current += 1
        else:
            stale_repo_duplicates.append(destination)
    status = "ok" if not missing and not mismatched and not stale_repo_duplicates else "needs_action"
    rows.append(
        status_row(
            "dataset_backed_state",
            "active dataset-backed assets",
            status,
            ""
            if status == "ok"
            else "Re-download the listed dataset paths, or remove stale duplicate files from the Git checkout and rerun.",
            repo_current=repo_current,
            dataset_backed=dataset_backed,
            missing_count=len(missing),
            mismatched_count=len(mismatched),
            stale_repo_duplicate_count=len(stale_repo_duplicates),
            missing_examples=missing[:25],
            mismatched_examples=mismatched[:25],
            stale_repo_duplicate_examples=stale_repo_duplicates[:25],
        )
    )
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    blocking_statuses = {"missing", "wrong_directory", "needs_action"}
    blocking = [row for row in rows if row.get("status") in blocking_statuses]
    warnings = [row for row in rows if row.get("status") in {"optional_missing"}]
    return {
        "status": "ready" if not blocking else "needs_action",
        "checked_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "blocking_count": len(blocking),
        "warning_count": len(warnings),
        "blocking": blocking,
        "warnings": warnings,
        "rows": rows,
    }


def rows_by_kind(report: dict[str, object], kind: str) -> list[dict[str, object]]:
    return [row for row in report.get("rows", []) if row.get("kind") == kind]


def missing_module_rows(report: dict[str, object]) -> list[dict[str, object]]:
    return [row for row in rows_by_kind(report, "python_module") if row.get("status") == "missing"]


def package_names(rows: list[dict[str, object]]) -> list[str]:
    return [str(row.get("package") or row.get("name")) for row in rows]


def wrapped_list(items: list[str], *, indent: str = "  ", width: int = 100) -> list[str]:
    import textwrap

    text = ", ".join(items)
    if not text:
        return [indent + "(none)"]
    return textwrap.wrap(text, width=width, initial_indent=indent, subsequent_indent=indent)


def print_setup_summary(report: dict[str, object], *, required_only: bool) -> None:
    print("")
    print("Reviewer setup check")
    print("====================")
    print("Status: " + ("READY" if report["status"] == "ready" else "NEEDS ACTION"))
    print("")

    repo_missing = [row for row in rows_by_kind(report, "repo_file") if row.get("status") != "ok"]
    dataset = rows_by_kind(report, "dataset_checksum")
    dataset_row = dataset[0] if dataset else {}
    node = rows_by_kind(report, "renderer")
    node_row = node[0] if node else {}
    dataset_backed = rows_by_kind(report, "dataset_backed_state")
    dataset_backed_row = dataset_backed[0] if dataset_backed else {}

    print("What is okay")
    print("- Repository files: " + ("all required files are present" if not repo_missing else f"{len(repo_missing)} required file(s) missing"))
    if dataset_row:
        print(
            "- Dataset bundle: "
            f"{dataset_row.get('checked', 0)}/{dataset_row.get('expected', 0)} files passed checksum"
            + (" (required files only)" if required_only else "")
        )
    print("- Figure renderer: " + ("Node.js found" if node_row.get("status") == "ok" else "Node.js not found"))
    if dataset_backed_row:
        print(
            "- Dataset-backed active assets: "
            f"{dataset_backed_row.get('dataset_backed', 0)} read directly from the dataset bundle; "
            f"{dataset_backed_row.get('repo_current', 0)} also present in the Git checkout"
        )
    print("")

    locations = list(report.get("dataset_locations") or [])
    if locations:
        print("Where the data is")
        for item in locations:
            if item.get("label") == "Manifest rows by dataset family":
                continue
            status = "ok" if item.get("status") == "present" else "missing"
            print(f"- {item.get('label')}: {item.get('path')} ({item.get('manifest_rows', 0)} files, {status})")
        print("")

    missing_modules = missing_module_rows(report)
    if missing_modules:
        print("Fix this first")
        print("- Install the Python environment:")
        print("  python -m pip install -r requirements.txt")
        print("- Missing packages:")
        for line in wrapped_list(package_names(missing_modules)):
            print(line)
        print("")

    if repo_missing:
        print("Missing repository files")
        for row in repo_missing[:20]:
            print(f"- {row.get('name')}: {row.get('fix', 'restore from the GitHub checkout')}")
        print("")

    if dataset_row and dataset_row.get("status") != "ok":
        print("Dataset bundle problem")
        print("- Re-download the companion dataset bundle or pass the correct --datasets-dir path.")
        for key in ("missing_examples", "mismatched_examples"):
            values = dataset_row.get(key) or []
            if values:
                print(f"- {key.replace('_', ' ')}:")
                for value in values[:10]:
                    print(f"  {value}")
        print("")

    if dataset_backed_row.get("status") == "needs_action":
        print("Dataset-backed asset problem")
        print("- The dataset bundle is the canonical location for these assets; fix the listed bundle files or stale Git duplicates.")
        for key in ("missing_examples", "mismatched_examples", "stale_repo_duplicate_examples"):
            values = dataset_backed_row.get(key) or []
            if values:
                print(f"- {key.replace('_', ' ')}:")
                for value in values[:10]:
                    print(f"  {value}")
        print("")

    print("Next commands")
    if report["status"] == "ready":
        print("1. Fast validation without expensive model jobs:")
        print("   python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets --dry-run")
        print("2. Full reproduction of manuscript results and figures:")
        print("   python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets")
        print("3. Optional GitHub cleanup after validation:")
        print("   python scripts/reproduce_manuscript.py --prepare-github-upload --datasets-dir ../github_exports_datasets")
    else:
        print("1. Fix the items above, then run this check again:")
        print("   python scripts/reviewer_workflow.py check --datasets-dir ../github_exports_datasets")
        print("2. Fast validation without expensive model jobs:")
        print("   python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets --dry-run")
        print("3. Full reproduction of manuscript results and figures:")
        print("   python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets")
    print("")
    print("Details saved to: results/logs/reviewer_setup_check.json" if not required_only else "Details saved to: results/logs/reviewer_required_setup_check.json")


def write_setup_report(report: dict[str, object], *, required_only: bool) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    name = "reviewer_required_setup_check.json" if required_only else "reviewer_setup_check.json"
    (LOG_DIR / name).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def run_setup_check(args: argparse.Namespace) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    rows.extend(check_repo_files())
    rows.extend(check_python_modules())
    rows.extend(check_node())
    rows.extend(check_dataset_assets(Path(args.datasets_dir), full=not args.required_only))
    rows.extend(check_dataset_backed_state(Path(args.datasets_dir)))
    report = summarize(rows)
    report["datasets_dir"] = manuscript.display_datasets_dir(Path(args.datasets_dir))
    report["full_dataset_check"] = not args.required_only
    report["dataset_locations"] = dataset_location_summary(Path(args.datasets_dir))
    report["next_commands"] = {
        "install_environment": "python -m pip install -r requirements.txt",
        "setup_check": "python scripts/reviewer_workflow.py check --datasets-dir ../github_exports_datasets",
        "fast_validation": "python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets --dry-run",
        "full_reproduction": "python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets",
    }
    write_setup_report(report, required_only=bool(args.required_only))
    if getattr(args, "emit", True):
        if getattr(args, "json", False):
            print(json.dumps(report, indent=2, default=str))
        else:
            print_setup_summary(report, required_only=bool(args.required_only))
    if args.strict_exit and report["status"] != "ready":
        raise SystemExit(1)
    return report


def print_reproduce_blocked(missing_modules: list[dict[str, object]], *, as_json: bool, dry_run: bool) -> None:
    payload = {
        "status": "blocked_missing_python_modules",
        "missing_packages": package_names(missing_modules),
        "fix": "python -m pip install -r requirements.txt",
        "rerun": "python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets"
        + (" --dry-run" if dry_run else ""),
    }
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return
    print("")
    print("Reviewer dry run cannot start yet" if dry_run else "Full reproduction cannot start yet")
    print("==================================" if dry_run else "==================================")
    print("The code and dataset bundle are present, but the Python environment is not installed.")
    print("")
    print("Run this once:")
    print("  python -m pip install -r requirements.txt")
    print("")
    print("Missing packages:")
    for line in wrapped_list(payload["missing_packages"]):
        print(line)
    print("")
    print("Then rerun:")
    print("  " + payload["rerun"])
    print("")
    print("Details saved to: results/logs/reviewer_required_setup_check.json")


def run_driver(command: list[str], *, dry_run: bool, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"running": command}, indent=2), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        return
    if dry_run:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        console_log = LOG_DIR / "reviewer_driver_dry_run_console.txt"
        result = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
        console_log.write_text((result.stdout or "") + ("\nStandard error output\n" + result.stderr if result.stderr else ""), encoding="utf-8")
        if result.returncode:
            if as_json:
                print(json.dumps({"status": "driver_failed", "returncode": result.returncode, "console_log": "results/logs/reviewer_driver_dry_run_console.txt"}, indent=2))
            else:
                print("")
                print("Driver dry run failed before completion")
                print("=======================================")
                print("The console output was saved to: results/logs/reviewer_driver_dry_run_console.txt")
                print("Most setup failures are fixed with:")
                print("  python -m pip install -r requirements.txt")
            raise SystemExit(result.returncode)
        return
    print("Running the full manuscript driver. This can take a long time.")
    subprocess.run(command, cwd=ROOT, check=True)


def run_reproduce(args: argparse.Namespace) -> None:
    check_args = argparse.Namespace(
        datasets_dir=args.datasets_dir,
        required_only=True,
        strict_exit=False,
        json=getattr(args, "json", False),
        emit=False,
    )
    setup_report = run_setup_check(check_args)
    missing_modules = [
        row for row in setup_report["rows"]
        if row.get("kind") == "python_module" and row.get("status") == "missing"
    ]
    minimum_missing = [row for row in missing_modules if row.get("name") in MINIMUM_DRIVER_MODULES]
    if missing_modules and (not args.dry_run or minimum_missing):
        print_reproduce_blocked(
            minimum_missing if args.dry_run else missing_modules,
            as_json=getattr(args, "json", False),
            dry_run=bool(args.dry_run),
        )
        raise SystemExit(1)

    if not getattr(args, "json", False):
        print("")
        print("Reviewer dry run" if args.dry_run else "Reviewer full reproduction")
        print("================" if args.dry_run else "==========================")
        if missing_modules:
            print("Package note: full-run packages are missing, but --dry-run can continue.")
            print("Fix for full reproduction: python -m pip install -r requirements.txt")
            print("")
        print("1. Verifying dataset-backed assets in ../github_exports_datasets ...")
    dataset_report = manuscript.validate_dataset_assets(Path(args.datasets_dir), required_only=False)
    if getattr(args, "json", False):
        print(json.dumps({"status": "datasets_verified", **dataset_report}, indent=2))
    else:
        print(
            f"   {dataset_report.get('checked_dataset_assets', 0)} dataset file(s) passed checksum; "
            "large assets will be read from the dataset bundle."
        )

    if not getattr(args, "json", False):
        print("2. Running strict manuscript validation ...")
    strict_report = manuscript.validate_strict(dataset_validation=dataset_report, datasets_dir=Path(args.datasets_dir))
    if getattr(args, "json", False):
        print(json.dumps(strict_report, indent=2))
    else:
        print(
            f"   strict validation {strict_report.get('status')}; "
            f"outside allowed roots: {strict_report.get('read_audit', {}).get('outside_root_reads')}."
        )

    run_all = [
        sys.executable,
        "src/run_all_experiments.py",
        "--config",
        "config/experiment_settings.strict_no_leakage.yaml",
        "--paths",
        "config/paths.yaml",
    ]
    if args.dry_run:
        run_all.append("--dry-run")
    if args.skip_figures:
        run_all.append("--skip-figures")
    if not getattr(args, "json", False):
        print("3. Running manuscript driver" + (" dry run ..." if args.dry_run else " ..."))
    run_driver(run_all, dry_run=bool(args.dry_run), as_json=getattr(args, "json", False))
    payload = {
        "status": "reviewer_reproduction_complete" if not args.dry_run else "reviewer_dry_run_complete",
        "strict_log": "results/logs/strict_validation_latest.json",
        "run_log": "results/logs/dry_run_all_experiments_manifest.json" if args.dry_run else "results/logs/run_all_experiments_manifest.json",
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))
    else:
        print("")
        print("Done")
        print("----")
        print("Status: " + payload["status"])
        print("Strict log: " + payload["strict_log"])
        print("Run log: " + payload["run_log"])
        if args.dry_run:
            print("Driver console log: results/logs/reviewer_driver_dry_run_console.txt")


def parse_args() -> argparse.Namespace:
    if len(sys.argv) == 1:
        sys.argv.append("check")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Check reviewer environment, configs, and dataset bundle.")
    check.add_argument("--datasets-dir", default=str(DEFAULT_DATASETS_DIR), help="Path to the companion dataset bundle.")
    check.add_argument("--required-only", action="store_true", help="Check only active dataset-backed assets instead of every dataset-bundle payload.")
    check.add_argument("--strict-exit", action="store_true", help="Exit nonzero when any blocking setup item is missing.")
    check.add_argument("--json", action="store_true", help="Print the full machine-readable JSON report.")

    reproduce = subparsers.add_parser("reproduce", help="Validate assets, run strict validation, then run the manuscript driver.")
    reproduce.add_argument("--datasets-dir", default=str(DEFAULT_DATASETS_DIR), help="Path to the companion dataset bundle.")
    reproduce.add_argument("--dry-run", action="store_true", help="Plan the full manuscript run without launching expensive model jobs.")
    reproduce.add_argument("--skip-figures", action="store_true", help="Do not regenerate manuscript figures during the full run.")
    reproduce.add_argument("--json", action="store_true", help="Print detailed machine-readable JSON as commands run.")

    for alias in ("start", "doctor"):
        alias_parser = subparsers.add_parser(alias, help="Alias for check.")
        alias_parser.add_argument("--datasets-dir", default=str(DEFAULT_DATASETS_DIR), help="Path to the companion dataset bundle.")
        alias_parser.add_argument("--required-only", action="store_true", help="Check only active dataset-backed assets instead of every dataset-bundle payload.")
        alias_parser.add_argument("--strict-exit", action="store_true", help="Exit nonzero when any blocking setup item is missing.")
        alias_parser.add_argument("--json", action="store_true", help="Print the full machine-readable JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command in {"check", "start", "doctor"}:
        run_setup_check(args)
    elif args.command == "reproduce":
        run_reproduce(args)
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
