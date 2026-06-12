"""Run the manuscript benchmark as resumable phases and gate completion."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import BUNDLE_ROOT, load_yaml, resolve_paths_map
from utils.endpoint_registry import load_endpoint_registry, muat_comparator_endpoints


ACTIVE_PHASES = [
    "main_manuscript_complete_panel",
    "muat_style_tcga_comparator",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment_settings.strict_no_leakage.yaml")
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--deadline-hours", type=float, default=12.0)
    parser.add_argument("--phase-attempts", type=int, default=2)
    parser.add_argument("--only", choices=ACTIVE_PHASES, nargs="*", help="Optional phase subset.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip strict figure regeneration and only run experiment phases.")
    return parser.parse_args()


def _log_dir(paths_arg: str) -> Path:
    paths = resolve_paths_map(load_yaml(paths_arg))
    log_dir = Path((paths.get("workspace") or {}).get("results_logs_dir") or (BUNDLE_ROOT / "results" / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _tables_dir(paths: dict) -> Path:
    return Path((paths.get("workspace") or {}).get("results_tables_dir") or (BUNDLE_ROOT / "results" / "tables"))


def _summary_completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    return bool(not frame.empty and "status" in frame.columns and frame["status"].astype(str).eq("completed").any())


def _settings_for(settings: dict, experiment_id: str) -> dict:
    return dict(((settings.get("experiments") or {}).get(experiment_id) or {}))


def _configured_muat_endpoints(settings: dict) -> set[str]:
    local = _settings_for(settings, "muat_style_tcga_comparator")
    if local.get("endpoints"):
        return {str(endpoint) for endpoint in local.get("endpoints", [])}
    primary = str(local.get("primary_endpoint") or local.get("primary_task") or "").strip()
    secondary = [str(endpoint) for endpoint in local.get("secondary_endpoints", [])]
    if primary:
        return {primary, *secondary}
    registry = load_endpoint_registry(settings.get("endpoint_registry"))
    return {str(endpoint) for endpoint in muat_comparator_endpoints(registry)}


def _phase_already_complete(phase: str, paths: dict, settings: dict) -> bool:
    tables = _tables_dir(paths)
    if phase == "main_manuscript_complete_panel":
        return _summary_completed(tables / "main_manuscript_complete_panel_summary.csv")
    if phase == "muat_style_tcga_comparator":
        local = _settings_for(settings, "muat_style_tcga_comparator")
        tag = str(local.get("output_tag") or "").strip()
        prefix = "muat_style_tcga_comparator" if not tag else f"muat_style_tcga_comparator_{tag}"
        results = tables / f"{prefix}_endpoint_results.csv"
        if not _summary_completed(tables / f"{prefix}_summary.csv") or not results.exists():
            return False
        try:
            frame = pd.read_csv(results)
        except Exception:
            return False
        observed = set(frame.get("endpoint", pd.Series(dtype=str)).astype(str))
        return _configured_muat_endpoints(settings).issubset(observed)
    return False


def _run_command(command: list[str], log_path: Path) -> int:
    started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{started}] START {' '.join(command)}\n")
        log.flush()
        proc = subprocess.Popen(command, cwd=BUNDLE_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        return_code = proc.wait()
        ended = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        log.write(f"[{ended}] EXIT {return_code} {' '.join(command)}\n")
        log.flush()
    return int(return_code)


def main() -> int:
    args = _parse_args()
    deadline = time.time() + max(float(args.deadline_hours), 0.01) * 3600.0
    settings = load_yaml(args.config)
    paths = resolve_paths_map(load_yaml(args.paths))
    log_dir = _log_dir(args.paths)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    phase_log = log_dir / f"benchmark_phase_runner_{stamp}.log"
    phases = list(args.only or ACTIVE_PHASES)

    for phase in phases:
        if _phase_already_complete(phase, paths, settings):
            print(f"[phase-runner] {phase} already complete; skipping", flush=True)
            continue
        attempts = max(1, int(args.phase_attempts))
        for attempt in range(1, attempts + 1):
            if time.time() >= deadline:
                print(f"[phase-runner] deadline reached before {phase}", flush=True)
                return 2
            command = [
                sys.executable,
                "src/run_all_experiments.py",
                "--config",
                args.config,
                "--paths",
                args.paths,
                "--only",
                phase,
                "--skip-figures",
            ]
            print(f"[phase-runner] {phase} attempt {attempt}/{attempts}", flush=True)
            rc = _run_command(command, phase_log)
            if rc == 0:
                break
            if attempt == attempts:
                print(f"[phase-runner] {phase} failed after {attempts} attempts; see {phase_log}", flush=True)
                return rc

    if not args.skip_figures:
        if time.time() >= deadline:
            print("[phase-runner] deadline reached before strict figure generation", flush=True)
            return 2
        figure_command = [
            sys.executable,
            "src/utils/make_all_figures.py",
            "--config",
            args.config,
            "--paths",
            args.paths,
            "--strict",
        ]
        print("[phase-runner] generating strict figures", flush=True)
        rc = _run_command(figure_command, phase_log)
        if rc != 0:
            print(f"[phase-runner] strict figure generation failed; see {phase_log}", flush=True)
            return rc

    gate_command = [
        sys.executable,
        "src/utils/check_benchmark_completion.py",
        "--config",
        args.config,
        "--paths",
        args.paths,
    ]
    print("[phase-runner] running completion gate", flush=True)
    rc = _run_command(gate_command, phase_log)
    if rc != 0:
        print(f"[phase-runner] completion gate failed; see {phase_log}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
