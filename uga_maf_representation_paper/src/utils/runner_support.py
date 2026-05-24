"""Shared runner helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from utils.checkpointing import atomic_write_json
from utils.config import BUNDLE_ROOT, REPRO_ROOT
from utils.feature_cache import FeatureCache, restore_feature_artifacts, snapshot_feature_artifacts


LEGACY_SOURCE = BUNDLE_ROOT / "src" / "legacy_source" / "cgr_validation"
LEGACY_WORKSPACE = BUNDLE_ROOT / "results" / "work" / "cgr_validation"
TEXT_SUFFIXES = {".csv", ".html", ".json", ".log", ".md", ".svg", ".tsv", ".txt", ".yaml", ".yml"}


@dataclass(frozen=True)
class RunnerContext:
    settings: dict
    paths: dict
    dry_run: bool = False
    refresh_cache: bool = False

    @property
    def tree_method(self) -> str:
        return str(self.settings.get("tree_method", "hist"))

    @property
    def xgb_n_jobs(self) -> int:
        return int(self.settings.get("xgb_n_jobs", 4))

    @property
    def cv_folds(self) -> int:
        return int(self.settings.get("cv_folds", 5))

    @property
    def cv_repeats(self) -> int:
        return int(self.settings.get("cv_repeats", 1))

    @property
    def bootstrap_iterations(self) -> int:
        return int(self.settings.get("bootstrap", 200))

    @property
    def optuna_trials(self) -> int:
        return int(self.settings.get("optuna_trials", 10))

    @property
    def xgb_estimators(self) -> int:
        return int(self.settings.get("xgb_estimators", 160))

    @property
    def run_id(self) -> str:
        return str(self.settings.get("_run_id", "unknown_run"))

    @property
    def tables_dir(self) -> Path:
        return Path(self.paths["workspace"]["results_tables_dir"])

    @property
    def figures_dir(self) -> Path:
        return Path(self.paths["workspace"]["results_figures_dir"])

    @property
    def logs_dir(self) -> Path:
        return Path(self.paths["workspace"]["results_logs_dir"])

    @property
    def repro_root(self) -> Path:
        return REPRO_ROOT

    @property
    def cache_settings(self) -> dict:
        return dict(self.settings.get("feature_cache") or {})

    @property
    def feature_cache_enabled(self) -> bool:
        return bool(self.cache_settings.get("enabled", True))

    @property
    def feature_cache_dir(self) -> Path:
        configured = self.cache_settings.get("dir", "results/cache/features")
        path = Path(configured)
        if not path.is_absolute():
            path = BUNDLE_ROOT / path
        return path.resolve()

    @property
    def feature_cache(self) -> FeatureCache:
        return FeatureCache(self.feature_cache_dir, enabled=self.feature_cache_enabled, refresh=self.refresh_cache)

    @property
    def validate_fasta(self) -> bool:
        return bool(self.cache_settings.get("validate_fasta", True))

    @property
    def optuna_storage_dir(self) -> Path:
        return BUNDLE_ROOT / "results" / "checkpoints" / "optuna"


def ensure_output_dirs(ctx: RunnerContext) -> None:
    for directory in (
        ctx.tables_dir,
        ctx.figures_dir,
        ctx.logs_dir,
        ctx.feature_cache_dir,
        ctx.optuna_storage_dir,
        BUNDLE_ROOT / "results" / "checkpoints" / "scripts",
        Path(ctx.paths["workspace"]["work_dir"]),
    ):
        directory.mkdir(parents=True, exist_ok=True)


def prepare_legacy_workspace(ctx: RunnerContext, *, require_inputs: bool) -> Path:
    """Create a disposable legacy-compatible project tree under results/work."""
    global _CURRENT_CONTEXT
    _CURRENT_CONTEXT = ctx
    ensure_output_dirs(ctx)
    workspace = legacy_workspace(ctx)
    shutil.copytree(LEGACY_SOURCE, workspace, dirs_exist_ok=True)
    _link_inputs(ctx, workspace=workspace, require_inputs=require_inputs)
    _patch_runtime_settings(workspace, ctx)
    return workspace


def restore_legacy_feature_snapshot(ctx: RunnerContext, experiment_id: str, exp_root: Path) -> dict[str, object]:
    return restore_feature_artifacts(ctx.feature_cache, f"legacy_{experiment_id}", exp_root)


def snapshot_legacy_feature_outputs(ctx: RunnerContext, experiment_id: str, exp_root: Path) -> dict[str, object]:
    if not bool(ctx.cache_settings.get("snapshot_legacy_features", True)):
        return {"namespace": experiment_id, "status": "disabled"}
    return snapshot_feature_artifacts(
        ctx.feature_cache,
        f"legacy_{experiment_id}",
        exp_root,
        [
            "data/**/*feature*.csv",
            "data/**/*feature*.csv.gz",
            "data/**/*features*.csv",
            "data/**/*features*.csv.gz",
            "data/**/*cache*.json",
            "data/**/*manifest*.csv",
            "data/**/*metadata*.json",
        ],
    )


def legacy_workspace(ctx: RunnerContext) -> Path:
    return Path(ctx.paths["workspace"]["work_dir"]) / "cgr_validation"


def _safe_link_or_copy(src: Path | None, dst: Path, *, require: bool) -> None:
    if src is None:
        if require:
            raise FileNotFoundError(f"No path configured for {dst}")
        return
    if not src.exists():
        if require:
            raise FileNotFoundError(f"Configured path does not exist: {src}")
        return
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        if src.is_dir() and dst.is_dir() and not dst.is_symlink():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            return
        if src.is_file() and dst.is_file() and not dst.is_symlink():
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _link_inputs(ctx: RunnerContext, *, workspace: Path, require_inputs: bool) -> None:
    raw = ctx.paths.get("raw_data", {})
    processed = ctx.paths.get("processed_helpers", {})
    exp_root = workspace / "cgr_validation_results" / "research" / "experiments"
    unified = exp_root / "supporting" / "2026_05_14_unified_uga_locked_manuscript_benchmark"

    mc3_source = raw.get("mc3_source_dir")
    hrd_assets = raw.get("hrd_assets_dir")
    kucab_raw = raw.get("kucab_raw_dir")
    pcawg = raw.get("pcawg_dir")

    _safe_link_or_copy(mc3_source, unified / "data" / "mc3_source", require=require_inputs)
    if processed.get("mc3_standard_features_dir") is not None:
        _safe_link_or_copy(processed["mc3_standard_features_dir"], unified / "data" / "mc3_source" / "features", require=require_inputs)
    _safe_link_or_copy(kucab_raw, unified / "data" / "raw", require=require_inputs)
    _safe_link_or_copy(hrd_assets, workspace / "cgr_validation_results" / "research" / "assets" / "EXP023_tcga_brca_hrd" / "TCGA-BRCA", require=require_inputs)
    _safe_link_or_copy(pcawg, workspace / "cgr_validation_results" / "research" / "data" / "pancan_pcawg_2020", require=require_inputs)


def _patch_runtime_settings(workspace: Path, ctx: RunnerContext) -> None:
    """Patch the disposable legacy workspace with the requested runtime profile."""
    replacements = {
        'TREE_METHOD = "gpu_hist"': f'TREE_METHOD = "{ctx.tree_method}"',
        'TREE_METHOD = "hist"': f'TREE_METHOD = "{ctx.tree_method}"',
        'default="gpu_hist"': f'default="{ctx.tree_method}"',
        '"tree_method": "gpu_hist"': f'"tree_method": "{ctx.tree_method}"',
        '"tree_method": args.tree_method': '"tree_method": args.tree_method',
        "REPEATS = 5": f"REPEATS = {ctx.cv_repeats}",
        "OUTER_FOLDS = 5": f"OUTER_FOLDS = {ctx.cv_folds}",
        "BOOTSTRAP = 1000": f"BOOTSTRAP = {ctx.bootstrap_iterations}",
        "N_ESTIMATORS = 250": f"N_ESTIMATORS = {ctx.xgb_estimators}",
        "N_ESTIMATORS = 100": f"N_ESTIMATORS = {ctx.xgb_estimators}",
        '"n_jobs": 1': f'"n_jobs": {ctx.xgb_n_jobs}',
        "XGB_N_JOBS = 4": f"XGB_N_JOBS = {ctx.xgb_n_jobs}",
        "from sklearn.linear_model import LogisticRegression, Ridge": "from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge",
        "from sklearn.linear_model import LogisticRegression\n": "from sklearn.linear_model import LogisticRegression\n",
        'Ridge(alpha=1.0)': 'ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=0)',
        'Ridge(alpha=1.0 / max(float(c_value), 1e-9))': 'ElasticNet(alpha=1.0 / max(float(c_value), 1e-9), l1_ratio=0.5, max_iter=5000, random_state=seed)',
        'solver="lbfgs",\n                    random_state=seed,': f'solver="saga",\n                    penalty="elasticnet",\n                    l1_ratio=0.5,\n                    n_jobs={ctx.xgb_n_jobs},\n                    random_state=seed,',
        'solver="lbfgs",\n                    random_state=stable_seed': f'solver="saga",\n                    penalty="elasticnet",\n                    l1_ratio=0.5,\n                    n_jobs={ctx.xgb_n_jobs},\n                    random_state=stable_seed',
    }
    for script in workspace.rglob("*.py"):
        text = script.read_text(encoding="utf-8")
        new_text = text
        for old, new in replacements.items():
            new_text = new_text.replace(old, new)
        if new_text != text:
            script.write_text(new_text, encoding="utf-8")


def run_python(script: Path, *, cwd: Path, log_path: Path, args: Iterable[str] = (), dry_run: bool = False) -> None:
    args_list = list(args)
    cmd = [sys.executable, str(script), *args_list]
    checkpoint_path = _script_checkpoint_path(script, args_list)

    def display_arg(value: str) -> str:
        try:
            path_value = Path(value)
        except TypeError:
            return str(value)
        if path_value.is_absolute():
            try:
                return str(path_value.relative_to(BUNDLE_ROOT))
            except ValueError:
                try:
                    return str(path_value.relative_to(LEGACY_WORKSPACE))
                except ValueError:
                    return str(value)
        return str(value)

    display_script = script
    try:
        display_script = script.relative_to(BUNDLE_ROOT)
    except ValueError:
        try:
            display_script = script.relative_to(LEGACY_WORKSPACE)
        except ValueError:
            pass
    display_cmd = ["python", str(display_script), *[display_arg(arg) for arg in args_list]]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(display_cmd) + "\n")
        if checkpoint_path.exists() and not dry_run:
            if _checkpoint_outputs_present(cwd, args_list):
                log.write(f"[checkpoint] skip completed script: {checkpoint_path.relative_to(BUNDLE_ROOT)}\n")
                return
            log.write(f"[checkpoint] stale completion marker without local outputs; rerunning: {checkpoint_path.relative_to(BUNDLE_ROOT)}\n")
    if dry_run:
        return
    env = os.environ.copy()
    src_path = str(BUNDLE_ROOT / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    warnings = env.get("PYTHONWARNINGS", "")
    pkg_warning_filter = "ignore:pkg_resources is deprecated"
    env["PYTHONWARNINGS"] = ",".join([part for part in [pkg_warning_filter, warnings] if part])
    env.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
    env.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 1))
    env.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 1))
    env.setdefault("NUMEXPR_NUM_THREADS", str(os.cpu_count() or 1))
    env.setdefault("CV_FOLDS", str(_CURRENT_CONTEXT.cv_folds if "_CURRENT_CONTEXT" in globals() else 5))
    env.setdefault("CV_REPEATS", str(_CURRENT_CONTEXT.cv_repeats if "_CURRENT_CONTEXT" in globals() else 1))
    env.setdefault("XGB_N_ESTIMATORS", str(_CURRENT_CONTEXT.xgb_estimators if "_CURRENT_CONTEXT" in globals() else 160))
    env.setdefault("BOOTSTRAP_ITERATIONS", str(_CURRENT_CONTEXT.bootstrap_iterations if "_CURRENT_CONTEXT" in globals() else 200))
    env.setdefault("KUCAB_ORIGINAL_DATA_CV_REPEATS", "1")
    env.setdefault("KUCAB_N_RESAMPLES", "1")
    pythonpath_parts = [str(script.parent), str(cwd), str(BUNDLE_ROOT / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    sanitize_text_file(log_path)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    atomic_write_json(
        checkpoint_path,
        {
            "args": args_list,
            "cwd": sanitize_text(str(cwd)),
            "returncode": int(proc.returncode),
            "script": sanitize_text(str(script)),
            "script_sha256": _file_sha256(script),
            "status": "completed",
        },
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_outputs_present(cwd: Path, args_list: list[str] | None = None) -> bool:
    """Guard against skipping after cleanup removed a legacy script's outputs."""
    root = Path(cwd)
    args_list = args_list or []
    if "--model-id" in args_list:
        idx = args_list.index("--model-id")
        if idx + 1 < len(args_list):
            model_id = args_list[idx + 1]
            safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in model_id)[:120]
            expected = root / "data" / f"full_panel_confirmation_results__{safe}.csv"
            return expected.exists() and expected.stat().st_size > 0
    for rel in ("data", "tables", "figures"):
        directory = root / rel
        if directory.exists() and any(path.is_file() for path in directory.rglob("*")):
            return True
    return False


def _script_checkpoint_path(script: Path, args_list: list[str]) -> Path:
    payload = {
        "args": args_list,
        "script": sanitize_text(str(script)),
        "script_sha256": _file_sha256(script) if script.exists() else "missing",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in Path(script).stem).strip("_")[:80]
    return BUNDLE_ROOT / "results" / "checkpoints" / "scripts" / f"{safe_name}_{digest}.json"


def sanitize_text(value: str) -> str:
    """Replace machine-local paths in generated text with portable placeholders."""
    replacements = {
        str(BUNDLE_ROOT): "${BUNDLE_ROOT}",
        str(REPRO_ROOT): "${REPRO_ROOT}",
        str(LEGACY_WORKSPACE): "${BUNDLE_ROOT}/results/work/cgr_validation",
        str(Path(sys.prefix)): "${CONDA_PREFIX}",
    }
    out = value
    for old, new in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        out = out.replace(old, new)
    return out


def sanitize_text_file(path: Path) -> None:
    if not path.exists() or path.suffix.lower() not in TEXT_SUFFIXES:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    new_text = sanitize_text(text)
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")


def sanitize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    object_columns = out.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        out[column] = out[column].map(lambda value: sanitize_text(value) if isinstance(value, str) else value)
    return out


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_artifacts(
    sources: Iterable[Path],
    destination: Path,
    *,
    prefix: str,
    source_root: Path,
    dry_run: bool = False,
) -> list[dict[str, str]]:
    """Copy regenerated artifacts into a flat, prefixed results folder."""
    rows: list[dict[str, str]] = []
    for src in sorted({path for path in sources if path.exists() and path.is_file()}):
        try:
            rel = src.relative_to(source_root)
        except ValueError:
            rel = Path(src.name)
        safe_name = "_".join(rel.parts)
        dst = destination / f"{prefix}_{safe_name}"
        rows.append({"source": str(rel), "output": str(dst.relative_to(BUNDLE_ROOT)), "status": "planned" if dry_run else "copied"})
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            sanitize_text_file(dst)
    return rows


def glob_many(root: Path, patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted({path for path in files if path.is_file()})


def add_source_file_column(frame: pd.DataFrame, source_file: str, experiment_id: str) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "source_file", source_file)
    out.insert(0, "experiment_id", experiment_id)
    return out


def read_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def write_summary_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"status": "no_rows"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows([{key: sanitize_text(value) if isinstance(value, str) else value for key, value in row.items()} for row in rows])


def write_endpoint_results(frames: list[pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        sanitize_frame(pd.concat(frames, ignore_index=True, sort=False)).to_csv(path, index=False)
    else:
        pd.DataFrame(columns=["experiment_id", "endpoint", "metric", "score"]).to_csv(path, index=False)
