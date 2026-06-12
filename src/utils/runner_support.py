"""Shared helpers for production experiment runners."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from utils.config import BUNDLE_ROOT, REPRO_ROOT
from utils.feature_cache import FeatureCache


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


def sanitize_text(value: str) -> str:
    """Replace machine-local paths in generated text with portable placeholders."""
    replacements: dict[str, str] = {}
    for old, new in (
        (str(BUNDLE_ROOT), "${BUNDLE_ROOT}"),
        (str(REPRO_ROOT), "${REPRO_ROOT}"),
        (str(Path(sys.prefix)), "${CONDA_PREFIX}"),
    ):
        replacements[old] = new
        replacements[old.replace("\\", "\\\\")] = new.replace("\\", "\\\\")
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


def write_summary_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"status": "no_rows"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows([{key: sanitize_text(value) if isinstance(value, str) else value for key, value in row.items()} for row in rows])
