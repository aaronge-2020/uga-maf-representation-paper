"""Configuration loading and path resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


BUNDLE_ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = BUNDLE_ROOT.parent


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file from an absolute path, cwd-relative path, or bundle-relative path."""
    raw = Path(os.path.expandvars(str(path))).expanduser()
    if raw.is_absolute():
        resolved = raw.resolve()
    elif raw.exists():
        resolved = raw.resolve()
    else:
        resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected mapping in {resolved}")
    return loaded


def resolve_path(path: str | Path | None, *, base: Path | None = None) -> Path | None:
    """Resolve a possibly relative path.

    Supported placeholders:
    - `${bundle_root}` for the top-level bundle folder.
    - `${repro_root}` for the self-contained cgr_validation reproduction root.
    - `${cwd}` for the current working directory.
    """
    if path is None:
        return None
    text = os.path.expandvars(str(path))
    text = (
        text.replace("${bundle_root}", str(BUNDLE_ROOT))
        .replace("${repro_root}", str(REPRO_ROOT))
        .replace("${cwd}", str(Path.cwd()))
    )
    out = Path(text).expanduser()
    if not out.is_absolute():
        out = (base or BUNDLE_ROOT) / out
    return out.resolve()


def resolve_paths_map(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve all path-like values in the paths config."""
    out = dict(config)
    for section in ("raw_data", "processed_helpers", "workspace", "feature_cache", "renderer"):
        values = dict(out.get(section) or {})
        for key, value in list(values.items()):
            values[key] = resolve_path(value) if value is not None else None
        out[section] = values
    return out


def enabled(settings: dict[str, Any], experiment_id: str) -> bool:
    """Return whether an experiment is enabled in settings."""
    return bool(((settings.get("experiments") or {}).get(experiment_id) or {}).get("enabled", False))


def experiment_settings(settings: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    """Return a mutable copy of one experiment's settings."""
    return dict(((settings.get("experiments") or {}).get(experiment_id) or {}))
