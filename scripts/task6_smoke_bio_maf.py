#!/usr/bin/env python3
"""Task 6 smoke test: build the Bio MAF feature matrix (no model training).

Exercises the full feature-build path (_load_mc3_maf_features) that the
Table-1 run depends on: MC3 MAF reading, OncoKB role loading, IntOGen/hotspot
resources. Writes to the worktree's results/cache (gitignored); the clean
checkout run uses its own empty cache.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ["CGR_DATASETS_DIR"] = str(REPO_ROOT.parent / "github_exports_datasets")

from utils.config import load_yaml, resolve_paths_map  # noqa: E402
from utils.runner_support import RunnerContext  # noqa: E402
import runners.run_main_manuscript_complete_panel as panel  # noqa: E402

settings = load_yaml(REPO_ROOT / "config" / "experiment_settings.table1.yaml")
panel_settings = dict(settings["experiments"]["main_manuscript_complete_panel"])
paths = resolve_paths_map(load_yaml(REPO_ROOT / "config" / "paths.yaml"))
ctx = RunnerContext(settings=settings, paths=paths)

t0 = time.time()
frame, cache_key = panel._load_mc3_maf_features(ctx, "maf_stack_bio_nested_fs", panel_settings)
elapsed = time.time() - t0
print(f"SMOKE OK: shape={frame.shape} cache_key={cache_key} wall={elapsed:.1f}s "
      f"per_tumour={elapsed / frame.shape[0]:.4f}s")
print("columns sample:", list(frame.columns[:3]), "...", list(frame.columns[-3:]))
print("candidate sets:", {k: len(v) for k, v in (frame.attrs.get('candidate_feature_set_counts') or {}).items()})
