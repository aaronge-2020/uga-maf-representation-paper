#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
EXP021_SCRIPT = REPO / "cgr_validation_results/research/scripts/EXP021_genome_wide_bicgr/EXP021_genome_wide_atlas_benchmark.py"
EXP021_ATLAS = REPO / "cgr_validation_results/research/reports/EXP021_genome_wide_bicgr/EXP021_atlas_genome_wide_45mer.json"
ARCHIVE_EXP021_ATLAS = REPO / "cgr_validation_results/research/archive/pre_EXP022_20260426/reports/EXP021_genome_wide_bicgr/EXP021_atlas_genome_wide_45mer.json"
EXP022_ATLAS = REPO / "cgr_validation_results/research/data/EXP022_atlas_genome_wide_45mer_universal.json"


def load_or_build_92(source: Path, rebuild: bool) -> dict[str, list[float]]:
    if source.is_file() and not rebuild:
        return json.loads(source.read_text(encoding="utf-8"))
    if source == EXP021_ATLAS and ARCHIVE_EXP021_ATLAS.is_file() and not rebuild:
        return json.loads(ARCHIVE_EXP021_ATLAS.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("exp021_atlas", EXP021_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import atlas builder from {EXP021_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_genome_wide_atlas(REPO, canonical=False)


def truncate_context(atlas92: dict[str, list[float]], d: int) -> dict[str, list[float]]:
    if not 1 <= d <= 23:
        raise ValueError("d must be in 1..23")
    out = {}
    for ctx, vec in atlas92.items():
        arr = np.asarray(vec, dtype=float)
        if arr.shape != (92,):
            raise ValueError(f"{ctx} has shape {arr.shape}, expected 92")
        out[ctx] = [round(float(x), 10) for x in np.concatenate([arr[i * 23 : i * 23 + d] for i in range(4)])]
    if len(out) != 64 or any(len(v) != 4 * d for v in out.values()):
        raise ValueError("Atlas integrity check failed")
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser(description="Build EXP-022 universal context atlas.")
    ap.add_argument("--source-atlas", type=Path, default=EXP021_ATLAS)
    ap.add_argument("--out", type=Path, default=EXP022_ATLAS)
    ap.add_argument("--depth", type=int, default=13)
    ap.add_argument("--rebuild", action="store_true", help="Rebuild source atlas from chromosome FASTAs.")
    args = ap.parse_args()

    atlas = truncate_context(load_or_build_92(args.source_atlas, args.rebuild), args.depth)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(atlas, separators=(",", ":")), encoding="utf-8")
    size = args.out.stat().st_size
    print(f"Wrote {args.out}")
    print(f"Contexts: {len(atlas)}; features/context: {4 * args.depth}; size: {size:,} bytes")
    if size >= 100_000:
        raise SystemExit("Atlas exceeds 100KB")


if __name__ == "__main__":
    main()
