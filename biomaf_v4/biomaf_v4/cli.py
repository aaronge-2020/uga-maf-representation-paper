"""Command-line entry point: ``biomaf-v4 --maf input.maf[.gz] --out features.csv.gz``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .features import featurize_maf, feature_names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biomaf-v4",
        description=(
            "Compute Bio MAF v4 features (948 per-tumor summaries: 943 core + "
            "5 optional burden controls) from an annotated MAF file."
        ),
    )
    parser.add_argument("--maf", required=True, help="Input MAF file (tab-separated, plain or .gz).")
    parser.add_argument("--out", required=True, help="Output feature matrix (.csv or .csv.gz).")
    parser.add_argument(
        "--samples",
        default=None,
        help="Optional comma-separated sample IDs to featurize (default: all Tumor_Sample_Barcode values, first 12 chars).",
    )
    parser.add_argument(
        "--sample-list",
        default=None,
        help="Optional text file with one sample ID per line.",
    )
    parser.add_argument(
        "--no-optional-controls",
        action="store_true",
        help="Emit only the 943 core features (drop the 5 optional burden controls).",
    )
    parser.add_argument("--chunksize", type=int, default=350_000, help="MAF rows per chunk (default: 350000).")
    parser.add_argument("--columns-out", default=None, help="Optional path to write the feature-name list (one per line).")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    maf_path = Path(args.maf)
    if not maf_path.exists():
        print(f"error: MAF not found: {maf_path}", file=sys.stderr)
        return 2

    sample_ids: list[str] | None = None
    if args.sample_list:
        sample_ids = [line.strip() for line in Path(args.sample_list).read_text().splitlines() if line.strip()]
    elif args.samples:
        sample_ids = [s.strip() for s in args.samples.split(",") if s.strip()]

    matrix = featurize_maf(
        maf_path,
        sample_ids,
        chunksize=args.chunksize,
        include_optional_controls=not args.no_optional_controls,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out_path, compression="gzip" if out_path.suffix == ".gz" else None)
    if args.columns_out:
        Path(args.columns_out).write_text("\n".join(matrix.columns) + "\n", encoding="utf-8")
    print(f"wrote {matrix.shape[0]} samples x {matrix.shape[1]} features -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
