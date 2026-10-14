"""Command-line entry point: ``biomaf --maf input.maf[.gz] --out features.csv.gz``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .features import featurize_maf, feature_names


# Maps MAF input columns to the feature groups they activate.
# Used by --check-input to tell the user what will be computed.
COLUMN_FEATURE_GROUPS: list[tuple[str, str, str]] = [
    ("Tumor_Sample_Barcode", "required", "Sample identity (always active)"),
    ("Hugo_Symbol", "gene identity", "802 per-gene columns + 16 driver-tier columns + 12 OncoKB role columns"),
    ("SYMBOL", "gene identity (alt)", "Fallback for Hugo_Symbol; same 802 per-gene columns"),
    ("Variant_Type", "VEP spectra", "Variant-type fractions within the 100 VEP-spectra features"),
    ("IMPACT", "VEP spectra", "vep_impact_* count and fraction columns (10)"),
    ("Consequence", "VEP spectra", "vep_consequence_* count and fraction columns (52)"),
    ("CANONICAL", "VEP spectra", "vep_canonical_* columns"),
    ("BIOTYPE", "VEP spectra", "vep_biotype_* columns"),
    ("SIFT", "VEP spectra", "vep_sift_* columns"),
    ("PolyPhen", "VEP spectra", "vep_polyphen_* columns"),
    ("Protein_position", "hotspots", "Residue-level hotspot matching (with Amino_acids / HGVSp_Short)"),
    ("Amino_acids", "hotspots", "Residue-level hotspot matching (with Protein_position)"),
    ("HGVSp_Short", "hotspots", "Exact-variant hotspot matching"),
    ("t_alt_count", "VAF summaries", "5 VAF columns (mean, SD, median, p90, max); needs t_depth"),
    ("t_depth", "VAF summaries", "5 VAF columns (mean, SD, median, p90, max); needs t_alt_count"),
]

# Feature groups that are always active regardless of input columns.
ALWAYS_ACTIVE = [
    "Burden controls (5): log10 SBS/DBS/ID burden, DBS and ID fractions — computed from row counts",
    "Driver-gene panel structure (401 genes): fixed reference, no input needed",
]


def check_input(maf_path: Path, chunksize: int = 350_000) -> int:
    """Inspect a MAF file and report which feature groups will be active.

    Reads only the header plus a small sample of rows. Returns 0 on success.
    """
    import gzip

    opener = gzip.open if str(maf_path).endswith(".gz") else open
    with opener(maf_path, "rt") as handle:
        header_line = handle.readline()
    if not header_line:
        print(f"error: empty file: {maf_path}", file=sys.stderr)
        return 2
    columns = [c.strip() for c in header_line.rstrip("\n").split("\t")]
    col_set = set(columns)

    # Count rows and distinct samples from a bounded sample.
    n_rows = 0
    samples: set[str] = set()
    for chunk in pd.read_csv(
        maf_path, sep="\t", usecols=["Tumor_Sample_Barcode"],
        dtype=str, chunksize=chunksize,
    ):
        n_rows += len(chunk)
        for barcode in chunk["Tumor_Sample_Barcode"].dropna().unique():
            samples.add(str(barcode)[:12])
        if n_rows >= chunksize:
            break

    print(f"MAF: {maf_path}")
    print(f"Columns present ({len(columns)}): {', '.join(columns)}")
    print(f"Rows scanned: {n_rows}{'+' if n_rows >= chunksize else ''}; "
          f"distinct samples (12-char): {len(samples)}")
    print()

    print("Feature groups:")
    for col, group, what in COLUMN_FEATURE_GROUPS:
        status = "PRESENT " if col in col_set else "missing "
        print(f"  [{status}] {col:22s} -> {group}: {what}")
    print()
    print("Always active (no input columns needed):")
    for item in ALWAYS_ACTIVE:
        print(f"  [active] {item}")
    print()

    # Summarize what will be zero-filled.
    missing_vep = [c for c, g, _ in COLUMN_FEATURE_GROUPS
                   if g == "VEP spectra" and c not in col_set]
    if missing_vep:
        print(f"Note: {len(missing_vep)} VEP annotation columns missing "
              f"({', '.join(missing_vep)}).")
        print("The corresponding VEP-spectra features will be zero-filled or "
              "fall back to 'unknown'. Note: without IMPACT/CONSEQUENCE the "
              "pipeline cannot identify functional events, which also disables "
              "the per-gene functional columns (see README 'Without VEP "
              "annotation').")
    if "Hugo_Symbol" not in col_set and "SYMBOL" not in col_set:
        print("WARNING: no Hugo_Symbol/SYMBOL column. The 802 per-gene "
              "identity columns will be all zeros, which cost 0.122 balanced "
              "accuracy points in the manuscript ablation. Gene symbols are "
              "strongly recommended.")
    if "t_alt_count" not in col_set or "t_depth" not in col_set:
        print("Note: t_alt_count/t_depth missing; the 5 VAF summary columns "
              "will be zero-filled.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biomaf",
        description=(
            "Compute Bio MAF features (948 per-tumor summaries: 943 core + "
            "5 optional burden controls) from an annotated MAF file."
        ),
    )
    parser.add_argument("--maf", required=True, help="Input MAF file (tab-separated, plain or .gz).")
    parser.add_argument("--out", required=False, default=None,
                        help="Output feature matrix (.csv or .csv.gz). Not needed with --check-input.")
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
    parser.add_argument(
        "--check-input",
        action="store_true",
        help="Inspect the MAF and report which columns are present/missing and "
             "which feature groups will be active vs zero-filled. No features are computed.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    maf_path = Path(args.maf)
    if not maf_path.exists():
        print(f"error: MAF not found: {maf_path}", file=sys.stderr)
        return 2

    if args.check_input:
        return check_input(maf_path, chunksize=args.chunksize)

    if not args.out:
        print("error: --out is required unless --check-input is given", file=sys.stderr)
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
