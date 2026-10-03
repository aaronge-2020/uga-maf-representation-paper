#!/usr/bin/env python3
"""Download the OncoKB cancer-gene role table used by the Bio MAF feature builder.

The derived TSV (config/feature_resources/oncokb_cancer_genes.tsv) is not
committed to the repository because OncoKB's Terms of Use
(https://www.oncokb.org/terms) restrict redistribution. Run this script once
before building features; the pipeline calls it automatically when the file
is missing.

Usage:
    python config/feature_resources/fetch_oncokb_cancer_genes.py [--output PATH]

See src/utils/oncokb_fetch.py for the API endpoint and the geneType mapping.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.oncokb_fetch import ONCOKB_API_URL, fetch_oncokb_tsv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "config" / "feature_resources" / "oncokb_cancer_genes.tsv"),
        help="Where to write the TSV.",
    )
    parser.add_argument("--url", default=ONCOKB_API_URL, help="OncoKB API endpoint.")
    args = parser.parse_args()
    out = fetch_oncokb_tsv(args.output, url=args.url)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
