"""Fetch OncoKB cancer-gene role annotations at build time.

The OncoKB gene-role table is NOT bundled with this package. OncoKB's Terms
of Use (https://www.oncokb.org/terms) limit use to academic/research settings
and restrict redistribution, so this module downloads the current public
cancer-gene list from the OncoKB public API (no key required) and converts it
to the TSV layout the feature builder reads. The file is cached under
``biomaf_v4/resources/oncokb_cancer_genes.tsv`` after the first download.

Source: https://public.api.oncokb.org/api/v1/utils/cancerGeneList
geneType mapping:
    ONCOGENE         -> oncogene=True
    TSG              -> tsg=True
    ONCOGENE_AND_TSG -> both True
    INSUFFICIENT_EVIDENCE / NEITHER / null -> both False

Note: OncoKB updates its gene list continuously and the API is not versioned,
so a fresh download may differ slightly from the snapshot used to build the
published feature matrices. The published matrices remain the artifact of
record for every number reported in the manuscript.
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

ONCOKB_API_URL = "https://public.api.oncokb.org/api/v1/utils/cancerGeneList"
ONCOKB_TERMS_URL = "https://www.oncokb.org/terms"

COLUMNS = ["hugo_symbol", "entrez_gene_id", "oncogene", "tsg", "source_url"]


def fetch_oncokb_tsv(path: str | Path, url: str = ONCOKB_API_URL) -> Path:
    """Download the OncoKB cancer-gene list and write it as a TSV to *path*."""
    path = Path(path)
    req = urllib.request.Request(url, headers={"User-Agent": "biomaf-v4/4.0.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        genes = json.load(resp)
    rows = []
    for g in genes:
        gene_type = (g.get("geneType") or "").upper()
        rows.append(
            {
                "hugo_symbol": g.get("hugoSymbol", ""),
                "entrez_gene_id": g.get("entrezGeneId", ""),
                "oncogene": gene_type in {"ONCOGENE", "ONCOGENE_AND_TSG"},
                "tsg": gene_type in {"TSG", "ONCOGENE_AND_TSG"},
                "source_url": url,
            }
        )
    rows.sort(key=lambda r: str(r["hugo_symbol"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


def ensure_oncokb_tsv(path: str | Path, url: str = ONCOKB_API_URL) -> Path:
    """Return the OncoKB TSV path, downloading it first if it is missing."""
    path = Path(path)
    if path.exists():
        return path
    try:
        return fetch_oncokb_tsv(path, url=url)
    except Exception as exc:
        raise RuntimeError(
            f"OncoKB gene-role file not found at {path} and the download from "
            f"{url} failed ({exc}). See OncoKB's Terms of Use: " + ONCOKB_TERMS_URL
        ) from exc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "resources" / "oncokb_cancer_genes.tsv"),
        help="Where to write the TSV.",
    )
    parser.add_argument("--url", default=ONCOKB_API_URL)
    args = parser.parse_args()
    out = fetch_oncokb_tsv(args.output, url=args.url)
    print(f"wrote {out}")
