"""Fetch OncoKB cancer-gene role annotations at build time.

The OncoKB gene-role table is intentionally NOT committed to this repository.
OncoKB's Terms of Use (https://www.oncokb.org/terms) limit use to
academic/research settings and restrict redistribution, so instead of
bundling the derived file, this module downloads the current public
cancer-gene list from the OncoKB public API (no key required) and converts
it to the TSV layout the feature builder reads.

Source: https://public.api.oncokb.org/api/v1/utils/cancerGeneList
geneType mapping (as documented in
config/feature_resources/oncokb_cancer_genes.PROVENANCE.md):
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


def download_oncokb_genes(url: str = ONCOKB_API_URL, timeout: int = 120) -> list[dict]:
    """Download the raw cancer-gene list JSON from the OncoKB public API."""
    req = urllib.request.Request(url, headers={"User-Agent": "uga-maf-representation-paper/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def genes_to_rows(genes: list[dict], source_url: str = ONCOKB_API_URL) -> list[dict]:
    """Map the API response to the TSV row layout the feature builder reads."""
    rows = []
    for g in genes:
        gene_type = (g.get("geneType") or "").upper()
        rows.append(
            {
                "hugo_symbol": g.get("hugoSymbol", ""),
                "entrez_gene_id": g.get("entrezGeneId", ""),
                "oncogene": gene_type in {"ONCOGENE", "ONCOGENE_AND_TSG"},
                "tsg": gene_type in {"TSG", "ONCOGENE_AND_TSG"},
                "source_url": source_url,
            }
        )
    rows.sort(key=lambda r: str(r["hugo_symbol"]))
    return rows


def write_tsv(rows: list[dict], path: Path) -> Path:
    """Write mapped rows to *path* as a tab-separated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


def fetch_oncokb_tsv(path: str | Path, url: str = ONCOKB_API_URL) -> Path:
    """Download the OncoKB cancer-gene list and write it as a TSV to *path*."""
    genes = download_oncokb_genes(url)
    rows = genes_to_rows(genes, source_url=url)
    return write_tsv(rows, path)


def ensure_oncokb_tsv(path: str | Path, url: str = ONCOKB_API_URL) -> Path:
    """Return the OncoKB TSV path, downloading it first if it is missing.

    Raises RuntimeError with build instructions if the download fails.
    """
    path = Path(path)
    if path.exists():
        return path
    try:
        return fetch_oncokb_tsv(path, url=url)
    except Exception as exc:  # network or API failure: fail loudly
        raise RuntimeError(
            f"OncoKB gene-role file not found at {path} and the download from "
            f"{url} failed ({exc}). Fetch it manually with "
            "`python config/feature_resources/fetch_oncokb_cancer_genes.py`, "
            "per OncoKB's Terms of Use: " + ONCOKB_TERMS_URL
        ) from exc
