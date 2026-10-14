#!/usr/bin/env python3
"""Export the exact feature list for every Table 1 representation. (Task 6)

Reads the ACTUAL artefacts, never reconstructed from memory:
  mutation_count    <- <bundle>/datasets/tcga_mc3/features/features_burden_only.csv
  mutation_patterns <- <bundle>/datasets/tcga_mc3/features/features_standard_sbs96_id83.csv.gz
  bio_maf_v4        <- results/tables/quick_bio_v4_maf_features.csv.gz
                       (the published 948-feature matrix; the current code
                       builder emits the corrected 910-feature set after
                       commit 7984d67 removed 38 undocumented role-matched
                       columns -- see task6_notes.md)
  sig_plus_maf      <- signature_matrix__-prefixed mutation_patterns columns
                       concatenated with the bio_maf_v4 columns, exactly as
                       runners.run_main_manuscript_complete_panel builds the
                       signatures_plus_MAF_stack matrix (182 + 948 = 1130)

Plus the 20 cancer types of the cancer_type_top20 endpoint from
src/utils/endpoint_registry.py.

Usage:
    python scripts/task6_export_feature_lists.py [--datasets-dir DIR]
                                                [--out-dir DIR]

Defaults: --datasets-dir ../github_exports_datasets,
          --out-dir results/tables/task6_feature_lists
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def read_header_csv(path: Path) -> list[str]:
    import pandas as pd

    frame = pd.read_csv(path, nrows=0)
    cols = [str(col) for col in frame.columns]
    # All three bundle/published matrices store sample ids in the first
    # column ('Unnamed: 0' / 'sample'); the feature list excludes it.
    return cols[1:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir).expanduser() if args.datasets_dir else (REPO_ROOT.parent / "github_exports_datasets")
    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "results" / "tables" / "task6_feature_lists")
    out_dir.mkdir(parents=True, exist_ok=True)

    mc3 = datasets_dir / "datasets" / "tcga_mc3"
    burden_path = mc3 / "features" / "features_burden_only.csv"
    standard_path = mc3 / "features" / "features_standard_sbs96_id83.csv.gz"
    bio_path = REPO_ROOT / "results" / "tables" / "quick_bio_v4_maf_features.csv.gz"
    for path in (burden_path, standard_path, bio_path):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")

    burden_cols = read_header_csv(burden_path)
    standard_cols = read_header_csv(standard_path)
    bio_feature_cols = read_header_csv(bio_path)

    sig_cols = [f"signature_matrix__{c}" for c in standard_cols]
    sig_plus_maf_cols = sig_cols + bio_feature_cols

    from utils.endpoint_registry import TCGA_CANCER_TYPE_TOP20

    artefacts = {
        "mutation_count.txt": (
            "Table 1 representation 'burden_only' (mutation count). "
            f"Source: {burden_path} (header row of the deposited bundle file).",
            burden_cols,
        ),
        "mutation_patterns.txt": (
            "Table 1 representation 'standard_sbs96_id83' (mutation patterns: "
            "COSMIC SBS96 + ID83 channels). "
            f"Source: {standard_path} (header row of the deposited bundle file).",
            standard_cols,
        ),
        "bio_maf_v4.txt": (
            "Table 1 representation 'MAF_stack_only' (Bio MAF v4, 948 features). "
            f"Source: {bio_path.name} (header row of the published feature matrix, "
            f"minus the sample-id column). NOTE: this is the "
            "historical 948-feature matrix; the current code builder emits the "
            "corrected 910-feature set (commit 7984d67). See task6_notes.md.",
            bio_feature_cols,
        ),
        "sig_plus_maf.txt": (
            "Table 1 representation 'signatures_plus_MAF_stack' (Sig+MAF, 1130 features = "
            "182 signature_matrix__-prefixed mutation-pattern columns + 948 Bio MAF v4 columns), "
            "concatenated in exactly the order the production runner builds it.",
            sig_plus_maf_cols,
        ),
        "task6_cancer_types.txt": (
            "The 20 cancer types of the cancer_type_top20 endpoint "
            "(src/utils/endpoint_registry.py::TCGA_CANCER_TYPE_TOP20).",
            list(TCGA_CANCER_TYPE_TOP20),
        ),
    }

    for name, (provenance, cols) in artefacts.items():
        out = out_dir / name
        with out.open("w", encoding="utf-8") as handle:
            handle.write(f"# {name}\n")
            handle.write(f"# provenance: {provenance}\n")
            handle.write(f"# n = {len(cols)}\n")
            for col in cols:
                handle.write(f"{col}\n")
        print(f"wrote {out.relative_to(REPO_ROOT)} ({len(cols)} entries)")

    # Sanity assertions against the manuscript's recorded counts.
    assert len(burden_cols) == 3, f"burden_only should have 3 features, got {len(burden_cols)}"
    assert len(standard_cols) == 182, f"standard_sbs96_id83 should have 182 features, got {len(standard_cols)}"
    assert len(bio_feature_cols) == 948, f"bio_maf_v4 should have 948 features, got {len(bio_feature_cols)}"
    assert len(sig_plus_maf_cols) == 1130, f"sig_plus_maf should have 1130 features, got {len(sig_plus_maf_cols)}"
    assert len(TCGA_CANCER_TYPE_TOP20) == 20
    print("counts verified: 3 / 182 / 948 / 1130 / 20")


if __name__ == "__main__":
    main()
