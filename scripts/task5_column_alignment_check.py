#!/usr/bin/env python3
"""Task 5 (PCAWG audit): column-alignment check.

Verifies that the PCAWG scoring tables line up with the frozen models'
training schemas:
  1. Sig+MAF model: results/frozen_models/cancer_type_top20__signatures_plus_MAF_stack
     (feature_order.txt, 1,130 names) vs the scored table
     pcawg_sigmaf_1130_features.csv.gz (default: data/pcawg/pcawg_sigmaf_1130_features.csv.gz;
     not deposited in the repo, copy it from the PCAWG transfer work dir first).
  2. patterns-only model: results/frozen_models/cancer_type_top20__standard_sbs96_id83
     (feature_order.txt, 182 names) vs
     results/external_validation/aaron_email/pcawg/pcawg_standard_sbs96_id83_features.csv.gz.

Checks: identical column SET (no missing, no extra), identical ORDER, and that
the name-based reindex used by the scoring code (X.reindex(columns=order)) is a
no-op on the actual tables (i.e. alignment by name == alignment by position here).

Usage: python3 scripts/task5_column_alignment_check.py [--sigmaf-features PATH]
Output: printed report; save with tee to results/tables/task5_column_alignment_check.txt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--sigmaf-features",
                    default=str(REPO / "data" / "pcawg" / "pcawg_sigmaf_1130_features.csv.gz"),
                    help="Sig+MAF scored PCAWG features table. Not deposited in the repo: "
                         "copy data/pcawg/pcawg_sigmaf_1130_features.csv.gz from the PCAWG "
                         "transfer work dir (~/workspace/pcawg_sigmaf_transfer/features/) or pass its path here.")
args = parser.parse_args()


def check(model_dir: Path, features_path: str, label: str) -> bool:
    ok = True
    order = [l.strip() for l in (model_dir / "feature_order.txt").read_text().splitlines() if l.strip()]
    X = pd.read_csv(features_path, index_col=0)
    cols = list(X.columns)
    missing = [c for c in order if c not in cols]
    extra = [c for c in cols if c not in order]
    order_match = cols == order
    reindexed = X.reindex(columns=order)
    noop = reindexed.columns.tolist() == cols and bool((reindexed.to_numpy() == X.to_numpy()).all())
    print(f"--- {label} ---")
    print(f"model dir        : {model_dir}")
    print(f"features table   : {features_path}")
    print(f"schema features  : {len(order)}")
    print(f"table shape      : {X.shape}")
    print(f"missing from table: {len(missing)} {missing[:5]}")
    print(f"extra in table     : {len(extra)} {extra[:5]}")
    print(f"table column order == schema order: {order_match}")
    print(f"name-based reindex is a no-op (name == position alignment): {noop}")
    if missing or extra or not order_match or not noop:
        ok = False
    print(f"ALIGNMENT_OK: {ok}\n")
    return ok


ok1 = check(REPO / "results/frozen_models/cancer_type_top20__signatures_plus_MAF_stack",
            args.sigmaf_features, "Sig+MAF (signatures_plus_MAF_stack)")
ok2 = check(REPO / "results/frozen_models/cancer_type_top20__standard_sbs96_id83",
            str(REPO / "results/external_validation/aaron_email/pcawg/pcawg_standard_sbs96_id83_features.csv.gz"),
            "patterns-only (standard_sbs96_id83)")
print("OVERALL:", "PASS" if (ok1 and ok2) else "FAIL")
raise SystemExit(0 if (ok1 and ok2) else 1)
