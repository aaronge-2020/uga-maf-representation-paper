#!/usr/bin/env python3
"""Build a channel-to-UGA basis matrix from a registered model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from uga_atlas.channels import build_uga_basis
from uga_atlas.context import ATLAS52, load_context_atlas
from uga_atlas.models import get_uga_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=Path, required=True, help="CSV/TSV file containing channel labels.")
    parser.add_argument("--column", default="Type", help="Column containing channel labels.")
    parser.add_argument("--model", default="master_spec_sbs_dbs_d22", help="Registered UGA model name.")
    parser.add_argument("--modality", choices=["SBS", "DBS", "ID"], default=None)
    parser.add_argument("--atlas", type=Path, default=ATLAS52)
    parser.add_argument("--out", type=Path, required=True, help="Output .npy basis path.")
    parser.add_argument("--diagnostics", type=Path, default=None, help="Optional diagnostics CSV path.")
    args = parser.parse_args()

    sep = "\t" if args.channels.suffix.lower() in {".tsv", ".txt"} else ","
    channels = pd.read_csv(args.channels, sep=sep)[args.column].astype(str).tolist()
    spec = get_uga_model(args.model)
    atlas = None
    if args.modality in {"SBS", "DBS"} and spec.context_source == "genome_atlas":
        atlas = load_context_atlas(args.atlas, spec.d_context)
    basis, diagnostics = build_uga_basis(channels, spec, atlas=atlas, modality=args.modality)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, basis)
    if args.diagnostics:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(args.diagnostics, index=False)
    print(f"Wrote {args.out} with shape {basis.shape}")


if __name__ == "__main__":
    main()
