#!/usr/bin/env python3
"""Inference for the frozen HRD_Score / signatures_plus_MAF_stack model.

Self-contained: needs only xgboost, numpy and pandas. No imports from the project repo.

    python predict.py --features my_cohort_features.csv --out predictions.csv

The features file must be a table with one row per sample and one column per feature. Column
ORDER does not matter (this script reorders), but the column NAMES must match feature_order.txt.
Missing columns are an error rather than a silent zero-fill, because a silently-zeroed feature
block is indistinguishable from a genuine measurement of zero and would corrupt the prediction.
"""

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb

HERE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="CSV/TSV with one row per sample")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--index-col", default=None, help="sample identifier column (default: first)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="fill absent feature columns with 0 instead of failing (use with care)")
    args = ap.parse_args()

    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    order = [line.strip() for line in (HERE / "feature_order.txt").read_text(encoding="utf-8").splitlines() if line.strip()]

    sep = "\t" if str(args.features).endswith((".tsv", ".txt", ".tsv.gz")) else ","
    frame = pd.read_csv(args.features, sep=sep)
    index_col = args.index_col or frame.columns[0]
    frame = frame.set_index(index_col)

    missing = [c for c in order if c not in frame.columns]
    if missing:
        if not args.allow_missing:
            raise SystemExit(
                f"{len(missing)} of {len(order)} required features are absent, e.g. {missing[:5]}.\n"
                "The model cannot be applied to a cohort that does not carry the same feature set.\n"
                "Rerun with --allow-missing only if you have decided zero-fill is defensible."
            )
        for c in missing:
            frame[c] = 0.0
        print(f"WARNING: zero-filled {len(missing)} absent features")

    # Reindexing to feature_order.txt IS the ordering contract: the booster is positional and
    # carries no feature names (XGBoost forbids names containing '[' or ']', and the SBS96
    # channels are named like "A[C>T]G").
    x = frame.reindex(columns=order).fillna(0.0).to_numpy(dtype=np.float32)
    booster = xgb.Booster(); booster.load_model(str(HERE / "model.ubj"))
    raw = booster.predict(xgb.DMatrix(x))

    task = manifest["task"]
    out = pd.DataFrame(index=frame.index)
    if task == "multiclass":
        classes = manifest["class_order"]
        for i, name in enumerate(classes):
            out[f"prob__{name}"] = raw[:, i]
        out["predicted_class"] = [classes[i] for i in np.argmax(raw, axis=1)]
    elif task == "binary":
        out["prob_positive"] = raw
        out["predicted_class"] = np.where(raw >= 0.5, manifest["class_order"][1], manifest["class_order"][0])
    else:
        out["prediction"] = raw

    out.to_csv(args.out)
    print(f"wrote {len(out)} predictions -> {args.out}")

if __name__ == "__main__":
    main()
