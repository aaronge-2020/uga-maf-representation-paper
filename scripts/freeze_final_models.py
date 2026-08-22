#!/usr/bin/env python3
"""Fit and freeze final models on the full TCGA cohort, for inference on external data.

The benchmark pipeline runs nested cross-validation: for each endpoint it fits five fold-models,
uses each to predict its held-out fold, and discards them. That produces pooled out-of-fold
metrics, which is what the manuscript reports -- but it leaves no fitted model on disk and no
model trained on all of TCGA. External validation (POG570, PCAWG) needs exactly that.

This script produces, per endpoint and representation:

    results/frozen_models/<endpoint>__<representation>/
        model.ubj            XGBoost booster, native format (portable across versions)
        manifest.json        feature order, class order, hyperparameters, provenance, checksums
        feature_order.txt    one feature name per line, in the order the model expects
        predict.py           standalone inference script, no repo imports required

    python scripts/freeze_final_models.py --endpoint cancer_type_top20 --representation standard_sbs96_id83
    python scripts/freeze_final_models.py --endpoint HRD_Score --representation signatures_plus_MAF_stack
    python scripts/freeze_final_models.py --all

Hyperparameter selection
------------------------
Nested CV selects a different hyperparameter set per fold, so there is no single "the" set to
export. This script instead runs k-fold cross-validation over a candidate grid on the *full*
cohort and takes the best-scoring setting, which is the standard way to arrive at a deployable
model. The selected setting is recorded in the manifest.

Important caveat for the manuscript: a model fitted on all of TCGA has never been evaluated on
held-out TCGA data. Every number in the paper comes from out-of-fold predictions of the fold
models. External results from a frozen model are a genuine out-of-sample test, but the frozen
model itself is a new artefact and should be described as such in Methods.

Preprocessing
-------------
XGBoost is fitted on raw, unscaled features in this pipeline (`_xgb_fit_predict` receives the
unscaled matrices; the StandardScaler is applied only to the linear and Cox learners). There is
therefore no scaler to export -- the only preprocessing that matters is column order and the
NaN-to-zero fill, both of which are recorded and reproduced by the generated predict.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.config import load_yaml, resolve_paths_map  # noqa: E402

SEED = 20260821
OUT_ROOT = REPO_ROOT / "results/frozen_models"

BIO_MAF_CACHE = REPO_ROOT / "results/tables/quick_bio_v4_maf_features.csv.gz"
SIGNATURE_REL = "datasets/tcga_mc3/features/features_standard_sbs96_id83.csv.gz"
COHORT_REL = "datasets/tcga_brca_hrd/cohort/final_analysis_cohort.tsv"

REPRESENTATIONS = ("standard_sbs96_id83", "MAF_stack_only", "signatures_plus_MAF_stack")

HRD_CONTINUOUS = {"HRD_Score", "HRD_TAI", "HRD_LST", "HRD_LOH", "PARPi7", "eCARD"}
HRD_BINARY = {"hrd_binary_24", "hrd_binary_33", "hrd_binary_42"}

# Modest grid: enough to pick a sensible operating point without a multi-hour search.
PARAM_GRID = [
    {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "reg_lambda": 1.0, "n_estimators": 400},
    {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "reg_lambda": 1.0, "n_estimators": 400},
    {"max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0, "n_estimators": 300},
    {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.5, "reg_lambda": 5.0, "n_estimators": 400},
]

DEFAULT_PAIRS = [
    ("cancer_type_top20", "standard_sbs96_id83"),
    ("cancer_type_top20", "signatures_plus_MAF_stack"),
    ("HRD_Score", "signatures_plus_MAF_stack"),
    ("hrd_binary_33", "signatures_plus_MAF_stack"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_signatures(datasets_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(datasets_dir / SIGNATURE_REL)
    index_col = frame.columns[0]
    frame[index_col] = frame[index_col].astype(str).str[:12]
    frame = frame.drop_duplicates(index_col).set_index(index_col)
    return frame.select_dtypes(include=[np.number]).astype(np.float32)


def load_bio_maf() -> pd.DataFrame:
    frame = pd.read_csv(BIO_MAF_CACHE)
    index_col = "sample" if "sample" in frame.columns else frame.columns[0]
    frame[index_col] = frame[index_col].astype(str).str[:12]
    frame = frame.drop_duplicates(index_col).set_index(index_col)
    return frame.select_dtypes(include=[np.number]).astype(np.float32)


def build_features(representation: str, datasets_dir: Path) -> pd.DataFrame:
    if representation == "standard_sbs96_id83":
        return load_signatures(datasets_dir)
    if representation == "MAF_stack_only":
        return load_bio_maf()
    if representation == "signatures_plus_MAF_stack":
        sig, bio = load_signatures(datasets_dir), load_bio_maf()
        sig = sig.rename(columns={c: f"sig__{c}" for c in sig.columns if c in set(bio.columns)})
        common = sig.index.intersection(bio.index)
        return pd.concat([sig.loc[common], bio.loc[common]], axis=1)
    raise SystemExit(f"unsupported representation: {representation}")


def build_labels(endpoint: str, datasets_dir: Path) -> tuple[pd.Series, str, list[str]]:
    """Return (labels, task, class_order). class_order is empty for regression."""
    if endpoint == "cancer_type_top20":
        from utils.endpoint_registry import CANCER_TYPE_ENDPOINT_CLASSES, build_cdr_cancer_type_labels

        classes = CANCER_TYPE_ENDPOINT_CLASSES.get("cancer_type_top20")
        mc3_dir = datasets_dir / "datasets/tcga_mc3"
        labels = build_cdr_cancer_type_labels(mc3_dir, classes=classes, endpoint_name=endpoint).astype(str)
        # XGBoost predicts columns 0..K-1 in the order of the sorted unique labels. Recording that
        # order is the only way a downstream user can map a probability column back to a tumour type.
        class_order = sorted(labels.unique())
        encoded = pd.Series(pd.Categorical(labels, categories=class_order).codes, index=labels.index, name=endpoint)
        return encoded, "multiclass", class_order

    cohort = pd.read_csv(datasets_dir / COHORT_REL, sep="\t")
    if endpoint in HRD_CONTINUOUS:
        data = cohort.dropna(subset=[endpoint]).copy()
        labels = pd.Series(data[endpoint].astype(float).to_numpy(), index=data["patient_id_12"].astype(str), name=endpoint)
        return labels, "regression", []
    if endpoint in HRD_BINARY:
        data = cohort[cohort[endpoint].isin(["HRD-high", "HRD-low"])].copy()
        labels = pd.Series((data[endpoint] == "HRD-high").astype(int).to_numpy(),
                           index=data["patient_id_12"].astype(str), name=endpoint)
        return labels, "binary", ["HRD-low", "HRD-high"]
    raise SystemExit(f"unsupported endpoint: {endpoint}")


def make_model(task: str, n_classes: int, params: dict, seed: int):
    import xgboost as xgb

    common = dict(random_state=int(seed), n_jobs=0, tree_method="hist", **params)
    if task == "multiclass":
        return xgb.XGBClassifier(objective="multi:softprob", num_class=int(n_classes), eval_metric="mlogloss", **common)
    if task == "binary":
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    return xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)


# XGBoost refuses feature names containing '[', ']' or '<', and the SBS96 channels are literally
# named like "A[C>T]G". The benchmark pipeline never hits this because it passes bare numpy
# arrays. We do the same: the model is positional, and feature_order.txt is the binding contract
# for which column goes where.
def as_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.to_numpy(dtype=np.float32)


def score(task: str, y_true: np.ndarray, model, x: pd.DataFrame) -> float:
    xm = as_matrix(x)
    if task == "multiclass":
        from sklearn.metrics import balanced_accuracy_score

        return float(balanced_accuracy_score(y_true, model.predict(xm)))
    if task == "binary":
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, model.predict_proba(xm)[:, 1]))
    from scipy.stats import spearmanr

    return float(spearmanr(y_true, model.predict(xm)).correlation)


PREDICT_TEMPLATE = '''#!/usr/bin/env python3
"""Inference for the frozen {endpoint} / {representation} model.

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

    sep = "\\t" if str(args.features).endswith((".tsv", ".txt", ".tsv.gz")) else ","
    frame = pd.read_csv(args.features, sep=sep)
    index_col = args.index_col or frame.columns[0]
    frame = frame.set_index(index_col)

    missing = [c for c in order if c not in frame.columns]
    if missing:
        if not args.allow_missing:
            raise SystemExit(
                f"{{len(missing)}} of {{len(order)}} required features are absent, e.g. {{missing[:5]}}.\\n"
                "The model cannot be applied to a cohort that does not carry the same feature set.\\n"
                "Rerun with --allow-missing only if you have decided zero-fill is defensible."
            )
        for c in missing:
            frame[c] = 0.0
        print(f"WARNING: zero-filled {{len(missing)}} absent features")

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
            out[f"prob__{{name}}"] = raw[:, i]
        out["predicted_class"] = [classes[i] for i in np.argmax(raw, axis=1)]
    elif task == "binary":
        out["prob_positive"] = raw
        out["predicted_class"] = np.where(raw >= 0.5, manifest["class_order"][1], manifest["class_order"][0])
    else:
        out["prediction"] = raw

    out.to_csv(args.out)
    print(f"wrote {{len(out)}} predictions -> {{args.out}}")

if __name__ == "__main__":
    main()
'''


def freeze(endpoint: str, representation: str, datasets_dir: Path, folds: int) -> Path:
    from sklearn.model_selection import KFold, StratifiedKFold

    print(f"\n=== {endpoint} / {representation} ===", flush=True)
    features = build_features(representation, datasets_dir)
    labels, task, class_order = build_labels(endpoint, datasets_dir)
    common = labels.index.intersection(features.index)
    labels, x = labels.loc[common], features.loc[common].fillna(0.0)
    feature_order = list(x.columns)
    n_classes = len(class_order) if task == "multiclass" else 2
    print(f"  n={len(common)}  features={len(feature_order)}  task={task}", flush=True)

    if task in {"multiclass", "binary"}:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
        splits = list(splitter.split(x, labels))
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=SEED)
        splits = list(splitter.split(x))

    print(f"  selecting hyperparameters over {len(PARAM_GRID)} settings, {folds}-fold CV", flush=True)
    best_params, best_score, grid_rows = None, -np.inf, []
    for i, params in enumerate(PARAM_GRID, start=1):
        scores = []
        for tr, va in splits:
            model = make_model(task, n_classes, params, SEED)
            model.fit(as_matrix(x.iloc[tr]), labels.iloc[tr], verbose=False)
            scores.append(score(task, labels.iloc[va].to_numpy(), model, x.iloc[va]))
        mean = float(np.mean(scores))
        grid_rows.append({"setting": i, **params, "cv_score_mean": mean, "cv_score_sd": float(np.std(scores, ddof=1))})
        print(f"    setting {i}: depth={params['max_depth']} lr={params['learning_rate']} -> {mean:.4f}", flush=True)
        if mean > best_score:
            best_params, best_score = dict(params), mean

    print(f"  refitting on all {len(common)} samples with the winning setting", flush=True)
    final = make_model(task, n_classes, best_params, SEED)
    final.fit(as_matrix(x), labels, verbose=False)

    out_dir = OUT_ROOT / f"{endpoint}__{representation}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.ubj"
    # Saved without feature names, for the reason noted above. Ordering is enforced by
    # feature_order.txt and by predict.py reindexing against it.
    final.get_booster().save_model(str(model_path))
    (out_dir / "feature_order.txt").write_text("\n".join(feature_order) + "\n", encoding="utf-8")
    (out_dir / "predict.py").write_text(
        PREDICT_TEMPLATE.format(endpoint=endpoint, representation=representation), encoding="utf-8"
    )
    pd.DataFrame(grid_rows).to_csv(out_dir / "hyperparameter_search.csv", index=False)

    manifest = {
        "endpoint": endpoint,
        "representation": representation,
        "task": task,
        "class_order": class_order,
        "n_features": len(feature_order),
        "n_training_samples": int(len(common)),
        "hyperparameters": best_params,
        "selection": {
            "method": f"{folds}-fold cross-validation over {len(PARAM_GRID)} candidate settings, fitted on the full cohort",
            "metric": {"multiclass": "balanced_accuracy", "binary": "auroc", "regression": "spearman"}[task],
            "cv_score": best_score,
        },
        "preprocessing": {
            "scaler": None,
            "note": "XGBoost is fitted on raw features in this pipeline; only the linear and Cox learners are standardised. Apply column reordering and NaN->0 fill only.",
            "nan_fill": 0.0,
        },
        "provenance": {
            "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "seed": SEED,
            "model_sha256": sha256_file(model_path),
            "xgboost_version": __import__("xgboost").__version__,
            "training_cohort": "TCGA (all samples with this endpoint and representation)",
        },
        "caveat": (
            "This model is fitted on the entire TCGA cohort and has NOT been evaluated on held-out "
            "TCGA data. All manuscript metrics come from nested out-of-fold predictions of separate "
            "fold models. Results from this frozen model on an external cohort are a genuine "
            "out-of-sample test, but the model itself is a new artefact and should be described as "
            "such in Methods."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  CV {manifest['selection']['metric']} = {best_score:.4f}", flush=True)
    print(f"  -> {out_dir.relative_to(REPO_ROOT)}", flush=True)
    return out_dir


def write_readme(dirs: list[Path]) -> None:
    lines = [
        "# Frozen models for external validation",
        "",
        "Each subdirectory holds one model fitted on the **entire TCGA cohort**, for inference on",
        "external data (POG570, PCAWG). Generated by `scripts/freeze_final_models.py`.",
        "",
        "## Why these did not exist before",
        "",
        "The benchmark runs nested cross-validation: five fold-models per endpoint, each predicting",
        "its held-out fold, then discarded. That yields the pooled out-of-fold metrics the manuscript",
        "reports, but no model trained on all of TCGA and nothing serialised to disk.",
        "",
        "## Contents",
        "",
        "| File | Purpose |",
        "|---|---|",
        "| `model.ubj` | XGBoost booster, native format |",
        "| `feature_order.txt` | Exact feature names, in the order the model expects |",
        "| `manifest.json` | Task, class order, hyperparameters, provenance, SHA-256 |",
        "| `hyperparameter_search.csv` | The CV grid and scores behind the selected setting |",
        "| `predict.py` | Standalone inference, no repo imports needed |",
        "",
        "## Running inference",
        "",
        "```bash",
        "cd <model directory>",
        "python predict.py --features external_cohort_features.csv --out predictions.csv",
        "```",
        "",
        "`predict.py` **fails** if any required feature is absent, rather than zero-filling. A",
        "silently zeroed feature block is indistinguishable from a genuine zero measurement and would",
        "corrupt predictions. Override with `--allow-missing` only deliberately.",
        "",
        "## Feature parity is the real constraint",
        "",
        "A frozen model is only usable if the external cohort can produce the same features:",
        "",
        "- **`standard_sbs96_id83`** — SBS96 and ID83 counts are computable from any MAF plus a",
        "  reference genome. Realistic for POG570 and PCAWG.",
        "- **`signatures_plus_MAF_stack`** — additionally requires all 948 Bio MAF v4 columns, which",
        "  depend on VEP consequence and impact, SIFT, PolyPhen, OncoKB role, hotspot annotation and",
        "  VAF. If the external cohort is not annotated identically these cannot be reconstructed,",
        "  and no amount of model export will help.",
        "",
        "## Caveat for Methods",
        "",
        "These models are fitted on all of TCGA and have never been evaluated on held-out TCGA data.",
        "Every manuscript metric comes from out-of-fold predictions of the fold models. External",
        "results from a frozen model are a genuine out-of-sample test, but the model is a new artefact",
        "and should be described as such.",
        "",
        "## Available models",
        "",
    ]
    for path in sorted(dirs):
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        lines.append(
            f"- `{path.name}` — {manifest['task']}, {manifest['n_features']} features, "
            f"n={manifest['n_training_samples']}, CV {manifest['selection']['metric']} "
            f"{manifest['selection']['cv_score']:.4f}"
        )
    (OUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint")
    parser.add_argument("--representation", choices=REPRESENTATIONS)
    parser.add_argument("--all", action="store_true", help="freeze the default set of endpoint/representation pairs")
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    paths = resolve_paths_map(load_yaml(args.paths))
    datasets_dir = Path(paths["raw_data"]["mc3_source_dir"]).parents[1]
    if not (datasets_dir / SIGNATURE_REL).exists():
        raise SystemExit(f"signature features not found: {datasets_dir / SIGNATURE_REL}")

    if args.all:
        pairs = DEFAULT_PAIRS
    elif args.endpoint and args.representation:
        pairs = [(args.endpoint, args.representation)]
    else:
        raise SystemExit("give --endpoint and --representation, or --all")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    made = [freeze(endpoint, representation, datasets_dir, args.folds) for endpoint, representation in pairs]
    write_readme(made)

    print("\n" + "=" * 74)
    print(f"{len(made)} model(s) frozen -> results/frozen_models/")
    for path in made:
        print(f"  {path.name}")
    print("\nSee results/frozen_models/README.md for inference instructions and the")
    print("feature-parity constraint that governs which of these can run on external data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
