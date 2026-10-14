#!/usr/bin/env python3
"""Task 3: HRD attribution baselines.

Question: is the HRD gain repair biology or scar-correlated summaries?

Runs three baselines in the SAME five outer folds as the existing HRD analysis
(n = 772), using the archived original fold assignments (nested fold
checkpoints results/tables/main_manuscript_complete_panel_nested_fold_checkpoints/;
fold order reconstructed from the original runner code path and verified
bit-for-bit against the archived sample_folds):

    (a) sig3_only    Signature 3 activity alone (SigProfilerAssignment 1.1.5,
                     COSMIC v3.4 SBS GRCh37, fitted on integer SBS96 counts
                     recovered exactly from features_standard_sbs96.csv.gz)
    (b) brca12_only  BRCA1/2 mutation status alone (pipeline role-matched
                     high-impact/protein-truncating event > 0 in BRCA1 or BRCA2)
    (c) sig3_brca12  Both together

Learner: XGBoost with one fixed, documented configuration per outer fold.
DEVIATION from the main panel's nested hyperparameter search (documented in
task3_notes.md): the nested inner search is degenerate for these 1-2-feature
baselines -- on several outer folds the inner validation split contains no
positive feature values at all (SBS3>0 in 9/772 samples; BRCA1/2-positive in
14/772), so every candidate predicts a constant and the inner Spearman is
undefined ("No valid nested-search candidate"). A fixed XGBoost config is used
instead; the outer folds are the archived originals, verified bit-for-bit.

Endpoints: HRD_Score (Spearman), hrd_binary_42 (AUROC), hrd_binary_33 (AUROC).

Outputs:
    results/tables/task3_hrd_attribution_baselines.csv
    results/tables/task3_hrd_attribution_oof_<baseline>.csv (per baseline)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS = Path.home() / "workspace" / "task3_inputs"
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.nested_oof import outer_splits  # noqa: E402

# Archived original outer-fold seeds (from the checkpoint manifests written by the
# original main-panel run). Verified: outer_splits(task, y, seed=...) in the
# reconstructed sample order reproduces the archived sample_folds exactly.
SEEDS = {
    "HRD_Score": 196932,
    "hrd_binary_33": 476778,
    "hrd_binary_42": 134427,
}
TASKS = {"HRD_Score": "regression", "hrd_binary_33": "binary", "hrd_binary_42": "binary"}
METRICS = {"HRD_Score": "spearman", "hrd_binary_33": "auroc", "hrd_binary_42": "auroc"}
CHECKPOINT_PREFIX = {
    "HRD_Score": "HRD_Score__signatures_plus_MAF_stack__xgboost__main_complete_panel_mc3_signatures_plus_maf_stack_bio_v4_nested_fs_ac11587791bbbd40db3991bb",
    "hrd_binary_33": "hrd_binary_33__signatures_plus_MAF_stack__xgboost__main_complete_panel_mc3_signatures_plus_maf_stack_bio_v4_nested_fs_ac11587791bbbd40db3991bb",
    "hrd_binary_42": "hrd_binary_42__signatures_plus_MAF_stack__xgboost__main_complete_panel_mc3_signatures_plus_maf_stack_bio_v4_nested_fs_ac11587791bbbd40db3991bb",
}
BASELINES = ("sig3_only", "brca12_only", "sig3_brca12")
BASELINE_FEATURES = {
    "sig3_only": ["sbs3_activity"],
    "brca12_only": ["brca12_status"],
    "sig3_brca12": ["sbs3_activity", "brca12_status"],
}

# Fixed XGBoost configuration (documented deviation from the nested search).
XGB_PARAMS = {
    "max_depth": 2,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "subsample": 0.9,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
    "n_jobs": 8,
}


def load_labels(endpoint: str) -> pd.Series:
    cohort = pd.read_csv(INPUTS / "final_analysis_cohort.tsv", sep="\t")
    cohort["patient_id_12"] = cohort["patient_id_12"].astype(str)
    task = TASKS[endpoint]
    if task == "regression":
        data = cohort.dropna(subset=[endpoint]).copy()
        return pd.Series(
            data[endpoint].astype(float).to_numpy(),
            index=data["patient_id_12"].astype(str),
            name=endpoint,
        )
    data = cohort[cohort[endpoint].isin(["HRD-high", "HRD-low"])].copy()
    return pd.Series(
        (data[endpoint] == "HRD-high").astype(int).to_numpy(),
        index=data["patient_id_12"].astype(str),
        name=endpoint,
    )


def load_baseline_features() -> pd.DataFrame:
    """Baseline feature matrix indexed by 12-char patient barcode."""
    sbs3 = pd.read_csv(INPUTS / "sbs3_activity_772.csv")
    sbs3 = sbs3.rename(columns={"sample": "patient"}).set_index("patient")["SBS3"]
    bio = pd.read_csv(
        REPO_ROOT / "results/tables/quick_bio_v4_maf_features.csv.gz",
        usecols=["sample", "driver_gene_role_matched_event_log_count__BRCA1",
                 "driver_gene_role_matched_event_log_count__BRCA2"],
    )
    bio["patient"] = bio["sample"].astype(str).str[:12]
    bio = bio.drop_duplicates("patient").set_index("patient")
    frame = pd.DataFrame(index=sbs3.index)
    frame["sbs3_activity"] = sbs3.astype(float)
    frame["brca12_status"] = (
        (bio["driver_gene_role_matched_event_log_count__BRCA1"] > 0)
        | (bio["driver_gene_role_matched_event_log_count__BRCA2"] > 0)
    ).astype(int).reindex(frame.index).fillna(0).astype(int)
    return frame


def fit_predict_xgb(task: str, x_train: np.ndarray, y_train: np.ndarray,
                    x_test: np.ndarray, seed: int) -> np.ndarray:
    import xgboost as xgb
    params = dict(XGB_PARAMS, random_state=seed)
    if task == "regression":
        model = xgb.XGBRegressor(**params)
        model.fit(x_train, y_train)
        return model.predict(x_test).astype(float)
    model = xgb.XGBClassifier(**params)
    model.fit(x_train, y_train.astype(int))
    return model.predict_proba(x_test)[:, 1].astype(float)


def main() -> int:
    features = load_baseline_features()
    print("baseline features:", features.shape)
    print("  SBS3>0 samples:", int((features["sbs3_activity"] > 0).sum()))
    print("  BRCA1/2-positive samples:", int(features["brca12_status"].sum()))

    tables = REPO_ROOT / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)
    ckpt_dir = tables / "main_manuscript_complete_panel_nested_fold_checkpoints"

    summary_rows: list[dict] = []
    oof_frames: dict[str, list[pd.DataFrame]] = {b: [] for b in BASELINES}

    for endpoint in ("HRD_Score", "hrd_binary_33", "hrd_binary_42"):
        task = TASKS[endpoint]
        metric = METRICS[endpoint]
        labels = load_labels(endpoint)
        # Verified original sample order (cohort TSV order intersected with feature index).
        order = pd.read_csv(INPUTS / f"order_{endpoint}.csv")["patient_id_12"].astype(str)
        labels = labels.loc[order]
        common = labels.index.intersection(features.index.astype(str))
        assert len(common) == len(labels), f"feature coverage gap for {endpoint}"
        labels = labels.loc[common]
        samples = common.astype(str)
        y = labels.to_numpy(dtype=float)
        print(f"\n=== {endpoint} ({task}) n={len(common)} seed={SEEDS[endpoint]} ===", flush=True)

        splits = outer_splits(task, y, groups=None, seed=SEEDS[endpoint], n_splits=5)
        # Bit-for-bit check against the archived original fold assignments.
        archived = np.load(ckpt_dir / f"{CHECKPOINT_PREFIX[endpoint]}.npz", allow_pickle=True)["sample_folds"]
        check = np.zeros(len(samples), dtype=int)
        for fold, _tr, te in splits:
            check[te] = fold
        assert (check == archived).all(), f"fold mismatch vs archive for {endpoint}"
        print("  archived folds reproduced exactly", flush=True)

        for baseline in BASELINES:
            cols = BASELINE_FEATURES[baseline]
            x = features.loc[common, cols].to_numpy(dtype=np.float32)
            oof_pred = np.zeros(len(samples), dtype=float)
            fold_id = np.zeros(len(samples), dtype=int)
            fold_scores: list[tuple[int, float, int]] = []
            for fold, tr, te in splits:
                p = fit_predict_xgb(task, x[tr], y[tr], x[te], SEEDS[endpoint] + fold)
                oof_pred[te] = p
                fold_id[te] = fold
                if task == "regression":
                    s = float(spearmanr(y[te], p).correlation)
                else:
                    s = float(roc_auc_score(y[te].astype(int), p))
                fold_scores.append((fold, s, len(te)))
            if task == "regression":
                pooled = float(spearmanr(y, oof_pred).correlation)
            else:
                pooled = float(roc_auc_score(y.astype(int), oof_pred))
            fold_mean = float(np.mean([s for _, s, _ in fold_scores]))
            print(f"  {baseline:14s} pooled {metric:9s} {pooled:.4f}   fold-mean {fold_mean:.4f}", flush=True)

            for fold, s, n_te in fold_scores:
                summary_rows.append({"baseline": baseline, "endpoint": endpoint, "metric": metric,
                                     "scope": "fold", "fold": fold, "score": s, "n_test": n_te,
                                     "n_features": len(cols), "n_samples": len(samples)})
            summary_rows.append({"baseline": baseline, "endpoint": endpoint, "metric": metric,
                                 "scope": "fold_mean", "fold": "", "score": fold_mean, "n_test": "",
                                 "n_features": len(cols), "n_samples": len(samples)})
            summary_rows.append({"baseline": baseline, "endpoint": endpoint, "metric": metric,
                                 "scope": "pooled", "fold": "", "score": pooled, "n_test": len(samples),
                                 "n_features": len(cols), "n_samples": len(samples)})
            oof_frames[baseline].append(pd.DataFrame({
                "tumour_barcode": samples.to_numpy(),
                "endpoint": endpoint,
                "true": y,
                "predicted_score": oof_pred,
                "fold": fold_id,
            }))

    for baseline in BASELINES:
        oof = pd.concat(oof_frames[baseline], ignore_index=True)
        oof.to_csv(tables / f"task3_hrd_attribution_oof_{baseline}.csv", index=False)
        print("wrote", tables / f"task3_hrd_attribution_oof_{baseline}.csv", oof.shape)

    pd.DataFrame(summary_rows).to_csv(tables / "task3_hrd_attribution_baselines.csv", index=False)
    print("wrote", tables / "task3_hrd_attribution_baselines.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
