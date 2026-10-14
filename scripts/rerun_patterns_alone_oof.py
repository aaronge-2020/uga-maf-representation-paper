#!/usr/bin/env python3
"""Re-run patterns-alone (standard_sbs96_id83) model on archived folds.

Generates OOF predictions for C2/C3/C4b/C5/C6 contrasts (Sig+MAF vs patterns-alone).
Uses the exact fold assignments from Task 1 OOF and per-fold hyperparameters
from main_manuscript_complete_panel_fold_metrics.csv.

Outputs:
- results/tables/patterns_alone_oof_cancer_type.csv
- results/tables/patterns_alone_oof_hrd.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "results" / "tables"
DATA = REPO_ROOT / "data"

# Per-fold XGBoost hyperparameters from main_manuscript_complete_panel_fold_metrics.csv
# (representation=standard_sbs96_id83, learner=xgboost)
CANCER_TYPE_PARAMS = {
    1: {"colsample_bytree": 0.7, "learning_rate": 0.1, "max_depth": 5, "min_child_weight": 5.0, "reg_alpha": 0.5, "reg_lambda": 5.0, "subsample": 0.85},
    2: {"colsample_bytree": 0.7, "learning_rate": 0.08, "max_depth": 4, "min_child_weight": 5.0, "reg_alpha": 0.0, "reg_lambda": 5.0, "subsample": 0.85},
    3: {"colsample_bytree": 0.85, "learning_rate": 0.1, "max_depth": 5, "min_child_weight": 5.0, "reg_alpha": 0.1, "reg_lambda": 0.5, "subsample": 0.85},
    4: {"colsample_bytree": 1.0, "learning_rate": 0.08, "max_depth": 3, "min_child_weight": 10.0, "reg_alpha": 0.05, "reg_lambda": 1.0, "subsample": 1.0},
    5: {"colsample_bytree": 1.0, "learning_rate": 0.1, "max_depth": 4, "min_child_weight": 1.0, "reg_alpha": 0.05, "reg_lambda": 1.0, "subsample": 0.85},
}

# HRD hyperparameters (from fold_metrics.csv)
HRD_SCORE_PARAMS = {
    1: {"colsample_bytree": 0.85, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 3.0, "reg_alpha": 0.0, "reg_lambda": 1.0, "subsample": 0.85},
    2: {"colsample_bytree": 0.7, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 3.0, "reg_alpha": 0.1, "reg_lambda": 1.0, "subsample": 0.85},
    3: {"colsample_bytree": 0.85, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 5.0, "reg_alpha": 0.0, "reg_lambda": 5.0, "subsample": 0.85},
    4: {"colsample_bytree": 1.0, "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 5.0, "reg_alpha": 0.05, "reg_lambda": 1.0, "subsample": 0.85},
    5: {"colsample_bytree": 0.85, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 1.0, "reg_alpha": 0.1, "reg_lambda": 0.5, "subsample": 0.85},
}

HRD33_PARAMS = {
    1: {"colsample_bytree": 0.85, "learning_rate": 0.08, "max_depth": 5, "min_child_weight": 5.0, "reg_alpha": 0.0, "reg_lambda": 1.0, "subsample": 0.85},
    2: {"colsample_bytree": 0.7, "learning_rate": 0.1, "max_depth": 4, "min_child_weight": 3.0, "reg_alpha": 0.1, "reg_lambda": 0.5, "subsample": 0.85},
    3: {"colsample_bytree": 0.85, "learning_rate": 0.08, "max_depth": 4, "min_child_weight": 5.0, "reg_alpha": 0.05, "reg_lambda": 1.0, "subsample": 0.85},
    4: {"colsample_bytree": 1.0, "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 10.0, "reg_alpha": 0.0, "reg_lambda": 5.0, "subsample": 1.0},
    5: {"colsample_bytree": 0.85, "learning_rate": 0.08, "max_depth": 5, "min_child_weight": 1.0, "reg_alpha": 0.1, "reg_lambda": 0.5, "subsample": 0.85},
}

HRD42_PARAMS = {
    1: {"colsample_bytree": 0.85, "learning_rate": 0.08, "max_depth": 4, "min_child_weight": 1.0, "reg_alpha": 0.05, "reg_lambda": 1.0, "subsample": 0.85},
    2: {"colsample_bytree": 0.7, "learning_rate": 0.1, "max_depth": 5, "min_child_weight": 3.0, "reg_alpha": 0.0, "reg_lambda": 0.5, "subsample": 0.85},
    3: {"colsample_bytree": 0.85, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 5.0, "reg_alpha": 0.1, "reg_lambda": 1.0, "subsample": 0.85},
    4: {"colsample_bytree": 1.0, "learning_rate": 0.08, "max_depth": 3, "min_child_weight": 5.0, "reg_alpha": 0.05, "reg_lambda": 5.0, "subsample": 1.0},
    5: {"colsample_bytree": 0.85, "learning_rate": 0.1, "max_depth": 4, "min_child_weight": 1.0, "reg_alpha": 0.0, "reg_lambda": 1.0, "subsample": 0.85},
}


def main():
    try:
        import xgboost as xgb
    except ImportError:
        print("ERROR: xgboost not installed", file=sys.stderr)
        sys.exit(1)

    print("Loading patterns-alone features...")
    feat = pd.read_csv(DATA / "task0_features" / "features_standard_sbs96_id83.csv.gz")
    feat = feat.set_index(feat.columns[0])
    print(f"  Features: {feat.shape}")

    print("Loading fold assignments from Task 1 OOF...")
    oof_task1 = pd.read_csv(TABLES / "task1_oof_sigmaf_a_full.csv", usecols=["sample", "fold", "true_label"])
    fold_map = dict(zip(oof_task1["sample"], oof_task1["fold"]))
    print(f"  Fold assignments: {len(fold_map)} samples")

    # Align features to fold map
    feat = feat.loc[feat.index.intersection(fold_map.keys())]
    print(f"  Aligned features: {feat.shape}")

    # Cancer typing: multiclass
    print("\n=== Cancer typing (patterns-alone XGBoost) ===")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(oof_task1.set_index("sample").loc[feat.index, "true_label"])
    classes = le.classes_

    oof_proba = np.zeros((len(feat), len(classes)))
    for fold in range(1, 6):
        train_idx = [i for i, s in enumerate(feat.index) if fold_map[s] != fold]
        test_idx = [i for i, s in enumerate(feat.index) if fold_map[s] == fold]
        X_train, y_train = feat.iloc[train_idx].values, y[train_idx]
        X_test = feat.iloc[test_idx].values
        params = CANCER_TYPE_PARAMS[fold]
        model = xgb.XGBClassifier(
            n_estimators=160, tree_method="hist", random_state=42,
            **params
        )
        model.fit(X_train, y_train)
        oof_proba[test_idx] = model.predict_proba(X_test)
        print(f"  Fold {fold}: trained on {len(train_idx)}, predicted {len(test_idx)}")

    # Save
    out = pd.DataFrame(oof_proba, index=feat.index, columns=[f"proba_{c}" for c in classes])
    out["fold"] = [fold_map[s] for s in feat.index]
    out["true_label"] = oof_task1.set_index("sample").loc[feat.index, "true_label"].values
    out.to_csv(TABLES / "patterns_alone_oof_cancer_type.csv", index_label="sample")
    print(f"  Saved: {TABLES / 'patterns_alone_oof_cancer_type.csv'}")

    # HRD endpoints
    print("\n=== HRD endpoints (patterns-alone XGBoost) ===")
    for endpoint, label_file, params_dict, is_binary in [
        ("HRD_Score", "order_HRD_Score.csv", HRD_SCORE_PARAMS, False),
        ("hrd_binary_33", "order_hrd_binary_33.csv", HRD33_PARAMS, True),
        ("hrd_binary_42", "order_hrd_binary_42.csv", HRD42_PARAMS, True),
    ]:
        print(f"\n  {endpoint}...")
        labels = pd.read_csv(DATA / "task3" / label_file, index_col=0)
        # Align
        common = feat.index.intersection(labels.index)
        X = feat.loc[common]
        y_hrd = labels.loc[common].iloc[:, 0].values
        # Fold assignments for HRD (use cancer typing folds as proxy; HRD uses same splits)
        folds_hrd = [fold_map[s] for s in common]

        oof_pred = np.zeros(len(common))
        for fold in range(1, 6):
            train_idx = [i for i in range(len(common)) if folds_hrd[i] != fold]
            test_idx = [i for i in range(len(common)) if folds_hrd[i] == fold]
            X_train, y_train = X.iloc[train_idx].values, y_hrd[train_idx]
            X_test = X.iloc[test_idx].values
            params = params_dict[fold]
            if is_binary:
                model = xgb.XGBClassifier(n_estimators=160, tree_method="hist", random_state=42, **params)
                model.fit(X_train, y_train)
                oof_pred[test_idx] = model.predict_proba(X_test)[:, 1]
            else:
                model = xgb.XGBRegressor(n_estimators=160, tree_method="hist", random_state=42, **params)
                model.fit(X_train, y_train)
                oof_pred[test_idx] = model.predict(X_test)
            print(f"    Fold {fold}: {len(train_idx)} train, {len(test_idx)} test")

        out_hrd = pd.DataFrame({
            "sample": common,
            "fold": folds_hrd,
            "true_label": y_hrd,
            "pred": oof_pred,
        })
        out_hrd.to_csv(TABLES / f"patterns_alone_oof_{endpoint}.csv", index=False)
        print(f"    Saved: {TABLES / f'patterns_alone_oof_{endpoint}.csv'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
