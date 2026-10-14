#!/usr/bin/env python3
"""Re-run patterns-alone HRD endpoints on archived folds."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "results" / "tables"
DATA = REPO_ROOT / "data"

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
    import xgboost as xgb
    
    print("Loading features...")
    feat = pd.read_csv(DATA / "task0_features" / "features_standard_sbs96_id83.csv.gz")
    feat = feat.set_index(feat.columns[0])
    
    print("Loading fold assignments...")
    oof_task1 = pd.read_csv(TABLES / "task1_oof_sigmaf_a_full.csv", usecols=["sample", "fold"])
    fold_map = dict(zip(oof_task1["sample"], oof_task1["fold"]))
    
    print("Loading HRD labels...")
    cohort = pd.read_csv(DATA / "task3" / "final_analysis_cohort.tsv", sep="\t", index_col="patient_id_12")
    
    for endpoint, label_col, params_dict, is_binary in [
        ("HRD_Score", "HRD_Score", HRD_SCORE_PARAMS, False),
        ("hrd_binary_33", "hrd_binary_33", HRD33_PARAMS, True),
        ("hrd_binary_42", "hrd_binary_42", HRD42_PARAMS, True),
    ]:
        print(f"\n{endpoint}...")
        # Get labels
        if is_binary:
            y_all = (cohort[label_col] == "HRD-high").astype(int)
        else:
            y_all = cohort[label_col]
        
        common = feat.index.intersection(y_all.index).intersection(fold_map.keys())
        X = feat.loc[common]
        y = y_all.loc[common].values
        folds = [fold_map[s] for s in common]
        
        print(f"  Samples: {len(common)}")
        oof_pred = np.zeros(len(common))
        for fold in range(1, 6):
            train_idx = [i for i in range(len(common)) if folds[i] != fold]
            test_idx = [i for i in range(len(common)) if folds[i] == fold]
            X_train, y_train = X.iloc[train_idx].values, y[train_idx]
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
            print(f"  Fold {fold}: done")
        
        out = pd.DataFrame({"sample": common, "fold": folds, "true_label": y, "pred": oof_pred})
        out.to_csv(TABLES / f"patterns_alone_oof_{endpoint}.csv", index=False)
        print(f"  Saved")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
