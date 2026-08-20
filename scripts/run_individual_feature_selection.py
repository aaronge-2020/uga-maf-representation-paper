#!/usr/bin/env python3
"""Individual-feature selection inside Bio MAF v4, with SHAP attribution.

Two requests from the 1 Aug 2026 meeting are handled together here because they share the same
fitted models.

Jeya, on why block-level selection is not enough:

    "Filtering at the individual feature level is necessary. Given the ratio of 943 features to
     the sample size in TCGA, there is a high risk of overfitting on noise."

His protocol, as stated in the meeting:

    1. Split the data 80% training / 20% held-out test.
    2. Within the 80%, run k-fold internal cross-validation to tune hyperparameters that control
       sparsity.
    3. Pick the most parsimonious setting that still gives comparable performance.
    4. Refit on the entire 80% with those winning hyperparameters.
    5. Evaluate once on the held-out 20%.

He explicitly rejected PCA for this: it compresses variance but does not remove features whose
apparent signal is noise, and removing those is the entire point.

Vijay's addition (same meeting): SHAP values over the selected model, so the paper can say which
features carry the signal rather than only how many survived.

    python scripts/run_individual_feature_selection.py --endpoint cancer_type_top20
    python scripts/run_individual_feature_selection.py --endpoint OS --no-shap
    python scripts/run_individual_feature_selection.py --endpoint hrd_binary_33 --inner-folds 5

Outputs, per endpoint, under results/tables/:
    feature_selection_<endpoint>_grid.csv            every candidate setting and its inner-CV score
    feature_selection_<endpoint>_selected.csv        surviving features, ranked by gain and SHAP
    feature_selection_<endpoint>_summary.json        winner, held-out score, feature counts
    feature_selection_<endpoint>_shap_summary.png    beeswarm plot, when SHAP is enabled
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.config import load_yaml, resolve_paths_map  # noqa: E402
from utils.endpoint_registry import (  # noqa: E402
    build_cdr_cancer_type_labels,
    build_cdr_survival_labels,
)

BIO_MAF_CACHE = REPO_ROOT / "results/tables/quick_bio_v4_maf_features.csv.gz"
SEED = 20260801

# Sparsity grid. Each setting trades feature count against fit. reg_alpha is the L1 penalty, which
# is what actually drives XGBoost to drop features entirely; colsample_bytree and min_child_weight
# restrict how many features can enter a tree at all.
SPARSITY_GRID = [
    {"reg_alpha": 0.0, "reg_lambda": 1.0, "colsample_bytree": 0.8, "min_child_weight": 1, "max_depth": 6},
    {"reg_alpha": 0.1, "reg_lambda": 1.0, "colsample_bytree": 0.6, "min_child_weight": 3, "max_depth": 5},
    {"reg_alpha": 1.0, "reg_lambda": 2.0, "colsample_bytree": 0.5, "min_child_weight": 5, "max_depth": 4},
    {"reg_alpha": 5.0, "reg_lambda": 5.0, "colsample_bytree": 0.4, "min_child_weight": 10, "max_depth": 4},
    {"reg_alpha": 10.0, "reg_lambda": 10.0, "colsample_bytree": 0.3, "min_child_weight": 20, "max_depth": 3},
    {"reg_alpha": 25.0, "reg_lambda": 20.0, "colsample_bytree": 0.3, "min_child_weight": 30, "max_depth": 3},
]


def load_bio_maf() -> pd.DataFrame:
    if not BIO_MAF_CACHE.exists():
        raise SystemExit(f"Bio MAF v4 feature table not found: {BIO_MAF_CACHE}")
    frame = pd.read_csv(BIO_MAF_CACHE)
    index_col = "sample" if "sample" in frame.columns else frame.columns[0]
    frame[index_col] = frame[index_col].astype(str).str[:12]
    frame = frame.drop_duplicates(index_col).set_index(index_col)
    return frame.select_dtypes(include=[np.number]).astype(np.float32)


def load_endpoint(endpoint: str, mc3_dir: Path, features: pd.DataFrame):
    """Return (y, task, n_classes) aligned to a subset of ``features``."""

    if endpoint == "cancer_type_top20":
        from utils.endpoint_registry import CANCER_TYPE_ENDPOINT_CLASSES

        classes = CANCER_TYPE_ENDPOINT_CLASSES.get("cancer_type_top20")
        labels = build_cdr_cancer_type_labels(mc3_dir, classes=classes, endpoint_name=endpoint)
        common = features.index.intersection(labels.index)
        labels = labels.loc[common].astype(str)
        codes, uniques = pd.factorize(labels, sort=True)
        return pd.Series(codes, index=labels.index), "multiclass", len(uniques)

    if endpoint in {"OS", "PFI", "DSS", "DFI"}:
        survival, _ = build_cdr_survival_labels(mc3_dir, [endpoint])
        if endpoint not in survival:
            raise SystemExit(f"survival endpoint {endpoint} could not be built")
        frame = survival[endpoint]
        common = features.index.intersection(frame.index)
        frame = frame.loc[common]
        # XGBoost's Cox objective takes a signed label: positive time for events, negative for
        # censored observations.
        y = frame["time"].where(frame["event"] == 1, -frame["time"])
        return y.astype(float), "survival", 1

    raise SystemExit(
        f"endpoint '{endpoint}' is not wired into this script. "
        "Supported: cancer_type_top20, OS, PFI, DSS, DFI."
    )


def make_model(task: str, n_classes: int, params: dict, n_estimators: int, seed: int):
    import xgboost as xgb

    common = dict(
        n_estimators=int(n_estimators),
        learning_rate=0.05,
        subsample=0.8,
        random_state=int(seed),
        n_jobs=0,
        tree_method="hist",
        **params,
    )
    if task == "multiclass":
        return xgb.XGBClassifier(objective="multi:softprob", num_class=int(n_classes), eval_metric="mlogloss", **common)
    if task == "survival":
        return xgb.XGBRegressor(objective="survival:cox", eval_metric="cox-nloglik", **common)
    return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)


def score_predictions(task: str, y_true: np.ndarray, model, x: np.ndarray) -> float:
    if task == "multiclass":
        from sklearn.metrics import balanced_accuracy_score

        return float(balanced_accuracy_score(y_true, model.predict(x)))
    if task == "survival":
        from sksurv.metrics import concordance_index_censored

        risk = np.asarray(model.predict(x), dtype=float)
        event = (np.asarray(y_true, dtype=float) > 0)
        time = np.abs(np.asarray(y_true, dtype=float))
        return float(concordance_index_censored(event, time, risk)[0])
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, model.predict_proba(x)[:, 1]))


def n_selected(model) -> int:
    """Features XGBoost actually used, i.e. those with non-zero total gain."""
    booster = model.get_booster()
    return int(sum(1 for value in booster.get_score(importance_type="total_gain").values() if value > 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default="cancer_type_top20")
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="Accept the sparsest setting whose inner-CV score is within this much "
                             "of the best. This operationalises Jeya's 'comparable performance'.")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--shap-max-samples", type=int, default=2000)
    args = parser.parse_args()

    from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

    paths = resolve_paths_map(load_yaml(args.paths))
    mc3_dir = Path(paths["raw_data"]["mc3_source_dir"])
    if not mc3_dir.exists():
        raise SystemExit(f"mc3_source_dir does not exist: {mc3_dir}")

    features = load_bio_maf()
    y, task, n_classes = load_endpoint(args.endpoint, mc3_dir, features)
    x = features.loc[y.index]
    feature_names = list(x.columns)
    print(f"endpoint={args.endpoint} task={task} n={len(y)} features={len(feature_names)}", flush=True)

    stratify = y if task == "multiclass" else None
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=args.test_size, random_state=SEED, stratify=stratify
    )
    x_train, y_train = x.iloc[idx_train], y.iloc[idx_train]
    x_test, y_test = x.iloc[idx_test], y.iloc[idx_test]
    print(f"split: train={len(idx_train)} heldout={len(idx_test)}", flush=True)

    if task == "multiclass":
        splitter = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=SEED)
        inner = list(splitter.split(x_train, y_train))
    else:
        splitter = KFold(n_splits=args.inner_folds, shuffle=True, random_state=SEED)
        inner = list(splitter.split(x_train))

    print(f"\ninner {args.inner_folds}-fold CV over {len(SPARSITY_GRID)} sparsity settings", flush=True)
    grid_rows = []
    for i, params in enumerate(SPARSITY_GRID, start=1):
        scores, counts = [], []
        for tr, va in inner:
            model = make_model(task, n_classes, params, args.n_estimators, SEED)
            model.fit(x_train.iloc[tr], y_train.iloc[tr], verbose=False)
            scores.append(score_predictions(task, y_train.iloc[va].to_numpy(), model, x_train.iloc[va]))
            counts.append(n_selected(model))
        row = {
            "setting": i,
            **params,
            "inner_score_mean": float(np.mean(scores)),
            "inner_score_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "n_features_mean": float(np.mean(counts)),
        }
        grid_rows.append(row)
        print(
            f"  setting {i}: alpha={params['reg_alpha']:<5g} colsample={params['colsample_bytree']} "
            f"score={row['inner_score_mean']:.4f} +/- {row['inner_score_sd']:.4f} features={row['n_features_mean']:.0f}",
            flush=True,
        )

    grid = pd.DataFrame(grid_rows)
    best = grid["inner_score_mean"].max()
    # "The most parsimonious setting that gives comparable performance": among settings within
    # tolerance of the best score, take the one using the fewest features.
    eligible = grid[grid["inner_score_mean"] >= best - args.tolerance]
    winner = eligible.sort_values("n_features_mean").iloc[0]
    winner_params = {key: winner[key] for key in SPARSITY_GRID[0]}
    winner_params["min_child_weight"] = int(winner_params["min_child_weight"])
    winner_params["max_depth"] = int(winner_params["max_depth"])
    print(
        f"\nwinner: setting {int(winner['setting'])} "
        f"(score {winner['inner_score_mean']:.4f} vs best {best:.4f}, "
        f"{winner['n_features_mean']:.0f} features)",
        flush=True,
    )

    print("refitting on the full training split", flush=True)
    final = make_model(task, n_classes, winner_params, args.n_estimators, SEED)
    final.fit(x_train, y_train, verbose=False)
    heldout = score_predictions(task, y_test.to_numpy(), final, x_test)
    gains = final.get_booster().get_score(importance_type="total_gain")
    selected = {name: float(value) for name, value in gains.items() if value > 0}
    print(f"held-out score = {heldout:.4f}   features used = {len(selected)} / {len(feature_names)}", flush=True)

    ranked = pd.DataFrame(
        {"feature": list(selected), "total_gain": list(selected.values())}
    ).sort_values("total_gain", ascending=False).reset_index(drop=True)
    ranked["gain_rank"] = ranked.index + 1

    shap_path = None
    if not args.no_shap:
        try:
            import xgboost as xgb

            n = min(len(x_test), args.shap_max_samples)
            sample = x_test.iloc[:n]
            print(f"computing SHAP values on {n} held-out samples", flush=True)

            # XGBoost's own exact TreeSHAP, via pred_contribs. The `shap` library's TreeExplainer
            # parses the model's JSON dump, which fails on current XGBoost versions with
            # "could not convert string to float" on the leaf-vector field. Going through the
            # booster avoids that parser entirely and gives identical values.
            contribs = final.get_booster().predict(xgb.DMatrix(sample), pred_contribs=True)
            contribs = np.asarray(contribs)
            # Shape is (n, n_features + 1) or (n, n_classes, n_features + 1); the trailing column
            # is the bias term, which is not a feature attribution.
            if contribs.ndim == 3:
                per_feature = contribs[:, :, :-1]
                mean_abs = np.abs(per_feature).mean(axis=(0, 1))
                plot_values = per_feature[:, 0, :]
            else:
                per_feature = contribs[:, :-1]
                mean_abs = np.abs(per_feature).mean(axis=0)
                plot_values = per_feature

            shap_frame = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            ranked = ranked.merge(shap_frame, on="feature", how="left")
            ranked["shap_rank"] = ranked["mean_abs_shap"].rank(ascending=False, method="min")
            shap_frame.sort_values("mean_abs_shap", ascending=False).to_csv(
                REPO_ROOT / f"results/tables/feature_selection_{args.endpoint}_shap_values.csv", index=False
            )

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                top = shap_frame.sort_values("mean_abs_shap", ascending=False).head(25).iloc[::-1]
                fig, ax = plt.subplots(figsize=(9, 8))
                ax.barh(range(len(top)), top["mean_abs_shap"], color="#2E5496")
                ax.set_yticks(range(len(top)))
                ax.set_yticklabels([name[:58] for name in top["feature"]], fontsize=8)
                ax.set_xlabel("mean |SHAP value|")
                ax.set_title(f"Bio MAF v4 feature attribution — {args.endpoint}", fontsize=11)
                ax.spines[["top", "right"]].set_visible(False)
                shap_path = REPO_ROOT / f"results/figures/feature_selection_{args.endpoint}_shap_summary.png"
                shap_path.parent.mkdir(parents=True, exist_ok=True)
                fig.tight_layout()
                fig.savefig(shap_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"SHAP summary plot -> {shap_path.relative_to(REPO_ROOT)}", flush=True)
            except ImportError:
                print("matplotlib not installed; SHAP values written but no plot", flush=True)
        except Exception as exc:  # pragma: no cover - attribution is best-effort
            print(f"SHAP step failed: {exc}", flush=True)

    tables = REPO_ROOT / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)
    grid.to_csv(tables / f"feature_selection_{args.endpoint}_grid.csv", index=False)
    ranked.to_csv(tables / f"feature_selection_{args.endpoint}_selected.csv", index=False)
    summary = {
        "endpoint": args.endpoint,
        "task": task,
        "n_samples": int(len(y)),
        "n_features_available": len(feature_names),
        "n_train": int(len(idx_train)),
        "n_heldout": int(len(idx_test)),
        "inner_folds": int(args.inner_folds),
        "tolerance": float(args.tolerance),
        "best_inner_score": float(best),
        "winner_setting": int(winner["setting"]),
        "winner_params": winner_params,
        "winner_inner_score": float(winner["inner_score_mean"]),
        "heldout_score": float(heldout),
        "n_features_selected": int(len(selected)),
        "selection_fraction": float(len(selected) / max(1, len(feature_names))),
        "shap_figure": str(shap_path.relative_to(REPO_ROOT)) if shap_path else None,
        "protocol": "80/20 split; inner k-fold tunes sparsity; sparsest within tolerance; refit on train; scored once on held-out",
    }
    (tables / f"feature_selection_{args.endpoint}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"{args.endpoint}: {len(selected)} of {len(feature_names)} features carry signal "
          f"({100 * len(selected) / len(feature_names):.1f}%)")
    print("=" * 74)
    print(ranked.head(20).to_string(index=False))
    print(f"\nwritten -> results/tables/feature_selection_{args.endpoint}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
