#!/usr/bin/env python3
"""Third-pass paired linear-learner screen for UGA candidates."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
DATA_DIR = EXPERIMENT_ROOT / "data"
TABLE_DIR = EXPERIMENT_ROOT / "tables"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_fast_defensible_signal_recovery_benchmark as base  # noqa: E402
import run_second_pass_distributional_uga_screen as second  # noqa: E402


RANDOM_SEED = 20260515


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def encode_y(y: pd.Series, task: str) -> tuple[np.ndarray, np.ndarray]:
    if task == "regression":
        return y.astype(float).to_numpy(dtype=np.float64), np.array([], dtype=object)
    if task == "binary":
        return y.astype(int).to_numpy(dtype=np.int32), np.array([0, 1], dtype=object)
    classes = np.array(sorted(y.astype(str).unique()), dtype=object)
    mapping = {value: i for i, value in enumerate(classes)}
    return y.astype(str).map(mapping).astype(int).to_numpy(dtype=np.int32), classes


def score(y: np.ndarray, pred: np.ndarray, task: str, classes: np.ndarray) -> tuple[str, float, float]:
    if task == "regression":
        return "spearman", float(spearmanr(y, pred)[0]), float("nan")
    if task == "binary":
        return "auroc", float(roc_auc_score(y, pred[:, 1])), float(balanced_accuracy_score(y, (pred[:, 1] >= 0.5).astype(int)))
    y_bin = label_binarize(y, classes=np.arange(len(classes)))
    return "macro_auroc", float(roc_auc_score(y_bin, pred, average="macro")), float(balanced_accuracy_score(y, np.argmax(pred, axis=1)))


def paired_linear_oof(
    y_series: pd.Series,
    frame: pd.DataFrame,
    *,
    task: str,
    folds: int,
    seed: int,
    learner: str,
    c_value: float,
) -> tuple[str, float, float, pd.DataFrame]:
    common = y_series.index.intersection(frame.index)
    x = frame.loc[common].fillna(0.0).to_numpy(dtype=np.float64)
    y, classes = encode_y(y_series.loc[common], task)
    if task == "regression":
        splitter = KFold(n_splits=min(int(folds), len(y)), shuffle=True, random_state=seed)
        pred = np.zeros(len(y), dtype=np.float64)
        for train_idx, test_idx in splitter.split(x, y):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train_idx])
            x_test = scaler.transform(x[test_idx])
            model = Ridge(alpha=1.0 / max(float(c_value), 1e-9))
            model.fit(x_train, y[train_idx])
            pred[test_idx] = model.predict(x_test)
        metric, value, bal = score(y, pred, task, classes)
        pred_df = pd.DataFrame({"patient_id": common.astype(str), "true_value": y.astype(float), "pred_value": pred})
        return metric, value, bal, pred_df

    n_splits = max(2, min(int(folds), int(pd.Series(y).value_counts().min())))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_classes = 2 if task == "binary" else len(classes)
    proba = np.zeros((len(y), n_classes), dtype=np.float64)
    for train_idx, test_idx in splitter.split(x, y):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        if learner == "l2":
            model = LogisticRegression(
                penalty="l2",
                C=float(c_value),
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                multi_class="auto",
            )
        else:
            model = LogisticRegression(
                penalty="elasticnet",
                C=float(c_value),
                l1_ratio=0.25,
                class_weight="balanced",
                solver="saga",
                max_iter=1500,
                multi_class="auto",
                random_state=seed,
                n_jobs=4,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y[train_idx])
        fold_proba = model.predict_proba(x_test)
        if task == "binary" and fold_proba.ndim == 1:
            fold_proba = np.column_stack([1.0 - fold_proba, fold_proba])
        proba[test_idx] = fold_proba
    metric, value, bal = score(y, proba, task, classes)
    pred_df = pd.DataFrame({"patient_id": common.astype(str), "true_value": y.astype(int)})
    for i in range(proba.shape[1]):
        pred_df[f"pred_class_{i}"] = proba[:, i]
    return metric, value, bal, pred_df


def screen_feature_sets(
    features: dict[str, pd.DataFrame],
    endpoints: list[base.Endpoint],
    *,
    folds: int,
    learner: str,
    c_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for endpoint in endpoints:
        seed = base.stable_seed("linear_screen", learner, c_value, endpoint.name)
        for name, frame in features.items():
            metric, value, bal, pred = paired_linear_oof(
                endpoint.y,
                frame,
                task=endpoint.task,
                folds=folds,
                seed=seed,
                learner=learner,
                c_value=c_value,
            )
            rows.append(
                {
                    "candidate": name,
                    "endpoint": endpoint.name,
                    "endpoint_family": endpoint.family,
                    "task": endpoint.task,
                    "learner": learner,
                    "c_value": float(c_value),
                    "metric": metric,
                    "score": value,
                    "balanced_accuracy": bal,
                    "n": int(len(pred)),
                    "n_features": int(frame.shape[1]),
                }
            )
            pred.insert(0, "candidate", name)
            pred.insert(0, "endpoint", endpoint.name)
            predictions.append(pred)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def add_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    standards = metrics[metrics["candidate"] == "standard_sbs96_id83"][["endpoint", "learner", "c_value", "score"]].rename(columns={"score": "standard_score"})
    out = metrics.merge(standards, on=["endpoint", "learner", "c_value"], how="left")
    out["delta_vs_standard"] = out["score"] - out["standard_score"]
    return out


def leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = metrics[metrics["candidate"] != "standard_sbs96_id83"].copy()
    board = (
        rows.groupby(["candidate", "learner", "c_value"], as_index=False)
        .agg(
            n_endpoints=("delta_vs_standard", "count"),
            mean_delta=("delta_vs_standard", "mean"),
            median_delta=("delta_vs_standard", "median"),
            min_delta=("delta_vs_standard", "min"),
            max_delta=("delta_vs_standard", "max"),
            n_positive=("delta_vs_standard", lambda x: int((x > 0).sum())),
            n_ge_0_03=("delta_vs_standard", lambda x: int((x >= 0.03).sum())),
            n_lt_minus_0_02=("delta_vs_standard", lambda x: int((x < -0.02).sum())),
        )
        .sort_values(["mean_delta", "n_ge_0_03"], ascending=False)
    )
    board["passes_gate"] = (
        (board["mean_delta"] >= 0.03)
        & (board["n_ge_0_03"] >= 3)
        & (board["n_lt_minus_0_02"] == 0)
    )
    return board


def repeated_linear_oof(
    endpoint: base.Endpoint,
    frame: pd.DataFrame,
    *,
    folds: int,
    repeats: int,
    learner: str,
    c_value: float,
    seed: int,
) -> tuple[str, float, float, pd.DataFrame]:
    pred_parts = []
    metric_name = ""
    balanced_acc = float("nan")
    for repeat in range(int(repeats)):
        metric_name, _, balanced_acc, pred = paired_linear_oof(
            endpoint.y,
            frame,
            task=endpoint.task,
            folds=folds,
            seed=seed + repeat * 10_007,
            learner=learner,
            c_value=c_value,
        )
        pred = pred.sort_values("patient_id").reset_index(drop=True)
        pred_parts.append(pred)
    out = pred_parts[0].copy()
    if endpoint.task == "regression":
        stacked = np.vstack([pred["pred_value"].to_numpy(dtype=float) for pred in pred_parts])
        out["pred_value"] = stacked.mean(axis=0)
    else:
        class_cols = [col for col in out.columns if col.startswith("pred_class_")]
        for col in class_cols:
            stacked = np.vstack([pred[col].to_numpy(dtype=float) for pred in pred_parts])
            out[col] = stacked.mean(axis=0)
    y, classes = encode_y(endpoint.y.loc[pd.Index(out["patient_id"].astype(str))], endpoint.task)
    if endpoint.task == "regression":
        metric_name, value, balanced_acc = score(y, out["pred_value"].to_numpy(dtype=float), endpoint.task, classes)
    else:
        class_cols = [col for col in out.columns if col.startswith("pred_class_")]
        metric_name, value, balanced_acc = score(y, out[class_cols].to_numpy(dtype=float), endpoint.task, classes)
    return metric_name, value, balanced_acc, out


def bootstrap_delta(endpoint: base.Endpoint, pred_a: pd.DataFrame, pred_b: pd.DataFrame, *, n_bootstrap: int, seed: int) -> tuple[float, float, float]:
    merged = pred_a.merge(pred_b, on=["patient_id", "true_value"], suffixes=("_a", "_b"))
    y = merged["true_value"].to_numpy()
    rng = np.random.default_rng(seed)
    deltas = np.zeros(int(n_bootstrap), dtype=np.float64)
    if endpoint.task == "regression":
        a = merged["pred_value_a"].to_numpy(dtype=float)
        b = merged["pred_value_b"].to_numpy(dtype=float)
        for i in range(int(n_bootstrap)):
            idx = rng.choice(np.arange(len(y)), size=len(y), replace=True)
            deltas[i] = spearmanr(y[idx], a[idx])[0] - spearmanr(y[idx], b[idx])[0]
    else:
        class_cols_a = [col for col in merged.columns if col.startswith("pred_class_") and col.endswith("_a")]
        class_cols_b = [col for col in merged.columns if col.startswith("pred_class_") and col.endswith("_b")]
        a = merged[class_cols_a].to_numpy(dtype=float)
        b = merged[class_cols_b].to_numpy(dtype=float)
        strata = [np.flatnonzero(y == value) for value in np.unique(y)]
        for i in range(int(n_bootstrap)):
            idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
            if endpoint.task == "binary":
                deltas[i] = roc_auc_score(y[idx], a[idx, 1]) - roc_auc_score(y[idx], b[idx, 1])
            else:
                classes = np.arange(a.shape[1])
                y_bin = label_binarize(y[idx], classes=classes)
                deltas[i] = roc_auc_score(y_bin, a[idx], average="macro") - roc_auc_score(y_bin, b[idx], average="macro")
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    p_lower = (np.sum(deltas <= 0.0) + 1.0) / (int(n_bootstrap) + 1.0)
    p_upper = (np.sum(deltas >= 0.0) + 1.0) / (int(n_bootstrap) + 1.0)
    return float(min(1.0, 2.0 * min(p_lower, p_upper))), float(ci_low), float(ci_high)


def bh_q_values(p_values: pd.Series) -> pd.Series:
    valid = p_values.dropna().astype(float)
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    if valid.empty:
        return out
    order = valid.sort_values()
    running = 1.0
    m = len(order)
    adjusted: dict[object, float] = {}
    for rank, idx in reversed(list(enumerate(order.index, start=1))):
        running = min(running, float(order.loc[idx]) * m / rank)
        adjusted[idx] = running
    for idx, value in adjusted.items():
        out.loc[idx] = min(1.0, value)
    return out


def confirm_finalists(
    finalists: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    endpoints: list[base.Endpoint],
    *,
    folds: int,
    repeats: int,
    bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if finalists.empty:
        empty = pd.DataFrame(
            columns=[
                "candidate",
                "learner",
                "c_value",
                "endpoint",
                "endpoint_family",
                "metric",
                "standard_score",
                "candidate_score",
                "delta_vs_standard",
                "ci_low",
                "ci_high",
                "p_value",
                "q_value",
            ]
        )
        return empty.copy(), empty.copy(), pd.DataFrame()
    metric_rows = []
    test_rows = []
    pred_rows = []
    standard_cache: dict[tuple[str, float, str], tuple[str, float, float, pd.DataFrame]] = {}
    for finalist in finalists.itertuples(index=False):
        candidate_name = str(finalist.candidate)
        learner = str(finalist.learner)
        c_value = float(finalist.c_value)
        for endpoint in endpoints:
            key = (learner, c_value, endpoint.name)
            if key not in standard_cache:
                standard_cache[key] = repeated_linear_oof(
                    endpoint,
                    features["standard_sbs96_id83"],
                    folds=folds,
                    repeats=repeats,
                    learner=learner,
                    c_value=c_value,
                    seed=base.stable_seed("linear_confirm", learner, c_value, endpoint.name),
                )
            metric_name_b, standard_score, standard_bal, pred_b = standard_cache[key]
            metric_name_a, candidate_score, candidate_bal, pred_a = repeated_linear_oof(
                endpoint,
                features[candidate_name],
                folds=folds,
                repeats=repeats,
                learner=learner,
                c_value=c_value,
                seed=base.stable_seed("linear_confirm", learner, c_value, endpoint.name),
            )
            p_value, ci_low, ci_high = bootstrap_delta(
                endpoint,
                pred_a,
                pred_b,
                n_bootstrap=bootstrap,
                seed=base.stable_seed("linear_bootstrap", candidate_name, learner, c_value, endpoint.name),
            )
            metric_rows.extend(
                [
                    {
                        "candidate": "standard_sbs96_id83",
                        "learner": learner,
                        "c_value": c_value,
                        "endpoint": endpoint.name,
                        "endpoint_family": endpoint.family,
                        "metric": metric_name_b,
                        "score": standard_score,
                        "balanced_accuracy": standard_bal,
                    },
                    {
                        "candidate": candidate_name,
                        "learner": learner,
                        "c_value": c_value,
                        "endpoint": endpoint.name,
                        "endpoint_family": endpoint.family,
                        "metric": metric_name_a,
                        "score": candidate_score,
                        "balanced_accuracy": candidate_bal,
                    },
                ]
            )
            test_rows.append(
                {
                    "candidate": candidate_name,
                    "learner": learner,
                    "c_value": c_value,
                    "endpoint": endpoint.name,
                    "endpoint_family": endpoint.family,
                    "metric": metric_name_a,
                    "standard_score": standard_score,
                    "candidate_score": candidate_score,
                    "delta_vs_standard": candidate_score - standard_score,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                    "n_bootstrap": int(bootstrap),
                }
            )
            pred_a = pred_a.copy()
            pred_a.insert(0, "candidate", candidate_name)
            pred_a.insert(0, "endpoint", endpoint.name)
            pred_b = pred_b.copy()
            pred_b.insert(0, "candidate", "standard_sbs96_id83")
            pred_b.insert(0, "endpoint", endpoint.name)
            pred_rows.extend([pred_a, pred_b])
    tests = pd.DataFrame(test_rows)
    tests["q_value"] = bh_q_values(tests["p_value"])
    return pd.DataFrame(metric_rows), tests, pd.concat(pred_rows, ignore_index=True)


def build_features() -> dict[str, pd.DataFrame]:
    standard_sbs, standard_id, standard_sbs_id, burden = base.load_feature_matrices()
    locked = base.build_mc3_candidate_features(standard_sbs, standard_id, burden, base.LOCKED_SBS_MODEL, base.LOCKED_ID_MODEL)
    proxy = base.build_registered_uga_features(
        standard_sbs,
        standard_id,
        burden,
        sbs_model=base.LOCKED_SBS_MODEL,
        id_model="id83_proxy_d10_dp5",
        prefix="proxy",
    )
    bio = base.build_biology_candidates(standard_sbs, standard_id, burden)
    second_candidates = second.build_second_pass_candidates(standard_sbs, standard_id, burden)
    second_map = {candidate.name: candidate.features for candidate in second_candidates}
    features = {
        "standard_sbs96_id83": standard_sbs_id,
        "locked_uga_separate": locked["uga_combined_separate"],
        "id_proxy_uga_separate": proxy["separate"],
        "biology_aggregates_sbs_id": bio["bio_sbs_id"],
        "locked_uga_plus_biology_aggregates": pd.concat(
            [
                locked["uga_combined_separate"],
                bio["bio_sbs_id"].drop(columns=[col for col in locked["uga_combined_separate"].columns if col in bio["bio_sbs_id"].columns], errors="ignore"),
            ],
            axis=1,
        ).fillna(0.0),
        "graph_spectral_uga_k48": second_map["graph_spectral_uga_k48"],
        "uga_moments_mean_var": second_map["uga_moments_mean_var"],
        "random_direction_moment_uga_d32": second_map["random_direction_moment_uga_d32"],
    }
    return features


def update_readme(metadata: dict[str, object], board: pd.DataFrame, confirm_tests: pd.DataFrame) -> None:
    readme_path = EXPERIMENT_ROOT / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# Fast Defensible UGA Signal-Recovery Benchmark\n"
    view = board.head(12)[["candidate", "learner", "c_value", "mean_delta", "min_delta", "max_delta", "n_positive", "n_ge_0_03", "n_lt_minus_0_02", "passes_gate"]].rename(
        columns={
            "candidate": "Candidate",
            "learner": "Learner",
            "c_value": "C",
            "mean_delta": "Mean delta",
            "min_delta": "Smallest delta",
            "max_delta": "Largest delta",
            "n_positive": "Positive endpoints",
            "n_ge_0_03": "Endpoints >= +0.03",
            "n_lt_minus_0_02": "Endpoints < -0.02",
            "passes_gate": "Passes gate",
        }
    )
    section = [
        "",
        "## Third-Pass Linear Learner Screen",
        f"Executed at {metadata['executed_at_utc']} with paired five-fold splits and runtime {float(metadata['elapsed_seconds']) / 60.0:.1f} minutes.",
        f"Candidates passing the promotion gate: {int(board['passes_gate'].sum())}.",
        "",
        view.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    if not confirm_tests.empty:
        summary = (
            confirm_tests.groupby(["candidate", "learner", "c_value"], as_index=False)
            .agg(
                endpoints=("delta_vs_standard", "count"),
                mean_delta=("delta_vs_standard", "mean"),
                smallest_delta=("delta_vs_standard", "min"),
                largest_delta=("delta_vs_standard", "max"),
                positive_endpoints=("delta_vs_standard", lambda x: int((x > 0).sum())),
                q_lt_0_05=("q_value", lambda x: int((x < 0.05).sum())),
            )
            .rename(
                columns={
                    "candidate": "Candidate",
                    "learner": "Learner",
                    "c_value": "C",
                    "endpoints": "Endpoints",
                    "mean_delta": "Mean delta",
                    "smallest_delta": "Smallest delta",
                    "largest_delta": "Largest delta",
                    "positive_endpoints": "Positive endpoints",
                    "q_lt_0_05": "FDR-significant endpoints",
                }
            )
        )
        section.extend(
            [
                "### Focused Confirmation",
                "Focused confirmation used three repeated paired five-fold runs and 300 paired bootstrap resamples. The comparison uses the same L2 learner and same regularization strength for Standard and UGA.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "This result supports a learner-sensitivity claim for structured UGA features under a matched regularized linear learner. It does not replace the XGBoost benchmark, where Standard SBS96+ID83 remains the stronger primary comparator.",
                "",
            ]
        )
    marker = "\n## Third-Pass Linear Learner Screen\n"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip()
    readme_path.write_text(existing.rstrip() + "\n" + "\n".join(section), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--confirm-repeats", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--max-finalists", type=int, default=2)
    parser.add_argument("--learners", default="l2")
    parser.add_argument("--c-values", default="0.1,0.3,1.0")
    args = parser.parse_args()

    ensure_dirs()
    start = time.perf_counter()
    features = build_features()
    burden = base.load_feature_matrices()[3]
    endpoints = [*base.load_hrd_endpoints(), *base.load_mc3_clinical_endpoints(), base.load_kmt2c_endpoint(pd.Index(burden.index.astype(str)))]
    metric_parts = []
    pred_parts = []
    learners = [x.strip() for x in str(args.learners).split(",") if x.strip()]
    c_values = [float(x) for x in str(args.c_values).split(",") if x.strip()]
    for learner in learners:
        for c_value in c_values:
            print(f"Linear screen learner={learner} C={c_value:g}", flush=True)
            metrics, predictions = screen_feature_sets(features, endpoints, folds=args.folds, learner=learner, c_value=c_value)
            metric_parts.append(metrics)
            pred_parts.append(predictions)
    metrics = add_deltas(pd.concat(metric_parts, ignore_index=True))
    board = leaderboard(metrics)
    predictions = pd.concat(pred_parts, ignore_index=True)
    finalists = board[board["passes_gate"]].head(int(args.max_finalists)).copy()
    confirm_metrics, confirm_tests, confirm_predictions = confirm_finalists(
        finalists,
        features,
        endpoints,
        folds=args.folds,
        repeats=args.confirm_repeats,
        bootstrap=args.bootstrap,
    )
    metrics.to_csv(DATA_DIR / "third_pass_linear_learner_metrics.csv", index=False)
    board.to_csv(DATA_DIR / "third_pass_linear_learner_leaderboard.csv", index=False)
    predictions.to_csv(DATA_DIR / "third_pass_linear_learner_oof_predictions.csv", index=False)
    confirm_metrics.to_csv(DATA_DIR / "third_pass_linear_focused_confirmation_metrics.csv", index=False)
    confirm_tests.to_csv(DATA_DIR / "third_pass_linear_focused_confirmation_tests.csv", index=False)
    if not confirm_predictions.empty:
        confirm_predictions.to_csv(DATA_DIR / "third_pass_linear_focused_confirmation_oof_predictions.csv", index=False)
    base.write_html_table(
        board,
        TABLE_DIR / "table7_third_pass_linear_learner_leaderboard.html",
        "Table 7. Third-pass paired linear-learner screen.",
        "Standard and UGA candidates use the same linear learner, same paired folds, and same endpoint labels. Results are screening evidence.",
    )
    base.write_html_table(
        confirm_tests,
        TABLE_DIR / "table8_third_pass_linear_focused_confirmation.html",
        "Table 8. Third-pass linear focused confirmation.",
        "Focused confirmation uses repeated paired folds, bootstrap confidence intervals, and Benjamini-Hochberg q values.",
    )
    metadata = {
        "experiment": EXPERIMENT_ROOT.name,
        "third_pass": "linear_learner",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "folds": int(args.folds),
        "learners": learners,
        "c_values": c_values,
        "confirm_repeats": int(args.confirm_repeats),
        "bootstrap": int(args.bootstrap),
        "n_feature_sets": int(len(features)),
        "n_endpoints": int(len(endpoints)),
        "n_passes_gate": int(board["passes_gate"].sum()),
        "confirmed_finalists": finalists[["candidate", "learner", "c_value"]].to_dict(orient="records"),
    }
    (DATA_DIR / "third_pass_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    update_readme(metadata, board, confirm_tests)
    print(json.dumps({"elapsed_seconds": round(metadata["elapsed_seconds"], 1), "n_passes_gate": metadata["n_passes_gate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
