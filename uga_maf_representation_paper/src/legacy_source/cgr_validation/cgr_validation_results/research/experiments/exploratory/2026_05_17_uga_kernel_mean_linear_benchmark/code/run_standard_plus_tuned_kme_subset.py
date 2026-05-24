#!/usr/bin/env python3
"""Minimal Standard + tuned KME add-on experiment for selected endpoints."""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_hrd33_kme_grid_tuned_panel as tuned
import run_uga_kernel_mean_linear_benchmark as base


SUBSET_ENDPOINTS = ["hrd_binary_33", "smoking_ever", "luad_kmt2c_mutated", "cancer_type_top10"]


def subset_endpoints(mc3: tuned.KMEUniverse) -> list[base.Endpoint]:
    endpoints = base.load_hrd_endpoints() + base.load_mc3_clinical_endpoints() + [base.load_kmt2c_endpoint(mc3.standard.frame.index)]
    endpoint_by_name = {endpoint.name: endpoint for endpoint in endpoints}
    missing = [name for name in SUBSET_ENDPOINTS if name not in endpoint_by_name]
    if missing:
        raise RuntimeError(f"Missing requested endpoints: {missing}")
    return [endpoint_by_name[name] for name in SUBSET_ENDPOINTS]


def selected_params() -> pd.DataFrame:
    path = base.DATA_DIR / "kme_grid_selected_params.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run HRD33 KME tuning first; missing {path}")
    return pd.read_csv(path)


def standard_plus_tuned_kme_feature_set(mc3: tuned.KMEUniverse, selected: pd.Series, learner: str) -> base.FeatureSet:
    tuned_feature_set, _diagnostics, _metadata = tuned.build_tuned_feature_set(
        mc3,
        representation=f"{learner}_tuned_kme_for_addon",
        kernel_width_multiplier=float(selected["kernel_width_multiplier"]),
        n_landmarks=int(selected["requested_n_landmarks"]),
    )
    kme_only_cols = [col for col in tuned_feature_set.frame.columns if col not in mc3.standard.frame.columns]
    combined = pd.concat([mc3.standard.frame, tuned_feature_set.frame.loc[:, kme_only_cols]], axis=1).fillna(0.0)
    return base.FeatureSet(
        f"standard_plus_{learner}_tuned_kme",
        "mc3_hrd_kmt2c",
        combined,
        (
            "Retained Standard SBS96+ID83 features concatenated with HRD33-frozen tuned KME landmark features; "
            "burden covariates are not duplicated."
        ),
        "standard_sbs96_id83 + HRD33-frozen tuned KME",
        kernel_sigma=float(tuned_feature_set.kernel_sigma) if tuned_feature_set.kernel_sigma is not None else None,
        n_encoded_channels=tuned_feature_set.n_encoded_channels,
        n_landmarks=tuned_feature_set.n_landmarks,
    )


def evaluate_subset(
    learner: str,
    mc3: tuned.KMEUniverse,
    selected: pd.Series,
    endpoints: list[base.Endpoint],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combo = standard_plus_tuned_kme_feature_set(mc3, selected, learner)
    result_rows = []
    prediction_frames = []
    probability_frames = []
    for endpoint in endpoints:
        for feature_set in [mc3.standard, combo]:
            result, predictions, probabilities = tuned.evaluate_with_learner(learner, endpoint, feature_set)
            result["learner"] = learner
            result_rows.append(result)
            prediction_frames.append(predictions.assign(learner=learner))
            if not probabilities.empty:
                probability_frames.append(probabilities.assign(learner=learner))
    results = pd.DataFrame(result_rows)
    results["delta_vs_standard"] = math.nan
    for endpoint, group in results.groupby("endpoint", dropna=False):
        standard = group.loc[group["representation"] == mc3.standard.name, "score"]
        if standard.empty:
            continue
        results.loc[group.index, "delta_vs_standard"] = group["score"] - float(standard.iloc[0])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    probabilities = pd.concat(probability_frames, ignore_index=True)
    return results, predictions, probabilities


def validate(results: pd.DataFrame, predictions: pd.DataFrame, probabilities: pd.DataFrame) -> dict[str, object]:
    validation: dict[str, object] = {
        "n_result_rows": int(len(results)),
        "expected_result_rows": 16,
        "missing_scores": int(results["score"].isna().sum()),
        "representation_counts_are_two": bool(
            (results.groupby(["learner", "endpoint"], dropna=False)["representation"].nunique() == 2).all()
        ),
        "duplicate_prediction_keys": int(
            predictions.duplicated(["learner", "benchmark", "endpoint", "representation", "sample"]).sum()
        ),
    }
    mismatches = []
    for (learner, endpoint), group in predictions.groupby(["learner", "endpoint"], dropna=False):
        folds = {
            rep: frame.set_index("sample")["fold"].sort_index()
            for rep, frame in group.groupby("representation", dropna=False)
        }
        reps = sorted(folds)
        if len(reps) == 2 and not folds[reps[0]].equals(folds[reps[1]]):
            mismatches.append(f"{learner}/{endpoint}")
    validation["fold_mismatch_count"] = int(len(mismatches))
    validation["fold_mismatch_examples"] = mismatches[:10]
    probability_bad_rows = 0
    if not probabilities.empty:
        sums = probabilities.groupby(["learner", "benchmark", "endpoint", "representation", "sample"], dropna=False)["probability"].sum()
        probability_bad_rows = int((np.abs(sums - 1.0) > 1e-6).sum())
    validation["probability_bad_row_count"] = probability_bad_rows
    return validation


def wide_results(results: pd.DataFrame) -> pd.DataFrame:
    standard_name = "standard_sbs96_id83"
    combo_names = {
        "linear": "standard_plus_linear_tuned_kme",
        "xgboost": "standard_plus_xgboost_tuned_kme",
    }
    rows = []
    for (learner, endpoint), group in results.groupby(["learner", "endpoint"], dropna=False):
        standard = group[group["representation"] == standard_name].iloc[0]
        combo = group[group["representation"] == combo_names[str(learner)]].iloc[0]
        rows.append(
            {
                "learner": learner,
                "family": combo["family"],
                "endpoint": endpoint,
                "metric": combo["metric"],
                "standard_score": float(standard["score"]),
                "standard_plus_kme_score": float(combo["score"]),
                "delta_standard_plus_kme_minus_standard": float(combo["score"] - standard["score"]),
                "standard_features": int(standard["n_features"]),
                "standard_plus_kme_features": int(combo["n_features"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["learner", "family", "endpoint"])


def update_readme(results: pd.DataFrame, wide: pd.DataFrame, validation: dict[str, object], elapsed: float) -> None:
    readme = base.EXPERIMENT_ROOT / "README.md"
    marker = "\n## Standard Plus Tuned KME Subset\n"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else "# UGA RBF Kernel Mean Linear Benchmark\n"
    head = existing.split(marker, 1)[0].rstrip()
    wins = wide.groupby("learner")["delta_standard_plus_kme_minus_standard"].apply(lambda x: int((x > 0).sum())).to_dict()
    lines = [
        "",
        "## Standard Plus Tuned KME Subset",
        "",
        f"Executed UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Elapsed seconds: {elapsed:.2f}",
        "",
        "This minimal add-on experiment asks whether HRD33-tuned KME features add useful information beyond the raw Standard spectra.",
        "",
        "### Methodology",
        "",
        "- Endpoints: `hrd_binary_33`, `smoking_ever`, `luad_kmt2c_mutated`, and `cancer_type_top10`.",
        "- Comparators: Standard SBS96+ID83 versus Standard SBS96+ID83 concatenated with the frozen tuned KME landmarks.",
        "- Burden covariates were not duplicated in the concatenated feature matrix.",
        "- Linear and XGBoost used the same locked settings and fixed folds as the tuned KME panel.",
        "",
        "### Results",
        "",
        wide.to_markdown(index=False, floatfmt=".6f"),
        "",
        "### Headline",
        "",
        f"Linear Standard+KME beat Standard on {wins.get('linear', 0)}/4 subset endpoints.",
        f"XGBoost Standard+KME beat Standard on {wins.get('xgboost', 0)}/4 subset endpoints.",
        "",
        "### Validation",
        "",
        "```json",
        json.dumps(validation, indent=2),
        "```",
    ]
    readme.write_text(head + marker + "\n".join(lines[2:]) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    base.ensure_dirs()
    mc3 = tuned.build_mc3_universe()
    endpoints = subset_endpoints(mc3)
    selected = selected_params()

    frames = []
    prediction_frames = []
    probability_frames = []
    for learner in ["linear", "xgboost"]:
        selected_row = selected[selected["learner"] == learner].iloc[0]
        results, predictions, probabilities = evaluate_subset(learner, mc3, selected_row, endpoints)
        frames.append(results)
        prediction_frames.append(predictions)
        probability_frames.append(probabilities)
    results = pd.concat(frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    probabilities = pd.concat(probability_frames, ignore_index=True)
    wide = wide_results(results)
    validation = validate(results, predictions, probabilities)

    results.to_csv(base.DATA_DIR / "standard_plus_tuned_kme_subset_results.csv", index=False)
    wide.to_csv(base.TABLE_DIR / "standard_plus_tuned_kme_subset_summary.csv", index=False)
    predictions.to_csv(base.DATA_DIR / "standard_plus_tuned_kme_subset_oof_predictions.csv.gz", index=False, compression="gzip")
    probabilities.to_csv(base.DATA_DIR / "standard_plus_tuned_kme_subset_oof_probabilities_long.csv.gz", index=False, compression="gzip")
    metadata = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - start,
        "endpoints": SUBSET_ENDPOINTS,
        "selected_params_source": str(base.DATA_DIR / "kme_grid_selected_params.csv"),
        "dyld_library_path": os.environ.get("DYLD_LIBRARY_PATH", ""),
        "validation": validation,
    }
    (base.DATA_DIR / "standard_plus_tuned_kme_subset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    update_readme(results, wide, validation, time.time() - start)

    print(
        json.dumps(
            {
                "experiment_root": str(base.EXPERIMENT_ROOT),
                "elapsed_seconds": round(time.time() - start, 2),
                "results": wide.to_dict("records"),
                "validation": validation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
