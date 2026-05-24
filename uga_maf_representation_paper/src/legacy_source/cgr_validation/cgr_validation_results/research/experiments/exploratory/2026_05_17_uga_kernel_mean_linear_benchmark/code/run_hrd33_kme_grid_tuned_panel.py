#!/usr/bin/env python3
"""Tune UGA-RBF KME parameters on HRD33, freeze, and rerun the full panel."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import run_uga_kernel_mean_linear_benchmark as base
import run_uga_kernel_mean_xgboost_sensitivity as xgb_runner


PRIMARY_ENDPOINT = "hrd_binary_33"
KERNEL_WIDTH_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0]
LANDMARK_COUNTS = [25, 50, 100, 179]


@dataclass(frozen=True)
class KMEUniverse:
    benchmark: str
    counts: pd.DataFrame
    basis: np.ndarray
    valid: np.ndarray
    channel_labels: list[str]
    burden: pd.DataFrame | None
    standard: base.FeatureSet
    previous: base.FeatureSet
    endpoint_frame: pd.DataFrame | None = None


def valid_mask(basis: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.asarray(valid, dtype=bool) & np.isfinite(basis).all(axis=1)


def farthest_point_landmarks(points: np.ndarray, n_landmarks: int) -> np.ndarray:
    n_points = int(points.shape[0])
    if n_landmarks >= n_points:
        return np.arange(n_points, dtype=int)
    if n_landmarks <= 0:
        raise ValueError("n_landmarks must be positive")
    centroid = points.mean(axis=0)
    start = int(np.argmin(np.sum((points - centroid) ** 2, axis=1)))
    selected = [start]
    min_dist2 = np.sum((points - points[start]) ** 2, axis=1)
    min_dist2[start] = -np.inf
    while len(selected) < n_landmarks:
        next_idx = int(np.argmax(min_dist2))
        selected.append(next_idx)
        dist2 = np.sum((points - points[next_idx]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
        min_dist2[selected] = -np.inf
    return np.asarray(selected, dtype=int)


def tuned_rbf_kme_features(
    counts: pd.DataFrame,
    basis: np.ndarray,
    valid: np.ndarray,
    *,
    prefix: str,
    channel_labels: Iterable[str],
    kernel_width_multiplier: float,
    n_landmarks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    mask = valid_mask(basis, valid)
    if basis.shape[0] != counts.shape[1]:
        raise ValueError(f"Basis rows ({basis.shape[0]}) do not match count columns ({counts.shape[1]})")
    if not bool(mask.any()):
        raise ValueError("No valid UGA channel coordinates are available for KME features")
    labels = np.asarray(list(channel_labels), dtype=object)
    valid_indices = np.flatnonzero(mask)
    valid_points = basis[valid_indices]
    median_distance = base.median_pairwise_scale(valid_points)
    local_landmark_idx = farthest_point_landmarks(valid_points, int(n_landmarks))
    landmark_indices = valid_indices[local_landmark_idx]
    landmarks = basis[landmark_indices]
    sigma = float(kernel_width_multiplier) * median_distance

    dist2 = np.sum((basis[:, None, :] - landmarks[None, :, :]) ** 2, axis=2)
    kernel = np.exp(-0.5 * dist2 / max(sigma * sigma, 1e-12))
    kernel[~mask, :] = 0.0
    raw = counts.to_numpy(dtype=np.float64)
    masked = raw * mask.astype(np.float64)
    denom = masked.sum(axis=1, keepdims=True)
    weights = np.divide(masked, denom, out=np.zeros_like(masked), where=denom > 0)
    embedded = weights @ kernel
    columns = [f"{prefix}__{labels[i]}" for i in landmark_indices]
    features = pd.DataFrame(embedded, index=counts.index, columns=columns)
    diagnostics = pd.DataFrame(
        {
            "feature_prefix": prefix,
            "channel": labels,
            "uga_encoded": mask,
            "used_as_landmark": np.isin(np.arange(len(labels)), landmark_indices),
        }
    )
    metadata = {
        "kernel_width_multiplier": float(kernel_width_multiplier),
        "median_pairwise_distance": float(median_distance),
        "kernel_sigma": float(sigma),
        "requested_n_landmarks": int(n_landmarks),
        "actual_n_landmarks": int(len(landmark_indices)),
        "n_input_channels": int(counts.shape[1]),
        "n_encoded_channels": int(mask.sum()),
        "feature_dimension": int(features.shape[1]),
    }
    return features, diagnostics, metadata


def build_mc3_universe() -> KMEUniverse:
    standard_sbs, standard_id, standard_sbs_id, burden = base.load_feature_matrices()
    sbs_counts = base.strip_prefix(standard_sbs, "SBS96__")
    id_counts = base.strip_prefix(standard_id, "ID83__")
    previous_uga = base.build_registered_uga_features(
        standard_sbs,
        standard_id,
        burden,
        sbs_model=base.LOCKED_SBS_MODEL,
        id_model=base.LOCKED_ID_MODEL,
        prefix="locked_mean",
    )["pooled"]
    sbs_basis, sbs_diag = base.build_uga_basis(
        sbs_counts.columns.astype(str).tolist(),
        base.LOCKED_SBS_MODEL,
        atlas=base.atlas_for_model(base.LOCKED_SBS_MODEL),
        modality="SBS",
    )
    id_basis, id_diag = base.build_uga_basis(id_counts.columns.astype(str).tolist(), base.LOCKED_ID_MODEL)
    pooled_counts = pd.concat([sbs_counts, id_counts], axis=1)
    pooled_basis = np.vstack([sbs_basis, id_basis])
    pooled_valid = np.concatenate(
        [
            sbs_diag["UGA_Encoded"].to_numpy(dtype=bool),
            id_diag["UGA_Encoded"].to_numpy(dtype=bool),
        ]
    )
    channel_labels = [f"SBS96:{col}" for col in sbs_counts.columns] + [f"ID83:{col}" for col in id_counts.columns]
    return KMEUniverse(
        benchmark="mc3_hrd_kmt2c",
        counts=pooled_counts,
        basis=pooled_basis,
        valid=pooled_valid,
        channel_labels=channel_labels,
        burden=burden,
        standard=base.FeatureSet(
            "standard_sbs96_id83",
            "mc3_hrd_kmt2c",
            standard_sbs_id,
            "Retained SBS96+ID83 channel fractions with retained burden covariates.",
            str(base.FEATURE_DIR / "features_standard_sbs96_id83.csv.gz"),
        ),
        previous=base.FeatureSet(
            "previous_uga_mean_pooled",
            "mc3_hrd_kmt2c",
            previous_uga,
            "Locked pooled UGA mean vector from SBS96 and ID83 channel distributions plus retained burden covariates.",
            "build_uga_basis + project_counts_to_uga using master_spec_sbs_dbs_d10_dp5 and id83_payload_only_d10_dp5",
        ),
    )


def build_kucab_universe() -> KMEUniverse:
    metadata, sbs, dbs, id_counts, _mapped = base.kucab.load_raw_counts()
    use = base.kucab.eligible_metadata(metadata)
    sbs = sbs.loc[use.index]
    dbs = dbs.loc[use.index]
    id_counts = id_counts.loc[use.index]
    raw_counts = np.concatenate([sbs.to_numpy(), dbs.to_numpy(), id_counts.to_numpy()], axis=1).astype(np.int64)
    keep = np.ones(len(use), dtype=bool)
    standard_columns = [f"SBS:{c}" for c in sbs.columns] + [f"DBS:{c}" for c in dbs.columns] + [f"ID:{c}" for c in id_counts.columns]
    standard = base.kucab.standard_features(raw_counts, keep, use.index, standard_columns)
    variant = base.kucab.build_variant(
        "previous_uga_mean_unified",
        base.LOCKED_SBS_MODEL,
        base.LOCKED_ID_MODEL,
        "unweighted_frac",
        1.0,
        sbs.columns.astype(str).tolist(),
        dbs.columns.astype(str).tolist(),
        id_counts.columns.astype(str).tolist(),
    )
    previous_uga = base.kucab.uga_features(raw_counts, keep, use.index, len(sbs.columns), len(dbs.columns), variant)
    endpoint = pd.DataFrame(
        {
            "sample": use.index.astype(str),
            "damage_class": use["damage_class"].astype(str).to_numpy(),
            "agent_core": use["agent_core"].astype(str).to_numpy(),
        }
    ).set_index("sample")
    return KMEUniverse(
        benchmark="kucab_damage_class",
        counts=pd.DataFrame(raw_counts, index=use.index, columns=standard_columns),
        basis=np.vstack([variant["sbsdbs_basis"], variant["id_basis"]]),
        valid=np.concatenate([variant["sbsdbs_valid"], variant["id_valid"]]),
        channel_labels=standard_columns,
        burden=None,
        standard=base.FeatureSet(
            "standard_sbs96_dbs78_id83",
            "kucab_damage_class",
            standard,
            "Original-data normalized SBS96+DBS78+ID83 channel distribution.",
            str(base.ACTIVE_ROOT / "data" / "raw"),
        ),
        previous=base.FeatureSet(
            "previous_uga_mean_unified",
            "kucab_damage_class",
            previous_uga,
            "Locked unified UGA mean-vector representation for SBS/DBS and ID83 channels with mutation-type fractions.",
            "run_locked_kucab_low_burden_benchmark.py::uga_features",
        ),
        endpoint_frame=endpoint,
    )


def build_tuned_feature_set(
    universe: KMEUniverse,
    *,
    representation: str,
    kernel_width_multiplier: float,
    n_landmarks: int,
) -> tuple[base.FeatureSet, pd.DataFrame, dict[str, float | int]]:
    kme, diagnostics, metadata = tuned_rbf_kme_features(
        universe.counts,
        universe.basis,
        universe.valid,
        prefix=representation,
        channel_labels=universe.channel_labels,
        kernel_width_multiplier=kernel_width_multiplier,
        n_landmarks=n_landmarks,
    )
    frame = pd.concat([universe.burden, kme], axis=1).fillna(0.0) if universe.burden is not None else kme
    diagnostics["benchmark"] = universe.benchmark
    diagnostics["representation"] = representation
    return (
        base.FeatureSet(
            representation,
            universe.benchmark,
            frame,
            (
                "HRD33-tuned UGA-RBF KME with frozen "
                f"kernel_width_multiplier={kernel_width_multiplier} and n_landmarks={n_landmarks}."
            ),
            "HRD33 grid-selected KME parameters",
            kernel_sigma=float(metadata["kernel_sigma"]),
            n_encoded_channels=int(metadata["n_encoded_channels"]),
            n_landmarks=int(metadata["actual_n_landmarks"]),
        ),
        diagnostics,
        metadata,
    )


def evaluate_with_learner(
    learner: str,
    endpoint: base.Endpoint,
    feature_set: base.FeatureSet,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    if learner == "linear":
        return base.evaluate_endpoint(endpoint, feature_set)
    if learner == "xgboost":
        return xgb_runner.evaluate_endpoint(endpoint, feature_set)
    raise ValueError(f"Unsupported learner: {learner}")


def evaluate_kucab_with_learner(
    learner: str,
    feature_set: base.FeatureSet,
    endpoint: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    if learner == "linear":
        return base.evaluate_kucab(feature_set, endpoint)
    if learner == "xgboost":
        return xgb_runner.evaluate_kucab(feature_set, endpoint)
    raise ValueError(f"Unsupported learner: {learner}")


def fold_signature(predictions: pd.DataFrame) -> tuple[tuple[str, int], ...]:
    pairs = predictions.loc[:, ["sample", "fold"]].copy()
    pairs["sample"] = pairs["sample"].astype(str)
    pairs = pairs.sort_values("sample")
    return tuple((row.sample, int(row.fold)) for row in pairs.itertuples(index=False))


def run_hrd33_tuning(mc3: KMEUniverse, endpoints: list[base.Endpoint]) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = next(endpoint for endpoint in endpoints if endpoint.name == PRIMARY_ENDPOINT)
    rows = []
    signatures: dict[str, tuple[tuple[str, int], ...]] = {}
    for learner in ["linear", "xgboost"]:
        reference_signature = None
        for multiplier in KERNEL_WIDTH_MULTIPLIERS:
            for n_landmarks in LANDMARK_COUNTS:
                feature_set, _diag, metadata = build_tuned_feature_set(
                    mc3,
                    representation=f"{learner}_grid_kme_m{str(multiplier).replace('.', 'p')}_l{n_landmarks}",
                    kernel_width_multiplier=multiplier,
                    n_landmarks=n_landmarks,
                )
                result, predictions, _probabilities = evaluate_with_learner(learner, primary, feature_set)
                sig = fold_signature(predictions)
                if reference_signature is None:
                    reference_signature = sig
                signatures[learner] = reference_signature
                rows.append(
                    {
                        "learner": learner,
                        "primary_endpoint": PRIMARY_ENDPOINT,
                        "metric": result["metric"],
                        "score": float(result["score"]),
                        "kernel_width_multiplier": float(multiplier),
                        "median_pairwise_distance": float(metadata["median_pairwise_distance"]),
                        "kernel_sigma": float(metadata["kernel_sigma"]),
                        "requested_n_landmarks": int(n_landmarks),
                        "actual_n_landmarks": int(metadata["actual_n_landmarks"]),
                        "n_features": int(feature_set.frame.shape[1]),
                        "n_samples": int(result["n_samples"]),
                        "fold_signature_matches_reference": bool(sig == reference_signature),
                    }
                )
    tuning = pd.DataFrame(rows)
    selected_rows = []
    for learner, group in tuning.groupby("learner", dropna=False):
        ranked = group.assign(
            tie_distance_to_unit=lambda x: (x["kernel_width_multiplier"] - 1.0).abs(),
        ).sort_values(
            ["score", "actual_n_landmarks", "tie_distance_to_unit", "kernel_width_multiplier"],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        best = ranked.iloc[0].to_dict()
        best["selection_rule"] = "max score; tie smaller landmarks; tie multiplier closest to 1.0; tie smaller multiplier"
        selected_rows.append(best)
    selected = pd.DataFrame(selected_rows).drop(columns=["tie_distance_to_unit"], errors="ignore")
    return tuning, selected


def add_deltas(results: pd.DataFrame, tuned_name: str) -> pd.DataFrame:
    out = results.copy()
    key_cols = ["benchmark", "endpoint", "family", "task"]
    standard_names = {
        "mc3_hrd_kmt2c": "standard_sbs96_id83",
        "kucab_damage_class": "standard_sbs96_dbs78_id83",
    }
    previous_names = {
        "mc3_hrd_kmt2c": "previous_uga_mean_pooled",
        "kucab_damage_class": "previous_uga_mean_unified",
    }
    out["delta_vs_standard"] = math.nan
    out["delta_vs_previous_uga_mean"] = math.nan
    for _, group in out.groupby(key_cols, dropna=False):
        benchmark = str(group["benchmark"].iloc[0])
        standard_score = group.loc[group["representation"] == standard_names[benchmark], "score"]
        previous_score = group.loc[group["representation"] == previous_names[benchmark], "score"]
        if standard_score.empty or previous_score.empty:
            continue
        out.loc[group.index, "delta_vs_standard"] = group["score"] - float(standard_score.iloc[0])
        out.loc[group.index, "delta_vs_previous_uga_mean"] = group["score"] - float(previous_score.iloc[0])
    out["is_tuned_kme"] = out["representation"].eq(tuned_name)
    return out


def run_full_panel(
    learner: str,
    mc3: KMEUniverse,
    kucab: KMEUniverse,
    endpoints: list[base.Endpoint],
    selected: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tuned_name = f"{learner}_tuned_kme"
    multiplier = float(selected["kernel_width_multiplier"])
    n_landmarks = int(selected["requested_n_landmarks"])
    mc3_tuned, mc3_diag, _mc3_meta = build_tuned_feature_set(
        mc3,
        representation=tuned_name,
        kernel_width_multiplier=multiplier,
        n_landmarks=n_landmarks,
    )
    kucab_tuned, kucab_diag, _kucab_meta = build_tuned_feature_set(
        kucab,
        representation=tuned_name,
        kernel_width_multiplier=multiplier,
        n_landmarks=n_landmarks,
    )
    result_rows = []
    prediction_frames = []
    probability_frames = []
    for endpoint in endpoints:
        for feature_set in [mc3.standard, mc3.previous, mc3_tuned]:
            result, predictions, probabilities = evaluate_with_learner(learner, endpoint, feature_set)
            result_rows.append(result)
            prediction_frames.append(predictions)
            if not probabilities.empty:
                probability_frames.append(probabilities)
    for feature_set in [kucab.standard, kucab.previous, kucab_tuned]:
        result, predictions, probabilities = evaluate_kucab_with_learner(learner, feature_set, kucab.endpoint_frame)
        result_rows.append(result)
        prediction_frames.append(predictions)
        if not probabilities.empty:
            probability_frames.append(probabilities)
    results = add_deltas(pd.DataFrame(result_rows), tuned_name)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    probabilities = pd.concat(probability_frames, ignore_index=True)
    diagnostics = pd.concat([mc3_diag, kucab_diag], ignore_index=True)
    return results, predictions, probabilities, diagnostics


def short_wide(results: pd.DataFrame, learner: str) -> pd.DataFrame:
    tuned_name = f"{learner}_tuned_kme"
    label = f"{learner}_tuned_kme"
    name_map = {
        "standard_sbs96_id83": "standard",
        "standard_sbs96_dbs78_id83": "standard",
        "previous_uga_mean_pooled": "previous_uga_mean",
        "previous_uga_mean_unified": "previous_uga_mean",
        tuned_name: label,
    }
    df = results.copy()
    df["rep_short"] = df["representation"].map(name_map)
    wide = (
        df.pivot_table(
            index=["benchmark", "family", "endpoint", "metric"],
            columns="rep_short",
            values="score",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    wide[f"{learner}_standard"] = wide["standard"]
    wide[f"{learner}_previous_uga_mean"] = wide["previous_uga_mean"]
    wide[f"{learner}_tuned_kme"] = wide[label]
    wide[f"{learner}_tuned_kme_minus_standard"] = wide[label] - wide["standard"]
    wide[f"{learner}_tuned_kme_minus_previous_uga_mean"] = wide[label] - wide["previous_uga_mean"]
    return wide[
        [
            "benchmark",
            "family",
            "endpoint",
            "metric",
            f"{learner}_standard",
            f"{learner}_previous_uga_mean",
            f"{learner}_tuned_kme",
            f"{learner}_tuned_kme_minus_standard",
            f"{learner}_tuned_kme_minus_previous_uga_mean",
        ]
    ]


def sign(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def build_tuned_sensitivity(linear: pd.DataFrame, xgboost: pd.DataFrame) -> pd.DataFrame:
    out = short_wide(linear, "linear").merge(
        short_wide(xgboost, "xgboost"),
        on=["benchmark", "family", "endpoint", "metric"],
        how="inner",
    )
    for comparison in ["tuned_kme_minus_standard", "tuned_kme_minus_previous_uga_mean"]:
        out[f"sign_agree_{comparison}"] = [
            sign(lv) == sign(xv)
            for lv, xv in zip(out[f"linear_{comparison}"], out[f"xgboost_{comparison}"], strict=False)
        ]
    return out


def validate_results(
    tuning: pd.DataFrame,
    selected: pd.DataFrame,
    result_sets: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> dict[str, object]:
    validation: dict[str, object] = {
        "tuning_rows": int(len(tuning)),
        "expected_tuning_rows": 40,
        "tuning_only_primary_endpoint": bool(set(tuning["primary_endpoint"]) == {PRIMARY_ENDPOINT}),
        "tuning_fold_signatures_match": bool(tuning["fold_signature_matches_reference"].all()),
        "selected_learners": sorted(selected["learner"].astype(str).tolist()),
    }
    selected_recomputed = []
    for learner, group in tuning.groupby("learner", dropna=False):
        best = group.assign(
            tie_distance_to_unit=lambda x: (x["kernel_width_multiplier"] - 1.0).abs(),
        ).sort_values(
            ["score", "actual_n_landmarks", "tie_distance_to_unit", "kernel_width_multiplier"],
            ascending=[False, True, True, True],
            kind="mergesort",
        ).iloc[0]
        chosen = selected[selected["learner"] == learner].iloc[0]
        selected_recomputed.append(
            bool(
                math.isclose(float(best["score"]), float(chosen["score"]))
                and int(best["actual_n_landmarks"]) == int(chosen["actual_n_landmarks"])
                and math.isclose(float(best["kernel_width_multiplier"]), float(chosen["kernel_width_multiplier"]))
            )
        )
    validation["selected_params_match_rule"] = bool(all(selected_recomputed))
    for learner, (results, predictions, probabilities) in result_sets.items():
        checked = xgb_runner.validate_outputs(results, predictions, probabilities)
        validation[f"{learner}_full_panel"] = checked
    return validation


def endpoint_table(results: pd.DataFrame, learner: str) -> str:
    wide = short_wide(results, learner)
    cols = [
        "benchmark",
        "family",
        "endpoint",
        "metric",
        f"{learner}_standard",
        f"{learner}_previous_uga_mean",
        f"{learner}_tuned_kme",
        f"{learner}_tuned_kme_minus_standard",
        f"{learner}_tuned_kme_minus_previous_uga_mean",
    ]
    return wide[cols].sort_values(["benchmark", "family", "endpoint"]).to_markdown(index=False, floatfmt=".6f")


def update_readme(
    tuning: pd.DataFrame,
    selected: pd.DataFrame,
    linear_results: pd.DataFrame,
    xgb_results: pd.DataFrame,
    sensitivity: pd.DataFrame,
    validation: dict[str, object],
    elapsed: float,
) -> None:
    readme = base.EXPERIMENT_ROOT / "README.md"
    marker = "\n## HRD33 KME Grid Tuning\n"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else "# UGA RBF Kernel Mean Linear Benchmark\n"
    head = existing.split(marker, 1)[0].rstrip()
    linear_tuned = linear_results[linear_results["representation"] == "linear_tuned_kme"]
    xgb_tuned = xgb_results[xgb_results["representation"] == "xgboost_tuned_kme"]
    lines = [
        "",
        "## HRD33 KME Grid Tuning",
        "",
        f"Executed UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Elapsed seconds: {elapsed:.2f}",
        "",
        "This is a single-endpoint exploratory KME parameter tuning pass. Only `hrd_binary_33` selected the KME parameters; the selected parameters were then frozen and applied once to the full panel.",
        "",
        "### Methodology",
        "",
        "- Primary tuning endpoint: `hrd_binary_33`.",
        "- Kernel-width grid: median UGA channel-distance multipliers `{0.25, 0.5, 1.0, 2.0, 4.0}`.",
        "- Landmark-count grid: `{25, 50, 100, 179}` KME landmarks.",
        "- Landmarks: deterministic farthest-point sampling over valid UGA channel coordinates, starting from the channel nearest the coordinate centroid.",
        "- Separate frozen KME parameters were selected for the fixed linear learner and the lightweight XGBoost learner.",
        "- Selection rule: highest HRD33 OOF AUROC; ties resolve by smaller landmark count, then multiplier closest to `1.0`, then smaller multiplier.",
        "",
        "### Selected Parameters",
        "",
        selected[
            [
                "learner",
                "score",
                "kernel_width_multiplier",
                "kernel_sigma",
                "requested_n_landmarks",
                "actual_n_landmarks",
                "n_features",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "### HRD33 Tuning Grid",
        "",
        tuning[
            [
                "learner",
                "kernel_width_multiplier",
                "requested_n_landmarks",
                "actual_n_landmarks",
                "kernel_sigma",
                "score",
                "fold_signature_matches_reference",
            ]
        ].sort_values(["learner", "score"], ascending=[True, False]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "### Frozen Linear Full-Panel Results",
        "",
        endpoint_table(linear_results, "linear"),
        "",
        "### Frozen XGBoost Full-Panel Results",
        "",
        endpoint_table(xgb_results, "xgboost"),
        "",
        "### Tuned KME Linear vs XGBoost Sensitivity",
        "",
        sensitivity.sort_values(["benchmark", "family", "endpoint"]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "### Headline",
        "",
        f"Frozen linear tuned KME beat Standard on {int((linear_tuned['delta_vs_standard'] > 0).sum())}/{len(linear_tuned)} endpoints and previous UGA mean on {int((linear_tuned['delta_vs_previous_uga_mean'] > 0).sum())}/{len(linear_tuned)}.",
        f"Frozen XGBoost tuned KME beat Standard on {int((xgb_tuned['delta_vs_standard'] > 0).sum())}/{len(xgb_tuned)} endpoints and previous UGA mean on {int((xgb_tuned['delta_vs_previous_uga_mean'] > 0).sum())}/{len(xgb_tuned)}.",
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
    mc3 = build_mc3_universe()
    kucab = build_kucab_universe()
    endpoints = base.load_hrd_endpoints() + base.load_mc3_clinical_endpoints() + [base.load_kmt2c_endpoint(mc3.standard.frame.index)]

    tuning, selected = run_hrd33_tuning(mc3, endpoints)
    linear_selected = selected[selected["learner"] == "linear"].iloc[0]
    xgb_selected = selected[selected["learner"] == "xgboost"].iloc[0]

    linear_results, linear_predictions, linear_probabilities, linear_diag = run_full_panel(
        "linear",
        mc3,
        kucab,
        endpoints,
        linear_selected,
    )
    xgb_results, xgb_predictions, xgb_probabilities, xgb_diag = run_full_panel(
        "xgboost",
        mc3,
        kucab,
        endpoints,
        xgb_selected,
    )
    sensitivity = build_tuned_sensitivity(linear_results, xgb_results)
    validation = validate_results(
        tuning,
        selected,
        {
            "linear": (linear_results, linear_predictions, linear_probabilities),
            "xgboost": (xgb_results, xgb_predictions, xgb_probabilities),
        },
    )

    tuning.to_csv(base.DATA_DIR / "kme_grid_hrd33_tuning_results.csv", index=False)
    selected.to_csv(base.DATA_DIR / "kme_grid_selected_params.csv", index=False)
    linear_results.to_csv(base.DATA_DIR / "linear_tuned_kme_endpoint_results.csv", index=False)
    xgb_results.to_csv(base.DATA_DIR / "xgboost_tuned_kme_endpoint_results.csv", index=False)
    linear_predictions.to_csv(base.DATA_DIR / "linear_tuned_kme_oof_predictions.csv.gz", index=False, compression="gzip")
    xgb_predictions.to_csv(base.DATA_DIR / "xgboost_tuned_kme_oof_predictions.csv.gz", index=False, compression="gzip")
    linear_probabilities.to_csv(base.DATA_DIR / "linear_tuned_kme_oof_probabilities_long.csv.gz", index=False, compression="gzip")
    xgb_probabilities.to_csv(base.DATA_DIR / "xgboost_tuned_kme_oof_probabilities_long.csv.gz", index=False, compression="gzip")
    pd.concat([linear_diag, xgb_diag], ignore_index=True).to_csv(base.DATA_DIR / "tuned_kme_basis_diagnostics.csv", index=False)
    sensitivity.to_csv(base.TABLE_DIR / "tuned_kme_model_sensitivity_summary.csv", index=False)

    metadata = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - start,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "kernel_width_multipliers": KERNEL_WIDTH_MULTIPLIERS,
        "landmark_counts": LANDMARK_COUNTS,
        "dyld_library_path": os.environ.get("DYLD_LIBRARY_PATH", ""),
        "validation": validation,
    }
    (base.DATA_DIR / "kme_grid_tuned_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    update_readme(tuning, selected, linear_results, xgb_results, sensitivity, validation, time.time() - start)

    payload = {
        "experiment_root": str(base.EXPERIMENT_ROOT),
        "elapsed_seconds": round(time.time() - start, 2),
        "selected_params": selected[
            [
                "learner",
                "score",
                "kernel_width_multiplier",
                "requested_n_landmarks",
                "actual_n_landmarks",
                "n_features",
            ]
        ].to_dict("records"),
        "linear_tuned_kme_wins_vs_standard": int(
            (linear_results.loc[linear_results["representation"] == "linear_tuned_kme", "delta_vs_standard"] > 0).sum()
        ),
        "xgboost_tuned_kme_wins_vs_standard": int(
            (xgb_results.loc[xgb_results["representation"] == "xgboost_tuned_kme", "delta_vs_standard"] > 0).sum()
        ),
        "validation": validation,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
