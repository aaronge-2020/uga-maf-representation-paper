#!/usr/bin/env python3
"""Second-pass rapid screen for distribution-preserving UGA features."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
DATA_DIR = EXPERIMENT_ROOT / "data"
TABLE_DIR = EXPERIMENT_ROOT / "tables"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_fast_defensible_signal_recovery_benchmark as base  # noqa: E402


RANDOM_SEED = 20260515


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def normalized_counts(counts: pd.DataFrame) -> np.ndarray:
    arr = counts.fillna(0.0).to_numpy(dtype=np.float64)
    row_sum = arr.sum(axis=1, keepdims=True)
    return np.divide(arr, row_sum, out=np.zeros_like(arr), where=row_sum > 1e-15)


def standardize_basis(basis: np.ndarray) -> np.ndarray:
    x = np.asarray(basis, dtype=np.float64)
    scale = x.std(axis=0, keepdims=True)
    return (x - x.mean(axis=0, keepdims=True)) / np.where(scale > 1e-12, scale, 1.0)


def squared_distances(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    y = x if y is None else y
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)


def median_sigma(z: np.ndarray, scale: float) -> float:
    d = np.sqrt(squared_distances(z))
    vals = d[np.triu_indices_from(d, k=1)]
    vals = vals[np.isfinite(vals) & (vals > 1e-12)]
    if len(vals) == 0:
        return float(scale)
    return float(np.median(vals) * scale)


def kernel_eigen_transform(basis: np.ndarray, *, k: int, sigma_scale: float) -> np.ndarray:
    z = standardize_basis(basis)
    sigma = median_sigma(z, sigma_scale)
    d2 = squared_distances(z)
    kernel = np.exp(-d2 / (2.0 * sigma * sigma))
    row_sum = kernel.sum(axis=1, keepdims=True)
    kernel = np.divide(kernel, row_sum, out=np.zeros_like(kernel), where=row_sum > 1e-15)
    vals, vecs = np.linalg.eigh((kernel + kernel.T) / 2.0)
    order = np.argsort(vals)[::-1]
    selected = order[: min(int(k), len(order))]
    transform = vecs[:, selected] * np.sqrt(np.maximum(vals[selected], 0.0))[None, :]
    return transform.astype(np.float64)


def farthest_landmarks(z: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), z.shape[0])
    selected = [int(np.argmin(np.sum((z - z.mean(axis=0, keepdims=True)) ** 2, axis=1)))]
    min_dist = squared_distances(z, z[selected]).reshape(-1)
    while len(selected) < k:
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, squared_distances(z, z[[nxt]]).reshape(-1))
    return np.asarray(selected, dtype=int)


def landmark_transform(basis: np.ndarray, *, k: int, sigma_scale: float) -> np.ndarray:
    z = standardize_basis(basis)
    idx = farthest_landmarks(z, k)
    landmarks = z[idx]
    sigma = median_sigma(z, sigma_scale)
    kernel = np.exp(-squared_distances(z, landmarks) / (2.0 * sigma * sigma))
    row_sum = kernel.sum(axis=1, keepdims=True)
    return np.divide(kernel, row_sum, out=np.zeros_like(kernel), where=row_sum > 1e-15)


def random_direction_moments(basis: np.ndarray, *, n_dirs: int, seed: int) -> np.ndarray:
    z = standardize_basis(basis)
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(z.shape[1], int(n_dirs)))
    w = w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-12)
    projected = z @ w
    return np.concatenate([projected, projected * projected, projected * projected * projected], axis=1)


def moment_features(counts: pd.DataFrame, basis: np.ndarray, prefix: str, *, include_third: bool) -> pd.DataFrame:
    weights = normalized_counts(counts)
    z = standardize_basis(basis)
    mean = weights @ z
    second = weights @ (z * z)
    var = np.maximum(second - mean * mean, 0.0)
    frames = [
        pd.DataFrame(mean, index=counts.index, columns=[f"{prefix}_mean_{i:03d}" for i in range(z.shape[1])]),
        pd.DataFrame(var, index=counts.index, columns=[f"{prefix}_var_{i:03d}" for i in range(z.shape[1])]),
    ]
    if include_third:
        third_raw = weights @ (z * z * z)
        third = third_raw - 3.0 * mean * second + 2.0 * mean * mean * mean
        frames.append(pd.DataFrame(third, index=counts.index, columns=[f"{prefix}_third_{i:03d}" for i in range(z.shape[1])]))
    return pd.concat(frames, axis=1).fillna(0.0)


def projected_features(counts: pd.DataFrame, transform: np.ndarray, prefix: str) -> pd.DataFrame:
    values = normalized_counts(counts) @ transform
    return pd.DataFrame(values, index=counts.index, columns=[f"{prefix}_{i:03d}" for i in range(values.shape[1])]).fillna(0.0)


def build_basis_pair(standard_sbs: pd.DataFrame, standard_id: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    sbs_counts = base.strip_prefix(standard_sbs, "SBS96__")
    id_counts = base.strip_prefix(standard_id, "ID83__")
    sbs_spec = base.get_uga_model(base.LOCKED_SBS_MODEL)
    atlas = base.load_context_atlas(base.CONTEXT_ATLAS, sbs_spec.d_context)
    sbs_basis, sbs_diag = base.build_uga_basis(sbs_counts.columns.astype(str).tolist(), base.LOCKED_SBS_MODEL, atlas=atlas, modality="SBS")
    id_basis, id_diag = base.build_uga_basis(id_counts.columns.astype(str).tolist(), "id83_token_pair_d10_dp5")
    if not bool(sbs_diag["UGA_Encoded"].all()):
        raise RuntimeError("SBS basis contains unencoded channels")
    if not bool(id_diag["UGA_Encoded"].all()):
        raise RuntimeError("ID basis contains unencoded channels")
    return sbs_counts, id_counts, sbs_basis, id_basis


def build_second_pass_candidates(
    standard_sbs: pd.DataFrame,
    standard_id: pd.DataFrame,
    burden: pd.DataFrame,
) -> list[base.Candidate]:
    sbs_counts, id_counts, sbs_basis, id_basis = build_basis_pair(standard_sbs, standard_id)
    candidates: list[base.Candidate] = []

    for k in (16, 32, 48, 64):
        sbs_t = kernel_eigen_transform(sbs_basis, k=k, sigma_scale=0.75)
        id_t = kernel_eigen_transform(id_basis, k=min(k, 48), sigma_scale=0.75)
        frame = pd.concat(
            [
                burden,
                projected_features(sbs_counts, sbs_t, f"sbs_graph_k{k}"),
                projected_features(id_counts, id_t, f"id_graph_k{min(k, 48)}"),
            ],
            axis=1,
        ).fillna(0.0)
        candidates.append(
            base.Candidate(
                f"graph_spectral_uga_k{k}",
                f"UGA graph spectral projection with {k} SBS components and {min(k, 48)} ID components.",
                frame,
                True,
                False,
                "distributional_uga",
            )
        )

    for k in (16, 32, 48):
        for sigma_scale in (0.35, 0.75, 1.25):
            sbs_t = landmark_transform(sbs_basis, k=k, sigma_scale=sigma_scale)
            id_t = landmark_transform(id_basis, k=min(k, 40), sigma_scale=sigma_scale)
            frame = pd.concat(
                [
                    burden,
                    projected_features(sbs_counts, sbs_t, f"sbs_landmark_k{k}_s{sigma_scale:g}"),
                    projected_features(id_counts, id_t, f"id_landmark_k{min(k, 40)}_s{sigma_scale:g}"),
                ],
                axis=1,
            ).fillna(0.0)
            candidates.append(
                base.Candidate(
                    f"soft_landmark_uga_k{k}_sigma{sigma_scale:g}",
                    f"Soft UGA landmark bins with {k} SBS landmarks, {min(k, 40)} ID landmarks, and sigma scale {sigma_scale:g}.",
                    frame,
                    True,
                    False,
                    "distributional_uga",
                )
            )

    for include_third in (False, True):
        frame = pd.concat(
            [
                burden,
                moment_features(sbs_counts, sbs_basis, "sbs_moment", include_third=include_third),
                moment_features(id_counts, id_basis, "id_moment", include_third=include_third),
            ],
            axis=1,
        ).fillna(0.0)
        name = "uga_moments_mean_var_third" if include_third else "uga_moments_mean_var"
        candidates.append(
            base.Candidate(
                name,
                "Weighted UGA coordinate moments by modality.",
                frame,
                True,
                False,
                "distributional_uga",
            )
        )

    for n_dirs in (32, 64, 96):
        sbs_t = random_direction_moments(sbs_basis, n_dirs=n_dirs, seed=RANDOM_SEED + n_dirs)
        id_t = random_direction_moments(id_basis, n_dirs=min(n_dirs, 64), seed=RANDOM_SEED + 1000 + n_dirs)
        frame = pd.concat(
            [
                burden,
                projected_features(sbs_counts, sbs_t, f"sbs_randmom_{n_dirs}"),
                projected_features(id_counts, id_t, f"id_randmom_{min(n_dirs, 64)}"),
            ],
            axis=1,
        ).fillna(0.0)
        candidates.append(
            base.Candidate(
                f"random_direction_moment_uga_d{n_dirs}",
                f"Random-direction first, second, and third UGA moments with {n_dirs} SBS directions.",
                frame,
                True,
                False,
                "distributional_uga",
            )
        )

    return candidates


def write_html_table(df: pd.DataFrame, path: Path, title: str, footnote: str) -> None:
    base.write_html_table(df, path, title, footnote)


def update_readme(metadata: dict[str, object], leaderboard: pd.DataFrame, finalists: list[str]) -> None:
    readme_path = EXPERIMENT_ROOT / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# Fast Defensible UGA Signal-Recovery Benchmark\n"
    leader_cols = ["candidate", "mean_delta", "min_delta", "max_delta", "n_positive", "n_ge_0_03", "n_lt_minus_0_02"]
    view = leaderboard.loc[:, leader_cols].head(12).rename(
        columns={
            "candidate": "Candidate",
            "mean_delta": "Mean delta",
            "min_delta": "Smallest delta",
            "max_delta": "Largest delta",
            "n_positive": "Positive endpoints",
            "n_ge_0_03": "Endpoints >= +0.03",
            "n_lt_minus_0_02": "Endpoints < -0.02",
        }
    )
    finalist_text = ", ".join(finalists) if finalists else "None"
    section = [
        "",
        "## Second-Pass Distributional UGA Screen",
        f"Executed at {metadata['executed_at_utc']} with paired five-fold splits, {metadata['rapid_trees']} XGBoost trees, and runtime {float(metadata['elapsed_seconds']) / 60.0:.1f} minutes.",
        f"Finalists passing the promotion gate: {finalist_text}.",
        "",
        view.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    marker = "\n## Second-Pass Distributional UGA Screen\n"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip()
    readme_path.write_text(existing.rstrip() + "\n" + "\n".join(section), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapid-folds", type=int, default=5)
    parser.add_argument("--rapid-trees", type=int, default=40)
    parser.add_argument("--tree-method", default="gpu_hist")
    parser.add_argument("--early-stop-delta", type=float, default=-0.02)
    parser.add_argument("--confirm-folds", type=int, default=5)
    parser.add_argument("--confirm-repeats", type=int, default=3)
    parser.add_argument("--confirm-trees", type=int, default=250)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--max-finalists", type=int, default=2)
    parser.add_argument("--xgb-n-jobs", type=int, default=4)
    args = parser.parse_args()

    ensure_dirs()
    base.XGB_N_JOBS = int(args.xgb_n_jobs)
    start = time.perf_counter()
    standard_sbs, standard_id, standard_sbs_id, burden = base.load_feature_matrices()
    patients = burden.index.astype(str)
    endpoints = [*base.load_hrd_endpoints(), *base.load_mc3_clinical_endpoints(), base.load_kmt2c_endpoint(pd.Index(patients))]
    candidates = build_second_pass_candidates(standard_sbs, standard_id, burden)
    manifest = pd.DataFrame(
        [
            {
                "candidate": candidate.name,
                "candidate_family": candidate.family,
                "description": candidate.description,
                "n_features": int(candidate.features.shape[1]),
                "promotable": candidate.promotable,
                "uses_standard_one_hot": candidate.uses_standard_one_hot,
            }
            for candidate in candidates
        ]
    )
    manifest.to_csv(DATA_DIR / "second_pass_model_manifest.csv", index=False)
    screen, leaderboard, predictions = base.run_candidate_screen(
        candidates,
        endpoints,
        standard_sbs_id,
        folds=args.rapid_folds,
        n_estimators=args.rapid_trees,
        tree_method=args.tree_method,
        early_stop_delta=args.early_stop_delta,
    )
    screen.to_csv(DATA_DIR / "second_pass_distributional_uga_screen.csv", index=False)
    leaderboard.to_csv(DATA_DIR / "second_pass_distributional_uga_leaderboard.csv", index=False)
    predictions.to_csv(DATA_DIR / "second_pass_distributional_uga_oof_predictions.csv", index=False)

    finalists = [
        str(candidate)
        for candidate in leaderboard["candidate"].tolist()
        if base.candidate_passes_gate(screen, str(candidate))
    ][: int(args.max_finalists)]
    confirm_metrics, confirm_tests, confirm_predictions = base.run_confirmation(
        finalists,
        candidates,
        endpoints,
        standard_sbs_id,
        folds=args.confirm_folds,
        repeats=args.confirm_repeats,
        n_estimators=args.confirm_trees,
        tree_method=args.tree_method,
        bootstrap=args.bootstrap,
    )
    confirm_metrics.to_csv(DATA_DIR / "second_pass_focused_confirmation_metrics.csv", index=False)
    confirm_tests.to_csv(DATA_DIR / "second_pass_focused_confirmation_tests.csv", index=False)
    if not confirm_predictions.empty:
        confirm_predictions.to_csv(DATA_DIR / "second_pass_focused_confirmation_oof_predictions.csv", index=False)

    write_html_table(
        leaderboard,
        TABLE_DIR / "table5_second_pass_distributional_uga_leaderboard.html",
        "Table 5. Second-pass distributional UGA rapid screen.",
        "Candidates are pure UGA-derived replacement features evaluated with paired five-fold XGBoost. Screening results are exploratory.",
    )
    write_html_table(
        confirm_tests,
        TABLE_DIR / "table6_second_pass_focused_confirmation.html",
        "Table 6. Second-pass focused confirmation.",
        "Focused confirmation is run only for candidates passing the predeclared promotion gate.",
    )
    metadata = {
        "experiment": EXPERIMENT_ROOT.name,
        "second_pass": "distributional_uga",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "rapid_folds": int(args.rapid_folds),
        "rapid_trees": int(args.rapid_trees),
        "tree_method": str(args.tree_method),
        "early_stop_delta": float(args.early_stop_delta),
        "confirm_folds": int(args.confirm_folds),
        "confirm_repeats": int(args.confirm_repeats),
        "confirm_trees": int(args.confirm_trees),
        "bootstrap": int(args.bootstrap),
        "n_candidates": int(len(candidates)),
        "n_endpoints": int(len(endpoints)),
        "finalists": finalists,
    }
    (DATA_DIR / "second_pass_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    update_readme(metadata, leaderboard, finalists)
    print(json.dumps({"elapsed_seconds": round(metadata["elapsed_seconds"], 1), "finalists": finalists, "n_candidates": len(candidates)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
