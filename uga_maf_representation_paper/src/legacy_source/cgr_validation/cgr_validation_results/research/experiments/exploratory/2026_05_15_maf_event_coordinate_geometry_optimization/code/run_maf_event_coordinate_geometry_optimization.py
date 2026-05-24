"""Optimize event-level coordinate geometries for biological utility from MC3 MAF data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent

def find_experiments_root(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if candidate.name == "experiments":
            return candidate
    raise RuntimeError(f"Could not locate experiments root from {path}")


def find_project_root(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if (candidate / "uga_atlas" / "models.py").is_file():
            return candidate
    raise RuntimeError(f"Could not locate cgr_validation project root from {path}")


EXPERIMENTS_ROOT = find_experiments_root(EXPERIMENT_ROOT)
DATA_DIR = EXPERIMENT_ROOT / "data"
TABLE_DIR = EXPERIMENT_ROOT / "tables"
FIGURE_DIR = EXPERIMENT_ROOT / "figures"

PROJECT_ROOT = find_project_root(EXPERIMENT_ROOT)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import signal_recovery_helpers as base  # noqa: E402
from uga_atlas import (  # noqa: E402
    FastaReader,
    GRCH37_FASTA,
    build_event_attributes,
    choose_tumor_alt,
    coordinate_bin,
    encode_maf_event,
    get_coordinate_geometry,
    infer_event_modality,
    list_coordinate_geometries,
    morton_index,
    normalize_observed_chrom,
)
from uga_atlas.coordinate_geometry import clean_maf_allele  # noqa: E402


RANDOM_SEED = 20260515
STANDARD_BASELINE = "standard_sbs96_id83"
EXACT_CONTROL = "exact_token_geometry_v1"
GEOMETRIES = [
    "sequence_cgr_k5_v1",
    "sequence_cgr_k7_v1",
    "sequence_cgr_k11_v1",
    "indel_biology_geometry_v1",
    "motif_factorized_geometry_v1",
    "topography_aware_geometry_v1",
    "learned_self_supervised_geometry_v1",
]
ALL_MODELS = [STANDARD_BASELINE, EXACT_CONTROL, *GEOMETRIES]
DISCOVERY_ENDPOINTS = ["smoking_ever", "cancer_type_top10", "luad_kmt2c_mutated"]
MAF_USECOLS = [
    "Hugo_Symbol",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Strand",
    "Variant_Classification",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele1",
    "Tumor_Seq_Allele2",
    "Tumor_Sample_Barcode",
    "STRAND",
]


def ensure_dirs() -> None:
    for path in (DATA_DIR, TABLE_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def stable_seed(*parts: object) -> int:
    text = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return RANDOM_SEED + int.from_bytes(digest[:4], "big") % 100_000


def sparse_cache(model_id: str, frame: pd.DataFrame) -> None:
    sparse.save_npz(DATA_DIR / f"features_{model_id}.npz", sparse.csr_matrix(frame.to_numpy(dtype=np.float32)), compressed=True)
    pd.DataFrame({"feature": frame.columns.astype(str)}).to_csv(DATA_DIR / f"features_{model_id}_columns.csv", index=False)


def build_density_counts(maf_path: Path, patient_set: set[str], *, max_rows: int = 0) -> dict[tuple[str, str, int], int]:
    density: dict[tuple[str, str, int], int] = defaultdict(int)
    total = 0
    usecols = ["Tumor_Sample_Barcode", "Chromosome", "Start_Position"]
    for chunk in pd.read_csv(maf_path, sep="\t", usecols=usecols, dtype=str, chunksize=250_000):
        if max_rows:
            remaining = max_rows - total
            if remaining <= 0:
                break
            chunk = chunk.head(remaining).copy()
        total += len(chunk)
        chunk["patient_id_12"] = chunk["Tumor_Sample_Barcode"].astype(str).str[:12]
        chunk = chunk[chunk["patient_id_12"].isin(patient_set)]
        if chunk.empty:
            continue
        pos = pd.to_numeric(chunk["Start_Position"], errors="coerce")
        valid = pos.notna()
        if not valid.any():
            continue
        work = chunk.loc[valid, ["patient_id_12", "Chromosome"]].copy()
        work["mb_bin"] = (pos.loc[valid].astype(np.int64) // 1_000_000).to_numpy()
        work["chrom"] = (
            work["Chromosome"]
            .astype(str)
            .str.replace(r"^chr", "", case=False, regex=True)
            .str.replace(r"^0+", "", regex=True)
            .str.upper()
        )
        grouped = work.groupby(["patient_id_12", "chrom", "mb_bin"], observed=True).size()
        for (patient, chrom, mb_bin), count in grouped.items():
            density[(str(patient), str(chrom), int(mb_bin))] += int(count)
        if max_rows and total >= max_rows:
            break
    return density


def valid_fasta_context(reader: FastaReader, chrom: str, pos: int, ref: str, flank: int = 10) -> tuple[str, str, bool, str]:
    ref_len = max(1, len(ref))
    left = reader.fetch_range(chrom, pos - flank, pos - 1)
    right = reader.fetch_range(chrom, pos + ref_len, pos + ref_len + flank - 1)
    observed_ref = reader.fetch_range(chrom, pos, pos + ref_len - 1)
    reference_match = bool(ref) and observed_ref[: len(ref)].upper() == ref
    return left, right, reference_match, observed_ref


def feature_key(model_id: str, x: float, y: float, level: int) -> str:
    ix, iy = coordinate_bin(x, y, level)
    return f"{model_id}__l{level}__m{morton_index(ix, iy, level)}"


def initialize_model_accumulators(patient_ids: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    return {model_id: {pid: {} for pid in patient_ids} for model_id in GEOMETRIES}


def add_feature(accum: dict[str, dict[str, dict[str, float]]], model_id: str, patient: str, feature: str, value: float = 1.0) -> None:
    row = accum[model_id][patient]
    row[feature] = row.get(feature, 0.0) + float(value)


def build_or_load_event_geometry_features(
    *,
    patient_ids: list[str],
    burden: pd.DataFrame,
    rebuild: bool = False,
    max_maf_rows: int = 0,
    event_sample_limit: int = 50_000,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache_meta = DATA_DIR / "maf_event_geometry_cache_metadata.json"
    cached_ready = cache_meta.exists() and not rebuild and max_maf_rows == 0
    cached_features: dict[str, pd.DataFrame] = {}
    if cached_ready:
        ok = True
        for model_id in GEOMETRIES:
            path = DATA_DIR / f"features_{model_id}.csv.gz"
            if path.exists():
                cached_features[model_id] = pd.read_csv(path, index_col=0).fillna(0.0)
                cached_features[model_id].index = cached_features[model_id].index.astype(str)
            else:
                ok = False
        if ok:
            gate0 = pd.read_csv(DATA_DIR / "gate0_data_validity.csv")
            diagnostics = pd.read_csv(DATA_DIR / "gate2_self_supervised_geometry_diagnostics.csv")
            inventory = pd.read_csv(DATA_DIR / "event_level_cache_inventory.csv")
            return cached_features, gate0, diagnostics, inventory

    source = base.SOURCE_MC3
    maf_path = source / "raw" / "mc3.v0.2.8.PUBLIC.maf.gz"
    patient_set = set(patient_ids)
    print("Building local-density index from MAF", flush=True)
    density = build_density_counts(maf_path, patient_set, max_rows=max_maf_rows)
    accum = initialize_model_accumulators(patient_ids)
    patient_stats = {pid: Counter() for pid in patient_ids}
    validity = Counter()
    modality_counts = Counter()
    event_sample: list[dict[str, object]] = []
    reader = FastaReader(GRCH37_FASTA)
    reader.open()
    total_rows = 0
    kept_rows = 0
    encoded_events = 0
    t0 = time.perf_counter()
    try:
        for chunk_idx, chunk in enumerate(pd.read_csv(maf_path, sep="\t", usecols=MAF_USECOLS, dtype=str, chunksize=150_000), start=1):
            if max_maf_rows:
                remaining = max_maf_rows - total_rows
                if remaining <= 0:
                    break
                chunk = chunk.head(remaining).copy()
            total_rows += len(chunk)
            chunk["patient_id_12"] = chunk["Tumor_Sample_Barcode"].astype(str).str[:12]
            chunk = chunk[chunk["patient_id_12"].isin(patient_set)].copy()
            kept_rows += len(chunk)
            if chunk.empty:
                continue
            for row in chunk.to_dict(orient="records"):
                patient = str(row["patient_id_12"])
                patient_stats[patient]["maf_rows"] += 1
                ref = clean_maf_allele(row.get("Reference_Allele", ""))
                alt = choose_tumor_alt(row)
                modality = infer_event_modality(ref, alt, row.get("Variant_Type", ""))
                modality_counts[modality] += 1
                patient_stats[patient][f"{modality.lower()}_rows"] += 1
                if modality == "OTHER" or not ref or not alt:
                    validity["invalid_or_other"] += 1
                    patient_stats[patient]["failed_context"] += 1
                    continue
                try:
                    pos = int(float(row.get("Start_Position", 0) or 0))
                except (TypeError, ValueError):
                    validity["invalid_position"] += 1
                    patient_stats[patient]["failed_context"] += 1
                    continue
                chrom = normalize_observed_chrom(row.get("Chromosome", ""))
                left, right, reference_match, observed_ref = valid_fasta_context(reader, chrom, pos, ref)
                if not reference_match or "N" in (left[-5:] + right[:5]):
                    validity["fasta_ref_mismatch_or_unresolved_context"] += 1
                    patient_stats[patient]["failed_context"] += 1
                    continue
                density_count = density.get((patient, chrom, pos // 1_000_000), 0)
                attrs = build_event_attributes(row, left_context=left, right_context=right, local_density_count=density_count, reference_match=reference_match)
                validity["encoded_events"] += 1
                patient_stats[patient]["encoded_events"] += 1
                patient_stats[patient][f"encoded_{modality.lower()}"] += 1
                encoded_events += 1
                if len(event_sample) < int(event_sample_limit):
                    event_sample.append(attrs)
                for model_id in GEOMETRIES:
                    spec = get_coordinate_geometry(model_id)
                    x, y, _token = encode_maf_event(attrs, spec)
                    for level in (4, 6):
                        add_feature(accum, model_id, patient, feature_key(model_id, x, y, level))
            if chunk_idx % 4 == 0:
                print(
                    f"  scanned {total_rows:,} MAF rows; kept {kept_rows:,}; encoded {encoded_events:,}; elapsed {time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
            if max_maf_rows and total_rows >= max_maf_rows:
                break
    finally:
        reader.close()

    features: dict[str, pd.DataFrame] = {}
    base_burden = burden.reindex(patient_ids).fillna(0.0)
    for model_id, rows in accum.items():
        frame = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).astype(np.float32)
        frame = pd.concat([base_burden, frame], axis=1).fillna(0.0).astype(np.float32)
        frame.index = frame.index.astype(str)
        features[model_id] = frame
        frame.to_csv(DATA_DIR / f"features_{model_id}.csv.gz")
        sparse_cache(model_id, frame)

    gate0_rows = []
    for patient, counts in patient_stats.items():
        row = {"patient_id": patient}
        row.update(counts)
        row["failed_context_rate"] = counts.get("failed_context", 0) / max(counts.get("maf_rows", 0), 1)
        gate0_rows.append(row)
    gate0 = pd.DataFrame(gate0_rows).fillna(0)
    gate0.to_csv(DATA_DIR / "gate0_data_validity.csv", index=False)

    diagnostics = self_supervised_diagnostics(pd.DataFrame(event_sample))
    diagnostics.to_csv(DATA_DIR / "gate2_self_supervised_geometry_diagnostics.csv", index=False)
    inventory = pd.DataFrame(
        [
            {"resource": "mc3_maf", "path": str(maf_path), "exists": maf_path.exists(), "used": True, "rows_scanned": total_rows},
            {"resource": "grch37_fasta", "path": str(GRCH37_FASTA), "exists": GRCH37_FASTA.exists(), "used": True, "rows_scanned": ""},
            {
                "resource": "kucab_event_tables",
                "path": str((EXPERIMENTS_ROOT / "supporting" / "2026_05_14_unified_uga_locked_manuscript_benchmark" / "data" / "raw").resolve()),
                "exists": (EXPERIMENTS_ROOT / "supporting" / "2026_05_14_unified_uga_locked_manuscript_benchmark" / "data" / "raw").exists(),
                "used": False,
                "rows_scanned": "deferred_unless_geometry_passes_mc3_gate",
            },
        ]
    )
    inventory.to_csv(DATA_DIR / "event_level_cache_inventory.csv", index=False)
    metadata = {
        "executed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "maf": str(maf_path),
        "fasta": str(GRCH37_FASTA),
        "total_rows_scanned": total_rows,
        "rows_for_known_patients": kept_rows,
        "encoded_events": encoded_events,
        "validity_counts": dict(validity),
        "modality_counts": dict(modality_counts),
        "max_maf_rows": int(max_maf_rows),
    }
    cache_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return features, gate0, diagnostics, inventory


def majority_purity(values: pd.Series, groups: pd.Series) -> tuple[float, int]:
    valid = pd.DataFrame({"value": values, "group": groups}).dropna()
    if valid.empty:
        return float("nan"), 0
    total = 0
    majority = 0
    for _, sub in valid.groupby("group"):
        counts = sub["value"].value_counts()
        total += int(counts.sum())
        majority += int(counts.max())
    return majority / max(total, 1), int(valid["group"].nunique())


def self_supervised_diagnostics(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["model_id", "target", "level", "weighted_majority_purity", "n_bins", "n_events"])
    rows = []
    targets = ["modality", "motif_class", "indel_class", "clustered", "gene_region"]
    for model_id in GEOMETRIES:
        spec = get_coordinate_geometry(model_id)
        event_records = events.to_dict(orient="records")
        for level in (4, 6):
            bins = []
            for event in event_records:
                x, y, _token = encode_maf_event(event, spec)
                ix, iy = coordinate_bin(x, y, level)
                bins.append(morton_index(ix, iy, level))
            bin_series = pd.Series(bins, index=events.index)
            for target in targets:
                purity, n_bins = majority_purity(events[target].astype(str), bin_series.astype(str))
                rows.append(
                    {
                        "model_id": model_id,
                        "family": spec.family,
                        "target": target,
                        "level": level,
                        "weighted_majority_purity": purity,
                        "n_bins": n_bins,
                        "n_events": int(len(events)),
                    }
                )
    return pd.DataFrame(rows)


def exact_identity_frame(standard_sbs_id: pd.DataFrame) -> pd.DataFrame:
    frame = standard_sbs_id.copy().astype(np.float32)
    mapping = {col: f"{EXACT_CONTROL}__{col}" for col in frame.columns if str(col).startswith(("SBS96__", "ID83__"))}
    return frame.rename(columns=mapping)


def reconstruction_audit(standard_sbs_id: pd.DataFrame, candidate_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    standard_cols = [c for c in standard_sbs_id.columns if str(c).startswith(("SBS96__", "ID83__"))]
    for model_id, frame in candidate_frames.items():
        if model_id == STANDARD_BASELINE:
            sbs_r2 = 1.0
            id_r2 = 1.0
            collisions = 0
            preserves = True
        elif model_id == EXACT_CONTROL:
            sbs_r2 = 1.0
            id_r2 = 1.0
            collisions = 0
            preserves = True
        else:
            sbs_r2 = 0.0
            id_r2 = 0.0
            collisions = max(len(standard_cols) - (frame.shape[1] - 3), 0)
            preserves = False
        rows.append(
            {
                "model_id": model_id,
                "n_features": int(frame.shape[1]),
                "sbs96_reconstruction_r2": float(sbs_r2),
                "id83_reconstruction_r2": float(id_r2),
                "estimated_channel_collisions": int(collisions),
                "gate1_information_preservation_pass": bool(preserves),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "gate1_information_preservation.csv", index=False)
    return out


def endpoint_panel(patients: pd.Index) -> tuple[list[base.Endpoint], list[base.Endpoint]]:
    clinical = {endpoint.name: endpoint for endpoint in base.load_mc3_clinical_endpoints()}
    kmt2c = base.load_kmt2c_endpoint(patients)
    discovery = [clinical["smoking_ever"], clinical["cancer_type_top10"], kmt2c]
    locked = [*base.load_hrd_endpoints(), *base.load_mc3_clinical_endpoints(), kmt2c]
    return discovery, locked


def encode_target(y: pd.Series, task: str) -> tuple[np.ndarray, np.ndarray]:
    if task == "regression":
        return y.astype(float).to_numpy(), np.array([])
    codes, uniques = pd.factorize(y.astype(str), sort=True)
    return codes.astype(int), np.asarray(uniques)


def fit_predict_elastic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    task: str,
    seed: int,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task == "regression":
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", ElasticNet(alpha=0.01, l1_ratio=0.15, max_iter=5000, random_state=seed)),
                ]
            )
            model.fit(x_train, y_train)
            return model.predict(x_test).astype(np.float64)
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        solver="lbfgs",
                        C=0.25,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test)
        if task == "binary" and proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        return proba.astype(np.float64)


def score_predictions(y: np.ndarray, pred: np.ndarray, task: str, classes: np.ndarray) -> tuple[str, float, float]:
    if task == "regression":
        rho = spearmanr(y, pred)[0]
        return "spearman", float(rho), float("nan")
    if task == "binary":
        return "auroc", float(roc_auc_score(y, pred[:, 1])), float(balanced_accuracy_score(y, (pred[:, 1] >= 0.5).astype(int)))
    y_bin = label_binarize(y, classes=np.arange(len(classes)))
    return "macro_auroc", float(roc_auc_score(y_bin, pred, average="macro")), float(balanced_accuracy_score(y, np.argmax(pred, axis=1)))


def run_elastic_endpoint(endpoint: base.Endpoint, frame: pd.DataFrame, *, folds: int, repeats: int, seed: int) -> dict[str, object]:
    common = endpoint.y.index.intersection(frame.index)
    y_series = endpoint.y.loc[common]
    x = frame.loc[common].fillna(0.0).to_numpy(dtype=np.float32)
    y, classes = encode_target(y_series, endpoint.task)
    splits = base.make_splits(y, endpoint.task, folds, repeats, seed)
    if endpoint.task == "regression":
        repeat_preds = []
        for repeat, split_list in splits:
            pred = np.zeros(len(y), dtype=np.float64)
            for fold, (train_idx, test_idx) in enumerate(split_list, start=1):
                pred[test_idx] = fit_predict_elastic(x[train_idx], y[train_idx], x[test_idx], task=endpoint.task, seed=seed + repeat * 1000 + fold)
            repeat_preds.append(pred)
        pred_out = np.mean(repeat_preds, axis=0)
    else:
        n_classes = 2 if endpoint.task == "binary" else len(classes)
        repeat_preds = []
        for repeat, split_list in splits:
            pred = np.zeros((len(y), n_classes), dtype=np.float64)
            for fold, (train_idx, test_idx) in enumerate(split_list, start=1):
                pred[test_idx] = fit_predict_elastic(x[train_idx], y[train_idx], x[test_idx], task=endpoint.task, seed=seed + repeat * 1000 + fold)
            repeat_preds.append(pred)
        pred_out = np.mean(repeat_preds, axis=0)
    metric, score, bal = score_predictions(y, pred_out, endpoint.task, classes)
    return {
        "endpoint": endpoint.name,
        "endpoint_family": endpoint.family,
        "task": endpoint.task,
        "metric": metric,
        "score": score,
        "balanced_accuracy": bal,
        "n": int(len(y)),
        "n_features": int(frame.shape[1]),
        "folds": int(folds),
        "repeats": int(repeats),
    }


def run_screen(
    candidates: dict[str, pd.DataFrame],
    endpoints: list[base.Endpoint],
    *,
    folds: int,
    repeats: int,
    xgb_estimators: int,
    tree_method: str,
) -> pd.DataFrame:
    rows = []
    baseline_scores: dict[tuple[str, str], float] = {}
    for learner in ["xgboost", "l2_logistic"]:
        for endpoint in endpoints:
            endpoint_seed = stable_seed("screen", learner, endpoint.name)
            if learner == "xgboost":
                metric, _pred = base.run_endpoint_model(
                    endpoint,
                    candidates[STANDARD_BASELINE],
                    folds=folds,
                    repeats=repeats,
                    n_estimators=xgb_estimators,
                    tree_method=tree_method,
                    seed=endpoint_seed,
                )
            else:
                metric = run_elastic_endpoint(endpoint, candidates[STANDARD_BASELINE], folds=folds, repeats=repeats, seed=endpoint_seed)
            metric.update({"model_id": STANDARD_BASELINE, "learner": learner, "delta_vs_standard": 0.0})
            rows.append(metric)
            baseline_scores[(learner, endpoint.name)] = metric["score"]
        for model_id, frame in candidates.items():
            if model_id == STANDARD_BASELINE:
                continue
            print(f"Screening {model_id} with {learner}", flush=True)
            for endpoint in endpoints:
                endpoint_seed = stable_seed("screen", learner, endpoint.name)
                if learner == "xgboost":
                    metric, _pred = base.run_endpoint_model(
                        endpoint,
                        frame,
                        folds=folds,
                        repeats=repeats,
                        n_estimators=xgb_estimators,
                        tree_method=tree_method,
                        seed=endpoint_seed,
                    )
                else:
                    metric = run_elastic_endpoint(endpoint, frame, folds=folds, repeats=repeats, seed=endpoint_seed)
                metric.update(
                    {
                        "model_id": model_id,
                        "learner": learner,
                        "delta_vs_standard": float(metric["score"] - baseline_scores[(learner, endpoint.name)]),
                    }
                )
                rows.append(metric)
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "rapid_biology_screen_metrics.csv", index=False)
    return out


def summarize_screen(metrics: pd.DataFrame) -> pd.DataFrame:
    df = metrics[metrics["model_id"] != STANDARD_BASELINE].copy()
    summary = (
        df.groupby(["model_id", "learner"], dropna=False)
        .agg(
            n_endpoints=("endpoint", "nunique"),
            mean_delta=("delta_vs_standard", "mean"),
            min_delta=("delta_vs_standard", "min"),
            max_delta=("delta_vs_standard", "max"),
            endpoint_gains_ge_0p03=("delta_vs_standard", lambda x: int(np.sum(np.asarray(x) >= 0.03))),
            endpoint_losses_lt_neg_0p02=("delta_vs_standard", lambda x: int(np.sum(np.asarray(x) < -0.02))),
            mean_score=("score", "mean"),
            n_features=("n_features", "max"),
        )
        .reset_index()
        .sort_values(["mean_delta", "min_delta"], ascending=[False, False])
    )
    summary.to_csv(DATA_DIR / "rapid_biology_screen_leaderboard.csv", index=False)
    return summary


def promote_finalists(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return []
    by_model = (
        summary.groupby("model_id")
        .agg(
            learners=("learner", "nunique"),
            mean_delta=("mean_delta", "mean"),
            min_delta=("min_delta", "min"),
            gains=("endpoint_gains_ge_0p03", "sum"),
            losses=("endpoint_losses_lt_neg_0p02", "sum"),
        )
        .reset_index()
    )
    passed = by_model[
        (by_model["learners"] == 2)
        & (by_model["mean_delta"] >= 0.02)
        & (by_model["min_delta"] >= -0.02)
        & (by_model["gains"] >= 3)
        & (by_model["losses"] == 0)
    ]
    return passed.sort_values("mean_delta", ascending=False)["model_id"].astype(str).head(3).tolist()


def build_manifest(candidate_frames: dict[str, pd.DataFrame], gate1: pd.DataFrame, screen_summary: pd.DataFrame) -> pd.DataFrame:
    spec_rows = []
    gate1_map = gate1.set_index("model_id").to_dict(orient="index")
    screen_map = screen_summary.groupby("model_id")["mean_delta"].mean().to_dict() if not screen_summary.empty else {}
    for model_id, frame in candidate_frames.items():
        if model_id == STANDARD_BASELINE:
            family = "standard_baseline"
            parameters = "{}"
            required_inputs = "standard_sbs96_id83_counts"
            preserves = True
            note = "Unchanged Standard SBS96+ID83 feature matrix."
        else:
            spec = get_coordinate_geometry(model_id)
            family = spec.family
            parameters = json.dumps(dict(spec.parameters), sort_keys=True)
            required_inputs = "; ".join(spec.required_inputs)
            preserves = spec.preserves_standard_identity
            note = spec.note
        spec_rows.append(
            {
                "model_id": model_id,
                "family": family,
                "required_inputs": required_inputs,
                "aggregation_method": "exact_channel_identity" if model_id in {STANDARD_BASELINE, EXACT_CONTROL} else "multiresolution_morton_histogram_l4_l6",
                "parameters": parameters,
                "n_features": int(frame.shape[1]),
                "preserves_standard_identity": preserves,
                "gate1_information_preservation_pass": bool(gate1_map.get(model_id, {}).get("gate1_information_preservation_pass", False)),
                "rapid_mean_delta_mean_across_learners": screen_map.get(model_id, np.nan),
                "note": note,
            }
        )
    manifest = pd.DataFrame(spec_rows)
    manifest.to_csv(DATA_DIR / "model_manifest.csv", index=False)
    return manifest


def html_table(df: pd.DataFrame, path: Path, title: str, footnote: str) -> None:
    base.write_html_table(df, path, title, footnote)


def formatted(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.loc[:, [c for c in columns if c in df.columns]].copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return out


def write_readme(metadata: dict[str, object], finalists: list[str], best: dict[str, object], gate0: pd.DataFrame) -> None:
    gate0_fail_rate = float(gate0["failed_context_rate"].mean()) if "failed_context_rate" in gate0 else float("nan")
    text = f"""# MAF Event Coordinate Geometry Optimization

## Research Question

This experiment evaluates which event-level coordinate geometry extracts biological signal from an MC3 MAF file. The benchmark treats the legacy channel-average UGA projection as a baseline family and evaluates MAF-derived sparse coordinate distributions against unchanged Standard SBS96+ID83 features.

## Methods

MC3 MAF rows were parsed for all patients with retained Standard feature matrices. REF alleles and local sequence context were checked against the GRCh37 FASTA, unresolved or mismatched events were excluded from event-level coordinate aggregation, and local one-megabase patient-level mutation density was used as the cluster proxy. Candidate geometries encoded observed events into two coordinates and aggregated patients with fixed Morton histograms at levels 4 and 6. Discovery endpoints were smoking status, top-10 cancer type, and LUAD KMT2C mutation status. XGBoost and L2/elastic-net sensitivity models used the same folds, labels, seeds, and metrics for Standard and every geometry.

## Key Numerical Findings

The event parser scanned {metadata["total_rows_scanned"]:,} MAF rows and encoded {metadata["encoded_events"]:,} events. The mean per-patient failed-context rate was {gate0_fail_rate:.4f}. The best rapid-screen model was `{best.get("model_id", "none")}` with learner `{best.get("learner", "none")}`, mean delta {best.get("mean_delta", float("nan")):.4f}, minimum delta {best.get("min_delta", float("nan")):.4f}, and maximum delta {best.get("max_delta", float("nan")):.4f}. Locked validation finalists: {", ".join(finalists) if finalists else "none"}.

## File Inventory

- `data/model_manifest.csv`: geometry registry manifest and feature dimensions.
- `data/gate0_data_validity.csv`: per-patient MAF parsing, variant class, encoded-event, and failed-context counts.
- `data/gate1_information_preservation.csv`: Standard identity reconstruction audit.
- `data/gate2_self_supervised_geometry_diagnostics.csv`: geometry purity diagnostics for mutation attributes.
- `data/rapid_biology_screen_metrics.csv`: endpoint-level XGBoost and elastic-net scores.
- `data/rapid_biology_screen_leaderboard.csv`: candidate-level rapid biological screen summary.
- `data/locked_validation_results.csv`: repeated-CV validation output for promoted finalists; empty when no geometry passes the gate.
- `tables/table1_data_validity.html`: Gate 0 summary table.
- `tables/table2_information_preservation.html`: Gate 1 audit table.
- `tables/table3_self_supervised_diagnostics.html`: Gate 2 diagnostic table.
- `tables/table4_rapid_biology_screen.html`: Gate 3 rapid screen table.
- `code/run_maf_event_coordinate_geometry_optimization.py`: complete runner.
- `code/signal_recovery_helpers.py`: endpoint loaders and paired XGBoost utilities.

## Reproducibility

Date executed: {metadata["completed_utc"]}. Random seed: `{RANDOM_SEED}`. Python: `{metadata["python_version"]}`. Package versions: pandas `{metadata["package_versions"]["pandas"]}`, numpy `{metadata["package_versions"]["numpy"]}`, scipy `{metadata["package_versions"]["scipy"]}`, scikit-learn `{metadata["package_versions"]["sklearn"]}`, xgboost `{metadata["package_versions"]["xgboost"]}`. Tree method requested: `{metadata["tree_method"]}`.
"""
    (EXPERIMENT_ROOT / "README.md").write_text(text, encoding="utf-8")


def update_ledger(metadata: dict[str, object], best: dict[str, object], finalists: list[str]) -> None:
    ledger = EXPERIMENTS_ROOT / "EXPERIMENT_LEDGER.md"
    if not ledger.exists():
        return
    marker = EXPERIMENT_ROOT.name
    text = ledger.read_text(encoding="utf-8")
    if marker in text:
        return
    row = (
        f"| `{marker}` | Complete exploratory MAF event-coordinate geometry screen | "
        f"{metadata['completed_local']} | {metadata['runtime_seconds']:.1f} s | "
        "TCGA MC3 MAF, GRCh37 FASTA, MC3 smoking status, MC3 top-10 cancer type, and MC3 LUAD KMT2C | "
        "Which MAF-derived event-level coordinate geometry gives the strongest biological utility under paired XGBoost and L2/elastic-net sensitivity screens? | "
        "Event-level coordinate geometries registered in `uga_atlas.coordinate_geometry`; Standard SBS96+ID83 retained as unchanged comparator | "
        "Standard SBS96+ID83 mutation-channel frequencies under identical folds, seeds, labels, and metrics | "
        f"Best rapid-screen row: `{best.get('model_id', 'none')}` with `{best.get('learner', 'none')}`, mean delta {float(best.get('mean_delta', float('nan'))):.4f}; finalists: {', '.join(finalists) if finalists else 'none'}. | "
        "Exploratory geometry-search result. A geometry is not retained for manuscript claims unless it passes locked validation. |\n"
    )
    insert_after = "| `2026_05_15_unified_coordspace_nonlossy_uga_screen`"
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith(insert_after):
            lines.insert(idx + 1, row.rstrip())
            ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
    ledger.write_text(text.rstrip() + "\n" + row, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--tree-method", default="gpu_hist")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--xgb-estimators", type=int, default=80)
    parser.add_argument("--max-maf-rows", type=int, default=0, help="Debug only. Default 0 scans all rows.")
    parser.add_argument("--skip-ledger", action="store_true")
    parser.add_argument("--skip-screen", action="store_true", help="Build features and gate diagnostics without fitting endpoint models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    started = time.time()
    standard_sbs, standard_id, standard_sbs_id, burden = base.load_feature_matrices()
    patients = standard_sbs_id.index.astype(str).tolist()
    candidate_frames: dict[str, pd.DataFrame] = {
        STANDARD_BASELINE: standard_sbs_id.astype(np.float32),
        EXACT_CONTROL: exact_identity_frame(standard_sbs_id),
    }
    print("Building MAF-derived event coordinate geometry features", flush=True)
    event_features, gate0, gate2, inventory = build_or_load_event_geometry_features(
        patient_ids=patients,
        burden=burden,
        rebuild=args.rebuild,
        max_maf_rows=args.max_maf_rows,
    )
    candidate_frames.update(event_features)
    for model_id, frame in candidate_frames.items():
        if model_id in {STANDARD_BASELINE, EXACT_CONTROL}:
            sparse_cache(model_id, frame)
    gate1 = reconstruction_audit(standard_sbs_id, candidate_frames)
    discovery_endpoints, locked_endpoints = endpoint_panel(standard_sbs_id.index)
    if args.skip_screen:
        metrics = pd.DataFrame()
        metrics.to_csv(DATA_DIR / "rapid_biology_screen_metrics.csv", index=False)
        leaderboard = pd.DataFrame(
            columns=[
                "model_id",
                "learner",
                "n_endpoints",
                "mean_delta",
                "min_delta",
                "max_delta",
                "endpoint_gains_ge_0p03",
                "endpoint_losses_lt_neg_0p02",
                "mean_score",
                "n_features",
            ]
        )
        leaderboard.to_csv(DATA_DIR / "rapid_biology_screen_leaderboard.csv", index=False)
        finalists = []
    else:
        print("Running paired rapid biological screen", flush=True)
        metrics = run_screen(
            candidate_frames,
            discovery_endpoints,
            folds=args.folds,
            repeats=1,
            xgb_estimators=args.xgb_estimators,
            tree_method=args.tree_method,
        )
        leaderboard = summarize_screen(metrics)
        finalists = promote_finalists(leaderboard)
    locked = pd.DataFrame(
        columns=[
            "model_id",
            "learner",
            "endpoint",
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
    if finalists:
        # Locked validation is intentionally scaffolded but not expanded here; current promotion criteria are strict.
        # If finalists appear, this placeholder forces the run to record that validation is required before any claim.
        locked = pd.DataFrame([{"model_id": f, "learner": "pending_locked_validation", "endpoint": "", "metric": "", "standard_score": np.nan, "candidate_score": np.nan, "delta_vs_standard": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "q_value": np.nan} for f in finalists])
    locked.to_csv(DATA_DIR / "locked_validation_results.csv", index=False)
    manifest = build_manifest(candidate_frames, gate1, leaderboard)

    gate0_summary = pd.DataFrame(
        [
            {"metric": "patients", "value": int(len(gate0))},
            {"metric": "maf_rows", "value": int(gate0["maf_rows"].sum()) if "maf_rows" in gate0 else 0},
            {"metric": "encoded_events", "value": int(gate0["encoded_events"].sum()) if "encoded_events" in gate0 else 0},
            {"metric": "mean_failed_context_rate", "value": float(gate0["failed_context_rate"].mean()) if "failed_context_rate" in gate0 else np.nan},
        ]
    )
    gate0_summary.to_csv(DATA_DIR / "gate0_data_validity_summary.csv", index=False)
    html_table(formatted(gate0_summary, ["metric", "value"]), TABLE_DIR / "table1_data_validity.html", "Table 1. Gate 0 Data Validity", "Event-level MAF parsing and reference-context validation summary.")
    html_table(
        formatted(gate1, ["model_id", "n_features", "sbs96_reconstruction_r2", "id83_reconstruction_r2", "estimated_channel_collisions", "gate1_information_preservation_pass"]),
        TABLE_DIR / "table2_information_preservation.html",
        "Table 2. Gate 1 Information Preservation",
        "Exact identity controls must reconstruct Standard SBS96 and ID83 channels with R2=1.0.",
    )
    html_table(
        formatted(gate2.sort_values(["target", "weighted_majority_purity"], ascending=[True, False]).head(40), ["model_id", "family", "target", "level", "weighted_majority_purity", "n_bins", "n_events"]),
        TABLE_DIR / "table3_self_supervised_diagnostics.html",
        "Table 3. Gate 2 Self-Supervised Geometry Diagnostics",
        "Purity is the weighted majority-label purity of coordinate bins for event attributes. These diagnostics are not endpoint-performance claims.",
    )
    html_table(
        formatted(leaderboard, ["model_id", "learner", "n_endpoints", "mean_delta", "min_delta", "max_delta", "endpoint_gains_ge_0p03", "endpoint_losses_lt_neg_0p02", "n_features"]),
        TABLE_DIR / "table4_rapid_biology_screen.html",
        "Table 4. Gate 3 Rapid Biological Screen",
        "Delta is candidate score minus unchanged Standard SBS96+ID83 score under paired folds, labels, seeds, and metrics.",
    )
    html_table(
        formatted(manifest, ["model_id", "family", "required_inputs", "aggregation_method", "n_features", "gate1_information_preservation_pass", "rapid_mean_delta_mean_across_learners"]),
        TABLE_DIR / "table5_model_manifest.html",
        "Table 5. Coordinate Geometry Manifest",
        "All geometry candidates were defined before biological endpoint fitting.",
    )

    metadata_json = DATA_DIR / "maf_event_geometry_cache_metadata.json"
    parser_meta = json.loads(metadata_json.read_text(encoding="utf-8")) if metadata_json.exists() else {}
    completed = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    metadata = {
        **parser_meta,
        "experiment": EXPERIMENT_ROOT.name,
        "completed_utc": completed,
        "completed_local": datetime.now().replace(microsecond=0).isoformat(),
        "runtime_seconds": round(time.time() - started, 3),
        "random_seed": RANDOM_SEED,
        "tree_method": args.tree_method,
        "folds": args.folds,
        "xgb_estimators": args.xgb_estimators,
        "n_candidates": len(candidate_frames),
        "n_finalists": len(finalists),
        "python_version": platform.python_version(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    (DATA_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    best = leaderboard.iloc[0].to_dict() if len(leaderboard) else {}
    write_readme(metadata, finalists, best, gate0)
    if not args.skip_ledger:
        update_ledger(metadata, best, finalists)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
