#!/usr/bin/env python3
"""UGA RBF kernel-mean embedding benchmark with fixed 5-fold linear OOF models.

This exploratory benchmark compares three representation families:

1. retained standard mutation-channel features,
2. the previous locked UGA mean-vector representation, and
3. a finite-dimensional RBF kernel mean embedding over locked UGA coordinates.

No endpoint-specific hyperparameter tuning is performed. Each endpoint is scored
from one aggregate set of out-of-fold predictions from a fixed 5-fold split.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.checkpointing import atomic_write_csv, atomic_write_json, merge_checkpoint_rows, read_completed_keys


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
DATA_DIR = EXPERIMENT_ROOT / "data"
TABLE_DIR = EXPERIMENT_ROOT / "tables"
RESULT_KEY_COLUMNS = ["benchmark", "endpoint", "representation"]
PREDICTION_KEY_COLUMNS = ["benchmark", "endpoint", "representation", "sample"]
PROBABILITY_KEY_COLUMNS = ["benchmark", "endpoint", "representation", "sample", "class_label"]


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
PROJECT_ROOT = find_project_root(EXPERIMENT_ROOT)
RESEARCH_ROOT = PROJECT_ROOT / "cgr_validation_results" / "research"

ACTIVE_ROOT = EXPERIMENTS_ROOT / "supporting" / "2026_05_14_unified_uga_locked_manuscript_benchmark"
ACTIVE_CODE = ACTIVE_ROOT / "code"
SOURCE_MC3 = ACTIVE_ROOT / "data" / "mc3_source"
FEATURE_DIR = SOURCE_MC3 / "features"
RAW_DIR = SOURCE_MC3 / "raw"
KUCAB_CODE = ACTIVE_CODE

HRD_COHORT = RESEARCH_ROOT / "assets" / "EXP023_tcga_brca_hrd" / "TCGA-BRCA" / "cohort" / "final_analysis_cohort.tsv"
CONTEXT_ATLAS = RESEARCH_ROOT / "data" / "EXP022_atlas_genome_wide_45mer_universal_d22.json"

for path in (PROJECT_ROOT, ACTIVE_CODE, KUCAB_CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from uga_atlas import build_uga_basis, get_uga_model, load_context_atlas, project_counts_to_uga  # noqa: E402

import run_locked_kucab_low_burden_benchmark as kucab  # noqa: E402


RANDOM_SEED = 20260517
N_SPLITS = 5
LOCKED_SBS_MODEL = "master_spec_sbs_dbs_d10_dp5"
LOCKED_ID_MODEL = "id83_payload_only_d10_dp5"

HRD_CONTINUOUS = ["HRD_Score", "eCARD", "HRD_TAI", "HRD_LST", "HRD_LOH", "PARPi7"]
HRD_BINARY = ["hrd_binary_24", "hrd_binary_33", "hrd_binary_42", "parpi7_binary"]
MC3_CLINICAL = {
    "cancer_type_top10": "multiclass",
    "smoking_ever": "binary",
    "high_purity": "binary",
    "high_stage": "binary",
    "os_event": "binary",
}


@dataclass(frozen=True)
class Endpoint:
    name: str
    family: str
    task: str
    y: pd.Series


@dataclass(frozen=True)
class FeatureSet:
    name: str
    benchmark: str
    frame: pd.DataFrame
    construction: str
    source: str
    kernel_sigma: float | None = None
    n_encoded_channels: int | None = None
    n_landmarks: int | None = None


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def stable_seed(*parts: object) -> int:
    text = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return RANDOM_SEED + int.from_bytes(digest[:4], "big") % 100_000


def strip_prefix(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [col for col in frame.columns if str(col).startswith(prefix)]
    out = frame.loc[:, cols].copy()
    out.columns = [str(col).replace(prefix, "", 1) for col in cols]
    return out


def load_feature_matrices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    standard_sbs = pd.read_csv(FEATURE_DIR / "features_standard_sbs96.csv.gz", index_col=0).fillna(0.0)
    standard_id = pd.read_csv(FEATURE_DIR / "features_standard_id83.csv.gz", index_col=0).fillna(0.0)
    standard_sbs_id = pd.read_csv(FEATURE_DIR / "features_standard_sbs96_id83.csv.gz", index_col=0).fillna(0.0)
    burden = pd.read_csv(FEATURE_DIR / "features_burden_only.csv", index_col=0).fillna(0.0)
    for frame in (standard_sbs, standard_id, standard_sbs_id, burden):
        frame.index = frame.index.astype(str)
    return standard_sbs, standard_id, standard_sbs_id, burden


def atlas_for_model(model_name: str) -> dict[str, np.ndarray] | None:
    spec = get_uga_model(model_name)
    if spec.context_source == "genome_atlas":
        return load_context_atlas(CONTEXT_ATLAS, spec.d_context)
    return None


def build_registered_uga_features(
    standard_sbs: pd.DataFrame,
    standard_id: pd.DataFrame,
    burden: pd.DataFrame,
    *,
    sbs_model: str,
    id_model: str,
    prefix: str,
) -> dict[str, pd.DataFrame]:
    sbs_counts = strip_prefix(standard_sbs, "SBS96__")
    id_counts = strip_prefix(standard_id, "ID83__")
    sbs_basis, sbs_diag = build_uga_basis(sbs_counts.columns.astype(str).tolist(), sbs_model, atlas=atlas_for_model(sbs_model), modality="SBS")
    id_basis, id_diag = build_uga_basis(id_counts.columns.astype(str).tolist(), id_model)
    sbs_uga = project_counts_to_uga(
        sbs_counts,
        sbs_basis,
        sbs_diag["UGA_Encoded"].to_numpy(dtype=bool),
        f"{prefix}_sbs",
    )
    id_uga = project_counts_to_uga(
        id_counts,
        id_basis,
        id_diag["UGA_Encoded"].to_numpy(dtype=bool),
        f"{prefix}_id",
    )
    out = {
        "separate": pd.concat([burden, sbs_uga, id_uga], axis=1).fillna(0.0),
    }
    if sbs_basis.shape[1] == id_basis.shape[1]:
        pooled_counts = pd.concat([sbs_counts, id_counts], axis=1)
        pooled_basis = np.vstack([sbs_basis, id_basis])
        pooled_valid = np.concatenate(
            [
                sbs_diag["UGA_Encoded"].to_numpy(dtype=bool),
                id_diag["UGA_Encoded"].to_numpy(dtype=bool),
            ]
        )
        pooled = project_counts_to_uga(pooled_counts, pooled_basis, pooled_valid, f"{prefix}_pooled")
        out["pooled"] = pd.concat([burden, pooled], axis=1).fillna(0.0)
    return out


def load_hrd_endpoints() -> list[Endpoint]:
    cohort = pd.read_csv(HRD_COHORT, sep="\t")
    cohort["patient_id_12"] = cohort["patient_id_12"].astype(str)
    endpoints: list[Endpoint] = []
    for endpoint in HRD_CONTINUOUS:
        data = cohort.dropna(subset=[endpoint]).copy()
        y = pd.Series(data[endpoint].astype(float).to_numpy(), index=data["patient_id_12"], name=endpoint)
        endpoints.append(Endpoint(endpoint, "hrd", "regression", y))
    for endpoint in HRD_BINARY:
        labels = ["PARPi-high", "PARPi-low"] if endpoint == "parpi7_binary" else ["HRD-high", "HRD-low"]
        positive = "PARPi-high" if endpoint == "parpi7_binary" else "HRD-high"
        data = cohort[cohort[endpoint].isin(labels)].copy()
        y = pd.Series((data[endpoint] == positive).astype(int).to_numpy(), index=data["patient_id_12"], name=endpoint)
        endpoints.append(Endpoint(endpoint, "hrd", "binary", y))
    return endpoints


def load_mc3_clinical_endpoints() -> list[Endpoint]:
    labels = pd.read_csv(SOURCE_MC3 / "biology_labels.csv", index_col=0)
    labels.index = labels.index.astype(str)
    endpoints: list[Endpoint] = []
    for endpoint, task in MC3_CLINICAL.items():
        y = labels[endpoint].dropna()
        if task == "binary":
            y = y.astype(int)
            if y.nunique() == 2 and y.value_counts().min() >= 25:
                endpoints.append(Endpoint(endpoint, "mc3_clinical", "binary", y))
        else:
            counts = y.astype(str).value_counts()
            y = y.astype(str)
            y = y[y.isin(counts[counts >= 50].index)]
            if y.nunique() >= 3:
                endpoints.append(Endpoint(endpoint, "mc3_clinical", "multiclass", y))
    return endpoints


def load_cancer_types(raw_dir: Path, patients: pd.Index) -> pd.Series:
    cdr = pd.read_excel(raw_dir / "TCGA-CDR-SupplementalTableS1.xlsx", usecols=["bcr_patient_barcode", "type"])
    cdr["patient"] = cdr["bcr_patient_barcode"].astype(str).str[:12]
    cancer_type = cdr.drop_duplicates("patient").set_index("patient")["type"].astype(str)
    return cancer_type.reindex(patients)


def build_driver_labels(source_dir: Path, patients: pd.Index) -> pd.DataFrame:
    out_path = source_dir / "driver_gene_labels_functional.csv"
    if not out_path.exists():
        raise FileNotFoundError(f"Expected cached driver label file not found: {out_path}")
    return pd.read_csv(out_path, index_col=0).reindex(patients).fillna(0).astype(int)


def load_kmt2c_endpoint(patients: pd.Index) -> Endpoint:
    cancer_type = load_cancer_types(RAW_DIR, patients)
    driver_labels = build_driver_labels(SOURCE_MC3, patients)
    luad_patients = cancer_type.index[cancer_type == "LUAD"]
    y = driver_labels.loc[driver_labels.index.intersection(luad_patients), "kmt2c_mutated"].astype(int)
    return Endpoint("luad_kmt2c_mutated", "mc3_kmt2c_supportive", "binary", y)


def median_pairwise_scale(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 1.0
    diffs = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2))
    vals = distances[np.triu_indices(points.shape[0], k=1)]
    vals = vals[np.isfinite(vals) & (vals > 1e-12)]
    if vals.size == 0:
        return 1.0
    return float(np.median(vals))


def rbf_kme_features(
    counts: pd.DataFrame,
    basis: np.ndarray,
    valid: np.ndarray,
    *,
    prefix: str,
    channel_labels: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    valid = np.asarray(valid, dtype=bool) & np.isfinite(basis).all(axis=1)
    if basis.shape[0] != counts.shape[1]:
        raise ValueError(f"Basis rows ({basis.shape[0]}) do not match count columns ({counts.shape[1]})")
    if not bool(valid.any()):
        raise ValueError("No valid UGA channel coordinates are available for KME features")

    labels = np.asarray(list(channel_labels), dtype=object)
    landmarks = basis[valid]
    landmark_labels = labels[valid]
    sigma = median_pairwise_scale(landmarks)
    dist2 = np.sum((basis[:, None, :] - landmarks[None, :, :]) ** 2, axis=2)
    kernel = np.exp(-0.5 * dist2 / max(sigma * sigma, 1e-12))
    kernel[~valid, :] = 0.0

    raw = counts.to_numpy(dtype=np.float64)
    masked = raw * valid.astype(np.float64)
    denom = masked.sum(axis=1, keepdims=True)
    weights = np.divide(masked, denom, out=np.zeros_like(masked), where=denom > 0)
    embedded = weights @ kernel
    columns = [f"{prefix}__{label}" for label in landmark_labels]
    features = pd.DataFrame(embedded, index=counts.index, columns=columns)
    diagnostics = pd.DataFrame(
        {
            "feature_prefix": prefix,
            "channel": labels,
            "uga_encoded": valid,
            "used_as_landmark": valid,
        }
    )
    metadata = {
        "kernel_sigma": sigma,
        "n_input_channels": int(counts.shape[1]),
        "n_encoded_channels": int(valid.sum()),
        "n_landmarks": int(valid.sum()),
        "feature_dimension": int(features.shape[1]),
    }
    return features, diagnostics, metadata


def build_mc3_feature_sets() -> tuple[list[FeatureSet], pd.DataFrame]:
    standard_sbs, standard_id, standard_sbs_id, burden = load_feature_matrices()
    sbs_counts = strip_prefix(standard_sbs, "SBS96__")
    id_counts = strip_prefix(standard_id, "ID83__")

    previous_uga = build_registered_uga_features(
        standard_sbs,
        standard_id,
        burden,
        sbs_model=LOCKED_SBS_MODEL,
        id_model=LOCKED_ID_MODEL,
        prefix="locked_mean",
    )["pooled"]

    sbs_basis, sbs_diag = build_uga_basis(sbs_counts.columns.astype(str).tolist(), LOCKED_SBS_MODEL, atlas=atlas_for_model(LOCKED_SBS_MODEL), modality="SBS")
    id_basis, id_diag = build_uga_basis(id_counts.columns.astype(str).tolist(), LOCKED_ID_MODEL)
    pooled_counts = pd.concat([sbs_counts, id_counts], axis=1)
    pooled_basis = np.vstack([sbs_basis, id_basis])
    pooled_valid = np.concatenate(
        [
            sbs_diag["UGA_Encoded"].to_numpy(dtype=bool),
            id_diag["UGA_Encoded"].to_numpy(dtype=bool),
        ]
    )
    channel_labels = [f"SBS96:{col}" for col in sbs_counts.columns] + [f"ID83:{col}" for col in id_counts.columns]
    kme, diagnostics, kme_meta = rbf_kme_features(
        pooled_counts,
        pooled_basis,
        pooled_valid,
        prefix="uga_rbf_kme",
        channel_labels=channel_labels,
    )
    kme_with_burden = pd.concat([burden, kme], axis=1).fillna(0.0)
    diagnostics["benchmark"] = "mc3_hrd_kmt2c"

    feature_sets = [
        FeatureSet(
            "standard_sbs96_id83",
            "mc3_hrd_kmt2c",
            standard_sbs_id,
            "Retained SBS96+ID83 channel fractions with retained burden covariates.",
            str(FEATURE_DIR / "features_standard_sbs96_id83.csv.gz"),
        ),
        FeatureSet(
            "previous_uga_mean_pooled",
            "mc3_hrd_kmt2c",
            previous_uga,
            "Locked pooled UGA mean vector from SBS96 and ID83 channel distributions plus retained burden covariates.",
            "build_uga_basis + project_counts_to_uga using master_spec_sbs_dbs_d10_dp5 and id83_payload_only_d10_dp5",
        ),
        FeatureSet(
            "uga_rbf_kernel_mean",
            "mc3_hrd_kmt2c",
            kme_with_burden,
            "Empirical kernel mean embedding: normalized SBS96+ID83 patient distribution multiplied by an RBF kernel over locked UGA channel coordinates; all valid channels are landmarks.",
            "features_standard_sbs96.csv.gz + features_standard_id83.csv.gz + locked UGA basis",
            kernel_sigma=float(kme_meta["kernel_sigma"]),
            n_encoded_channels=int(kme_meta["n_encoded_channels"]),
            n_landmarks=int(kme_meta["n_landmarks"]),
        ),
    ]
    return feature_sets, diagnostics


def make_regressor() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])


def make_classifier(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def encode_target(y: pd.Series, task: str) -> tuple[np.ndarray, np.ndarray]:
    if task == "regression":
        return y.astype(float).to_numpy(dtype=np.float64), np.array([], dtype=object)
    if task == "binary":
        return y.astype(int).to_numpy(dtype=np.int32), np.array([0, 1], dtype=object)
    classes = np.array(sorted(y.astype(str).unique()), dtype=object)
    mapping = {value: i for i, value in enumerate(classes)}
    return y.astype(str).map(mapping).astype(int).to_numpy(dtype=np.int32), classes


def make_standard_splits(y: np.ndarray, task: str, seed: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    if task == "regression":
        splitter = KFold(n_splits=min(N_SPLITS, len(y)), shuffle=True, random_state=seed)
        return [(fold, train_idx, test_idx) for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y))), start=1)]
    min_class = int(pd.Series(y).value_counts().min())
    n_splits = min(N_SPLITS, min_class)
    if n_splits < 2:
        raise ValueError("Not enough samples in every class for cross-validation")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(fold, train_idx, test_idx) for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y), start=1)]


def score_oof(
    y: np.ndarray,
    pred: np.ndarray,
    task: str,
    classes: np.ndarray,
) -> tuple[str, float, dict[str, float]]:
    aux: dict[str, float] = {}
    if task == "regression":
        rho = spearmanr(y, pred).statistic
        return "spearman", float(rho) if np.isfinite(rho) else math.nan, aux
    if task == "binary":
        score = float(roc_auc_score(y, pred[:, 1]))
        aux["balanced_accuracy_at_0p5"] = float(balanced_accuracy_score(y, (pred[:, 1] >= 0.5).astype(int)))
        return "auroc", score, aux
    score = float(roc_auc_score(y, pred, average="macro", multi_class="ovr"))
    aux["balanced_accuracy_argmax"] = float(balanced_accuracy_score(y, np.argmax(pred, axis=1)))
    return "macro_auroc", score, aux


def evaluate_endpoint(endpoint: Endpoint, feature_set: FeatureSet) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    common = endpoint.y.index.astype(str).intersection(feature_set.frame.index.astype(str))
    y_series = endpoint.y.loc[common]
    x = feature_set.frame.loc[common].fillna(0.0).to_numpy(dtype=np.float64)
    y, classes = encode_target(y_series, endpoint.task)
    seed = stable_seed(endpoint.name)
    splits = make_standard_splits(y, endpoint.task, seed)

    folds = np.zeros(len(y), dtype=int)
    if endpoint.task == "regression":
        pred = np.zeros(len(y), dtype=np.float64)
        prob_long = pd.DataFrame()
        for fold, train_idx, test_idx in splits:
            model = make_regressor()
            model.fit(x[train_idx], y[train_idx])
            pred[test_idx] = model.predict(x[test_idx])
            folds[test_idx] = fold
    else:
        n_classes = 2 if endpoint.task == "binary" else len(classes)
        pred = np.zeros((len(y), n_classes), dtype=np.float64)
        for fold, train_idx, test_idx in splits:
            model = make_classifier(seed + fold)
            model.fit(x[train_idx], y[train_idx])
            fold_proba = model.predict_proba(x[test_idx])
            model_classes = model.named_steps["model"].classes_
            for local_col, class_id in enumerate(model_classes):
                pred[test_idx, int(class_id)] = fold_proba[:, local_col]
            folds[test_idx] = fold
        long_rows = []
        for i, sample in enumerate(common):
            for j in range(n_classes):
                class_label = str(classes[j]) if endpoint.task == "multiclass" else str(j)
                long_rows.append(
                    {
                        "benchmark": feature_set.benchmark,
                        "endpoint": endpoint.name,
                        "family": endpoint.family,
                        "task": endpoint.task,
                        "representation": feature_set.name,
                        "sample": sample,
                        "fold": int(folds[i]),
                        "class_label": class_label,
                        "probability": float(pred[i, j]),
                    }
                )
        prob_long = pd.DataFrame(long_rows)

    metric_name, score, aux = score_oof(y, pred, endpoint.task, classes)
    row = {
        "benchmark": feature_set.benchmark,
        "endpoint": endpoint.name,
        "family": endpoint.family,
        "task": endpoint.task,
        "representation": feature_set.name,
        "metric": metric_name,
        "score": score,
        "n_samples": int(len(y)),
        "n_classes": int(pd.Series(y).nunique()) if endpoint.task != "regression" else math.nan,
        "n_features": int(feature_set.frame.shape[1]),
        "n_folds": int(len(splits)),
        "model": "Ridge(alpha=1.0)" if endpoint.task == "regression" else "LogisticRegression(L2,C=1.0,class_weight=balanced)",
        "split_strategy": "5-fold KFold" if endpoint.task == "regression" else "5-fold StratifiedKFold",
        "tuning": "none",
    }
    row.update(aux)

    pred_rows = []
    for i, sample in enumerate(common):
        item = {
            "benchmark": feature_set.benchmark,
            "endpoint": endpoint.name,
            "family": endpoint.family,
            "task": endpoint.task,
            "representation": feature_set.name,
            "sample": sample,
            "fold": int(folds[i]),
            "y_true": str(y_series.iloc[i]) if endpoint.task == "multiclass" else float(y_series.iloc[i]),
        }
        if endpoint.task == "regression":
            item["prediction"] = float(pred[i])
        elif endpoint.task == "binary":
            item["prediction"] = float(pred[i, 1])
            item["predicted_label"] = int(pred[i, 1] >= 0.5)
        else:
            pred_id = int(np.argmax(pred[i]))
            item["prediction"] = str(classes[pred_id])
            item["predicted_label"] = str(classes[pred_id])
        pred_rows.append(item)

    return row, pd.DataFrame(pred_rows), prob_long


def build_kucab_feature_sets() -> tuple[list[FeatureSet], pd.DataFrame, pd.DataFrame]:
    metadata, sbs, dbs, id_counts, _mapped = kucab.load_raw_counts()
    use = kucab.eligible_metadata(metadata)
    sbs = sbs.loc[use.index]
    dbs = dbs.loc[use.index]
    id_counts = id_counts.loc[use.index]
    raw_counts = np.concatenate([sbs.to_numpy(), dbs.to_numpy(), id_counts.to_numpy()], axis=1).astype(np.int64)
    keep = np.ones(len(use), dtype=bool)
    standard_columns = [f"SBS:{c}" for c in sbs.columns] + [f"DBS:{c}" for c in dbs.columns] + [f"ID:{c}" for c in id_counts.columns]
    standard = kucab.standard_features(raw_counts, keep, use.index, standard_columns)

    variant = kucab.build_variant(
        "previous_uga_mean_unified",
        LOCKED_SBS_MODEL,
        LOCKED_ID_MODEL,
        "unweighted_frac",
        1.0,
        sbs.columns.astype(str).tolist(),
        dbs.columns.astype(str).tolist(),
        id_counts.columns.astype(str).tolist(),
    )
    previous_uga = kucab.uga_features(raw_counts, keep, use.index, len(sbs.columns), len(dbs.columns), variant)

    pooled_counts = pd.DataFrame(raw_counts, index=use.index, columns=standard_columns)
    pooled_basis = np.vstack([variant["sbsdbs_basis"], variant["id_basis"]])
    pooled_valid = np.concatenate([variant["sbsdbs_valid"], variant["id_valid"]])
    kme, diagnostics, kme_meta = rbf_kme_features(
        pooled_counts,
        pooled_basis,
        pooled_valid,
        prefix="uga_rbf_kme",
        channel_labels=standard_columns,
    )
    diagnostics["benchmark"] = "kucab_damage_class"

    feature_sets = [
        FeatureSet(
            "standard_sbs96_dbs78_id83",
            "kucab_damage_class",
            standard,
            "Original-data normalized SBS96+DBS78+ID83 channel distribution.",
            str(ACTIVE_ROOT / "data" / "raw"),
        ),
        FeatureSet(
            "previous_uga_mean_unified",
            "kucab_damage_class",
            previous_uga,
            "Locked unified UGA mean-vector representation for SBS/DBS and ID83 channels with mutation-type fractions.",
            "run_locked_kucab_low_burden_benchmark.py::uga_features",
        ),
        FeatureSet(
            "uga_rbf_kernel_mean",
            "kucab_damage_class",
            kme,
            "Empirical kernel mean embedding over locked UGA coordinates for the original SBS96+DBS78+ID83 Kucab distribution; all valid channels are landmarks.",
            "Kucab raw SBS/DBS/ID counts + locked UGA basis",
            kernel_sigma=float(kme_meta["kernel_sigma"]),
            n_encoded_channels=int(kme_meta["n_encoded_channels"]),
            n_landmarks=int(kme_meta["n_landmarks"]),
        ),
    ]
    endpoint = pd.DataFrame(
        {
            "sample": use.index.astype(str),
            "damage_class": use["damage_class"].astype(str).to_numpy(),
            "agent_core": use["agent_core"].astype(str).to_numpy(),
        }
    ).set_index("sample")
    return feature_sets, endpoint, diagnostics


def evaluate_kucab(feature_set: FeatureSet, endpoint: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    meta = endpoint.loc[feature_set.frame.index.astype(str)]
    x = feature_set.frame.loc[meta.index].fillna(0.0).to_numpy(dtype=np.float64)
    classes = np.array(sorted(meta["damage_class"].unique()), dtype=object)
    mapping = {value: i for i, value in enumerate(classes)}
    y = meta["damage_class"].map(mapping).astype(int).to_numpy()
    groups = meta["agent_core"].to_numpy()
    seed = stable_seed("kucab_damage_class")
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    pred = np.zeros((len(y), len(classes)), dtype=np.float64)
    folds = np.zeros(len(y), dtype=int)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        model = make_classifier(seed + fold)
        model.fit(x[train_idx], y[train_idx])
        fold_proba = model.predict_proba(x[test_idx])
        model_classes = model.named_steps["model"].classes_
        for local_col, class_id in enumerate(model_classes):
            pred[test_idx, int(class_id)] = fold_proba[:, local_col]
        folds[test_idx] = fold

    predicted_ids = np.argmax(pred, axis=1)
    macro_auc = float(roc_auc_score(y, pred, average="macro", multi_class="ovr"))
    balanced = float(balanced_accuracy_score(y, predicted_ids))
    accuracy = float(accuracy_score(y, predicted_ids))
    macro_f1 = float(f1_score(y, predicted_ids, average="macro", zero_division=0))
    row = {
        "benchmark": feature_set.benchmark,
        "endpoint": "damage_class",
        "family": "kucab",
        "task": "multiclass_grouped",
        "representation": feature_set.name,
        "metric": "macro_auroc",
        "score": macro_auc,
        "balanced_accuracy_argmax": balanced,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "n_samples": int(len(y)),
        "n_classes": int(len(classes)),
        "n_features": int(feature_set.frame.shape[1]),
        "n_folds": N_SPLITS,
        "model": "LogisticRegression(L2,C=1.0,class_weight=balanced)",
        "split_strategy": "5-fold StratifiedGroupKFold grouped by agent_core",
        "tuning": "none",
    }

    pred_rows = []
    prob_rows = []
    for i, sample in enumerate(meta.index):
        pred_id = int(predicted_ids[i])
        pred_rows.append(
            {
                "benchmark": feature_set.benchmark,
                "endpoint": "damage_class",
                "family": "kucab",
                "task": "multiclass_grouped",
                "representation": feature_set.name,
                "sample": sample,
                "fold": int(folds[i]),
                "y_true": str(meta["damage_class"].iloc[i]),
                "prediction": str(classes[pred_id]),
                "predicted_label": str(classes[pred_id]),
            }
        )
        for j, label in enumerate(classes):
            prob_rows.append(
                {
                    "benchmark": feature_set.benchmark,
                    "endpoint": "damage_class",
                    "family": "kucab",
                    "task": "multiclass_grouped",
                    "representation": feature_set.name,
                    "sample": sample,
                    "fold": int(folds[i]),
                    "class_label": str(label),
                    "probability": float(pred[i, j]),
                }
            )
    return row, pd.DataFrame(pred_rows), pd.DataFrame(prob_rows)


def add_delta_columns(results: pd.DataFrame) -> pd.DataFrame:
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
        standard_score = group.loc[group["representation"] == standard_names.get(benchmark), "score"]
        previous_score = group.loc[group["representation"] == previous_names.get(benchmark), "score"]
        if not standard_score.empty:
            out.loc[group.index, "delta_vs_standard"] = group["score"] - float(standard_score.iloc[0])
        if not previous_score.empty:
            out.loc[group.index, "delta_vs_previous_uga_mean"] = group["score"] - float(previous_score.iloc[0])
    return out


def build_feature_manifest(feature_sets: list[FeatureSet]) -> pd.DataFrame:
    rows = []
    for fs in feature_sets:
        rows.append(
            {
                "benchmark": fs.benchmark,
                "representation": fs.name,
                "n_features": int(fs.frame.shape[1]),
                "n_samples_available": int(fs.frame.shape[0]),
                "construction": fs.construction,
                "source": fs.source,
                "kernel_sigma": fs.kernel_sigma,
                "n_encoded_channels": fs.n_encoded_channels,
                "n_landmarks": fs.n_landmarks,
            }
        )
    return pd.DataFrame(rows)


def write_readme(results: pd.DataFrame, feature_manifest: pd.DataFrame, elapsed: float) -> None:
    kme = results[results["representation"] == "uga_rbf_kernel_mean"].copy()
    wins_standard = int((kme["delta_vs_standard"] > 0).sum())
    ties_standard = int((kme["delta_vs_standard"].abs() <= 1e-12).sum())
    total = int(kme["delta_vs_standard"].notna().sum())
    wins_previous = int((kme["delta_vs_previous_uga_mean"] > 0).sum())
    endpoint_table = kme[
        [
            "benchmark",
            "family",
            "endpoint",
            "metric",
            "score",
            "delta_vs_standard",
            "delta_vs_previous_uga_mean",
        ]
    ].sort_values(["benchmark", "family", "endpoint"])
    endpoint_table = endpoint_table.rename(
        columns={
            "score": "kme_score",
            "delta_vs_standard": "kme_minus_standard",
            "delta_vs_previous_uga_mean": "kme_minus_previous_uga_mean",
        }
    )
    previous_ahead = endpoint_table.loc[
        endpoint_table["kme_minus_previous_uga_mean"] < 0,
        "endpoint",
    ].tolist()
    previous_ahead_text = ", ".join(previous_ahead) if previous_ahead else "none"
    lines = [
        "# UGA RBF Kernel Mean Linear Benchmark",
        "",
        f"Executed UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Elapsed seconds: {elapsed:.2f}",
        "",
        "## Protocol",
        "",
        "- Representations: retained standard channel features, previous locked UGA mean features, and an RBF kernel mean embedding over locked UGA coordinates.",
        "- Model: fixed Ridge(alpha=1.0) for regression and fixed L2 LogisticRegression(C=1.0, class_weight=balanced) for classification.",
        "- Validation: one 5-fold out-of-fold prediction pass per endpoint; all representations share the same endpoint-specific folds. Kucab uses grouped 5-fold CV by agent_core.",
        "- Tuning: none.",
        "",
        "## Headline",
        "",
        f"UGA-RBF KME beat the standard representation on {wins_standard}/{total} scored endpoint comparisons"
        + (f" with {ties_standard} exact ties" if ties_standard else "")
        + f", and beat the previous UGA mean on {wins_previous}/{total} comparisons.",
        "",
        "Interpretation: in this lightweight fixed-linear protocol, the KME representation is a strong upgrade over the retained standard channel baseline for HRD and KMT2C, but it is not a uniform replacement for the previous UGA mean. The previous mean remains ahead on: "
        + previous_ahead_text
        + ".",
        "",
        "## UGA-RBF KME Endpoint Deltas",
        "",
        endpoint_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Outputs",
        "",
        "- `data/endpoint_results.csv`: endpoint-level scores and deltas.",
        "- `data/oof_predictions.csv.gz`: aggregate out-of-fold predictions.",
        "- `data/oof_probabilities_long.csv.gz`: class probabilities for binary/multiclass endpoints.",
        "- `data/feature_manifest.csv`: representation construction and dimensions.",
        "- `data/kernel_basis_diagnostics.csv`: channels encoded into each KME kernel basis.",
        "- `tables/family_summary.csv`: average deltas by endpoint family.",
        "",
        "## Feature Manifest",
        "",
        feature_manifest.to_markdown(index=False),
        "",
    ]
    (EXPERIMENT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.time()
    ensure_dirs()

    all_results = []
    all_feature_sets: list[FeatureSet] = []
    diagnostics = []
    endpoint_results_path = DATA_DIR / "endpoint_results.csv"
    prediction_checkpoint_path = DATA_DIR / "oof_predictions_checkpoint.csv"
    probability_checkpoint_path = DATA_DIR / "oof_probabilities_long_checkpoint.csv"
    completed = read_completed_keys(endpoint_results_path, RESULT_KEY_COLUMNS)

    mc3_feature_sets, mc3_diagnostics = build_mc3_feature_sets()
    all_feature_sets.extend(mc3_feature_sets)
    diagnostics.append(mc3_diagnostics)
    standard_index = mc3_feature_sets[0].frame.index
    endpoints = load_hrd_endpoints() + load_mc3_clinical_endpoints() + [load_kmt2c_endpoint(standard_index)]
    for endpoint in endpoints:
        for feature_set in mc3_feature_sets:
            key = (str(feature_set.benchmark), str(endpoint.name), str(feature_set.name))
            if key in completed:
                print(f"[checkpoint] skip {key}", flush=True)
                continue
            result, predictions, probabilities = evaluate_endpoint(endpoint, feature_set)
            all_results.append(result)
            merge_checkpoint_rows(endpoint_results_path, [result], key_columns=RESULT_KEY_COLUMNS, sort_columns=RESULT_KEY_COLUMNS)
            merge_checkpoint_rows(
                prediction_checkpoint_path,
                predictions.to_dict("records"),
                key_columns=PREDICTION_KEY_COLUMNS,
                sort_columns=PREDICTION_KEY_COLUMNS,
            )
            if not probabilities.empty:
                merge_checkpoint_rows(
                    probability_checkpoint_path,
                    probabilities.to_dict("records"),
                    key_columns=PROBABILITY_KEY_COLUMNS,
                    sort_columns=PROBABILITY_KEY_COLUMNS,
                )
            completed.add(key)
            print(f"[checkpoint] wrote {key}", flush=True)

    kucab_feature_sets, kucab_endpoint, kucab_diagnostics = build_kucab_feature_sets()
    all_feature_sets.extend(kucab_feature_sets)
    diagnostics.append(kucab_diagnostics)
    for feature_set in kucab_feature_sets:
        key = (str(feature_set.benchmark), "damage_class", str(feature_set.name))
        if key in completed:
            print(f"[checkpoint] skip {key}", flush=True)
            continue
        result, predictions, probabilities = evaluate_kucab(feature_set, kucab_endpoint)
        all_results.append(result)
        merge_checkpoint_rows(endpoint_results_path, [result], key_columns=RESULT_KEY_COLUMNS, sort_columns=RESULT_KEY_COLUMNS)
        merge_checkpoint_rows(
            prediction_checkpoint_path,
            predictions.to_dict("records"),
            key_columns=PREDICTION_KEY_COLUMNS,
            sort_columns=PREDICTION_KEY_COLUMNS,
        )
        merge_checkpoint_rows(
            probability_checkpoint_path,
            probabilities.to_dict("records"),
            key_columns=PROBABILITY_KEY_COLUMNS,
            sort_columns=PROBABILITY_KEY_COLUMNS,
        )
        completed.add(key)
        print(f"[checkpoint] wrote {key}", flush=True)

    if not endpoint_results_path.exists():
        raise RuntimeError("No endpoint checkpoint rows were written")
    results = add_delta_columns(pd.read_csv(endpoint_results_path))
    predictions = pd.read_csv(prediction_checkpoint_path) if prediction_checkpoint_path.exists() else pd.DataFrame()
    probabilities = pd.read_csv(probability_checkpoint_path) if probability_checkpoint_path.exists() else pd.DataFrame()
    feature_manifest = build_feature_manifest(all_feature_sets)
    basis_diagnostics = pd.concat(diagnostics, ignore_index=True)

    family_summary = (
        results[results["representation"] == "uga_rbf_kernel_mean"]
        .groupby(["benchmark", "family"], dropna=False)
        .agg(
            n_endpoints=("endpoint", "nunique"),
            mean_delta_vs_standard=("delta_vs_standard", "mean"),
            median_delta_vs_standard=("delta_vs_standard", "median"),
            wins_vs_standard=("delta_vs_standard", lambda x: int((x > 0).sum())),
            mean_delta_vs_previous_uga_mean=("delta_vs_previous_uga_mean", "mean"),
            median_delta_vs_previous_uga_mean=("delta_vs_previous_uga_mean", "median"),
            wins_vs_previous_uga_mean=("delta_vs_previous_uga_mean", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )

    atomic_write_csv(results, DATA_DIR / "endpoint_results.csv", index=False)
    predictions.to_csv(DATA_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")
    probabilities.to_csv(DATA_DIR / "oof_probabilities_long.csv.gz", index=False, compression="gzip")
    atomic_write_csv(feature_manifest, DATA_DIR / "feature_manifest.csv", index=False)
    atomic_write_csv(basis_diagnostics, DATA_DIR / "kernel_basis_diagnostics.csv", index=False)
    atomic_write_csv(family_summary, TABLE_DIR / "family_summary.csv", index=False)

    run_metadata = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - start,
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "sbs_model": LOCKED_SBS_MODEL,
        "id_model": LOCKED_ID_MODEL,
        "context_atlas": str(CONTEXT_ATLAS),
        "feature_sources": {
            "mc3": str(FEATURE_DIR),
            "hrd": str(HRD_COHORT),
            "kucab": str(ACTIVE_ROOT / "data" / "raw"),
        },
    }
    atomic_write_json(DATA_DIR / "run_metadata.json", run_metadata)
    write_readme(results, feature_manifest, time.time() - start)

    kme = results[results["representation"] == "uga_rbf_kernel_mean"]
    payload = {
        "experiment_root": str(EXPERIMENT_ROOT),
        "elapsed_seconds": round(time.time() - start, 2),
        "n_endpoint_rows": int(len(results)),
        "n_scored_kme_comparisons": int(kme["delta_vs_standard"].notna().sum()),
        "kme_wins_vs_standard": int((kme["delta_vs_standard"] > 0).sum()),
        "kme_wins_vs_previous_uga_mean": int((kme["delta_vs_previous_uga_mean"] > 0).sum()),
        "mean_kme_delta_vs_standard": float(kme["delta_vs_standard"].mean()),
        "mean_kme_delta_vs_previous_uga_mean": float(kme["delta_vs_previous_uga_mean"].mean()),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
