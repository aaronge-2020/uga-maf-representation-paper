#!/usr/bin/env python3
"""Fast diagnostic and screening benchmark for UGA signal recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from xgboost import XGBClassifier, XGBRegressor


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

ACTIVE_ROOT = EXPERIMENTS_ROOT / "supporting" / "2026_05_14_unified_uga_locked_manuscript_benchmark"
ACTIVE_CODE = ACTIVE_ROOT / "code"
SOURCE_MC3 = ACTIVE_ROOT / "data" / "mc3_source"
FEATURE_DIR = SOURCE_MC3 / "features"
RAW_DIR = SOURCE_MC3 / "raw"

PROJECT_ROOT = find_project_root(EXPERIMENT_ROOT)
RESEARCH_ROOT = PROJECT_ROOT / "cgr_validation_results" / "research"
HRD_COHORT = RESEARCH_ROOT / "assets" / "EXP023_tcga_brca_hrd" / "TCGA-BRCA" / "cohort" / "final_analysis_cohort.tsv"

for path in (ACTIVE_CODE, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from unified_mc3_helpers import (  # noqa: E402
    build_driver_labels,
    build_mc3_candidate_features,
    load_cancer_types,
)
from uga_atlas import build_uga_basis, get_uga_model, load_context_atlas, project_counts_to_uga  # noqa: E402


RANDOM_SEED = 20260515
XGB_N_JOBS = 4
LOCKED_SBS_MODEL = "master_spec_sbs_dbs_d10_dp5"
LOCKED_ID_MODEL = "id83_payload_only_d10_dp5"
CONTEXT_ATLAS = RESEARCH_ROOT / "data" / "EXP022_atlas_genome_wide_45mer_universal_d22.json"

HRD_CONTINUOUS = ["HRD_Score", "eCARD", "HRD_TAI", "HRD_LST", "HRD_LOH", "PARPi7"]
HRD_BINARY = ["hrd_binary_24", "hrd_binary_33", "hrd_binary_42", "parpi7_binary"]
MC3_CLINICAL = {
    "cancer_type_top10": "multiclass",
    "smoking_ever": "binary",
    "high_purity": "binary",
    "high_stage": "binary",
    "os_event": "binary",
}
LIMITING_HRD_ENDPOINTS = ["HRD_Score", "hrd_binary_24", "hrd_binary_33", "hrd_binary_42"]


@dataclass(frozen=True)
class Endpoint:
    name: str
    family: str
    task: str
    y: pd.Series


@dataclass(frozen=True)
class Candidate:
    name: str
    description: str
    features: pd.DataFrame
    promotable: bool
    uses_standard_one_hot: bool
    family: str


def ensure_dirs() -> None:
    for directory in (DATA_DIR, TABLE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


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
    sbs_spec = get_uga_model(sbs_model)
    atlas = None
    if sbs_spec.context_source == "genome_atlas":
        atlas = load_context_atlas(CONTEXT_ATLAS, sbs_spec.d_context)
    sbs_basis, sbs_diag = build_uga_basis(sbs_counts.columns.astype(str).tolist(), sbs_model, atlas=atlas, modality="SBS")
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
        "sbs": pd.concat([burden, sbs_uga], axis=1).fillna(0.0),
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


def parse_sbs96(channel: str) -> tuple[str, str, str, str]:
    ch = str(channel)
    return ch[0], ch[2], ch[4], ch[6]


def parse_id83(channel: str) -> tuple[str, str, int, int]:
    parts = str(channel).split(":")
    if len(parts) < 4:
        return "other", "other", 0, 0
    if parts[0].upper() in {"DEL", "INS"}:
        event = parts[0].upper()
        motif = parts[1].upper().replace("REPEATS", "R").replace("MH", "M")
        length_s = parts[2]
        aux_s = parts[3]
    else:
        length_s, event_s, motif, aux_s = parts[:4]
        event = "DEL" if event_s.lower() == "del" else "INS"
        motif = motif.upper().replace("REPEATS", "R").replace("MH", "M")
    length = 5 if str(length_s).endswith("+") else int(length_s) if str(length_s).isdigit() else 0
    aux = 5 if str(aux_s).endswith("+") else int(aux_s) if str(aux_s).isdigit() else 0
    return event, motif, length, aux


def channel_signal_class(channel: str) -> str:
    ch = str(channel)
    if ch.startswith("SBS96__"):
        left, ref, alt, right = parse_sbs96(ch.replace("SBS96__", "", 1))
        sub = f"{ref}>{alt}"
        if ref == "C" and alt == "T" and right == "G":
            return "SBS CpG C>T"
        if ref == "C" and alt in {"T", "G"} and left == "T" and right in {"A", "T"}:
            return "SBS APOBEC-like C>T/C>G"
        if ref == "C" and alt == "A":
            return "SBS C>A substitution"
        return f"SBS {sub} substitution"
    if ch.startswith("ID83__"):
        event, motif, length, aux = parse_id83(ch.replace("ID83__", "", 1))
        if motif in {"M", "MH"}:
            return "ID microhomology"
        if motif in {"C", "T"} and aux >= 5:
            return "ID homopolymer 5+"
        if motif in {"C", "T"}:
            return "ID homopolymer"
        if motif == "R":
            return "ID repeat"
        if length >= 5:
            return "ID length 5+"
        return f"ID {event}"
    return "burden"


def add_sum_feature(frames: list[pd.Series], counts: pd.DataFrame, columns: list[str], name: str) -> None:
    if columns:
        frames.append(counts.loc[:, columns].sum(axis=1).rename(name))
    else:
        frames.append(pd.Series(0.0, index=counts.index, name=name))


def build_sbs_biology_aggregates(standard_sbs: pd.DataFrame) -> pd.DataFrame:
    counts = strip_prefix(standard_sbs, "SBS96__")
    frames: list[pd.Series] = []
    parsed = {col: parse_sbs96(col) for col in counts.columns}
    substitutions = sorted({f"{ref}>{alt}" for _, ref, alt, _ in parsed.values()})
    for sub in substitutions:
        cols = [col for col, (_, ref, alt, _) in parsed.items() if f"{ref}>{alt}" == sub]
        add_sum_feature(frames, counts, cols, f"sbs_substitution_{sub}")
    for base in "ACGT":
        add_sum_feature(frames, counts, [col for col, (left, _, _, _) in parsed.items() if left == base], f"sbs_left_{base}")
        add_sum_feature(frames, counts, [col for col, (_, _, _, right) in parsed.items() if right == base], f"sbs_right_{base}")
    for gc_count in range(3):
        cols = [col for col, (left, _, _, right) in parsed.items() if int(left in "GC") + int(right in "GC") == gc_count]
        add_sum_feature(frames, counts, cols, f"sbs_flank_gc_count_{gc_count}")
    add_sum_feature(frames, counts, [col for col, (_, ref, alt, right) in parsed.items() if ref == "C" and alt == "T" and right == "G"], "sbs_cpg_ct")
    add_sum_feature(frames, counts, [col for col, (left, ref, alt, right) in parsed.items() if ref == "C" and alt in {"T", "G"} and left == "T" and right in {"A", "T"}], "sbs_apobec_like")
    add_sum_feature(frames, counts, [col for col, (_, ref, alt, _) in parsed.items() if ref == "C" and alt == "A"], "sbs_ca_total")
    for sub in substitutions:
        for base in "ACGT":
            cols = [col for col, (left, ref, alt, _) in parsed.items() if f"{ref}>{alt}" == sub and left == base]
            add_sum_feature(frames, counts, cols, f"sbs_{sub}_left_{base}")
            cols = [col for col, (_, ref, alt, right) in parsed.items() if f"{ref}>{alt}" == sub and right == base]
            add_sum_feature(frames, counts, cols, f"sbs_{sub}_right_{base}")
    return pd.concat(frames, axis=1).fillna(0.0)


def build_id_biology_aggregates(standard_id: pd.DataFrame) -> pd.DataFrame:
    counts = strip_prefix(standard_id, "ID83__")
    frames: list[pd.Series] = []
    parsed = {col: parse_id83(col) for col in counts.columns}
    for event in ["DEL", "INS"]:
        add_sum_feature(frames, counts, [col for col, (ev, _, _, _) in parsed.items() if ev == event], f"id_event_{event.lower()}")
    for motif in ["C", "T", "R", "M"]:
        add_sum_feature(frames, counts, [col for col, (_, mo, _, _) in parsed.items() if mo == motif], f"id_motif_{motif.lower()}")
    for length in range(1, 6):
        add_sum_feature(frames, counts, [col for col, (_, _, le, _) in parsed.items() if le == length], f"id_length_{length if length < 5 else '5plus'}")
        for event in ["DEL", "INS"]:
            cols = [col for col, (ev, _, le, _) in parsed.items() if ev == event and le == length]
            add_sum_feature(frames, counts, cols, f"id_{event.lower()}_length_{length if length < 5 else '5plus'}")
    for aux in range(0, 6):
        label = aux if aux < 5 else "5plus"
        add_sum_feature(frames, counts, [col for col, (_, mo, _, au) in parsed.items() if mo in {"C", "T"} and au == aux], f"id_homopolymer_aux_{label}")
        add_sum_feature(frames, counts, [col for col, (_, mo, _, au) in parsed.items() if mo == "M" and au == aux], f"id_microhomology_aux_{label}")
        add_sum_feature(frames, counts, [col for col, (_, mo, _, au) in parsed.items() if mo == "R" and au == aux], f"id_repeat_aux_{label}")
    add_sum_feature(frames, counts, [col for col, (_, mo, _, au) in parsed.items() if mo in {"C", "T"} and au >= 5], "id_homopolymer_5plus")
    add_sum_feature(frames, counts, [col for col, (_, mo, _, au) in parsed.items() if mo == "M" and au >= 5], "id_microhomology_5plus")
    for event in ["DEL", "INS"]:
        for motif in ["C", "T", "R", "M"]:
            add_sum_feature(frames, counts, [col for col, (ev, mo, _, _) in parsed.items() if ev == event and mo == motif], f"id_{event.lower()}_{motif.lower()}")
    return pd.concat(frames, axis=1).fillna(0.0)


def build_biology_candidates(
    standard_sbs: pd.DataFrame,
    standard_id: pd.DataFrame,
    burden: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    sbs_agg = build_sbs_biology_aggregates(standard_sbs)
    id_agg = build_id_biology_aggregates(standard_id)
    joint = pd.concat([sbs_agg, id_agg], axis=1).fillna(0.0)
    key_cols = [col for col in joint.columns if col in {"sbs_cpg_ct", "sbs_apobec_like", "sbs_ca_total", "id_homopolymer_5plus", "id_microhomology_5plus"}]
    interactions = []
    for col in key_cols:
        interactions.append((joint[col] * burden["log10_sbs_burden"]).rename(f"{col}_x_log10_sbs_burden"))
        interactions.append((joint[col] * burden["log10_id_burden"]).rename(f"{col}_x_log10_id_burden"))
    interaction_df = pd.concat(interactions, axis=1) if interactions else pd.DataFrame(index=burden.index)
    return {
        "bio_sbs": pd.concat([burden, sbs_agg], axis=1).fillna(0.0),
        "bio_id": pd.concat([burden, id_agg], axis=1).fillna(0.0),
        "bio_sbs_id": pd.concat([burden, joint, interaction_df], axis=1).fillna(0.0),
    }


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


def load_kmt2c_endpoint(patients: pd.Index) -> Endpoint:
    cancer_type = load_cancer_types(RAW_DIR, patients)
    driver_labels = build_driver_labels(SOURCE_MC3, patients)
    luad_patients = cancer_type.index[cancer_type == "LUAD"]
    y = driver_labels.loc[driver_labels.index.intersection(luad_patients), "kmt2c_mutated"].astype(int)
    return Endpoint("luad_kmt2c_mutated", "mc3_kmt2c_supportive", "binary", y)


def encode_target(y: pd.Series, task: str) -> tuple[np.ndarray, np.ndarray]:
    if task == "regression":
        return y.astype(float).to_numpy(dtype=np.float64), np.array([], dtype=object)
    if task == "binary":
        return y.astype(int).to_numpy(dtype=np.int32), np.array([0, 1], dtype=object)
    classes = np.array(sorted(y.astype(str).unique()), dtype=object)
    mapping = {value: i for i, value in enumerate(classes)}
    return y.astype(str).map(mapping).astype(int).to_numpy(dtype=np.int32), classes


def make_splits(y: np.ndarray, task: str, folds: int, repeats: int, seed: int) -> list[tuple[int, list[tuple[np.ndarray, np.ndarray]]]]:
    out: list[tuple[int, list[tuple[np.ndarray, np.ndarray]]]] = []
    for repeat in range(repeats):
        rs = seed + repeat * 10_007
        if task == "regression":
            splitter = KFold(n_splits=min(int(folds), len(y)), shuffle=True, random_state=rs)
            out.append((repeat + 1, list(splitter.split(np.zeros(len(y)), y))))
        else:
            n_splits = max(2, min(int(folds), int(pd.Series(y).value_counts().min())))
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
            out.append((repeat + 1, list(splitter.split(np.zeros(len(y)), y))))
    return out


def xgb_classifier_params(y_train: np.ndarray, task: str, n_estimators: int, seed: int, tree_method: str) -> dict[str, object]:
    params: dict[str, object] = {
        "n_estimators": int(n_estimators),
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "reg_lambda": 2.0,
        "reg_alpha": 0.05,
        "random_state": int(seed),
        "n_jobs": XGB_N_JOBS,
        "tree_method": tree_method,
        "verbosity": 0,
    }
    if task == "binary":
        positives = float(np.sum(y_train == 1))
        negatives = float(np.sum(y_train == 0))
        params.update({"objective": "binary:logistic", "eval_metric": "auc", "scale_pos_weight": negatives / max(positives, 1.0)})
    else:
        params.update({"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": int(len(np.unique(y_train)))})
    if tree_method == "gpu_hist":
        params["predictor"] = "gpu_predictor"
    return params


def xgb_regressor_params(n_estimators: int, seed: int, tree_method: str) -> dict[str, object]:
    params: dict[str, object] = {
        "n_estimators": int(n_estimators),
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "reg_lambda": 2.0,
        "reg_alpha": 0.05,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": int(seed),
        "n_jobs": XGB_N_JOBS,
        "tree_method": tree_method,
        "verbosity": 0,
    }
    if tree_method == "gpu_hist":
        params["predictor"] = "gpu_predictor"
    return params


def fit_predict_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    task: str,
    n_estimators: int,
    seed: int,
    tree_method: str,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task == "regression":
            model = XGBRegressor(**xgb_regressor_params(n_estimators, seed, tree_method))
            try:
                model.fit(x_train, y_train)
            except Exception:
                if tree_method == "gpu_hist":
                    model = XGBRegressor(**xgb_regressor_params(n_estimators, seed, "hist"))
                    model.fit(x_train, y_train)
                else:
                    raise
            return model.predict(x_test).astype(np.float64)
        model = XGBClassifier(**xgb_classifier_params(y_train, task, n_estimators, seed, tree_method))
        try:
            model.fit(x_train, y_train)
        except Exception:
            if tree_method == "gpu_hist":
                model = XGBClassifier(**xgb_classifier_params(y_train, task, n_estimators, seed, "hist"))
                model.fit(x_train, y_train)
            else:
                raise
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


def run_endpoint_model(
    endpoint: Endpoint,
    frame: pd.DataFrame,
    *,
    folds: int,
    repeats: int,
    n_estimators: int,
    tree_method: str,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    common = endpoint.y.index.intersection(frame.index)
    y_series = endpoint.y.loc[common]
    x = frame.loc[common].fillna(0.0).to_numpy(dtype=np.float32)
    y, classes = encode_target(y_series, endpoint.task)
    splits = make_splits(y, endpoint.task, folds, repeats, seed)
    if endpoint.task == "regression":
        repeat_preds = []
        for repeat, split_list in splits:
            pred = np.zeros(len(y), dtype=np.float64)
            for fold, (train_idx, test_idx) in enumerate(split_list, start=1):
                pred[test_idx] = fit_predict_fold(
                    x[train_idx],
                    y[train_idx],
                    x[test_idx],
                    task=endpoint.task,
                    n_estimators=n_estimators,
                    seed=seed + repeat * 1000 + fold,
                    tree_method=tree_method,
                )
            repeat_preds.append(pred)
        pred_out = np.mean(repeat_preds, axis=0)
    else:
        n_classes = 2 if endpoint.task == "binary" else len(classes)
        repeat_preds = []
        for repeat, split_list in splits:
            pred = np.zeros((len(y), n_classes), dtype=np.float64)
            for fold, (train_idx, test_idx) in enumerate(split_list, start=1):
                pred[test_idx] = fit_predict_fold(
                    x[train_idx],
                    y[train_idx],
                    x[test_idx],
                    task=endpoint.task,
                    n_estimators=n_estimators,
                    seed=seed + repeat * 1000 + fold,
                    tree_method=tree_method,
                )
            repeat_preds.append(pred)
        pred_out = np.mean(repeat_preds, axis=0)
    metric_name, score, balanced_acc = score_predictions(y, pred_out, endpoint.task, classes)
    metric = {
        "endpoint": endpoint.name,
        "endpoint_family": endpoint.family,
        "task": endpoint.task,
        "metric": metric_name,
        "score": score,
        "balanced_accuracy": balanced_acc,
        "n": int(len(y)),
        "n_features": int(frame.shape[1]),
        "folds": int(folds),
        "repeats": int(repeats),
        "n_estimators": int(n_estimators),
    }
    if endpoint.task == "regression":
        pred_df = pd.DataFrame({"patient_id": common.astype(str), "true_value": y.astype(float), "pred_value": pred_out.astype(float)})
    else:
        pred_df = pd.DataFrame({"patient_id": common.astype(str), "true_value": y.astype(int)})
        for i in range(pred_out.shape[1]):
            pred_df[f"pred_class_{i}"] = pred_out[:, i]
    return metric, pred_df


def run_candidate_screen(
    candidates: list[Candidate],
    endpoints: list[Endpoint],
    standard_frame: pd.DataFrame,
    *,
    folds: int,
    n_estimators: int,
    tree_method: str,
    early_stop_delta: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    standard_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    endpoint_map = {endpoint.name: endpoint for endpoint in endpoints}
    for endpoint in endpoints:
        metric, pred = run_endpoint_model(
            endpoint,
            standard_frame,
            folds=folds,
            repeats=1,
            n_estimators=n_estimators,
            tree_method=tree_method,
            seed=stable_seed("rapid_screen", endpoint.name),
        )
        metric.update({"candidate": "standard_sbs96_id83", "promotable": False, "screen_status": "baseline"})
        standard_rows.append(metric)
        pred.insert(0, "candidate", "standard_sbs96_id83")
        pred.insert(0, "endpoint", endpoint.name)
        prediction_rows.append(pred)
    standard_scores = pd.DataFrame(standard_rows).set_index("endpoint")["score"].to_dict()

    rows = list(standard_rows)
    for candidate in candidates:
        print(f"Tier 2 candidate: {candidate.name} ({candidate.features.shape[1]} features)", flush=True)
        limiting_deltas = []
        status = "screened_full_panel"
        for endpoint_name in LIMITING_HRD_ENDPOINTS:
            endpoint = endpoint_map[endpoint_name]
            metric, pred = run_endpoint_model(
                endpoint,
                candidate.features,
                folds=folds,
                repeats=1,
                n_estimators=n_estimators,
                tree_method=tree_method,
                seed=stable_seed("rapid_screen", endpoint.name),
            )
            delta = metric["score"] - standard_scores[endpoint.name]
            limiting_deltas.append(delta)
            metric.update(
                {
                    "candidate": candidate.name,
                    "candidate_family": candidate.family,
                    "promotable": candidate.promotable,
                    "uses_standard_one_hot": candidate.uses_standard_one_hot,
                    "delta_vs_standard": delta,
                    "screen_status": "limiting_family",
                }
            )
            rows.append(metric)
            pred.insert(0, "candidate", candidate.name)
            pred.insert(0, "endpoint", endpoint.name)
            prediction_rows.append(pred)
        if float(np.mean(limiting_deltas)) < early_stop_delta:
            status = "rejected_after_limiting_hrd"
            print(f"  early reject: mean limiting HRD delta={np.mean(limiting_deltas):.4f}", flush=True)
            for row in rows:
                if row.get("candidate") == candidate.name:
                    row["screen_status"] = status
            continue
        already_done = set(LIMITING_HRD_ENDPOINTS)
        for endpoint in endpoints:
            if endpoint.name in already_done:
                continue
            metric, pred = run_endpoint_model(
                endpoint,
                candidate.features,
                folds=folds,
                repeats=1,
                n_estimators=n_estimators,
                tree_method=tree_method,
                seed=stable_seed("rapid_screen", endpoint.name),
            )
            delta = metric["score"] - standard_scores[endpoint.name]
            metric.update(
                {
                    "candidate": candidate.name,
                    "candidate_family": candidate.family,
                    "promotable": candidate.promotable,
                    "uses_standard_one_hot": candidate.uses_standard_one_hot,
                    "delta_vs_standard": delta,
                    "screen_status": status,
                }
            )
            rows.append(metric)
            pred.insert(0, "candidate", candidate.name)
            pred.insert(0, "endpoint", endpoint.name)
            prediction_rows.append(pred)
        if status == "screened_full_panel":
            for row in rows:
                if row.get("candidate") == candidate.name:
                    row["screen_status"] = status
        cand_rows = [row for row in rows if row.get("candidate") == candidate.name]
        mean_delta = np.mean([row["delta_vs_standard"] for row in cand_rows if "delta_vs_standard" in row])
        print(f"  {status}: mean observed delta={mean_delta:.4f}; endpoints={len(cand_rows)}", flush=True)

    screen = pd.DataFrame(rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    candidate_rows = screen[screen["candidate"] != "standard_sbs96_id83"].copy()
    leaderboard = (
        candidate_rows.groupby(["candidate", "candidate_family", "promotable", "uses_standard_one_hot", "screen_status"], as_index=False)
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
        .sort_values(["promotable", "mean_delta", "n_ge_0_03"], ascending=False)
    )
    return screen, leaderboard, predictions


def recoverability_scores(standard_channels: pd.DataFrame, uga_frame: pd.DataFrame) -> pd.Series:
    common = standard_channels.index.intersection(uga_frame.index)
    x = uga_frame.loc[common].fillna(0.0).to_numpy(dtype=np.float64)
    y = standard_channels.loc[common].fillna(0.0).to_numpy(dtype=np.float64)
    pred = np.zeros_like(y)
    splitter = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    for train_idx, test_idx in splitter.split(x):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        model = Ridge(alpha=10.0)
        model.fit(x_train, y[train_idx])
        pred[test_idx] = model.predict(x_test)
    ss_res = np.sum((y - pred) ** 2, axis=0)
    ss_tot = np.sum((y - y.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = np.zeros(y.shape[1], dtype=np.float64)
    valid = ss_tot > 1e-15
    r2[valid] = 1.0 - (ss_res[valid] / ss_tot[valid])
    return pd.Series(np.clip(r2, -1.0, 1.0), index=standard_channels.columns, name="uga_recoverability_r2")


def endpoint_importance_scores(standard_channels: pd.DataFrame, endpoints: list[Endpoint]) -> pd.Series:
    scores = pd.Series(0.0, index=standard_channels.columns, dtype=float)
    counts = pd.Series(0, index=standard_channels.columns, dtype=int)
    for endpoint in endpoints:
        common = endpoint.y.index.intersection(standard_channels.index)
        if len(common) < 50:
            continue
        x = standard_channels.loc[common].fillna(0.0).to_numpy(dtype=np.float64)
        y, _ = encode_target(endpoint.y.loc[common], endpoint.task)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if endpoint.task == "regression":
                vals = mutual_info_regression(x, y.astype(float), random_state=stable_seed("mi", endpoint.name))
            else:
                vals = mutual_info_classif(x, y.astype(int), random_state=stable_seed("mi", endpoint.name))
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        if vals.max() > 0:
            vals = vals / vals.max()
        scores += pd.Series(vals, index=standard_channels.columns)
        counts += 1
    return (scores / counts.replace(0, np.nan)).fillna(0.0).rename("endpoint_importance")


def run_tier1_diagnostics(
    standard_sbs: pd.DataFrame,
    standard_id: pd.DataFrame,
    standard_sbs_id: pd.DataFrame,
    locked_uga: pd.DataFrame,
    bio_sbs_id: pd.DataFrame,
    endpoints: list[Endpoint],
    *,
    folds: int,
    n_estimators: int,
    tree_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    standard_channels = pd.concat([strip_prefix(standard_sbs, "SBS96__").add_prefix("SBS96__"), strip_prefix(standard_id, "ID83__").add_prefix("ID83__")], axis=1)
    importance = endpoint_importance_scores(standard_channels, endpoints)
    recoverability = recoverability_scores(standard_channels, locked_uga)
    channel_diag = pd.DataFrame(
        {
            "channel": standard_channels.columns,
            "signal_class": [channel_signal_class(col) for col in standard_channels.columns],
            "endpoint_importance": importance.reindex(standard_channels.columns).to_numpy(),
            "uga_recoverability_r2": recoverability.reindex(standard_channels.columns).to_numpy(),
        }
    )
    channel_diag["missing_signal_score"] = channel_diag["endpoint_importance"] * (1.0 - channel_diag["uga_recoverability_r2"].clip(lower=0.0, upper=1.0))
    block_diag = (
        channel_diag.groupby("signal_class", as_index=False)
        .agg(
            n_channels=("channel", "count"),
            mean_endpoint_importance=("endpoint_importance", "mean"),
            mean_uga_recoverability_r2=("uga_recoverability_r2", "mean"),
            mean_missing_signal_score=("missing_signal_score", "mean"),
            max_missing_signal_score=("missing_signal_score", "max"),
        )
        .sort_values(["mean_missing_signal_score", "max_missing_signal_score"], ascending=False)
    )

    top_channels = channel_diag.sort_values("missing_signal_score", ascending=False).head(20)["channel"].tolist()
    sbs_cols = [col for col in standard_sbs_id.columns if str(col).startswith("SBS96__")]
    id_cols = [col for col in standard_sbs_id.columns if str(col).startswith("ID83__")]
    addback_sets = {
        "locked_uga_base": locked_uga,
        "diagnostic_uga_plus_standard_sbs": pd.concat([locked_uga, standard_sbs_id.loc[:, sbs_cols]], axis=1),
        "diagnostic_uga_plus_standard_id": pd.concat([locked_uga, standard_sbs_id.loc[:, id_cols]], axis=1),
        "diagnostic_uga_plus_top20_standard_channels": pd.concat([locked_uga, standard_sbs_id.loc[:, top_channels]], axis=1),
        "candidate_uga_plus_biology_aggregates": pd.concat([locked_uga, bio_sbs_id.drop(columns=[col for col in locked_uga.columns if col in bio_sbs_id.columns], errors="ignore")], axis=1),
    }
    rows: list[dict[str, object]] = []
    for feature_name, frame in addback_sets.items():
        for endpoint in [ep for ep in endpoints if ep.name in LIMITING_HRD_ENDPOINTS]:
            metric, _ = run_endpoint_model(
                endpoint,
                frame,
                folds=folds,
                repeats=1,
                n_estimators=n_estimators,
                tree_method=tree_method,
                seed=stable_seed("tier1", endpoint.name),
            )
            metric.update(
                {
                    "feature_set": feature_name,
                    "diagnostic_only": feature_name.startswith("diagnostic_"),
                    "uses_standard_one_hot": "standard" in feature_name,
                }
            )
            rows.append(metric)
    addback = pd.DataFrame(rows)
    standard_rows = []
    for endpoint in [ep for ep in endpoints if ep.name in LIMITING_HRD_ENDPOINTS]:
        metric, _ = run_endpoint_model(
            endpoint,
            standard_sbs_id,
            folds=folds,
            repeats=1,
            n_estimators=n_estimators,
            tree_method=tree_method,
            seed=stable_seed("tier1", endpoint.name),
        )
        metric.update({"feature_set": "standard_sbs96_id83", "diagnostic_only": False, "uses_standard_one_hot": True})
        standard_rows.append(metric)
    addback = pd.concat([pd.DataFrame(standard_rows), addback], ignore_index=True)
    standards = addback[addback["feature_set"] == "standard_sbs96_id83"][["endpoint", "score"]].rename(columns={"score": "standard_score"})
    addback = addback.merge(standards, on="endpoint", how="left")
    addback["delta_vs_standard"] = addback["score"] - addback["standard_score"]
    return channel_diag.sort_values("missing_signal_score", ascending=False), block_diag, addback


def candidate_passes_gate(screen: pd.DataFrame, candidate: str) -> bool:
    rows = screen[(screen["candidate"] == candidate) & (screen["promotable"] == True)].copy()
    rows = rows[rows["screen_status"] == "screened_full_panel"]
    if rows.empty:
        return False
    families = ["hrd", "mc3_clinical", "all"]
    for family in families:
        subset = rows if family == "all" else rows[rows["endpoint_family"] == family]
        if len(subset) < 3:
            continue
        mean_delta = float(subset["delta_vs_standard"].mean())
        n_ge = int((subset["delta_vs_standard"] >= 0.03).sum())
        min_delta = float(subset["delta_vs_standard"].min())
        if mean_delta >= 0.03 and n_ge >= 3 and min_delta >= -0.02:
            return True
    return False


def bootstrap_delta(
    endpoint: Endpoint,
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    merged = pred_a.merge(pred_b, on=["patient_id", "true_value"], suffixes=("_a", "_b"))
    y = merged["true_value"].to_numpy()
    rng = np.random.default_rng(seed)
    deltas = np.zeros(n_bootstrap, dtype=np.float64)
    if endpoint.task == "regression":
        pred_a_arr = merged["pred_value_a"].to_numpy(dtype=float)
        pred_b_arr = merged["pred_value_b"].to_numpy(dtype=float)
        for i in range(n_bootstrap):
            idx = rng.choice(np.arange(len(y)), size=len(y), replace=True)
            deltas[i] = spearmanr(y[idx], pred_a_arr[idx])[0] - spearmanr(y[idx], pred_b_arr[idx])[0]
    else:
        class_cols_a = [col for col in merged.columns if col.startswith("pred_class_") and col.endswith("_a")]
        class_cols_b = [col for col in merged.columns if col.startswith("pred_class_") and col.endswith("_b")]
        proba_a = merged[class_cols_a].to_numpy(dtype=float)
        proba_b = merged[class_cols_b].to_numpy(dtype=float)
        strata = [np.flatnonzero(y == value) for value in np.unique(y)]
        for i in range(n_bootstrap):
            idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
            if endpoint.task == "binary":
                deltas[i] = roc_auc_score(y[idx], proba_a[idx, 1]) - roc_auc_score(y[idx], proba_b[idx, 1])
            else:
                classes = np.arange(proba_a.shape[1])
                y_bin = label_binarize(y[idx], classes=classes)
                deltas[i] = roc_auc_score(y_bin, proba_a[idx], average="macro") - roc_auc_score(y_bin, proba_b[idx], average="macro")
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    p_lower = (np.sum(deltas <= 0.0) + 1.0) / (n_bootstrap + 1.0)
    p_upper = (np.sum(deltas >= 0.0) + 1.0) / (n_bootstrap + 1.0)
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


def run_confirmation(
    finalist_names: list[str],
    candidates: list[Candidate],
    endpoints: list[Endpoint],
    standard_frame: pd.DataFrame,
    *,
    folds: int,
    repeats: int,
    n_estimators: int,
    tree_method: str,
    bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not finalist_names:
        empty = pd.DataFrame(
            columns=[
                "candidate",
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
        return empty, empty.copy(), pd.DataFrame()
    candidate_map = {candidate.name: candidate for candidate in candidates}
    metric_rows: list[dict[str, object]] = []
    pred_rows: list[pd.DataFrame] = []
    for endpoint in endpoints:
        metric, pred = run_endpoint_model(
            endpoint,
            standard_frame,
            folds=folds,
            repeats=repeats,
            n_estimators=n_estimators,
            tree_method=tree_method,
            seed=stable_seed("focused_confirmation", endpoint.name),
        )
        metric.update({"candidate": "standard_sbs96_id83"})
        metric_rows.append(metric)
        pred.insert(0, "candidate", "standard_sbs96_id83")
        pred.insert(0, "endpoint", endpoint.name)
        pred_rows.append(pred)
        for candidate_name in finalist_names:
            candidate = candidate_map[candidate_name]
            metric, pred = run_endpoint_model(
                endpoint,
                candidate.features,
                folds=folds,
                repeats=repeats,
                n_estimators=n_estimators,
                tree_method=tree_method,
                seed=stable_seed("focused_confirmation", endpoint.name),
            )
            metric.update({"candidate": candidate.name})
            metric_rows.append(metric)
            pred.insert(0, "candidate", candidate.name)
            pred.insert(0, "endpoint", endpoint.name)
            pred_rows.append(pred)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(pred_rows, ignore_index=True)
    standards = metrics[metrics["candidate"] == "standard_sbs96_id83"][["endpoint", "score"]].rename(columns={"score": "standard_score"})
    tests: list[dict[str, object]] = []
    endpoint_map = {endpoint.name: endpoint for endpoint in endpoints}
    for candidate_name in finalist_names:
        cand_metrics = metrics[metrics["candidate"] == candidate_name].merge(standards, on="endpoint", how="left")
        for _, row in cand_metrics.iterrows():
            endpoint = endpoint_map[str(row["endpoint"])]
            pred_a = predictions[(predictions["candidate"] == candidate_name) & (predictions["endpoint"] == endpoint.name)].drop(columns=["candidate", "endpoint"])
            pred_b = predictions[(predictions["candidate"] == "standard_sbs96_id83") & (predictions["endpoint"] == endpoint.name)].drop(columns=["candidate", "endpoint"])
            p_value, ci_low, ci_high = bootstrap_delta(
                endpoint,
                pred_a,
                pred_b,
                n_bootstrap=bootstrap,
                seed=stable_seed("bootstrap", candidate_name, endpoint.name),
            )
            tests.append(
                {
                    "candidate": candidate_name,
                    "endpoint": endpoint.name,
                    "endpoint_family": endpoint.family,
                    "metric": row["metric"],
                    "standard_score": float(row["standard_score"]),
                    "candidate_score": float(row["score"]),
                    "delta_vs_standard": float(row["score"] - row["standard_score"]),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                }
            )
    tests_df = pd.DataFrame(tests)
    if not tests_df.empty:
        tests_df["q_value"] = bh_q_values(tests_df["p_value"])
    return metrics, tests_df, predictions


def write_html_table(df: pd.DataFrame, path: Path, title: str, footnote: str) -> None:
    headers = "".join(f"<th>{escape(str(col))}</th>" for col in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                text = f"{value:.6g}"
            elif pd.isna(value):
                text = "NA"
            else:
                text = str(value)
            cells.append(f"<td>{escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#111}}
table{{border-collapse:collapse;width:100%;font-size:12px}}
caption{{caption-side:top;text-align:left;font-weight:700;margin-bottom:8px}}
thead th{{border-top:1px solid #111;border-bottom:1px solid #111;padding:6px 8px;text-align:left}}
tbody td{{border-bottom:0.5px solid #bbb;padding:5px 8px;text-align:left}}
tfoot td{{border-top:1px solid #111;padding:6px 8px;font-size:11px;line-height:1.35}}
</style>
</head>
<body>
<table>
<caption>{escape(title)}</caption>
<thead><tr>{headers}</tr></thead>
<tbody>{''.join(rows)}</tbody>
<tfoot><tr><td colspan="{len(df.columns)}">{escape(footnote)}</td></tr></tfoot>
</table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_readme(
    *,
    metadata: dict[str, object],
    block_diag: pd.DataFrame,
    leaderboard: pd.DataFrame,
    confirm_tests: pd.DataFrame,
    finalists: list[str],
) -> None:
    top_blocks = block_diag.head(5)
    top_block_text = "; ".join(
        f"{row.signal_class} (mean missing-signal score {row.mean_missing_signal_score:.3f})"
        for row in top_blocks.itertuples(index=False)
    )
    promotable = leaderboard[leaderboard["promotable"] == True].copy()
    best_text = "No promotable candidate was fully screened."
    if not promotable.empty:
        best = promotable.sort_values(["mean_delta", "n_ge_0_03"], ascending=False).iloc[0]
        best_text = (
            f"The best promotable rapid-screen candidate was {best['candidate']} "
            f"with mean delta {best['mean_delta']:.4f} across {int(best['n_endpoints'])} evaluated endpoints."
        )
    if finalists:
        confirm_text = f"Focused confirmation was run for {', '.join(finalists)}."
    else:
        confirm_text = "No candidate passed the rapid promotion gate, so focused confirmation was not run."
    leader_cols = ["candidate", "mean_delta", "min_delta", "max_delta", "n_positive", "n_ge_0_03", "n_lt_minus_0_02"]
    leader_view = leaderboard.loc[:, leader_cols].head(8).copy() if not leaderboard.empty else pd.DataFrame(columns=leader_cols)
    if not leader_view.empty:
        leader_view = leader_view.rename(
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
        leaderboard_markdown = leader_view.to_markdown(index=False, floatfmt=".4f")
    else:
        leaderboard_markdown = "No rapid-screen candidates were evaluated."
    lines = [
        "# Fast Defensible UGA Signal-Recovery Benchmark",
        "",
        "## Research Question",
        "Which biological signals used by Standard SBS96/ID83 features are poorly recovered by the current locked UGA representation, and can targeted pure replacement feature schemes recover enough signal to justify full manuscript validation?",
        "",
        "## Methods",
        "The benchmark used retained TCGA/MC3 feature matrices and labels with fixed patients, paired folds, fixed seeds, identical XGBoost learners, and unchanged Standard SBS96+ID83 baselines. Tier 1 ranked missing Standard signals by endpoint association and recoverability from locked UGA features, then ran diagnostic add-back tests. Tier 2 screened pure replacement candidates on all eligible samples with one five-fold paired split and stopped candidates early when the limiting HRD family lost by more than 0.02 mean metric units. Tier 3 was reserved for candidates passing a predeclared promotion gate of mean delta at least 0.03, at least three endpoint gains of 0.03 or greater, and no endpoint loss worse than -0.02.",
        "",
        "## Key Numerical Findings",
        f"The highest ranked missing signal blocks were {top_block_text}.",
        best_text,
        confirm_text,
        "",
        "## Rapid Screen Summary",
        leaderboard_markdown,
        "",
        "## File Inventory",
        "| File | Purpose |",
        "|---|---|",
        "| `data/tier1_missing_signal_channels.csv` | Channel-level endpoint association, UGA recoverability, and missing-signal score. |",
        "| `data/tier1_missing_signal_blocks.csv` | Missing-signal summary by biological channel block. |",
        "| `data/tier1_addback_diagnostics.csv` | Diagnostic-only Standard add-back tests and biology-aggregate rescue tests. |",
        "| `data/tier2_rapid_candidate_screen.csv` | Rapid paired endpoint screen for Standard and candidate representations. |",
        "| `data/tier2_candidate_leaderboard.csv` | Candidate-level rapid-screen summary and promotion-gate inputs. |",
        "| `data/tier2_oof_predictions.csv` | Rapid-screen out-of-fold predictions. |",
        "| `data/tier3_focused_confirmation_metrics.csv` | Focused confirmation metrics for finalists, if any passed the gate. |",
        "| `data/tier3_focused_confirmation_tests.csv` | Paired bootstrap confidence intervals, p values, and q values for finalists. |",
        "| `data/model_manifest.csv` | Candidate feature definitions and promotion eligibility. |",
        "| `tables/*.html` | Standalone manuscript-ready HTML tables. |",
        "| `code/run_fast_defensible_signal_recovery_benchmark.py` | Reproducible benchmark script. |",
        "",
        "## Reproducibility",
        f"Executed at {metadata['executed_at_utc']} with random seed {metadata['random_seed']}, rapid folds={metadata['rapid_folds']}, rapid trees={metadata['rapid_trees']}, tree_method={metadata['tree_method']}, and elapsed runtime {float(metadata['elapsed_seconds']) / 60.0:.1f} minutes.",
        "",
    ]
    (EXPERIMENT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier1-folds", type=int, default=3)
    parser.add_argument("--tier1-trees", type=int, default=60)
    parser.add_argument("--rapid-folds", type=int, default=5)
    parser.add_argument("--rapid-trees", type=int, default=100)
    parser.add_argument("--tree-method", default="gpu_hist")
    parser.add_argument("--early-stop-delta", type=float, default=-0.02)
    parser.add_argument("--confirm-folds", type=int, default=5)
    parser.add_argument("--confirm-repeats", type=int, default=3)
    parser.add_argument("--confirm-trees", type=int, default=250)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--max-finalists", type=int, default=2)
    parser.add_argument("--xgb-n-jobs", type=int, default=4)
    args = parser.parse_args()

    ensure_dirs()
    global XGB_N_JOBS
    XGB_N_JOBS = int(args.xgb_n_jobs)
    start = time.perf_counter()
    standard_sbs, standard_id, standard_sbs_id, burden = load_feature_matrices()
    locked = build_mc3_candidate_features(standard_sbs, standard_id, burden, LOCKED_SBS_MODEL, LOCKED_ID_MODEL)
    registered_proxy = build_registered_uga_features(
        standard_sbs,
        standard_id,
        burden,
        sbs_model=LOCKED_SBS_MODEL,
        id_model="id83_proxy_d10_dp5",
        prefix="proxy",
    )
    registered_token = build_registered_uga_features(
        standard_sbs,
        standard_id,
        burden,
        sbs_model=LOCKED_SBS_MODEL,
        id_model="id83_token_pair_d10_dp5",
        prefix="token",
    )
    label_proxy = build_registered_uga_features(
        standard_sbs,
        standard_id,
        burden,
        sbs_model="label_sbs96_d10",
        id_model="id83_proxy_d10_dp5",
        prefix="label_proxy",
    )
    bio = build_biology_candidates(standard_sbs, standard_id, burden)

    locked_separate = locked["uga_combined_separate"]
    locked_pooled = locked.get("uga_combined_pooled", locked_separate)
    proxy_separate = registered_proxy["separate"]
    token_separate = registered_token["separate"]
    label_proxy_separate = label_proxy["separate"]
    bio_sbs_id = bio["bio_sbs_id"]
    locked_plus_bio = pd.concat([locked_separate, bio_sbs_id.drop(columns=[col for col in locked_separate.columns if col in bio_sbs_id.columns], errors="ignore")], axis=1).fillna(0.0)
    proxy_plus_bio = pd.concat([proxy_separate, bio_sbs_id.drop(columns=[col for col in proxy_separate.columns if col in bio_sbs_id.columns], errors="ignore")], axis=1).fillna(0.0)

    patients = burden.index.astype(str)
    endpoints = [*load_hrd_endpoints(), *load_mc3_clinical_endpoints(), load_kmt2c_endpoint(pd.Index(patients))]
    endpoint_summary = pd.DataFrame(
        [
            {
                "endpoint": endpoint.name,
                "endpoint_family": endpoint.family,
                "task": endpoint.task,
                "n": int(len(endpoint.y)),
                "positive_or_classes": int(endpoint.y.sum()) if endpoint.task == "binary" else int(endpoint.y.nunique()) if endpoint.task == "multiclass" else np.nan,
            }
            for endpoint in endpoints
        ]
    )
    endpoint_summary.to_csv(DATA_DIR / "endpoint_manifest.csv", index=False)

    candidates = [
        Candidate("locked_uga_pooled", "Locked manuscript UGA pooled SBS+ID coordinate projection.", locked_pooled, True, False, "current_locked_uga"),
        Candidate("locked_uga_separate", "Locked manuscript UGA with SBS and ID kept as separate blocks.", locked_separate, True, False, "current_locked_uga"),
        Candidate("id_proxy_uga_separate", "Locked SBS UGA plus ID83 repeat/microhomology proxy UGA as separate blocks.", proxy_separate, True, False, "registered_uga_proxy"),
        Candidate("id_token_uga_separate", "Locked SBS UGA plus deterministic token-pair ID83 UGA as separate blocks.", token_separate, True, False, "registered_uga_proxy"),
        Candidate("label_sbs_id_proxy_uga", "Label-derived SBS96 UGA plus ID83 proxy UGA as separate blocks.", label_proxy_separate, True, False, "registered_uga_proxy"),
        Candidate("biology_aggregates_sbs_id", "Grouped SBS motif and ID biology aggregates without SBS96/ID83 one-hot channels.", bio_sbs_id, True, False, "biology_aggregate"),
        Candidate("locked_uga_plus_biology_aggregates", "Locked separate-block UGA plus grouped SBS motif and ID biology aggregates.", locked_plus_bio, True, False, "uga_biology_hybrid"),
        Candidate("id_proxy_uga_plus_biology_aggregates", "ID proxy separate-block UGA plus grouped SBS motif and ID biology aggregates.", proxy_plus_bio, True, False, "uga_biology_hybrid"),
    ]
    model_manifest = pd.DataFrame(
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
        + [
            {
                "candidate": "standard_sbs96_id83",
                "candidate_family": "standard_baseline",
                "description": "Standard SBS96+ID83 mutation-channel feature matrix with shared burden covariates.",
                "n_features": int(standard_sbs_id.shape[1]),
                "promotable": False,
                "uses_standard_one_hot": True,
            }
        ]
    )
    model_manifest.to_csv(DATA_DIR / "model_manifest.csv", index=False)

    print("Tier 1 diagnostics", flush=True)
    channel_diag, block_diag, addback = run_tier1_diagnostics(
        standard_sbs,
        standard_id,
        standard_sbs_id,
        locked_separate,
        bio_sbs_id,
        endpoints,
        folds=args.tier1_folds,
        n_estimators=args.tier1_trees,
        tree_method=args.tree_method,
    )
    channel_diag.to_csv(DATA_DIR / "tier1_missing_signal_channels.csv", index=False)
    block_diag.to_csv(DATA_DIR / "tier1_missing_signal_blocks.csv", index=False)
    addback.to_csv(DATA_DIR / "tier1_addback_diagnostics.csv", index=False)

    print("Tier 2 rapid candidate screen", flush=True)
    screen, leaderboard, rapid_predictions = run_candidate_screen(
        candidates,
        endpoints,
        standard_sbs_id,
        folds=args.rapid_folds,
        n_estimators=args.rapid_trees,
        tree_method=args.tree_method,
        early_stop_delta=args.early_stop_delta,
    )
    screen.to_csv(DATA_DIR / "tier2_rapid_candidate_screen.csv", index=False)
    leaderboard.to_csv(DATA_DIR / "tier2_candidate_leaderboard.csv", index=False)
    rapid_predictions.to_csv(DATA_DIR / "tier2_oof_predictions.csv", index=False)

    finalists = [
        str(candidate)
        for candidate in leaderboard["candidate"].tolist()
        if candidate_passes_gate(screen, str(candidate))
    ][: int(args.max_finalists)]
    print(f"Finalists: {finalists if finalists else 'none'}", flush=True)
    confirm_metrics, confirm_tests, confirm_predictions = run_confirmation(
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
    confirm_metrics.to_csv(DATA_DIR / "tier3_focused_confirmation_metrics.csv", index=False)
    confirm_tests.to_csv(DATA_DIR / "tier3_focused_confirmation_tests.csv", index=False)
    if not confirm_predictions.empty:
        confirm_predictions.to_csv(DATA_DIR / "tier3_oof_predictions.csv", index=False)

    write_html_table(
        block_diag.head(25),
        TABLE_DIR / "table1_tier1_missing_signal_blocks.html",
        "Table 1. Tier 1 missing Standard signal blocks.",
        "Missing-signal score is endpoint association multiplied by one minus UGA recoverability from the current locked UGA representation.",
    )
    write_html_table(
        addback,
        TABLE_DIR / "table2_tier1_addback_diagnostics.html",
        "Table 2. Tier 1 diagnostic add-back tests.",
        "Standard add-back rows are diagnostic-only and are not eligible for promotion as UGA replacements.",
    )
    write_html_table(
        leaderboard,
        TABLE_DIR / "table3_tier2_candidate_leaderboard.html",
        "Table 3. Tier 2 rapid pure-replacement candidate screen.",
        "Rapid screening uses all eligible samples with paired folds and fixed XGBoost settings. Screening results are exploratory.",
    )
    write_html_table(
        confirm_tests,
        TABLE_DIR / "table4_tier3_focused_confirmation.html",
        "Table 4. Tier 3 focused confirmation for promoted finalists.",
        "Focused confirmation uses paired out-of-fold predictions, bootstrap confidence intervals, and Benjamini-Hochberg q values. Empty tables indicate that no candidate passed the rapid promotion gate.",
    )

    metadata = {
        "experiment": EXPERIMENT_ROOT.name,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "random_seed": RANDOM_SEED,
        "rapid_folds": int(args.rapid_folds),
        "rapid_trees": int(args.rapid_trees),
        "tier1_folds": int(args.tier1_folds),
        "tier1_trees": int(args.tier1_trees),
        "confirm_folds": int(args.confirm_folds),
        "confirm_repeats": int(args.confirm_repeats),
        "confirm_trees": int(args.confirm_trees),
        "bootstrap": int(args.bootstrap),
        "tree_method": str(args.tree_method),
        "early_stop_delta": float(args.early_stop_delta),
        "n_endpoints": int(len(endpoints)),
        "n_candidates": int(len(candidates)),
        "finalists": finalists,
    }
    (DATA_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_readme(metadata=metadata, block_diag=block_diag, leaderboard=leaderboard, confirm_tests=confirm_tests, finalists=finalists)
    print(json.dumps({"elapsed_seconds": round(metadata["elapsed_seconds"], 1), "finalists": finalists, "n_endpoints": len(endpoints)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
