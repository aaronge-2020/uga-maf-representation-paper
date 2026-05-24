#!/usr/bin/env python3
"""Shared MC3 helpers for the locked unified UGA benchmark."""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


RANDOM_SEED = 20260514

DRIVER_GENES = [
    "APC",
    "ARID1A",
    "ATM",
    "BAP1",
    "BRAF",
    "BRCA1",
    "BRCA2",
    "CDH1",
    "CDKN2A",
    "CTNNB1",
    "EGFR",
    "EP300",
    "FBXW7",
    "FGFR3",
    "GATA3",
    "IDH1",
    "IDH2",
    "KDM6A",
    "KEAP1",
    "KMT2C",
    "KMT2D",
    "KRAS",
    "MAP2K1",
    "MET",
    "MTOR",
    "NF1",
    "NOTCH1",
    "NRAS",
    "PIK3CA",
    "PIK3R1",
    "PTEN",
    "RB1",
    "SETD2",
    "SMAD4",
    "SMARCA4",
    "TERT",
    "TP53",
    "TSC1",
    "TSC2",
    "VHL",
]

GENE_GROUPS = {
    "brca_pathway_mutated": ["BRCA1", "BRCA2", "PALB2", "ATM", "ATR", "CHEK2", "RAD51C", "RAD51D"],
    "mmr_pathway_mutated": ["MLH1", "MSH2", "MSH6", "PMS2"],
    "pole_pold1_mutated": ["POLE", "POLD1"],
    "ras_pathway_mutated": ["KRAS", "NRAS", "HRAS", "BRAF", "MAP2K1"],
    "pi3k_pathway_mutated": ["PIK3CA", "PIK3R1", "PTEN", "AKT1", "MTOR", "TSC1", "TSC2"],
    "chromatin_mutated": ["ARID1A", "KMT2C", "KMT2D", "KDM6A", "BAP1", "SMARCA4", "EP300"],
}

FUNCTIONAL_CLASSES = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Nonstop_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Splice_Site",
    "Translation_Start_Site",
}


def find_cgr_root() -> Path:
    path = Path(__file__).resolve()
    for candidate in [path.parent, *path.parents]:
        if (candidate / "uga_atlas" / "models.py").is_file() and (candidate / "data" / "Signatures").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate cgr_validation root from {path}")


CGR_ROOT = find_cgr_root()
if str(CGR_ROOT) not in sys.path:
    sys.path.insert(0, str(CGR_ROOT))

from uga_atlas import build_uga_basis, get_uga_model, load_context_atlas, project_counts_to_uga  # noqa: E402


RESEARCH_ROOT = CGR_ROOT / "cgr_validation_results" / "research"
CONTEXT_ATLAS = RESEARCH_ROOT / "data" / "EXP022_atlas_genome_wide_45mer_universal_d22.json"


def strip_prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    out = df[cols].copy()
    out.columns = [str(c).replace(prefix, "", 1) for c in cols]
    return out


def build_mc3_candidate_features(
    standard_sbs: pd.DataFrame,
    standard_id: pd.DataFrame,
    burden: pd.DataFrame,
    sbs_model: str,
    id_model: str,
) -> dict[str, pd.DataFrame]:
    sbs_counts = strip_prefix_columns(standard_sbs, "SBS96__")
    id_counts = strip_prefix_columns(standard_id, "ID83__")
    sbs_spec = get_uga_model(sbs_model)
    atlas = load_context_atlas(CONTEXT_ATLAS, sbs_spec.d_context)
    sbs_basis, sbs_diag = build_uga_basis(sbs_counts.columns.tolist(), sbs_model, atlas=atlas, modality="SBS")
    id_basis, id_diag = build_uga_basis(id_counts.columns.tolist(), id_model)
    sbs_uga = project_counts_to_uga(
        sbs_counts,
        sbs_basis,
        sbs_diag["UGA_Encoded"].to_numpy(dtype=bool),
        "uga_sbs",
    )
    id_uga = project_counts_to_uga(
        id_counts,
        id_basis,
        id_diag["UGA_Encoded"].to_numpy(dtype=bool),
        "uga_id",
    )
    out = {
        "uga_sbs": pd.concat([burden, sbs_uga], axis=1),
        "uga_combined_separate": pd.concat([burden, sbs_uga, id_uga], axis=1),
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
        pooled = project_counts_to_uga(pooled_counts, pooled_basis, pooled_valid, "uga_pooled")
        out["uga_combined_pooled"] = pd.concat([burden, pooled], axis=1)
    return out


def load_cancer_types(raw_dir: Path, patients: pd.Index) -> pd.Series:
    cdr = pd.read_excel(raw_dir / "TCGA-CDR-SupplementalTableS1.xlsx", usecols=["bcr_patient_barcode", "type"])
    cdr["patient"] = cdr["bcr_patient_barcode"].astype(str).str[:12]
    cancer_type = cdr.drop_duplicates("patient").set_index("patient")["type"].astype(str)
    return cancer_type.reindex(patients)


def build_driver_labels(source_dir: Path, patients: pd.Index) -> pd.DataFrame:
    out_path = source_dir / "driver_gene_labels_functional.csv"
    if out_path.exists():
        return pd.read_csv(out_path, index_col=0).reindex(patients).fillna(0).astype(int)

    raw_dir = source_dir / "raw"
    all_genes = set(DRIVER_GENES)
    for genes in GENE_GROUPS.values():
        all_genes.update(genes)
    patient_set = set(patients.astype(str))
    hits = {gene: set() for gene in sorted(all_genes)}
    usecols = ["Tumor_Sample_Barcode", "Hugo_Symbol", "Variant_Classification"]
    total = 0
    t0 = time.time()
    for chunk_idx, chunk in enumerate(
        pd.read_csv(raw_dir / "mc3.v0.2.8.PUBLIC.maf.gz", sep="\t", usecols=usecols, dtype=str, chunksize=250_000),
        start=1,
    ):
        total += len(chunk)
        chunk = chunk[chunk["Variant_Classification"].isin(FUNCTIONAL_CLASSES)].copy()
        chunk["patient"] = chunk["Tumor_Sample_Barcode"].astype(str).str[:12]
        chunk = chunk[chunk["patient"].isin(patient_set)]
        chunk["gene"] = chunk["Hugo_Symbol"].fillna("").astype(str).str.upper()
        chunk = chunk[chunk["gene"].isin(all_genes)]
        for gene, sub in chunk.groupby("gene"):
            hits[gene].update(sub["patient"].unique())
        if chunk_idx % 10 == 0:
            print(f"  scanned {total:,} MAF rows in {time.time() - t0:.1f}s", flush=True)

    labels = pd.DataFrame(index=pd.Index(patients.astype(str), name="patient"))
    for gene in DRIVER_GENES:
        labels[f"{gene.lower()}_mutated"] = [1 if patient in hits.get(gene, set()) else 0 for patient in labels.index]
    for group_name, genes in GENE_GROUPS.items():
        group_hits = set()
        for gene in genes:
            group_hits.update(hits.get(gene, set()))
        labels[group_name] = [1 if patient in group_hits else 0 for patient in labels.index]
    labels.to_csv(out_path)
    return labels


def total_burden(features: dict[str, pd.DataFrame]) -> pd.Series:
    burden = features["burden_only"]
    sbs = np.power(10.0, burden["log10_sbs_burden"].astype(float)) - 1.0
    indel = np.power(10.0, burden["log10_id_burden"].astype(float)) - 1.0
    return pd.Series(sbs + indel, index=burden.index, name="total_mutation_burden")


def patients_for_task(task: pd.Series, driver_labels: pd.DataFrame, cancer_type: pd.Series, features: dict[str, pd.DataFrame]) -> pd.Index:
    scenario = str(task["scenario"])
    if scenario == "full":
        return driver_labels.index
    if scenario == "low_burden_q50":
        burden = total_burden(features)
        return burden[burden <= burden.quantile(0.50)].index
    if scenario.startswith("within_cancer_type_"):
        ct = str(task["cancer_type"])
        return cancer_type.index[cancer_type == ct]
    raise ValueError(f"Unknown scenario: {scenario}")


def xgb_params(y_train: np.ndarray, *, n_estimators: int, seed: int, tree_method: str) -> dict[str, object]:
    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    params: dict[str, object] = {
        "n_estimators": int(n_estimators),
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "reg_lambda": 2.0,
        "reg_alpha": 0.05,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": negatives / max(positives, 1.0),
        "random_state": int(seed),
        "n_jobs": 1,
        "tree_method": tree_method,
        "verbosity": 0,
    }
    if tree_method == "gpu_hist":
        params["predictor"] = "gpu_predictor"
    return params


def fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, *, n_estimators: int, seed: int, tree_method: str) -> np.ndarray:
    model = XGBClassifier(**xgb_params(y_train, n_estimators=n_estimators, seed=seed, tree_method=tree_method))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(X_train, y_train)
        except Exception:
            if tree_method == "gpu_hist":
                model = XGBClassifier(**xgb_params(y_train, n_estimators=n_estimators, seed=seed, tree_method="hist"))
                model.fit(X_train, y_train)
            else:
                raise
    return model.predict_proba(X_test)[:, 1].astype(np.float64)


def run_oof(
    y: pd.Series,
    X_df: pd.DataFrame,
    *,
    folds: int,
    n_estimators: int,
    seed: int,
    tree_method: str,
) -> tuple[float, float, np.ndarray, list[dict[str, object]]]:
    common = y.index.intersection(X_df.index)
    y = y.loc[common].astype(int)
    X = X_df.loc[common].to_numpy(dtype=np.float32)
    y_arr = y.to_numpy(dtype=np.int32)
    n_splits = max(2, min(int(folds), int(pd.Series(y_arr).value_counts().min())))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = np.zeros(len(y_arr), dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y_arr), start=1):
        fold_proba = fit_predict(
            X[train_idx],
            y_arr[train_idx],
            X[test_idx],
            n_estimators=n_estimators,
            seed=seed + fold * 101,
            tree_method=tree_method,
        )
        proba[test_idx] = fold_proba
        pred = (fold_proba >= 0.5).astype(int)
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "fold_auroc": float(roc_auc_score(y_arr[test_idx], fold_proba)),
                "fold_balanced_accuracy": float(balanced_accuracy_score(y_arr[test_idx], pred)),
            }
        )
    pred_all = (proba >= 0.5).astype(int)
    return float(roc_auc_score(y_arr, proba)), float(balanced_accuracy_score(y_arr, pred_all)), proba, fold_rows


def stratified_bootstrap_delta(y: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray, *, n_bootstrap: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    deltas = np.zeros(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample_idx = np.concatenate([rng.choice(idx0, len(idx0), replace=True), rng.choice(idx1, len(idx1), replace=True)])
        deltas[i] = roc_auc_score(y[sample_idx], proba_a[sample_idx]) - roc_auc_score(y[sample_idx], proba_b[sample_idx])
    p_lower = (np.sum(deltas <= 0.0) + 1.0) / (n_bootstrap + 1.0)
    p_upper = (np.sum(deltas >= 0.0) + 1.0) / (n_bootstrap + 1.0)
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return float(min(1.0, 2.0 * min(p_lower, p_upper))), float(ci_low), float(ci_high)


def bh_q_values(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 1.0
    for i in range(len(p_values) - 1, -1, -1):
        running = min(running, ranked[i] * len(p_values) / (i + 1))
        adjusted[order[i]] = running
    return np.minimum(adjusted, 1.0)
