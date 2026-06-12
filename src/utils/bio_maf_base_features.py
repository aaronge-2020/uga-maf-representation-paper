"""Shared fixed Bio MAF feature helpers used by the Bio MAF v4 builder.

This module keeps only the reusable fixed biological feature construction and
nested feature-selection utilities required by Bio MAF v4 diagnostics. It uses
the standalone vendored resource paths under ``config/feature_resources``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from utils.endpoint_registry import CANCER_TYPE_ENDPOINT_CLASSES, build_cdr_cancer_type_labels
except ImportError:  # pragma: no cover - direct execution from src/utils.
    from endpoint_registry import CANCER_TYPE_ENDPOINT_CLASSES, build_cdr_cancer_type_labels


BUNDLE_ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = BUNDLE_ROOT

IMPACTS = ["high", "moderate", "low", "modifier", "unknown"]
CONSEQUENCES = [
    "missense_variant",
    "synonymous_variant",
    "stop_gained",
    "frameshift_variant",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "splice_region_variant",
    "inframe_deletion",
    "inframe_insertion",
    "start_lost",
    "stop_lost",
    "stop_retained_variant",
    "protein_altering_variant",
    "coding_sequence_variant",
    "transcript_ablation",
    "intron_variant",
    "3_prime_utr_variant",
    "5_prime_utr_variant",
    "upstream_gene_variant",
    "downstream_gene_variant",
    "non_coding_transcript_exon_variant",
    "non_coding_transcript_variant",
    "mature_mirna_variant",
    "incomplete_terminal_codon_variant",
    "other",
    "unknown",
]
CANONICAL = ["yes", "no", "unknown"]
BIOTYPES = ["protein_coding", "lncrna", "nmd", "retained_intron", "processed_transcript", "other", "unknown"]
SIFT = ["deleterious", "deleterious_low_confidence", "tolerated", "tolerated_low_confidence", "unknown"]
POLYPHEN = ["probably_damaging", "possibly_damaging", "benign", "unknown"]

ROLE_LABELS = ["oncogene", "tumor_suppressor", "oncogene_and_tumor_suppressor"]
ROLE_METRICS = [
    "vep_high_or_moderate_impact_log_count",
    "vep_high_impact_log_count",
    "unique_high_or_moderate_impact_genes_log_count",
    "max_vaf_high_or_moderate_impact",
]
OPTIONAL_CONTROLS = ["log10_sbs_burden", "log10_dbs_burden", "log10_id_burden", "dbs_fraction", "id_fraction"]

MAIN_ENDPOINTS = [
    "HRD_Score",
    "HRD_TAI",
    "HRD_LST",
    "HRD_LOH",
    "PARPi7",
    "eCARD",
    "hrd_binary_24",
    "hrd_binary_33",
    "hrd_binary_42",
    "parpi7_binary",
    "cancer_type_top20",
    "high_stage",
    "smoking_ever",
    "high_purity",
]


@dataclass(frozen=True)
class Endpoint:
    name: str
    task: str
    labels: pd.Series


def _resolve_path(value: str | Path) -> Path:
    text = str(value).replace("${bundle_root}", str(BUNDLE_ROOT)).replace("${repro_root}", str(REPRO_ROOT))
    path = Path(text)
    if not path.is_absolute():
        path = BUNDLE_ROOT / path
    return path.resolve()


def load_paths(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    out = {}
    for section, values in raw.items():
        out[section] = {}
        for key, value in (values or {}).items():
            out[section][key] = None if value is None else _resolve_path(value)
    return out


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("||".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def normalize_token(value: object, fallback: str = "unknown") -> str:
    text = "" if value is None else str(value).strip().lower()
    if text in {"", "-", ".", "nan", "none", "na", "n/a"}:
        text = fallback
    text = text.replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or fallback


def normalize_gene(value: object) -> str:
    text = "" if value is None else str(value).strip().upper()
    if text in {"", "-", ".", "NAN", "NONE", "NA", "N/A", "UNKNOWN"}:
        return "UNKNOWN"
    return re.sub(r"[^A-Z0-9]+", "_", text).strip("_") or "UNKNOWN"


def first_12(value: object) -> str:
    return str(value).strip()[:12]


def _canonical(value: object) -> str:
    token = normalize_token(value)
    if token in {"yes", "y", "true", "1"}:
        return "yes"
    if token in {"no", "n", "false", "0"}:
        return "no"
    return "unknown"


def _biotype(value: object) -> str:
    token = normalize_token(value)
    if token == "lincrna":
        return "lncrna"
    if token == "nonsense_mediated_decay":
        return "nmd"
    return token if token in BIOTYPES else "other" if token != "unknown" else "unknown"


def _sift(value: object) -> str:
    token = normalize_token(re.sub(r"\(.*$", "", "" if value is None else str(value)).strip())
    return token if token in SIFT else "unknown"


def _polyphen(value: object) -> str:
    token = normalize_token(re.sub(r"\(.*$", "", "" if value is None else str(value)).strip())
    return token if token in POLYPHEN else "unknown"


def _impact(value: object) -> str:
    token = normalize_token(value)
    return token if token in IMPACTS else "unknown"


def _consequence(value: object) -> str:
    token = normalize_token(str(value).split(",")[0].split(";")[0] if value is not None else value)
    return token if token in CONSEQUENCES else "other" if token != "unknown" else "unknown"


def _variant_event_class(value: object) -> str:
    token = normalize_token(value)
    if token in {"snp", "snv"}:
        return "sbs"
    if token in {"dnp", "dnv"}:
        return "dbs"
    if token in {"ins", "del", "insertion", "deletion"}:
        return "id"
    return "other"


def feature_columns() -> tuple[list[str], list[str], dict[str, list[str]]]:
    vep = []
    for prefix, labels in [
        ("vep_impact", IMPACTS),
        ("vep_consequence", CONSEQUENCES),
        ("vep_canonical", CANONICAL),
        ("vep_biotype", BIOTYPES),
        ("vep_sift", SIFT),
        ("vep_polyphen", POLYPHEN),
    ]:
        for label in labels:
            vep.extend([f"{prefix}_count__{label}", f"{prefix}_fraction__{label}"])
    vaf = ["vaf_mean", "vaf_std", "vaf_median", "vaf_q90", "vaf_max"]
    oncokb = [f"oncokb_role_{metric}__{role}" for role in ROLE_LABELS for metric in ROLE_METRICS]
    residual = [f"residual_non_oncokb_positive_role_{metric}" for metric in ROLE_METRICS]
    core = [*vep, *vaf, *oncokb, *residual]
    all_cols = [*core, *OPTIONAL_CONTROLS]
    blocks = {
        "vep_annotations": vep,
        "vep_plus_vaf": [*vep, *vaf],
        "oncokb_roles": oncokb,
        "oncokb_plus_residual": [*oncokb, *residual],
        "core_bio_maf": core,
        "core_bio_maf_plus_burden_controls": all_cols,
    }
    return core, all_cols, blocks


def load_oncokb_roles(path: Path) -> dict[str, str]:
    table = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    roles: dict[str, str] = {}
    for row in table.to_dict(orient="records"):
        symbol = normalize_gene(row.get("hugo_symbol") or row.get("Hugo Symbol"))
        if symbol == "UNKNOWN":
            continue
        oncogene = str(row.get("oncogene", "")).strip().lower() == "true"
        tsg = str(row.get("tsg", "")).strip().lower() == "true"
        gene_type = str(row.get("Gene Type", "")).strip().upper()
        if gene_type:
            oncogene = oncogene or gene_type in {"ONCOGENE", "ONCOGENE_AND_TSG"}
            tsg = tsg or gene_type in {"TSG", "ONCOGENE_AND_TSG"}
        if oncogene and tsg:
            roles[symbol] = "oncogene_and_tumor_suppressor"
        elif oncogene:
            roles[symbol] = "oncogene"
        elif tsg:
            roles[symbol] = "tumor_suppressor"
    return roles


def _add_group_counts(
    matrix: pd.DataFrame,
    frame: pd.DataFrame,
    sample_col: str,
    token_col: str,
    prefix: str,
    valid_tokens: Iterable[str],
) -> None:
    counts = frame.groupby([sample_col, token_col], observed=True).size()
    for (sample, token), count in counts.items():
        if token in valid_tokens:
            matrix.at[sample, f"{prefix}_count__{token}"] += float(count)


def build_base_bio_features(paths: dict, output_dir: Path, *, chunksize: int, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    core_cols, all_cols, _blocks = feature_columns()
    cache_path = output_dir / "bio_maf_base_features.csv.gz"
    audit_path = output_dir / "bio_maf_base_features_audit.json"
    schema_path = BUNDLE_ROOT / "results" / "tables" / "proposed_clinical_bio_v4_feature_summary.csv"
    if cache_path.exists() and not force:
        features = pd.read_csv(cache_path, index_col=0)
        schema = pd.read_csv(schema_path) if schema_path.exists() else pd.DataFrame({"Feature added": all_cols})
        return features.astype(np.float32), schema

    mc3_dir = Path(paths["raw_data"]["mc3_source_dir"])
    maf_path = mc3_dir / "raw" / "mc3.v0.2.8.PUBLIC.maf.gz"
    standard_path = mc3_dir / "features" / "features_standard_sbs96_id83.csv.gz"
    sample_ids = pd.read_csv(standard_path, index_col=0, usecols=[0]).index.astype(str).tolist()
    sample_set = set(sample_ids)
    role_lookup = load_oncokb_roles(BUNDLE_ROOT / "config" / "feature_resources" / "oncokb_cancer_genes.tsv")

    raw_counts = pd.DataFrame(0.0, index=pd.Index(sample_ids, name="sample"), columns=all_cols, dtype=np.float64)
    total_events = pd.Series(0.0, index=raw_counts.index)
    broad_counts = {kind: pd.Series(0.0, index=raw_counts.index) for kind in ["sbs", "dbs", "id"]}
    vaf_values: dict[str, list[np.ndarray]] = defaultdict(list)
    unique_gene_sets: dict[str, dict[str, set[str]]] = {role: defaultdict(set) for role in [*ROLE_LABELS, "residual"]}
    qc = Counter()
    usecols = [
        "Tumor_Sample_Barcode",
        "Hugo_Symbol",
        "SYMBOL",
        "Variant_Type",
        "IMPACT",
        "Consequence",
        "CANONICAL",
        "BIOTYPE",
        "SIFT",
        "PolyPhen",
        "t_alt_count",
        "t_depth",
    ]
    for chunk in pd.read_csv(maf_path, sep="\t", usecols=usecols, dtype=str, chunksize=chunksize):
        qc["rows_scanned"] += int(len(chunk))
        chunk["sample"] = chunk["Tumor_Sample_Barcode"].map(first_12)
        chunk = chunk[chunk["sample"].isin(sample_set)].copy()
        if chunk.empty:
            continue
        qc["rows_used"] += int(len(chunk))
        total_events = total_events.add(chunk.groupby("sample", observed=True).size(), fill_value=0.0)

        chunk["_impact"] = chunk["IMPACT"].map(_impact)
        chunk["_consequence"] = chunk["Consequence"].map(_consequence)
        chunk["_canonical"] = chunk["CANONICAL"].map(_canonical)
        chunk["_biotype"] = chunk["BIOTYPE"].map(_biotype)
        chunk["_sift"] = chunk["SIFT"].map(_sift)
        chunk["_polyphen"] = chunk["PolyPhen"].map(_polyphen)
        for token_col, prefix, labels in [
            ("_impact", "vep_impact", IMPACTS),
            ("_consequence", "vep_consequence", CONSEQUENCES),
            ("_canonical", "vep_canonical", CANONICAL),
            ("_biotype", "vep_biotype", BIOTYPES),
            ("_sift", "vep_sift", SIFT),
            ("_polyphen", "vep_polyphen", POLYPHEN),
        ]:
            _add_group_counts(raw_counts, chunk, "sample", token_col, prefix, labels)

        alt = pd.to_numeric(chunk["t_alt_count"], errors="coerce")
        depth = pd.to_numeric(chunk["t_depth"], errors="coerce")
        vaf = alt / depth.replace(0, np.nan)
        chunk["_vaf"] = vaf.where(np.isfinite(vaf), np.nan)
        valid_vaf = chunk[chunk["_vaf"].notna()]
        for sample, values in valid_vaf.groupby("sample", observed=True)["_vaf"]:
            vaf_values[sample].append(values.to_numpy(dtype=np.float64))

        chunk["_gene"] = chunk["Hugo_Symbol"].where(chunk["Hugo_Symbol"].notna(), chunk["SYMBOL"]).map(normalize_gene)
        chunk["_role"] = chunk["_gene"].map(role_lookup).fillna("residual")
        for role in [*ROLE_LABELS, "residual"]:
            role_frame = chunk[chunk["_role"] == role]
            if role_frame.empty:
                continue
            hm = role_frame[role_frame["_impact"].isin(["high", "moderate"])]
            hi = role_frame[role_frame["_impact"] == "high"]
            hm_counts = hm.groupby("sample", observed=True).size()
            hi_counts = hi.groupby("sample", observed=True).size()
            for sample, count in hm_counts.items():
                col = (
                    f"oncokb_role_vep_high_or_moderate_impact_log_count__{role}"
                    if role != "residual"
                    else "residual_non_oncokb_positive_role_vep_high_or_moderate_impact_log_count"
                )
                raw_counts.at[sample, col] += float(count)
            for sample, count in hi_counts.items():
                col = (
                    f"oncokb_role_vep_high_impact_log_count__{role}"
                    if role != "residual"
                    else "residual_non_oncokb_positive_role_vep_high_impact_log_count"
                )
                raw_counts.at[sample, col] += float(count)
            for sample, genes in hm.groupby("sample", observed=True)["_gene"]:
                unique_gene_sets[role][sample].update(g for g in genes if g != "UNKNOWN")
            for sample, max_vaf in hm.groupby("sample", observed=True)["_vaf"].max().dropna().items():
                col = (
                    f"oncokb_role_max_vaf_high_or_moderate_impact__{role}"
                    if role != "residual"
                    else "residual_non_oncokb_positive_role_max_vaf_high_or_moderate_impact"
                )
                raw_counts.at[sample, col] = max(raw_counts.at[sample, col], float(max_vaf))

        chunk["_event_class"] = chunk["Variant_Type"].map(_variant_event_class)
        for kind in ["sbs", "dbs", "id"]:
            counts = chunk[chunk["_event_class"] == kind].groupby("sample", observed=True).size()
            broad_counts[kind] = broad_counts[kind].add(counts, fill_value=0.0)

    for role in [*ROLE_LABELS, "residual"]:
        for sample, genes in unique_gene_sets[role].items():
            col = (
                f"oncokb_role_unique_high_or_moderate_impact_genes_log_count__{role}"
                if role != "residual"
                else "residual_non_oncokb_positive_role_unique_high_or_moderate_impact_genes_log_count"
            )
            raw_counts.at[sample, col] = float(len(genes))

    features = raw_counts.copy()
    for col in [c for c in features.columns if "_count__" in c or c.endswith("_log_count") or "unique_high_or_moderate" in c]:
        if col.startswith("log10_"):
            continue
        features[col] = np.log1p(raw_counts[col].to_numpy(dtype=np.float64))
    for prefix, labels in [
        ("vep_impact", IMPACTS),
        ("vep_consequence", CONSEQUENCES),
        ("vep_canonical", CANONICAL),
        ("vep_biotype", BIOTYPES),
        ("vep_sift", SIFT),
        ("vep_polyphen", POLYPHEN),
    ]:
        for label in labels:
            count_col = f"{prefix}_count__{label}"
            frac_col = f"{prefix}_fraction__{label}"
            features[frac_col] = np.divide(
                raw_counts[count_col].to_numpy(dtype=np.float64),
                total_events.to_numpy(dtype=np.float64),
                out=np.zeros(len(total_events), dtype=np.float64),
                where=total_events.to_numpy(dtype=np.float64) > 0,
            )

    for sample in features.index:
        arrays = vaf_values.get(sample, [])
        if not arrays:
            continue
        vals = np.concatenate(arrays)
        if vals.size:
            features.at[sample, "vaf_mean"] = float(np.mean(vals))
            features.at[sample, "vaf_std"] = float(np.std(vals))
            features.at[sample, "vaf_median"] = float(np.median(vals))
            features.at[sample, "vaf_q90"] = float(np.quantile(vals, 0.9))
            features.at[sample, "vaf_max"] = float(np.max(vals))

    broad_total = broad_counts["sbs"] + broad_counts["dbs"] + broad_counts["id"]
    features["log10_sbs_burden"] = np.log10(1.0 + broad_counts["sbs"])
    features["log10_dbs_burden"] = np.log10(1.0 + broad_counts["dbs"])
    features["log10_id_burden"] = np.log10(1.0 + broad_counts["id"])
    broad_total_arr = broad_total.to_numpy(dtype=np.float64)
    features["dbs_fraction"] = np.divide(
        broad_counts["dbs"].to_numpy(dtype=np.float64),
        broad_total_arr,
        out=np.zeros(len(broad_total_arr), dtype=np.float64),
        where=broad_total_arr > 0,
    )
    features["id_fraction"] = np.divide(
        broad_counts["id"].to_numpy(dtype=np.float64),
        broad_total_arr,
        out=np.zeros(len(broad_total_arr), dtype=np.float64),
        where=broad_total_arr > 0,
    )

    features = features.loc[:, all_cols].fillna(0.0).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(cache_path, compression="gzip")
    schema = pd.read_csv(schema_path) if schema_path.exists() else pd.DataFrame({"Feature added": all_cols})
    audit = {
        "feature_count": int(features.shape[1]),
        "sample_count": int(features.shape[0]),
        "core_feature_count": len(core_cols),
        "optional_control_count": len(OPTIONAL_CONTROLS),
        "rows_scanned": int(qc["rows_scanned"]),
        "rows_used": int(qc["rows_used"]),
        "total_events": int(total_events.sum()),
        "nonzero_feature_count": int((features.sum(axis=0) != 0).sum()),
        "cache_path": str(cache_path),
        "resource_path": "config/feature_resources/oncokb_cancer_genes.tsv",
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return features, schema


def load_endpoints(paths: dict, endpoints: Iterable[str], min_binary: int = 25, min_multiclass: int = 50) -> list[Endpoint]:
    wanted = set(endpoints)
    out: list[Endpoint] = []
    hrd_path = Path(paths["raw_data"]["hrd_assets_dir"]) / "cohort" / "final_analysis_cohort.tsv"
    cohort = pd.read_csv(hrd_path, sep="\t")
    cohort["patient_id_12"] = cohort["patient_id_12"].astype(str)
    hrd_continuous = {"HRD_Score", "HRD_TAI", "HRD_LST", "HRD_LOH", "PARPi7", "eCARD"}
    hrd_binary = {"hrd_binary_24", "hrd_binary_33", "hrd_binary_42", "parpi7_binary"}
    for endpoint in [e for e in MAIN_ENDPOINTS if e in wanted]:
        if endpoint in hrd_continuous and endpoint in cohort.columns:
            labels = cohort.dropna(subset=[endpoint]).set_index("patient_id_12")[endpoint].astype(float)
            out.append(Endpoint(endpoint, "regression", labels))
        elif endpoint in hrd_binary and endpoint in cohort.columns:
            allowed = ["PARPi-high", "PARPi-low"] if endpoint == "parpi7_binary" else ["HRD-high", "HRD-low"]
            positive = "PARPi-high" if endpoint == "parpi7_binary" else "HRD-high"
            data = cohort[cohort[endpoint].isin(allowed)].copy()
            labels = pd.Series((data[endpoint] == positive).astype(int).to_numpy(), index=data["patient_id_12"].astype(str), name=endpoint)
            if labels.value_counts().min() >= min_binary:
                out.append(Endpoint(endpoint, "binary", labels))

    mc3_labels = pd.read_csv(Path(paths["raw_data"]["mc3_source_dir"]) / "biology_labels.csv", index_col=0)
    mc3_labels.index = mc3_labels.index.astype(str)
    for endpoint in [e for e in MAIN_ENDPOINTS if e in wanted]:
        if endpoint in CANCER_TYPE_ENDPOINT_CLASSES:
            fixed_classes = list(CANCER_TYPE_ENDPOINT_CLASSES[endpoint])
            feature_patients = pd.read_csv(
                Path(paths["raw_data"]["mc3_source_dir"]) / "features" / "features_burden_only.csv",
                usecols=[0],
                index_col=0,
            ).index.astype(str)
            labels = build_cdr_cancer_type_labels(
                Path(paths["raw_data"]["mc3_source_dir"]),
                feature_patients,
                fixed_classes,
                endpoint_name=endpoint,
            )
            counts = labels.astype(str).value_counts().reindex(fixed_classes, fill_value=0)
            if labels.nunique() == len(fixed_classes) and int(counts.min()) >= min_multiclass:
                out.append(Endpoint(endpoint, "multiclass", labels))
        elif endpoint in {"high_stage", "smoking_ever", "high_purity"} and endpoint in mc3_labels.columns:
            labels = mc3_labels[endpoint].dropna().astype(int)
            if labels.nunique() == 2 and labels.value_counts().min() >= min_binary:
                out.append(Endpoint(endpoint, "binary", labels))
    return out


def align_xy(features: pd.DataFrame, endpoint: Endpoint) -> tuple[pd.Index, np.ndarray, pd.Series]:
    labels = endpoint.labels.dropna()
    common = features.index.intersection(labels.index)
    x = features.loc[common].to_numpy(dtype=np.float32)
    y = labels.loc[common]
    return common, x, y


def candidate_params(task: str, profile: str) -> list[dict[str, float]]:
    if task == "regression":
        alphas = [0.001, 0.01, 0.1] if profile == "tiny" else [0.0001, 0.001, 0.01, 0.1, 1.0]
        l1s = [0.0, 0.5] if profile == "tiny" else [0.0, 0.25, 0.5, 0.75, 1.0]
        return [{"alpha": a, "l1_ratio": l1} for a in alphas for l1 in l1s]
    cs = [0.01, 0.1, 1.0] if profile == "tiny" else [0.001, 0.01, 0.1, 1.0, 10.0]
    l1s = [0.0, 0.5] if profile == "tiny" else [0.0, 0.25, 0.5, 0.75, 1.0]
    return [{"C": c, "l1_ratio": l1} for c in cs for l1 in l1s]


def encode_y(endpoint: Endpoint, y: pd.Series) -> tuple[np.ndarray, LabelEncoder | None]:
    if endpoint.task == "regression":
        return y.astype(float).to_numpy(), None
    if endpoint.task == "binary":
        return y.astype(int).to_numpy(), None
    enc = LabelEncoder()
    return enc.fit_transform(y.astype(str).to_numpy()), enc


def outer_splits(task: str, y: np.ndarray, folds: int, seed: int):
    if len(y) < 2:
        raise ValueError("At least two samples are required for outer cross-validation")
    n_splits = max(2, min(int(folds), int(len(y))))
    if task == "regression":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y))))
    counts = pd.Series(y).value_counts()
    if len(counts) > 1 and int(counts.min()) >= 2:
        stratified_splits = min(n_splits, int(counts.min()))
        splitter = StratifiedKFold(n_splits=stratified_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), y))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y))))


def inner_split(task: str, y: np.ndarray, train_idx: np.ndarray, seed: int):
    sub_y = y[train_idx]
    if task != "regression" and pd.Series(sub_y).value_counts().min() >= 2:
        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
            tr, val = next(splitter.split(np.zeros(len(train_idx)), sub_y))
            return train_idx[tr], train_idx[val]
        except ValueError:
            pass
    splitter = ShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, val = next(splitter.split(np.zeros(len(train_idx))))
    return train_idx[tr], train_idx[val]


def primary_metric_for_endpoint(endpoint: Endpoint) -> str:
    if endpoint.task == "regression":
        return "spearman"
    if endpoint.task == "binary":
        return "auroc"
    if endpoint.name == "cancer_type_top20":
        return "balanced_accuracy"
    return "macro_auroc"


def safe_score(task: str, y_true: np.ndarray, pred: np.ndarray, metric: str | None = None) -> float:
    metric = str(metric or "").lower()
    try:
        if task == "regression":
            score = spearmanr(y_true, pred).statistic
        elif task == "binary":
            if len(np.unique(y_true)) < 2:
                return float("nan")
            klass = (pred[:, 1] >= 0.5).astype(int)
            if metric == "balanced_accuracy":
                score = balanced_accuracy_score(y_true, klass)
            elif metric == "macro_f1":
                score = f1_score(y_true, klass, average="macro", zero_division=0)
            else:
                score = roc_auc_score(y_true, pred[:, 1])
        else:
            klass = np.argmax(pred, axis=1)
            if metric == "balanced_accuracy":
                score = balanced_accuracy_score(y_true, klass)
            elif metric == "macro_f1":
                score = f1_score(y_true, klass, average="macro", zero_division=0)
            else:
                classes = np.unique(y_true)
                scores = []
                for cls in classes:
                    binary = (y_true == cls).astype(int)
                    if binary.min() == binary.max():
                        continue
                    scores.append(roc_auc_score(binary, pred[:, int(cls)]))
                score = float(np.mean(scores)) if scores else float("nan")
        return float(score) if np.isfinite(score) else float("nan")
    except Exception:
        return float("nan")


def fit_predict(task: str, x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray, params: dict[str, float], seed: int) -> np.ndarray:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_pred = scaler.transform(x_pred)
    if task == "regression":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = ElasticNet(
                alpha=float(params["alpha"]),
                l1_ratio=float(params["l1_ratio"]),
                max_iter=2000,
                tol=1e-4,
                random_state=seed,
            )
            model.fit(x_train, y_train)
        return model.predict(x_pred)
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=float(params["C"]),
        l1_ratio=float(params["l1_ratio"]),
        max_iter=500,
        tol=1e-3,
        n_jobs=1,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    return model.predict_proba(x_pred)


def metric_bundle(task: str, y: np.ndarray, pred: np.ndarray, metric: str) -> dict[str, float]:
    score = safe_score(task, y, pred, metric)
    out = {"primary_score": score}
    if task == "regression":
        out.update({"spearman": score, "mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred))})
    elif task == "binary":
        klass = (pred[:, 1] >= 0.5).astype(int)
        out.update(
            {
                "auroc": score,
                "auprc": float(average_precision_score(y, pred[:, 1])),
                "balanced_accuracy": float(balanced_accuracy_score(y, klass)),
                "macro_f1": float(f1_score(y, klass, average="macro", zero_division=0)),
            }
        )
    else:
        klass = np.argmax(pred, axis=1)
        out.update(
            {
                "macro_auroc": safe_score(task, y, pred, "macro_auroc"),
                "balanced_accuracy": float(balanced_accuracy_score(y, klass)),
                "macro_f1": float(f1_score(y, klass, average="macro", zero_division=0)),
            }
        )
    return out


def run_nested_feature_selection(
    endpoint: Endpoint,
    features_by_name: dict[str, pd.DataFrame],
    candidate_sets: dict[str, tuple[str, list[str]]],
    *,
    folds: int,
    seed: int,
    profile: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    base_features = next(iter(features_by_name.values()))
    samples, _unused, labels = align_xy(base_features, endpoint)
    y, encoder = encode_y(endpoint, labels)
    primary_metric = primary_metric_for_endpoint(endpoint)
    params_grid = candidate_params(endpoint.task, profile)
    splits = outer_splits(endpoint.task, y, folds, seed)
    pred = (
        np.zeros(len(samples), dtype=np.float64)
        if endpoint.task == "regression"
        else np.zeros((len(samples), 2 if endpoint.task == "binary" else len(np.unique(y))), dtype=np.float64)
    )
    fold_rows = []
    sample_folds = np.full(len(samples), -1, dtype=int)
    for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
        if np.any(sample_folds[test_idx] != -1):
            raise RuntimeError(f"Outer fold {fold_no} overlaps a previous Bio MAF diagnostic test fold")
        inner_train, inner_val = inner_split(endpoint.task, y, train_idx, seed + fold_no * 1009)
        best = {"score": float("-inf"), "feature_set": None, "params": None}
        for feature_set_name, (matrix_name, columns) in candidate_sets.items():
            full = features_by_name[matrix_name].loc[samples, columns].to_numpy(dtype=np.float32)
            for params in params_grid:
                try:
                    val_pred = fit_predict(endpoint.task, full[inner_train], y[inner_train], full[inner_val], params, seed + fold_no)
                    score = safe_score(endpoint.task, y[inner_val], val_pred, primary_metric)
                except Exception:
                    score = float("nan")
                if np.isfinite(score) and score > float(best["score"]):
                    best = {"score": float(score), "feature_set": feature_set_name, "params": dict(params)}
        if best["feature_set"] is None:
            raise RuntimeError(f"No valid candidate for {endpoint.name}")
        matrix_name, columns = candidate_sets[str(best["feature_set"])]
        full = features_by_name[matrix_name].loc[samples, columns].to_numpy(dtype=np.float32)
        test_pred = fit_predict(endpoint.task, full[train_idx], y[train_idx], full[test_idx], dict(best["params"]), seed + 99_991 + fold_no)
        pred[test_idx] = test_pred
        sample_folds[test_idx] = int(fold_no)
        fold_score = safe_score(endpoint.task, y[test_idx], test_pred, primary_metric)
        fold_rows.append(
            {
                "endpoint": endpoint.name,
                "task": endpoint.task,
                "fold": fold_no,
                "outer_train_n": int(len(train_idx)),
                "outer_test_n": int(len(test_idx)),
                "inner_train_n": int(len(inner_train)),
                "inner_val_n": int(len(inner_val)),
                "selected_feature_set": best["feature_set"],
                "selected_feature_count": int(len(columns)),
                "selected_inner_score": float(best["score"]),
                "outer_fold_score": float(fold_score) if np.isfinite(fold_score) else float("nan"),
                "selected_params_json": json.dumps(best["params"], sort_keys=True),
                "candidate_feature_sets": ",".join(candidate_sets),
                "candidate_param_count": int(len(params_grid)),
            }
        )
    if np.any(sample_folds < 0):
        missing = samples[sample_folds < 0].astype(str).tolist()
        raise RuntimeError(f"Bio MAF diagnostic OOF split left {len(missing)} sample(s) without predictions: {missing[:10]}")
    metrics = metric_bundle(endpoint.task, y, pred, primary_metric)
    summary = {
        "endpoint": endpoint.name,
        "task": endpoint.task,
        "n_samples": int(len(samples)),
        "requested_folds": int(folds),
        "n_folds": int(len(splits)),
        "primary_metric": primary_metric,
        "primary_score": metrics["primary_score"],
        **metrics,
    }
    pred_df = pd.DataFrame({"sample": samples.astype(str), "true": labels.to_numpy(), "fold": sample_folds})
    if endpoint.task == "regression":
        pred_df["pred"] = pred
    elif endpoint.task == "binary":
        pred_df["pred_class_1"] = pred[:, 1]
    else:
        classes = encoder.classes_ if encoder is not None else [str(i) for i in range(pred.shape[1])]
        for i, cls in enumerate(classes):
            pred_df[f"pred__{cls}"] = pred[:, i]
    return summary, pred_df, pd.DataFrame(fold_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--endpoints", nargs="*", default=MAIN_ENDPOINTS)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--chunksize", type=int, default=350_000)
    parser.add_argument("--profile", choices=["tiny", "standard"], default="tiny")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    started = time.time()
    paths = load_paths(Path(args.paths))
    out_dir = BUNDLE_ROOT / "results" / "tables"
    features, _schema = build_base_bio_features(paths, out_dir, chunksize=args.chunksize)
    endpoints = load_endpoints(paths, args.endpoints)
    print(json.dumps({"status": "ok", "features": list(features.shape), "endpoints": [item.name for item in endpoints], "elapsed_seconds": round(time.time() - started, 3)}, indent=2))
