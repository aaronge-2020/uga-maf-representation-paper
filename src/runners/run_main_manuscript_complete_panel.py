"""Complete the measured main-manuscript benchmark panel.

This runner fills the definitive main manuscript endpoint panel. It writes after
every endpoint/representation/model job so the same command can be resumed after
interruption without losing completed folds.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils.checkpointing import atomic_write_csv, atomic_write_json
from utils.endpoint_data import EndpointData
from utils.feature_cache import code_fingerprint, directory_fingerprint, file_fingerprint, make_cache_key
from utils.feature_evaluation import evaluate_features
from utils.kucab_event_features import (
    FastaReader,
    build_kucab_inventory,
    burden_features_from_covariates,
    find_fasta,
    read_cosmic_channels,
    standard_features_from_counts,
)
from utils.nested_oof import evaluate_nested_feature_set_oof, stable_seed
from utils.quick_bio_v4_nested_feature_selection import build_bio_v4_features, v4_feature_columns
from utils.endpoint_registry import (
    CANCER_TYPE_ENDPOINT_CLASSES,
    HRD_BINARY,
    HRD_CONTINUOUS,
    build_cdr_cancer_type_labels,
    build_cdr_survival_labels,
    build_luad_kmt2c_labels,
    load_endpoint_registry,
    main_kucab_endpoints,
    main_non_survival_endpoints,
    main_survival_endpoints,
    registry_minimums,
)
from utils.runner_support import (
    RunnerContext,
    sanitize_frame,
    write_summary_csv,
)


EXPERIMENT_ID = "main_manuscript_complete_panel"
_REGISTRY = load_endpoint_registry()
MAIN_ENDPOINTS = [*main_kucab_endpoints(_REGISTRY), *main_non_survival_endpoints(_REGISTRY), *main_survival_endpoints(_REGISTRY)]
MC3_MAIN_ENDPOINTS = main_non_survival_endpoints(_REGISTRY)
MC3_SURVIVAL_ENDPOINTS = main_survival_endpoints(_REGISTRY)
LEARNERS = ["linear", "xgboost"]
SURVIVAL_LEARNERS = ["cox_ph"]
MAF_MAIN_ONLY_LEARNERS = LEARNERS
BIO_MAF_V4_MODEL_IDS = [
    "maf_stack_bio_v4_nested_fs",
    "signatures_plus_maf_stack_bio_v4_nested_fs",
]
MAF_REPRESENTATION = {
    "maf_stack_bio_v4_nested_fs": "MAF_stack_only",
    "signatures_plus_maf_stack_bio_v4_nested_fs": "signatures_plus_MAF_stack",
}
KUCAB_TABLES = [
    "denovo_subclone_subs_final.txt",
    "denovo_subclone_doublesub_final.txt",
    "denovo_subclone_indels.final.txt",
    "README.txt",
]


def _stable_bucket(namespace: str, token: object, n_buckets: int) -> int:
    if int(n_buckets) <= 0:
        raise ValueError("n_buckets must be positive")
    digest = hashlib.sha256(f"{namespace}::{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(n_buckets)


def _settings(ctx: RunnerContext) -> dict:
    local = dict(((ctx.settings.get("experiments") or {}).get(EXPERIMENT_ID) or {}))
    merged = dict(local)
    registry = load_endpoint_registry(local.get("endpoint_registry") or ctx.settings.get("endpoint_registry"))
    merged.setdefault("d_context", 6)
    merged.setdefault("d_payload", 6)
    merged.setdefault("xgb_estimators", ctx.xgb_estimators)
    merged.setdefault("xgb_max_depth", 2)
    merged.setdefault("xgb_learning_rate", 0.05)
    merged.setdefault("random_search_trials", int(ctx.optuna_trials))
    merged["main_mc3_endpoints"] = list(local.get("main_mc3_endpoints") or main_non_survival_endpoints(registry))
    merged["survival_endpoints"] = list(local.get("survival_endpoints") or main_survival_endpoints(registry))
    merged["kucab_endpoints"] = list(local.get("kucab_endpoints") or main_kucab_endpoints(registry))
    merged["endpoint_registry_minimums"] = registry_minimums(registry)
    return merged


def _maf_model_ids(settings: dict) -> list[str]:
    if bool(settings.get("allow_transductive_maf_features", False)):
        raise ValueError("The standalone manuscript export requires fold-isolated MAF feature construction.")
    builder = str(settings.get("maf_stack_builder", "frozen_biological_v4_nested_fs"))
    if builder != "frozen_biological_v4_nested_fs":
        raise ValueError(f"Unsupported active MAF stack builder for this export: {builder}")
    return list(BIO_MAF_V4_MODEL_IDS)


def _load_mc3_endpoints(ctx: RunnerContext, requested: Iterable[str], survival_requested: Iterable[str], settings: dict) -> tuple[list[EndpointData], pd.DataFrame]:
    requested = [str(value) for value in requested]
    hrd_path = Path(ctx.paths["raw_data"]["hrd_assets_dir"]) / "cohort" / "final_analysis_cohort.tsv"
    cohort = pd.read_csv(hrd_path, sep="\t")
    cohort["patient_id_12"] = cohort["patient_id_12"].astype(str)
    labels_path = Path(ctx.paths["raw_data"]["mc3_source_dir"]) / "biology_labels.csv"
    mc3_labels = pd.read_csv(labels_path, index_col=0)
    mc3_labels.index = mc3_labels.index.astype(str)
    endpoints: list[EndpointData] = []
    audit_rows: list[dict[str, object]] = []
    minimums = dict(settings.get("endpoint_registry_minimums") or {})
    binary_min = int(minimums.get("binary_min_per_class", 25))
    multiclass_min = int(minimums.get("multiclass_min_per_class", 50))
    feature_patient_index: pd.Index | None = None

    def matched_feature_patients() -> pd.Index:
        nonlocal feature_patient_index
        if feature_patient_index is None:
            burden_path = Path(ctx.paths["raw_data"]["mc3_source_dir"]) / "features" / "features_burden_only.csv"
            feature_patient_index = pd.read_csv(burden_path, usecols=[0], index_col=0).index.astype(str)
        return feature_patient_index

    for endpoint in requested:
        if endpoint in HRD_CONTINUOUS and endpoint in cohort.columns:
            data = cohort.dropna(subset=[endpoint]).copy()
            labels = pd.Series(data[endpoint].astype(float).to_numpy(), index=data["patient_id_12"].astype(str), name=endpoint)
            endpoints.append(EndpointData(endpoint, "mc3_main", "regression", labels))
            audit_rows.append({"endpoint": endpoint, "task": "regression", "status": "included", "n_samples": int(len(labels)), "reason": ""})
        elif endpoint in HRD_BINARY and endpoint in cohort.columns:
            allowed = ["PARPi-high", "PARPi-low"] if endpoint == "parpi7_binary" else ["HRD-high", "HRD-low"]
            positive = "PARPi-high" if endpoint == "parpi7_binary" else "HRD-high"
            data = cohort[cohort[endpoint].isin(allowed)].copy()
            labels = pd.Series((data[endpoint] == positive).astype(int).to_numpy(), index=data["patient_id_12"].astype(str), name=endpoint)
            counts = labels.value_counts()
            if len(counts) == 2 and int(counts.min()) >= binary_min:
                endpoints.append(EndpointData(endpoint, "mc3_main", "binary", labels))
                audit_rows.append({"endpoint": endpoint, "task": "binary", "status": "included", "n_samples": int(len(labels)), "min_class_n": int(counts.min()), "reason": ""})
            else:
                audit_rows.append({"endpoint": endpoint, "task": "binary", "status": "excluded", "n_samples": int(len(labels)), "min_class_n": int(counts.min() if len(counts) else 0), "reason": "below_minimum_support"})
        elif endpoint == "luad_kmt2c_mutated":
            driver_index = pd.read_csv(Path(ctx.paths["raw_data"]["mc3_source_dir"]) / "driver_gene_labels_functional.csv", index_col=0).index.astype(str)
            labels = build_luad_kmt2c_labels(Path(ctx.paths["raw_data"]["mc3_source_dir"]), driver_index)
            counts = labels.value_counts()
            if len(counts) == 2 and int(counts.min()) >= binary_min:
                endpoints.append(EndpointData(endpoint, "mc3_kmt2c_supportive", "binary", labels))
                audit_rows.append({"endpoint": endpoint, "task": "binary", "status": "included", "n_samples": int(len(labels)), "min_class_n": int(counts.min()), "reason": ""})
            else:
                audit_rows.append({"endpoint": endpoint, "task": "binary", "status": "excluded", "n_samples": int(len(labels)), "min_class_n": int(counts.min() if len(counts) else 0), "reason": "below_minimum_support"})
        elif endpoint in CANCER_TYPE_ENDPOINT_CLASSES:
            fixed_classes = list(CANCER_TYPE_ENDPOINT_CLASSES[endpoint])
            labels = build_cdr_cancer_type_labels(
                Path(ctx.paths["raw_data"]["mc3_source_dir"]),
                matched_feature_patients(),
                fixed_classes,
                endpoint_name=endpoint,
            )
            counts = labels.astype(str).value_counts().reindex(fixed_classes, fill_value=0)
            task = "multiclass"
            if labels.nunique() == len(fixed_classes) and int(counts.min()) >= multiclass_min:
                endpoints.append(EndpointData(endpoint, "mc3_main", task, labels.astype(str)))
                audit_rows.append(
                    {
                        "endpoint": endpoint,
                        "task": task,
                        "status": "included",
                        "n_samples": int(len(labels)),
                        "n_classes": int(labels.nunique()),
                        "min_class_n": int(counts.min()),
                        "class_counts_json": json.dumps({key: int(value) for key, value in counts.items()}),
                        "reason": "",
                    }
                )
            else:
                audit_rows.append(
                    {
                        "endpoint": endpoint,
                        "task": task,
                        "status": "excluded",
                        "n_samples": int(len(labels)),
                        "n_classes": int(labels.nunique()),
                        "min_class_n": int(counts.min() if len(counts) else 0),
                        "class_counts_json": json.dumps({key: int(value) for key, value in counts.items()}),
                        "reason": "below_fixed_top20_support",
                    }
                )
        elif endpoint in mc3_labels.columns:
            labels = mc3_labels[endpoint].dropna()
            labels = labels.astype(int)
            task = "binary"
            counts = labels.value_counts()
            if len(counts) == 2 and int(counts.min()) >= binary_min:
                endpoints.append(EndpointData(endpoint, "mc3_main", task, labels))
                audit_rows.append({"endpoint": endpoint, "task": task, "status": "included", "n_samples": int(len(labels)), "min_class_n": int(counts.min()), "reason": ""})
            else:
                audit_rows.append({"endpoint": endpoint, "task": task, "status": "excluded", "n_samples": int(len(labels)), "min_class_n": int(counts.min() if len(counts) else 0), "reason": "below_minimum_support"})
        else:
            audit_rows.append({"endpoint": endpoint, "task": "unknown", "status": "excluded", "reason": "missing_label_column"})
    survival_labels, survival_audit = build_cdr_survival_labels(
        Path(ctx.paths["raw_data"]["mc3_source_dir"]),
        survival_requested,
        min_samples=int(minimums.get("survival_min_samples", 200)),
        min_events=int(minimums.get("survival_min_events", 25)),
    )
    for endpoint, labels in survival_labels.items():
        endpoints.append(EndpointData(endpoint, "tcga_cdr_survival", "survival", labels))
    if not survival_audit.empty:
        audit_rows.extend(survival_audit.to_dict("records"))
    return endpoints, pd.DataFrame(audit_rows)


def _load_mc3_standard_features(ctx: RunnerContext) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    feature_dir = Path(ctx.paths["raw_data"]["mc3_source_dir"]) / "features"
    burden_path = feature_dir / "features_burden_only.csv"
    standard_path = feature_dir / "features_standard_sbs96_id83.csv.gz"
    cache_key = make_cache_key(
        "main_complete_panel_mc3_standard_features",
        params={"representations": ["burden_only", "signatures_only"]},
        inputs={
            "burden": file_fingerprint(burden_path),
            "standard": file_fingerprint(standard_path),
            "code": code_fingerprint([Path(__file__)]),
        },
    )
    cache = ctx.feature_cache
    cached_burden = cache.load_frame(cache_key, "burden_only.csv.gz")
    cached_standard = cache.load_frame(cache_key, "signatures_only.csv.gz")
    if cached_burden is not None and cached_standard is not None:
        for frame in (cached_burden, cached_standard):
            frame.index = frame.index.astype(str)
        return cached_burden.astype(np.float32), cached_standard.astype(np.float32), cache_key
    burden = pd.read_csv(burden_path, index_col=0).fillna(0.0).astype(np.float32)
    standard = pd.read_csv(standard_path, index_col=0).fillna(0.0).astype(np.float32)
    burden.index = burden.index.astype(str)
    standard.index = standard.index.astype(str)
    metadata = {
        "namespace": EXPERIMENT_ID,
        "representation": "mc3_standard_burden_signature_features",
        "sample_count": int(standard.shape[0]),
        "feature_count": {"burden_only": int(burden.shape[1]), "signatures_only": int(standard.shape[1])},
    }
    cache.save_frame(cache_key, burden, "burden_only.csv.gz", metadata=metadata)
    cache.save_frame(cache_key, standard, "signatures_only.csv.gz", metadata=metadata)
    return burden, standard, cache_key


def _load_mc3_maf_features(ctx: RunnerContext, model_id: str, settings: dict | None = None) -> tuple[pd.DataFrame, str]:
    settings = dict(settings or {})
    raw_dir = Path(ctx.paths["raw_data"]["mc3_source_dir"])
    maf_path = raw_dir / "raw" / "mc3.v0.2.8.PUBLIC.maf.gz"
    is_combined = model_id.startswith("id_plus_") or model_id.startswith("signatures_plus_")
    if model_id in BIO_MAF_V4_MODEL_IDS:
        standard_path = raw_dir / "features" / "features_standard_sbs96_id83.csv.gz"
        _core_cols, _all_cols, v4_blocks = v4_feature_columns()
        requested_sets = settings.get("maf_bio_v4_candidate_sets")
        if requested_sets:
            requested = {str(name) for name in requested_sets}
            v4_blocks = {name: cols for name, cols in v4_blocks.items() if name in requested}
            if not v4_blocks:
                raise ValueError(f"No valid Bio v4 candidate feature sets requested: {requested_sets}")
        tables_dir = Path(__file__).parents[2] / "results" / "tables"
        v4_resource_dir = Path(__file__).parents[2] / "config" / "feature_resources" / "clinical_bio_v4_sources"
        cache_key = make_cache_key(
            f"main_complete_panel_mc3_{model_id}",
            params={
                "model_id": model_id,
                "schema": 1,
                "feature_builder": "frozen_biological_v4_nested_fs",
                "candidate_feature_sets": sorted(v4_blocks),
                "include_signatures_only_candidate": bool(settings.get("maf_bio_v4_include_signatures_only_candidate", True)),
            },
            inputs={
                "maf": file_fingerprint(maf_path),
                "standard": file_fingerprint(standard_path) if is_combined else None,
                "v4_manifest": file_fingerprint(tables_dir / "proposed_clinical_bio_v4_manifest.json") if (tables_dir / "proposed_clinical_bio_v4_manifest.json").exists() else None,
                "v4_feature_table": file_fingerprint(tables_dir / "proposed_clinical_bio_v4_feature_table_exact.csv") if (tables_dir / "proposed_clinical_bio_v4_feature_table_exact.csv").exists() else None,
                "v4_resources": directory_fingerprint(v4_resource_dir, patterns=("*.tsv", "*.txt", "*.xlsx", "*.json", "*.zip")) if v4_resource_dir.exists() else None,
                "code": code_fingerprint(
                    [
                        Path(__file__),
                        Path(__file__).parents[1] / "utils" / "nested_oof.py",
                        Path(__file__).parents[1] / "utils" / "bio_maf_base_features.py",
                        Path(__file__).parents[1] / "utils" / "quick_bio_v4_nested_feature_selection.py",
                    ]
                ),
            },
        )

        def attach_candidate_sets(frame: pd.DataFrame) -> pd.DataFrame:
            candidate_sets: dict[str, list[str]] = {}
            if is_combined:
                sig_cols = [col for col in frame.columns.astype(str) if col.startswith("signature_matrix__")]
                if bool(settings.get("maf_bio_v4_include_signatures_only_candidate", True)):
                    candidate_sets["signatures_only"] = list(sig_cols)
                for name, cols in v4_blocks.items():
                    candidate_sets[f"signatures_plus_{name}"] = [*sig_cols, *cols]
            else:
                candidate_sets = {name: list(cols) for name, cols in v4_blocks.items()}
            missing = {name: [col for col in cols if col not in frame.columns] for name, cols in candidate_sets.items()}
            missing = {name: cols for name, cols in missing.items() if cols}
            if missing:
                first_name = next(iter(missing))
                raise RuntimeError(f"Bio v4 feature matrix is missing columns for {first_name}: {missing[first_name][:10]}")
            frame.attrs["candidate_feature_sets"] = candidate_sets
            frame.attrs["candidate_feature_set_counts"] = {name: len(cols) for name, cols in candidate_sets.items()}
            frame.attrs["feature_builder"] = "frozen_biological_v4_nested_fs"
            return frame

        cached = ctx.feature_cache.load_frame(cache_key, "features.csv.gz")
        if cached is not None:
            cached.index = cached.index.astype(str)
            return attach_candidate_sets(cached.astype(np.float32)), cache_key

        bio = build_bio_v4_features(
            ctx.paths,
            ctx.tables_dir,
            chunksize=int(settings.get("maf_bio_chunksize", 350_000)),
            force=bool(ctx.refresh_cache or settings.get("maf_bio_v4_force_features", False)),
        ).astype(np.float32)
        bio.index = bio.index.astype(str)
        frame = bio
        if is_combined:
            standard = pd.read_csv(standard_path, index_col=0).fillna(0.0).astype(np.float32)
            standard.index = standard.index.astype(str)
            standard = standard.add_prefix("signature_matrix__")
            frame = pd.concat([standard.reindex(bio.index), bio], axis=1).fillna(0.0).astype(np.float32)
        frame = attach_candidate_sets(frame)
        candidate_counts = dict(frame.attrs.get("candidate_feature_set_counts") or {})
        ctx.feature_cache.save_frame(
            cache_key,
            frame,
            "features.csv.gz",
            metadata={
                "namespace": EXPERIMENT_ID,
                "representation": model_id,
                "sample_count": int(frame.shape[0]),
                "feature_count": int(frame.shape[1]),
                "model_id": model_id,
                "feature_builder": "frozen_biological_v4_nested_fs",
                "leakage_status": "fixed external Bio MAF v4 schema/resources; predeclared feature blocks selected only inside each outer fold's inner validation split",
                "schema_type": "named collision-free Bio MAF v4 features with nested feature-block selection",
                "candidate_feature_set_counts": candidate_counts,
                "include_signatures": bool(is_combined),
            },
        )
        return frame, cache_key

    raise ValueError(f"Unsupported active Bio MAF v4 model_id: {model_id}")


def _normalize_token(value: object, fallback: str = "unknown") -> str:
    text = str(value).strip().lower()
    if text in {"", "-", ".", "nan", "none"}:
        text = fallback
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or fallback


def _normalize_gene(value: object) -> str:
    return _normalize_token(value, "no_gene").upper()


def _normalize_chrom(value: object) -> str:
    text = str(value).replace("chr", "").replace("CHR", "").strip().upper()
    return text.lstrip("0") or "0"


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _event_rows_for_kucab(raw_dir: Path, patient_set: set[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def base_row(record: dict[str, object], modality: str) -> dict[str, object] | None:
        sample = str(record.get("Sample", ""))
        if sample not in patient_set:
            return None
        chrom = _normalize_chrom(record.get("Chrom"))
        pos = _safe_float(record.get("Pos"))
        mb = "unknown_mb" if not math.isfinite(pos) else f"chr{chrom}_mb{int(max(pos, 0.0) // 1_000_000):04d}"
        ref = str(record.get("Ref", "")).upper()
        alt = str(record.get("Alt", "")).upper()
        gene = _normalize_gene(record.get("Gene"))
        return {
            "sample": sample,
            "modality": modality,
            "chrom": f"chr{chrom}",
            "position": int(pos) if math.isfinite(pos) else 0,
            "chrom_modality": f"{modality}__chr{chrom}",
            "mb_bin": mb,
            "gene": gene,
            "gene_modality": f"{modality}__{gene}",
            "ref_alt": _normalize_token(f"{ref}>{alt}", "unknown_change"),
            "context_5p": _normalize_token(record.get("pre_context"), "unknown_context"),
            "context_3p": _normalize_token(record.get("rear_context"), "unknown_context"),
            "pm_tum": _safe_float(record.get("PM.Tum")),
            "clpm": _safe_float(record.get("CLPM")),
            "asmd": _safe_float(record.get("ASMD")),
        }

    subs = pd.read_csv(raw_dir / "denovo_subclone_subs_final.txt", sep="\t")
    for record in subs.to_dict(orient="records"):
        row = base_row(record, "SBS")
        if row is not None:
            row["event_class"] = f"SBS__{row['ref_alt']}"
            rows.append(row)

    dbs = pd.read_csv(raw_dir / "denovo_subclone_doublesub_final.txt", sep="\t")
    for record in dbs.to_dict(orient="records"):
        row = base_row(record, "DBS")
        if row is not None:
            row["event_class"] = "DBS__" + _normalize_token(record.get("dinuc_mutation"), str(row["ref_alt"]))
            row["neighbor_distance"] = _safe_float(record.get("neigbor_dist"))
            rows.append(row)

    indels = pd.read_csv(raw_dir / "denovo_subclone_indels.final.txt", sep="\t")
    for record in indels.to_dict(orient="records"):
        row = base_row(record, "ID")
        if row is not None:
            indel_type = _normalize_token(record.get("Type"), "unknown_indel_type")
            effect = _normalize_token(record.get("Effect"), "unknown_effect")
            classification = _normalize_token(record.get("classification"), "unclassified")
            row.update(
                {
                    "event_class": f"ID__{indel_type}__{effect}__{classification}",
                    "indel_type": indel_type,
                    "indel_effect": effect,
                    "indel_classification": classification,
                    "indel_change": _normalize_token(record.get("change"), "unknown_change"),
                    "indel_slice5": _normalize_token(record.get("slice5_1bp"), "unknown_slice"),
                    "indel_slice3": _normalize_token(record.get("slice3_1bp"), "unknown_slice"),
                    "vaf_tum": _safe_float(record.get("VAF.Tum")),
                    "vaf_tum_cal": _safe_float(record.get("VAF.Tum_Cal")),
                    "repcount": _safe_float(record.get("repcount")),
                    "indel_length": _safe_float(record.get("indel.length")),
                    "nr_tum": _safe_float(record.get("NR.Tum")),
                    "nu_tum": _safe_float(record.get("NU.Tum")),
                    "pr_tum": _safe_float(record.get("PR.Tum")),
                    "pu_tum": _safe_float(record.get("PU.Tum")),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _count_block(events: pd.DataFrame, patient_ids: list[str], column: str, prefix: str, *, top_n: int | None = None) -> pd.DataFrame:
    if events.empty or column not in events.columns:
        return pd.DataFrame(index=pd.Index(patient_ids, name="sample"))
    temp = events.loc[:, ["sample", column]].copy()
    temp[column] = temp[column].fillna("unknown").astype(str)
    if top_n is not None:
        patient_to_row = {str(patient): idx for idx, patient in enumerate(patient_ids)}
        matrix = np.zeros((len(patient_ids), int(top_n)), dtype=np.float32)
        for sample, value in temp.itertuples(index=False):
            row = patient_to_row.get(str(sample))
            if row is None:
                continue
            matrix[row, _stable_bucket(prefix, value, int(top_n))] += 1.0
        columns = [f"{prefix}__hash_{idx:04d}" for idx in range(int(top_n))]
        return pd.DataFrame(matrix, index=pd.Index(patient_ids, name="sample"), columns=columns).astype(np.float32)
    table = pd.crosstab(temp["sample"].astype(str), temp[column])
    table = table.reindex(patient_ids).fillna(0.0).astype(np.float32)
    table.columns = [f"{prefix}__{col}" for col in table.columns.astype(str)]
    table.index.name = "sample"
    return table


def _numeric_summary(events: pd.DataFrame, patient_ids: list[str]) -> pd.DataFrame:
    numeric_cols = [
        "pm_tum",
        "clpm",
        "asmd",
        "neighbor_distance",
        "vaf_tum",
        "vaf_tum_cal",
        "repcount",
        "indel_length",
        "nr_tum",
        "nu_tum",
        "pr_tum",
        "pu_tum",
    ]
    pieces: list[pd.DataFrame] = []
    for col in numeric_cols:
        if col not in events.columns:
            continue
        values = pd.to_numeric(events[col], errors="coerce")
        frame = pd.DataFrame({"sample": events["sample"].astype(str), col: values}).dropna(subset=[col])
        if frame.empty:
            continue
        grouped = frame.groupby("sample")[col]
        summary = pd.DataFrame(
            {
                f"kucab_num__{col}__mean": grouped.mean(),
                f"kucab_num__{col}__std": grouped.std().fillna(0.0),
                f"kucab_num__{col}__median": grouped.median(),
                f"kucab_num__{col}__q90": grouped.quantile(0.90),
            }
        )
        pieces.append(summary.reindex(patient_ids).fillna(0.0).astype(np.float32))
    return pd.concat(pieces, axis=1).fillna(0.0).astype(np.float32) if pieces else pd.DataFrame(index=pd.Index(patient_ids, name="sample"))


def _build_kucab_event_stack(ctx: RunnerContext, endpoint: EndpointData) -> tuple[pd.DataFrame, str]:
    raw_dir = Path(ctx.paths["raw_data"]["kucab_raw_dir"])
    patient_ids = endpoint.labels.index.astype(str).tolist()
    cache_key = make_cache_key(
        "main_complete_panel_kucab_event_stack",
        params={"schema": 1, "patient_ids": patient_ids},
        inputs={
            "raw": {name: file_fingerprint(raw_dir / name) for name in KUCAB_TABLES},
            "code": code_fingerprint([Path(__file__)]),
        },
    )
    cached = ctx.feature_cache.load_frame(cache_key, "features.csv.gz")
    if cached is not None:
        cached.index = cached.index.astype(str)
        return cached.astype(np.float32), cache_key
    events = _event_rows_for_kucab(raw_dir, set(patient_ids))
    if events.empty:
        raise RuntimeError("Kucab event-stack construction produced no eligible events")
    cov_rows = []
    modality_counts = events.groupby(["sample", "modality"]).size().unstack(fill_value=0.0).reindex(patient_ids).fillna(0.0)
    total = modality_counts.sum(axis=1).replace(0.0, np.nan)
    cov_rows.append(pd.DataFrame({"kucab_event__log_total_burden": np.log1p(modality_counts.sum(axis=1))}, index=modality_counts.index))
    for modality in ["SBS", "DBS", "ID"]:
        if modality not in modality_counts.columns:
            modality_counts[modality] = 0.0
        cov_rows.append(modality_counts[[modality]].rename(columns={modality: f"kucab_event__count_{modality.lower()}"}))
        cov_rows.append((modality_counts[[modality]].div(total, axis=0).fillna(0.0)).rename(columns={modality: f"kucab_event__fraction_{modality.lower()}"}))
    categorical_blocks = [
        ("modality", "kucab_modality", None),
        ("chrom", "kucab_chrom", None),
        ("chrom_modality", "kucab_chrom_modality", None),
        ("mb_bin", "kucab_mb_bin", 512),
        ("gene", "kucab_gene", 512),
        ("gene_modality", "kucab_gene_modality", 512),
        ("event_class", "kucab_event_class", 512),
        ("ref_alt", "kucab_ref_alt", 128),
        ("context_5p", "kucab_context_5p", 64),
        ("context_3p", "kucab_context_3p", 64),
        ("indel_type", "kucab_indel_type", None),
        ("indel_effect", "kucab_indel_effect", 128),
        ("indel_classification", "kucab_indel_classification", None),
        ("indel_change", "kucab_indel_change", 256),
        ("indel_slice5", "kucab_indel_slice5", 64),
        ("indel_slice3", "kucab_indel_slice3", 64),
    ]
    pieces = [*cov_rows]
    pieces.extend(_count_block(events, patient_ids, column, prefix, top_n=top_n) for column, prefix, top_n in categorical_blocks)
    pieces.append(_numeric_summary(events, patient_ids))
    frame = pd.concat(pieces, axis=1).reindex(patient_ids).fillna(0.0).astype(np.float32)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    ctx.feature_cache.save_frame(
        cache_key,
        frame,
        "features.csv.gz",
        metadata={
            "namespace": EXPERIMENT_ID,
            "representation": "Kucab event-stack analogue",
            "sample_count": int(frame.shape[0]),
            "feature_count": int(frame.shape[1]),
            "event_count": int(len(events)),
            "status": "measured_event_stack_no_tcga_maf_atlas",
        },
    )
    return frame, cache_key


def _load_kucab_features(ctx: RunnerContext, settings: dict) -> tuple[EndpointData, dict[str, tuple[pd.DataFrame, str]]]:
    grch37_dir = Path(ctx.paths["raw_data"]["grch37_dir"])
    fasta_path = find_fasta(grch37_dir)
    fasta = FastaReader(fasta_path)
    sbs, dbs, ids = read_cosmic_channels(grch37_dir)
    try:
        inventory, endpoint, inventory_cache_key = build_kucab_inventory(ctx, settings, fasta, fasta_path, sbs, dbs, ids)
    finally:
        fasta.close()
    standard = standard_features_from_counts(inventory.standard_counts, inventory.covariates)
    burden = burden_features_from_covariates(inventory.covariates)
    event_stack, event_cache_key = _build_kucab_event_stack(ctx, endpoint)
    combined = pd.concat([standard.reindex(event_stack.index), event_stack], axis=1).fillna(0.0).astype(np.float32)
    combined_cache_key = make_cache_key(
        "main_complete_panel_kucab_signatures_plus_event_stack",
        params={"schema": 1, "standard_cache_key": inventory_cache_key, "event_cache_key": event_cache_key},
        inputs={"code": code_fingerprint([Path(__file__)])},
    )
    if not ctx.feature_cache.has(combined_cache_key):
        ctx.feature_cache.save_frame(
            combined_cache_key,
            combined,
            "features.csv.gz",
            metadata={
                "namespace": EXPERIMENT_ID,
                "representation": "Kucab signatures plus event-stack analogue",
                "sample_count": int(combined.shape[0]),
                "feature_count": int(combined.shape[1]),
                "atlas_status": "no TCGA gene-consequence MAF atlas; Kucab raw event annotations plus spectra",
            },
        )
    features = {
        "burden_only": (burden.astype(np.float32), inventory_cache_key),
        "standard_sbs96_dbs78_id83": (standard.astype(np.float32), inventory_cache_key),
        "MAF_stack_only": (event_stack.astype(np.float32), event_cache_key),
        "signatures_plus_MAF_stack": (combined.astype(np.float32), combined_cache_key),
    }
    return endpoint, features


def _expected_linear_solver(task: str, learner: str, n_features: int, settings: dict) -> str | None:
    if learner != "linear" or task in {"regression", "survival"}:
        return None
    solver = str(settings.get("linear_high_dim_classification_solver", settings.get("linear_high_dim_multiclass_solver", ""))).strip().lower()
    threshold = int(settings.get("linear_high_dim_classification_solver_min_features", settings.get("linear_high_dim_multiclass_solver_min_features", settings.get("linear_process_backend_min_features", 500))))
    if task in {"binary", "multiclass"} and int(n_features) >= threshold and solver in {"sgd", "sgd_log_loss", "sgd_log_loss_elasticnet"}:
        return "sgd_log_loss_elasticnet_v1"
    return "logistic_saga_elasticnet_nested_v1"


def _high_purity_vaf_safe_candidate_sets(
    endpoint: EndpointData,
    representation: str,
    candidate_feature_sets: object,
) -> tuple[dict[str, list[str]] | None, bool]:
    if endpoint.name != "high_purity" or representation not in {"MAF_stack_only", "signatures_plus_MAF_stack"}:
        return None, False
    if not candidate_feature_sets:
        return None, False
    filtered: dict[str, list[str]] = {}
    removed_any = False
    for name, columns in dict(candidate_feature_sets).items():
        kept = [str(col) for col in list(columns) if "vaf" not in str(col).lower()]
        removed_any = removed_any or len(kept) != len(list(columns))
        if not kept:
            raise ValueError(f"High-purity VAF-safe candidate set {name} has no remaining features")
        filtered[str(name)] = kept
    return filtered, removed_any


def _same_cache_namespace(row_cache_key: object, slot_cache_key: object) -> bool:
    row_key = str(row_cache_key or "")
    slot_key = str(slot_cache_key or "")
    if row_key == slot_key:
        return True
    row_base, row_suffix = row_key.split("::", 1) if "::" in row_key else (row_key, "")
    slot_base, slot_suffix = slot_key.split("::", 1) if "::" in slot_key else (slot_key, "")
    if row_suffix or slot_suffix:
        if row_suffix != slot_suffix:
            return False
        row_key, slot_key = row_base, slot_base
    if "_" not in row_key or "_" not in slot_key:
        return False
    return row_key.rsplit("_", 1)[0] == slot_key.rsplit("_", 1)[0]


def _current_row(row: dict[str, object], ctx: RunnerContext, *, tuned: bool = False, expected_linear_solver: str | None = None) -> bool:
    if str(row.get("primary_metric_scope", "")) != "pooled_global_oof":
        return False
    if str(row.get("endpoint", "")) == "cancer_type_top20" and str(row.get("metric", "")).strip() != "balanced_accuracy":
        return False
    if str(row.get("tuning", "")) != "outer_train_inner_validation_randomized_search":
        return False
    folds = pd.to_numeric(pd.Series([row.get("n_folds", row.get("folds"))]), errors="coerce").iloc[0]
    repeats = pd.to_numeric(pd.Series([row.get("repeats", 1)]), errors="coerce").iloc[0]
    if not pd.notna(folds) or int(folds) != int(ctx.cv_folds):
        return False
    if not pd.notna(repeats) or int(repeats) != int(ctx.cv_repeats):
        return False
    if tuned:
        trials = pd.to_numeric(pd.Series([row.get("optuna_trials_completed")]), errors="coerce").iloc[0]
        if not pd.notna(trials) or int(trials) != int(ctx.optuna_trials):
            return False
    score = pd.to_numeric(pd.Series([row.get("score")]), errors="coerce").iloc[0]
    if str(row.get("learner")) == "cox_ph" and str(row.get("linear_solver", "")) not in {
        "lifelines_penalized_cox_ph_nested_v3_conditioned_variance_stepcap",
        "lifelines_penalized_cox_ph_nested_v4_conditioned_variance_highdim_stepcap",
        "fast_breslow_elasticnet_cox_nested_v1",
    }:
        return False
    if str(row.get("learner")) == "linear" and str(row.get("linear_solver", "")) not in {"sgd_log_loss_elasticnet_v1", "elastic_net_coordinate_descent_v1", "logistic_saga_elasticnet_nested_v1", "elastic_net_coordinate_descent_nested_v1"}:
        return False
    if expected_linear_solver and str(row.get("linear_solver", "")) != expected_linear_solver:
        return False
    return pd.notna(score) and np.isfinite(float(score))


def _main_model_family(learner: str) -> str:
    if learner == "linear":
        return "elastic_net"
    if learner == "cox_ph":
        return "cox_ph"
    return "XGBoost"


def _active_endpoint_names(settings: dict[str, object]) -> set[str]:
    return {
        str(endpoint)
        for endpoint in [
            *settings.get("kucab_endpoints", ["damage_class"]),
            *settings.get("main_mc3_endpoints", MC3_MAIN_ENDPOINTS),
            *settings.get("survival_endpoints", MC3_SURVIVAL_ENDPOINTS),
        ]
    }


def _filter_existing_frame(frame: pd.DataFrame, active_endpoints: set[str]) -> pd.DataFrame:
    if frame.empty or "endpoint" not in frame.columns:
        return frame
    return frame[frame["endpoint"].astype(str).isin(active_endpoints)].copy()


def run(ctx: RunnerContext) -> None:
    settings = _settings(ctx)
    ctx.tables_dir.mkdir(parents=True, exist_ok=True)
    ctx.logs_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    log_path = ctx.logs_dir / f"{EXPERIMENT_ID}.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"{EXPERIMENT_ID} started {datetime.now(timezone.utc).isoformat()}\n")
        log.write(json.dumps({k: str(v) for k, v in settings.items()}, indent=2, sort_keys=True) + "\n")
    endpoint_results_path = ctx.tables_dir / f"{EXPERIMENT_ID}_endpoint_results.csv"
    oof_predictions_path = ctx.tables_dir / f"{EXPERIMENT_ID}_oof_predictions.csv"
    fold_metrics_path = ctx.tables_dir / f"{EXPERIMENT_ID}_fold_metrics.csv"
    feature_manifest_path = ctx.tables_dir / f"{EXPERIMENT_ID}_feature_manifest.csv"
    endpoint_audit_path = ctx.tables_dir / f"{EXPERIMENT_ID}_endpoint_audit.csv"
    summary_path = ctx.tables_dir / f"{EXPERIMENT_ID}_summary.csv"
    active_endpoints = _active_endpoint_names(settings)

    if ctx.dry_run:
        planned = [*settings.get("kucab_endpoints", ["damage_class"]), *settings.get("main_mc3_endpoints", MC3_MAIN_ENDPOINTS), *settings.get("survival_endpoints", MC3_SURVIVAL_ENDPOINTS)]
        write_summary_csv([{"experiment_id": EXPERIMENT_ID, "status": "planned", "main_endpoints": ";".join(planned)}], ctx.logs_dir / f"dry_run_{EXPERIMENT_ID}_summary.csv")
        return

    endpoint_rows: list[dict[str, object]] = (
        _filter_existing_frame(
            pd.read_csv(endpoint_results_path),
            active_endpoints,
        ).to_dict("records")
        if endpoint_results_path.exists()
        else []
    )
    prediction_frames: list[pd.DataFrame] = (
        [
            _filter_existing_frame(
                pd.read_csv(oof_predictions_path),
                active_endpoints,
            )
        ]
        if oof_predictions_path.exists()
        else []
    )
    fold_frames: list[pd.DataFrame] = (
        [
            _filter_existing_frame(
                pd.read_csv(fold_metrics_path),
                active_endpoints,
            )
        ]
        if fold_metrics_path.exists()
        else []
    )
    feature_rows: list[dict[str, object]] = pd.read_csv(feature_manifest_path).to_dict("records") if feature_manifest_path.exists() else []

    def save_current() -> None:
        if endpoint_rows:
            frame = pd.DataFrame(endpoint_rows).drop_duplicates(["endpoint", "representation", "learner"], keep="last")
            atomic_write_csv(sanitize_frame(frame), endpoint_results_path, index=False)
        if prediction_frames:
            pred = pd.concat(prediction_frames, ignore_index=True, sort=False).drop_duplicates(
                ["benchmark", "endpoint", "representation", "learner", "sample"],
                keep="last",
            )
            atomic_write_csv(sanitize_frame(pred), oof_predictions_path, index=False)
        if fold_frames:
            folds = pd.concat(fold_frames, ignore_index=True, sort=False).drop_duplicates(
                ["benchmark", "endpoint", "representation", "learner", "repeat", "fold"],
                keep="last",
            )
            atomic_write_csv(sanitize_frame(folds), fold_metrics_path, index=False)
        if feature_rows:
            features = pd.DataFrame(feature_rows).drop_duplicates(["benchmark", "representation"], keep="last")
            atomic_write_csv(sanitize_frame(features), feature_manifest_path, index=False)

    def evaluate_slot(endpoint: EndpointData, features: pd.DataFrame, representation: str, learner: str, cache_key: str, construction: str) -> None:
        slot_features = features
        high_purity_vaf_safe = False
        if endpoint.name == "high_purity":
            vaf_columns = [col for col in features.columns.astype(str) if "vaf" in col.lower()]
            if vaf_columns:
                slot_features = features.drop(columns=vaf_columns, errors="ignore").copy()
                slot_features.attrs.update(features.attrs)
                high_purity_vaf_safe = True
        raw_candidate_feature_sets = features.attrs.get("candidate_feature_sets")
        filtered_candidate_feature_sets, candidate_vaf_removed = _high_purity_vaf_safe_candidate_sets(
            endpoint,
            representation,
            raw_candidate_feature_sets,
        )
        high_purity_vaf_safe = high_purity_vaf_safe or candidate_vaf_removed
        candidate_feature_sets = filtered_candidate_feature_sets if candidate_vaf_removed else raw_candidate_feature_sets
        if candidate_feature_sets:
            slot_features.attrs["candidate_feature_sets"] = candidate_feature_sets
            slot_features.attrs["candidate_feature_set_counts"] = {name: len(cols) for name, cols in dict(candidate_feature_sets).items()}
        slot_cache_key = f"{cache_key}::high_purity_no_vaf_v1" if high_purity_vaf_safe else cache_key
        slot_construction = (
            f"{construction}; high-purity endpoint excludes VAF-derived MAF columns to avoid tumor-purity proxy leakage"
            if high_purity_vaf_safe
            else construction
        )
        expected_linear_solver = _expected_linear_solver(endpoint.task, learner, int(slot_features.shape[1]), settings)
        existing = [
            row
            for row in endpoint_rows
            if str(row.get("endpoint")) == endpoint.name
            and str(row.get("representation")) == representation
            and str(row.get("learner")) == learner
            and _same_cache_namespace(row.get("cache_key", ""), slot_cache_key)
            and _current_row(row, ctx, expected_linear_solver=expected_linear_solver)
        ]
        if existing:
            print(f"[{EXPERIMENT_ID}] [checkpoint] reuse {endpoint.name} {representation} {learner}", flush=True)
            return
        print(f"[{EXPERIMENT_ID}] evaluating {endpoint.name} {representation} {learner}", flush=True)
        slot_start = time.time()
        if candidate_feature_sets:
            common = endpoint.labels.index.astype(str).intersection(slot_features.index.astype(str))
            labels = endpoint.labels.loc[common]
            groups = endpoint.groups.loc[common] if endpoint.groups is not None else None
            nested_settings = {
                **settings,
                "tree_method": str(ctx.tree_method),
                "xgb_n_jobs": int(ctx.xgb_n_jobs),
                "xgb_estimators": int(settings.get("xgb_estimators", ctx.xgb_estimators)),
                "optuna_trials": int(settings.get("random_search_trials", settings.get("optuna_trials", ctx.optuna_trials))),
                "fold_checkpoint_enabled": bool(settings.get("fold_checkpoint_enabled", True)),
                "fold_checkpoint_resume": bool(settings.get("fold_checkpoint_resume", True)),
                "fold_checkpoint_dir": str(ctx.tables_dir / f"{EXPERIMENT_ID}_nested_fold_checkpoints"),
                "fold_checkpoint_key": f"{endpoint.name}__{representation}__{learner}__{slot_cache_key}",
                "fold_progress_label": f"{EXPERIMENT_ID} {endpoint.name} {representation} {learner}",
            }
            matrices = {
                str(name): slot_features.loc[common, list(cols)].fillna(0.0).to_numpy(dtype=np.float32)
                for name, cols in dict(candidate_feature_sets).items()
            }
            result = evaluate_nested_feature_set_oof(
                samples=common.astype(str),
                feature_sets=matrices,
                labels=labels,
                task=endpoint.task,
                benchmark=endpoint.benchmark,
                endpoint=endpoint.name,
                representation=representation,
                learner=learner,
                groups=groups,
                settings=nested_settings,
                n_splits=int(ctx.cv_folds),
                seed=stable_seed(endpoint.benchmark, endpoint.name, learner, representation, slot_cache_key, "bio_v4_feature_sets"),
            )
            row, pred_frame, fold_frame = dict(result.summary), result.predictions, result.fold_diagnostics
        else:
            row, pred_frame, fold_frame = evaluate_features(endpoint, slot_features, representation, learner, ctx, settings)
        elapsed = round(time.time() - slot_start, 3)
        row.update(
            {
                "experiment_id": EXPERIMENT_ID,
                "model_family": _main_model_family(learner),
                "cache_key": slot_cache_key,
                "oof_prediction_file": oof_predictions_path.name,
                "fold_metrics_file": fold_metrics_path.name,
                "optuna_trials_completed": 0,
                "run_id": ctx.run_id,
                "status": "measured",
                "atlas_status_note": slot_construction,
                "runtime_seconds": elapsed,
            }
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"completed endpoint={endpoint.name} representation={representation} learner={learner} seconds={elapsed}\n")
        endpoint_rows.append(row)
        prediction_frames.append(pred_frame)
        if not fold_frame.empty:
            fold_frames.append(fold_frame)
        feature_rows.append(
            {
                "benchmark": endpoint.benchmark,
                "representation": representation,
                "n_features": int(slot_features.shape[1]),
                "n_samples": int(slot_features.shape[0]),
                "cache_key": slot_cache_key,
                "construction": slot_construction,
                "candidate_feature_sets": ";".join(sorted(dict(candidate_feature_sets))) if candidate_feature_sets else "",
                "candidate_feature_set_counts_json": json.dumps(slot_features.attrs.get("candidate_feature_set_counts") or {}, sort_keys=True) if candidate_feature_sets else "",
            }
        )
        save_current()

    mc3_endpoints, endpoint_audit = _load_mc3_endpoints(
        ctx,
        settings.get("main_mc3_endpoints", MC3_MAIN_ENDPOINTS),
        settings.get("survival_endpoints", MC3_SURVIVAL_ENDPOINTS),
        settings,
    )
    if not endpoint_audit.empty:
        atomic_write_csv(sanitize_frame(endpoint_audit), endpoint_audit_path, index=False)
    burden, standard, mc3_standard_cache_key = _load_mc3_standard_features(ctx)
    mc3_features: dict[str, tuple[pd.DataFrame, str, str]] = {
        "burden_only": (burden, mc3_standard_cache_key, "MC3 cached burden-only matrix"),
        "standard_sbs96_id83": (standard, mc3_standard_cache_key, "MC3 cached SBS96+ID83 signature matrix"),
    }
    for model_id in _maf_model_ids(settings):
        frame, cache_key = _load_mc3_maf_features(ctx, model_id, settings)
        representation = MAF_REPRESENTATION[model_id]
        if model_id in BIO_MAF_V4_MODEL_IDS:
            counts = frame.attrs.get("candidate_feature_set_counts") or {}
            construction = (
                f"MC3/HRD {model_id}; Bio MAF v4 with fixed external resources and nested inner-loop "
                f"feature-block selection ({json.dumps(counts, sort_keys=True)})"
            )
        else:
            raise ValueError(f"Unsupported active Bio MAF v4 model_id: {model_id}")
        mc3_features[representation] = (frame, cache_key, construction)

    for endpoint in mc3_endpoints:
        for representation, (features, cache_key, construction) in mc3_features.items():
            if endpoint.task == "survival":
                learners = SURVIVAL_LEARNERS
            else:
                learners = MAF_MAIN_ONLY_LEARNERS if representation in {"MAF_stack_only", "signatures_plus_MAF_stack"} else LEARNERS
            for learner in learners:
                evaluate_slot(endpoint, features, representation, learner, cache_key, construction)

    kucab_endpoint, kucab_features = _load_kucab_features(ctx, settings)
    for representation, (features, cache_key) in kucab_features.items():
        construction = (
            "Kucab event-stack analogue; fixed capped categorical event-token features; no TCGA gene-consequence MAF atlas"
            if "MAF" in representation
            else "Kucab raw-event spectra/burden from FASTA-validated event inventory"
        )
        for learner in LEARNERS:
            evaluate_slot(kucab_endpoint, features, representation, learner, cache_key, construction)

    save_current()
    measured = pd.DataFrame(endpoint_rows)
    main_slots = {
        (endpoint, representation, learner)
        for endpoint in [*settings.get("kucab_endpoints", ["damage_class"]), *settings.get("main_mc3_endpoints", MC3_MAIN_ENDPOINTS), *settings.get("survival_endpoints", MC3_SURVIVAL_ENDPOINTS)]
        for representation in ["burden_only", "standard_sbs96_id83", "standard_sbs96_dbs78_id83", "MAF_stack_only", "signatures_plus_MAF_stack"]
        for learner in (SURVIVAL_LEARNERS if endpoint in set(settings.get("survival_endpoints", MC3_SURVIVAL_ENDPOINTS)) else MAF_MAIN_ONLY_LEARNERS if endpoint != "damage_class" and representation in {"MAF_stack_only", "signatures_plus_MAF_stack"} else LEARNERS)
        if not (endpoint != "damage_class" and representation == "standard_sbs96_dbs78_id83")
        if not (endpoint == "damage_class" and representation == "standard_sbs96_id83")
    }
    observed = {(str(row["endpoint"]), str(row["representation"]), str(row["learner"])) for _, row in measured.iterrows()}
    missing = sorted(main_slots - observed)
    summary_rows = [
        {
            "experiment_id": EXPERIMENT_ID,
            "status": "completed" if not missing else "incomplete",
            "elapsed_seconds": round(time.time() - start, 3),
            "rows": int(len(measured)),
            "missing_slots": len(missing),
        }
    ]
    if missing:
        for endpoint, representation, learner in missing:
            summary_rows.append({"experiment_id": EXPERIMENT_ID, "status": "missing", "endpoint": endpoint, "representation": representation, "learner": learner})
    write_summary_csv(summary_rows, summary_path)
    atomic_write_json(
        ctx.logs_dir / f"{EXPERIMENT_ID}_manifest.json",
        {
            "completed_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "elapsed_seconds": round(time.time() - start, 3),
            "endpoint_results": endpoint_results_path.name,
            "oof_predictions": oof_predictions_path.name,
            "fold_metrics": fold_metrics_path.name,
            "feature_manifest": feature_manifest_path.name,
            "endpoint_audit": endpoint_audit_path.name,
            "missing_slots": ["/".join(item) for item in missing],
        },
    )
