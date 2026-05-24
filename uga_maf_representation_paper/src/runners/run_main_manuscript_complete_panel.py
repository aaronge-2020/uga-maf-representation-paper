"""Complete the measured main-manuscript benchmark panel.

This runner fills visible main-figure slots that are not covered by the
legacy XGBoost-only MAF runner or by the one-hot KME scout.  It writes after
every endpoint/representation/model job so the same command can be resumed
after interruption without losing completed folds.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from runners.run_one_hot_event_kme_scout import (
    EndpointData,
    FastaReader,
    build_kucab_inventory,
    burden_features_from_covariates,
    evaluate_features,
    find_fasta,
    read_cosmic_channels,
    standard_features_from_counts,
)
from utils.checkpointing import atomic_write_csv, atomic_write_json
from utils.feature_cache import code_fingerprint, directory_fingerprint, file_fingerprint, make_cache_key
from utils.runner_support import (
    RunnerContext,
    prepare_legacy_workspace,
    restore_legacy_feature_snapshot,
    sanitize_frame,
    write_summary_csv,
)


EXPERIMENT_ID = "main_manuscript_complete_panel"
MAIN_ENDPOINTS = ["damage_class", "HRD_Score", "hrd_binary_33", "cancer_type_top10", "os_event"]
MC3_MAIN_ENDPOINTS = ["HRD_Score", "hrd_binary_33", "cancer_type_top10", "os_event"]
LEARNERS = ["linear", "xgboost"]
MAF_MAIN_ONLY_LEARNERS = LEARNERS
MAF_MODEL_IDS = [
    "maf_only_best_gene_locus_multiscale_stack",
    "id_plus_best_gene_locus_multiscale_stack",
]
MAF_REPRESENTATION = {
    "maf_only_best_gene_locus_multiscale_stack": "MAF_stack_only",
    "id_plus_best_gene_locus_multiscale_stack": "signatures_plus_MAF_stack",
}
KUCAB_TABLES = [
    "denovo_subclone_subs_final.txt",
    "denovo_subclone_doublesub_final.txt",
    "denovo_subclone_indels.final.txt",
    "README.txt",
]


def _settings(ctx: RunnerContext) -> dict:
    one_hot = dict(((ctx.settings.get("experiments") or {}).get("one_hot_event_kme_scout") or {}))
    local = dict(((ctx.settings.get("experiments") or {}).get(EXPERIMENT_ID) or {}))
    merged = {**one_hot, **local}
    merged.setdefault("d_context", 6)
    merged.setdefault("d_payload", 6)
    merged.setdefault("xgb_estimators", ctx.xgb_estimators)
    merged.setdefault("xgb_max_depth", 2)
    merged.setdefault("xgb_learning_rate", 0.05)
    merged["main_mc3_endpoints"] = list(local.get("main_mc3_endpoints") or one_hot.get("main_mc3_endpoints") or MC3_MAIN_ENDPOINTS)
    return merged


def _load_mc3_endpoints(ctx: RunnerContext, requested: Iterable[str]) -> list[EndpointData]:
    requested = [str(value) for value in requested]
    hrd_path = Path(ctx.paths["raw_data"]["hrd_assets_dir"]) / "cohort" / "final_analysis_cohort.tsv"
    cohort = pd.read_csv(hrd_path, sep="\t")
    cohort["patient_id_12"] = cohort["patient_id_12"].astype(str)
    labels_path = Path(ctx.paths["raw_data"]["mc3_source_dir"]) / "biology_labels.csv"
    mc3_labels = pd.read_csv(labels_path, index_col=0)
    mc3_labels.index = mc3_labels.index.astype(str)
    endpoints: list[EndpointData] = []
    for endpoint in requested:
        if endpoint in {"HRD_Score", "eCARD", "HRD_TAI", "HRD_LST", "HRD_LOH", "PARPi7"} and endpoint in cohort.columns:
            data = cohort.dropna(subset=[endpoint]).copy()
            labels = pd.Series(data[endpoint].astype(float).to_numpy(), index=data["patient_id_12"].astype(str), name=endpoint)
            endpoints.append(EndpointData(endpoint, "mc3_main", "regression", labels))
        elif endpoint in {"hrd_binary_24", "hrd_binary_33", "hrd_binary_42", "parpi7_binary"} and endpoint in cohort.columns:
            allowed = ["PARPi-high", "PARPi-low"] if endpoint == "parpi7_binary" else ["HRD-high", "HRD-low"]
            positive = "PARPi-high" if endpoint == "parpi7_binary" else "HRD-high"
            data = cohort[cohort[endpoint].isin(allowed)].copy()
            labels = pd.Series((data[endpoint] == positive).astype(int).to_numpy(), index=data["patient_id_12"].astype(str), name=endpoint)
            endpoints.append(EndpointData(endpoint, "mc3_main", "binary", labels))
        elif endpoint in mc3_labels.columns:
            labels = mc3_labels[endpoint].dropna()
            if endpoint == "cancer_type_top10":
                counts = labels.astype(str).value_counts()
                labels = labels.astype(str)
                labels = labels[labels.isin(counts[counts >= 50].index)]
                task = "multiclass"
            else:
                labels = labels.astype(int)
                task = "binary"
            endpoints.append(EndpointData(endpoint, "mc3_main", task, labels))
    return endpoints


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


def _load_maf_search_module(ctx: RunnerContext):
    workspace = prepare_legacy_workspace(ctx, require_inputs=not ctx.dry_run)
    exp_root = workspace / "cgr_validation_results" / "research" / "experiments" / "exploratory" / "2026_05_16_xgboost_maf_event_gene_coordinate_search"
    restore_legacy_feature_snapshot(ctx, "maf_event_gene_locus", exp_root)
    code_dir = exp_root / "code"
    prior_dir = workspace / "cgr_validation_results" / "research" / "experiments" / "exploratory" / "2026_05_15_maf_event_coordinate_geometry_optimization" / "code"
    project_root = workspace
    for path in (code_dir, prior_dir, project_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module_path = code_dir / "run_xgboost_maf_event_gene_coordinate_search.py"
    module_name = "_main_complete_panel_maf_search"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import MAF search module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, exp_root


def _load_mc3_maf_features(ctx: RunnerContext, model_id: str) -> tuple[pd.DataFrame, str]:
    raw_dir = Path(ctx.paths["raw_data"]["mc3_source_dir"])
    cache_key = make_cache_key(
        f"main_complete_panel_mc3_{model_id}",
        params={"model_id": model_id, "schema": 1},
        inputs={
            "mc3_source": directory_fingerprint(raw_dir, patterns=("features/*", "raw/mc3.v0.2.8.PUBLIC.maf.gz")),
            "code": code_fingerprint([Path(__file__)]),
        },
    )
    cached = ctx.feature_cache.load_frame(cache_key, "features.csv.gz")
    if cached is not None:
        cached.index = cached.index.astype(str)
        return cached.astype(np.float32), cache_key
    search, _exp_root = _load_maf_search_module(ctx)
    _standard_sbs, _standard_id, standard_sbs_id, _burden = search.base.load_feature_matrices()
    patient_ids = standard_sbs_id.index.astype(str).tolist()
    identity = search.exact_identity_frame(standard_sbs_id.astype(np.float32))
    raw_blocks = search.build_gene_coordinate_blocks(patient_ids)
    raw_blocks.update(search.load_reference_locus_blocks(patient_ids))
    blocks = search.build_transformed_blocks(raw_blocks)
    spec_map = {spec.model_id: spec for spec in search.make_candidate_specs()}
    if model_id not in spec_map:
        raise ValueError(f"Unknown MAF feature model_id: {model_id}")
    frame = search.build_candidate_frame(spec_map[model_id], identity, blocks).fillna(0.0).astype(np.float32)
    frame.index = frame.index.astype(str)
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
        },
    )
    return frame, cache_key


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
    table = pd.crosstab(temp["sample"].astype(str), temp[column])
    table = table.reindex(patient_ids).fillna(0.0).astype(np.float32)
    if top_n is not None and table.shape[1] > top_n:
        keep = table.var(axis=0).sort_values(ascending=False).head(top_n).index
        table = table.loc[:, keep]
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
    sbs, dbs, ids = read_cosmic_channels()
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


def _current_row(row: dict[str, object], ctx: RunnerContext, *, tuned: bool = False) -> bool:
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
    if str(row.get("learner")) == "linear" and str(row.get("linear_solver", "")) not in {"sgd_log_loss_elasticnet_v1", "elastic_net_coordinate_descent_v1"}:
        return False
    return pd.notna(score) and np.isfinite(float(score))


def _main_model_family(learner: str) -> str:
    return "elastic_net" if learner == "linear" else "XGBoost"


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
    summary_path = ctx.tables_dir / f"{EXPERIMENT_ID}_summary.csv"

    if ctx.dry_run:
        write_summary_csv([{"experiment_id": EXPERIMENT_ID, "status": "planned", "main_endpoints": ";".join(MAIN_ENDPOINTS)}], summary_path)
        return

    endpoint_rows: list[dict[str, object]] = pd.read_csv(endpoint_results_path).to_dict("records") if endpoint_results_path.exists() else []
    prediction_frames: list[pd.DataFrame] = [pd.read_csv(oof_predictions_path)] if oof_predictions_path.exists() else []
    fold_frames: list[pd.DataFrame] = [pd.read_csv(fold_metrics_path)] if fold_metrics_path.exists() else []
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
        existing = [
            row
            for row in endpoint_rows
            if str(row.get("endpoint")) == endpoint.name
            and str(row.get("representation")) == representation
            and str(row.get("learner")) == learner
            and _current_row(row, ctx)
        ]
        if existing:
            print(f"[{EXPERIMENT_ID}] [checkpoint] reuse {endpoint.name} {representation} {learner}", flush=True)
            return
        print(f"[{EXPERIMENT_ID}] evaluating {endpoint.name} {representation} {learner}", flush=True)
        slot_start = time.time()
        row, pred_frame, fold_frame = evaluate_features(endpoint, features, representation, learner, ctx, settings)
        elapsed = round(time.time() - slot_start, 3)
        row.update(
            {
                "experiment_id": EXPERIMENT_ID,
                "model_family": _main_model_family(learner),
                "cache_key": cache_key,
                "oof_prediction_file": oof_predictions_path.name,
                "fold_metrics_file": fold_metrics_path.name,
                "optuna_trials_completed": 0,
                "run_id": ctx.run_id,
                "status": "measured",
                "atlas_status_note": construction,
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
                "n_features": int(features.shape[1]),
                "n_samples": int(features.shape[0]),
                "cache_key": cache_key,
                "construction": construction,
            }
        )
        save_current()

    mc3_endpoints = _load_mc3_endpoints(ctx, settings.get("main_mc3_endpoints", MC3_MAIN_ENDPOINTS))
    burden, standard, mc3_standard_cache_key = _load_mc3_standard_features(ctx)
    mc3_features: dict[str, tuple[pd.DataFrame, str, str]] = {
        "burden_only": (burden, mc3_standard_cache_key, "MC3 cached burden-only matrix"),
        "standard_sbs96_id83": (standard, mc3_standard_cache_key, "MC3 cached SBS96+ID83 signature matrix"),
    }
    for model_id in MAF_MODEL_IDS:
        frame, cache_key = _load_mc3_maf_features(ctx, model_id)
        representation = MAF_REPRESENTATION[model_id]
        construction = f"MC3/HRD {model_id}; TCGA MAF gene/locus/event stack"
        mc3_features[representation] = (frame, cache_key, construction)

    for endpoint in mc3_endpoints:
        for representation, (features, cache_key, construction) in mc3_features.items():
            learners = MAF_MAIN_ONLY_LEARNERS if representation in {"MAF_stack_only", "signatures_plus_MAF_stack"} else LEARNERS
            for learner in learners:
                evaluate_slot(endpoint, features, representation, learner, cache_key, construction)

    kucab_endpoint, kucab_features = _load_kucab_features(ctx, settings)
    for representation, (features, cache_key) in kucab_features.items():
        construction = "Kucab event-stack analogue; no TCGA gene-consequence MAF atlas" if "MAF" in representation else "Kucab raw-event spectra/burden from FASTA-validated event inventory"
        for learner in LEARNERS:
            evaluate_slot(kucab_endpoint, features, representation, learner, cache_key, construction)

    save_current()
    measured = pd.DataFrame(endpoint_rows)
    main_slots = {
        (endpoint, representation, learner)
        for endpoint in MAIN_ENDPOINTS
        for representation in ["burden_only", "standard_sbs96_id83", "standard_sbs96_dbs78_id83", "MAF_stack_only", "signatures_plus_MAF_stack"]
        for learner in (MAF_MAIN_ONLY_LEARNERS if endpoint != "damage_class" and representation in {"MAF_stack_only", "signatures_plus_MAF_stack"} else LEARNERS)
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
            "missing_slots": ["/".join(item) for item in missing],
        },
    )
