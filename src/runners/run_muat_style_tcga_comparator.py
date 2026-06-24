"""MuAt-compatible TCGA WES comparator and smoke-run harness."""

from __future__ import annotations

import json
import os
import time
import hashlib
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

from utils.checkpointing import atomic_write_csv, atomic_write_json
from utils.endpoint_registry import (
    CANCER_TYPE_ENDPOINT_CLASSES,
    HRD_CONTINUOUS,
    build_cdr_cancer_type_labels,
    build_luad_kmt2c_labels,
    load_endpoint_registry,
    muat_comparator_endpoints,
)
from utils.muat_compatible import (
    OfficialMuAtCLI,
    build_dictionaries,
    build_fixed_hash_dictionaries,
    build_muat_event_table,
    collect_maf_patients,
    dictionary_summary,
    encode_events,
    expected_calibration_error,
    hash_muat_event_tokens,
    load_tcga_20_type_labels,
    sample_smoke_labels,
    stable_sha256,
    topk_accuracy,
)
from utils.nested_oof import assign_oof_fold, assert_all_oof_assigned, inner_train_val_split, outer_splits, safe_macro_auroc, safe_micro_auroc
from utils.runner_support import RunnerContext, sanitize_frame, sanitize_text, write_summary_csv


EXPERIMENT_ID = "muat_style_tcga_comparator"
PRIMARY_ENDPOINT = "tcga_20_type"
LOCAL_REPRESENTATION = "MuAt-compatible reimplementation"
TCGA_WES_TOP20_ENDPOINTS = {PRIMARY_ENDPOINT, "cancer_type_top20"}
SURVIVAL_ENDPOINTS = {"OS", "DSS", "PFI", "DFI"}
MUAT_PAPER_TCGA_WES_N_SAMPLES = 7352
MUAT_PAPER_TCGA_WES_N_CLASSES = 20
MUAT_PAPER_TCGA_WES_ACCURACY = 0.641
MUAT_PAPER_TCGA_WES_TOP5_ACCURACY = 0.906


@dataclass(frozen=True)
class LocalMuAtSettings:
    embed_dim: int = 32
    attention_heads: int = 1
    num_layers: int = 1
    feature_dim: int = 24
    dropout: float = 0.1
    count_feature_mode: str = "none"
    pooling_mode: str = "attention_weighted"
    scheduler: str = "none"
    lr: float = 6e-4
    momentum: float = 0.9
    weight_decay: float = 0.0
    epochs: int = 1
    batch_size: int = 1


def _load_secondary_endpoint_labels(ctx: RunnerContext, endpoints: list[str]) -> dict[str, tuple[str, pd.Series | pd.DataFrame]]:
    """Load manuscript endpoints that can be modeled from MuAt-compatible bags."""

    out: dict[str, tuple[str, pd.Series | pd.DataFrame]] = {}
    endpoints = [endpoint for endpoint in endpoints if str(endpoint) != "damage_class"]
    survival_requested = sorted(str(endpoint) for endpoint in endpoints if str(endpoint) in SURVIVAL_ENDPOINTS)
    if survival_requested:
        raise ValueError(
            "MuAt-compatible neural survival endpoints are disabled; "
            f"use the main scikit-survival Cox pipeline for {survival_requested}"
        )
    if not endpoints:
        return out
    mc3_dir = Path(ctx.paths["raw_data"]["mc3_source_dir"])
    labels = pd.read_csv(mc3_dir / "biology_labels.csv", index_col=0)
    labels.index = labels.index.astype(str)
    feature_patient_index = pd.read_csv(mc3_dir / "features" / "features_burden_only.csv", usecols=[0], index_col=0).index.astype(str)
    hrd = pd.read_csv(Path(ctx.paths["raw_data"]["hrd_assets_dir"]) / "cohort" / "final_analysis_cohort.tsv", sep="\t")
    hrd["patient_id_12"] = hrd["patient_id_12"].astype(str)
    for endpoint in endpoints:
        if endpoint in out:
            continue
        if endpoint in CANCER_TYPE_ENDPOINT_CLASSES:
            y = build_cdr_cancer_type_labels(
                mc3_dir,
                feature_patient_index,
                CANCER_TYPE_ENDPOINT_CLASSES[endpoint],
                endpoint_name=endpoint,
            ).astype(str)
            out[endpoint] = ("multiclass", y)
        elif endpoint in HRD_CONTINUOUS and endpoint in hrd.columns:
            data = hrd.dropna(subset=[endpoint]).drop_duplicates("patient_id_12").copy()
            out[endpoint] = ("regression", pd.Series(data[endpoint].astype(float).to_numpy(), index=data["patient_id_12"], name=endpoint))
        elif endpoint in hrd.columns and str(endpoint).startswith("hrd_binary"):
            data = hrd[hrd[endpoint].isin(["HRD-high", "HRD-low"])].drop_duplicates("patient_id_12").copy()
            out[endpoint] = ("binary", pd.Series((data[endpoint] == "HRD-high").astype(int).to_numpy(), index=data["patient_id_12"], name=endpoint))
        elif endpoint in labels.columns:
            out[endpoint] = ("binary", labels[endpoint].dropna().astype(int))
        elif endpoint == "luad_kmt2c_mutated":
            driver_index = pd.read_csv(mc3_dir / "driver_gene_labels_functional.csv", index_col=0).index.astype(str)
            out[endpoint] = ("binary", build_luad_kmt2c_labels(mc3_dir, driver_index))
    return out


def _clean_kucab_token(value: object, default: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "na", "n/a", "<na>"}:
        return default
    return "".join(ch if ch.isalnum() or ch in {"_", "-", ">", "[", "]"} else "_" for ch in text)


def _load_kucab_muat_endpoint(ctx: RunnerContext, settings: dict[str, Any]) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Build Kucab damage-class labels and MuAt-compatible event rows."""

    from runners.run_main_manuscript_complete_panel import _event_rows_for_kucab, _load_kucab_features

    kucab_settings = dict(settings)
    kucab_settings.setdefault("d_context", 3)
    kucab_settings.setdefault("d_payload", 8)
    kucab_settings.setdefault("kucab_min_total_events", 20)
    endpoint, _features = _load_kucab_features(ctx, kucab_settings)
    labels = endpoint.labels.astype(str)
    groups = endpoint.groups.astype(str) if endpoint.groups is not None else pd.Series(labels.index.astype(str), index=labels.index)
    raw_events = _event_rows_for_kucab(Path(ctx.paths["raw_data"]["kucab_raw_dir"]), set(labels.index.astype(str)))
    if raw_events.empty:
        raise RuntimeError("Kucab MuAt-compatible event construction produced no events")
    raw_events = raw_events[raw_events["sample"].astype(str).isin(labels.index.astype(str))].copy()
    raw_events["position"] = pd.to_numeric(raw_events.get("position", 0), errors="coerce").fillna(0).astype(np.int64)

    def motif(row: pd.Series) -> str:
        event_class = _clean_kucab_token(row.get("event_class"), "unknown_event")
        left = _clean_kucab_token(row.get("context_5p"), "N")
        right = _clean_kucab_token(row.get("context_3p"), "N")
        return f"{left}__{event_class}__{right}"

    def annotation(row: pd.Series) -> str:
        modality = _clean_kucab_token(row.get("modality"), "unknown_modality")
        gene = _clean_kucab_token(row.get("gene"), "unknown_gene")
        indel_type = _clean_kucab_token(row.get("indel_type"), "")
        indel_effect = _clean_kucab_token(row.get("indel_effect"), "")
        return f"{modality}__gene_{gene}__itype_{indel_type}__ieffect_{indel_effect}"

    events = pd.DataFrame(
        {
            "sample": raw_events["sample"].astype(str),
            "chromosome": raw_events.get("chrom", pd.Series("chrNA", index=raw_events.index)).astype(str).str.replace("^chr", "", regex=True),
            "position": raw_events["position"].astype(np.int64),
            "motif": raw_events.apply(motif, axis=1),
            "position_token": raw_events.get("mb_bin", pd.Series("unknown_mb", index=raw_events.index)).astype(str),
            "annotation": raw_events.apply(annotation, axis=1),
            "variant_type": raw_events.get("modality", pd.Series("", index=raw_events.index)).astype(str),
            "variant_classification": raw_events.get("event_class", pd.Series("", index=raw_events.index)).astype(str),
            "gene": raw_events.get("gene", pd.Series("", index=raw_events.index)).astype(str),
            "ref": "",
            "alt": "",
            "context": raw_events.get("context_5p", pd.Series("", index=raw_events.index)).astype(str)
            + "N"
            + raw_events.get("context_3p", pd.Series("", index=raw_events.index)).astype(str),
        }
    ).sort_values(["sample", "chromosome", "position", "motif"], kind="mergesort")
    audit = _endpoint_audit("damage_class", labels)
    event_counts = events.groupby("sample").size().rename("n_run_events")
    audit = audit.merge(
        pd.DataFrame({"sample": labels.index.astype(str), "tumour_type": labels.astype(str).to_numpy()})
        .join(event_counts, on="sample")
        .groupby("tumour_type", as_index=False)
        .agg(n_run_patients=("sample", "nunique"), n_run_events=("n_run_events", "sum")),
        how="left",
        on="tumour_type",
    )
    audit["n_run_patients"] = audit["n_run_patients"].fillna(0).astype(int)
    audit["n_run_events"] = audit["n_run_events"].fillna(0).astype(int)
    return labels, groups, events.reset_index(drop=True), audit


def _endpoint_audit(endpoint: str, labels: pd.Series, *, included: bool = True) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame(
            [{"endpoint": endpoint, "tumour_type": "", "n_patients": 0, "included": False, "reason": "no eligible labels"}]
        )
    rows = []
    for value, n_patients in labels.astype(str).value_counts().sort_index().items():
        rows.append(
            {
                "endpoint": endpoint,
                "tumour_type": str(value),
                "n_patients": int(n_patients),
                "included": bool(included),
                "reason": "primary MuAt-compatible task",
            }
        )
    return pd.DataFrame(rows)


def _settings(settings: dict[str, Any]) -> LocalMuAtSettings:
    return LocalMuAtSettings(
        embed_dim=int(settings.get("embed_dim", settings.get("embedding_dim", 32))),
        attention_heads=int(settings.get("attention_heads", 1)),
        num_layers=int(settings.get("num_layers", 1)),
        feature_dim=int(settings.get("feature_dim", 24)),
        dropout=float(settings.get("dropout", 0.1)),
        count_feature_mode=str(settings.get("count_feature_mode", "none")),
        pooling_mode=str(settings.get("pooling_mode", "attention_weighted")),
        scheduler=str(settings.get("scheduler", "none")),
        lr=float(settings.get("lr", 6e-4)),
        momentum=float(settings.get("momentum", 0.9)),
        weight_decay=float(settings.get("weight_decay", 0.0)),
        epochs=int(settings.get("epochs", 1)),
        batch_size=int(settings.get("batch_size", 1)),
    )


def _setting_int_list(settings: dict[str, Any], key: str, default: list[int]) -> list[int]:
    value = settings.get(key)
    if value is None:
        return [int(item) for item in default]
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]
    out: list[int] = []
    for item in raw_items:
        parsed = int(item)
        if parsed not in out:
            out.append(parsed)
    return out or [int(item) for item in default]


def _architecture_candidates(settings: dict[str, Any], base: LocalMuAtSettings) -> list[LocalMuAtSettings]:
    if not bool(settings.get("architecture_search_enabled", False)):
        return [base]
    embed_dims = _setting_int_list(settings, "architecture_search_embed_dims", [128, 256, 512])
    num_layers = _setting_int_list(settings, "architecture_search_num_layers", [1, 2, 4])
    attention_heads = _setting_int_list(settings, "architecture_search_attention_heads", [1, 2])
    candidates: list[LocalMuAtSettings] = []
    seen: set[tuple[int, int, int]] = set()
    for embed_dim, layers, heads in product(embed_dims, num_layers, attention_heads):
        width = int(embed_dim) * 3
        if width % int(heads) != 0:
            raise ValueError(f"Invalid MuAt architecture candidate: 3 * embed_dim={width} is not divisible by attention_heads={heads}")
        key = (int(embed_dim), int(layers), int(heads))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            LocalMuAtSettings(
                embed_dim=int(embed_dim),
                attention_heads=int(heads),
                num_layers=int(layers),
                feature_dim=base.feature_dim,
                dropout=base.dropout,
                count_feature_mode=base.count_feature_mode,
                pooling_mode=base.pooling_mode,
                scheduler=base.scheduler,
                lr=base.lr,
                momentum=base.momentum,
                weight_decay=base.weight_decay,
                epochs=base.epochs,
                batch_size=base.batch_size,
            )
        )
    if len(candidates) < 18 and set(embed_dims) >= {128, 256, 512} and set(num_layers) >= {1, 2, 4} and set(attention_heads) >= {1, 2}:
        raise ValueError("MuAt architecture search grid unexpectedly contains fewer than the required 18 candidates")
    return candidates or [base]


def _most_common_model_settings(rows: list[dict[str, object]], fallback: LocalMuAtSettings) -> LocalMuAtSettings:
    values = [str(row.get("model_settings_json", "")) for row in rows if row.get("model_settings_json")]
    if not values:
        return fallback
    counts = pd.Series(values).value_counts(sort=True)
    payload = json.loads(str(counts.index[0]))
    return LocalMuAtSettings(**{key: payload.get(key, getattr(fallback, key)) for key in asdict(fallback)})


def _cache_key(maf_path: Path, patients: list[str], settings: dict[str, Any]) -> str:
    stat = maf_path.stat()
    payload = {
        "maf_path": str(maf_path),
        "maf_size": int(stat.st_size),
        "maf_mtime_ns": int(stat.st_mtime_ns),
        "patients": stable_sha256(list(map(str, patients))),
        "settings": {
            "max_events_per_sample": settings.get("max_events_per_sample"),
            "maf_chunksize": settings.get("maf_chunksize"),
            "preprocessor": "muat_compatible_v1",
            "dictionary_mode": settings.get("dictionary_mode", "observed"),
            "motif_hash_buckets": settings.get("motif_hash_buckets"),
            "position_hash_buckets": settings.get("position_hash_buckets"),
            "annotation_hash_buckets": settings.get("annotation_hash_buckets"),
        },
    }
    return stable_sha256(payload)[:24]


def _write_fidelity_report(path: Path, *, official_available: bool, smoke_run: bool, n_types: int, n_samples: int) -> None:
    lines = [
        "# MuAt-Compatible Fidelity Report",
        "",
        f"- Official MuAt CLI available: `{official_available}`.",
        f"- Local run mode: `{'smoke' if smoke_run else 'full_local'}`.",
        f"- TCGA task: `{n_types}` tumour types, `{n_samples}` patients in this run.",
        "",
        "## Supported",
        "- TCGA WES SNV/MNV/indel mutation rows from bundled MC3.",
        "- Explicit motif, 1-Mb position, and genic/exonic/strand dictionaries.",
        "- Separate modality embeddings, stacked Q/K/V self-attention, residual/normalization blocks, attention-weighted set pooling, and a 24-dimensional tumour-feature layer.",
        "- Limited architecture search over embedding size, self-attention layer count, and attention-head count before outer-fold refitting.",
        "",
        "## Partial",
        "- Motif recovery uses MC3 `CONTEXT` where available; rows without reference context fall back to `N` flanks.",
        "- `cancer_type_top20` uses the fixed manuscript class list and canonical matched-feature folds; `tcga_20_type` uses the MuAt-style top-20-by-count label loader.",
        "- The source paper clearly describes sequence-context encodings, but the exact auxiliary biological annotation vocabulary is not fully specified in the bundled manuscript sources. This implementation uses deterministic MAF/VEP-derived genic, exonic, and strand tokens; broader Bio MAF v4 external-resource features are part of the tabular benchmark rather than this MuAt-compatible event bag.",
        "",
        "## Unsupported In Bundled TCGA WES",
        "- PCAWG WGS training, GEL/ICGC/CRC external validation, SV/MEI modalities, and official pretrained checkpoint claims unless the official MuAt CLI/checkpoints are configured and executed.",
        "",
        "## MuAt Paper Reference",
        f"- TCGA-WES reference: `{MUAT_PAPER_TCGA_WES_N_SAMPLES}` tumours, `{MUAT_PAPER_TCGA_WES_N_CLASSES}` tumour types, accuracy `{MUAT_PAPER_TCGA_WES_ACCURACY:.3f}`, top-5 accuracy `{MUAT_PAPER_TCGA_WES_TOP5_ACCURACY:.3f}`.",
        "- The local result is a same-fold MuAt-compatible comparator, not an official MuAt replication, unless the official CLI status above is true.",
        "",
        "## Naming Rule",
        "- Results from this local model must be labelled `MuAt-compatible reimplementation`, not official `MuAt`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_name(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def _torch_autocast(torch_module: Any, device: Any, enabled: bool):
    if not enabled or str(device).split(":", 1)[0] != "cuda":
        return nullcontext()
    if hasattr(torch_module, "amp") and hasattr(torch_module.amp, "autocast"):
        return torch_module.amp.autocast("cuda")
    return torch_module.cuda.amp.autocast()


def _new_grad_scaler(torch_module: Any, enabled: bool):
    if not enabled:
        return None
    try:
        return torch_module.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch_module.cuda.amp.GradScaler(enabled=True)


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or len(np.unique(np.asarray(y_true, dtype=float))) < 2:
        return float("nan")
    value = spearmanr(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float), nan_policy="omit").correlation
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def _checkpoint_paths(checkpoint_dir: Path, endpoint: str, fold: int) -> dict[str, Path]:
    stem = f"{_safe_name(endpoint)}_fold{int(fold):02d}"
    return {
        "meta": checkpoint_dir / f"{stem}_meta.json",
        "pred": checkpoint_dir / f"{stem}_predictions.csv",
        "fold": checkpoint_dir / f"{stem}_fold_metrics.csv",
        "attention": checkpoint_dir / f"{stem}_attention.csv",
        "features": checkpoint_dir / f"{stem}_features.csv",
        "split": checkpoint_dir / f"{stem}_split.csv",
        "training": checkpoint_dir / f"{stem}_training.pt",
    }


def _write_fold_checkpoint(paths: dict[str, Path], *, fingerprint: str, frames: dict[str, pd.DataFrame], meta: dict[str, object]) -> None:
    for key, frame in frames.items():
        if key in paths:
            atomic_write_csv(sanitize_frame(frame), paths[key], index=False)
    atomic_write_json(paths["meta"], {**meta, "fingerprint": fingerprint})


def _model_settings_match(fold_frame: pd.DataFrame, expected_model_settings: dict[str, object] | None) -> bool:
    if expected_model_settings is None:
        return True
    if fold_frame.empty or "model_settings_json" not in fold_frame.columns:
        return False
    values = fold_frame["model_settings_json"].dropna().astype(str)
    if values.empty:
        return False
    try:
        observed = json.loads(values.iloc[0])
    except Exception:
        return False
    return observed == expected_model_settings


def _checkpoint_split_matches(split_frame: pd.DataFrame, expected_test_samples: set[str] | None) -> bool:
    if expected_test_samples is None:
        return True
    if split_frame.empty or not {"sample", "split"}.issubset(split_frame.columns):
        return False
    observed = set(split_frame.loc[split_frame["split"].astype(str).eq("test"), "sample"].astype(str))
    return observed == expected_test_samples


def _read_fold_checkpoint(
    paths: dict[str, Path],
    *,
    fingerprint: str,
    endpoint: str | None = None,
    fold: int | None = None,
    expected_model_settings: dict[str, object] | None = None,
    expected_test_samples: set[str] | None = None,
    compatible_fingerprints: set[str] | None = None,
) -> dict[str, pd.DataFrame] | None:
    if not paths["meta"].exists():
        return None
    try:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    except Exception:
        return None
    checkpoint_fingerprint = str(meta.get("fingerprint", ""))
    exact_match = checkpoint_fingerprint == fingerprint
    compatible_match = checkpoint_fingerprint in (compatible_fingerprints or set())
    if not exact_match and not compatible_match:
        return None
    required = ["pred", "fold", "attention", "features", "split"]
    if not all(paths[key].exists() for key in required):
        return None
    frames = {key: pd.read_csv(paths[key]) for key in required}
    if exact_match:
        return frames
    if endpoint is not None and str(meta.get("endpoint")) != str(endpoint):
        return None
    if fold is not None and int(meta.get("fold", -1)) != int(fold):
        return None
    if not _checkpoint_split_matches(frames["split"], expected_test_samples):
        return None
    if not _model_settings_match(frames["fold"], expected_model_settings):
        return None
    if "endpoint" in frames["fold"].columns and not frames["fold"]["endpoint"].astype(str).eq(str(endpoint)).all():
        return None
    if "endpoint" in frames["pred"].columns and not frames["pred"]["endpoint"].astype(str).eq(str(endpoint)).all():
        return None
    return frames


def _trim_to_observed_events(x_batch: Any, mask_batch: Any, *, enabled: bool = True) -> tuple[Any, Any]:
    """Drop padded event positions from a batch before self-attention."""

    if not enabled:
        return x_batch, mask_batch
    try:
        observed = int(mask_batch.long().sum(dim=1).max().item())
    except Exception:
        return x_batch, mask_batch
    observed = max(1, observed)
    return x_batch[:, :observed, :], mask_batch[:, :observed]


def _classification_score(metric: str, y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    metric = str(metric or "").lower()
    pred = np.argmax(proba, axis=1)
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, pred))
    if metric == "accuracy":
        return float(accuracy_score(y_true, pred))
    if metric == "macro_f1":
        return float(f1_score(y_true, pred, average="macro", zero_division=0))
    if n_classes == 2 and metric in {"auroc", "macro_auroc"}:
        try:
            return float(roc_auc_score(y_true, proba[:, 1]))
        except ValueError:
            return float("nan")
    return safe_macro_auroc(y_true, proba, n_classes)


def _endpoint_validation_score(metric: str, task: str, y_true: Any, prediction: np.ndarray, n_classes: int) -> float:
    if task in {"binary", "multiclass"}:
        return _classification_score(metric, np.asarray(y_true), prediction, n_classes)
    if task == "regression":
        return _safe_spearman(np.asarray(y_true, dtype=float), prediction.reshape(-1))
    if task == "survival":
        raise ValueError("MuAt-compatible survival endpoints are disabled; use the main scikit-survival Cox pipeline")
    return float("nan")


def _default_selection_metric(endpoint: str, task: str) -> str:
    if task == "regression":
        return "spearman"
    if task == "survival":
        raise ValueError("MuAt-compatible survival endpoints are disabled; use the main scikit-survival Cox pipeline")
    if task == "multiclass" and str(endpoint) in TCGA_WES_TOP20_ENDPOINTS:
        return "balanced_accuracy"
    return "macro_auroc"


def _resolve_output_relative_path(path_value: object, ctx: RunnerContext) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    bundle_root = ctx.tables_dir.parent.parent
    return (bundle_root / path).resolve()


def _canonical_split_signature(split_frame: pd.DataFrame) -> str:
    payload = split_frame.loc[:, ["sample", "fold"]].sort_values(["sample", "fold"]).to_dict(orient="records")
    return stable_sha256(payload)


def _ndarray_digest(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()[:24]


def _label_digest(task: str, labels: object) -> str:
    if task == "survival":
        frame = pd.DataFrame(labels).loc[:, ["time", "event"]].copy()
        payload = {
            "time": pd.to_numeric(frame["time"], errors="coerce").fillna(-1).astype(float).round(10).tolist(),
            "event": pd.to_numeric(frame["event"], errors="coerce").fillna(-1).astype(int).tolist(),
        }
        return stable_sha256(payload)[:24]
    return _ndarray_digest(np.asarray(labels))


def _load_canonical_split_samples(
    source_path: str | Path,
    *,
    endpoint: str,
    representation: str,
    learner: str,
) -> pd.Index:
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Configured split_manifest_source does not exist: {path}")
    required = {"sample", "endpoint", "representation", "learner"}
    header = pd.read_csv(path, nrows=0)
    missing = required - set(header.columns)
    if missing:
        raise ValueError(f"Split manifest source {path} is missing columns: {sorted(missing)}")
    frame = pd.read_csv(path, usecols=sorted(required), low_memory=False)
    frame = frame[
        frame["endpoint"].astype(str).eq(str(endpoint))
        & frame["representation"].astype(str).eq(str(representation))
        & frame["learner"].astype(str).eq(str(learner))
    ].copy()
    if frame.empty:
        raise ValueError(
            "Split manifest source has no rows for "
            f"endpoint={endpoint}, representation={representation}, learner={learner}"
        )
    return pd.Index(frame["sample"].drop_duplicates().astype(str), name="sample")


def _load_canonical_split_manifest(
    source_path: str | Path,
    *,
    endpoint: str,
    representation: str,
    learner: str,
    samples: pd.Index,
    expected_folds: int,
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], pd.DataFrame, str]:
    """Load exact outer-fold assignments from a prior OOF prediction file."""

    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Configured split_manifest_source does not exist: {path}")
    required = {"sample", "endpoint", "representation", "learner", "fold"}
    header = pd.read_csv(path, nrows=0)
    missing = required - set(header.columns)
    if missing:
        raise ValueError(f"Split manifest source {path} is missing columns: {sorted(missing)}")
    frame = pd.read_csv(path, usecols=sorted(required), low_memory=False)
    frame = frame[
        frame["endpoint"].astype(str).eq(str(endpoint))
        & frame["representation"].astype(str).eq(str(representation))
        & frame["learner"].astype(str).eq(str(learner))
    ].copy()
    if frame.empty:
        raise ValueError(
            "Split manifest source has no rows for "
            f"endpoint={endpoint}, representation={representation}, learner={learner}"
        )
    frame["sample"] = frame["sample"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="coerce").astype("Int64")
    if frame["fold"].isna().any():
        raise ValueError("Split manifest source contains non-numeric fold values")
    duplicate_samples = frame.loc[frame["sample"].duplicated(), "sample"].unique()
    if len(duplicate_samples):
        example = ", ".join(map(str, duplicate_samples[:5]))
        raise ValueError(f"Split manifest source has duplicate sample rows after filtering; examples: {example}")

    sample_index = pd.Index(samples.astype(str), name="sample")
    frame = frame.set_index("sample")
    missing_samples = sample_index.difference(frame.index)
    extra_samples = frame.index.difference(sample_index)
    if len(missing_samples) or len(extra_samples):
        raise ValueError(
            "Canonical split samples do not match MuAt endpoint labels "
            f"(missing={len(missing_samples)}, extra={len(extra_samples)})"
        )
    frame = frame.loc[sample_index].reset_index()
    folds = sorted(int(value) for value in frame["fold"].unique())
    if len(folds) != int(expected_folds):
        raise ValueError(f"Canonical split has {len(folds)} folds, expected {expected_folds}: {folds}")
    splits: list[tuple[int, np.ndarray, np.ndarray]] = []
    all_idx = np.arange(len(sample_index), dtype=int)
    fold_array = frame["fold"].astype(int).to_numpy()
    for fold in folds:
        test_idx = np.flatnonzero(fold_array == int(fold))
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        if len(test_idx) == 0 or len(train_idx) == 0:
            raise ValueError(f"Canonical split fold {fold} is empty")
        splits.append((int(fold), train_idx, test_idx))
    return splits, frame, _canonical_split_signature(frame)


def _write_main_panel_model_comparison(ctx: RunnerContext, output_prefix: str, muat_results: pd.DataFrame) -> None:
    main_path = ctx.tables_dir / "main_manuscript_complete_panel_endpoint_results.csv"
    if not main_path.exists() or muat_results.empty:
        return
    endpoints = set(muat_results.get("endpoint", pd.Series(dtype=str)).dropna().astype(str))
    main = pd.read_csv(main_path)
    main = main[main["endpoint"].astype(str).isin(endpoints)].copy()
    muat = muat_results[muat_results["endpoint"].astype(str).isin(endpoints)].copy()
    if main.empty or muat.empty:
        return
    keep = [
        "benchmark",
        "endpoint",
        "task",
        "representation",
        "learner",
        "metric",
        "score",
        "macro_auroc",
        "micro_auroc",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "cohen_kappa",
        "top3_accuracy",
        "top5_accuracy",
        "n_samples",
        "n_classes",
        "n_features",
        "n_folds",
        "primary_metric_scope",
        "validation_protocol",
        "canonical_split_source",
        "canonical_split_signature",
    ]
    for frame in (main, muat):
        for column in keep:
            if column not in frame.columns:
                frame[column] = np.nan
        macro_mask = frame["metric"].astype(str).eq("macro_auroc") & frame["macro_auroc"].isna()
        frame.loc[macro_mask, "macro_auroc"] = pd.to_numeric(frame.loc[macro_mask, "score"], errors="coerce")
    comparison = pd.concat([main[keep], muat[keep]], ignore_index=True, sort=False)
    comparison["comparison_note"] = np.where(
        comparison["representation"].astype(str).eq(LOCAL_REPRESENTATION),
        "MuAt-compatible direct comparator on canonical endpoint-specific outer folds; balanced accuracy is primary for cancer_type_top20, while accuracy/top-5 columns are retained for MuAt-paper TCGA-WES reference",
        "Existing main-panel tabular result",
    )
    comparison_path = ctx.tables_dir / f"{output_prefix}_main_panel_comparison.csv"
    atomic_write_csv(sanitize_frame(comparison), comparison_path, index=False)


def _sanitize_json_like(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [_sanitize_json_like(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_json_like(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_json_like(item) for key, item in value.items()}
    return value


def _official_cli_audit(ctx: RunnerContext, settings: dict[str, Any], maf_path: Path) -> dict[str, object]:
    cli = OfficialMuAtCLI(str(settings.get("muat_executable")) if settings.get("muat_executable") else None)
    info = _sanitize_json_like(cli.version())
    result_dir = ctx.tables_dir / f"{EXPERIMENT_ID}_official_muat"
    command = cli.predict_pretrained_command(
        assay="wes",
        mutation_type=str(settings.get("official_mutation_type", "snv+mnv+indel")),
        input_filepath=maf_path,
        result_dir=result_dir,
        ensemble=True,
    )
    record = _sanitize_json_like(cli.command_record(command, input_paths=[maf_path]))
    return {"status": "available" if cli.available else "not_available", "version": info, "pretrained_wes_predict_ensemble_command": record}


def _fit_predict_local(
    *,
    endpoint: str,
    task: str,
    benchmark: str = "tcga_mc3_wes",
    labels: pd.Series | pd.DataFrame,
    groups: pd.Series | np.ndarray | None = None,
    patients: list[str],
    x: np.ndarray,
    mask: np.ndarray,
    dictionaries: Any,
    settings: dict[str, Any],
    ctx: RunnerContext,
    device: Any,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if task == "survival":
        raise ValueError("MuAt-compatible survival endpoints are disabled; use the main scikit-survival Cox pipeline")

    import torch
    from torch.utils.data import DataLoader, Subset, TensorDataset

    from utils.muat_compatible import MuAtCompatibleModel

    local = _settings(settings)
    architecture_candidates = _architecture_candidates(settings, local)
    architecture_search_enabled = len(architecture_candidates) > 1
    architecture_search_epochs = max(
        1,
        min(
            int(local.epochs),
            int(settings.get("architecture_search_epochs", local.epochs)),
        ),
    )
    smoke_run = bool(settings.get("smoke_run", True))
    primary_scope = "smoke_oof" if smoke_run else "pooled_global_oof"
    patient_index = pd.Index(patients, name="sample")
    label_index = pd.Index(pd.DataFrame(labels).index.astype(str))
    split_manifest_cfg = dict((settings.get("split_manifest_by_endpoint") or {}).get(endpoint) or {})
    split_manifest_source = split_manifest_cfg.get("source", settings.get("split_manifest_source"))
    split_manifest_endpoint = split_manifest_cfg.get("endpoint", settings.get("split_manifest_endpoint") or endpoint)
    split_manifest_representation = split_manifest_cfg.get("representation", settings.get("split_manifest_representation") or "")
    split_manifest_learner = split_manifest_cfg.get("learner", settings.get("split_manifest_learner") or "")
    canonical_split_source = str(_resolve_output_relative_path(split_manifest_source, ctx)) if split_manifest_source and not smoke_run else ""
    if canonical_split_source:
        manifest_samples = _load_canonical_split_samples(
            canonical_split_source,
            endpoint=str(split_manifest_endpoint),
            representation=str(split_manifest_representation),
            learner=str(split_manifest_learner),
        )
        missing_from_labels = manifest_samples.difference(label_index)
        missing_from_patients = manifest_samples.difference(patient_index)
        if len(missing_from_labels) or len(missing_from_patients):
            raise ValueError(
                "Canonical split samples are not available for MuAt endpoint "
                f"{endpoint} (missing_labels={len(missing_from_labels)}, missing_encoded_patients={len(missing_from_patients)})"
            )
        common = manifest_samples
    else:
        common = patient_index.intersection(label_index)
    row_idx = np.array([patient_index.get_loc(patient) for patient in common], dtype=np.int64)
    x_np = x[row_idx]
    mask_np = mask[row_idx]
    group_arr = None
    if groups is not None:
        if isinstance(groups, pd.Series):
            group_series = groups.copy()
            group_series.index = group_series.index.astype(str)
            missing_groups = common.difference(pd.Index(group_series.index.astype(str)))
            if len(missing_groups):
                raise ValueError(f"Missing grouped split labels for endpoint={endpoint}: {len(missing_groups)} samples")
            group_arr = group_series.loc[common.astype(str)].astype(str).to_numpy()
        else:
            group_arr = np.asarray(groups)
            if len(group_arr) != len(common):
                raise ValueError(f"Grouped split array for endpoint={endpoint} has length {len(group_arr)} but expected {len(common)}")
    if task == "multiclass":
        y_series = pd.Series(labels.loc[common]).astype(str)
        classes = np.array(sorted(y_series.unique()), dtype=object)
        y_np = y_series.map({label: i for i, label in enumerate(classes)}).to_numpy(dtype=np.int64)
        y_split = y_np
    elif task == "binary":
        classes = np.array([0, 1], dtype=object)
        y_np = pd.Series(labels.loc[common]).astype(int).to_numpy(dtype=np.int64)
        y_split = y_np
    elif task == "regression":
        classes = np.array(["value"], dtype=object)
        y_np = pd.Series(labels.loc[common]).astype(float).to_numpy(dtype=np.float32)
        y_split = y_np
    else:
        raise ValueError(f"Unsupported MuAt-compatible task for endpoint={endpoint}: {task}")

    n_splits = int(settings.get("smoke_folds" if smoke_run else "folds", 2 if smoke_run else ctx.cv_folds))
    if task in {"binary", "multiclass"}:
        n_splits = max(2, min(n_splits, int(pd.Series(y_np).value_counts().min())))
    else:
        n_splits = max(2, min(n_splits, int(len(common))))
    split_task = task
    canonical_split_frame = pd.DataFrame()
    canonical_split_signature = ""
    if canonical_split_source:
        splits, canonical_split_frame, canonical_split_signature = _load_canonical_split_manifest(
            canonical_split_source,
            endpoint=str(split_manifest_endpoint),
            representation=str(split_manifest_representation),
            learner=str(split_manifest_learner),
            samples=common.astype(str),
            expected_folds=n_splits,
        )
    else:
        splits = outer_splits(split_task, y_split, groups=group_arr, seed=int(settings.get("seed", 20260524)), n_splits=n_splits)

    checkpoint_dir = ctx.feature_cache_dir / EXPERIMENT_ID / "fold_checkpoints" / ("smoke" if smoke_run else "full")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    default_selection_metric = _default_selection_metric(endpoint, task)
    fingerprint_payload = {
            "endpoint": endpoint,
            "benchmark": benchmark,
            "task": task,
            "scope": primary_scope,
            "samples": list(map(str, common)),
            "labels_digest": _label_digest(task, y_np),
            "groups": stable_sha256(list(map(str, group_arr))) if group_arr is not None else "",
            "classes": list(map(str, classes)),
            "model": asdict(local),
            "architecture_search_enabled": bool(architecture_search_enabled),
            "architecture_search_epochs": int(architecture_search_epochs),
            "architecture_candidates": [asdict(candidate) for candidate in architecture_candidates],
            "nested_validation": bool(settings.get("nested_validation", False)) and not smoke_run,
            "selection_metric": str((settings.get("selection_metric_by_endpoint") or {}).get(endpoint) or settings.get("selection_metric") or settings.get("primary_metric") or default_selection_metric).lower(),
            "canonical_split_source": canonical_split_source,
            "canonical_split_signature": canonical_split_signature,
            "max_events": int(x_np.shape[1]),
            "encoded_events_digest": _ndarray_digest(x_np),
            "encoded_mask_digest": _ndarray_digest(mask_np.astype(np.uint8)),
            "dictionary_sizes": dictionaries.sizes(),
            "dictionary_digest": stable_sha256(dictionaries.to_jsonable())[:24],
            "dictionary_mode": str(settings.get("dictionary_mode", "observed")),
            "motif_hash_buckets": settings.get("motif_hash_buckets"),
            "position_hash_buckets": settings.get("position_hash_buckets"),
            "annotation_hash_buckets": settings.get("annotation_hash_buckets"),
            "trim_to_observed_events": bool(settings.get("trim_to_observed_events", True)),
            "n_splits": int(n_splits),
        }
    if task == "regression":
        fingerprint_payload["regression_target_standardization"] = "fold_local_zscore_v1"
        fingerprint_payload["regression_prediction_scale"] = "outer_train_inverse_transform_v1"
    fingerprint = stable_sha256(fingerprint_payload)

    fold_rows: list[dict[str, object]] = []
    attention_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    preload_tensors = bool(settings.get("preload_tensors_to_device", getattr(device, "type", str(device)) == "cuda"))
    effective_batch_size = int(local.batch_size)
    if task == "regression":
        effective_batch_size = int(settings.get("regression_batch_size", local.batch_size))
    effective_batch_size = max(1, effective_batch_size)
    if preload_tensors:
        x_tensor = torch.from_numpy(x_np).long().to(device)
        mask_tensor = torch.from_numpy(mask_np).bool().to(device)
        y_tensor = None
        if task in {"binary", "multiclass"}:
            y_tensor = torch.from_numpy(np.asarray(y_np, dtype=np.int64)).long().to(device)
        elif task == "regression":
            y_tensor = torch.from_numpy(np.asarray(y_np, dtype=np.float32)).float().to(device)
        base_dataset = None
    else:
        x_tensor = None
        mask_tensor = None
        y_tensor = None
        if task in {"binary", "multiclass"}:
            base_dataset = TensorDataset(torch.from_numpy(x_np).long(), torch.from_numpy(mask_np).bool(), torch.from_numpy(np.asarray(y_np, dtype=np.int64)).long())
        elif task == "regression":
            base_dataset = TensorDataset(torch.from_numpy(x_np).long(), torch.from_numpy(mask_np).bool(), torch.from_numpy(np.asarray(y_np, dtype=np.float32)).float())
        else:
            raise ValueError(f"Unsupported MuAt-compatible task for endpoint={endpoint}: {task}")
    use_amp = bool(settings.get("amp", True)) and getattr(device, "type", str(device)) == "cuda"
    trim_to_observed = bool(settings.get("trim_to_observed_events", True))
    nested_validation = bool(settings.get("nested_validation", False)) and (
        not smoke_run or bool(settings.get("nested_validation_in_smoke", False))
    )
    selection_metric = str((settings.get("selection_metric_by_endpoint") or {}).get(endpoint) or settings.get("selection_metric") or settings.get("primary_metric") or default_selection_metric).lower()
    dataloader_workers = int(settings.get("dataloader_workers", 0))
    train_loader_args: dict[str, object] = {
        "batch_size": effective_batch_size,
        "shuffle": True,
        "pin_memory": getattr(device, "type", str(device)) == "cuda",
        "num_workers": max(0, dataloader_workers),
    }
    if dataloader_workers > 0:
        train_loader_args["persistent_workers"] = True
        train_loader_args["prefetch_factor"] = int(settings.get("prefetch_factor", 2))

    def make_model(model_settings: LocalMuAtSettings) -> Any:
        return MuAtCompatibleModel(
            motif_vocab_size=len(dictionaries.motif),
            position_vocab_size=len(dictionaries.position),
            annotation_vocab_size=len(dictionaries.annotation),
            n_outputs=len(classes) if task in {"binary", "multiclass"} else 1,
            embed_dim=model_settings.embed_dim,
            attention_heads=model_settings.attention_heads,
            num_layers=model_settings.num_layers,
            feature_dim=model_settings.feature_dim,
            dropout=model_settings.dropout,
            count_feature_mode=model_settings.count_feature_mode,
            pooling_mode=model_settings.pooling_mode,
        ).to(device)

    def make_scheduler(optimizer: Any, model_settings: LocalMuAtSettings, max_epochs: int) -> Any:
        mode = str(model_settings.scheduler or "none").strip().lower()
        if mode in {"", "none", "off", "disabled"}:
            return None
        if mode in {"cosine", "cosine_annealing"}:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(max_epochs)))
        if mode in {"step", "steplr"}:
            step_size = max(1, int(settings.get("scheduler_step_size", max(1, int(max_epochs) // 3))))
            gamma = float(settings.get("scheduler_gamma", 0.5))
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        raise ValueError(f"Unsupported MuAt scheduler: {mode}")

    def train_one_epoch(
        model: Any,
        optimizer: Any,
        scaler: Any,
        indices: np.ndarray,
        *,
        fold_seed: int,
        epoch: int,
        loader: Any = None,
        target_mean: float = 0.0,
        target_scale: float = 1.0,
    ) -> float:
        model.train()
        losses = []
        if preload_tensors:
            assert x_tensor is not None and mask_tensor is not None
            order = np.asarray(indices, dtype=np.int64).copy()
            np.random.default_rng(int(settings.get("seed", 20260524)) + int(fold_seed) * 100_000 + int(epoch)).shuffle(order)
            assert y_tensor is not None
            batches = (
                (
                    x_tensor[torch.as_tensor(order[start : start + effective_batch_size], dtype=torch.long, device=device)],
                    mask_tensor[torch.as_tensor(order[start : start + effective_batch_size], dtype=torch.long, device=device)],
                    y_tensor[torch.as_tensor(order[start : start + effective_batch_size], dtype=torch.long, device=device)],
                )
                for start in range(0, len(order), effective_batch_size)
            )
        else:
            assert loader is not None
            batches = loader
        for batch in batches:
            batch_x, batch_mask, batch_y = batch
            batch_x = batch_x.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_x, batch_mask = _trim_to_observed_events(batch_x, batch_mask, enabled=trim_to_observed)
            optimizer.zero_grad(set_to_none=True)
            with _torch_autocast(torch, device, use_amp):
                logits, _features = model(batch_x, batch_mask)
                if task in {"binary", "multiclass"}:
                    loss = loss_fn(logits, batch_y)
                elif task == "regression":
                    target = (batch_y.float().reshape(-1) - float(target_mean)) / max(float(target_scale), 1e-6)
                    loss = loss_fn(logits.reshape(-1), target)
                else:
                    raise ValueError(f"Unsupported MuAt-compatible task for endpoint={endpoint}: {task}")
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else float("nan")

    def predict_arrays(
        model: Any,
        indices: np.ndarray,
        *,
        model_settings: LocalMuAtSettings,
        return_attention: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        model.eval()
        eval_batch_size = max(1, int(settings.get("eval_batch_size", model_settings.batch_size)))
        outputs_np = np.zeros((len(indices), len(classes) if task in {"binary", "multiclass"} else 1), dtype=float)
        features = np.zeros((len(indices), model_settings.feature_dim), dtype=float)
        attentions: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        with torch.no_grad():
            for start in range(0, len(indices), eval_batch_size):
                batch_positions = np.asarray(indices[start : start + eval_batch_size], dtype=np.int64)
                if preload_tensors:
                    assert x_tensor is not None and mask_tensor is not None
                    batch_index = torch.as_tensor(batch_positions, dtype=torch.long, device=device)
                    test_x = x_tensor[batch_index]
                    test_mask = mask_tensor[batch_index]
                else:
                    test_x = torch.as_tensor(x_np[batch_positions], dtype=torch.long, device=device)
                    test_mask = torch.as_tensor(mask_np[batch_positions], dtype=torch.bool, device=device)
                test_x, test_mask = _trim_to_observed_events(test_x, test_mask, enabled=trim_to_observed)
                with _torch_autocast(torch, device, use_amp):
                    outputs = model(test_x, test_mask, return_attention=return_attention)
                if return_attention:
                    logits, batch_features, attention = outputs
                    attentions.append((batch_positions, attention.float().cpu().numpy(), mask_np[batch_positions]))
                else:
                    logits, batch_features = outputs
                if task in {"binary", "multiclass"}:
                    batch_output = torch.softmax(logits, dim=1).float().cpu().numpy()
                else:
                    batch_output = logits.reshape(-1, 1).float().cpu().numpy()
                outputs_np[start : start + len(batch_positions)] = batch_output
                features[start : start + len(batch_positions)] = batch_features.float().cpu().numpy()
        return outputs_np, features, attentions

    sample_folds = np.full(len(common), -1, dtype=int)
    for fold, train_idx, test_idx in splits:
        paths = _checkpoint_paths(checkpoint_dir, endpoint, int(fold))
        compatible_fingerprints = set()
        if bool(settings.get("allow_compatible_checkpoint_fingerprints", False)):
            compatible_fingerprints = set(
                str(value)
                for value in (
                    ((settings.get("compatible_checkpoint_fingerprints") or {}).get(endpoint))
                    or settings.get("compatible_checkpoint_fingerprints")
                    or []
                )
            )
        expected_test_samples = set(map(str, common[np.asarray(test_idx, dtype=int)]))
        cached = (
            _read_fold_checkpoint(
                paths,
                fingerprint=fingerprint,
                endpoint=endpoint,
                fold=int(fold),
                expected_model_settings=asdict(local),
                expected_test_samples=expected_test_samples,
                compatible_fingerprints=compatible_fingerprints,
            )
            if bool(settings.get("resume_checkpoints", True))
            else None
        )
        if cached is not None:
            print(f"[{EXPERIMENT_ID}] reused checkpoint endpoint={endpoint} fold={fold}/{n_splits}", flush=True)
            assign_oof_fold(sample_folds, test_idx, int(fold))
            pred_frames.append(cached["pred"])
            fold_rows.extend(cached["fold"].to_dict(orient="records"))
            attention_rows.extend(cached["attention"].to_dict(orient="records"))
            feature_frames.append(cached["features"])
            split_rows.extend(cached["split"].to_dict(orient="records"))
            continue

        print(
            f"[{EXPERIMENT_ID}] training endpoint={endpoint} fold={fold}/{n_splits} "
            f"train={len(train_idx)} test={len(test_idx)} epochs={local.epochs}",
            flush=True,
        )
        torch.manual_seed(int(settings.get("seed", 20260524)) + int(fold))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(settings.get("seed", 20260524)) + int(fold))
        if task in {"binary", "multiclass"}:
            loss_fn = torch.nn.CrossEntropyLoss()
        elif task == "regression":
            loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported MuAt-compatible task for endpoint={endpoint}: {task}")

        inner_train_n = float("nan")
        inner_val_n = float("nan")
        selected_epoch = int(local.epochs)
        selected_inner_score = float("nan")
        selected_local = architecture_candidates[0]
        candidate_results: list[dict[str, object]] = []
        selected_architecture_candidate_id = 1
        inner_train_idx = np.asarray(train_idx)
        inner_val_idx = np.asarray([], dtype=int)
        if nested_validation:
            inner_train_idx, inner_val_idx = inner_train_val_split(split_task, y_split, np.asarray(train_idx), groups=group_arr, seed=int(settings.get("seed", 20260524)) + int(fold))
            inner_train_n = int(len(inner_train_idx))
            inner_val_n = int(len(inner_val_idx))
            progress_interval = max(1, int(settings.get("progress_interval_epochs", 10)))
            early_patience = int((settings.get("early_stopping_patience_by_endpoint") or {}).get(endpoint, settings.get("early_stopping_patience", 0) or 0))
            best_candidate_score = -np.inf
            best_candidate_epoch = 1
            print(
                f"[{EXPERIMENT_ID}] endpoint={endpoint} fold={fold}/{n_splits} "
                f"architecture_search_candidates={len(architecture_candidates)} search_epochs={architecture_search_epochs}",
                flush=True,
            )
            for candidate_id, candidate_local in enumerate(architecture_candidates, start=1):
                torch.manual_seed(int(settings.get("seed", 20260524)) + int(fold) * 1000 + int(candidate_id))
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(int(settings.get("seed", 20260524)) + int(fold) * 1000 + int(candidate_id))
                selector_model = make_model(candidate_local)
                selector_optimizer = torch.optim.SGD(
                    selector_model.parameters(),
                    lr=candidate_local.lr,
                    momentum=candidate_local.momentum,
                    weight_decay=candidate_local.weight_decay,
                )
                selector_scheduler = make_scheduler(selector_optimizer, candidate_local, architecture_search_epochs)
                selector_scaler = _new_grad_scaler(torch, use_amp)
                selector_target_mean = 0.0
                selector_target_scale = 1.0
                if task == "regression":
                    selector_targets = np.asarray(y_np, dtype=float)[inner_train_idx]
                    selector_target_mean = float(np.nanmean(selector_targets))
                    selector_target_scale = float(np.nanstd(selector_targets))
                    if not np.isfinite(selector_target_scale) or selector_target_scale <= 0:
                        selector_target_scale = 1.0
                if base_dataset is not None:
                    selector_ds = Subset(base_dataset, list(map(int, inner_train_idx)))
                    selector_generator = torch.Generator()
                    selector_generator.manual_seed(int(settings.get("seed", 20260524)) + int(fold) * 10 + int(candidate_id))
                    selector_loader = DataLoader(selector_ds, generator=selector_generator, **train_loader_args)
                else:
                    selector_loader = None
                candidate_best_epoch = 1
                candidate_best_score = -np.inf
                selection_history: list[dict[str, object]] = []
                for epoch in range(1, architecture_search_epochs + 1):
                    train_loss = train_one_epoch(
                        selector_model,
                        selector_optimizer,
                        selector_scaler,
                        inner_train_idx,
                        fold_seed=int(fold) * 10 + int(candidate_id),
                        epoch=epoch,
                        loader=selector_loader,
                        target_mean=selector_target_mean,
                        target_scale=selector_target_scale,
                    )
                    if selector_scheduler is not None:
                        selector_scheduler.step()
                    val_prediction, _val_features, _val_attention = predict_arrays(
                        selector_model,
                        inner_val_idx,
                        model_settings=candidate_local,
                        return_attention=False,
                    )
                    val_truth = np.asarray(y_np)[inner_val_idx]
                    val_score = _endpoint_validation_score(selection_metric, task, val_truth, val_prediction, len(classes))
                    selection_history.append({"epoch": int(epoch), "train_loss": float(train_loss), "validation_score": float(val_score)})
                    if np.isfinite(val_score) and (val_score > candidate_best_score or (val_score == candidate_best_score and epoch < candidate_best_epoch)):
                        candidate_best_score = float(val_score)
                        candidate_best_epoch = int(epoch)
                    if epoch == 1 or epoch == architecture_search_epochs or epoch % progress_interval == 0:
                        print(
                            f"[{EXPERIMENT_ID}] endpoint={endpoint} fold={fold}/{n_splits} "
                            f"candidate={candidate_id}/{len(architecture_candidates)} "
                            f"embed={candidate_local.embed_dim} layers={candidate_local.num_layers} heads={candidate_local.attention_heads} "
                            f"selection_epoch={epoch}/{architecture_search_epochs} inner_{selection_metric}={val_score:.6g}",
                            flush=True,
                        )
                    if early_patience > 0 and epoch - candidate_best_epoch >= early_patience:
                        print(
                            f"[{EXPERIMENT_ID}] endpoint={endpoint} fold={fold}/{n_splits} "
                            f"candidate={candidate_id} selection_early_stop_epoch={epoch} best_epoch={candidate_best_epoch} patience={early_patience}",
                            flush=True,
                        )
                        break
                candidate_results.append(
                    {
                        "candidate_id": int(candidate_id),
                        "params": asdict(candidate_local),
                        "selected_m_star": int(candidate_best_epoch),
                        "selected_inner_score": float(candidate_best_score),
                        "selection_history": selection_history,
                    }
                )
                if np.isfinite(candidate_best_score) and (
                    candidate_best_score > best_candidate_score
                    or (candidate_best_score == best_candidate_score and candidate_id < selected_architecture_candidate_id)
                ):
                    best_candidate_score = float(candidate_best_score)
                    best_candidate_epoch = int(candidate_best_epoch)
                    selected_local = candidate_local
                    selected_architecture_candidate_id = int(candidate_id)
                del selector_model, selector_optimizer
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            selected_epoch = int(best_candidate_epoch)
            selected_inner_score = float(best_candidate_score)
            if not np.isfinite(selected_inner_score):
                raise RuntimeError(f"No valid MuAt architecture candidate for endpoint={endpoint} fold={fold}")
            print(
                f"[{EXPERIMENT_ID}] endpoint={endpoint} fold={fold}/{n_splits} "
                f"selected_candidate={selected_architecture_candidate_id} embed={selected_local.embed_dim} "
                f"layers={selected_local.num_layers} heads={selected_local.attention_heads} "
                f"selected_epoch={selected_epoch} selected_inner_{selection_metric}={selected_inner_score:.6g}",
                flush=True,
            )
        else:
            candidate_results = [
                {
                    "candidate_id": 1,
                    "params": asdict(selected_local),
                    "selected_m_star": int(selected_epoch),
                    "selected_inner_score": selected_inner_score,
                    "selection_history": [],
                }
            ]

        model = make_model(selected_local)
        optimizer = torch.optim.SGD(model.parameters(), lr=selected_local.lr, momentum=selected_local.momentum, weight_decay=selected_local.weight_decay)
        scheduler = make_scheduler(optimizer, selected_local, selected_epoch)
        if base_dataset is not None:
            train_ds = Subset(base_dataset, list(map(int, train_idx)))
            generator = torch.Generator()
            generator.manual_seed(int(settings.get("seed", 20260524)) + int(fold))
            loader = DataLoader(train_ds, generator=generator, **train_loader_args)
        else:
            loader = None
        epoch_losses = []
        scaler = _new_grad_scaler(torch, use_amp)
        start_epoch = 1
        if bool(settings.get("resume_checkpoints", True)) and paths["training"].exists():
            try:
                state = torch.load(paths["training"], map_location=device)
                if state.get("fingerprint") == fingerprint and int(state.get("selected_epoch", local.epochs)) == int(selected_epoch):
                    model.load_state_dict(state["model_state"])
                    optimizer.load_state_dict(state["optimizer_state"])
                    if scheduler is not None and state.get("scheduler_state") is not None:
                        scheduler.load_state_dict(state["scheduler_state"])
                    if scaler is not None and state.get("scaler_state") is not None:
                        scaler.load_state_dict(state["scaler_state"])
                    epoch_losses = [float(v) for v in state.get("epoch_losses", [])]
                    start_epoch = int(state.get("epoch", 0)) + 1
            except Exception:
                start_epoch = 1
                epoch_losses = []
        progress_interval = max(1, int(settings.get("progress_interval_epochs", 10)))
        train_target_mean = 0.0
        train_target_scale = 1.0
        if task == "regression":
            train_targets = np.asarray(y_np, dtype=float)[np.asarray(train_idx)]
            train_target_mean = float(np.nanmean(train_targets))
            train_target_scale = float(np.nanstd(train_targets))
            if not np.isfinite(train_target_scale) or train_target_scale <= 0:
                train_target_scale = 1.0
        for epoch in range(start_epoch, selected_epoch + 1):
            epoch_losses.append(
                train_one_epoch(
                    model,
                    optimizer,
                    scaler,
                    np.asarray(train_idx),
                    fold_seed=int(fold),
                    epoch=epoch,
                    loader=loader,
                    target_mean=train_target_mean,
                    target_scale=train_target_scale,
                )
            )
            if scheduler is not None:
                scheduler.step()
            if epoch == 1 or epoch == selected_epoch or epoch % progress_interval == 0:
                print(
                    f"[{EXPERIMENT_ID}] endpoint={endpoint} fold={fold}/{n_splits} "
                    f"epoch={epoch}/{selected_epoch} loss={epoch_losses[-1]:.6g}",
                    flush=True,
                )
            if bool(settings.get("epoch_checkpoints", True)):
                tmp_path = paths["training"].with_suffix(".tmp.pt")
                torch.save(
                    {
                        "fingerprint": fingerprint,
                        "epoch": int(epoch),
                        "selected_epoch": int(selected_epoch),
                        "selected_inner_score": float(selected_inner_score),
                        "selection_metric": selection_metric,
                        "epoch_losses": epoch_losses,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                        "scaler_state": scaler.state_dict() if scaler is not None else None,
                    },
                    tmp_path,
                )
                os.replace(tmp_path, paths["training"])
        model.eval()
        fold_output = np.zeros((len(test_idx), len(classes) if task in {"binary", "multiclass"} else 1), dtype=float)
        fold_features = np.zeros((len(test_idx), selected_local.feature_dim), dtype=float)
        fold_attention_rows: list[dict[str, object]] = []
        eval_batch_size = max(1, int(settings.get("eval_batch_size", selected_local.batch_size)))
        with torch.no_grad():
            for start in range(0, len(test_idx), eval_batch_size):
                batch_positions = np.asarray(test_idx[start : start + eval_batch_size], dtype=np.int64)
                if preload_tensors:
                    assert x_tensor is not None and mask_tensor is not None
                    batch_index = torch.as_tensor(batch_positions, dtype=torch.long, device=device)
                    test_x = x_tensor[batch_index]
                    test_mask = mask_tensor[batch_index]
                else:
                    test_x = torch.as_tensor(x_np[batch_positions], dtype=torch.long, device=device)
                    test_mask = torch.as_tensor(mask_np[batch_positions], dtype=torch.bool, device=device)
                test_x, test_mask = _trim_to_observed_events(test_x, test_mask, enabled=trim_to_observed)
                with _torch_autocast(torch, device, use_amp):
                    logits, features, attention = model(test_x, test_mask, return_attention=True)
                if task in {"binary", "multiclass"}:
                    batch_prediction = torch.softmax(logits, dim=1).float().cpu().numpy()
                else:
                    batch_prediction = logits.reshape(-1, 1).float().cpu().numpy()
                batch_features = features.float().cpu().numpy()
                attn = attention.float().cpu().numpy()
                fold_output[start : start + len(batch_positions)] = batch_prediction
                fold_features[start : start + len(batch_positions)] = batch_features
                for local_i, sample_pos in enumerate(batch_positions):
                    valid = mask_np[sample_pos]
                    if attn.ndim == 4:
                        per_event = attn[local_i].mean(axis=(0, 1))[: len(valid)]
                    elif attn.ndim == 3:
                        per_event = attn[local_i].mean(axis=0)[: len(valid)]
                    else:
                        per_event = np.zeros(len(valid), dtype=float)
                    # Evaluation batches can be dynamically trimmed to the
                    # largest observed event count in the batch, while
                    # ``mask_np`` still has the configured global max length.
                    # Align the boolean mask to the attention vector length.
                    per_event = per_event[valid[: len(per_event)]]
                    if per_event.size:
                        per_event = per_event / max(float(per_event.sum()), 1e-12)
                        entropy = float(-(per_event * np.log(per_event + 1e-12)).sum())
                        max_attention = float(per_event.max())
                    else:
                        entropy = 0.0
                        max_attention = 0.0
                    fold_attention_rows.append(
                        {
                            "endpoint": endpoint,
                            "sample": str(common[sample_pos]),
                            "fold": int(fold),
                            "attention_max": max_attention,
                            "attention_entropy": entropy,
                            "n_events": int(valid.sum()),
                        }
                    )
        if task in {"binary", "multiclass"}:
            pred = np.argmax(fold_output, axis=1)
            fold_truth = np.asarray(y_np)[test_idx]
            fold_score = _endpoint_validation_score(selection_metric, task, fold_truth, fold_output, len(classes))
            fold_pred_frame = pd.DataFrame(
                {
                    "benchmark": benchmark,
                    "endpoint": endpoint,
                    "representation": LOCAL_REPRESENTATION,
                    "learner": "muat_compatible_qkv_attention",
                    "sample": common[test_idx].astype(str),
                    "true_value": fold_truth,
                    "predicted_class": pred,
                    "fold": int(fold),
                }
            )
            for i, cls in enumerate(classes):
                fold_pred_frame[f"pred_class_{i}"] = fold_output[:, i]
                fold_pred_frame[f"class_label_{i}"] = str(cls)
        elif task == "regression":
            fold_pred_z = fold_output[:, 0]
            fold_pred = fold_pred_z * float(train_target_scale) + float(train_target_mean)
            fold_truth = np.asarray(y_np, dtype=float)[test_idx]
            fold_score = _safe_spearman(fold_truth, fold_pred)
            fold_pred_frame = pd.DataFrame(
                {
                    "benchmark": benchmark,
                    "endpoint": endpoint,
                    "representation": LOCAL_REPRESENTATION,
                    "learner": "muat_compatible_qkv_attention",
                    "sample": common[test_idx].astype(str),
                    "true_value": fold_truth,
                    "pred_value": fold_pred,
                    "pred_value_z": fold_pred_z,
                    "fold": int(fold),
                }
            )
        else:
            raise ValueError(f"Unsupported MuAt-compatible task for endpoint={endpoint}: {task}")
        fold_feature_frame = pd.DataFrame(fold_features, columns=[f"muat_feature_{i + 1}" for i in range(fold_features.shape[1])])
        fold_feature_frame.insert(0, "sample", common[test_idx].astype(str))
        fold_feature_frame.insert(0, "endpoint", endpoint)
        fold_split_rows = []
        for sample_pos in train_idx:
            fold_split_rows.append({"endpoint": endpoint, "sample": str(common[sample_pos]), "fold": int(fold), "split": "train"})
        if nested_validation:
            for sample_pos in inner_train_idx:
                fold_split_rows.append({"endpoint": endpoint, "sample": str(common[sample_pos]), "fold": int(fold), "split": "inner_train"})
            for sample_pos in inner_val_idx:
                fold_split_rows.append({"endpoint": endpoint, "sample": str(common[sample_pos]), "fold": int(fold), "split": "inner_val"})
        for sample_pos in test_idx:
            fold_split_rows.append({"endpoint": endpoint, "sample": str(common[sample_pos]), "fold": int(fold), "split": "test"})
        fold_row = pd.DataFrame(
            [
                {
                    "endpoint": endpoint,
                    "fold": int(fold),
                    "outer_train_n": int(len(train_idx)),
                    "outer_test_n": int(len(test_idx)),
                    "inner_train_n": inner_train_n,
                    "inner_val_n": inner_val_n,
                    "epochs": int(selected_local.epochs),
                    "selected_m_star": int(selected_epoch),
                    "selected_inner_score": selected_inner_score,
                    "selection_metric": selection_metric,
                    "candidate_count": int(len(candidate_results)),
                    "architecture_candidate_count": int(len(architecture_candidates)),
                    "architecture_search_enabled": bool(architecture_search_enabled),
                    "architecture_search_epochs": int(architecture_search_epochs),
                    "selected_architecture_candidate_id": int(selected_architecture_candidate_id),
                    "candidate_results_json": json.dumps(candidate_results, sort_keys=True),
                    "nested_validation": bool(nested_validation),
                    "batch_size": int(effective_batch_size),
                    "configured_batch_size": int(selected_local.batch_size),
                    "eval_batch_size": int(eval_batch_size),
                    "train_loss_last": epoch_losses[-1] if epoch_losses else float("nan"),
                    "fold_score": fold_score,
                    "accuracy": fold_score if task in {"binary", "multiclass"} and selection_metric == "accuracy" else float("nan"),
                    "balanced_accuracy": fold_score if task in {"binary", "multiclass"} and selection_metric == "balanced_accuracy" else float("nan"),
                    "macro_auroc": fold_score if task in {"binary", "multiclass"} and selection_metric == "macro_auroc" else float("nan"),
                    "spearman": fold_score if task == "regression" else float("nan"),
                    "c_index": float("nan"),
                    "target_mean": train_target_mean if task == "regression" else float("nan"),
                    "target_scale": train_target_scale if task == "regression" else float("nan"),
                    "checkpoint_path": str(paths["meta"]),
                    "model_settings_json": json.dumps(asdict(selected_local), sort_keys=True),
                }
            ]
        )
        fold_attention_frame = pd.DataFrame(fold_attention_rows)
        fold_split_frame = pd.DataFrame(fold_split_rows)
        _write_fold_checkpoint(
            paths,
            fingerprint=fingerprint,
            frames={"pred": fold_pred_frame, "fold": fold_row, "attention": fold_attention_frame, "features": fold_feature_frame, "split": fold_split_frame},
            meta={"endpoint": endpoint, "fold": int(fold), "completed_epoch": int(selected_epoch), "primary_metric_scope": primary_scope},
        )
        print(f"[{EXPERIMENT_ID}] completed endpoint={endpoint} fold={fold}/{n_splits} {selection_metric}={fold_score:.6g}", flush=True)
        assign_oof_fold(sample_folds, test_idx, int(fold))
        pred_frames.append(fold_pred_frame)
        fold_rows.extend(fold_row.to_dict(orient="records"))
        attention_rows.extend(fold_attention_rows)
        feature_frames.append(fold_feature_frame)
        split_rows.extend(fold_split_rows)

    assert_all_oof_assigned(sample_folds, common)
    pred_frame = pd.concat(pred_frames, ignore_index=True, sort=False)
    pred_frame = pred_frame.drop_duplicates(["endpoint", "sample"], keep="last").set_index("sample").loc[common.astype(str)]
    pred_frame.index.name = "sample"
    pred_frame = pred_frame.reset_index()
    if "sample" not in pred_frame.columns and "index" in pred_frame.columns:
        pred_frame = pred_frame.rename(columns={"index": "sample"})
    feature_frame = pd.concat(feature_frames, ignore_index=True, sort=False).drop_duplicates(["endpoint", "sample"], keep="last")
    feature_frame = feature_frame.set_index("sample").loc[common.astype(str)]
    feature_frame.index.name = "sample"
    feature_frame = feature_frame.reset_index()
    if "sample" not in feature_frame.columns and "index" in feature_frame.columns:
        feature_frame = feature_frame.rename(columns={"index": "sample"})

    metric_override = str((settings.get("primary_metric_by_endpoint") or {}).get(endpoint) or settings.get("primary_metric") or "").strip().lower()
    accuracy = float("nan")
    balanced_accuracy = float("nan")
    top3 = float("nan")
    top5 = float("nan")
    macro_f1 = float("nan")
    macro_auroc = float("nan")
    micro_auroc = float("nan")
    cohen_kappa = float("nan")
    spearman = float("nan")
    c_index = float("nan")
    calibration: dict[str, float] = {}
    if task == "binary" and len(np.unique(y_np)) == 2:
        proba = pred_frame[[f"pred_class_{i}" for i in range(len(classes))]].to_numpy(dtype=float)
        pred = pred_frame["predicted_class"].to_numpy(dtype=int)
        primary_metric = "auroc"
        score = float(roc_auc_score(y_np, proba[:, 1]))
        micro_auroc = float(roc_auc_score(y_np, proba[:, 1]))
        accuracy = float(accuracy_score(y_np, pred))
        balanced_accuracy = float(balanced_accuracy_score(y_np, pred))
        top3 = topk_accuracy(y_np, proba, 3)
        top5 = topk_accuracy(y_np, proba, 5)
        macro_f1 = float(f1_score(y_np, pred, average="macro", zero_division=0))
        cohen_kappa = float(cohen_kappa_score(y_np, pred))
        calibration = expected_calibration_error(y_np, proba)
    elif task == "multiclass":
        proba = pred_frame[[f"pred_class_{i}" for i in range(len(classes))]].to_numpy(dtype=float)
        pred = pred_frame["predicted_class"].to_numpy(dtype=int)
        macro_auroc = safe_macro_auroc(y_np, proba, len(classes))
        micro_auroc = safe_micro_auroc(y_np, proba, len(classes))
        accuracy = float(accuracy_score(y_np, pred))
        balanced_accuracy = float(balanced_accuracy_score(y_np, pred))
        primary_metric = metric_override or ("balanced_accuracy" if str(endpoint) in TCGA_WES_TOP20_ENDPOINTS else "macro_auroc")
        score = _classification_score(primary_metric, y_np, proba, len(classes))
        top3 = topk_accuracy(y_np, proba, 3)
        top5 = topk_accuracy(y_np, proba, 5)
        macro_f1 = float(f1_score(y_np, pred, average="macro", zero_division=0))
        cohen_kappa = float(cohen_kappa_score(y_np, pred))
        calibration = expected_calibration_error(y_np, proba)
    elif task == "regression":
        primary_metric = "spearman"
        pred_values = pred_frame["pred_value"].to_numpy(dtype=float)
        spearman = _safe_spearman(np.asarray(y_np, dtype=float), pred_values)
        score = spearman
    else:
        raise ValueError(f"Unsupported MuAt-compatible task for endpoint={endpoint}: {task}")
    selected_overall = _most_common_model_settings(fold_rows, local)
    selected_architecture_counts: dict[str, int] = {}
    for row in fold_rows:
        value = str(row.get("model_settings_json", ""))
        if value:
            selected_architecture_counts[value] = selected_architecture_counts.get(value, 0) + 1
    result_row = {
        "experiment_id": EXPERIMENT_ID,
        "benchmark": benchmark,
        "endpoint": endpoint,
        "task": task,
        "representation": LOCAL_REPRESENTATION,
        "learner": "muat_compatible_qkv_attention",
        "model_family": LOCAL_REPRESENTATION,
        "metric": primary_metric,
        "score": score,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "top3_accuracy": top3,
        "top5_accuracy": top5,
        "macro_f1": macro_f1,
        "macro_auroc": macro_auroc,
        "micro_auroc": micro_auroc,
        "cohen_kappa": cohen_kappa,
        "spearman": spearman,
        "c_index": c_index,
        "n_samples": int(len(common)),
        "n_classes": int(len(classes) if task in {"binary", "multiclass"} else 1),
        "primary_metric_scope": primary_scope,
        "validation_protocol": "outer_train_inner_validation_architecture_and_epoch_selection" if nested_validation else "fixed_epoch_outer_oof",
        "official_muat": False,
        "n_folds": int(n_splits),
        "checkpoint_fingerprint": fingerprint,
        "checkpoint_dir": str(checkpoint_dir),
        "amp": bool(use_amp),
        "dataloader_workers": int(dataloader_workers),
        "trim_to_observed_events": bool(trim_to_observed),
        "preload_tensors_to_device": bool(preload_tensors),
        "canonical_split_source": canonical_split_source,
        "canonical_split_signature": canonical_split_signature,
        "dictionary_mode": str(settings.get("dictionary_mode", "observed")),
        "dictionary_sizes_json": json.dumps(dictionaries.sizes(), sort_keys=True),
        "architecture_search_enabled": bool(architecture_search_enabled),
        "architecture_candidate_count": int(len(architecture_candidates)),
        "architecture_search_epochs": int(architecture_search_epochs),
        "architecture_candidates_json": json.dumps([asdict(candidate) for candidate in architecture_candidates], sort_keys=True),
        "selected_architecture_counts_json": json.dumps(selected_architecture_counts, sort_keys=True),
        **{f"calibration_{k}": v for k, v in calibration.items()},
        **asdict(selected_overall),
    }
    if task == "multiclass" and str(endpoint) in TCGA_WES_TOP20_ENDPOINTS:
        result_row.update(
            {
                "paper_reference": "MuAt TCGA WES 20 tumour types",
                "paper_reference_n_samples": MUAT_PAPER_TCGA_WES_N_SAMPLES,
                "paper_reference_n_classes": MUAT_PAPER_TCGA_WES_N_CLASSES,
                "paper_reference_accuracy": MUAT_PAPER_TCGA_WES_ACCURACY,
                "paper_reference_top5_accuracy": MUAT_PAPER_TCGA_WES_TOP5_ACCURACY,
                "accuracy_minus_paper_reference": accuracy - MUAT_PAPER_TCGA_WES_ACCURACY,
                "top5_accuracy_minus_paper_reference": top5 - MUAT_PAPER_TCGA_WES_TOP5_ACCURACY,
                "paper_reference_comparability": (
                    "same TCGA-WES 20-type task family; not an official MuAt checkpoint "
                    "replication unless official_muat is true"
                ),
            }
        )
    return (
        {
            **result_row,
        },
        pred_frame,
        pd.DataFrame(fold_rows),
        pd.DataFrame(attention_rows),
        feature_frame.assign(endpoint=endpoint),
        pd.DataFrame(split_rows),
    )


def run(ctx: RunnerContext) -> None:
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
    settings = dict(((ctx.settings.get("experiments") or {}).get(EXPERIMENT_ID) or {}))
    registry = load_endpoint_registry(settings.get("endpoint_registry") or ctx.settings.get("endpoint_registry"))
    primary_endpoint = str(settings.get("primary_endpoint") or settings.get("primary_task") or PRIMARY_ENDPOINT)
    secondary_endpoints = list(settings.get("secondary_endpoints") or ([] if not settings.get("run_manuscript_endpoints", False) else muat_comparator_endpoints(registry)))
    secondary_endpoints = [endpoint for endpoint in secondary_endpoints if str(endpoint) != primary_endpoint]
    requested_endpoints = [primary_endpoint, *secondary_endpoints]
    kucab_endpoints = [endpoint for endpoint in requested_endpoints if str(endpoint) == "damage_class"]
    tcga_secondary_endpoints = [endpoint for endpoint in secondary_endpoints if str(endpoint) != "damage_class"]
    output_tag = str(settings.get("output_tag") or "").strip()
    output_prefix = f"{EXPERIMENT_ID}_{_safe_name(output_tag)}" if output_tag else EXPERIMENT_ID
    summary_path = ctx.tables_dir / f"{output_prefix}_summary.csv"
    results_path = ctx.tables_dir / f"{output_prefix}_endpoint_results.csv"
    predictions_path = ctx.tables_dir / f"{output_prefix}_oof_predictions.csv"
    folds_path = ctx.tables_dir / f"{output_prefix}_fold_metrics.csv"
    attention_path = ctx.tables_dir / f"{output_prefix}_attention_summary.csv"
    calibration_path = ctx.tables_dir / f"{output_prefix}_calibration_summary.csv"
    event_table_path = ctx.tables_dir / f"{output_prefix}_muat_compatible_events.tsv.gz"
    dictionary_path = ctx.tables_dir / f"{output_prefix}_dictionary_summary.csv"
    task_audit_path = ctx.tables_dir / f"{output_prefix}_task_audit.csv"
    split_path = ctx.tables_dir / f"{output_prefix}_split_manifest.csv"
    feature_path = ctx.tables_dir / f"{output_prefix}_tumour_level_features.csv"
    fidelity_path = ctx.tables_dir / f"{output_prefix}_fidelity_report.md"
    cli_audit_path = ctx.logs_dir / f"{output_prefix}_official_cli_audit.json"
    manifest_path = ctx.logs_dir / f"{output_prefix}_manifest.json"

    if ctx.dry_run:
        write_summary_csv(
            [
                {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "planned",
                    "primary_endpoint": primary_endpoint,
                    "secondary_endpoints": ";".join(secondary_endpoints),
                    "smoke_run": bool(settings.get("smoke_run", True)),
                }
            ],
            ctx.logs_dir / f"dry_run_{EXPERIMENT_ID}_summary.csv",
        )
        return

    try:
        import torch
    except ImportError:
        write_summary_csv([{"experiment_id": EXPERIMENT_ID, "status": "skipped_missing_torch"}], summary_path)
        return

    t0 = time.time()
    seed = int(settings.get("seed", 20260524))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch_threads = int(settings.get("torch_num_threads", min(8, os.cpu_count() or 1)))
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    device_setting = str(settings.get("device", "auto")).lower()
    device = torch.device("cuda" if device_setting == "auto" and torch.cuda.is_available() else ("cpu" if device_setting == "auto" else device_setting))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    mc3_dir = Path(ctx.paths["raw_data"]["mc3_source_dir"])
    maf_path = mc3_dir / "raw" / "mc3.v0.2.8.PUBLIC.maf.gz"
    cli_audit = _official_cli_audit(ctx, settings, maf_path)
    atomic_write_json(cli_audit_path, cli_audit)

    maf_patients = collect_maf_patients(maf_path, chunksize=int(settings.get("maf_chunksize", 250_000)))
    tcga20_labels, tcga20_task_audit = load_tcga_20_type_labels(
        mc3_dir,
        maf_patients=maf_patients,
        min_count=int(settings.get("tcga_min_type_n", 100)),
        max_types=int(settings.get("tcga_max_types", 20)),
    )
    tcga20_task = "multiclass"
    if bool(settings.get("smoke_run", True)):
        tcga20_labels = sample_smoke_labels(tcga20_labels, per_class=int(settings.get("smoke_samples_per_type", 3)), seed=seed)

    labels_by_endpoint: dict[str, tuple[str, pd.Series | pd.DataFrame]] = {}
    if primary_endpoint == "damage_class":
        primary_labels = pd.Series(dtype=str, name="damage_class")
        task_audit = pd.DataFrame(columns=["endpoint", "tumour_type", "n_patients", "included", "reason"])
    elif primary_endpoint == PRIMARY_ENDPOINT:
        labels_by_endpoint[primary_endpoint] = (tcga20_task, tcga20_labels)
        primary_labels = tcga20_labels
        task_audit = tcga20_task_audit.copy()
    else:
        loaded_primary = _load_secondary_endpoint_labels(ctx, [primary_endpoint])
        if primary_endpoint not in loaded_primary:
            raise ValueError(f"Unsupported MuAt-compatible primary endpoint: {primary_endpoint}")
        primary_task, primary_labels = loaded_primary[primary_endpoint]
        if bool(settings.get("smoke_run", True)) and primary_task == "multiclass":
            primary_labels = sample_smoke_labels(primary_labels, per_class=int(settings.get("smoke_samples_per_type", 3)), seed=seed)
        labels_by_endpoint[primary_endpoint] = (primary_task, primary_labels)
        task_audit = _endpoint_audit(primary_endpoint, primary_labels)

    result_rows = []
    pred_frames = []
    fold_frames = []
    attention_frames = []
    feature_frames = []
    split_frames = []
    task_audit_frames = [task_audit]

    labels_by_endpoint.update(_load_secondary_endpoint_labels(ctx, tcga_secondary_endpoints))
    patients = sorted({patient for _task, endpoint_labels in labels_by_endpoint.values() for patient in endpoint_labels.index.astype(str)})

    cache_dir = ctx.feature_cache_dir / EXPERIMENT_ID
    cache_dir.mkdir(parents=True, exist_ok=True)
    dictionary_mode = str(settings.get("dictionary_mode", "observed")).strip().lower()
    motif_buckets = int(settings.get("motif_hash_buckets", 1024))
    position_buckets = int(settings.get("position_hash_buckets", 2048))
    annotation_buckets = int(settings.get("annotation_hash_buckets", 256))
    cache_key = ""
    cache_hit = False
    events_cache_path: Path | None = None
    events = pd.DataFrame(columns=["sample", "chromosome", "position", "motif", "position_token", "annotation"])
    event_counts = np.asarray([], dtype=np.int32)
    manifest_total_events = 0
    manifest_event_counts: list[np.ndarray] = []
    manifest_dictionary_sizes: dict[str, int] = {}

    if labels_by_endpoint:
        cache_key = _cache_key(maf_path, patients, settings)
        events_cache_path = cache_dir / f"muat_compatible_events_{cache_key}.csv.gz"
        if events_cache_path.exists() and not ctx.refresh_cache:
            events = pd.read_csv(events_cache_path, sep="\t")
            cache_hit = True
        else:
            events = build_muat_event_table(maf_path, patients, chunksize=int(settings.get("maf_chunksize", 250_000)))
            events.to_csv(events_cache_path, sep="\t", index=False, compression="gzip")
            cache_hit = False
        events.to_csv(event_table_path, sep="\t", index=False, compression="gzip")
        if dictionary_mode in {"fixed_hash", "hash", "hashed"}:
            events_for_encoding = hash_muat_event_tokens(
                events,
                motif_buckets=motif_buckets,
                position_buckets=position_buckets,
                annotation_buckets=annotation_buckets,
            )
            dictionaries = build_fixed_hash_dictionaries(
                motif_buckets=motif_buckets,
                position_buckets=position_buckets,
                annotation_buckets=annotation_buckets,
            )
        else:
            events_for_encoding = events
            dictionaries = build_dictionaries(events)
        atomic_write_csv(dictionary_summary(dictionaries).assign(benchmark="tcga_mc3_wes"), dictionary_path, index=False)
        manifest_dictionary_sizes = dictionaries.sizes()
        if not primary_labels.empty:
            sample_event_counts = events.groupby("sample").size().rename("n_events")
            label_event_summary = (
                pd.DataFrame({"sample": primary_labels.index.astype(str), "tumour_type": primary_labels.astype(str).to_numpy()})
                .join(sample_event_counts, on="sample")
                .groupby("tumour_type", as_index=False)
                .agg(n_run_patients=("sample", "nunique"), n_run_events=("n_events", "sum"))
            )
            task_audit = task_audit.merge(label_event_summary, how="left", on="tumour_type")
            task_audit["n_run_patients"] = task_audit["n_run_patients"].fillna(0).astype(int)
            task_audit["n_run_events"] = task_audit["n_run_events"].fillna(0).astype(int)
            task_audit_frames[0] = task_audit

        x, mask, event_counts = encode_events(events_for_encoding, patients, dictionaries, max_events=int(settings.get("max_events_per_sample", 128)))
        manifest_total_events += int(len(events))
        manifest_event_counts.append(event_counts)
        for endpoint, (endpoint_task, endpoint_labels) in labels_by_endpoint.items():
            row, pred, folds, attention, features, splits = _fit_predict_local(
                endpoint=endpoint,
                task=endpoint_task,
                benchmark="tcga_mc3_wes",
                labels=endpoint_labels,
                patients=patients,
                x=x,
                mask=mask,
                dictionaries=dictionaries,
                settings=settings,
                ctx=ctx,
                device=device,
            )
            result_rows.append(row)
            pred_frames.append(pred)
            fold_frames.append(folds)
            attention_frames.append(attention)
            feature_frames.append(features)
            split_frames.append(splits)

    if kucab_endpoints:
        kucab_labels, kucab_groups, kucab_events, kucab_audit = _load_kucab_muat_endpoint(ctx, settings)
        kucab_events.to_csv(ctx.tables_dir / f"{output_prefix}_kucab_muat_compatible_events.tsv.gz", sep="\t", index=False, compression="gzip")
        if dictionary_mode in {"fixed_hash", "hash", "hashed"}:
            kucab_events_for_encoding = hash_muat_event_tokens(
                kucab_events,
                motif_buckets=motif_buckets,
                position_buckets=position_buckets,
                annotation_buckets=annotation_buckets,
            )
            kucab_dictionaries = build_fixed_hash_dictionaries(
                motif_buckets=motif_buckets,
                position_buckets=position_buckets,
                annotation_buckets=annotation_buckets,
            )
        else:
            kucab_events_for_encoding = kucab_events
            kucab_dictionaries = build_dictionaries(kucab_events)
        dict_summary = dictionary_summary(kucab_dictionaries).assign(benchmark="kucab_wgs")
        manifest_dictionary_sizes = kucab_dictionaries.sizes()
        if dictionary_path.exists():
            prior_dict = pd.read_csv(dictionary_path)
            dict_summary = pd.concat([prior_dict, dict_summary], ignore_index=True, sort=False)
        atomic_write_csv(dict_summary, dictionary_path, index=False)
        kucab_patients = kucab_labels.index.astype(str).tolist()
        kucab_x, kucab_mask, kucab_event_counts = encode_events(kucab_events_for_encoding, kucab_patients, kucab_dictionaries, max_events=int(settings.get("kucab_max_events_per_sample", settings.get("max_events_per_sample", 5000))))
        manifest_total_events += int(len(kucab_events))
        manifest_event_counts.append(kucab_event_counts)
        row, pred, folds, attention, features, splits = _fit_predict_local(
            endpoint="damage_class",
            task="multiclass",
            benchmark="kucab_wgs",
            labels=kucab_labels,
            groups=kucab_groups,
            patients=kucab_patients,
            x=kucab_x,
            mask=kucab_mask,
            dictionaries=kucab_dictionaries,
            settings=settings,
            ctx=ctx,
            device=device,
        )
        result_rows.append(row)
        pred_frames.append(pred)
        fold_frames.append(folds)
        attention_frames.append(attention)
        feature_frames.append(features)
        split_frames.append(splits)
        task_audit_frames.append(kucab_audit)

    if task_audit_frames:
        atomic_write_csv(sanitize_frame(pd.concat(task_audit_frames, ignore_index=True, sort=False)), task_audit_path, index=False)
    manifest_counts = np.concatenate(manifest_event_counts) if manifest_event_counts else np.asarray([], dtype=np.int32)

    results = pd.DataFrame(result_rows)
    expected_endpoints = set(map(str, requested_endpoints))
    observed_endpoints = set(results.get("endpoint", pd.Series(dtype=str)).astype(str))
    missing_expected = sorted(expected_endpoints - observed_endpoints)
    if missing_expected:
        raise RuntimeError(f"MuAt-compatible run did not produce requested endpoint rows: {missing_expected}")
    if "cancer_type_top20" in expected_endpoints and not bool(settings.get("smoke_run", True)):
        top20_rows = results[results.get("endpoint", pd.Series(dtype=str)).astype(str).eq("cancer_type_top20")]
        if top20_rows.empty:
            raise RuntimeError("MuAt-compatible top-20 run is missing cancer_type_top20 results")
        n_samples = int(pd.to_numeric(top20_rows.iloc[0].get("n_samples"), errors="coerce"))
        n_classes = int(pd.to_numeric(top20_rows.iloc[0].get("n_classes"), errors="coerce"))
        if n_samples != 8800 or n_classes != 20:
            raise RuntimeError(f"MuAt-compatible cancer_type_top20 expected n=8800/classes=20, got n={n_samples}/classes={n_classes}")
        metrics = set(top20_rows.get("metric", pd.Series(dtype=str)).astype(str))
        if metrics != {"balanced_accuracy"}:
            raise RuntimeError(f"MuAt-compatible cancer_type_top20 must use balanced_accuracy, got metrics={sorted(metrics)}")
    atomic_write_csv(sanitize_frame(results), results_path, index=False)
    _write_main_panel_model_comparison(ctx, output_prefix, results)
    if pred_frames:
        atomic_write_csv(sanitize_frame(pd.concat(pred_frames, ignore_index=True, sort=False)), predictions_path, index=False)
    if fold_frames:
        atomic_write_csv(sanitize_frame(pd.concat(fold_frames, ignore_index=True, sort=False)), folds_path, index=False)
    if attention_frames:
        atomic_write_csv(sanitize_frame(pd.concat(attention_frames, ignore_index=True, sort=False)), attention_path, index=False)
    if feature_frames:
        atomic_write_csv(sanitize_frame(pd.concat(feature_frames, ignore_index=True, sort=False)), feature_path, index=False)
    if split_frames:
        atomic_write_csv(sanitize_frame(pd.concat(split_frames, ignore_index=True, sort=False)), split_path, index=False)
    calibration_cols = [column for column in results.columns if column.startswith("calibration_")]
    if calibration_cols:
        atomic_write_csv(results[["endpoint", *calibration_cols]], calibration_path, index=False)

    _write_fidelity_report(
        fidelity_path,
        official_available=bool(cli_audit.get("status") == "available"),
        smoke_run=bool(settings.get("smoke_run", True)),
        n_types=int(primary_labels.nunique()) if not primary_labels.empty else int(results.get("n_classes", pd.Series(dtype=float)).max() or 0),
        n_samples=int(len(primary_labels)) if not primary_labels.empty else int(results.get("n_samples", pd.Series(dtype=float)).sum() or 0),
    )
    manifest = {
        "elapsed_seconds": round(time.time() - t0, 3),
        "primary_endpoint": primary_endpoint,
        "secondary_endpoints": secondary_endpoints,
        "smoke_run": bool(settings.get("smoke_run", True)),
        "representation": LOCAL_REPRESENTATION,
        "official_cli_status": cli_audit.get("status"),
        "torch_device": str(device),
        "torch_num_threads": int(torch.get_num_threads()),
        "n_patients": int(results.get("n_samples", pd.Series(dtype=float)).max() or len(patients)),
        "n_events": int(manifest_total_events),
        "event_cache_key": cache_key,
        "event_cache_path": sanitize_text(str(events_cache_path)) if events_cache_path is not None else "",
        "event_cache_hit": cache_hit,
        "max_events_per_sample": int(settings.get("max_events_per_sample", 128)),
        "encoded_event_count_min": int(manifest_counts.min()) if len(manifest_counts) else 0,
        "encoded_event_count_median": float(np.median(manifest_counts)) if len(manifest_counts) else 0.0,
        "encoded_event_count_max": int(manifest_counts.max()) if len(manifest_counts) else 0,
        "dictionary_sizes": manifest_dictionary_sizes,
        "dictionary_mode": dictionary_mode,
        "count_feature_mode": str(settings.get("count_feature_mode", "none")),
        "pooling_mode": str(settings.get("pooling_mode", "attention_weighted")),
        "architecture_search_enabled": bool(settings.get("architecture_search_enabled", False)),
        "architecture_search_embed_dims": settings.get("architecture_search_embed_dims"),
        "architecture_search_num_layers": settings.get("architecture_search_num_layers"),
        "architecture_search_attention_heads": settings.get("architecture_search_attention_heads"),
        "architecture_search_epochs": settings.get("architecture_search_epochs"),
    }
    atomic_write_json(manifest_path, manifest)
    write_summary_csv(
        [
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "completed",
                "elapsed_seconds": round(time.time() - t0, 3),
                "primary_endpoint": primary_endpoint,
                "secondary_endpoints": ";".join(secondary_endpoints),
                "smoke_run": bool(settings.get("smoke_run", True)),
                "official_cli_status": cli_audit.get("status"),
                "n_patients": len(patients),
                "n_events": int(len(events)),
            }
        ],
        summary_path,
    )
