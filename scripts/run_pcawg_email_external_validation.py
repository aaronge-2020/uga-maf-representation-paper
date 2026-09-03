#!/usr/bin/env python3
"""Run Aaron's 913-sample PCAWG ICGC-only frozen-model validation.

The DIG coordinates are zero-based half-open. SigProfilerMatrixGenerator expects
one-based coordinates, so both START and END are shifted by one during conversion.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = Path(
    os.environ.get(
        "PCAWG_RAW_DIR",
        str(REPO_ROOT / "data" / "external_validation" / "pcawg_icgc_only"),
    )
)
DEFAULT_OUT = REPO_ROOT / "results" / "external_validation" / "aaron_email" / "pcawg"
DEFAULT_MODEL = REPO_ROOT / "results" / "frozen_models" / "cancer_type_top20__standard_sbs96_id83"
SEED = 20260822

SOURCE_MAP = {
    "Breast-AdenoCa": ("BRCA", 110),
    "Breast-LobularCa": ("BRCA", 7),
    "Head-SCC": ("HNSC", 13),
    "Prost-AdenoCA": ("PRAD", 180),
    "Skin-Melanoma": ("SKCM", 70),
    "Stomach-AdenoCA": ("STAD", 32),
    "Ovary-AdenoCA": ("OV", 69),
    "Kidney-RCC": ("KIRC", 74),
    "Liver-HCC": ("LIHC", 261),
    "Eso-AdenoCa": ("ESCA", 97),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return path.name


def sigprofiler_skip_audit(output_dir: Path) -> dict[str, object]:
    logs = sorted(
        (output_dir / "sigprofiler_output" / "logs").glob(
            "SigProfilerMatrixGenerator_PCAWG_ICGC_only_913_GRCh37*.out"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not logs:
        return {
            "sigprofiler_log": None,
            "sigprofiler_skipped_mutations": None,
            "sigprofiler_version_from_log": None,
        }
    latest_log = logs[-1]
    log_text = latest_log.read_text(encoding="utf-8", errors="replace")
    skipped = log_text.count("Skipping this mutation")
    version_match = re.search(r"SigProfilerMatrixGenerator Version:\s*(\S+)", log_text)
    return {
        "sigprofiler_log": portable_path(latest_log),
        "sigprofiler_skipped_mutations": skipped,
        "sigprofiler_version_from_log": version_match.group(1) if version_match else None,
    }


def load_cached_inputs(output_dir: Path, model_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping_path = output_dir / "pcawg_913_sample_mapping.csv"
    features_path = output_dir / "pcawg_standard_sbs96_id83_features.csv.gz"
    if not mapping_path.exists() or not features_path.exists():
        raise RuntimeError(
            "--reuse-features requires pcawg_913_sample_mapping.csv and "
            "pcawg_standard_sbs96_id83_features.csv.gz in the output directory"
        )
    mapping = pd.read_csv(mapping_path, dtype={"sample_id": str})
    features = pd.read_csv(features_path, index_col=0)
    features.index = features.index.astype(str)
    feature_order = [
        line.strip()
        for line in (model_dir / "feature_order.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(mapping) != 913 or mapping["sample_id"].nunique() != 913:
        raise RuntimeError(f"Invalid cached PCAWG mapping: shape={mapping.shape}")
    if len(features) != 913 or features.index.nunique() != 913:
        raise RuntimeError(f"Invalid cached PCAWG features: shape={features.shape}")
    if set(mapping["sample_id"]) != set(features.index):
        raise RuntimeError("Cached PCAWG mapping and feature sample IDs do not match")
    missing = [column for column in feature_order if column not in features]
    extra = [column for column in features if column not in feature_order]
    if missing or extra:
        raise RuntimeError(f"Cached feature schema mismatch: missing={missing[:8]}, extra={extra[:8]}")
    features = features.reindex(columns=feature_order).astype(np.float32)
    if not np.isfinite(features.to_numpy()).all():
        raise RuntimeError("Cached PCAWG feature matrix contains non-finite values")
    return mapping, features


def prepare_inputs(raw_dir: Path, output_dir: Path) -> pd.DataFrame:
    simple_dir = output_dir / "sigprofiler_input"
    simple_dir.mkdir(parents=True, exist_ok=True)
    mapping_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    all_samples: set[str] = set()
    header = "project\tsample\tunused1\tunused2\tunused3\tchrom\tstart\tend\tref\talt\n"

    for source_type, (true_label, expected_n) in SOURCE_MAP.items():
        source = raw_dir / f"{source_type}_SNV_MNV_INDEL.ICGC.annot.txt.gz"
        destination = simple_dir / f"{source_type}.txt"
        samples: set[str] = set()
        rows = 0
        with gzip.open(source, "rt", newline="") as source_handle, destination.open("w", encoding="utf-8") as output_handle:
            output_handle.write(header)
            reader = csv.reader(source_handle, delimiter="\t")
            for line_number, row in enumerate(reader, start=1):
                if len(row) != 10:
                    raise ValueError(f"{source.name}:{line_number} has {len(row)} fields; expected 10")
                chrom, start, end, ref, alt, sample = row[:6]
                samples.add(sample)
                rows += 1
                output_handle.write(
                    f"PCAWG\t{sample}\t.\t.\t.\t{chrom}\t{int(start) + 1}\t{int(end) + 1}\t{ref}\t{alt}\n"
                )
        if len(samples) != expected_n:
            raise RuntimeError(f"{source_type}: expected {expected_n} samples; found {len(samples)}")
        overlap = all_samples.intersection(samples)
        if overlap:
            raise RuntimeError(f"Sample IDs repeated across leaf files: {sorted(overlap)[:5]}")
        all_samples.update(samples)
        mapping_rows.extend(
            {
                "sample_id": sample,
                "pcawg_cancer_type": source_type,
                "true_label": true_label,
                "source_file": source.name,
                "included": True,
                "exclusion_reason": "",
            }
            for sample in sorted(samples)
        )
        manifest_rows.append(
            {
                "source_file": source.name,
                "source_sha256": sha256(source),
                "source_bytes": source.stat().st_size,
                "converted_file": destination.name,
                "converted_sha256": sha256(destination),
                "mutation_rows": rows,
                "samples": len(samples),
                "coordinate_conversion": "DIG zero-based START/END -> one-based by adding 1 to both",
            }
        )

    if len(all_samples) != 913:
        raise RuntimeError(f"Expected 913 unique PCAWG samples; found {len(all_samples)}")
    mapping = pd.DataFrame(mapping_rows).sort_values(["true_label", "pcawg_cancer_type", "sample_id"])
    mapping.to_csv(output_dir / "pcawg_913_sample_mapping.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(output_dir / "input_manifest.csv", index=False)
    return mapping


def generate_features(output_dir: Path, model_dir: Path) -> pd.DataFrame:
    from SigProfilerMatrixGenerator.scripts.SigProfilerMatrixGeneratorFunc import SigProfilerMatrixGeneratorFunc

    matrices = SigProfilerMatrixGeneratorFunc(
        "PCAWG_ICGC_only_913",
        "GRCh37",
        str(output_dir / "sigprofiler_input"),
        plot=False,
        seqInfo=False,
        tsb_stat=False,
        output_directory=str(output_dir / "sigprofiler_output"),
    )
    sbs = matrices["96"].astype(float)
    ids = matrices["ID"].astype(float)
    sample_ids = sorted(set(sbs.columns).union(ids.columns))
    sbs = sbs.reindex(columns=sample_ids, fill_value=0.0)
    ids = ids.reindex(columns=sample_ids, fill_value=0.0)
    sbs_burden = sbs.sum(axis=0)
    id_burden = ids.sum(axis=0)
    total = sbs_burden + id_burden

    features = pd.DataFrame(index=pd.Index(sample_ids, name="sample_id"))
    features["log10_sbs_burden"] = np.log10(1.0 + sbs_burden)
    features["log10_id_burden"] = np.log10(1.0 + id_burden)
    features["id_fraction"] = np.divide(id_burden, total, out=np.zeros(len(total)), where=total > 0)
    for channel in sbs.index:
        features[f"SBS96__{channel}"] = np.divide(
            sbs.loc[channel], sbs_burden, out=np.zeros(len(sbs_burden)), where=sbs_burden > 0
        )
    for channel in ids.index:
        features[f"ID83__{channel}"] = np.divide(
            ids.loc[channel], id_burden, out=np.zeros(len(id_burden)), where=id_burden > 0
        )

    feature_order = [line.strip() for line in (model_dir / "feature_order.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [column for column in feature_order if column not in features]
    extra = [column for column in features if column not in feature_order]
    if missing or extra:
        raise RuntimeError(f"Feature schema mismatch: missing={missing[:8]}, extra={extra[:8]}")
    features = features.reindex(columns=feature_order).astype(np.float32)
    if len(features) != 913 or not np.isfinite(features.to_numpy()).all():
        raise RuntimeError(f"Invalid PCAWG feature matrix: shape={features.shape}")
    features.to_csv(output_dir / "pcawg_standard_sbs96_id83_features.csv.gz", compression="gzip")
    sbs.to_csv(output_dir / "pcawg_sigprofiler_sbs96_counts.csv.gz", compression="gzip")
    ids.to_csv(output_dir / "pcawg_sigprofiler_id83_counts.csv.gz", compression="gzip")
    return features


def predict(features: pd.DataFrame, mapping: pd.DataFrame, model_dir: Path, output_dir: Path) -> pd.DataFrame:
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    model_hash = sha256(model_dir / "model.ubj")
    if model_hash != manifest["provenance"]["model_sha256"]:
        raise RuntimeError("Frozen model SHA-256 does not match its manifest")
    booster = xgb.Booster()
    booster.load_model(model_dir / "model.ubj")
    probabilities = booster.predict(xgb.DMatrix(features.to_numpy(dtype=np.float32)))
    classes = manifest["class_order"]
    if probabilities.shape != (len(features), len(classes)):
        raise RuntimeError(f"Unexpected probability shape: {probabilities.shape}")
    predictions = mapping.set_index("sample_id").loc[features.index].copy()
    for index, label in enumerate(classes):
        predictions[f"prob_{label}"] = probabilities[:, index]
    predictions["pred_label"] = [classes[index] for index in np.argmax(probabilities, axis=1)]
    predictions["model_sha256"] = model_hash
    predictions.reset_index().to_csv(output_dir / "pcawg_frozen_model_predictions.csv", index=False)
    return predictions.reset_index()


def bootstrap_metrics(frame: pd.DataFrame, classes: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    probability_columns = [f"prob_{label}" for label in classes]
    mapped_labels = sorted(frame["true_label"].unique())

    def top3(data: pd.DataFrame) -> float:
        probabilities = data[probability_columns].to_numpy(float)
        top = np.argsort(-probabilities, axis=1)[:, :3]
        truth = data["true_label"].to_numpy(str)
        return float(np.mean([truth[i] in {classes[j] for j in top[i]} for i in range(len(data))]))

    def mapped_balanced_accuracy(data: pd.DataFrame) -> float:
        recalls = [
            data.loc[data["true_label"].eq(label), "pred_label"].eq(label).mean()
            for label in mapped_labels
        ]
        return float(np.mean(recalls))

    functions = {
        "overall_accuracy": lambda data: accuracy_score(data["true_label"], data["pred_label"]),
        "balanced_accuracy": mapped_balanced_accuracy,
        "macro_f1_mapped9": lambda data: f1_score(
            data["true_label"], data["pred_label"], labels=mapped_labels, average="macro", zero_division=0
        ),
        "macro_f1_union_sensitivity": lambda data: f1_score(
            data["true_label"], data["pred_label"], average="macro", zero_division=0
        ),
        "top3_accuracy": top3,
    }
    rng = np.random.default_rng(seed)
    strata = [group.index.to_numpy() for _, group in frame.groupby("true_label", sort=True)]
    rows: list[dict[str, object]] = []
    for name, function in functions.items():
        values = []
        for _ in range(n_boot):
            sampled = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in strata])
            values.append(float(function(frame.loc[sampled])))
        rows.append(
            {
                "metric": name,
                "class_label": "",
                "n": len(frame),
                "estimate": float(function(frame)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
            }
        )
    for label, group in frame.groupby("true_label", sort=True):
        observed = group["pred_label"].eq(label).to_numpy(float)
        values = [float(rng.choice(observed, len(observed), replace=True).mean()) for _ in range(n_boot)]
        rows.append(
            {
                "metric": "per_cancer_recall",
                "class_label": label,
                "n": len(group),
                "estimate": float(observed.mean()),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def evaluate(predictions: pd.DataFrame, model_dir: Path, output_dir: Path, n_boot: int) -> None:
    model_manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    classes = model_manifest["class_order"]
    metrics = bootstrap_metrics(predictions, classes, n_boot, SEED)
    metrics.to_csv(output_dir / "pcawg_metrics_with_95ci.csv", index=False)
    matrix = confusion_matrix(predictions["true_label"], predictions["pred_label"], labels=classes)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(output_dir / "pcawg_confusion_matrix_20class.csv")

    versions = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "git_commit": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "model_file": portable_path(model_dir / "model.ubj"),
        "model_sha256": sha256(model_dir / "model.ubj"),
        "class_order": classes,
        "usable_samples": len(predictions),
        "excluded_sample_rows": 913 - len(predictions),
        "preprocessing": [
            "Only the ten prespecified individual ICGC-only leaf files mapping to nine TCGA labels were used",
            "Breast-DCIS and Pancan/combined files were excluded a priori and never ingested",
            "DIG zero-based half-open coordinates converted to one-based coordinates by adding 1 to START and END",
            "SBS96 and ID83 channels normalized within sample; burdens transformed as log10(1+n)",
            "features reordered exactly as feature_order.txt; no scaling",
            "frozen 20-class TCGA model applied once without fitting or tuning on PCAWG/ICGC",
        ],
        "model_caveat": "Vijay's freeze script selected hyperparameters by TCGA-only CV and refit this new artifact on all eligible TCGA samples.",
    }
    try:
        installed_sigprofiler = importlib.metadata.version("SigProfilerMatrixGenerator")
    except importlib.metadata.PackageNotFoundError:
        installed_sigprofiler = "not installed"
    sigprofiler_audit = sigprofiler_skip_audit(output_dir)
    versions["SigProfilerMatrixGenerator"] = (
        sigprofiler_audit["sigprofiler_version_from_log"] or installed_sigprofiler
    )
    versions["SigProfilerMatrixGenerator_current_environment"] = installed_sigprofiler
    versions.update(sigprofiler_audit)
    (output_dir / "software_versions.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")

    exclusions = pd.DataFrame(
        [
            {
                "excluded_source": "Breast-DCIS",
                "sample_count": "not assessed; file not ingested",
                "reason": "excluded a priori from the primary analysis per Aaron's email",
            },
            {
                "excluded_source": "Pancan and other combined files",
                "sample_count": "not assessed; files not ingested",
                "reason": "excluded a priori because combined files repeat patients from individual cancer files",
            },
        ]
    )
    exclusions.to_csv(output_dir / "pcawg_exclusions_by_design.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW,
        help="PCAWG raw-data directory (default: $PCAWG_RAW_DIR or data/external_validation/pcawg_icgc_only)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="reuse and validate the saved 913-sample mapping and 182-column feature matrix",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_features:
        mapping, features = load_cached_inputs(args.output_dir, args.model_dir)
    else:
        mapping = prepare_inputs(args.raw_dir, args.output_dir)
        features = generate_features(args.output_dir, args.model_dir)
    predictions = predict(features, mapping, args.model_dir, args.output_dir)
    evaluate(predictions, args.model_dir, args.output_dir, args.bootstrap)


if __name__ == "__main__":
    main()
