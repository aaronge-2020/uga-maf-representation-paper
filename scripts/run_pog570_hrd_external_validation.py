#!/usr/bin/env python3
"""Run frozen-model HRD validation on the independent POG570 cohort.

POG570 MAF coordinates are standard one-based MAF coordinates. The script builds
the same SBS96/ID83 and Bio-MAF v4 feature blocks used to fit the frozen TCGA
model, predicts once, and evaluates the prespecified HRD cutoffs 24, 33, and 42.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.bio_maf_base_features import first_12  # noqa: E402
from utils.quick_bio_v4_nested_feature_selection import build_bio_v4_features  # noqa: E402


DEFAULT_RAW = Path(
    os.environ.get(
        "POG570_RAW_DIR",
        str(REPO_ROOT / "data" / "external_validation" / "pog570"),
    )
)
DEFAULT_OUT = REPO_ROOT / "results" / "external_validation" / "aaron_email" / "pog570"
DEFAULT_MODEL = REPO_ROOT / "results" / "frozen_models" / "HRD_Score__signatures_plus_MAF_stack"
SEED = 20260822
CUTOFFS = (24, 33, 42)


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
            "SigProfilerMatrixGenerator_POG570_GRCh37*.out"
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


def load_clinical(raw_dir: Path) -> pd.DataFrame:
    clinical = pd.read_csv(raw_dir / "data_clinical_sample.txt", sep="\t", comment="#", dtype=str)
    clinical["HRD_SCORE"] = pd.to_numeric(clinical["HRD_SCORE"], errors="coerce")
    clinical["model_sample_id"] = clinical["SAMPLE_ID"].astype(str) + "_T"
    clinical["analysis_group"] = np.where(clinical["CANCER_TYPE"].eq("Breast Cancer"), "primary_breast", "exploratory_other")
    if len(clinical) != 570 or clinical["SAMPLE_ID"].nunique() != 570:
        raise RuntimeError(f"Expected 570 unique clinical samples; found {len(clinical)} rows")
    if clinical["HRD_SCORE"].isna().any():
        raise RuntimeError("POG570 contains missing/non-numeric HRD scores")
    if clinical["analysis_group"].eq("primary_breast").sum() != 147:
        raise RuntimeError("Expected 147 primary-analysis breast cancers")
    return clinical


def prepare_sigprofiler_input(raw_dir: Path, output_dir: Path) -> tuple[Path, set[str]]:
    source = raw_dir / "POG570_mutations.maf.gz"
    simple_dir = output_dir / "sigprofiler_input"
    simple_dir.mkdir(parents=True, exist_ok=True)
    destination = simple_dir / "POG570.txt"
    samples: set[str] = set()
    rows = 0
    with gzip.open(source, "rt") as source_handle, destination.open("w", encoding="utf-8") as output_handle:
        header = source_handle.readline().rstrip("\n").split("\t")
        columns = {name: index for index, name in enumerate(header)}
        required = [
            "Tumor_Sample_Barcode", "Chromosome", "Start_Position", "End_Position",
            "Reference_Allele", "Tumor_Seq_Allele2",
        ]
        missing = [name for name in required if name not in columns]
        if missing:
            raise RuntimeError(f"POG MAF lacks required columns: {missing}")
        output_handle.write("project\tsample\tunused1\tunused2\tunused3\tchrom\tstart\tend\tref\talt\n")
        for line_number, line in enumerate(source_handle, start=2):
            row = line.rstrip("\n").split("\t")
            if len(row) != len(header):
                raise RuntimeError(f"POG MAF line {line_number} has {len(row)} fields; expected {len(header)}")
            sample = first_12(row[columns["Tumor_Sample_Barcode"]])
            samples.add(sample)
            output_handle.write(
                "POG570\t{}\t.\t.\t.\t{}\t{}\t{}\t{}\t{}\n".format(
                    sample,
                    row[columns["Chromosome"]],
                    row[columns["Start_Position"]],
                    row[columns["End_Position"]],
                    row[columns["Reference_Allele"]],
                    row[columns["Tumor_Seq_Allele2"]],
                )
            )
            rows += 1
    manifest = {
        "source_file": source.name,
        "source_sha256": sha256(source),
        "source_bytes": source.stat().st_size,
        "converted_file": portable_path(destination),
        "converted_sha256": sha256(destination),
        "mutation_rows": rows,
        "samples_in_maf": len(samples),
        "coordinate_conversion": "none; POG MAF Start_Position/End_Position are one-based",
    }
    (output_dir / "pog570_input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return simple_dir, samples


def generate_signature_features(simple_dir: Path, output_dir: Path) -> pd.DataFrame:
    from SigProfilerMatrixGenerator.scripts.SigProfilerMatrixGeneratorFunc import SigProfilerMatrixGeneratorFunc

    matrices = SigProfilerMatrixGeneratorFunc(
        "POG570",
        "GRCh37",
        str(simple_dir),
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
    features = features.astype(np.float32)
    features.to_csv(output_dir / "pog570_standard_sbs96_id83_features.csv.gz", compression="gzip")
    sbs.to_csv(output_dir / "pog570_sigprofiler_sbs96_counts.csv.gz", compression="gzip")
    ids.to_csv(output_dir / "pog570_sigprofiler_id83_counts.csv.gz", compression="gzip")
    return features


def stage_bio_inputs(raw_dir: Path, output_dir: Path, signatures: pd.DataFrame) -> Path:
    stage = output_dir / "bio_maf_staging"
    raw_stage = stage / "raw"
    feature_stage = stage / "features"
    raw_stage.mkdir(parents=True, exist_ok=True)
    feature_stage.mkdir(parents=True, exist_ok=True)
    target = raw_stage / "mc3.v0.2.8.PUBLIC.maf.gz"
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(raw_dir / "POG570_mutations.maf.gz")
    signatures.to_csv(feature_stage / "features_standard_sbs96_id83.csv.gz", compression="gzip")
    return stage


def generate_model_features(
    raw_dir: Path,
    signatures: pd.DataFrame,
    model_dir: Path,
    output_dir: Path,
    chunksize: int,
    force: bool = False,
) -> pd.DataFrame:
    stage = stage_bio_inputs(raw_dir, output_dir, signatures)
    bio_dir = output_dir / "bio_maf_features"
    paths = {"raw_data": {"mc3_source_dir": stage}}
    bio = build_bio_v4_features(paths, bio_dir, chunksize=chunksize, force=force)
    signature_block = signatures.rename(columns={column: f"sig__{column}" for column in signatures if column in set(bio.columns)})
    common = signature_block.index.intersection(bio.index)
    combined = pd.concat([signature_block.loc[common], bio.loc[common]], axis=1)
    order = [line.strip() for line in (model_dir / "feature_order.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [column for column in order if column not in combined]
    extra = [column for column in combined if column not in order]
    if missing or extra:
        raise RuntimeError(f"Feature schema mismatch: missing={missing[:8]}, extra={extra[:8]}")
    combined = combined.reindex(columns=order).fillna(0.0).astype(np.float32)
    if not np.isfinite(combined.to_numpy()).all():
        raise RuntimeError("POG570 model feature matrix contains non-finite values")
    combined.to_csv(output_dir / "pog570_signatures_plus_MAF_stack_features.csv.gz", compression="gzip")
    return combined


def load_cached_model_features(
    output_dir: Path,
    model_dir: Path,
    clinical: pd.DataFrame,
) -> pd.DataFrame:
    features_path = output_dir / "pog570_signatures_plus_MAF_stack_features.csv.gz"
    if not features_path.exists():
        raise RuntimeError(
            "--reuse-features requires pog570_signatures_plus_MAF_stack_features.csv.gz "
            "in the output directory"
        )
    features = pd.read_csv(features_path, index_col=0)
    features.index = features.index.astype(str)
    expected_samples = set(clinical["model_sample_id"])
    if len(features) != 570 or features.index.nunique() != 570:
        raise RuntimeError(f"Invalid cached POG570 features: shape={features.shape}")
    if set(features.index) != expected_samples:
        raise RuntimeError("Cached POG570 feature and clinical sample IDs do not match exactly")
    feature_order = [
        line.strip()
        for line in (model_dir / "feature_order.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing = [column for column in feature_order if column not in features]
    extra = [column for column in features if column not in feature_order]
    if missing or extra:
        raise RuntimeError(f"Cached feature schema mismatch: missing={missing[:8]}, extra={extra[:8]}")
    features = features.reindex(columns=feature_order).astype(np.float32)
    if not np.isfinite(features.to_numpy()).all():
        raise RuntimeError("Cached POG570 feature matrix contains non-finite values")
    return features


def predict(features: pd.DataFrame, clinical: pd.DataFrame, model_dir: Path, output_dir: Path) -> pd.DataFrame:
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    model_hash = sha256(model_dir / "model.ubj")
    if model_hash != manifest["provenance"]["model_sha256"]:
        raise RuntimeError("Frozen model SHA-256 does not match manifest")
    booster = xgb.Booster()
    booster.load_model(model_dir / "model.ubj")
    prediction = booster.predict(xgb.DMatrix(features.to_numpy(dtype=np.float32)))
    if prediction.shape != (len(features),) or not np.isfinite(prediction).all():
        raise RuntimeError(f"Unexpected HRD prediction shape/content: {prediction.shape}")

    indexed = clinical.set_index("model_sample_id")
    if set(features.index) != set(indexed.index):
        raise RuntimeError("POG570 feature and clinical sample IDs do not match exactly")
    frame = indexed.loc[features.index].copy()
    frame["predicted_HRD_SCORE"] = prediction
    frame["model_sha256"] = model_hash
    frame.reset_index().to_csv(output_dir / "pog570_frozen_model_predictions.csv", index=False)
    return frame.reset_index()


def continuous_metrics(data: pd.DataFrame) -> dict[str, float]:
    truth = data["HRD_SCORE"].to_numpy(float)
    pred = data["predicted_HRD_SCORE"].to_numpy(float)
    return {
        "pearson_r": float(pearsonr(truth, pred).statistic) if len(data) >= 3 else math.nan,
        "spearman_rho": float(spearmanr(truth, pred).statistic) if len(data) >= 3 else math.nan,
        "mae": float(mean_absolute_error(truth, pred)),
        "rmse": float(mean_squared_error(truth, pred) ** 0.5),
        "r2": float(r2_score(truth, pred)) if len(data) >= 2 else math.nan,
    }


def cutoff_metrics(data: pd.DataFrame, cutoff: int) -> dict[str, float]:
    truth = data["HRD_SCORE"].ge(cutoff).to_numpy(int)
    score = data["predicted_HRD_SCORE"].to_numpy(float)
    pred = (score >= cutoff).astype(int)
    specificity = float(np.mean(pred[truth == 0] == 0)) if np.any(truth == 0) else math.nan
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)) if len(np.unique(truth)) == 2 else math.nan,
        "sensitivity": float(recall_score(truth, pred, zero_division=0)),
        "specificity": specificity,
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "auroc": float(roc_auc_score(truth, score)) if len(np.unique(truth)) == 2 else math.nan,
        "average_precision": float(average_precision_score(truth, score)) if len(np.unique(truth)) == 2 else math.nan,
        "prevalence": float(truth.mean()),
    }


def bootstrap_rows(data: pd.DataFrame, cohort: str, n_boot: int, seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    continuous_observed = continuous_metrics(data)
    cutoff_observed = {cutoff: cutoff_metrics(data, cutoff) for cutoff in CUTOFFS}
    continuous_boot = {metric: [] for metric in continuous_observed}
    cutoff_boot = {
        cutoff: {metric: [] for metric in metrics}
        for cutoff, metrics in cutoff_observed.items()
    }
    for _ in range(n_boot):
        sample = data.iloc[rng.integers(0, len(data), len(data))]
        for metric, value in continuous_metrics(sample).items():
            continuous_boot[metric].append(value)
        for cutoff in CUTOFFS:
            for metric, value in cutoff_metrics(sample, cutoff).items():
                cutoff_boot[cutoff][metric].append(value)

    rows: list[dict[str, object]] = []
    for metric, estimate in continuous_observed.items():
        values = np.asarray(continuous_boot[metric], dtype=float)
        values = values[np.isfinite(values)]
        rows.append({
            "cohort": cohort, "analysis": "continuous", "cutoff": "", "metric": metric,
            "n": len(data), "estimate": estimate,
            "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)),
        })
    for cutoff in CUTOFFS:
        for metric, estimate in cutoff_observed[cutoff].items():
            values = np.asarray(cutoff_boot[cutoff][metric], dtype=float)
            values = values[np.isfinite(values)]
            rows.append({
                "cohort": cohort, "analysis": "fixed_cutoff", "cutoff": cutoff, "metric": metric,
                "n": len(data), "estimate": estimate,
                "ci_low": float(np.quantile(values, 0.025)) if len(values) else math.nan,
                "ci_high": float(np.quantile(values, 0.975)) if len(values) else math.nan,
            })
    return rows


def evaluate(predictions: pd.DataFrame, output_dir: Path, n_boot: int) -> None:
    primary = predictions[predictions["analysis_group"].eq("primary_breast")]
    rows = bootstrap_rows(primary, "primary_breast", n_boot, SEED)
    rows.extend(bootstrap_rows(predictions, "all_pog570", n_boot, SEED + 1))
    pd.DataFrame(rows).to_csv(output_dir / "pog570_primary_and_all_metrics_with_95ci.csv", index=False)

    cancer_rows: list[dict[str, object]] = []
    exploratory = predictions[predictions["analysis_group"].eq("exploratory_other")]
    for cancer_type, group in exploratory.groupby("CANCER_TYPE", sort=True):
        for metric, value in continuous_metrics(group).items():
            cancer_rows.append({
                "cancer_type": cancer_type, "n": len(group), "analysis": "continuous", "cutoff": "",
                "metric": metric, "estimate": value,
            })
        for cutoff in CUTOFFS:
            for metric, value in cutoff_metrics(group, cutoff).items():
                cancer_rows.append({
                    "cancer_type": cancer_type, "n": len(group), "analysis": "fixed_cutoff", "cutoff": cutoff,
                    "metric": metric, "estimate": value,
                })
    pd.DataFrame(cancer_rows).to_csv(output_dir / "pog570_exploratory_metrics_by_cancer_type.csv", index=False)


def save_reproducibility(raw_dir: Path, model_dir: Path, clinical: pd.DataFrame, feature_samples: set[str], output_dir: Path) -> None:
    mapping = clinical.copy()
    mapping["included"] = mapping["model_sample_id"].isin(feature_samples)
    mapping["exclusion_reason"] = np.where(mapping["included"], "", "no usable SBS96/ID83 and Bio-MAF feature row")
    mapping.to_csv(output_dir / "pog570_sample_mapping_and_exclusions.csv", index=False)
    versions = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "model_file": portable_path(model_dir / "model.ubj"),
        "model_sha256": sha256(model_dir / "model.ubj"),
        "clinical_sha256": sha256(raw_dir / "data_clinical_sample.txt"),
        "mutation_maf_sha256": sha256(raw_dir / "POG570_mutations.maf.gz"),
        "copy_number_sha256": (
            sha256(raw_dir / "POG570_copy_number_cbioportal.txt.gz")
            if (raw_dir / "POG570_copy_number_cbioportal.txt.gz").exists()
            else None
        ),
        "copy_number_used_as_model_input": False,
        "prespecified_cutoffs": list(CUTOFFS),
        "usable_samples": int(mapping["included"].sum()),
        "excluded_samples": int((~mapping["included"]).sum()),
        "preprocessing": [
            "POG standard one-based MAF coordinates passed unchanged to SigProfilerMatrixGenerator GRCh37",
            "SBS96 and ID83 channels normalized within sample; burdens transformed as log10(1+n)",
            "Bio-MAF v4 generated with the fixed repo resources and feature schema",
            "signature columns colliding with Bio-MAF controls prefixed with sig__",
            "columns reordered exactly as feature_order.txt; NaN filled with zero; no scaling",
            "frozen model applied once without fitting or tuning on POG570",
        ],
        "model_caveat": "Vijay's freeze script selected hyperparameters by TCGA-only CV and refit this new artifact on all eligible TCGA samples.",
        "cohort_difference_note": (
            "POG570 comprises advanced/metastatic or recurrent cancers profiled by whole-genome sequencing, whereas the "
            "TCGA training cohort predominantly comprises untreated primary tumors represented by MC3 exome mutation calls."
        ),
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
    (output_dir / "software_versions_and_preprocessing.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW,
        help="POG570 raw-data directory (default: $POG570_RAW_DIR or data/external_validation/pog570)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--chunksize", type=int, default=350_000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="rebuild Bio-MAF feature caches instead of reusing compatible cached files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clinical = load_clinical(args.raw_dir)
    if args.reuse_features and not args.force_features:
        features = load_cached_model_features(args.output_dir, args.model_dir, clinical)
    else:
        simple_dir, maf_samples = prepare_sigprofiler_input(args.raw_dir, args.output_dir)
        if set(clinical["model_sample_id"]) != maf_samples:
            raise RuntimeError("Clinical and MAF sample identifiers do not match exactly")
        signatures = generate_signature_features(simple_dir, args.output_dir)
        features = generate_model_features(
            args.raw_dir,
            signatures,
            args.model_dir,
            args.output_dir,
            args.chunksize,
            force=args.force_features,
        )
    predictions = predict(features, clinical, args.model_dir, args.output_dir)
    evaluate(predictions, args.output_dir, args.bootstrap)
    save_reproducibility(args.raw_dir, args.model_dir, clinical, set(features.index), args.output_dir)
    print(f"POG570 complete: {len(predictions)} usable samples; outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
