#!/usr/bin/env python3
"""Do genomic features add survival signal beyond standard clinical covariates?

Extension of scripts/run_survival_cancer_type_baseline.py for reviewer rigor:
standard TCGA survival analyses adjust for age, sex, and stage. This script adds
clinical-covariate arms on the IDENTICAL cohort (n=9,986) and IDENTICAL outer
folds (PAIRED_SEED=20260801) as the committed three-arm run, so every C-index is
directly comparable.

Arms
----
    clinical                              age + gender + AJCC stage (one-hot w/ missing indicator)
    clinical_plus_cancer_type             clinical + one-hot cancer type
    clinical_plus_cancer_type_plus_bio_maf  clinical + cancer type + Bio MAF v4

Missing clinical values are median/mode-imputed with a missing indicator, so the
cohort is exactly the committed n=9,986 (no sample is dropped for missing stage).
Age is z-scored on the full cohort (a label-free transform; no leakage).

Writes results/tables/survival_clinical_baseline_summary.csv and
results/tables/survival_clinical_baseline_comparisons.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from utils.config import load_yaml, resolve_paths_map  # noqa: E402
from utils.endpoint_registry import build_cdr_cancer_type_labels, build_cdr_survival_labels  # noqa: E402
from utils.nested_oof import evaluate_nested_oof  # noqa: E402
from run_survival_cancer_type_baseline import (  # noqa: E402
    PAIRED_SEED,
    concordance,
    load_bio_maf_features,
    one_hot_cancer_type,
    paired_bootstrap,
    run_arm,
)

ARMS = ("clinical", "clinical_plus_cancer_type", "clinical_plus_cancer_type_plus_bio_maf")


def load_clinical_features(mc3_dir: Path) -> pd.DataFrame:
    """Age + gender + AJCC stage design matrix, indexed by 12-char patient barcode."""
    cdr = pd.read_excel(
        mc3_dir / "raw" / "TCGA-CDR-SupplementalTableS1.xlsx",
        usecols=["bcr_patient_barcode", "age_at_initial_pathologic_diagnosis",
                 "gender", "ajcc_pathologic_tumor_stage"],
    )
    cdr["patient"] = cdr["bcr_patient_barcode"].astype(str).str[:12]
    cdr = cdr.drop_duplicates("patient").set_index("patient")

    out = pd.DataFrame(index=cdr.index)
    age = pd.to_numeric(cdr["age_at_initial_pathologic_diagnosis"], errors="coerce")
    out["age_missing"] = age.isna().astype(np.float32)
    age = age.fillna(age.median())
    out["age_z"] = ((age - age.mean()) / age.std()).astype(np.float32)

    gender = cdr["gender"].astype(str).str.upper()
    out["gender_male"] = (gender == "MALE").astype(np.float32)
    out["gender_missing"] = (~gender.isin(["MALE", "FEMALE"])).astype(np.float32)

    stage = cdr["ajcc_pathologic_tumor_stage"].fillna("[Not Available]").astype(str)
    # Collapse substages to I/II/III/IV; keep explicit missing levels as their own indicator.
    def collapse(s: str) -> str:
        s = s.strip()
        if s in ("[Not Available]", "[Not Applicable]", "[Unknown]", "nan", ""):
            return "stage_missing"
        m = s.replace("Stage ", "")
        if m.startswith("0"):
            return "stage_0"
        for roman in ("IV", "III", "II", "I"):
            if m.startswith(roman):
                return f"stage_{roman}"
        return "stage_other"
    stage_simple = stage.map(collapse)
    dummies = pd.get_dummies(stage_simple, prefix="", prefix_sep="", dtype=np.float32)
    dummies.columns = ["stage_" + str(c).replace("stage_", "") if not str(c).startswith("stage_") else str(c) for c in dummies.columns]
    out = pd.concat([out, dummies.reindex(sorted(dummies.columns), axis=1)], axis=1)
    return out.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--config", default="config/experiment_settings.strict_no_leakage.yaml")
    parser.add_argument("--endpoints", nargs="+", default=["OS"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    paths = resolve_paths_map(load_yaml(args.paths))
    mc3_dir = Path(paths["raw_data"]["mc3_source_dir"])
    if not mc3_dir.exists():
        raise SystemExit(f"mc3_source_dir does not exist: {mc3_dir}")

    full = load_yaml(args.config)
    panel = ((full.get("experiments") or {}).get("main_manuscript_complete_panel") or {})
    settings = {key: value for key, value in {**full, **panel}.items() if not isinstance(value, dict)}

    bio_maf = load_bio_maf_features()
    survival_labels, _audit = build_cdr_survival_labels(mc3_dir, args.endpoints)
    if not survival_labels:
        raise SystemExit(f"no survival labels built for {args.endpoints}")

    summary_rows: list[dict] = []
    comparison_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for endpoint, labels in survival_labels.items():
        print(f"\n=== {endpoint} ===", flush=True)
        cancer_type = build_cdr_cancer_type_labels(mc3_dir, endpoint_name="cancer_type_all")
        clinical = load_clinical_features(mc3_dir)

        # Identical cohort to the committed three-arm run: intersection of survival,
        # cancer type, and Bio MAF. Clinical missingness is imputed, never dropped.
        common = labels.index.astype(str).intersection(cancer_type.index.astype(str))
        common = common.intersection(bio_maf.index.astype(str))
        common = pd.Index(sorted(set(map(str, common))))
        print(f"  cohort after intersection: n={len(common)}", flush=True)

        surv = labels.loc[common]
        ct_design = one_hot_cancer_type(cancer_type.loc[common])
        clin_design = clinical.reindex(common)
        print(f"  clinical features: {clin_design.shape[1]} (age_z, age_missing, gender x2, stage x{clin_design.shape[1]-4})", flush=True)

        designs = {}
        if "clinical" in args.arms:
            designs["clinical"] = clin_design
        if "clinical_plus_cancer_type" in args.arms:
            designs["clinical_plus_cancer_type"] = pd.concat([clin_design, ct_design], axis=1)
        if "clinical_plus_cancer_type_plus_bio_maf" in args.arms:
            designs["clinical_plus_cancer_type_plus_bio_maf"] = pd.concat(
                [clin_design, ct_design, bio_maf.loc[common]], axis=1)

        risks: dict[str, pd.Series] = {}
        for arm in args.arms:
            row, preds = run_arm(arm, endpoint, designs[arm], surv, settings, args.folds)
            summary_rows.append(row)
            preds = preds.copy()
            preds["arm"] = arm
            prediction_frames.append(preds)
            score_col = "risk_score" if "risk_score" in preds.columns else "pred_value"
            risks[arm] = preds.set_index("sample")[score_col].reindex(common)
            print(f"  [{arm}] C-index = {row.get('score'):.4f}", flush=True)

        time = surv["time"].to_numpy(dtype=float)
        event = surv["event"].to_numpy(dtype=int)
        for reference, candidate in (
            ("clinical", "clinical_plus_cancer_type"),
            ("clinical_plus_cancer_type", "clinical_plus_cancer_type_plus_bio_maf"),
            ("clinical", "clinical_plus_cancer_type_plus_bio_maf"),
        ):
            if reference not in risks or candidate not in risks:
                continue
            stats = paired_bootstrap(
                time, event,
                risks[reference].to_numpy(dtype=float),
                risks[candidate].to_numpy(dtype=float),
                n_boot=args.bootstrap,
                seed=PAIRED_SEED,
            )
            comparison_rows.append({"endpoint": endpoint, "reference": reference, "candidate": candidate, **stats})

    tables = REPO_ROOT / "results/tables"
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables / "survival_clinical_baseline_summary.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            tables / "survival_clinical_baseline_predictions.csv.gz", index=False, compression="gzip")
    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        comparisons.to_csv(tables / "survival_clinical_baseline_comparisons.csv", index=False)

    print("\n" + "=" * 74)
    print("RESULT (Harrell C-index, pooled OOF, identical folds to the 3-arm run)")
    print("=" * 74)
    for endpoint in summary["endpoint"].unique():
        block = summary[summary["endpoint"] == endpoint]
        for _, row in block.iterrows():
            print(f"  {row['arm']:42s} {float(row['score']):.4f}   ({int(row['n_features'])} features)")
        if not comparisons.empty:
            for _, row in comparisons[comparisons["endpoint"] == endpoint].iterrows():
                sig = "" if row["p_two_sided"] >= 0.05 else "  *"
                print(f"  {row['candidate']} vs {row['reference']}: delta={row['delta']:+.4f} "
                      f"[{row['ci_low']:+.4f},{row['ci_high']:+.4f}] p={row['p_two_sided']:.3f}{sig}")
    print("\nwritten -> results/tables/survival_clinical_baseline_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
