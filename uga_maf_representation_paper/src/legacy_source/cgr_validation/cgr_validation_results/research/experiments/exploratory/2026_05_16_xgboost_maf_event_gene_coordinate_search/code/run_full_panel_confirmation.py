"""Full-panel repeated-CV confirmation for the retained MAF event-coordinate candidate."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost

from utils.checkpointing import atomic_write_csv, atomic_write_json, merge_checkpoint_rows

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
DATA_DIR = EXPERIMENT_ROOT / "data"
TABLE_DIR = EXPERIMENT_ROOT / "tables"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_xgboost_maf_event_gene_coordinate_search as search  # noqa: E402


DEFAULT_MODEL = "id_plus_best_gene_locus_multiscale_stack"


def safe_model_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:120]


def load_full_panel_endpoints(patients: pd.Index) -> list[search.base.Endpoint]:
    endpoints: list[search.base.Endpoint] = []
    endpoints.extend(search.base.load_hrd_endpoints())
    endpoints.extend(search.base.load_mc3_clinical_endpoints())
    endpoints.append(search.base.load_kmt2c_endpoint(patients))
    ordered = [
        "HRD_Score",
        "eCARD",
        "HRD_TAI",
        "HRD_LST",
        "HRD_LOH",
        "PARPi7",
        "hrd_binary_24",
        "hrd_binary_33",
        "hrd_binary_42",
        "parpi7_binary",
        "os_event",
        "high_stage",
        "high_purity",
        "smoking_ever",
        "cancer_type_top10",
        "luad_kmt2c_mutated",
    ]
    by_name = {endpoint.name: endpoint for endpoint in endpoints}
    return [by_name[name] for name in ordered if name in by_name]


def html_table(df: pd.DataFrame, title: str, footnote: str) -> str:
    return search.html_table(df, title, footnote)


def format_for_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return search.format_for_table(df, columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--tree-method", default="gpu_hist")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--xgb-estimators", type=int, default=160)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    _standard_sbs, _standard_id, standard_sbs_id, _burden = search.base.load_feature_matrices()
    patient_ids = standard_sbs_id.index.astype(str).tolist()
    identity = search.exact_identity_frame(standard_sbs_id)
    raw_blocks = search.build_gene_coordinate_blocks(patient_ids)
    raw_blocks.update(search.load_reference_locus_blocks(patient_ids))
    blocks = search.build_transformed_blocks(raw_blocks)
    specs = search.make_candidate_specs()
    if args.model_id not in {spec.model_id for spec in specs}:
        raise ValueError(f"Unknown model_id: {args.model_id}")
    endpoints = load_full_panel_endpoints(standard_sbs_id.index)

    results = search.run_confirmation(
        [args.model_id],
        specs,
        endpoints,
        standard_sbs_id.astype(np.float32),
        identity,
        blocks,
        folds=args.folds,
        repeats=args.repeats,
        n_estimators=args.xgb_estimators,
        tree_method=args.tree_method,
        bootstrap=args.bootstrap,
    )
    model_tag = safe_model_name(args.model_id)
    model_results_path = DATA_DIR / f"full_panel_confirmation_results__{model_tag}.csv"
    atomic_write_csv(results, model_results_path, index=False)
    merge_checkpoint_rows(
        DATA_DIR / "full_panel_confirmation_results.csv",
        results.to_dict("records"),
        key_columns=["model_id", "endpoint"],
        sort_columns=["model_id", "endpoint"],
    )
    summary = (
        results.groupby("endpoint_family", as_index=False)
        .agg(
            n_endpoints=("endpoint", "nunique"),
            mean_delta=("delta_vs_standard", "mean"),
            min_delta=("delta_vs_standard", "min"),
            max_delta=("delta_vs_standard", "max"),
            positive_endpoints=("delta_vs_standard", lambda x: int(np.sum(np.asarray(x) > 0))),
            fdr_significant_positive=("q_value", lambda x: int(np.sum((results.loc[x.index, "delta_vs_standard"] > 0) & (x < 0.05)))),
        )
        .sort_values("mean_delta", ascending=False)
    )
    summary.insert(0, "model_id", args.model_id)
    atomic_write_csv(summary, DATA_DIR / f"full_panel_confirmation_summary__{model_tag}.csv", index=False)
    merge_checkpoint_rows(
        DATA_DIR / "full_panel_confirmation_summary.csv",
        summary.to_dict("records"),
        key_columns=["model_id", "endpoint_family"],
        sort_columns=["model_id", "endpoint_family"],
    )

    (TABLE_DIR / f"table5_full_panel_confirmation__{model_tag}.html").write_text(
        html_table(
            format_for_table(
                results,
                [
                    "endpoint",
                    "endpoint_family",
                    "metric",
                    "standard_score",
                    "candidate_score",
                    "delta_vs_standard",
                    "delta_ci_low",
                    "delta_ci_high",
                    "p_value",
                    "q_value",
                    "n",
                ],
            ),
            "Full-panel repeated-CV confirmation",
            "The candidate is compared with unchanged Standard SBS96+ID83 using identical XGBoost settings, folds, patients, labels, and metrics. P values and q values are paired bootstrap estimates.",
        ),
        encoding="utf-8",
    )
    (TABLE_DIR / f"table6_full_panel_family_summary__{model_tag}.html").write_text(
        html_table(
            format_for_table(summary, ["endpoint_family", "n_endpoints", "mean_delta", "min_delta", "max_delta", "positive_endpoints", "fdr_significant_positive"]),
            "Full-panel endpoint-family summary",
            "Positive endpoint counts use delta >0. FDR-significant positive counts use q<0.05 within endpoint family.",
        ),
        encoding="utf-8",
    )

    metadata = {
        "completed_local": datetime.now().replace(microsecond=0).isoformat(),
        "completed_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "model_id": args.model_id,
        "folds": args.folds,
        "repeats": args.repeats,
        "xgb_estimators": args.xgb_estimators,
        "bootstrap": args.bootstrap,
        "tree_method": args.tree_method,
        "n_endpoints": int(results["endpoint"].nunique()),
        "mean_delta": float(results["delta_vs_standard"].mean()),
        "min_delta": float(results["delta_vs_standard"].min()),
        "max_delta": float(results["delta_vs_standard"].max()),
        "positive_endpoints": int(np.sum(results["delta_vs_standard"] > 0)),
        "fdr_significant_positive": int(np.sum((results["delta_vs_standard"] > 0) & (results["q_value"] < 0.05))),
        "python_version": platform.python_version(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    atomic_write_json(DATA_DIR / f"full_panel_confirmation_metadata__{model_tag}.json", metadata)
    atomic_write_json(DATA_DIR / "full_panel_confirmation_metadata_latest.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
