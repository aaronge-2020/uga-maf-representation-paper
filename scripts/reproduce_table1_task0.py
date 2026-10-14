#!/usr/bin/env python
"""Task 0: Reproduce Table 1 (cancer typing, 20 types, n=8800) from deposited inputs.

Uses the EXACT nested-OOF code path from src/utils/nested_oof.py, the EXACT
seeds from the checkpoint manifests (recomputed via stable_seed and verified),
and the pinned software environment (xgboost 1.6.2, sklearn 1.7.2, ...).

Inputs (all verified against checkpoint digests before running):
- samples/labels: reconstructed from features_burden_only.csv order + TCGA CDR,
  verified via samples_digest + labels_digest.
- burden_only (3): features_burden_only.csv (git blob)
- standard_sbs96_id83 (182): features_standard_sbs96_id83.csv.gz (git blob)
- MAF_stack_only (948): results/tables/quick_bio_v4_maf_features.csv.gz
- signatures_plus_MAF_stack (1130): 182 + 948
- feature sets: original 948-column v4 blocks (reconstructed, digests verified)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from utils.nested_oof import (  # noqa: E402
    _label_digest,
    _matrix_digest,
    _sample_digest,
    encode_labels,
    evaluate_nested_feature_set_oof,
    evaluate_nested_oof,
    stable_seed,
)
# FIX (2026-10-13): the four single-matrix slots (burden_only,
# standard_sbs96_id83) were run in production by evaluate_features() in
# utils.feature_evaluation, which uses its OWN module-local stable_seed
# (20260518-based), not the nested_oof one (1729-based). The rerun script
# previously used nested_oof.stable_seed for all eight slots, producing wrong
# seeds for the four single-matrix slots. Feature-set slots are unaffected.
from utils.feature_evaluation import stable_seed as fe_stable_seed  # noqa: E402

OUT = REPO / "results" / "tables" / "task0_rerun_outputs"
OUT.mkdir(parents=True, exist_ok=True)
FEAT = REPO / "data" / "task0_features"  # canonical in-repo input copies

FIXED_CLASSES = ["BRCA","UCEC","LUAD","LGG","HNSC","PRAD","THCA","LUSC","SKCM","STAD",
                 "BLCA","OV","COAD","GBM","KIRC","LIHC","CESC","KIRP","SARC","ESCA"]

EXPECTED_BAL_ACC = {
    ("burden_only", "linear"): 0.2473,
    ("burden_only", "xgboost"): 0.2549,
    ("standard_sbs96_id83", "linear"): 0.5263,
    ("standard_sbs96_id83", "xgboost"): 0.5381,
    ("MAF_stack_only", "linear"): 0.5712,
    ("MAF_stack_only", "xgboost"): 0.6618,
    ("signatures_plus_MAF_stack", "linear"): 0.6723,
    ("signatures_plus_MAF_stack", "xgboost"): 0.7378,
}

# Seeds: MAF ones verified against checkpoint manifests; burden/standard from
# utils.feature_evaluation.stable_seed (the 20260518-based formula actually used
# by evaluate_features() in the August production run), NOT nested_oof's.
CACHE_MAF = "main_complete_panel_mc3_maf_stack_bio_v4_nested_fs_15d698606a129584e3f82812"
CACHE_SIGMAF = "main_complete_panel_mc3_signatures_plus_maf_stack_bio_v4_nested_fs_ac11587791bbbd40db3991bb"
SEEDS = {
    ("burden_only", "linear"): fe_stable_seed("mc3_main", "cancer_type_top20", "linear", "burden_only"),
    ("burden_only", "xgboost"): fe_stable_seed("mc3_main", "cancer_type_top20", "xgboost", "burden_only"),
    ("standard_sbs96_id83", "linear"): fe_stable_seed("mc3_main", "cancer_type_top20", "linear", "standard_sbs96_id83"),
    ("standard_sbs96_id83", "xgboost"): fe_stable_seed("mc3_main", "cancer_type_top20", "xgboost", "standard_sbs96_id83"),
    ("MAF_stack_only", "linear"): stable_seed("mc3_main", "cancer_type_top20", "linear", "MAF_stack_only", CACHE_MAF, "bio_v4_feature_sets"),
    ("MAF_stack_only", "xgboost"): stable_seed("mc3_main", "cancer_type_top20", "xgboost", "MAF_stack_only", CACHE_MAF, "bio_v4_feature_sets"),
    ("signatures_plus_MAF_stack", "linear"): stable_seed("mc3_main", "cancer_type_top20", "linear", "signatures_plus_MAF_stack", CACHE_SIGMAF, "bio_v4_feature_sets"),
    ("signatures_plus_MAF_stack", "xgboost"): stable_seed("mc3_main", "cancer_type_top20", "xgboost", "signatures_plus_MAF_stack", CACHE_SIGMAF, "bio_v4_feature_sets"),
}
# Verified against the August production values (investigation 2026-10-13).
assert SEEDS[("burden_only", "linear")] == 20284067
assert SEEDS[("burden_only", "xgboost")] == 20287593
assert SEEDS[("standard_sbs96_id83", "linear")] == 20295293
assert SEEDS[("standard_sbs96_id83", "xgboost")] == 20311142
# Verified against checkpoint manifests: 243603, 615371, 566310, 750978.
assert SEEDS[("MAF_stack_only", "linear")] == 243603
assert SEEDS[("MAF_stack_only", "xgboost")] == 615371
assert SEEDS[("signatures_plus_MAF_stack", "linear")] == 566310
assert SEEDS[("signatures_plus_MAF_stack", "xgboost")] == 750978


def base_settings() -> dict:
    """Math-relevant settings matching config/experiment_settings.strict_no_leakage.yaml
    and the runner's nested_settings. Checkpointing disabled for a clean rerun."""
    return {
        "tree_method": "gpu_hist",
        "xgb_n_jobs": 32,
        "xgb_estimators": 160,
        "random_search_trials": 10,
        "optuna_trials": 10,
        "fold_checkpoint_enabled": False,
        "candidate_checkpoint_enabled": False,
        # linear model settings from config
        "linear_high_dim_classification_solver": "sgd_log_loss",
        "linear_high_dim_classification_solver_min_features": 500,
        "linear_sgd_max_iter": 200,
        "linear_sgd_tol": 1e-3,
        "linear_sgd_n_iter_no_change": 5,
        "linear_sgd_average": True,
        "linear_candidate_n_jobs": 32,
        "linear_high_dim_candidate_n_jobs": 16,
        "linear_process_backend_min_features": 500,
        "linear_multiclass_process_backend": False,
        "candidate_memmap_max_nbytes": "10M",
    }


def load_labels() -> tuple[pd.Index, pd.Series]:
    burden = pd.read_csv(FEAT / "features_burden_only.csv", index_col=0)
    patients = burden.index.astype(str)
    cdr = pd.read_excel(FEAT / "TCGA-CDR-SupplementalTableS1.xlsx",
                        usecols=["bcr_patient_barcode", "type"])
    cdr["patient"] = cdr["bcr_patient_barcode"].astype(str).str[:12]
    labels_all = cdr.drop_duplicates("patient").set_index("patient")["type"].astype(str)
    labels = labels_all.reindex(pd.Index([str(v) for v in patients])).dropna()
    labels = labels[labels.isin(FIXED_CLASSES)]
    labels.name = "cancer_type_top20"
    samples = pd.Index([str(v) for v in labels.index], name="sample")
    return samples, labels


def verify_inputs(samples, labels, feature_sets_by_rep: dict) -> None:
    y, _ = encode_labels(labels, "multiclass")
    assert _sample_digest(samples) == "1bec742b3353fad4c756e167", "samples digest mismatch"
    assert _label_digest("multiclass", y) == "1e050a771d361c6e1d93d535", "labels digest mismatch"
    print("[verify] samples + labels digests OK (n=8800)", flush=True)
    # Verify MAF feature-set digests against the Sig+MAF/xgboost checkpoint manifest.
    d = json.load(open(REPO / "results/tables/main_manuscript_complete_panel_nested_fold_checkpoints/cancer_type_top20__signatures_plus_MAF_stack__xgboost__main_complete_panel_mc3_signatures_plus_maf_stack_bio_v4_nested_fs_ac11587791bbbd40db3991bb.json"))
    expected = d["manifest"]["feature_digests"]
    got_sets = feature_sets_by_rep["signatures_plus_MAF_stack"]
    for name, exp in expected.items():
        arr = np.nan_to_num(np.asarray(got_sets[name], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        assert _matrix_digest(arr) == exp, f"feature digest mismatch: {name}"
    print("[verify] all 7 Sig+MAF feature-set digests OK", flush=True)
    d2 = json.load(open(REPO / "results/tables/main_manuscript_complete_panel_nested_fold_checkpoints/cancer_type_top20__MAF_stack_only__xgboost__main_complete_panel_mc3_maf_stack_bio_v4_nested_fs_15d698606a129584e3f82812.json"))
    for name, exp in d2["manifest"]["feature_digests"].items():
        arr = np.nan_to_num(np.asarray(feature_sets_by_rep["MAF_stack_only"][name], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        assert _matrix_digest(arr) == exp, f"feature digest mismatch: {name}"
    print("[verify] all 6 MAF feature-set digests OK", flush=True)


def build_feature_sets(samples) -> dict:
    """Build the four representation feature-set dicts, aligned to samples."""
    from utils.quick_bio_v4_nested_feature_selection import v4_feature_columns, load_v4_resources
    from utils.bio_maf_base_features import OPTIONAL_CONTROLS

    maf = pd.read_csv(REPO / "results/tables/quick_bio_v4_maf_features.csv.gz", index_col=0)
    maf.index = maf.index.astype(str)
    std = pd.read_csv(FEAT / "features_standard_sbs96_id83.csv.gz", index_col=0)
    std.index = std.index.astype(str)
    burden = pd.read_csv(FEAT / "features_burden_only.csv", index_col=0)
    burden.index = burden.index.astype(str)

    # Reconstruct the ORIGINAL 948-column v4 blocks (current code excludes 38
    # role-matched columns; the original run interleaved them per gene).
    _, _, blocks = v4_feature_columns()
    panel, *_ = load_v4_resources()
    genes = panel["gene"].astype(str).tolist()
    orig_exact: list[str] = []
    for gene in genes:
        orig_exact.append(f"driver_gene_functional_event_log_count__{gene}")
        orig_exact.append(f"driver_gene_role_matched_event_log_count__{gene}")
    base_core = blocks["v4_compact_biology_only"]
    tier_cols = blocks["v4_compact_plus_driver_evidence_tiers"][len(base_core):]
    hotspot_cols = blocks["v4_compact_plus_hotspot_summary"][len(base_core):]
    orig_blocks = {
        "v4_compact_biology_only": base_core,
        "v4_compact_plus_driver_evidence_tiers": [*base_core, *tier_cols],
        "v4_compact_plus_hotspot_summary": [*base_core, *hotspot_cols],
        "v4_compact_plus_exact_driver_genes": [*base_core, *orig_exact],
        "v4_full_core": [*base_core, *tier_cols, *orig_exact, *hotspot_cols],
        "v4_full_core_plus_optional_controls": [*base_core, *tier_cols, *orig_exact, *hotspot_cols, *OPTIONAL_CONTROLS],
    }

    maf_al = maf.reindex(samples)
    std_al = std.add_prefix("signature_matrix__").reindex(samples)
    X_sig = std_al.to_numpy(dtype=np.float32)
    X_burden = burden.reindex(samples).to_numpy(dtype=np.float32)

    maf_sets = {name: maf_al[cols].to_numpy(dtype=np.float32) for name, cols in orig_blocks.items()}
    sigmaf_sets = {"signatures_only": X_sig}
    for name, cols in orig_blocks.items():
        sigmaf_sets[f"signatures_plus_{name}"] = np.concatenate(
            [X_sig, maf_al[cols].to_numpy(dtype=np.float32)], axis=1)

    return {
        "burden_only": {"burden_only": X_burden},
        "standard_sbs96_id83": {"standard_sbs96_id83": std_al.to_numpy(dtype=np.float32)},
        "MAF_stack_only": maf_sets,
        "signatures_plus_MAF_stack": sigmaf_sets,
    }


def run_combo(representation: str, learner: str, samples, labels, feature_sets: dict) -> dict:
    seed = SEEDS[(representation, learner)]
    settings = base_settings()
    print(f"[run] {representation}/{learner} seed={seed} n_sets={len(feature_sets)}", flush=True)
    t0 = time.time()
    if representation in ("MAF_stack_only", "signatures_plus_MAF_stack"):
        result = evaluate_nested_feature_set_oof(
            samples=samples, feature_sets=feature_sets, labels=labels,
            task="multiclass", benchmark="mc3_main", endpoint="cancer_type_top20",
            representation=representation, learner=learner,
            settings=settings, n_splits=5, seed=seed,
        )
    else:
        (only_name, only_x), = feature_sets.items()
        result = evaluate_nested_oof(
            samples=samples, x=only_x, labels=labels,
            task="multiclass", benchmark="mc3_main", endpoint="cancer_type_top20",
            representation=representation, learner=learner,
            settings=settings, n_splits=5, seed=seed,
        )
    elapsed = time.time() - t0
    summary = dict(result.summary)
    pred_frame = result.predictions
    fold_frame = result.fold_diagnostics

    # Verify splits match the archived checkpoints where available.
    ck_map = {
        ("MAF_stack_only", "linear"): "cancer_type_top20__MAF_stack_only__linear__main_complete_panel_mc3_maf_stack_bio_v4_nested_fs_15d698606a129584e3f82812.npz",
        ("MAF_stack_only", "xgboost"): "cancer_type_top20__MAF_stack_only__xgboost__main_complete_panel_mc3_maf_stack_bio_v4_nested_fs_15d698606a129584e3f82812.npz",
        ("signatures_plus_MAF_stack", "linear"): "cancer_type_top20__signatures_plus_MAF_stack__linear__main_complete_panel_mc3_signatures_plus_maf_stack_bio_v4_nested_fs_ac11587791bbbd40db3991bb.npz",
        ("signatures_plus_MAF_stack", "xgboost"): "cancer_type_top20__signatures_plus_MAF_stack__xgboost__main_complete_panel_mc3_signatures_plus_maf_stack_bio_v4_nested_fs_ac11587791bbbd40db3991bb.npz",
    }
    if (representation, learner) in ck_map:
        z = np.load(REPO / "results/tables/main_manuscript_complete_panel_nested_fold_checkpoints" / ck_map[(representation, learner)])
        archived_folds = z["sample_folds"]
        assert np.array_equal(pred_frame["fold"].to_numpy(), archived_folds), \
            f"split mismatch for {representation}/{learner}"
        print(f"[run] {representation}/{learner}: regenerated splits match archived checkpoint", flush=True)

    tag = f"cancer_type_top20__{representation}__{learner}"
    pred_frame.to_csv(OUT / f"{tag}__oof_predictions.csv", index=False)
    fold_frame.to_csv(OUT / f"{tag}__fold_metrics.csv", index=False)
    with open(OUT / f"{tag}__summary.json", "w") as f:
        json.dump({"representation": representation, "learner": learner, "seed": seed,
                   "elapsed_seconds": elapsed, "summary": summary}, f, indent=1, default=str)

    bal_acc = float(summary["balanced_accuracy"])
    exp = EXPECTED_BAL_ACC[(representation, learner)]
    status = "MATCH" if round(bal_acc, 4) == exp else "DIFFER"
    print(f"[result] {representation}/{learner}: balanced_accuracy={bal_acc:.6f} "
          f"(expected {exp:.4f}) [{status}] elapsed={elapsed:.1f}s", flush=True)
    return {"representation": representation, "learner": learner, "seed": seed,
            "balanced_accuracy": bal_acc, "expected": exp, "status": status,
            "elapsed_seconds": elapsed}


def main() -> None:
    only = sys.argv[1:] or None
    combos = [
        ("burden_only", "linear"), ("burden_only", "xgboost"),
        ("standard_sbs96_id83", "linear"), ("standard_sbs96_id83", "xgboost"),
        ("MAF_stack_only", "linear"), ("MAF_stack_only", "xgboost"),
        ("signatures_plus_MAF_stack", "linear"), ("signatures_plus_MAF_stack", "xgboost"),
    ]
    if only:
        combos = [c for c in combos if f"{c[0]}/{c[1]}" in only or c[0] in only]
    samples, labels = load_labels()
    print(f"[init] n_samples={len(samples)}", flush=True)
    feature_sets_by_rep = build_feature_sets(samples)
    verify_inputs(samples, labels, feature_sets_by_rep)
    results = []
    for representation, learner in combos:
        results.append(run_combo(representation, learner, samples, labels,
                                 feature_sets_by_rep[representation]))
    with open(OUT / "rerun_summary.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['representation']:25s} {r['learner']:8s} {r['balanced_accuracy']:.6f} "
              f"(exp {r['expected']:.4f}) [{r['status']}]")


if __name__ == "__main__":
    main()
