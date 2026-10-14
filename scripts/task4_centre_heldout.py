#!/usr/bin/env python3
"""Task 4: centre-held-out evaluation for cancer typing (reviewer analysis).

Leave-one-centre-out WITHIN each multi-centre type (CESC, OV, COAD, KIRC,
KIRP): train the headline 20-class cancer-typing model on all centres EXCEPT
the held-out centre, test on the held-out centre's tumours of that type.
Compares signatures_plus_MAF_stack vs standard_sbs96_id83, XGBoost, headline
nested-validation settings (inner stratified 75/25 split, inner-loop
feature-block selection over the same predeclared Bio MAF v4 blocks, 5
XGBoost hyperparameter candidates scored by balanced accuracy on the inner
validation split).

Centre assignment: tumour-sample barcodes from
results/tables/batch_center_sample_mapping.csv (same counting convention as
scripts/run_batch_center_analysis.py: the two dual-aliquot patients
(TCGA-DV-A4W0 KIRC, TCGA-UZ-A9PS KIRP, both centre 10) contribute two
tumour-barcode rows each; features are patient-level so both aliquots share
one feature row and one prediction).

SMALL-CENTRE RULE (fixed before running): centres with fewer than 20
tumours of the type are pooled into a single "small" test fold per type
(train on everything else). Uniform across the five types; no data is
discarded. In this cohort only OV has a sub-20 centre (centre 01, 12
tumours).

Outputs (results/tables/):
  task4_centre_heldout.csv        per-fold + pooled recall per type/representation
  task4_centre_composition.csv    tumour counts per centre for all 20 types
  task4_centre_heldout_predictions.csv  out-of-fold predictions (tumour barcode level)
  task4_notes.md                 method notes (written at the end)

Deviations from the headline pipeline (documented in the notes):
  - tree_method=hist instead of gpu_hist (no GPU on this VM); the pipeline's
    own fallback does gpu_hist->hist anyway.
  - xgb_n_jobs=2 (config: 32; 2-core VM).
  - Precomputed feature matrices instead of rebuilding from the MAF:
      * 948-col MAF stack (results/tables/quick_bio_v4_maf_features.csv.gz)
      * 182-col SBS96+ID83 signatures (downloaded from the companion HF
        dataset bundle; column set verified identical to the frozen model's
        feature_order.txt)
    The headline nested candidate blocks are reconstructed exactly
    (counts 182/303/319/307/1105/1125/1130 verified against the endpoint
    results table).
  - Outer folds are leave-one-centre-out by design; pooled held-out metric
    is recall (test sets are single-class, so balanced accuracy is
    undefined there).
  - Whole held-out centre excluded from training, including other cancer
    types (so some folds train without single-centre types such as LIHC
    when centre 10 is held out, or BRCA/ESCA/SARC/UCEC when centre 09 is
    held out).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from utils.nested_oof import _search_and_predict_feature_set_fold, stable_seed  # noqa: E402

TOP20 = ["BRCA", "UCEC", "LUAD", "LGG", "HNSC", "PRAD", "THCA", "LUSC",
         "SKCM", "STAD", "BLCA", "OV", "COAD", "GBM", "KIRC", "LIHC",
         "CESC", "KIRP", "SARC", "ESCA"]
TASK_TYPES = ["CESC", "OV", "COAD", "KIRC", "KIRP"]
SMALL_CENTRE_MIN = 20
DEFAULT_ASSETS = Path(os.environ.get("TASK4_ASSETS",
                                      str(REPO / "data" / "task4")))  # in-repo copy of task4 assets (2026-10-14)
ASSETS = DEFAULT_ASSETS  # overridden by --data-dir in main()
SIG_BUNDLE = ASSETS / "feats_std.csv.gz"   # 182-col SBS96+ID83, from HF dataset bundle
CDR_XLSX = ASSETS / "cdr.xlsx"
BURDEN = ASSETS / "burden.csv"  # defines the pipeline's matched-feature patient set
MAF_MATRIX = REPO / "results/tables/quick_bio_v4_maf_features.csv.gz"  # 948-col MAF stack
MAPPING = REPO / "results/tables/batch_center_sample_mapping.csv"
DEFAULT_TABLES = REPO / "results/tables"
TABLES = DEFAULT_TABLES  # overridden by --output-dir in main()
OUT_SUMMARY = TABLES / "task4_centre_heldout.csv"
OUT_COMPOSITION = TABLES / "task4_centre_composition.csv"
OUT_PREDS = TABLES / "task4_centre_heldout_predictions.csv"
CKPT_DIR = REPO / "results/logs/task4_centre_heldout_checkpoints"

SETTINGS = {
    "xgb_estimators": 160,
    "xgb_max_depth": 2,
    "xgb_learning_rate": 0.05,
    "xgb_feature_set_random_search_trials": 5,
    "random_search_trials": 10,
    "optuna_trials": 10,
    "tree_method": "hist",          # deviation: no GPU on this VM (headline: gpu_hist)
    "xgb_n_jobs": 1,                # deviation: config 32; 2-core VM, 2 concurrent workers
    "candidate_checkpoint_enabled": True,
    "fold_checkpoint_enabled": True,
    "fold_checkpoint_dir": str(CKPT_DIR),
    "candidate_progress_every": 5,
}
JOB_RESULTS_DIR = CKPT_DIR / "job_results"
JOB_PREDS_DIR = CKPT_DIR / "job_predictions"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Task 4: centre-held-out evaluation for cancer typing")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_ASSETS,
                   help="directory with feats_std.csv.gz, cdr.xlsx and burden.csv "
                        "(overrides TASK4_ASSETS; default: %(default)s)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_TABLES,
                   help="directory for the task4 CSV outputs (default: %(default)s)")
    return p.parse_args(argv)


def configure_paths(args):
    """Point the module-level input/output paths at the CLI-chosen dirs."""
    global ASSETS, SIG_BUNDLE, CDR_XLSX, BURDEN, TABLES
    global OUT_SUMMARY, OUT_COMPOSITION, OUT_PREDS
    ASSETS = args.data_dir
    SIG_BUNDLE = ASSETS / "feats_std.csv.gz"
    CDR_XLSX = ASSETS / "cdr.xlsx"
    BURDEN = ASSETS / "burden.csv"
    TABLES = args.output_dir
    TABLES.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY = TABLES / "task4_centre_heldout.csv"
    OUT_COMPOSITION = TABLES / "task4_centre_composition.csv"
    OUT_PREDS = TABLES / "task4_centre_heldout_predictions.csv"


def load_cohort():
    """Exact headline cohort: CDR top-20 labels restricted to patients with
    matched MC3 feature rows (n=8800)."""
    cdr = pd.read_excel(CDR_XLSX, usecols=["bcr_patient_barcode", "type"])
    cdr["patient"] = cdr["bcr_patient_barcode"].astype(str).str[:12]
    labels = cdr.drop_duplicates("patient").set_index("patient")["type"].astype(str)
    feat_patients = pd.read_csv(BURDEN, usecols=[0], index_col=0).index.astype(str)
    labels = labels.reindex(feat_patients).dropna()
    labels = labels[labels.isin(TOP20)]
    assert len(labels) == 8800, f"cohort n={len(labels)}, expected 8800"
    assert "TCGA-36-2539" not in labels.index
    return labels


def load_features(patients, frozen_block=False):
    sig = pd.read_csv(SIG_BUNDLE, index_col=0)
    sig.index = sig.index.astype(str)
    maf = pd.read_csv(MAF_MATRIX, index_col=0)
    maf.index = maf.index.astype(str)
    assert sig.shape[1] == 182 and maf.shape[1] == 948
    # headline block reconstruction: the current code's v4 blocks are
    # 121/137/125/885/905/910; the headline run's blocks were
    # 121/137/125/923/943/948. The 38 extra
    # driver_gene_role_matched_event_log_count__* columns in the 948-col
    # matrix (genes whose OncoKB role became INSUFFICIENT_EVIDENCE after
    # the headline run) belong to the exact-driver-genes family, so they
    # are added to the three exact-driver-containing blocks.
    from utils.quick_bio_v4_nested_feature_selection import v4_feature_columns
    _core, _all, code_blocks = v4_feature_columns()
    maf_cols = set(maf.columns)
    for _name, _cols in code_blocks.items():
        assert set(_cols) <= maf_cols, _name
    extra = [c for c in maf.columns
             if c.startswith("driver_gene_role_matched_event_log_count__")
             and c not in set(code_blocks["v4_compact_plus_exact_driver_genes"])]
    assert len(extra) == 38, len(extra)
    blocks = {
        "v4_compact_biology_only": code_blocks["v4_compact_biology_only"],
        "v4_compact_plus_driver_evidence_tiers":
            code_blocks["v4_compact_plus_driver_evidence_tiers"],
        "v4_compact_plus_hotspot_summary":
            code_blocks["v4_compact_plus_hotspot_summary"],
        "v4_compact_plus_exact_driver_genes":
            code_blocks["v4_compact_plus_exact_driver_genes"] + extra,
        "v4_full_core": code_blocks["v4_full_core"] + extra,
        "v4_full_core_plus_optional_controls":
            code_blocks["v4_full_core_plus_optional_controls"] + extra,
    }
    sig_pref = sig.add_prefix("signature_matrix__")
    combined = pd.concat([sig_pref, maf], axis=1)
    cand = {"signatures_only": list(sig_pref.columns)}
    # frozen-block design (see notes): only the 1130-col block, the superset
    # of every block the headline run selected for this endpoint and the
    # winner of the one faithful Task 4 job completed.
    block_names = (["v4_full_core_plus_optional_controls"] if frozen_block
                   else list(blocks))
    for name in block_names:
        cand[f"signatures_plus_{name}"] = (list(sig_pref.columns)
                                           + blocks[name])
    counts = {k: len(v) for k, v in cand.items()}
    expected = {"signatures_only": 182,
                "signatures_plus_v4_full_core_plus_optional_controls": 1130}
    if not frozen_block:
        expected.update({
            "signatures_plus_v4_compact_biology_only": 303,
            "signatures_plus_v4_compact_plus_driver_evidence_tiers": 319,
            "signatures_plus_v4_compact_plus_hotspot_summary": 307,
            "signatures_plus_v4_compact_plus_exact_driver_genes": 1105,
            "signatures_plus_v4_full_core": 1125,
        })
    assert counts == expected, counts
    pats = pd.Index([str(p) for p in patients])
    sig = sig.loc[pats].fillna(0.0).to_numpy(dtype=np.float32)
    all_cols = list(combined.columns)
    pos = {c: i for i, c in enumerate(all_cols)}
    combined = combined.loc[pats].fillna(0.0).to_numpy(dtype=np.float32)
    feature_sets = {
        "standard_sbs96_id83": {"patterns_only": sig},
        "signatures_plus_MAF_stack": {
            k: combined[:, [pos[c] for c in v]] for k, v in cand.items()
        },
    }
    return feature_sets


def load_centre_info(patients):
    """Tumour-barcode -> centre mapping; returns patient->centres and
    tumour-barcode frame for the cohort."""
    m = pd.read_csv(MAPPING, dtype=str)
    pats = set(str(p) for p in patients)
    m = m[m["sample"].isin(pats)].copy()
    missing = pats - set(m["sample"].unique())
    print(f"[task4] cohort patients without centre mapping: {len(missing)}")
    # patient -> sorted unique centres (dual-aliquot patients: both at centre 10)
    pat_centres = m.groupby("sample")["center"].apply(lambda s: sorted(s.unique()))
    multi = pat_centres[pat_centres.str.len() > 1]
    print(f"[task4] patients with tumours at >1 centre: {len(multi)}")
    return m, pat_centres


def define_folds(m, labels):
    """Per type: held-out groups = centres with >=20 tumours, plus one pooled
    'small' group for the rest. Returns dict type -> list of
    (group_label, set_of_centres)."""
    folds = {}
    for t in TASK_TYPES:
        dd = m[m["cancer_type_cdr"] == t]
        cc = dd["center"].value_counts()
        big = sorted(c for c in cc.index if cc[c] >= SMALL_CENTRE_MIN)
        small = sorted(c for c in cc.index if cc[c] < SMALL_CENTRE_MIN)
        groups = [(c, {c}) for c in big]
        if small:
            groups.append(("small(" + ",".join(small) + ")", set(small)))
        folds[t] = groups
        print(f"[task4] {t}: " + "; ".join(
            f"{g} n={dd[dd['center'].isin(cs)].shape[0]}" for g, cs in groups))
    return folds


def _job_id(t, group_label, rep):
    safe = "".join(c if c.isalnum() else "_" for c in group_label)
    return f"{t}__{safe}__{rep}"


def run_jobs():
    """Run (a slice of) the fold x representation jobs. Each job writes a
    result JSON + predictions CSV; completed jobs are skipped (resume-safe).
    Slicing via env: TASK4_SLICE='i/n' runs jobs[i::n]; TASK4_MAX_JOBS=k
    caps the number of jobs (for timing probes)."""
    t0 = time.time()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    JOB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    JOB_PREDS_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_cohort()
    patients = labels.index
    frozen = os.environ.get("TASK4_FROZEN_BLOCK") == "1"
    feature_sets = load_features(patients, frozen_block=frozen)
    print(f"[task4] frozen_block={frozen}", flush=True)
    m, pat_centres = load_centre_info(patients)
    folds = define_folds(m, labels)

    y = labels.astype(str).to_numpy()
    classes = np.array(sorted(set(y)))
    n_classes = len(classes)
    class_id = {c: i for i, c in enumerate(classes)}
    y_enc = np.array([class_id[c] for c in y])
    pat_pos = {p: i for i, p in enumerate(patients)}
    pat_centre1 = {p: (cs[0] if len(cs) == 1 else "MULTI:" + ",".join(cs))
                   for p, cs in pat_centres.items()}

    jobs = []
    for t in TASK_TYPES:
        t_pats = [p for p in patients if labels[p] == t]
        for group_label, centres in folds[t]:
            for rep in ["signatures_plus_MAF_stack", "standard_sbs96_id83"]:
                jobs.append((t, group_label, centres, t_pats, rep))

    sl = os.environ.get("TASK4_SLICE", "0/1")
    si, sn = (int(x) for x in sl.split("/"))
    my_jobs = jobs[si::sn]
    max_jobs = os.environ.get("TASK4_MAX_JOBS")
    if max_jobs:
        my_jobs = my_jobs[: int(max_jobs)]
    print(f"[task4] slice {sl}: {len(my_jobs)} jobs of {len(jobs)} total",
          flush=True)

    for ji, (t, group_label, centres, t_pats, rep) in enumerate(my_jobs):
        jid = _job_id(t, group_label, rep)
        res_path = JOB_RESULTS_DIR / f"{jid}.json"
        if res_path.exists():
            print(f"[task4] [{ji+1}/{len(my_jobs)}] {jid}: already done, skip",
                  flush=True)
            continue
        test_pats = [p for p in t_pats if pat_centre1.get(p) in centres]
        train_pats = [p for p in patients if pat_centre1.get(p) not in centres]
        train_idx = np.array([pat_pos[p] for p in train_pats], dtype=int)
        test_idx = np.array([pat_pos[p] for p in test_pats], dtype=int)
        n_train_classes = len(set(y_enc[train_idx]))
        print(f"[task4] [{ji+1}/{len(my_jobs)}] {jid}: "
              f"train_n={len(train_idx)} train_classes={n_train_classes} "
              f"test_patients={len(test_idx)}", flush=True)
        seed = int(stable_seed("task4_centre_heldout", t, group_label, rep))
        settings = dict(SETTINGS)
        ckpt_tag = "__frozen1130" if (frozen and rep == "signatures_plus_MAF_stack") else ""
        settings["fold_checkpoint_key"] = f"task4_centre_heldout__{jid}{ckpt_tag}"
        settings["fold_progress_label"] = f"task4 {t}/{group_label}/{rep}"
        jt0 = time.time()
        pred_test, diag = _search_and_predict_feature_set_fold(
            task="multiclass", learner="xgboost",
            x_by_feature_set=feature_sets[rep],
            y=y_enc, train_idx=train_idx, test_idx=test_idx,
            groups=None, seed=seed, n_classes=n_classes,
            primary_metric="balanced_accuracy", settings=settings)
        pred_label = classes[np.argmax(pred_test, axis=1)]
        true_label = np.array([labels[p] for p in test_pats])
        pred_rows = []
        for p, pl, tl, proba in zip(test_pats, pred_label, true_label,
                                    pred_test):
            barcodes = m.loc[m["sample"] == p, "mc3_tumor_barcode"].tolist()
            for bc in barcodes:
                pred_rows.append({
                    "tumour_barcode": bc,
                    "true_label": tl,
                    "predicted_label": pl,
                    "pred_score": float(np.max(proba)),
                    "type": t,
                    "heldout_centre": group_label,
                    "representation": rep,
                })
        n_corr = int(np.sum(pred_label == true_label))
        pd.DataFrame(pred_rows).to_csv(JOB_PREDS_DIR / f"{jid}.csv",
                                       index=False)
        result = {
            "job_id": jid, "type": t, "representation": rep,
            "heldout_centre": group_label,
            "heldout_centres": sorted(centres),
            "design": ("frozen_1130_block" if (frozen and rep == "signatures_plus_MAF_stack")
                       else "faithful_nested_block_search"),
            "n_test_patients": len(test_idx),
            "n_test_tumours": len(pred_rows),
            "n_correct_patients": n_corr,
            "recall_patients": n_corr / len(test_idx) if len(test_idx) else None,
            "train_n": len(train_idx),
            "train_classes": n_train_classes,
            "selected_feature_set": diag["selected_feature_set"],
            "selected_feature_count": diag["selected_feature_count"],
            "selected_params": diag["selected_params"],
            "selected_m_star": diag["selected_m_star"],
            "inner_val_balanced_accuracy": float(diag["selected_score"]),
            "inner_train_n": diag["inner_train_n"],
            "inner_val_n": diag["inner_val_n"],
            "seconds": round(time.time() - jt0, 1),
            "seed": seed,
        }
        with open(res_path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[task4] [{ji+1}/{len(my_jobs)}] {jid}: done in "
              f"{result['seconds']}s recall={result['recall_patients']:.4f} "
              f"selected={diag['selected_feature_set']} "
              f"({diag['selected_feature_count']})", flush=True)
    print(f"[task4] slice {sl} finished in {time.time()-t0:.0f}s", flush=True)


def finalize():
    """Assemble the three deliverable CSVs from per-job outputs."""
    import glob
    res_files = sorted(glob.glob(str(JOB_RESULTS_DIR / "*.json")))
    assert res_files, "no job results found"
    print(f"[task4] finalizing from {len(res_files)} job results", flush=True)
    rows, all_preds = [], []
    for fp in res_files:
        with open(fp) as fh:
            r = json.load(fh)
        jid = r["job_id"]
        preds = pd.read_csv(JOB_PREDS_DIR / f"{jid}.csv")
        # restore the exact held-out group label ("08" not 8): read_csv
        # infers ints for pure-digit labels
        preds["heldout_centre"] = r["heldout_centre"]
        all_preds.append(preds)
        n_corr_t = int((preds["predicted_label"] == preds["true_label"]).sum())
        design = r.get("design", "") or ""
        if r["representation"] == "standard_sbs96_id83":
            # patterns-alone has a single 182-col feature set; there was never
            # a block search, regardless of the frozen-block flag.
            design = "single_feature_set"
        rows.append({
            "type": r["type"], "representation": r["representation"],
            "heldout_centre": r["heldout_centre"],
            "design": design,
            "n_test_patients": r["n_test_patients"],
            "n_test_tumours": len(preds),
            "n_correct": n_corr_t,
            "recall": n_corr_t / len(preds) if len(preds) else float("nan"),
            "train_n": r["train_n"],
            "train_classes": r["train_classes"],
            "selected_feature_set": r["selected_feature_set"],
            "selected_feature_count": r["selected_feature_count"],
            "inner_val_balanced_accuracy": round(r["inner_val_balanced_accuracy"], 6),
            "seconds": r["seconds"],
        })
    summ = pd.DataFrame(rows)
    pool = []
    for (t, rep), g in summ.groupby(["type", "representation"]):
        n = int(g["n_test_tumours"].sum())
        c = int(g["n_correct"].sum())
        pool.append({"type": t, "representation": rep,
                     "heldout_centre": "pooled",
                     "design": ";".join(sorted(set(g["design"]))),
                     "n_test_patients": int(g["n_test_patients"].sum()),
                     "n_test_tumours": n, "n_correct": c,
                     "recall": c / n if n else float("nan"),
                     "train_n": "", "train_classes": "",
                     "selected_feature_set": ";".join(g["selected_feature_set"]),
                     "selected_feature_count": "",
                     "inner_val_balanced_accuracy": "",
                     "seconds": int(g["seconds"].sum())})
    summ = pd.concat([summ, pd.DataFrame(pool)], ignore_index=True)
    summ = summ.sort_values(["type", "representation", "heldout_centre"])
    summ.to_csv(OUT_SUMMARY, index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(OUT_PREDS, index=False)
    print(f"[task4] wrote {OUT_SUMMARY} ({len(summ)} rows) and {OUT_PREDS}",
          flush=True)
    labels = load_cohort()
    m, _ = load_centre_info(labels.index)
    comp = pd.crosstab(m["cancer_type_cdr"], m["center"])
    comp = comp.reindex(TOP20).fillna(0).astype(int)
    comp["n"] = comp.sum(axis=1)
    comp.index.name = "cancer_type"
    comp.to_csv(OUT_COMPOSITION)
    print(f"[task4] wrote {OUT_COMPOSITION}", flush=True)


def main():
    configure_paths(parse_args())
    if os.environ.get("TASK4_FINALIZE") == "1":
        finalize()
    else:
        run_jobs()

if __name__ == "__main__":
    main()
