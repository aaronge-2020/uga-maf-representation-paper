#!/usr/bin/env python
"""Default verification for the reproducibility package. Fits no models.

Sections:
  1. artifact hashes (SHA-256 pins)
  2. feature rebuild in a shadow tree + TF-IDF smoke check
  3. canonical splits (structure + regeneration from the main OOF file)
  4. extension grid scores + A8 contrasts, recomputed from saved predictions
  5. trial-selection audit (runner's own selection rule vs saved FINAL rows)
  6. manuscript numbers recomputed from saved repo artifacts
  7. KNOWN DISCREPANCIES (expected findings; never counted as failures)
Exit code is nonzero iff any check in sections 1-6 FAILs.
"""
import ast
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repro_common as rc

NS = rc.NS
ROOT = rc.ROOT
COUNTS = {"pass": 0, "fail": 0}
MAX_CI_DEV = [0.0]


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f" | {detail}" if detail else ""), flush=True)
    COUNTS["pass" if ok else "fail"] += 1
    return ok


def section(title):
    print(f"\n=== {title} ===", flush=True)


def stream_digest(path, chunksize=1000):
    """SHA-256 over a feature CSV's parsed values, streamed in row chunks so
    the 8,800 x 19,672 all-genes matrices never sit in memory whole."""
    import hashlib
    h = hashlib.sha256()
    shape = [0, 0]
    for chunk in pd.read_csv(path, index_col=0, chunksize=chunksize):
        if shape[1] == 0:
            h.update("\x00".join(map(str, chunk.columns)).encode())
            shape[1] = chunk.shape[1]
        h.update("\x00".join(map(str, chunk.index)).encode())
        h.update(np.ascontiguousarray(chunk.values).tobytes())
        shape[0] += chunk.shape[0]
    return h.hexdigest(), tuple(shape)


LEDGER = {}
with open(NS / "results_ledger.csv", newline="") as f:
    for r in csv.DictReader(f):
        LEDGER[r["claim_id"]] = r


def tol_for(stored):
    """Half a unit of the last printed digit of the stored value string."""
    s = str(stored).strip().lstrip("+")
    if "e" in s or "E" in s:
        return None  # relative comparison
    if "." in s:
        return 0.5 * 10 ** (-len(s.split(".")[1])) + 1e-12
    return 0.5 + 1e-12


def compare(cid, recomputed):
    row = LEDGER[cid]
    stored = row["value"]
    if stored.startswith("<"):
        ok = recomputed < float(stored[1:])
        return check(f"ledger {cid} = {stored}", ok, f"recomputed {recomputed}")
    tol = tol_for(stored)
    if tol is None:
        ok = abs(recomputed / float(stored) - 1) < 0.01
        return check(f"ledger {cid} = {stored}", ok, f"recomputed {recomputed:.3e}")
    ok = abs(recomputed - float(stored)) <= tol
    return check(f"ledger {cid} = {stored}", ok, f"recomputed {recomputed:.6f}")


# ------------------------------------------------------------------ hashes
def sec_hashes():
    section("1. Artifact hashes (SHA-256)")
    with open(NS / "metrics" / "artifact_hashes.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    bad = 0
    for r in rows:
        p = ROOT / r["path"]
        ok = p.exists() and rc.sha256_file(p) == r["sha256"] and p.stat().st_size == int(r["bytes"])
        if not ok:
            bad += 1
            check(f"hash {r['path']}", False, "missing or hash/size mismatch")
    check(f"hashes: {len(rows)} pinned artifacts verified", bad == 0,
          f"{bad} mismatches")


# ------------------------------------------------------------------ features
def sec_features_and_splits():
    section("2. Feature rebuild (shadow tree) and canonical splits")
    oof = rc.T / "main_manuscript_complete_panel_oof_predictions.csv"
    head = oof.read_bytes()[:200]
    if b"oid sha256" in head:
        check("git-lfs smudge for main OOF file", False,
              "file is an LFS pointer; install git-lfs and run `git lfs pull`")
        return
    check("git-lfs smudge for main OOF file", True, f"{oof.stat().st_size} bytes")
    tmp = Path(tempfile.mkdtemp(prefix="repro_shadow_"))
    try:
        shadow = tmp / "repo"
        ext = shadow / "analyses" / "representation_extension"
        (ext / "scripts").mkdir(parents=True)
        (ext / "config").mkdir(parents=True)
        (ext / "features").mkdir(parents=True)
        for name in ["build_gene_features.py", "make_canonical_splits.py"]:
            shutil.copy(NS / "scripts" / name, ext / "scripts" / name)
        shutil.copy(NS / "config" / "panel_genes_401.csv", ext / "config" / "panel_genes_401.csv")
        os.symlink(ROOT / "results", shadow / "results")
        os.symlink(ROOT / "data", shadow / "data")
        env = dict(os.environ)
        r = subprocess.run([sys.executable, "scripts/build_gene_features.py"],
                           cwd=ext, env=env, capture_output=True, text=True)
        if r.returncode == 0:
            detail = (r.stdout.strip().splitlines() or [""])[-1]
        else:
            tail = (r.stderr.strip().splitlines() or ["<no stderr>"])[-1]
            detail = f"exit {r.returncode}: {tail}"
        check("feature builder runs in shadow tree", r.returncode == 0, detail)
        if r.returncode != 0:
            return
        man = pd.read_csv(ext / "config" / "feature_build_manifest.csv").set_index("task")
        committed = pd.read_csv(NS / "config" / "feature_build_manifest.csv").set_index("task")
        for task in ["cancer", "hrd"]:
            check(f"events sha256 matches manifest ({task})",
                  man.loc[task, "events_sha256"] == committed.loc[task, "events_sha256"])
        for name in sorted(p.name for p in (NS / "features").glob("*.csv.gz")):
            a, b = ext / "features" / name, NS / "features" / name
            if not a.exists():
                check(f"rebuilt feature matrix identical: {name}", False, "not rebuilt")
                continue
            da, db = stream_digest(a), stream_digest(b)
            check(f"rebuilt feature matrix identical: {name}", da == db,
                  "chunked value digest over rows in file order")
        # TF-IDF is a run-time transform of the R3 panel matrix; smoke check
        # against metrics/integration_check.json.
        sys.path.insert(0, str(NS / "scripts"))
        import runner
        split = pd.read_csv(NS / "config" / "canonical_split_cancer.csv",
                            dtype={"patient": str})
        patients = list(split["patient"])
        counts = runner.load_matrix("cancer", "r3_counts_panel").reindex(patients)
        tr_idx = np.where(split["fold"].values != 1)[0]
        _, vocab = runner.tfidf(counts, tr_idx)
        integ = json.load(open(NS / "metrics" / "integration_check.json"))
        check("TF-IDF vocabulary genes (fold-1 train)",
              len(vocab) == integ["tfidf_vocab_genes"],
              f"vocab {len(vocab)}")
        # canonical splits: structure
        for task in ["cancer", "hrd"]:
            sp = pd.read_csv(NS / "config" / f"canonical_split_{task}.csv",
                             dtype={"patient": str})
            sizes = sp.groupby("fold")["patient"].count()
            check(f"canonical split {task}: every patient in exactly one fold",
                  sp["patient"].is_unique and set(sp["fold"].unique()) == {1, 2, 3, 4, 5},
                  f"n={len(sp)}, fold sizes {sorted(sizes.tolist())}")
        r2 = subprocess.run([sys.executable, "scripts/make_canonical_splits.py"],
                            cwd=ext, env=env, capture_output=True, text=True)
        check("canonical split regeneration runs", r2.returncode == 0,
              (r2.stderr.strip().splitlines() or ["<no stderr>"])[-1] if r2.returncode else "")
        if r2.returncode == 0:
            for task in ["cancer", "hrd"]:
                a = ext / "config" / f"canonical_split_{task}.csv"
                b = NS / "config" / f"canonical_split_{task}.csv"
                check(f"regenerated canonical split identical ({task})",
                      rc.sha256_file(a) == rc.sha256_file(b))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ extension
def sec_extension():
    section("3. Extension grid scores recomputed from saved predictions")
    cache = {}
    for cid, row in LEDGER.items():
        if not cid.startswith("EXT-CELL-"):
            continue
        parts = cid.split("-")
        cell = "-".join(parts[2:-1])
        task, rep, learner = cell.split("__")
        if cell not in cache:
            cache[cell] = rc.cell_scores(task, rep, learner)
        folds, primary = cache[cell]
        if parts[-1] == "PRIMARY":
            recomputed = primary
        else:
            recomputed = folds[int(parts[-1].replace("FOLD", ""))]
        ok = round(recomputed, 4) == float(row["value"])
        check(f"ledger {cid} = {row['value']}", ok, f"recomputed {recomputed:.6f}")
    section("4. A8 primary contrasts recomputed (bootstrap seed 20261025)")
    a8 = rc.a8_contrasts()
    for cid, row in LEDGER.items():
        if not cid.startswith("EXT-A8-"):
            continue
        rest = cid[len("EXT-A8-"):]
        kind = rest.rsplit("-", 1)[1]
        body = rest.rsplit("-", 1)[0]
        task, learner = body.split("-", 2)[0], body.split("-", 2)[1]
        rep = body.split("-", 2)[2]
        entry = a8[f"{task}/{learner}"]
        stored = float(row["value"])
        if kind == "PRIMARY":
            recomputed = entry["primary"][rep]
            ok = round(recomputed, 4) == stored
            check(f"ledger {cid} = {row['value']}", ok, f"recomputed {recomputed:.6f}")
        elif kind == "DIFF":
            recomputed = entry["contrasts"][rep]["mean_per_fold_diff"]
            ok = round(recomputed, 4) == stored
            check(f"ledger {cid} = {row['value']}", ok, f"recomputed {recomputed:+.6f}")
        else:
            lo_hi = entry["contrasts"][rep]["bootstrap_ci95"]
            recomputed = lo_hi[0] if kind == "CILO" else lo_hi[1]
            dev = abs(recomputed - stored)
            MAX_CI_DEV[0] = max(MAX_CI_DEV[0], dev)
            check(f"ledger {cid} = {row['value']} (CI tol +/-0.002)", dev <= 0.002,
                  f"recomputed {recomputed:+.6f}, deviation {dev:.6f}")
    sc = rc.sbs3_brca_counts()
    compare("EXT-SBS3-NONZERO", float(sc["sbs3_nonzero"]))
    compare("EXT-SBS3-MEAN", sc["sbs3_mean"])
    compare("EXT-SBS3-MAX", sc["sbs3_max"])
    compare("EXT-BRCA-POS772", float(sc["brca_positive_772"]))
    compare("EXT-BRCA-POSALL", float(sc["brca_positive_all"]))


# ------------------------------------------------------------------ trials
def sec_trials():
    section("5. Trial-selection audit (runner selection rule vs saved FINAL rows)")
    sys.path.insert(0, str(NS / "scripts"))
    import runner
    for task, rep, learner in rc.list_cells():
        cell = rc.cell_name(task, rep, learner)
        t = pd.read_csv(NS / "metrics" / "trials" / f"{cell}.csv", dtype={"trial": str})
        n_trials = int((t["trial"] != "FINAL").groupby(t["fold"]).sum().iloc[0])
        ok_all, detail = True, ""
        for fold in sorted(t["fold"].unique()):
            sub = t[t["fold"] == fold]
            records, configs = [], [None] * n_trials
            for _, r in sub.iterrows():
                cfg = ast.literal_eval(r["config"])
                records.append({"cell": cell, "fold": int(fold), "trial": r["trial"],
                                "config": r["config"], "status": r["status"],
                                "inner_score": (None if pd.isna(r["inner_score"])
                                                else float(r["inner_score"]))})
                if r["trial"] != "FINAL":
                    configs[int(r["trial"])] = cfg
            ok = runner.trial_file_matches(records, cell, int(fold), n_trials, configs)
            if not ok:
                ok_all, detail = False, f"fold {fold} FINAL does not match selection rule"
                break
        check(f"trial selection {cell}", ok_all, detail)


# ------------------------------------------------------------------ manuscript
def sec_manuscript():
    section("6. Manuscript numbers recomputed from saved artifacts")
    V = {}
    t1 = rc.table1_fold_means()
    for (ep, rep, learner), (mean, folds) in t1.items():
        V[f"MS-T1-{ep}-{rep}-{learner}"] = mean
        if rep == "signatures_plus_MAF_stack" and learner == "xgboost" and ep != "OS":
            lo, hi = rc.t_interval(folds)
            V[f"MS-T1CI-{ep}-LO"], V[f"MS-T1CI-{ep}-HI"] = lo, hi
    sv = rc.survival_fold_means()
    for (ep, rep, learner), mean in sv.items():
        if ep == "PFI":
            V[f"MS-PFI-{rep}-cox_ph"] = mean
    mc, mh, pw = rc.muat_cancer(), rc.muat_hrd(), rc.paired_vs_muat()
    V["MS-MUAT-CANCER-BA"] = np.mean(list(mc["fold_ba"].values()))
    for k, v in mc["fold_ba"].items():
        V[f"MS-MUAT-CANCER-FOLD{k}"] = v
    V["MS-MUAT-CANCER-ACC"] = mc["accuracy"]
    V["MS-MUAT-CANCER-TOP5"] = mc["top5"]
    V["MS-MUAT-CANCER-BACI-LO"], V["MS-MUAT-CANCER-BACI-HI"] = mc["ba_ci"]
    V["MS-MUAT-HRD-SPEARMAN"] = mh["HRD_Score"]["mean"]
    V["MS-MUAT-HRD-CI-LO"], V["MS-MUAT-HRD-CI-HI"] = mh["HRD_Score"]["ci"]
    V["MS-MUAT-HRD33-AUROC"] = mh["hrd_binary_33"]["mean"]
    for k, v in mh["hrd_binary_33"]["folds"].items():
        V[f"MS-MUAT-HRD33-FOLD{k}"] = v
    V["MS-MUAT-HRD33-CI-LO"], V["MS-MUAT-HRD33-CI-HI"] = mh["hrd_binary_33"]["ci"]
    V["MS-PAIR-HRD-DIFF"] = pw["hrd_spearman"]["mean"]
    V["MS-PAIR-HRD-P"] = pw["hrd_spearman"]["p"]
    V["MS-PAIR-HRD-CI-LO"], V["MS-PAIR-HRD-CI-HI"] = pw["hrd_spearman"]["ci_lo"], pw["hrd_spearman"]["ci_hi"]
    V["MS-PAIR-HRD33-DIFF"] = pw["hrd33_auroc"]["mean"]
    V["MS-PAIR-HRD33-P"] = pw["hrd33_auroc"]["p"]
    V["MS-PAIR-HRD33-CI-LO"], V["MS-PAIR-HRD33-CI-HI"] = pw["hrd33_auroc"]["ci_lo"], pw["hrd33_auroc"]["ci_hi"]
    V["MS-PAIR-CANCER-DIFF"] = pw["cancer_ba"]["mean"]
    V["MS-PAIR-CANCER-CI-LO"], V["MS-PAIR-CANCER-CI-HI"] = pw["cancer_ba"]["ci_lo"], pw["cancer_ba"]["ci_hi"]
    V["MS-PAIR-CANCER-P"] = pw["cancer_ba"]["p"]
    abl = rc.ablations()
    for (rep, var), v in abl.items():
        V[f"MS-ABL-{rep}-{var}"] = v
    V["MS-ABL-KNOWLEDGE-DELTA"] = abs(abl[("sigmaf", "b_no_knowledge")] - abl[("sigmaf", "a_full")])
    pc = rc.pcawg()
    V["MS-PCAWG-ACC"], V["MS-PCAWG-BA"] = pc["accuracy"], pc["balanced_accuracy"]
    for typ, v in pc["recalls"].items():
        V[f"MS-PCAWG-RECALL-{typ}"] = v
    sp = pd.read_csv(ROOT / "results/external_validation/aaron_email/pcawg/pcawg_frozen_model_predictions.csv")
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    V["MS-PCAWG-SPECTRA-ACC"] = accuracy_score(sp["true_label"], sp["pred_label"])
    V["MS-PCAWG-SPECTRA-BA"] = balanced_accuracy_score(sp["true_label"], sp["pred_label"])
    for typ in ["BRCA", "KIRC"]:
        sub = sp[sp["true_label"] == typ]
        V[f"MS-PCAWG-SPECTRA-RECALL-{typ}"] = float((sub["pred_label"] == typ).mean())
    pg = rc.pog570()
    V["MS-POG-SPEARMAN"], V["MS-POG-PEARSON"] = pg["spearman"], pg["pearson"]
    V["MS-POG-R2"], V["MS-POG-SPEC24"] = pg["r2"], pg["specificity24"]
    centre = rc.centre_heldout()
    for (typ, rep), d in centre.items():
        V[f"MS-CENTRE-{typ}-{rep}"] = d["pooled_tumours"]
    svv = rc.survival_covariate()
    for rep, v in svv["scores"].items():
        V[f"MS-SURVCOV-{rep}"] = v
    V["MS-SURVCOV-GAIN"] = svv["gain"]
    V["MS-SURVCOV-P"] = svv["p"]
    V["MS-SURVCOV-CI-LO"], V["MS-SURVCOV-CI-HI"] = svv["ci"]
    hc = rc.hrd_controls()
    FULL = "signatures_plus_MAF_stack__full"
    NOG = "signatures_plus_MAF_stack__no_hrd_genes"
    NOV = "signatures_plus_MAF_stack__no_vaf"
    V["MS-HRDCTL-FULL-SCORE"] = hc["scores"][("HRD_Score", FULL)]
    V["MS-HRDCTL-NOGENES-SCORE"] = hc["scores"][("HRD_Score", NOG)]
    V["MS-HRDCTL-NOGENES-P"] = hc["comparisons"][("HRD_Score", "no_hrd_genes")]["p"]
    V["MS-HRDCTL-FULL-33"] = hc["scores"][("hrd_binary_33", FULL)]
    V["MS-HRDCTL-NOGENES-33"] = hc["scores"][("hrd_binary_33", NOG)]
    V["MS-HRDCTL-NOGENES-P33"] = hc["comparisons"][("hrd_binary_33", "no_hrd_genes")]["p"]
    V["MS-HRDCTL-NOVAF-33"] = hc["scores"][("hrd_binary_33", NOV)]
    V["MS-HRDCTL-NOVAF-P33"] = hc["comparisons"][("hrd_binary_33", "no_vaf")]["p"]
    t3 = rc.task3_baselines()
    for cid, key in [("MS-T3-SIG3-SPEARMAN", ("sig3_only", "HRD_Score", "spearman")),
                     ("MS-T3-SIG3-AUROC33", ("sig3_only", "hrd_binary_33", "auroc")),
                     ("MS-T3-SIG3-AUROC42", ("sig3_only", "hrd_binary_42", "auroc")),
                     ("MS-T3-BRCA-SPEARMAN", ("brca12_only", "HRD_Score", "spearman")),
                     ("MS-T3-BRCA-AUROC33", ("brca12_only", "hrd_binary_33", "auroc")),
                     ("MS-T3-BRCA-AUROC42", ("brca12_only", "hrd_binary_42", "auroc"))]:
        V[cid] = t3[key]
    g3 = rc.fig3_gains()
    for k, d in g3.items():
        V[f"MS-FIG3-{d['label']}-GAIN"] = d["gain"]
        V[f"MS-FIG3-{d['label']}-N"] = float(d["n"])
    cancer_shap = rc.T / "feature_selection_cancer_type_top20_shap_values.csv"
    text_sh = rc.shap_text_shares(cancer_shap)
    for name, v in text_sh.items():
        V[f"MS-FIG6-TEXT-{name}"] = v
    for name, v in rc.shap_shares(cancer_shap).items():
        V[f"MS-FIG6-FIG-{name}"] = v
    for name, v in rc.shap_shares(rc.T / "feature_selection_OS_shap_values.csv").items():
        V[f"MS-FIG6OS-{name}"] = v
    V["MS-FIG6-MAXFEATURE"] = rc.shap_max_feature_share(cancer_shap)
    kuc = rc.kucab_pooled()
    V["MS-KUCAB-SPECTRA-LINEAR"] = kuc[("standard_sbs96_dbs78_id83", "linear")]
    V["MS-KUCAB-SPECTRA-XGB"] = kuc[("standard_sbs96_dbs78_id83", "xgboost")]
    V["MS-KUCAB-EVENTSTACK-XGB"] = kuc[("MAF_stack_only", "xgboost")]
    ep = rc.endpoint_results()
    def pooled(ep_id, rep, learner):
        r = ep[(ep["endpoint"] == ep_id) & (ep["representation"] == rep)
               & (ep["learner"] == learner)].iloc[0]
        return float(r["score"])
    V["MS-TEXT-SIGMAF-VS-MAF-DIFF"] = (pooled("cancer_type_top20", "signatures_plus_MAF_stack", "xgboost")
                                       - pooled("cancer_type_top20", "MAF_stack_only", "xgboost"))
    V["MS-TEXT-HRD33-GAIN"] = (t1[("hrd_binary_33", "signatures_plus_MAF_stack", "xgboost")][0]
                               - t1[("hrd_binary_33", "standard_sbs96_id83", "xgboost")][0])
    V["MS-TEXT-HRD-GAIN"] = (t1[("HRD_Score", "signatures_plus_MAF_stack", "xgboost")][0]
                             - t1[("HRD_Score", "standard_sbs96_id83", "xgboost")][0])
    missing = []
    for cid, row in LEDGER.items():
        if not cid.startswith("MS-"):
            continue
        if row["status"] == "reproduced":
            if cid in V:
                compare(cid, V[cid])
            else:
                missing.append(cid)
        elif row["status"] == "sourced_not_rerun":
            srcs = [s.strip() for s in row["source_file"].split(";")]
            ok = all((ROOT / s).exists() for s in srcs)
            check(f"source present for {cid}", ok, row["source_file"])
    check("every reproduced manuscript row has a recomputation", not missing,
          f"missing: {missing}" if missing else f"{sum(1 for c in V if c in LEDGER)} rows")


# ------------------------------------------------------------------ discrepancies
def sec_discrepancies():
    section("7. KNOWN DISCREPANCIES (expected findings, not failures)")
    scan = rc.oof_vs_endpoint()
    over = scan[scan["abs_diff"] > 0.002].sort_values("abs_diff", ascending=False)
    print(f"DISCREPANCY oof-vs-endpoint: {len(over)} of {len(scan)} groups differ by > 0.002 "
          f"({int((scan['abs_diff'] > 0.005).sum())} by > 0.005, {int((scan['abs_diff'] > 0.01).sum())} by > 0.01)")
    for _, r in over.head(10).iterrows():
        print(f"  {r['endpoint']} / {r['representation']} / {r['learner']} ({r['metric']}): "
              f"endpoint {r['endpoint_score']:.4f} vs OOF-recomputed {r['oof_score']:.4f} "
              f"(|diff| {r['abs_diff']:.4f})")
    doc = (ROOT / "docs" / "results_summary_2026-10.md").read_text()
    fig = rc.shap_shares(rc.T / "feature_selection_cancer_type_top20_shap_values.csv")
    if all(s in doc for s in ["21.9", "19.9", "15.2"]):
        print(f"DISCREPANCY shap-doc-vs-csv: docs/results_summary_2026-10.md prints VEP consequence 21.9, "
              f"per-gene driver counts 19.9, VEP impact 15.2; the SHAP CSV gives "
              f"{fig['VEP consequence']:.1f}, {fig['Gene-specific features']:.1f}, {fig['VEP impact']:.1f} "
              f"(the CSV matches the manuscript; the summary doc is wrong)")
    stale = json.load(open(rc.T / "cancer_typing_primary_fold_differences.json"))
    print(f"DISCREPANCY stale-fold-differences: cancer_typing_primary_fold_differences.json stores "
          f"mean_difference {stale['mean_difference']:.4f} from a superseded MuAt pairing; "
          f"the faithful pairing recomputes to +0.2094 (MS-PAIR-CANCER-DIFF)")
    man = json.load(open(ROOT / "results/manuscript/plot_data/plot_data_manifest.json"))
    refs = set()
    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str) and "canonical/" in o:
            refs.add(o)
    walk(man)
    missing = [r for r in sorted(refs) if not (ROOT / "results/manuscript/plot_data" / r).exists()]
    print(f"DISCREPANCY plot-data-manifest: plot_data_manifest.json references {sorted(refs)}; "
          f"missing on disk: {missing}")


def main():
    sec_hashes()
    sec_features_and_splits()
    sec_extension()
    sec_trials()
    sec_manuscript()
    sec_discrepancies()
    section("SUMMARY")
    n_uns = sum(1 for r in LEDGER.values() if r["status"] == "unsourced")
    print(f"checks passed: {COUNTS['pass']}, failed: {COUNTS['fail']}")
    print(f"largest A8 bootstrap CI deviation: {MAX_CI_DEV[0]:.6f} (tolerance +/-0.002)")
    print(f"ledger rows: {len(LEDGER)} (unsourced rows listed in claim_audit.md: {n_uns})")
    print("RESULT:", "PASS" if COUNTS["fail"] == 0 else "FAIL")
    return 0 if COUNTS["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
