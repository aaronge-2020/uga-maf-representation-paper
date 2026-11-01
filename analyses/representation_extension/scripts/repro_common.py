"""Shared computations for the reproducibility package (workstreams G1 to G3).

Every number the package reports is computed here, once, from the saved
artifacts. scripts/build_ledger.py writes those values into results_ledger.csv;
scripts/verify_reproduce.py recomputes them through the same functions and
compares. Nothing in this module fits a model.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
NS = ROOT / "analyses" / "representation_extension"
SEED = 20261025

# ---------------------------------------------------------------- extension

REPS_A8 = ["spectra+maf_stack", "spectra+r3_counts_panel", "spectra+r2_binary_panel",
           "spectra+r4_consequence_panel", "spectra+r5_tfidf_panel"]
REF_A8 = "spectra+maf_stack"
NBOOT = 2000
CI_TOL = 0.002          # stated Monte Carlo tolerance for bootstrap CI bounds
SCORE_TOL = 5e-5        # ledger stores extension scores rounded to 4 dp


def cell_name(task, rep, learner):
    return f"{task}__{rep}__{learner}"


def list_cells():
    cells = []
    for p in sorted((NS / "predictions").glob("*.csv")):
        task, rep, learner = p.stem.split("__")
        cells.append((task, rep, learner))
    return cells


def load_predictions(task, rep, learner):
    df = pd.read_csv(NS / "predictions" / f"{cell_name(task, rep, learner)}.csv",
                     float_precision="round_trip")
    return df.sort_values("patient").reset_index(drop=True)


def metric(task, y, p):
    if task == "cancer":
        return balanced_accuracy_score(y, p)
    return float(spearmanr(y, p).statistic)


def cell_scores(task, rep, learner):
    """Per-fold scores and the primary score (mean within-fold) for one cell."""
    df = load_predictions(task, rep, learner)
    y = df["true_value"].values
    y = y.astype(float) if task == "hrd" else y
    folds = {}
    for k in sorted(df["fold"].unique()):
        m = df["fold"].values == k
        folds[int(k)] = metric(task, y[m], df["prediction"].values[m])
    return folds, float(np.mean(list(folds.values())))


def a8_contrasts():
    """The A8 primary contrasts, replicating the reference implementation's
    bootstrap stream exactly (one generator, fixed iteration order), so the
    confidence intervals reproduce the delivered table to the digit."""
    rng = np.random.default_rng(SEED)
    out = {}
    for task in ["cancer", "hrd"]:
        for learner in ["xgboost", "elasticnet"]:
            frames = {rep: load_predictions(task, rep, learner) for rep in REPS_A8}
            base = frames[REF_A8]
            y = base["true_value"].values
            y = y.astype(float) if task == "hrd" else y
            fold_scores, primary = {}, {}
            for rep in REPS_A8:
                fs = {}
                for k in sorted(base["fold"].unique()):
                    m = base["fold"].values == k
                    fs[int(k)] = metric(task, y[m], frames[rep]["prediction"].values[m])
                fold_scores[rep] = fs
                primary[rep] = float(np.mean(list(fs.values())))
            entry = {"primary": primary, "fold_scores": fold_scores, "contrasts": {}}
            ref_p = frames[REF_A8]["prediction"].values
            n = len(y)
            for rep in REPS_A8:
                if rep == REF_A8:
                    continue
                p = frames[rep]["prediction"].values
                per_fold_diff = float(np.mean(
                    [fold_scores[rep][k] - fold_scores[REF_A8][k] for k in fold_scores[rep]]))
                diffs = np.empty(NBOOT)
                for b in range(NBOOT):
                    idx = rng.integers(0, n, n)
                    diffs[b] = metric(task, y[idx], p[idx]) - metric(task, y[idx], ref_p[idx])
                lo, hi = np.percentile(diffs, [2.5, 97.5])
                entry["contrasts"][rep] = {
                    "mean_per_fold_diff": per_fold_diff,
                    "bootstrap_ci95": [float(lo), float(hi)],
                }
            out[f"{task}/{learner}"] = entry
    return out


def frame_digest(df):
    """Order-independent digest of a feature frame's values."""
    df = df.sort_index().sort_index(axis=1)
    h = hashlib.sha256()
    h.update("\x00".join(map(str, df.index)).encode())
    h.update("\x00".join(map(str, df.columns)).encode())
    h.update(np.ascontiguousarray(df.values).tobytes())
    return h.hexdigest()


def load_feature_frame(path):
    return pd.read_csv(path, index_col=0)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sbs3_brca_counts():
    """Audit report section 6: SBS3 nonzero count and BRCA1/2 indicator count,
    recomputed from the raw files."""
    sbs = pd.read_csv(ROOT / "data/task3/sbs3_activity_772.csv", index_col=0)
    col = [c for c in sbs.columns if "SBS3" in c.upper()][0]
    sbs3_n = int((sbs[col] > 0).sum())
    sbs3_mean = float(sbs[col].mean())
    sbs3_max = float(sbs[col].max())
    cohort = pd.read_csv(ROOT / "data/task3/final_analysis_cohort.tsv", sep="\t")
    patients772 = set(cohort["patient_id_12"].astype(str))
    maf = pd.read_csv(ROOT / "results/tables/quick_bio_maf_features.csv.gz", index_col=0)
    maf.index = maf.index.astype(str).str.slice(0, 12)
    role_cols = [c for c in maf.columns if "role_matched" in c and
                 (c.endswith("__BRCA1") or c.endswith("__BRCA2"))]
    assert len(role_cols) == 2, role_cols
    pos = (maf[role_cols] > 0).any(axis=1)
    brca_all = int(pos.sum())
    brca_772 = int(pos.reindex(sorted(patients772)).fillna(False).sum())
    return {"sbs3_nonzero": sbs3_n, "sbs3_mean": sbs3_mean, "sbs3_max": sbs3_max,
            "brca_positive_772": brca_772, "brca_positive_all": brca_all,
            "brca_columns": role_cols}

# ---------------------------------------------------------------- manuscript

T = ROOT / "results" / "tables"
REP_PRINT = {"burden_only": "Burden", "standard_sbs96_id83": "Spectra",
             "MAF_stack_only": "Bio MAF", "signatures_plus_MAF_stack": "Sig+MAF"}
T1_ENDPOINTS = ["HRD_Score", "hrd_binary_24", "hrd_binary_33", "hrd_binary_42",
                "cancer_type_top20", "OS"]
T_CRIT = float(stats.t.ppf(0.975, 4))


def t_interval(fold_scores):
    x = np.asarray(fold_scores, dtype=float)
    half = T_CRIT * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() - half), float(x.mean() + half)


def fold_metrics():
    return pd.read_csv(T / "main_manuscript_complete_panel_fold_metrics.csv")


def endpoint_results():
    return pd.read_csv(T / "main_manuscript_complete_panel_endpoint_results.csv")


def table1_fold_means():
    """{(endpoint, representation, learner): (fold_mean, [fold scores])} for the
    six Table 1 endpoints, from the fold metrics file."""
    fm = fold_metrics()
    out = {}
    for (ep, rep, learner), g in fm[fm["endpoint"].isin(T1_ENDPOINTS)].groupby(
            ["endpoint", "representation", "learner"]):
        scores = [float(g.loc[g["fold"] == k, "score"].iloc[0]) for k in range(1, 6)]
        out[(ep, rep, learner)] = (float(np.mean(scores)), scores)
    return out


def survival_fold_means():
    fm = fold_metrics()
    out = {}
    for (ep, rep, learner), g in fm[fm["endpoint"].isin(["OS", "PFI"])].groupby(
            ["endpoint", "representation", "learner"]):
        scores = [float(g.loc[g["fold"] == k, "score"].iloc[0]) for k in range(1, 6)]
        out[(ep, rep, learner)] = float(np.mean(scores))
    return out


def load_oof():
    return pd.read_csv(T / "main_manuscript_complete_panel_oof_predictions.csv",
                       low_memory=False)


def _oof_metric(ep, metric_name, g):
    y = g["true_value"].values
    if metric_name == "balanced_accuracy":
        return balanced_accuracy_score(y, g["predicted_class"].values)
    if metric_name == "auroc":
        # binary endpoints store the positive-class probability in pred_class_1
        s = g["pred_value"] if g["pred_value"].notna().all() else g["pred_class_1"]
        return roc_auc_score(y.astype(int), s.values.astype(float))
    if metric_name == "spearman":
        return float(spearmanr(y.astype(float), g["pred_value"].values.astype(float)).statistic)
    if metric_name == "c_index":
        return harrell_c(g["time"].values.astype(float), g["event"].values.astype(int),
                         g["risk_score"].values.astype(float))
    raise ValueError(metric_name)


def harrell_c(time, event, risk):
    """Harrell's concordance, vectorized by time groups. A pair is comparable
    when the earlier time carries an event (time ties count when both have
    events, or the tied partner is an event against a longer-lived subject)."""
    order = np.argsort(time, kind="mergesort")
    t, e, r = time[order], event[order], risk[order]
    n = len(t)
    conc = tied = comp = 0.0
    # group boundaries of equal times
    bounds = np.searchsorted(t, np.unique(t), side="right")
    starts = np.concatenate([[0], bounds[:-1]])
    ends = bounds
    suffix_events = np.concatenate([np.cumsum(e[::-1])[::-1][1:], [0]])  # events after each pos
    for s, en in zip(starts, ends):
        idx = np.arange(s, en)
        ev = idx[e[idx] == 1]
        if len(ev) == 0:
            continue
        later = np.arange(en, n)
        for i in ev:
            if len(later):
                conc += float((r[later] < r[i]).sum())
                tied += float((r[later] == r[i]).sum())
                comp += float(len(later))
            peers = idx[idx != i]
            peers_ev = peers[e[peers] == 1]
            if len(peers_ev):
                conc += float((r[peers_ev] < r[i]).sum())
                tied += float((r[peers_ev] == r[i]).sum())
                comp += float(len(peers_ev))
    return (conc + 0.5 * tied) / comp if comp else float("nan")


def oof_vs_endpoint():
    """Pooled primary metric recomputed from the saved OOF file for every
    (endpoint, representation, learner) group in the endpoint results file,
    beside the endpoint file's stored pooled score."""
    oof = load_oof()
    ep = endpoint_results()
    rows = []
    oof_groups = {(ep_, rep, lr): g for (ep_, rep, lr), g in
                  oof.groupby(["endpoint", "representation", "learner"])}
    for _, r in ep.iterrows():
        key = (r["endpoint"], r["representation"], r["learner"])
        if key not in oof_groups:
            rows.append({**dict(zip(["endpoint", "representation", "learner"], key)),
                         "metric": r["metric"], "endpoint_score": r["score"],
                         "oof_score": None, "abs_diff": None})
            continue
        val = _oof_metric(r["endpoint"], r["metric"], oof_groups[key])
        rows.append({**dict(zip(["endpoint", "representation", "learner"], key)),
                     "metric": r["metric"], "endpoint_score": float(r["score"]),
                     "oof_score": float(val), "abs_diff": abs(float(val) - float(r["score"]))})
    return pd.DataFrame(rows)


def _paired_t(diffs):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    mean = float(d.mean())
    se = d.std(ddof=1) / np.sqrt(n)
    lo, hi = mean - T_CRIT * se, mean + T_CRIT * se
    p = float(stats.ttest_1samp(d, 0.0).pvalue)
    return mean, float(lo), float(hi), p


MUAT_CANCER_OOF = T / "muat_style_tcga_comparator_cancer_type_top20_faithful_fixed_arch_oof_predictions.csv"
MUAT_HRD_OOF = T / "muat_style_tcga_comparator_hrd_faithful_fixed_arch_oof_predictions.csv"


def muat_cancer():
    df = pd.read_csv(MUAT_CANCER_OOF)
    y = df["true_value"].values
    pred = df["predicted_class"].values
    proba_cols = sorted([c for c in df.columns if c.startswith("pred_class_")],
                        key=lambda c: int(c.rsplit("_", 1)[1]))
    proba = df[proba_cols].values
    fold_ba = {}
    for k in sorted(df["fold"].unique()):
        m = df["fold"].values == k
        fold_ba[int(k)] = balanced_accuracy_score(y[m], pred[m])
    top5 = float(np.mean([y[i] in np.argsort(-proba[i])[:5] for i in range(len(y))]))
    return {"fold_ba": fold_ba, "ba_mean": float(np.mean(list(fold_ba.values()))),
            "accuracy": float((pred == y).mean()), "top5": top5,
            "ba_ci": t_interval(list(fold_ba.values()))}


def muat_hrd():
    df = pd.read_csv(MUAT_HRD_OOF)
    out = {}
    for ep, met in [("HRD_Score", "spearman"), ("hrd_binary_33", "auroc")]:
        g = df[df["endpoint"] == ep]
        folds = {}
        for k in sorted(g["fold"].unique()):
            gg = g[g["fold"] == k]
            if met == "spearman":
                folds[int(k)] = float(spearmanr(gg["true_value"], gg["pred_value"]).statistic)
            else:
                folds[int(k)] = roc_auc_score(gg["true_value"].astype(int), gg["pred_class_1"])
        out[ep] = {"folds": folds, "mean": float(np.mean(list(folds.values()))),
                   "ci": t_interval(list(folds.values()))}
    return out


def paired_vs_muat():
    """Paired per-fold differences, Sig+MAF XGBoost (fold metrics file) minus
    MuAt faithful (recomputed from its OOF files), with paired t statistics."""
    fm = fold_metrics()
    def sig_folds(ep, metric):
        g = fm[(fm["endpoint"] == ep) & (fm["representation"] == "signatures_plus_MAF_stack")
               & (fm["learner"] == "xgboost") & (fm["metric"] == metric)]
        return [float(g.loc[g["fold"] == k, "score"].iloc[0]) for k in range(1, 6)]
    mc = muat_cancer()
    mh = muat_hrd()
    out = {}
    d = np.array(sig_folds("cancer_type_top20", "balanced_accuracy")) - \
        np.array([mc["fold_ba"][k] for k in range(1, 6)])
    out["cancer_ba"] = {"diffs": list(d), **dict(zip(["mean", "ci_lo", "ci_hi", "p"], _paired_t(d)))}
    d = np.array(sig_folds("HRD_Score", "spearman")) - \
        np.array([mh["HRD_Score"]["folds"][k] for k in range(1, 6)])
    out["hrd_spearman"] = {"diffs": list(d), **dict(zip(["mean", "ci_lo", "ci_hi", "p"], _paired_t(d)))}
    d = np.array(sig_folds("hrd_binary_33", "auroc")) - \
        np.array([mh["hrd_binary_33"]["folds"][k] for k in range(1, 6)])
    out["hrd33_auroc"] = {"diffs": list(d), **dict(zip(["mean", "ci_lo", "ci_hi", "p"], _paired_t(d)))}
    return out


def ablations():
    df = pd.read_csv(T / "task1_gene_knowledge_ablations.csv")
    out = {}
    for (rep, var), g in df.groupby(["representation", "variant"]):
        out[(rep, var)] = float(g["balanced_accuracy"].mean())
    return out


def pcawg():
    pred = pd.read_csv(ROOT / "external_validation/pcawg_transfer/predictions/pcawg_sigmaf_predictions.csv")
    y, p = pred["true_label"].values, pred["pred_label"].values
    recalls = {}
    for t in sorted(set(y)):
        m = y == t
        recalls[t] = float((p[m] == t).mean())
    return {"accuracy": float((p == y).mean()),
            "balanced_accuracy": balanced_accuracy_score(y, p), "recalls": recalls,
            "n": int(len(y))}


def pog570():
    pred = pd.read_csv(ROOT / "results/external_validation/aaron_email/pog570/pog570_frozen_model_predictions.csv")
    g = pred[pred["analysis_group"] == "primary_breast"]
    y, p = g["HRD_SCORE"].values.astype(float), g["predicted_HRD_SCORE"].values.astype(float)
    r2 = 1 - float(np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    spec24 = float((p[y < 24] < 24).mean())
    return {"n": int(len(g)), "spearman": float(spearmanr(y, p).statistic),
            "pearson": float(np.corrcoef(y, p)[0, 1]), "r2": r2,
            "mae": float(np.mean(np.abs(y - p))), "specificity24": spec24}


def centre_heldout():
    df = pd.read_csv(T / "task4_centre_heldout.csv")
    out = {}
    for (typ, rep), g in df.groupby(["type", "representation"]):
        out[(typ, rep)] = {
            "pooled_tumours": float(g["n_correct"].sum() / g["n_test_tumours"].sum()),
            "pooled_patients": float(g["n_correct"].sum() / g["n_test_patients"].sum())}
    return out


def survival_covariate():
    s = pd.read_csv(T / "survival_cancer_type_baseline_summary.csv")
    c = pd.read_csv(T / "survival_cancer_type_baseline_comparisons.csv")
    scores = {r["representation"]: float(r["score"]) for _, r in s.iterrows()}
    row = c[(c["reference"] == "cancer_type") & (c["candidate"] == "cancer_type_plus_bio_maf")].iloc[0]
    return {"scores": scores, "gain": float(row["delta"]), "p": float(row["p_two_sided"]),
            "ci": (float(row["ci_low"]), float(row["ci_high"]))}


def hrd_controls():
    s = pd.read_csv(T / "hrd_shortcut_ablation_summary.csv")
    c = pd.read_csv(T / "hrd_shortcut_ablation_comparisons.csv")
    scores = {}
    for _, r in s.iterrows():
        scores[(r["endpoint"], r["representation"])] = float(r["score"])
    comps = {}
    for _, r in c.iterrows():
        comps[(r["endpoint"], r["candidate"])] = {
            "delta": float(r["delta"]), "p": float(r["p_two_sided"]),
            "ci": (float(r["ci_low"]), float(r["ci_high"]))}
    return {"scores": scores, "comparisons": comps}


def task3_baselines():
    df = pd.read_csv(T / "task3_hrd_attribution_baselines.csv")
    out = {}
    for _, r in df[df["scope"] == "pooled"].iterrows():
        out[(r["baseline"], r["endpoint"], r["metric"])] = float(r["score"])
    return out


def fig3_gains():
    oof = load_oof()
    g = oof[oof["endpoint"] == "cancer_type_top20"]
    labels = {}
    for k in range(20):
        col = f"class_label_{k}"
        if col in g.columns:
            labels[k] = str(g[col].dropna().iloc[0])
    out = {}
    for rep in ["standard_sbs96_id83", "signatures_plus_MAF_stack"]:
        gg = g[(g["representation"] == rep) & (g["learner"] == "xgboost")]
        for k in range(20):
            m = gg["true_value"].values == k
            out.setdefault(k, {})[rep] = float(
                (gg["predicted_class"].values[m] == k).mean())
            out[k]["n"] = int(m.sum())
    for k in out:
        out[k]["gain"] = out[k]["signatures_plus_MAF_stack"] - out[k]["standard_sbs96_id83"]
        out[k]["label"] = labels.get(k, str(k))
    return out


def _shap_block(f):
    """The block grouping the regenerated Figure 6 uses (fixed prefix rules)."""
    f = str(f)
    if f.startswith("vep_consequence"):
        return "VEP consequence"
    if f.startswith("driver_gene"):
        return "Gene-specific features"
    if f.startswith(("vep_sift", "vep_polyphen", "vep_biotype", "vep_canonical")):
        return "VEP other (SIFT, PolyPhen, biotype)"
    if f.startswith("vep_impact"):
        return "VEP impact"
    if f.startswith("vaf"):
        return "Allele fraction (VAF)"
    if f.startswith("driver_evidence"):
        return "Driver evidence / tiers"
    if f.startswith("oncokb"):
        return "OncoKB role"
    if f.startswith("residual"):
        return "Residual annotations"
    if f.startswith("cancer_hotspot"):
        return "Hotspots"
    if f.startswith("id_") or f == "id_fraction":
        return "Mutation type / burden"
    return "Other"


def shap_shares(path):
    df = pd.read_csv(path)
    total = df["mean_abs_shap"].sum()
    shares = {}
    for feat, val in zip(df["feature"], df["mean_abs_shap"]):
        grp = _shap_block(feat)
        shares[grp] = shares.get(grp, 0.0) + float(val)
    return {k: 100.0 * v / total for k, v in shares.items()}


def shap_text_shares(path):
    """The nine groups the manuscript Results text cites. This is the figure
    grouping with two merges: residual annotations fold into the OncoKB group
    and the burden columns (log10 burdens, dbs_fraction) fold into mutation
    type / count. Under this partition the nine printed shares sum to 100.0
    and each matches the CSV to the printed digit."""
    fig = shap_shares(path)
    return {
        "Mutation consequence summaries": fig["VEP consequence"],
        "Per-gene driver counts": fig["Gene-specific features"],
        "Functional-effect predictions": fig["VEP other (SIFT, PolyPhen, biotype)"],
        "Impact": fig["VEP impact"],
        "OncoKB gene roles": fig["OncoKB role"] + fig["Residual annotations"],
        "Allele-fraction summaries": fig["Allele fraction (VAF)"],
        "Driver evidence tiers": fig["Driver evidence / tiers"],
        "Hotspot summaries": fig["Hotspots"],
        "Mutation type / count": fig["Mutation type / burden"] + fig["Other"],
    }


def shap_max_feature_share(path):
    df = pd.read_csv(path)
    return float(100.0 * df["mean_abs_shap"].max() / df["mean_abs_shap"].sum())


def kucab_pooled():
    ep = endpoint_results()
    out = {}
    for _, r in ep[ep["endpoint"] == "damage_class"].iterrows():
        out[(r["representation"], r["learner"])] = float(r["score"])
    return out
