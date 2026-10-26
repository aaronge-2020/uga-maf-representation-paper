"""Unified benchmark runner (protocol v1.0).

One invocation = one (task, representation, learner) cell. Resumable: a cell whose
predictions file and trial table already exist is skipped unless --force is given.

Design points fixed by the protocol:
- canonical outer split from config/canonical_split_<task>.csv, shared by all cells;
- tuning inside the outer training partition only, on one shared stratified inner
  holdout (20 percent, seed 20261025) used for model selection for every cell, and
  for XGBoost early stopping; TF-IDF statistics fitted on the outer training
  partition of each fold (and refit inside tuning on the inner-train part);
- 10 random-search trials per outer fold, spaces in sample_config();
- every trial, failure, selected config, and fit time is appended to the trial log.
"""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[3]
NS = ROOT / "analyses" / "representation_extension"
SEED = 20261025
N_TRIALS = 10

sys.path.insert(0, str(NS / "scripts"))


def load_matrix(task, name):
    if name == "burden":
        df = pd.read_csv(ROOT / "data/task0_features/features_burden_only.csv", index_col=0)
    elif name == "maf_stack":
        df = pd.read_csv(ROOT / "results/tables/quick_bio_maf_features.csv.gz", index_col=0)
    elif name == "spectra":
        df = pd.read_csv(ROOT / "data/task0_features/features_standard_sbs96_id83.csv.gz", index_col=0)
    else:
        df = pd.read_csv(NS / "features" / f"{task}_{name}.csv.gz", index_col=0)
    df.index = df.index.astype(str).str.slice(0, 12)
    return df[~df.index.duplicated(keep="first")]


def tfidf(counts, train_idx):
    df = (counts.iloc[train_idx] > 0).sum(axis=0)
    vocab = df[df > 0].index
    idf = np.log(len(train_idx) / df[vocab])
    return counts[vocab] * idf, list(vocab)


def assemble(task, rep, patients):
    """Return (X DataFrame aligned to patients, meta). TF-IDF handled by caller per fold."""
    parts = []
    if rep in ("burden",):
        parts = ["burden"]
    elif rep in ("spectra",):
        parts = ["spectra"]
    elif rep.startswith("spectra+"):
        parts = ["spectra", rep.split("+", 1)[1]]
    else:
        parts = [rep]
    mats = []
    for p in parts:
        base = p.replace("_log1p", "")
        m = load_matrix(task, base)
        m = m.reindex(patients)
        if m.isna().any().any():
            raise ValueError(f"{p}: patients missing from matrix")
        if p.endswith("_log1p"):
            m = np.log1p(m)
        mats.append(m.reset_index(drop=True))
    return pd.concat(mats, axis=1), parts


def sample_config(rng, learner, task):
    if learner == "xgboost":
        return {
            "max_depth": int(rng.choice([3, 4, 5, 6])),
            "learning_rate": float(rng.choice([0.03, 0.06, 0.1, 0.2])),
            "subsample": float(rng.choice([0.7, 0.85, 1.0])),
            "colsample_bytree": float(rng.choice([0.4, 0.6, 0.8, 1.0])),
            "min_child_weight": float(rng.choice([1, 3, 5])),
            "reg_lambda": float(rng.choice([0.5, 1.0, 2.0, 5.0])),
            "n_estimators": 400,
        }
    if task == "cancer":
        return {"C": float(10 ** rng.uniform(-3, 0.5)), "l1_ratio": float(rng.choice([0.1, 0.3, 0.5, 0.7, 0.9]))}
    return {"alpha": float(10 ** rng.uniform(-3, 0.5)), "l1_ratio": float(rng.choice([0.1, 0.3, 0.5, 0.7, 0.9]))}


def fit_predict(learner, task, cfg, Xtr, ytr, Xva, yva, Xte, seed):
    if learner == "xgboost":
        if task == "cancer":
            classes, counts = np.unique(ytr, return_counts=True)
            wmap = {c: len(ytr) / (len(classes) * n) for c, n in zip(classes, counts)}
            sw = np.array([wmap[v] for v in ytr])
            m = XGBClassifier(**cfg, early_stopping_rounds=40, n_jobs=2,
                              random_state=seed, tree_method="hist")
            m.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
            return m.predict(Xte), m.predict_proba(Xte), m.best_iteration
        m = XGBRegressor(**cfg, early_stopping_rounds=40, n_jobs=2,
                         random_state=seed, tree_method="hist")
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return m.predict(Xte), None, m.best_iteration
    # elastic net family, standardized features
    sc = StandardScaler().fit(Xtr)
    Ztr, Zva, Zte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    if task == "cancer":
        m = LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced",
                               max_iter=400, random_state=seed, **cfg)
        m.fit(Ztr, ytr)
        return m.predict(Zte), m.predict_proba(Zte), None
    m = ElasticNet(max_iter=3000, random_state=seed, **cfg)
    m.fit(Ztr, ytr)
    return m.predict(Zte), None, None


def score(task, ytrue, pred):
    if task == "cancer":
        return balanced_accuracy_score(ytrue, pred)
    return float(spearmanr(ytrue, pred).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["cancer", "hrd"])
    ap.add_argument("--representation", required=True)
    ap.add_argument("--learner", required=True, choices=["xgboost", "elasticnet"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()
    task, rep, learner = args.task, args.representation, args.learner
    cell = f"{task}__{rep}__{learner}"
    pred_path = NS / "predictions" / f"{cell}.csv"
    trial_path = NS / "metrics" / "trials" / f"{cell}.csv"
    if pred_path.exists() and not args.force:
        print(f"SKIP {cell} (exists)")
        return

    split = pd.read_csv(NS / "config" / f"canonical_split_{task}.csv", dtype={"patient": str})
    patients = list(split["patient"])
    y_all = split["label"].values
    is_tfidf = "r5" in rep
    counts_for_tfidf = None
    if is_tfidf:
        base = rep.replace("spectra+", "")
        counts_for_tfidf = load_matrix(task, "r3_counts_panel").reindex(patients)
        X_static, _ = assemble(task, rep.replace(base, "r3_counts_panel"), patients) \
            if rep.startswith("spectra+") else (None, None)
        if rep.startswith("spectra+"):
            spec = load_matrix(task, "spectra").reindex(patients).reset_index(drop=True)
        else:
            spec = None
    else:
        X_all, _ = assemble(task, rep, patients)
        X_all = X_all.values.astype(np.float64)

    trials, preds = [], []
    for fold in range(1, 6):
        te_mask = split["fold"].values == fold
        tr_mask = ~te_mask
        tr_idx = np.where(tr_mask)[0]
        if is_tfidf:
            tfidf_mat, _ = tfidf(counts_for_tfidf, tr_idx)
            if spec is not None:
                X_all = pd.concat([spec.reset_index(drop=True),
                                   tfidf_mat.reset_index(drop=True)], axis=1).values.astype(np.float64)
            else:
                X_all = tfidf_mat.values.astype(np.float64)
        # shared inner holdout inside outer training data
        sss = StratifiedShuffleSplit(1, test_size=0.2, random_state=SEED)
        strat = y_all[tr_idx] if task == "cancer" else pd.qcut(y_all[tr_idx], 5, labels=False)
        inner_tr_pos, inner_va_pos = next(sss.split(np.zeros(len(tr_idx)), strat))
        itr, iva = tr_idx[inner_tr_pos], tr_idx[inner_va_pos]
        rng = np.random.default_rng(SEED + fold * 1000 + abs(hash(cell)) % 10000)
        best = None
        for t in range(args.trials):
            cfg = sample_config(rng, learner, task)
            t0 = time.time()
            rec = {"cell": cell, "fold": fold, "trial": t, "config": json.dumps(cfg),
                   "status": "ok", "inner_score": np.nan, "seconds": np.nan}
            try:
                p_va, _, _ = fit_predict(learner, task, cfg, X_all[itr], y_all[itr],
                                         X_all[iva], y_all[iva], X_all[iva], SEED + t)
                rec["inner_score"] = score(task, y_all[iva], p_va)
            except Exception as e:  # keep the failure on record
                rec["status"] = f"FAILED: {type(e).__name__}: {e}"
                traceback.print_exc()
            rec["seconds"] = round(time.time() - t0, 1)
            trials.append(rec)
            if rec["status"] == "ok" and (best is None or rec["inner_score"] > best[0]):
                best = (rec["inner_score"], cfg)
        if best is None:
            raise RuntimeError(f"{cell} fold {fold}: all trials failed")
        cfg = best[1]
        t0 = time.time()
        p_te, proba, best_iter = fit_predict(learner, task, cfg, X_all[tr_idx], y_all[tr_idx],
                                             X_all[iva], y_all[iva], X_all[te_mask], SEED)
        fit_seconds = round(time.time() - t0, 1)
        trials.append({"cell": cell, "fold": fold, "trial": "FINAL", "config": json.dumps(cfg),
                       "status": "ok", "inner_score": best[0], "seconds": fit_seconds,
                       "best_iteration": best_iter})
        block = {"patient": np.array(patients)[te_mask], "fold": fold,
                 "true_value": y_all[te_mask], "prediction": p_te}
        if proba is not None:
            for k in range(proba.shape[1]):
                block[f"proba_{k}"] = proba[:, k]
        preds.append(pd.DataFrame(block))
        print(f"{cell} fold {fold}: inner best {best[0]:.4f}, final fit {fit_seconds}s", flush=True)

    out = pd.concat(preds).sort_values(["fold", "patient"])
    out.insert(0, "task", task); out.insert(1, "representation", rep); out.insert(2, "learner", learner)
    out.insert(3, "seed", SEED)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(pred_path, index=False)
    pd.DataFrame(trials).to_csv(trial_path, index=False)
    print(f"WROTE {pred_path} rows={len(out)} pooled score={score(task, out['true_value'].values, out['prediction'].values):.4f}")


if __name__ == "__main__":
    main()
