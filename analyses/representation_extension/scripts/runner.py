"""Unified benchmark runner (protocol v1.0).

One invocation = one (task, representation, learner) cell. Resumable: a cell whose
predictions file and trial table already exist is skipped unless --force is given.
Per-fold checkpointing: each completed outer fold's predictions and trial records
are persisted atomically under predictions_partial/<cell>/, together with the
trial configurations sampled for every fold (drawn at cell start from the same
per-fold seeded generators the fold loop has always used). A restarted invocation
validates and replays those partials instead of recomputing finished folds, so a
killed cell resumes at its first unfinished fold with identical configurations.
Per-trial checkpointing: within an unfinished fold, every tuning trial's record
is persisted atomically to predictions_partial/<cell>/fold<k>_trial<t>.csv as soon
as that trial completes. One fold's tuning can outlast this machine's process
lifetime (cancer elastic net on the wide matrix), so a killed fold resumes at its
first unfinished trial: persisted trial records are validated against the fold's
persisted configurations and replayed, and the fold's best-configuration
selection, final fit, and partial write proceed exactly as before.

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
import os
import shutil
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
            m = XGBClassifier(**cfg, early_stopping_rounds=40, n_jobs=int(os.environ.get('RUNNER_N_JOBS', 2)),
                              random_state=seed, tree_method="hist")
            m.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
            return m.predict(Xte), m.predict_proba(Xte), m.best_iteration
        m = XGBRegressor(**cfg, early_stopping_rounds=40, n_jobs=int(os.environ.get('RUNNER_N_JOBS', 2)),
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


def atomic_write_csv(df, path, float_format=None):
    """Write a CSV via a temp file in the same directory, then rename over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, float_format=float_format)
    os.replace(tmp, path)


def atomic_write_text(text, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def fold_rng(cell, fold):
    """The per-fold tuning generator, seeded exactly as the fold loop seeds it."""
    return np.random.default_rng(SEED + fold * 1000 + abs(hash(cell)) % 10000)


def load_or_make_configs(partial_dir, cell, learner, task, n_trials):
    """Trial configurations for all five folds, persisted at cell start.

    Each fold's seed uses Python's salted string hash, so a fresh process would
    sample different configurations for folds it has not run yet. Each fold keeps
    its own seeded generator and the generator feeds only sample_config, so
    drawing every fold's configurations up front yields lists identical to
    drawing them inside the loop; persisting them makes a restarted invocation
    replay the identical tuning.
    """
    path = partial_dir / "trial_configs.json"
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            configs = {int(k): v for k, v in raw.items()}
            ok = set(configs) == set(range(1, 6)) and all(
                len(configs[f]) == n_trials
                and all(isinstance(c, dict) for c in configs[f])
                for f in configs)
            if ok:
                print(f"{cell} checkpoint: trial configs loaded from partial", flush=True)
                return configs
            print(f"{cell} checkpoint: trial configs partial invalid, redrawing", flush=True)
        except Exception as e:
            print(f"{cell} checkpoint: trial configs partial unreadable ({e}), redrawing",
                  flush=True)
    configs = {fold: [sample_config(fold_rng(cell, fold), learner, task)
                      for _ in range(n_trials)]
               for fold in range(1, 6)}
    atomic_write_text(json.dumps(configs), path)
    print(f"{cell} checkpoint: trial configs written", flush=True)
    return configs


def _same_config(stored, cfg):
    """A stored config string parses back to exactly this configuration dict."""
    try:
        return json.loads(stored) == cfg
    except Exception:
        return False


def trial_record_matches(rec, cell, fold, trial, cfg):
    """One stored trial record belongs to this cell/fold/trial and carries the
    persisted configuration sampled for that trial. The inner score is NaN
    exactly for failed trials, matching how the trial loop writes them."""
    return (
        rec["cell"] == cell
        and int(rec["fold"]) == fold
        and str(rec["trial"]) == str(trial)
        and _same_config(rec["config"], cfg)
        and isinstance(rec["status"], str)
        and (pd.isna(rec["inner_score"]) if rec["status"].startswith("FAILED")
             else not pd.isna(rec["inner_score"]))
    )


def trial_file_matches(trials, cell, fold, n_trials, configs):
    """A fold's stored trial records replay the fold only if they are exactly
    the fold's per-trial records for the persisted configurations, in order,
    ending in a FINAL record whose score and configuration match the best
    trial under the trial loop's own selection rule."""
    if len(trials) != n_trials + 1:
        return False
    best = None
    for t in range(n_trials):
        if not trial_record_matches(trials[t], cell, fold, t, configs[t]):
            return False
        if trials[t]["status"] == "ok" and (best is None or trials[t]["inner_score"] > best[0]):
            best = (trials[t]["inner_score"], configs[t])
    if best is None:
        return False
    fin = trials[-1]
    return (
        fin["cell"] == cell
        and int(fin["fold"]) == fold
        and str(fin["trial"]) == "FINAL"
        and fin["status"] == "ok"
        and not pd.isna(fin["inner_score"])
        and fin["inner_score"] == best[0]
        and _same_config(fin["config"], best[1])
    )


def load_trial_checkpoint(partial_dir, cell, fold, t, cfg):
    """Return trial t's persisted record for this fold, else None.

    A per-trial checkpoint is valid only if its file is readable with the
    round-trip parser, holds exactly one row with exactly the trial record
    columns, and that row matches this cell, fold, and trial index and carries
    exactly the configuration persisted for the trial. Anything else is logged
    and the trial is recomputed.
    """
    path = partial_dir / f"fold{fold}_trial{t}.csv"
    if not path.exists():
        return None
    try:
        tri = pd.read_csv(path, float_precision="round_trip")
    except Exception:
        print(f"{cell} checkpoint: fold {fold} trial {t} checkpoint unreadable, recomputing",
              flush=True)
        return None
    cols = {"cell", "fold", "trial", "config", "status", "inner_score", "seconds"}
    if (set(tri.columns) != cols or len(tri) != 1
            or not trial_record_matches(tri.iloc[0], cell, fold, t, cfg)):
        print(f"{cell} checkpoint: fold {fold} trial {t} checkpoint failed validation, "
              f"recomputing", flush=True)
        return None
    return tri.to_dict("records")[0]


def load_valid_partial(partial_dir, cell, fold, split, patients, y_all, task,
                       n_trials, configs):
    """Return (fold predictions, fold trial records) from a validated partial, else None.

    A partial is valid only if the predictions file is readable, carries exactly
    the columns the fold block would have, holds one row per canonical test
    patient of this fold with no duplicates, and the trial file is readable and
    holds exactly this fold's trial records for the persisted configurations,
    ending in a FINAL record that matches the fold's best trial.
    """
    pred_file = partial_dir / f"fold{fold}.csv"
    trial_file = partial_dir / f"fold{fold}_trials.csv"
    if not (pred_file.exists() and trial_file.exists()):
        return None
    try:
        # float_precision="round_trip": the default CSV parser can land one ulp off
        # on 17-digit floats; the round-trip parser recovers partial values exactly.
        df = pd.read_csv(pred_file, dtype={"patient": str}, float_precision="round_trip")
        tri = pd.read_csv(trial_file, float_precision="round_trip")
    except Exception:
        return None
    te_patients = list(np.array(patients)[split["fold"].values == fold])
    base_cols = {"patient", "fold", "true_value", "prediction"}
    proba_cols = ({f"proba_{k}" for k in range(len(np.unique(y_all)))}
                  if task == "cancer" else set())
    valid = (
        set(df.columns) == base_cols | proba_cols
        and len(df) == len(te_patients)
        and df["patient"].is_unique
        and set(df["patient"]) == set(te_patients)
        and (df["fold"] == fold).all()
        and df["prediction"].notna().all()
        and df["true_value"].notna().all()
    )
    records = tri.to_dict("records")
    if not valid or not trial_file_matches(records, cell, fold, n_trials, configs):
        print(f"{cell} checkpoint: fold {fold} partial failed validation, recomputing",
              flush=True)
        return None
    return df, records


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

    partial_dir = NS / "predictions_partial" / cell
    if args.force and partial_dir.exists():
        shutil.rmtree(partial_dir)
        print(f"{cell} checkpoint: --force given, partials cleared", flush=True)
    fold_configs = load_or_make_configs(partial_dir, cell, learner, task, args.trials)

    trials, preds = [], []
    n_loaded = 0
    for fold in range(1, 6):
        te_mask = split["fold"].values == fold
        tr_mask = ~te_mask
        tr_idx = np.where(tr_mask)[0]
        loaded = load_valid_partial(partial_dir, cell, fold, split, patients, y_all, task,
                                    args.trials, fold_configs[fold])
        if loaded is not None:
            block_df, fold_trials = loaded
            preds.append(block_df)
            trials.extend(fold_trials)
            n_loaded += 1
            print(f"{cell} checkpoint: fold {fold} loaded from partial", flush=True)
            continue
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
        fold_trials = []
        best = None
        n_trials_loaded = 0
        for t in range(args.trials):
            cfg = fold_configs[fold][t]
            rec = load_trial_checkpoint(partial_dir, cell, fold, t, cfg)
            if rec is not None:
                n_trials_loaded += 1
            else:
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
                # persist the trial as soon as it finishes, so a killed process
                # loses at most the trial in flight, never the whole fold
                atomic_write_csv(pd.DataFrame([rec]),
                                 partial_dir / f"fold{fold}_trial{t}.csv",
                                 float_format="%.17g")
            fold_trials.append(rec)
            if rec["status"] == "ok" and (best is None or rec["inner_score"] > best[0]):
                best = (rec["inner_score"], cfg)
        if n_trials_loaded:
            print(f"{cell} checkpoint: fold {fold}: {n_trials_loaded} of {args.trials} "
                  f"trials loaded from per-trial checkpoints", flush=True)
        if best is None:
            raise RuntimeError(f"{cell} fold {fold}: all trials failed")
        cfg = best[1]
        t0 = time.time()
        p_te, proba, best_iter = fit_predict(learner, task, cfg, X_all[tr_idx], y_all[tr_idx],
                                             X_all[iva], y_all[iva], X_all[te_mask], SEED)
        fit_seconds = round(time.time() - t0, 1)
        fold_trials.append({"cell": cell, "fold": fold, "trial": "FINAL", "config": json.dumps(cfg),
                            "status": "ok", "inner_score": best[0], "seconds": fit_seconds,
                            "best_iteration": best_iter})
        block = {"patient": np.array(patients)[te_mask], "fold": fold,
                 "true_value": y_all[te_mask], "prediction": p_te}
        if proba is not None:
            for k in range(proba.shape[1]):
                block[f"proba_{k}"] = proba[:, k]
        block_df = pd.DataFrame(block)
        preds.append(block_df)
        trials.extend(fold_trials)
        # %.17g so a reloaded partial reproduces the in-memory floats bit-exactly
        atomic_write_csv(block_df, partial_dir / f"fold{fold}.csv", float_format="%.17g")
        atomic_write_csv(pd.DataFrame(fold_trials), partial_dir / f"fold{fold}_trials.csv",
                         float_format="%.17g")
        print(f"{cell} fold {fold}: inner best {best[0]:.4f}, final fit {fit_seconds}s", flush=True)
        print(f"{cell} checkpoint: fold {fold} written", flush=True)

    out = pd.concat(preds).sort_values(["fold", "patient"])
    out.insert(0, "task", task); out.insert(1, "representation", rep); out.insert(2, "learner", learner)
    out.insert(3, "seed", SEED)
    atomic_write_csv(out, pred_path)
    atomic_write_csv(pd.DataFrame(trials), trial_path)
    print(f"WROTE {pred_path} rows={len(out)} pooled score={score(task, out['true_value'].values, out['prediction'].values):.4f}")
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
        print(f"{cell} checkpoint: assembled final outputs "
              f"({n_loaded} of 5 folds loaded from partials), partials removed", flush=True)


if __name__ == "__main__":
    main()
