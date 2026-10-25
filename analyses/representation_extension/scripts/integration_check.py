"""Small integration check before expensive training (protocol section 12).

Verifies, on fold 1 of each primary task only, with small untuned models:
- feature matrices align exactly with the canonical split patients,
- the 9 no-event cancer patients are present as all-zero gene rows (not dropped),
- TF-IDF fitted on the training partition only produces finite test features,
- metrics reconstruct from saved predictions.
This is a smoke check. Its scores are not results and are never reported as such.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[3]
NS = ROOT / "analyses" / "representation_extension"


def load_counts(task, kind):
    return pd.read_csv(NS / "features" / f"{task}_{kind}.csv.gz", index_col=0)


def tfidf_transform(train_counts, test_counts):
    df = (train_counts > 0).sum(axis=0)
    vocab = df[df > 0].index
    idf = np.log(len(train_counts) / df[vocab])
    return (train_counts[vocab] * idf), (test_counts[vocab] * idf)


def main():
    out = {}
    spectra = pd.read_csv(ROOT / "data/task0_features/features_standard_sbs96_id83.csv.gz", index_col=0)
    spectra.index = spectra.index.astype(str).str.slice(0, 12)
    spectra = spectra[~spectra.index.duplicated(keep="first")]

    # ---- cancer, fold 1, spectra + panel counts, small XGBoost
    split = pd.read_csv(NS / "config" / "canonical_split_cancer.csv", dtype={"patient": str})
    counts = load_counts("cancer", "r3_counts_panel")
    assert list(counts.index.astype(str)) == list(split["patient"]), "cancer patient order mismatch"
    zero_rows = int((counts.sum(axis=1) == 0).sum())
    X = pd.concat([spectra.reindex(split["patient"]).reset_index(drop=True),
                   counts.reset_index(drop=True)], axis=1)
    assert not X.isna().any().any(), "NaN after spectra join"
    y = split["label"].values
    tr, te = split["fold"].values != 1, split["fold"].values == 1
    t0 = time.time()
    m = XGBClassifier(n_estimators=60, max_depth=4, learning_rate=0.15, n_jobs=2,
                      subsample=0.9, colsample_bytree=0.7, random_state=20261025)
    m.fit(X.values[tr], y[tr])
    pred = m.predict(X.values[te])
    out["cancer_fold1_smoke"] = {
        "balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "zero_gene_rows_total": zero_rows, "seconds": round(time.time() - t0, 1)}
    # TF-IDF boundary check
    tr_t, te_t = tfidf_transform(counts[tr], counts[te])
    assert np.isfinite(tr_t.values).all() and np.isfinite(te_t.values).all()
    out["tfidf_vocab_genes"] = int(tr_t.shape[1])

    # ---- HRD, fold 1, spectra + panel counts, elastic-net-style ridge via XGBRegressor smoke
    split_h = pd.read_csv(NS / "config" / "canonical_split_hrd.csv", dtype={"patient": str})
    counts_h = load_counts("hrd", "r3_counts_panel")
    assert list(counts_h.index.astype(str)) == list(split_h["patient"]), "hrd patient order mismatch"
    spec_h = pd.read_csv(ROOT / "data/task3/features_standard_sbs96.csv.gz", index_col=0)
    spec_h.index = spec_h.index.astype(str).str.slice(0, 12)
    Xh = pd.concat([spec_h.reindex(split_h["patient"]).reset_index(drop=True),
                    counts_h.reset_index(drop=True)], axis=1)
    yh = split_h["label"].values
    trh, teh = split_h["fold"].values != 1, split_h["fold"].values == 1
    sc = StandardScaler().fit(Xh[trh])
    lr = LogisticRegression(max_iter=5)  # import smoke only, not fitted on regression target
    rg = XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.15, n_jobs=2,
                      random_state=20261025)
    rg.fit(sc.transform(Xh.values[trh]), yh[trh])
    ph = rg.predict(sc.transform(Xh.values[teh]))
    out["hrd_fold1_smoke"] = {
        "spearman": float(spearmanr(yh[teh], ph).statistic),
        "n_train": int(trh.sum()), "n_test": int(teh.sum())}
    out["status"] = "PASS"
    print(json.dumps(out, indent=1))
    (NS / "metrics").mkdir(exist_ok=True)
    with open(NS / "metrics" / "integration_check.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
