#!/usr/bin/env python3
"""Stage 7: score frozen Sig+MAF model on PCAWG 913; bootstrap metrics.

- Verifies model.ubj sha256 against manifest.
- Predicts with the frozen booster (predict.py logic).
- Metrics: accuracy, balanced accuracy, macro-F1, top-3 accuracy, per-type recall,
  each with 95% bootstrap CI (n=2000, seed 42).
- Compares vs spectra-only PCAWG (acc 0.479, bal-acc 0.365) and internal TCGA CV (0.738).
Saves predictions/pcawg_sigmaf_predictions.csv and predictions/pcawg_sigmaf_metrics.json
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

WORK = Path("/home/hatch/workspace/pcawg_sigmaf_transfer")
MODEL = Path("/home/hatch/workspace/somatic-mutation-repo/results/frozen_models/cancer_type_top20__signatures_plus_MAF_stack")
SEED = 42
N_BOOT = 2000

# 1. verify sha256
manifest = json.loads((MODEL / "manifest.json").read_text())
expected = manifest["provenance"]["model_sha256"]
h = hashlib.sha256((MODEL / "model.ubj").read_bytes()).hexdigest()
print(f"sha256 match: {h == expected}", flush=True)
assert h == expected, f"sha256 mismatch: {h} vs {expected}"

# 2. load features + labels
X = pd.read_csv(WORK / "features" / "pcawg_sigmaf_1130_features.csv.gz", index_col=0)
X.index = X.index.astype(str)
smap = pd.read_csv(WORK / "raw" / "pcawg_913_sample_mapping.csv")
smap = smap[smap["included"] == True].copy()
smap["sample_id"] = smap["sample_id"].astype(str)
y_true = smap.set_index("sample_id")["true_label"]
X = X.loc[y_true.index]
print(f"X={X.shape}", flush=True)

# 3. predict (mirror predict.py: booster is positional, pass numpy in order)
order = [l.strip() for l in (MODEL / "feature_order.txt").read_text().splitlines() if l.strip()]
X = X.reindex(columns=order)
bst = xgb.Booster()
bst.load_model(str(MODEL / "model.ubj"))
proba = bst.predict(xgb.DMatrix(X.to_numpy(dtype=np.float32)))
classes = manifest["class_order"]
print(f"proba={proba.shape} n_classes={len(classes)}", flush=True)
assert proba.shape == (len(X), len(classes))
pred_idx = proba.argmax(axis=1)
y_pred = [classes[i] for i in pred_idx]
# top-3
top3_idx = np.argsort(-proba, axis=1)[:, :3]
top3 = [[classes[i] for i in row] for row in top3_idx]

pred_df = pd.DataFrame({
    "sample_id": X.index,
    "true_label": y_true.values,
    "pred_label": y_pred,
    "top3": [";".join(t) for t in top3],
})
pred_df.to_csv(WORK / "predictions" / "pcawg_sigmaf_predictions.csv", index=False)

# 4. metrics
yt = y_true.values
yp = np.array(y_pred)
acc = accuracy_score(yt, yp)
bal = balanced_accuracy_score(yt, yp)
macro_f1 = f1_score(yt, yp, average="macro")
top3_labels = np.array([[classes[i] for i in row] for row in top3_idx])
top3_acc = np.mean([t in row for t, row in zip(yt, top3_labels)])
per_type_recall = {}
for cls in sorted(set(yt)):
    m = yt == cls
    per_type_recall[cls] = float(np.mean(yp[m] == cls))

# 5. bootstrap CIs
rng = np.random.default_rng(SEED)
n = len(yt)
boot = {"acc": [], "bal": [], "macro_f1": [], "top3": []}
for b in range(N_BOOT):
    idx = rng.integers(0, n, n)
    ytb, ypb = yt[idx], yp[idx]
    boot["acc"].append(accuracy_score(ytb, ypb))
    boot["bal"].append(balanced_accuracy_score(ytb, ypb))
    boot["macro_f1"].append(f1_score(ytb, ypb, average="macro"))
    boot["top3"].append(np.mean([t in row for t, row in zip(ytb, top3_labels[idx])]))
ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}

metrics = {
    "n_samples": n,
    "accuracy": float(acc), "accuracy_ci95": ci["acc"],
    "balanced_accuracy": float(bal), "balanced_accuracy_ci95": ci["bal"],
    "macro_f1": float(macro_f1), "macro_f1_ci95": ci["macro_f1"],
    "top3_accuracy": float(top3_acc), "top3_accuracy_ci95": ci["top3"],
    "per_type_recall": per_type_recall,
    "sha256_ok": True,
    "comparators": {
        "spectra_only_pcawg_accuracy": 0.479,
        "spectra_only_pcawg_balanced_accuracy": 0.365,
        "internal_tcga_cv_accuracy": 0.738,
    },
}
(WORK / "predictions" / "pcawg_sigmaf_metrics.json").write_text(json.dumps(metrics, indent=1))
print(json.dumps({k: v for k, v in metrics.items() if k != "per_type_recall"}, indent=1), flush=True)
print("SCORE_DONE", flush=True)
