"""Canonical outer splits from the repo convention: the main panel OOF file,
filtered to representation=signatures_plus_MAF_stack, learner=xgboost, per endpoint.
This is the triple every MuAt config names as the canonical split source.
Supersedes the seed-20261025 StratifiedKFold split (protocol amendment v1.1)."""
import hashlib, json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
NS = ROOT / "analyses" / "representation_extension"
oof = pd.read_csv(ROOT / "results/tables/main_manuscript_complete_panel_oof_predictions.csv",
                  usecols=["endpoint", "representation", "learner", "sample", "true_value", "fold"],
                  dtype={"sample": str})
cohort = pd.read_csv(ROOT / "data/task3/final_analysis_cohort.tsv", sep="\t")
hrd_labels = cohort.set_index("patient_id_12")["HRD_Score"]
out = {}
for task, ep in [("cancer", "cancer_type_top20"), ("hrd", "HRD_Score")]:
    sub = oof[(oof.endpoint == ep) & (oof.representation == "signatures_plus_MAF_stack")
              & (oof.learner == "xgboost")].drop_duplicates("sample")
    assert sub["sample"].is_unique
    df = pd.DataFrame({"patient": sub["sample"].values, "fold": sub["fold"].values.astype(int)})
    if task == "cancer":
        df["label"] = sub["true_value"].values.astype(int)
    else:
        df["label"] = hrd_labels.reindex(df["patient"]).values
        assert not df["label"].isna().any()
        assert (df.set_index("patient")["label"].sort_index()
                == pd.Series(sub.set_index("sample")["true_value"]).sort_index()).all(), "HRD label mismatch OOF vs cohort"
    df = df.sort_values("patient").reset_index(drop=True)
    p = NS / "config" / f"canonical_split_{task}.csv"
    df.to_csv(p, index=False)
    out[task] = {"n": len(df), "fold_sizes": df["fold"].value_counts().sort_index().to_dict(),
                 "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
                 "source": "main_manuscript_complete_panel_oof_predictions.csv @ signatures_plus_MAF_stack/xgboost"}
    print(task, out[task])
json.dump(out, open(NS / "config" / "split_provenance.json", "w"), indent=1)
