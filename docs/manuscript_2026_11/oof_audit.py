import pandas as pd, numpy as np, pathlib, collections, sys, json, re
root = pathlib.Path.home()/"workspace/projects/uga-maf-representation-paper"

def patient12(x):
    s=str(x)
    return s[:12] if s.startswith("TCGA-") else s

def check_oof(path, sample_col_candidates=("sample","tumour_barcode","sample_id","donor_id","patient_id","patient_id_12","patient","id")):
    try:
        df=pd.read_csv(path, low_memory=False, nrows=5)
        cols=list(df.columns)
    except Exception as e:
        return {"file":str(path.relative_to(root)), "error":str(e)}
    # need full read if sample+fold present
    sample_col=None
    for c in sample_col_candidates:
        if c in cols:
            sample_col=c; break
    if sample_col is None:
        # first column maybe?
        return {"file":str(path.relative_to(root)), "cols":cols[:12], "note":"no sample col found"}
    # read only needed cols
    fold_col="fold" if "fold" in cols else None
    usecols=[sample_col]+([fold_col] if fold_col else [])
    # grouping cols
    group_cols=[c for c in ["benchmark","endpoint","representation","learner","repeat","model","analysis","variant"] if c in cols]
    usecols+=group_cols
    try:
        df=pd.read_csv(path, usecols=usecols, low_memory=False, dtype=str)
    except Exception as e:
        return {"file":str(path.relative_to(root)), "error":str(e)}
    results=[]
    if not group_cols:
        group_cols=["_all"]
        df["_all"]="all"
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple): keys=(keys,)
        samples=g[sample_col].astype(str)
        patients=samples.map(patient12)
        # sample duplicates
        dup_sample = int((samples.value_counts()>1).sum())
        dup_patient = int((patients.value_counts()>1).sum())
        # patient across folds
        cross_fold_patients=0
        cross_fold_samples=0
        folds_per_patient=None
        if fold_col:
            pf=g.assign(_p=patients).groupby("_p")[fold_col].nunique()
            cross_fold_patients=int((pf>1).sum())
            sf=g.groupby(sample_col)[fold_col].nunique()
            cross_fold_samples=int((sf>1).sum())
        results.append({
            "group":dict(zip(group_cols, [str(k) for k in keys])),
            "n_rows":int(len(g)),
            "n_unique_sample":int(samples.nunique()),
            "n_unique_patient12_or_id":int(patients.nunique()),
            "dup_sample_ids":dup_sample,
            "dup_patient_ids":dup_patient,
            "patients_in_multiple_folds":cross_fold_patients,
            "samples_in_multiple_folds":cross_fold_samples,
            "sample_len_mode": int(samples.str.len().mode().iloc[0]) if len(samples) else None,
            "sample_examples": samples.head(3).tolist(),
            "n_folds": int(g[fold_col].nunique()) if fold_col else None,
        })
    return {"file":str(path.relative_to(root)), "sample_col":sample_col, "fold_col":fold_col, "groups":results}

files=[
 "results/tables/main_manuscript_complete_panel_oof_predictions.csv",
 "results/tables/task1_oof_maf_a_full.csv",
 "results/tables/task3_hrd_attribution_oof_sig3_only.csv",
 "results/tables/task3_hrd_attribution_oof_sig3_brca12.csv",
 "results/tables/task3_hrd_attribution_oof_brca12_only.csv",
 "results/tables/task4_centre_heldout_predictions.csv",
 "results/tables/patterns_alone_oof_HRD_Score.csv",
 "results/tables/patterns_alone_oof_cancer_type.csv",
 "results/tables/patterns_alone_oof_hrd_binary_33.csv",
 "results/tables/patterns_alone_oof_hrd_binary_42.csv",
]
# add muat oof files
files+= [str(p.relative_to(root)) for p in sorted(root.glob("results/tables/muat_*oof_predictions.csv"))]
files+= [str(p.relative_to(root)) for p in sorted(root.glob("results/tables/task1_oof_*.csv"))]
out=[]
for f in sorted(set(files)):
    print("CHECK",f, flush=True)
    out.append(check_oof(root/f))
print(json.dumps(out, indent=2))
pathlib.Path.home().joinpath("workspace/scratch/uga-maf-audit/oof_audit.json").write_text(json.dumps(out,indent=2))
