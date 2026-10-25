"""Build R2-R5 gene feature matrices from the repo MuAt-compatible events files.

Representations (protocol v1.0, eligibility amended pre-execution, see deviations log):
  R2 binary presence per gene, R3 event counts per gene,
  R4 counts per gene x consequence group (missense / truncating / other protein-affecting),
  on (a) the 401-gene Bio MAF panel and (b) all genes observed (R2/R3 only).
Eligible events: protein-affecting Variant_Classification values only (Silent excluded).
Events are deduplicated on (sample, chromosome, position, ref, alt) before counting.
TF-IDF (R5) is a transform of R3 counts fitted inside training partitions at run time,
so only count matrices are stored here.
Also writes the single canonical outer split used by every new run (protocol amendment:
the archived OOF file carries a different random partition per representation, verified
2026-10-25, so no shared canonical split exists to reuse).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[3]
NS = ROOT / "analyses" / "representation_extension"
SEED = 20261025

PROTEIN_AFFECTING = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "In_Frame_Del", "In_Frame_Ins", "Splice_Site", "Nonstop_Mutation",
    "Translation_Start_Site",
}
GROUPS = {
    "missense": {"Missense_Mutation"},
    "truncating": {"Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site"},
    "other_pa": PROTEIN_AFFECTING - {"Missense_Mutation", "Nonsense_Mutation",
                                     "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site"},
}

EVENTS = {
    "cancer": ROOT / "results/tables/muat_style_tcga_comparator_cancer_type_top20_faithful_fixed_arch_muat_compatible_events.tsv.gz",
    "hrd": ROOT / "results/tables/muat_style_tcga_comparator_ab_d_faithful_muat_compatible_events.tsv.gz",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def panel_genes():
    g = pd.read_csv(ROOT / "results/manuscript/tables/technical/table_s5_bio_maf_feature_guide_technical.csv")
    feats = g.loc[g["Bio MAF v4 block"] == "exact_consensus_driver_gene_identity", "Feature"]
    genes = sorted({f.rsplit("__", 1)[1] for f in feats})
    assert len(genes) == 401, len(genes)
    return genes


def load_events(path):
    ev = pd.read_csv(path, sep="\t", dtype=str,
                     usecols=["sample", "chromosome", "position", "variant_classification", "gene", "ref", "alt"])
    ev["patient"] = ev["sample"].str.slice(0, 12)
    n_raw = len(ev)
    ev = ev.drop_duplicates(subset=["sample", "chromosome", "position", "ref", "alt"])
    ev = ev[ev["variant_classification"].isin(PROTEIN_AFFECTING)].copy()
    return ev, n_raw


def count_matrix(ev, patients, genes):
    idx = pd.Index(patients)
    gidx = pd.Index(genes)
    sub = ev[ev["gene"].isin(gidx)]
    ct = sub.groupby(["patient", "gene"]).size().unstack(fill_value=0)
    ct = ct.reindex(index=idx, columns=gidx, fill_value=0)
    return ct.astype(np.int32)


def main():
    genes_panel = panel_genes()
    (NS / "features").mkdir(parents=True, exist_ok=True)
    (NS / "config").mkdir(parents=True, exist_ok=True)
    pd.Series(genes_panel).to_csv(NS / "config" / "panel_genes_401.csv", index=False, header=["gene"])

    oof = pd.read_csv(ROOT / "results/tables/main_manuscript_complete_panel_oof_predictions.csv",
                      usecols=["endpoint", "representation", "learner", "sample", "true_value"],
                      dtype={"sample": str})
    cancer_lab = (oof[(oof.endpoint == "cancer_type_top20") &
                      (oof.representation == "signatures_plus_MAF_stack") &
                      (oof.learner == "xgboost")]
                  .drop_duplicates("sample").set_index("sample")["true_value"].astype(int))

    cohort = pd.read_csv(ROOT / "data/task3/final_analysis_cohort.tsv", sep="\t")
    hrd_lab = cohort.set_index("patient_id_12")["HRD_Score"]

    manifest_rows = []
    for task, labels in [("cancer", cancer_lab), ("hrd", hrd_lab)]:
        patients = sorted(labels.index)
        ev, n_raw = load_events(EVENTS[task])
        ev_patients = sorted(ev["patient"].unique())
        missing = sorted(set(patients) - set(ev_patients))
        extra = sorted(set(ev_patients) - set(patients))
        print(f"{task}: label patients={len(patients)} event patients={len(ev_patients)} "
              f"missing={len(missing)} extra={len(extra)} raw_events={n_raw} eligible={len(ev)}")
        counts_panel = count_matrix(ev, patients, genes_panel)
        counts_panel.to_csv(NS / "features" / f"{task}_r3_counts_panel.csv.gz", compression="gzip")
        (counts_panel > 0).astype(np.int8).to_csv(
            NS / "features" / f"{task}_r2_binary_panel.csv.gz", compression="gzip")
        # R4: per gene x consequence group
        r4 = {}
        for gname, gset in GROUPS.items():
            sub = ev[ev["variant_classification"].isin(gset)]
            r4[gname] = count_matrix(sub, patients, genes_panel)
        r4df = pd.concat({g: r4[g] for g in GROUPS}, axis=1)
        r4df.columns = [f"{gene}__{grp}" for grp, gene in r4df.columns]
        r4df = r4df.reindex(columns=sorted(r4df.columns))
        r4df.to_csv(NS / "features" / f"{task}_r4_consequence_panel.csv.gz", compression="gzip")
        all_genes = sorted(ev["gene"].unique())
        counts_all = count_matrix(ev, patients, all_genes)
        counts_all.to_csv(NS / "features" / f"{task}_r3_counts_allgenes.csv.gz", compression="gzip")
        (counts_all > 0).astype(np.int8).to_csv(
            NS / "features" / f"{task}_r2_binary_allgenes.csv.gz", compression="gzip")
        # canonical split
        if task == "cancer":
            strat = labels.loc[patients].values
        else:
            strat = pd.qcut(labels.loc[patients].values, 5, labels=False)
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        fold = np.zeros(len(patients), dtype=int)
        for k, (_, te) in enumerate(skf.split(np.zeros(len(patients)), strat), start=1):
            fold[te] = k
        sdf = pd.DataFrame({"patient": patients, "fold": fold,
                            "label": labels.loc[patients].values})
        sdf.to_csv(NS / "config" / f"canonical_split_{task}.csv", index=False)
        manifest_rows.append({
            "task": task, "n_patients": len(patients), "n_label_only_no_events": len(missing),
            "n_event_only": len(extra), "raw_events": n_raw, "eligible_events": int(len(ev)),
            "events_sha256": sha256(EVENTS[task]),
            "split_sha256": sha256(NS / "config" / f"canonical_split_{task}.csv"),
            "seed": SEED,
        })
    pd.DataFrame(manifest_rows).to_csv(NS / "config" / "feature_build_manifest.csv", index=False)
    print(json.dumps(manifest_rows, indent=1))


if __name__ == "__main__":
    main()
