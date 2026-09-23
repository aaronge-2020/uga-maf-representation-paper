#!/usr/bin/env python3
"""Batch-center (sequencing-center) confounding analysis.

Reproduces the "Batch effects and confounding" analysis of the manuscript:
  (i)   eta-squared of sequencing center for six per-tumor summaries, across
        centers and within cancer types;
  (ii)  multinomial logistic regression predicting the four largest centers
        from the six summaries (5-fold stratified CV) vs majority baseline;
  (iii) within-type center comparisons for OV, KIRC, KIRP.

Inputs (all paths relative to repo root):
  results/tables/quick_bio_v4_maf_features_910.csv.gz
      910-feature matrix (sample + 910 features).
  results/tables/batch_center_sample_mapping.csv
      Per-tumor-sample mapping: sample (patient barcode) -> full-length MC3
      Tumor_Sample_Barcode, sequencing center (final two barcode characters),
      cancer-type labels from GDC (project_id) and from the TCGA CDR
      (Supplemental Table S1). 10,226 rows: one per tumor-sample barcode.
      Two patients (TCGA-DV-A4W0, KIRC; TCGA-UZ-A9PS, KIRP) have two tumor
      aliquots each in the MC3 MAF, both at center 10, so they contribute
      two rows. All other patients contribute one row.

Outputs:
  results/tables/batch_center_etasq.csv
  results/tables/batch_center_within_type.csv
  results/tables/batch_center_classification.csv

Method notes (see docs/batch_center_analysis.md for full provenance):
  * The six summaries are modifier fraction, 3' UTR fraction, indel fraction,
    log10 SBS burden, mean VAF, max VAF. Indel fraction uses the stored
    `id_fraction` column: it reproduces the manuscript's eta-squared values,
    whereas re-deriving 10**log burdens from the log columns does not
    (documented deviation from the manuscript's "derived from burden
    logs" phrasing).
  * Eta-squared is SS_between / SS_total from a one-way ANOVA on center.
  * Analysis unit: the across-centers eta-squared (i) and the center
    classifier (ii) are patient-level (one row per patient: n = 8,858 and
    n = 10,206), which reproduces the manuscript's cohort sizes and the
    69.7% / 58.3% classifier numbers exactly. The within-type comparisons
    (iii) count tumor-sample barcodes (n = 1,051): the two dual-aliquot
    patients above are counted twice, which is the only reading that
    reproduces the manuscript's KIRC (08:207, 10:163) and KIRP (08:114,
    10:168) center splits exactly.
  * "Median/maximum within type" are computed over the three eligible
    within-type comparisons (OV, KIRC, KIRP; largest two centers each), which
    is the only reading that reproduces the manuscript's medians and maxima.
  * The center classifier is fit on ALL patients (not only top-20),
    restricted to the four largest centers; this reproduces the 58.3%
    baseline and 69.7% CV accuracy exactly.
"""

import csv
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(REPO, "results", "tables", "quick_bio_v4_maf_features_910.csv.gz")
MAPPING = os.path.join(REPO, "results", "tables", "batch_center_sample_mapping.csv")
OUT_ETASQ = os.path.join(REPO, "results", "tables", "batch_center_etasq.csv")
OUT_WITHIN = os.path.join(REPO, "results", "tables", "batch_center_within_type.csv")
OUT_CLASSIF = os.path.join(REPO, "results", "tables", "batch_center_classification.csv")

SUMMARIES = [
    ("modifier_frac", "vep_impact_fraction__modifier", "Modifier fraction"),
    ("utr3_frac", "vep_consequence_fraction__3_prime_utr_variant", "3' UTR fraction"),
    ("indel_frac", "id_fraction", "Indel fraction"),
    ("log_sbs", "log10_sbs_burden", "Log SBS burden"),
    ("mean_vaf", "vaf_mean", "Mean VAF"),
    ("max_vaf", "vaf_max", "Max VAF"),
]
TOP20 = ["BRCA", "UCEC", "LUAD", "LGG", "HNSC", "PRAD", "THCA",
         "LUSC", "SKCM", "STAD", "BLCA", "OV", "COAD", "GBM",
         "KIRC", "LIHC", "CESC", "KIRP", "SARC", "ESCA"]
ELIGIBLE_TYPES = ["OV", "KIRC", "KIRP"]
CENTER_NAMES = {"08": "Broad Institute", "09": "Washington University",
                "10": "Baylor", "32": "HudsonAlpha", "01": "Other"}

# Manuscript values: (across, median_within, max_within, max_type)
MANUSCRIPT = {
    "modifier_frac": (0.293, 0.134, 0.292, "KIRP"),
    "utr3_frac": (0.308, 0.158, 0.180, "KIRP"),
    "indel_frac": (0.033, 0.024, 0.045, "KIRP"),
    "log_sbs": (0.003, 0.002, 0.033, "KIRC"),
    "mean_vaf": (0.035, 0.011, 0.063, "KIRP"),
    "max_vaf": (0.021, 0.002, 0.007, "KIRP"),
}
# Manuscript within-type center splits: type -> (n, {center: n})
MANUSCRIPT_WITHIN = {
    "OV": (399, {"08": 206, "09": 193}),
    "KIRC": (370, {"08": 207, "10": 163}),
    "KIRP": (282, {"08": 114, "10": 168}),
}


def eta_squared(values, groups):
    """One-way ANOVA eta-squared: SS_between / SS_total."""
    x = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    grand = x.mean()
    ss_total = ((x - grand) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = sum(
        ((x[g == k].mean() - grand) ** 2) * (g == k).sum()
        for k in np.unique(g)
    )
    return float(ss_between / ss_total)


def main():
    for p in (MATRIX, MAPPING):
        if not os.path.isfile(p):
            sys.exit(f"missing input: {p}")

    usecols = ["sample"] + [c for _, c, _ in SUMMARIES]
    df = pd.read_csv(MATRIX, usecols=usecols)
    df = df.rename(columns={c: s for s, c, _ in SUMMARIES})
    mapping = pd.read_csv(MAPPING, dtype=str)
    # one-to-many: two patients have two tumor-sample barcodes each
    df = df.merge(mapping[["sample", "center", "cancer_type_gdc"]],
                  on="sample", how="left", validate="one_to_many")
    assert df["center"].notna().all(), "samples without center barcode"
    df = df.rename(columns={"cancer_type_gdc": "cancer_type"})
    summaries = [s for s, _, _ in SUMMARIES]

    # Patient-level view (one row per patient) for the across-centers
    # eta-squared and the classifier.
    df_patient = df.drop_duplicates("sample").copy()
    # Tumor-sample (barcode) view for the within-type comparisons.
    # Cohort definition (shared): top-20 cancer types by GDC project label.
    # TCGA-36-2539 is excluded: GDC labels it OV, but it is absent from the
    # TCGA CDR (the manuscript's cited clinical source); excluding it gives
    # n=8858 and OV n=399, exactly the manuscript's reported cohort sizes.
    cohort_patients = df_patient[
        df_patient["cancer_type"].isin(TOP20)
        & (df_patient["sample"] != "TCGA-36-2539")
    ]["sample"]
    cohort = df_patient[df_patient["sample"].isin(cohort_patients)].copy()
    cohort_bc = df[df["sample"].isin(cohort_patients)].copy()
    print(f"top-20 cohort: {len(cohort)} patients, {len(cohort_bc)} tumor samples "
          f"(manuscript: 8858; TCGA-36-2539 excluded, see docs)")

    # ---- (i) eta-squared across centers (patient-level) ----
    etasq_rows = []
    within_for_med = {}
    for key, _, label in SUMMARIES:
        e_across = eta_squared(cohort[key].values, cohort["center"].values)
        within = {}
        for t in ELIGIBLE_TYPES:
            dd = cohort_bc[cohort_bc["cancer_type"] == t]
            top2 = dd["center"].value_counts().nlargest(2).index
            dd2 = dd[dd["center"].isin(top2)]
            within[t] = eta_squared(dd2[key].values, dd2["center"].values)
        within_for_med[key] = within
        e_med = float(np.median(list(within.values())))
        t_max = max(within, key=within.get)
        m = MANUSCRIPT[key]
        etasq_rows.append({
            "summary": label,
            "eta2_across_centers": round(e_across, 3),
            "eta2_median_within_type": round(e_med, 3),
            "eta2_max_within_type": round(within[t_max], 3),
            "eta2_max_type": t_max,
            "manuscript_across": m[0], "manuscript_median": m[1],
            "manuscript_max": m[2], "manuscript_max_type": m[3],
        })
        print(f"  {label}: across={e_across:.3f} (ms {m[0]}) "
              f"median_within={e_med:.3f} (ms {m[1]}) "
              f"max_within={within[t_max]:.3f} {t_max} (ms {m[2]} {m[3]})")
    pd.DataFrame(etasq_rows).to_csv(OUT_ETASQ, index=False)
    print(f"wrote {OUT_ETASQ}")

    # ---- (iii) within-type eligible comparisons (tumor-sample level) ----
    within_rows = []
    for t in ELIGIBLE_TYPES:
        dd = cohort_bc[cohort_bc["cancer_type"] == t]
        top2 = dd["center"].value_counts().nlargest(2).index.tolist()
        dd2 = dd[dd["center"].isin(top2)]
        cc = dd2["center"].value_counts()
        row = {"cancer_type": t, "n": len(dd2)}
        for c in sorted(top2):
            row[f"center_{c}_n"] = int(cc[c])
            row[f"center_{c}_name"] = CENTER_NAMES.get(c, c)
        for key, _, label in SUMMARIES:
            row[f"eta2_{key}"] = round(
                eta_squared(dd2[key].values, dd2["center"].values), 3)
        # manuscript check
        ms_n, ms_cc = MANUSCRIPT_WITHIN[t]
        ok = (len(dd2) == ms_n
              and all(int(cc[c]) == ms_cc[c] for c in ms_cc))
        row["matches_manuscript"] = ok
        within_rows.append(row)
        print(f"  {t}: n={len(dd2)} (ms {ms_n}), " +
              ", ".join(f"{c}={cc[c]} (ms {ms_cc[c]})" for c in top2) +
              f" -> {'MATCH' if ok else 'MISMATCH'}")
    within_cols = (["cancer_type", "n"]
                   + [f"center_{c}_{s}" for c in ("08", "09", "10")
                      for s in ("n", "name")]
                   + [f"eta2_{s}" for s, _, _ in SUMMARIES]
                   + ["matches_manuscript"])
    pd.DataFrame(within_rows).reindex(columns=within_cols).to_csv(OUT_WITHIN, index=False)
    print(f"wrote {OUT_WITHIN}")

    # ---- (ii) center classification: patient-level, four largest centers ----
    dfp = df_patient
    top4 = dfp["center"].value_counts().nlargest(4).index.tolist()
    d4 = dfp[dfp["center"].isin(top4)].copy()
    majority = d4["center"].value_counts().iloc[0] / len(d4)
    X = StandardScaler().fit_transform(d4[summaries].values)
    y = d4["center"].values
    clf = LogisticRegression(max_iter=5000, solver="lbfgs")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    acc = float(cross_val_score(clf, X, y, cv=cv).mean())
    print(f"  centers={top4} n={len(d4)} baseline={majority:.3f} (ms 0.583) "
          f"CV accuracy={acc:.3f} (ms 0.697)")
    pd.DataFrame([{
        "model": "multinomial_logistic_regression_lbfgs_standardized_5fold_stratified",
        "n_samples": len(d4),
        "centers": ";".join(top4),
        "cv_accuracy": round(acc, 3),
        "majority_baseline": round(majority, 3),
        "manuscript_cv_accuracy": 0.697,
        "manuscript_majority_baseline": 0.583,
    }]).to_csv(OUT_CLASSIF, index=False)
    print(f"wrote {OUT_CLASSIF}")


if __name__ == "__main__":
    main()
