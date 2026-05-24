from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from exp23_config import WorkflowConfig
from exp23_utils import (
    canonical_96_channels,
    clean_object_columns,
    df_to_md_table,
    ensure_stage_dirs,
    make_subtype_columns,
    safe_numeric,
    sbs96_channel,
    write_json,
)


MAF_USECOLS = [
    "Tumor_Sample_Barcode",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
    "CONTEXT",
    "Chromosome",
    "Start_Position",
    "Hugo_Symbol",
    "Variant_Classification",
    "NCBI_Build",
]


def load_ddr_cohort(cfg: WorkflowConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.ddr_path, sep="	")
    df = clean_object_columns(df)
    ac = cfg.cohort_acronym
    sub = df[df["acronym"].astype(str).str.strip('"') == ac].copy()
    sub["patient_id_12"] = sub["patient_id"].astype(str).str[:12]
    return sub


def load_clinical_cohort(cfg: WorkflowConfig) -> pd.DataFrame | None:
    if cfg.clinical_path is None or not cfg.clinical_path.exists():
        return None
    df = pd.read_csv(
        cfg.clinical_path,
        sep="	",
        low_memory=False,
        encoding="latin-1",
        on_bad_lines="skip",
    )
    df = clean_object_columns(df)
    ac = cfg.cohort_acronym
    clin = df[df["acronym"].astype(str).str.strip('"') == ac].copy()
    clin["patient_id_12"] = clin["bcr_patient_barcode"].astype(str).str[:12]
    return clin


def load_maf_minimal(cfg: WorkflowConfig) -> pd.DataFrame:
    comp: str | None = "infer"
    if str(cfg.maf_path).lower().endswith(".maf"):
        comp = None
    try:
        return pd.read_csv(
            cfg.maf_path,
            sep="	",
            compression=comp,
            usecols=MAF_USECOLS,
            on_bad_lines="skip",
            low_memory=False,
        )
    except EOFError as e:
        raise RuntimeError(
            f"MAF gzip stream is incomplete or corrupt: {cfg.maf_path}\n"
            "If the file is on a cloud drive, download it fully (e.g. make it available offline) and retry."
        ) from e


def inspect_inputs(
    ddr_cohort: pd.DataFrame,
    clin_cohort: pd.DataFrame | None,
    maf_df: pd.DataFrame,
) -> dict:
    maf_sample_ids = maf_df["Tumor_Sample_Barcode"].dropna().astype(str)
    maf_patients_12 = set(maf_sample_ids.str[:12])
    ddr_patients_12 = set(ddr_cohort["patient_id_12"].dropna().astype(str))
    clin_patients_12 = (
        set(clin_cohort["patient_id_12"].dropna().astype(str)) if clin_cohort is not None else set()
    )
    overlap_maf_ddr = maf_patients_12 & ddr_patients_12
    overlap_all3 = overlap_maf_ddr & clin_patients_12 if clin_patients_12 else overlap_maf_ddr

    manifest = {
        "cohort_acronym": str(ddr_cohort["acronym"].iloc[0]).strip('"') if len(ddr_cohort) else "",
        "ddr_cohort_rows": int(len(ddr_cohort)),
        "clinical_cohort_rows": int(len(clin_cohort)) if clin_cohort is not None else 0,
        "maf_rows": int(len(maf_df)),
        "maf_unique_samples": int(maf_df["Tumor_Sample_Barcode"].nunique()),
        "maf_unique_patients_12": int(len(maf_patients_12)),
        "ddr_unique_patients_12": int(len(ddr_patients_12)),
        "clinical_unique_patients_12": int(len(clin_patients_12)),
        "overlap_maf_ddr": int(len(overlap_maf_ddr)),
        "overlap_all3": int(len(overlap_all3)),
        "ncbi_build": str(maf_df["NCBI_Build"].iloc[0]) if "NCBI_Build" in maf_df.columns and len(maf_df) else "unknown",
        "context_present": bool("CONTEXT" in maf_df.columns),
    }
    return manifest


def build_catalogs_from_maf(maf_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    total_rows = len(maf_df)
    
    # SBS
    snp = maf_df[maf_df["Variant_Type"] == "SNP"].copy()
    snp["sbs96"] = snp.apply(
        lambda row: sbs96_channel(
            str(row["Reference_Allele"]),
            str(row["Tumor_Seq_Allele2"]),
            str(row["CONTEXT"]),
        ),
        axis=1,
    )
    valid_sbs = snp.dropna(subset=["sbs96"]).copy()
    valid_sbs["patient_id_12"] = valid_sbs["Tumor_Sample_Barcode"].astype(str).str[:12]
    valid_sbs["sample_16"] = valid_sbs["Tumor_Sample_Barcode"].astype(str).str[:16]

    # DBS (DNPs)
    dnp = maf_df[maf_df["Variant_Type"] == "DNP"].copy()
    dnp["dbs78"] = [
        dbs78_channel(str(r), str(a)) 
        for r, a in zip(dnp["Reference_Allele"], dnp["Tumor_Seq_Allele2"])
    ]
    valid_dbs = dnp.dropna(subset=["dbs78"]).copy()
    valid_dbs["patient_id_12"] = valid_dbs["Tumor_Sample_Barcode"].astype(str).str[:12]
    valid_dbs["sample_16"] = valid_dbs["Tumor_Sample_Barcode"].astype(str).str[:16]

    # Combine for sample selection
    all_valid = pd.concat([valid_sbs, valid_dbs], ignore_index=True)
    sample_counts = all_valid.groupby(["patient_id_12", "sample_16"]).size().reset_index(name="n_muts")
    sample_counts["is_primary"] = sample_counts["sample_16"].str[13:15] == "01"

    best_rows = []
    for patient_id, group in sample_counts.groupby("patient_id_12"):
        primary = group[group["is_primary"]]
        best = primary.sort_values("n_muts", ascending=False).iloc[0] if len(primary) else group.sort_values("n_muts", ascending=False).iloc[0]
        best_rows.append(best)
    best_df = pd.DataFrame(best_rows)
    selected_samples = set(best_df["sample_16"].astype(str))

    # Build SBS Catalog
    p_sbs = valid_sbs[valid_sbs["sample_16"].isin(selected_samples)].copy()
    sbs_wide = p_sbs.groupby(["patient_id_12", "sbs96"]).size().unstack(fill_value=0)
    for c in canonical_96_channels():
        if c not in sbs_wide.columns: sbs_wide[c] = 0
    sbs_wide = sbs_wide[canonical_96_channels()]

    # Build DBS Catalog
    p_dbs = valid_dbs[valid_dbs["sample_16"].isin(selected_samples)].copy()
    dbs_wide = p_dbs.groupby(["patient_id_12", "dbs78"]).size().unstack(fill_value=0) if not p_dbs.empty else pd.DataFrame(index=sbs_wide.index)

    # Burden
    burden = pd.DataFrame(index=sbs_wide.index)
    burden["total_sbs"] = sbs_wide.sum(axis=1)
    burden["total_dbs"] = dbs_wide.sum(axis=1) if not dbs_wide.empty else 0
    burden = burden.reset_index().rename(columns={"index": "patient_id_12"})
    burden["patient_id"] = burden["patient_id_12"]
    burden["log10_burden"] = np.log10((burden["total_sbs"] + burden["total_dbs"]).clip(lower=1))

    harmonization = best_df[["patient_id_12", "sample_16", "n_muts", "is_primary"]].copy()
    harmonization.columns = ["patient_id_12", "selected_sample_16", "n_muts", "is_primary_tumor"]

    stats = {
        "total_maf_rows": int(total_rows),
        "snp_rows": int(len(snp)),
        "dnp_rows": int(len(dnp)),
        "valid_sbs": int(len(valid_sbs)),
        "valid_dbs": int(len(valid_dbs)),
        "mean_sbs": float(burden["total_sbs"].mean()),
        "mean_dbs": float(burden["total_dbs"].mean()),
    }
    return sbs_wide, dbs_wide, burden, harmonization, stats


def prepare_continuous_labels(ddr_cohort: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "patient_id_12",
        "HRD_Score",
        "HRD_TAI",
        "HRD_LST",
        "HRD_LOH",
        "PARPi7",
        "PARPi7_bin",
        "eCARD",
        "RPS",
        "tp53_score",
        "purity",
        "ploidy",
        "mutLoad_nonsilent",
    ]
    out = ddr_cohort[cols].copy()
    numeric_cols = [
        "HRD_Score",
        "HRD_TAI",
        "HRD_LST",
        "HRD_LOH",
        "PARPi7",
        "eCARD",
        "RPS",
        "tp53_score",
        "purity",
        "ploidy",
        "mutLoad_nonsilent",
    ]
    for col in numeric_cols:
        out[col] = safe_numeric(out[col])
    out = out.dropna(subset=["HRD_Score"]).drop_duplicates(subset=["patient_id_12"])
    return out


def prepare_binary_labels(cont: pd.DataFrame, thresholds: tuple[int, ...]) -> pd.DataFrame:
    out = cont[["patient_id_12", "HRD_Score", "PARPi7_bin"]].copy()
    for threshold in thresholds:
        out[f"hrd_binary_{threshold}"] = np.where(
            out["HRD_Score"] >= threshold,
            "HRD-high",
            "HRD-low",
        )
    q25 = out["HRD_Score"].quantile(0.25)
    q75 = out["HRD_Score"].quantile(0.75)
    out["hrd_binary_quartile"] = np.where(
        out["HRD_Score"] >= q75,
        "HRD-high",
        np.where(out["HRD_Score"] <= q25, "HRD-low", "ambiguous"),
    )
    parpi = safe_numeric(out["PARPi7_bin"])
    out["parpi7_binary"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out.loc[parpi == 1, "parpi7_binary"] = "PARPi-high"
    out.loc[parpi == 0, "parpi7_binary"] = "PARPi-low"
    return out


def prepare_clinical_covariates(clin_cohort: pd.DataFrame) -> pd.DataFrame:
    er_col = "breast_carcinoma_estrogen_receptor_status"
    pr_col = "breast_carcinoma_progesterone_receptor_status"
    her2_col = "lab_proc_her2_neu_immunohistochemistry_receptor_status"

    out = clin_cohort[["patient_id_12"]].copy()
    mapping = {
        er_col: "ER_status",
        pr_col: "PR_status",
        her2_col: "HER2_IHC_status",
    }
    for raw_col, clean_col in mapping.items():
        if raw_col in clin_cohort.columns:
            vals = clin_cohort[raw_col].copy()
            vals = vals.replace({
                "[Not Available]": np.nan,
                "[Not Evaluated]": np.nan,
                "[Performed but Not Available]": np.nan,
            })
            out[clean_col] = vals
    if "pathologic_stage" in clin_cohort.columns:
        out["stage"] = clin_cohort["pathologic_stage"].replace(
            {"[Not Available]": np.nan, "[Discrepancy]": np.nan}
        )
    return out.drop_duplicates(subset=["patient_id_12"])


def build_base_cohort(
    cont: pd.DataFrame,
    binary: pd.DataFrame,
    clin_cov: pd.DataFrame | None,
    burden: pd.DataFrame,
    cfg: WorkflowConfig,
) -> tuple[pd.DataFrame, dict]:
    eligible = set(burden["patient_id_12"].astype(str))
    cohort = cont[cont["patient_id_12"].isin(eligible)].copy()
    cohort = cohort.merge(
        burden[["patient_id_12", "total_sbs", "log10_burden"]],
        on="patient_id_12",
        how="left",
    )
    binary_cols = [
        "patient_id_12",
        *[f"hrd_binary_{t}" for t in cfg.hrd_thresholds],
        "hrd_binary_quartile",
        "parpi7_binary",
    ]
    cohort = cohort.merge(binary[binary_cols], on="patient_id_12", how="left")
    if clin_cov is not None and not clin_cov.empty:
        cohort = cohort.merge(clin_cov, on="patient_id_12", how="left")
    cohort = cohort.dropna(subset=["HRD_Score", "total_sbs"]).drop_duplicates(subset=["patient_id_12"])
    cohort = make_subtype_columns(cohort)

    stats = {
        "patients_with_hrd_scores": int(len(cont)),
        "patients_with_catalogs": int(len(eligible)),
        "base_analysis_cohort": int(len(cohort)),
    }
    for threshold in cfg.hrd_thresholds:
        label_col = f"hrd_binary_{threshold}"
        stats[f"{label_col}_high"] = int((cohort[label_col] == "HRD-high").sum())
        stats[f"{label_col}_low"] = int((cohort[label_col] == "HRD-low").sum())
    stats["quartile_extremes"] = int((cohort["hrd_binary_quartile"] != "ambiguous").sum())
    return cohort, stats


def _kv_table(title: str, d: dict) -> str:
    rows = [{"key": str(k), "value": d[k]} for k in d]
    return f"## {title}\n\n{df_to_md_table(pd.DataFrame(rows))}"


def write_prepare_reports(
    cfg: WorkflowConfig,
    input_manifest: dict,
    catalog_stats: dict,
    cohort_stats: dict,
    burden: pd.DataFrame,
) -> None:
    validation_lines = [
        "# EXP023 input validation",
        "",
        "## Core file checks",
        f"- Cohort acronym: {input_manifest.get('cohort_acronym', '')}",
        f"- DDR cohort rows: {input_manifest['ddr_cohort_rows']}",
        f"- Clinical cohort rows: {input_manifest['clinical_cohort_rows']}",
        f"- MAF rows: {input_manifest['maf_rows']:,}",
        f"- MAF unique patients (12-char): {input_manifest['maf_unique_patients_12']}",
        f"- MAF ∩ DDR (12-char): {input_manifest['overlap_maf_ddr']}",
        f"- Full overlap (incl. clinical if used): {input_manifest['overlap_all3']}",
        f"- CONTEXT present: {input_manifest['context_present']}",
        "",
        "## Barcode rule",
        "- Join on 12-character patient barcode.",
        "- For patients with multiple tumor samples, prefer primary tumor (`-01`), then highest SNV count.",
        "",
        _kv_table("Input validation (tabular)", input_manifest),
    ]
    (cfg.reports_dir / "input_validation.md").write_text("\n".join(validation_lines), encoding="utf-8")

    bq = burden["total_sbs"]
    burden_tbl = pd.DataFrame(
        [
            {"statistic": "min", "total_sbs": float(bq.min())},
            {"statistic": "Q25", "total_sbs": float(bq.quantile(0.25))},
            {"statistic": "median", "total_sbs": float(bq.median())},
            {"statistic": "Q75", "total_sbs": float(bq.quantile(0.75))},
            {"statistic": "max", "total_sbs": float(bq.max())},
        ]
    )
    prepare_lines = [
        "# EXP023 data preparation summary",
        "",
        "## Catalog construction",
        f"- SNP rows retained: {catalog_stats['snp_rows']:,}",
        f"- DNP rows retained: {catalog_stats['dnp_rows']:,}",
        f"- Valid SBS mutations: {catalog_stats['valid_sbs']:,}",
        f"- Valid DBS mutations: {catalog_stats['valid_dbs']:,}",
        f"- Mean SBS burden: {catalog_stats['mean_sbs']:.1f}",
        f"- Mean DBS burden: {catalog_stats['mean_dbs']:.1f}",
        "",
        _kv_table("Catalog manifest", catalog_stats),
        "",
        "## Cohort assembly",
        f"- Patients with HRD scores: {cohort_stats['patients_with_hrd_scores']}",
        f"- Patients with catalogs: {cohort_stats['patients_with_catalogs']}",
        f"- Base analysis cohort: {cohort_stats['base_analysis_cohort']}",
        "",
        _kv_table("Cohort manifest", cohort_stats),
        "",
        "## Burden distribution (total SBS)",
        "",
        df_to_md_table(burden_tbl),
    ]
    (cfg.reports_dir / "data_preparation.md").write_text("\n".join(prepare_lines), encoding="utf-8")


def run_prepare(cfg: WorkflowConfig) -> dict:
    ensure_stage_dirs(cfg)

    ddr_cohort = load_ddr_cohort(cfg)
    clin_cohort = load_clinical_cohort(cfg)
    maf_df = load_maf_minimal(cfg)
    ddr_ids = set(ddr_cohort["patient_id_12"].astype(str))
    maf_df = maf_df[maf_df["Tumor_Sample_Barcode"].astype(str).str[:12].isin(ddr_ids)].copy()
    if maf_df.empty:
        raise RuntimeError(
            f"No MAF rows for DDR acronym={cfg.cohort_acronym} after 12-char barcode filter. "
            "Non-BRCA cohorts need a MAF that actually contains those tumor barcodes "
            "(e.g. TCGA-OV uses prefixes like TCGA-04/09/24/29/…; a file named allCohortMAF may still be breast-only). "
            "Use a GDC TCGA-OV open-access MAF or a true pan-cohort MAF that includes OV participants."
        )

    input_manifest = inspect_inputs(ddr_cohort, clin_cohort, maf_df)
    write_json(cfg.metadata_dir / "input_validation.json", input_manifest)

    sbs_wide, dbs_wide, burden, harmonization, catalog_stats = build_catalogs_from_maf(maf_df)
    if len(sbs_wide) == 0:
        raise RuntimeError(
            f"No SBS96 patient catalogs for acronym={cfg.cohort_acronym}."
        )
    sbs_wide.to_csv(cfg.catalogs_dir / "sbs96_counts.tsv", sep="	")
    if not dbs_wide.empty:
        dbs_wide.to_csv(cfg.catalogs_dir / "dbs78_counts.tsv", sep="	")
    burden.to_csv(cfg.catalogs_dir / "sample_burden_summary.tsv", sep="	", index=False)
    harmonization.to_csv(cfg.catalogs_dir / "barcode_harmonization.tsv", sep="	", index=False)
    write_json(cfg.metadata_dir / "catalog_manifest.json", catalog_stats)

    cont = prepare_continuous_labels(ddr_cohort)
    binary = prepare_binary_labels(cont, cfg.hrd_thresholds)
    clin_cov = prepare_clinical_covariates(clin_cohort) if clin_cohort is not None else None

    cont.to_csv(cfg.labels_dir / "hrd_labels_continuous.tsv", sep="	", index=False)
    binary.to_csv(cfg.labels_dir / "hrd_labels_binary.tsv", sep="	", index=False)
    if clin_cov is not None:
        clin_cov.to_csv(cfg.labels_dir / "clinical_covariates.tsv", sep="	", index=False)

    base_cohort, cohort_stats = build_base_cohort(cont, binary, clin_cov, burden, cfg)
    base_cohort.to_csv(cfg.cohort_dir / "base_analysis_cohort.tsv", sep="	", index=False)
    write_json(cfg.metadata_dir / "cohort_manifest.json", cohort_stats)

    write_prepare_reports(cfg, input_manifest, catalog_stats, cohort_stats, burden)

    return {
        "input_manifest": input_manifest,
        "catalog_stats": catalog_stats,
        "cohort_stats": cohort_stats,
    }
