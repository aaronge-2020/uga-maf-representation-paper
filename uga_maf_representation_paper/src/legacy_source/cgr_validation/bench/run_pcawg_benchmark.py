#!/usr/bin/env python3
import argparse
import json
import time
import warnings
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.spatial.distance import cosine
from scipy.stats import friedmanchisquare, pearsonr, wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from uga_atlas import (  # noqa: E402
    FastaReader,
    build_bicgr52_ref as _uga_build_bicgr52_ref,
    build_standard_ref as _uga_build_standard_ref,
    build_universal_ref as _uga_build_universal_ref,
    encode_variant_universal,
    get_uga_model,
    load_context_atlas as _uga_load_atlas,
    normalize_rows_l1,
    sig_columns as _uga_sig_columns,
    stack_channel_matrix as _stack_channel_matrix,
    universal_context_slice as _uga_universal_context_slice,
    universal_fdim as _uga_universal_fdim,
    variant_vec as _uga_variant_vec,
    variant_vec_bicgr52_context_only as _uga_variant_vec_bicgr52_context_only,
    weighted_patient_profiles,
)


REPO = Path(__file__).resolve().parents[1]
PCAWG = REPO / "cgr_validation_results/research/data/pancan_pcawg_2020"
MUTATIONS = PCAWG / "data_mutations.txt"
SBS_COUNTS = PCAWG / "data_mutational_signatures_counts_SBS.txt"
DBS_COUNTS = PCAWG / "data_mutational_signatures_counts_DBS.txt"
SBS_TRUTH = PCAWG / "data_mutational_signatures_contribution_SBS.txt"
DBS_TRUTH = PCAWG / "data_mutational_signatures_contribution_DBS.txt"
CLINICAL = PCAWG / "data_clinical_sample.txt"
COSMIC_SBS = Path(__file__).resolve().parents[1] / "data/Signatures/COSMIC_v3.5_SBS_GRCh37.txt"
COSMIC_DBS = Path(__file__).resolve().parents[1] / "data/Signatures/COSMIC_v3.5_DBS_GRCh37.txt"
GRCH37_FASTA = REPO / "data/GRCH37/GCF_000001405.13/GCF_000001405.13_GRCh37_genomic.fna"

ATLAS52 = REPO / "cgr_validation_results/research/data/EXP022_atlas_genome_wide_45mer_universal.json"
ATLAS92 = REPO / "cgr_validation_results/research/reports/EXP021_genome_wide_bicgr/EXP021_atlas_genome_wide_45mer.json"
EXP021_PATIENTS = REPO / "cgr_validation_results/research/reports/EXP021_genome_wide_bicgr/EXP-021_PCAWG_RAW_PATIENT_METRICS.csv"
ARCHIVE_EXP021_PATIENTS = REPO / "cgr_validation_results/research/archive/pre_EXP022_20260426/reports/EXP021_genome_wide_bicgr/EXP-021_PCAWG_RAW_PATIENT_METRICS.csv"
UNIVERSAL_MODEL = "Universal-BiCGR"  # fdim = 4*d_context + 4*d_payload; name must not embed a fixed d
MODELS = ["Standard", "BiCGR-52", UNIVERSAL_MODEL]
MODEL_PAIRS = [(UNIVERSAL_MODEL, "BiCGR-52"), (UNIVERSAL_MODEL, "Standard"), ("BiCGR-52", "Standard")]

DEFAULT_PAYLOAD_SCHEMA = "masked"
DEFAULT_UGA_MODEL = get_uga_model("compact_sbs_dbs_d10")


def build_bicgr52_patients(
    counts: pd.DataFrame, 
    atlas: dict[str, np.ndarray], 
    d: int, 
    modality: str,
    patients_list: list[str] | None = None,
    universal_out: np.ndarray | None = None,
    burdens: list[int] | None = None,
    d_payload: int | None = None,
    payload_schema: str = DEFAULT_PAYLOAD_SCHEMA,
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    BiCGR-52: Context bits only.
    If universal_out is provided, we just take the context part of it.
    This ensures we use the SAME dynamic walks as the Universal model.
    """
    if universal_out is not None:
        # Use existing dynamic walks from Universal-BiCGR
        dp = int(d_payload if d_payload is not None else d)
        out = universal_context_slice(universal_out, d, dp, payload_schema).copy()
        return out, patients_list, burdens

    # Fallback to categorical if no universal results available
    channels = counts["NAME"].tolist()
    patients = [c for c in counts.columns if c.startswith("SP")]
    chan_vecs = [variant_vec_bicgr52_context_only(c, atlas, d, modality, d_payload, payload_schema) for c in channels]
    fdim = 4 * d
    V, valid = _stack_channel_matrix(chan_vecs, fdim)
    C = counts[patients].fillna(0).to_numpy(dtype=np.float64)
    out = weighted_patient_profiles(V, valid, C)
    burdens = C.sum(axis=0).astype(int).tolist()
    return out, patients, burdens


def build_universal_patients(
    counts: pd.DataFrame,
    atlas: dict[str, np.ndarray],
    d_context: int,
    modality: str,
    d_payload: int | None = None,
    payload_schema: str = DEFAULT_PAYLOAD_SCHEMA,
) -> tuple[np.ndarray, list[str], list[int]]:
    dp = int(d_payload if d_payload is not None else d_context)
    fdim = universal_fdim(d_context, dp, payload_schema)
    patients = [c for c in counts.columns if c.startswith("SP")]
    
    # Persistent Cache Logic: Check for pre-encoded profiles to avoid expensive FASTA walks
    cache_dir = REPO / "cgr_validation_results/research/data/patient_profiles_cache_clean_context"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Cache name includes modality, depth parameters, and cohort size
    cache_path = cache_dir / f"PCAWG_{modality}_d{d_context}_dp{dp}_{payload_schema}_N{len(patients)}.npz"
    
    if cache_path.exists():
        print(f"Loading cached {modality} patient profiles from {cache_path.name}...")
        data = np.load(cache_path, allow_pickle=True)
        # Verify patient list matches the current counts dataframe
        if list(data["patients"]) == patients:
            return data["out"], list(data["patients"]), list(data["burdens"])
        else:
            print("Cache patient list mismatch. Re-encoding...")
    
    global _MUTATIONS_CACHE
    if '_MUTATIONS_CACHE' not in globals() or _MUTATIONS_CACHE is None:
        print(f"Loading raw mutations from {MUTATIONS}...")
        _MUTATIONS_CACHE = pd.read_csv(MUTATIONS, sep="\t", comment="#", low_memory=False)
        
    df_muts = _MUTATIONS_CACHE
    
    if modality == "SBS":
        df_mod = df_muts[df_muts["Variant_Type"] == "SNP"]
    elif modality == "DBS":
        df_mod = df_muts[df_muts["Variant_Type"] == "DNP"]
    else:
        df_mod = df_muts[df_muts["Variant_Type"] == modality]
        
    out = np.zeros((len(patients), fdim), dtype=np.float64)
    burdens = []
    
    df_mod = df_mod[df_mod["Tumor_Sample_Barcode"].isin(patients)]
    pat_groups = df_mod.groupby("Tumor_Sample_Barcode")
    
    print(f"Executing dynamic sequence walks for {len(patients)} patients ({modality})...")
    
    reader = FastaReader(GRCH37_FASTA)
    reader.open()
    
    for idx, pid in enumerate(patients):
        if pid not in pat_groups.groups:
            out[idx] = np.zeros(fdim, dtype=np.float64)
            burdens.append(0)
            continue
            
        group = pat_groups.get_group(pid)
        vecs = []
        for _, row in group.iterrows():
            chrom = str(row["Chromosome"])
            pos = int(row["Start_Position"])
            ref = str(row["Reference_Allele"]).upper()
            alt = str(row["Tumor_Seq_Allele2"]).upper()
            
            seq_45 = reader.fetch(chrom, pos, window_len=45)
            vec = encode_variant_universal(
                seq_45,
                alt,
                d_context=d_context,
                d_payload=dp,
                ref_allele=ref,
                canonicalize=True,
                payload_schema=payload_schema,
            )
            vecs.append(vec)
            
        if vecs:
            out[idx] = np.mean(vecs, axis=0)
            burdens.append(len(vecs))
        else:
            out[idx] = np.zeros(fdim, dtype=np.float64)
            burdens.append(0)
            
    reader.close()
    
    # Save to cache for future runs
    print(f"Caching {modality} patient profiles to {cache_path.name}...")
    np.savez_compressed(cache_path, out=out, patients=np.array(patients), burdens=np.array(burdens))
    
    return out, patients, burdens


# Canonical UGA construction lives in uga_atlas. These assignments preserve the
# historical bench.run_pcawg_benchmark import surface for older scripts.
load_atlas = _uga_load_atlas
universal_fdim = _uga_universal_fdim
universal_context_slice = _uga_universal_context_slice
variant_vec = _uga_variant_vec
variant_vec_bicgr52_context_only = _uga_variant_vec_bicgr52_context_only
build_universal_ref = _uga_build_universal_ref
build_bicgr52_ref = _uga_build_bicgr52_ref
build_standard_ref = _uga_build_standard_ref
sig_columns = _uga_sig_columns




def parse_truth(path: Path) -> dict[str, dict[str, float]]:
    df, truth = pd.read_csv(path, sep="\t"), {}
    patients = [c for c in df.columns if c.startswith("SP")]
    for p in patients:
        truth[p] = {}
    for _, row in df.iterrows():
        sig = row["NAME"].split(" ")[0]
        for p in patients:
            if float(row[p]) > 0:
                truth[p][sig] = float(row[p])
    return truth


def load_clinical(path: Path) -> dict[str, str]:
    try:
        df = pd.read_csv(path, sep="\t", skiprows=4)
        return {str(r["SAMPLE_ID"]): str(r.get("CANCER_TYPE", "Unknown")) for _, r in df.iterrows()}
    except Exception:
        return {}


def burden_tier(n: int) -> str:
    return "Low (<100)" if n < 100 else ("Medium (100-1000)" if n <= 1000 else "High (>1000)")


def lasso_nonneg_coef(
    A: np.ndarray, b: np.ndarray, alpha: float = 1e-5, tol: float = 1e-4, cap: int = 2_000_000
) -> np.ndarray:
    """Nonnegative LASSO with coordinate descent run until sklearn reports convergence (n_iter < max_iter)."""
    max_iter = 4_000
    while True:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m = Lasso(positive=True, alpha=alpha, fit_intercept=False, max_iter=max_iter, tol=tol)
            m.fit(A, b)
        n_iter = int(np.ravel(m.n_iter_)[0])
        if n_iter < max_iter:
            return m.coef_.astype(float, copy=False)
        if max_iter >= cap:
            raise RuntimeError(
                f"LASSO (positive, α={alpha}, tol={tol}) did not meet the duality-gap tolerance before max_iter={max_iter}."
            )
        max_iter = min(max_iter * 4, cap)


def evaluate(A: np.ndarray, B: np.ndarray, patients: list[str], burdens: list[int], truth: dict, sigs: list[str], model: str, modality: str, clinical: dict, fitter: str) -> pd.DataFrame:
    rows: list[dict] = []
    t0 = time.perf_counter()
    
    # Identify indices for split normalization if Unified
    sbs_mask = np.array([s.startswith("SBS") for s in sigs])
    dbs_mask = np.array([s.startswith("DBS") for s in sigs])
    has_both = modality == "Unified" and sbs_mask.any() and dbs_mask.any()

    for pid, b, burden in zip(patients, B, burdens):
        if b.sum() <= 1e-12:
            continue
        w = nnls(A, b)[0] if fitter == "nnls" else lasso_nonneg_coef(A, b)
        
        if has_both:
            p = np.zeros_like(w)
            w_sbs = w[sbs_mask]
            w_dbs = w[dbs_mask]
            if w_sbs.sum() > 1e-15:
                p[sbs_mask] = w_sbs / w_sbs.sum()
            if w_dbs.sum() > 1e-15:
                p[dbs_mask] = w_dbs / w_dbs.sum()
        else:
            p = w / w.sum() if w.sum() > 1e-15 else np.zeros_like(w)
            
        t = np.array([truth.get(pid, {}).get(s, 0.0) for s in sigs])
        pred, true = p > 0.01, t > 0.01
        tp, fp, fn = np.sum(pred & true), np.sum(pred & ~true), np.sum(~pred & true)
        rec_err = float(np.linalg.norm(A @ w - b) / (np.linalg.norm(b) + 1e-15))
        res_global = {
            "Patient": pid, "Cancer_Type": clinical.get(pid, "Unknown"), "Burden_Tier": burden_tier(burden),
            "Fitter": fitter, "Model": model, "Modality": modality, "MAE": float(np.mean(np.abs(t - p))),
            "Cosine_Similarity": float(1.0 - cosine(t, p)) if np.linalg.norm(t) and np.linalg.norm(p) else 0.0,
            "Pearson_Correlation": float(pearsonr(t, p)[0]) if np.std(t) and np.std(p) else 0.0,
            "Reconstruction_Error": rec_err, "Precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "Recall": float(tp / (tp + fn)) if tp + fn else 0.0, "Active_Signatures": int(np.sum(p > 0.01)),
        }
        rows.append(res_global)

        if has_both:
            # Also record SBS-specific and DBS-specific metrics from the unified solution
            for mask, label in [(sbs_mask, f"{modality}-SBS"), (dbs_mask, f"{modality}-DBS")]:
                # Identify features belonging to this modality block in A
                feat_mask = A[:, mask].any(axis=1)
                # Skip if the target vector for this patient has no signal in this modality's features
                # (Equivalent to skipping 0-count patients in independent modality runs)
                if b[feat_mask].sum() <= 1e-12:
                    continue
                    
                tm, pm = t[mask], p[mask]
                if np.sum(tm) <= 1e-12: continue # Truth has no signatures for this modality
                
                # Metrics for this sub-modality
                # Note: pm and tm are already sum-to-1 normalized in the block above
                pred_m, true_m = pm > 0.01, tm > 0.01
                tp_m, fp_m, fn_m = np.sum(pred_m & true_m), np.sum(pred_m & ~true_m), np.sum(~pred_m & true_m)
                
                rows.append({
                    "Patient": pid, "Cancer_Type": clinical.get(pid, "Unknown"), "Burden_Tier": burden_tier(burden),
                    "Fitter": fitter, "Model": model, "Modality": label, "MAE": float(np.mean(np.abs(tm - pm))),
                    "Cosine_Similarity": float(1.0 - cosine(tm, pm)) if np.linalg.norm(tm) and np.linalg.norm(pm) else 0.0,
                    "Pearson_Correlation": float(pearsonr(tm, pm)[0]) if np.std(tm) and np.std(pm) else 0.0,
                    "Reconstruction_Error": rec_err, 
                    "Precision": float(tp_m / (tp_m + fp_m)) if tp_m + fp_m else 0.0,
                    "Recall": float(tp_m / (tp_m + fn_m)) if tp_m + fn_m else 0.0, 
                    "Active_Signatures": int(np.sum(pm > 0.01)),
                })
    df = pd.DataFrame(rows)
    df["Runtime_Sec"] = time.perf_counter() - t0
    return df


def bh_fdr(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order, out, prev = np.argsort(p), np.empty(len(p)), 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        q = min(prev, p[idx] * len(p) / (len(p) - rank + 1))
        out[idx], prev = q, q
    return out.tolist()


def add_q_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Q"] = np.nan
    mask = out["P"].notna()
    if mask.any():
        out.loc[mask, "Q"] = bh_fdr(out.loc[mask, "P"].tolist())
    return out


def paired_model_tests(
    df: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str],
    pairs: list[tuple[str, str]] = MODEL_PAIRS,
    min_n: int = 10,
) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_cols, keys))
        for metric in metrics:
            wide = g.pivot_table(index="Patient", columns="Model", values=metric, aggfunc="mean")
            for a, b in pairs:
                if a not in wide or b not in wide:
                    continue
                x = wide[[a, b]].dropna()
                diff = x[a] - x[b]
                nonzero = diff[diff != 0]
                p = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= min_n else np.nan
                rows.append({
                    **base,
                    "Metric": metric,
                    "Test": "paired Wilcoxon signed-rank",
                    "Model_A": a,
                    "Model_B": b,
                    "Comparison": f"{a} - {b}",
                    "N": len(x),
                    "Nonzero_N": len(nonzero),
                    "Delta_Mean": float(diff.mean()) if len(diff) else np.nan,
                    "Delta_Median": float(diff.median()) if len(diff) else np.nan,
                    "P": p,
                    "Note": "" if len(nonzero) >= min_n else f"not tested: <{min_n} nonzero paired differences",
                })
    return add_q_values(pd.DataFrame(rows)) if rows else pd.DataFrame()


def omnibus_model_tests(
    df: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str],
    models: list[str] = MODELS,
    min_n: int = 10,
) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_cols, keys))
        for metric in metrics:
            wide = g.pivot_table(index="Patient", columns="Model", values=metric, aggfunc="mean")
            if not all(m in wide for m in models):
                continue
            x = wide[models].dropna()
            stat = p = np.nan
            note = "" if len(x) >= min_n else f"not tested: n<{min_n}"
            if len(x) >= min_n:
                stat, p = friedmanchisquare(*(x[m].to_numpy() for m in models))
            rows.append({
                **base,
                "Metric": metric,
                "Test": "Friedman repeated-measures omnibus",
                "Models": " vs ".join(models),
                "N": len(x),
                "Statistic": float(stat) if pd.notna(stat) else np.nan,
                "P": float(p) if pd.notna(p) else np.nan,
                "Note": note,
            })
    return add_q_values(pd.DataFrame(rows)) if rows else pd.DataFrame()


def paired_tests(df: pd.DataFrame, a: str, b: str, metric: str = "Cosine_Similarity") -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(["Modality", "Fitter", "Cancer_Type"]):
        wide = g.pivot_table(index="Patient", columns="Model", values=metric, aggfunc="mean")
        if a not in wide or b not in wide:
            continue
        x = wide[[a, b]].dropna()
        if len(x) < 10:
            continue
        diff = x[a] - x[b]
        p = float(wilcoxon(diff[diff != 0]).pvalue) if np.sum(diff != 0) >= 10 else np.nan
        rows.append({"Modality": keys[0], "Fitter": keys[1], "Cancer_Type": keys[2], "N": len(x), "Delta": float(diff.mean()), "P": p})
    return add_q_values(pd.DataFrame(rows)) if rows else pd.DataFrame()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["Modality", "Fitter", "Model"]).agg(
        N=("Patient", "count"), Mean_Cos=("Cosine_Similarity", "mean"), Median_Cos=("Cosine_Similarity", "median"),
        MAE=("MAE", "mean"), Reconstruction_Error=("Reconstruction_Error", "mean"),
        Precision=("Precision", "mean"), Recall=("Recall", "mean"), Active_Signatures=("Active_Signatures", "mean"),
        Runtime_Sec=("Runtime_Sec", "mean"),
    ).reset_index()


def run_pcawg(
    out_dir: Path,
    atlas_path: Path = ATLAS52,
    include_standard: bool = True,
    include_bicgr52: bool = True,
    unified: bool = True,
    dry_run: bool = False,
    use_fasta: bool = True,
    uga_model: str = DEFAULT_UGA_MODEL.name,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = get_uga_model(uga_model)
    d_context = int(model.d_context)
    dp = int(model.d_payload)
    payload_schema = str(model.payload_schema or DEFAULT_PAYLOAD_SCHEMA)
    atlas = load_atlas(atlas_path, d_context)
    dfs, clinical = [], load_clinical(CLINICAL)
    df_sbs, df_dbs = pd.read_csv(COSMIC_SBS, sep="\t"), pd.read_csv(COSMIC_DBS, sep="\t")
    counts_sbs, counts_dbs = pd.read_csv(SBS_COUNTS, sep="\t"), pd.read_csv(DBS_COUNTS, sep="\t")
    sbs_sigs, dbs_sigs = sig_columns(df_sbs, "SBS"), sig_columns(df_dbs, "DBS")
    truth_sbs, truth_dbs = parse_truth(SBS_TRUTH), parse_truth(DBS_TRUTH)

    if dry_run:
        print(f"!!! DRY RUN MODE: Limiting to 10 common patients !!!")
        p_sbs = [c for c in counts_sbs.columns if c.startswith("SP")]
        p_dbs = [c for c in counts_dbs.columns if c.startswith("SP")]
        common = sorted(list(set(p_sbs) & set(p_dbs)))[:10]
        # Subset counts_sbs and counts_dbs to only these patients (+ NAME/Type columns)
        counts_sbs = counts_sbs[["NAME"] + [c for c in counts_sbs.columns if c in common]]
        counts_dbs = counts_dbs[["NAME"] + [c for c in counts_dbs.columns if c in common]]

    base_modalities = [
        (counts_sbs, df_sbs, sbs_sigs, truth_sbs, "SBS"),
        (counts_dbs, df_dbs, dbs_sigs, truth_dbs, "DBS")
    ]

    if include_standard:
        for counts, cosmic, sigs, truth, mod in base_modalities:
            patients = [c for c in counts.columns if c.startswith("SP")]
            burdens = [int(counts[p].fillna(0).sum()) for p in patients]
            B = np.stack([v / v.sum() if v.sum() else v for v in [counts[p].fillna(0).to_numpy() for p in patients]])
            A = build_standard_ref(cosmic, sigs)
            for fitter in ("nnls", "lasso"):
                dfs.append(evaluate(A, B, patients, burdens, truth, sigs, "Standard", mod, clinical, fitter))

    if unified:
        # Unified mode: pool SBS and DBS
        print("Unified Mode: Pooling SBS and DBS...")
        # For truth, we need to combine SBS and DBS truth dictionaries
        combined_truth = {}
        all_pids = set(truth_sbs.keys()) | set(truth_dbs.keys())
        for pid in all_pids:
            combined_truth[pid] = {**truth_sbs.get(pid, {}), **truth_dbs.get(pid, {})}
        
        # For evaluation, we need to handle the pooled counts correctly.
        # Since we use dynamic walks, we can just call build_universal_patients with 'all' or similar.
        # Actually, let's just run them separately and then mean-pool them by burden.
        A_univ_sbs = build_universal_ref(df_sbs, atlas, d_context, sbs_sigs, "SBS", dp, payload_schema)
        A_univ_dbs = build_universal_ref(df_dbs, atlas, d_context, dbs_sigs, "DBS", dp, payload_schema)
        # Stacked-Universal: Block-diagonal to prevent modal swamping
        A_pool = np.block([
            [A_univ_sbs, np.zeros((A_univ_sbs.shape[0], len(dbs_sigs)))],
            [np.zeros((A_univ_dbs.shape[0], len(sbs_sigs))), A_univ_dbs],
        ])
        sigs_all = sbs_sigs + dbs_sigs
        fdim_univ = universal_fdim(d_context, dp, payload_schema)
        
        B_sbs_raw, patients_sbs, burdens_sbs = build_universal_patients(counts_sbs, atlas, d_context, "SBS", dp, payload_schema)
        B_dbs_raw, patients_dbs, burdens_dbs = build_universal_patients(counts_dbs, atlas, d_context, "DBS", dp, payload_schema)
        
        # Convert to DataFrames for easy alignment
        df_B_sbs = pd.DataFrame(B_sbs_raw, index=patients_sbs)
        df_B_dbs = pd.DataFrame(B_dbs_raw, index=patients_dbs)
        df_burdens_sbs = pd.Series(burdens_sbs, index=patients_sbs)
        df_burdens_dbs = pd.Series(burdens_dbs, index=patients_dbs)
        
        common_patients = sorted(list(set(patients_sbs) & set(patients_dbs)))
        print(f"Unified Mode: Pooling SBS and DBS for {len(common_patients)} patients...")
        
        p_sbs = df_B_sbs.loc[common_patients].to_numpy()
        p_dbs = df_B_dbs.loc[common_patients].to_numpy()
        b_sbs = df_burdens_sbs.loc[common_patients].to_numpy()[:, np.newaxis]
        b_dbs = df_burdens_dbs.loc[common_patients].to_numpy()[:, np.newaxis]
        b_total = b_sbs + b_dbs
        # Stacked profiles: Concatenate modality centroids with equal weighting (0.5 each)
        # to prevent numerical swamping of low-burden modalities (e.g., DBS) by high-burden ones (SBS).
        B_pool = np.column_stack([
            p_sbs * 0.5,
            p_dbs * 0.5,
        ])
        total_burdens_list = b_total.flatten().astype(int).tolist()
        
        for fitter in ("nnls", "lasso"):
            dfs.append(evaluate(A_pool, B_pool, common_patients, total_burdens_list, combined_truth, sigs_all, UNIVERSAL_MODEL, "Unified", clinical, fitter))
            
        if include_bicgr52:
            A_ctx_sbs = build_bicgr52_ref(df_sbs, atlas, d_context, sbs_sigs, "SBS", payload_schema)
            A_ctx_dbs = build_bicgr52_ref(df_dbs, atlas, d_context, dbs_sigs, "DBS", payload_schema)
            # Stacked BiCGR-52
            A_ctx_pool = np.block([
                [A_ctx_sbs, np.zeros((A_ctx_sbs.shape[0], len(dbs_sigs)))],
                [np.zeros((A_ctx_dbs.shape[0], len(sbs_sigs))), A_ctx_dbs],
            ])

            # BiCGR-52 is just the context part of the stacked profile
            B_ctx_pool = np.column_stack([
                universal_context_slice(p_sbs, d_context, dp, payload_schema) * 0.5,
                universal_context_slice(p_dbs, d_context, dp, payload_schema) * 0.5,
            ])

            for fitter in ("nnls", "lasso"):
                dfs.append(evaluate(A_ctx_pool, B_ctx_pool, common_patients, total_burdens_list, combined_truth, sigs_all, "BiCGR-52", "Unified", clinical, fitter))

        if include_standard:
            # the same burden-weighted pooled patient profiles as Universal/BiCGR-52.
            A_std_sbs = build_standard_ref(df_sbs, sbs_sigs)   # (96, n_sbs_sigs)
            A_std_dbs = build_standard_ref(df_dbs, dbs_sigs)   # (78, n_dbs_sigs)
            A_std_pool = np.block([
                [A_std_sbs, np.zeros((A_std_sbs.shape[0], len(dbs_sigs)))],
                [np.zeros((A_std_dbs.shape[0], len(sbs_sigs))), A_std_dbs],
            ])  # (174, n_sbs + n_dbs)

            # Build Standard patient profiles (L1-normalised count spectra)
            def _std_profiles(counts_df: pd.DataFrame) -> pd.DataFrame:
                pids = [c for c in counts_df.columns if c.startswith("SP")]
                vecs = [v / v.sum() if v.sum() else v
                        for v in [counts_df[p].fillna(0).to_numpy() for p in pids]]
                return pd.DataFrame(np.stack(vecs), index=pids)

            df_B_std_sbs = _std_profiles(counts_sbs)
            df_B_std_dbs = _std_profiles(counts_dbs)

            # Burden-weighted column-stack to form 174-dimensional combined profile
            B_std_pool = np.column_stack([
                df_B_std_sbs.loc[common_patients].to_numpy() * b_sbs / np.where(b_total > 0, b_total, 1.0),
                df_B_std_dbs.loc[common_patients].to_numpy() * b_dbs / np.where(b_total > 0, b_total, 1.0),
            ])  # (n_common, 174)

            for fitter in ("nnls", "lasso"):
                dfs.append(evaluate(A_std_pool, B_std_pool, common_patients, total_burdens_list,
                                    combined_truth, sigs_all, "Standard", "Unified", clinical, fitter))

    # Independent evaluation for Universal models
    print(f"DEBUG: Starting independent evaluation for Universal models. Base modalities: {[m[4] for m in base_modalities]}")
    for counts, cosmic, sigs, truth, mod in base_modalities:
        print(f"DEBUG: Evaluating {mod} for {UNIVERSAL_MODEL}...")
        A_univ = build_universal_ref(cosmic, atlas, d_context, sigs, mod, dp, payload_schema)
        if use_fasta:
            B_univ, patients, burdens = build_universal_patients(counts, atlas, d_context, mod, dp, payload_schema)
        else:
            print(f"Using atlas-based patient profiling for {mod}...")
            patients = [c for c in counts.columns if c.startswith("SP")]
            chan_vecs = [variant_vec(c, atlas, d_context, mod, dp, payload_schema) for c in counts["NAME"].tolist()]
            V, valid = _stack_channel_matrix(chan_vecs, universal_fdim(d_context, dp, payload_schema))
            C = counts[patients].fillna(0).to_numpy(dtype=np.float64)
            B_univ = weighted_patient_profiles(V, valid, C)
            burdens = C.sum(axis=0).astype(int).tolist()
        
        for fitter in ("nnls", "lasso"):
            res = evaluate(A_univ, B_univ, patients, burdens, truth, sigs, UNIVERSAL_MODEL, mod, clinical, fitter)
            dfs.append(res)
        if include_bicgr52:
            A_ctx = build_bicgr52_ref(cosmic, atlas, d_context, sigs, mod, payload_schema)
            B_ctx, _, _ = build_bicgr52_patients(
                counts,
                atlas,
                d_context,
                mod,
                patients,
                B_univ,
                burdens,
                dp,
                payload_schema,
            )
            for fitter in ("nnls", "lasso"):
                res = evaluate(A_ctx, B_ctx, patients, burdens, truth, sigs, "BiCGR-52", mod, clinical, fitter)
                dfs.append(res)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Final results compiled: {len(df)} rows across {df['Modality'].nunique()} modalities.")
    paths = {
        "patients": out_dir / "EXP-022_PCAWG_PATIENT_METRICS.csv",
        "summary": out_dir / "EXP-022_TABLE2_THREE_WAY_METRICS.csv",
        "burden": out_dir / "EXP-022_TABLE3_BURDEN_TIER_LASSO.csv",
        "strata": out_dir / "EXP-022_STRATIFIED_RESULTS.csv",
        "wilcoxon": out_dir / "EXP-022_WILCOXON_UNIVERSAL_BICGR_VS_BICGR52.csv",
        "overall_omnibus": out_dir / "EXP-022_OVERALL_OMNIBUS_TESTS.csv",
        "overall_pairwise": out_dir / "EXP-022_OVERALL_PAIRWISE_TESTS.csv",
        "burden_omnibus": out_dir / "EXP-022_BURDEN_LASSO_OMNIBUS_TESTS.csv",
        "burden_pairwise": out_dir / "EXP-022_BURDEN_LASSO_PAIRWISE_TESTS.csv",
        "strata_pairwise": out_dir / "EXP-022_STRATA_PAIRWISE_TESTS.csv",
    }
    df.to_csv(paths["patients"], index=False)
    summary = summarize(df)
    summary.to_csv(paths["summary"], index=False)
    print("\n--- Final Summary Table ---")
    print(summary.to_string(index=False))
    print("---------------------------\n")
    df[df["Fitter"] == "lasso"].groupby(["Modality", "Burden_Tier", "Model"]).agg(
        N=("Patient", "count"), Mean_Cos=("Cosine_Similarity", "mean"), MAE=("MAE", "mean")
    ).reset_index().to_csv(paths["burden"], index=False)
    df.groupby(["Modality", "Fitter", "Cancer_Type", "Model"]).agg(
        N=("Patient", "count"), Mean_Cos=("Cosine_Similarity", "mean"), MAE=("MAE", "mean")
    ).reset_index().query("N >= 10").to_csv(paths["strata"], index=False)
    paired_tests(df[df["Model"].isin([UNIVERSAL_MODEL, "BiCGR-52"])], UNIVERSAL_MODEL, "BiCGR-52").to_csv(paths["wilcoxon"], index=False)
    metrics = ["Cosine_Similarity", "MAE"]
    omnibus_model_tests(df, ["Modality", "Fitter"], metrics).to_csv(paths["overall_omnibus"], index=False)
    paired_model_tests(df, ["Modality", "Fitter"], metrics).to_csv(paths["overall_pairwise"], index=False)
    omnibus_model_tests(df, ["Modality", "Fitter", "Burden_Tier"], metrics).to_csv(paths["burden_omnibus"], index=False)
    paired_model_tests(df, ["Modality", "Fitter", "Burden_Tier"], metrics).to_csv(paths["burden_pairwise"], index=False)
    paired_model_tests(df, ["Modality", "Fitter", "Cancer_Type"], metrics).to_csv(paths["strata_pairwise"], index=False)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--atlas", type=Path, default=ATLAS52)
    ap.add_argument("--uga-model", default=DEFAULT_UGA_MODEL.name)
    ap.add_argument("--no-unified", action="store_true", help="solve SBS/DBS individually")
    ap.add_argument("--dry-run", action="store_true", help="limit to 5 patients for quick testing")
    args = ap.parse_args()
    for k, p in run_pcawg(args.out_dir, args.atlas,
                          unified=not args.no_unified, dry_run=args.dry_run,
                          uga_model=args.uga_model).items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()
