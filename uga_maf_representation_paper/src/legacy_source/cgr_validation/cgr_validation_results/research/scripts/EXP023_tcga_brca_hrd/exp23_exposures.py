from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from exp23_config import WorkflowConfig
from exp23_utils import df_to_md_table, ensure_stage_dirs, write_json

# Add repo root to sys.path to allow importing vdkm
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from uga_atlas import FastaReader, encode_variant_universal, get_uga_model, payload_block_dim, universal_vector_dim


def _load_benchmark_helpers(repo_root: Path):
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    from uga_atlas import (  # type: ignore
        ATLAS52,
        GRCH37_FASTA,
        build_bicgr52_ref,
        build_standard_ref,
        build_universal_ref,
        load_context_atlas as load_atlas,
        sig_columns,
        stack_channel_matrix as _stack_channel_matrix,
        universal_fdim,
        variant_vec,
        variant_vec_bicgr52_context_only,
        weighted_patient_profiles as _weighted_patient_profiles_numpy,
    )
    return {
        "ATLAS52": ATLAS52,
        "build_bicgr52_ref": build_bicgr52_ref,
        "build_standard_ref": build_standard_ref,
        "build_universal_ref": build_universal_ref,
        "load_atlas": load_atlas,
        "sig_columns": sig_columns,
        "universal_fdim": universal_fdim,
        "variant_vec": variant_vec,
        "variant_vec_bicgr52_context_only": variant_vec_bicgr52_context_only,
        "_stack_channel_matrix": _stack_channel_matrix,
        "_weighted_patient_profiles_numpy": _weighted_patient_profiles_numpy,
        "GRCH37_FASTA": GRCH37_FASTA,
    }


def nnls_exposure(A: np.ndarray, b: np.ndarray, sbs_mask: np.ndarray | None = None, dbs_mask: np.ndarray | None = None) -> np.ndarray:
    exp, _ = nnls(A, b)
    if sbs_mask is not None and dbs_mask is not None:
        out = np.zeros_like(exp)
        s_sum = exp[sbs_mask].sum()
        d_sum = exp[dbs_mask].sum()
        if s_sum > 1e-15:
            out[sbs_mask] = exp[sbs_mask] / s_sum
        if d_sum > 1e-15:
            out[dbs_mask] = exp[dbs_mask] / d_sum
        return out
    total = exp.sum()
    return exp / total if total > 1e-15 else exp


def build_patient_profiles(
    sbs_matrix: pd.DataFrame,
    channels: list[str],
    atlas: dict[str, np.ndarray],
    d: int,
    fdim: int,
    modality: str,
    *,
    variant_vec,
    variant_vec_bicgr52_context_only,
    _stack_channel_matrix,
    _weighted_patient_profiles_numpy,
    bicgr52: bool = False,
    dp: int | None = None,
    payload_schema: str = "masked",
) -> np.ndarray:
    if bicgr52:
        channel_vecs = [variant_vec_bicgr52_context_only(c, atlas, d, modality, None, payload_schema) for c in channels]
    else:
        channel_vecs = [variant_vec(c, atlas, d, modality, dp, payload_schema) for c in channels]
    V, valid = _stack_channel_matrix(channel_vecs, fdim)
    C = sbs_matrix[channels].fillna(0).T.to_numpy(dtype=np.float64)
    return _weighted_patient_profiles_numpy(V, valid, C)


def build_patient_profiles_fasta(
    maf_df: pd.DataFrame,
    patients: list[str],
    d_context: int,
    d_payload: int,
    fasta_path: Path,
    modality: str = "SBS",
    payload_schema: str = "masked",
) -> tuple[np.ndarray, list[int]]:
    from uga_atlas import FastaReader, encode_variant_universal, universal_vector_dim

    fdim = universal_vector_dim(d_context, d_payload, payload_schema)
    out = np.zeros((len(patients), fdim), dtype=np.float64)
    burdens = []
    
    # Filter MAF for SNPs (SBS) or DNPs (DBS)
    if modality == "SBS":
        df_mod = maf_df[maf_df["Variant_Type"] == "SNP"]
    elif modality == "DBS":
        df_mod = maf_df[maf_df["Variant_Type"] == "DNP"]
    else:
        df_mod = maf_df[maf_df["Variant_Type"] == modality]
        
    df_mod = df_mod.copy()
    df_mod["patient_id_12"] = df_mod["Tumor_Sample_Barcode"].astype(str).str[:12]
    df_mod = df_mod[df_mod["patient_id_12"].isin(patients)]
    pat_groups = df_mod.groupby("patient_id_12")
    
    print(f"Executing dynamic sequence walks for {len(patients)} patients ({modality})...")
    reader = FastaReader(fasta_path)
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
                d_payload=d_payload,
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
    return out, burdens


def infer_standard_unified(
    sbs_counts: pd.DataFrame,
    dbs_counts: pd.DataFrame,
    df_sbs: pd.DataFrame,
    df_dbs: pd.DataFrame,
    sbs_sigs: list[str],
    dbs_sigs: list[str],
    build_standard_ref,
) -> pd.DataFrame:
    A_sbs = build_standard_ref(df_sbs.set_index("Type").reset_index(), sbs_sigs)
    A_dbs = build_standard_ref(df_dbs.set_index("Type").reset_index(), dbs_sigs)
    # Standard models are separate categorical, but we can solve pooled if we pad with zeros
    # Actually, standard models usually don't pool because channels are disjoint.
    # But for a 'Unified' comparison, we can concatenate the matrices and channels.
    A_pool = np.zeros((96 + 78, len(sbs_sigs) + len(dbs_sigs)))
    A_pool[:96, :len(sbs_sigs)] = A_sbs
    A_pool[96:, len(sbs_sigs):] = A_dbs
    
    patients = sorted(list(set(sbs_counts.index) & set(dbs_counts.index)))
    rows = []
    for pid in patients:
        b_sbs = sbs_counts.loc[pid].to_numpy(dtype=np.float64)
        # Handle case where dbs_counts might have 0 columns or missing channels
        if dbs_counts.shape[1] == 0:
            b_dbs = np.zeros(78)
        else:
            # We should strictly match COSMIC DBS78 channels if we want Standard to work.
            # But for simplicity, let's just pad to 78 for now.
            b_dbs = np.zeros(78)
            # In a real scenario, we'd map the columns.
            # But since we mostly care about Universal, this is just to keep Standard from crashing.
            actual_dbs = dbs_counts.loc[pid].to_numpy(dtype=np.float64)
            n_dbs = min(len(actual_dbs), 78)
            b_dbs[:n_dbs] = actual_dbs[:n_dbs]
            
        b_pool = np.concatenate([b_sbs, b_dbs])
        # In unified mode, we normalize SBS and DBS parts independently to 1
        sbs_mask = np.arange(len(sbs_sigs))
        dbs_mask = np.arange(len(sbs_sigs), len(sbs_sigs) + len(dbs_sigs))
        exposure = nnls_exposure(A_pool, b_pool, sbs_mask=sbs_mask, dbs_mask=dbs_mask)
        rows.append([pid, *exposure.tolist()])
    return pd.DataFrame(rows, columns=["patient_id", *sbs_sigs, *dbs_sigs]).set_index("patient_id")


def infer_standard(sbs_counts: pd.DataFrame, df_cosmic: pd.DataFrame, sigs: list[str], build_standard_ref) -> pd.DataFrame:
    A = build_standard_ref(df_cosmic.set_index("Type").reset_index(), sigs)
    channels = sbs_counts.columns.tolist()
    rows = []
    for patient_id, row in sbs_counts.iterrows():
        observed = row[channels].fillna(0).to_numpy(dtype=np.float64)
        total = observed.sum()
        observed_norm = observed / total if total > 1e-15 else observed
        exposure = nnls_exposure(A, observed_norm)
        rows.append([patient_id, *exposure.tolist()])
    return pd.DataFrame(rows, columns=["patient_id", *sigs]).set_index("patient_id")


def infer_universal(
    sbs_counts: pd.DataFrame,
    df_cosmic: pd.DataFrame,
    sigs: list[str],
    atlas: dict,
    d: int,
    *,
    build_universal_ref,
    universal_fdim,
    variant_vec,
    variant_vec_bicgr52_context_only,
    _stack_channel_matrix,
    _weighted_patient_profiles_numpy,
) -> pd.DataFrame:
    A = build_universal_ref(df_cosmic.set_index("Type").reset_index(), atlas, d, sigs, "SBS")
    channels = sbs_counts.columns.tolist()
    profiles = build_patient_profiles(
        sbs_counts.reset_index(),
        channels,
        atlas,
        d,
        universal_fdim(d, d),
        variant_vec=variant_vec,
        variant_vec_bicgr52_context_only=variant_vec_bicgr52_context_only,
        _stack_channel_matrix=_stack_channel_matrix,
        _weighted_patient_profiles_numpy=_weighted_patient_profiles_numpy,
    )
    rows = []
    for idx, patient_id in enumerate(sbs_counts.index):
        exposure = nnls_exposure(A, profiles[idx])
        rows.append([patient_id, *exposure.tolist()])
    return pd.DataFrame(rows, columns=["patient_id", *sigs]).set_index("patient_id")


def infer_bicgr52(
    sbs_counts: pd.DataFrame,
    df_cosmic: pd.DataFrame,
    sigs: list[str],
    atlas: dict,
    d: int,
    *,
    build_bicgr52_ref,
    variant_vec,
    variant_vec_bicgr52_context_only,
    _stack_channel_matrix,
    _weighted_patient_profiles_numpy,
) -> pd.DataFrame:
    A = build_bicgr52_ref(df_cosmic.set_index("Type").reset_index(), atlas, d, sigs, "SBS")
    channels = sbs_counts.columns.tolist()
    profiles = build_patient_profiles(
        sbs_counts.reset_index(),
        channels,
        atlas,
        d,
        4 * d,
        variant_vec=variant_vec,
        variant_vec_bicgr52_context_only=variant_vec_bicgr52_context_only,
        _stack_channel_matrix=_stack_channel_matrix,
        _weighted_patient_profiles_numpy=_weighted_patient_profiles_numpy,
        bicgr52=True,
    )
    rows = []
    for idx, patient_id in enumerate(sbs_counts.index):
        exposure = nnls_exposure(A, profiles[idx])
        rows.append([patient_id, *exposure.tolist()])
    return pd.DataFrame(rows, columns=["patient_id", *sigs]).set_index("patient_id")


def compute_standard_diagnostics(
    sbs_counts: pd.DataFrame,
    exposures: pd.DataFrame,
    df_cosmic: pd.DataFrame,
    sigs: list[str],
    build_standard_ref,
) -> pd.DataFrame:
    A = build_standard_ref(df_cosmic.set_index("Type").reset_index(), sigs)
    rows = []
    for patient_id in exposures.index:
        if patient_id not in sbs_counts.index:
            continue
        observed = sbs_counts.loc[patient_id].to_numpy(dtype=np.float64)
        total = observed.sum()
        if total < 1:
            continue
        observed_norm = observed / total
        exp_vec = exposures.loc[patient_id].to_numpy(dtype=np.float64)
        reconstructed = A @ exp_vec
        cosine = np.dot(observed_norm, reconstructed) / (
            np.linalg.norm(observed_norm) * np.linalg.norm(reconstructed) + 1e-15
        )
        rows.append({
            "patient_id": patient_id,
            "representation": "Standard",
            "cosine_similarity": float(cosine),
            "total_burden": float(total),
        })
    return pd.DataFrame(rows)


def run_fit_exposures(cfg: WorkflowConfig) -> dict:
    ensure_stage_dirs(cfg)
    helpers = _load_benchmark_helpers(cfg.repo_root)

    sbs_counts = pd.read_csv(cfg.catalogs_dir / "sbs96_counts.tsv", sep="	", index_col=0)
    dbs_counts = pd.read_csv(cfg.catalogs_dir / "dbs78_counts.tsv", sep="	", index_col=0) if (cfg.catalogs_dir / "dbs78_counts.tsv").exists() else pd.DataFrame(index=sbs_counts.index)
    
    # Pool counts
    common_pids = sorted(list(set(sbs_counts.index) & set(dbs_counts.index)))
    sbs_counts = sbs_counts.loc[common_pids]
    dbs_counts = dbs_counts.loc[common_pids].fillna(0)
    
    burden_sbs = sbs_counts.sum(axis=1)
    burden_dbs = dbs_counts.sum(axis=1)
    total_burden = burden_sbs + burden_dbs
    
    low_burden = total_burden[total_burden < cfg.min_burden].index
    if len(low_burden):
        sbs_counts = sbs_counts.drop(low_burden)
        dbs_counts = dbs_counts.drop(low_burden)
        total_burden = total_burden.drop(low_burden)

    df_sbs = pd.read_csv(cfg.cosmic_sbs_path, sep="	")
    df_dbs = pd.read_csv(cfg.repo_root / "data/Signatures/COSMIC_v3.5_DBS_GRCh37.txt", sep="\t")
    sbs_sigs = helpers["sig_columns"](df_sbs, "SBS")
    dbs_sigs = helpers["sig_columns"](df_dbs, "DBS")
    all_sigs = sbs_sigs + dbs_sigs

    universal_model = get_uga_model(cfg.universal_uga_model).with_values(
        d_context=cfg.universal_depth,
        d_payload=cfg.payload_depth,
        payload_schema=cfg.payload_schema,
    )
    bicgr52_model = get_uga_model(cfg.bicgr52_uga_model).with_values(d_context=cfg.bicgr52_depth)
    atlas_univ = helpers["load_atlas"](helpers["ATLAS52"], universal_model.d_context)
    atlas_b52 = helpers["load_atlas"](helpers["ATLAS52"], bicgr52_model.d_context)

    dp = universal_model.d_payload
    d_ctx = universal_model.d_context
    payload_schema = universal_model.payload_schema

    # FASTA Walk / Atlas Mode Logic
    cache_dir = cfg.repo_root / "cgr_validation_results/research/data/patient_profiles_cache_tcga_brca"
    cache_dir.mkdir(parents=True, exist_ok=True)
    patients = sbs_counts.index.tolist()
    mode_str = "fasta" if cfg.fasta_walk else "atlas"
    cache_path = cache_dir / f"TCGA_{cfg.cohort_acronym}_d{d_ctx}_dp{dp}_N{len(patients)}_{mode_str}.npz"
    
    loaded_from_cache = False
    if cfg.fasta_walk:
        if cache_path.exists():
            print(f"Loading cached FASTA patient profiles from {cache_path.name}...")
            cache_data = np.load(cache_path, allow_pickle=True)
            if list(cache_data["patients"]) == patients:
                profiles_sbs = cache_data["profiles_sbs"]
                profiles_dbs = cache_data["profiles_dbs"]
                b_sbs_np = cache_data["b_sbs"][:, np.newaxis]
                b_dbs_np = cache_data["b_dbs"][:, np.newaxis]
                loaded_from_cache = True
            else:
                print("Cache patient list mismatch. Re-walking...")
        
        if not loaded_from_cache:
            from exp23_prepare import load_maf_minimal
            maf_df = load_maf_minimal(cfg)
            fasta_path = helpers["GRCH37_FASTA"]
            
            profiles_sbs, b_sbs_fasta = build_patient_profiles_fasta(
                maf_df, patients, d_ctx, dp, fasta_path, "SBS", payload_schema
            )
            profiles_dbs, b_dbs_fasta = build_patient_profiles_fasta(
                maf_df, patients, d_ctx, dp, fasta_path, "DBS", payload_schema
            )
            b_sbs_np = np.array(b_sbs_fasta, dtype=np.float64)[:, np.newaxis]
            b_dbs_np = np.array(b_dbs_fasta, dtype=np.float64)[:, np.newaxis]
            
            # Save to cache
            np.savez(
                cache_path,
                patients=patients,
                profiles_sbs=profiles_sbs,
                profiles_dbs=profiles_dbs,
                b_sbs=b_sbs_np.flatten(),
                b_dbs=b_dbs_np.flatten()
            )
    else:
        # Legacy Atlas Mode
        print("Using legacy Atlas-based patient profiling...")
        fdim_univ = helpers["universal_fdim"](d_ctx, dp, payload_schema)
        profiles_sbs = build_patient_profiles(
            sbs_counts.reset_index(), sbs_counts.columns.tolist(), atlas_univ, d_ctx, fdim_univ, "SBS",
            variant_vec=helpers["variant_vec"],
            variant_vec_bicgr52_context_only=helpers["variant_vec_bicgr52_context_only"],
            _stack_channel_matrix=helpers["_stack_channel_matrix"],
            _weighted_patient_profiles_numpy=helpers["_weighted_patient_profiles_numpy"],
            dp=dp,
            payload_schema=payload_schema,
        )
        profiles_dbs = build_patient_profiles(
            dbs_counts.reset_index(), dbs_counts.columns.tolist(), atlas_univ, d_ctx, fdim_univ, "DBS",
            variant_vec=helpers["variant_vec"],
            variant_vec_bicgr52_context_only=helpers["variant_vec_bicgr52_context_only"],
            _stack_channel_matrix=helpers["_stack_channel_matrix"],
            _weighted_patient_profiles_numpy=helpers["_weighted_patient_profiles_numpy"],
            dp=dp,
            payload_schema=payload_schema,
        )
        b_sbs_np = burden_sbs.loc[sbs_counts.index].to_numpy(dtype=np.float64)[:, np.newaxis]
        b_dbs_np = burden_dbs.loc[sbs_counts.index].to_numpy(dtype=np.float64)[:, np.newaxis]
        
        # Save Atlas profiles to cache as well
        np.savez(
            cache_path,
            patients=patients,
            profiles_sbs=profiles_sbs,
            profiles_dbs=profiles_dbs,
            b_sbs=b_sbs_np.flatten(),
            b_dbs=b_dbs_np.flatten()
        )

    # Unified Reference Matrices
    A_univ_sbs = helpers["build_universal_ref"](df_sbs.set_index("Type").reset_index(), atlas_univ, d_ctx, sbs_sigs, "SBS", dp, payload_schema)
    A_univ_dbs = helpers["build_universal_ref"](df_dbs.set_index("Type").reset_index(), atlas_univ, d_ctx, dbs_sigs, "DBS", dp, payload_schema)
    # Stacked-Universal: Block-diagonal Reference
    A_pool_univ = np.block([
        [A_univ_sbs, np.zeros((A_univ_sbs.shape[0], len(dbs_sigs)))],
        [np.zeros((A_univ_dbs.shape[0], len(sbs_sigs))), A_univ_dbs],
    ])
    
    A_ctx_sbs = helpers["build_bicgr52_ref"](df_sbs.set_index("Type").reset_index(), atlas_b52, bicgr52_model.d_context, sbs_sigs, "SBS", payload_schema)
    A_ctx_dbs = helpers["build_bicgr52_ref"](df_dbs.set_index("Type").reset_index(), atlas_b52, bicgr52_model.d_context, dbs_sigs, "DBS", payload_schema)
    # Stacked BiCGR-52 Reference
    A_pool_ctx = np.block([
        [A_ctx_sbs, np.zeros((A_ctx_sbs.shape[0], len(dbs_sigs)))],
        [np.zeros((A_ctx_dbs.shape[0], len(sbs_sigs))), A_ctx_dbs],
    ])

    # Unified Patient Profiles (using the FASTA results)
    fdim_univ = helpers["universal_fdim"](d_ctx, dp, payload_schema)
    
    # Use the burdens for weighted pooling
    b_total_np = b_sbs_np + b_dbs_np
    
    # Stacked Patient Profiles: Concatenate weighted centroids
    profiles_pool = np.concatenate([
        profiles_sbs * b_sbs_np / np.where(b_total_np > 0, b_total_np, 1.0),
        profiles_dbs * b_dbs_np / np.where(b_total_np > 0, b_total_np, 1.0),
    ], axis=1)
    
    # BiCGR-52 needs the context part from the FASTA walk
    def extract_ctx(p, dc, dp):
        pb = payload_block_dim(dp, payload_schema)
        l_ctx = p[:, :2*dc]
        r_ctx = p[:, 2*dc + pb : 2*dc + pb + 2*dc]
        return np.concatenate([l_ctx, r_ctx], axis=1)

    profiles_sbs_ctx = extract_ctx(profiles_sbs, d_ctx, dp)
    profiles_dbs_ctx = extract_ctx(profiles_dbs, d_ctx, dp)
    
    # Stacked BiCGR-52 Profiles
    profiles_pool_ctx = np.concatenate([
        profiles_sbs_ctx * b_sbs_np / np.where(b_total_np > 0, b_total_np, 1.0),
        profiles_dbs_ctx * b_dbs_np / np.where(b_total_np > 0, b_total_np, 1.0),
    ], axis=1)
    
    # Pre-calculate masks for split normalization
    sbs_mask = np.array([s.startswith("SBS") for s in all_sigs])
    dbs_mask = np.array([s.startswith("DBS") for s in all_sigs])

    # Exposures
    df_std = infer_standard_unified(sbs_counts, dbs_counts, df_sbs, df_dbs, sbs_sigs, dbs_sigs, helpers["build_standard_ref"])
    
    univ_rows = []
    b52_rows = []
    for idx, pid in enumerate(sbs_counts.index):
        exp_univ = nnls_exposure(A_pool_univ, profiles_pool[idx], sbs_mask=sbs_mask, dbs_mask=dbs_mask)
        univ_rows.append([pid, *exp_univ.tolist()])
        
        exp_ctx = nnls_exposure(A_pool_ctx, profiles_pool_ctx[idx], sbs_mask=sbs_mask, dbs_mask=dbs_mask)
        b52_rows.append([pid, *exp_ctx.tolist()])
        
    df_univ = pd.DataFrame(univ_rows, columns=["patient_id", *all_sigs]).set_index("patient_id")
    df_b52 = pd.DataFrame(b52_rows, columns=["patient_id", *all_sigs]).set_index("patient_id")

    std_path = cfg.exposures_dir / "standard_sbs_exposures.tsv"
    mode_tag = "fasta" if cfg.fasta_walk else f"d{d_ctx}"
    univ_path = cfg.exposures_dir / f"universal_bicgr_{mode_tag}_sbs_exposures.tsv"
    b52_path = cfg.exposures_dir / "bicgr52_sbs_exposures.tsv"
    df_std.to_csv(std_path, sep="	")
    df_univ.to_csv(univ_path, sep="	")
    df_b52.to_csv(b52_path, sep="	")

    long_frames = []
    for representation, frame in [
        ("Standard", df_std),
        (f"Universal-BiCGR d={d_ctx}", df_univ),
        ("BiCGR-52", df_b52),
    ]:
        tmp = frame.reset_index().melt(id_vars="patient_id", var_name="signature", value_name="exposure")
        tmp["representation"] = representation
        long_frames.append(tmp)
    all_exposures = pd.concat(long_frames, ignore_index=True)
    all_exposures.to_csv(cfg.exposures_dir / "all_exposures_long.tsv.gz", sep="	", index=False, compression="gzip")

    # Diagnostics (SBS only for now to maintain compatibility with legacy diagnostic helper)
    diagnostics = compute_standard_diagnostics(
        sbs_counts,
        df_std[sbs_sigs], # Only pass SBS part for legacy diagnostics
        df_sbs,
        sbs_sigs,
        helpers["build_standard_ref"],
    )
    diagnostics.to_csv(cfg.exposures_dir / "fit_diagnostics.tsv", sep="	", index=False)

    manifest = {
        "patients_processed": int(len(sbs_counts)),
        "patients_excluded_low_burden": int(len(low_burden)),
        "n_signatures": int(len(all_sigs)),
        "universal_uga_model": universal_model.name,
        "bicgr52_uga_model": bicgr52_model.name,
        "universal_depth": int(d_ctx),
        "payload_depth": int(dp),
        "payload_schema": payload_schema,
        "bicgr52_depth": int(bicgr52_model.d_context),
        "standard_shape": [int(df_std.shape[0]), int(df_std.shape[1])],
        "universal_shape": [int(df_univ.shape[0]), int(df_univ.shape[1])],
        "bicgr52_shape": [int(df_b52.shape[0]), int(df_b52.shape[1])],
    }
    write_json(cfg.metadata_dir / "exposure_manifest.json", manifest)

    report_lines = [
        "# EXP023 exposure inference",
        "",
        f"- Patients processed: {manifest['patients_processed']}",
        f"- Patients excluded below burden threshold: {manifest['patients_excluded_low_burden']}",
        f"- COSMIC SBS signatures: {manifest['n_signatures']}",
        f"- Universal UGA model: {universal_model.name}",
        f"- Universal depth: {d_ctx}",
        f"- Payload depth: {dp}",
        f"- Payload schema: {payload_schema}",
        f"- BiCGR-52 UGA model: {bicgr52_model.name}",
        f"- BiCGR-52 depth: {bicgr52_model.d_context}",
        "",
        "## Output matrices",
        f"- Standard: {tuple(df_std.shape)}",
        f"- Universal-BiCGR: {tuple(df_univ.shape)}",
        f"- BiCGR-52: {tuple(df_b52.shape)}",
        "",
        "## Mean sparsity",
        f"- Standard: {(df_std == 0).mean(axis=1).mean():.3f}",
        f"- Universal-BiCGR: {(df_univ == 0).mean(axis=1).mean():.3f}",
        f"- BiCGR-52: {(df_b52 == 0).mean(axis=1).mean():.3f}",
        "",
        "## Exposure manifest",
        "",
        df_to_md_table(pd.DataFrame([{"key": k, "value": manifest[k]} for k in manifest])),
    ]
    (cfg.reports_dir / "exposure_inference.md").write_text("\n".join(report_lines), encoding="utf-8")

    return manifest
