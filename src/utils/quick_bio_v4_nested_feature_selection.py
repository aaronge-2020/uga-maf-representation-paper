"""Quick nested-CV smoke test for Clinical Bio v4 MAF feature selection.

This runner intentionally stays outside the manuscript pipeline. It builds the
frozen Bio MAF v4 feature matrix, then tests inner-loop feature-block selection
against prior quick benchmark arms without rewriting canonical manuscript files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from utils import bio_maf_base_features as base
except ImportError:  # pragma: no cover - direct script execution from src/utils.
    import bio_maf_base_features as base


BUNDLE_ROOT = base.BUNDLE_ROOT
ROLE_MATCH_LOF = {
    "stop_gained",
    "frameshift_variant",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "start_lost",
    "transcript_ablation",
}
ROLE_MATCH_ACT = {
    "missense_variant",
    "inframe_deletion",
    "inframe_insertion",
    "protein_altering_variant",
}
PROTEIN_AFFECTING = {
    *ROLE_MATCH_LOF,
    *ROLE_MATCH_ACT,
    "stop_lost",
    "coding_sequence_variant",
}
DRIVER_TIER_METRICS = [
    "high_or_moderate_impact_log_count",
    "high_impact_log_count",
    "unique_high_or_moderate_genes_log_count",
    "max_vaf_high_or_moderate_impact",
]
HOTSPOT_FEATURES = [
    "cancer_hotspot_exact_variant_log_count",
    "cancer_hotspot_residue_log_count",
    "cancer_hotspot_unique_gene_log_count",
    "cancer_hotspot_max_vaf",
]


def parse_protein_position(value: object) -> int | None:
    text = "" if value is None else str(value)
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None


def parse_amino_acids(row: pd.Series) -> tuple[str, str]:
    aa = "" if row.get("Amino_acids") is None else str(row.get("Amino_acids"))
    if "/" in aa:
        ref, alt = aa.split("/", 1)
        return ref.strip().upper(), alt.strip().upper()
    hgvsp = "" if row.get("HGVSp_Short") is None else str(row.get("HGVSp_Short"))
    match = re.search(r"p\.?([A-Za-z\*]+)(\d+)([A-Za-z\*]+)", hgvsp)
    if match:
        return match.group(1).upper(), match.group(3).upper()
    return "", ""


def load_v4_resources() -> tuple[pd.DataFrame, dict[str, int], dict[str, str], set[tuple[str, int]], set[tuple[str, int, str, str]]]:
    tables_dir = BUNDLE_ROOT / "results" / "tables"
    resource_dir = BUNDLE_ROOT / "config" / "feature_resources" / "clinical_bio_v4_sources"
    evidence = pd.read_csv(tables_dir / "proposed_clinical_bio_v4_driver_gene_evidence_all_sources.csv")
    evidence["gene"] = evidence["gene"].map(base.normalize_gene)
    tiers = evidence.set_index("gene")["primary_driver_evidence_count"].fillna(0).astype(int).to_dict()
    panel = pd.read_csv(tables_dir / "proposed_clinical_bio_v4_driver_gene_panel.csv")
    panel["gene"] = panel["gene"].map(base.normalize_gene)
    role_lookup = panel.set_index("gene")["role_interpretation"].fillna("").astype(str).to_dict()

    residues = pd.read_csv(resource_dir / "v3_multi_type_residue.txt", sep="\t")
    residues["gene"] = residues["Hugo_Symbol"].map(base.normalize_gene)
    residue_set = {
        (str(row.gene), int(pos))
        for row in residues.itertuples(index=False)
        for pos in [parse_protein_position(getattr(row, "Amino_Acid_Position", None))]
        if pos is not None and str(row.gene) != "UNKNOWN"
    }
    variants = pd.read_csv(resource_dir / "v3_multi_type_variant_file.txt", sep="\t")
    variants["gene"] = variants["Hugo_Symbol"].map(base.normalize_gene)
    variant_set = set()
    for row in variants.itertuples(index=False):
        pos = parse_protein_position(getattr(row, "Amino_Acid_Position", None))
        ref = str(getattr(row, "Reference_Amino_Acid", "")).strip().upper()
        alt = str(getattr(row, "Variant_Amino_Acid", "")).strip().upper()
        gene = str(row.gene)
        if pos is not None and gene != "UNKNOWN" and ref and alt:
            variant_set.add((gene, pos, ref, alt))
    return panel, tiers, role_lookup, residue_set, variant_set


def v4_feature_columns() -> tuple[list[str], list[str], dict[str, list[str]]]:
    base_core, _base_all, _base_blocks = base.feature_columns()
    panel, _tiers, _roles, _residues, _variants = load_v4_resources()
    tier_cols = [f"driver_evidence_{metric}__source_count_{tier}" for tier in range(1, 5) for metric in DRIVER_TIER_METRICS]
    exact_cols = []
    for gene in panel["gene"].astype(str).tolist():
        exact_cols.extend(
            [
                f"driver_gene_functional_event_log_count__{gene}",
                f"driver_gene_role_matched_event_log_count__{gene}",
            ]
        )
    hotspot_cols = list(HOTSPOT_FEATURES)
    core = [*base_core, *tier_cols, *exact_cols, *hotspot_cols]
    all_cols = [*core, *base.OPTIONAL_CONTROLS]
    blocks = {
        "v4_compact_biology_only": base_core,
        "v4_compact_plus_driver_evidence_tiers": [*base_core, *tier_cols],
        "v4_compact_plus_hotspot_summary": [*base_core, *hotspot_cols],
        "v4_compact_plus_exact_driver_genes": [*base_core, *exact_cols],
        "v4_full_core": core,
        "v4_full_core_plus_optional_controls": all_cols,
    }
    return core, all_cols, blocks


def is_role_matched(row: pd.Series, role_lookup: dict[str, str]) -> bool:
    role = role_lookup.get(str(row["_gene"]), "")
    consequence = str(row["_consequence"])
    impact = str(row["_impact"])
    hotspot = bool(row["_hotspot_residue"] or row["_hotspot_exact"])
    lof = impact == "high" or consequence in ROLE_MATCH_LOF
    activating = hotspot or consequence in ROLE_MATCH_ACT
    if "tumor-suppressor" in role:
        return lof
    if "activating/oncogene" in role:
        return activating
    if "both-role" in role or "ambiguous" in role:
        return lof or activating
    return impact in {"high", "moderate"} or hotspot


def build_bio_v4_features(paths: dict, output_dir: Path, *, chunksize: int, force: bool = False) -> pd.DataFrame:
    cache_path = output_dir / "quick_bio_v4_maf_features.csv.gz"
    audit_path = output_dir / "quick_bio_v4_maf_features_audit.json"
    if cache_path.exists() and not force:
        features = pd.read_csv(cache_path, index_col=0)
        features.index = features.index.astype(str)
        return features.astype(np.float32)

    base_features, _schema = base.build_base_bio_features(paths, output_dir, chunksize=chunksize, force=force)
    core_cols, all_cols, _blocks = v4_feature_columns()
    panel, tier_lookup, role_lookup, residue_set, variant_set = load_v4_resources()
    selected_genes = set(panel["gene"].astype(str))
    sample_ids = base_features.index.astype(str).tolist()
    matrix = pd.DataFrame(0.0, index=pd.Index(sample_ids, name="sample"), columns=all_cols, dtype=np.float64)
    matrix.loc[:, base_features.columns] = base_features.reindex(matrix.index).astype(np.float64)

    tier_unique_genes = {tier: defaultdict(set) for tier in range(1, 5)}
    hotspot_unique_genes: dict[str, set[str]] = defaultdict(set)
    qc = Counter()

    mc3_dir = Path(paths["raw_data"]["mc3_source_dir"])
    maf_path = mc3_dir / "raw" / "mc3.v0.2.8.PUBLIC.maf.gz"
    sample_set = set(sample_ids)
    usecols = [
        "Tumor_Sample_Barcode",
        "Hugo_Symbol",
        "SYMBOL",
        "IMPACT",
        "Consequence",
        "Protein_position",
        "Amino_acids",
        "HGVSp_Short",
        "t_alt_count",
        "t_depth",
    ]
    for chunk in pd.read_csv(maf_path, sep="\t", usecols=usecols, dtype=str, chunksize=int(chunksize)):
        qc["rows_scanned"] += int(len(chunk))
        chunk["sample"] = chunk["Tumor_Sample_Barcode"].map(base.first_12)
        chunk = chunk[chunk["sample"].isin(sample_set)].copy()
        if chunk.empty:
            continue
        qc["rows_used"] += int(len(chunk))
        chunk["_gene"] = chunk["Hugo_Symbol"].where(chunk["Hugo_Symbol"].notna(), chunk["SYMBOL"]).map(base.normalize_gene)
        chunk["_impact"] = chunk["IMPACT"].map(base._impact)
        chunk["_consequence"] = chunk["Consequence"].map(base._consequence)
        chunk["_vaf"] = pd.to_numeric(chunk["t_alt_count"], errors="coerce") / pd.to_numeric(chunk["t_depth"], errors="coerce").replace(0, np.nan)
        chunk["_protein_pos"] = chunk["Protein_position"].map(parse_protein_position)
        aa = chunk.apply(parse_amino_acids, axis=1, result_type="expand")
        chunk["_ref_aa"] = aa[0]
        chunk["_alt_aa"] = aa[1]
        chunk["_tier"] = chunk["_gene"].map(tier_lookup).fillna(0).astype(int).clip(0, 4)
        chunk["_hotspot_residue"] = [
            (gene, pos) in residue_set if pos is not None else False
            for gene, pos in zip(chunk["_gene"], chunk["_protein_pos"])
        ]
        chunk["_hotspot_exact"] = [
            (gene, pos, ref, alt) in variant_set if pos is not None else False
            for gene, pos, ref, alt in zip(chunk["_gene"], chunk["_protein_pos"], chunk["_ref_aa"], chunk["_alt_aa"])
        ]
        chunk["_functional"] = chunk["_impact"].isin(["high", "moderate"]) | chunk["_consequence"].isin(PROTEIN_AFFECTING)
        chunk["_high_mod"] = chunk["_impact"].isin(["high", "moderate"])
        chunk["_high"] = chunk["_impact"].eq("high")
        chunk["_role_matched"] = chunk.apply(is_role_matched, axis=1, role_lookup=role_lookup)

        for tier in range(1, 5):
            tier_frame = chunk[chunk["_tier"].eq(tier)]
            hm = tier_frame[tier_frame["_high_mod"]]
            hi = tier_frame[tier_frame["_high"]]
            for sample, count in hm.groupby("sample", observed=True).size().items():
                matrix.at[sample, f"driver_evidence_high_or_moderate_impact_log_count__source_count_{tier}"] += float(count)
            for sample, count in hi.groupby("sample", observed=True).size().items():
                matrix.at[sample, f"driver_evidence_high_impact_log_count__source_count_{tier}"] += float(count)
            for sample, genes in hm.groupby("sample", observed=True)["_gene"]:
                tier_unique_genes[tier][sample].update(g for g in genes if g != "UNKNOWN")
            for sample, max_vaf in hm.groupby("sample", observed=True)["_vaf"].max().dropna().items():
                col = f"driver_evidence_max_vaf_high_or_moderate_impact__source_count_{tier}"
                matrix.at[sample, col] = max(float(matrix.at[sample, col]), float(max_vaf))

        gene_frame = chunk[chunk["_gene"].isin(selected_genes)]
        functional = gene_frame[gene_frame["_functional"]]
        role_matched = gene_frame[gene_frame["_role_matched"]]
        for (sample, gene), count in functional.groupby(["sample", "_gene"], observed=True).size().items():
            matrix.at[sample, f"driver_gene_functional_event_log_count__{gene}"] += float(count)
        for (sample, gene), count in role_matched.groupby(["sample", "_gene"], observed=True).size().items():
            matrix.at[sample, f"driver_gene_role_matched_event_log_count__{gene}"] += float(count)

        hs_residue = chunk[chunk["_hotspot_residue"]]
        hs_exact = chunk[chunk["_hotspot_exact"]]
        for sample, count in hs_exact.groupby("sample", observed=True).size().items():
            matrix.at[sample, "cancer_hotspot_exact_variant_log_count"] += float(count)
        for sample, count in hs_residue.groupby("sample", observed=True).size().items():
            matrix.at[sample, "cancer_hotspot_residue_log_count"] += float(count)
        for sample, genes in hs_residue.groupby("sample", observed=True)["_gene"]:
            hotspot_unique_genes[sample].update(g for g in genes if g != "UNKNOWN")
        for sample, max_vaf in hs_residue.groupby("sample", observed=True)["_vaf"].max().dropna().items():
            matrix.at[sample, "cancer_hotspot_max_vaf"] = max(float(matrix.at[sample, "cancer_hotspot_max_vaf"]), float(max_vaf))

    for tier in range(1, 5):
        for sample, genes in tier_unique_genes[tier].items():
            matrix.at[sample, f"driver_evidence_unique_high_or_moderate_genes_log_count__source_count_{tier}"] = float(len(genes))
    for sample, genes in hotspot_unique_genes.items():
        matrix.at[sample, "cancer_hotspot_unique_gene_log_count"] = float(len(genes))

    log_count_cols = [
        col
        for col in matrix.columns
        if col.endswith("_log_count") or "_log_count__" in col
    ]
    for col in log_count_cols:
        matrix[col] = np.log1p(matrix[col].to_numpy(dtype=np.float64))
    matrix = matrix.loc[:, all_cols].fillna(0.0).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(cache_path, compression="gzip")
    audit = {
        "feature_count": int(matrix.shape[1]),
        "sample_count": int(matrix.shape[0]),
        "core_feature_count": len(core_cols),
        "optional_control_count": len(base.OPTIONAL_CONTROLS),
        "driver_gene_count": int(len(selected_genes)),
        "rows_scanned": int(qc["rows_scanned"]),
        "rows_used": int(qc["rows_used"]),
        "nonzero_feature_count": int((matrix.sum(axis=0) != 0).sum()),
        "cache_path": str(cache_path),
        "leakage_note": "Bio v4 schema and gene/hotspot resources are fixed before training; feature-set inclusion is selected inside the inner CV loop.",
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return matrix


def make_v4_candidate_sets(bio_v4: pd.DataFrame, signatures: pd.DataFrame, mode: str):
    _core, _all, blocks = v4_feature_columns()
    features = {"bio_v4": bio_v4, "signatures": signatures}
    if mode == "bio_v4_only":
        return features, {name: ("bio_v4", cols) for name, cols in blocks.items()}
    signatures_prefixed = signatures.add_prefix("signature_matrix__")
    combined = pd.concat([signatures_prefixed, bio_v4], axis=1).fillna(0.0)
    features["combined_v4"] = combined
    sig_cols = list(signatures_prefixed.columns)
    candidates = {"signatures_only": ("combined_v4", sig_cols)}
    for name, cols in blocks.items():
        candidates[f"signatures_plus_{name}"] = ("combined_v4", [*sig_cols, *cols])
    return features, candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--endpoints", nargs="*", default=base.MAIN_ENDPOINTS)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--chunksize", type=int, default=350_000)
    parser.add_argument("--profile", choices=["tiny", "standard"], default="tiny")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--include-signatures", action="store_true")
    parser.add_argument("--output-prefix", default="quick_bio_v4_nested_fs_smoke")
    parser.add_argument("--seed-key", default=None, help="Reuse another run's split seed key for paired smoke comparisons.")
    parser.add_argument(
        "--candidate-sets",
        nargs="*",
        default=None,
        help="Optional exact candidate-set names to keep for bounded diagnostic runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    paths = base.load_paths(Path(args.paths))
    out_dir = BUNDLE_ROOT / "results" / "tables"
    bio_v4 = build_bio_v4_features(paths, out_dir, chunksize=args.chunksize, force=bool(args.force_features))
    standard = pd.read_csv(Path(paths["raw_data"]["mc3_source_dir"]) / "features" / "features_standard_sbs96_id83.csv.gz", index_col=0).fillna(0.0).astype(np.float32)
    standard.index = standard.index.astype(str)
    common = bio_v4.index.intersection(standard.index)
    bio_v4 = bio_v4.loc[common]
    standard = standard.loc[common]
    endpoints = base.load_endpoints(paths, args.endpoints)

    summaries = []
    pred_frames = []
    fold_frames = []
    summary_path = out_dir / f"{args.output_prefix}_summary.csv"
    fold_path = out_dir / f"{args.output_prefix}_folds.csv"
    pred_path = out_dir / f"{args.output_prefix}_predictions.csv.gz"
    wide_path = out_dir / f"{args.output_prefix}_wide_comparison.csv"
    manifest_path = out_dir / f"{args.output_prefix}_manifest.json"
    seed_key = args.seed_key or args.output_prefix
    modes = [("bio_v4_only", "bio_v4_nested_fs_linear")]
    if bool(args.include_signatures):
        modes.append(("signatures_plus_bio_v4", "signatures_plus_bio_v4_nested_fs_linear"))

    def write_checkpoint(status: str) -> None:
        summary_df = pd.DataFrame(summaries)
        fold_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
        pred_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
        if not summary_df.empty:
            summary_df.to_csv(summary_path, index=False)
            wide = summary_df.pivot_table(index=["endpoint", "task", "primary_metric"], columns="representation", values="primary_score", aggfunc="first").reset_index()
            wide.to_csv(wide_path, index=False)
        if not fold_df.empty:
            fold_df.to_csv(fold_path, index=False)
        if not pred_df.empty:
            pred_df.to_csv(pred_path, index=False, compression="gzip")
        manifest = {
            "status": status,
            "elapsed_seconds": round(time.time() - started, 3),
            "folds": int(args.folds),
            "profile": args.profile,
            "seed_key": seed_key,
            "endpoints_requested": list(args.endpoints),
            "endpoints_completed": sorted(set(summary_df["endpoint"])) if not summary_df.empty else [],
            "bio_v4_feature_count": int(bio_v4.shape[1]),
            "signature_feature_count": int(standard.shape[1]),
            "outputs": {
                "summary": str(summary_path),
                "folds": str(fold_path),
                "predictions": str(pred_path),
                "wide_comparison": str(wide_path),
            },
            "leakage_note": "Feature schema/resources are fixed before training; feature-set selection is performed within each outer fold's inner validation split.",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for endpoint in endpoints:
        for mode, representation in modes:
            features_by_name, candidate_sets = make_v4_candidate_sets(bio_v4, standard, mode)
            if args.candidate_sets:
                keep = [name for name in args.candidate_sets if name in candidate_sets]
                if not keep:
                    raise ValueError(f"No requested candidate sets are valid for mode {mode}: {args.candidate_sets}")
                candidate_sets = {name: candidate_sets[name] for name in keep}
            summary, pred_df, folds_df = base.run_nested_feature_selection(
                endpoint,
                features_by_name,
                candidate_sets,
                folds=int(args.folds),
                seed=base.stable_seed(seed_key, endpoint.name),
                profile=str(args.profile),
            )
            summary["representation"] = representation
            summary["candidate_feature_set_count"] = len(candidate_sets)
            summary["profile"] = args.profile
            summaries.append(summary)
            pred_df.insert(0, "representation", representation)
            pred_df.insert(0, "endpoint", endpoint.name)
            folds_df.insert(0, "representation", representation)
            pred_frames.append(pred_df)
            fold_frames.append(folds_df)
            write_checkpoint("running")
            print(json.dumps({k: summary[k] for k in ["endpoint", "representation", "primary_metric", "primary_score", "n_samples"]}, sort_keys=True), flush=True)

    write_checkpoint("completed")
    print(manifest_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
