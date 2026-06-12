"""Frozen, interpretable biological MAF feature construction.

This module builds a fixed named schema from external cancer-gene resources,
curated pathways, fixed GRCh37 chromosome bins, and standard MAF/VEP annotation
fields. No feature vocabulary is learned from the cohort being cross-validated.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from utils.config import BUNDLE_ROOT, load_yaml


DEFAULT_RESOURCE_DIR = BUNDLE_ROOT / "config" / "feature_resources"

STANDARD_CHROMS = [str(idx) for idx in range(1, 23)] + ["X", "Y", "MT"]
GRCH37_CHROM_LENGTHS = {
    "1": 249_250_621,
    "2": 243_199_373,
    "3": 198_022_430,
    "4": 191_154_276,
    "5": 180_915_260,
    "6": 171_115_067,
    "7": 159_138_663,
    "8": 146_364_022,
    "9": 141_213_431,
    "10": 135_534_747,
    "11": 135_006_516,
    "12": 133_851_895,
    "13": 115_169_878,
    "14": 107_349_540,
    "15": 102_531_392,
    "16": 90_354_753,
    "17": 81_195_210,
    "18": 78_077_248,
    "19": 59_128_983,
    "20": 63_025_520,
    "21": 48_129_895,
    "22": 51_304_566,
    "X": 155_270_560,
    "Y": 59_373_566,
    "MT": 16_569,
}

MAF_BIO_USECOLS = [
    "Tumor_Sample_Barcode",
    "Hugo_Symbol",
    "SYMBOL",
    "HGNC_ID",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Variant_Type",
    "Variant_Classification",
    "Reference_Allele",
    "Tumor_Seq_Allele1",
    "Tumor_Seq_Allele2",
    "IMPACT",
    "Consequence",
    "CANONICAL",
    "BIOTYPE",
    "STRAND",
    "EXON",
    "INTRON",
    "SIFT",
    "PolyPhen",
    "COSMIC",
    "CLIN_SIG",
    "t_alt_count",
    "t_ref_count",
    "t_depth",
    "CONTEXT",
]

VARIANT_TYPES = ["snp", "dnp", "tnp", "onp", "ins", "del", "other", "unknown"]
VARIANT_CLASSES = [
    "missense_mutation",
    "silent",
    "nonsense_mutation",
    "frame_shift_del",
    "frame_shift_ins",
    "splice_site",
    "translation_start_site",
    "nonstop_mutation",
    "in_frame_del",
    "in_frame_ins",
    "rna",
    "intron",
    "3utr",
    "5utr",
    "igr",
    "other",
    "unknown",
]
IMPACTS = ["high", "moderate", "low", "modifier", "unknown"]
CONSEQUENCES = [
    "missense_variant",
    "synonymous_variant",
    "stop_gained",
    "frameshift_variant",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "splice_region_variant",
    "inframe_deletion",
    "inframe_insertion",
    "start_lost",
    "stop_lost",
    "stop_retained_variant",
    "protein_altering_variant",
    "coding_sequence_variant",
    "transcript_ablation",
    "intron_variant",
    "3_prime_utr_variant",
    "5_prime_utr_variant",
    "upstream_gene_variant",
    "downstream_gene_variant",
    "non_coding_transcript_exon_variant",
    "non_coding_transcript_variant",
    "mature_mirna_variant",
    "incomplete_terminal_codon_variant",
    "other",
    "unknown",
]
CANONICAL = ["yes", "no", "unknown"]
BIOTYPES = ["protein_coding", "lncrna", "nmd", "retained_intron", "processed_transcript", "other", "unknown"]
SIFT = ["deleterious", "deleterious_low_confidence", "tolerated", "tolerated_low_confidence", "unknown"]
POLYPHEN = ["probably_damaging", "possibly_damaging", "benign", "unknown"]
STRANDS = ["plus", "minus", "unknown"]
SUBSTITUTIONS = ["C_A", "C_G", "C_T", "T_A", "T_C", "T_G"]
INDEL_TYPES = ["insertion", "deletion", "unknown"]
INDEL_LENGTH_BINS = ["1", "2_5", "6_10", "gt10", "unknown"]

DAMAGING_IMPACTS = {"high", "moderate"}
DAMAGING_VARIANT_CLASSES = {
    "missense_mutation",
    "nonsense_mutation",
    "frame_shift_del",
    "frame_shift_ins",
    "splice_site",
    "translation_start_site",
    "nonstop_mutation",
    "in_frame_del",
    "in_frame_ins",
}
TRUNCATING_VARIANT_CLASSES = {
    "nonsense_mutation",
    "frame_shift_del",
    "frame_shift_ins",
    "splice_site",
    "translation_start_site",
    "nonstop_mutation",
}
TRUNCATING_CONSEQUENCES = {
    "stop_gained",
    "frameshift_variant",
    "splice_donor_variant",
    "splice_acceptor_variant",
    "start_lost",
    "stop_lost",
    "transcript_ablation",
}

COMPLEMENT = str.maketrans("ACGT", "TGCA")


@dataclass(frozen=True)
class FeatureResources:
    resources_dir: Path
    gene_panel: tuple[str, ...]
    pathways: Mapping[str, tuple[str, ...]]
    hgnc_alias_map: Mapping[str, str]
    manifest: Mapping[str, object]


@dataclass
class FeatureLayout:
    columns: list[str]
    schema_rows: list[dict[str, object]]
    log_count_columns: set[str]
    fraction_specs: list[tuple[str, str, str]]

    @property
    def column_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.columns)}


@dataclass
class BiologicalMafResult:
    features: pd.DataFrame
    schema: pd.DataFrame
    audit: pd.DataFrame
    resources: FeatureResources


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_token(value: object, fallback: str = "unknown") -> str:
    text = "" if value is None else str(value).strip().lower()
    if text in {"", "-", ".", "nan", "none", "na", "n/a"}:
        text = fallback
    text = text.replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or fallback


def normalize_gene_symbol(value: object, fallback: str = "UNKNOWN") -> str:
    text = "" if value is None else str(value).strip().upper()
    if text in {"", "-", ".", "NAN", "NONE", "NA", "N/A", "UNKNOWN"}:
        return fallback
    return re.sub(r"[^A-Z0-9]+", "_", text).strip("_") or fallback


def normalize_chromosome(value: object) -> str:
    text = "" if value is None else str(value).strip().upper()
    text = text.replace("CHR", "")
    if text in {"M", "MT", "MTR", "MITO"}:
        return "MT"
    text = text.lstrip("0")
    return text if text in set(STANDARD_CHROMS) else "unknown"


def _present(value: object) -> bool:
    text = "" if value is None else str(value).strip()
    return text.upper() not in {"", "-", ".", "NAN", "NONE", "NA", "N/A", "UNKNOWN"}


def _canonical_token(value: object) -> str:
    text = normalize_token(value)
    if text in {"yes", "y", "1", "true"}:
        return "yes"
    if text in {"no", "n", "0", "false"}:
        return "no"
    return "unknown"


def _biotype_token(value: object) -> str:
    text = normalize_token(value)
    if text in {"protein_coding", "lncrna", "retained_intron", "processed_transcript"}:
        return text
    if text in {"nonsense_mediated_decay", "nmd"}:
        return "nmd"
    return "unknown" if text == "unknown" else "other"


def _sift_token(value: object) -> str:
    text = normalize_token(str(value).split("(")[0])
    if text in set(SIFT):
        return text
    if text.startswith("deleterious"):
        return "deleterious_low_confidence" if "low_confidence" in text else "deleterious"
    if text.startswith("tolerated"):
        return "tolerated_low_confidence" if "low_confidence" in text else "tolerated"
    return "unknown"


def _polyphen_token(value: object) -> str:
    text = normalize_token(str(value).split("(")[0])
    if text in set(POLYPHEN):
        return text
    if text.startswith("probably"):
        return "probably_damaging"
    if text.startswith("possibly"):
        return "possibly_damaging"
    if text.startswith("benign"):
        return "benign"
    return "unknown"


def _strand_token(value: object) -> str:
    text = str(value).strip()
    if text in {"1", "+", "+1"}:
        return "plus"
    if text in {"-1", "-"}:
        return "minus"
    return "unknown"


def _first_consequence(value: object) -> str:
    text = "" if value is None else str(value)
    if not _present(text):
        return "unknown"
    parts = [normalize_token(part) for part in re.split(r"[,;|&]", text) if normalize_token(part)]
    for part in parts:
        if part in set(CONSEQUENCES):
            return part
    return "other"


def _variant_type_token(value: object) -> str:
    text = normalize_token(value)
    return text if text in set(VARIANT_TYPES) else "other"


def _variant_class_token(value: object) -> str:
    text = normalize_token(value)
    return text if text in set(VARIANT_CLASSES) else "other"


def _impact_token(value: object) -> str:
    text = normalize_token(value)
    return text if text in set(IMPACTS) else "unknown"


def _safe_numeric(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _split_pipe_values(value: object) -> list[str]:
    if not _present(value):
        return []
    return [part.strip() for part in str(value).split("|") if _present(part)]


def ensure_feature_resources(resources_dir: str | Path | None = None, *, refresh: bool = False) -> FeatureResources:
    """Validate bundled Bio MAF resources without network access."""

    if refresh:
        raise RuntimeError(
            "Active manuscript reproduction does not download or refresh Bio MAF resources. "
            "Restore the vendored resource bundle before running the production pipeline."
        )
    return load_feature_resources(resources_dir)


def load_feature_resources(resources_dir: str | Path | None = None) -> FeatureResources:
    resources_dir = Path(resources_dir or DEFAULT_RESOURCE_DIR)
    oncokb_path = resources_dir / "oncokb_cancer_genes.tsv"
    hgnc_alias_path = resources_dir / "hgnc_symbol_aliases.tsv"
    pathway_path = resources_dir / "curated_pathways.yaml"
    manifest_path = resources_dir / "resource_manifest.json"
    required = [oncokb_path, hgnc_alias_path, pathway_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing vendored Bio MAF resource file(s): "
            + ", ".join(missing)
            + ". Restore the dataset/resource bundle instead of downloading during manuscript reproduction."
        )

    alias_frame = pd.read_csv(hgnc_alias_path, sep="\t", dtype=str).fillna("")
    alias_map = {normalize_gene_symbol(row.alias): normalize_gene_symbol(row.approved_symbol) for row in alias_frame.itertuples(index=False)}
    pathways_raw = load_yaml(pathway_path)
    pathways: dict[str, tuple[str, ...]] = {}
    for name, genes in pathways_raw.items():
        clean = sorted({alias_map.get(normalize_gene_symbol(gene), normalize_gene_symbol(gene)) for gene in (genes or [])})
        pathways[normalize_token(name)] = tuple(gene for gene in clean if gene != "UNKNOWN")

    oncokb = pd.read_csv(oncokb_path, sep="\t", dtype=str).fillna("")
    oncokb_genes = {alias_map.get(normalize_gene_symbol(value), normalize_gene_symbol(value)) for value in oncokb["hugo_symbol"]}
    pathway_genes = {gene for genes in pathways.values() for gene in genes}
    gene_panel = tuple(sorted(gene for gene in oncokb_genes.union(pathway_genes) if gene != "UNKNOWN"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return FeatureResources(resources_dir=resources_dir, gene_panel=gene_panel, pathways=pathways, hgnc_alias_map=alias_map, manifest=manifest)


def _maf_header_columns(maf_path: str | Path) -> list[str]:
    path = Path(maf_path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            return line.rstrip("\n").split("\t")
    return []


def _available_maf_columns(maf_path: str | Path) -> list[str]:
    available = set(_maf_header_columns(maf_path))
    usecols = [column for column in MAF_BIO_USECOLS if column in available]
    if "Tumor_Sample_Barcode" not in usecols:
        raise ValueError("MAF input is missing Tumor_Sample_Barcode")
    return usecols


def _add_feature(layout: FeatureLayout, name: str, family: str, source_field: str, transform: str, notes: str = "", *, log_count: bool = False) -> None:
    layout.columns.append(name)
    layout.schema_rows.append(
        {
            "feature": name,
            "family": family,
            "source_field": source_field,
            "transform": transform,
            "interpretability": "named",
            "notes": notes,
        }
    )
    if log_count:
        layout.log_count_columns.add(name)


def _add_count_fraction_family(layout: FeatureLayout, family: str, tokens: Iterable[str], source_field: str, denominator: str) -> None:
    for token in tokens:
        count_col = f"{family}_count__{token}"
        fraction_col = f"{family}_fraction__{token}"
        _add_feature(layout, count_col, family, source_field, "log1p(count)", log_count=True)
        _add_feature(layout, fraction_col, family, source_field, f"count / {denominator}")
        layout.fraction_specs.append((fraction_col, count_col, denominator))


def build_feature_layout(resources: FeatureResources, *, include_locus_bins: bool = True) -> FeatureLayout:
    layout = FeatureLayout(columns=[], schema_rows=[], log_count_columns=set(), fraction_specs=[])
    for gene in resources.gene_panel:
        _add_feature(layout, f"gene_any__{gene}", "explicit_gene", "Hugo_Symbol/SYMBOL/HGNC_ID", "log1p(count of mutations in gene)", log_count=True)
        _add_feature(layout, f"gene_damaging__{gene}", "explicit_gene", "IMPACT/Variant_Classification", "log1p(damaging count in gene)", "damaging = IMPACT HIGH/MODERATE or damaging class", log_count=True)
        _add_feature(layout, f"gene_high_impact__{gene}", "explicit_gene", "IMPACT", "log1p(HIGH-impact count in gene)", "HIGH means VEP IMPACT == HIGH", log_count=True)
        _add_feature(layout, f"gene_truncating_splice__{gene}", "explicit_gene", "Variant_Classification/Consequence", "log1p(truncating/splice count in gene)", log_count=True)
        _add_feature(layout, f"gene_max_vaf__{gene}", "explicit_gene", "t_alt_count/t_depth", "maximum VAF in gene; 0 if absent")

    for name, source, transform in [
        ("residual_gene_total_log_count", "Hugo_Symbol/SYMBOL/HGNC_ID", "log1p(non-panel mutation count)"),
        ("residual_gene_damaging_log_count", "IMPACT/Variant_Classification", "log1p(damaging non-panel mutation count)"),
        ("residual_gene_high_impact_log_count", "IMPACT", "log1p(HIGH-impact non-panel mutation count)"),
        ("residual_gene_unique_log_count", "Hugo_Symbol/SYMBOL/HGNC_ID", "log1p(unique non-panel genes mutated)"),
    ]:
        _add_feature(layout, name, "residual_gene", source, transform, log_count=True)
    _add_feature(layout, "residual_gene_fraction_of_events", "residual_gene", "Hugo_Symbol/SYMBOL/HGNC_ID", "non-panel events / total events")
    _add_feature(layout, "residual_gene_max_vaf", "residual_gene", "t_alt_count/t_depth", "maximum VAF among non-panel mutations; 0 if absent")

    for pathway in sorted(resources.pathways):
        _add_feature(layout, f"pathway_total__{pathway}", "pathway", "curated_pathways.yaml", "log1p(count of mutations in pathway genes)", log_count=True)
        _add_feature(layout, f"pathway_damaging__{pathway}", "pathway", "IMPACT/Variant_Classification", "log1p(damaging mutations in pathway genes)", log_count=True)
        _add_feature(layout, f"pathway_high_impact__{pathway}", "pathway", "IMPACT", "log1p(HIGH-impact mutations in pathway genes)", log_count=True)
        _add_feature(layout, f"pathway_unique_genes__{pathway}", "pathway", "curated_pathways.yaml", "log1p(unique mutated pathway genes)", log_count=True)
        _add_feature(layout, f"pathway_fraction__{pathway}", "pathway", "curated_pathways.yaml", "pathway event count / total events")
        _add_feature(layout, f"pathway_max_vaf__{pathway}", "pathway", "t_alt_count/t_depth", "maximum VAF among pathway mutations; 0 if absent")
        layout.fraction_specs.append((f"pathway_fraction__{pathway}", f"pathway_total__{pathway}", "total_events"))

    for chrom in STANDARD_CHROMS:
        count_col = f"chrom_count__chr{chrom}"
        fraction_col = f"chrom_fraction__chr{chrom}"
        _add_feature(layout, count_col, "chromosome", "Chromosome", "log1p(count on fixed GRCh37 chromosome)", log_count=True)
        _add_feature(layout, fraction_col, "chromosome", "Chromosome", "chromosome count / total events")
        layout.fraction_specs.append((fraction_col, count_col, "total_events"))
        if include_locus_bins:
            n_bins = int(math.ceil(GRCH37_CHROM_LENGTHS[chrom] / 1_000_000))
            for bin_idx in range(n_bins):
                _add_feature(layout, f"locus_mb__chr{chrom}__{bin_idx:04d}", "locus_1mb", "Chromosome/Start_Position", "log1p(count in fixed GRCh37 1-Mb bin)", log_count=True)

    _add_count_fraction_family(layout, "variant_type", VARIANT_TYPES, "Variant_Type", "total_events")
    _add_count_fraction_family(layout, "variant_class", VARIANT_CLASSES, "Variant_Classification", "total_events")
    _add_count_fraction_family(layout, "impact", IMPACTS, "IMPACT", "total_events")
    _add_count_fraction_family(layout, "consequence", CONSEQUENCES, "Consequence", "total_events")
    _add_count_fraction_family(layout, "canonical", CANONICAL, "CANONICAL", "total_events")
    _add_count_fraction_family(layout, "biotype", BIOTYPES, "BIOTYPE", "total_events")
    _add_count_fraction_family(layout, "sift", SIFT, "SIFT", "total_events")
    _add_count_fraction_family(layout, "polyphen", POLYPHEN, "PolyPhen", "total_events")
    _add_count_fraction_family(layout, "strand", STRANDS, "STRAND", "total_events")

    for name, field in [("cosmic_present", "COSMIC"), ("clin_sig_present", "CLIN_SIG"), ("exon_present", "EXON"), ("intron_present", "INTRON")]:
        count_col = f"{name}_count"
        fraction_col = f"{name}_fraction"
        _add_feature(layout, count_col, "annotation_presence", field, "log1p(count where field is present)", log_count=True)
        _add_feature(layout, fraction_col, "annotation_presence", field, "present count / total events")
        layout.fraction_specs.append((fraction_col, count_col, "total_events"))

    for token in SUBSTITUTIONS:
        count_col = f"substitution_count__{token}"
        fraction_col = f"substitution_fraction__{token}"
        _add_feature(layout, count_col, "coarse_allele", "Reference_Allele/Tumor_Seq_Allele2", "log1p(count of pyrimidine-normalized SNV class)", log_count=True)
        _add_feature(layout, fraction_col, "coarse_allele", "Reference_Allele/Tumor_Seq_Allele2", "substitution count / total SNV events")
        layout.fraction_specs.append((fraction_col, count_col, "snv_events"))
    for token in INDEL_TYPES:
        count_col = f"indel_count__{token}"
        _add_feature(layout, count_col, "coarse_indel", "Variant_Type", "log1p(insertion/deletion count)", log_count=True)
    for token in INDEL_LENGTH_BINS:
        _add_feature(layout, f"indel_length_bin__{token}", "coarse_indel", "Reference_Allele/Tumor_Seq_Allele2/Start_End", "log1p(indel count in length bin)", log_count=True)

    for name, source, transform in [
        ("summary_total_events_log", "Tumor_Sample_Barcode", "log1p(total events)"),
        ("summary_snv_events_log", "Variant_Type/alleles", "log1p(SNV/SNP events)"),
        ("summary_indel_events_log", "Variant_Type", "log1p(INS/DEL events)"),
        ("summary_valid_vaf_count_log", "t_alt_count/t_depth", "log1p(events with valid VAF)"),
    ]:
        _add_feature(layout, name, "numeric_summary", source, transform, log_count=True)
    for name, transform in [
        ("summary_vaf_mean", "mean VAF"),
        ("summary_vaf_std", "VAF standard deviation"),
        ("summary_vaf_median", "median VAF"),
        ("summary_vaf_q90", "90th percentile VAF"),
        ("summary_vaf_max", "maximum VAF"),
        ("summary_vaf_high_fraction", "fraction with VAF >= 0.35"),
        ("summary_vaf_low_fraction", "fraction with VAF <= 0.10"),
        ("summary_depth_mean", "mean tumour depth"),
        ("summary_depth_median", "median tumour depth"),
        ("summary_depth_q90", "90th percentile tumour depth"),
    ]:
        _add_feature(layout, name, "numeric_summary", "t_alt_count/t_depth/t_ref_count", transform)
    return layout


def _preferred_gene(record: Mapping[str, object], alias_map: Mapping[str, str]) -> str:
    for field in ("Hugo_Symbol", "SYMBOL", "HGNC_ID"):
        gene = normalize_gene_symbol(record.get(field))
        if gene != "UNKNOWN":
            return alias_map.get(gene, gene)
    return "UNKNOWN"


def _is_damaging(impact: str, variant_class: str) -> bool:
    return impact in DAMAGING_IMPACTS or variant_class in DAMAGING_VARIANT_CLASSES


def _is_truncating_splice(variant_class: str, consequence: str) -> bool:
    return variant_class in TRUNCATING_VARIANT_CLASSES or consequence in TRUNCATING_CONSEQUENCES


def _alternate_allele(record: Mapping[str, object]) -> str:
    ref = str(record.get("Reference_Allele", "")).upper()
    allele2 = str(record.get("Tumor_Seq_Allele2", "")).upper()
    allele1 = str(record.get("Tumor_Seq_Allele1", "")).upper()
    if allele2 and allele2 not in {"-", ".", "NAN"} and allele2 != ref:
        return allele2
    return allele1


def _substitution_token(ref: object, alt: object) -> str | None:
    ref_text = str(ref).upper()
    alt_text = str(alt).upper()
    if len(ref_text) != 1 or len(alt_text) != 1 or ref_text not in "ACGT" or alt_text not in "ACGT" or ref_text == alt_text:
        return None
    if ref_text in {"A", "G"}:
        ref_text = ref_text.translate(COMPLEMENT)
        alt_text = alt_text.translate(COMPLEMENT)
    token = f"{ref_text}_{alt_text}"
    return token if token in set(SUBSTITUTIONS) else None


def _indel_type_token(variant_type: str) -> str:
    if variant_type == "ins":
        return "insertion"
    if variant_type == "del":
        return "deletion"
    return "unknown"


def _indel_length_bin(record: Mapping[str, object]) -> str:
    ref = str(record.get("Reference_Allele", ""))
    alt = _alternate_allele(record)
    length = abs(len(ref) - len(alt))
    if length <= 0:
        start = _safe_numeric(record.get("Start_Position"))
        end = _safe_numeric(record.get("End_Position"))
        if math.isfinite(start) and math.isfinite(end) and end >= start:
            length = int(end - start + 1)
    if length <= 0:
        return "unknown"
    if length == 1:
        return "1"
    if length <= 5:
        return "2_5"
    if length <= 10:
        return "6_10"
    return "gt10"


def _safe_divide(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=np.float32)
    valid = denom > 0
    out[valid] = num[valid] / denom[valid]
    return out


def build_biological_maf_stack(
    maf_path: str | Path,
    patient_ids: Iterable[str],
    *,
    resources_dir: str | Path | None = None,
    chunksize: int = 350_000,
    include_locus_bins: bool = True,
    target_gene_exclusions: Iterable[str] | None = None,
) -> BiologicalMafResult:
    """Build a frozen named MAF event feature matrix.

    ``target_gene_exclusions`` is intended only for explicit target-gene
    ablation QC endpoints.  The main non-circular benchmark leaves it empty.
    """

    resources = load_feature_resources(resources_dir)
    layout = build_feature_layout(resources, include_locus_bins=include_locus_bins)
    col_index = layout.column_index
    patients = pd.Index([str(patient) for patient in patient_ids], name="sample")
    patient_to_row = {patient: idx for idx, patient in enumerate(patients)}
    patient_set = set(patient_to_row)
    matrix = np.zeros((len(patients), len(layout.columns)), dtype=np.float32)
    total_events = np.zeros(len(patients), dtype=np.float32)
    snv_events = np.zeros(len(patients), dtype=np.float32)
    indel_events = np.zeros(len(patients), dtype=np.float32)
    valid_vaf_counts = np.zeros(len(patients), dtype=np.float32)
    vaf_values: list[list[float]] = [[] for _ in patients]
    depth_values: list[list[float]] = [[] for _ in patients]
    residual_gene_sets: list[set[str]] = [set() for _ in patients]
    pathway_gene_sets: dict[str, list[set[str]]] = {pathway: [set() for _ in patients] for pathway in resources.pathways}
    gene_panel = set(resources.gene_panel)
    target_exclusions = {normalize_gene_symbol(value) for value in (target_gene_exclusions or [])}
    pathway_gene_lookup = {pathway: set(genes) for pathway, genes in resources.pathways.items()}
    standard_chrom_set = set(STANDARD_CHROMS)
    target_excluded_events = 0
    unknown_gene_events = 0
    panel_events = 0
    residual_events = 0
    usecols = _available_maf_columns(maf_path)

    def add(row: int, column: str, value: float = 1.0) -> None:
        idx = col_index.get(column)
        if idx is not None:
            matrix[row, idx] += float(value)

    def max_update(row: int, column: str, value: float) -> None:
        if not math.isfinite(value):
            return
        idx = col_index.get(column)
        if idx is not None and value > matrix[row, idx]:
            matrix[row, idx] = float(value)

    for chunk in pd.read_csv(maf_path, sep="\t", usecols=usecols, dtype=str, chunksize=int(chunksize), low_memory=False):
        chunk["patient_id"] = chunk["Tumor_Sample_Barcode"].astype(str).str[:12]
        chunk = chunk[chunk["patient_id"].isin(patient_set)]
        if chunk.empty:
            continue
        for record in chunk.to_dict(orient="records"):
            sample = str(record.get("patient_id", ""))
            row = patient_to_row.get(sample)
            if row is None:
                continue
            gene = _preferred_gene(record, resources.hgnc_alias_map)
            if gene in target_exclusions:
                target_excluded_events += 1
                continue
            if gene == "UNKNOWN":
                unknown_gene_events += 1
            variant_type = _variant_type_token(record.get("Variant_Type"))
            variant_class = _variant_class_token(record.get("Variant_Classification"))
            impact = _impact_token(record.get("IMPACT"))
            consequence = _first_consequence(record.get("Consequence"))
            damaging = _is_damaging(impact, variant_class)
            high_impact = impact == "high"
            truncating_splice = _is_truncating_splice(variant_class, consequence)
            alt = _safe_numeric(record.get("t_alt_count"))
            depth = _safe_numeric(record.get("t_depth"))
            vaf = alt / depth if math.isfinite(alt) and math.isfinite(depth) and depth > 0 else float("nan")

            total_events[row] += 1.0
            total_col = col_index.get("summary_total_events_log")
            if total_col is not None:
                matrix[row, total_col] += 1.0

            is_panel_gene = gene in gene_panel
            if is_panel_gene:
                panel_events += 1
                add(row, f"gene_any__{gene}")
                if damaging:
                    add(row, f"gene_damaging__{gene}")
                if high_impact:
                    add(row, f"gene_high_impact__{gene}")
                if truncating_splice:
                    add(row, f"gene_truncating_splice__{gene}")
                max_update(row, f"gene_max_vaf__{gene}", vaf)
            else:
                residual_events += 1
                add(row, "residual_gene_total_log_count")
                if damaging:
                    add(row, "residual_gene_damaging_log_count")
                if high_impact:
                    add(row, "residual_gene_high_impact_log_count")
                if gene != "UNKNOWN":
                    residual_gene_sets[row].add(gene)
                max_update(row, "residual_gene_max_vaf", vaf)

            for pathway, genes in pathway_gene_lookup.items():
                if gene not in genes:
                    continue
                add(row, f"pathway_total__{pathway}")
                if damaging:
                    add(row, f"pathway_damaging__{pathway}")
                if high_impact:
                    add(row, f"pathway_high_impact__{pathway}")
                pathway_gene_sets[pathway][row].add(gene)
                max_update(row, f"pathway_max_vaf__{pathway}", vaf)

            chrom = normalize_chromosome(record.get("Chromosome"))
            if chrom in standard_chrom_set:
                add(row, f"chrom_count__chr{chrom}")
                position = _safe_numeric(record.get("Start_Position"))
                if include_locus_bins and math.isfinite(position) and position > 0:
                    bin_idx = int((position - 1) // 1_000_000)
                    if bin_idx < int(math.ceil(GRCH37_CHROM_LENGTHS[chrom] / 1_000_000)):
                        add(row, f"locus_mb__chr{chrom}__{bin_idx:04d}")

            add(row, f"variant_type_count__{variant_type}")
            add(row, f"variant_class_count__{variant_class}")
            add(row, f"impact_count__{impact}")
            add(row, f"consequence_count__{consequence}")
            add(row, f"canonical_count__{_canonical_token(record.get('CANONICAL'))}")
            add(row, f"biotype_count__{_biotype_token(record.get('BIOTYPE'))}")
            add(row, f"sift_count__{_sift_token(record.get('SIFT'))}")
            add(row, f"polyphen_count__{_polyphen_token(record.get('PolyPhen'))}")
            add(row, f"strand_count__{_strand_token(record.get('STRAND'))}")
            for field, column in [("COSMIC", "cosmic_present_count"), ("CLIN_SIG", "clin_sig_present_count"), ("EXON", "exon_present_count"), ("INTRON", "intron_present_count")]:
                if _present(record.get(field)):
                    add(row, column)

            substitution = _substitution_token(record.get("Reference_Allele"), _alternate_allele(record))
            if variant_type == "snp" or substitution is not None:
                snv_events[row] += 1.0
                add(row, "summary_snv_events_log")
            if substitution is not None:
                add(row, f"substitution_count__{substitution}")
            if variant_type in {"ins", "del"}:
                indel_events[row] += 1.0
                add(row, "summary_indel_events_log")
                add(row, f"indel_count__{_indel_type_token(variant_type)}")
                add(row, f"indel_length_bin__{_indel_length_bin(record)}")

            if math.isfinite(vaf):
                valid_vaf_counts[row] += 1.0
                add(row, "summary_valid_vaf_count_log")
                vaf_values[row].append(float(vaf))
            if math.isfinite(depth) and depth > 0:
                depth_values[row].append(float(depth))

    for row, genes in enumerate(residual_gene_sets):
        matrix[row, col_index["residual_gene_unique_log_count"]] = float(len(genes))
    for pathway, sets in pathway_gene_sets.items():
        col = col_index[f"pathway_unique_genes__{pathway}"]
        for row, genes in enumerate(sets):
            matrix[row, col] = float(len(genes))

    denom_map = {"total_events": total_events, "snv_events": snv_events, "indel_events": indel_events}
    for fraction_col, numerator_col, denom_name in layout.fraction_specs:
        frac_idx = col_index[fraction_col]
        numerator = matrix[:, col_index[numerator_col]].astype(np.float32, copy=False)
        matrix[:, frac_idx] = _safe_divide(numerator, denom_map[denom_name])
    matrix[:, col_index["residual_gene_fraction_of_events"]] = _safe_divide(matrix[:, col_index["residual_gene_total_log_count"]], total_events)

    for row, values in enumerate(vaf_values):
        if values:
            arr = np.asarray(values, dtype=np.float32)
            matrix[row, col_index["summary_vaf_mean"]] = float(np.mean(arr))
            matrix[row, col_index["summary_vaf_std"]] = float(np.std(arr))
            matrix[row, col_index["summary_vaf_median"]] = float(np.median(arr))
            matrix[row, col_index["summary_vaf_q90"]] = float(np.quantile(arr, 0.90))
            matrix[row, col_index["summary_vaf_max"]] = float(np.max(arr))
            matrix[row, col_index["summary_vaf_high_fraction"]] = float(np.mean(arr >= 0.35))
            matrix[row, col_index["summary_vaf_low_fraction"]] = float(np.mean(arr <= 0.10))
    for row, values in enumerate(depth_values):
        if values:
            arr = np.asarray(values, dtype=np.float32)
            matrix[row, col_index["summary_depth_mean"]] = float(np.mean(arr))
            matrix[row, col_index["summary_depth_median"]] = float(np.median(arr))
            matrix[row, col_index["summary_depth_q90"]] = float(np.quantile(arr, 0.90))

    if layout.log_count_columns:
        log_indices = [col_index[name] for name in layout.log_count_columns]
        matrix[:, log_indices] = np.log1p(matrix[:, log_indices])

    features = pd.DataFrame(matrix.astype(np.float32, copy=False), index=patients, columns=layout.columns)
    schema = pd.DataFrame(layout.schema_rows)
    audit = pd.DataFrame(
        [
            {
                "profile": "explicit_biological_maf_v2",
                "sample_count": int(len(patients)),
                "feature_count": int(features.shape[1]),
                "panel_gene_count": int(len(resources.gene_panel)),
                "pathway_count": int(len(resources.pathways)),
                "total_events": int(total_events.sum()),
                "panel_events": int(panel_events),
                "residual_events": int(residual_events),
                "unknown_gene_events": int(unknown_gene_events),
                "target_excluded_events": int(target_excluded_events),
                "include_locus_bins": bool(include_locus_bins),
                "schema_type": "collision_free_named_columns_plus_residual_aggregates",
                "high_impact_definition": "IMPACT == HIGH",
                "damaging_definition": "IMPACT in {HIGH, MODERATE} or damaging/truncating variant class",
                "resource_manifest": json.dumps(dict(resources.manifest), sort_keys=True),
            }
        ]
    )
    return BiologicalMafResult(features=features, schema=schema, audit=audit, resources=resources)
