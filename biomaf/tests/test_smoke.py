"""Smoke test: build a tiny synthetic MAF, run the full Bio MAF pipeline,
and verify the output schema against the manuscript feature manifest."""

import gzip
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biomaf import feature_names, featurize_maf  # noqa: E402
from biomaf.cli import main as cli_main  # noqa: E402

REPO_TABLES = Path(__file__).resolve().parents[1].parent / "results" / "tables"


def make_synthetic_maf(path: Path) -> None:
    cols = [
        "Tumor_Sample_Barcode", "Hugo_Symbol", "SYMBOL", "Variant_Type",
        "IMPACT", "Consequence", "CANONICAL", "BIOTYPE", "SIFT", "PolyPhen",
        "Protein_position", "Amino_acids", "HGVSp_Short",
        "t_alt_count", "t_depth",
    ]
    rows = [
        # sample 1: TP53 missense (hotspot-ish), high VAF
        ["TCGA-AA-0001-01", "TP53", "TP53", "SNP", "MODERATE", "missense_variant",
         "YES", "protein_coding", "deleterious", "probably_damaging",
         "273", "R/H", "p.R273H", "45", "100"],
        # sample 1: APC truncating
        ["TCGA-AA-0001-01", "APC", "APC", "DEL", "HIGH", "frameshift_variant",
         "YES", "protein_coding", "", "",
         "1309", "", "", "30", "90"],
        # sample 2: KRAS G12 hotspot
        ["TCGA-BB-0002-01", "KRAS", "KRAS", "SNP", "MODERATE", "missense_variant",
         "YES", "protein_coding", "deleterious", "probably_damaging",
         "12", "G/D", "p.G12D", "60", "120"],
        # sample 2: synonymous, non-panel gene
        ["TCGA-BB-0002-01", "TTN", "TTN", "SNP", "LOW", "synonymous_variant",
         "YES", "protein_coding", "tolerated", "benign",
         "1234", "A/A", "", "10", "100"],
        # sample 3: no mutations at all (all-zero row expected)
    ]
    frame = pd.DataFrame(rows, columns=cols)
    with gzip.open(path, "wt") as handle:
        frame.to_csv(handle, sep="\t", index=False)


def test_schema_matches_manifest(tmp_path=None):
    manifest = pd.read_csv(REPO_TABLES / "proposed_clinical_bio_v4_feature_table_exact.csv")["Feature"].tolist()
    names = feature_names()
    assert len(names) == 948, f"expected 948 features, got {len(names)}"
    assert names == manifest, "feature names do not match the manuscript manifest"


def test_end_to_end(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        maf = tmp / "synthetic.maf.gz"
        out = tmp / "features.csv.gz"
        make_synthetic_maf(maf)
        matrix = featurize_maf(maf, sample_ids=["TCGA-AA-0001", "TCGA-BB-0002", "TCGA-CC-0003"])
        assert matrix.shape == (3, 948), f"unexpected shape {matrix.shape}"
        # sample 3 has no mutations: all zeros
        assert (matrix.loc["TCGA-CC-0003"] == 0).all(), "empty sample should be all zeros"
        # sample 1: TP53/APC events counted
        assert matrix.loc["TCGA-AA-0001", "driver_gene_functional_event_log_count__TP53"] > 0
        assert matrix.loc["TCGA-AA-0001", "driver_gene_functional_event_log_count__APC"] > 0
        # sample 2: KRAS hotspot residue hit
        assert matrix.loc["TCGA-BB-0002", "cancer_hotspot_residue_log_count"] > 0
        # VEP fractions sum sensibly; burden controls present
        assert matrix.loc["TCGA-AA-0001", "log10_sbs_burden"] > 0
        assert matrix["vep_impact_count__moderate"].sum() > 0
        # no NaNs or infs
        assert not matrix.isna().any().any(), "NaNs in output"
        assert (matrix.abs() != float("inf")).all().all(), "infs in output"
        # CLI round-trip
        rc = cli_main(["--maf", str(maf), "--out", str(out),
                       "--samples", "TCGA-AA-0001,TCGA-BB-0002"])
        assert rc == 0
        reloaded = pd.read_csv(out, index_col=0)
        assert reloaded.shape == (2, 948)
        pd.testing.assert_frame_equal(reloaded, matrix.loc[["TCGA-AA-0001", "TCGA-BB-0002"]],
                                      check_dtype=False)
    print("smoke test passed: 3 samples x 948 features, schema matches manifest")


def make_minimal_maf(path: Path) -> None:
    """Minimal MAF: barcode + gene symbol only, no VEP annotation columns."""
    cols = ["Tumor_Sample_Barcode", "Hugo_Symbol"]
    rows = [
        ["TCGA-AA-0001-01", "TP53"],
        ["TCGA-AA-0001-01", "APC"],
        ["TCGA-BB-0002-01", "KRAS"],
    ]
    frame = pd.DataFrame(rows, columns=cols)
    with gzip.open(path, "wt") as handle:
        frame.to_csv(handle, sep="\t", index=False)


def test_no_vep_graceful_degradation(tmp_path=None):
    """No-VEP mode: pipeline completes on barcode+symbol only without crashing.
    VEP-dependent columns fall back to "unknown"; per-gene functional columns
    are zero because functionality cannot be determined without IMPACT."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        maf = tmp / "minimal.maf.gz"
        make_minimal_maf(maf)
        matrix = featurize_maf(maf, sample_ids=["TCGA-AA-0001", "TCGA-BB-0002"])
        assert matrix.shape == (2, 948), f"unexpected shape {matrix.shape}"
        # VEP-dependent columns fall back to "unknown" without annotation columns.
        vep_cols = [c for c in matrix.columns if c.startswith("vep_")]
        assert len(vep_cols) > 0, "expected VEP columns in schema"
        specific_impact = [c for c in vep_cols if "unknown" not in c and c.startswith("vep_impact_count__")]
        assert (matrix[specific_impact] == 0).all().all(), "specific VEP impact categories should be zero without VEP input"
        assert (matrix["vep_impact_count__unknown"] > 0).all(), "unknown impact should absorb events without VEP input"
        # VAF summaries are zeros without t_alt_count/t_depth.
        vaf_cols = ["vaf_mean", "vaf_std", "vaf_median", "vaf_q90", "vaf_max"]
        assert (matrix[vaf_cols] == 0).all().all(), "VAF columns should be zero-filled without depth columns"
        # Per-gene functional columns are zero without VEP: functionality
        # requires IMPACT high/moderate or protein-affecting consequence.
        # This documents the limitation; VEP is required for meaningful output.
        assert matrix.loc["TCGA-AA-0001", "driver_gene_functional_event_log_count__TP53"] == 0
        # No NaNs or infs.
        assert not matrix.isna().any().any(), "NaNs in output"
        assert (matrix.abs() != float("inf")).all().all(), "infs in output"
        # --check-input runs without error on the minimal MAF.
        rc = cli_main(["--maf", str(maf), "--check-input"])
        assert rc == 0, "--check-input should exit 0"
    print("no-VEP test passed: pipeline completes; VEP-dependent columns degrade gracefully")


if __name__ == "__main__":
    test_schema_matches_manifest()
    test_end_to_end()
    test_no_vep_graceful_degradation()
