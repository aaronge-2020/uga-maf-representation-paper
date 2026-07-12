#!/usr/bin/env python3
"""Check the MuAt-faithful tokenisers against the worked examples in the paper.

    python scripts/test_muat_tokens.py

Sanjaya et al., Genome Medicine 2023;15:47, Methods, "Preparing MuAt inputs from somatic variant
callsets" gives three concrete encodings:

    "the substitution ApCpG>ApTpG would be encoded as A[C>T]G, a diadenine deletion preceded by a
     cytosine as C[del A][del A], and a deletion breakpoint with a C>G substitution followed by a
     thymine as [SV_del][C>G]T"

The third involves an SV breakpoint symbol, which whole-exome data cannot supply, so only the
first two are checkable here. The strand attribute is checked against their definition:

    "mutation's pyrimidine reference base is on the (1) same or (2) opposite strand as a gene, or
     (3) mutation overlaps two genes on opposite strands, or (4) mutation is intergenic"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.muat_compatible import (  # noqa: E402
    annotation_token,
    enumerate_annotation_tokens,
    enumerate_position_tokens,
    mutation_motif_token,
    muat_symbolic_motif_token,
    transcriptional_strand_class,
)

failures: list[str] = []


def check(label: str, got: object, expected: object) -> None:
    ok = str(got) == str(expected)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label:34s} got {got!s:24s} expected {expected!s}")
    if not ok:
        failures.append(label)


print("MuAt-symbolic motif grammar, against the paper's worked examples")
print("-" * 78)

# "the substitution ApCpG>ApTpG would be encoded as A[C>T]G"
check(
    "ApCpG>ApTpG",
    muat_symbolic_motif_token(
        {"Reference_Allele": "C", "Tumor_Seq_Allele1": "C", "Tumor_Seq_Allele2": "T",
         "Variant_Type": "SNP", "CONTEXT": "ACG"}
    ),
    "A[C>T]G",
)

# Purine reference must be canonicalised to a pyrimidine, taking the reverse complement of the
# whole trinucleotide. TpGpA>TpApA is the reverse complement of TpCpA>TpTpA.
check(
    "TpGpA>TpApA (purine ref)",
    muat_symbolic_motif_token(
        {"Reference_Allele": "G", "Tumor_Seq_Allele1": "G", "Tumor_Seq_Allele2": "A",
         "Variant_Type": "SNP", "CONTEXT": "TGA"}
    ),
    "T[C>T]A",
)

# "a diadenine deletion preceded by a cytosine as C[del A][del A]"
check(
    "del AA preceded by C",
    muat_symbolic_motif_token(
        {"Reference_Allele": "AA", "Tumor_Seq_Allele1": "-", "Tumor_Seq_Allele2": "-",
         "Variant_Type": "DEL", "CONTEXT": "CAA"}
    ),
    "C[delA][delA]",
)

print()
print("Bounded vs unbounded grammar on the same variant")
print("-" * 78)
long_mnv = {
    "Reference_Allele": "AGGGAGTTAATG",
    "Tumor_Seq_Allele2": "GGGAGTTAATGC",
    "Variant_Type": "DNP",
    "CONTEXT": "AAG",
}
print(f"  legacy  : {mutation_motif_token(long_mnv)}")
print(f"  muat    : {muat_symbolic_motif_token(long_mnv)}")
print("  (the legacy grammar makes a 12-base substitution a single vocabulary entry)")

print()
print("Transcriptional strand, against the paper's four-class definition")
print("-" * 78)
gene_plus_c = {"Hugo_Symbol": "TP53", "Reference_Allele": "C", "STRAND": "1", "Variant_Classification": "Missense_Mutation"}
gene_minus_c = {"Hugo_Symbol": "TP53", "Reference_Allele": "C", "STRAND": "-1", "Variant_Classification": "Missense_Mutation"}
gene_minus_g = {"Hugo_Symbol": "TP53", "Reference_Allele": "G", "STRAND": "-1", "Variant_Classification": "Missense_Mutation"}
gene_plus_g = {"Hugo_Symbol": "TP53", "Reference_Allele": "G", "STRAND": "1", "Variant_Classification": "Missense_Mutation"}
intergenic = {"Hugo_Symbol": "", "Reference_Allele": "C", "STRAND": "1", "Variant_Classification": "IGR"}

# C is a pyrimidine and MAF reports the reference on the plus strand, so a C reference in a
# plus-strand gene puts the pyrimidine on the same strand as the gene.
check("pyrimidine ref (C), gene +", transcriptional_strand_class(gene_plus_c), "same")
check("pyrimidine ref (C), gene -", transcriptional_strand_class(gene_minus_c), "opposite")
# G is a purine on the plus strand, so its pyrimidine partner (C) sits on the minus strand.
check("purine ref (G), gene -", transcriptional_strand_class(gene_minus_g), "same")
check("purine ref (G), gene +", transcriptional_strand_class(gene_plus_g), "opposite")
check("intergenic", transcriptional_strand_class(intergenic), "intergenic")

print()
print("  The legacy encoder collapses all four of these to the gene's own orientation:")
for label, row in [("C/gene+", gene_plus_c), ("C/gene-", gene_minus_c), ("G/gene-", gene_minus_g), ("G/gene+", gene_plus_g)]:
    print(f"    {label:9s} legacy -> {annotation_token(row):16s}  muat -> {transcriptional_strand_class(row)}")
print("  So C/gene- and G/gene- are the same legacy token but opposite transcriptional classes.")

print()
print("Vocabulary sizes")
print("-" * 78)
ann = enumerate_annotation_tokens(strand_mode="transcriptional")
pos = enumerate_position_tokens()
check("annotation vocabulary", len(ann), 16)          # MuAt: 2 x 2 x 4 = 16
print(f"  [info] position vocabulary        {len(pos)} 1-Mb bins   (MuAt reports 2,915)")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
    raise SystemExit(1)
print("all checks passed")
