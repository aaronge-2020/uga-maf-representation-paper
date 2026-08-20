#!/usr/bin/env python3
"""Check every concrete input file the pipeline opens, in one pass.

Written after four separate runs each died on a different missing file, hours apart, because the
published Hugging Face bundle is incomplete. Rather than discovering gaps one crash at a time,
this lists every required path up front and says which endpoints each gap blocks.

    python scripts/audit_bundle_inputs.py
    python scripts/audit_bundle_inputs.py --datasets-dir D:\\path\\to\\github_exports_datasets

Exit code is 0 only when nothing required is missing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_datasets_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("CGR_DATASETS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    fetched = REPO_ROOT.parent / "hf_bundle" / "github_exports_datasets"
    if fetched.exists():
        return fetched
    return REPO_ROOT.parent / "github_exports_datasets"


# (relative path under the bundle, what needs it, required?)
BUNDLE_INPUTS: list[tuple[str, str, bool]] = [
    # --- TCGA MC3: every MC3 endpoint ---
    ("datasets/tcga_mc3/raw/mc3.v0.2.8.PUBLIC.maf.gz", "all MC3 endpoints, MuAt", True),
    ("datasets/tcga_mc3/raw/TCGA-CDR-SupplementalTableS1.xlsx", "cancer type, OS, PFI", True),
    ("datasets/tcga_mc3/raw/clinical_PANCAN_patient_with_followup.tsv", "cancer-type label loader", True),
    ("datasets/tcga_mc3/raw/TCGA_mastercalls.abs_tables_JSedit.fixed.txt", "purity/ploidy controls", False),

    # --- HRD ---
    ("datasets/tcga_brca_hrd/cohort/final_analysis_cohort.tsv", "HRD endpoints", True),
    ("datasets/tcga_brca_hrd/DDRscores.txt", "HRD scores (also enables pan-cancer/OV HRD)", False),

    # --- Kucab damage_class ---
    ("datasets/kucab_mendeley/raw_mutation_tables/README.txt", "Kucab damage_class (treatment map)", True),
    ("datasets/kucab_mendeley/raw_mutation_tables/denovo_subclone_subs_final.txt", "Kucab damage_class (SBS)", True),
    ("datasets/kucab_mendeley/raw_mutation_tables/denovo_subclone_doublesub_final.txt", "Kucab damage_class (DBS)", True),
    ("datasets/kucab_mendeley/raw_mutation_tables/denovo_subclone_indels.final.txt", "Kucab damage_class (indels)", True),

    # --- Reference + signature channel definitions ---
    ("references/grch37/cgr_validation/data/Signatures/COSMIC_v3.5_SBS_GRCh37.txt", "Kucab spectra (96 channels)", True),
    ("references/grch37/cgr_validation/data/Signatures/COSMIC_v3.5_DBS_GRCh37.txt", "Kucab spectra (78 channels)", True),
    ("references/grch37/cgr_validation/data/Signatures/COSMIC_v3.5_ID_GRCh37.txt", "Kucab spectra (83 channels)", True),
]

# Files the repo itself must carry (not part of the dataset bundle).
REPO_INPUTS: list[tuple[str, str, bool]] = [
    ("results/tables/quick_bio_v4_maf_features.csv.gz", "Bio MAF v4 features: survival baseline, feature selection", True),
    ("results/tables/main_manuscript_complete_panel_oof_predictions.csv", "canonical split manifest for every MuAt config", True),
    ("config/feature_resources/oncokb_cancer_genes.tsv", "Bio MAF v4 driver annotations", True),
    ("config/feature_resources/curated_pathways.yaml", "Bio MAF v4 pathway blocks", True),
]


def check_fasta(datasets_dir: Path) -> tuple[bool, str]:
    ref = datasets_dir / "references/grch37"
    if not ref.exists():
        return False, "references/grch37 does not exist"
    fastas = sorted(ref.rglob("*.fna")) + sorted(ref.rglob("*.fa")) + sorted(ref.rglob("*.fasta"))
    if not fastas:
        return False, "no *.fna under references/grch37 (fetch with --with-reference)"
    fasta = fastas[0]
    if not fasta.with_name(fasta.name + ".fai").exists():
        return False, f"{fasta.name} present but its .fai index is missing"
    return True, f"{fasta.name} ({fasta.stat().st_size / 1024**3:.1f} GB) + .fai"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-dir", default=None)
    args = parser.parse_args()

    datasets_dir = resolve_datasets_dir(args.datasets_dir)
    print("=" * 86)
    print("BUNDLE INPUT AUDIT")
    print("=" * 86)
    print(f"datasets dir : {datasets_dir}")
    print(f"repo root    : {REPO_ROOT}\n")

    missing_required: list[tuple[str, str]] = []
    missing_optional: list[tuple[str, str]] = []

    def report(group: str, items: list[tuple[str, str, bool]], base: Path) -> None:
        print(f"--- {group} " + "-" * (82 - len(group)))
        for rel, needed_by, required in items:
            path = base / rel
            if path.exists():
                size = path.stat().st_size
                unit = f"{size / 1024**2:.1f} MB" if size >= 1024**2 else f"{size / 1024:.0f} KB"
                print(f"  ok       {rel}  ({unit})")
            else:
                tag = "MISSING " if required else "absent  "
                print(f"  {tag} {rel}")
                print(f"           needed by: {needed_by}")
                (missing_required if required else missing_optional).append((rel, needed_by))
        print()

    report("dataset bundle", BUNDLE_INPUTS, datasets_dir)

    ok, detail = check_fasta(datasets_dir)
    print("--- reference FASTA " + "-" * 66)
    print(f"  {'ok      ' if ok else 'MISSING '} {detail}\n")
    if not ok:
        missing_required.append(("references/grch37/*.fna", "Kucab damage_class event features"))

    report("repository", REPO_INPUTS, REPO_ROOT)

    print("=" * 86)
    if missing_required:
        print(f"{len(missing_required)} REQUIRED input(s) missing:\n")
        blocked: dict[str, list[str]] = {}
        for rel, needed_by in missing_required:
            blocked.setdefault(needed_by, []).append(rel)
        for needed_by, paths in blocked.items():
            print(f"  {needed_by}")
            for rel in paths:
                print(f"      {rel}")
        print("\nTry:  python scripts/fetch_datasets.py --minimal --with-reference")
        print("If a file is still missing afterwards it is absent from the published bundle and")
        print("must be sourced from the original export.")
    else:
        print("All required inputs present. The full pipeline can run.")
    if missing_optional:
        print(f"\n{len(missing_optional)} optional input(s) absent (fine unless you need them):")
        for rel, needed_by in missing_optional:
            print(f"  {rel}  -> {needed_by}")
    print("=" * 86)
    return 1 if missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
