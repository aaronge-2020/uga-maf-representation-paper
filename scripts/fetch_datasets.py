#!/usr/bin/env python3
"""Fetch the dataset bundle from Hugging Face instead of the sibling-folder convention.

The manuscript pipeline expects a ``../github_exports_datasets`` bundle that was previously
distributed by hand through Google Drive. The same bundle is published as a Hugging Face
dataset, so reproduction no longer depends on anyone re-sharing a Drive folder.

    python scripts/fetch_datasets.py                     # bundle only (no 3.2 GB reference)
    python scripts/fetch_datasets.py --with-reference    # + GRCh37 FASTA
    python scripts/fetch_datasets.py --verify-only       # check an existing bundle

Two artefacts need special handling, because neither lives where the code expects it:

1. ``main_manuscript_complete_panel_oof_predictions.csv`` is the canonical split manifest. Every
   MuAt config sets ``require_canonical_split: true`` and points at
   ``results/tables/main_manuscript_complete_panel_oof_predictions.csv`` inside the *repository*.
   That file is listed in ``.gitignore``, so it is absent from any clone. In the bundle it lives
   under ``results/large_tables/``. Without this step the MuAt comparator cannot run at all.

2. The GRCh37 FASTA is published at the dataset root as
   ``GCF_000001405.13_GRCh37_genomic-002.fna`` -- the ``-002`` is a Google Drive duplicate-name
   artefact. The pipeline globs ``*.fna`` under ``references/grch37``, so the file has to be
   placed there under its canonical name, with its ``.fai`` index alongside it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPO_ID = "vjdeara/cgr_tensorlab_umsom"
BUNDLE_PREFIX = "github_exports_datasets"

# Published at the dataset root rather than inside the bundle.
REMOTE_FASTA = "GCF_000001405.13_GRCh37_genomic-002.fna"
CANONICAL_FASTA_NAME = "GCF_000001405.13_GRCh37_genomic.fna"
FASTA_DEST_SUBDIR = Path("references/grch37/GCF_000001405.13")

# Bundle-relative source -> repository-relative destination.
REPO_LOCAL_ARTEFACTS: dict[str, str] = {
    "results/large_tables/main_manuscript_complete_panel_oof_predictions.csv":
        "results/tables/main_manuscript_complete_panel_oof_predictions.csv",
}

# Precomputed MuAt event bags. Optional, but they skip the expensive MAF -> event-table build.
MUAT_EVENT_BAGS: dict[str, str] = {
    "results/large_tables/muat_style_tcga_comparator_cancer_type_top20_comparable_muat_compatible_events.tsv.gz":
        "results/tables/muat_style_tcga_comparator_cancer_type_top20_comparable_muat_compatible_events.tsv.gz",
    "results/large_tables/muat_style_tcga_comparator_main_endpoints_comparable_muat_compatible_events.tsv.gz":
        "results/tables/muat_style_tcga_comparator_main_endpoints_comparable_muat_compatible_events.tsv.gz",
}

REQUIRED_BUNDLE_PATHS = [
    "datasets/tcga_mc3",
    "datasets/tcga_brca_hrd",
    "datasets/kucab_mendeley",
    "datasets/pcawg_pancan",
    "references/grch37",
    "resources",
]

MINIMAL_REQUIRED_PATHS = [
    "datasets/tcga_mc3",
    "datasets/tcga_brca_hrd",
]


def _human(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:,.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def _require_hub():
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
    except ImportError:
        sys.exit(
            "huggingface_hub is not installed.\n"
            "    pip install 'huggingface_hub[hf_transfer]'\n"
            "Set HF_HUB_ENABLE_HF_TRANSFER=1 for substantially faster downloads."
        )
    from huggingface_hub import snapshot_download

    return snapshot_download


# Everything the MuAt comparator actually touches. The comparator reads MC3 and the HRD assets
# only -- it builds mutation motifs from the MC3 ``CONTEXT`` column, so it needs no reference
# FASTA -- plus the canonical split manifest. This skips `caches/`, `optional_archived/` and
# `benchmarks/`, which are large and irrelevant to a MuAt rerun.
MINIMAL_PATTERNS = [
    f"{BUNDLE_PREFIX}/datasets/tcga_mc3/**",
    f"{BUNDLE_PREFIX}/datasets/tcga_brca_hrd/**",
    f"{BUNDLE_PREFIX}/references/grch37/**",
    f"{BUNDLE_PREFIX}/results/large_tables/main_manuscript_complete_panel_oof_predictions.csv",
    f"{BUNDLE_PREFIX}/*.csv",
    f"{BUNDLE_PREFIX}/*.json",
    f"{BUNDLE_PREFIX}/README.md",
]


def download(args: argparse.Namespace) -> Path:
    snapshot_download = _require_hub()

    if args.minimal:
        allow = list(MINIMAL_PATTERNS)
        if args.with_event_bags:
            allow += [f"{BUNDLE_PREFIX}/results/large_tables/*muat_compatible_events.tsv.gz"]
    else:
        allow = [f"{BUNDLE_PREFIX}/**"]
    if args.with_reference:
        allow.append(REMOTE_FASTA)

    print(f"[fetch] repo      : {args.repo_id} (revision={args.revision or 'main'})")
    print(f"[fetch] target    : {args.download_root}")
    print(f"[fetch] patterns  : {allow}")
    if not args.with_reference:
        print("[fetch] note      : skipping the 3.2 GB GRCh37 FASTA (pass --with-reference to include it)")

    local = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(args.download_root),
        allow_patterns=allow,
        token=args.token,
        max_workers=args.workers,
    )
    return Path(local)


def place_reference(download_root: Path, datasets_dir: Path, *, move: bool) -> None:
    source = download_root / REMOTE_FASTA
    if not source.exists():
        print(f"[fasta ] not downloaded ({REMOTE_FASTA}); skipping placement")
        return

    dest_dir = datasets_dir / FASTA_DEST_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / CANONICAL_FASTA_NAME

    if dest.exists() and dest.stat().st_size == source.stat().st_size:
        print(f"[fasta ] already in place: {dest}")
    else:
        print(f"[fasta ] {'moving' if move else 'copying'} -> {dest}  ({_human(source.stat().st_size)})")
        if move:
            shutil.move(str(source), str(dest))
        else:
            shutil.copy2(source, dest)

    # The pipeline looks for "<fasta>.fai" beside the FASTA. The repository tracks a .fai under
    # data/GRCH37/, so reuse it rather than requiring samtools.
    fai_dest = dest.with_name(dest.name + ".fai")
    if not fai_dest.exists():
        tracked_fai = REPO_ROOT / "data/GRCH37/GCF_000001405.13/GCF_000001405.13_GRCh37_genomic.fna.fai"
        if tracked_fai.exists():
            shutil.copy2(tracked_fai, fai_dest)
            print(f"[fasta ] copied tracked index -> {fai_dest.name}")
        else:
            print("[fasta ] WARNING: no .fai index found. Build one with: samtools faidx <fasta>")


def place_repo_artefacts(datasets_dir: Path, *, include_event_bags: bool) -> list[str]:
    problems: list[str] = []
    wanted = dict(REPO_LOCAL_ARTEFACTS)
    if include_event_bags:
        wanted.update(MUAT_EVENT_BAGS)

    for rel_source, rel_dest in wanted.items():
        source = datasets_dir / rel_source
        dest = REPO_ROOT / rel_dest
        if not source.exists():
            problems.append(f"missing in bundle: {rel_source}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == source.stat().st_size:
            print(f"[place ] up to date : {rel_dest}")
            continue
        shutil.copy2(source, dest)
        print(f"[place ] {rel_dest}  ({_human(source.stat().st_size)})")
    return problems


def verify(datasets_dir: Path, *, minimal: bool = False) -> int:
    print(f"\n[verify] bundle root: {datasets_dir}")
    problems: list[str] = []

    if not datasets_dir.exists():
        print(f"[verify] FAIL  bundle directory does not exist")
        return 1

    required = MINIMAL_REQUIRED_PATHS if minimal else REQUIRED_BUNDLE_PATHS
    for rel in required:
        path = datasets_dir / rel
        status = "ok  " if path.exists() else "MISS"
        if not path.exists():
            problems.append(f"missing bundle path: {rel}")
        print(f"[verify] {status}  {rel}")

    fasta = sorted((datasets_dir / "references/grch37").rglob("*.fna")) if (datasets_dir / "references/grch37").exists() else []
    if fasta:
        print(f"[verify] ok    reference FASTA: {fasta[0].name} ({_human(fasta[0].stat().st_size)})")
        if not fasta[0].with_name(fasta[0].name + ".fai").exists():
            problems.append("reference FASTA has no .fai index")
    else:
        print("[verify] WARN  no *.fna under references/grch37 (rerun with --with-reference if a runner needs it)")

    canonical = REPO_ROOT / REPO_LOCAL_ARTEFACTS["results/large_tables/main_manuscript_complete_panel_oof_predictions.csv"]
    if canonical.exists():
        print(f"[verify] ok    canonical split manifest ({_human(canonical.stat().st_size)})")
    else:
        problems.append(
            "results/tables/main_manuscript_complete_panel_oof_predictions.csv is absent. "
            "Every MuAt config sets require_canonical_split: true and will fail without it."
        )

    if problems:
        print("\n[verify] PROBLEMS:")
        for item in problems:
            print(f"   - {item}")
        return 1

    print("\n[verify] bundle looks complete.")
    print("[verify] point the runners at it (src/utils/config.py reads CGR_DATASETS_DIR):")
    print(f"[verify]   PowerShell :  $env:CGR_DATASETS_DIR = '{datasets_dir}'")
    print(f"[verify]   bash       :  export CGR_DATASETS_DIR='{datasets_dir}'")
    print(f"[verify] scripts/reproduce_manuscript.py instead takes:  --datasets-dir {datasets_dir}")
    print("[verify] scripts/run_muat_ablation.py finds this location automatically.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=None, help="Branch, tag, or commit sha (default: main).")
    parser.add_argument(
        "--download-root",
        type=Path,
        default=REPO_ROOT.parent / "hf_bundle",
        help="Where the Hugging Face snapshot is materialised (default: ../hf_bundle).",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Only fetch what the MuAt comparator needs (MC3, HRD assets, canonical split manifest). "
             "Skips caches/, optional_archived/ and benchmarks/.",
    )
    parser.add_argument("--with-reference", action="store_true", help="Also download the 3.2 GB GRCh37 FASTA.")
    parser.add_argument("--with-event-bags", action="store_true", help="Also stage the precomputed MuAt event tables.")
    parser.add_argument("--move-reference", action="store_true", help="Move rather than copy the FASTA (saves 3.2 GB).")
    parser.add_argument("--verify-only", action="store_true", help="Do not download; just check an existing bundle.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--token", default=None, help="HF token (only needed if the dataset is private).")
    args = parser.parse_args()

    args.download_root = args.download_root.expanduser().resolve()
    datasets_dir = args.download_root / BUNDLE_PREFIX

    if args.verify_only:
        return verify(datasets_dir, minimal=args.minimal)

    download(args)
    if args.with_reference:
        place_reference(args.download_root, datasets_dir, move=args.move_reference)
    problems = place_repo_artefacts(datasets_dir, include_event_bags=args.with_event_bags)
    for item in problems:
        print(f"[place ] WARNING: {item}")

    return verify(datasets_dir, minimal=args.minimal)


if __name__ == "__main__":
    raise SystemExit(main())
