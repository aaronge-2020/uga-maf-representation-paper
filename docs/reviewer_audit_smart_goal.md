# SMART Goal: Reviewer-Ready Manuscript Reproducibility Export

## Specific

Make `github_exports` understandable and reproducible for a new manuscript reviewer who has only:

- the GitHub code checkout at `github_exports/`;
- the Hugging Face-style dataset bundle at `github_exports_datasets/`;
- Python, pip, and Node.js.

The reviewer must be able to discover where TCGA/MC3, TCGA-BRCA HRD, KUCAB/Mendeley, PCAWG/PanCancer, GRCh37, Bio MAF v4, COSMIC/signature resources, caches, and regenerated manuscript outputs live without reading internal manifests first.

## Measurable

Completion requires all of the following evidence from inside `github_exports`:

- `python scripts/reviewer_workflow.py start --datasets-dir ../github_exports_datasets` reports `Status: READY`.
- The setup output lists the key dataset locations in `../github_exports_datasets`.
- `python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets --dry-run` completes.
- `results/logs/strict_validation_latest.json` reports `status: passed` and `outside_root_reads: 0`.
- `results/logs/dry_run_all_experiments_manifest.json` is generated.
- Active Python files under `scripts/` and `src/` compile with zero syntax failures.
- No active code depends on the old external `cgr_validation` project folder or any other local folder outside the export.
- Public docs name `cancer_type_top20`, include HRD binary endpoints, and avoid retired smaller-cancer-type or retired hashed-feature wording except validator guard patterns.
- Bio MAF v4 source tables/resources remain present for regenerating S4/S5/S6 outputs, and COSMIC/signature resources remain separate from Bio MAF v4 resources.

## Achievable

The export keeps large assets in `github_exports_datasets` and validates/reads required runtime files from that bundle automatically. Production documentation should point reviewers to the readable dataset-bundle layout rather than to old local project paths.

## Relevant

This directly supports the publication handoff: scripts go to GitHub, large assets go to Hugging Face or similar storage, and collaborators can reproduce or validate manuscript findings without hunting through old project folders.

## Time-Bound

Before public upload, rerun the setup check, dry-run reproduction, strict validation, large-file scan, and `git status --short`, then update `docs/final_qc_report.md` with the command results and any remaining limitations.
