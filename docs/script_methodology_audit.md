# Script Methodology Audit

This audit covers the active scripts in `github_exports` after cleanup. Source snapshots that were only retained for provenance have been removed from the GitHub-ready export.

## Active Surface

- `scripts/reproduce_manuscript.py`
- `scripts/reviewer_workflow.py`
- `scripts/validate_static_bundle.py`
- `scripts/check_restricted_data_privacy.py`
- Active runners under `src/runners/`
- Active utilities under `src/utils/`
- Active manuscript rendering code under `src/visualization/`

## Findings Addressed

- Removed active dependence on the historical validation project folder and external project folders.
- Removed active retired hash-derived MAF and stale feature-ranking paths from the release workflow.
- Removed obsolete provenance-only scripts and scratch reports from the GitHub-ready export.
- Ensured large/downloaded assets are validated and read from `../github_exports_datasets` instead of hidden local paths.
- Restricted Bio MAF v4 resource loading to bundled/local resources; no network download is attempted by active code.
- Strengthened nested OOF split handling, grouped fallback behavior, and OOF assignment checks.
- Strengthened checkpoint provenance with label, feature, sample, encoded-event, dictionary, mask, and configuration fingerprints.
- Removed implicit compatible-checkpoint reuse from active configs.
- Confirmed manuscript rendering uses the active endpoint set, including `cancer_type_top20`, PFI, and HRD binary endpoints.

## Methodology Guards

- Bio MAF v4 feature selection occurs only inside inner-loop tuning.
- No endpoint labels or held-out fold performance are used to define Bio MAF features.
- COSMIC/signature features remain separate from Bio MAF resources.
- Nested cross-validation remains intact for active model comparison workflows.
- Binary and multiclass endpoints use task-appropriate metrics and preserved out-of-fold predictions.
- Active configs avoid stale feature-ranking claims and old Bio MAF/hash-derived MAF wording.

## Validation

Run from `github_exports`:

```bash
python scripts/reproduce_manuscript.py --datasets-dir ../github_exports_datasets --strict
python scripts/reviewer_workflow.py reproduce --datasets-dir ../github_exports_datasets --dry-run
```

The strict command validates active manuscript artifacts, manifests, dataset-backed large assets, and read boundaries. The dry run confirms a collaborator can follow the published workflow without launching expensive model jobs.
