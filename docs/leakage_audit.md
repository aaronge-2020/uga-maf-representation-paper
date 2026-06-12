# Leakage Audit

This audit summarizes the leakage controls in the active manuscript export.

## Controls

- Nested cross-validation is retained for active model evaluation.
- Bio MAF v4 feature selection is performed inside inner-loop tuning only.
- Held-out fold labels and held-out fold performance are not used for feature-definition decisions.
- Checkpoints are accepted only when the recorded data, label, sample, feature, encoded-event, dictionary, mask, and configuration fingerprints match the current run.
- COSMIC/signature resources are modeled as their own feature family and are not merged into Bio MAF resources.
- Retired hash-derived MAF workflows are not active in the release export.
- Stale feature-ranking claims are excluded from public docs and active outputs.

## Re-Curation Boundary

The release export is designed for manuscript reproduction, not upstream dataset re-curation. If future re-curation is needed, use a separate curation workspace and then import the resulting approved resources into `github_exports_datasets` with updated manifests and checksums.

## Required Validation

```bash
python scripts/reproduce_manuscript.py --datasets-dir ../github_exports_datasets --strict
```

The strict validator must pass with zero reads outside `github_exports` after restoration from the sibling dataset bundle.
