# UGA Signal-Recovery Variants

## Inputs

- Standard SBS96/ID83 MC3 features.
- Locked UGA projections from the same channel counts.
- Candidate UGA add-back and distributional variants from the supporting 2026-05-15 signal-recovery experiment.

## Construction

1. Build baseline Standard and locked UGA feature families.
2. Screen candidate UGA variants for missing-signal recovery under paired folds.
3. Run a third-pass linear-learner screen for finalist variants.
4. Summarize candidate leaderboard and focused confirmation metrics.

## Dimensionality

Dimensionality is candidate-specific and recorded in regenerated manifests and diagnostics. Candidate families include locked mean projections, add-back features, and distributional summaries.

## Preserved, Added, Lost

- Preserved: original Standard features for paired comparisons and delta calculations.
- Added: UGA-derived recovery features intended to diagnose information loss from mean projection.
- Lost or smoothed: depends on candidate; mean-style variants smooth channel identity, while add-back variants restore selected signals.

## Leakage Controls

Screens use paired folds and promotion gates. Candidate selection is separated from focused confirmation, and endpoint labels are not used to construct UGA coordinates.
