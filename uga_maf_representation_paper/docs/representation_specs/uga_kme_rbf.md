# UGA RBF Kernel Mean Embedding

## Inputs

- Locked UGA event or channel coordinates.
- Patient-level mutation weights from the same retained SBS96/ID83 inputs.
- Fixed or tuned RBF kernel parameters from the experiment config.

## Construction

1. Build locked UGA coordinates for mutation channels.
2. Select landmark coordinates from the locked channel/event set.
3. Compute RBF similarities from each UGA coordinate to the landmarks.
4. Average those similarities with patient mutation counts as weights.
5. Evaluate untuned KME, HRD33-tuned KME, and Standard+KME subset variants.

## Dimensionality

Dimensionality equals the selected number of RBF landmarks. Tuned runs record requested and actual landmark counts in `kme_grid_selected_params.csv` and related diagnostics.

## Preserved, Added, Lost

- Preserved: patient-level mutation mass distribution over the locked UGA coordinate system.
- Added: nonlinear similarity features around UGA landmarks.
- Lost or smoothed: exact channel counts are replaced by kernel-smoothed landmark responses.

## Leakage Controls

The tuned KME profile is selected on the HRD33 endpoint and then frozen for full-panel evaluation. Full-panel folds are paired with Standard features and reuse identical patients, labels, seeds, and learner settings.
