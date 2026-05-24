# SA-UGA / Observed Atlas Variants

## Inputs

- Observed event contexts or genome-atlas contexts.
- UGA model specs in `uga_atlas/models.py`.
- Optional observed-context atlas when running observed-atlas variants.

## Construction

1. Encode each event with the selected UGA context and payload schema.
2. Use observed contexts where the model spec requests `observed_context`; otherwise use the genome-wide atlas.
3. Aggregate encoded events to patient-level means, blocks, or downstream candidate features depending on the experiment.

## Dimensionality

Dimensionality follows the selected UGA model spec. For d10/dp10 masked observed-context variants, the vector has 100 dimensions: 40 context coordinates plus two 30D payload blocks.

## Preserved, Added, Lost

- Preserved: event-level allele payload encoding according to the selected schema.
- Added: observed-context geometry when available.
- Lost or smoothed: patient-level aggregation smooths individual events unless a downstream representation keeps distributional summaries.

## Leakage Controls

Atlas construction and event encoding are independent of endpoint labels. Observed-atlas variants are treated as sensitivity analyses unless explicitly promoted by a locked protocol.
