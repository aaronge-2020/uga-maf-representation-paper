# Brief — Figure 1, study design schematic (UGA/MAF representation benchmark)

Mode: design. Route: flat. Depth: full. Format: journal, 170 mm wide.

## What the figure must say (one idea)
One tumour's mutation list is encoded four ways, and every encoding is scored on
the same five held-out test groups, so the representation is the only thing that
varies. A MuAt-compatible rebuild, which reads mutation event bags directly,
joins the same comparison from outside the tabular path.

A reader who gives it ten seconds should be able to say: the same mutations go
in, four feature tables come out, two model types are fitted on training folds
only, and everything is judged on identical test groups and four endpoints.

## Frozen content (from the manuscript caption; nothing may be added or dropped)
- Input: one tumour's mutation list, MAF rows with locus, alleles, gene,
  consequence, VAF.
- Four feature tables: Burden (3 features), Spectra SBS96+ID83 (182),
  Bio MAF (948), Signatures + Bio MAF (1,130).
- Paired elastic net and XGBoost models, tuned inside training folds only,
  scored on five held-out test groups.
- Endpoints: cancer type, 20 types (n=8,800); HRD score and HRD high/low at
  24, 33, 42 (n=772); overall survival (n=9,986).
- MuAt-compatible rebuild consumes mutation event bags directly, benchmarked
  on the same five groups, outside the tabular path.

## Must not appear
- Any other representation (no KME, no channel view), no "v4" naming.
- No invented patient data: the MAF table shows column structure and elided
  rows, never fabricated gene names, alleles, or VAF values.
- No performance numbers (results live in Figures 2-8), no logos, no photos.
- No legend: every colour is labelled where it sits, and the same colour key
  as Figures 2-8 is used (burden grey, spectra blue, Bio MAF teal,
  Sig+MAF dark blue, MuAt orange).
