# B2 decision brief, 2026-11-01

## The hole

The paper's core premise, as Aaron stated it on 2026-11-01, is a decomposition: previous authors never measured how much each component of their pipeline contributed to prediction, and this paper does.

The evidence for that decomposition currently comes from two different protocols:

- The 49-cell grid decomposes whole encodings under the fair protocol (identical patients, folds, learners; each encoding retuned). It prices gene identity, spectra, consequence counts, TF-IDF, and the full Bio MAF stack against each other.
- The component removals inside Bio MAF (gene features out: cancer 0.662 to 0.540; curated knowledge out: 0.662 to 0.659; VAF summaries out: HRD33 0.900 to 0.874; DNA repair genes out: HRD unchanged) ran at archived settings under the main-panel protocol, not retuned under the fair protocol.

B2 is the one frozen workstream that closes that gap: retuned additions to spectra plus R3 counts, and retuned removals from Bio MAF, using the Bio MAF matrix at results/tables/quick_bio_maf_features.csv.gz (status table, workstream B2). Everything else frozen (B3, C1 to C3, D2, D3, F1, F3) is not needed for the premise.

## Option 1: scope the claim, run nothing (fits the analysis freeze)

Rewrite the premise sentence so it claims only what the two existing protocols support:

- "Decomposition" refers to whole encodings under the grid protocol, plus component removals at archived settings in the main panel, with the protocol difference stated once where the removal numbers appear.
- Add the copy-number scope sentence Aaron already approved in substance on 2026-11-01: the benchmark is built from mutation calls only, so CPEM's copy-number block (13,040 of its 35,565 features, credited by CPEM's own stepwise table with +2.5 points) was not tested.

Cost: editing only. Risk: a reviewer can still ask for retuned removals; the paper answers with the archived-setting removals and the stated limitation.

## Option 2: run a minimal B2 (one exception to the freeze)

Arms, from the protocol's four Bio MAF blocks (gene event, consequence/functional, VAF, curated):

- Removals from the full Bio MAF stack: minus gene event, minus consequence/functional, minus VAF, minus curated. 4 arms.
- Additions to spectra plus R3 counts: plus consequence/functional, plus VAF, plus curated. 3 arms. (Gene event is already in R3 counts, so it is not an addition arm.)

Scope choices:

- Cancer and HRD, XGBoost only (the learner where the annotation claim lives): 7 arms x 2 tasks = 14 cells.
- Cancer, XGBoost only (narrowest version that prices the annotation layer where it pays): 7 cells.
- Adding elastic net doubles either count and mainly re-tests a learner where TF-IDF already won on cancer and everything was weak on HRD.

Compute, honestly: each cell is 10 tuning trials per fold across 5 folds under protocol v1.0. The 49-cell grid took about a week of interrupted wall time on this VM; Aaron's laptop finished its final 9 to 12 cells in well under an hour on 2026-11-01. So 14 cells is days on the VM and plausibly hours on the laptop. It is not a handful of quick runs on this machine, whatever was said in chat on 2026-11-01.

## Recommendation

Option 1 now, Option 2 only if Vijay or Cindy, or a reviewer, presses the decomposition claim. The reproducibility package (G1 ledger, G2 claim audit, G3 FAIR) already underway does not depend on B2 either way, and B2's result would not change the headline finding: gene identity and mutation patterns carry the signal; curated annotation pays only for cancer typing with boosted trees.
