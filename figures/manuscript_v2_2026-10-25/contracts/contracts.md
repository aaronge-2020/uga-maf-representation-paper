# Figure contracts — UGA/MAF representation benchmark (Oxford Bioinformatics)

Shared system (all figures): width 170 mm; Arial/DejaVu Sans; base text 7.5–8 pt,
5 pt glyph floor; bold lowercase panel labels; white background; top/right spines off.
Colour key (one role per hue, identical in every figure incl. Figure 1):
Burden = neutral grey #767676 (baseline), Spectra = #3775BA, Bio MAF = #42949E (teal),
Signatures + Bio MAF = #0F4D92 (hero dark blue), MuAt-compatible = #E28E2C (warm accent),
Published MuAt = light grey, hatched (external reference, not ours).
Green #2E9E44 / red #B64342 only for directional difference cues.
All values frozen (verified in the prior audit round); craft only, no number changes.
Export: 300-dpi PNG + vector PDF (editable text, pdf.fonttype 42) at final size.

## Figure 1 — schematic-led (built with the schematic-figure skill, separate project)
Claim: one mutation list is encoded four ways and every encoding is scored on the
same five held-out groups, so representation is the only thing that varies; the
MuAt-compatible rebuild joins the same test groups from outside the tabular path.

## Figure 2 — quantitative grid, 1 × 3 grouped bars (a Accuracy, b Balanced accuracy, c Top-5)
Claim: on cancer typing (XGBoost, n = 8,800), adding Bio MAF features to spectra
(Sig+MAF) beats spectra alone and the faithful MuAt-compatible rebuild on all three
metrics; published MuAt is shown hatched as external context only.
Evidence chain: three panels are the same comparison under three metrics of rising
leniency; hero series = Sig+MAF (dark blue). Whiskers: per-fold 95% t intervals.
Risk: published MuAt (different cohort/framework) must not read as a like-for-like bar —
hatching + footnote carry that.

## Figure 3 — single-panel paired horizontal bars, sorted by gain
Claim: Sig+MAF improves per-type recall over spectra alone for every one of the
20 cancer types (gain +0.60 ESCA to +0.00 SKCM); no type is harmed.
Evidence: paired bars per type (same patients, two representations), gain printed
at the end of each pair; per-type n in the row label. No uncertainty shown in the
source spec (pooled out-of-fold recalls).

## Figure 4 — quantitative grid, 2 learners × (bars + differences)
Claim: spectra beat burden alone on nearly every endpoint under both learners
(counts are the weak baseline; patterns carry the signal).
Rows: elastic net / XGBoost. Left: pooled out-of-fold bars (burden vs spectra) for
Kucab damage class, HRD score, HRD24/33/42, cancer type. Right: descriptive paired
differences (spectra − burden) as dots, green positive / red negative.
Subtitle (frozen): bars are pooled out-of-fold; differences descriptive, no CIs/tests.

## Figure 5 — quantitative grid, 2 learners × (bars + differences), 3 series
Claim: gene features add information patterns lack: Bio MAF alone and Sig+MAF sit
above spectra on most endpoints, and the combination is the strongest tabular arm.
Same grid as Figure 4 for direct comparability (controlled-comparison template).
Differences: Bio MAF − spectra, Sig+MAF − spectra, Sig+MAF − Bio MAF.
Kucab rows use event tokens (2,980) / combination (3,241) — disclosed in the legend.

## Figure 6 — quantitative grid, 1 × 2 horizontal SHAP-share bars
Claim: the cancer-type model's attribution is spread across annotation blocks
(consequence 23.8% largest, no single dominant block), and the overall-survival
model leans on a different mix (VEP other 20.6%).
Panels: a cancer type, b overall survival. Single blue family; shares printed.

## Figure 7 — quantitative grid, 2 endpoints × (bars + forest difference)
Claim: on HRD the faithful MuAt-compatible rebuild trails the best tabular model
by a wide, significant margin (Spearman −0.270, AUROC −0.165; CIs exclude 0).
Left: fold-mean bars with 95% t intervals (Best tabular, Sig+MAF default,
MuAt-compatible). Right: MuAt − best tabular differences with 95% CIs (paired file).

## Figure 8 — asymmetric 1 × 2 (bars + delta forest)
Claim: for overall survival, Bio MAF alone (0.638) cannot replace cancer type
(0.742), and adding Bio MAF to cancer type adds nothing (0.743; Δ +0.001, CI spans 0).
Left: pooled out-of-fold Harrell C-index bars (axis from 0.50, stated on the axis).
Right: paired bootstrap Δ C-index vs cancer type only, with 95% CIs.
