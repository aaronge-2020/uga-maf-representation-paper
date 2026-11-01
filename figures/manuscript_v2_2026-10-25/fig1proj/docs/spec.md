# Spec — Figure 1

## Structure first
Reading order follows the analysis itself (P8): mutations in, encodings made,
models fitted, endpoints scored. Two things must be visible at a glance:
(1) the fork from one input into four feature tables; (2) the shared measuring
stick, the same five test groups, touched by both the tabular path and the
MuAt lane. The comparison's fairness is the figure's claim, so the test-group
strip is a drawn object, not a sentence.

## Concepts considered
**A. Horizontal pipeline, MuAt as a dashed lane below.** Input table at left,
four feature-table glyphs stacked centre-left, models frame centre-right,
endpoints at right; a test-group strip runs under models and endpoints; the
MuAt lane sits in a dashed frame along the bottom and rises to the same strip.
Scores: reading direction 5 (causality left to right), comparison clarity 5
(one strip, both paths touch it), label fit 4 (endpoints column is narrow but
entries wrap to two short lines), density 4. Total 18.
**B. Vertical pipeline, MuAt lane at right.** Input on top, four glyphs in a
row, models below, endpoints at bottom; MuAt column at the right edge.
Scores: reading direction 3 (top to bottom fights the paper's other figures),
comparison clarity 3 (shared splits become a vertical bracket, easy to miss),
label fit 5 (glyphs get full width each), density 4. Total 15.
**C. Hub and spokes (input centre, four tables around it).** Rejected before
drafting: reading order is ambiguous and the shared-test-group claim has no
single object to attach to (P8, P10).

Kept for drafting: A and B. A leads on the claim; B is the control that proves
the horizontal choice is doing work.

## Spec card
**Takeaway (one sentence):** One mutation list becomes four feature tables, and
every table, plus a MuAt-compatible rebuild that reads event bags directly, is
scored on the same five held-out test groups.
**Size:** 170 mm × 96 mm
**Hero:** none. The subject is an analysis pipeline, an architecture-type
subject where boxes and tables are the right register (P1 exception); no
physical object exists to draw. The fork into four glyphs carries the eye.

## Panels
| Panel | Job (one line) | Position / share of the figure |
|---|---|---|
| a | The whole design: input, encodings, models, endpoints, MuAt lane | full figure |

## Components
| Panel | Component | Label (exact text) | Look (type, colour role, 2D/3D) | Source |
|---|---|---|---|---|
| a | MAF table | "One tumour" heading; "Mutation list (MAF)"; columns "Locus", "Alleles", "Gene", "Consequence", "VAF"; cells "…" | ruled mini-table, flat, ink on white | draw (structure only, no values) |
| a | Feature glyph 1 | "Burden", "3 features" | small-multiple table glyph, burden grey #767676 cells | draw |
| a | Feature glyph 2 | "Spectra (SBS96 + ID83)", "182 features" | same glyph template, spectra blue #3775BA | draw |
| a | Feature glyph 3 | "Bio MAF", "948 features" | same glyph template, Bio MAF teal #42949E | draw |
| a | Feature glyph 4 | "Signatures + Bio MAF", "1,130 features" | same glyph template, mixed blue and teal cells (it is the combination) | draw |
| a | Models frame | "Elastic net", "XGBoost", "Tuned inside training folds only" | dashed frame, two pills, ink | draw |
| a | Test-group strip | "Same five held-out test groups" + five squares | thin strip with 5 square marks | draw (count 5 from caption) |
| a | Endpoints box | "Endpoints"; "Cancer type (20 types)", "n=8,800"; "HRD score", "n=772"; "HRD high/low at 24, 33, 42", "n=772"; "Overall survival", "n=9,986" | list box, ink, endpoint names bold | draw |
| a | MuAt lane | "Mutation event bags", "MuAt-compatible rebuild", "Same five test groups" | dashed frame, bag glyph (loose dots), MuAt orange #E28E2C | draw |

## Relations
| From | To | Kind | Drawn as |
|---|---|---|---|
| MAF table | each feature glyph | flow (encodes as) | fork, four arrows |
| each feature glyph | models frame | flow (fitted by) | four wires joining to one arrow |
| models frame | endpoints box | flow (scored on) | arrow crossing the test-group strip |
| event bags | MuAt rebuild | flow (consumes) | arrow inside the lane |
| MuAt rebuild | test-group strip | benchmarked on | rising arrow to the same strip |

## Conventions
- Colour key identical to Figures 2-8: burden grey #767676, spectra #3775BA,
  Bio MAF #42949E, Sig+MAF #0F4D92, MuAt #E28E2C, ink #272727.
- Labels sit next to what they name; no legend.
- Feature counts print next to their glyph, never inside a bar or cell.
- The four glyphs share one template (P3); cell density rises with feature
  count but cell count is schematic, not proportional (the counts are printed).

## Negative constraints (must not appear)
- No KME or fifth representation; no "v4"; no results numbers; no fabricated
  gene, allele, or VAF values; no legend; no photographs or logos.
