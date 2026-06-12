# Supplemental Figure Improvement Log

Generated: 2026-06-02

Scope: Supplementary Figures S1-S3 in `results/manuscript/supplement/`. All changes were implemented in the reproducible figure renderer and regenerated through `make_all_figures.py --strict`.

## Supplementary Figure S1: Representation Construction

1. Switched from hard-coded lower-card item positions to card-relative geometry; this fixes the prior overflow from the checkpointed-output card.
2. Increased card width and used consistent card dimensions; this improves alignment and prevents cramped interior text.
3. Reduced card corner radius from decorative rounded blocks to a more restrained figure style; this looks more manuscript-ready.
4. Added split provenance to the subtitle; this clarifies why the figure matters for leakage-safe reproducibility.
5. Added a small figure-purpose note below the subtitle; this helps readers understand the schematic without cluttering the cards.
6. Rewrote the Spectra card text to mention within-sample normalization; this makes the feature construction more precise.
7. Rewrote the FASTA card text to mention REF validation; this clarifies a key quality-control step.
8. Rewrote the KME card text to mention locked settings; this reduces ambiguity about exploratory vs final KME settings.
9. Rewrote the MAF-stack card text to include burden covariates; this more accurately describes the final feature set.
10. Rewrote the UGA-variants card to identify it as supplementary; this reduces confusion with required main outputs.
11. Rewrote the checkpointed-output card to emphasize traceable artifacts; this improves reproducibility framing.
12. Moved the KME point cloud upward; this fixes visual crowding against the caption.
13. Shortened the KME caption; this prevents text from competing with the illustration.
14. Tightened checkpoint file-list spacing; this keeps every artifact label inside the output card.
15. Reduced checkpoint file-row height; this eliminates bottom-edge crowding.
16. Added a footer callout about strict rerun cache invalidation; this explains how checkpointing avoids unnecessary reruns.
17. Added a footer note that artifact names are kept inside card bounds; this documents the layout constraint.
18. Aligned top-row arrows to card centers; this improves flow readability.
19. Aligned bottom-row arrows to card centers; this makes the workflow easier to follow.
20. Added a vertical connector from KME to checkpointed outputs; this clarifies that KME also flows into final artifacts.
21. Used row-aligned card placements; this removes the visually uneven schematic spacing.
22. Increased white space between rows; this reduces visual crowding.
23. Kept illustrations within each card; this avoids the previous box-overflow problem.
24. Preserved color semantics by feature family; this links the schematic to main figures.
25. Simplified captions to one technical claim per card; this improves scanability.
26. Removed long file labels from the card edge; this prevents Word/PDF clipping risk.
27. Added artifact examples with real output-like names; this makes the checkpoint concept concrete.
28. Kept the UGA line-plot graphic inside its card; this fixes possible visual spillover.
29. Kept MAF feature chips in a fixed grid; this avoids label drift across renderers.
30. Confirmed S1 passes SVG text-overlap and clipping QA; this verifies the fixes rather than relying on visual guesswork.

## Supplementary Figure S2: Calibration and Reliability

1. Fixed the Kucab calibration data bug by dropping all-empty probability columns before multiclass `argmax`; this removes the impossible all-zero observed accuracy line.
2. Treats missing probability values as invalid rather than as candidate classes; this prevents empty classes from winning prediction.
3. Computes multiclass confidence from finite probabilities only; this keeps confidence bins meaningful.
4. Compares predicted class indices against numeric true labels for Kucab; this matches how the OOF table encodes the endpoint.
5. Keeps binary HRD calibration as positive-class probability; this preserves the correct binary reliability interpretation.
6. Adds `calibration_y_label`; this lets each panel say whether y is accuracy or positive fraction.
7. Adds endpoint-level ECE; this gives a compact calibration error summary.
8. Adds total sample count per endpoint; this improves interpretability.
9. Adds class count for multiclass endpoints; this clarifies why Kucab and cancer type use confidence/correctness.
10. Replaced the 2-by-2 layout with a three-panel row; this removes the unused empty panel and makes the figure balanced.
11. Removed the legacy `os_event` panel; this avoids repeating the old binary-survival analysis.
12. Moved panel titles into consistent top lanes; this prevents title/subtitle overlap.
13. Replaced rotated y-axis labels with horizontal panel labels; this prevents tick-label collisions and clipped text.
14. Added a two-line explanatory note; this avoids a long subtitle running into the panels.
15. Uses circle size for bin sample count; this makes sparse bins visible.
16. Labels sparse Kucab bins; this warns that high-confidence Kucab bins have low n.
17. Adds a clear perfect-calibration legend; this makes the dashed diagonal self-explanatory.
18. Adds an observed-bin legend; this explains the blue points without relying on a caption.
19. Increases chart height; this makes calibration curvature easier to see.
20. Uses consistent 0.25 tick spacing; this improves cross-panel comparison.
21. Keeps x-axis labels identical across panels; this reinforces the confidence/probability interpretation.
22. Keeps y-axis range fixed at 0-1; this prevents misleading rescaling.
23. Uses tooltip metadata in SVG output; this preserves detailed bin values in the HTML asset.
24. Writes corrected plot data to CSV; this makes the fix reproducible and auditable.
25. Adds ECE and n to each panel header; this reduces the need for a separate calibration table.
26. Removes all empty probability classes from calibration only; this fixes plotting without changing benchmark scores.
27. Preserves canonical OOF source rows; this maintains comparability with the main benchmark.
28. Makes Kucab high-confidence sparsity visible instead of hiding it; this improves scientific honesty.
29. Regenerated PNG/PDF/SVG/HTML assets from the corrected CSV; this ensures all formats match.
30. Confirmed S2 passes SVG text-overlap and clipping QA; this verifies both data and layout fixes.

## Supplementary Figure S3: Supplementary Representation Checks

1. Replaced inline long point labels with a fixed detail column; this removes repeated text from the plotting field.
2. Added a legend for representation families; this makes color meaning explicit.
3. Increased figure width; this gives labels, plot, and detail column separate lanes.
4. Shifted the score axis right; this prevents group-header collisions.
5. Added panel boxes around each analysis family; this makes the three supplementary analyses visually distinct.
6. Added a top note explaining measured-only rows; this clarifies why missing combinations are not plotted.
7. Added a second note explaining the detail column; this helps readers parse the redesigned chart.
8. Uses alternating row bands; this improves readability across long endpoint lists.
9. Keeps endpoint labels in the left lane; this avoids overplotting text near score markers.
10. Shows score values next to dots only; this keeps numeric interpretation immediate without long labels.
11. Draws faint leader lines from zero to each dot; this improves visual comparison of score magnitude.
12. Keeps selected supplementary model details in a fixed right lane; this prevents dot-label collisions.
13. Wraps detail-column text to two lines maximum; this avoids row-height overflow.
14. Uses consistent x-axis ticks across panels; this supports cross-panel comparison.
15. Keeps unsupported combinations in Table S3 instead of plotting placeholders; this reduces visual clutter.
16. Adds a footer explaining S3's scoped role; this reduces redundancy with Figure 5.
17. Uses representation-family colors consistently with other figures; this improves visual continuity.
18. Removes repeated "model score - representation - model" labels; this makes the figure more professional.
19. Separates alternative geometry, COSMIC/NNLS, and mechanistic controls; this prevents unlike analyses from being overinterpreted as one ranking.
20. Preserves exact measured rows from the plot-data CSV; this avoids changing results while improving layout.
21. Uses compact point markers; this keeps dense sections readable.
22. Prevents long group title overlap with axis labels; this fixes the specific renderer QA issue.
23. Adds a source-like selected-row column; this makes each dot traceable.
24. Keeps main endpoint ranking out of S3; this reduces repetition with main Figures 3-5.
25. Keeps mechanistic controls visibly separate and compact; this avoids overstating their scope.
26. Uses a cleaner title/subtitle hierarchy; this improves manuscript polish.
27. Reduces empty white space compared with the old dot labels; this makes better use of the page.
28. Ensures all group panels use the same plotting grammar; this reduces reader relearning.
29. Regenerated PNG/PDF/SVG/HTML assets from the redesigned renderer; this keeps all export formats synchronized.
30. Confirmed S3 passes SVG text-overlap and clipping QA; this verifies the redesigned layout.

## Redundancy-Reduction Plan

1. Keep Figure 2 as the only main figure focused on burden versus signatures; do not repeat this contrast in supplementary figures.
2. Keep Figure 3 as the only main MuAt-compatible comparator figure; do not duplicate MuAt rows in supplementary calibration or supplementary checks.
3. Keep Figure 4 as the only MAF-stack incremental-benefit figure; avoid repeating the same deltas in S3.
4. Keep Figure 5 as the executive cross-endpoint summary; it should be the only all-main-endpoint heatmap.
5. Keep S1 as a methods/provenance schematic only; it should contain no performance numbers.
6. Keep S2 as calibration/reliability only for the canonical Signatures + MAF stack XGBoost rows; it should not become another performance-ranking figure.
7. Keep S3 as supplementary representation checks only; it should not repeat main endpoint rankings already in Figure 5.
8. Move unsupported or missing supplementary combinations to Table S3 instead of plotting empty cells.
9. Keep detailed machine-readable benchmark rows in technical CSVs, not in manuscript-facing tables.
10. Keep Table 2 as a compact top-line performance table, with full exhaustive rows retained only in technical tables.
11. Keep Table 3 as methods/provenance and dimensionality, not another score table.
12. Keep Table S1 as endpoint/class distribution only, not model performance.
13. Keep Table S2 as sensitivity analyses only, not main benchmark repetition.
14. Keep Table S3 as completeness/missingness documentation only, not result ranking.
15. Use captions to cross-reference where repeated information lives instead of redrawing it.
16. In the manuscript text, cite Figure 5 for broad ranking, Figure 3 for MuAt, Figure 4 for MAF-stack gain, and S2 for calibration.
17. Avoid placing identical endpoint-by-representation score grids in both a main figure and supplement.
18. For publication, make all exhaustive raw-result tables available as supplemental CSV/HTML files rather than page-width Word tables.
19. Use one "best model per endpoint" text summary in Results rather than repeating every cell value in prose.
20. Keep statistical pairwise-test details in plot data and technical tables unless they are needed to interpret a visible claim.
