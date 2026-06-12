# Figure Readiness Report

Generated after the strict no-leakage benchmark and count-aware MuAt-compatible rerun. Each item below is implemented or verified in the regenerated manuscript figures under `results/manuscript/figures`.

## Figure 1. Mutation Catalogues to Benchmarked Representations

1. Clarify the main visual thesis: the title now states that mutation catalogues become benchmarked representations, because readers need to know this is an evaluation workflow, not only feature engineering.
2. Separate the primary tabular path from the direct event-set path with solid versus dashed arrows, because MuAt-compatible is not a tabular feature family.
3. Update the subtitle to say MuAt-compatible is a separate measured comparator, because the earlier "conceptual alternative" wording was outdated.
4. Use panel letters A-D, because multi-step workflow figures are easier to cite in the text.
5. Keep a concrete MAF/VCF event-row mockup, because abstract feature families are otherwise hard to connect to raw input data.
6. Show retained event fields as chips, because locus, allele, gene, VAF, effect, and FASTA context are key signals in the benchmark.
7. Use distinct colors for signatures, KME, MAF stack, and combined stack, because representation families need to be visually separable throughout the manuscript.
8. Add miniature spectra bars in the signature card, because signatures are histograms and should look like channel summaries.
9. Add FASTA-base boxes and points in the KME card, because the reader needs to see sequence-context geometry rather than another histogram.
10. Add gene/locus/consequence chips in the MAF-stack card, because this distinguishes MAF features from spectra.
11. Show the combined tabular matrix as joined blocks, because the signatures-plus-MAF stack is an explicit concatenation.
12. Label the tabular model card as paired tabular models, because Elastic net and XGBoost share the tabular feature path.
13. Add metric chips to the model card, because the endpoints use different primary metrics.
14. Change the direct event-set card to "Direct event-set comparator", because only MuAt-compatible is in the measured manuscript figures.
15. Remove unsupported ATGC wording from the measured-flow caption, because ATGC is not benchmarked in these outputs.
16. Put the MuAt-compatible path outside the tabular path, because otherwise it can be misread as another engineered representation.
17. Use a bottom legend for arrow semantics, because the static PNG/PDF cannot rely on hover tooltips.
18. Keep the panel uncluttered with four major columns, because the figure is a conceptual overview, not a result table.
19. Use light cards and high-contrast labels, because the figure must remain legible when reduced in a PDF.
20. Export PNG, SVG, PDF, and HTML versions, because journals and coauthors need both raster preview and vector submission assets.

## Figure 2. Burden and Signature Baselines

1. Keep the figure focused only on burden versus mutational signatures, because it establishes the baseline before richer feature families.
2. Show endpoint metric/task/n chips on the left, because cross-endpoint values are not directly comparable without metric context.
3. Keep Elastic net and XGBoost as separate facets for non-survival endpoints, because they answer different linear/nonlinear baseline questions.
4. Label the right facet "XGBoost / Cox PH for OS", because OS is survival modeling, not XGBoost classification.
5. Remove OS bars from the Elastic net facet, because Cox PH should not masquerade as Elastic net.
6. Add an explicit OS note in the Elastic net facet, because blank survival cells need interpretation.
7. Keep paired delta panels beside each learner, because absolute scores alone do not show whether signatures improved over burden.
8. Use green/red signed deltas, because directionality matters for interpretation.
9. Include paired-test FDR markers, because readers need statistical context in addition to point estimates.
10. Draw confidence intervals where available, because effect-size uncertainty is more informative than stars alone.
11. Use a zero reference line in delta panels, because it anchors positive and negative changes.
12. Keep bars and deltas aligned by endpoint row, because visual pairing reduces lookup errors.
13. Use row banding, because there are multiple endpoints and small labels.
14. Use a muted gray for burden and blue for signatures, because the baseline and candidate are visually distinct.
15. Preserve exact OOF values as bar-end labels, because readers often need approximate numbers without consulting a table.
16. Use common x-axis ticks across facets, because scores should be visually comparable.
17. Include Harrell C-index in the subtitle, because survival now uses a different primary metric.
18. Keep the legend below the plot, because it avoids competing with the endpoint labels.
19. Avoid placing MuAt in this figure, because this figure is strictly a burden/signature baseline.
20. Pass SVG text-overlap QA, because any clipped or overlapping label weakens submission readiness.

## Figure 3. MuAt-Compatible Event-Bag Comparator

1. Make Figure 3 a dedicated MuAt-compatible comparator, because the prior Elastic net/XGBoost faceting made MuAt look like a tabular learner.
2. Show MuAt-compatible only once per endpoint, because it is one transformer-style event-bag model.
3. Compare against the best tabular result in each row, because that is the most scientifically conservative benchmark.
4. Also show the signatures-plus-MAF default, because it is the manuscript's practical combined tabular baseline.
5. Add a right-side MuAt-minus-best-tabular delta, because the key question is whether MuAt beats the strongest tabular result.
6. Use the same endpoint order as the other figures, because repeated row order reduces cognitive load.
7. Keep the same metric/task/n chips as the other main figures, because endpoint context remains essential.
8. Remove per-row series labels that overlapped sample counts, because the color legend is cleaner and sufficient.
9. Use a compact three-color legend, because the figure has only three compared series.
10. Use red/green delta direction colors, because the sign of MuAt's difference is the main interpretation.
11. Use a zero reference line for the MuAt delta, because near-zero differences should be visually obvious.
12. Keep the subtitle explicit about shared five outer folds, because comparability depends on identical OOF evaluation.
13. State that MuAt is outside Elastic net/XGBoost sections, because that directly addresses possible model-family confusion.
14. Use best-tabular bars in black, because they are the benchmark target in each row.
15. Use signatures-plus-MAF in orange, because that color is consistent across manuscript figures.
16. Use MuAt-compatible in slate, because it is visually distinct from tabular feature colors.
17. Avoid paired-test stars for MuAt deltas unless explicitly computed, because unsupported significance markers would overclaim.
18. Keep full provenance in SVG tooltips, because HTML review can still expose model and representation details.
19. Preserve the existing file stem for reproducibility, because downstream manuscript references already point to Figure 3 assets.
20. Export regenerated PNG/SVG/PDF/HTML after strict visual QA, because the displayed and source assets must match.

## Figure 4. MAF-Stack Biology vs Signatures

1. Keep Figure 4 focused on signatures, MAF stack, and signatures-plus-MAF, because this is the biological annotation comparison.
2. Exclude MuAt from this figure, because MuAt is not a MAF-stack tabular representation.
3. Preserve separate Elastic net and XGBoost facets for non-survival endpoints, because model family changes the MAF-stack effect.
4. Label the right facet "XGBoost / Cox PH for OS", because survival is evaluated with Cox PH.
5. Remove OS bars from the Elastic net facet, because linear survival is represented by Cox PH, not Elastic net.
6. Add the OS Cox note in the left facet, because the missing survival bars otherwise look like an omission.
7. Keep three tabular bars per endpoint, because the comparison needs signatures alone, MAF alone, and combined features.
8. Use a right-side contrast panel, because the figure emphasizes deltas rather than only absolute scores.
9. Show MAF minus signatures, because it isolates event-level biology from mutational process spectra.
10. Show signatures-plus-MAF minus signatures, because it tests whether event features add to spectra.
11. Show signatures-plus-MAF minus MAF, because it tests whether spectra add beyond event biology.
12. Use green/red contrast cards, because signed differences are the main result.
13. Keep FDR stars in contrast cards, because multiple endpoint/model contrasts need correction context.
14. Show intervals where bootstrapped, because uncertainty matters for manuscript interpretation.
15. Keep DeLong p/q behavior separate from bootstrap intervals, because binary AUROC tests differ from bootstrap deltas.
16. Use consistent colors with Table/Figure 5, because readers follow representations by color.
17. Align contrast cards to endpoint rows, because the three contrasts are otherwise easy to misread.
18. Keep the legend below the plot, because the right-side contrast panels already occupy visual space.
19. Use endpoint n labels, because MAF-stack performance can be sample-size sensitive.
20. Pass strict renderer QA, because the figure is wide and especially susceptible to label overlap.

## Figure 5. Cross-Endpoint Representation Summary

1. Separate MuAt-compatible into a direct event-set panel, because it is a transformer-style event-bag comparator, not Elastic net or XGBoost.
2. Split tabular results into linear and nonlinear/survival groups, because model family is a separate design axis from representation family.
3. Label the linear group as Elastic net only for non-survival endpoints, because OS is not evaluated there.
4. Label the nonlinear/survival group as XGBoost with Cox PH for OS, because this accurately describes the mixed endpoint modeling.
5. Mark Elastic net survival cells as "Cox only", because blank cells need a clear reason.
6. Show MuAt-compatible once per endpoint, because duplicating it under both tabular learner blocks was misleading.
7. Use one row per endpoint, because the figure is now a true cross-endpoint summary rather than repeated learner facets.
8. Use short representation headers, because full labels were cramped and hurt readability.
9. Keep endpoint metric/task/n chips, because each row uses a different endpoint definition or metric.
10. Use a unified heat scale for all displayed scores, because the figure summarizes the full main panel.
11. Keep exact numeric scores inside cells, because heat color alone is not precise enough.
12. Outline the best displayed score within each endpoint, because readers need the winner without scanning all numbers.
13. Add MuAt-minus-best-tabular text in the MuAt cells, because it directly answers whether the event-bag model beats tabular baselines.
14. Use red text when MuAt is below best tabular, because direction is the core interpretation.
15. Keep Cox PH labels inside OS cells, because survival values should not be mistaken for XGBoost.
16. Remove the old feature-richness arrow, because MuAt is not merely a richer tabular feature matrix.
17. Add a not-applicable legend, because survival cells intentionally differ from non-survival cells.
18. Use group header cards, because the column layout now encodes model family and representation family.
19. Avoid significance stars in the summary heatmap, because this figure prioritizes absolute score structure and Figure 2/4 carry paired tests.
20. Export all four asset formats and pass the completion gate, because Figure 5 is the main visual summary for submission.
