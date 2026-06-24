# Main Figure Overlap And Collision Improvement Proposals

Scope: main manuscript Figures 1-5 in `results/manuscript/figures`. This second pass focuses on overlapping text, crowded labels, element collisions, and static-export readability. Each proposed fix has been implemented in `src/visualization/render_manuscript_figures.mjs`.

## Figure 1. Conceptual Overview

| # | What was wrong | Why it needs to be fixed | How to fix it | Why the proposed fix is better |
|---|---|---|---|---|
| 1 | Gene text in the event table sat close to the colored row dots. | The dot could be read as part of the gene label at small sizes. | Move row dots farther right and extend row rules to match. | The gene names and sample glyphs are visually separate. |
| 2 | Panel badges and headings were close to the main content. | Tight badge spacing makes panel labels compete with card titles. | Reserve a clear header lane above each panel. | Captions can reference panels without interfering with the diagram. |
| 3 | The first feature card had histogram bars close to its description text. | Data glyphs should not visually touch explanatory text. | Keep glyphs in the lower half of the card. | Text reads first, then visual encoding reads second. |
| 4 | Sequence tiles and KME points shared the same horizontal band. | The sequence-to-embedding transformation was visually crowded. | Separate sequence tiles left and point cloud right. | The transformation is legible without collisions. |
| 5 | MAF chips were close to the lower card edge. | Small chips can look clipped in PDF export. | Increase internal padding around chip rows. | The card remains clean in raster and vector exports. |
| 6 | The joined matrix blocks were close to the explanatory paragraph. | The mini matrix could look attached to the paragraph. | Put the matrix in a dedicated lower visual zone. | The combined representation reads as an object, not decoration. |
| 7 | Solid arrows entered cards near text. | Arrowheads can be mistaken for bullets or labels. | Align arrowheads to card edges and away from text baselines. | Direction is clear without text collision. |
| 8 | The dashed comparator arrow crossed the lower canvas near the legend. | Curves and legends can compete when they share a lane. | Keep the dashed path above the footer legend. | The comparator path and legend are visually distinct. |
| 9 | The bottom legend had text close to line glyphs. | Tight legend spacing hurts print readability. | Increase the line-to-label gap in the footer. | Arrow semantics are easier to decode. |
| 10 | Metric chips in the model card were tightly packed. | Adjacent pills can look like one combined label. | Add consistent chip gaps. | Each metric is individually readable. |
| 11 | The row-rule endpoints in the input table stopped before the dot lane. | A truncated rule made dots look outside the row structure. | Extend rules behind the dot lane. | Dots now belong to the event rows without touching text. |
| 12 | Long subtitle text could crowd panel headers. | Header hierarchy should remain stable across export sizes. | Keep the subtitle in a bounded top text block. | The figure keeps a clean reading order. |
| 13 | Feature numbers and titles were close in B panel cards. | Number bubbles can collide with title text if fonts differ. | Use fixed title starts after the number bubbles. | Rendering is robust to font substitution. |
| 14 | Direct-event-set card was visually close to the dashed arrowhead. | Arrowheads near text boxes can feel like collision. | Anchor the arrow to the card edge, not inside the card. | The dashed path ends cleanly. |
| 15 | The input field chips had narrow spacing. | Pill labels can blur together in print. | Use uniform chip widths and gutters. | Field names remain discrete. |
| 16 | Small provenance labels were close to visual glyphs. | Provenance text should not compete with schematic marks. | Place provenance in text lanes above or below graphics. | The figure is more readable when scaled down. |
| 17 | Card borders and colored glyphs sometimes used similar positions. | Border and content collisions weaken grouping. | Add more internal card padding. | Borders frame content instead of touching it. |
| 18 | The main workflow path and conceptual comparator path had similar vertical proximity near the input card. | Readers could confuse primary and dashed branches. | Start the dashed comparator below the primary branch. | Primary and comparator paths are visually separated. |
| 19 | The mini matrix labels could crowd their block borders. | Labels near borders are hard to read in SVG/PDF. | Center labels with more top padding. | The joined feature blocks are clearer. |
| 20 | The canvas had several unstructured white regions that made arrows long. | Long arrows increase the chance of crossing other elements. | Tighten the stage layout while preserving gutters. | Fewer long connections means fewer collisions and faster scanning. |

## Figure 2. Burden And Signature Baselines

| # | What was wrong | Why it needs to be fixed | How to fix it | Why the proposed fix is better |
|---|---|---|---|---|
| 1 | Delta labels sat directly on CI whiskers. | Text over a whisker obscures both the value and interval. | Move delta labels into fixed side label lanes. | The interval and label can be read independently. |
| 2 | Negative delta labels were close to the zero line. | Near-zero losses can look clipped or ambiguous. | Put negative labels in a left-side lane with a pale red backing. | Losses are readable and clearly negative. |
| 3 | Positive delta labels were close to points. | Labels near markers can hide the marker location. | Put positive labels in a right-side lane with a pale green backing. | The marker position remains visible. |
| 4 | CI whiskers and labels shared the same vertical baseline. | Shared baselines create visual collisions. | Draw whiskers above the label baseline. | Intervals and numeric labels no longer overlap. |
| 5 | Rows were too compressed for bars plus delta glyphs. | Dense rows make labels feel stacked. | Increase row height for bar-and-delta figures. | Each endpoint has a clean visual lane. |
| 6 | Bar value labels were close to bar edges. | Values can look attached to the bar stroke. | Add a consistent offset after each bar. | Scores are readable without covering bars. |
| 7 | The model tags were near headings. | Tight heading chips can merge with titles. | Preserve a fixed gap between model title and tag. | Facet headers read cleanly. |
| 8 | Bottom axis ticks were close to legend text. | Axis and legend text can overlap in wide figures. | Keep axis titles above the legend and use more bottom padding. | The decoding aids are separated. |
| 9 | Endpoint chips and n labels were compact. | Metric, task, and sample count could visually run together. | Use chip gutters and a separate n text position. | Row metadata is easier to scan. |
| 10 | Row bands ended close to data marks. | Bands should guide rows, not compete with plotted marks. | Extend bands across full figure width with low contrast. | Rows are easier to follow without adding clutter. |
| 11 | The delta zero band was too subtle when labels crossed it. | A subtle zero line can vanish beneath labels. | Keep labels out of the zero band. | The zero reference remains visible. |
| 12 | Delta tick labels used the same lane as label text. | Tick labels can conflict with row annotations. | Keep delta labels inside row lanes and ticks in the axis lane. | Axis scale and row values are distinct. |
| 13 | Significance stars were appended to raw delta text. | Stars can touch numerals and reduce clarity. | Put the full delta-plus-star label inside a pill. | The combined label is visually grouped. |
| 14 | CI cap lines touched small labels in near-zero rows. | Caps can look like punctuation. | Offset label baselines below caps. | Whisker caps read as interval marks. |
| 15 | Legend items were near the left edge. | Edge-adjacent legends can look cramped. | Keep a consistent left margin for legend entries. | The legend aligns with the row label column. |
| 16 | The blue and gray bars had similar vertical spacing to text. | Adjacent bars need clear separation from row metadata. | Increase bar-row spacing slightly. | The two representations are easier to compare. |
| 17 | The delta panel did not reserve room for all possible labels. | Long labels with stars could overflow or collide. | Reserve fixed label lanes independent of marker position. | Labels remain stable across endpoints. |
| 18 | Some values near the right edge were close to panel boundaries. | Edge crowding makes values look clipped. | Expand the figure width and delta panel width. | End labels have breathing room. |
| 19 | The positive/negative legend was close to plotted data. | Legends can be mistaken for data marks. | Put the sign legend in the header lane. | The legend decodes the plot without touching data. |
| 20 | The previous QA only caught text-text overlap. | Text can collide with lines and markers too. | Use layout lanes that avoid element classes by construction. | The figure remains readable beyond automated text checks. |

## Figure 3. One-Hot KME Versus Signatures

| # | What was wrong | Why it needs to be fixed | How to fix it | Why the proposed fix is better |
|---|---|---|---|---|
| 1 | KME delta labels overlapped CI whiskers in several rows. | The effect size and interval could not be separated visually. | Move labels into fixed side lanes. | Delta values no longer sit on interval lines. |
| 2 | Near-zero XGBoost deltas sat on the zero reference. | Near-zero effects are the most sensitive to collision. | Offset marker glyphs and place labels below. | Null-like effects remain legible. |
| 3 | Red loss labels touched marker glyphs. | A loss marker can be obscured by its label. | Use red label pills away from the marker. | Both loss magnitude and point location are visible. |
| 4 | Positive KME labels could overlap right-side CI caps. | Labels at cap endpoints obscure uncertainty. | Put positive labels in a right lane beyond the glyph zone. | CI caps remain readable. |
| 5 | The delta panel was too narrow for signed labels. | Long labels with stars can collide with ticks. | Increase delta panel width. | Labels and axis ticks have separate spaces. |
| 6 | The score and delta panels were visually crowded within each facet. | Crowding weakens the paired comparison. | Increase facet width and row height. | Bars, values, CIs, and labels fit comfortably. |
| 7 | KME version text in the subtitle competed with the legend. | Repeated details can make the top region busy. | Keep configuration detail as a short note, not a long legend item. | The header stays readable. |
| 8 | The same horizontal baseline was used for marker and label. | Same-baseline marks collide in dense rows. | Use two vertical lanes per delta row. | Whiskers and labels are visually independent. |
| 9 | Metric chips were close to endpoint labels. | Metadata could merge with endpoint names. | Keep endpoint labels above metric/task chips. | Row labels scan as a structured stack. |
| 10 | Value labels near bar ends were small and close to strokes. | Tiny labels can be swallowed by bar outlines. | Use consistent offsets after bar ends. | Scores are more legible. |
| 11 | Winner outlines could touch adjacent bars. | Outlines need enough clearance not to look like overlap. | Add slightly taller row spacing. | Winner cues remain clear. |
| 12 | Delta axis tick labels were close to panel title. | Axis and title should not share a line. | Keep title in header and ticks in bottom axis lane. | The delta scale is easier to parse. |
| 13 | Positive and negative direction legend was near the data. | It could be mistaken for a plotted row. | Move sign legend to the header lane. | The legend reads as guidance, not data. |
| 14 | Labels with q stars were not visually bounded. | Stars can drift into nearby elements. | Put signed labels and stars inside compact pills. | Statistical text is grouped. |
| 15 | Some labels were placed based on marker position. | Position-based labels collide when points are near zero. | Use fixed lanes independent of delta value. | Near-zero and large effects both render safely. |
| 16 | The zero-band background overlapped numeric labels. | Background bands should not sit under text. | Keep labels outside the zero-band region. | The zero reference remains clean. |
| 17 | The row banding did not fully solve per-row crowding. | Row guidance does not prevent local collision. | Add vertical separation inside each row. | Each row has separate sublanes for bars and deltas. |
| 18 | KME and signature bars were tightly stacked. | Adjacent bars can visually merge. | Increase bar pitch from 22 to 24 pixels. | The paired bars are distinct. |
| 19 | Long endpoint names could compete with chips. | Left metadata should be compact but not cramped. | Use fixed left label block and metadata chip positions. | Labels remain aligned across rows. |
| 20 | Static exports relied on tooltip detail for exact q values. | Readers cannot hover over manuscript figures. | Render signed delta and star labels directly. | The exported figure is self-contained. |

## Figure 4. MAF-Stack Biology Versus Signatures

| # | What was wrong | Why it needs to be fixed | How to fix it | Why the proposed fix is better |
|---|---|---|---|---|
| 1 | Contrast labels were inside the same lane as contrast glyphs. | Text could sit on mini-bars and zero lines. | Split each contrast cell into glyph and label sublanes. | Deltas and labels no longer collide. |
| 2 | Contrast cells were shallow. | Shallow boxes force text, points, and intervals together. | Increase contrast-cell height. | Each contrast has room for interval, marker, and label. |
| 3 | Three contrast columns were cramped. | Narrow columns make labels and CIs compete. | Increase contrast panel width. | Multi-contrast comparisons are easier to read. |
| 4 | The contrast panel heading was long. | A long heading can overlap sign legends or columns. | Shorten heading to "Pairwise deltas; CI if bootstrapped". | The title communicates the rule without crowding. |
| 5 | CI lines could pass through label text. | Intervals should not obscure numeric deltas. | Draw CI/glyphs above labels. | Uncertainty and effect size are both visible. |
| 6 | Zero lines ran through the label area. | Zero references can look like minus signs. | Limit zero reference to the glyph lane. | Labels are not visually contaminated. |
| 7 | Labels with stars could exceed the cell width. | Long statistical labels can spill into neighbors. | Use fixed-width label pills inside each cell. | Labels stay contained. |
| 8 | Row spacing was tight for three bars plus three contrasts. | Dense rows make cross-row scanning difficult. | Increase row height for contrast figures. | Each endpoint has a cleaner horizontal band. |
| 9 | Contrast background fills were close to row bands. | Similar backgrounds can blend. | Keep contrast pills outlined and slightly inset. | Data cells remain distinct from row striping. |
| 10 | Model facet title and contrast legend were close. | Header collisions reduce hierarchy. | Move sign legend to a separate header lane. | Titles and legends do not compete. |
| 11 | Score labels near bars were close to value strokes. | Text should not touch bar outlines. | Add a consistent score label offset. | Values remain readable. |
| 12 | Bar row pitch was too tight for three representations. | Three bars can visually merge if pitch is low. | Increase bar pitch and bar height carefully. | The three representations are distinguishable. |
| 13 | The right panel had no reserved label zone. | Labels followed data positions and could collide. | Reserve one label position per contrast cell. | Labels remain stable across endpoints. |
| 14 | Positive and negative cells had similar text treatment. | Direction could rely too much on color. | Pair color with signed labels and a zero line. | Direction is clear in color and text. |
| 15 | DeLong rows without CIs could look incomplete. | Missing CIs should not be confused with rendering failure. | Keep the note but render p/q labels cleanly. | Readers understand why no interval appears. |
| 16 | The contrast columns could be mistaken for data bars. | Mini-bars need their own visual grammar. | Use boxed contrast cells with center zero lines. | Contrasts read separately from score bars. |
| 17 | Long comparison labels wrapped tightly. | Headers can collide with adjacent columns. | Use short column headers and explain in tooltips/title. | Column labels stay compact. |
| 18 | XGBoost contrast cells were close to the figure edge. | Edge crowding can clip labels in export. | Increase total figure width. | The rightmost contrast has safe margins. |
| 19 | Stars appended to labels could look like stray marks. | Statistical marks should be grouped with the value. | Put stars inside the same label pill. | Stars stay attached to the relevant delta. |
| 20 | Previous element spacing was manually tuned for one export size. | Manuscript figures may be resized. | Use fixed sublanes and larger gutters. | The figure is robust to scaling. |

## Removed Cross-Endpoint Summary

The former cross-endpoint heatmap duplicated the endpoint/model matrix already covered by the main model-comparison panels. The manuscript now keeps that non-survival information in Figures 2-4 plus Table 2 and uses the Figure 5 slot for a survival-specific CoxNet C-index panel.
