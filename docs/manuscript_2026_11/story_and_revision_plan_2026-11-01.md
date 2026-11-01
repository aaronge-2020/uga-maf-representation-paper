# Story and revision plan for the mutation representation manuscript

Written against the current Google Doc draft (read in full on 2026-11-01) and the completed 49-cell representation grid (primary contrasts in the extension goal's files/primary_contrasts.md).

## The central story

Current draft's implied story: Sig+MAF is the best representation, and annotations add value. The grid does not support the strong version of that claim, and the draft's own boundary results (survival null, Kucab reversal, PRAD transfer failure, HRD42 pooled reversal) already argue against it. A reviewer who reads the whole draft can see the tension.

Recommended story, one sentence: **The value of mutation annotation is specific, not general. It depends on the task, the learner, and the cohort, and this paper maps where it pays and where it does not.**

Under that story, the paper is a measurement paper, not a method paper. Its product is the map:

- Cancer typing with gradient-boosted trees: annotation pays a lot. Sig+MAF beats the best simpler encoding by about 0.08 balanced accuracy in the grid (0.7565 vs 0.6742 for TF-IDF) and beats spectra alone by about 0.20 in both benchmarks.
- Cancer typing with a linear model: the ranking flips. TF-IDF beats Sig+MAF (0.6483 vs 0.6215), and raw panel counts tie it.
- HRD: with XGBoost, panel counts and TF-IDF tie Sig+MAF (0.7599 and 0.7651 vs 0.7610). The annotation advantage on HRD is real against spectra alone and against the MuAt rebuild, but it is not separable from simple gene encodings.
- Survival: mutation features add nothing beyond cancer type (+0.0013, p=0.512).
- Transfer: the typing advantage survives centre holdout on all five tested types and frozen-model transfer to PCAWG overall (0.614 vs 0.479), and fails in named places (PRAD recall 0.03, driven by exome-to-genome count scale).
- Kucab damage classification: spectra alone beat the event representation. The ranking reverses outside cancer.

The draft already contains every one of these results. The revision promotes them from caveats to the thesis. The Discussion's current first sentence ("The predictive value of a somatic mutation representation depended on the outcome and evaluation setting") is the thesis already; the rest of the paper just does not commit to it yet.

This story is also the defensible one. A benchmark that reports where its own featured method loses is hard to dismiss as advocacy, and it converts the grid from a threat to the paper into the paper's strongest evidence.

Core premise (Aaron, 2026-11-01): Bio MAF is a representation optimized for these tasks, but neither the prior authors nor Bio MAF's own benchmarking ever priced how much each component of a representation contributes to prediction under one fixed evaluation. This paper performs that decomposition. One precision: CPEM did add blocks stepwise, but inside a single pipeline, in one fixed order, with plain accuracy on one task, which prices the pipeline rather than the components.

Pitch sentence (settled with Aaron, 2026-11-01, integrated version): Şimşek et al. and CPEM tested single encodings for cancer typing within TCGA, never pricing what each component contributed, never facing the shallow models against a per-mutation token model, and never testing transfer as a designed experiment; this paper does all three, and finds that gene identity and mutation patterns carry the predictive signal, that curated annotation adds value only for cancer typing with boosted trees, that the best tabular model beats a rebuilt MuAt attention model on identical splits for typing and HRD, and that its advantage survives centre holdout and external cohorts in most cancer types, with the failures named.

## The decomposition, component by component

What does the work, priced by removal or addition:

- Gene identity is the largest single component. Removing the gene features from Bio MAF drops cancer balanced accuracy from 0.662 to 0.540. In the grid, adding any gene-identity encoding to spectra adds about 0.11 on cancer (0.558 to 0.67) and about 0.09 on HRD (0.673 to 0.76).
- Mutation patterns are the base layer and win outright where the labels are mutation processes (Kucab damage classes, 0.331 and 0.250 for spectra alone).
- Allele-fraction summaries carry the HRD-specific gain. Removing them drops HRD33 from 0.900 to 0.874 (p<0.001).
- Mutation consequence summaries are the largest SHAP group (23.8%) but are redundant: gene identity columns carry the same signal, so removing the curated knowledge columns costs almost nothing (next item).

What was credited but does not do the work:

- Curated knowledge. Removing all 32 curated-knowledge columns changes cancer balanced accuracy by 0.003 (0.662 to 0.659).
- Class-relevance weighting, Şimşek's credited mechanism. The safe form of their weighting (TF-IDF) adds 0.003 over raw counts on cancer with XGBoost (0.6742 vs 0.6711), and the class-supervised form cannot score an unlabeled patient at all. The counts under the weighting were doing the work.
- The known HRD mechanism. Removing the 16 DNA repair genes' features leaves HRD unchanged (0.760 to 0.763, p=0.622), and SBS3 alone and BRCA1/2 status alone sit at random (AUROC 0.501 to 0.524). Caveat the draft already carries: the SBS3 and BRCA controls are sparse (9 and 14 positive tumors of 772), so the claim is that these proxies carry none of the model's signal as measured, not that the biology is irrelevant.

The one hole: the component removals above were run at archived model settings under the main-panel protocol. The grid decomposes at the level of whole encodings under the fair protocol, not yet at the level of components inside Bio MAF. Workstream B2 (retuned additions to spectra plus counts and retuned removals from Bio MAF) is the piece that closes that hole, and the premise sentence needs it if the paper claims a full decomposition.

Scope limit (Aaron, 2026-11-01): copy number is the one major component of a prior pipeline this study cannot price. The benchmark is built from mutation calls only, so CPEM's copy-number block (13,040 of its 35,565 features; CPEM's own stepwise table credits it with +2.5 points, 58.5 to 61.0) stands untested here. Every decomposition claim in the paper must be scoped to mutation-derived components, and the manuscript should carry one sentence saying the copy-number layer was not tested. Şimşek is priced in full, and CPEM's mutation-side blocks (per-gene counts, trinucleotide channels) are priced; only its copy-number block is not.

## Where the 49 cells go

Not as a 49-row table. The full grid is a supplementary object (all 49 cells, per-fold scores, alone arms, all-genes and log1p arms) and belongs in the supplement as one table with the protocol stated above it.

What enters the main text is the combined-representation contrast: five combined representations, two tasks, two learners, 20 primary scores with paired differences against Sig+MAF and bootstrap intervals. That is one compact main table (new Table 2). It is the fairest single object in the paper and it carries the story's central contrast.

To make room, move Figure 4 (mutation count vs patterns baselines) to the supplement. Its content is already carried by the Burden and Spectra columns of Table 1 and by the alone arms in the supplementary grid table. Figure 5 can stay, since its threshold-versus-continuous HRD contrast is a distinct point, but its caption should stop presenting pooled values next to Table 1 fold means without the current clarifying note.

Nothing else comes out. Table 1 stays as the spine (endpoints by representations, MuAt column, survival row). Figures 2, 3, 6, 7, and 8 each carry a distinct claim.

## One integration problem to solve before any of this lands

The grid and Table 1 are different evaluations. Sig+MAF with XGBoost on cancer typing scores 0.738 (fold mean) in Table 1 and 0.7565 (mean within fold) in the grid. The split sets differ (the grid uses its own canonical splits), the tuning budgets differ, and the XGBoost versions differ (1.6.2 GPU hist for Table 1, 3.0.5 for the grid). A reviewer will find this in minutes.

Handle it openly:

1. Present the grid as a second, purpose-built benchmark for the encoding question, with its own short protocol paragraph in Methods (canonical splits, tuning inside training folds, the 0.01 and 0.02 practical-similarity margins, paired bootstrap).
2. Never mix numbers from the two benchmarks in one sentence or one comparison. Each table carries its protocol in its note.
3. Say once, in the new Results section, that the two benchmarks agree on the cancer-typing ranking and differ in level because the splits and tuning differ. Agreement of ranking across two protocols is evidence for the story, not an embarrassment.

## Section-by-section revision

**Title.** Keep. "Benchmarking Somatic Mutation Representations" already promises a benchmark, which is what the revised story delivers.

**Abstract.** Rewrite around the map, not the trophy. Open with the question (when does gene and annotation information improve on counts and spectra), then give the four headline findings in story order: the typing gain with boosted trees, the learner flip and HRD tie with simpler encodings, the survival null, and the uneven transfer with the named PRAD failure. Close on the map sentence. Drop the current closing claim ("added predictive value of gene specific mutation summaries") as the lead conclusion; it is true only in part of the map.

**Introduction.** Change the gap statement. The current gap is "when gene information improves on counts and spectra remains unclear." The sharper gap, now supported by the crosswalk: prior comparisons each tested one encoding idea, on one task, under their own evaluation, and the strongest published weighting scheme cannot score an unlabeled patient as published, so the field has never had a like-for-like answer. The paper's offer: one fixed, inference-safe evaluation across encodings, tasks, learners, and cohorts. Şimşek and CPEM move from background citations to named foils, with the CPEM counts arm and the TF-IDF arm identified as their ideas rebuilt under the shared protocol (with the crosswalk naming discipline kept: "in the style of", never the pipeline names).

**Results.** Insert one new section after "Mutation count and mutation patterns baselines": "Simpler gene encodings under matched tuning." It reports Table 2, the three headline contrasts (typing XGBoost gap, typing elastic-net flip, HRD tie), the alone-arm pricing (spectra alone 0.5582 cancer and 0.6734 HRD under the grid protocol; the gene block's added value over spectra for each encoding), and the one-sentence protocol note from the integration section above. Keep it to one section and one table; the all-genes and log1p results get one sentence each with pointers to the supplement (all-genes adds nothing over panel: 0.6704 and 0.6819 vs 0.6711 and 0.6725).

**Discussion.** Replace the archived-settings controls paragraph (binary 0.654, role-matched counts 0.614, with its "retuning could change the differences" caveat) with the grid results, which answer that caveat directly. Soften the HRD language to match: the combination's HRD advantage holds against spectra and against the MuAt rebuild; against retuned simple encodings it does not separate. Keep the MuAt protocol discussion, the calibration discussion, and the limitations; they already serve the measurement story. Add one short paragraph on inference safety as a reporting standard for representation papers: a representation should be computable for an unlabeled patient from statistics fitted on training data only, and prior headline methods should be checked against that standard before they are cited as baselines.

**Methods.** Add the grid protocol paragraph (splits, learners, tuning budgets, margins, bootstrap, seed) and one line pointing to the reproduction package. The one-line reproduction command and the results ledger belong in Data and Code Availability, which strengthens the fairness claim the story rests on.

## Novelty, stated the way a reviewer will test it

1. **The comparison itself is new.** Şimşek 2019 varies a weighting scheme on one task (10 grouped types, micro-averaged accuracy). CPEM 2019 builds one accuracy-maximizing pipeline that requires copy-number data. Salvadores 2019 combines blocks inside one model for typing. None swaps encodings one at a time on identical patients, folds, and learners across typing, HRD, and survival. The unit of novelty is the fixed evaluation, not Bio MAF.
2. **The inference-safety audit is new and checkable.** Şimşek's winning BM25-tf-rf weighting is defined through the patient's true class, so no faithful implementation can score a new patient. Naming that, with the formula-level reason, and supplying the safe substitute (training-fitted TF-IDF) is a methodological contribution any reader can verify from the 2019 paper's text.
3. **Retuned controls are new.** Prior comparisons, including this draft's own archived-settings controls, priced alternatives at the winner's settings. Here every encoding is tuned inside its training folds, so the gaps in Table 2 are encoding gaps, not tuning gaps.
4. **The boundary map is new.** The survival null against a clinical baseline, the Kucab ranking reversal, the PRAD transfer failure with its count-scale explanation, the threshold-versus-continuous HRD boundary, and the learner-dependent flip are reported as findings with numbers, not as limitations prose.
5. **Reproducibility is part of the claim.** A one-command package that recomputes every reported number from hash-pinned artifacts, with a public ledger of which numbers reproduce and which do not, exceeds what either 2019 paper or MuAt provides.
6. **The shallow-versus-neural comparison is new (Aaron's emphasis, 2026-11-01).** Şimşek's neural network and CPEM's deep ensemble both consumed tabular encodings. Neither faced tabular representations against a per-mutation token model. This paper trains a MuAt-compatible attention model from scratch on identical splits and beats it on typing (0.738 vs 0.528 balanced accuracy; margin 0.209) and HRD (0.777 vs 0.507 Spearman; 0.911 vs 0.746 AUROC). The claim must stay scoped to the evaluated rebuilds: no published pretrained weights were used, and the architecture-search rebuild (0.590) is the fairer comparator and still loses. Claim "the evaluated tabular pipeline beat both MuAt rebuilds on identical splits," never "tabular beats neural."
7. **External validation is a designed experiment, not an afterthought (Aaron's emphasis, 2026-11-01).** Şimşek ran no external cohort; CPEM's single cross-cohort application collapsed (6.79% on ICGC esophageal cancer). This paper tests transfer three ways with per-type reporting: centre holdout (combination ahead on all five types), frozen models on PCAWG (0.614 vs 0.479 accuracy, with PRAD at 0.03 named as the main failure and its count-scale explanation), and POG570 (ranking kept at Spearman 0.667, calibration lost, specificity 0.034 at the original cutoff). The failures are part of the evidence.

What not to claim: best overall accuracy (CPEM's 84.09% is pipeline accuracy on a different metric and cohort definition, and the crosswalk bars the comparison), first use of any single encoding, and universal superiority of Sig+MAF. The draft's current text mostly respects these; the revision should check every comparative sentence against them.

## Order of work

1. Freeze the story above with Vijay and Cindy (this memo is the discussion document).
2. Rewrite the Abstract and the Introduction gap paragraphs.
3. Insert the new Results section and Table 2; move Figure 4 to the supplement; add the full grid as a supplementary table.
4. Rewrite the Discussion controls paragraph and soften the HRD claims; add the inference-safety paragraph.
5. Add the Methods protocol paragraph and the availability update once the reproduction package passes its fresh-clone check.
6. Run the claim audit over the revised text before submission.
