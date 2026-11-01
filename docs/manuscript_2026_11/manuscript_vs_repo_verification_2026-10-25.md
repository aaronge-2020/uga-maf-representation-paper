# Manuscript vs repository verification, 2026-10-25

Scope: the uploaded PDF draft "Benchmarking Somatic Mutation Representations (full manuscript draft)" (18 pages, Google Docs render) checked against github.com/aaronge-2020/uga-maf-representation-paper.

## Latest branch and commit

GitHub has exactly four branches. The newest commit overall is on **main**:

| Branch | Head | Date |
|---|---|---|
| main | ca13132462fab9d1be364a2477ea8e753d7d5050 | 2026-10-14 14:12 EDT, "Rename bio_maf_v4.txt to bio_maf.txt (drop v4)" |
| reviewer-analyses-2026-10 | c738064ac7b50e2e07227e3c4395a4e924a3339e | 2026-10-14 08:28 EDT |
| codex/coxnet-muat-manuscript-updates | b3d830cfed249e451accd290a6c1bef54ff75b16 | 2026-10-05 (Vijay) |
| main-backup-2026-09-23 | 4838bf9b844211ba0ba018b3aa1119f81373c19c | 2026-09-23 |

The local clone was configured to fetch main only, which hid the other three branches; they were cloned separately for this check. The two result files that carry the headline numbers, `main_manuscript_complete_panel_endpoint_results.csv` and `main_manuscript_complete_panel_fold_metrics.csv`, are byte-identical on all four branches, so the newest numbers do not depend on which branch you pick.

## What the PDF gets right (verified against repo files on main @ ca13132)

**Table 1 is exact.** Every one of the 33 cells matches the mean of the five per-fold scores in `results/tables/main_manuscript_complete_panel_fold_metrics.csv`, to rounding. The PDF is explicit that Table 1 reports fold means, and it does. Verified cells include: HRD score Sig+MAF XGBoost 0.777 (folds average 0.7765), HRD33 0.911 (0.9111), HRD42 0.925 (0.9252), cancer type Sig+MAF 0.738 / Bio MAF 0.662 / spectra 0.538, OS 0.651 / 0.634 / 0.607 / 0.584. The three printed confidence intervals also match t-intervals computed from the five fold scores: HRD score (0.745, 0.808), HRD33 (0.886, 0.936), cancer type (0.721, 0.755).

**MuAt column is exact, and it is the faithful run.** Cancer: 0.528 balanced accuracy, 0.549 accuracy, 0.861 top-5, per-fold balanced accuracies 0.540, 0.528, 0.519, 0.540, 0.515 (all recomputed from `muat_style_tcga_comparator_cancer_type_top20_faithful_fixed_arch_oof_predictions.csv`). HRD: 0.507 Spearman and 0.746 AUROC, fold means of the faithful fixed-architecture run; per-fold AUROCs recomputed from the OOF file are 0.718, 0.713, 0.714, 0.776, 0.807. The paired differences (+0.270 Spearman, p=0.0001, CI 0.220 to 0.319; +0.165 AUROC, p=0.003, CI 0.094 to 0.236) match `hrd_muat_paired_tests.json` exactly. The architecture-search run (0.602 accuracy, 0.590 balanced accuracy, 0.886 top-5) matches `muat_style_tcga_comparator_cancer_type_top20_comparable_fold3_rerun_v2_endpoint_results.csv`.

**Ablations are exact (pooled, from `task1_gene_knowledge_ablations.csv`).** Bio MAF alone 0.662, without gene identity 0.540, without knowledge summaries 0.659. Sig+MAF 0.738, without gene identity 0.677, binary-gene control + spectra 0.654, driver-count control + spectra 0.614.

**External validation is exact.** PCAWG Sig+MAF: accuracy 0.614 (CI 0.583 to 0.645), balanced accuracy 0.557, and all nine per-type recalls match `external_validation/pcawg_transfer/predictions/pcawg_sigmaf_metrics.json` and `task5_pcawg_audit_per_type.csv` (BRCA 0.85, LIHC 0.83, KIRC 0.82, ESCA 0.77, SKCM 0.73, OV 0.67, PRAD 0.03, HNSC 0.15, STAD 0.16). Spectra-only PCAWG: 0.479 accuracy, 0.365 balanced accuracy, BRCA 0.077, KIRC 0.000, from `pcawg_metrics_with_95ci.csv`. POG570 breast (n=147): Spearman 0.667, Pearson 0.670, R2 -0.225, specificity at cutoff 24 = 0.034, all from `pog570_primary_and_all_metrics_with_95ci.csv`.

**Centre-held-out is exact (pooled recall, `task4_centre_heldout.csv`).** OV 0.64 vs 0.40, COAD 0.57 vs 0.13, KIRC 0.73 vs 0.46, KIRP 0.43 vs 0.22, CESC 0.12 vs 0.10, Sig+MAF ahead on all five types.

**Survival is exact.** Main panel (Table 1): OS 0.584 / 0.607 / 0.634 / 0.651; PFI 0.556 / 0.570 / 0.621 / 0.631 (fold means). Cancer-type covariate run (`survival_cancer_type_baseline_summary.csv`): type alone 0.742, Bio MAF alone 0.638, both 0.743, gain +0.0013, p=0.512.

**HRD controls are exact.** Repair-gene removal: HRD score 0.760 to 0.763 (p=0.622), HRD33 0.900 to 0.898 (p=0.637); VAF removal: HRD33 to 0.874 (p<0.001), from `hrd_shortcut_ablation_summary.csv`. Signature 3 alone: -0.068 Spearman, 0.501 / 0.510 AUROC; BRCA1/2 alone: -0.065, 0.507 / 0.524, from `task3_hrd_attribution_baselines.csv`.

**Figure 3 per-type gains are exact** (recomputed from the main OOF predictions): ESCA +0.60, SARC +0.42, PRAD +0.36, CESC +0.29, SKCM +0.00, LIHC +0.06; no type worse; per-type n from 184 to 1,020.

**Figure 6 SHAP shares match** `feature_selection_cancer_type_top20_shap_values.csv` under the natural block grouping (consequence 23.8%, per-gene driver counts 17.3%, impact 10.9%, OncoKB 9.1%, mutation type/count 4.4%, and the rest consistent). Feature selection: 347 of 948 retained, held-out 0.697, matches the summary JSON.

**Kucab is handled correctly in the PDF.** Table 1 has no Kucab row, the Additional results give spectra 0.331 / 0.250 (the repo pooled values), and the Discussion states plainly that the Kucab event features were not the full Bio MAF representation. This fixes the older repo manuscript, which reported the Kucab event-stack numbers as "Bio MAF".

## Mismatches and problems

### In the PDF itself

1. **Wrong figure cross-reference.** The Discussion says adding Bio MAF to cancer type produced no improvement in "the separate overall-survival analysis (Figure 6)". Figure 6 is the SHAP figure. The survival figure is Figure 8, and the Results section cites it correctly.
2. **Centre-held-out contradiction.** Results: Sig+MAF "beat mutation patterns alone on all five multi-center cancer types". Discussion: "the combination improved pooled recall for four of the five evaluated cancer types". The repo pooled recalls support "all five" (CESC 0.118 vs 0.104 is the fifth). One of the two sentences needs to change; if "four of five" came from a per-centre reading, it should not say pooled.
3. **Missing confidence interval values.** The HRD paragraph reads "At thresholds 24 and 42 the AUROC was 0.860 and 0.925 (95% CI, ahead of mutation patterns alone (0.839 and 0.915)". The interval values are absent and a parenthesis never closes. The fold-based t-intervals from the repo are (0.831, 0.889) for threshold 24 and (0.895, 0.955) for threshold 42.
4. **Wrong package version in the Abstract.** "the Bio MAF command-line package (biomaf-v4, version 4.0.0)". The repo package is `biomaf`, version **1.0.0** (`biomaf/pyproject.toml`). The PDF's own Data availability section says 1.0.0, so the Abstract contradicts both the repo and the PDF itself.
5. **"q=0.0"** for the cancer Sig+MAF vs Bio MAF contrast. A q value is never exactly zero; this should read as a bound (for example q<0.001). The difference itself (+0.076, CI 0.067 to 0.085) matches the pooled difference 0.7378 - 0.6618 = 0.0760.

### Numbers in the PDF that the repo cannot support (no source file found)

6. The newer secondary-contrast statistics: HRD33 gain +0.037 (raw p=0.058, BH q=0.068) and HRD score gain +0.082 (BH q=0.0026). The differences match the fold-mean differences (0.9111 - 0.8742 = 0.0369; 0.7765 - 0.6943 = 0.0822), but no repo file carries these p or q values. The repo's only stored version of this contrast is the older pooled bootstrap in the plot data (q=0.109 for HRD33), which the repo's own audit already flagged for grouping BH corrections globally rather than within families.
7. PRAD rescaling in PCAWG, "raised PRAD recall from 0.033 to 0.417": the 0.033 is in the repo; no file in the repo reports 0.417 or a rescaling run.
8. SIFT missing for 55% of PCAWG missense calls (vs 9% in TCGA) and PolyPhen for 43% (vs 2%): no repo file reports these percentages.
9. "Only nine tumors had nonzero fitted SBS3 activity and fourteen met the mutation-indicator definition": no repo file reports these counts.
10. HRD42 "bootstrap interval of 0.902 to 0.946" in the Discussion: no repo file reports it. The fold t-interval from the repo is (0.895, 0.955).
11. Zenodo DOI 10.5281/zenodo.23122728 in the Abstract: the DOI does not appear anywhere in the repo.
12. MuAt runtimes (18.5 hours cancer, 3.0 hours HRD): not recorded in the result files checked.

### Inconsistencies inside the repo (the PDF mostly avoids them, but they are live)

13. **The saved OOF predictions file does not reproduce the tables for a subset of groups.** `main_manuscript_complete_panel_oof_predictions.csv` and the endpoint/fold tables agree exactly for many groups (all Kucab rows, all burden and spectra regression rows), but not for others. Largest gaps, recomputed from the OOF file vs the endpoint table (pooled): Sig+MAF elastic net cancer type 0.586 vs 0.672; HRD42 Sig+MAF elastic net AUROC 0.905 vs 0.825 (here the OOF file is *higher*); HRD33 Bio MAF elastic net 0.830 vs 0.769; HRD33 Sig+MAF XGBoost 0.867 vs 0.901; OS Sig+MAF C-index 0.633 vs 0.651. The fold metrics and endpoint tables agree with each other and with the PDF, so the OOF file is the odd artifact out, most likely written by a different run for those groups. Anyone who recomputes from the OOF file, as a reviewer would, will not get Table 1 for those cells.
14. **The SHAP table in `docs/results_summary_2026-10.md` (all branches) does not match the SHAP data file.** The doc gives consequence 21.9%, per-gene 19.9%, impact 15.2%, VEP other 13.1%. Aggregating `feature_selection_cancer_type_top20_shap_values.csv` gives the PDF's numbers (23.8 / 17.3 / 10.9), not the doc's. The doc's aggregation is stale or used a different grouping and should be fixed or deleted.
15. **`cancer_typing_primary_fold_differences.json` is stale.** It pairs Sig+MAF with the superseded MuAt comparable run (including the diverged fold 3 at 0.358) and reports a mean difference of 0.188, p=0.017. The PDF's +0.209 (CI 0.197 to 0.222) comes from the faithful run's folds and is the correct current pairing; the JSON says in its own note that a rerun "will supersede these numbers", and it was never replaced.
16. **The manuscript stored in the repo is an older draft than the PDF.** `submission_manuscript.docx` on main and `results/manuscript/text/manuscript_captions_and_results.md` report pooled numbers throughout (HRD Sig+MAF 0.767, HRD33 0.901), use the superseded MuAt comparable run (0.519 Spearman, 0.729 AUROC), include a Kucab row labelled as Bio MAF, and describe Bio MAF as 910 features in one Methods paragraph and 948 elsewhere. None of these should be quoted tonight; the PDF supersedes them, but the repo should be brought into line before anyone else reads it as the current draft.
17. The plot data manifest references `canonical/main_panel_pairwise_tests.csv`, which does not exist in the repo.

## Edits made to the Google Doc, 2026-10-25

At Aaron's direction, the live Google Doc (the source of the PDF above) was edited in place and read back to confirm. The Introduction was not touched (verified unchanged, paragraph by paragraph).

Sources are recorded as Google Docs comments, one anchored to each figure or table caption, each containing clickable links to the GitHub files at commit ca13132 (per Aaron's follow-up instruction, the caption text itself carries no source lines):

- Figure 1: `biomaf/README.md`; `results/tables/main_manuscript_complete_panel_feature_manifest.csv`
- Figure 2: `results/tables/main_manuscript_complete_panel_endpoint_results.csv`; `results/tables/muat_style_tcga_comparator_cancer_type_top20_faithful_fixed_arch_endpoint_results.csv` (published MuAt bar noted as from reference 1, not the repo)
- Figure 3: `results/tables/main_manuscript_complete_panel_oof_predictions.csv`
- Figure 4: `results/tables/main_manuscript_complete_panel_endpoint_results.csv`
- Figure 5: `results/tables/main_manuscript_complete_panel_endpoint_results.csv`
- Figure 6: `results/tables/feature_selection_cancer_type_top20_shap_values.csv`
- Figure 7: `results/tables/muat_style_tcga_comparator_hrd_faithful_fixed_arch_oof_predictions.csv`; `results/tables/hrd_muat_paired_tests.json`; `results/tables/main_manuscript_complete_panel_fold_metrics.csv`
- Figure 8: `results/tables/survival_cancer_type_baseline_summary.csv`; `results/tables/survival_cancer_type_baseline_comparisons.csv`
- Table 1: `results/tables/main_manuscript_complete_panel_fold_metrics.csv`; the two faithful MuAt OOF files; `results/tables/hrd_muat_paired_tests.json`

Number and reference corrections applied (items 1 to 4 from the mismatch list): the survival cross-reference now reads Figure 8; the centre-held-out sentence now reads "all five evaluated cancer types"; the missing HRD intervals were filled in as 0.831-0.889 (threshold 24) and 0.895-0.955 (threshold 42); the Abstract package line now reads "(biomaf, version 1.0.0)". Item 5 ("q=0.0") was left unchanged on purpose: the correct adjusted value is not stored in the repository, and replacing it would mean inventing a bound. Items 6 to 12 still need a source the repo does not contain.

One caption wording point left for the authors: the Figure 6 caption and the Results text describe the SHAP analysis as the Sig+MAF model, while its source file comes from the Bio MAF feature-selection run and the Discussion calls it a Bio MAF-only model. The shares themselves match the file; the label should be settled by the authors.

## Figure images checked against the referenced files, 2026-10-25 (evening)

The earlier pass verified the numbers in the text, captions, and Table 1. This pass extracted the actual chart images from the manuscript and compared their plotted values with the files named in the caption comments.

- Figure 1: DOES NOT MATCH. The image shows three candidate feature views, including a "Channel KME" view (UGA channel contexts summarized by kernel means) that is not one of the four benchmarked representations and appears in no referenced file; burden is absent; the combined view is labelled "Signatures + Bio MAF v4" (stale v4 naming). The referenced README and feature manifest describe only the Bio MAF construction.
- Figure 2: PARTIAL MISMATCH. All twelve bars for burden, spectra, Bio MAF, and Sig+MAF match the pooled values in `main_manuscript_complete_panel_endpoint_results.csv` exactly, and the published MuAt bars match the text (0.641 / 0.612 / 0.906, external). The MuAt-compatible bars (0.570 accuracy, 0.550 balanced, 0.871 top-5) do NOT come from the referenced faithful fixed-architecture file (0.549 / 0.528 / 0.861, the values in Table 1). They match `muat_style_tcga_comparator_cancer_type_top20_comparable_endpoint_results.csv`, the architecture-search run.
- Figure 3: MATCHES. All 20 per-type gains and all 20 per-type sample sizes recomputed from `main_manuscript_complete_panel_oof_predictions.csv` agree with the chart.
- Figure 4: VALUES MATCH, CAPTION ESTIMATOR WRONG. The bars agree with the referenced files, but for TCGA they are fold means (the image subtitle itself says "mean across five test folds"; Kucab rows are pooled), while the manuscript caption says plotted values are pooled out-of-fold estimates. The image's internal title also reads "Figure 2", a stale number.
- Figure 5: VALUES MATCH, CAPTION ESTIMATOR WRONG. Same pattern: HRD score 0.777 and HRD33 0.911 in the chart are the fold means; the pooled values in the referenced endpoint file are 0.767 and 0.901. The caption's "pooled out-of-fold" wording is backwards for the TCGA bars. Internal title reads "Figure 4". The Kucab green bar is the event-token stack; the legend discloses this (2,980 event tokens), and its values (0.293 / 0.174) match the repo.
- Figure 6: LEFT PANEL MATCHES the referenced SHAP file for the shares the manuscript cites (23.8 / 17.3 / 10.9). The figure also contains a second panel, overall-survival SHAP, which the caption never mentions; it comes from `results/tables/feature_selection_OS_shap_values.csv`, a file named neither in the caption nor in the audit comment (the comment should gain that link).
- Figure 7: DOES NOT MATCH. The MuAt-compatible bar for HRD score (about 0.52) and both difference panels (-0.249 and -0.160) match the superseded ab_c_full MuAt configuration, not the referenced faithful run (0.507 Spearman, 0.746 AUROC, differences -0.270 and -0.165). The note inside `hrd_muat_paired_tests.json` itself records that the ab_c_full paired test (0.527 / 0.751, diff 0.249 / 0.160) was superseded by the faithful run and states the faithful values are the ones displayed in the manuscript; that is true for Table 1 and the Results text but false for this figure, which was not regenerated.
- Figure 8: MATCHES. 0.742 / 0.638 / 0.743 and both plotted differences agree with the referenced survival files. Bars are labelled "Bio MAF v4" (stale name only).
- Table 1: MATCHES. All 33 cells equal the fold means in the referenced fold metrics file; the MuAt-compatible column equals the faithful run.

## Regeneration of the outdated figures, 2026-10-25 (evening, at Aaron's direction)

All seven defective figures were regenerated from the referenced repo files (script and extracted values in `~/workspace/scratch/uga-maf-audit/figregen/`) and replaced in the Google Doc; the Doc was then exported and the new images read back from the export to confirm the values in place.

- Figure 1: new schematic matching the caption and the manifest: one tumour's mutation list feeds four feature tables (Burden 3, Spectra 182, Bio MAF 948, Signatures + Bio MAF 1,130), paired elastic net and XGBoost models, and the endpoints; the MuAt-compatible rebuild is shown as a separate dashed path. No Channel KME, no v4 naming.
- Figure 2: MuAt-compatible bars now from the faithful fixed-architecture run (0.549 / 0.528 / 0.861); the other twelve bars and the published MuAt bars are unchanged in value.
- Figure 4: regenerated with pooled out-of-fold scores from the endpoint file, as its caption states (elastic net HRD rows now use the endpoint file's pooled values, e.g. spectra 0.431; internal title corrected to Figure 4).
- Figure 5: same treatment (HRD score 0.767, HRD33 0.901 for Sig+MAF XGBoost, the pooled values; internal title corrected to Figure 5).
- Figure 6: regenerated from both SHAP files under one block grouping (cancer panel reproduces 23.8 / 17.3 / 16.3 / 10.9 exactly; survival panel 20.6 / 16.2 / 14.0 / 12.3). Caption rewritten to cover both panels and to name the model Bio MAF XGBoost; the Results sentence now says the SHAP analysis describes the Bio MAF model's predictions; a reply on the Figure 6 comment adds the overall-survival SHAP file link.
- Figure 7: regenerated from the faithful run and the paired-tests file: MuAt-compatible 0.507 (HRD) and 0.746 (HRD33), differences -0.270 (CI -0.319 to -0.220) and -0.165 (CI -0.236 to -0.094).
- Figure 8: same values, labels now read Bio MAF (no v4).
- Captions: the sentence "Error bars are 95% confidence intervals." was removed from the Figure 3, 4, and 5 captions, because those charts show no error bars (Figures 2, 7, and 8 do, and their captions keep the sentence).
- Figure 3 and Table 1 were correct and were not touched.
- The new PNGs also live in Aaron's Drive in a folder named "Manuscript figures (regenerated 2026-10-25)"; the folder was briefly shared so Docs could fetch the images and is private again (the Doc keeps its own copies, verified by a second export).

One new observation from the regeneration: in the endpoint file, the pooled elastic-net HRD scores (burden 0.497, spectra 0.431) sit far below their fold means (0.528, 0.508), another instance of the pooled-versus-fold discrepancy family noted earlier for the OOF file. The figures now print the pooled values, as the captions promise, but the gap is large enough that the authors may want to know which estimator the endpoint file's pooled linear HRD scores actually came from.

## Bottom line

The PDF's reported numbers are, with the exceptions listed above, an accurate report of the newest results in the repository: Table 1 and every headline comparison were recomputed from the repo's own fold metrics, OOF files, ablation tables, and external-validation outputs on main @ ca13132 and matched to rounding. The fixes needed are two wrong sentences in the PDF (Figure 6 cross-reference; four-of-five centre claim), one incomplete sentence (missing CI values), one wrong version string (biomaf 4.0.0), sourcing or softening for the six numbers the repo does not contain, and repo housekeeping, above all regenerating or replacing the OOF predictions file so it reproduces Table 1.

## Task 8 addendum (2026-10-25 late): all eight figures rebuilt with the new figure skills

At Aaron's direction every figure was rebuilt for craft only, values frozen from this report and figregen/values.json: Figure 1 with the schematic-figure engine (design mode, flat, full depth, gate green with no FAIL), Figures 2-8 with the nature-figure system (written contract per figure, panel-alignment gate PASS, validate READY, PDF collision audit PASS on all but Figure 4, which closed REVIEW REQUIRED on two warning-level label touches at the difference panel's right edge, accepted rather than shrinking type below the 5 pt floor). One visual system across all eight (Arial, bold lowercase panel labels, burden grey, spectra blue, Bio MAF teal, Sig+MAF dark blue, MuAt orange). Figure 3 was restyled too (values untouched). All eight were swapped into the Google Doc on 2026-10-25 and verified by PDF export: 8 images present at the new dimensions, Figure 1 read back in place on page 3. New PNGs are in the Drive folder "Manuscript figures (regenerated 2026-10-25)" as figureN-v2.png; sources and QA outputs in ~/workspace/scratch/uga-maf-audit/figregen2/. Note: the Drive folder's temporary anyone-with-link sharing could not be removed on 2026-10-25 because the permission-delete call kept dying with service restarts; it holds only figure PNGs and still needs unsharing.
