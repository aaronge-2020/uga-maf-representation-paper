# Manuscript numbers triage: what Aaron reviews by hand, and what he does not

Built 2026-11-01 from three sources: the current manuscript draft, the story and revision plan of the same date (the pitch sentence and its three pillars: decomposition, shallow versus token-neural, external validation as a designed experiment), and the reproducibility ledger (`results_ledger.csv`, 999 rows: 594 reproduced, 389 sourced_not_rerun, 16 unsourced).

## The headline finding

No story-critical number is unsourced. Every number in Tier A below is marked reproduced in the ledger, recomputed from a named file in the GitHub repository. Aaron's manual review is therefore a check that the printed number matches that one named file, not a hunt through the repo.

Out of 999 ledger rows, Aaron needs to review about 93 atomic values by hand. They sit in 13 groups, each checked against one or two named files. Everything else either reproduces on its own and carries no story weight, belongs in the supplement, or should leave the main text.

One rule for the manual check: for the Table 1 spine, check against the fold metrics file named below, not against the main out-of-fold predictions file. The saved out-of-fold file disagrees with the fold and endpoint tables for 56 of 96 groups (known discrepancy 1 in the claim audit). The printed Table 1 values match the fold file. That discrepancy still needs an answer from Aaron or Vijay before submission, but it does not change which file is the source for the printed numbers.

## Tier A: belongs in the main manuscript, review by hand

Each group states the claim it carries, the values, and the one GitHub file to check them in.

### A1. Table 1 spine, XGBoost rows (the benchmark itself)
Claim: Sig+MAF leads on typing and HRD in the main panel, and the size of each lead.
Source: `results/tables/main_manuscript_complete_panel_fold_metrics.csv`, plus the two MuAt files in A5.

- Cancer type, balanced accuracy: Burden is not the story; the four that matter are Spectra 0.538, Bio MAF 0.662, Sig+MAF 0.738, MuAt 0.528.
- HRD score, Spearman: Spectra 0.694, Bio MAF 0.739, Sig+MAF 0.777, MuAt 0.507.
- HRD33, AUROC: Spectra 0.874, Bio MAF 0.873, Sig+MAF 0.911, MuAt 0.746.
- HRD42, AUROC: Spectra 0.915, Bio MAF 0.880, Sig+MAF 0.925.

15 values. If any one of these is wrong, a headline sentence is wrong.

### A2. New Table 2, the grid contrasts (the decomposition at encoding level)
Claim: annotation pays only for cancer typing with boosted trees; it ties simpler encodings on HRD and loses to TF-IDF under a linear model on cancer.
Source: the extension predictions on branch `analysis/representation-extension`, collected in `files/primary_contrasts.md` (goal workspace) and ledger rows EXT-A8.

- Cancer, XGBoost: Sig+MAF 0.7565; TF-IDF 0.6742; binary 0.6725; counts 0.6711; consequence 0.6678. Key difference: best simple encoding trails by 0.082.
- Cancer, elastic net: TF-IDF 0.6483; Sig+MAF 0.6215; counts 0.6186; consequence 0.6129; binary 0.5847. Key difference: TF-IDF leads by 0.027.
- HRD, XGBoost: TF-IDF 0.7651; Sig+MAF 0.7610; counts 0.7599; consequence 0.7502; binary 0.7466. Key reading: three-way tie at the top.
- HRD, elastic net: binary 0.6184; Sig+MAF 0.5952; TF-IDF 0.5608; consequence 0.5222; counts 0.5166. Key reading: weak across the board, binary lead not established.

20 scores plus the 4 key differences above. Note in the table note: this benchmark uses its own splits and tuning, so its Sig+MAF cancer score (0.7565) is not the Table 1 score (0.738). Never mix the two in one comparison.

### A3. Component decomposition, ablations (what does the work inside Bio MAF)
Claim: gene identity does the work; curated knowledge does almost nothing.
Source: `results/tables/task1_gene_knowledge_ablations.csv`.

- Bio MAF full 0.662; without gene identity 0.540; without curated knowledge 0.659.
- Sig+MAF full 0.738; without gene identity 0.677.
- Derived: removing curated knowledge changes accuracy by 0.003.

6 values. The two archived-settings controls in the same file, binary gene 0.654 and driver count 0.614, are NOT in this tier: the story plan replaces that Discussion paragraph with the A2 grid numbers, so those two values leave the main text and need no review.

### A4. HRD controls (the known mechanism does not carry the signal; allele fraction does)
Claims: removing the repair genes changes nothing; removing allele-fraction summaries costs real HRD performance; the named HRD proxies alone sit at random.
Sources: `results/tables/hrd_shortcut_ablation_summary.csv` and its comparisons file; `results/tables/task3_hrd_attribution_baselines.csv`.

- Repair-gene removal, HRD score: 0.760 to 0.763, p = 0.622.
- Allele-fraction removal, HRD33: 0.900 to 0.874, p < 0.001.
- SBS3 alone, HRD33 AUROC 0.501; BRCA1/2 alone, HRD33 AUROC 0.507.
- Sparsity counts that bound the claim: 9 of 772 tumors with nonzero SBS3, 14 of 772 with a BRCA1/2 indicator.

8 values plus 2 counts. The counts matter: without them the at-random claim reads stronger than the data support.

### A5. MuAt margins (shallow versus token-neural pillar)
Claim: the tabular pipeline beats both MuAt rebuilds on identical splits, for typing and HRD.
Source: `results/tables/muat_style_tcga_comparator_cancer_type_top20_faithful_fixed_arch_oof_predictions.csv` and the matching HRD file.

- Cancer balanced accuracy difference: +0.209 (Sig+MAF 0.738 vs MuAt 0.528; points already in A1).
- HRD score Spearman difference: +0.270 (0.777 vs 0.507).
- HRD33 AUROC difference: +0.165 (0.911 vs 0.746).

3 differences. The claim stays scoped to the rebuilds; the architecture-search rebuild (0.590) is the fairer comparator and is a Tier B number below.

### A6. PCAWG transfer headline (external validation pillar)
Claim: the typing advantage transfers to a frozen external cohort overall, and fails in one named type.
Sources: `external_validation/pcawg_transfer/predictions/pcawg_sigmaf_predictions.csv`; `results/external_validation/aaron_email/pcawg/pcawg_frozen_model_predictions.csv`.

- Sig+MAF accuracy 0.614 vs spectra alone 0.479.
- Sig+MAF balanced accuracy 0.557 vs spectra alone 0.365.
- PRAD recall 0.03, the named failure.

5 values. Cohort size 913 (ICGC samples) frames them; check it once in the same files.

### A7. POG570 transfer headline (ranking holds, calibration fails)
Claim: transfer keeps the ranking and loses the calibration.
Source: `results/external_validation/aaron_email/pog570/pog570_frozen_model_predictions.csv`.

- Spearman 0.667; R2 -0.225; specificity 0.034 at the original HRD cutoff of 24.
- Subset size: breast, n = 147 of 570.

3 values plus the subset size.

### A8. Centre holdout (transfer inside TCGA)
Claim: Sig+MAF beats spectra on all five tested cancer types.
Source: `results/tables/task4_centre_heldout.csv`.

- Pooled recall, Sig+MAF vs spectra: OV 0.64 vs 0.40; COAD 0.57 vs 0.13; KIRC 0.73 vs 0.46; KIRP 0.43 vs 0.22; CESC 0.12 vs 0.10.

10 values, one file, one claim. The claim fails if any pair flips, which is why all ten stay in this tier.

### A9. Survival null (the boundary of the map)
Claim: mutation features add nothing beyond cancer type.
Sources: `results/tables/survival_cancer_type_baseline_summary.csv` and its comparisons file.

- Cancer type alone 0.742; Bio MAF alone 0.638; cancer type plus Bio MAF 0.743.
- Gain from adding Bio MAF: +0.0013, p = 0.512.

5 values.

### A10. HRD gains over spectra in the main panel (point values only)
Claim: adding Bio MAF to spectra improves the continuous HRD score and HRD33 in the main panel.
Source: `results/tables/main_manuscript_complete_panel_fold_metrics.csv`.

- HRD score gain +0.082; HRD33 gain +0.037.

2 values. Their p and q values are unsourced; see the decision list. The story needs the gains, not the q values, because A2 already tempers the HRD claim.

### A11. Kucab reversal (the ranking reverses outside cancer)
Claim: on damage classes, spectra alone beat the event representation.
Source: `results/tables/main_manuscript_complete_panel_endpoint_results.csv`.

- Spectra pooled: 0.331 (elastic net) and 0.250 (XGBoost); event stack 0.174 (XGBoost).

3 values. Naming rule: the 0.174 arm is a Kucab event stack, never Bio MAF.

### A12. SHAP headline (what the model leans on, and its limit)
Claim: consequence summaries are the largest feature group, and no single feature dominates.
Source: `results/tables/feature_selection_cancer_type_top20_shap_values.csv`.

- Mutation consequence group share 23.8%.
- Largest single feature share 5.85% (printed as: no single feature exceeds 6%).

2 values. The full nine-group breakdown is Tier B.

### A13. Framing counts (every score above rests on these)
Check once, in Methods and the feature dictionary:

- Cancer typing n = 8,800; HRD cohort n = 772; Bio MAF feature count 948.

3 values. A wrong cohort size or feature count undermines every table at once, which is why these sit in Tier A despite being Methods numbers.

**Tier A total: about 93 values in 13 groups, checked against 17 source files.** At a few minutes per file, this is about an hour of manual review.

## Tier B: belongs in the main manuscript, no manual review

These numbers stay in the paper, and the reproduction package already recomputes them (227 of the 231 manuscript ledger rows are reproduced; the other 4 are stored values read from a named file). A third-decimal error in any one of them changes no sentence in the story, so Aaron's time buys nothing here.

- The rest of Table 1: all elastic net (linear) rows, the Burden column, the HRD24 row, and the overall survival and PFI rows (0.556, 0.570, 0.621, 0.631).
- All confidence intervals except where a Tier A claim quotes one: the Table 1 t-intervals, the MuAt intervals, the PCAWG accuracy interval 0.583 to 0.645 (sourced, not rerun).
- Figure 3: the 20 per-type recall gains and 20 sample sizes. The figure's claim (gains are broad across types) is carried by the pattern, not by any single type.
- Figure 6: the full SHAP group breakdown beyond A12 (per-gene 17.3%, functional effect 16.3%, impact 10.9%, OncoKB roles 9.1%, allele fraction 7.8%, driver tiers 6.1%, hotspots 4.3%, mutation type and count 4.4%), and the survival SHAP panel.
- Secondary per-type PCAWG recalls beyond PRAD (BRCA 0.85, LIHC 0.83, KIRC 0.82, ESCA 0.77, SKCM 0.73, OV 0.67, HNSC 0.15, STAD 0.16), the POG570 Pearson value 0.670, and the HRD34 and HRD42 forms of the SBS3 and BRCA controls (0.510, 0.524, and the two negative Spearman values).
- The architecture-search MuAt rebuild score 0.590, quoted once as the fairer comparator.
- The two all-genes grid values quoted in one sentence (0.6704 and 0.6819, against panel 0.6711 and 0.6725): they support a no-added-value sentence and are reproduced from the extension predictions.
- Secondary p values that sit beside Tier A point values (for example HRD33 repair removal p = 0.637, the MuAt paired p values).

## Tier C: does not belong in the main manuscript, no review at all

- The full 49-cell grid beyond Table 2: 29 remaining primary scores, all 245 per-fold scores, and all trial tables (ledger EXT family beyond the A8 rows, about 299 rows). One supplementary table, generated from the ledger, with the protocol stated above it. Nobody checks per-fold values by hand; the package recomputes every one.
- Every per-fold score in the main panel, including the 10 MuAt per-fold values in the ledger.
- The reports and crosswalk families (RPT and XW, 385 ledger rows): protocol settings, timings, and the published numbers of Simsek et al., Lee et al. (CPEM), and the MuAt paper. These are other papers' numbers and this project's settings. Their check is against the original papers, which the literature crosswalk already did; they are not GitHub results and Aaron gains nothing by re-checking them in the repo.
- Figure 4 (mutation count versus patterns baselines): moves to the supplement per the story plan; its content is already in the Burden and Spectra columns of Table 1.
- Feature-selection detail: the count of features retained by SHAP selection and the held-out score of the selected set. Methods or supplement detail; no story sentence rests on them.

## The 16 unsourced numbers, decided by tier

Rule applied: an unsourced number needs sourcing work only if a Tier A claim rests on it. None does. The list below sorts all 16 (claim audit IDs UNS-01 to UNS-16).

**Cut, no replacement needed (11):**
- UNS-01 and UNS-02: HRD33 secondary contrast p = 0.058 and q = 0.068. The gain (+0.037, A10) is reproduced; at p = 0.058 the test adds nothing.
- UNS-04, UNS-05, UNS-06: cancer contrast q = 0.0 as printed, and the +0.076 contrast interval 0.067 to 0.085. Keep the +0.076 point value (reproduced, from the endpoint file); drop its q and interval. A q printed as 0.0 cannot stay in any case.
- UNS-07: PRAD rescaled recall 0.417. A post-hoc exploratory check. The story number is the raw recall 0.03 (A6).
- UNS-12 and UNS-13: HRD42 bootstrap interval 0.902 to 0.946 in the Discussion. It conflicts with both stored intervals (bootstrap 0.871 to 0.920; Table 1 t-interval 0.895 to 0.955). The point value 0.925 (A1) and the Table 1 interval stay; this duplicate goes.
- UNS-15 and UNS-16: MuAt runtimes 18.5 h and 3.0 h in the Table 1 note. No timing log exists, and runtime carries no story claim.

**Optional, only if Aaron wants the sentence at full strength (4):**
- UNS-08 to UNS-11: annotation missingness, SIFT 55% in PCAWG and 9% in TCGA, PolyPhen 43% and 2%. These support one explanatory sentence about why PCAWG annotation degrades. They are not a headline. Either compute the four shares once from the annotation files and add that file to the repo, or soften the sentence to a qualitative statement with no percentages. No other number depends on this choice.

**Verify outside GitHub, not a results review (1):**
- UNS-14: Zenodo DOI 10.5281/zenodo.23122728 in the availability lines. It appears nowhere in the repo because it cannot be checked there. Before submission, confirm the DOI resolves to this paper's record on Zenodo, or replace it with the real one.

**One that needs a decision, not sourcing work:**
- UNS-03: HRD score q = 0.0026, attached to the reproduced +0.082 gain (A10). Options: drop the q and report the gain alone (recommended, because the A2 grid tie already tempers the HRD claim, so the paper does not need this q at full strength), or ask Vijay whether a stored test file exists under another name. Do not spend manual review time hunting for it in the repo; the claim audit already established that no file in the repo holds it.

## What this means for Aaron's review session

1. Open the 17 Tier A source files, check the 13 groups, done. About an hour.
2. Spend no time on Tier B or Tier C numbers; the package's recomputation is the check for those.
3. Make the two unsourced decisions: cut the 11, and choose on the missingness percentages (compute once, or soften the sentence).
4. Separately, and not part of number review: the main out-of-fold discrepancy (56 of 96 groups) needs an answer before submission, because a reviewer who downloads that file will find it. It does not block items 1 to 3, since no Tier A number is sourced from that file.
