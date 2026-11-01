# Claim audit (G2)

Verdicts rest on `results_ledger.csv`. A claim is **reproduced** when this
package recomputes its number from the named source file and it matches at
printed precision; **sourced_not_rerun** when the number is read from a
named source but not recomputed; **unsourced** when no file in the
repository holds it. Unsourced numbers are never dropped silently: they
are listed first, in full.

## Unsourced numbers (complete list)

Sixteen manuscript numbers have no source file in the repository
(ledger family UNS):

| ID | Location | Value | What exists instead |
|---|---|---|---|
| UNS-01 | Results, secondary contrasts | p = 0.058 (HRD33 raw) | the +0.037 gain is reproduced (MS-TEXT-HRD33-GAIN) |
| UNS-02 | Results, secondary contrasts | q = 0.068 (HRD33 BH) | the older pooled q = 0.109 is stored (RPT-AUDIT-31) |
| UNS-03 | Results, secondary contrasts | q = 0.0026 (HRD score BH) | the +0.082 gain is reproduced (MS-TEXT-HRD-GAIN) |
| UNS-04 | Results, primary cancer contrast | q = 0.0 as printed | no adjusted value is stored anywhere; the script that would produce it writes a file that does not exist in the repo |
| UNS-05 | Results, primary cancer contrast | CI lower 0.067 | the +0.076 difference is reproduced (MS-TEXT-SIGMAF-VS-MAF-DIFF) |
| UNS-06 | Results, primary cancer contrast | CI upper 0.085 | same as UNS-05 |
| UNS-07 | Results, PCAWG PRAD | rescaled recall 0.417 | the starting recall 0.033 is reproduced (MS-PCAWG-RECALL-PRAD) |
| UNS-08 | Results, annotation missingness | SIFT missing 55% (PCAWG) | no file stores the share |
| UNS-09 | Results, annotation missingness | SIFT missing 9% (TCGA) | no file stores the share |
| UNS-10 | Results, annotation missingness | PolyPhen missing 43% (PCAWG) | no file stores the share |
| UNS-11 | Results, annotation missingness | PolyPhen missing 2% (TCGA) | no file stores the share |
| UNS-12 | Results, HRD42 interval | bootstrap CI lower 0.902 | stored bootstrap interval is 0.871 to 0.920 (RPT-AUDIT-06/07); fold t-interval is 0.895 to 0.955 (MS-T1CI-hrd_binary_42) |
| UNS-13 | Results, HRD42 interval | bootstrap CI upper 0.946 | same as UNS-12 |
| UNS-14 | Data availability | Zenodo DOI 10.5281/zenodo.23122728 | the string appears nowhere in the repository |
| UNS-15 | Methods, MuAt runtime | 18.5 h (cancer) | no timing log exists |
| UNS-16 | Methods, MuAt runtime | 3.0 h (HRD) | no timing log exists |

## Naming rules (crosswalk), enforced

- Per-gene counts (R3) are **in the style of CPEM**, never "CPEM". CPEM
  is a full pipeline (gene counts plus copy-number ratios, mutation
  rates, spectra, and a random-forest/deep-network ensemble); this
  package benchmarks the count encoding only. Verdict: the ledger, the
  feature dictionary, and the crosswalk all use the "in the style of"
  form. No claim compares against CPEM's published accuracy.
- TF-IDF (R5) is the **inference-safe substitute** for the tf-rf family
  of Simsek et al. 2019, never their winning method. Their rf weight is
  defined per true class, so the winning BM25-tf-rf representation
  cannot score an unlabeled patient as published; no faithful
  implementation exists. Verdict: the package never presents R5 as the
  Simsek method and never compares scores against their headline
  numbers.
- The **Kucab event stack is never called Bio MAF**. Kucab's release has
  no VEP annotation, so Bio MAF cannot be constructed there. Verdict:
  ledger rows MS-KUCAB-* name the event stack as an event stack; the
  endpoint file's `MAF_stack_only` label for Kucab is a file convention
  and is flagged in the row notes wherever it appears.

## Verdicts by claim

### Extension grid (49 cells)
Every cell's five fold scores and primary score reproduce from the saved
predictions, hash-pinned (ledger EXT-CELL-*, 294 rows). The trial tables
replay exactly under the runner's own selection rule for all 49 cells
(verify section 5). The feature matrices rebuild exactly from tracked
inputs. The canonical splits regenerate exactly from the main OOF file.
**Verdict: supported.**

### A8 primary contrasts
All 20 primary scores, 16 paired differences, and 32 bootstrap interval
bounds reproduce (ledger EXT-A8-*). The bootstrap replicates the
reference random stream (one generator, seed 20261025); interval bounds
are checked at a stated Monte Carlo tolerance of plus or minus 0.002 and
land far inside it. The headline reading stands: Sig+MAF's advantage is
learner- and task-specific (cancer typing with XGBoost), TF-IDF leads
with elastic net on cancer, raw panel counts are statistically
indistinguishable from Sig+MAF there, and HRD with XGBoost is a tie at
the top. **Verdict: supported.**

### Manuscript Table 1
All 44 representation cells reproduce as fold means from
`main_manuscript_complete_panel_fold_metrics.csv`, and the three MuAt
cells reproduce from the faithful run's OOF files (MS-T1-*, MS-MUAT-*).
The printed Sig+MAF XGBoost intervals reproduce as t-intervals from the
five fold scores, including the Doc-corrected HRD24 and HRD42 intervals.
**Verdict: supported in full.**

Two presentation gaps inside otherwise reproduced contrasts:
- The cancer paired difference prints CI 0.197 to 0.222 and p<0.000001.
  Recomputation from the saved folds gives 0.196 to 0.222 and
  p = 1.5e-06. The difference itself (+0.209) reproduces exactly. The
  printed lower bound is one unit high in the third decimal and the
  printed p bound is marginally stronger than the recomputation
  supports (ledger MS-PAIR-CANCER-*).
- The MuAt cancer interval prints an upper bound of 0.543; the
  t-interval from the faithful OOF folds gives 0.5425, stored as 0.542
  (MS-MUAT-CANCER-BACI-HI).

### Ablations
All seven printed ablation values reproduce from
`task1_gene_knowledge_ablations.csv` (MS-ABL-*), including the claim
that removing all curated-knowledge columns moves balanced accuracy by
0.003. Caveat carried from the audit: control refits reuse the full
run's winner hyperparameters. **Verdict: supported.**

### Transfer: PCAWG and POG570
PCAWG Sig+MAF accuracy 0.614, balanced accuracy 0.557, and all nine
per-type recalls reproduce from the saved predictions; the spectra-only
accuracy 0.479, balanced accuracy 0.365, and the BRCA and KIRC recalls
reproduce from the frozen spectra predictions file (MS-PCAWG-*). The
accuracy interval is stored, not rerun. POG570 breast (n=147) Spearman
0.667, Pearson 0.670, R2 -0.225, and specificity 0.034 reproduce
(MS-POG-*). Both cohorts are previously examined, not untouched
validation. The PRAD rescaled recall 0.417 is unsourced (UNS-07).
**Verdict: supported except UNS-07.**

### Centre-held-out
Sig+MAF leads spectra on all five types; the ten pooled recalls
reproduce (MS-CENTRE-*). Two dual-aliquot patients are double-weighted
in the KIRC and KIRP centre counts, a weighting artefact documented in
the audit report, not leakage. **Verdict: supported.**

### Survival
PFI fold means reproduce (0.556, 0.570, 0.621, 0.631). The covariate run
reproduces: cancer type 0.742, Bio MAF 0.638, combined 0.743, gain
+0.0013, p = 0.512 (MS-PFI-*, MS-SURVCOV-*). **Verdict: supported.**

### HRD controls
Repair-gene removal (0.760 to 0.763, p = 0.622; HRD33 p = 0.637) and VAF
removal (HRD33 to 0.874, p stored as 0.000, printed p<0.001) reproduce
from the shortcut ablation files (MS-HRDCTL-*). SBS3-alone and
BRCA1/2-alone values reproduce from the attribution baselines file
(MS-T3-*), and their patient counts reproduce from raw files: 9 of 772
patients with nonzero SBS3 activity, 14 of 772 with a BRCA1/2 indicator
(EXT-SBS3-*, EXT-BRCA-*). **Verdict: supported.**

### Figures 3 and 6
Figure 3's twenty per-type gains and sample sizes reproduce from the
main OOF file (MS-FIG3-*). Figure 6's text shares reproduce from the
SHAP CSV under the manuscript's nine-group grouping (MS-FIG6-TEXT-*):
consequence 23.8, per-gene 17.3, functional-effect 16.3, impact 10.9,
OncoKB roles 9.1, allele fraction 7.8, driver tiers 6.1, hotspots 4.3,
mutation type/count 4.4, summing to 100.0. Stated openly: the text
grouping equals the figure's block grouping with two merges (residual
annotations into the OncoKB group, the remainder into mutation
type/count); that merge definition is inferred from the printed values
as the finest partition consistent with them. No single feature exceeds
6% (largest 5.85%). **Verdict: supported, with the grouping inference
disclosed.**

### Kucab
Pooled spectra 0.331 and 0.250 and the event-stack 0.174 reproduce from
the endpoint file (MS-KUCAB-*). No Kucab row appears in Table 1, which
is correct: Bio MAF was not evaluated on Kucab. **Verdict: supported,
under the naming rule above.**

### Extension reports and crosswalk numbers
The 354 numbers extracted from the deviations log, protocol, status
table, integration check, and literature crosswalk are ledgered as
sourced_not_rerun (RPT-*, XW-*): they are protocol settings, measured
timings, and the published values of Simsek et al., Lee et al., and
Sanjaya et al. (MuAt), recorded so every number in the reports has a
ledger row. Conflicts inside the reports are preserved as printed and
noted in the rows: a 46-cell queue figure versus 49 completed cells, and
the protocol's 10,000-resample prescription versus the 2,000 resamples
the A8 contrasts actually used (the ledger's reproduced A8 rows use
2,000, matching the computed table). **Verdict: recorded faithfully;
nothing in this family is claimed as recomputed.**

## Known discrepancies (expected findings, not failures)

1. **Main OOF file vs fold and endpoint tables.** Recomputing pooled
   scores from `main_manuscript_complete_panel_oof_predictions.csv`
   disagrees with the endpoint file for 56 of 96 groups by more than
   0.002 (44 by more than 0.005, 31 by more than 0.01). Worst three:
   cancer Sig+MAF elastic net, endpoint 0.6723 vs OOF 0.5858
   (difference 0.0865); HRD42 Sig+MAF linear AUROC, 0.8245 vs 0.9053
   (0.0808); HRD33 Bio MAF linear AUROC, 0.7691 vs 0.8304 (0.0613). The
   manuscript's printed values check against the fold and endpoint
   files, which are the ledger sources; the OOF file appears to come
   from a different (likely earlier) run for the affected groups.
2. **Results summary SHAP table.** `docs/results_summary_2026-10.md`
   prints 21.9, 19.9, 15.2 for the top SHAP groups; its own CSV gives
   23.8, 17.3, 10.9, which matches the manuscript. The document is
   wrong; the CSV is authoritative.
3. **Stale fold-differences JSON.** `cancer_typing_primary_fold_differences.json`
   stores a mean difference of 0.1877 from a superseded MuAt pairing.
   The faithful pairing recomputes to +0.2094.
4. **Plot-data manifest.** `plot_data_manifest.json` references
   `canonical/main_panel_pairwise_tests.csv` and
   `canonical/main_panel_results.csv`; the `canonical/` directory does
   not exist in the repository.

## Spot check (bounded refit)

Four folds refit end to end from the saved trial tables, one per
task/learner combination. On every fold, the FINAL configuration the
runner's selection rule picks is identical to the banked FINAL record,
and the refit trial inner scores match the trial table (exactly 0
deviation on three folds, 5.6e-17 on the fourth). Prediction agreement
with the banked files (maximum absolute deviation):

| Fold | Predictions | Probabilities | Inner scores |
|---|---|---|---|
| hrd spectra+maf_stack elasticnet, fold 1 (October proof cell) | 3.6e-15 | n/a | 0 |
| hrd spectra+maf_stack xgboost, fold 1 | 3.8e-06 | n/a | 0 |
| cancer spectra+maf_stack xgboost, fold 5 | 0 | 0 | 0 |
| cancer spectra elasticnet, fold 2 | 0 | 0 | 5.6e-17 |

The shipped pass bound is 1e-5 for predictions (class labels are
integers, so the bound still requires identical labels) and 1e-9 for
inner scores. The two nonzero deviations are floating-point effects in
the final fit only; the tuning layer reproduces bit-exactly, and no
fold score moves at printed precision. Byte identity belongs to the
October kill-and-resume proof of the checkpoint machinery (same
process resumed), not to a cold refit, and this audit does not claim
it for the spot check. No spot check exists for the main panel; its
pipeline kept no persisted configurations.
