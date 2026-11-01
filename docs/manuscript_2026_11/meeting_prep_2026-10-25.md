# Prep for tonight's paper meeting with Vijay Vankadaru and Cindy Liu

Sunday, October 25, 2026, 7:00 to 8:00 pm ET
Meet link: https://meet.google.com/khj-tbtb-jqd
Status: both guests have accepted the calendar invite (Vijay accepted Oct 24 evening ET, his first reply in the whole thread; Cindy accepted overnight).

Paper: the UGA/MAF representation paper.

## Decisions Vijay's own summary says are needed

From `docs/results_summary_2026-10.md` in the uga-maf-representation-paper repo, written by Vijay Vankadaru, dated 4 October 2026, section 12. His recommendations are in the right column.

1. Which MuAt protocol the main text reports for cancer typing. Context: sections 3.1, 3.2. Recommendation: faithful fixed architecture in the main text (verified against the official trainer, no fold diverged); the search protocol goes in the supplement as the ablation it already is.
2. Whether the HRD column also moves to the faithful protocol. Context: section 3.3. Recommendation: yes. Run 12 is clean (0.484 / 0.736, no fold diverged); the whole MuAt column becomes one protocol and the arm C versus main row question disappears.
3. Weight decay wording. Context: section 3.7. Recommendation: "0.001 per the official software; the 2023 value is not recoverable from the published checkpoints."
4. Do not attribute attention-weighted pooling to MuAt in Methods. Context: section 3.5. The paper does not specify pooling; it is the team's own design choice.
5. Pretrained weights: state why they were not used. Context: section 3.7. One sentence: PCAWG whole-genome token distributions do not match TCGA exome.
6. Fold pairing: main-panel outer folds differ across representations (open since August). Either state the limitation, or re-run paired.
7. Metastatic status as a survival confounder (open since 1 August). Methods or Limitations.
8. Refresh the Hugging Face OOF table before submission. Context: section 8. One upload; needs a write token.

## Also in the same summary, relevant to Cindy

Item 8 of the "What is new" list: a fifth frozen model, `HRD_Score__standard_sbs96_id83`, was produced for Cindy's external validation.

## Reviewer-analysis status (separate workstream, from reviewer-analyses-notes.md in the same repo)

- Task 5 (PCAWG audit): done. 913 tumours, 9 types; Sig+MAF gains largest on KIRC (+0.824) and BRCA (+0.769), loses on LIHC (-0.146) and ESCA (-0.083).
- Task 0 (reproduce Table 1): accepted with a GPU caveat. burden_only/linear matches bit for bit; burden_only/xgboost differs by 0.001 because production ran on GPU and the rerun ran on CPU. Root cause was a seed bug: two different functions both named `stable_seed` in nested_oof.py and feature_evaluation.py. Pre-submission recommendation in the notes: rename one or consolidate before submission.
- Task 3 (HRD attribution baselines): accepted as run. Sig3 alone, BRCA1/2 alone, and both together all sit near chance (AUROC about 0.50 to 0.53) next to Sig+MAF at 0.90.
- Tasks 2, 4, 6 were recorded as pending Task 0 in the notes.
