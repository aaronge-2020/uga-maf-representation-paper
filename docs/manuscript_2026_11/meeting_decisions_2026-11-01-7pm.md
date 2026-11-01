# Vijay and Cindy meeting, Sun Nov 1, 2026, 7:00 to 8:00 PM: decision list
Prepared 2026-11-01 for Aaron's 3:15 PM prep block. Everything below is already documented in the triage, the B2 brief, or the reproducibility package report; this file only puts the decisions that need another person in one place, in the order to raise them.

## What is settled, say it in one minute

- Grid complete, 49 of 49. Headline: Bio MAF's advantage is real but specific; it earns its complexity on cancer typing with XGBoost (Sig+MAF 0.7565, every simple encoding trails by 0.082 to 0.089), ties TF-IDF and counts on HRD XGBoost, and loses to TF-IDF on cancer with elastic net.
- Reproducibility package built and pushed at c2bcbb3: 999 ledger numbers, 594 reproduced, 389 sourced but not rerun, 16 unsourced.
- Aaron's manual review is bounded: about 93 Tier A values in 13 groups, about an hour, each against a named file. No story-critical number is unsourced. Full list: manuscript-numbers-triage-2026-11-01.md in this folder.
- Still not done: the independent fresh-clone verify was killed by a runtime restart mid run (hashes passed, no failures) and relaunched. Do not call the package independently verified until that run finishes.

## Decisions that need Vijay or Cindy tonight

1. **The out-of-fold discrepancy (biggest item).** The saved main out-of-fold predictions file disagrees with the published fold tables in 56 of 96 groups; worst case cancer Sig+MAF elastic net, 0.672 printed vs 0.586 recomputed from that file. The printed values reproduce from the fold files, so the tables stand, but the OOF file cannot be the one that generated them. Ask: does anyone know which run produced the fold tables, or where the matching OOF file is? This needs an answer before submission.
2. **B2: scope the claim or run 7 to 14 cells.** The decomposition premise rests on two protocols: whole encodings in the retuned grid, component removals inside Bio MAF at archived settings. Option 1: scope the premise sentence, state the protocol difference once, add the copy-number sentence (benchmark is mutation calls only; CPEM's copy-number block, 13,040 of 35,565 features, was not tested), run nothing. Option 2: 7 cells (cancer XGBoost) or 14 cells (adding HRD), hours on Aaron's laptop, days on the VM. Recommendation in the brief: Option 1 now, Option 2 only if they or a reviewer press the claim. Full brief: ~/workspace/goals/uga-maf-representation-extension-analysis/files/b2-decision-brief-2026-11-01.md
3. **HRD score q=0.0026 (UNS-03).** No repo file holds it. Decision, not a search: drop the q from the paper, or Vijay supplies its source.
4. **SIFT and PolyPhen missingness (55% and 9%, 43% and 2%).** Compute once from the data, or soften that one sentence. Everything else unsourced (11 items, including both HRD33 p/q values, q=0.0 and its interval, PRAD rescale 0.417, the HRD42 duplicate interval, both MuAt runtimes) is already recommended cut with no replacement; the Zenodo DOI is checked on Zenodo, not in GitHub.
5. **Kucab label.** The Kucab arm ran a MAF-stack-only analogue (2,980 features) because Kucab releases no VEP fields for Bio MAF, but the manuscript reports it as "Bio MAF" for Kucab damage class (for example 0.293 and 0.174 in Table 1). That label has to change before submission; tonight is the cheapest time to agree the replacement wording.

## If time remains

- Confirm no more new analyses: Aaron decided 2026-11-01 that remaining effort goes to fairness and reproducibility of the whole paper, and B2 is the only proposed exception.
- Dinner with Jane at 6:00 PM is before this meeting and untouched; the meeting replaces the free-rest hour.
