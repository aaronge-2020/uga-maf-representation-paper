# Submission readiness: what stands between the current draft and submission

Read against the live Google Doc on 2026-11-01 (export byte-identical to the October figure-swap version, so the draft has not changed since the audit), the 17-page supplementary file (last modified 2026-10-12), the reproducibility ledger and claim audit, the numbers triage, and the story and revision plan. Target journal: Oxford Bioinformatics.

Verdict: not ready. The science is done and verified; what remains is one story decision, a set of text fixes that follow from it, the unsourced-number cuts, two repository problems, and author sign-off items. Nothing below needs a new analysis except the optional items marked as such.

## 1. The one decision that gates most of the work (Aaron, Vijay, Cindy)

The completed 49-cell grid is not in the manuscript at all. The words TF-IDF and counts panel never appear. As written, the paper still tells the old story:

- The Abstract closes on "the added predictive value of gene specific mutation summaries" as the lead conclusion. The grid shows that value is specific to cancer typing with boosted trees, ties simple counts and TF-IDF on HRD, and reverses under a linear model on cancer.
- The Discussion still carries the archived-settings controls paragraph (binary 0.654, role-matched counts 0.614) with its own caveat that retuning could change the differences. The grid is the retuned answer to that caveat, and it narrows the gap the paragraph claims.
- No copy-number scope sentence, no inference-safety paragraph, no grid protocol paragraph in Methods.

Two ways to submit, and the meeting should pick one:

- **Integrate the grid (recommended in the story plan).** New Table 2, one new Results section, Abstract and Introduction rewritten around the map, Discussion controls paragraph replaced, Figure 4 to the supplement, full grid as a supplementary table. This is the stronger paper and the honest one, because the authors now hold results that qualify the draft's headline claim.
- **Submit the current story.** Defensible only in the narrow sense that the current text caveats its own controls. It leaves the paper exposed: a reviewer who asks for retuned simple-encoding controls is asking for exactly the analysis that already exists and is not in the paper.

The B2 question rides on this choice. The current draft's decomposition claims are modest and caveated, so it can be submitted without B2. The stronger decomposition framing in the story plan needs B2 or the grid integration to support it.

## 2. Text fixes that must happen under either choice

**Unsourced numbers: all 16 are still in the text** (checked line by line today). Per the triage:

- Cut: HRD33 p = 0.058 and q = 0.068 (Results, HRD boundary paragraph); q = 0.0 and the 0.067 to 0.085 interval around the +0.076 cancer contrast (Results, cancer typing); PRAD rescaled recall 0.417 (keep the raw 0.03); the HRD42 bootstrap interval 0.902 to 0.946 in the Discussion (keep the Table 1 interval); MuAt runtimes 18.5 h and 3.0 h in the Table 1 note.
- Decide: the four SIFT and PolyPhen missingness percentages (55, 9, 43, 2) in the PCAWG paragraph. Compute once and deposit the file, or soften the sentence.
- Decide: q = 0.0026 on the +0.082 HRD gain. Drop the q or ask Vijay for its source. The gain itself is reproduced.

**References and pointers:**

- Simsek et al. 2019 and Salvadores et al. 2019 are cited by name in the Introduction and appear nowhere in the 19-item reference list. Add both, and settle one citation style: the draft mixes author-year, bracketed numbers, and superscripts. If the grid is integrated, Lee et al. 2019 (CPEM) must be cited and added as well.
- The Methods cite "Supplementary Table S7" for the MuAt model's per-mutation inputs. The supplement has Tables S1 to S6 only; S7 is a section (Data provenance), not a table. Add the table or fix the pointer.
- The supplement's Table S4 still labels a row "Signatures + Bio MAF v4", the naming the main text corrected to Sig+MAF and biomaf version 1.0.0. Fix in the supplement.
- The supplement's knowledge-base ablation (910-feature matrix, n = 8,861 center cohort) concludes the Bio MAF advantage "depends on curated prior knowledge", while the main text reports that removing the 32 curated-knowledge columns costs 0.003. Both can be true (the supplement removes 784 columns including per-gene driver features; the main text removes 32 summary columns), but as written a reviewer will read them as contradicting each other. Add one sentence in each place naming exactly which columns each removal covers.

**Availability and the Zenodo claim:**

- The Abstract says feature tables, trained models, and reference files "are archived at Zenodo", and the supplement refers to "the deposited Zenodo matrices". No DOI or link appears anywhere in either document. Either publish the Zenodo record and put its DOI in the Abstract availability line and the Data availability section, or remove both Zenodo claims. A submission cannot claim a deposit it does not identify.
- The reproduction package (ledger, claim audit, reproduce.sh) lives on branch `analysis/representation-extension`. The Data availability section links the repository's main branch. Before submission, merge the branch or cut a release so a reader following the availability link reaches the package.

**Comments and title page:**

- Ten unresolved comments sit on the Doc: the nine audit source comments (working notes, useful until the numbers freeze, then resolve them) and one author comment, "I dont know if the affiliations are correct?" The affiliations question must be answered by the authors and the comment closed; a submission exports cannot carry it.
- Confirm the author block itself: order, equal-contribution and co-senior notes, Almeida's contribution footnote, and Maron as corresponding author, with every author approving the final version. The contributions section already states all authors reviewed and approved; that needs to be true of the revised final text, not the current draft.

## 3. Repository problems to answer before submission

- **The main out-of-fold file.** It disagrees with the fold and endpoint tables in 56 of 96 groups. The printed numbers match the fold files, so the tables stand, but the repo as linked cannot reproduce its own headline tables from its main predictions file. Fix by replacing the file with the one that generated the tables if it exists, or by adding a repo note that names the fold and endpoint files as authoritative and explains the OOF file's provenance. Silence is the one bad option: the claim audit already documents the mismatch, and a reviewer who recomputes will find it.
- **Repo hygiene, three small items** from the same audit: the results summary document whose SHAP table (21.9, 19.9, 15.2) contradicts its own CSV (23.8, 17.3, 10.9, the manuscript's values); the stale fold-differences JSON (0.1877 for a superseded MuAt pairing; the faithful value is 0.2094); the plot-data manifest pointing at a `canonical/` directory that does not exist. Fix or delete each; all three are one-line jobs, and all three sit in the repo a reviewer will browse.

## 4. Aaron's own checks (from the triage)

- The Tier A manual pass: about 93 values in 13 groups against 17 named files, about an hour. This is the only number-checking left; Tier B and C are covered by the verified reproduction package (independently re-verified 2026-11-01: 669 checks, 0 failed; spot check 4 of 4).
- Verify the funding statement (AIM-AHEAD, NIH Agreement 1OT2OD032581) and the competing-interests statement with the senior authors, since both are author attestations, not results.
- One ethics line: the draft has no ethics statement. The data are public and de-identified, which the Data availability section describes; if the journal's form asks, that sentence is the answer, and adding it to the manuscript costs one line.

## 5. Formatting for the journal (last, after the text freezes)

- Rebuild the submission files from the frozen Doc: journal template and reference style for Oxford Bioinformatics, figures as separate files at print resolution (the regenerated set already exists as 300 dpi PNG plus vector PDF), the supplement as one combined file including any new grid table.
- The Abstract runs about 300 words in the current draft and will be rewritten under the story decision in any case; write the new one to the journal's abstract limit rather than trimming this one.
- Cover letter and submission metadata: article type, keywords (present), corresponding author contact (present), and the availability statement matching whatever section 2 above settles.

## Order of work

1. Tonight's meeting: pick the story (section 1), settle the missingness and q = 0.0026 decisions, answer the affiliations question.
2. Text revision per sections 1 and 2, in the Doc.
3. Repository items (section 3) and the Zenodo record, in parallel.
4. Aaron's Tier A pass over the revised numbers; claim audit re-run over the revised text.
5. Author sign-off on the frozen text; formatting and submission files (section 5).
