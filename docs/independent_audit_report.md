# Independent Code & Manuscript Audit — UGA/MAF Representation Paper

**Prepared for:** Aaron Ge (and the team)
**Prepared by:** Vijay Vankadaru
**Date:** 2026-06-20
**Repo audited:** `uga-maf-representation-paper` @ `main` (committed run_id `20260604T021234Z`)
**Reference paper for the MuAt comparator:** Sanjaya et al., *Mutation-Attention (MuAt): deep representation learning of somatic mutations for tumour typing and subtyping*, **Genome Medicine 2023** (PMC10326961).

> **Scope / method.** This is a line-by-line read of the entire Python codebase (`src/**`, `scripts/**`), all configs, all docs, the committed result tables, the full MuAt paper, and the manuscript draft. Line numbers are from the current HEAD; treat them as precise-but-verify if the tree moves. **Nothing in the code or manuscript was changed** — this is an evaluation only.

---

## 0. TL;DR

The **code is methodologically sound and largely leakage-safe** — in several respects *more* rigorous than the manuscript claims. The problem is **synchronization and reproducibility**, not (mostly) correctness:

1. **The manuscript and the committed pipeline describe two different benchmarks.** Across all five main endpoints, the manuscript's Table 2 numbers do **not** reproduce from the current code — different representation set, different cancer-type cohort, different survival model, different CV, different feature counts. The "reproduce" command actually *validates the code's design and forbids the manuscript's terminology.*
2. **The statistics as run differ from the manuscript** (bootstrap 500 not 10,000; four comparison families not five; no One-hot-KME family; OS excluded from testing).
3. **The MuAt comparator is a faithful, leakage-clean "MuAt-compatible reimplementation" but not a replication** — and it's honestly labeled as such. Specific architecture/CV deviations are listed in §4.
4. **A handful of engine-level concerns** (high-dimensional "Cox" is a 5-step non-converged approximation; silent exception/warning swallowing) and **manuscript-internal issues** (the reference list and Declarations belong to a different paper).

Severity legend used below: 🔴 blocking for reproducibility/claims · 🟠 should fix before submission · 🟡 disclose/clean up.

---

## 1. CRITICAL — Manuscript ⇄ code divergence

Every cited line was verified against `config/endpoint_registry.yaml`, `config/experiment_settings.strict_no_leakage.yaml`, the committed `results/tables/*`, and the source modules.

| # | Manuscript says | Committed code produces | Evidence (file:line) | Sev |
|---|---|---|---|---|
| 1 | **5 representations**, incl. **One-hot sequence-window KME** (Fig 3 + "KME wins Kucab") | **4 representations** (burden, signatures, MAF stack, sig+MAF); KME **removed** | `config/experiment_settings.strict_no_leakage.yaml:102-106`; `scripts/reproduce_manuscript.py:75-76,87-98` **forbid** the `one_hot_event_kme` token | 🔴 |
| 2 | Cancer type **top-10**, **n=5,462**, macro-AUROC | Cancer type **top-20**, **n=8,800**, 20 classes | `src/utils/endpoint_registry.py:16-40`; `config/endpoint_registry.yaml:43-69`; `scripts/reproduce_manuscript.py:82` **forbids** `top-10`/`top10` | 🔴 |
| 3 | OS = **binary classification**, AUROC, **XGBoost = 0.667** | OS = **Cox PH time-to-event**, Harrell **c-index = 0.633**, **no XGBoost** | `src/utils/endpoint_registry.py:127-160`; `src/runners/run_main_manuscript_complete_panel.py:62,940-941`; `src/utils/make_all_figures.py:3165` (drops `os_event`); `src/utils/check_benchmark_completion.py:166-184,312` (bans `os_event`) | 🔴 |
| 4 | **Single stratified 5-fold** CV, pooled OOF | **Nested** CV: 5 outer folds + inner 60/20 validation split selecting hyperparameters *and* a Bio-MAF feature block | `src/utils/nested_oof.py:1-6` (docstring), `:188-235`; `split_strategy` column = "5 outer KFold; inner 60/20 validation split" | 🟠 |
| 5 | MAF stack **~1,823–2,255** feats; sig+MAF **~2,005–2,516** (Table 3) | MAF stack **948** (MC3) / 2,980 (Kucab); sig+MAF **1,130** / 3,241 | `results/tables/main_manuscript_complete_panel_feature_manifest.csv`; schema in `src/utils/quick_bio_v4_nested_feature_selection.py:112-135` | 🟠 |

**Matches:** burden (1–3), signatures (182 MC3 / 261 Kucab ✓ Table 3's 182–261), Kucab n=259, HRD n=772.

### 1.1 Exact-number reconciliation (manuscript Table 2 vs committed results)

Pulled directly from `results/tables/main_manuscript_complete_panel_endpoint_results.csv`. **No endpoint matches the manuscript exactly** — confirming Table 2 came from an earlier pipeline.

| Endpoint (XGBoost, best rep) | Manuscript | Committed code | Note |
|---|---|---|---|
| Kucab damage class (signatures) | 0.651 | **0.636** | + manuscript's winning "One-hot KME 0.66" rep doesn't exist in code |
| HRD score (sig+MAF, Spearman) | 0.775 | **0.781** | close, not equal |
| HRD33 binary (sig+MAF, AUROC) | 0.902 | **0.867** | all binary-HRD cells lower in code |
| Cancer type (sig+MAF) | 0.958 (top-10) | **0.976** (top-20, macro-AUROC) | different cohort *and* value; main-table metric is balanced-accuracy (§3/§8) |
| Overall survival (sig+MAF) | 0.667 (AUROC) | **0.633** (Cox c-index) | different model *and* metric |

### 1.2 The reproduction command does not reproduce the paper

`scripts/reproduce_manuscript.py --strict` is a **validator, not a regenerator** — it checksums committed files + the dataset bundle and asserts the committed tables contain required endpoints. It runs **no models** (model regeneration is the separate `src/run_all_experiments.py`). Critically its `FORBIDDEN_PATTERNS` scan **fails the build** if the active tree contains `top-10`/`top10` (`scripts/reproduce_manuscript.py:82`) or `one_hot_event_kme` (`:75-76,87-98`). So the repo *actively rejects the manuscript's two headline elements.* Aaron's note that this command "reproduces the manuscript results" is misleading and should be reworded.

---

## 2. The MuAt comparator (the assigned review)

**Files:** `src/utils/muat_compatible.py`, `src/runners/run_muat_style_tcga_comparator.py`, `docs/muat_replication_rigor.md`.

**Verdict: a credible, leakage-clean "MuAt-compatible" event-bag model — correctly NOT called a MuAt replication.** It is a genuine PyTorch attention network and a fair *same-fold internal* baseline. It cannot reproduce the paper's numbers, and the repo says so.

**Faithful to the MuAt paper (good):**
- Three per-mutation modalities — 3-nt motif, **1-Mb** position bin, genic/exonic/strand annotation: `muat_compatible.py:84-90,104-159`.
- Optimizer/schedule **exactly match the paper**: SGD, lr 6e-4, momentum 0.9, batch size 1, 150 epochs (`config/experiment_settings.muat_top20_comparable.yaml:34-43`; paper §Methods).
- **5,000-mutation cap matches the paper's** O(n²) cap (`muat_compatible.py` `max_events_per_sample=5000`).
- **Dropping SV/MEI is faithful for TCGA exomes** — the paper also excludes SVs for the WES experiment.
- Fixed-hash token dictionaries are **genuinely fold-independent → no vocabulary leakage** (`muat_compatible.py:252-269`), and it reuses the *same canonical folds* as the tabular benchmark (`run_muat_style_tcga_comparator.py:518-579`).

**Deviations from the paper (must be disclosed if compared head-to-head):**
| Deviation | Paper | Code | file:line |
|---|---|---|---|
| Folds | **10-fold** | **5-fold** | `config/experiment_settings.muat_top20_comparable.yaml:33`; CSV `n_folds=5` |
| Fold ensemble | **logit summation** across 10 models | none (independent per-fold OOF) | only an *unexecuted* CLI string in `muat_compatible.py:649-655` |
| Architecture search | embed{128,256,512}×layers{1,2,4}×heads{1,2} | fixed embed=128, **1 layer, 1 head**, no search | `muat_compatible.py:519,525-527`; `run_*:1337` (`candidate_count=1`) |
| Set pooling | attention-weighted | **mean pooling** | `muat_compatible.py:569-571` |
| Primary metric | **top-1 accuracy** (0.641) | **macro-AUROC** (0.920) is the displayed `score`; accuracy reported separately as **0.539** | `run_*:1432`; CSV `score=0.9202`, `accuracy=0.5389` |
| Motif when ref context missing | from FASTA | falls back to `N` flanks | `muat_compatible.py:95-96` |

**Result:** top-1 **0.539** vs paper 0.641; top-5 **0.861** vs 0.906; macro-AUROC 0.920 (no paper analogue — *the paper reports no AUROC*). It also **underperforms** the repo's own tabular sig+MAF+XGBoost (0.744 acc; `docs/muat_replication_rigor.md:7`), which is the actual reviewer-facing point.

**Concerns:**
- 🟠 **Stale fidelity report:** `results/tables/muat_style_tcga_comparator_fidelity_report.md` describes a **60-patient smoke run**, not the full 8,800-patient run (regenerated by `run_muat_style_tcga_comparator.py:248-275` — the committed copy is just out of sync).
- 🟡 Displayed primary metric (macro-AUROC 0.92) sitting next to the paper's 0.641 accuracy invites an apples-to-oranges read; show **accuracy-vs-accuracy** (0.539 vs 0.641).
- 🟡 `max_events` truncation keeps the **first 5,000 by genomic order** (`muat_compatible.py:331`), not a random sample → bias for hypermutators (POLE/MSI). Paper caps but doesn't genomic-order-truncate.
- 🟡 Unweighted `CrossEntropyLoss` (`run_*:1034`) while selection metric is balanced-accuracy on an imbalanced 20-class set.

**Recommendations:** run the 10-fold `muat_full.yaml` **with logit-summation ensembling**, make **top-1 accuracy** the displayed primary, add **attention-weighted pooling** + a small layer/head search, regenerate the fidelity report from the full run, and (as Aaron suggested) **contact the MuAt authors for the official TCGA splits/checkpoints** — the code already has an `OfficialMuAtCLI` hook ready (`muat_compatible.py:422-495`).

---

## 3. Statistics — deviations from the manuscript

**File:** `src/utils/stat_tests.py` (implementations are individually correct) + `src/utils/make_all_figures.py` (wiring).

- 🔴 **Bootstrap = 500, not 10,000.** `config/experiment_settings.strict_no_leakage.yaml:7` → flows to `make_all_figures.py:5341` → `stat_tests.paired_bootstrap_delta`. Default fallback is 200, never 10,000. Generated `n_resamples` column prints 500.
- 🔴 **Four comparison families, not five — the "One-hot KME vs signatures" family is absent.** Coded families: `signatures_vs_burden`, `maf_stack_vs_signatures`, `sig_maf_vs_signatures`, `sig_maf_vs_maf_stack` (`make_all_figures.py:86-94`).
- 🟠 **BH-FDR is grouped globally and per-figure, not within the five pre-specified families** (`make_all_figures.py:4295-4299`; text/markers use the global `q_value`).
- 🟠 **OS gets no paired test at all** — survival is skipped (`make_all_figures.py:4216-4217`).
- 🟡 Spearman bootstrap is **unstratified** while macro-AUROC is stratified (`make_all_figures.py:4277`); DeLong reports a p-value but **no CI** (`stat_tests.py:95`).

---

## 4. Engine / methodology concerns

**File:** `src/utils/nested_oof.py` (1,914 lines). The core nested-CV design is correct and leakage-safe (final refit on full outer-train fold, single pooled-OOF metric, fold-isolated scaling/selection, explicit split-integrity asserts at `:238-297`). Concerns:

- 🔴 **High-dimensional "Cox PH" is not a converged Cox fit.** For feature sets ≥500 features (i.e. the MAF stack at 948 and sig+MAF at 1,130 — *all* the headline survival numbers except burden/signatures), survival uses the hand-rolled `fast_breslow` backend, which runs **~5 fixed gradient-descent steps with no convergence check** (`nested_oof.py:614-637`, esp. `:625`). The reported OS c-index for MAF/sig+MAF is therefore a heavily-shrunk approximation, not a partial-likelihood MLE. Calling it "Cox proportional hazards" overstates it — disclose or switch to a proper penalized solver.
- 🟠 **Silent exception/warning swallowing** can mask failures behind plausible numbers:
  - `harrell_c_index` falls back to a (non-identical) pure-Python c-index if lifelines raises, with no log (`nested_oof.py:152-157`).
  - Candidate fits swallow any exception → `score=-inf` (`:908-912`, `:1091-1095`); a systematically failing learner degrades silently.
  - Convergence warnings suppressed (`:572,1260,1675`); high-dim linear **classification stays on `saga` by default** (the SGD switch only fires if configured), so it may report non-converged models. → Audit the saved `status`/`candidate_results` columns to confirm reported numbers came from the intended, converged estimator.
- 🟠 **Elastic-net description mismatch.** Manuscript says "α=0.01, L1 ratio=0.5, max_iter=5000". Code **tunes** α/L1 over grids (`nested_oof.py:53-55,307-314`) and uses `max_iter=10000` for **regression** (`:449`) vs `5000` for classification (`:476`). Reword the Methods.
- 🟡 `GroupKFold` fallback is unseeded (`:203`); `n_classes` summary field is mislabeled for survival (`:1899`).

---

## 5. Label / cohort construction concerns

**Files:** `src/utils/endpoint_registry.py`, `src/runners/run_main_manuscript_complete_panel.py`.

- 🟠 **HRD binary thresholds (24/33/42) are NOT computed in this repo** — they're read from precomputed `HRD-high`/`HRD-low` string columns in `final_analysis_cohort.tsv` (`run_main_manuscript_complete_panel.py:141-149`). The thresholding lives in an **upstream builder not in this checkout**.
- 🟠 **The manuscript's BRCA input-validation paragraph is not verifiable from the repo.** The funnel (981 DDR/clinical rows → 71,772 MAF → 807 samples → 792 patients → 67,370 SBS96 → final 772) and the **"highest-SNV primary-tumor tie-break"** appear **nowhere** in the audited code. Patient collapse is a blunt `drop_duplicates("patient")` that keeps the **first** row with no tie-break (`endpoint_registry.py:93,116,152`). Either this logic is upstream of the repo or the manuscript over-specifies it.
- 🟡 **Cross-index joins can silently shrink the cohort** with no assertion that n stays 772/8,800 — HRD labels keyed by `patient_id_12` are intersected with the feature index (`feature_evaluation.py:32`; `run_main_manuscript_complete_panel.py:841`); a barcode-format drift would quietly reduce n.
- 🟡 **Silent endpoint exclusion**: endpoints below support minimums become an audit row, never an error (`run_main_manuscript_complete_panel.py:150-151,159-160,185-197`); the run is marked "incomplete" but does not fail.
- 🟡 **Latent footgun**: `MAF_MAIN_ONLY_LEARNERS = LEARNERS` (`run_main_manuscript_complete_panel.py:63`) — the name implies a restriction that isn't active; a future edit could silently drop a learner from MAF reps while the completeness check desyncs.

---

## 6. Feature pipeline — leakage-safe, with caveats

**Verdict: no leakage.** All representations are deterministic per-sample functions of (a) that sample's own MAF/event rows and (b) **static vendored resources** (OncoKB/IntOGen/CGC/Vogelstein/CancerHotspots v3/COSMIC channel defs). Feature-block selection and scaling are strictly in-fold; transductive MAF construction is hard-blocked (`run_main_manuscript_complete_panel.py:104-106`). Caveats (fidelity, not leakage):

- 🟠 **`biological_maf_features.py` is a divergent LEGACY builder** (`"explicit_biological_maf_v2"`, ~3,100 locus-bin columns) — **not** the v4 stack. The canonical 948-feature v4 builder is `quick_bio_v4_nested_feature_selection.py`, whose docstring misleadingly says it "stays outside the manuscript pipeline" even though the production runner imports it (`run_main_manuscript_complete_panel.py:35`). Clarify which module is canonical to prevent a wrong-feature run.
- 🟡 **Kucab ID83 sets `repeat=0` hard-coded** (`kucab_event_features.py:326,337,339`) → all indels collapse into repeat-0 channels; it is a *coarse* ID83 that does not reproduce SigProfiler repeat/microhomology stratification. Disclose.
- 🟡 **Feature-cache fingerprint hashes only head+tail 1 MiB + size + mtime** (`feature_cache.py:38-50`) → a middle-of-file edit preserving size+mtime could serve a stale cache. Unlikely in practice; document or use a full hash for the MAF.
- 🟡 The signature count **182 = 179 channels + 3 burden covariates is inferred** (the MC3 feature bundle isn't in this checkout, so the 3 columns weren't directly read). Confirm against `features_standard_sbs96_id83.csv.gz` if exactness matters.

---

## 7. Reproduction / figure-generation concerns

**Files:** `scripts/reproduce_manuscript.py`, `src/utils/make_all_figures.py`, `src/utils/build_submission_docx.py`.

- 🔴 **Table 2 columns ≠ manuscript columns.** Code emits **Burden / Signatures / Bio MAF v4 / Signatures+bio MAF / MuAt-compatible** (`make_all_figures.py:71-77,403-409`). The manuscript's 3rd column "One-hot KME" has **no code path**, and the manuscript has **no MuAt column** — so the two Table 2 designs are structurally different.
- 🟠 **Cancer-type main-table metric is balanced-accuracy, not macro-AUROC** (`make_all_figures.py:3340-3341,4057-4063`), while the committed results store macro-AUROC as the row metric — an internal inconsistency to verify (strict figure validation may reject the committed table). Macro-AUROC for cancer type only appears in Suppl Table S4.
- 🟠 **`build_submission_docx.py` is stale and would break**: it hard-codes prose numbers decoupled from the regenerated tables (`:650-651,663-664`) and reads Table-2 columns by **keys that no longer match the generated headers** (`row["MAF stack"]`, `row["Signatures + MAF"]` vs generated `"Bio MAF v4"`, `"Signatures + bio MAF"`) → `KeyError`/mis-key (`:379-396`).
- 🟡 `figure_s2_calibration_thresholds.png` is actually an **OOF-coverage heatmap**, not reliability curves, despite the S2 caption (`make_all_figures.py:4744`). Real reliability data is only written to `plot_data/`.

---

## 8. Manuscript-internal issues (no code involved)

- 🔴 **The reference list belongs to a different paper.** In-text [2]=Kucab → ref[2]=Alexandrov 2013; [3]=MC3 → ref[3]=Koh review; [4]=MuAt → ref[4]=deconstructSigs; [5]=ATGC → ref[5]=Medo; [6]=HRDetect → ref[6]=MutationalPatterns; [9]=XGBoost(Chen&Guestrin) → ref[9]=Chaos Game Representation; [10]=SigProfilerMatrixGenerator → ref[10]=USM. The actual clinical refs (HRDetect/Telli/Knijnenburg/Davies…) appear as [12]–[21] but are never cited by number. The bibliography looks imported from an unrelated CGR/USM project. **Full reference rebuild needed.**
- 🟠 **Declarations describe a different study** — "Universal-BiCGR", "synthetic depth-sweep framework", "PCAWG benchmark" — none are in this paper's Methods (TCGA-MC3 / TCGA-BRCA / Kucab).
- 🟡 Author/corresponding inconsistency (byline marks Maron `*`; the block names Aaron Ge with an NIH Bethesda address vs a `@som.umaryland.edu` email).
- 🟡 Abstract/captions cite **"964 benchmark rows"** / "914 supplementary rows"; the committed main results table is **98 rows** (`results/tables/main_manuscript_complete_panel_summary.csv`). Verify whether a combined export exists or the counts are stale.

---

## 9. Organization / dead code (light cleanup)

- `src/utils/` mixes the **core package** (`nested_oof.py`, `endpoint_registry.py`, `feature_evaluation.py`, `make_all_figures.py`, `muat_compatible.py`) with **standalone tools**.
- **Unreferenced / dead w.r.t. the documented reproduce path:** `clean_generated_outputs.py` (referenced nowhere), `build_submission_docx.py` (referenced nowhere, and stale per §7), `analyze_bio_maf_v4_interpretability.py` (standalone post-hoc), and the `run_benchmark_phases.py` + `check_benchmark_completion.py` pair (an *alternate* orchestration the reviewer flow bypasses).
- Duplicate manifests (auto-mirrored by the script: root `DATA_MANIFEST.csv`/`LFS_MANIFEST.csv` vs `manifests/*`), duplicate `requirements.txt`, **4 orphaned `experiment_settings.muat_*.yaml`** configs, ~17 docs (several are process/cleanup artifacts), a shallow README, and **no tests/CI**.
- **Hugging Face data** (`vjdeara/cgr_tensorlab_umsom`) is public but **nothing in the code pulls from it** — reproduction needs a manually placed `../github_exports_datasets` sibling bundle. (Note: the HF repo's top level is `Benchmark/` + `github_exports_datasets/` + a loose FASTA — its layout would need a small path map to match `manifests/dataset_assets_manifest.csv` before auto-download works.)

---

## 10. What is genuinely strong (credit where due)

- Textbook **nested CV** with final refit on the full outer-train fold and a single **pooled-OOF** metric; preprocessing and feature-block selection are rigorously fold-isolated with **active split-integrity assertions** (`nested_oof.py:238-297`). This is *more* rigorous than the manuscript's stated "single 5-fold".
- Correct **Cox c-index sign convention** and pooled concordance; **degeneracy-safe macro/micro-AUROC**; deterministic seeding with order-independent candidate selection.
- Strong **reproducibility engineering**: checksum manifests, dataset-bundle verification, COSMIC-kept-separate guard, git-ignore policy for generated outputs, atomic writes, robust checkpoint-resume.
- The MuAt comparator is **honest** — every row is stamped `MuAt-compatible reimplementation`, leakage is closed with fixed-hash dictionaries, and it reuses the tabular benchmark's exact folds.

---

## 11. Recommended next steps for Aaron (prioritized)

1. **Decide the source of truth** (manuscript vs current code). The cleanest path is to **rewrite Methods/Table 2/Figures 2–5 to the current validated pipeline** (4 reps, top-20, Cox/c-index OS, nested CV, corrected feature counts) — reverting the code re-opens retired/forbidden paths.
2. **Fix the manuscript bibliography and Declarations** (§8) regardless of #1.
3. **Correct the stats text** to bootstrap = 500 and the four implemented families (or re-run at 10,000 with five families incl. KME if KME is reinstated).
4. **MuAt:** 10-fold + logit ensemble + top-1 primary + attention pooling; regenerate the stale fidelity report; contact the authors for official splits/checkpoints.
5. **Engine:** disclose or replace the 5-step high-dim "Cox"; surface swallowed exceptions/convergence warnings; align the elastic-net description.
6. **Cohort:** bring the BRCA funnel + primary-tumor/highest-SNV tie-break into the repo (or document it's upstream) and assert cohort n.
7. **Light cleanup** + **HF auto-download** wiring.

---

## Appendix A — File:line concern index

| Sev | Concern | File:line |
|---|---|---|
| 🔴 | KME representation forbidden / removed | `scripts/reproduce_manuscript.py:75-76,87-98`; `config/experiment_settings.strict_no_leakage.yaml:102-106` |
| 🔴 | top-10 cancer type forbidden; code is top-20 | `scripts/reproduce_manuscript.py:82`; `src/utils/endpoint_registry.py:16-40`; `config/endpoint_registry.yaml:43-69` |
| 🔴 | OS modeled as Cox/c-index, not binary/XGBoost; os_event banned | `src/utils/endpoint_registry.py:127-160`; `src/runners/run_main_manuscript_complete_panel.py:62,940-941`; `src/utils/make_all_figures.py:3165`; `src/utils/check_benchmark_completion.py:166-184,312` |
| 🔴 | High-dim "Cox" = ~5-step non-converged GD | `src/utils/nested_oof.py:614-637` (esp. `:625`) |
| 🔴 | Bootstrap 500 not 10,000 | `config/experiment_settings.strict_no_leakage.yaml:7`; `src/utils/make_all_figures.py:5341`; `src/utils/stat_tests.py:117,127` |
| 🔴 | Only 4 comparison families; KME family absent | `src/utils/make_all_figures.py:86-94` |
| 🔴 | Table 2 columns (MuAt-compatible, no KME) ≠ manuscript | `src/utils/make_all_figures.py:71-77,403-409` |
| 🔴 | Reference list mismatched / from another paper | manuscript bibliography [2]–[10] vs in-text |
| 🟠 | Nested CV (not single 5-fold) | `src/utils/nested_oof.py:1-6,188-235` |
| 🟠 | MAF/sig+MAF feature counts ≠ Table 3 | `results/tables/main_manuscript_complete_panel_feature_manifest.csv`; `src/utils/quick_bio_v4_nested_feature_selection.py:112-135` |
| 🟠 | Silent exception/warning swallowing | `src/utils/nested_oof.py:152-157,908-912,1091-1095,572,1260,1675` |
| 🟠 | Elastic-net params tuned + max_iter 10000/5000 split | `src/utils/nested_oof.py:53-55,307-314,449,476` |
| 🟠 | BH-FDR grouped globally/per-figure, not per-family; OS skipped | `src/utils/make_all_figures.py:4295-4299,4216-4217` |
| 🟠 | HRD thresholds read from precomputed cols (upstream) | `src/runners/run_main_manuscript_complete_panel.py:141-149` |
| 🟠 | BRCA funnel + highest-SNV tie-break absent from repo | `src/utils/endpoint_registry.py:93,116,152` |
| 🟠 | Legacy vs canonical MAF builder confusion | `src/utils/biological_maf_features.py:781`; `src/utils/quick_bio_v4_nested_feature_selection.py:1-6` |
| 🟠 | Cancer-type main-table metric is balanced-accuracy | `src/utils/make_all_figures.py:3340-3341,4057-4063` |
| 🟠 | build_submission_docx stale prose + column-key mismatch | `src/utils/build_submission_docx.py:379-396,650-651,663-664` |
| 🟠 | MuAt: 5-fold, no logit ensemble, 1 layer/head, mean pooling | `config/experiment_settings.muat_top20_comparable.yaml:33`; `src/utils/muat_compatible.py:519,525-527,569-571` |
| 🟠 | MuAt fidelity report stale (smoke 60 pts) | `results/tables/muat_style_tcga_comparator_fidelity_report.md` |
| 🟡 | Cohort join can silently shrink n | `src/utils/feature_evaluation.py:32`; `src/runners/run_main_manuscript_complete_panel.py:841` |
| 🟡 | Silent endpoint exclusion | `src/runners/run_main_manuscript_complete_panel.py:150-151,159-160,185-197` |
| 🟡 | MAF_MAIN_ONLY_LEARNERS == LEARNERS footgun | `src/runners/run_main_manuscript_complete_panel.py:63` |
| 🟡 | Kucab ID83 repeat hard-coded 0 | `src/utils/kucab_event_features.py:326,337,339` |
| 🟡 | Feature-cache fingerprint head+tail+size+mtime only | `src/utils/feature_cache.py:38-50` |
| 🟡 | Spearman bootstrap unstratified; DeLong has no CI | `src/utils/make_all_figures.py:4277`; `src/utils/stat_tests.py:95` |
| 🟡 | figure_s2 is a coverage heatmap, not reliability curves | `src/utils/make_all_figures.py:4744` |
| 🟡 | MuAt: macro-AUROC shown as primary vs paper's accuracy; first-5000 truncation; unweighted loss | `src/runners/run_muat_style_tcga_comparator.py:1034,1432`; `src/utils/muat_compatible.py:331` |
| 🟡 | Dead/standalone scripts | `src/utils/clean_generated_outputs.py`, `build_submission_docx.py`, `run_benchmark_phases.py`, `check_benchmark_completion.py`, `analyze_bio_maf_v4_interpretability.py` |
| 🟡 | Declarations describe a different study; author/corresponding mismatch; 964 vs 98 rows | manuscript Declarations / front matter |

*Line numbers are from the audited HEAD; verify after any refactor. Items marked "upstream" depend on dataset-prep code outside this checkout.*
