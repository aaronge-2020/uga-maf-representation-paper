# Results Summary — August 2026 Rerun

**Vijay Vankadaru · 21 August 2026**

Covers every deliverable from the 1 August meeting. All numbers come from a clean rebuild: the
previous results table and fold checkpoints were archived first, so no row was reused from the
June run. 96 of 96 endpoint × representation × learner slots completed.

---

## 1. Survival: does Bio MAF v4 predict outcome beyond cancer type?

Jeya's question, answered directly. Three CoxNet arms on TCGA CDR overall survival, **identical
outer folds across arms** (one fixed seed), cohort = survival ∩ cancer type ∩ Bio MAF v4,
n = 9,986, 3,061 events.

| Arm | Features | C-index |
|---|---|---|
| Cancer type only (one-hot, 33 types) | 33 | **0.7416** |
| Bio MAF v4 only | 948 | 0.6377 |
| Cancer type + Bio MAF v4 | 981 | **0.7429** |

Paired bootstrap, 2,000 resamples:

| Comparison | Δ C-index | 95% CI | p |
|---|---|---|---|
| Bio MAF v4 vs cancer type | −0.1039 | [−0.1153, −0.0923] | 0.000 |
| **Cancer type + Bio MAF vs cancer type** | **+0.0013** | **[−0.0028, +0.0054]** | **0.512** |
| Cancer type + Bio MAF vs Bio MAF | +0.1052 | [+0.0952, +0.1150] | 0.000 |

**Adding 948 genomic features to cancer type buys +0.0013 C-index.** The interval is tight enough
to exclude any effect above ~0.005, so this is a precise null rather than an underpowered one.
Cancer type alone *outperforms* the genomic representation by 0.10.

The interpretation is that the survival signal in Bio MAF v4 is cancer type, recovered
imperfectly: the representation predicts tumour type well, tumour type predicts mortality, and
once tumour type is known the representation contributes nothing further.

**Manuscript implication.** The survival endpoint cannot be presented as evidence that
representation choice matters for clinical outcome. It should be reframed as a negative result:
mutation-derived features add no incremental prognostic value beyond tumour type in TCGA.

---

## 2. Main panel, clean rebuild

96/96 slots, no cached rows. Survival numbers moved only marginally against the submitted draft,
which confirms the survival finding above is not an artefact of the older code:

| Endpoint | Representation | New | Submitted draft |
|---|---|---|---|
| OS | burden only | 0.5838 | 0.584 |
| OS | signatures | 0.6069 | 0.607 |
| OS | Bio MAF v4 | 0.6335 | 0.632 |
| OS | signatures + Bio MAF v4 | 0.6505 | 0.647 |
| PFI | burden only | 0.5552 | 0.548 |
| PFI | signatures | 0.5697 | 0.568 |
| PFI | Bio MAF v4 | 0.6220 | 0.606 |
| PFI | signatures + Bio MAF v4 | 0.6305 | 0.614 |

Also resolved by the rebuild: `damage_class` now reports **balanced accuracy** in
`endpoint_results`, matching Table 1. It previously reported macro-AUROC, which was a
documentation inconsistency.

Table S3 (non-applicability summary) now reports **zero unmeasured combinations**. The submitted
draft listed five representation families with no measured results.

---

## 3. Individual-feature selection

Jeya's protocol, implemented as specified: 80/20 split, 5-fold internal CV over six sparsity
settings, select the sparsest setting within tolerance of the best inner score, refit on the full
training partition, evaluate once on the held-out 20%.

| Endpoint | Selected | Inner-CV (best) | Held-out |
|---|---|---|---|
| cancer_type_top20 | **347 / 948 (36.6%)** | 0.6490 (0.6490) | 0.6968 |
| OS | **92 / 948 (9.7%)** | 0.6296 (0.6391) | 0.6108 |

For cancer type the selected setting **matched the best inner-CV score exactly** — a 63% reduction
in feature count at zero measurable cost. That is a direct answer to the overfitting concern: most
of the 948-feature space is not carrying signal.

PCA was considered and rejected per Jeya's reasoning: it compresses variance but does not remove
features whose apparent signal is noise.

---

## 4. SHAP attribution by annotation block

Exact TreeSHAP via XGBoost's `pred_contribs`, computed on held-out samples only, aggregated to
annotation families rather than individual features.

| Endpoint | Largest block | Share |
|---|---|---|
| cancer_type_top20 | VEP consequence | 21.9% |
| OS | VEP other (SIFT, PolyPhen, biotype) | 20.6% |

No block exceeds 22%. Attribution spreads across VEP consequence and impact, per-gene driver
counts, OncoKB role, hotspots and allele-fraction summaries. The winning model is therefore not
resting on a single unstable feature.

Worth noting: gain and SHAP rankings disagree in informative ways. `IDH1` ranks 5th by total gain
but 41st by mean |SHAP| — consistent with a feature that is decisive for a small subset (IDH1
mutation is near-pathognomonic for glioma) rather than broadly informative.

---

## 5. MuAt comparator: fidelity ablation

Four arms on the HRD endpoints, each adding one fidelity fix, plus the tumour-typing benchmark.

### HRD endpoints (n = 772)

| Arm | HRD_Score (Spearman) | HRD33 (AUROC) |
|---|---|---|
| A — as committed (training-budget bug, hashed dictionaries) | 0.4766 | 0.7582 |
| B — + epoch-budget fix | 0.5032 | 0.7274 |
| C — + exact token dictionaries | 0.5041 | 0.7028 |
| D — + MuAt-faithful motif grammar and transcriptional strand | 0.4815 | 0.7335 |
| *tuned tabular baseline* | *0.781* | *0.867* |

Total spread across four arms: 0.028 Spearman, 0.055 AUROC — both within per-fold standard
deviation, and directionally inconsistent. **Progressive fidelity fixes did not improve HRD
performance.**

Diagnosis: given a 150-epoch budget the model selects stopping epochs of 1–34, i.e. it peaks
almost immediately and then overfits. HRD is data-limited (~600 training samples per fold), not
representation-limited.

### Tumour typing — MuAt's own benchmark (n = 8,800, 20 classes)

| Model | Accuracy | Top-5 |
|---|---|---|
| Signatures + Bio MAF v4 / XGBoost | **0.744** | **0.955** |
| MuAt, published (Sanjaya et al. 2023, n = 7,352) | 0.641 | 0.906 |
| MuAt-compatible, after fixes | **0.570** | 0.871 |
| MuAt-compatible, before fixes | 0.539 | 0.861 |

Balanced accuracy after fixes: 0.5502.

**The fixes gained +0.031 accuracy here, having gained nothing on HRD.** That is exactly what the
data-starvation diagnosis predicts: correcting the training budget and the token dictionaries pays
off when there are 8,800 samples to exploit them, and not when there are 772. The gap to published
MuAt narrowed from 0.102 to 0.071.

### Remaining, documented deviations from the source paper

- **Architecture search is ranked on a short budget.** MuAt trained every candidate to completion
  within each fold. In practice our search selected embedding 128 with 1–4 layers on every fold,
  whereas MuAt's published model was embedding 512 with 2 layers — the ranking budget biases
  toward small, fast-converging models.
- **No ensembling.** MuAt reported an ensemble summing logits across the ten fold models; we
  report single models under nested 5-fold CV. This depresses our score relative to theirs and is
  a stricter protocol, but it must be stated.
- **No SV/MEI modalities**, which whole-exome data cannot supply.
- **Pooling is unverified.** The paper does not specify how mutation features are pooled into a
  tumour representation. Attention-weighted set pooling is our design choice and should not be
  attributed to the source paper in Methods.

---

## 6. HRD shortcut ablation

Tests whether the HRD result reflects mutation biology or two obvious shortcuts: direct hits in
HRD/HRR genes, and VAF features that can proxy tumour purity or copy-number state (HRD scores are
themselves copy-number derived, so this is a route to partial circularity).

Same five outer folds across arms, paired bootstrap with 2,000 resamples.

| Endpoint | Full | − HRD/HRR genes | Δ (p) | − VAF | Δ (p) |
|---|---|---|---|---|---|
| HRD_Score | 0.7604 | 0.7628 | **+0.0023 (0.622)** | 0.7468 | −0.0136 (0.072) |
| hrd_binary_24 | 0.8633 | 0.8514 | **−0.0119 (0.180)** | 0.8690 | +0.0057 (0.183) |
| hrd_binary_33 | 0.8995 | 0.8981 | **−0.0014 (0.637)** | 0.8742 | −0.0254 (0.000) |
| hrd_binary_42 | 0.8966 | 0.8976 | **+0.0010 (0.725)** | 0.9018 | +0.0052 (0.261) |

**Removing all per-gene features for the 16 HRD/HRR genes (32 columns, including BRCA1, BRCA2,
PALB2, ATM) costs nothing on any of the four endpoints.** Every comparison is non-significant and
two are positive. The HRD result is therefore not a restatement of "a known HR gene is mutated".

VAF removal is mixed: significant only on hrd_binary_33 (−0.025), non-significant elsewhere, and
positive on two endpoints. There is no systematic purity or clonality dependence, though the
hrd_binary_33 effect should be reported.

This converts the HRD claim from "Bio MAF v4 predicts HRD" into "Bio MAF v4 still carries
HRD-associated signal after removing direct HRD-gene and allele-fraction shortcuts", which is
considerably more defensible.

---

## 7. Defects found and fixed

| Defect | Effect | Status |
|---|---|---|
| Attention pooling raised on any GPU run (fp16/fp32 mask-fill mismatch) | The pooling described in Methods had never completed a training step on a GPU; it could only run with AMP disabled, i.e. on CPU | fixed |
| Architecture-search epoch budget reused as the final training budget | The final model trained for ≤3 epochs instead of the configured 150 | fixed |
| Token dictionaries hashed into fewer buckets than the vocabulary contains | ~96% of motif tokens and ~76% of position bins collided | fixed (`dictionary_mode: exact`) |
| Event truncation sorted on a string chromosome column | Lexicographic order discarded chromosomes 3–9, X and Y in hypermutators | fixed (seeded subsample); only 2 of 772 HRD tumours exceed the cap |
| CoxNet fitted at a single alpha without a warm-start path | `ArithmeticError` on high-dimensional survival designs; degenerate all-zero coefficient folds | fixed (regularisation-path fallback) |
| Fold-checkpoint fingerprint omitted the training protocol | A fixed rerun would silently reuse pre-fix checkpoints and report them as fixed | fixed |
| Table S3 written only when combinations were missing | A fully complete benchmark failed strict validation — the run failed because it succeeded | fixed |

---

## 8. Reproducibility: the published bundle is incomplete

Four inputs required by the pipeline are absent from the published Hugging Face dataset. Each was
discovered only by running from a clean state, and each cost a run that failed hours in.

| Missing input | Blocks |
|---|---|
| `TCGA-CDR-SupplementalTableS1.xlsx` | cancer type, OS, PFI |
| `COSMIC_v3.5_SBS_GRCh37.txt` | Kucab spectra (96 channels) |
| `COSMIC_v3.5_ID_GRCh37.txt` | Kucab spectra (83 channels) |
| `datasets/kucab_mendeley/raw_mutation_tables/` (entire directory) | Kucab damage_class |

**As published, the bundle cannot reproduce the main panel.** The CDR table has been uploaded; the
remaining three were sourced locally and should be uploaded before submission.

`scripts/audit_bundle_inputs.py` now checks every required input in about five seconds and reports
which endpoints each gap blocks. It should be step one of the reproduction instructions.

Separately: the longest tracked path in the repository is 249 characters
(`results/tables/main_manuscript_complete_panel_nested_fold_checkpoints/candidate_rows/…`), which
exceeds the Windows 260-character limit once a normal checkout prefix is added. This is why the
repository could not be cloned in June. Workaround is `git config core.longpaths true`; the real
fix is to shorten those checkpoint filenames.

---

## 9. Suggested manuscript changes

1. **Survival.** Reframe from predictive performance to incremental value over tumour type, and
   report the cancer-type baseline alongside the representation results.
2. **Methods, "training for 150 epochs".** True of the configuration, but the value was
   unreachable in the committed code. Now accurate.
3. **Methods, attention-weighted pooling.** Do not attribute to the source paper; it is our design
   choice and the paper does not specify pooling.
4. **Methods, "limited architecture search".** Describe the ranking budget honestly, or make the
   search faithful, or adopt MuAt's published architecture and say so.
5. **Fold pairing.** Outer folds differ across representations in the main panel because the seed
   is derived from the representation name. Cross-representation comparisons are pooled-cohort,
   not strictly paired. State it, or re-run paired.
6. **Metastatic status.** Raised on 1 August and unresolved. It is a survival confounder and
   belongs in Methods or Limitations either way.
7. **Add the HRD shortcut ablation** as a supplementary analysis; it pre-empts the most obvious
   reviewer objection to the HRD result.

---

## 10. Reproducing these results

```bash
python scripts/audit_bundle_inputs.py          # verify every required input first
python scripts/run_all_deliverables.py         # survival, feature selection, MuAt, figures
python src/run_all_experiments.py --config config/experiment_settings.strict_no_leakage.yaml \
    --paths config/paths.yaml --datasets-dir <bundle>   # main panel + manuscript figures
python scripts/run_hrd_shortcut_ablation.py    # HRD shortcut ablation
python scripts/make_new_analysis_figures.py    # figures for the analyses above
```

Outputs: `results/manuscript/` (manuscript figures and tables), `results/figures/` (new-analysis
figures), `results/tables/` (result tables).
