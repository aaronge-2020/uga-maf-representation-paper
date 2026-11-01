# ANALYSIS AGENT BRIEF (from Aaron, 2026-10-25)
Mutation representations, annotation value, and transfer beyond TCGA

## ROLE AND OBJECTIVE

Act as a computational cancer genomics researcher and methods reviewer. Audit, implement, run, and document a reproducible comparison of mutation representations in the repository aaronge-2020/uga-maf-representation-paper.

The scientific objective is to determine what information mutation representations add beyond established alternatives, what explains any advantage, and whether that information remains useful outside the development cohort. Bio MAF is a candidate representation, not a predetermined winner. A result showing that a simpler representation matches or exceeds it is acceptable and must be reported.

The intended journal is Oxford Bioinformatics. Do not promise publication or manufacture a novelty claim. Establish what the evidence supports relative to prior methods. Adding endpoints, software packaging, or validation checks alone does not establish a methodological advance.

This is an implementation and analysis assignment, not merely a planning assignment. Inspect existing work first, reuse valid outputs, implement the missing comparisons, execute the prioritized analyses that available data and resources support, and deliver a documented evidence package. Do not stop after producing a plan. If an analysis is blocked, finish independent work and identify the exact missing prerequisite without inventing data or results.

## 1. CONTEXT TO VERIFY

The manuscript reports four representations: burden, spectra, Bio MAF, and spectra plus Bio MAF. Bio MAF is described as 948 annotation features; spectra as 182 features, including SBS96, ID83, and count summaries. Verify these definitions against actual code and manifests.

Reported cohorts are 8,800 TCGA patients across 20 cancer types for cancer typing, 772 TCGA breast cancers for genomic scar HRD prediction, and 9,986 patients for overall survival. External evaluations include 913 ICGC only PCAWG samples spanning nine mapped training types and 147 POG570 breast cancers with the required HRD outcome. Do not confuse these 147 cases with the complete 570 patient POG cohort.

An earlier repository audit at commit ca13132 found unique TCGA patient rows before splitting, canonical shared splits, and donor disjoint training and test data. This is useful context, not a substitute for inspecting the current checkout. Do not repeatedly redesign splitting merely because some call sites use groups=None: splitting unique patient rows is valid. Check the patient identity invariants directly.

Known issues and existing work to locate:

1. Binary gene and role matched gene controls already exist, but the manuscript says they reused settings selected for the full representation. Determine whether independently tuned comparisons already exist before repeating work.
2. The manuscript describes both fixed recipe and architecture searched MuAt implementations. Verify their actual patients, seeds, inputs, checkpoints, and estimators.
3. Some HRD controls were sparse: nine nonzero fitted SBS3 exposures and fourteen positive BRCA indicators were reported. Audit before drawing biological conclusions.
4. External Bio MAF features were reconstructed differently from TCGA, with substantial annotation missingness. This is a potential measurement mismatch, not proof of its cause.
5. Some center outputs contain two repeated aliquot rows for TCGA-DV-A4W0 and TCGA-UZ-A9PS. Previous inspection found no split leakage. Calculate primary metrics once per patient and document the effect of aliquot weighting if present.
6. The Kucab event stack is a different representation, not Bio MAF. Do not include it in matched Bio MAF comparisons or use it to establish a reversal of Bio MAF performance across tasks.
7. Figures, tables, prose, software versions, and fold versus pooled estimators have contained discrepancies. Produce an authoritative results ledger rather than silently choosing preferred values.

## 2. WORKING PRACTICES

Read applicable AGENTS.md files and repository documentation. Inspect git status and preserve unrelated user work. Record the current branch, full commit SHA, remote, dirty state, and any working tree diff required to reproduce your run. Do not assume the checkout still matches ca13132. Verify LFS objects are real data rather than pointer files.

Work on an isolated branch or worktree if appropriate. Do not modify raw data or overwrite archived manuscript outputs. Use a new analysis namespace, for example analyses/representation_extension/, adapted to repository conventions. Separate scripts, configuration, raw inputs, derived features, predictions, and reports.

Never fabricate missing annotations, read counts, purity, clinical variables, or labels. Never silently drop failed models or unsupported patients. Do not publish or upload data, push branches, submit manuscripts, change access permissions, or create new licenses without explicit authorization. Publicly available does not automatically mean redistributable.

Use existing environments and workflows when suitable. Separate faithful reproduction of archived results from the new unified benchmark. Record dependencies and hardware. Do not mix prediction files from incompatible cohort definitions or silently upgrade libraries halfway through a comparison.

## 3. FIRST DELIVERABLE: AUDIT AND FIXED ANALYSIS PROTOCOL

Before executing new outcome comparisons, write an audit report and a versioned protocol containing:

1. What analyses already exist, which outputs are reusable, and what is missing.
2. Exact cohort definitions, patient counts, endpoints, label mappings, gene vocabularies, feature definitions, and split hashes.
3. Primary questions, representations, learners, metrics, contrasts, tuning budgets, and external evaluation rules.
4. A resource estimate and execution order. State which analyses are essential, conditional, or optional.
5. Known prior exposure to test results. These are follow up analyses of an existing study; do not call the protocol preregistered or the previously examined external cohorts untouched validation data.

Create an analysis status table with planned, running, completed, failed, and blocked states. Make routine reversible choices autonomously. Ask only for genuinely necessary access, spending authorization, or scientific decisions that cannot reasonably be resolved from the repository.

## 4. LITERATURE AND REPRESENTATION CROSSWALK

Read these papers and relevant available supplements or code:

1. Ozcan Simsek et al. 2019, Statistical representation models for mutation information within genomic data. DOI: 10.1186/s12859-019-2868-4.
2. Lee et al. 2019, CPEM: Accurate cancer type classification based on somatic alterations using an ensemble of a random forest and a deep neural network. DOI: 10.1038/s41598-019-53034-3.
3. Sanjaya et al. 2023, Mutation-Attention (MuAt). DOI: 10.1186/s13073-023-01204-4.

Create a crosswalk covering input mutation eligibility, gene definitions, consequence filtering, aggregation formula, normalization, additional modalities, training derived transformations, inference requirements, and task.

Important distinctions:

1. CPEM counts mutations that alter protein sequence within genes and combines these with other features, including copy number alterations. Implementing its mutation feature idea is not reproducing the complete CPEM model. Its 96 trinucleotide channels should not be mistaken for fitted signature exposures merely because of the paper's terminology.
2. The statistical encoding paper uses exonic and intronic variants and broad cancer categories. Its published performance is not directly comparable to this manuscript's coding mutation cohort and twenty labels.
3. Class relevance weighting depends on cancer labels. Resolve how the original method transforms an unlabeled patient before reproducing it. Never select a test patient's transformation using the patient's true class. If an unambiguous valid implementation cannot be established, use an inference safe alternative such as TF-IDF, document the limitation, and do not label it a faithful reproduction of the ambiguous method.
4. Fit all learned weighting, scaling, feature selection, and vocabulary filtering inside the appropriate training partition. This applies within inner validation as well as outer testing.

Do not compare published accuracy values against this manuscript's balanced accuracy as evidence of superiority. Literature claims must link to primary sources and exact sections, tables, or equations. Distinguish faithful reproduction, representation adaptation, and a new control.

## 5. DATA AND VALIDATION CONTRACT

Reuse canonical outer patient splits when valid. Identify patients by the appropriate dataset specific schema. TCGA barcode truncation to twelve characters must not be applied blindly to ICGC or POG identifiers.

For each task and representation verify:

1. One analytical row per patient, or an explicitly documented grouping rule.
2. No patient overlap between training, inner validation, and outer test sets at the relevant stage.
3. Each eligible patient has one outer test prediction per complete run. Repetition across representations and explicitly labeled seeds is expected, not a duplicate error.
4. Identical evaluation patients and outcomes for paired contrasts. Use explicit joins and report exclusions; equal fold numbers do not establish patient matching.
5. No endpoint labels enter fixed Bio MAF features. Any supervised selection or encoding is fitted using training labels only.
6. External samples do not enter tuning, feature selection, threshold selection, or model fitting for the frozen transfer analysis. Retain the audited ICGC only restriction and verify donor independence without assuming that distinct sample labels alone prove independence.

Record raw input checksums, provenance, reference builds, filtering, duplicate handling, transcript rules, and exclusion counts. Distinguish an absent mutation from missing annotation and from unavailable sequencing coverage.

## 6. PRIORITY A: FAIR COMPARISON WITH ESTABLISHED GENE ENCODINGS

Primary tasks are cancer typing and continuous genomic scar HRD prediction. Use the existing survival cohort later for incremental analyses. Begin with a fixed gene panel identical to Bio MAF's panel to isolate encoding from gene selection.

Implement or verify:

R0. Existing burden baseline.
R1. Existing SBS and indel spectra baseline, with count summaries explicitly identified.
R2. Binary mutation presence per gene.
R3. Eligible mutation counts per gene.
R4. Counts within genes separated by defined consequence or impact categories. Explicitly document how unknown and multiple transcript consequences are handled.
R5. A published statistical gene weighting representation. TF-IDF is a reasonable default, fitted in training data. Record the formula and all choices; do not claim an exact published implementation unless verified.
R6. Existing full Bio MAF, audited against its manifest.

Run R2 through R6 alone and combined with R1. Prioritize the combinations with R1 if resources are constrained, because the primary question is added value beyond spectra. Preserve labels identifying pure representation and combined representation.

Add a functional impact weighted representation, such as sums of CADD scores within genes, if verified scores and sufficient coverage can be obtained. Log coverage by variant type, gene, cohort, and consequence. Do not zero fill unavailable scores and call the result a valid CADD reproduction. If this is blocked, document it and retain consequence aware counts as an available biological encoding control, without claiming equivalence.

Also run binary presence and gene counts on all eligible genes. Learn any prevalence filters inside training data. This small extension separates the effect of the curated panel from the encoding. Do not derive a gene vocabulary from external outcomes.

For count based representations assess a documented normalization sensitivity, such as log1p counts or burden normalized counts with total burden retained separately. Choose transformations inside training data or fix them in the protocol. Raw spectra and fitted signature activities are different representations and must remain distinct.

Use elastic net classification or regression as appropriate and XGBoost. Tune each representation independently with common learner specific search spaces, the same inner folds, endpoint appropriate objectives, and comparable search budgets. Use at least the original ten trials per outer fold as the initial budget if adequate; document any budget changes before execution. Do not reuse full Bio MAF hyperparameters as the primary evaluation of reduced representations. Existing fixed setting ablations may remain sensitivities.

Hold early stopping, probability threshold selection, class weighting decisions, and feature selection inside training data. Record every trial, seed, selected configuration, fit time, and failure. Equal trial counts are a resource convention, not proof that every method is fully optimized; examine optimization stability for the leading alternatives if conclusions depend on small differences.

Primary contrasts:

1. Spectra plus Bio MAF versus spectra plus binary gene indicators.
2. Spectra plus Bio MAF versus spectra plus gene counts.
3. Spectra plus Bio MAF versus spectra plus consequence aware gene counts.
4. Spectra plus Bio MAF versus spectra plus the specified published weighting representation.

Report the panel restricted and all eligible gene comparisons separately. Do not choose the most favorable baseline retrospectively. If summarizing the best baseline, specify its selection procedure and acknowledge selection uncertainty.

## 7. PRIORITY B: WHAT INFORMATION EXPLAINS THE RESULT?

Define nonoverlapping feature blocks from the actual code: gene event information, consequence and functional prediction summaries, variant allele fraction summaries, and curated role, hotspot, or driver summaries. Some features encode several concepts. Document these overlaps rather than pretending the blocks remove all related biology.

Use planned additions to spectra plus a simple gene representation and removals from full Bio MAF. Select any reference gene representation through training validation or report the planned controls individually. Retune each configuration.

Test whether:

1. Consequence information adds value beyond gene identity and count.
2. Allele fraction summaries add value beyond spectra and gene features.
3. Curated summaries add value after those inputs are present.
4. A compact subset retains practically similar performance and improves transfer or resource requirements.

Use addition and removal evidence together. A sequential addition order alone does not uniquely attribute correlated information. SHAP describes fitted model use, not causality or necessity. Avoid claiming independence from biological knowledge merely because a small set of curated columns can be removed.

If selecting a compact feature subset, select it within training data at every validation level. Do not choose a subset using full cohort importance and then report ordinary cross validation as an unbiased estimate. Any claim of practical equivalence requires an explicit margin and uncertainty analysis, not a nonsignificant superiority test.

## 8. PRIORITY C: CONTROLLED EXTERNAL TRANSFER

Evaluate the leading simple gene representations and Bio MAF using frozen source trained models. Include the main prespecified candidates so source ranking does not dictate which competitors are reported externally.

Compare the existing pipeline with:

1. Matched coding territory or documented common genomic intervals across TCGA and external calls, with callable coverage information where available.
2. Consistent annotation release, genome build, allele normalization, transcript prioritization, and feature construction across cohorts.
3. A restricted representation using features reliably measurable in both cohorts.

Change one factor at a time where possible. If changing feature definitions or variant inclusion, retrain and internally evaluate the corresponding TCGA model before freezing it. Do not feed redefined features into old models and interpret the change as a valid method comparison.

Audit feature distributions and annotation missingness by cohort and cancer type. Compare zero, missing, and unknown values. A diagnostic masking experiment in TCGA can test the sensitivity to external annotation availability, but it does not prove that missingness is the only cause of transfer failure.

For PCAWG retain the original classifier's full label space in the primary analysis. Predictions into one of the other training types remain errors. Compute balanced accuracy over the observed true types and report the exact rule. Do not silently restrict and renormalize predictions to nine labels. Report per type recall, sample size, uncertainty, and a confusion matrix.

For POG570 report rank correlation, absolute error, R squared, calibration, and threshold performance where target definitions are comparable. Document differences in score construction and ascertainment. If recalibrating on target data, use separate calibration and evaluation patients and label that analysis as adaptation, not frozen transfer.

Previously inspected cohorts are not pristine prospective validation. Any adaptations informed by PCAWG or POG results must be disclosed. A genuinely new cohort would strengthen confirmation if available; do not block the primary comparison while searching indefinitely for one.

## 9. PRIORITY D: HRD CONTROLS AND INTERPRETATION

Keep prediction of genomic scar scores separate from functional repair deficiency, BRCA dysfunction, and treatment response. Do not claim clinical response prediction without the appropriate outcomes.

Audit the nine nonzero SBS3 exposures and the fourteen positive BRCA indicators previously reported. Check label joins, signature fitting, catalogue version, mutation counts, fitting constraints, opportunity correction, and reconstruction error. Establish which germline, somatic, biallelic, copy number, and methylation information is actually available. Do not equate absence of a somatic mutation indicator with BRCA proficiency.

Repeat the core representation contrasts within breast cancer. Where data permit, assess added information beyond mutation burden, purity, ploidy, breast subtype, and technical covariates. Use matched patient subsets and report missingness; reductions in sample size can change comparisons. Stratified estimates should not be overinterpreted when sample sizes are small.

Determine whether allele fraction summaries remain useful after these variables are included. Association with purity or copy number is not automatically invalid for prediction, but it changes the biological interpretation.

Review current relevant HRD methods, including SigMA and methods with a valid exome implementation. Select a representative comparator only after checking required inputs, training cohort overlap, and target compatibility. Do not force WGS methods requiring structural variants onto exome inputs, and do not evaluate pretrained models as independent if their training donors overlap the test cohort.

Treat the continuous scar score as a primary outcome unless an existing scientific justification specifies otherwise. Retain thresholds 24, 33, and 42 transparently; justify their assay and disease context. Do not make a threshold primary merely because another pipeline happened to produce it. A difference in statistical significance between regression and classification does not itself establish a difference between their effects.

## 9A. ENDPOINT INVENTORY AND ONE OPTIONAL COMPLEMENTARY TASK

Before adding endpoints, inventory actual local label files and scripts. Separate labels already present, labels retrievable from a cited public resource, labels requiring new computation, and unavailable labels. Do not assume that every annotation published for TCGA, PCAWG, or POG is present locally or available for every donor.

The earlier audit reported outputs for HRD score, TAI, LST, LOH, PARPi7, eCARD, HRD thresholds, OS, and PFI, plus an unspecified LUAD endpoint. Verify definitions and availability. TAI, LST, and LOH are related components of the scar target; they are useful diagnostics rather than independent biological validation. Expression signatures named PARPi7 or eCARD must not be described as observed treatment response without the relevant clinical outcomes.

Candidate complementary endpoints, in priority order for feasibility review:

1. Documented smoking history within LUAD: ever versus never smoking as a primary candidate, with pack years secondary if sufficiently complete. Verify the existing LUAD endpoint before adding a duplicate task. Treat unknown as missing, not never smoking. Use clinical history, not a label generated from tobacco mutation signatures. LUSC may have too few never smokers for stable validation; assess it separately rather than pooling histologies to create an easy contrast. If combining histologies, include histology only and clinical baselines and report within histology results. Smoking history and molecular tobacco signatures measure related but different phenomena, so disagreement is not necessarily prediction error from poor representation.
2. Breast subtype defined using RNA, such as PAM50: an independently measured molecular phenotype. Verify subtype release, specimen matching, normal like and unknown handling, and class sizes. Do not supply expression data to a model intended to evaluate mutation representations.
3. Microsatellite instability using a documented clinical or assay label in a suitable cancer cohort, such as colorectal or endometrial cancer. Record whether labels come from an independent assay or are inferred from the same sequencing data. Do not define the outcome using the very count or feature being benchmarked. Analyze within cancer before pooling.
4. HPV status within HNSC from a defined assay or published viral call set. Document assay differences, missingness, anatomical subsite, and smoking information. Do not infer HPV status from TP53 or other mutation features and then evaluate prediction of that derived label.

For each candidate report the label source, definition, eligible patients, class counts or outcome distribution, missingness, overlap with feature data, potential circularity, covariates, split feasibility, and a scientific reason it answers a distinct question. Do not use outer test performance to select which endpoint becomes the new main result.

Prioritize one complementary endpoint only if labels and class sizes support it. Smoking within LUAD is a good candidate for testing exposure related information; PAM50 is an alternative if the goal is an independently measured molecular state. Run the complete core representation comparison for the chosen endpoint, not just Bio MAF. Use appropriate AUROC, precision recall, balanced accuracy, and calibration or regression metrics. Avoid automatic expansion to every available outcome. Additional endpoints do not replace strong baseline comparisons or establish novelty by themselves.

## 10. CONDITIONAL WORK: ATTENTION, LEARNING CURVES, SURVIVAL

If the neural comparison remains a major claim, use the verified architecture searched MuAt implementation as the primary local comparator and the fixed recipe as a sensitivity. Verify canonical patient matching. Audit input content, mutation subsampling, truncation, checkpoint objectives, and training stability. Run at least three training seeds for the main neural comparison if feasible. Do not attribute a pipeline gap solely to architecture.

If claiming that explicit features help with limited training data, run training fractions such as 25%, 50%, 75%, and 100% for the main tabular alternatives and attention model. Keep outer test patients fixed, use outcome aware sampling, and perform tuning and early stopping within the available training fraction. Include multiple sampling seeds. Otherwise omit learning curves rather than imply a sample efficiency claim.

For survival compare the same clinical baseline against clinical plus simple gene features and clinical plus spectra and Bio MAF. Use a common clinical complete case or documented imputation strategy. Include cancer type, age, sex, and stage where valid. Address event counts, time origin, censoring, and cancer specific stage definitions. Report incremental performance rather than representation only rankings. Use within cancer or stratified sensitivity analyses when informative. Do not pool incomparable Cox risk scores from independently fitted folds without a justified scoring approach. Do not extend an overall survival conclusion to PFI without the corresponding analysis.

## 11. STATISTICAL ANALYSIS

Prespecify primary contrasts and avoid treating five fold scores as independent experiments. Preserve patient pairing in all comparisons. Use a limited multiplicity family for the primary comparisons and label additional analyses exploratory.

Recommended primary estimators, unless a documented scientific reason favors another:

1. Cancer typing: balanced accuracy from complete outer predictions, with macro F1, ordinary accuracy, per type recall, and probability calibration as secondary measures.
2. HRD regression: mean within fold Spearman correlation, with pooled correlation as a labeled sensitivity; include absolute error and R squared.
3. HRD classification: mean within fold AUROC and area under the precision recall curve; report class prevalence and pooled scoring separately. Explain any ranking reversal.
4. External endpoints: metrics on the frozen predictions, with patient based uncertainty and explicit class weighting.

Use paired patient bootstrap intervals for appropriate contrasts, retaining folds and stratifying by cancer type where appropriate. When predictions are held fixed, label the interval as conditional on the fitted models. It does not capture all training or model selection uncertainty. Summarize retraining variability across prespecified seeds for leading comparisons; do not pretend repeated seeds are new independent patients. Do not report bootstrap probabilities as exactly zero. Avoid unsupported equivalence claims.

Use final source model selection procedures based only on source training validation. Do not select hyperparameters or representations on external outcomes. Do not reserve test observations retrospectively and describe them as never examined if earlier runs included them.

## 12. REPRODUCIBILITY, VALIDATION, AND COMPUTE

Provide a single documented workflow with stages for data validation, features, inner tuning, outer prediction, external prediction, metrics, and report generation. Support resuming completed jobs by configuration and input hashes. Run a small integration check before expensive training. Test the meaningful invariants: donor separation, fit versus transform boundaries, exact feature alignment, missingness handling, and metric reconstruction.

Record random seeds, library versions, operating system, CPU/GPU, thread counts, deterministic settings, commands, exit codes, wall time, memory where measurable, and all fit failures. If exact GPU reproducibility is unavailable, document expected numerical tolerances and compare results using appropriate precision. Do not claim bitwise reproducibility without testing it.

Run existing required repository checks. Have a fresh process rebuild selected metrics and one representative model run from saved inputs and configuration. Do not expand smoke tests into unnecessary full reruns when no new issue warrants them.

Resource priority is A, B, C, then D. Neural learning curves and expanded survival analyses are conditional on the main claims. Diagnose failures before spending resources on further runs. Record unsupported and blocked analyses explicitly, including a safe next action and the consequence for the manuscript claim.

## 13. FAIR DELIVERABLES

Findable: Give the code release, feature definitions, dataset versions, and analysis runs stable identifiers. Include manuscript title, authors where known, keywords, version, dates, source commit, and related primary literature in machine readable metadata. Prepare a citation file and deposition metadata; do not publish or mint a DOI without authorization.

Accessible: Document retrieval URLs, access dates, credentials or application procedures required for controlled inputs, and restrictions. Provide small legally redistributable examples or explicitly labeled synthetic fixtures. FAIR does not mean that controlled donor data, predictions, or trained models may be redistributed without review.

Interoperable: Use documented TSV or Parquet tables, JSON or YAML configuration, UTF-8 text, stable identifiers, explicit missing values, and a schema with units and allowed values. Declare genome assembly, transcript release, gene identifier mapping, and sequence conventions. Avoid undocumented pickled objects as the only reusable output. Retain software native models with version requirements and preferably accessible tabular predictions where permitted.

Reusable: Record exact aggregation formulas, source provenance, checksums, reference snapshots or reproducible retrieval, parameter searches, preprocessing, and software dependencies. Preserve original licenses and terms for TCGA, reference annotations, OncoKB, CADD, code, and derived outputs as applicable. Do not apply the repository license indiscriminately to third party data.

## 14. REQUIRED OUTPUT PACKAGE

Use repository conventions where practical. At minimum provide:

1. README with scientific scope, exact reproduction commands, expected outputs, resource requirements, and limitations.
2. Audit report, fixed protocol, analysis status table, and deviations log.
3. Cohort and split manifests, provenance and checksum manifest, a literature to implementation crosswalk, and an endpoint availability inventory.
4. Versioned feature dictionary with formula, gene scope, prerequisites, missingness conventions, and feature block membership.
5. Dependency lock or equivalent environment specification; container recipe if appropriate and feasible.
6. Tuning trial tables and final model configuration tables.
7. Patient prediction tables containing dataset, task, patient key, true outcome, fold, seed, representation, learner, prediction, full class probabilities where applicable, and source model identifier. Respect access restrictions.
8. Metrics and paired contrast tables with cohort N, estimator, units, uncertainty method, and multiplicity family.
9. Figures generated entirely from those tables: main representation comparison, feature additions/removals, external per type performance, and HRD calibration or error. Add learning curves only if run.
10. A main results ledger mapping every proposed manuscript number to its source file, exact cohort, model, metric, and estimator.
11. A concise collaborator report covering the findings, their uncertainty, what was already known, what is new, and what remains unresolved.
12. A claim audit table with columns: proposed claim; closest precedent; new evidence; limitations; supported wording; unsupported wording.

## 15. COMPLETION AND CLAIM RULES

Do not stop with a literature summary or a list of proposed experiments. Complete the feasible prioritized work, verify outputs, and report exactly what ran.

The final report must answer:

1. Does Bio MAF add value beyond simple and established gene representations on matched data?
2. How much of any gain depends on the gene panel, consequence information, allele fractions, or curated summaries?
3. Which findings persist after consistent external feature construction?
4. Are HRD improvements robust to adequate controls and relevant alternative methods?
5. What comparisons remain limited by tuning, sample size, input mismatch, prior use of test results, or missing data?
6. What substantive contribution can now be defended for publication, and what claim should be narrowed or abandoned?

If Bio MAF wins, quantify the gain and its robustness. If simpler representations match it, assess whether unnecessary annotation can be removed and what practical benefit follows. If rankings change across settings, identify the evidence for those conditions without implying universal rules. If none of these results supports a substantial advance, say so plainly and recommend a narrower benchmark or resource framing.

Do not claim clinical readiness, causality, universal superiority over transformers, absence of useful BRCA biology, or independence from all prior knowledge. Do not claim novelty solely because no previous paper used this exact combination of cohorts or feature names. Report a conclusion the experiments establish, not the conclusion the manuscript originally hoped to establish.
