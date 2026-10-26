# Analysis status table

Project: mutation representation extension benchmark (brief dated 2026-10-25). Branch: `analysis/representation-extension`. Last updated: 2026-10-25, first-deliverable checkpoint.

| ID | Workstream | State | Evidence / blocker |
|---|---|---|---|
| A0 | Audit report and fixed protocol v1.0 (with v1.1 amendments) | completed | reports/audit_report.md, reports/protocol_v1.0.md, reports/deviations_log.md |
| A1 | Literature and representation crosswalk (Ozcan Simsek 2019, CPEM 2019, MuAt 2023) | completed | reports/literature_crosswalk.md |
| A2 | Feature construction R2 to R5 from events files, both tasks, panel and all-gene arms | completed | features/*.csv.gz, config/feature_build_manifest.csv (cancer 2,039,980 eligible events, HRD 67,440; 9 no-event cancer patients retained as zero rows) |
| A3 | Canonical splits (repo convention triple) and split provenance | completed | config/canonical_split_cancer.csv, config/canonical_split_hrd.csv, config/split_provenance.json |
| A4 | Integration check (alignment, TF-IDF boundary, metric reconstruction) | completed | metrics/integration_check.json, status PASS |
| A5 | Priority A runs, primary grid (2 tasks x 2 learners x 8 representations) | running | queue2, predictions/ fills cell by cell; trials in metrics/trials/; per-fold checkpointing added to runner.py 2026-10-26 (deviations log entry 9), so an interrupted cell resumes at its first unfinished fold with identical tuning |
| A6 | Priority A extension cells (all-gene combos, log1p sensitivity, alone arms for R2/R4/R5) | planned | queued after A5 in queue2 |
| A7 | Burden (R0) cells under the unified protocol | running | included in the A5 grid |
| A8 | Paired contrasts and primary metrics tables (pooled primary, fold means labelled) | planned | starts when A5 cells for both sides of a contrast exist |
| A9 | CADD-weighted representation | blocked | missing prerequisite: a licensed CADD score source matched to the event calls, with coverage logging. R4 is the available control and is not equivalent |
| B1 | Priority B feature blocks: definitions from the feature guide | completed | audit report section 1 (802 per-gene, 121 compact biology, 16 tiers, 4 hotspot, 5 controls) |
| B2 | Priority B additions to spectra plus R3 and removals from Bio MAF, retuned | planned | after A5; uses the Bio MAF matrix at results/tables/quick_bio_maf_features.csv.gz |
| B3 | Compact subset selection inside training data only, with equivalence margin analysis | planned | margin fixed in protocol (0.01 balanced accuracy, 0.02 Spearman) |
| C1 | Priority C frozen transfer, PCAWG (full 20-label space, per-type recall, confusion matrix) | planned | external matrices present: data/pcawg/pcawg_bio_v4_features.csv.gz, pcawg_sigmaf_1130_features.csv.gz |
| C2 | Priority C frozen transfer, POG570 breast n=147 (Spearman, MAE, R squared, calibration, thresholds) | planned | matrix present (570 x 1,130); cohort rule from the archived analysis |
| C3 | Priority C annotation missingness and distribution audit by cohort | planned | table work on saved matrices and events |
| D1 | Priority D SBS3 and BRCA sparsity audit | completed | SBS3 nonzero 9 of 772 (data/task3/sbs3_activity_772.csv); BRCA indicator positive 14 of 772 (task3 rule on the Bio MAF matrix) |
| D2 | Priority D covariate analyses (burden, purity, ploidy, subtype) on matched subsets | planned | cohort table carries the covariates (771 of 772 complete) |
| D3 | SigMA or other exome-valid HRD comparator | planned | subject to input and training-overlap checks; no WGS-only method |
| E1 | Complementary endpoint screen (LUAD smoking first) | blocked | CDR supplement verified locally: 34 columns, no smoking or tobacco field. Missing prerequisite: a documented smoking history label source (for example the TCGA LUAD clinical record via GDC or cBioPortal) plus a download decision. PAM50 subtype calls also lack a documented local source. No complementary endpoint is adopted until labels exist locally and pass the circularity screen |
| E2 | Complementary endpoint full core comparison (if adopted) | planned | only if E1 supports it, decided on label quality and counts, never on test performance |
| F1 | Survival incremental analysis (clinical baseline vs plus gene features vs plus spectra and Bio MAF) | planned | conditional, after A to D per brief priority |
| F2 | MuAt reruns (3 seeds, architecture searched primary) | blocked | missing prerequisite: GPU hardware. Archived runs audited instead (audit report section 5) |
| F3 | Learning curves (25/50/75/100 percent) | planned | conditional on a sample-efficiency claim being retained |
| G1 | Results ledger (every manuscript number to source file, cohort, model, estimator) | planned | seeded by the same-day verification report; unsourced numbers listed in audit report section 9 |
| G2 | Claim audit table and collaborator report answering brief section 15 | planned | final package item |
| G3 | FAIR package: README, feature dictionary, environment spec, citation metadata | planned | no publication or DOI without Aaron's authorization |

## Resource note

Sequential execution on 2 CPU cores. The A5/A6 queue (46 cells) runs in the protocol's priority order, so the primary contrasts are covered first if the queue is interrupted.
