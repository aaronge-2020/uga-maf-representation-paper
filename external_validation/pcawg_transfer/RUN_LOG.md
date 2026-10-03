# PCAWG Sig+MAF transfer — run log (2026-10-01)

## Provenance
- Source: PCAWG public consensus MAF (final_consensus_passonly.snv_mnv_indel.icgc.public.maf.gz),
  mirrored from https://object.genomeinformatics.org, downloaded 2026-10-01.
- Filtered to 913 samples with ICGC histology mappable to the 20 TCGA training classes.
- Frozen model: results/frozen_models/cancer_type_top20__signatures_plus_MAF_stack (1,130 features).

## Pipeline stages
1. **Filter** (done): dedup/pcawg_913.maf.gz = 17,471,084 rows, 913 samples.
2. **GRCh37 fasta** (done): raw/human_g1k_v37.fasta (+ .fai.json).
3. **Hybrid annotation** (done): stage3_hybrid_annot.py -> annot/pcawg_913.hybrid.maf.gz.
   Consequence/IMPACT from ICGC Variant_Classification; BIOTYPE from GENCODE v19;
   CANONICAL yes/unknown; VAF from t_alt_count/t_ref_count.
4. **Hotspot VEP** (done): 2,714 coding variants in 248 hotspot genes annotated via
   Ensembl GRCh37 REST VEP (curl) -> annot/vep_rest_annotations.tsv.gz.
   Joined -> annot/pcawg_913.annotated.maf.gz (2,481 rows hit).
5. **Bio features** (running): stage5_build_bio.py. Barcodes rewritten to synth_id
   (raw/pcawg_913_synth_mapping.csv) to avoid first_12 prefix collisions (6 samples).
6. **Assemble** (pending): stage6_assemble.py -> features/pcawg_sigmaf_1130_features.csv.gz.
7. **Score** (pending): stage7_score.py -> predictions/pcawg_sigmaf_predictions.csv,
   predictions/pcawg_sigmaf_metrics.json (bootstrap 95% CIs, n=2000, seed 42).

## Key approximations (documented in METHODS_paragraph.txt)
- Consequence/IMPACT: deterministic ICGC Variant_Classification map (0 unmapped of 17.47M).
- BIOTYPE: GENCODE v19 gene lookup.
- SIFT/PolyPhen/protein positions: only for 2,714 hotspot-gene coding variants (REST VEP);
  non-hotspot missense left as unknown (vep_sift/vep_polyphen gain ~4.7, minimal impact).
- CANONICAL: yes if genic else unknown.
