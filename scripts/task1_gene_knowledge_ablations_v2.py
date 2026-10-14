"""Task 1 (v2): Gene identity and knowledge controls, XGBoost, Task 0 folds.

Replaces the superseded scripts/task1_gene_knowledge_ablations.py and
scripts/task1_part2_ef_and_summary.py, which used xgboost 3.4.1 and a
non-native refit harness (see reviewer-analyses-notes.md "Tasks 1, 2, 4, 6").

Design (all verified before any model trains):
  * Cohort/labels/feature matrices rebuilt from results/tables/ and checked
    byte-for-byte against the archived checkpoint manifests via
    _matrix_digest (sha256 of shape + float32 bytes, hex[:24]).
  * Outer folds: archived sample_folds from the frozen checkpoints
    (MAF_stack_only seed 615371; signatures_plus_MAF_stack seed 750978).
  * (a) controls are the ARCHIVED OOF predictions (no refit): they match
    Task 0 by construction.
  * (b)-(f) refits use the archived per-fold nested-search winners
    (feature set, hyperparameters, m_star) with NO re-tuning, and the exact
    production final-refit seed schedule: split_seed + fold*1009 + 99991
    (src/utils/nested_oof.py _search_and_predict_feature_set_fold).
  * Pinned Table 1 environment: xgboost 1.6.2, sklearn 1.7.2, numpy 1.26.4,
    pandas 2.3.1, scipy 1.15.3 (~/workspace/task0_rerun/venv312/).
    tree_method='hist': production ran gpu_hist with the pipeline's automatic
    hist fallback on CPU (same fallback, same numerics).

Variants (rep in {maf, sigmaf}):
  a_full                  archived OOF (control; matches Task 0)
  b_no_knowledge          winner set minus 32 knowledge columns
  c_no_gene_identity      winner set minus 802 per-gene columns
  d_no_knowledge_no_gene  winner set minus both
  e_binary_genes          401 per-gene binary mutated/not indicators
                          (sigmaf: 182 signatures + 401 binaries)
  f_driver_gene_counts    401 per-gene role-matched driver log-counts
                          (sigmaf: 182 signatures + 401 counts)

Outputs (results/tables/):
  task1_gene_knowledge_ablations.csv   per-variant per-fold + pooled metrics
  task1_oof_<rep>_<variant>.csv       OOF predictions per variant
"""

import gzip
import hashlib
import json
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from xgboost import XGBClassifier

REPO = Path('/home/hatch/workspace/somatic-mutation-repo')
TABLES = REPO / 'results' / 'tables'
CKPT = TABLES / 'main_manuscript_complete_panel_nested_fold_checkpoints'
SIGCSV = Path('/home/hatch/workspace/task0_rerun/features/features_standard_sbs96_id83.csv.gz')
CLASSES = ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'GBM', 'HNSC', 'KIRC',
           'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PRAD', 'SARC',
           'SKCM', 'STAD', 'THCA', 'UCEC']
SEEDS = {'maf': 615371, 'sigmaf': 750978}
REPS = {'maf': 'MAF_stack_only', 'sigmaf': 'signatures_plus_MAF_stack'}

SET_BLOCKS = {
    'v4_compact_biology_only': ['v3_core_compact_biology'],
    'v4_compact_plus_driver_evidence_tiers':
        ['v3_core_compact_biology', 'external_driver_evidence_tier_aggregates'],
    'v4_compact_plus_exact_driver_genes':
        ['v3_core_compact_biology', 'exact_consensus_driver_gene_identity'],
    'v4_compact_plus_hotspot_summary':
        ['v3_core_compact_biology', 'cancer_hotspot_summary'],
    'v4_full_core': ['v3_core_compact_biology',
                     'external_driver_evidence_tier_aggregates',
                     'cancer_hotspot_summary',
                     'exact_consensus_driver_gene_identity'],
    'v4_full_core_plus_optional_controls':
        ['v3_core_compact_biology', 'external_driver_evidence_tier_aggregates',
         'cancer_hotspot_summary', 'exact_consensus_driver_gene_identity',
         'optional_burden_and_event_composition_controls'],
}

# MECHANICAL feature-classification rules (no case-by-case calls):
#   gene-identity: column name starts with one of the two per-gene prefixes.
#     Verified identical to the feature table's 'exact_consensus_driver_gene_identity' block.
#   knowledge: column name starts with oncokb_role_ / cancer_hotspot_ / driver_evidence_.
#     Covers OncoKB role features (12), hotspot flags (4), driver-evidence tiers (16).
KB_PREFIXES = ('oncokb_role_', 'cancer_hotspot_', 'driver_evidence_')
GENE_PREFIXES = ('driver_gene_functional_event_log_count__',
                 'driver_gene_role_matched_event_log_count__')


def matrix_digest(matrix):
    arr = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    d = hashlib.sha256()
    d.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    d.update(arr.tobytes())
    return d.hexdigest()[:24]


def log(msg):
    print(msg, flush=True)


def load_everything():
    t0 = time.time()
    m = pd.read_csv(TABLES / 'batch_center_sample_mapping.csv')
    with gzip.open(TABLES / 'quick_bio_v4_maf_features.csv.gz', 'rt') as f:
        cols = f.readline().strip().split(',')
        rows = {}
        for line in f:
            p = line.split(',')
            rows[p[0]] = p[1:]
    maf_cols = np.array(cols[1:])
    co = m[m['cancer_type_cdr'].isin(CLASSES)
           & m['sample'].isin(rows)].drop_duplicates('sample')
    samples = np.sort(co['sample'].unique())
    assert len(samples) == 8800, len(samples)
    y = pd.Categorical(co.set_index('sample').loc[samples, 'cancer_type_cdr'],
                       categories=CLASSES).codes
    X_maf = np.nan_to_num(
        np.array([[float(v) for v in rows[s]] for s in samples], dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0)

    # signature block: 182 cols, order from the frozen feature_order.txt
    order = [l.strip() for l in open(
        REPO / 'results/frozen_models/cancer_type_top20__signatures_plus_MAF_stack/feature_order.txt')
        if l.strip()]
    sig_order = [c for c in order if c.startswith(('sig__', 'SBS96__', 'ID83__'))]
    assert len(sig_order) == 182
    assert [c for c in order if c not in sig_order] == list(maf_cols), \
        'MAF column order mismatch vs feature_order.txt'
    with gzip.open(SIGCSV, 'rt') as f:
        scols = f.readline().strip().split(',')
        srows = {}
        for line in f:
            p = line.split(',')
            srows[p[0]] = p[1:]
    assert not [s for s in samples if s not in srows], 'sig file misses samples'
    sidx = {c: i for i, c in enumerate(scols[1:])}
    inv = {'sig__log10_sbs_burden': 'log10_sbs_burden',
           'sig__log10_id_burden': 'log10_id_burden',
           'sig__id_fraction': 'id_fraction'}
    X_sig = np.nan_to_num(
        np.array([[float(srows[s][sidx[inv.get(c, c)]]) for c in sig_order]
                  for s in samples], dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0)

    # digest verification against archived production manifests
    t = pd.read_csv(TABLES / 'proposed_clinical_bio_v4_feature_table_exact.csv')
    blocks = t.set_index('Feature').loc[maf_cols, 'Bio MAF v4 block'].values
    for rep, seed in SEEDS.items():
        fj = glob.glob(str(CKPT / f'cancer_type_top20__{REPS[rep]}__xgboost__*.json'))[0]
        man = json.load(open(fj))['manifest']
        assert man['seed'] == seed and man['n_samples'] == 8800
        if rep == 'maf':
            assert matrix_digest(X_maf) == \
                man['feature_digests']['v4_full_core_plus_optional_controls']
            assert matrix_digest(X_maf[:, np.isin(
                blocks, SET_BLOCKS['v4_full_core'])]) == \
                man['feature_digests']['v4_full_core']
        else:
            assert matrix_digest(X_sig) == man['feature_digests']['signatures_only']
            full1130 = np.concatenate([X_sig, X_maf], axis=1)
            assert matrix_digest(full1130) == \
                man['feature_digests']['signatures_plus_v4_full_core_plus_optional_controls']
            core943 = np.isin(blocks, SET_BLOCKS['v4_full_core'])
            assert matrix_digest(np.concatenate([X_sig, X_maf[:, core943]], axis=1)) == \
                man['feature_digests']['signatures_plus_v4_full_core']
    log(f'digest verification PASSED (maf 948 + sig 182), load {time.time()-t0:.0f}s')

    # mechanical classification + cross-checks
    kb_mask = np.array([c.startswith(KB_PREFIXES) for c in maf_cols])
    gene_mask = np.array([c.startswith(GENE_PREFIXES) for c in maf_cols])
    assert (gene_mask == (blocks == 'exact_consensus_driver_gene_identity')).all(), \
        'gene-identity rule != exact_consensus_driver_gene_identity block'
    assert kb_mask.sum() == 32 and gene_mask.sum() == 802, \
        (kb_mask.sum(), gene_mask.sum())
    assert (kb_mask & gene_mask).sum() == 0
    func_mask = np.array([c.startswith('driver_gene_functional_event_log_count__')
                          for c in maf_cols])
    role_mask = np.array([c.startswith('driver_gene_role_matched_event_log_count__')
                          for c in maf_cols])
    genes = sorted(c.split('__')[-1] for c in maf_cols[func_mask])
    assert len(genes) == 401 and genes == sorted(
        c.split('__')[-1] for c in maf_cols[role_mask])
    log(f'knowledge cols: 32, gene-identity cols: 802, genes: 401, overlap: 0')
    return samples, y, X_maf, X_sig, maf_cols, blocks, kb_mask, gene_mask, func_mask, role_mask, genes


def load_rep_info(rep):
    fj = glob.glob(str(CKPT / f'cancer_type_top20__{REPS[rep]}__xgboost__*.json'))[0]
    npz = glob.glob(str(CKPT / f'cancer_type_top20__{REPS[rep]}__xgboost__*.npz'))[0]
    winners = json.load(open(fj))['fold_rows']
    r = np.load(npz, allow_pickle=True)
    folds = r['sample_folds']
    assert folds.shape == (8800,)
    info = {}
    for row in winners:
        f = int(row['fold'])
        info[f] = {'set': row['selected_feature_set'],
                   'params': json.loads(row['selected_params_json']),
                   'm_star': int(row['selected_m_star'])}
    return folds, info, r['pred']


def final_seed(rep, fold):
    return SEEDS[rep] + fold * 1009 + 99991


def fit_fold(X_tr, y_tr, X_te, params, m_star, seed):
    clf = XGBClassifier(
        objective='multi:softprob', eval_metric='mlogloss', num_class=20,
        n_estimators=int(m_star),
        max_depth=int(params['max_depth']),
        learning_rate=float(params['learning_rate']),
        subsample=float(params['subsample']),
        colsample_bytree=float(params['colsample_bytree']),
        min_child_weight=float(params['min_child_weight']),
        reg_alpha=float(params['reg_alpha']),
        reg_lambda=float(params['reg_lambda']),
        random_state=int(seed), n_jobs=4, tree_method='hist', verbosity=0)
    clf.fit(np.ascontiguousarray(X_tr), y_tr)
    assert list(clf.classes_) == list(range(20)), clf.classes_
    proba = np.asarray(clf.predict_proba(np.ascontiguousarray(X_te)), dtype=np.float64)
    # mirror src/utils/nested_oof.py::_coerce_proba (NaN->1/20, then renormalize)
    proba = np.nan_to_num(proba, nan=1.0 / 20, posinf=1.0, neginf=0.0)
    rs = proba.sum(axis=1, keepdims=True)
    proba = np.divide(proba, rs, out=np.full_like(proba, 1.0 / 20), where=rs > 0)
    return proba


def write_oof_csv(path, samples, y, proba, folds):
    pred_lab = proba.argmax(axis=1)
    df = pd.DataFrame({'sample': samples,
                       'true_label': [CLASSES[c] for c in y],
                       'pred_label': [CLASSES[c] for c in pred_lab],
                       'fold': folds.astype(int)})
    for i, c in enumerate(CLASSES):
        df[f'proba_{c}'] = proba[:, i]
    df.to_csv(path, index=False)


def main():
    t_start = time.time()
    samples, y, X_maf, X_sig, maf_cols, blocks, kb_mask, gene_mask, \
        func_mask, role_mask, genes = load_everything()
    X_binary = (X_maf[:, func_mask] > 0).astype(np.float32)
    X_driver = X_maf[:, role_mask]

    variants = ['a_full', 'b_no_knowledge', 'c_no_gene_identity',
                'd_no_knowledge_no_gene', 'e_binary_genes', 'f_driver_gene_counts']
    summary_rows = []

    for rep in ['maf', 'sigmaf']:
        folds, info, arch_proba = load_rep_info(rep)
        sig = rep == 'sigmaf'
        seed = SEEDS[rep]

        # per-fold base column masks (over 948 MAF cols; sig block always kept)
        base_masks = {}
        for f in range(1, 6):
            set_name = info[f]['set']
            maf_set = set_name.replace('signatures_plus_', '', 1) if sig else set_name
            base_masks[f] = np.isin(blocks, SET_BLOCKS[maf_set])

        def fold_matrix(f, variant):
            bm = base_masks[f]
            if variant == 'a_full':
                return None  # archived OOF, no refit
            if variant == 'b_no_knowledge':
                mm = bm & ~kb_mask
            elif variant == 'c_no_gene_identity':
                mm = bm & ~gene_mask
            elif variant == 'd_no_knowledge_no_gene':
                mm = bm & ~kb_mask & ~gene_mask
            elif variant == 'e_binary_genes':
                return np.concatenate([X_sig, X_binary], axis=1) if sig else X_binary
            elif variant == 'f_driver_gene_counts':
                return np.concatenate([X_sig, X_driver], axis=1) if sig else X_driver
            Xv = X_maf[:, mm]
            return np.concatenate([X_sig, Xv], axis=1) if sig else Xv

        for v in variants:
            tv = time.time()
            if v == 'a_full':
                proba, vfolds = arch_proba, folds
                n_feat = 948 if not sig else 1130
            else:
                proba = np.zeros((8800, 20))
                n_feat_f = {}
                for f in range(1, 6):
                    Xv = fold_matrix(f, v)
                    te = np.where(folds == f)[0]
                    tr = np.where(folds != f)[0]
                    n_feat_f[f] = Xv.shape[1]
                    pf = fit_fold(Xv[tr], y[tr], Xv[te], info[f]['params'],
                                  info[f]['m_star'], final_seed(rep, f))
                    proba[te] = pf
                    ba = balanced_accuracy_score(y[te], pf.argmax(1))
                    acc = accuracy_score(y[te], pf.argmax(1))
                    summary_rows.append({'representation': rep, 'variant': v,
                                         'fold': f, 'n': len(te),
                                         'n_features': Xv.shape[1],
                                         'balanced_accuracy': ba, 'accuracy': acc})
                    log(f'{rep}/{v} fold {f}: BA={ba:.6f} acc={acc:.6f} '
                        f'({time.time()-tv:.0f}s)')
                vfolds = folds
                n_feat = int(round(float(np.mean(list(n_feat_f.values())))))
            pred_lab = proba.argmax(1)
            for f in range(1, 6):
                te = np.where(vfolds == f)[0]
                if v != 'a_full':
                    continue  # already recorded per-fold above
                summary_rows.append({'representation': rep, 'variant': v,
                                     'fold': f, 'n': len(te),
                                     'n_features': n_feat,
                                     'balanced_accuracy': balanced_accuracy_score(
                                         y[te], pred_lab[te]),
                                     'accuracy': accuracy_score(y[te], pred_lab[te])})
            summary_rows.append({'representation': rep, 'variant': v,
                                 'fold': 'pooled', 'n': 8800,
                                 'n_features': n_feat,
                                 'balanced_accuracy': balanced_accuracy_score(y, pred_lab),
                                 'accuracy': accuracy_score(y, pred_lab)})
            write_oof_csv(TABLES / f'task1_oof_{rep}_{v}.csv',
                          samples, y, proba, vfolds)
            log(f'{rep}/{v} pooled BA={summary_rows[-1]["balanced_accuracy"]:.6f} '
                f'({"archived" if v == "a_full" else "refit"}) '
                f'total {time.time()-t_start:.0f}s')

    summary = pd.DataFrame(summary_rows,
                           columns=['representation', 'variant', 'fold', 'n',
                                    'n_features', 'balanced_accuracy', 'accuracy'])
    summary.to_csv(TABLES / 'task1_gene_knowledge_ablations.csv', index=False)
    log(summary.to_string(index=False))
    log(f'DONE in {time.time()-t_start:.0f}s')


if __name__ == '__main__':
    main()
