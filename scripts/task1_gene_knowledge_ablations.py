"""Task 1: Gene identity and knowledge controls (cancer typing, XGBoost).

Question: how much of the Sig+MAF gain is gene identity and curated cancer
knowledge, and how much is mutation annotation that carries no gene name?

Uses ONLY the archived original five-fold stratified splits that produced
manuscript Table 1 (n = 8,800 tumours, 20 cancer types), recovered from the
frozen fold checkpoints:
  results/tables/main_manuscript_complete_panel_nested_fold_checkpoints/
    cancer_type_top20__MAF_stack_only__xgboost__*.{json,npz}
Cohort/label/fold/feature-matrix digests all verified against the checkpoint
manifests before any model is trained (see results/tables/task1_notes.md).

XGBoost construction mirrors src/utils/nested_oof.py::_xgb_model /
_xgb_fit_predict for the final outer refit:
  XGBClassifier(objective='multi:softprob', eval_metric='mlogloss',
                num_class=20, n_estimators=<per-fold archived m_star>,
                <per-fold archived winner hyperparameters>,
                random_state=615371+fold*1009+99991, tree_method='hist')
on NaN->0-filled float32 features. Per-fold winner feature sets and
hyperparameters are the archived Table 1 nested-search selections; ablations
drop columns from those sets without re-tuning.

Variants (all MAF_stack_only unless noted):
  a_maf_full        full features, per-fold archived winner sets (control refit)
  b_maf_kb          (a) minus knowledge features (32 cols)
  c_maf_gene        (a) minus gene-identity features (802 cols)
  d_maf_both        (a) minus both (834 cols)
  e_binary          401 binary per-gene indicators (functional event count > 0)
  f_driver_counts   401 per-gene role-matched driver log-counts
Archived OOF (no refit) is exported for a_maf_full and a_sigmaf_full.

Sig+MAF ablations and the signatures_only baseline cannot run on this machine:
the 182-channel TCGA signature matrix is not on disk (see task1_notes.md).

Outputs (results/tables/):
  task1_gene_knowledge_ablations.csv      per-variant per-fold + pooled metrics
  task1_oof_<variant>.csv                out-of-fold predictions per variant
  task1_notes.md                         method, feature lists, deviations
"""

import gzip
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
CLASSES = ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'GBM', 'HNSC', 'KIRC',
           'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PRAD', 'SARC',
           'SKCM', 'STAD', 'THCA', 'UCEC']
# Archived outer-split seeds (from the frozen fold-checkpoint manifests):
#   MAF_stack_only            -> 615371
#   signatures_plus_MAF_stack -> 750978
# Per-fold seed = split seed + fold*1009; final outer refit adds 99991
# (src/utils/nested_oof.py).
MAF_SEED = 615371
SIGMAF_SEED = 750978


def maf_final_seed(fold: int) -> int:
    return MAF_SEED + fold * 1009 + 99991

# Candidate feature-set block membership (verified by digest against the
# archived checkpoint manifests).
SET_BLOCKS = {
    'compact_biology_only': ['v3_core_compact_biology'],
    'v4_compact_plus_driver_evidence_tiers':
        ['v3_core_compact_biology', 'external_driver_evidence_tier_aggregates'],
    'v4_compact_plus_exact_driver_genes':
        ['v3_core_compact_biology', 'exact_consensus_driver_gene_identity'],
    'v4_compact_plus_hotspot_summary':
        ['v3_core_compact_biology', 'cancer_hotspot_summary'],
    'full_core': ['v3_core_compact_biology',
                     'external_driver_evidence_tier_aggregates',
                     'cancer_hotspot_summary',
                     'exact_consensus_driver_gene_identity'],
    'full_core_plus_optional_controls':
        ['v3_core_compact_biology', 'external_driver_evidence_tier_aggregates',
         'cancer_hotspot_summary', 'exact_consensus_driver_gene_identity',
         'optional_burden_and_event_composition_controls'],
}

KB_PREFIXES = ('oncokb_role_', 'cancer_hotspot_', 'driver_evidence_')
GENE_PREFIXES = ('driver_gene_functional_event_log_count__',
                 'driver_gene_role_matched_event_log_count__')


def is_knowledge(col: str) -> bool:
    return col.startswith(KB_PREFIXES)


def is_gene_identity(col: str) -> bool:
    return col.startswith(GENE_PREFIXES)


def load_cohort():
    """Reconstruct the 8,800-tumour cohort, labels, folds, feature matrix."""
    t0 = time.time()
    m = pd.read_csv(TABLES / 'batch_center_sample_mapping.csv')
    with gzip.open(TABLES / 'quick_bio_maf_features.csv.gz', 'rt') as f:
        cols = f.readline().strip().split(',')
        rows = {}
        for line in f:
            p = line.split(',')
            rows[p[0]] = p[1:]
    feat_cols = cols[1:]
    co = m[m['cancer_type_cdr'].isin(CLASSES)
           & m['sample'].isin(rows)].drop_duplicates('sample')
    samples = np.sort(co['sample'].unique())
    assert len(samples) == 8800, len(samples)
    y = pd.Categorical(co.set_index('sample').loc[samples, 'cancer_type_cdr'],
                       categories=CLASSES).codes
    X = np.array([[float(v) for v in rows[s]] for s in samples],
                 dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    ckpt = glob.glob(str(TABLES / 'main_manuscript_complete_panel_nested_fold_checkpoints'
                         / 'cancer_type_top20__MAF_stack_only__xgboost__*.npz'))[0]
    r = np.load(ckpt, allow_pickle=True)
    folds = r['sample_folds']
    assert folds.shape == (8800,)
    print(f'cohort loaded in {time.time()-t0:.1f}s', flush=True)
    return samples, y, X, feat_cols, folds


def load_archived_oof(rep):
    ckpt = glob.glob(str(TABLES / 'main_manuscript_complete_panel_nested_fold_checkpoints'
                         / f'cancer_type_top20__{rep}__xgboost__*.npz'))[0]
    r = np.load(ckpt, allow_pickle=True)
    return r['pred'], r['sample_folds']


def load_fold_winners():
    fj = glob.glob(str(TABLES / 'main_manuscript_complete_panel_nested_fold_checkpoints'
                       / 'cancer_type_top20__MAF_stack_only__xgboost__*.json'))[0]
    return json.load(open(fj))['fold_rows']


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
        random_state=seed, n_jobs=2, tree_method='hist', verbosity=0)
    clf.fit(X_tr, y_tr)
    assert list(clf.classes_) == list(range(20)), clf.classes_
    proba = np.nan_to_num(np.asarray(clf.predict_proba(X_te), dtype=np.float64),
                          nan=1.0 / 20, posinf=1.0, neginf=0.0)
    return proba


def write_oof_csv(path, samples, y, proba, folds):
    pred_lab = proba.argmax(axis=1)
    df = pd.DataFrame({'sample': samples,
                       'true_label': [CLASSES[c] for c in y],
                       'pred_label': [CLASSES[c] for c in pred_lab],
                       'fold': folds})
    for i, c in enumerate(CLASSES):
        df[f'proba_{c}'] = proba[:, i]
    df.to_csv(path, index=False)


def main():
    t_start = time.time()
    samples, y, X, feat_cols, folds = load_cohort()
    feat_cols = np.array(feat_cols)

    t = pd.read_csv(TABLES / 'proposed_clinical_bio_feature_table_exact.csv')
    blocks = t.set_index('Feature').loc[feat_cols, 'Bio MAF block'].values

    kb_mask = np.array([is_knowledge(c) for c in feat_cols])
    gene_mask = np.array([is_gene_identity(c) for c in feat_cols])
    print(f'knowledge cols: {kb_mask.sum()}, gene-identity cols: {gene_mask.sum()}, '
          f'overlap: {(kb_mask & gene_mask).sum()}', flush=True)
    # cross-check gene rule against the feature-table block (must be identical)
    block_gene = blocks == 'exact_consensus_driver_gene_identity'
    assert (gene_mask == block_gene).all(), 'gene-identity rule != block definition'

    func_mask = np.array([c.startswith('driver_gene_functional_event_log_count__')
                          for c in feat_cols])
    role_mask = np.array([c.startswith('driver_gene_role_matched_event_log_count__')
                          for c in feat_cols])
    genes_func = sorted(c.split('__')[-1] for c in feat_cols[func_mask])
    genes_role = sorted(c.split('__')[-1] for c in feat_cols[role_mask])
    assert genes_func == genes_role and len(genes_func) == 401
    print('per-gene genes:', len(genes_func), flush=True)

    X_binary = (X[:, func_mask] > 0).astype(np.float32)   # (e)
    X_driver = X[:, role_mask]                            # (f)

    winners = load_fold_winners()
    fold_info = {}
    for row in winners:
        f = int(row['fold'])
        fold_info[f] = {'set': row['selected_feature_set'],
                        'params': json.loads(row['selected_params_json']),
                        'm_star': int(row['selected_m_star'])}

    # variant -> per-fold column mask (over the 948 columns), or special
    col_masks = {}
    for f in range(1, 6):
        base = np.isin(blocks, SET_BLOCKS[fold_info[f]['set']])
        col_masks[f] = {
            'a_maf_full': base,
            'b_maf_kb': base & ~kb_mask,
            'c_maf_gene': base & ~gene_mask,
            'd_maf_both': base & ~kb_mask & ~gene_mask,
        }
    variants = ['a_maf_full', 'b_maf_kb', 'c_maf_gene', 'd_maf_both',
                'e_binary', 'f_driver_counts']

    oof = {v: np.zeros((8800, 20)) for v in variants}
    nfeat = {v: {} for v in variants}
    summary_rows = []

    for v in variants:
        tv = time.time()
        for f in range(1, 6):
            te = np.where(folds == f)[0]
            tr = np.where(folds != f)[0]
            info = fold_info[f]
            if v in col_masks[f]:
                mask = col_masks[f][v]
                Xtr, Xte = X[tr][:, mask], X[te][:, mask]
            elif v == 'e_binary':
                Xtr, Xte = X_binary[tr], X_binary[te]
            else:
                Xtr, Xte = X_driver[tr], X_driver[te]
            nfeat[v][f] = Xtr.shape[1]
            proba = fit_fold(Xtr, y[tr], Xte, info['params'], info['m_star'],
                             maf_final_seed(f))
            oof[v][te] = proba
            ba = balanced_accuracy_score(y[te], proba.argmax(1))
            acc = accuracy_score(y[te], proba.argmax(1))
            summary_rows.append({'variant': v, 'fold': f, 'n': len(te),
                                 'n_features': Xtr.shape[1],
                                 'balanced_accuracy': ba, 'accuracy': acc})
            print(f'{v} fold {f}: BA={ba:.6f} acc={acc:.6f} '
                  f'({time.time()-tv:.0f}s)', flush=True)
        pred_lab = oof[v].argmax(1)
        summary_rows.append({'variant': v, 'fold': 'pooled', 'n': 8800,
                             'n_features': int(np.mean(list(nfeat[v].values()))),
                             'balanced_accuracy': balanced_accuracy_score(y, pred_lab),
                             'accuracy': accuracy_score(y, pred_lab)})
        write_oof_csv(TABLES / f'task1_oof_{v}.csv', samples, y, oof[v], folds)
        print(f'{v} pooled BA={summary_rows[-1]["balanced_accuracy"]:.6f} '
              f'written ({time.time()-t_start:.0f}s total)', flush=True)
    # archived OOF exports (no refit): (a) controls.
    # NOTE: each representation has its own archived outer splits; the fold
    # column in each CSV reflects that variant's own folds.
    for rep, v in [('MAF_stack_only', 'a_maf_full_archived'),
                   ('signatures_plus_MAF_stack', 'a_sigmaf_full_archived')]:
        proba, rep_folds = load_archived_oof(rep)
        pred_lab = proba.argmax(1)
        for f in range(1, 6):
            te = np.where(rep_folds == f)[0]
            summary_rows.append({
                'variant': v, 'fold': f, 'n': len(te),
                'n_features': 948 if rep == 'MAF_stack_only' else 1130,
                'balanced_accuracy': balanced_accuracy_score(y[te], pred_lab[te]),
                'accuracy': accuracy_score(y[te], pred_lab[te])})
        summary_rows.append({
            'variant': v, 'fold': 'pooled', 'n': 8800,
            'n_features': 948 if rep == 'MAF_stack_only' else 1130,
            'balanced_accuracy': balanced_accuracy_score(y, pred_lab),
            'accuracy': accuracy_score(y, pred_lab)})
        write_oof_csv(TABLES / f'task1_oof_{v}.csv', samples, y, proba, rep_folds)
        print(f'{v} pooled BA={summary_rows[-1]["balanced_accuracy"]:.6f} '
              f'(archived)', flush=True)

    summary = pd.DataFrame(summary_rows,
                           columns=['variant', 'fold', 'n', 'n_features',
                                    'balanced_accuracy', 'accuracy'])
    summary.to_csv(TABLES / 'task1_gene_knowledge_ablations.csv', index=False)
    print(summary.to_string(index=False), flush=True)
    print(f'DONE in {time.time()-t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
