"""Task 1, part 2: train (e)/(f), export archived OOFs, write the summary CSV.

Part 1 (scripts/task1_gene_knowledge_ablations.py) completed (a)-(d) refits and
their OOF CSVs before the worker was killed; this script finishes the rest
without retraining (a)-(d): their summary rows are recomputed from the
already-written OOF CSVs.
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
LOGS = REPO / 'results' / 'logs'
CLASSES = ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'GBM', 'HNSC', 'KIRC',
           'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PRAD', 'SARC',
           'SKCM', 'STAD', 'THCA', 'UCEC']
MAF_SEED = 615371


def maf_final_seed(fold: int) -> int:
    return MAF_SEED + fold * 1009 + 99991


def log(msg):
    print(msg, flush=True)


def load_cohort():
    t0 = time.time()
    m = pd.read_csv(TABLES / 'batch_center_sample_mapping.csv')
    with gzip.open(TABLES / 'quick_bio_maf_features.csv.gz', 'rt') as f:
        cols = f.readline().strip().split(',')
        rows = {}
        for line in f:
            p = line.split(',')
            rows[p[0]] = p[1:]
    feat_cols = np.array(cols[1:])
    co = m[m['cancer_type_cdr'].isin(CLASSES)
           & m['sample'].isin(rows)].drop_duplicates('sample')
    samples = np.sort(co['sample'].unique())
    assert len(samples) == 8800
    y = pd.Categorical(co.set_index('sample').loc[samples, 'cancer_type_cdr'],
                       categories=CLASSES).codes
    X = np.array([[float(v) for v in rows[s]] for s in samples],
                 dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    ckpt = glob.glob(str(TABLES / 'main_manuscript_complete_panel_nested_fold_checkpoints'
                         / 'cancer_type_top20__MAF_stack_only__xgboost__*.npz'))[0]
    folds = np.load(ckpt, allow_pickle=True)['sample_folds']
    log(f'cohort loaded in {time.time()-t0:.1f}s')
    return samples, y, X, feat_cols, folds


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


def oof_from_csv(path, samples, y):
    df = pd.read_csv(path)
    assert (df['sample'].values == samples).all()
    assert (df['true_label'].values == np.array(CLASSES)[y]).all()
    proba = df[[f'proba_{c}' for c in CLASSES]].values
    return proba, df['fold'].values.astype(int)


def main():
    t_start = time.time()
    samples, y, X, feat_cols, folds = load_cohort()

    func_mask = np.array([c.startswith('driver_gene_functional_event_log_count__')
                          for c in feat_cols])
    role_mask = np.array([c.startswith('driver_gene_role_matched_event_log_count__')
                          for c in feat_cols])
    assert func_mask.sum() == 401 and role_mask.sum() == 401
    X_binary = (X[:, func_mask] > 0).astype(np.float32)
    X_driver = X[:, role_mask]

    fj = glob.glob(str(TABLES / 'main_manuscript_complete_panel_nested_fold_checkpoints'
                       / 'cancer_type_top20__MAF_stack_only__xgboost__*.json'))[0]
    winners = json.load(open(fj))['fold_rows']
    fold_info = {int(r['fold']): {'params': json.loads(r['selected_params_json']),
                                  'm_star': int(r['selected_m_star']),
                                  'set': r['selected_feature_set']}
                 for r in winners}

    # per-fold feature counts for (a)-(d), same masks as part 1
    t = pd.read_csv(TABLES / 'proposed_clinical_bio_feature_table_exact.csv')
    blocks = t.set_index('Feature').loc[feat_cols, 'Bio MAF block'].values
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
    kb_mask = np.array([c.startswith(('oncokb_role_', 'cancer_hotspot_',
                                      'driver_evidence_')) for c in feat_cols])
    gene_mask = np.array([c.startswith(('driver_gene_functional_event_log_count__',
                                        'driver_gene_role_matched_event_log_count__'))
                          for c in feat_cols])
    nfeat = {}
    for f in range(1, 6):
        base = np.isin(blocks, SET_BLOCKS[fold_info[f]['set']])
        nfeat[f] = {'a_maf_full': int(base.sum()),
                    'b_maf_kb': int((base & ~kb_mask).sum()),
                    'c_maf_gene': int((base & ~gene_mask).sum()),
                    'd_maf_both': int((base & ~kb_mask & ~gene_mask).sum())}
    log(f'per-fold n_features: {nfeat}')

    summary_rows = []

    # (a)-(d): recompute summary rows from already-written OOF CSVs (no retrain)
    for v in ['a_maf_full', 'b_maf_kb', 'c_maf_gene', 'd_maf_both']:
        proba, f_ = oof_from_csv(TABLES / f'task1_oof_{v}.csv', samples, y)
        pred_lab = proba.argmax(1)
        for f in range(1, 6):
            te = np.where(f_ == f)[0]
            summary_rows.append({
                'variant': v, 'fold': f, 'n': len(te),
                'n_features': nfeat[f][v],
                'balanced_accuracy': balanced_accuracy_score(y[te], pred_lab[te]),
                'accuracy': accuracy_score(y[te], pred_lab[te])})
        summary_rows.append({
            'variant': v, 'fold': 'pooled', 'n': 8800,
            'n_features': int(round(float(np.mean([nfeat[f][v]
                                                   for f in range(1, 6)])))),
            'balanced_accuracy': balanced_accuracy_score(y, pred_lab),
            'accuracy': accuracy_score(y, pred_lab)})
        log(f'{v} recomputed from OOF CSV: pooled BA='
            f'{summary_rows[-1]["balanced_accuracy"]:.6f}')

    # (e), (f): train
    for v, Xv in [('e_binary', X_binary), ('f_driver_counts', X_driver)]:
        oof = np.zeros((8800, 20))
        for f in range(1, 6):
            te = np.where(folds == f)[0]
            tr = np.where(folds != f)[0]
            info = fold_info[f]
            tv = time.time()
            proba = fit_fold(Xv[tr], y[tr], Xv[te], info['params'],
                             info['m_star'], maf_final_seed(f))
            oof[te] = proba
            ba = balanced_accuracy_score(y[te], proba.argmax(1))
            acc = accuracy_score(y[te], proba.argmax(1))
            summary_rows.append({'variant': v, 'fold': f, 'n': len(te),
                                 'n_features': 401,
                                 'balanced_accuracy': ba, 'accuracy': acc})
            log(f'{v} fold {f}: BA={ba:.6f} acc={acc:.6f} ({time.time()-tv:.0f}s)')
        pred_lab = oof.argmax(1)
        summary_rows.append({'variant': v, 'fold': 'pooled', 'n': 8800,
                             'n_features': 401,
                             'balanced_accuracy': balanced_accuracy_score(y, pred_lab),
                             'accuracy': accuracy_score(y, pred_lab)})
        write_oof_csv(TABLES / f'task1_oof_{v}.csv', samples, y, oof, folds)
        log(f'{v} pooled BA={summary_rows[-1]["balanced_accuracy"]:.6f} written')

    # archived OOF exports (no refit)
    for rep, v, nf in [('MAF_stack_only', 'a_maf_full_archived', 948),
                       ('signatures_plus_MAF_stack', 'a_sigmaf_full_archived', 1130)]:
        ckpt = glob.glob(str(TABLES / 'main_manuscript_complete_panel_nested_fold_checkpoints'
                             / f'cancer_type_top20__{rep}__xgboost__*.npz'))[0]
        r = np.load(ckpt, allow_pickle=True)
        proba, rep_folds = r['pred'], r['sample_folds']
        pred_lab = proba.argmax(1)
        for f in range(1, 6):
            te = np.where(rep_folds == f)[0]
            summary_rows.append({
                'variant': v, 'fold': f, 'n': len(te), 'n_features': nf,
                'balanced_accuracy': balanced_accuracy_score(y[te], pred_lab[te]),
                'accuracy': accuracy_score(y[te], pred_lab[te])})
        summary_rows.append({
            'variant': v, 'fold': 'pooled', 'n': 8800, 'n_features': nf,
            'balanced_accuracy': balanced_accuracy_score(y, pred_lab),
            'accuracy': accuracy_score(y, pred_lab)})
        write_oof_csv(TABLES / f'task1_oof_{v}.csv', samples, y, proba, rep_folds)
        log(f'{v} pooled BA={summary_rows[-1]["balanced_accuracy"]:.6f} (archived)')

    summary = pd.DataFrame(summary_rows,
                           columns=['variant', 'fold', 'n', 'n_features',
                                    'balanced_accuracy', 'accuracy'])
    summary.to_csv(TABLES / 'task1_gene_knowledge_ablations.csv', index=False)
    log(summary.to_string(index=False))
    log(f'DONE in {time.time()-t_start:.0f}s')


if __name__ == '__main__':
    main()
