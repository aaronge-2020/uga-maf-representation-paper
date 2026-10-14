"""Task 1 pilot: refit archived Table 1 fold 1 (MAF_stack_only, xgboost) and compare.

Verifies that the local training harness reproduces the archived OOF predictions
within the accepted environmental tolerance before running ablations.
"""
import gzip
import json
import glob
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

REPO = '/home/hatch/workspace/somatic-mutation-repo'
CLASSES = ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'GBM', 'HNSC', 'KIRC', 'KIRP',
           'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PRAD', 'SARC', 'SKCM', 'STAD',
           'THCA', 'UCEC']
SEED = 750978
FINAL_SEED = SEED + 99991  # per src/utils/nested_oof.py final refit


def load_cohort():
    t0 = time.time()
    m = pd.read_csv(f'{REPO}/results/tables/batch_center_sample_mapping.csv')
    with gzip.open(f'{REPO}/results/tables/quick_bio_maf_features.csv.gz', 'rt') as f:
        cols = f.readline().strip().split(',')
        rows = {}
        for line in f:
            p = line.split(',')
            rows[p[0]] = p[1:]
    feat_cols = cols[1:]
    co = m[m['cancer_type_cdr'].isin(CLASSES) & m['sample'].isin(rows)].drop_duplicates('sample')
    samples = np.sort(co['sample'].unique())
    y = pd.Categorical(co.set_index('sample').loc[samples, 'cancer_type_cdr'],
                       categories=CLASSES).codes
    X = np.array([[float(v) for v in rows[s]] for s in samples], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f'load: {time.time()-t0:.1f}s  X={X.shape}', flush=True)
    return samples, y, X, feat_cols


def main():
    samples, y, X, feat_cols = load_cohort()
    fn = glob.glob(f'{REPO}/results/tables/main_manuscript_complete_panel_nested_fold_checkpoints/'
                    f'cancer_type_top20__MAF_stack_only__xgboost__*.npz')[0]
    r = np.load(fn, allow_pickle=True)
    sf = r['sample_folds']
    arch_pred = r['pred']
    fj = glob.glob(f'{REPO}/results/tables/main_manuscript_complete_panel_nested_fold_checkpoints/'
                    f'cancer_type_top20__MAF_stack_only__xgboost__*.json')[0]
    fold1 = json.load(open(fj))['fold_rows'][0]
    te = np.where(sf == 1)[0]
    print('archived fold1 outer BA:',
          round(balanced_accuracy_score(y[te], arch_pred[te].argmax(1)), 6), flush=True)

    t = pd.read_csv(f'{REPO}/results/tables/proposed_clinical_bio_feature_table_exact.csv').set_index('Feature')
    blocks = t.loc[feat_cols, 'Bio MAF block']
    mask = blocks.isin(['v3_core_compact_biology',
                        'exact_consensus_driver_gene_identity']).values
    print('fold1 winner cols:', mask.sum(), flush=True)
    params = json.loads(fold1['selected_params_json'])
    m_star = int(fold1['selected_m_star'])
    tr = np.where(sf != 1)[0]

    t1 = time.time()
    clf = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss',
                        num_class=20, n_estimators=m_star,
                        max_depth=int(params['max_depth']),
                        learning_rate=float(params['learning_rate']),
                        subsample=float(params['subsample']),
                        colsample_bytree=float(params['colsample_bytree']),
                        min_child_weight=float(params['min_child_weight']),
                        reg_alpha=float(params['reg_alpha']),
                        reg_lambda=float(params['reg_lambda']),
                        random_state=FINAL_SEED, n_jobs=2,
                        tree_method='hist', verbosity=0)
    clf.fit(X[tr][:, mask], y[tr])
    proba = np.nan_to_num(np.asarray(clf.predict_proba(X[te][:, mask]),
                                     dtype=np.float64),
                          nan=1.0 / 20, posinf=1.0, neginf=0.0)
    print(f'fit+predict: {time.time()-t1:.1f}s', flush=True)
    print('refit fold1 BA:', round(balanced_accuracy_score(y[te], proba.argmax(1)), 6), flush=True)
    print('max abs proba diff vs archived:',
          float(np.abs(arch_pred[te] - proba).max()), flush=True)


if __name__ == '__main__':
    main()
