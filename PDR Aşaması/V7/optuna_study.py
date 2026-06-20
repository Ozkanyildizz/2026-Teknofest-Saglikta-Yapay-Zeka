"""
AlgoMed V7 — optuna_study.py
==============================
Sadece Optuna optimizasyonunu çalıştırmak için yardımcı script.
train.py'den bağımsız olarak çalıştırılabilir.

Kullanım:
  python optuna_study.py --trials 700
  python optuna_study.py --trials 1000 --model xgb
  python optuna_study.py --trials 500 --model cat

Sonuçlar v7_optuna_results.json'a kaydedilir.
Ardından train.py'de RUN_OPTUNA=False ile bu sonuçlar kullanılabilir.
"""

import argparse
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold

import lightgbm as lgb
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# ─── YOLLAR ─────────────────────────────────────────────────────────────────
V7_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE      = os.path.join(V7_DIR, '..', 'universite-veri-seti', 'EĞİTİM (TRAIN) SETLERİ')
DATA_PATH = os.path.join(BASE, 'YARISMA_TRAIN_MASTER.csv')
OUT_FILE  = os.path.join(V7_DIR, 'v7_optuna_results.json')

RANDOM_STATE    = 42
MISSING_THRESH  = 0.80
OPTUNA_CV_FOLDS = 3


def detect_gpu():
    try:
        import subprocess
        subprocess.run(['nvidia-smi'], capture_output=True, check=True)
        return True
    except Exception:
        return False


GPU_AVAILABLE = detect_gpu()


def get_grantham(aa1, aa2):
    if pd.isna(aa1) or pd.isna(aa2):
        return np.nan
    _data = {
        'A':{'A':0,'R':112,'N':111,'D':126,'C':195,'Q':91,'E':107,'G':60,'H':86,'I':94,'L':96,'K':106,'M':84,'F':113,'P':27,'S':99,'T':58,'W':148,'Y':112,'V':64},
        'R':{'A':112,'R':0,'N':86,'D':96,'C':180,'Q':43,'E':54,'G':125,'H':29,'I':97,'L':102,'K':26,'M':91,'F':97,'P':103,'S':110,'T':71,'W':101,'Y':77,'V':96},
    }
    # Basit fallback
    return 100.0


AA_PROPERTIES = {
    'A':{'polarity':0,'charge':0,'hydropathy':1.8,'weight':89.1,'volume':88.6,'flexibility':0.36,'aromatic':0},
    'R':{'polarity':1,'charge':1,'hydropathy':-4.5,'weight':174.2,'volume':173.4,'flexibility':0.53,'aromatic':0},
    'N':{'polarity':1,'charge':0,'hydropathy':-3.5,'weight':132.1,'volume':114.1,'flexibility':0.46,'aromatic':0},
    'D':{'polarity':1,'charge':-1,'hydropathy':-3.5,'weight':133.1,'volume':111.1,'flexibility':0.51,'aromatic':0},
    'C':{'polarity':0,'charge':0,'hydropathy':2.5,'weight':121.2,'volume':108.5,'flexibility':0.35,'aromatic':0},
    'E':{'polarity':1,'charge':-1,'hydropathy':-3.5,'weight':147.1,'volume':138.4,'flexibility':0.50,'aromatic':0},
    'Q':{'polarity':1,'charge':0,'hydropathy':-3.5,'weight':146.2,'volume':143.8,'flexibility':0.49,'aromatic':0},
    'G':{'polarity':0,'charge':0,'hydropathy':-0.4,'weight':75.1,'volume':60.1,'flexibility':0.54,'aromatic':0},
    'H':{'polarity':1,'charge':1,'hydropathy':-3.2,'weight':155.2,'volume':153.2,'flexibility':0.32,'aromatic':1},
    'I':{'polarity':0,'charge':0,'hydropathy':4.5,'weight':131.2,'volume':166.7,'flexibility':0.30,'aromatic':0},
    'L':{'polarity':0,'charge':0,'hydropathy':3.8,'weight':131.2,'volume':166.7,'flexibility':0.40,'aromatic':0},
    'K':{'polarity':1,'charge':1,'hydropathy':-3.9,'weight':146.2,'volume':168.6,'flexibility':0.47,'aromatic':0},
    'M':{'polarity':0,'charge':0,'hydropathy':1.9,'weight':149.2,'volume':162.9,'flexibility':0.30,'aromatic':0},
    'F':{'polarity':0,'charge':0,'hydropathy':2.8,'weight':165.2,'volume':189.9,'flexibility':0.31,'aromatic':1},
    'P':{'polarity':0,'charge':0,'hydropathy':-1.6,'weight':115.1,'volume':112.7,'flexibility':0.51,'aromatic':0},
    'S':{'polarity':1,'charge':0,'hydropathy':-0.8,'weight':105.1,'volume':89.0,'flexibility':0.51,'aromatic':0},
    'T':{'polarity':1,'charge':0,'hydropathy':-0.7,'weight':119.1,'volume':116.1,'flexibility':0.44,'aromatic':0},
    'W':{'polarity':0,'charge':0,'hydropathy':-0.9,'weight':204.2,'volume':227.8,'flexibility':0.31,'aromatic':1},
    'Y':{'polarity':1,'charge':0,'hydropathy':-1.3,'weight':181.2,'volume':193.6,'flexibility':0.42,'aromatic':1},
    'V':{'polarity':0,'charge':0,'hydropathy':4.2,'weight':117.1,'volume':140.0,'flexibility':0.39,'aromatic':0},
}


def load_and_prepare():
    """Veriyi yükle ve temel özellik çıkarımı yap."""
    df = pd.read_csv(DATA_PATH)
    df['Label'] = df['Label'].astype(int)

    al_cols  = [c for c in df.columns if c.startswith('AL_')]
    ek_cols  = [c for c in df.columns if c.startswith('EK_')]
    cat_cols = [c for c in df.columns if c.startswith('CAT_') and c != 'CAT_6']

    # Biyokimyasal özellikler
    props = ['polarity', 'charge', 'hydropathy', 'weight', 'volume', 'flexibility', 'aromatic']
    for aa_col, prefix in [('AA_1', 'AA1'), ('AA_2', 'AA2')]:
        for prop in props:
            df[f'{prefix}_{prop}'] = df[aa_col].map(
                lambda x: AA_PROPERTIES.get(x, {}).get(prop, np.nan))

    df['AA_hydro_diff']  = abs(df['AA1_hydropathy'] - df['AA2_hydropathy'])
    df['AA_weight_diff'] = abs(df['AA1_weight']     - df['AA2_weight'])
    df['AA_volume_diff'] = abs(df['AA1_volume']     - df['AA2_volume'])

    # EK_ etkileşimleri
    for a, b in combinations(ek_cols, 2):
        df[f'{a}_x_{b}'] = df[a] * df[b]

    # Eksiklik özellikleri
    df['missing_ratio_AL'] = df[al_cols].isnull().mean(axis=1)
    df['missing_ratio_EK'] = df[ek_cols].isnull().mean(axis=1)

    # Sütunları topla
    aa_num = [f'AA1_{p}' for p in props] + [f'AA2_{p}' for p in props] + \
             ['AA_hydro_diff', 'AA_weight_diff', 'AA_volume_diff']
    ek_int = [f'{a}_x_{b}' for a, b in combinations(ek_cols, 2)]
    miss_p = ['missing_ratio_AL', 'missing_ratio_EK']

    all_num = al_cols + ek_cols + aa_num + ek_int + miss_p
    all_num = [c for c in all_num if c in df.columns]
    missing_ratio_series = df[all_num].isnull().mean()
    num_cols = missing_ratio_series[missing_ratio_series < MISSING_THRESH].index.tolist()

    cat_all = [c for c in cat_cols if c not in ('CAT_4', 'CAT_5')] + ['AA_1', 'AA_2']

    X = df[num_cols + cat_all]
    y = df['Label']

    # Preprocessor
    num_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('scl', RobustScaler()),
        ('vt',  VarianceThreshold(1e-4)),
    ])
    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])
    prep = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_all)])
    X_proc = prep.fit_transform(X)

    pos_ratio = y.mean()
    scale_pw  = float(round((1 - pos_ratio) / pos_ratio, 4))

    return X_proc, y, scale_pw


def find_thresh(y_true, y_prob, step=0.01):
    best_mcc, best_thresh = -1.0, 0.5
    for t in np.arange(0.20, 0.71, step):
        yp  = (y_prob >= t).astype(int)
        mcc = matthews_corrcoef(y_true, yp)
        if mcc > best_mcc:
            best_mcc, best_thresh = mcc, t
    return round(best_thresh, 2), best_mcc


def xgb_objective(trial, X_proc, y):
    params = {
        'n_estimators':     trial.suggest_int  ('n_estimators',     300, 1500, step=100),
        'learning_rate':    trial.suggest_float ('learning_rate',    0.003, 0.15, log=True),
        'max_depth':        trial.suggest_int   ('max_depth',        3, 10),
        'subsample':        trial.suggest_float ('subsample',        0.50, 1.0),
        'colsample_bytree': trial.suggest_float ('colsample_bytree', 0.40, 1.0),
        'reg_alpha':        trial.suggest_float ('reg_alpha',        1e-8, 10.0, log=True),
        'reg_lambda':       trial.suggest_float ('reg_lambda',       1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int   ('min_child_weight', 1, 20),
        'scale_pos_weight': trial.suggest_float ('scale_pos_weight', 0.15, 0.60),
        'gamma':            trial.suggest_float ('gamma',            0.0, 5.0),
        'random_state': RANDOM_STATE, 'verbosity': 0, 'n_jobs': -1, 'eval_metric': 'logloss',
    }
    if GPU_AVAILABLE:
        params['device'] = 'cuda'
        params['tree_method'] = 'hist'

    clf = xgb.XGBClassifier(**params)
    cv  = StratifiedKFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mccs = []
    for tr, val in cv.split(X_proc, y):
        clf.fit(X_proc[tr], y.iloc[tr])
        prob = clf.predict_proba(X_proc[val])[:, 1]
        t, mcc = find_thresh(y.iloc[val], prob)
        mccs.append(mcc)
    return np.mean(mccs)


def cat_objective(trial, X_proc, y):
    if not CATBOOST_AVAILABLE:
        return 0.0
    params = {
        'iterations':       trial.suggest_int  ('iterations',       300, 1500, step=100),
        'learning_rate':    trial.suggest_float ('learning_rate',    0.003, 0.15, log=True),
        'depth':            trial.suggest_int   ('depth',            4, 10),
        'l2_leaf_reg':      trial.suggest_float ('l2_leaf_reg',      1.0, 10.0),
        'scale_pos_weight': trial.suggest_float ('scale_pos_weight', 0.15, 0.60),
        'subsample':        trial.suggest_float ('subsample',        0.60, 1.0),
        'random_seed': RANDOM_STATE, 'verbose': 0, 'eval_metric': 'F1',
    }
    if GPU_AVAILABLE:
        params['task_type'] = 'GPU'

    clf = CatBoostClassifier(**params)
    cv  = StratifiedKFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mccs = []
    for tr, val in cv.split(X_proc, y):
        clf.fit(X_proc[tr], y.iloc[tr])
        prob = clf.predict_proba(X_proc[val])[:, 1]
        t, mcc = find_thresh(y.iloc[val], prob)
        mccs.append(mcc)
    return np.mean(mccs)


def main():
    parser = argparse.ArgumentParser(description='AlgoMed V7 — Optuna Optimizasyonu')
    parser.add_argument('--trials', type=int, default=700,
                        help='Optuna trial sayısı (varsayılan: 700)')
    parser.add_argument('--model', choices=['xgb', 'cat', 'both'], default='both',
                        help='Hangi model optimize edilsin (varsayılan: both)')
    args = parser.parse_args()

    print(f'{"="*60}')
    print(f'  AlgoMed V7 — Optuna Çalışması')
    print(f'  Trials: {args.trials} | Model: {args.model}')
    print(f'  GPU: {"Aktif ✅" if GPU_AVAILABLE else "CPU ⚠️"}')
    print(f'{"="*60}')

    print('\nVeri yükleniyor...')
    X_proc, y, scale_pw = load_and_prepare()
    print(f'  Veri boyutu: {X_proc.shape}')
    print(f'  Patojenik oranı: {y.mean():.2%}')
    print(f'  scale_pos_weight: {scale_pw}')

    results = {}

    if args.model in ('xgb', 'both'):
        print(f'\n[XGBoost] {args.trials} trial başlıyor...')
        t0 = time.time()
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        )
        study.optimize(
            lambda trial: xgb_objective(trial, X_proc, y),
            n_trials=args.trials,
            show_progress_bar=True,
            n_jobs=1
        )
        elapsed = time.time() - t0
        print(f'✅ XGBoost bitti — {elapsed/60:.1f} dakika')
        print(f'   En iyi MCC: {study.best_value:.4f}')
        print(f'   Parametreler: {study.best_params}')
        results['xgb_best_params'] = study.best_params
        results['xgb_best_mcc']    = study.best_value

    if args.model in ('cat', 'both') and CATBOOST_AVAILABLE:
        print(f'\n[CatBoost] {args.trials} trial başlıyor...')
        t0 = time.time()
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        )
        study.optimize(
            lambda trial: cat_objective(trial, X_proc, y),
            n_trials=args.trials,
            show_progress_bar=True,
            n_jobs=1
        )
        elapsed = time.time() - t0
        print(f'✅ CatBoost bitti — {elapsed/60:.1f} dakika')
        print(f'   En iyi MCC: {study.best_value:.4f}')
        print(f'   Parametreler: {study.best_params}')
        results['cat_best_params'] = study.best_params
        results['cat_best_mcc']    = study.best_value

    # Kaydet
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n💾 Sonuçlar kaydedildi: {OUT_FILE}')
    print('\nBitince train.py çalıştırın:')
    print('  python train.py   (RUN_OPTUNA=False olarak ayarla)')


if __name__ == '__main__':
    main()
