"""
AlgoMed V7 — Sızdırmaz Pipeline + Optuna GPU Optimizasyonu
===========================================================
GÖREV: V5 MİMARİSİNİN SINIRLARINI ZORLAMAK

V5'in mimarisi değişmeden korunmaktadır:
  LightGBM + XGBoost + CatBoost → StackingClassifier (LogisticRegression meta)
  GPU hızlandırmalı (CUDA/hist)
  V3'ten biyokimyasal AA özellikleri (polarity, charge, hydropathy, weight)
  V4'ten EK_i x EK_j etkileşim terimleri
  V4'ten eksiklik örüntüsü özellikleri

V7'de V5'e eklenenler:
  1. Optuna 500-1000 trial (L1/L2 reg, max_depth, scale_pos_weight odaklı)
     V5'te sadece XGBoost için 100 trial yapılmıştı.
     V7'de XGBoost için 700 trial, CatBoost için 500 trial GPU üzerinde yapıldı.
  2. SMOTE / ADASYN / BorderlineSMOTE ile sınıf dengesizliği çözümü
     imblearn kullanılarak yalnızca CV eğitim katmanında uygulanmaktadır.

Kritik Kural — Data Leakage Önleme:
  SMOTE işlemi tüm veriye kesinlikle uygulanmaz.
  Her fold'da:
    - X_train → SMOTE ile büyütülür (sentetik veri üretilir)
    - X_val   → dokunulmaz olarak kalır (orijinal, sentetik veri görmez)

Bağımlılıklar:
  pip install lightgbm xgboost catboost optuna imbalanced-learn scikit-learn pandas numpy
"""

import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, confusion_matrix,
    precision_score, recall_score
)

import lightgbm as lgb
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("imbalanced-learn kurulu degil! Kur: pip install imbalanced-learn")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost kurulu degil: pip install catboost")

# ─── VERİ YOLLARI (V5 ile aynı) ─────────────────────────────────────────────
BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'universite-veri-seti', 'EĞİTİM (TRAIN) SETLERİ'
)
PATHS = {
    'MASTER': os.path.join(BASE, 'YARISMA_TRAIN_MASTER.csv'),
    'KANSER': os.path.join(BASE, 'YARISMA_TRAIN_KANSER.csv'),
    'PAH':    os.path.join(BASE, 'YARISMA_TRAIN_PAH.csv'),
    'CFTR':   os.path.join(BASE, 'YARISMA_TRAIN_CFTR.csv'),
}

V7_DIR       = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(V7_DIR, 'v7_optuna_results.json')
DB_FILE      = os.path.join(V7_DIR, 'v7_optuna.db')

SEP  = '=' * 65
SEP2 = '-' * 50

# ─── AYARLAR ─────────────────────────────────────────────────────────────────
# =============================================================================
OPTUNA_TRIALS   = 500    # XGB için 700, CatBoost için 500 deneme yapıldı. Tamamlandı.
RUN_XGB_OPTUNA  = False  # XGBoost Optuna optimizasyonu tamamlandı (700 deneme, 91.4 dakika). Parametreler JSON'dan yükleniyor.
RUN_CAT_OPTUNA  = False  # CatBoost Optuna optimizasyonu tamamlandı (500 deneme, 631.6 dakika). Parametreler JSON'dan yükleniyor.
RUN_OPTUNA      = False  # False olduğunda parametreler v7_optuna_results.json dosyasından yüklenir. True yapılırsa sıfırdan başlar.
SMOTE_METHOD    = 'auto' # Kullanılacak aşırı örnekleme yöntemi. 'smote', 'adasyn', 'borderline' veya 'auto' (üçü de test edilir)
USE_GPU         = True   # NVIDIA GPU mevcutsa True olarak bırakılmalıdır
MISSING_THRESH  = 0.80   # V5 ile aynı — %80 ve üzeri eksik olan sayısal sütunlar veri setinden çıkarılır
CV_FOLDS        = 5      # V5 ile aynı — 5-Fold Stratified Cross Validation
OPTUNA_CV_FOLDS = 3      # Optuna içinde hız için 3-fold kullanılmaktadır (V5'te de 3 kullanılmıştı)
RANDOM_STATE    = 42
# =============================================================================

# XGBoost Optuna optimizasyonu sırasında bulunan en iyi parametreler (700 deneme, MCC: 0.5685)
XGB_BEST_FROM_RUN = {
    'n_estimators': 1100,
    'learning_rate': 0.02586734932530263,
    'max_depth': 7,
    'subsample': 0.8664228716565388,
    'colsample_bytree': 0.5119883363882134,
    'reg_alpha': 6.651253573074137,
    'reg_lambda': 0.034194964864740884,
    'min_child_weight': 2,
    'scale_pos_weight': 0.5252072954262788,
    'gamma': 0.10625792313196786,
}

# V5 Optuna sonuclarından elde edilmiş LightGBM parametreleri (varsayılan olarak kullanılır)
V5_BEST_LGB_PARAMS = {
    'n_estimators': 800,
    'learning_rate': 0.07917491343504915,
    'num_leaves': 74,
    'max_depth': 11,
    'subsample': 0.597804620160675,
    'colsample_bytree': 0.6513603446650886,
    'min_child_samples': 25,
    'scale_pos_weight': 0.33067145309795315,
    'random_state': RANDOM_STATE,
    'verbose': -1,
    'n_jobs': -1,
}
V5_BEST_XGB_PARAMS = {
    'n_estimators': 800,
    'learning_rate': 0.024122636498078342,
    'max_depth': 8,
    'subsample': 0.6960781394323559,
    'colsample_bytree': 0.759442862395421,
    'min_child_weight': 10,
    'scale_pos_weight': 0.4775383149015521,
}

# Amino asit biyokimyasal özellik sözlüğü (V3'ten, V5'te değişmeden korunmuştur)
AA_PROPERTIES = {
    'A': {'polarity': 0, 'charge':  0, 'hydropathy':  1.8, 'weight':  89.1},
    'R': {'polarity': 1, 'charge':  1, 'hydropathy': -4.5, 'weight': 174.2},
    'N': {'polarity': 1, 'charge':  0, 'hydropathy': -3.5, 'weight': 132.1},
    'D': {'polarity': 1, 'charge': -1, 'hydropathy': -3.5, 'weight': 133.1},
    'C': {'polarity': 0, 'charge':  0, 'hydropathy':  2.5, 'weight': 121.2},
    'E': {'polarity': 1, 'charge': -1, 'hydropathy': -3.5, 'weight': 147.1},
    'Q': {'polarity': 1, 'charge':  0, 'hydropathy': -3.5, 'weight': 146.2},
    'G': {'polarity': 0, 'charge':  0, 'hydropathy': -0.4, 'weight':  75.1},
    'H': {'polarity': 1, 'charge':  1, 'hydropathy': -3.2, 'weight': 155.2},
    'I': {'polarity': 0, 'charge':  0, 'hydropathy':  4.5, 'weight': 131.2},
    'L': {'polarity': 0, 'charge':  0, 'hydropathy':  3.8, 'weight': 131.2},
    'K': {'polarity': 1, 'charge':  1, 'hydropathy': -3.9, 'weight': 146.2},
    'M': {'polarity': 0, 'charge':  0, 'hydropathy':  1.9, 'weight': 149.2},
    'F': {'polarity': 0, 'charge':  0, 'hydropathy':  2.8, 'weight': 165.2},
    'P': {'polarity': 0, 'charge':  0, 'hydropathy': -1.6, 'weight': 115.1},
    'S': {'polarity': 1, 'charge':  0, 'hydropathy': -0.8, 'weight': 105.1},
    'T': {'polarity': 1, 'charge':  0, 'hydropathy': -0.7, 'weight': 119.1},
    'W': {'polarity': 0, 'charge':  0, 'hydropathy': -0.9, 'weight': 204.2},
    'Y': {'polarity': 1, 'charge':  0, 'hydropathy': -1.3, 'weight': 181.2},
    'V': {'polarity': 0, 'charge':  0, 'hydropathy':  4.2, 'weight': 117.1},
}


# ─── YARDIMCI FONKSİYONLAR (V5'ten aynen alınmıştır) ───────────────────────

def extract_aa_features(df):
    """AA_1 ve AA_2 sütunlarından biyokimyasal sayısal özellikler hesaplanır (V3'ten, V5'te değişmeden korunmuştur)."""
    df = df.copy()
    for aa_col, prefix in [('AA_1', 'AA1'), ('AA_2', 'AA2')]:
        df[f'{prefix}_polarity'] = df[aa_col].map(
            lambda x: AA_PROPERTIES.get(x, {}).get('polarity', np.nan))
        df[f'{prefix}_charge']   = df[aa_col].map(
            lambda x: AA_PROPERTIES.get(x, {}).get('charge', np.nan))
        df[f'{prefix}_hydro']    = df[aa_col].map(
            lambda x: AA_PROPERTIES.get(x, {}).get('hydropathy', np.nan))
        df[f'{prefix}_weight']   = df[aa_col].map(
            lambda x: AA_PROPERTIES.get(x, {}).get('weight', np.nan))

    df['AA_hydro_diff']      = abs(df['AA1_hydro']  - df['AA2_hydro'])
    df['AA_weight_diff']     = abs(df['AA1_weight'] - df['AA2_weight'])
    df['AA_charge_change']   = (df['AA1_charge']   != df['AA2_charge']).astype(float)
    df['AA_polarity_change'] = (df['AA1_polarity'] != df['AA2_polarity']).astype(float)
    df.loc[df['AA1_charge'].isna() | df['AA2_charge'].isna(), 'AA_charge_change'] = np.nan
    df.loc[df['AA1_polarity'].isna() | df['AA2_polarity'].isna(), 'AA_polarity_change'] = np.nan
    return df


def add_interaction_features(df, ek_cols):
    """EK_ sütunları arasında ikili çarpım etkileşim terimleri üretilir (V4'ten, V5'te değişmeden korunmuştur)."""
    df = df.copy()
    for a, b in combinations(ek_cols, 2):
        df[f'{a}_x_{b}'] = df[a] * df[b]
    return df


def add_missing_pattern_features(df, al_cols, ek_cols):
    """Satır bazında eksiklik istatistikleri hesaplanır (V4'ten, V5'te değişmeden korunmuştur)."""
    df = df.copy()
    df['missing_count_AL']     = df[al_cols].isnull().sum(axis=1)
    df['missing_ratio_AL']     = df[al_cols].isnull().mean(axis=1)
    df['missing_ratio_AL_log'] = np.log1p(df['missing_ratio_AL'])
    df['missing_count_EK']     = df[ek_cols].isnull().sum(axis=1)
    df['missing_ratio_EK']     = df[ek_cols].isnull().mean(axis=1)
    df['missing_count_ALL']    = df[al_cols + ek_cols].isnull().sum(axis=1)
    if 'EK_9' in df.columns:
        df['EK9_x_miss_AL'] = df['EK_9'] * df['missing_ratio_AL']
    return df


def find_best_threshold(y_true, y_prob, step=0.01):
    """F1 skorunu en üst düzeye çıkaran karar eşiği belirlenir (V5'ten değişmeden alınmıştır)."""
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.20, 0.71, step):
        yp  = (y_prob >= thresh).astype(int)
        f1t = f1_score(y_true, yp, zero_division=0)
        if f1t > best_f1:
            best_f1, best_thresh = f1t, thresh
    return round(best_thresh, 2), best_f1


def detect_gpu():
    if not USE_GPU:
        return False
    try:
        import subprocess
        subprocess.run(['nvidia-smi'], capture_output=True, check=True)
        return True
    except Exception:
        return False


GPU_AVAILABLE = detect_gpu()


# ─── V5 MIMARISI: STACKING ENSEMBLE ─────────────────────────────────────────

def build_v5_stack(lgb_params, xgb_params, cat_params):
    """
    V5'in orijinal StackingClassifier mimarisi:
      LightGBM + XGBoost + CatBoost → LogisticRegression meta
    V7'de aynı mimari, yalnızca Optuna'nın bulduğu parametrelerle çalışmaktadır.
    """
    lgb_full = lgb_params.copy()
    if GPU_AVAILABLE:
        lgb_full['device_type'] = 'gpu'
    lgbm_clf = lgb.LGBMClassifier(**lgb_full)

    xgb_full = {'eval_metric': 'logloss', 'random_state': RANDOM_STATE,
                 'verbosity': 0, 'n_jobs': -1}
    xgb_full.update(xgb_params)
    if GPU_AVAILABLE:
        xgb_full['device'] = 'cuda'
        xgb_full['tree_method'] = 'hist'
    xgb_clf = xgb.XGBClassifier(**xgb_full)

    base_estimators = [('lgbm', lgbm_clf), ('xgb', xgb_clf)]

    if CATBOOST_AVAILABLE:
        cat_full = {
            'iterations': 800,
            'learning_rate': 0.05,
            'depth': 7,
            'eval_metric': 'F1',
            'random_seed': RANDOM_STATE,
            'verbose': 0,
            'task_type': 'GPU' if GPU_AVAILABLE else 'CPU'
        }
        if isinstance(cat_params, dict):
            cat_full.update(cat_params)
        elif isinstance(cat_params, (int, float)):
            cat_full['scale_pos_weight'] = cat_params
        
        cat_clf = CatBoostClassifier(**cat_full)
        base_estimators.append(('catboost', cat_clf))

    meta_clf = LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE)

    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_clf,
        cv=5,
        passthrough=False,  # V5 ile aynı
        n_jobs=1            # CatBoost GPU çakışmasını önlemek amacıyla 1 olarak bırakılmıştır
    )


class SaveBestParamsCallback:
    def __init__(self, json_path, study_type, scale_pw):
        self.json_path = json_path
        self.study_type = study_type
        self.scale_pw = scale_pw

    def __call__(self, study, trial):
        if study.best_trial.number == trial.number:
            try:
                saved_data = {}
                if os.path.exists(self.json_path):
                    with open(self.json_path, 'r', encoding='utf-8') as f:
                        saved_data = json.load(f)
                
                if self.study_type == 'xgb':
                    saved_data['xgb_best_params'] = study.best_params
                    saved_data['xgb_best_mcc'] = study.best_value
                elif self.study_type == 'cat':
                    saved_data['cat_best_params'] = study.best_params
                    saved_data['cat_best_mcc'] = study.best_value
                
                saved_data['trials'] = study.best_trial.number + 1
                saved_data['gpu'] = True
                
                with open(self.json_path, 'w', encoding='utf-8') as f:
                    json.dump(saved_data, f, indent=2, ensure_ascii=False)
                print(f"\n  [YENİ EN İYİ {self.study_type.upper()}] MCC: {study.best_value:.4f} (Parametreler JSON'a kaydedildi)")
            except Exception as e:
                print(f"Callback kaydetme hatasi: {e}")


# ─── OPTUNA OBJEKTİF FONKSİYONLARI ─────────────────────────────────────────

def make_xgb_objective(X_proc, y):
    """
    XGBoost için Optuna hedef fonksiyonu.
    V7'nin ana odağı: L1 (reg_alpha), L2 (reg_lambda), max_depth ve scale_pos_weight parametreleridir.
    V5'te yalnızca 100 deneme yapılmıştı; V7'de bu sayı 700'e çıkarıldı.
    """
    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int  ('n_estimators',     300, 1500, step=100),
            'learning_rate':    trial.suggest_float ('learning_rate',    0.003, 0.15, log=True),
            'max_depth':        trial.suggest_int   ('max_depth',        3, 10),
            'subsample':        trial.suggest_float ('subsample',        0.50, 1.0),
            'colsample_bytree': trial.suggest_float ('colsample_bytree', 0.40, 1.0),
            'reg_alpha':        trial.suggest_float ('reg_alpha',        1e-8, 10.0, log=True),  # L1 düzenlileştirme
            'reg_lambda':       trial.suggest_float ('reg_lambda',       1e-8, 10.0, log=True),  # L2 düzenlileştirme
            'min_child_weight': trial.suggest_int   ('min_child_weight', 1, 20),
            'scale_pos_weight': trial.suggest_float ('scale_pos_weight', 0.15, 0.60),
            'gamma':            trial.suggest_float ('gamma',            0.0, 5.0),
            'random_state': RANDOM_STATE, 'verbosity': 0, 'n_jobs': -1,
            'eval_metric': 'logloss',
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
            thresh, _ = find_best_threshold(y.iloc[val], prob)
            mccs.append(matthews_corrcoef(y.iloc[val], (prob >= thresh).astype(int)))
        return np.mean(mccs)
    return objective


def make_cat_objective(X_proc, y, scale_pw):
    """
    CatBoost için Optuna hedef fonksiyonu.
    V5'te CatBoost hiç Optuna'ya girmemişti; V7'de ilk kez optimize edilmiştir.
    L2 (l2_leaf_reg), depth ve scale_pos_weight parametreleri odaklanılan alanlardır.
    """
    def objective(trial):
        if not CATBOOST_AVAILABLE:
            return 0.0
        params = {
            'iterations':       trial.suggest_int  ('iterations',       300, 1000, step=100),
            'learning_rate':    trial.suggest_float ('learning_rate',    0.003, 0.15, log=True),
            'depth':            trial.suggest_int   ('depth',            4, 8),  # Aşırı öğrenmeyi ve yavaşlamayı önlemek amacıyla maksimum 8 olarak sınırlandırılmıştır
            'l2_leaf_reg':      trial.suggest_float ('l2_leaf_reg',      1.0, 10.0),  # L2 düzenlileştirme
            'scale_pos_weight': trial.suggest_float ('scale_pos_weight', 0.15, 0.60),
            'random_seed': RANDOM_STATE, 'verbose': 0, 'eval_metric': 'F1',
        }
        
        # CatBoost'ta Bayesian bootstrap tipi subsample parametresini desteklemediğinden,
        # bootstrap türüne göre farklı parametreler kullanılmaktadır.
        bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli'])
        params['bootstrap_type'] = bootstrap_type
        if bootstrap_type == 'Bernoulli':
            params['subsample'] = trial.suggest_float('subsample', 0.60, 1.0)
        elif bootstrap_type == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 10.0)

        if GPU_AVAILABLE:
            params['task_type'] = 'GPU'

        clf = CatBoostClassifier(**params)
        cv  = StratifiedKFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        mccs = []
        for tr, val in cv.split(X_proc, y):
            clf.fit(X_proc[tr], y.iloc[tr])
            prob = clf.predict_proba(X_proc[val])[:, 1]
            thresh, _ = find_best_threshold(y.iloc[val], prob)
            mccs.append(matthews_corrcoef(y.iloc[val], (prob >= thresh).astype(int)))
        return np.mean(mccs)
    return objective


# ─── SIZMAZ BORU HATTI: CV + SMOTE ──────────────────────────────────────────

def get_oversampler(method):
    """
    Aşırı örnekleme (oversampling) nesnesi oluşturulur.
    Kritik Not: Bu nesne her fold'da yalnızca X_train'e uygulanmaktadır. X_val'e hiçbir şekilde dokunulmaz.
    """
    if not IMBLEARN_AVAILABLE:
        raise RuntimeError("imbalanced-learn kurulu degil: pip install imbalanced-learn")
    if method == 'smote':
        return SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    elif method == 'adasyn':
        return ADASYN(random_state=RANDOM_STATE, n_neighbors=5)
    elif method == 'borderline':
        return BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=5,
                               m_neighbors=10, kind='borderline-1')
    else:
        raise ValueError(f"Bilinmeyen yontem: {method}. 'smote'/'adasyn'/'borderline' kullanin.")


def run_leak_free_cv_with_smote(X, y, num_cols, cat_all, lgb_params, xgb_params,
                                 cat_params, smote_method='smote'):
    """
    Sızdırmaz Boru Hattı — Ana CV Fonksiyonu

    Her fold'da şu sırayla çalışmaktadır:
      1. X_train ve X_val ayrılır
      2. Ön işleme adımları (preprocessor) yalnızca X_train'e fit edilir
      3. SMOTE yalnızca X_train_proc'a uygulanır (sentetik veri üretilir)
      4. X_val_proc dokunulmaz kalır (orijinaldir, sentetik veri görmez)
      5. V5 Stacking modeli SMOTE'lu X_train ile eğitilir
      6. Tahmin yalnızca orijinal X_val üzerinde yapılır

    Döndürür: (oof_preds, fold_scores)
    """
    def make_preprocessor():
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler',  RobustScaler()),
            ('vt',      VarianceThreshold(threshold=1e-4)),
        ])
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ])
        return ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_all)])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds   = np.zeros(len(y))
    fold_scores = []

    print(f'\n  [SMOTE={smote_method.upper()}] Sizdirmaz {CV_FOLDS}-Fold CV basliyor...')
    print(f'  {"Fold":<6} {"Train_once":>11} {"Train_sonra":>11} {"Val":>6} {"F1":>8} {"MCC":>8}')
    print(f'  {"-"*55}')

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, y_tr   = X.iloc[tr_idx].copy(), y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]

        # Adım 1: Ön işleme adımları yalnızca eğitim verisine (X_train) fit edilir
        preprocessor = make_preprocessor()
        X_tr_proc  = preprocessor.fit_transform(X_tr)
        X_val_proc = preprocessor.transform(X_val)   # Kritik: doğrulama seti yalnızca dönüştürülür, sızdırma yapılmaz

        n_before = len(y_tr)

        # Adım 2: SMOTE yalnızca eğitim verisine (X_train_proc) uygulanır
        # Doğrulama seti (X_val_proc) bu işleme hiçbir şekilde dahil edilmez
        oversampler = get_oversampler(smote_method)
        try:
            X_tr_res, y_tr_res = oversampler.fit_resample(X_tr_proc, y_tr)
        except Exception as e:
            print(f'    SMOTE hatasi fold {fold}: {e} -- orijinal train kullaniliyor')
            X_tr_res, y_tr_res = X_tr_proc, y_tr

        n_after = len(y_tr_res)

        # Adım 3: V5 Stacking modeli SMOTE uygulanmış eğitim verisiyle eğitilir
        stack = build_v5_stack(lgb_params, xgb_params, cat_params)
        stack.fit(X_tr_res, y_tr_res)

        # Adım 4: Tahmin yalnızca orijinal doğrulama seti üzerinde yapılır (sentetik veri görmez)
        oof_preds[val_idx] = stack.predict_proba(X_val_proc)[:, 1]

        thresh, f1_f = find_best_threshold(y_val, oof_preds[val_idx])
        mcc_f = matthews_corrcoef(y_val, (oof_preds[val_idx] >= thresh).astype(int))
        fold_scores.append({'f1': f1_f, 'mcc': mcc_f})

        print(f'  Fold {fold:<4} {n_before:>11,d} {n_after:>11,d} '
              f'{len(y_val):>6d} {f1_f:>8.4f} {mcc_f:>8.4f}')

    avg_f1  = np.mean([s['f1']  for s in fold_scores])
    avg_mcc = np.mean([s['mcc'] for s in fold_scores])
    print(f'  {"-"*55}')
    print(f'  {"Ortalama":<35} {avg_f1:>8.4f} {avg_mcc:>8.4f}')

    return oof_preds, fold_scores


# ═══════════════════════════════════════════════════════════════════════════════
#  ANA CALISMA
# ═══════════════════════════════════════════════════════════════════════════════

print(SEP)
print('  AlgoMed V7 -- Sizdirmaz Pipeline + Optuna GPU Optimizasyonu')
print('  V5 Mimarisinin Sinirlarini Zorlama:')
print(f'  Optuna {OPTUNA_TRIALS} trial (L1/L2/depth/scale_pos_weight)')
print('  + SMOTE/ADASYN/BorderlineSMOTE Sizdirmaz Pipeline')
print(SEP)
print(f'  GPU      : {"Aktif" if GPU_AVAILABLE else "CPU modu"}')
print(f'  Optuna   : {"CALISACAK (" + str(OPTUNA_TRIALS) + " trial)" if RUN_OPTUNA else "ATLANACAK -- JSON yükle"}')
print(f'  imblearn : {"Kurulu" if IMBLEARN_AVAILABLE else "EKSIK -- pip install imbalanced-learn"}')

if not IMBLEARN_AVAILABLE:
    print('\nHATA: imbalanced-learn kurulu degil!')
    print('  pip install imbalanced-learn')
    import sys; sys.exit(1)

# ─── ADIM 1: VERI YUKLEME VE OZELLIK CIKARIMI (V5 ile birebir ayni) ─────────
print(f'\n{SEP2}')
print('ADIM 1 -- Veri Yukleme ve Ozellik Cikarimi (V5 mimarisi)')
print(SEP2)

df = pd.read_csv(PATHS['MASTER'])
df['Label'] = df['Label'].astype(int)

al_cols  = [c for c in df.columns if c.startswith('AL_')]
ek_cols  = [c for c in df.columns if c.startswith('EK_')]
cat_cols = [c for c in df.columns if c.startswith('CAT_') and c != 'CAT_6']
aa_cols  = ['AA_1', 'AA_2']

# V3'ten gelen biyokimyasal özellikler (V5'te değişmeden korunmuştur)
df = extract_aa_features(df)
new_aa_num_cols = [
    'AA1_polarity', 'AA1_charge', 'AA1_hydro', 'AA1_weight',
    'AA2_polarity', 'AA2_charge', 'AA2_hydro', 'AA2_weight',
    'AA_hydro_diff', 'AA_weight_diff', 'AA_charge_change', 'AA_polarity_change'
]

# V4'ten EK_ etkileşim terimleri (V5'te değişmeden korunmuştur)
df = add_interaction_features(df, ek_cols)
ek_interaction_cols = [f'{a}_x_{b}' for a, b in combinations(ek_cols, 2)]

# V4'ten eksiklik örüntüsü özellikleri (V5'te değişmeden korunmuştur)
df = add_missing_pattern_features(df, al_cols, ek_cols)
missing_pattern_cols = [
    'missing_count_AL', 'missing_ratio_AL', 'missing_ratio_AL_log',
    'missing_count_EK', 'missing_ratio_EK', 'missing_count_ALL', 'EK9_x_miss_AL'
]

# Eksiklik oranı %80 ve üzeri olan sayısal sütunlar veri setinden çıkarılır (V5 ile aynı)
all_num_base = al_cols + ek_cols + new_aa_num_cols + ek_interaction_cols + missing_pattern_cols
all_num_base = [c for c in all_num_base if c in df.columns]
missing_ratio_series = df[all_num_base].isnull().mean()
num_cols = missing_ratio_series[missing_ratio_series < MISSING_THRESH].index.tolist()

# CAT_4 ve CAT_5 sütunları veri setinden çıkarıldı (V5'te de aynı karar alınmıştı)
cat_all = [c for c in cat_cols if c not in ('CAT_4', 'CAT_5')] + aa_cols

X = df[num_cols + cat_all]
y = df['Label']

pos_ratio = y.mean()
scale_pw  = float(round((1 - pos_ratio) / pos_ratio, 4))

print(f'  Biyokimyasal ozellikler : {len(new_aa_num_cols)} (V3ten)')
print(f'  EK_ etkilesim terimleri : {len(ek_interaction_cols)} (V4ten)')
miss_pat_in_df = [c for c in missing_pattern_cols if c in df.columns]
print(f'  Eksiklik oruntu ozellik : {len(miss_pat_in_df)} (V4ten)')
print(f'  Tutulan sayisal sutun   : {len(num_cols)}')
print(f'  Kategorik sutun         : {len(cat_all)}')
print(f'  MASTER boyutu           : {df.shape}')
print(f'  Patojenik orani         : {pos_ratio:.2%} | scale_pos_weight: {scale_pw}')

# ─── ADIM 2: OPTUNA OPTIMIZASYONU ────────────────────────────────────────────
print(f'\n{SEP2}')
print(f'ADIM 2 -- Optuna Optimizasyonu ({OPTUNA_TRIALS} trial)')
print('  Odak: L1/L2 regularizasyon + max_depth + scale_pos_weight')
print(SEP2)

# Optuna için geçici bir ön işleme adımı uygulanmaktadır (tam CV dışında, hız amaçlıdır)
print('  Optuna icin gecici on isleme...')
_prep_opt = ColumnTransformer([
    ('num', Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('scl', RobustScaler()),
        ('vt',  VarianceThreshold(1e-4)),
    ]), num_cols),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ]), cat_all),
])
X_proc_for_optuna = _prep_opt.fit_transform(X)
print(f'  Optuna X boyutu: {X_proc_for_optuna.shape}')

if RUN_OPTUNA:
    final_xgb_params = XGB_BEST_FROM_RUN.copy()
    xgb_best_mcc = 0.5685
    final_cat_params = {'scale_pos_weight': scale_pw}
    cat_optuna_mcc = 0.0
    
    storage_url = f"sqlite:///{DB_FILE.replace(os.sep, '/')}"

    if RUN_XGB_OPTUNA:
        print(f'\n  [XGBoost] {OPTUNA_TRIALS} trial basliyor (GPU: {GPU_AVAILABLE})...')
        t0 = time.time()
        xgb_callback = SaveBestParamsCallback(RESULTS_FILE, 'xgb', scale_pw)
        xgb_study = optuna.create_study(
            direction='maximize',
            study_name='v7_xgb',
            storage=storage_url,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        )
        completed_trials = len([t for t in xgb_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        trials_to_run = max(0, OPTUNA_TRIALS - completed_trials)
        if trials_to_run > 0:
            xgb_study.optimize(
                make_xgb_objective(X_proc_for_optuna, y),
                n_trials=trials_to_run,
                show_progress_bar=True,
                n_jobs=1,
                callbacks=[xgb_callback]
            )
        xgb_elapsed = time.time() - t0
        print(f'  XGBoost Optuna tamamlandi: {xgb_elapsed/60:.1f} dakika')
        print(f'  En iyi MCC ({OPTUNA_CV_FOLDS}-fold): {xgb_study.best_value:.4f}')
        print('  En iyi XGB parametreler:')
        for k, v in xgb_study.best_params.items():
            print(f'    {k}: {v}')
        final_xgb_params = xgb_study.best_params
        xgb_best_mcc = xgb_study.best_value
    else:
        print(f'\n  [XGBoost] Optuna optimizasyonu atlandi, onceki basarili parametreler yuklendi.')
        print(f'  Kullanilan XGB MCC: {xgb_best_mcc:.4f}')

    # CatBoost Optuna optimizasyonu (V5'te hiç yapılmamıştı; V7'de ilk kez uygulandı)
    if CATBOOST_AVAILABLE:
        if RUN_CAT_OPTUNA:
            print(f'\n  [CatBoost] {OPTUNA_TRIALS} trial basliyor...')
            t0 = time.time()
            cat_callback = SaveBestParamsCallback(RESULTS_FILE, 'cat', scale_pw)
            cat_study = optuna.create_study(
                direction='maximize',
                study_name='v7_cat',
                storage=storage_url,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
            )
            completed_trials = len([t for t in cat_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            trials_to_run = max(0, OPTUNA_TRIALS - completed_trials)
            if trials_to_run > 0:
                cat_study.optimize(
                    make_cat_objective(X_proc_for_optuna, y, scale_pw),
                    n_trials=trials_to_run,
                    show_progress_bar=True,
                    n_jobs=1,
                    callbacks=[cat_callback]
                )
            cat_elapsed = time.time() - t0
            print(f'  CatBoost Optuna tamamlandi: {cat_elapsed/60:.1f} dakika')
            print(f'  En iyi MCC ({OPTUNA_CV_FOLDS}-fold): {cat_study.best_value:.4f}')
            print('  En iyi CAT parametreler:')
            for k, v in cat_study.best_params.items():
                print(f'    {k}: {v}')
            final_cat_params = cat_study.best_params
            cat_optuna_mcc   = cat_study.best_value
        else:
            print(f'\n  [CatBoost] Optuna optimizasyonu atlandi.')
            final_cat_params = {'scale_pos_weight': scale_pw}
            cat_optuna_mcc = 0.0

    # Tüm Optuna sonuçları JSON dosyasına kaydedilir (her ihtimale karşı eğitim sonunda da yazılır)
    optuna_out = {
        'xgb_best_params': final_xgb_params,
        'xgb_best_mcc':    xgb_best_mcc,
        'cat_best_params': final_cat_params,
        'cat_best_mcc':    cat_optuna_mcc,
        'trials': OPTUNA_TRIALS, 'gpu': GPU_AVAILABLE,
    }
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(optuna_out, f, indent=2, ensure_ascii=False)
    print(f'\n  Optuna sonuclari kaydedildi: {RESULTS_FILE}')

    final_lgb_params = V5_BEST_LGB_PARAMS.copy()

else:
    if os.path.exists(RESULTS_FILE):
        print(f'  Kayitli Optuna sonuclari yukleniyor: {RESULTS_FILE}')
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        final_lgb_params = V5_BEST_LGB_PARAMS.copy()
        final_xgb_params = saved.get('xgb_best_params', V5_BEST_XGB_PARAMS.copy())
        final_cat_params = saved.get('cat_best_params', {})
        print(f'  XGB Optuna MCC: {saved.get("xgb_best_mcc", "N/A")}')
        print(f'  CAT Optuna MCC: {saved.get("cat_best_mcc", "N/A")}')
    else:
        print(f'  {RESULTS_FILE} bulunamadi -- V5 varsayilan parametreler kullaniliyor')
        final_lgb_params = V5_BEST_LGB_PARAMS.copy()
        final_xgb_params = V5_BEST_XGB_PARAMS.copy()
        final_cat_params = {'scale_pos_weight': scale_pw}

# ─── ADIM 3: SIZMAZ SMOTE + V5 STACKING CV ───────────────────────────────────
print(f'\n{SEP2}')
print('ADIM 3 -- Sizdirmaz SMOTE Pipeline + V5 Stacking CV')
print('  KRITIK: SMOTE SADECE her foldin TRAIN kismina uygulanir!')
print('  KRITIK: VAL seti orijinal kalir -- sentetik veri GORMEZ!')
print(SEP2)

if SMOTE_METHOD == 'auto':
    methods    = ['smote', 'adasyn', 'borderline']
    method_res = {}
    print('\n  Tum oversampling yontemleri karsilastiriliyor...')

    for method in methods:
        print(f'\n  {"─"*45}')
        print(f'  Test: {method.upper()}')
        t0 = time.time()
        try:
            oof, fold_sc = run_leak_free_cv_with_smote(
                X, y, num_cols, cat_all,
                final_lgb_params, final_xgb_params, final_cat_params,
                method
            )
            thresh, _ = find_best_threshold(y, oof)
            preds = (oof >= thresh).astype(int)
            res = {
                'f1':      f1_score(y, preds, zero_division=0),
                'mcc':     matthews_corrcoef(y, preds),
                'pr':      average_precision_score(y, oof),
                'roc':     roc_auc_score(y, oof),
                'thresh':  thresh,
                'time_min': (time.time() - t0) / 60,
                'oof':     oof.tolist(),
            }
            method_res[method] = res
            print(f'  {method.upper()} -> F1:{res["f1"]:.4f}  MCC:{res["mcc"]:.4f}  '
                  f'PR-AUC:{res["pr"]:.4f}  ({res["time_min"]:.1f} dk)')
        except Exception as e:
            print(f'  {method.upper()} basarisiz: {e}')
            method_res[method] = None

    valid    = [m for m in method_res if method_res[m] is not None]
    best_smote   = max(valid, key=lambda m: method_res[m]['mcc'])
    best_res     = method_res[best_smote]
    oof_probs    = np.array(best_res['oof'])
    final_thresh = best_res['thresh']
    print(f'\n  KAZANAN: {best_smote.upper()} (MCC={best_res["mcc"]:.4f})')

else:
    best_smote = SMOTE_METHOD
    oof_probs, fold_scores = run_leak_free_cv_with_smote(
        X, y, num_cols, cat_all,
        final_lgb_params, final_xgb_params, final_cat_params,
        SMOTE_METHOD
    )
    final_thresh, _ = find_best_threshold(y, oof_probs)
    method_res = {SMOTE_METHOD: {
        'oof': oof_probs.tolist(), 'thresh': final_thresh,
        'f1':  f1_score(y, (oof_probs >= final_thresh).astype(int), zero_division=0),
        'mcc': matthews_corrcoef(y, (oof_probs >= final_thresh).astype(int)),
        'pr':  average_precision_score(y, oof_probs),
        'roc': roc_auc_score(y, oof_probs),
    }}
    best_res = method_res[SMOTE_METHOD]

# ─── ADIM 4: SONUCLAR ────────────────────────────────────────────────────────
print(f'\n{SEP2}')
print('ADIM 4 -- Sonuclar ve Degerlendirme')
print(SEP2)

final_preds = (oof_probs >= final_thresh).astype(int)
v7_f1  = f1_score(y, final_preds, zero_division=0)
v7_mcc = matthews_corrcoef(y, final_preds)
v7_pr  = average_precision_score(y, oof_probs)
v7_roc = roc_auc_score(y, oof_probs)

print(f'\n  MASTER 5-Fold CV Sonuclari (SMOTE={best_smote.upper()}, Esik={final_thresh:.2f}):')
print(f'  {"Metrik":<20} {"Deger":>10}')
print(f'  {"-"*32}')
print(f'  {"F1 Skoru":<20} {v7_f1:>10.4f}')
print(f'  {"MCC":<20} {v7_mcc:>10.4f}')
print(f'  {"PR-AUC":<20} {v7_pr:>10.4f}')
print(f'  {"ROC-AUC":<20} {v7_roc:>10.4f}')

cm = confusion_matrix(y, final_preds)
print('\n  Confusion Matrix:')
print(f'                  Tahmin')
print(f'                  Benign  Patojenik')
print(f'  Gercek Benign  {cm[0,0]:6d}  {cm[0,1]:9d}')
print(f'  Gercek Patojen {cm[1,0]:6d}  {cm[1,1]:9d}')
print(f'\n  FN={cm[1,0]}  FP={cm[0,1]}')

print('\n  Esik Analizi:')
print(f'  {"Esik":>6} {"Precision":>10} {"Recall":>8} {"F1":>8} {"MCC":>8}')
for t in np.arange(0.25, 0.66, 0.05):
    yp   = (oof_probs >= t).astype(int)
    prec = precision_score(y, yp, zero_division=0)
    rec  = recall_score(y, yp, zero_division=0)
    f1t  = f1_score(y, yp, zero_division=0)
    mct  = matthews_corrcoef(y, yp)
    mark = ' <-- optimal' if abs(t - final_thresh) < 0.001 else ''
    print(f'  {t:>6.2f} {prec:>10.4f} {rec:>8.4f} {f1t:>8.4f} {mct:>8.4f}{mark}')

# ─── ADIM 5: OVERSAMPLING KARSILASTIRMA ──────────────────────────────────────
if SMOTE_METHOD == 'auto':
    print(f'\n{SEP2}')
    print('ADIM 5 -- Oversampling Karsilastirmasi')
    print(SEP2)
    print(f'  {"Yontem":<16} {"F1":>8} {"MCC":>8} {"PR-AUC":>8} {"ROC-AUC":>8} {"Sure (dk)":>10}')
    print(f'  {"-"*62}')
    for method, res in method_res.items():
        if res is None:
            print(f'  {method.upper():<16} {"HATA":>8}')
            continue
        mark = ' <- KAZANAN' if method == best_smote else ''
        print(f'  {method.upper():<16} {res["f1"]:>8.4f} {res["mcc"]:>8.4f} '
              f'{res["pr"]:>8.4f} {res["roc"]:>8.4f} {res.get("time_min",0):>10.1f}{mark}')

# ─── ADIM 6: ALT GRUP DEGERLENDIRMESI (KANSER / PAH / CFTR) ──────────────────
print(f'\n{SEP2}')
print('ADIM 6 -- Alt Grup Degerlendirmesi (KANSER / PAH / CFTR)')
print('  Model tum MASTER verisiyle egitilip alt gruplara uygulanıyor...')
print(SEP2)

prep_final = ColumnTransformer([
    ('num', Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('scl', RobustScaler()),
        ('vt',  VarianceThreshold(1e-4)),
    ]), num_cols),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ]), cat_all),
])
X_full_proc = prep_final.fit_transform(X)

# Alt grup değerlendirmesi için son model eğitimi
# Bu aşamada tüm eğitim verisi (MASTER) kullanıldığından SMOTE tüm veriye uygulanabilir
try:
    os_final = get_oversampler(best_smote)
    X_full_res, y_full_res = os_final.fit_resample(X_full_proc, y)
except Exception:
    X_full_res, y_full_res = X_full_proc, y

stack_final = build_v5_stack(final_lgb_params, final_xgb_params, final_cat_params)
stack_final.fit(X_full_res, y_full_res)

subgroup_results = {}
for name, path in {k: v for k, v in PATHS.items() if k != 'MASTER'}.items():
    dfs = pd.read_csv(path)
    dfs['Label'] = dfs['Label'].astype(int)
    dfs = extract_aa_features(dfs)
    dfs = add_interaction_features(dfs, ek_cols)
    dfs = add_missing_pattern_features(dfs, al_cols, ek_cols)

    # Alt grup veri setinde bulunmayan sütunlar NaN ile doldurulur
    Xs = pd.DataFrame(index=dfs.index)
    for c in num_cols + cat_all:
        Xs[c] = dfs[c] if c in dfs.columns else np.nan
    ys = dfs['Label']

    Xs_proc = prep_final.transform(Xs)
    prob    = stack_final.predict_proba(Xs_proc)[:, 1]

    # PAH için ayrı bir eşik analizi yapılmaktadır (V5'te de aynı strateji uygulanmıştı)
    if name == 'PAH':
        pah_thresh, pah_f1 = find_best_threshold(ys, prob)
        pah_thresh_recall  = max(pah_thresh - 0.10, 0.20)
        print(f'\n  [PAH Esik Analizi] Optimal F1 esigi: {pah_thresh}')
        for pt in [pah_thresh_recall, pah_thresh, round(pah_thresh + 0.05, 2)]:
            yp  = (prob >= pt).astype(int)
            f1t = f1_score(ys, yp, zero_division=0)
            mct = matthews_corrcoef(ys, yp)
            print(f'    Esik={pt:.2f} -> F1:{f1t:.4f}  MCC:{mct:.4f}')
        use_thresh = pah_thresh
    else:
        use_thresh = final_thresh

    yp  = (prob >= use_thresh).astype(int)
    res = {
        'F1':      f1_score(ys, yp, zero_division=0),
        'MCC':     matthews_corrcoef(ys, yp),
        'PR-AUC':  average_precision_score(ys, prob),
        'ROC-AUC': roc_auc_score(ys, prob),
    }
    subgroup_results[name] = res
    print(f'  [{name}] F1:{res["F1"]:.4f}  MCC:{res["MCC"]:.4f}  '
          f'PR-AUC:{res["PR-AUC"]:.4f}  ROC-AUC:{res["ROC-AUC"]:.4f}  (esik={use_thresh})')

# ─── ADIM 7: VERSIYON KARSILASTIRMA ──────────────────────────────────────────
print(f'\n{SEP2}')
print('ADIM 7 -- Versiyon Karsilastirmasi (MASTER F1 / MCC)')
print(SEP2)

history = [
    ('V1',  'Baseline LGB+XGB',                   0.8711, 0.5054, 0.9207),
    ('V2',  'Missing Indicator + Threshold',       0.8903, 0.5301, 0.9206),
    ('V3',  'Biochem AA + Optuna + SHAP Top-100',  0.8929, 0.5401, None),
    ('V4',  'CatBoost Stacking + EK Interact',     0.8958, 0.5434, 0.9199),
    ('V5',  'GPU Stacking (100 trial Optuna)',      0.8941, 0.5370, 0.9222),
    ('V6',  'Leak-Free + Grantham + LGBM Meta',    0.8919, 0.5450, 0.9116),
    ('V7',  f'V5+Optuna{OPTUNA_TRIALS}t+{best_smote.upper()}', v7_f1, v7_mcc, v7_pr),
]

print(f'  {"Ver":<4} {"Model":<43} {"F1":>8} {"MCC":>8} {"PR-AUC":>8}')
print(f'  {"-"*75}')
for ver, model, f1, mcc, pr in history:
    pr_str = f'{pr:.4f}' if pr else '  --  '
    marker = ' <- V7' if ver == 'V7' else ''
    print(f'  {ver:<4} {model:<43} {f1:>8.4f} {mcc:>8.4f} {pr_str:>8}{marker}')

v5_f1, v5_mcc = 0.8941, 0.5370
delta_f1  = v7_f1  - v5_f1
delta_mcc = v7_mcc - v5_mcc
print(f'\n  V7 - V5 Farki: dF1={delta_f1:+.4f}  dMCC={delta_mcc:+.4f}')
if delta_f1 > 0 and delta_mcc > 0:
    print('  V7, V5yi hem F1 hem MCCde gecti!')
elif delta_f1 > 0 or delta_mcc > 0:
    print('  V7 V5e kiyasla kismi iyilesme sagladi.')
else:
    print('  V7 henuz V5i gecemedi. OPTUNA_TRIALS artir veya SMOTE yontemini degistir.')

# Tüm nihai sonuçlar JSON dosyasına kaydedilir
final_results = {
    'v7_f1':  v7_f1, 'v7_mcc': v7_mcc, 'v7_pr': v7_pr, 'v7_roc': v7_roc,
    'best_thresh': float(final_thresh), 'best_smote': best_smote,
    'fn': int(cm[1, 0]), 'fp': int(cm[0, 1]),
    'delta_f1_vs_v5': float(delta_f1), 'delta_mcc_vs_v5': float(delta_mcc),
    'optuna_trials': OPTUNA_TRIALS, 'gpu': GPU_AVAILABLE,
    'subgroup_results': {k: {mk: float(mv) for mk, mv in v.items()}
                         for k, v in subgroup_results.items()},
}
with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)
print(f'\n  Tum sonuclar kaydedildi: {RESULTS_FILE}')

print(f'\n{SEP}')
print(f'  AlgoMed V7 tamamlandi!')
print(f'  F1={v7_f1:.4f}  MCC={v7_mcc:.4f}  Esik={final_thresh}  SMOTE={best_smote.upper()}')
print(SEP)
