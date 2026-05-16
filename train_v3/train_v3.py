# -*- coding: utf-8 -*-

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from category_encoders import TargetEncoder

# --- YOLLAR (Yeni Klasor Adina Gore Guncellendi) ---
# Eger klasor baska bir yerdeyse (Orn: Masaustu) buradaki yolu kendi bilgisayarina gore duzenleyebilirsin.
BASE = r"C:\Users\admin\Downloads\universite-veri-seti\EGITIM_TRAIN_SETLERI"

# --- YOLLAR ---
PATHS = {
    'MASTER': r"C:\Users\admin\Downloads\universite-veri-seti\EGITIM_ TRAIN_ SETLERI\YARISMA_TRAIN_MASTER.csv"
}

SEP = '=' * 60

# --- 1. VERI YUKLEME VE V2 MIRASI FILTRELEME ---
print(SEP)
print('1. VERI YUKLEME VE FILTRELEME (V2 + V3)')
print(SEP)

df = pd.read_csv(PATHS['MASTER'])
df['Label'] = df['Label'].astype(int)

al_cols  = [c for c in df.columns if c.startswith('AL_')]
cat_cols = [c for c in df.columns if c.startswith('CAT_')]
ek_cols  = [c for c in df.columns if c.startswith('EK_')]
aa_cols  = [c for c in df.columns if c.startswith('AA_')]

num_cols = al_cols + ek_cols
cat_all  = cat_cols + aa_cols

# V2 Kazanimi: CAT_6 Kaldirildi (%97+ eksik)
if 'CAT_6' in cat_all:
    cat_all.remove('CAT_6')
    print('  [V2] CAT_6 kaldirildi (%97+ missing)')

# V2 Kazanimi: %80'den fazla eksik olan sayisal sutunlari kaldir
missing_ratio = df[num_cols].isnull().mean()
keep_num_cols = missing_ratio[missing_ratio < 0.80].index.tolist()
drop_num_cols = missing_ratio[missing_ratio >= 0.80].index.tolist()

print(f'  [V2] Silinen yuksek eksikli sayisal sutun: {len(drop_num_cols)}')
num_cols = keep_num_cols

X = df[num_cols + cat_all]
y = df['Label']

# --- 2. ON ISLEME PIPELINE ---
print(f'\n{SEP}')
print('2. ON ISLEME (Target Encoding Eklendi)')
print(SEP)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('scaler', RobustScaler()),
    ('vt', VarianceThreshold(threshold=1e-4)),
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('encoder', TargetEncoder(smoothing=10)) 
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_all),
])

X_proc = preprocessor.fit_transform(X, y)
print('  [V2] Missing Indicator eklendi')
print('  [V3] Target Encoding uygulandi')


# --- 3. MODEL TANIMI (UCLU ENSEMBLE) ---
print(f'\n{SEP}')
print('3. MODEL TANIMI (LightGBM + XGBoost + CatBoost)')
print(SEP)

lgbm_model = lgb.LGBMClassifier(
    n_estimators=800, learning_rate=0.03, max_depth=6, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1
)

xgb_model = xgb.XGBClassifier(
    n_estimators=800, learning_rate=0.03, max_depth=5, subsample=0.8,
    colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=-1
)

cb_model = CatBoostClassifier(
    iterations=800, learning_rate=0.03, depth=6,
    random_state=42, verbose=0, thread_count=-1
)

ensemble = VotingClassifier(
    estimators=[('lgbm', lgbm_model), ('xgb', xgb_model), ('cat', cb_model)],
    voting='soft',
    weights=[1, 1, 1.2] 
)

# --- 4. CAPRAZ DOGRULAMA ---
print(f'\n{SEP}')
print('4. CAPRAZ DOGRULAMA (5-Fold Stratified)')
print(SEP)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# TRAFIK SIKISIKLIGINI COZEN KISIM (Bilgisayar donmayacak)
y_prob_all = cross_val_predict(ensemble, X_proc, y, cv=cv, method='predict_proba', n_jobs=1)[:, 1]

# --- 5. KARAR ESIGI (THRESHOLD) OPTIMIZASYONU ---
print(f'\n{SEP}')
print('5. KARAR ESIGI ANALIZI')
print(SEP)

best_f1, best_mcc, best_thresh = 0, 0, 0.5

print(f'  {"Esik":>6} | {"Precision":>10} | {"Recall":>8} | {"F1":>8} | {"MCC":>8}')
print('  ' + '-'*50)

for thresh in np.arange(0.20, 0.71, 0.05):
    yp = (y_prob_all >= thresh).astype(int)
    from sklearn.metrics import precision_score, recall_score
    prec = precision_score(y, yp, zero_division=0)
    rec  = recall_score(y, yp, zero_division=0)
    f1t  = f1_score(y, yp, zero_division=0)
    mcct = matthews_corrcoef(y, yp)
    
    if f1t > best_f1:
        best_f1, best_thresh, best_mcc = f1t, thresh, mcct
        
    print(f'  {thresh:>6.2f} | {prec:>10.4f} | {rec:>8.4f} | {f1t:>8.4f} | {mcct:>8.4f}')

print(f'\n  [SONUC] En iyi esik: {best_thresh:.2f}  --> F1: {best_f1:.4f} | MCC: {best_mcc:.4f}')