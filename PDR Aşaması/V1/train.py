"""
========================================================
TEKNOFEST 2026 – Sağlıkta Yapay Zeka Yarışması
V1 — Baseline Model Eğitimi
Model: LightGBM + XGBoost Soft-Voting Ensemble
Çalıştırma: PDR Aşaması/V1/ klasöründen  →  python train.py
========================================================
"""

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
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

import lightgbm as lgb
import xgboost as xgb

# ─── YOLLAR ───────────────────────────────────────────
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'universite-veri-seti', 'EĞİTİM (TRAIN) SETLERİ')

PATHS = {
    'MASTER': os.path.join(BASE, 'YARISMA_TRAIN_MASTER.csv'),
    'KANSER': os.path.join(BASE, 'YARISMA_TRAIN_KANSER.csv'),
    'PAH':    os.path.join(BASE, 'YARISMA_TRAIN_PAH.csv'),
    'CFTR':   os.path.join(BASE, 'YARISMA_TRAIN_CFTR.csv'),
}

SEP = '=' * 60
SEP2 = '-' * 50

# ─── 1. VERİ YÜKLEME ──────────────────────────────────
print(SEP)
print('1. VERİ YÜKLEME')
print(SEP)

df = pd.read_csv(PATHS['MASTER'])
df['Label'] = df['Label'].astype(int)

# Sütun grupları
al_cols  = [c for c in df.columns if c.startswith('AL_')]
cat_cols = [c for c in df.columns if c.startswith('CAT_')]
ek_cols  = [c for c in df.columns if c.startswith('EK_')]
aa_cols  = [c for c in df.columns if c.startswith('AA_')]
num_cols = al_cols + ek_cols           # 343 sayısal sütun
cat_all  = cat_cols + aa_cols          # 8 kategorik sütun

print(f'  MASTER boyutu    : {df.shape}')
print(f'  Patojenik (1)    : {(df.Label==1).sum()} ({(df.Label==1).mean()*100:.1f}%)')
print(f'  Benign    (0)    : {(df.Label==0).sum()} ({(df.Label==0).mean()*100:.1f}%)')
print(f'  Sayısal özellik  : {len(num_cols)}')
print(f'  Kategorik özellik: {len(cat_all)}')

X = df[num_cols + cat_all]
y = df['Label']

# ─── 2. ÖN İŞLEME PIPELINE ────────────────────────────
print(f'\n{SEP}')
print('2. ÖN İŞLEME')
print(SEP)

# Sayısal pipeline:
#   1. Median imputation  (eksik ~%90 olan sütunlar dahil)
#   2. RobustScaler       (aykırı değerlere karşı dayanıklı)
#   3. VarianceThreshold  (sabit sütunları at)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  RobustScaler()),
    ('vt',      VarianceThreshold(threshold=1e-4)),
])

# Kategorik pipeline:
#   1. Eksik → 'MISSING' ile doldur
#   2. OrdinalEncoder → integer (LightGBM native cat desteği için)
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('encoder', OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )),
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_all),
])

X_proc = preprocessor.fit_transform(X)
n_features_after = X_proc.shape[1]

print(f'  İşlem öncesi özellik: {len(num_cols) + len(cat_all)}')
print(f'  İşlem sonrası özellik: {n_features_after}')
print(f'  Median imputation → eksik değerler dolduruldu ✅')
print(f'  RobustScaler → ölçekleme uygulandı ✅')
print(f'  VarianceThreshold → sabit sütunlar temizlendi ✅')
print(f'  OrdinalEncoder → kategorikler sayıya çevrildi ✅')

# ─── 3. MODEL TANIMI ──────────────────────────────────
print(f'\n{SEP}')
print('3. MODEL TANIMI')
print(SEP)

# Sınıf dengesi için ağırlık oranı
n_benign     = (y == 0).sum()
n_pathogenic = (y == 1).sum()
spw = n_benign / n_pathogenic   # ~0.36  → Patojenik ağırlığını azalt
print(f'  Sınıf oranı (benign/pathogenic): {spw:.4f}')
print(f'  scale_pos_weight = {spw:.4f}  (dengesizliği dengeler)')

lgbm_model = lgb.LGBMClassifier(
    n_estimators      = 500,
    learning_rate     = 0.05,
    max_depth         = 6,
    num_leaves        = 31,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    scale_pos_weight  = spw,
    random_state      = 42,
    verbose           = -1,
    n_jobs            = -1,
)

xgb_model = xgb.XGBClassifier(
    n_estimators     = 500,
    learning_rate    = 0.05,
    max_depth        = 5,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    scale_pos_weight = spw,
    eval_metric      = 'logloss',
    random_state     = 42,
    verbosity        = 0,
    n_jobs           = -1,
)

ensemble = VotingClassifier(
    estimators=[('lgbm', lgbm_model), ('xgb', xgb_model)],
    voting='soft',
    weights=[1, 1],
)

print('  LightGBM + XGBoost Soft-Voting Ensemble tanımlandı ✅')

# ─── 4. ÇAPRAZ DOĞRULAMA (MASTER) ─────────────────────
print(f'\n{SEP}')
print('4. ÇAPRAZ DOĞRULAMA — MASTER (5-Fold Stratified)')
print(SEP)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

f1_list, auc_list, prauc_list, mcc_list, bacc_list = [], [], [], [], []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_proc, y), 1):
    X_tr, X_val = X_proc[tr_idx], X_proc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    ensemble.fit(X_tr, y_tr)
    y_prob = ensemble.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    f1   = f1_score(y_val, y_pred)
    auc  = roc_auc_score(y_val, y_prob)
    prau = average_precision_score(y_val, y_prob)
    mcc  = matthews_corrcoef(y_val, y_pred)
    bacc = balanced_accuracy_score(y_val, y_pred)

    f1_list.append(f1);  auc_list.append(auc)
    prauc_list.append(prau); mcc_list.append(mcc); bacc_list.append(bacc)

    print(f'  Fold {fold} → F1: {f1:.4f}  ROC-AUC: {auc:.4f}  '
          f'PR-AUC: {prau:.4f}  MCC: {mcc:.4f}')

print(f'\n{SEP2}')
print(f'  Ortalama F1       : {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}')
print(f'  Ortalama ROC-AUC  : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}')
print(f'  Ortalama PR-AUC   : {np.mean(prauc_list):.4f} ± {np.std(prauc_list):.4f}')
print(f'  Ortalama MCC      : {np.mean(mcc_list):.4f} ± {np.std(mcc_list):.4f}')
print(f'  Dengeli Doğruluk  : {np.mean(bacc_list):.4f} ± {np.std(bacc_list):.4f}')

# ─── 5. FİNAL MODEL + SINIFLANDIRMA RAPORU ────────────
print(f'\n{SEP}')
print('5. FİNAL MODEL — Tüm MASTER Verisiyle CV Tahminleri')
print(SEP)

ensemble.fit(X_proc, y)
y_prob_all = cross_val_predict(ensemble, X_proc, y,
                                cv=cv, method='predict_proba')[:, 1]
y_pred_all = (y_prob_all >= 0.5).astype(int)

print('\nSınıflandırma Raporu (CV üzerinden):')
print(classification_report(y, y_pred_all, target_names=['Benign', 'Patojenik']))

cm = confusion_matrix(y, y_pred_all)
print(f'Confusion Matrix:')
print(f'  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}')
print(f'  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}')
print(f'\n  Yanlış Negatif (FN – gizli patoloji): {cm[1,0]}  ← klinik risk')
print(f'  Yanlış Pozitif (FP – gereksiz alarm) : {cm[0,1]}')

# ─── 6. EŞİK ANALİZİ ─────────────────────────────────
print(f'\n{SEP}')
print('6. KARAR EŞİĞİ ANALİZİ')
print(SEP)

print(f'\n  {"Eşik":>6} | {"Precision":>10} | {"Recall":>8} | {"F1":>8} | {"MCC":>8}')
print('  ' + '-'*50)

best_f1, best_thresh = 0, 0.5
for thresh in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    yp = (y_prob_all >= thresh).astype(int)
    from sklearn.metrics import precision_score, recall_score
    prec = precision_score(y, yp, zero_division=0)
    rec  = recall_score(y, yp, zero_division=0)
    f1t  = f1_score(y, yp, zero_division=0)
    mcct = matthews_corrcoef(y, yp)
    marker = ' ←' if f1t > best_f1 else ''
    if f1t > best_f1:
        best_f1, best_thresh = f1t, thresh
    print(f'  {thresh:>6.2f} | {prec:>10.4f} | {rec:>8.4f} | {f1t:>8.4f} | {mcct:>8.4f}{marker}')

print(f'\n  En iyi eşik: {best_thresh}  (F1={best_f1:.4f})')

# ─── 7. ALT GRUP DEĞERLENDİRME ───────────────────────
print(f'\n{SEP}')
print('7. ALT GRUP DEĞERLENDİRME (MASTER modeliyle)')
print(SEP)

subgroups = {k: v for k, v in PATHS.items() if k != 'MASTER'}

print(f'\n  {"Grup":>8} | {"N":>5} | {"F1":>8} | {"MCC":>8} | {"PR-AUC":>8} | {"ROC-AUC":>8}')
print('  ' + '-'*60)

for name, path in subgroups.items():
    dfs   = pd.read_csv(path)
    dfs['Label'] = dfs['Label'].astype(int)
    Xs    = dfs[num_cols + cat_all]
    ys    = dfs['Label']
    Xs_p  = preprocessor.transform(Xs)

    yp    = ensemble.predict_proba(Xs_p)[:, 1]
    yc    = (yp >= best_thresh).astype(int)

    f1s   = f1_score(ys, yc, zero_division=0)
    mccs  = matthews_corrcoef(ys, yc)
    praus = average_precision_score(ys, yp)
    aucs  = roc_auc_score(ys, yp)

    print(f'  {name:>8} | {len(dfs):>5} | {f1s:>8.4f} | {mccs:>8.4f} | '
          f'{praus:>8.4f} | {aucs:>8.4f}')

# ─── 8. SHAP ÖNEMLİLİĞİ ──────────────────────────────
print(f'\n{SEP}')
print('8. AÇIKLANABİLİRLİK — SHAP (LightGBM)')
print(SEP)

try:
    import shap
    lgbm_fitted = ensemble.named_estimators_['lgbm']
    explainer   = shap.TreeExplainer(lgbm_fitted)
    # Bellek için örnekle çalış
    sample_idx  = np.random.RandomState(42).choice(len(X_proc), size=min(500, len(X_proc)), replace=False)
    sv          = explainer.shap_values(X_proc[sample_idx])
    sv_arr      = sv[1] if isinstance(sv, list) else sv

    # Orijinal özellik isimlerini yeniden oluştur
    num_feats_after = preprocessor.named_transformers_['num'].named_steps['vt'].get_support()
    kept_num = [c for c, keep in zip(num_cols, num_feats_after) if keep]
    feat_names = kept_num + cat_all

    mean_shap = np.abs(sv_arr).mean(axis=0)
    shap_df = pd.DataFrame({'Ozellik': feat_names[:len(mean_shap)],
                             'Ort_SHAP': mean_shap})
    shap_df = shap_df.sort_values('Ort_SHAP', ascending=False)
    print('\n  Top 15 En Önemli Özellik (SHAP):')
    print(f'  {"Özellik":>12} | {"Ort |SHAP|":>12}')
    print('  ' + '-'*28)
    for _, row in shap_df.head(15).iterrows():
        print(f'  {row.Ozellik:>12} | {row.Ort_SHAP:>12.5f}')
except Exception as e:
    print(f'  SHAP hesaplanamadı: {e}')

# ─── 9. ÖZET ──────────────────────────────────────────
print(f'\n{SEP}')
print('9. V1 SONUÇ ÖZETİ')
print(SEP)
print(f'  Model      : LightGBM + XGBoost Soft-Voting Ensemble')
print(f'  Veri       : MASTER ({len(df)} varyant)')
print(f'  CV         : 5-Fold Stratified')
print(f'  Eşik       : {best_thresh}')
print(f'  F1  (ort.) : {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}')
print(f'  MCC (ort.) : {np.mean(mcc_list):.4f} ± {np.std(mcc_list):.4f}')
print(f'  PR-AUC     : {np.mean(prauc_list):.4f} ± {np.std(prauc_list):.4f}')
print(f'  ROC-AUC    : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}')
print(f'\n✅ V1 train.py tamamlandı. Sonuçları V1/README.md dosyasına yazınız.')
