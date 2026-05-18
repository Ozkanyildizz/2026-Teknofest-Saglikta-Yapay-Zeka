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
import optuna
from imblearn.over_sampling import SMOTE

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

# Ayarlar
RUN_OPTUNA = True
OPTUNA_TRIALS = 10  # Hızlı göstermek için kısa tutuldu. Gerçekte 50-100 olabilir.
TOP_N_FEATURES = 100

# ─── BİYOKİMYASAL ÖZELLİK SÖZLÜKLERİ ─────────────────
AA_PROPERTIES = {
    'A': {'polarity': 0, 'charge': 0, 'hydropathy': 1.8, 'weight': 89.1},
    'R': {'polarity': 1, 'charge': 1, 'hydropathy': -4.5, 'weight': 174.2},
    'N': {'polarity': 1, 'charge': 0, 'hydropathy': -3.5, 'weight': 132.1},
    'D': {'polarity': 1, 'charge': -1, 'hydropathy': -3.5, 'weight': 133.1},
    'C': {'polarity': 0, 'charge': 0, 'hydropathy': 2.5, 'weight': 121.2},
    'E': {'polarity': 1, 'charge': -1, 'hydropathy': -3.5, 'weight': 147.1},
    'Q': {'polarity': 1, 'charge': 0, 'hydropathy': -3.5, 'weight': 146.2},
    'G': {'polarity': 0, 'charge': 0, 'hydropathy': -0.4, 'weight': 75.1},
    'H': {'polarity': 1, 'charge': 1, 'hydropathy': -3.2, 'weight': 155.2},
    'I': {'polarity': 0, 'charge': 0, 'hydropathy': 4.5, 'weight': 131.2},
    'L': {'polarity': 0, 'charge': 0, 'hydropathy': 3.8, 'weight': 131.2},
    'K': {'polarity': 1, 'charge': 1, 'hydropathy': -3.9, 'weight': 146.2},
    'M': {'polarity': 0, 'charge': 0, 'hydropathy': 1.9, 'weight': 149.2},
    'F': {'polarity': 0, 'charge': 0, 'hydropathy': 2.8, 'weight': 165.2},
    'P': {'polarity': 0, 'charge': 0, 'hydropathy': -1.6, 'weight': 115.1},
    'S': {'polarity': 1, 'charge': 0, 'hydropathy': -0.8, 'weight': 105.1},
    'T': {'polarity': 1, 'charge': 0, 'hydropathy': -0.7, 'weight': 119.1},
    'W': {'polarity': 0, 'charge': 0, 'hydropathy': -0.9, 'weight': 204.2},
    'Y': {'polarity': 1, 'charge': 0, 'hydropathy': -1.3, 'weight': 181.2},
    'V': {'polarity': 0, 'charge': 0, 'hydropathy': 4.2, 'weight': 117.1},
}

def extract_aa_features(df):
    """AA_1 ve AA_2 sütunlarından biyokimyasal özellikleri çıkarır"""
    df = df.copy()
    
    for aa_col, prefix in [('AA_1', 'AA1'), ('AA_2', 'AA2')]:
        df[f'{prefix}_polarity'] = df[aa_col].map(lambda x: AA_PROPERTIES.get(x, {}).get('polarity', np.nan))
        df[f'{prefix}_charge'] = df[aa_col].map(lambda x: AA_PROPERTIES.get(x, {}).get('charge', np.nan))
        df[f'{prefix}_hydro'] = df[aa_col].map(lambda x: AA_PROPERTIES.get(x, {}).get('hydropathy', np.nan))
        df[f'{prefix}_weight'] = df[aa_col].map(lambda x: AA_PROPERTIES.get(x, {}).get('weight', np.nan))
        
    # Farklar
    df['AA_hydro_diff'] = abs(df['AA1_hydro'] - df['AA2_hydro'])
    df['AA_weight_diff'] = abs(df['AA1_weight'] - df['AA2_weight'])
    df['AA_charge_change'] = (df['AA1_charge'] != df['AA2_charge']).astype(float)
    df.loc[df['AA1_charge'].isna() | df['AA2_charge'].isna(), 'AA_charge_change'] = np.nan
    df['AA_polarity_change'] = (df['AA1_polarity'] != df['AA2_polarity']).astype(float)
    df.loc[df['AA1_polarity'].isna() | df['AA2_polarity'].isna(), 'AA_polarity_change'] = np.nan
    
    return df

# ─── 1. VERİ YÜKLEME ──────────────────────────────────
print(SEP)
print('1. VERİ YÜKLEME VE ÖZELLİK ÇIKARIMI (V3)')
print(SEP)

df = pd.read_csv(PATHS['MASTER'])
df['Label'] = df['Label'].astype(int)

# Biyokimyasal özellikleri çıkar
df = extract_aa_features(df)
new_aa_num_cols = [
    'AA1_polarity', 'AA1_charge', 'AA1_hydro', 'AA1_weight',
    'AA2_polarity', 'AA2_charge', 'AA2_hydro', 'AA2_weight',
    'AA_hydro_diff', 'AA_weight_diff', 'AA_charge_change', 'AA_polarity_change'
]

# Sütun grupları
al_cols  = [c for c in df.columns if c.startswith('AL_')]
cat_cols = [c for c in df.columns if c.startswith('CAT_')]
ek_cols  = [c for c in df.columns if c.startswith('EK_')]
aa_cols  = ['AA_1', 'AA_2']

num_cols = al_cols + ek_cols + new_aa_num_cols
cat_all  = cat_cols + aa_cols

if 'CAT_6' in cat_all:
    cat_all.remove('CAT_6')

# Çok eksik sütunları kaldır (%80+)
missing_ratio = df[num_cols].isnull().mean()
keep_num_cols = missing_ratio[missing_ratio < 0.80].index.tolist()
num_cols = keep_num_cols

print(f'  Yeni eklenen Biyokimyasal Özellikler: {len(new_aa_num_cols)}')
print(f'  Tutulan sayısal sütun      : {len(num_cols)}')
print(f'  MASTER boyutu              : {df.shape}')

X = df[num_cols + cat_all]
y = df['Label']

# ─── 2. ÖN İŞLEME PIPELINE ────────────────────────────
print(f'\n{SEP}')
print('2. ÖN İŞLEME')
print(SEP)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('scaler', RobustScaler()),
    ('vt', VarianceThreshold(threshold=1e-4)),
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_all),
])

X_proc = preprocessor.fit_transform(X)

print(f'  İşlem sonrası özellik: {X_proc.shape[1]}')

# --- SHAP HATASI ÇÖZÜMÜ İÇİN İSİMLENDİRME ---
num_feats_after = preprocessor.named_transformers_['num'].named_steps['vt'].get_support()
kept_num = [c for c, keep in zip(num_cols, num_feats_after) if keep]

# Doğru indikatör isimleri
imputer = preprocessor.named_transformers_['num'].named_steps['imputer']
indicator_indices = imputer.indicator_.features_
indicator_features = [f"{num_cols[idx]}_missing" for idx in indicator_indices]
# Varyans filtresinden geçenleri al (indikatörlerin vt'den geçip geçmediğine dikkat)
indicator_mask = num_feats_after[len(num_cols):] # vt maskesinin indicator kısmı
kept_indicators = [f for f, keep in zip(indicator_features, indicator_mask) if keep]

feat_names = kept_num + kept_indicators + cat_all

# ─── 3. HİPERPARAMETRE OPTİMİZASYONU (OPTUNA) ─────────
best_lgb_params = {
    'n_estimators': 800, 'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'verbose': -1
}

if RUN_OPTUNA:
    print(f'\n{SEP}')
    print(f'3. OPTUNA İLE HİPERPARAMETRE OPTİMİZASYONU (Trials: {OPTUNA_TRIALS})')
    print(SEP)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        clf = lgb.LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        mcc_scores = []
        for tr_idx, val_idx in cv.split(X_proc, y):
            clf.fit(X_proc[tr_idx], y.iloc[tr_idx])
            y_pred = clf.predict(X_proc[val_idx])
            mcc_scores.append(matthews_corrcoef(y.iloc[val_idx], y_pred))
            
        return np.mean(mcc_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    print(f"\n  En iyi Optuna MCC Skoru: {study.best_value:.4f}")
    best_lgb_params.update(study.best_params)
    print(f"  Bulunan En İyi Parametreler: {study.best_params}")

# ─── 4. MODEL TANIMI ──────────────────────────────────
print(f'\n{SEP}')
print('4. MODEL TANIMI VE EĞİTİMİ')
print(SEP)

lgbm_model = lgb.LGBMClassifier(**best_lgb_params, n_jobs=-1)

xgb_model = xgb.XGBClassifier(
    n_estimators=800, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
    random_state=42, verbosity=0, n_jobs=-1
)

ensemble = VotingClassifier(
    estimators=[('lgbm', lgbm_model), ('xgb', xgb_model)],
    voting='soft', weights=[1, 1]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_prob_all = cross_val_predict(ensemble, X_proc, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]

# En iyi eşik
best_f1, best_thresh = 0, 0.5
for thresh in np.arange(0.20, 0.71, 0.01):
    yp = (y_prob_all >= thresh).astype(int)
    f1t = f1_score(y, yp, zero_division=0)
    if f1t > best_f1:
        best_f1, best_thresh = f1t, thresh

y_pred_all = (y_prob_all >= best_thresh).astype(int)
print(f"  En İyi Eşik Değeri: {best_thresh:.2f}")
print(f"  MASTER (Cross-Val) F1 : {best_f1:.4f}")
print(f"  MASTER (Cross-Val) MCC: {matthews_corrcoef(y, y_pred_all):.4f}")

# ─── 5. SHAP VE TOP-N ÖZELLİK SEÇİMİ ──────────────────
print(f'\n{SEP}')
print('5. SHAP ANALİZİ VE TOP-N SEÇİMİ')
print(SEP)

ensemble.fit(X_proc, y)
try:
    import shap
    lgbm_fitted = ensemble.named_estimators_['lgbm']
    explainer = shap.TreeExplainer(lgbm_fitted)
    sample_idx = np.random.RandomState(42).choice(len(X_proc), size=min(500, len(X_proc)), replace=False)
    sv = explainer.shap_values(X_proc[sample_idx])
    sv_arr = sv[1] if isinstance(sv, list) else sv
    
    mean_shap = np.abs(sv_arr).mean(axis=0)
    shap_df = pd.DataFrame({'Ozellik': feat_names[:len(mean_shap)], 'Ort_SHAP': mean_shap})
    shap_df = shap_df.sort_values('Ort_SHAP', ascending=False)
    
    print('\n  Top 15 En Önemli Özellik (SHAP):')
    for _, row in shap_df.head(15).iterrows():
        print(f'  {row.Ozellik:>18} | {row.Ort_SHAP:>10.5f}')
        
    top_n_features_idx = shap_df.head(TOP_N_FEATURES).index
    X_proc_topn = X_proc[:, top_n_features_idx]
    print(f"\n  Top {TOP_N_FEATURES} özellik seçildi. Model bu alt küme ile tekrar değerlendiriliyor...")
    
    y_prob_topn = cross_val_predict(ensemble, X_proc_topn, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    y_pred_topn = (y_prob_topn >= best_thresh).astype(int)
    print(f"  Top-{TOP_N_FEATURES} Model F1 : {f1_score(y, y_pred_topn):.4f}")
    print(f"  Top-{TOP_N_FEATURES} Model MCC: {matthews_corrcoef(y, y_pred_topn):.4f}")
    
except Exception as e:
    print(f'  SHAP hesaplanamadı veya hata oluştu: {e}')

# ─── 6. ALT GRUP (ÖZEL PAH - SMOTE DEĞERLENDİRMESİ) ───
print(f'\n{SEP}')
print('6. ALT GRUPLAR VE PAH ÖZEL MODELİ')
print(SEP)

subgroups = {k: v for k, v in PATHS.items() if k != 'MASTER'}

for name, path in subgroups.items():
    dfs = pd.read_csv(path)
    dfs['Label'] = dfs['Label'].astype(int)
    dfs = extract_aa_features(dfs)
    Xs = dfs[num_cols + cat_all]
    ys = dfs['Label']
    Xs_p = preprocessor.transform(Xs)
    
    yp = ensemble.predict_proba(Xs_p)[:, 1]
    yc = (yp >= best_thresh).astype(int)
    f1s = f1_score(ys, yc, zero_division=0)
    mccs = matthews_corrcoef(ys, yc)
    print(f'  [MASTER MODELİ] {name:>8} | F1: {f1s:.4f} | MCC: {mccs:.4f}')

print('\n  [PAH ÖZEL MODELİ - SMOTE İLE]')
df_pah = pd.read_csv(PATHS['PAH'])
df_pah['Label'] = df_pah['Label'].astype(int)
df_pah = extract_aa_features(df_pah)
X_pah = df_pah[num_cols + cat_all]
y_pah = df_pah['Label']
X_pah_proc = preprocessor.transform(X_pah)

# SMOTE Sadece Eğitim Verisinde (CV içinde)
pah_mccs = []
pah_f1s = []
for tr_idx, val_idx in cv.split(X_pah_proc, y_pah):
    X_tr, y_tr = X_pah_proc[tr_idx], y_pah.iloc[tr_idx]
    X_val, y_val = X_pah_proc[val_idx], y_pah.iloc[val_idx]
    
    smote = SMOTE(random_state=42)
    X_tr_smote, y_tr_smote = smote.fit_resample(X_tr, y_tr)
    
    ensemble.fit(X_tr_smote, y_tr_smote)
    y_prob = ensemble.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= best_thresh).astype(int)
    
    pah_mccs.append(matthews_corrcoef(y_val, y_pred))
    pah_f1s.append(f1_score(y_val, y_pred))

print(f"  PAH Özel SMOTE Modeli CV MCC: {np.mean(pah_mccs):.4f}")
print(f"  PAH Özel SMOTE Modeli CV F1 : {np.mean(pah_f1s):.4f}")

print(f'\n{SEP}')
print('✅ V3 tamamlandı!')
print(SEP)
