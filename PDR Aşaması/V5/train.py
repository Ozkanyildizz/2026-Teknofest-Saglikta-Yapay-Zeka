import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

import lightgbm as lgb
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost kurulu değil. pip install catboost")

# ─── YOLLAR ───────────────────────────────────────────────────────────────────
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

SEP  = '=' * 65
SEP2 = '-' * 50

# ─── AYARLAR ──────────────────────────────────────────────────────────────────
RUN_OPTUNA      = False  # GPU ile 100 trial denemesi tamamlandı
OPTUNA_TRIALS   = 100
TOP_N_FEATURES  = 100
MISSING_THRESH  = 0.80

# Önceki Optuna çalışmasından en iyi parametreler
BEST_LGB_FROM_OPTUNA = {
    'n_estimators': 800,
    'learning_rate': 0.07917491343504915,
    'num_leaves': 74,
    'max_depth': 11,
    'subsample': 0.597804620160675,
    'colsample_bytree': 0.6513603446650886,
    'min_child_samples': 25,
    'scale_pos_weight': 0.33067145309795315,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}  # %80+ eksik sütunları kaldır

# ─── BİYOKİMYASAL ÖZELLİK SÖZLÜĞÜ (V3'ten) ───────────────────────────────────
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


# ─── YARDIMCI FONKSİYONLAR ────────────────────────────────────────────────────

def extract_aa_features(df):
    """AA_1/AA_2 → biyokimyasal sayısal özellikler (V3'ten alındı, aynen korundu)."""
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

    df['AA_hydro_diff']      = abs(df['AA1_hydro']    - df['AA2_hydro'])
    df['AA_weight_diff']     = abs(df['AA1_weight']   - df['AA2_weight'])
    df['AA_charge_change']   = (df['AA1_charge']   != df['AA2_charge']).astype(float)
    df['AA_polarity_change'] = (df['AA1_polarity'] != df['AA2_polarity']).astype(float)

    df.loc[df['AA1_charge'].isna()   | df['AA2_charge'].isna(),   'AA_charge_change']   = np.nan
    df.loc[df['AA1_polarity'].isna() | df['AA2_polarity'].isna(), 'AA_polarity_change'] = np.nan
    return df


def add_interaction_features(df, ek_cols):
    """
    YENİ (V4): EK_ sütunları arasında 2-yönlü çarpım etkileşim terimleri.
    EK_7 her versiyonda SHAP #1 — diğerleriyle etkileşimi bilgi taşıyabilir.
    """
    df = df.copy()
    for a, b in combinations(ek_cols, 2):
        df[f'{a}_x_{b}'] = df[a] * df[b]
    return df


def add_missing_pattern_features(df, al_cols, ek_cols):
    """
    YENİ (V4): Satır bazında eksiklik istatistikleri.
    EDA Bulgusu: missing_ratio_AL korelasyon=0.2258 — patojenikte çok daha yüksek!
    Log dönüşümü ile bu sinyali daha belirgin hale getiriyoruz.
    """
    df = df.copy()
    df['missing_count_AL']      = df[al_cols].isnull().sum(axis=1)
    df['missing_ratio_AL']      = df[al_cols].isnull().mean(axis=1)
    # Log transform: 0.42 → 0.62 farkını daha belirgin kılar
    df['missing_ratio_AL_log']  = np.log1p(df['missing_ratio_AL'])
    df['missing_count_EK']      = df[ek_cols].isnull().sum(axis=1)
    df['missing_ratio_EK']      = df[ek_cols].isnull().mean(axis=1)
    df['missing_count_ALL']     = df[al_cols + ek_cols].isnull().sum(axis=1)
    # EK_9 × missing_ratio_AL — çarpışan iki güçlü sinyal
    if 'EK_9' in df.columns:
        df['EK9_x_miss_AL'] = df['EK_9'] * df['missing_ratio_AL']
    return df


def find_best_threshold(y_true, y_prob, step=0.01):
    """F1'i maksimize eden karar eşiğini bul."""
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.20, 0.71, step):
        yp = (y_prob >= thresh).astype(int)
        f1t = f1_score(y_true, yp, zero_division=0)
        if f1t > best_f1:
            best_f1, best_thresh = f1t, thresh
    return round(best_thresh, 2), best_f1


def evaluate(y_true, y_prob, threshold, label=''):
    """Metrikleri hesapla ve yazdır."""
    yp = (y_prob >= threshold).astype(int)
    f1  = f1_score(y_true, yp, zero_division=0)
    mcc = matthews_corrcoef(y_true, yp)
    roc = roc_auc_score(y_true, y_prob)
    pr  = average_precision_score(y_true, y_prob)
    ba  = balanced_accuracy_score(y_true, yp)
    print(f'  {"[" + label + "]":<12} F1:{f1:.4f}  MCC:{mcc:.4f}  '
          f'PR-AUC:{pr:.4f}  ROC-AUC:{roc:.4f}  BalAcc:{ba:.4f}  Eşik:{threshold}')
    return {'F1': f1, 'MCC': mcc, 'PR-AUC': pr, 'ROC-AUC': roc}


# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print('  AlgoMed V5 — GPU Hızlandırılmış Stacking Ensemble (LGBM+XGB+CatBoost)')
print(SEP)

# ─── 1. VERİ YÜKLEME VE ÖZELLİK ÇIKARIMI ─────────────────────────────────────
print(f'\n{SEP2}')
print('ADIM 1 — Veri Yükleme ve Özellik Çıkarımı')
print(SEP2)

df = pd.read_csv(PATHS['MASTER'])
df['Label'] = df['Label'].astype(int)

# Sütun grupları
al_cols  = [c for c in df.columns if c.startswith('AL_')]
ek_cols  = [c for c in df.columns if c.startswith('EK_')]
cat_cols = [c for c in df.columns if c.startswith('CAT_') and c != 'CAT_6']
aa_cols  = ['AA_1', 'AA_2']

# V3'ten gelen biyokimyasal özellikler
df = extract_aa_features(df)
new_aa_num_cols = [
    'AA1_polarity', 'AA1_charge', 'AA1_hydro', 'AA1_weight',
    'AA2_polarity', 'AA2_charge', 'AA2_hydro', 'AA2_weight',
    'AA_hydro_diff', 'AA_weight_diff', 'AA_charge_change', 'AA_polarity_change'
]

# YENİ V4: EK_ etkileşim terimleri
df = add_interaction_features(df, ek_cols)
ek_interaction_cols = [f'{a}_x_{b}' for a, b in combinations(ek_cols, 2)]

# YENİ V4: Eksiklik örüntüsü özellikleri
df = add_missing_pattern_features(df, al_cols, ek_cols)
missing_pattern_cols = [
    'missing_count_AL', 'missing_ratio_AL', 'missing_ratio_AL_log',
    'missing_count_EK', 'missing_ratio_EK',
    'missing_count_ALL', 'EK9_x_miss_AL'
]

# %80+ eksik sayısal sütunları kaldır
all_num_base = al_cols + ek_cols + new_aa_num_cols + ek_interaction_cols + missing_pattern_cols
missing_ratio = df[all_num_base].isnull().mean()
num_cols = missing_ratio[missing_ratio < MISSING_THRESH].index.tolist()

# EDA Bulgusu: CAT_3, CAT_4, CAT_5 neredeyse özdeş patojenite oranları
# → CAT_4 ve CAT_5 kaldırılıyor (CAT_3 temsil edici)
cat_all = [c for c in cat_cols if c not in ('CAT_4', 'CAT_5')] + aa_cols

print(f'  Biyokimyasal özellikler : {len(new_aa_num_cols)}')
print(f'  EK_ etkileşim terimleri : {len(ek_interaction_cols)}')
print(f'  Eksiklik örüntü özellik : {len(missing_pattern_cols)}')
print(f'  Tutulan sayısal sütun   : {len(num_cols)}')
print(f'  Kategorik sütun         : {len(cat_all)}')
print(f'  MASTER boyutu           : {df.shape}')

X = df[num_cols + cat_all]
y = df['Label']

pos_ratio = y.mean()
scale_pw   = float(round((1 - pos_ratio) / pos_ratio, 4))  # CatBoost için düz Python float
print(f'\n  Sınıf dengesi — Patojenik: {pos_ratio:.2%} | scale_pos_weight: {scale_pw}')

# ─── 2. ÖN İŞLEME ─────────────────────────────────────────────────────────────
print(f'\n{SEP2}')
print('ADIM 2 — Ön İşleme Pipeline')
print(SEP2)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('scaler',  RobustScaler()),
    ('vt',      VarianceThreshold(threshold=1e-4)),
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
print(f'  İşlem sonrası özellik sayısı: {X_proc.shape[1]}')

# Özellik isim çıkarımı (SHAP için)
num_feats_mask = preprocessor.named_transformers_['num'].named_steps['vt'].get_support()
kept_num       = [c for c, k in zip(num_cols, num_feats_mask) if k]
imputer_step   = preprocessor.named_transformers_['num'].named_steps['imputer']
indicator_idx  = imputer_step.indicator_.features_
indicator_feats = [f'{num_cols[i]}_missing' for i in indicator_idx]
indicator_mask = num_feats_mask[len(num_cols):]
kept_indicators = [f for f, k in zip(indicator_feats, indicator_mask) if k]
feat_names = kept_num + kept_indicators + cat_all

# ─── 3. OPTUNA HİPERPARAMETRE OPTİMİZASYONU ──────────────────────────────────
print(f'\n{SEP2}')
print(f'ADIM 3 — Optuna ({OPTUNA_TRIALS} trial)')
print(SEP2)

best_lgb_params = BEST_LGB_FROM_OPTUNA.copy()
best_xgb_params = {
    'n_estimators': 800, 
    'learning_rate': 0.024122636498078342, 
    'max_depth': 8, 
    'subsample': 0.6960781394323559, 
    'colsample_bytree': 0.759442862395421, 
    'min_child_weight': 10, 
    'scale_pos_weight': 0.4775383149015521
}

if RUN_OPTUNA:
    def objective(trial):
        params = {
            'n_estimators':    trial.suggest_int  ('n_estimators',    300, 1200, step=100),
            'learning_rate':   trial.suggest_float('learning_rate',   0.005, 0.1, log=True),
            'max_depth':       trial.suggest_int  ('max_depth',       3, 9),
            'subsample':       trial.suggest_float('subsample',       0.5, 1.0),
            'colsample_bytree':trial.suggest_float('colsample_bytree',0.5, 1.0),
            'min_child_weight':trial.suggest_int  ('min_child_weight', 1, 10),
            'scale_pos_weight':trial.suggest_float('scale_pos_weight', 0.2, 0.6),
            'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
            'device': 'cuda', 'tree_method': 'hist'
        }
        clf = xgb.XGBClassifier(**params)
        cv  = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr, val in cv.split(X_proc, y):
            clf.fit(X_proc[tr], y.iloc[tr])
            prob = clf.predict_proba(X_proc[val])[:, 1]
            thresh, _ = find_best_threshold(y.iloc[val], prob)
            pred = (prob >= thresh).astype(int)
            scores.append(matthews_corrcoef(y.iloc[val], pred))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    print(f'  En iyi Optuna MCC (XGBoost): {study.best_value:.4f}')
    print(f'  En iyi parametreler: {study.best_params}')
    best_xgb_params.update(study.best_params)

# ─── 4. MODEL TANIMI — STACKING ───────────────────────────────────────────────
print(f'\n{SEP2}')
print('ADIM 4 — Stacking Ensemble Tanımı')
print(SEP2)

lgbm_clf = lgb.LGBMClassifier(**best_lgb_params)  # n_jobs best_lgb_params içinde

xgb_base_params = {
    'n_estimators': 800, 'learning_rate': 0.05, 'max_depth': 5,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'scale_pos_weight': scale_pw,
    'eval_metric': 'logloss', 'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
    'device': 'cuda', 'tree_method': 'hist'
}
xgb_base_params.update(best_xgb_params)
xgb_clf  = xgb.XGBClassifier(**xgb_base_params)

base_estimators = [('lgbm', lgbm_clf), ('xgb', xgb_clf)]

if CATBOOST_AVAILABLE:
    cat_clf = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=7,
        scale_pos_weight=scale_pw,
        eval_metric='F1',
        random_seed=42,
        verbose=0,
        task_type='GPU'
    )
    base_estimators.append(('catboost', cat_clf))
    print('  Base modeller: LightGBM + XGBoost + CatBoost')
else:
    print('  Base modeller: LightGBM + XGBoost (CatBoost eksik)')

# Meta-öğrenci: L2 düzenlemeli Lojistik Regresyon
meta_clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)

stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_clf,
    cv=5,
    passthrough=False,   # meta-model sadece base tahminlerini alır
    n_jobs=1  # CatBoost GPU çakışmasını engellemek için
)

# ─── 5. CROSS-VAL DEĞERLENDİRME ───────────────────────────────────────────────
print(f'\n{SEP2}')
print('ADIM 5 — 5-Fold CV Değerlendirme (MASTER)')
print(SEP2)

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_prob_master = cross_val_predict(
    stack, X_proc, y, cv=cv5, method='predict_proba', n_jobs=1  # CatBoost clone uyumu için
)[:, 1]

best_thresh, best_f1 = find_best_threshold(y, y_prob_master)
y_pred_master = (y_prob_master >= best_thresh).astype(int)

master_mcc = matthews_corrcoef(y, y_pred_master)
master_pr  = average_precision_score(y, y_prob_master)
master_roc = roc_auc_score(y, y_prob_master)

print(f'\n  MASTER 5-Fold CV Sonuçları (eşik={best_thresh}):')
print(f'  {"Metrik":<15} {"Değer":>10}')
print(f'  {"-"*27}')
print(f'  {"F1 Skoru":<15} {best_f1:>10.4f}')
print(f'  {"MCC":<15} {master_mcc:>10.4f}')
print(f'  {"PR-AUC":<15} {master_pr:>10.4f}')
print(f'  {"ROC-AUC":<15} {master_roc:>10.4f}')

print('\n  Confusion Matrix:')
cm = confusion_matrix(y, y_pred_master)
print(f'                Tahmin')
print(f'                Benign  Patojenik')
print(f'  Gerçek Benign  {cm[0,0]:5d}  {cm[0,1]:9d}')
print(f'  Gerçek Patojen {cm[1,0]:5d}  {cm[1,1]:9d}')
print(f'\n  FN={cm[1,0]}  FP={cm[0,1]}')

print('\n  Eşik Analizi:')
print(f'  {"Eşik":>6} {"Precision":>10} {"Recall":>8} {"F1":>8} {"MCC":>8}')
for t in np.arange(0.25, 0.66, 0.05):
    yp = (y_prob_master >= t).astype(int)
    from sklearn.metrics import precision_score, recall_score
    pr  = precision_score(y, yp, zero_division=0)
    rec = recall_score(y, yp, zero_division=0)
    f1t = f1_score(y, yp, zero_division=0)
    mct = matthews_corrcoef(y, yp)
    mark = ' <-- optimal' if abs(t - best_thresh) < 0.001 else ''
    print(f'  {t:>6.2f} {pr:>10.4f} {rec:>8.4f} {f1t:>8.4f} {mct:>8.4f}{mark}')

# ─── 6. ALT GRUP DEĞERLENDİRMESİ ─────────────────────────────────────────────
print(f'\n{SEP2}')
print('ADIM 6 — Alt Grup Değerlendirme (KANSER / PAH / CFTR)')
print(SEP2)

# Modeli tüm MASTER üzerinde eğit
stack.fit(X_proc, y)

subgroup_results = {}
for name, path in {k: v for k, v in PATHS.items() if k != 'MASTER'}.items():
    dfs = pd.read_csv(path)
    dfs['Label'] = dfs['Label'].astype(int)
    dfs = extract_aa_features(dfs)
    dfs = add_interaction_features(dfs, ek_cols)
    dfs = add_missing_pattern_features(dfs, al_cols, ek_cols)

    Xs = dfs[num_cols + cat_all]
    ys = dfs['Label']
    Xs_p = preprocessor.transform(Xs)

    prob = stack.predict_proba(Xs_p)[:, 1]

    # PAH için özel eşik (daha düşük eşik → recall artışı → FN azalır)
    if name == 'PAH':
        # PAH'ta özel threshold aramak (FN'i minimize etmek öncelik)
        pah_thresh, pah_f1 = find_best_threshold(ys, prob)
        # Recall odaklı alternatif
        pah_thresh_recall = max(pah_thresh - 0.10, 0.20)
        print(f'\n  [PAH Eşik Analizi] Optimal F1 eşiği: {pah_thresh}')
        for pt in [pah_thresh_recall, pah_thresh, round(pah_thresh + 0.05, 2)]:
            yp = (prob >= pt).astype(int)
            f1t = f1_score(ys, yp, zero_division=0)
            mct = matthews_corrcoef(ys, yp)
            print(f'    Eşik={pt:.2f} -> F1:{f1t:.4f}  MCC:{mct:.4f}')
        use_thresh = pah_thresh
    else:
        use_thresh = best_thresh

    res = evaluate(ys, prob, use_thresh, name)
    subgroup_results[name] = res

print(f'\n  Alt Grup Özet Tablosu (MASTER eşiği={best_thresh}):')
print(f'  {"Grup":<10} {"F1":>8} {"MCC":>8} {"PR-AUC":>8} {"ROC-AUC":>8}')
print(f'  {"-"*44}')
for name, res in subgroup_results.items():
    print(f'  {name:<10} {res["F1"]:>8.4f} {res["MCC"]:>8.4f} '
          f'{res["PR-AUC"]:>8.4f} {res["ROC-AUC"]:>8.4f}')

# ─── 7. V1 / V2 / V3 KARŞILAŞTIRMA ───────────────────────────────────────────
print(f'\n{SEP2}')
print('ADIM 7 — Versiyon Karşılaştırması (MASTER F1 / MCC)')
print(SEP2)

history = [
    ('V1', 'Baseline',                               0.8711, 0.5054, 0.9207),
    ('V2', 'Missing Indicator + Threshold',          0.8903, 0.5301, 0.9206),
    ('V3', 'Biochem AA + Optuna + SHAP Top-100',     0.8929, 0.5401,  None),
    ('V4', 'CatBoost Stacking + EK Interact + Patt.',0.8958, 0.5434, 0.9199),
    ('V5', 'GPU Hızlandırılmış Stacking',            best_f1, master_mcc, master_pr),
]

print(f'  {"Ver":<4} {"Model":<42} {"F1":>8} {"MCC":>8} {"PR-AUC":>8}')
print(f'  {"-"*72}')
for ver, model, f1, mcc, pr in history:
    pr_str = f'{pr:.4f}' if pr else '  —   '
    marker = ' <-- V5' if ver == 'V5' else ''
    print(f'  {ver:<4} {model:<42} {f1:>8.4f} {mcc:>8.4f} {pr_str:>8}{marker}')

v4_f1, v4_mcc = 0.8958, 0.5434
delta_f1  = best_f1   - v4_f1
delta_mcc = master_mcc - v4_mcc
print(f'\n  V5 - V4 Farkı: ΔF1={delta_f1:+.4f}  ΔMCC={delta_mcc:+.4f}')
if delta_f1 >= 0:
    print('  ✅ V5, V4 skorlarını yakaladı/geçti — GPU eğitimi başarılı!')
else:
    print('  ⚠️  V5 henüz V4\'ü geçemedi — parametreleri veya GPU seed\'lerini optimize et.')

print(f'\n{SEP}')
print('  ✅ V5 tamamlandı!')
print(SEP)
