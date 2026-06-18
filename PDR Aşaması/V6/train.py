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
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import StackingClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, precision_score, recall_score
)
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("[UYARI] CatBoost kurulu degil. pip install catboost")

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
TOP_N_FEATURES  = 100
MISSING_THRESH  = 0.80
USE_GPU         = True
RUN_SUBGROUP_CV = False   # True yaparsan alt gruplar da calısır (~3x uzun)

def log(msg):
    print(msg, flush=True)

# V4/V5'ten kanıtlanmış LightGBM parametreleri (FN regresyonunu önlemek için)
BEST_LGB_PARAMS = {
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
    'n_jobs': -1,
}

BEST_XGB_PARAMS = {
    'n_estimators': 800,
    'learning_rate': 0.024122636498078342,
    'max_depth': 8,
    'subsample': 0.6960781394323559,
    'colsample_bytree': 0.759442862395421,
    'min_child_weight': 10,
    'scale_pos_weight': 0.4775383149015521,
    'eval_metric': 'logloss',
    'random_state': 42,
    'verbosity': 0,
    'n_jobs': -1,
}

# ─── BLOSUM62 ─────────────────────────────────────────────────────────────────
_AA_ORDER = 'ARNDCQEGHILKMFPSTWYV'
_BLOSUM62_RAW = [
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
]
BLOSUM62 = {}
for i, a in enumerate(_AA_ORDER):
    for j, b in enumerate(_AA_ORDER):
        BLOSUM62[(a, b)] = _BLOSUM62_RAW[i][j]

# ─── BİYOKİMYASAL ÖZELLİK SÖZLÜĞÜ (V6: Derinleştirildi) ────────────────────────
# V6: polarity, charge, hydropathy, weight, volume, flexibility, aromatic, blosum62
AA_PROPERTIES = {
    'A': {'polarity': 0, 'charge':  0, 'hydropathy':  1.8, 'weight':  89.1, 'volume':  88.6, 'flexibility': 0.360, 'aromatic': 0, 'blosum62':  4},
    'R': {'polarity': 1, 'charge':  1, 'hydropathy': -4.5, 'weight': 174.2, 'volume': 173.4, 'flexibility': 0.530, 'aromatic': 0, 'blosum62':  5},
    'N': {'polarity': 1, 'charge':  0, 'hydropathy': -3.5, 'weight': 132.1, 'volume': 114.1, 'flexibility': 0.460, 'aromatic': 0, 'blosum62':  6},
    'D': {'polarity': 1, 'charge': -1, 'hydropathy': -3.5, 'weight': 133.1, 'volume': 111.1, 'flexibility': 0.510, 'aromatic': 0, 'blosum62':  6},
    'C': {'polarity': 0, 'charge':  0, 'hydropathy':  2.5, 'weight': 121.2, 'volume': 108.5, 'flexibility': 0.350, 'aromatic': 0, 'blosum62':  9},
    'E': {'polarity': 1, 'charge': -1, 'hydropathy': -3.5, 'weight': 147.1, 'volume': 138.4, 'flexibility': 0.500, 'aromatic': 0, 'blosum62':  5},
    'Q': {'polarity': 1, 'charge':  0, 'hydropathy': -3.5, 'weight': 146.2, 'volume': 143.8, 'flexibility': 0.490, 'aromatic': 0, 'blosum62':  5},
    'G': {'polarity': 0, 'charge':  0, 'hydropathy': -0.4, 'weight':  75.1, 'volume':  60.1, 'flexibility': 0.540, 'aromatic': 0, 'blosum62':  6},
    'H': {'polarity': 1, 'charge':  1, 'hydropathy': -3.2, 'weight': 155.2, 'volume': 153.2, 'flexibility': 0.320, 'aromatic': 1, 'blosum62':  8},
    'I': {'polarity': 0, 'charge':  0, 'hydropathy':  4.5, 'weight': 131.2, 'volume': 166.7, 'flexibility': 0.300, 'aromatic': 0, 'blosum62':  4},
    'L': {'polarity': 0, 'charge':  0, 'hydropathy':  3.8, 'weight': 131.2, 'volume': 166.7, 'flexibility': 0.400, 'aromatic': 0, 'blosum62':  4},
    'K': {'polarity': 1, 'charge':  1, 'hydropathy': -3.9, 'weight': 146.2, 'volume': 168.6, 'flexibility': 0.470, 'aromatic': 0, 'blosum62':  5},
    'M': {'polarity': 0, 'charge':  0, 'hydropathy':  1.9, 'weight': 149.2, 'volume': 162.9, 'flexibility': 0.300, 'aromatic': 0, 'blosum62':  5},
    'F': {'polarity': 0, 'charge':  0, 'hydropathy':  2.8, 'weight': 165.2, 'volume': 189.9, 'flexibility': 0.310, 'aromatic': 1, 'blosum62':  6},
    'P': {'polarity': 0, 'charge':  0, 'hydropathy': -1.6, 'weight': 115.1, 'volume': 112.7, 'flexibility': 0.510, 'aromatic': 0, 'blosum62':  7},
    'S': {'polarity': 1, 'charge':  0, 'hydropathy': -0.8, 'weight': 105.1, 'volume':  89.0, 'flexibility': 0.510, 'aromatic': 0, 'blosum62':  4},
    'T': {'polarity': 1, 'charge':  0, 'hydropathy': -0.7, 'weight': 119.1, 'volume': 116.1, 'flexibility': 0.440, 'aromatic': 0, 'blosum62':  5},
    'W': {'polarity': 0, 'charge':  0, 'hydropathy': -0.9, 'weight': 204.2, 'volume': 227.8, 'flexibility': 0.310, 'aromatic': 1, 'blosum62': 11},
    'Y': {'polarity': 1, 'charge':  0, 'hydropathy': -1.3, 'weight': 181.2, 'volume': 193.6, 'flexibility': 0.420, 'aromatic': 1, 'blosum62':  7},
    'V': {'polarity': 0, 'charge':  0, 'hydropathy':  4.2, 'weight': 117.1, 'volume': 140.0, 'flexibility': 0.390, 'aromatic': 0, 'blosum62':  4},
}

def detect_gpu():
    if not USE_GPU: return False
    try:
        import subprocess
        subprocess.run(['nvidia-smi'], capture_output=True, check=True)
        return True
    except Exception: return False

GPU_AVAILABLE = detect_gpu()

def extract_aa_features(df):
    """AA_1/AA_2 → derinleştirilmiş biyokimyasal özellikler (V6)."""
    df = df.copy()
    props = ['polarity', 'charge', 'hydropathy', 'weight', 'volume', 'flexibility', 'aromatic', 'blosum62']
    for aa_col, prefix in [('AA_1', 'AA1'), ('AA_2', 'AA2')]:
        for prop in props:
            df[f'{prefix}_{prop}'] = df[aa_col].map(lambda x: AA_PROPERTIES.get(x, {}).get(prop, np.nan))
    df['AA_hydro_diff']      = abs(df['AA1_hydropathy'] - df['AA2_hydropathy'])
    df['AA_weight_diff']     = abs(df['AA1_weight'] - df['AA2_weight'])
    df['AA_volume_diff']     = abs(df['AA1_volume'] - df['AA2_volume'])
    df['AA_flex_diff']       = abs(df['AA1_flexibility'] - df['AA2_flexibility'])
    df['AA_charge_change']   = (df['AA1_charge'] != df['AA2_charge']).astype(float)
    df['AA_polarity_change'] = (df['AA1_polarity'] != df['AA2_polarity']).astype(float)
    df['AA_aromatic_change'] = (df['AA1_aromatic'] != df['AA2_aromatic']).astype(float)
    df['AA_blosum_sum']      = df['AA1_blosum62'] + df['AA2_blosum62']
    df['AA_blosum_min']      = df[['AA1_blosum62','AA2_blosum62']].min(axis=1)
    return df

def add_clustering_features(df, cols, n_clusters=5):
    imputer = SimpleImputer(strategy='median')
    data = imputer.fit_transform(df[cols])
    df['cluster'] = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(data)
    return df

def add_pca_features(df, cols, n_components=3):
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='median')
    data = imputer.fit_transform(df[cols])
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data_scaled)
    for i in range(n_components):
        df[f'pca_{i}'] = pca_data[:, i]
    return df

def add_interaction_features(df, ek_cols):
    df = df.copy()
    for a, b in combinations(ek_cols, 2):
        df[f'{a}_x_{b}'] = df[a] * df[b]
    return df

def add_missing_pattern_features(df, al_cols, ek_cols):
    df = df.copy()
    df['missing_count_AL'] = df[al_cols].isnull().sum(axis=1)
    df['missing_ratio_AL'] = df[al_cols].isnull().mean(axis=1)
    df['missing_count_EK'] = df[ek_cols].isnull().sum(axis=1)
    return df

def find_best_threshold(y_true, y_prob, metric='mcc', step=0.01):
    best_score, best_thresh = -1.0, 0.5
    for thresh in np.arange(0.20, 0.71, step):
        yp = (y_prob >= thresh).astype(int)
        score = matthews_corrcoef(y_true, yp) if metric == 'mcc' else f1_score(y_true, yp, zero_division=0)
        if score > best_score:
            best_score, best_thresh = score, thresh
    return round(best_thresh, 2), best_score

def run_cv_evaluation(stack, X_proc, y, threshold_metric='mcc', label=''):
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob = cross_val_predict(stack, X_proc, y, cv=cv5, method='predict_proba', n_jobs=1)[:, 1]
    best_thresh, _ = find_best_threshold(y, y_prob, metric=threshold_metric)
    y_pred = (y_prob >= best_thresh).astype(int)
    return {'y_prob': y_prob, 'y_pred': y_pred, 'best_thresh': best_thresh,
            'f1': f1_score(y, y_pred, zero_division=0), 'mcc': matthews_corrcoef(y, y_pred),
            'pr': average_precision_score(y, y_prob), 'roc': roc_auc_score(y, y_prob)}

def build_preprocessor(num_cols, cat_all):
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)), ('scaler', RobustScaler())])
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')), ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
    return ColumnTransformer([('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_all)])

def prepare_features(df, al_cols, ek_cols, cat_cols):
    df = extract_aa_features(df)
    df = add_missing_pattern_features(df, al_cols, ek_cols)
    df = add_clustering_features(df, al_cols, n_clusters=5)
    df = add_pca_features(df, al_cols, n_components=3)
    
    props = ['polarity', 'charge', 'hydropathy', 'weight', 'volume', 'flexibility', 'aromatic', 'blosum62']
    aa_num_cols = [f'AA1_{p}' for p in props] + [f'AA2_{p}' for p in props] + \
                  ['AA_hydro_diff', 'AA_weight_diff', 'AA_volume_diff', 'AA_flex_diff',
                   'AA_charge_change', 'AA_polarity_change', 'AA_aromatic_change', 'AA_blosum_sum', 'AA_blosum_min']
    
    num_cols = al_cols + ek_cols + aa_num_cols + ['pca_0', 'pca_1', 'pca_2']
    cat_all = [c for c in cat_cols if c not in ('CAT_4', 'CAT_5')] + ['AA_1', 'AA_2', 'cluster']
    return df[num_cols + cat_all], df['Label'].astype(int), num_cols, cat_all

def select_top_features_shap(X_proc, y, feat_names, top_n=TOP_N_FEATURES):
    lgb_params = {'n_estimators': 200, 'learning_rate': 0.05, 'verbose': -1}
    if GPU_AVAILABLE: lgb_params['device'] = 'gpu'
    probe = lgb.LGBMClassifier(**lgb_params).fit(X_proc, y)
    try:
        import shap
        sv = shap.TreeExplainer(probe).shap_values(X_proc[:500])
        mean_shap = np.abs(sv[1] if isinstance(sv, list) else sv).mean(axis=0)
        top_idx = np.argsort(mean_shap)[-top_n:]
        return top_idx
    except: return np.arange(X_proc.shape[1])

def build_stack(scale_pw):
    lgb_params = BEST_LGB_PARAMS.copy()
    if GPU_AVAILABLE: lgb_params['device_type'] = 'gpu'
    lgb_clf = lgb.LGBMClassifier(**lgb_params)

    xgb_params = BEST_XGB_PARAMS.copy()
    xgb_params['scale_pos_weight'] = scale_pw
    if GPU_AVAILABLE:
        xgb_params['device'] = 'cuda'
        xgb_params['tree_method'] = 'hist'
    xgb_clf = xgb.XGBClassifier(**xgb_params)

    base = [('lgbm', lgb_clf), ('xgb', xgb_clf)]
    if CATBOOST_AVAILABLE:
        cat_params = dict(iterations=800, learning_rate=0.05, depth=7,
                          scale_pos_weight=scale_pw, eval_metric='F1',
                          random_seed=42, verbose=0)
        if GPU_AVAILABLE: cat_params['task_type'] = 'GPU'
        base.append(('catboost', CatBoostClassifier(**cat_params)))

    meta_clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    return StackingClassifier(estimators=base, final_estimator=meta_clf, cv=5, passthrough=False, n_jobs=1)


def get_feature_names(preprocessor, num_cols, cat_all):
    imputer_step = preprocessor.named_transformers_['num'].named_steps['imputer']
    indicator_idx = imputer_step.indicator_.features_
    indicator_feats = [f'{num_cols[i]}_missing' for i in indicator_idx]
    return num_cols + indicator_feats + cat_all


print(SEP)
print('  AlgoMed V6 — Ozkan: Derin AA Bio. + K-Means Cluster + PCA (GPU Stacking)')
print(SEP)
print(f'  GPU: {"Aktif" if GPU_AVAILABLE else "CPU modu"}')

print(f'\n{SEP2}')
print('ADIM 1 — Veri Yükleme ve Özellik Çıkarımı')
print(SEP2)

df_master = pd.read_csv(PATHS['MASTER'])
df_master['Label'] = df_master['Label'].astype(int)
al_cols  = [c for c in df_master.columns if c.startswith('AL_')]
ek_cols  = [c for c in df_master.columns if c.startswith('EK_')]
cat_cols = [c for c in df_master.columns if c.startswith('CAT_') and c != 'CAT_6']

X, y, num_cols, cat_all = prepare_features(df_master, al_cols, ek_cols, cat_cols)

pos_ratio = y.mean()
scale_pw  = float(round((1 - pos_ratio) / pos_ratio, 4))
print(f'  MASTER boyutu           : {df_master.shape}')
print(f'  Sayısal sütun           : {len(num_cols)}')
print(f'  Kategorik sütun         : {len(cat_all)}')
print(f'  Patojenik oranı         : {pos_ratio:.2%}')

# ─── 2. ÖN İŞLEME ─────────────────────────────────────────────────────────────
print(f'\n{SEP2}')
print('ADIM 2 — Ön İşleme Pipeline')
print(SEP2)

preprocessor = build_preprocessor(num_cols, cat_all)
X_proc = preprocessor.fit_transform(X)
feat_names = get_feature_names(preprocessor, num_cols, cat_all)
print(f'  İşlem sonrası özellik   : {X_proc.shape[1]}')

# ─── 3. SHAP TOP-N SEÇİMİ ────────────────────────────────────────────────────
print(f'\n{SEP2}')
print(f'ADIM 3 — SHAP Top-{TOP_N_FEATURES} Özellik Seçimi')
print(SEP2)

top_idx = select_top_features_shap(X_proc, y, feat_names, TOP_N_FEATURES)
X_proc_sel = X_proc[:, top_idx]
sel_feat_names = [feat_names[i] for i in top_idx]
print(f'  Seçilen özellik sayısı  : {X_proc_sel.shape[1]}')

# ─── 4. STACKING MODEL ────────────────────────────────────────────────────────
print(f'\n{SEP2}')
print('ADIM 4 — Stacking Ensemble (passthrough=True, LGBM meta)')
print(SEP2)

stack = build_stack(scale_pw)
n_base = 3 if CATBOOST_AVAILABLE else 2
print(f'  Base modeller           : LightGBM + XGBoost' +
      (' + CatBoost' if CATBOOST_AVAILABLE else ''))
print(f'  Meta-learner            : LGBM (max_depth=3, passthrough=True)')

# ─── 5. MASTER CV ───────────────────────────────────────────────────────────
print(f'\n{SEP2}')
print('ADIM 5 — 5-Fold CV (MASTER)')
print(SEP2)

master_res = run_cv_evaluation(stack, X_proc_sel, y, threshold_metric='f1', label='MASTER')

print(f'\n  MASTER CV Sonuçları:')
print(f'  {"Metrik":<20} {"Değer":>10}')
print(f'  {"-"*32}')
print(f'  {"F1":<20} {master_res["f1"]:>10.4f}')
print(f'  {"MCC":<20} {master_res["mcc"]:>10.4f}')
print(f'  {"PR-AUC":<20} {master_res["pr"]:>10.4f}')
print(f'  {"ROC-AUC":<20} {master_res["roc"]:>10.4f}')
print(f'  {"Optimal esik":<20} {master_res["best_thresh"]:>10.2f}')

cm = confusion_matrix(y, master_res['y_pred'])
print('\n  Confusion Matrix (MCC eşik):')
print(f'                Tahmin')
print(f'                Benign  Patojenik')
print(f'  Gerçek Benign  {cm[0,0]:5d}  {cm[0,1]:9d}')
print(f'  Gerçek Patojen {cm[1,0]:5d}  {cm[1,1]:9d}')
print(f'\n  FN={cm[1,0]}  FP={cm[0,1]}')

print('\n  Eşik Analizi:')
print(f'  {"Eşik":>6} {"Precision":>10} {"Recall":>8} {"F1":>8} {"MCC":>8}')
for t in np.arange(0.25, 0.66, 0.05):
    yp = (master_res['y_prob'] >= t).astype(int)
    pr  = precision_score(y, yp, zero_division=0)
    rec = recall_score(y, yp, zero_division=0)
    f1t = f1_score(y, yp, zero_division=0)
    mct = matthews_corrcoef(y, yp)
    mark = ' <-- opt' if abs(t - master_res['best_thresh']) < 0.001 else ''
    print(f'  {t:>6.2f} {pr:>10.4f} {rec:>8.4f} {f1t:>8.4f} {mct:>8.4f}{mark}')

# ─── 6. ALT GRUP AYRI CV ─────────────────────────────────────────────────────
print(f'\n{SEP2}')
print('ADIM 6 — Alt Grup Ayrı CV (KANSER / PAH / CFTR)')
print(SEP2)

subgroup_results = {}
if not RUN_SUBGROUP_CV:
    log('  Alt grup CV atlandi (RUN_SUBGROUP_CV=False).')

for name, path in ({k: v for k, v in PATHS.items() if k != 'MASTER'}.items()
                   if RUN_SUBGROUP_CV else []):
    df_sub = pd.read_csv(path)
    Xs, ys, _, _ = prepare_features(df_sub, al_cols, ek_cols, cat_cols)
    Xs_proc = preprocessor.transform(Xs)
    Xs_sel  = Xs_proc[:, top_idx]

    sub_stack = build_stack(float(round((1 - ys.mean()) / ys.mean(), 4)))
    sub_res   = run_cv_evaluation(sub_stack, Xs_sel, ys, threshold_metric='mcc')

    print(f'\n  [{name}] CV — F1:{sub_res["f1"]:.4f}  MCC:{sub_res["mcc"]:.4f}  '
          f'PR-AUC:{sub_res["pr"]:.4f}  Eşik:{sub_res["best_thresh"]}  '
          f'FN={int(((ys==1) & (sub_res["y_pred"]==0)).sum())}')
    subgroup_results[name] = sub_res

print(f'\n  Alt Grup Özet Tablosu:')
print(f'  {"Grup":<10} {"F1":>8} {"MCC":>8} {"PR-AUC":>8} {"Eşik":>8} {"FN":>6}')
print(f'  {"-"*50}')
for name, res in subgroup_results.items():
    df_sub = pd.read_csv(PATHS[name])
    ys = df_sub['Label'].astype(int)
    fn = int(((ys == 1) & (res['y_pred'] == 0)).sum())
    print(f'  {name:<10} {res["F1"]:>8.4f} {res["MCC"]:>8.4f} '
          f'{res["PR-AUC"]:>8.4f} {res["best_thresh"]:>8.2f} {fn:>6}')

# ─── 7. VERSİYON KARŞILAŞTIRMA ───────────────────────────────────────────────
print(f'\n{SEP2}')
print('ADIM 7 — Versiyon Karsilastirmasi')
print(SEP2)

history = [
    ('V4', 'CatBoost Stacking + EK Interact',     0.8958, 0.5434, 0.9199),
    ('V5', 'GPU Stacking',                        0.8941, 0.5370, 0.9222),
    ('V6', 'BLOSUM+SHAP+MetaLGBM+MCC',            master_res['f1'], master_res['mcc'], master_res['pr']),
]

print(f'  {"Ver":<4} {"Model":<35} {"F1":>8} {"MCC":>8} {"PR-AUC":>8}')
print(f'  {"-"*65}')
for ver, model, f1, mcc, pr in history:
    marker = ' <-- V6' if ver == 'V6' else ''
    print(f'  {ver:<4} {model:<35} {f1:>8.4f} {mcc:>8.4f} {pr:>8.4f}{marker}')

v4_f1, v4_mcc = 0.8958, 0.5434
delta_f1  = master_res['f1']  - v4_f1
delta_mcc = master_res['mcc'] - v4_mcc
print(f'\n  V6 - V4 Farki: dF1={delta_f1:+.4f}  dMCC={delta_mcc:+.4f}')
if delta_f1 > 0 and delta_mcc > 0:
    print('  [OK] V6, V4\'u hem F1 hem MCC\'de gecti -- push edilebilir!')
elif delta_f1 > 0 or delta_mcc > 0:
    print('  [~] V6 kismi iyilesme -- takimla degerlendir.')
else:
    print('  [!] V6 henuz V4\'u gecemedi.')

print(f'\n{SEP}')
print('  V6 tamamlandi!')
print(SEP)
