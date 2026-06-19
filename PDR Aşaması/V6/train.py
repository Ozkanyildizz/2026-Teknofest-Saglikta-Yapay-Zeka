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
import shap
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

# ─── TAM GRANTHAM MESAFESİ MATRİSİ ──────────────────────────────────────────
# Değer aralığı: 0 (Aynı) ile 215 (En radikal değişim: C -> W) arasındadır.
_GRANTHAM_DATA = {
    'A': {'A': 0, 'R': 112, 'N': 111, 'D': 126, 'C': 195, 'Q': 91, 'E': 107, 'G': 60, 'H': 86, 'I': 94, 'L': 96, 'K': 106, 'M': 84, 'F': 113, 'P': 27, 'S': 99, 'T': 58, 'W': 148, 'Y': 112, 'V': 64},
    'R': {'A': 112, 'R': 0, 'N': 86, 'D': 96, 'C': 180, 'Q': 43, 'E': 54, 'G': 125, 'H': 29, 'I': 97, 'L': 102, 'K': 26, 'M': 91, 'F': 97, 'P': 103, 'S': 110, 'T': 71, 'W': 101, 'Y': 77, 'V': 96},
    'N': {'A': 111, 'R': 86, 'N': 0, 'D': 23, 'C': 139, 'Q': 46, 'E': 42, 'G': 80, 'H': 68, 'I': 149, 'L': 153, 'K': 94, 'M': 142, 'F': 158, 'P': 91, 'S': 46, 'T': 65, 'W': 174, 'Y': 143, 'V': 133},
    'D': {'A': 126, 'R': 96, 'N': 23, 'D': 0, 'C': 154, 'Q': 61, 'E': 45, 'G': 94, 'H': 81, 'I': 168, 'L': 172, 'K': 101, 'M': 160, 'F': 177, 'P': 108, 'S': 65, 'T': 85, 'W': 181, 'Y': 160, 'V': 152},
    'C': {'A': 195, 'R': 180, 'N': 139, 'D': 154, 'C': 0, 'Q': 154, 'E': 170, 'G': 159, 'H': 174, 'I': 198, 'L': 198, 'K': 202, 'M': 196, 'F': 205, 'P': 169, 'S': 112, 'T': 149, 'W': 215, 'Y': 194, 'V': 192},
    'Q': {'A': 91, 'R': 43, 'N': 46, 'D': 61, 'C': 154, 'Q': 0, 'E': 29, 'G': 87, 'H': 24, 'I': 134, 'L': 130, 'K': 53, 'M': 101, 'F': 116, 'P': 76, 'S': 68, 'T': 42, 'W': 130, 'Y': 99, 'V': 109},
    'E': {'A': 107, 'R': 54, 'N': 42, 'D': 45, 'C': 170, 'Q': 29, 'E': 0, 'G': 98, 'H': 40, 'I': 134, 'L': 138, 'K': 56, 'M': 126, 'F': 140, 'P': 93, 'S': 80, 'T': 65, 'W': 152, 'Y': 122, 'V': 121},
    'G': {'A': 60, 'R': 125, 'N': 80, 'D': 94, 'C': 159, 'Q': 87, 'E': 98, 'G': 0, 'H': 98, 'I': 135, 'L': 138, 'K': 127, 'M': 127, 'F': 153, 'P': 42, 'S': 56, 'T': 59, 'W': 184, 'Y': 147, 'V': 109},
    'H': {'A': 86, 'R': 29, 'N': 68, 'D': 81, 'C': 174, 'Q': 24, 'E': 40, 'G': 98, 'H': 0, 'I': 94, 'L': 99, 'K': 32, 'M': 87, 'F': 100, 'P': 77, 'S': 89, 'T': 47, 'W': 115, 'Y': 83, 'V': 90},
    'I': {'A': 94, 'R': 97, 'N': 149, 'D': 168, 'C': 198, 'Q': 134, 'E': 134, 'G': 135, 'H': 94, 'I': 0, 'L': 5, 'K': 102, 'M': 10, 'F': 21, 'P': 95, 'S': 142, 'T': 89, 'W': 61, 'Y': 33, 'V': 29},
    'L': {'A': 96, 'R': 102, 'N': 153, 'D': 172, 'C': 198, 'Q': 130, 'E': 138, 'G': 138, 'H': 99, 'I': 5, 'L': 0, 'K': 107, 'M': 15, 'F': 22, 'P': 98, 'S': 145, 'T': 92, 'W': 61, 'Y': 36, 'V': 32},
    'K': {'A': 106, 'R': 26, 'N': 94, 'D': 101, 'C': 202, 'Q': 53, 'E': 56, 'G': 127, 'H': 32, 'I': 102, 'L': 107, 'K': 0, 'M': 95, 'F': 102, 'P': 103, 'S': 121, 'T': 78, 'W': 110, 'Y': 85, 'V': 97},
    'M': {'A': 84, 'R': 91, 'N': 142, 'D': 160, 'C': 196, 'Q': 101, 'E': 126, 'G': 127, 'H': 87, 'I': 10, 'L': 15, 'K': 95, 'M': 0, 'F': 28, 'P': 87, 'S': 135, 'T': 81, 'W': 67, 'Y': 36, 'V': 21},
    'F': {'A': 113, 'R': 97, 'N': 158, 'D': 177, 'C': 205, 'Q': 116, 'E': 140, 'G': 153, 'H': 100, 'I': 21, 'L': 22, 'K': 102, 'M': 28, 'F': 0, 'P': 114, 'S': 155, 'T': 103, 'W': 40, 'Y': 22, 'V': 50},
    'P': {'A': 27, 'R': 103, 'N': 91, 'D': 108, 'C': 169, 'Q': 76, 'E': 93, 'G': 42, 'H': 77, 'I': 95, 'L': 98, 'K': 103, 'M': 87, 'F': 114, 'P': 0, 'S': 74, 'T': 38, 'W': 147, 'Y': 110, 'V': 68},
    'S': {'A': 99, 'R': 110, 'N': 46, 'D': 65, 'C': 112, 'Q': 68, 'E': 80, 'G': 56, 'H': 89, 'I': 142, 'L': 145, 'K': 121, 'M': 135, 'F': 155, 'P': 74, 'S': 0, 'T': 42, 'W': 177, 'Y': 144, 'V': 124},
    'T': {'A': 58, 'R': 71, 'N': 65, 'D': 85, 'C': 149, 'Q': 42, 'E': 65, 'G': 59, 'H': 47, 'I': 89, 'L': 92, 'K': 78, 'M': 81, 'F': 103, 'P': 38, 'S': 42, 'T': 0, 'W': 128, 'Y': 92, 'V': 69},
    'W': {'A': 148, 'R': 101, 'N': 174, 'D': 181, 'C': 215, 'Q': 130, 'E': 152, 'G': 184, 'H': 115, 'I': 61, 'L': 61, 'K': 110, 'M': 67, 'F': 40, 'P': 147, 'S': 177, 'T': 128, 'W': 0, 'Y': 37, 'V': 88},
    'Y': {'A': 112, 'R': 77, 'N': 143, 'D': 160, 'C': 194, 'Q': 99, 'E': 122, 'G': 147, 'H': 83, 'I': 33, 'L': 36, 'K': 85, 'M': 36, 'F': 22, 'P': 110, 'S': 144, 'T': 92, 'W': 37, 'Y': 0, 'V': 55},
    'V': {'A': 64, 'R': 96, 'N': 133, 'D': 152, 'C': 192, 'Q': 109, 'E': 121, 'G': 109, 'H': 90, 'I': 29, 'L': 32, 'K': 97, 'M': 21, 'F': 50, 'P': 68, 'S': 124, 'T': 69, 'W': 88, 'Y': 55, 'V': 0}
}

def get_grantham(aa1, aa2):
    """İki amino asit arasındaki Grantham (biyokimyasal radikallik) mesafesini döndürür."""
    if pd.isna(aa1) or pd.isna(aa2) or aa1 not in _GRANTHAM_DATA or aa2 not in _GRANTHAM_DATA:
        return np.nan
    return _GRANTHAM_DATA[aa1][aa2]

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
            
    df['AA_grantham_dist']   = df.apply(lambda row: get_grantham(row['AA_1'], row['AA_2']), axis=1)
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
    
    al_block_cols = [f'AL_{i}' for i in range(27, 39) if f'AL_{i}' in al_cols]
    if al_block_cols:
        df['missing_ratio_AL_block1'] = df[al_block_cols].isnull().mean(axis=1)
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

def build_base_preprocessor(num_cols, cat_all):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)), 
        ('scaler', RobustScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')), 
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    return ColumnTransformer([('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_all)])

def prepare_features(df, al_cols, ek_cols, cat_cols):
    df = extract_aa_features(df)
    df = add_missing_pattern_features(df, al_cols, ek_cols)
    
    # Sızıntıyı önlemek için K-Means ve PCA burada YAPILMIYOR
    # CV döngüsü içinde yapılacak
    
    props = ['polarity', 'charge', 'hydropathy', 'weight', 'volume', 'flexibility', 'aromatic', 'blosum62']
    aa_num_cols = [f'AA1_{p}' for p in props] + [f'AA2_{p}' for p in props] + \
                  ['AA_grantham_dist', 'AA_hydro_diff', 'AA_weight_diff', 'AA_volume_diff', 'AA_flex_diff',
                   'AA_charge_change', 'AA_polarity_change', 'AA_aromatic_change', 'AA_blosum_sum', 'AA_blosum_min']
    
    # PCA ve Cluster özellikleri döngüde ekleneceği için num_cols ve cat_all'a şimdilik eklemiyoruz
    num_cols = al_cols + ek_cols + aa_num_cols + ['missing_count_AL', 'missing_ratio_AL', 'missing_count_EK']
    if 'missing_ratio_AL_block1' in df.columns:
        num_cols.append('missing_ratio_AL_block1')
        
    cat_all = [c for c in cat_cols if c not in ('CAT_4', 'CAT_5')] + ['AA_1', 'AA_2']
    return df[num_cols + cat_all], df['Label'].astype(int), num_cols, cat_all

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

    meta_clf = lgb.LGBMClassifier(max_depth=3, num_leaves=8, random_state=42, verbose=-1)
    return StackingClassifier(estimators=base, final_estimator=meta_clf, cv=5, passthrough=True, n_jobs=1)

def run_leak_free_cv(X, y, num_cols, cat_all, scale_pw, al_cols):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    
    print(f"\n[!] Sizintisiz 5-Fold Custom CV Basliyor...")
    
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
        print(f"  --> Fold {fold}/5 egitiliyor...")
        
        X_tr, y_tr = X.iloc[tr_idx].copy(), y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]
        
        preprocessor = build_base_preprocessor(num_cols, cat_all)
        X_tr_proc = preprocessor.fit_transform(X_tr)
        X_val_proc = preprocessor.transform(X_val)
        
        al_indices = [num_cols.index(col) for col in al_cols if col in num_cols]
        
        pca = PCA(n_components=3, random_state=42)
        kmeans = KMeans(n_clusters=5, random_state=42)
        
        X_tr_al = X_tr_proc[:, al_indices]
        X_val_al = X_val_proc[:, al_indices]
        
        X_tr_pca = pca.fit_transform(X_tr_al)
        X_val_pca = pca.transform(X_val_al)
        
        X_tr_km = kmeans.fit_predict(X_tr_al).reshape(-1, 1)
        X_val_km = kmeans.predict(X_val_al).reshape(-1, 1)
        
        X_tr_final = np.hstack((X_tr_proc, X_tr_pca, X_tr_km))
        X_val_final = np.hstack((X_val_proc, X_val_pca, X_val_km))
        
        probe = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=42)
        if GPU_AVAILABLE: probe.set_params(device_type='gpu')
        probe.fit(X_tr_final, y_tr)
        
        explainer = shap.TreeExplainer(probe)
        shap_vals = explainer.shap_values(X_tr_final)
        mean_shap = np.abs(shap_vals[1] if isinstance(shap_vals, list) else shap_vals).mean(axis=0)
        
        top_idx = np.argsort(mean_shap)[-TOP_N_FEATURES:]
        X_tr_sel = X_tr_final[:, top_idx]
        X_val_sel = X_val_final[:, top_idx]
        
        stack = build_stack(scale_pw)
        stack.fit(X_tr_sel, y_tr)
        
        oof_preds[val_idx] = stack.predict_proba(X_val_sel)[:, 1]
    
    return oof_preds


print(SEP)
print('  AlgoMed V6 — Ozkan: Derin AA Bio. + K-Means Cluster + PCA (GPU Stacking)')
print(SEP)
print(f'  GPU: {"Aktif" if GPU_AVAILABLE else "CPU modu"}')

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

# ─── YÜRÜTME (Custom CV & Optimizasyon) ───────────────────────────────────────
print(f'\n{SEP2}')
print('ADIM 2 — Sızıntısız Özel Cross-Validation')
print(SEP2)

oof_probabilities = run_leak_free_cv(X, y, num_cols, cat_all, scale_pw, al_cols)

print(f'\n{SEP2}')
print('ADIM 3 — Sonuçlar ve Eşik Analizi')
print(SEP2)

best_f1, best_thresh = 0, 0.5
for thresh in np.arange(0.20, 0.70, 0.01):
    yp = (oof_probabilities >= thresh).astype(int)
    f1t = f1_score(y, yp, zero_division=0)
    if f1t > best_f1:
        best_f1, best_thresh = f1t, thresh

final_preds = (oof_probabilities >= best_thresh).astype(int)

master_res = {
    'f1': best_f1,
    'mcc': matthews_corrcoef(y, final_preds),
    'pr': average_precision_score(y, oof_probabilities),
    'roc': roc_auc_score(y, oof_probabilities),
    'best_thresh': best_thresh,
    'y_pred': final_preds,
    'y_prob': oof_probabilities
}

print(f'\n  MASTER CV Sonuclari:')
print(f'  {"Metrik":<20} {"Deger":>10}')
print(f'  {"-"*32}')
print(f'  {"F1":<20} {master_res["f1"]:>10.4f}')
print(f'  {"MCC":<20} {master_res["mcc"]:>10.4f}')
print(f'  {"PR-AUC":<20} {master_res["pr"]:>10.4f}')
print(f'  {"ROC-AUC":<20} {master_res["roc"]:>10.4f}')
print(f'  {"Optimal esik":<20} {master_res["best_thresh"]:>10.2f}')

cm = confusion_matrix(y, master_res['y_pred'])
print('\n  Confusion Matrix (F1 esik):')
print(f'                Tahmin')
print(f'                Benign  Patojenik')
print(f'  Gercek Benign  {cm[0,0]:5d}  {cm[0,1]:9d}')
print(f'  Gercek Patojen {cm[1,0]:5d}  {cm[1,1]:9d}')
print(f'\n  FN={cm[1,0]}  FP={cm[0,1]}')

print('\n  Esik Analizi:')
print(f'  {"Esik":>6} {"Precision":>10} {"Recall":>8} {"F1":>8} {"MCC":>8}')
for t in np.arange(0.25, 0.66, 0.05):
    yp = (master_res['y_prob'] >= t).astype(int)
    pr  = precision_score(y, yp, zero_division=0)
    rec = recall_score(y, yp, zero_division=0)
    f1t = f1_score(y, yp, zero_division=0)
    mct = matthews_corrcoef(y, yp)
    mark = ' <-- opt' if abs(t - master_res['best_thresh']) < 0.001 else ''
    print(f'  {t:>6.2f} {pr:>10.4f} {rec:>8.4f} {f1t:>8.4f} {mct:>8.4f}{mark}')

# ─── ALT GRUP AYRI CV (ATLANDI) ──────────────────────────────────────────────
print(f'\n{SEP2}')
print('ADIM 4 — Alt Grup Ayrı CV (KANSER / PAH / CFTR)')
print(SEP2)
print('  Alt grup CV atlandi (Sızıntısız model test ediliyor).')

# ─── VERSİYON KARŞILAŞTIRMA ───────────────────────────────────────────────
print(f'\n{SEP2}')
print('ADIM 5 — Versiyon Karsilastirmasi')
print(SEP2)

history = [
    ('V4', 'CatBoost Stacking + EK Interact',     0.8958, 0.5434, 0.9199),
    ('V5', 'GPU Stacking',                        0.8941, 0.5370, 0.9222),
    ('V6(L)', 'Sizintisiz CV + Derin AA',         master_res['f1'], master_res['mcc'], master_res['pr']),
]

print(f'  {"Ver":<6} {"Model":<35} {"F1":>8} {"MCC":>8} {"PR-AUC":>8}')
print(f'  {"-"*65}')
for ver, model, f1, mcc, pr in history:
    marker = ' <-- V6 Leak-Free' if ver == 'V6(L)' else ''
    print(f'  {ver:<6} {model:<35} {f1:>8.4f} {mcc:>8.4f} {pr:>8.4f}{marker}')

v4_f1, v4_mcc = 0.8958, 0.5434
delta_f1  = master_res['f1']  - v4_f1
delta_mcc = master_res['mcc'] - v4_mcc
print(f'\n  V6(L) - V4 Farki: dF1={delta_f1:+.4f}  dMCC={delta_mcc:+.4f}')
if delta_f1 > 0 and delta_mcc > 0:
    print('  [OK] Sizintisiz V6, V4\'u hem F1 hem MCC\'de gecti!')
elif delta_f1 > 0 or delta_mcc > 0:
    print('  [~] Sizintisiz V6 kismi iyilesme.')
else:
    print('  [!] Sizintisiz V6 henuz V4\'u gecemedi.')

print(f'\n{SEP}')
print('  V6 tamamlandi!')
print(SEP)
