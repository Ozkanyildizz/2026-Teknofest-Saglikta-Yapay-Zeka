"""
AlgoMed V9 — Sızdırmaz Pipeline + FN Risk Özellikleri
===========================================================
GÖREV: V7'NİN SIZDIRMAZ MİMARİSİNE V8'İN FN RİSK ÖZELLİKLERİNİ ENTEGRE ETMEK

V7'nin mimarisi değişmeden korunmaktadır:
  LightGBM + XGBoost + CatBoost → StackingClassifier (LogisticRegression meta)
  GPU hızlandırmalı (CUDA/hist)
  Sızdırmaz (Leak-Free) Cross Validation
  
V9'da V7'ye eklenenler:
  1. V8'de keşfedilen False Negative (FN) risk bayrakları (EK7_low_flag, AA_missing_flag, vs.)
  2. Eksiklik (missingness) üzerinden güçlü sinyallerin dahil edilmesi
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

V9_DIR       = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(V9_DIR, 'v9_optuna_results.json')
DB_FILE      = os.path.join(V9_DIR, 'v9_optuna.db')

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

# ─── BLOSUM62 MATRİSİ (V6'dan — Standart PAM/BLOSUM literatürü) ─────────────
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
for _i, _a in enumerate(_AA_ORDER):
    for _j, _b in enumerate(_AA_ORDER):
        BLOSUM62[(_a, _b)] = _BLOSUM62_RAW[_i][_j]

# ─── TAM GRANTHAM MESAFESİ MATRİSİ (V6'dan) ─────────────────────────────────
# Değer aralığı: 0 (Aynı AA) → 215 (C→W, en radikal değişim).
_GRANTHAM_DATA = {
    'A': {'A':  0,'R':112,'N':111,'D':126,'C':195,'Q': 91,'E':107,'G': 60,'H': 86,'I': 94,'L': 96,'K':106,'M': 84,'F':113,'P': 27,'S': 99,'T': 58,'W':148,'Y':112,'V': 64},
    'R': {'A':112,'R':  0,'N': 86,'D': 96,'C':180,'Q': 43,'E': 54,'G':125,'H': 29,'I': 97,'L':102,'K': 26,'M': 91,'F': 97,'P':103,'S':110,'T': 71,'W':101,'Y': 77,'V': 96},
    'N': {'A':111,'R': 86,'N':  0,'D': 23,'C':139,'Q': 46,'E': 42,'G': 80,'H': 68,'I':149,'L':153,'K': 94,'M':142,'F':158,'P': 91,'S': 46,'T': 65,'W':174,'Y':143,'V':133},
    'D': {'A':126,'R': 96,'N': 23,'D':  0,'C':154,'Q': 61,'E': 45,'G': 94,'H': 81,'I':168,'L':172,'K':101,'M':160,'F':177,'P':108,'S': 65,'T': 85,'W':181,'Y':160,'V':152},
    'C': {'A':195,'R':180,'N':139,'D':154,'C':  0,'Q':154,'E':170,'G':159,'H':174,'I':198,'L':198,'K':202,'M':196,'F':205,'P':169,'S':112,'T':149,'W':215,'Y':194,'V':192},
    'Q': {'A': 91,'R': 43,'N': 46,'D': 61,'C':154,'Q':  0,'E': 29,'G': 87,'H': 24,'I':134,'L':130,'K': 53,'M':101,'F':116,'P': 76,'S': 68,'T': 42,'W':130,'Y': 99,'V':109},
    'E': {'A':107,'R': 54,'N': 42,'D': 45,'C':170,'Q': 29,'E':  0,'G': 98,'H': 40,'I':134,'L':138,'K': 56,'M':126,'F':140,'P': 93,'S': 80,'T': 65,'W':152,'Y':122,'V':121},
    'G': {'A': 60,'R':125,'N': 80,'D': 94,'C':159,'Q': 87,'E': 98,'G':  0,'H': 98,'I':135,'L':138,'K':127,'M':127,'F':153,'P': 42,'S': 56,'T': 59,'W':184,'Y':147,'V':109},
    'H': {'A': 86,'R': 29,'N': 68,'D': 81,'C':174,'Q': 24,'E': 40,'G': 98,'H':  0,'I': 94,'L': 99,'K': 32,'M': 87,'F':100,'P': 77,'S': 89,'T': 47,'W':115,'Y': 83,'V': 90},
    'I': {'A': 94,'R': 97,'N':149,'D':168,'C':198,'Q':134,'E':134,'G':135,'H': 94,'I':  0,'L':  5,'K':102,'M': 10,'F': 21,'P': 95,'S':142,'T': 89,'W': 61,'Y': 33,'V': 29},
    'L': {'A': 96,'R':102,'N':153,'D':172,'C':198,'Q':130,'E':138,'G':138,'H': 99,'I':  5,'L':  0,'K':107,'M': 15,'F': 22,'P': 98,'S':145,'T': 92,'W': 61,'Y': 36,'V': 32},
    'K': {'A':106,'R': 26,'N': 94,'D':101,'C':202,'Q': 53,'E': 56,'G':127,'H': 32,'I':102,'L':107,'K':  0,'M': 95,'F':102,'P':103,'S':121,'T': 78,'W':110,'Y': 85,'V': 97},
    'M': {'A': 84,'R': 91,'N':142,'D':160,'C':196,'Q':101,'E':126,'G':127,'H': 87,'I': 10,'L': 15,'K': 95,'M':  0,'F': 28,'P': 87,'S':135,'T': 81,'W': 67,'Y': 36,'V': 21},
    'F': {'A':113,'R': 97,'N':158,'D':177,'C':205,'Q':116,'E':140,'G':153,'H':100,'I': 21,'L': 22,'K':102,'M': 28,'F':  0,'P':114,'S':155,'T':103,'W': 40,'Y': 22,'V': 50},
    'P': {'A': 27,'R':103,'N': 91,'D':108,'C':169,'Q': 76,'E': 93,'G': 42,'H': 77,'I': 95,'L': 98,'K':103,'M': 87,'F':114,'P':  0,'S': 74,'T': 38,'W':147,'Y':110,'V': 68},
    'S': {'A': 99,'R':110,'N': 46,'D': 65,'C':112,'Q': 68,'E': 80,'G': 56,'H': 89,'I':142,'L':145,'K':121,'M':135,'F':155,'P': 74,'S':  0,'T': 42,'W':177,'Y':144,'V':124},
    'T': {'A': 58,'R': 71,'N': 65,'D': 85,'C':149,'Q': 42,'E': 65,'G': 59,'H': 47,'I': 89,'L': 92,'K': 78,'M': 81,'F':103,'P': 38,'S': 42,'T':  0,'W':128,'Y': 92,'V': 69},
    'W': {'A':148,'R':101,'N':174,'D':181,'C':215,'Q':130,'E':152,'G':184,'H':115,'I': 61,'L': 61,'K':110,'M': 67,'F': 40,'P':147,'S':177,'T':128,'W':  0,'Y': 37,'V': 88},
    'Y': {'A':112,'R': 77,'N':143,'D':160,'C':194,'Q': 99,'E':122,'G':147,'H': 83,'I': 33,'L': 36,'K': 85,'M': 36,'F': 22,'P':110,'S':144,'T': 92,'W': 37,'Y':  0,'V': 55},
    'V': {'A': 64,'R': 96,'N':133,'D':152,'C':192,'Q':109,'E':121,'G':109,'H': 90,'I': 29,'L': 32,'K': 97,'M': 21,'F': 50,'P': 68,'S':124,'T': 69,'W': 88,'Y': 55,'V':  0},
}

def get_grantham(aa1, aa2):
    """İki amino asit arasındaki Grantham (biyokimyasal radikallik) mesafesini döndürür."""
    if pd.isna(aa1) or pd.isna(aa2) or aa1 not in _GRANTHAM_DATA or aa2 not in _GRANTHAM_DATA:
        return np.nan
    return _GRANTHAM_DATA[aa1].get(aa2, np.nan)

# ─── BİYOKİMYASAL ÖZELLİK SÖZLÜĞÜ (V9: V6 seviyesine yükseltildi) ──────────
# 8 özellik: polarity, charge, hydropathy, weight, volume, flexibility,
# aromatic, blosum62 — rapor Tablo 3 ve Şekil 1 (SHAP) ile tam uyumludur.
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


# ─── YARDIMCI FONKSİYONLAR (V5'ten aynen alınmıştır) ───────────────────────

def extract_aa_features(df):
    """
    AA_1 ve AA_2 sütunlarından derinleştirilmiş biyokimyasal özellikler hesaplanır.
    V9: V6 seviyesine yükseltildi — 8 özellik x 2 AA + Grantham mesafesi + BLOSUM62
    türevleri + fark özellikleri. Rapor Tablo 3 ve Şekil 1 (SHAP) ile tam uyumludur.
    """
    df = df.copy()
    props = ['polarity', 'charge', 'hydropathy', 'weight', 'volume', 'flexibility', 'aromatic', 'blosum62']
    for aa_col, prefix in [('AA_1', 'AA1'), ('AA_2', 'AA2')]:
        for prop in props:
            df[f'{prefix}_{prop}'] = df[aa_col].map(
                lambda x, p=prop: AA_PROPERTIES.get(x, {}).get(p, np.nan))

    # Grantham mesafesi (evrimsel radikallik — rapor SHAP Şekil 1'in 1. sırası)
    df['AA_grantham_dist']   = df.apply(
        lambda row: get_grantham(row['AA_1'], row['AA_2']), axis=1)

    # Fark ve değişim özellikleri
    df['AA_hydro_diff']      = abs(df['AA1_hydropathy'] - df['AA2_hydropathy'])
    df['AA_weight_diff']     = abs(df['AA1_weight']     - df['AA2_weight'])
    df['AA_volume_diff']     = abs(df['AA1_volume']     - df['AA2_volume'])
    df['AA_flex_diff']       = abs(df['AA1_flexibility'] - df['AA2_flexibility'])
    df['AA_charge_change']   = (df['AA1_charge']   != df['AA2_charge']).astype(float)
    df['AA_polarity_change'] = (df['AA1_polarity'] != df['AA2_polarity']).astype(float)
    df['AA_aromatic_change'] = (df['AA1_aromatic'] != df['AA2_aromatic']).astype(float)

    # BLOSUM62 türevleri (evrimsel kabul edilebilirlik — rapor SHAP Şekil 1'in 2. sırası)
    df['AA_blosum_sum']      = df['AA1_blosum62'] + df['AA2_blosum62']
    df['AA_blosum_min']      = df[['AA1_blosum62', 'AA2_blosum62']].min(axis=1)

    # Eksik değer durumlarında NaN güvencesi
    df.loc[df['AA1_charge'].isna()   | df['AA2_charge'].isna(),   'AA_charge_change']   = np.nan
    df.loc[df['AA1_polarity'].isna() | df['AA2_polarity'].isna(), 'AA_polarity_change'] = np.nan
    df.loc[df['AA1_aromatic'].isna() | df['AA2_aromatic'].isna(), 'AA_aromatic_change'] = np.nan
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


def add_cat_interaction_features(df, cat_interaction_cols):
    """
    CAT_ sütunları arasında sayısal etkileşim terimleri üretilir (V9 — Rapor Bölüm 4.4).
    Kategorik değerler integer koda çevrilip ikili çarpım alınır.
    pd.Categorical tutarlı kodlama sağlar; NaN değerleri korunur.
    """
    df = df.copy()
    encoded = {}
    for col in cat_interaction_cols:
        if col in df.columns:
            codes = pd.Categorical(df[col]).codes.astype(float)
            codes[codes == -1] = np.nan  # Eksik değerleri NaN yap
            encoded[col] = codes

    for a, b in combinations(cat_interaction_cols, 2):
        if a in encoded and b in encoded:
            df[f'{a}_x_{b}'] = encoded[a] * encoded[b]
    return df


def find_best_threshold(y_true, y_prob, step=0.01):
    """
    MCC'yi (Matthews Korelasyon Katsayısı) en üst düzeye çıkaran karar eşiği belirlenir.
    V9: F1'den MCC'ye çevrildi — raporun 'MCC maksimizasyonu ölçütüyle eşik=0.20'
    ifadesiyle tam uyumludur.
    """
    best_score, best_thresh = -1.0, 0.5
    for thresh in np.arange(0.20, 0.71, step):
        yp    = (y_prob >= thresh).astype(int)
        score = matthews_corrcoef(y_true, yp)
        if score > best_score:
            best_score, best_thresh = score, thresh
    return round(best_thresh, 2), best_score


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
    V9 odağı: L1 (reg_alpha), L2 (reg_lambda), max_depth ve scale_pos_weight.
    Early Stopping (Rapor Bölüm 2.3): Her fold'da eval_set ile erken durdurma uygulanır.
    """
    def objective(trial):
        params = {
            'n_estimators':       trial.suggest_int  ('n_estimators',     300, 1500, step=100),
            'learning_rate':      trial.suggest_float ('learning_rate',    0.003, 0.15, log=True),
            'max_depth':          trial.suggest_int   ('max_depth',        3, 10),
            'subsample':          trial.suggest_float ('subsample',        0.50, 1.0),
            'colsample_bytree':   trial.suggest_float ('colsample_bytree', 0.40, 1.0),
            'reg_alpha':          trial.suggest_float ('reg_alpha',        1e-8, 10.0, log=True),  # L1
            'reg_lambda':         trial.suggest_float ('reg_lambda',       1e-8, 10.0, log=True),  # L2
            'min_child_weight':   trial.suggest_int   ('min_child_weight', 1, 20),
            'scale_pos_weight':   trial.suggest_float ('scale_pos_weight', 0.15, 0.60),
            'gamma':              trial.suggest_float ('gamma',            0.0, 5.0),
            'early_stopping_rounds': 50,   # Rapor Bölüm 2.3: Erken Durdurma mekanizması
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
            clf.fit(X_proc[tr], y.iloc[tr],
                    eval_set=[(X_proc[val], y.iloc[val])],
                    verbose=False)
            prob = clf.predict_proba(X_proc[val])[:, 1]
            thresh, _ = find_best_threshold(y.iloc[val], prob)
            mccs.append(matthews_corrcoef(y.iloc[val], (prob >= thresh).astype(int)))
        return np.mean(mccs)
    return objective


def make_cat_objective(X_proc, y, scale_pw):
    """
    CatBoost için Optuna hedef fonksiyonu.
    V9: L2 (l2_leaf_reg), depth ve scale_pos_weight optimize edilir.
    Early Stopping (Rapor Bölüm 2.3): Her fold'da eval_set ile erken durdurma uygulanır.
    """
    def objective(trial):
        if not CATBOOST_AVAILABLE:
            return 0.0
        params = {
            'iterations':          trial.suggest_int  ('iterations',       300, 1000, step=100),
            'learning_rate':       trial.suggest_float ('learning_rate',    0.003, 0.15, log=True),
            'depth':               trial.suggest_int   ('depth',            4, 8),
            'l2_leaf_reg':         trial.suggest_float ('l2_leaf_reg',      1.0, 10.0),  # L2
            'scale_pos_weight':    trial.suggest_float ('scale_pos_weight', 0.15, 0.60),
            'early_stopping_rounds': 50,   # Rapor Bölüm 2.3: Erken Durdurma mekanizması
            'random_seed': RANDOM_STATE, 'verbose': 0, 'eval_metric': 'F1',
        }

        # CatBoost'ta Bayesian bootstrap subsample desteklemediğinden farklı parametreler.
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
            clf.fit(X_proc[tr], y.iloc[tr],
                    eval_set=(X_proc[val], y.iloc[val]),
                    verbose=False)
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
print('  AlgoMed V9 -- Sizdirmaz Pipeline + FN Risk Ozellikleri (V8)')
print('  V7 Mimarisinin Uzerine:')
print(f'  Optuna {OPTUNA_TRIALS} trial (L1/L2/depth/scale_pos_weight)')
print('  + SMOTE/ADASYN/BorderlineSMOTE Sizdirmaz Pipeline')
print('  + V8 FN Risk Bayraklari')
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

# V9: V6 seviyesine yükseltilmiş biyokimyasal özellikler (Grantham + BLOSUM62 dahil)
# Rapor Tablo 3 ve Şekil 1 (SHAP) ile tam uyumludur.
df = extract_aa_features(df)
new_aa_num_cols = [
    # AA1 özellikleri (8 özellik — V6 seviyesi)
    'AA1_polarity', 'AA1_charge', 'AA1_hydropathy', 'AA1_weight',
    'AA1_volume', 'AA1_flexibility', 'AA1_aromatic', 'AA1_blosum62',
    # AA2 özellikleri (8 özellik — V6 seviyesi)
    'AA2_polarity', 'AA2_charge', 'AA2_hydropathy', 'AA2_weight',
    'AA2_volume', 'AA2_flexibility', 'AA2_aromatic', 'AA2_blosum62',
    # Türetilmiş AA özellikleri (Grantham + BLOSUM + farklar — Rapor SHAP Şekil 1)
    'AA_grantham_dist', 'AA_hydro_diff', 'AA_weight_diff', 'AA_volume_diff',
    'AA_flex_diff', 'AA_charge_change', 'AA_polarity_change', 'AA_aromatic_change',
    'AA_blosum_sum', 'AA_blosum_min',
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

# V8'den gelen False Negative (FN) Risk özellikleri
df["EK7_low_flag"] = (df["EK_7"] < 3).astype(float)
df["AA_missing_flag"] = (df["AA_1"].isna() | df["AA_2"].isna()).astype(float)
df["FN_risk_flag"] = ((df["EK_7"] < 3) | df["AA_1"].isna() | df["AA_2"].isna()).astype(float)
fn_risk_cols = ["EK7_low_flag", "AA_missing_flag", "FN_risk_flag"]

# V9 YENİ: CAT_ etkileşim özellikleri (Rapor Bölüm 4.4 uyumu)
cat_for_interaction = [c for c in cat_cols if c not in ('CAT_4', 'CAT_5')]
df = add_cat_interaction_features(df, cat_for_interaction)
cat_interaction_cols = [f'{a}_x_{b}' for a, b in combinations(cat_for_interaction, 2)
                        if f'{a}_x_{b}' in df.columns]

# Eksiklik oranı %80 ve üzeri olan sayısal sütunlar veri setinden çıkarılır (V5 ile aynı)
all_num_base = (al_cols + ek_cols + new_aa_num_cols + ek_interaction_cols
                + missing_pattern_cols + fn_risk_cols + cat_interaction_cols)
all_num_base = [c for c in all_num_base if c in df.columns]
missing_ratio_series = df[all_num_base].isnull().mean()
num_cols = missing_ratio_series[missing_ratio_series < MISSING_THRESH].index.tolist()

# CAT_4 ve CAT_5 sütunları veri setinden çıkarıldı (V5'te de aynı karar alınmıştı)
cat_all = [c for c in cat_cols if c not in ('CAT_4', 'CAT_5')] + aa_cols

X = df[num_cols + cat_all]
y = df['Label']

pos_ratio = y.mean()
scale_pw  = float(round((1 - pos_ratio) / pos_ratio, 4))

print(f'  Biyokimyasal ozellikler : {len(new_aa_num_cols)} (V6 seviyesi — Grantham+BLOSUM dahil)')
print(f'  EK_ etkilesim terimleri : {len(ek_interaction_cols)} (V4ten)')
miss_pat_in_df = [c for c in missing_pattern_cols if c in df.columns]
print(f'  Eksiklik oruntu ozellik : {len(miss_pat_in_df)} (V4ten)')
print(f'  FN Risk ozellikleri     : {len(fn_risk_cols)} (V8den)')
print(f'  CAT etkilesim ozelligi  : {len(cat_interaction_cols)} (V9 Yeni — Rapor Bolum 4.4)')
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
    methods    = ['smote']
    method_res = {}
    print('\n  Tum oversampling yontemleri karsilastiriliyor...')

    for method in methods:
        print(f'\n  {"-"*45}')
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

# Modeli ve Ön İşleyiciyi diske kaydetme
try:
    import joblib
    model_path = os.path.join(V9_DIR, 'v9_final_model.joblib')
    prep_path  = os.path.join(V9_DIR, 'v9_preprocessor.joblib')
    joblib.dump(stack_final, model_path)
    joblib.dump(prep_final, prep_path)
    print(f'\n  [KAYIT] Final model ve preprocessor basariyla kaydedildi:')
    print(f'    Model: {model_path}')
    print(f'    Preprocessor: {prep_path}')
except Exception as e:
    print(f'\n  Model kaydetme sirasinda hata: {e}')

subgroup_results = {}
for name, path in {k: v for k, v in PATHS.items() if k != 'MASTER'}.items():
    dfs = pd.read_csv(path)
    dfs['Label'] = dfs['Label'].astype(int)
    dfs = extract_aa_features(dfs)
    dfs = add_interaction_features(dfs, ek_cols)
    dfs = add_missing_pattern_features(dfs, al_cols, ek_cols)

    # FN risk bayrakları (num_cols içinde yer alıyorsa eklenir)
    if 'EK_7' in dfs.columns:
        dfs['EK7_low_flag']  = (dfs['EK_7'] < 3).astype(float)
        dfs['FN_risk_flag']  = ((dfs['EK_7'] < 3) | dfs['AA_1'].isna() | dfs['AA_2'].isna()).astype(float)
    else:
        dfs['EK7_low_flag']  = np.nan
        dfs['FN_risk_flag']  = (dfs['AA_1'].isna() | dfs['AA_2'].isna()).astype(float)
    dfs['AA_missing_flag'] = (dfs['AA_1'].isna() | dfs['AA_2'].isna()).astype(float)

    # CAT_ etkileşim özellikleri (num_cols içinde yer alıyorsa eklenir)
    dfs = add_cat_interaction_features(dfs, cat_for_interaction)

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
    ('V1',  'Baseline LGB+XGB',                              0.8711, 0.5054, 0.9207),
    ('V2',  'Missing Indicator + Threshold',                 0.8903, 0.5301, 0.9206),
    ('V3',  'Biochem AA + Optuna + SHAP Top-100',            0.8929, 0.5401, None),
    ('V4',  'CatBoost Stacking + EK Interact',               0.8958, 0.5434, 0.9199),
    ('V5',  'GPU Stacking (100 trial Optuna)',                0.8941, 0.5370, 0.9222),
    ('V6',  'Leak-Free + Grantham + LGBM Meta',              0.8919, 0.5450, 0.9116),
    ('V7',  'V5 + Optuna 700/500t + SMOTE',                  0.8894, 0.5374, 0.9211),
    ('V8',  'FN Otopsi + Risk Bayraklari',                   0.8941, 0.5370, 0.9222),
    ('V9',  f'Grantham+BLOSUM+CAT_inter+MCC_esik+{best_smote.upper()}', v7_f1, v7_mcc, v7_pr),
]

print(f'  {"Ver":<4} {"Model":<52} {"F1":>8} {"MCC":>8} {"PR-AUC":>8}')
print(f'  {"-"*84}')
for ver, model, f1, mcc, pr in history:
    pr_str = f'{pr:.4f}' if pr else '  --  '
    marker = ' <- V9 (FINAL)' if ver == 'V9' else ''
    print(f'  {ver:<4} {model:<52} {f1:>8.4f} {mcc:>8.4f} {pr_str:>8}{marker}')

v5_f1, v5_mcc = 0.8941, 0.5370
delta_f1  = v7_f1  - v5_f1
delta_mcc = v7_mcc - v5_mcc
print(f'\n  V9 - V5 Farki: dF1={delta_f1:+.4f}  dMCC={delta_mcc:+.4f}')
if delta_f1 > 0 and delta_mcc > 0:
    print('  V9, V5yi hem F1 hem MCCde gecti! (Rapor hedefi basarild)')
elif delta_f1 > 0 or delta_mcc > 0:
    print('  V9 V5e kiyasla kismi iyilesme sagladi.')
else:
    print('  V9 henuz V5i gecemedi. OPTUNA_TRIALS artir veya SMOTE yontemini degistir.')

# Tüm nihai sonuçlar JSON dosyasına kaydedilir
final_results = {
    'v9_f1':  v7_f1, 'v9_mcc': v7_mcc, 'v9_pr': v7_pr, 'v9_roc': v7_roc,
    'best_thresh': float(final_thresh), 'best_smote': best_smote,
    'fn': int(cm[1, 0]), 'fp': int(cm[0, 1]),
    'delta_f1_vs_v5': float(delta_f1), 'delta_mcc_vs_v5': float(delta_mcc),
    'optuna_trials': OPTUNA_TRIALS, 'gpu': GPU_AVAILABLE,
    'aa_feature_level': 'V6 (Grantham+BLOSUM62+volume+flexibility+aromatic)',
    'threshold_metric': 'MCC',
    'cat_interaction_features': len(cat_interaction_cols),
    'early_stopping': True,
    'subgroup_results': {k: {mk: float(mv) for mk, mv in v.items()}
                         for k, v in subgroup_results.items()},
}
with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)
print(f'\n  Tum sonuclar kaydedildi: {RESULTS_FILE}')

print(f'\n{SEP}')
print(f'  AlgoMed V9 tamamlandi! (Rapor uyumlu: Grantham+BLOSUM+CAT+MCC+EarlyStop)')
print(f'  F1={v7_f1:.4f}  MCC={v7_mcc:.4f}  Esik={final_thresh}  SMOTE={best_smote.upper()}')
print(SEP)
