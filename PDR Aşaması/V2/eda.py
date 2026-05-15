"""
========================================================
TEKNOFEST 2026 – Sağlıkta Yapay Zeka Yarışması
V1 — EDA (Keşifsel Veri Analizi)
Çalıştırma: PDR Aşaması/V1/ klasöründen  →  python eda.py
========================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── Yollar ───────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), '..', 'universite-veri-seti', 'EĞİTİM (TRAIN) SETLERİ')

PATHS = {
    'MASTER': os.path.join(BASE, 'YARISMA_TRAIN_MASTER.csv'),
    'KANSER': os.path.join(BASE, 'YARISMA_TRAIN_KANSER.csv'),
    'PAH':    os.path.join(BASE, 'YARISMA_TRAIN_PAH.csv'),
    'CFTR':   os.path.join(BASE, 'YARISMA_TRAIN_CFTR.csv'),
}

SEP = '=' * 60

# ─── 1. GENEL ÖZET ────────────────────────────────────
print(SEP)
print('1. TÜM VERİ SETLERİ — GENEL ÖZET')
print(SEP)

for name, path in PATHS.items():
    df = pd.read_csv(path)
    total = len(df)
    pat   = (df['Label'] == 1).sum()
    ben   = (df['Label'] == 0).sum()
    miss  = df.isnull().sum().sum()
    miss_pct = miss / (df.shape[0] * df.shape[1]) * 100
    print(f'\n  {name}:')
    print(f'    Satır      : {total}')
    print(f'    Patojenik  : {pat} ({pat/total*100:.1f}%)')
    print(f'    Benign     : {ben} ({ben/total*100:.1f}%)')
    print(f'    Eksik hücre: {miss:,} ({miss_pct:.1f}%)')

# ─── 2. MASTER — DETAYLI ANALİZ ───────────────────────
print(f'\n{SEP}')
print('2. MASTER VERİ SETİ — DETAYLI ANALİZ')
print(SEP)

df = pd.read_csv(PATHS['MASTER'])

al_cols  = [c for c in df.columns if c.startswith('AL_')]
cat_cols = [c for c in df.columns if c.startswith('CAT_')]
ek_cols  = [c for c in df.columns if c.startswith('EK_')]
aa_cols  = [c for c in df.columns if c.startswith('AA_')]
num_cols = al_cols + ek_cols

print(f'\n  Sütun grupları:')
print(f'    AL_ (sayısal) : {len(al_cols)}')
print(f'    EK_ (sayısal) : {len(ek_cols)}')
print(f'    CAT_ (kateg.) : {len(cat_cols)}')
print(f'    AA_  (amino)  : {len(aa_cols)}')

# ─── Sayısal sütunlarda eksik analizi ─────────────────
miss_by_col = df[num_cols].isnull().sum()
miss_pct    = miss_by_col / len(df) * 100

print(f'\n  Sayısal sütunlarda eksik değer dağılımı:')
print(f'    %0   eksik sütun sayısı : {(miss_pct == 0).sum()}')
print(f'    %0-50 eksik            : {((miss_pct > 0) & (miss_pct <= 50)).sum()}')
print(f'    %50+ eksik             : {(miss_pct > 50).sum()}')

# En çok eksik olan 10 sütun
top_miss = miss_pct.sort_values(ascending=False).head(10)
print(f'\n  En çok eksik olan 10 sayısal sütun:')
for col, pct in top_miss.items():
    print(f'    {col}: {pct:.1f}%')

# Hiç eksik olmayan sütun sayısı
zero_miss_cols = miss_pct[miss_pct == 0].index.tolist()
print(f'\n  Hiç eksik değer olmayan sayısal sütun: {len(zero_miss_cols)}')

# ─── Kategorik sütunlar ───────────────────────────────
print(f'\n  CAT_ sütunları:')
for c in cat_cols:
    n_unique = df[c].nunique()
    miss_n   = df[c].isnull().sum()
    print(f'    {c}: {n_unique} unique, {miss_n} eksik -> {df[c].dropna().value_counts().head(3).to_dict()}')

print(f'\n  AA_ sütunları:')
for c in aa_cols:
    print(f'    {c}: {df[c].nunique()} unique amino asit, {df[c].isnull().sum()} eksik')
    print(f'       Dağılım: {df[c].value_counts().head(5).to_dict()}')

# ─── 3. KORELASYON ANALİZİ (Sadece tam dolu sütunlar) ─
print(f'\n{SEP}')
print('3. ETİKET-ÖZELLİK KORELASYONU (Sıfır eksikli sütunlar)')
print(SEP)

complete_num = [c for c in num_cols if df[c].isnull().sum() == 0]
if complete_num:
    corrs = df[complete_num + ['Label']].corr()['Label'].drop('Label').abs()
    top10 = corrs.sort_values(ascending=False).head(10)
    print('\n  Etiketle en yüksek korelasyonlu 10 özellik:')
    for col, val in top10.items():
        print(f'    {col}: {val:.4f}')
else:
    print('  (Tüm sayısal sütunlarda eksik değer var)')

# ─── 4. AYKIRI DEĞER ──────────────────────────────────
print(f'\n{SEP}')
print('4. AYKIRI DEĞER ANALİZİ (EK_ sütunları — tam dolu)')
print(SEP)

complete_ek = [c for c in ek_cols if df[c].isnull().sum() == 0]
print(f'  Eksik olmayan EK_ sütunları: {len(complete_ek)}/{len(ek_cols)}')
if complete_ek:
    for c in complete_ek[:3]:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr     = q3 - q1
        out_n   = ((df[c] < q1 - 1.5*iqr) | (df[c] > q3 + 1.5*iqr)).sum()
        print(f'    {c}: min={df[c].min():.3f}, max={df[c].max():.3f}, '
              f'IQR aykırı={out_n} ({out_n/len(df)*100:.1f}%)')

print(f'\n{SEP}')
print('EDA TAMAMLANDI')
print(SEP)
