"""
V4 EDA — AlgoMed TEKNOFEST 2026
Odak: EK_ sütun analizi, etkileşim keşfi, eksiklik örüntüsü
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy import stats

# ─── YOLLAR ───────────────────────────────────────────────────────────────────
BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'universite-veri-seti', 'EĞİTİM (TRAIN) SETLERİ'
)
MASTER_PATH = os.path.join(BASE, 'YARISMA_TRAIN_MASTER.csv')

SEP = '=' * 60

print(SEP)
print('  V4 EDA — EK Analizi, Etkileşim, Eksiklik Örüntüsü')
print(SEP)

df = pd.read_csv(MASTER_PATH)
df['Label'] = df['Label'].astype(int)

al_cols  = [c for c in df.columns if c.startswith('AL_')]
ek_cols  = [c for c in df.columns if c.startswith('EK_')]
cat_cols = [c for c in df.columns if c.startswith('CAT_')]

# ─── 1. EK_ SÜTUN ANALİZİ ─────────────────────────────────────────────────────
print(f'\n{"─"*50}')
print('1. EK_ Sütunları — Patojenik vs Benign Karşılaştırması')
print(f'{"─"*50}')

print(f'\n  {"Sütun":<10} {"Benign_ort":>12} {"Patojen_ort":>12} '
      f'{"Fark":>10} {"Mann-Whitney p":>15}')
print(f'  {"-"*62}')

ek_stats = []
for col in ek_cols:
    ben = df[df['Label'] == 0][col].dropna()
    pat = df[df['Label'] == 1][col].dropna()
    if len(ben) < 2 or len(pat) < 2:
        continue
    stat, p = stats.mannwhitneyu(ben, pat, alternative='two-sided')
    diff = pat.mean() - ben.mean()
    ek_stats.append({'col': col, 'benign_mean': ben.mean(), 'patojen_mean': pat.mean(),
                     'diff': diff, 'p_value': p})
    sig = ' ***' if p < 0.001 else (' **' if p < 0.01 else (' *' if p < 0.05 else ''))
    print(f'  {col:<10} {ben.mean():>12.4f} {pat.mean():>12.4f} '
          f'{diff:>10.4f} {p:>15.6f}{sig}')

# ─── 2. EK_ ETKİLEŞİM TERİMLERİ KORELASYONu ──────────────────────────────────
print(f'\n{"─"*50}')
print('2. EK_ × EK_ Etkileşim Terimlerinin Hedefle Korelasyonu (Top 10)')
print(f'{"─"*50}')

interact_corrs = []
for a, b in combinations(ek_cols, 2):
    inter = df[a] * df[b]
    corr  = inter.corr(df['Label'])
    if not np.isnan(corr):
        interact_corrs.append({'term': f'{a}×{b}', 'corr': corr, 'abs_corr': abs(corr)})

interact_df = pd.DataFrame(interact_corrs).sort_values('abs_corr', ascending=False)
print(f'\n  {"Etkileşim Terimi":<20} {"Korelasyon":>12}')
print(f'  {"-"*35}')
for _, row in interact_df.head(10).iterrows():
    print(f'  {row["term"]:<20} {row["corr"]:>12.4f}')

# ─── 3. EKSİKLİK ÖRÜNTÜSÜ ANALİZİ ───────────────────────────────────────────
print(f'\n{"─"*50}')
print('3. Satır Bazında Eksiklik Örüntüsü — Label ile İlişki')
print(f'{"─"*50}')

df['missing_ratio_AL']  = df[al_cols].isnull().mean(axis=1)
df['missing_count_EK']  = df[ek_cols].isnull().sum(axis=1)
df['missing_count_ALL'] = df[al_cols + ek_cols].isnull().sum(axis=1)

for feat in ['missing_ratio_AL', 'missing_count_EK', 'missing_count_ALL']:
    corr = df[feat].corr(df['Label'])
    ben_mean = df[df['Label'] == 0][feat].mean()
    pat_mean = df[df['Label'] == 1][feat].mean()
    print(f'\n  {feat}:')
    print(f'    Benign ortalama   : {ben_mean:.4f}')
    print(f'    Patojenik ortalama: {pat_mean:.4f}')
    print(f'    Label korelasyonu : {corr:.4f}')

# ─── 4. CAT_ KATEGORİK SÜTUN ANALİZİ ─────────────────────────────────────────
print(f'\n{"─"*50}')
print('4. Kategorik Sütunlar (CAT_) — Patojenite Oranları')
print(f'{"─"*50}')

for col in cat_cols:
    if df[col].isna().mean() > 0.5:
        print(f'\n  {col}: %{df[col].isna().mean()*100:.1f} eksik — atlandı')
        continue
    top_vals = df[col].value_counts().head(5).index
    print(f'\n  {col} (top 5 değer):')
    print(f'    {"Değer":<30} {"N":>6} {"Patojenik%":>12}')
    for val in top_vals:
        mask  = df[col] == val
        n     = mask.sum()
        pat_r = df.loc[mask, 'Label'].mean()
        print(f'    {str(val):<30} {n:>6} {pat_r*100:>11.1f}%')

# ─── 5. GÖRSEL: EK_7 DAĞILIMI ─────────────────────────────────────────────────
try:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('V4 EDA — EK_ Önemli Sütunlar', fontsize=13, fontweight='bold')

    # EK_7 dağılımı
    ax = axes[0]
    for label, color, name in [(0, '#2196F3', 'Benign'), (1, '#F44336', 'Patojenik')]:
        data = df[df['Label'] == label]['EK_7'].dropna()
        ax.hist(data, bins=40, alpha=0.6, color=color, label=name, density=True)
    ax.set_title('EK_7 Dağılımı')
    ax.set_xlabel('EK_7')
    ax.legend()

    # Eksiklik oranı vs label
    ax = axes[1]
    df.boxplot(column='missing_ratio_AL', by='Label', ax=ax,
               boxprops=dict(color='steelblue'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_title('AL_ Eksiklik Oranı (Satır Bazı)')
    ax.set_xlabel('Label (0=Benign, 1=Patojenik)')
    plt.sca(ax)
    plt.title('AL_ Eksiklik Oranı (Satır Bazı)')

    # EK_ etkileşim korelasyonları (top 8)
    ax = axes[2]
    top8 = interact_df.head(8)
    colors = ['#F44336' if c > 0 else '#2196F3' for c in top8['corr']]
    ax.barh(top8['term'], top8['abs_corr'], color=colors)
    ax.set_title('EK×EK Etkileşim |Korelasyon| (Top 8)')
    ax.set_xlabel('|Korelasyon|')
    ax.invert_yaxis()

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v4_eda.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\n  Görsel kaydedildi: {out_path}')
    plt.close()
except Exception as e:
    print(f'\n  Görsel oluşturulamadı: {e}')

# ─── 6. ÖZET ──────────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print('  EDA Özeti — V4 İçin Öneriler')
print(SEP)

# EK_ en anlamlı
top_ek = sorted(ek_stats, key=lambda x: abs(x['diff']), reverse=True)
print(f'\n  En ayırt edici EK_ sütunu: {top_ek[0]["col"]} '
      f'(Δort={top_ek[0]["diff"]:+.4f})')

# Etkileşim
print(f'  En güçlü EK×EK etkileşimi: {interact_df.iloc[0]["term"]} '
      f'(corr={interact_df.iloc[0]["corr"]:+.4f})')

# Eksiklik örüntüsü
miss_corr = df['missing_ratio_AL'].corr(df['Label'])
print(f'  AL_ eksiklik oranı korelasyonu: {miss_corr:+.4f}')

print('\n✅ V4 EDA tamamlandı!')
