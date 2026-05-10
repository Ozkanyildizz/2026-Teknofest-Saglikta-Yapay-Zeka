"""
Veri kalite analizi — missense_dataset.csv
"""
import pandas as pd
import numpy as np

df = pd.read_csv("missense_dataset.csv")

print("=" * 60)
print("VERİ BOYUTU:", df.shape)
print("=" * 60)

# Sütun tipleri
print("\n--- SÜTUN TİPLERİ ---")
print(df.dtypes)

# Etiket → sayısal yap
df['label'] = df['label'].astype(int)

print("\n--- ETİKET DAĞILIMI ---")
print(df['label'].value_counts())
print(f"Sınıf oranı (Patojenik/Benign): {df['label'].mean():.3f}")

# Özellik sütunları
feature_cols = [c for c in df.columns if c not in ['chrom','pos','ref','alt','label']]
print(f"\n--- ÖZELLİKLER ({len(feature_cols)} adet) ---")
print(feature_cols)

X = df[feature_cols]

print("\n--- EKSİK DEĞERLER ---")
missing = X.isnull().sum()
missing_pct = (missing / len(X) * 100).round(2)
miss_df = pd.DataFrame({'Eksik Sayı': missing, 'Eksik %': missing_pct})
print(miss_df[miss_df['Eksik Sayı'] > 0])

print("\n--- İSTATİSTİKSEL ÖZET ---")
print(X.describe().round(3))

print("\n--- AYKIRI DEĞER ANALİZİ (IQR) ---")
for col in feature_cols:
    q1 = X[col].quantile(0.25)
    q3 = X[col].quantile(0.75)
    iqr = q3 - q1
    outliers = ((X[col] < q1 - 3*iqr) | (X[col] > q3 + 3*iqr)).sum()
    if outliers > 0:
        print(f"  {col}: {outliers} aykırı değer ({outliers/len(X)*100:.1f}%)")

print("\n--- ÖZELLİK-ETİKET KORELASYONU ---")
corr = X.corrwith(df['label']).round(3)
print(corr.sort_values(ascending=False))

print("\n--- YINELENMİŞ SATIR ---")
dupes = df.duplicated().sum()
print(f"Yinelenen satır sayısı: {dupes}")

print("\n✅ Analiz tamamlandı.")
