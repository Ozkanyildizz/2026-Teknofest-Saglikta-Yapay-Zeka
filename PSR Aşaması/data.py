"""
Sağlıkta YZ Yarışması - PSR için veri hazırlama scripti
Kaynak: HuggingFace - songlab/clinvar  (ClinVar missense varyantlar)
"""

import pandas as pd
from datasets import load_dataset

print("=== VERİ İNDİRİLİYOR ===")
print("Kaynak: HuggingFace - songlab/clinvar")
print("Lütfen bekleyin...\n")

# HuggingFace'den indir
dataset = load_dataset("songlab/clinvar", split="test")
df = dataset.to_pandas()

print(f"Toplam varyant sayısı: {len(df)}")
print(f"\nSütunlar:\n{list(df.columns)}\n")
print(f"İlk 3 satır:\n{df.head(3)}\n")

# Etiket dağılımını göster
if 'label' in df.columns:
    print(f"Etiket dağılımı:\n{df['label'].value_counts()}\n")
    # 0 = Benign, 1 = Pathogenic (genel kabul)
    pathogenic = df[df['label'] == 1]
    benign     = df[df['label'] == 0]
    print(f"Patojenik varyant sayısı : {len(pathogenic)}")
    print(f"Benign varyant sayısı    : {len(benign)}")

# Yarışma boyutuna benzer dengeli alt küme oluştur (1500 + 1500)
n = min(1500, len(pathogenic), len(benign))
df_balanced = pd.concat([
    pathogenic.sample(n=n, random_state=42),
    benign.sample(n=n, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n=== DENGELİ VERİ SETİ ===")
print(f"Toplam: {len(df_balanced)} varyant ({n} Patojenik + {n} Benign)")

# CSV olarak kaydet
out_path = "missense_dataset.csv"
df_balanced.to_csv(out_path, index=False)
print(f"\n✅ Veri '{out_path}' olarak kaydedildi.")

# Eksik değer analizi
print(f"\n=== EKSİK DEĞER ANALİZİ ===")
missing = df_balanced.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "Eksik değer yok ✅")

print("\n=== TAMAMLANDI ===")