
import pandas as pd
import os

base = r'c:\Users\ozkan\OneDrive\Masaüstü\algomed\PDR Aşaması\universite-veri-seti\EĞİTİM (TRAIN) SETLERİ'

files = {
    'MASTER': os.path.join(base, 'YARISMA_TRAIN_MASTER.csv'),
    'KANSER': os.path.join(base, 'YARISMA_TRAIN_KANSER.csv'),
    'PAH':    os.path.join(base, 'YARISMA_TRAIN_PAH.csv'),
    'CFTR':   os.path.join(base, 'YARISMA_TRAIN_CFTR.csv'),
}

for name, path in files.items():
    df = pd.read_csv(path)
    print(f'=== {name} ===')
    print(f'Boyut: {df.shape}')
    label_dist = df['Label'].value_counts().to_dict()
    print(f'Label (0=Benign, 1=Pathogenic): {label_dist}')
    total_missing = df.isnull().sum().sum()
    print(f'Eksik deger toplam: {total_missing}')
    al_cols  = [c for c in df.columns if c.startswith('AL_')]
    cat_cols = [c for c in df.columns if c.startswith('CAT_')]
    ek_cols  = [c for c in df.columns if c.startswith('EK_')]
    aa_cols  = [c for c in df.columns if c.startswith('AA_')]
    print(f'AL_ (sayisal): {len(al_cols)}  |  CAT_ (kategorik): {len(cat_cols)}  |  EK_: {len(ek_cols)}  |  AA_: {len(aa_cols)}')
    if name == 'MASTER':
        for c in cat_cols:
            print(f'  {c} uniques: {sorted(df[c].dropna().unique()[:10].tolist())}')
        for c in ek_cols:
            print(f'  {c} uniques: {sorted(df[c].dropna().unique()[:10].tolist())}')
        for c in aa_cols:
            print(f'  {c} uniques: {df[c].dropna().unique()[:5].tolist()}')
        print(f'  Variant_ID ornekler: {df["Variant_ID"].head(3).tolist()}')
    print()
