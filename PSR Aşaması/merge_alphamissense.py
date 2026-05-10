"""
AlphaMissense Skorlarını ClinVar Veri Setiyle Birleştirme
==========================================================
Bu script:
  1. missense_dataset.csv dosyasındaki varyantları okur (chrom, pos, ref, alt)
  2. AlphaMissense hg38 dosyasından ilgili am_pathogenicity skorlarını çeker
  3. Birleştirilmiş veri setini missense_dataset_enriched.csv olarak kaydeder

AlphaMissense dosyası çok büyük olduğundan (71M satır), akış (streaming)
yöntemiyle okunur — dosyanın tamamı RAM'e yüklenmez.
"""

import gzip
import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── AYARLAR ───────────────────────────────────────────
CLINVAR_PATH = Path("missense_dataset.csv")
AM_HG38_PATH = Path("data/alphamissense/AlphaMissense_hg38.tsv.gz")
AM_GENE_PATH = Path("data/alphamissense/AlphaMissense_gene_hg38.tsv.gz")
OUTPUT_PATH  = Path("missense_dataset_enriched.csv")


def load_clinvar_variants(csv_path):
    """ClinVar veri setini yükle ve eşleşme anahtarlarını oluştur."""
    print(f"📂 ClinVar veri seti yükleniyor: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Satır sayısı: {len(df)}")
    print(f"   Sütunlar: {list(df.columns)}")

    # Eşleşme anahtarı: (chrom, pos, ref, alt)
    # ClinVar'da chrom = "3", AlphaMissense'te = "chr3" → normalize et
    lookup = {}
    for idx, row in df.iterrows():
        chrom = str(row['chrom']).replace('chr', '')
        key = (chrom, int(row['pos']), str(row['ref']), str(row['alt']))
        lookup[key] = idx

    print(f"   Benzersiz varyant anahtarı: {len(lookup)}")
    return df, lookup


def stream_alphamissense_hg38(gz_path, lookup):
    """
    AlphaMissense hg38 dosyasını satır satır oku,
    eşleşen varyantların am_pathogenicity'sini topla.
    """
    print(f"\n📂 AlphaMissense hg38 dosyası taranıyor: {gz_path}")
    print(f"   (Bu işlem dosya boyutuna bağlı olarak birkaç dakika sürebilir...)")

    matches = {}
    total_lines = 0
    matched_count = 0

    with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
        for line in f:
            # Yorum satırlarını atla
            if line.startswith('#'):
                continue

            total_lines += 1
            if total_lines % 5_000_000 == 0:
                print(f"   İşlenen: {total_lines:,} satır, Eşleşen: {matched_count}")

            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue

            # Sütunlar: CHROM, POS, REF, ALT, genome, uniprot_id,
            #           transcript_id, protein_variant, am_pathogenicity, am_class
            chrom = parts[0].replace('chr', '')
            try:
                pos = int(parts[1])
            except ValueError:
                continue
            ref = parts[2]
            alt = parts[3]

            key = (chrom, pos, ref, alt)
            if key in lookup:
                try:
                    am_score = float(parts[8])
                    am_class = parts[9] if len(parts) > 9 else ""
                    protein_var = parts[7] if len(parts) > 7 else ""
                    uniprot = parts[5] if len(parts) > 5 else ""
                    transcript = parts[6] if len(parts) > 6 else ""

                    matches[key] = {
                        'am_pathogenicity': am_score,
                        'am_class': am_class,
                        'protein_variant': protein_var,
                        'uniprot_id': uniprot,
                        'transcript_id': transcript,
                    }
                    matched_count += 1
                except (ValueError, IndexError):
                    pass

    print(f"\n   Toplam taranan satır: {total_lines:,}")
    print(f"   Eşleşen varyant: {matched_count}")
    return matches


def load_gene_averages(gz_path):
    """Gen düzeyinde ortalama AlphaMissense skorlarını yükle."""
    print(f"\n📂 Gen ortalama skorları yükleniyor: {gz_path}")
    gene_scores = {}

    with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or line.startswith('transcript_id'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                transcript_id = parts[0]
                try:
                    mean_score = float(parts[1])
                    gene_scores[transcript_id] = mean_score
                except ValueError:
                    pass

    print(f"   Yüklenen gen sayısı: {len(gene_scores)}")
    return gene_scores


def main():
    print("=" * 60)
    print("AlphaMissense ↔ ClinVar Birleştirme")
    print("=" * 60)

    # 1. ClinVar veri setini yükle
    df, lookup = load_clinvar_variants(CLINVAR_PATH)

    # 2. AlphaMissense hg38 dosyasını tara
    if not AM_HG38_PATH.exists():
        print(f"\n❌ AlphaMissense hg38 dosyası bulunamadı: {AM_HG38_PATH}")
        print("   Önce download_alphamissense.py'yi çalıştırın!")
        print("   Not: hg38 dosyası genomik koordinat eşleşmesi için gereklidir.")
        print("   download_alphamissense.py'ye bu dosyayı da ekleyebilirsiniz:")
        print('   "AlphaMissense_hg38.tsv.gz": {')
        print('       "md5": "9fd167735f16a1b87da6eb3e4c25fcb5",')
        print('       "size": "643 MB",')
        print('       "desc": "71M SNV (genomik koordinatlar, hg38)"')
        print("   }")
        return

    matches = stream_alphamissense_hg38(AM_HG38_PATH, lookup)

    # 3. Eşleşmeleri DataFrame'e ekle
    print(f"\n{'─' * 50}")
    print("Birleştirme sonuçları:")

    am_pathogenicity = []
    am_class_col = []
    protein_variant = []
    uniprot_id = []
    transcript_id = []

    for idx, row in df.iterrows():
        chrom = str(row['chrom']).replace('chr', '')
        key = (chrom, int(row['pos']), str(row['ref']), str(row['alt']))

        if key in matches:
            m = matches[key]
            am_pathogenicity.append(m['am_pathogenicity'])
            am_class_col.append(m['am_class'])
            protein_variant.append(m['protein_variant'])
            uniprot_id.append(m['uniprot_id'])
            transcript_id.append(m['transcript_id'])
        else:
            am_pathogenicity.append(None)
            am_class_col.append(None)
            protein_variant.append(None)
            uniprot_id.append(None)
            transcript_id.append(None)

    df['am_pathogenicity'] = am_pathogenicity
    df['am_class'] = am_class_col
    df['protein_variant'] = protein_variant
    df['uniprot_id'] = uniprot_id
    df['transcript_id'] = transcript_id

    # 4. İstatistikler
    matched = df['am_pathogenicity'].notna().sum()
    total = len(df)
    match_pct = matched / total * 100

    print(f"  Toplam varyant     : {total}")
    print(f"  Eşleşen varyant   : {matched} ({match_pct:.1f}%)")
    print(f"  Eşleşemeyen       : {total - matched} ({100-match_pct:.1f}%)")

    if matched > 0:
        print(f"\n  am_pathogenicity istatistikleri (eşleşenler):")
        am_stats = df['am_pathogenicity'].dropna()
        print(f"    Ortalama : {am_stats.mean():.4f}")
        print(f"    Std      : {am_stats.std():.4f}")
        print(f"    Min      : {am_stats.min():.4f}")
        print(f"    Max      : {am_stats.max():.4f}")

        # am_class dağılımı
        print(f"\n  am_class dağılımı:")
        print(f"    {df['am_class'].value_counts().to_string()}")

        # Etiket ile korelasyon
        df_matched = df.dropna(subset=['am_pathogenicity'])
        if 'label' in df.columns:
            df_matched_copy = df_matched.copy()
            df_matched_copy['label_int'] = df_matched_copy['label'].astype(int)
            corr = df_matched_copy['am_pathogenicity'].corr(df_matched_copy['label_int'])
            print(f"\n  am_pathogenicity ↔ label korelasyonu: {corr:.4f}")

    # 5. Kaydet
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Birleştirilmiş veri seti kaydedildi: {OUTPUT_PATH}")
    print(f"   Yeni sütunlar: am_pathogenicity, am_class, protein_variant, uniprot_id, transcript_id")

    # 6. Sonraki adımlar
    print(f"\n{'=' * 60}")
    print("SONRAKİ ADIMLAR:")
    print("  1. 'am_pathogenicity' özelliğini model.py'deki feature_cols'a ekleyin")
    print("  2. Modeli yeniden eğitin ve performans artışını gözlemleyin")
    print("  3. SHAP ile am_pathogenicity'nin etkisini analiz edin")
    print("=" * 60)


if __name__ == "__main__":
    main()
