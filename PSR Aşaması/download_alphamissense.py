"""
AlphaMissense Tahmin Dosyalarını Zenodo'dan İndirme
====================================================
Kaynak: https://zenodo.org/records/10813168
Lisans: CC-BY 4.0 (DeepMind Technologies Limited)

İndirilen dosyalar:
  1. AlphaMissense_aa_substitutions.tsv.gz  (1.2 GB) — Protein varyant skorları
  2. AlphaMissense_gene_hg38.tsv.gz         (254 KB) — Gen düzeyinde ortalama skor
"""

import os
import hashlib
import requests
from pathlib import Path

# ── AYARLAR ───────────────────────────────────────────
OUTPUT_DIR = Path("data/alphamissense")
ZENODO_RECORD = "10813168"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

# İndirilecek dosyalar ve md5 hash'leri
FILES = {
    "AlphaMissense_hg38.tsv.gz": {
        "md5": "9fd167735f16a1b87da6eb3e4c25fcb5",
        "size": "643 MB",
        "desc": "71M SNV tahminleri (genomik koordinatlar, hg38) — ClinVar eşleşmesi için"
    },
    "AlphaMissense_gene_hg38.tsv.gz": {
        "md5": "529e3b51cec079272c571affde6f0376",
        "size": "254 KB",
        "desc": "Gen düzeyinde ortalama AlphaMissense skoru"
    },
    "AlphaMissense_aa_substitutions.tsv.gz": {
        "md5": "b9ccb339e0de6cb0a8d1973ad2026576",
        "size": "1.2 GB",
        "desc": "216M protein varyantı (UniProt ID + amino asit değişikliği)"
    },
}


def calculate_md5(filepath, chunk_size=8192):
    """Dosyanın MD5 hash'ini hesapla."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def download_file(url, dest_path, expected_md5=None):
    """
    Dosyayı URL'den indir, ilerleme göster, md5 doğrula.
    Dosya zaten mevcutsa ve md5 doğruysa tekrar indirme.
    """
    if dest_path.exists():
        if expected_md5:
            print(f"  ℹ Dosya mevcut, md5 kontrol ediliyor...")
            actual_md5 = calculate_md5(dest_path)
            if actual_md5 == expected_md5:
                print(f"  ✅ Dosya zaten indirilmiş ve doğrulanmış: {dest_path.name}")
                return True
            else:
                print(f"  ⚠ MD5 uyuşmuyor, yeniden indiriliyor...")
        else:
            print(f"  ✅ Dosya zaten mevcut: {dest_path.name}")
            return True

    print(f"  ⬇ İndiriliyor: {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r  İlerleme: {downloaded_mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)

    print()  # Yeni satır

    # MD5 kontrolü
    if expected_md5:
        print(f"  🔍 MD5 doğrulanıyor...")
        actual_md5 = calculate_md5(dest_path)
        if actual_md5 == expected_md5:
            print(f"  ✅ MD5 doğrulandı!")
            return True
        else:
            print(f"  ❌ MD5 uyuşmuyor!")
            print(f"     Beklenen : {expected_md5}")
            print(f"     Hesaplanan: {actual_md5}")
            return False
    return True


def main():
    print("=" * 60)
    print("AlphaMissense Tahmin Dosyaları — Zenodo İndirici")
    print("=" * 60)
    print(f"Zenodo Record: {ZENODO_RECORD}")
    print(f"Çıktı dizini : {OUTPUT_DIR.resolve()}\n")

    # Çıktı dizini oluştur
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for filename, info in FILES.items():
        print(f"\n{'─' * 50}")
        print(f"📦 {filename} ({info['size']})")
        print(f"   {info['desc']}")

        url = f"{BASE_URL}/{filename}"
        dest = OUTPUT_DIR / filename

        ok = download_file(url, dest, expected_md5=info["md5"])
        if ok:
            success_count += 1

    # Özet
    print(f"\n{'=' * 60}")
    print(f"SONUÇ: {success_count}/{len(FILES)} dosya başarıyla indirildi")
    print(f"Dosyalar: {OUTPUT_DIR.resolve()}")

    if success_count == len(FILES):
        print("\n✅ Tüm dosyalar hazır!")
        print("Sonraki adım: merge_alphamissense.py çalıştırarak")
        print("AlphaMissense skorlarını mevcut veri setinizle birleştirin.")
    else:
        print("\n⚠ Bazı dosyalar indirilemedi, lütfen tekrar deneyin.")

    print("=" * 60)


if __name__ == "__main__":
    main()
