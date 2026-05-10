# AlgoMed — Missense Varyant Patojenite Sınıflandırması

> **TEKNOFEST 2026 – Sağlıkta Yapay Zeka Yarışması**  
> Missense varyantların patojenik (hastalık yapıcı) veya benign (zararsız) olarak sınıflandırılması.

---

## 📁 Proje Yapısı

```
algomed/
├── data.py                      # Adım 1: ClinVar veri setini indir
├── eda.py                       # Adım 2: Keşifsel veri analizi (EDA)
├── model.py                     # Adım 3: Model eğitimi ve değerlendirme
├── download_alphamissense.py    # Adım 4: AlphaMissense verilerini indir
├── merge_alphamissense.py       # Adım 5: AlphaMissense skorlarını birleştir
├── tsv_to_csv.py                # Yardımcı: TSV → CSV dönüştürme
├── missense_dataset.csv         # Üretilen dengeli veri seti (3000 varyant)
├── data/
│   └── alphamissense/           # İndirilen AlphaMissense dosyaları
│       ├── AlphaMissense_hg38.tsv.gz           (643 MB)
│       ├── AlphaMissense_gene_hg38.tsv.gz      (254 KB)
│       └── AlphaMissense_aa_substitutions.tsv.gz (1.2 GB)
└── rapor/
    └── generate_psr.py          # PSR rapor üretici
```

---

## 🚀 Adım Adım Çalıştırma Rehberi

### Adım 1: Veri İndirme ve Hazırlama — `data.py`

```bash
python data.py
```

**Ne yapar:**
- HuggingFace'den `songlab/clinvar` veri setini indirir
- Tüm ClinVar missense varyantlarını yükler
- **Dengeli alt küme** oluşturur: 1500 Patojenik + 1500 Benign = 3000 varyant
- `missense_dataset.csv` olarak kaydeder

**Çıktı:** `missense_dataset.csv`

**Sütunlar:**
| Sütun | Açıklama |
|---|---|
| `chrom` | Kromozom numarası |
| `pos` | Genomik pozisyon |
| `ref` | Referans allel |
| `alt` | Alternatif allel |
| `GPN-MSA` | GPN-MSA evrimsel skor |
| `CADD` | CADD in-silico risk skoru |
| `phyloP-100v` | phyloP korunmuşluk (100 omurgalı) |
| `phyloP-241m` | phyloP korunmuşluk (241 memeli) |
| `phastCons-100v` | phastCons korunmuşluk skoru |
| `ESM-1b` | ESM-1b protein dil modeli skoru |
| `NT` | Nucleotide Transformer skoru |
| `HyenaDNA` | HyenaDNA skoru (sıfır varyans → çıkarıldı) |
| `label` | 0 = Benign, 1 = Patojenik |

---

### Adım 2: Keşifsel Veri Analizi (EDA) — `eda.py`

```bash
python eda.py
```

**Ne yapar:**
- Veri boyutu ve sütun tiplerini raporlar
- Etiket dağılımını kontrol eder (dengeli mi?)
- **Eksik değer analizi** (ESM-1b ~%11.5 eksik)
- İstatistiksel özet (min, max, ortalama, std)
- **Aykırı değer tespiti** (IQR yöntemi, 3×IQR)
- **Özellik-etiket korelasyonu** (hangi özellikler patojenisite ile ilişkili?)
- Yinelenmiş satır kontrolü

---

### Adım 3: Model Eğitimi — `model.py`

```bash
python model.py
```

**Ne yapar:**

1. **Ön İşleme Pipeline:**
   - Eksik değerler → Median imputation (ESM-1b için)
   - RobustScaler → Outlier'lara dayanıklı ölçekleme
   - VarianceThreshold → Sıfır varyanslı sütunları çıkarma (HyenaDNA)

2. **Model: LightGBM + XGBoost Soft-Voting Ensemble**
   - LightGBM (500 ağaç, lr=0.05, max_depth=6)
   - XGBoost (500 ağaç, lr=0.05, max_depth=5)
   - İki modelin olasılık çıktılarının ortalaması (soft-voting)

3. **Değerlendirme: 5-Fold Stratified Cross-Validation**
   - F1 Score, ROC-AUC, PR-AUC, Balanced Accuracy
   - Confusion Matrix (FN = klinik risk)
   - Classification Report

4. **Açıklanabilirlik: SHAP**
   - Her özelliğin model kararına etkisi (|SHAP| ortalaması)
   - En etkili özellikler: CADD, phyloP, phastCons, GPN-MSA

**Model Seçim Gerekçesi:**
| Alternatif | Neden Elendi |
|---|---|
| Derin Öğrenme | 3000 örnek için overfit riski, tabular veride GBDT'den düşük |
| Lojistik Regresyon | Non-linear etkileşimleri yakalayamaz |

---

### Adım 4: AlphaMissense Verilerini İndirme — `download_alphamissense.py`

```bash
python download_alphamissense.py
```

**Ne yapar:**
- [Zenodo](https://zenodo.org/records/10813168)'dan AlphaMissense tahmin dosyalarını indirir
- **MD5 hash doğrulaması** ile dosya bütünlüğünü kontrol eder
- Daha önce indirilmiş dosyaları tekrar indirmez

**İndirilen dosyalar:**
| Dosya | Boyut | İçerik |
|---|---|---|
| `AlphaMissense_hg38.tsv.gz` | 643 MB | 71M SNV tahmini (genomik koordinatlar) |
| `AlphaMissense_gene_hg38.tsv.gz` | 254 KB | Gen düzeyinde ortalama skor |
| `AlphaMissense_aa_substitutions.tsv.gz` | 1.2 GB | 216M protein varyantı |

> **Not:** AlphaMissense, DeepMind tarafından geliştirilen %90+ doğruluklu bir derin öğrenme modelidir.  
> Lisans: CC-BY 4.0 (DeepMind Technologies Limited)

---

### Adım 5: AlphaMissense Skorlarını Birleştirme — `merge_alphamissense.py`

```bash
python merge_alphamissense.py
```

**Ne yapar:**

1. `missense_dataset.csv` dosyasını yükler
2. `AlphaMissense_hg38.tsv.gz` dosyasını **streaming** yöntemiyle tarar (71M satır, RAM'e yüklemeden)
3. **(chrom, pos, ref, alt)** anahtarıyla ClinVar ↔ AlphaMissense eşleşmesi yapar
4. Eşleşen varyantlara 5 yeni sütun ekler:
   - `am_pathogenicity` — AlphaMissense patojenisite skoru (0-1)
   - `am_class` — benign / ambiguous / pathogenic
   - `protein_variant` — Amino asit değişikliği (ör. A123V)
   - `uniprot_id` — UniProt protein ID
   - `transcript_id` — Transkript ID
5. `am_pathogenicity ↔ label` korelasyonunu hesaplar
6. Zenginleştirilmiş veri setini kaydeder

**Çıktı:** `missense_dataset_enriched.csv`

---

## 📌 Sonraki Adımlar

Merge işlemi tamamlandıktan sonra:

1. **`model.py`'deki `feature_cols`'a yeni özelliği ekle:**
   ```python
   feature_cols = ['GPN-MSA', 'CADD', 'phyloP-100v', 'phyloP-241m',
                   'phastCons-100v', 'ESM-1b', 'NT', 'am_pathogenicity']  # ← EKLENDİ
   ```

2. **Modeli `missense_dataset_enriched.csv` ile yeniden eğit**

3. **SHAP analizi ile `am_pathogenicity`'nin etkisini gözlemle**

---

## 🛠️ Gereksinimler

```bash
pip install pandas numpy scikit-learn lightgbm xgboost shap requests datasets
```

| Kütüphane | Kullanım |
|---|---|
| `pandas`, `numpy` | Veri işleme |
| `scikit-learn` | Pipeline, CV, metrikler |
| `lightgbm` | LightGBM modeli |
| `xgboost` | XGBoost modeli |
| `shap` | Açıklanabilirlik |
| `requests` | Dosya indirme |
| `datasets` | HuggingFace veri yükleme |

---

## 📚 Veri Kaynakları

| Kaynak | Açıklama | Link |
|---|---|---|
| ClinVar (songlab) | Missense varyant veri seti | [HuggingFace](https://huggingface.co/datasets/songlab/clinvar) |
| AlphaMissense | DeepMind patojenisite tahminleri | [Zenodo](https://zenodo.org/records/10813168) |
