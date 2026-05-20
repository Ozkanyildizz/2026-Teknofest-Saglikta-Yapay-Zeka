# AlgoMed — PDR Aşaması

> **TEKNOFEST 2026 – Sağlıkta Yapay Zeka Yarışması**  
> Missense Varyant Patojenite Sınıflandırması — Proje Tasarım Raporu (PDR)

---

## 📌 Genel Bakış

Bu klasör, TEKNOFEST 2026 Sağlıkta Yapay Zeka yarışmasının **PDR (Proje Tasarım Raporu) aşamasına** ait tüm çalışmaları içermektedir. PSR aşaması başarıyla tamamlanmış olup bu aşamada yarışma organizasyonu tarafından sağlanan **gerçek yarışma veri seti** üzerinde model geliştirme çalışmaları yürütülmektedir.

---

## 🗂️ Klasör Yapısı

```
PDR Aşaması/
├── README.md                        ← Bu dosya (takım kuralları & genel bilgi)
├── INTEGRATION_GUIDE.md             ← V4 entegrasyon rehberi (YENİ)
├── Rapor/
│   └── 2026_PDR_Şablon_*.docx       ← Yarışma rapor şablonu
├── universite-veri-seti/
│   └── EĞİTİM (TRAIN) SETLERİ/
│       ├── YARISMA_TRAIN_MASTER.csv  ← Ana veri seti (2931 varyant)
│       ├── YARISMA_TRAIN_KANSER.csv  ← Kanser alt grubu (388 varyant)
│       ├── YARISMA_TRAIN_PAH.csv     ← PAH alt grubu (372 varyant)
│       └── YARISMA_TRAIN_CFTR.csv   ← CFTR alt grubu (111 varyant)
├── V1/                              ← 1. versiyon (Baseline)
│   ├── train.py, eda.py, README.md
│   └── F1: 0.8711, MCC: 0.5054
├── V2/                              ← 2. versiyon (Missing Indicator)
│   ├── train.py, eda.py, README.md
│   └── F1: 0.8903, MCC: 0.5301
├── V3/                              ← 3. versiyon (Amino Asit Bio.)
│   ├── train.py, eda.py, README.md
│   └── F1: 0.8929, MCC: 0.5401
├── V4/                              ← 4. versiyon (CatBoost Stacking)
│   ├── train.py, eda.py, README.md
│   └── F1: 0.8958, MCC: 0.5434
├── V5/                              ← 5. versiyon (GPU Accelerated XGB)
│   ├── train.py, eda.py, README.md
│   └── F1: 0.8941, KANSER F1: 0.9069
└── ...
```

---

## 🧬 Yarışma Veri Seti — Detaylı Yapı

### Dosyalar ve Boyutlar

| Dosya | Satır | Sütun | Patojenik (1) | Benign (0) | Eksik Değer |
|---|---|---|---|---|---|
| `MASTER` | 2.931 | 353 | 2.149 (%73) | 782 (%27) | ~568.000 |
| `KANSER` | 388 | 353 | 268 (%69) | 120 (%31) | ~78.000 |
| `PAH` | 372 | 353 | 310 (%83) | 62 (%17) | ~71.000 |
| `CFTR` | 111 | 353 | 90 (%81) | 21 (%19) | ~12.000 |

> ⚠️ **Önemli:** Veri **dengesizdir** (Patojenik >> Benign). Doğruluk (accuracy) metriği yanıltıcı olur. F1, MCC ve PR-AUC kullanılmalıdır.

### Sütun Grupları (353 sütun)

| Prefix | Adet | Tür | Açıklama |
|---|---|---|---|
| `AL_1..AL_334` | 334 | float | Şifreli sayısal genomik özellikler |
| `CAT_1..CAT_6` | 6 | string (kategorik) | Popülasyon, genotip, bölge tipi |
| `EK_1..EK_9` | 9 | float | Ek sayısal özellikler |
| `AA_1, AA_2` | 2 | string | Ref. ve alt. amino asit (tek harf kodu) |
| `Variant_ID` | 1 | string | Örnek kimliği (VAR_XXXXXX) |
| `Label` | 1 | int | **Hedef değişken** — 0: Benign, 1: Patojenik |

### Kategorik Sütunlar (CAT_)

| Sütun | Değerler | Anlamı |
|---|---|---|
| `CAT_1` | gnomADe_AFR/ASJ/EAS/FIN/MID/NFE/SAS/EUR... | gnomAD popülasyon grubu |
| `CAT_2` | AllofUs_AFR/AMR/EAS/EUR/MID/OTH/SAS | All of Us popülasyon grubu |
| `CAT_3` | A/A, C/C, G/G, T/T, ./. | Referans genotip |
| `CAT_4` | A/A, C/C, G/G, T/T, ./. | Alternatif genotip |
| `CAT_5` | A/A, C/C, G/G, T/T, ./. | 3. genotip bilgisi |
| `CAT_6` | decoy&segdup, lcr, segdup | Genomik bölge kalite filtresi |

---

## 📋 PDR Rapor Şablonu — Ne İsteniyor?

Rapor **100 puan** üzerinden değerlendiriliyor:

| Bölüm | Puan | Anahtar Beklentiler |
|---|---|---|
| **1. Giriş** | 10 | Problem tanımı, klinik önem, veri yapısı açıklaması |
| **2. Yöntem** | 25 | Veri mühendisliği, model seçimi, hiperparametre, CV, SHAP, eşik belirleme |
| **3. Bulgular** | 30 | **F1 + MCC + PR-AUC** zorunlu; karmaşıklık matrisi; **alt grup bazında ayrı sonuçlar** |
| **4. Sonuç** | 25 | FP/FN klinik yorum, güçlü/zayıf yönler, son aşama zorluları |
| **5. Kaynakça** | 10 | Akademik yazım, görseller, Türkçe dil kuralları |

> 💡 **Bulgular bölümü için MUTLAKA şunlar olmalı:**
> - MASTER, KANSER, PAH, CFTR setleri için **ayrı ayrı** sonuç tabloları
> - Farklı karar eşiklerinin (0.3, 0.4, 0.5, 0.6...) performans karşılaştırması
> - Precision-Recall eğrisi görseli

---

## 👥 Takım Çalışma Kuralları

### Temel Prensipler

Bu proje takım halinde yürütülmektedir. Kod çakışmalarını ve karmaşayı önlemek için **versiyon klasörü** yaklaşımı benimsenmiştir. Her takım üyesi kendi çalışmasını bağımsız bir klasörde geliştirerek takıma katkıda bulunur.

---

### Versiyon Klasörü Sistemi

Her takım üyesi çalışmaya başlamadan önce bir **Vx/** klasörü oluşturur:

```
PDR Aşaması/
├── V1/   ← 1. takım üyesi çalışıyor (Özkan)
├── V2/   ← 2. takım üyesi çalışacak
├── V3/   ← 3. takım üyesi çalışacak
└── ...
```

**Kurallar:**
1. Çalışmaya başlamadan önce mevcut versiyonlara bak — hangi numaraya kadar gidilmiş?
2. Bir sonraki numarayı al ve o klasörü oluştur (ör. V3 varsa sen V4 açarsın)
3. **Tüm çalışmanı kendi Vx/ klasörünün içinde yap** — başkasının klasörüne dokunma
4. Her Vx/ klasörünün kendi `README.md` dosyası olmalıdır (bkz. aşağıda)

---

### GitHub'a Push Kuralı

> ⚠️ **Altın Kural:** Yalnızca **önceki versiyondan daha iyi sonuç alan** çalışmalar GitHub'a pushlanır.

| Durum | Yapılacak İşlem |
|---|---|
| Sonuçların önceki versiyondan **iyiyse** | Commit & Push — `git push origin main` |
| Sonuçların önceki versiyonla **eşitse** | Tercihen push etme, takımla tartış |
| Sonuçların önceki versiyondan **kötüyse** | Push etme — çalışmaları yerel tut, geliştir |

**Karşılaştırma metriği:** MASTER veri seti üzerinde **F1 skoru** ana kıyaslama kriteri olarak kullanılır. MCC ve PR-AUC ikincil kriterlerdir.

---

### Git Commit Mesajı Formatı

```
[Vx] Kısa açıklama

Örnek:
[V1] LightGBM baseline — F1: 0.87, MCC: 0.71
[V2] Feature engineering + SMOTE — F1: 0.89, MCC: 0.74
[V3] Optuna hyperparameter tuning — F1: 0.91, MCC: 0.78
```

---

### Her Versiyon README.md'si Ne İçermeli?

Her `Vx/README.md` dosyası aşağıdaki şablona göre doldurulmalıdır:

```markdown
# Vx — [Takım Üyesi Adı] — [Tarih]

## Özet
Kısaca ne denedin, hedefin ne?

## Yöntem
- Hangi modeli kullandın?
- Veri ön işleme adımların neler?
- Herhangi bir feature engineering yaptın mı?
- Hiperparametre optimizasyonu yaptın mı? Nasıl?

## Sonuçlar

### MASTER Veri Seti
| Metrik | Değer |
|---|---|
| F1 Skoru | ? |
| MCC | ? |
| PR-AUC | ? |
| ROC-AUC | ? |

### Alt Grup Sonuçları
| Grup | F1 | MCC | PR-AUC |
|---|---|---|---|
| KANSER | ? | ? | ? |
| PAH | ? | ? | ? |
| CFTR | ? | ? | ? |

## Karar Eşiği Analizi
Hangi eşik değerinde en iyi sonucu aldın?

## Önceki Versiyonla Karşılaştırma
V(x-1)'e göre iyileşme: ...

## Öğrendiklerin / Notlar
Sonraki versiyona ne öneriryorsun?

## GitHub'a Pushlandı mı?
[ ] Evet / [ ] Hayır — Neden?
```

---

## 🚀 Yeni Başlayan Takım Üyesi İçin Hızlı Başlangıç

1. **Repoyu güncelle:**
   ```bash
   git pull origin main
   ```

2. **Mevcut versiyonları kontrol et:**
   ```bash
   ls "PDR Aşaması/"
   ```

3. **Yeni klasörünü oluştur:**
   ```bash
   mkdir "PDR Aşaması/V2"
   ```

4. **V1/README.md'yi oku** — önceki sonuçları ve önerileri öğren

5. **Kendi klasöründe çalış:**
   ```
   PDR Aşaması/V2/
   ├── README.md        ← sonuçlarını buraya yaz
   ├── model.py         ← modelini buraya yaz
   ├── eda.py           ← analizlerini buraya yaz
   └── ...
   ```

6. **Sonuçlar V1'den iyiyse push et:**
   ```bash
   git add "PDR Aşaması/V2/"
   git commit -m "[V2] Açıklayıcı mesaj — F1: X.XX, MCC: X.XX"
   git push origin main
   ```

---

## 📊 Sürüm Karşılaştırma Tablosu

Bu tablo, yeni bir versiyon tamamlandıkça güncellenecektir:

| Versiyon | Geliştirici | Tarih | Model | MASTER F1 | MCC | PR-AUC | Özellikler | Push? |
|---|---|---|---|---|---|---|---|---|
| **V1** | - | Mayıs 2026 | LGB + XGB (2-lü) | 0.8711 | 0.5054 | 0.9207 | Baseline | ✅ |
| **V2** | - | Mayıs 2026 | LGB + XGB (2-lü) | 0.8903 | 0.5301 | 0.9206 | + Missing Indicator | ✅ |
| **V3** | - | Mayıs 2026 | LGB + XGB (2-lü) | 0.8929 | 0.5401 | — | + Amino Asit Bio. + Optuna | ✅ |
| **V4** | - | Mayıs 2026 | LGB+XGB+CatBoost | **0.8958** | **0.5434** | 0.9199 | + CatBoost Stacking + EK_ Interact | ✅ |
| **V5** ⭐| - | Mayıs 2026 | LGB+XGB+CatBoost | 0.8941 | 0.5370 | **0.9222** | **Tam GPU Desteği**, XGB Optuna, KANSER F1: 0.9069 | ✅ |

> ✅ = Push edildi | 🔄 = Beklemede | — = Henüz belirlenmedi

**Rapor**: Detaylı bilgi için vx/README.md dosyalarını okuyunuz.

---

## 🛠️ Gereksinimler

```bash
pip install rewuirements.txt
```

| Kütüphane | Kullanım |
|---|---|
| `pandas`, `numpy` | Veri işleme |
| `scikit-learn` | Pipeline, CV, metrikler |
| `lightgbm`, `xgboost` | Gradient boosting modelleri |
| `shap` | Açıklanabilirlik (SHAP) |
| `optuna` | Hiperparametre optimizasyonu |
| `imbalanced-learn` | SMOTE — veri dengeleme |
| `matplotlib`, `seaborn` | Görselleştirme |

---

## 📚 Faydalı Kaynaklar

- [PSR Aşaması README](../PSR%20A%C5%9Famas%C4%B1/README.md) — önceki aşama modeli ve yaklaşımı
- [PDR Rapor Şablonu](./Rapor/) — yarışma rapor formatı
- [ClinVar Veri Kaynağı](https://huggingface.co/datasets/songlab/clinvar)
- [AlphaMissense (DeepMind)](https://zenodo.org/records/10813168)
- [LightGBM Dokümantasyonu](https://lightgbm.readthedocs.io/)
- [SHAP Dokümantasyonu](https://shap.readthedocs.io/)
