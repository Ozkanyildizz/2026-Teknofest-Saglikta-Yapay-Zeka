# AlgoMed: Missense Varyant Patojenite Tahmininde Klinik Odaklı Karar Destek Sistemi
**TEKNOFEST 2026 - Sağlıkta Yapay Zeka Yarışması**

![AlgoMed Projesi](https://img.shields.io/badge/Status-PDR_Phase_Completed-success) ![Model](https://img.shields.io/badge/Model-Stacking_Ensemble-blue) ![Metric](https://img.shields.io/badge/False_Negative-Minimization-red)

AlgoMed, klinik önemi henüz belirlenememiş (VUS) missense varyantların patojenitesini tahmin etmek üzere geliştirilmiş, veri sızıntısından arındırılmış (leak-free) ve klinik güvenliği temel alan açıklanabilir bir yapay zeka (XAI) sistemidir.

## 🧬 Problem ve Çözüm Yaklaşımı
Genomik varyant verilerindeki en büyük iki engel **yüksek eksik değer oranları** ve **keskin sınıf dengesizliğidir**. Patojenik bir varyantın "sağlıklı" (benign) olarak hatalı etiketlenmesi (Yalancı Negatif - FN) hastalar için hayati risk oluşturur.

AlgoMed sistemi bu sorunu salt bir makine öğrenmesi metrik maksimizasyonu olarak değil, bir **"Klinik Risk Yönetimi"** problemi olarak ele alır.

### Öne Çıkan Özellikler (AlgoMed V9 Mimarisi)
1. **Klinik Güvenlik (Maliyet-Duyarlı Karar Eşiği):** Standart 0.50 eşiği terk edilerek klinik stres testine (%86 Benign) uygun olarak MCC optimizasyonu ile eşik 0.56'da, **Klinik Güvenlik** için ise eşik **0.25**'e çekilmiştir. Bu dinamik yaklaşım sayesinde klinik hata olan FN (Hastayı Kaçırma) riski minimize edilmiştir (Eşik=0.25 için Recall %92.7).
2. **Sızdırmaz Boru Hattı (Leak-Free Custom CV):** Tüm aşırı örnekleme (SMOTE) ve ön işleme adımları, özel yazılmış Cross-Validation döngüsüyle sadece eğitim setine izole edilmiş, modelin test ortamındaki genellenebilirliği %100 garanti altına alınmıştır.
3. **Biyokimyasal Özellik Mühendisliği:** Amino asitlerin moleküler ağırlık, polarite, hidrofobisite farkları ve evrimsel (Grantham, BLOSUM62) metrikleri modele "domain-expertise" olarak dahil edilmiştir.
4. **Hibrit Stacking Mimarisi:** XGBoost, LightGBM ve CatBoost algoritmalarının OOF (Out-of-Fold) tahminleri Lojistik Regresyon meta-modeli ile birleştirilerek yüksek boyutlu verideki kararlılık maksimize edilmiştir.

---

## 📊 Final Performans Sonuçları (V9)

*Yarışma kapsamındaki asimetrik test setine ("Klinik Stres Testi") uygun olarak genel ayrıştırma gücü için MCC maksimizasyonu hedeflenmiş, klinik güvenlik için ise FN minimizasyonuna gidilmiştir.*

- **MASTER F1 Skoru:** 0.8836
- **MASTER MCC:** 0.5594 (Proje İçi En Yüksek MCC)
- **MASTER PR-AUC:** 0.9268
- **Yanlış Negatif (FN):** 243 (Eşik=0.56) / Klinik Güvenlikte Çok Daha Düşük
- **Kullanılan Threshold (Karar Eşiği):** 0.56 (Genel MCC) / 0.25 (Klinik Güvenlik - Recall: %92.7)

### Alt Grup Başarıları (MCC)
- **KANSER Grubu:** 0.7169
- **CFTR Grubu:** 0.6561
- **PAH Grubu:** 0.5399

---

## 📂 Proje Yapısı

```text
algomed/
├── PDR Aşaması/
│   ├── Rapor/                   # TEKNOFEST'e sunulan resmi PDR Raporları ve PDF çıktıları
│   ├── V9/                      # Projenin nihai üretim hattı (Leak-Free Custom CV, SMOTE)
│   ├── V1 - V8/                 # Geliştirme, keşif ve Optuna hiperparametre optimizasyon aşamaları
│   └── PROJE_GELISIM_RAPORU.md  # V1'den V9'a kadar adım adım alınan tüm mimari kararlar
├── README.md                    # Proje kök dizin açıklaması
└── PSR Aşaması/                  # TEKNOFEST PSR raporunu başarılı bir şekilde geçtiğimiz çalışmaların bulunduğu dizin 
```

## ⚙️ Kurulum ve Çalıştırma

Projenin final kodları `PDR Aşaması/V9/` dizininde yer almaktadır.

```bash
# Gerekli kütüphanelerin yüklenmesi
pip install requirements.txt

# Modelin eğitimi ve sızdırmaz CV döngüsünün başlatılması
python "PDR Aşaması/V9/train.py"
```

*Not: Sistem GPU hızlandırmasını (CUDA) otomatik olarak algılar ve kullanır.*

---
*Bu proje TEKNOFEST 2026 Sağlıkta Yapay Zeka yarışması PSR ve PDR aşaması için geliştirilmiştir.*
