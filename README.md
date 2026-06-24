# AlgoMed: Missense Varyant Patojenite Tahmininde Klinik Odaklı Karar Destek Sistemi
**TEKNOFEST 2026 - Sağlıkta Yapay Zeka Yarışması**

![AlgoMed Projesi](https://img.shields.io/badge/Status-PDR_Phase_Completed-success) ![Model](https://img.shields.io/badge/Model-Stacking_Ensemble-blue) ![Metric](https://img.shields.io/badge/False_Negative-Minimization-red)

AlgoMed, klinik önemi henüz belirlenememiş (VUS) missense varyantların patojenitesini tahmin etmek üzere geliştirilmiş, veri sızıntısından arındırılmış (leak-free) ve klinik güvenliği temel alan açıklanabilir bir yapay zeka (XAI) sistemidir.

## 🧬 Problem ve Çözüm Yaklaşımı
Genomik varyant verilerindeki en büyük iki engel **yüksek eksik değer oranları** ve **keskin sınıf dengesizliğidir**. Patojenik bir varyantın "sağlıklı" (benign) olarak hatalı etiketlenmesi (Yalancı Negatif - FN) hastalar için hayati risk oluşturur.

AlgoMed sistemi bu sorunu salt bir makine öğrenmesi metrik maksimizasyonu olarak değil, bir **"Klinik Risk Yönetimi"** problemi olarak ele alır.

### Öne Çıkan Özellikler (AlgoMed V9 Mimarisi)
1. **Klinik Güvenlik (Maliyet-Duyarlı Karar Eşiği):** Standart 0.50 eşiği yerine, algoritmik Kesinlik-Duyarlılık dengesi kurularak eşik **0.20**'ye çekilmiştir. Bu dinamik yaklaşım sayesinde klinik hata olan FN (Hastayı Kaçırma) oranı proje başından beri en düşük seviye olan **64**'e indirilmiştir.
2. **Sızdırmaz Boru Hattı (Leak-Free Custom CV):** Tüm aşırı örnekleme (SMOTE) ve ön işleme adımları, özel yazılmış Cross-Validation döngüsüyle sadece eğitim setine izole edilmiş, modelin test ortamındaki genellenebilirliği %100 garanti altına alınmıştır.
3. **Biyokimyasal Özellik Mühendisliği:** Amino asitlerin moleküler ağırlık, polarite, hidrofobisite farkları ve Grantham mesafeleri gibi evrimsel metrikleri modele "domain-expertise" olarak dahil edilmiştir.
4. **Hibrit Stacking Mimarisi:** XGBoost, LightGBM ve CatBoost algoritmalarının OOF (Out-of-Fold) tahminleri Lojistik Regresyon meta-modeli ile birleştirilerek yüksek boyutlu verideki kararlılık maksimize edilmiştir.

---

## 📊 Final Performans Sonuçları (V9)

*Klinik risk önceliklendirildiği için F1/MCC maksimizasyonu yerine False Negative (FN) minimizasyonuna gidilmiştir.*

- **MASTER F1 Skoru:** %89.58
- **MASTER MCC:** 0.5425
- **False Negative (FN):** 64 (Mükemmel Gelişme)
- **Kullanılan Threshold:** 0.20

### Alt Grup Başarıları (F1)
- **CFTR Grubu:** %96.13
- **PAH Grubu:** %94.03
- **KANSER Grubu:** %88.74

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
