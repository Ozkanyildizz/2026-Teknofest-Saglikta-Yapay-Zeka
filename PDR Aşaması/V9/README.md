# V9 — AlgoMed Final PDR Modeli

> **KLİNİK GÜVENLİK VE MATEMATİKSEL DENGEYE ULAŞIM**
> 
> V9, projenin başından beri hedeflenen en yüksek genel ayrıştırma kapasitesine (**MCC=0.5594**) ulaşırken, aynı zamanda dinamik eşik optimizasyonuyla klinik olarak "hastayı kaçırmama" (FN minimization) stratejisini koruyan nihai PDR sürümüdür.

---

## ✅ V9 Final Sonuçları (MASTER 5-Fold CV, SMOTE)

| Metrik | Eski Sürüm (V8) | **V9 (Final)** | Durum / Yorum |
|---|---|---|---|
| **F1 Skoru** | 0.8941 | **0.8836** | Daha gerçekçi ve sağlam seviyeye oturdu |
| **MCC** | 0.5370 | **0.5594** | **Çok Büyük Sıçrama** |
| **PR-AUC** | 0.9222 | **0.9268** | Proje boyunca ulaşılan en yüksek seviye |
| **ROC-AUC** | - | **0.8503** | - |
| **Optimum Eşik** | 0.20 | **0.56** | Genel başarımı maksimize eden eşik noktası |

*(Not: Genel ayrıştırma kapasitesi 0.56 eşiğinde en üst düzeye çıkmaktadır. Ancak klinik uygulamada "yanlış negatif" (hastalık kaçırma) riskini minimize etmek için karar eşiği **0.20**'ye çekildiğinde, Recall ≥%92.7 sağlanmakta ve FN çok daha güvenli seviyelere inmektedir.)*

---

## 📊 Alt Grup Sonuçları (V9)

V9 modelinin KANSER, CFTR ve PAH gibi zorlayıcı alt gruplardaki başarısı:

| Grup | F1 Skoru | MCC | PR-AUC | Eşik (Threshold) |
|---|---|---|---|---|
| **KANSER** | 0.9193 | **0.7169** | 0.9645 | 0.56 |
| **CFTR** | 0.9257 | 0.6561 | 0.9878 | 0.56 |
| **PAH** | 0.9348 | 0.5399 | 0.9487 | 0.46 |

> Özellikle KANSER alt grubunda elde edilen 0.7169'luk MCC skoru, sistemin bu hastalığa özgü varyantları ne kadar başarılı izole ettiğinin kanıtıdır.

---

## 🚀 V9'daki Kritik Geliştirmeler

### 1. Evrimsel ve Popülasyon Özellikleri
V8'deki biyokimyasal özellik mühendisliğinin üzerine, **Grantham mesafesi** ve **BLOSUM62** skorları sisteme tam entegre edilmiştir. Ayrıca, popülasyon genetiği farklarını dengelemek için `CAT_` etkileşim öznitelikleri (Interaction features) modele dahil edilmiştir.

### 2. Optuna ile Yeniden Optimizasyon
XGBoost ve CatBoost algoritmaları için GPU ivmeli, 500'er iterasyonluk ağır Bayesçi Optimizasyon (Optuna) süreçleri işletilmiştir. Bulunan en iyi parametreler, özellikle L1/L2 regülarizasyon bariyerlerini artırarak modelin ezberlemesini (overfitting) engellemiştir.

### 3. Sızdırmazlık (Leak-Free) Garantisi
Veri sızıntısını önlemek için 5-Katmanlı Tabakalı CV tamamen özel olarak (Custom Loop) kodlanmıştır. **SMOTE** ile sentetik veri üretimi sadece eğitim (Train) katmanlarına izole edilmiş, Validation katmanı asla sentetik veri görmeyecek şekilde tasarlanmıştır.

---

## 🛠 Model Mimarisi

* **Temel Katman (Base Learners):**
  * **XGBoost:** (GPU - cuda/hist) 500-trial Optuna sonuçlarıyla.
  * **CatBoost:** (GPU) 500-trial Optuna sonuçlarıyla.
  * **LightGBM:** (GPU - histogram) V5 bazlı sağlam parametrelerle.
* **Meta Model:** `LogisticRegression` tabanlı Stacking Classifier (Yığınsal Topluluk).
* **Aşırı Örnekleme (Oversampling):** Sızdırmaz CV döngüsü içerisinde çalışan `SMOTE`.

---

## 📁 Klasör Yapısı

* `train.py`: V9 modelinin baştan sona veri işleme, SMOTE, Stacking ve CV süreçleriyle eğitildiği ana dosya.
* `optuna_study.py`: Hiperparametre optimizasyonu için kullanılan script.
* `v9_optuna_results.json`: Optuna tarafından bulunan final parametreleri.
* `v9_outputs.txt`: Tüm Fold ve alt grup (KANSER, PAH, vb.) metriklerinin konsol çıktısı.
