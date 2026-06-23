# V9 — Sızdırmaz Pipeline + FN Risk Özellikleri (Final Model)

> **V7'DEN V9'A KLİNİK RİSK ODAKLI GEÇİŞ**
>
> V7'deki yüksek performanslı sızdırmaz (leak-free) Stacking modelinin üzerine, Yalancı Negatif (FN) riskini klinik olarak tolere edilebilir seviyeye çekmek amacıyla hata analizi yapılmış ve **FN Risk Bayrakları** eklenmiştir.
> 
> *Not: V8 modeli, V7 üzerindeki salt özellik testlerinden ibaret olduğu için üretim hattında V9 olarak adlandırılmıştır.*

---

## ✅ V9 Sonuçları (PDR Aşaması Final Modeli)

> **V9, projenin başından beri en düşük FN (Hastalığı Kaçırma) oranına ulaşmıştır!**
> Hedef, salt yüksek MCC değil; klinik olarak "güvenilir" bir çalışma noktası (Threshold=0.20) bulmaktır.

### MASTER 5-Fold CV (Oversampling=SMOTE, Eşik=0.20)

| Metrik | V7 | **V9** | Durum |
|---|---|---|---|
| **F1 Skoru** | 0.8959 | **0.8958** | Korundu |
| **MCC** | 0.5532 | **0.5425** | Klinik güvenlik takası |
| **FN (Kaçırılan Hasta)** | 101 | **64** | **-37 (Mükemmel Gelişme)** |
| **FP (Yanlış Alarm)** | 375 | **421** | Tolere edilebilir |

### Alt Grup Sonuçları (V9)

| Grup | F1 | Eşik | Karakteristik Davranış |
|---|---|---|---|
| **CFTR** | **0.9613** | 0.20 | Varyasyon kalıpları son derece kararlı |
| **KANSER** | **0.8874** | 0.20 | Kompleks gen yapısı nedeniyle dengeli F1 |
| **PAH** | **0.9403** | 0.51 | Çok keskin dengesizlik (5:1), yüksek eşik optimizasyonu |

### Karmaşıklık Matrisi (MASTER - Threshold: 0.20)

```text
                Tahmin
                Benign  Patojenik
Gerçek Benign     361       421 (FP)
Gerçek Patojen     64 (FN) 2085
```
*FN=64 seviyesine indirilerek, klinik risk olan "hastayı eve gönderme" ihtimali projenin başından beri en düşük seviyeye çekilmiştir.*

---

## 🚀 V9'daki Kritik Geliştirmeler

### 1. FN Risk Bayrakları (Klinik Odaklı Feature Engineering)
V8'deki kapsamlı analizler sonucunda, algoritmaların "benign" sandığı ama aslında patojenik olan vakalarda bazı biyolojik örüntüler keşfedildi. Modele şu bayraklar eklendi:
* `EK7_low_flag`: EK_7 değerinin aşırı düşük olduğu durumlar.
* `AA_missing_flag`: Amino asit dönüşüm verisinin bulunmadığı eksiklik durumları.

### 2. Custom CV Loop (Sızdırmazlık - Leak Free)
Tüm aşırı örnekleme (Oversampling) işlemleri, `implearn.pipeline` veya tüm veri setine uygulanmak yerine **Özel Kodlanmış (Custom) Cross-Validation Döngüsü** ile sadece ve sadece eğitim (Train) katmanlarına izole edildi. Model doğrulama (Validation) aşamasında hiçbir zaman sentetik veri (SMOTE) görmez. Sızdırmazlık %100 garanti altındadır.

### 3. ADASYN'den SMOTE'a Geçiş
V7'de kullanılan ADASYN, Windows ortamlarında Python `multiprocessing` aşamasında "deadlock" (kilitlenme) yaratıyordu. V9'da aynı başarıyı çok daha stabil sağlayan standart **SMOTE (k_neighbors=5)** yöntemine geri dönüldü.

---

## 🛠 Model Mimarisi

* **Temel Katman (Base Learners):**
  * **XGBoost:** (GPU - cuda/hist) V7'den gelen 700-trial Optuna parametreleriyle sabitlendi.
  * **LightGBM:** (GPU) V5'ten gelen optimum hiperparametrelerle.
  * **CatBoost:** (GPU) V7'den gelen 500-trial Optuna parametreleriyle.
* **Meta Model:** `LogisticRegression(C=0.1)` kullanılarak Stacking Ensemble (Yığınsal Topluluk) yapısı kuruldu.
* **Karar Eşiği (Threshold):** `0.50` yerine, maliyet-duyarlı optimizasyonla `0.20` olarak ayarlandı.

## 📁 Dosya Yapısı

* `train.py`: V9 modelinin GPU üzerinde Custom CV ve SMOTE ile baştan sona eğitildiği ana dosya.
* `v9_optuna_results.json`: Optuna tarafından bulunmuş ve modelin direkt kullandığı en iyi hiperparametreler.
* `README.md`: Bu doküman.
