V8

## Proje Özeti

AlgoMed V6, genomik varyantların patojenik (Pathogenic) veya benign olarak sınıflandırılması amacıyla geliştirilmiş bir makine öğrenmesi sistemidir.

Model, yarışma kapsamında sağlanan eğitim verileri üzerinde eğitilmiş ve özellikle False Negative (FN) hatalarını azaltmaya odaklanacak şekilde geliştirilmiştir.

V6 sürümü önceki versiyonlara göre:

* Amino asit biyokimyasal özellikleri
* EK değişkenleri arası etkileşim terimleri
* Eksiklik (missingness) örüntüsü özellikleri
* False Negative risk göstergeleri
* Stacking ensemble mimarisi

içermektedir.

---

## Kullanılan Modeller

Stacking Ensemble:

1. LightGBM
2. XGBoost
3. CatBoost
4. Logistic Regression (Meta Learner)

Bu yapı sayesinde farklı ağaç tabanlı modellerin güçlü yönleri birleştirilmiştir.

---

## Özellik Mühendisliği

### 1. Amino Asit Biyokimyasal Özellikleri

AA_1 ve AA_2 sütunlarından:

* Polarite
* Yük (Charge)
* Hidropati (Hydropathy)
* Moleküler ağırlık

özellikleri çıkarılmıştır.

Ek olarak:

* Hidropati farkı
* Ağırlık farkı
* Yük değişimi
* Polarite değişimi

hesaplanmıştır.

---

### 2. EK Etkileşim Terimleri

Tüm EK sütunları arasında ikili çarpım etkileşimleri oluşturulmuştur.

Örnek:

EK_3 × EK_7

EK_4 × EK_9

Bu sayede doğrusal olmayan ilişkilerin modele aktarılması amaçlanmıştır.

---

### 3. Missing Pattern Özellikleri

Verideki eksiklik örüntülerinin önemli sinyal taşıdığı gözlemlenmiştir.

Bu nedenle:

* missing_count_AL
* missing_ratio_AL
* missing_ratio_AL_log
* missing_count_EK
* missing_ratio_EK
* missing_count_ALL
* EK9_x_miss_AL

özellikleri eklenmiştir.

---

### 4. False Negative Risk Özellikleri

FN analizi sonucunda bazı ortak örüntüler belirlenmiştir.

Eklenen özellikler:

* EK7_low_flag
* AA_missing_flag
* FN_risk_flag

Bu özellikler modelin patojenik varyantları kaçırma riskini azaltmak amacıyla tasarlanmıştır.

---

## Ön İşleme

Sayısal değişkenler:

* Median imputasyon
* Missing indicator üretimi
* RobustScaler
* VarianceThreshold

Kategorik değişkenler:

* Sabit değer ile doldurma
* Ordinal Encoding

---

## Değerlendirme

Ana değerlendirme yöntemi:

* 5-Fold Stratified Cross Validation

Kullanılan metrikler:

* F1 Score
* MCC (Matthews Correlation Coefficient)
* PR-AUC
* ROC-AUC
* Balanced Accuracy

---

## Alt Grup Analizi

Model aşağıdaki alt veri setlerinde ayrıca değerlendirilmiştir:

* KANSER
* PAH
* CFTR

PAH alt grubunda ayrıca özel eşik optimizasyonu uygulanmıştır.

---

## False Negative Analizi

V6 sürümünde False Negative örnekleri ayrı olarak incelenmiştir.

Analiz edilen noktalar:

* EK_7 dağılımı
* Missing Ratio dağılımı
* Amino asit dönüşümleri
* AA eksiklik oranları
* Alt grup dağılımları

FN kayıtları ayrıca:

false_negatives_v5.csv

dosyasına kaydedilmektedir.

---



