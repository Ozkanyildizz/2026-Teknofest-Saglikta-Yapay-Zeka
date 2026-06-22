# PDR Raporu — Hazırlanan Görseller ve Açıklamaları

Aşağıdaki 6 grafik rapordaki eksik görsel gereksinimlerini karşılamaktadır.  
Tüm dosyalar şu klasöre kaydedildi: `PDR Aşaması/Rapor/gorseller/`

---

## Şekil 1 — Versiyon Bazlı Model Karşılaştırması

![Şekil 1](file:///C:/Users/ozkan/.gemini/antigravity/brain/2df6adbd-085e-4591-99b3-964c5865b7b5/sekil1_versiyon_karsilastirma.png)

**Açıklama:** Geliştirilen V1–V7 modellerinin MASTER veri seti üzerinde elde ettiği F1 Skoru ve Matthews Korelasyon Katsayısı (MCC) değerleri karşılaştırmalı olarak sunulmaktadır. V3'ten V7'ye kadar her versiyonda biyoinformatik özellik mühendisliği, Optuna hiperparametre optimizasyonu ve sızıntısız (leak-free) pipeline tasarımı ile kademeli iyileşme sağlanmıştır. V7 modeli F1=0.8959 ve MCC=0.5532 ile en yüksek genel performansı elde etmiştir.

---

## Şekil 2 — Alt Grup Bazında Performans Analizi

![Şekil 2](file:///C:/Users/ozkan/.gemini/antigravity/brain/2df6adbd-085e-4591-99b3-964c5865b7b5/sekil2_altgrup_performans.png)

**Açıklama:** V7 modelinin dört farklı veri alt grubu üzerindeki (MASTER, KANSER, PAH, CFTR) F1 Skoru, MCC ve PR-AUC değerleri gösterilmektedir. CFTR grubunda en yüksek MCC (0.765) ve PR-AUC (0.988) elde edilmiştir. PAH grubunun düşük MCC değeri (0.559), bu gruptaki yüksek sınıf dengesizliğinden kaynaklanmaktadır. Her alt grup için ayrı karar eşiği (threshold) optimizasyonu uygulanmıştır.

---

## Şekil 3 — Karmaşıklık Matrisi (Confusion Matrix)

![Şekil 3](file:///C:/Users/ozkan/.gemini/antigravity/brain/2df6adbd-085e-4591-99b3-964c5865b7b5/sekil3_karisiklik_matrisi.png)

**Açıklama:** V7 modelinin MASTER veri seti üzerinde 5-Katlı Çapraz Doğrulama (5-Fold CV) sonucu elde edilen karmaşıklık matrisi sunulmaktadır. Klinik açıdan en kritik hata türü olan Yanlış Negatif (FN=101), yani aslında patojenik olan varyantların benign olarak sınıflandırılması, kırmızı kesik çizgi ile vurgulanmıştır. Düşük karar eşiği (0.21) kullanılması sayesinde Recall oranı %97.2'ye yükseltilmiş ve FN sayısı minimize edilmiştir.

---

## Şekil 4 — Karar Eşiği (Threshold) Analizi

![Şekil 4](file:///C:/Users/ozkan/.gemini/antigravity/brain/2df6adbd-085e-4591-99b3-964c5865b7b5/sekil4_esik_analizi.png)

**Açıklama:** Farklı karar eşiği değerlerinde V7 modelinin F1 Skoru, Kesinlik (Precision), Duyarlılık (Recall) ve MCC değerlerinin değişimi gösterilmektedir. Yüksek eşik değerlerinde kesinlik artarken duyarlılık düşmekte; düşük eşik değerlerinde ise duyarlılık artarken daha fazla Yanlış Pozitif üretilmektedir. MCC maksimizasyonu kriteri ile seçilen optimum eşik 0.21 olarak belirlenmiş; bu değer özellikle klinik uygulamalarda hasta kaçırma riskini minimize etmek açısından tercih edilmiştir.

---

## Şekil 5 — Alt Grup Radar Grafiği

![Şekil 5](file:///C:/Users/ozkan/.gemini/antigravity/brain/2df6adbd-085e-4591-99b3-964c5865b7b5/sekil5_radar_altgrup.png)

**Açıklama:** V7 modelinin MASTER, KANSER, PAH ve CFTR alt gruplarındaki çok boyutlu performansı (F1, MCC, PR-AUC, ROC-AUC, Recall) radar grafiği ile sunulmaktadır. CFTR grubunun en geniş alana sahip olması bu gruptaki model tutarlılığını gösterirken, PAH grubunun görece daha küçük alanı ROC-AUC'daki düşüklüğü yansıtmaktadır. Bu görsel, modelin genel güçlü yönlerini ve alt grup bazındaki kör noktalarını bütüncül biçimde ortaya koymaktadır.

---

## Şekil 6 — SHAP Özellik Önem Analizi

![Şekil 6](file:///C:/Users/ozkan/.gemini/antigravity/brain/2df6adbd-085e-4591-99b3-964c5865b7b5/sekil6_shap_onem.png)

**Açıklama:** V7 modelinde en yüksek karar katkısına sahip 15 özellik SHAP (SHapley Additive exPlanations) yöntemiyle sıralanmaktadır. Yeşil çubuklar amino asit biyokimyasal özelliklerini (AA_), mavi çubuklar destekleyici biyolojik ölçümleri (EK_), turuncu çubuklar ise şifreli genomik özellikleri (AL_) temsil etmektedir. Grantham mesafesi (`AA_grantham_dist`) ve hidrofobisite farkı (`AA_hydro_diff`) en kritik özellikler olarak öne çıkmıştır. Bu bulgu, biyokimyasal özellik mühendisliğinin model başarısına kritik katkı sağladığını kanıtlamaktadır.

---

> [!NOTE]
> Şekil açıklamaları rapor içindeki tablo altı açıklamalar olarak aynen kullanılabilir.
> Tüm görseller 200 DPI çözünürlüklü PNG formatında `PDR Aşaması/Rapor/gorseller/` klasörüne kaydedilmiştir.
