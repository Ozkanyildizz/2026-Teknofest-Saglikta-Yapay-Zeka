# PDR Raporu — Kapsamlı Değişiklik Listesi

> Tüm değişiklikler v9_outputs.txt, README'ler ve Şartname V2'ye göre doğrulanmıştır.

---

## 📍 V9 Seçim Gerekçesi — NEREYE EKLENECEK?

**Yer:** Bölüm 3.1 — "Genel sonuç:" paragrafının hemen altına

**Mevcut metin:**
> *"Genel sonuç: V9(L) modeli, en yüksek MCC değerini ve klinik risk odaklı en dengeli performansı sağlamıştır."*

**Yeni metin (bunu ekle):**
> *"Genel sonuç: Sızıntısız (leak-free) koşulda değerlendirildiğinde V7, F1=0.8959 ile en yüksek F1 skoruna ulaşmış; ancak MCC=0.5532 düzeyinde kalmıştır. V9'da entegre edilen Grantham+BLOSUM62 evrimsel matrisleri ve CAT_ etkileşim özellikleri sayesinde MCC proje boyunca ulaşılan en yüksek değere (0.5594) yükselmiş, PR-AUC ise 0.9268 ile V7 seviyesinde korunmuştur. Şartnamede tanımlanan asimetrik test yapısında (Klinik Stres Testi: MASTER test seti %86 Benign) doğru pozitif ayrıştırma kapasitesini en güvenilir biçimde yansıtan MCC metriğindeki bu gelişme, V9'un final model olarak seçilmesinin temel gerekçesini oluşturmaktadır."*

---

## 🔴 ZORUNLU DEĞİŞİKLİKLER (Hata veya Çelişki İçeriyor)

---

### 1. Bölüm 2.3 — Şekil referansı yanlış
**Sayfa 5, satır:** *"Şekil 1'de SHAP incelenmektedir [6]."*

❌ Şekil 1 = Pipeline şeması. SHAP Şekil 2.

✏️ Değiştir:
> *"Şekil 1'de SHAP incelenmektedir"* → **"Şekil 2'de SHAP incelenmektedir"**

---

### 2. Bölüm 2.1 — Tablo 2'de yazım hatası
**Satır:** *"(1:patejonik, 0:benign)"*

❌ "patejonik" yanlış yazım

✏️ Değiştir: **"patojenik"**

---

### 3. Bölüm 3.1 — V1–V2 versiyonlar grafikte yok
**Satır:** *"Geliştirilen modellerin (V3–V9) performansları"*

❌ V1 ve V2 de grafik/tabloda gösterilmeli çünkü "gelişim hikayesi" V1'den başlıyor.

✏️ Değiştir: **(V3–V9)** → **(V1–V9)**

---

### 4. Bölüm 3.3 — Klinik eşik 0.20 çıktıda yok
**Satır:** *"0.20–0.25 eşiğinde Recall ≥%92.7 sağlanmaktadır"*

❌ v9_outputs.txt'te eşik tablosu **0.25'ten** başlıyor. 0.20 değeri gösterilmiyor, %92.7 Recall aslında **eşik=0.25**'e ait.

✏️ Değiştir:
> *"0.20–0.25 eşiğinde Recall ≥%92.7"* → **"eşik=0.25'te Recall=0.9269 (%92.7)"**

Ve klinik eşik tartışmasını şöyle düzelt:
> *"Klinik güvenlik eşiği olarak 0.25 veya altı önerilmekte olup eşik=0.25'te Recall=0.9269 (%92.7) ile patojenik varyantların büyük çoğunluğu yakalanmaktadır."*

---

### 5. Bölüm 4.2 — Aynı 0.20 eşiği sorunu
**Satır:** *"klinik odaklı 0.20 eşiğinde Recall≥%92.7 ile hasta kaçırma riski minimize edilmektedir"*

✏️ Aynı düzeltme: **0.20** → **0.25**

---

### 6. Bölüm 3.2 — Radar grafiği metni (önceki konuşmadan)
**Mevcut:**
> *"Her alt grup için ayrı eşik optimizasyonu uygulanmıştır. MASTER, KANSER, PAH ve CFTR alt gruplarının F1, MCC, PR-AUC, ROC-AUC ve Recall boyutlarındaki performansı radar grafiğiyle sunulmaktadır. CFTR grubunun en geniş alana sahip olması..."*

✏️ Değiştir (önceki kararımıza göre):
> *"Her alt grup için ayrı eşik optimizasyonu uygulanmıştır. CFTR grubunda en yüksek model tutarlılığı elde edilirken (MCC=0.6561, PR-AUC=0.9878), PAH grubunun görece düşük MCC değeri (0.5399) bu gruptaki sınıf heterojenliğini yansıtmaktadır."*

---

### 7. Kaynakça [3] — Dergi adı yanlış
**Mevcut:** *"Bioinformatics, 2020"* ← Önceki seferden biliyoruz

✏️ Değiştir: **"Genome Medicine, 2020"**

---

### 8. Kaynakça [19] — Makale konu dışı
**Mevcut:** *"Z. Lin et al., ESMFold (protein yapısı)"*

✏️ Değiştir:
> *"N. Brandes et al., 'Genome-wide prediction of disease variant effects with a deep protein language model,' Nature Genetics, 2023. DOI: 10.1038/s41588-023-01465-0"*

---

## 🟡 GÜÇLÜ KISALTICI EKLEMELER (Skor Artırır)

---

### 9. Bölüm 1.1 — Şartname Klinik Stres Testi bağlantısı
**Nereye:** 1.1'in son cümlesinden sonra yeni paragraf

**Eklenecek metin:**
> *"Yarışma şartnamesinde 'Klinik Stres Testi' olarak tanımlanan asimetrik test yapısı (MASTER test seti: 500 Patojenik / 3000 Benign, %86 Benign oranı), gerçek klinik koşulları simüle etmektedir. Bu yapı, yüksek Recall (Duyarlılık) öncelikli dinamik eşik stratejisinin zorunluluğunu doğrudan ortaya koymaktadır."*

---

### 10. Bölüm 2.1 — Özellik kategorileri şartname uyumu
**Nereye:** Tablo 2'nin hemen altına

**Eklenecek:**
> *"Bu özellik kategorileri (AL_, EK_, CAT_, AA_), şartname tarafından tanımlanan varyant profili yapısıyla birebir örtüşmektedir."*

---

### 11. Bölüm 3.4 — FN gelişim tablosu ekle
**Nereye:** Confusion Matrix açıklamasından sonra

**Eklenecek tablo:**

| Versiyon | FN (Opt. Eşik) | Eşik | Yöntem |
|---|---|---|---|
| V3 | 109 | 0.45 | Biyokimyasal AA |
| V4 | 68 | 0.35 | Stacking + Eşik |
| V7 | 101 | 0.21 | ADASYN + Sızıntısız |
| V9 (0.56) | 243 | 0.56 | MCC Optimize |
| V9 (0.25) | ~%92.7 Recall | 0.25 | Klinik Güvenlik |

> *"V9'da MCC optimize eşiğinde FN=243 gözlemlenmiş; ancak klinik güvenlik eşiği (0.25) uygulandığında Recall %92.7'ye yükselmektedir. V4'teki düşük FN (68) değeri ise veri sızıntısı içerdiğinden klinik açıdan savunulamaz."*

---

### 12. Bölüm 4.1 — V7 ve V8 için sayısal veri ekle
**Mevcut:** *"V7–V8 ile hiperparametre optimizasyonu ve dengeleme stratejileriyle performans güçlendirilmiştir."*

✏️ Genişlet:
> *"V7'de sızıntısız ADASYN entegrasyonu ve 700/500 trial Optuna optimizasyonuyla F1=0.8959, MCC=0.5532 elde edilmiştir. V8'de ise FN otopsisi yapılarak EK_7 düşüklüğü ve AA eksikliğinin kaçırma riskini artırdığı tespit edilmiştir."*

---

### 13. Bölüm 4.5 — F1'in şartname ana metriği olduğunu belirt
**Nereye:** "Genel olarak AlgoMed..." paragrafının başına

**Eklenecek:**
> *"Yarışma şartnamesinin belirlediği temel değerlendirme metriği F1 skoru olup V9 modeli MASTER veri setinde F1=0.8836 ile rekabetçi bir konumdadır."*

---

## ✅ DOĞRU OLAN — DOKUNMA

| Bölüm | Neden Doğru |
|---|---|
| 1.2 tüm metin | FN ve dengesizlik anlatımı doğru |
| 2.2 Mimari Katmanları | Doğru teknik detay |
| 3.2 KANSER/CFTR/PAH sayıları | v9_outputs.txt ile eşleşiyor |
| 3.3 Eşik tablosu (0.25-0.65) | v9_outputs.txt ile eşleşiyor |
| 3.4 FN=243 (eşik=0.56) | v9_outputs.txt ile eşleşiyor |
| 4.5 F1=0.8836, MCC=0.5594, PR-AUC=0.9268 | v9_outputs.txt ile eşleşiyor |
| Kaynakça [20] AnnotateMissense DOI | Doğrulandı |
| Kaynakça [19] (yeni Brandes) | Önceki oturumda zaten düzeltilmiş |
| Kaynakça [3] (Genome Medicine) | Önceki oturumda zaten düzeltilmiş |

---

## 📋 ÖNCELİK SIRASI

| # | Değişiklik | Öncelik | Süre |
|---|---|---|---|
| 1 | V9 seçim gerekçesi → 3.1'e ekle | 🔴 Yüksek | 3 dk |
| 2 | "Şekil 1'de SHAP" → "Şekil 2" | 🔴 Hata | 30 sn |
| 3 | "patejonik" → "patojenik" | 🔴 Hata | 10 sn |
| 4 | "0.20 eşiği" → "0.25 eşiği" (2 yerde) | 🔴 Çelişki | 1 dk |
| 5 | Radar grafiği metni → yeni metin | 🔴 Yüksek | 1 dk |
| 6 | V3–V9 → V1–V9 | 🟡 Orta | 10 sn |
| 7 | F1'in şartname metriği olduğu → 4.5'e | 🟡 Orta | 1 dk |
| 8 | Klinik Stres Testi → 1.1'e | 🟡 Orta | 2 dk |
| 9 | V7 sayıları → 4.1'e | 🟡 Orta | 1 dk |
| 10 | FN gelişim tablosu → 3.4'e | 🟢 Bonus | 3 dk |
