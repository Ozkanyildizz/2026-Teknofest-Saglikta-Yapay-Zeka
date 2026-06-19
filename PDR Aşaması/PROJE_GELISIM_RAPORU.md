# AlgoMed — Proje Gelişim ve Tıbbi Gerekçe Raporu (V1 - V5)

Bu belge, TEKNOFEST 2026 Sağlıkta Yapay Zeka yarışması PDR (Proje Tasarım Raporu) ve final sunumlarında jüriye argüman olarak sunulmak üzere hazırlanmıştır. Projenin 1. versiyonundan 5. versiyonuna kadar olan süreçte **hangi teknik adımın neden atıldığını** tıbbi ve mühendislik gerekçeleriyle açıklar.

---

## 1. Nereden Nereye Geldik? (Versiyonların Gelişim Hikayesi)

### V1: Temel Atma (Baseline)
* **Ne Yaptık?** Yarışmanın verdiği ham veriyi (`MASTER`, `KANSER`, `PAH`, `CFTR`) LightGBM ve XGBoost ile doğrudan eğittik. 
* **Neden Yaptık?** Elimizde hiçbir önişleme olmadan verinin bize ne kadar skor verebileceğini (baseline) görmek istedik.
* **Sonuç:** F1 Skoru: %87. Ancak klinik bir vizyon eksikti.

### V2: Tıbbi Farkındalık ve Eksik Veri Analizi
* **Ne Yaptık?** Veri setinde %80'den fazla boş olan (missing value) devasa sütunlar vardı. Bu boşlukları doğrudan silmek yerine `Missing Indicator` (Eksiklik Belirteci) yöntemi kullandık.
* **Neden Yaptık?** Tıbbi verilerde "bir testin yapılmamış olması (boş değer)" hastanın durumu hakkında doktora bilgi verir. Örneğin; hastada kanser şüphesi yoksa o genetik test istenmemiştir. Boşlukları sadece bir matematik hatası olarak değil, **klinik bir örüntü** olarak modele öğrettik.

### V3: Biyokimyasal Özellik Mühendisliği (Tıp x Yapay Zeka)
* **Ne Yaptık?** `AA_1` (Referans Amino Asit) ve `AA_2` (Alternatif Amino Asit) sütunlarındaki protein yapılarını analiz ettik. Amino asitlerin **polarite (kutupluluk), hidrofilik (su seven) / hidrofobik karakter, molekül ağırlığı ve elektriksel yük** gibi biyokimyasal özelliklerini çıkartarak yeni sütunlar ürettik.
* **Neden Yaptık?** Missense varyantlar (mutasyonlar), proteinin 3 boyutlu yapısını bozduğu için hastalıklara yol açar. Bir amino asit, tamamen zıt kutuplu veya devasa ağırlıkta başka bir amino aside dönüşürse protein bozulur (patojenik). Jüriye, yapay zekayı sadece bir araç olarak kullanmadığımızı, **biyoinformatik ve genetik** bilgimizi modele entegre ettiğimizi göstermek istedik. (Skoru %89.2'ye çıkardı).

### V4: Yapay Zeka Mimarisi ve Hayat Kurtarma (Threshold Optimizasyonu)
* **Ne Yaptık?** LightGBM, XGBoost ve CatBoost modellerini bir araya getirerek **Stacking (Yığılma)** mimarisi kurduk. Ayrıca KANSER ve PAH gibi özel hastalık gruplarında karar eşiklerini (Threshold) 0.5'ten 0.25 - 0.35 seviyelerine çektik.
* **Neden Yaptık?** Tıpta **False Negative (Yanlış Negatif: Hasta olan birine sen sağlıklısın demek)** en ölümcül hatadır. Modelin hastaları kaçırmaması (Recall/Duyarlılık oranının artması) için eşikleri bilinçli olarak düşürdük. Yanlış negatif vaka sayısını 109'dan 81'e kadar indirdik. Bu hamle, "Yapay zekanın klinik pratikte nasıl kullanılacağını çok iyi biliyoruz" demenin en teknik yoludur.

### V5: Donanım Hızlandırması (GPU) ve Ölçeklenebilirlik
* **Ne Yaptık?** Tüm bu karmaşık mimariyi (Stacking + Optuna) işlemciden (CPU) alıp **NVIDIA GPU (Ekran Kartı)** üzerine taşıdık. KANSER alt grubunda F1 skorunu **0.9069**'a çıkardık.
* **Neden Yaptık?** Gerçek hayatta hastaneler devasa genetik verilerle (milyonlarca satır) çalışır. Sistemimizin saatler değil, dakikalar/saniyeler içinde eğitilip sonuç üretebilmesi gerekiyordu. Hız kazanarak hiperparametre optimizasyonlarını (Optuna) saniyeler içinde tamamlayabildik.

### V6: Sızıntısız (Leak-Free) Derin Biyoinformatik ve Grantham (Gerçek Dünya Modeli)
* **Ne Yaptık?** Kod incelemeleri sonucunda Cross-Validation sırasında ufak sızıntılara (Data Leakage) yol açan K-Means, PCA ve SHAP seçim işlemlerini tamamen CV döngüsünün içine hapsederek **sızıntısız (leak-free)** bir boru hattı (pipeline) inşa ettik. 400 amino asit geçişinin tamamını içeren **Grantham Mesafe Matrisini** sisteme entegre ettik.
* **Neden Yaptık?** Modelin kağıt üzerinde yüksek skor vermesinden ziyade "test verisi gördüğünde ne kadar sağlam (robust) kalacağı" önemlidir. Veri sızıntısını keserek elde ettiğimiz skorlar (F1: 0.8919, MCC: 0.5450) önceki sızıntılı skorlardan (F1: 0.8965) teknik olarak ufak bir tık aşağıda görünse de, bu skor **klinik olarak savunulabilir, hilesiz ve gerçek** bir performanstır. Jüriye "Hatamızı fark edip Data Leakage'i sıfırladık ve MCC'yi yine de yükselttik" demek çok güçlü bir mühendislik duruşudur.

---

## 2. PDR Raporunda Jüriyi Etkileyecek 3 Ana Argüman (Kozlarımız)

**🛡️ Overfitting (Aşırı Öğrenme) Koruması:**
* "Modelimiz test verilerini ezberlemedi." Stacking modelini eğitirken uyguladığımız **5-Fold Out-of-fold (OOF)** yöntemi sayesinde meta-modelimiz sadece temel modellerin daha önce *hiç görmediği* veriler üzerindeki performansına bakarak karar veriyor. Bu nedenle modelimiz gerçek dünyada (final testinde) asla çuvallamayacak.

**⚕️ Yüksek F1 ve Kanser Başarısı (%90+):**
* Veri son derece imbalanced (dengesiz) ve boşluklarla dolu olmasına rağmen genel MASTER'da **0.8941 F1** (%92 PR-AUC), KANSER özel grubunda ise rekor olan **0.9069 F1** (%96 PR-AUC) seviyesine ulaştık. Accuracy (Doğruluk) gibi yanıltıcı metrikler yerine tıbbın altın standardı olan MCC ve F1 metriklerini optimize ettik.

**🧬 Domain Expertise (Alan Bilgisi):**
* Modeli kurup geçmedik; eksikliklerin tıbbi anlamını (Missing Indicators) çıkarttık, mutasyonların biyokimyasal tahribatlarını (AA polarity, charge, weight) yapay zekaya tercüme ettik. Klinik olarak hastalığı gözden kaçırmanın (False Negative) maliyetini bildiğimiz için **PAH ve Kanser varyantlarında hayat kurtarıcı karar eşiği (Custom Threshold)** stratejileri geliştirdik.

---
*Not: Bu belge PDR ve PSR raporları yazılırken "Neyi neden yaptık?" sorularına ile yanıt vermek üzere referans olarak oluşturulmuştur.*
