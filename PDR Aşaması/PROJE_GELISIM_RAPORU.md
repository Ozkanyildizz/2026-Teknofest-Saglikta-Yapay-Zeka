# AlgoMed — Proje Gelişim ve Tıbbi Gerekçe Raporu (V1 - V9)

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

### V7: Sınıf Dengeleme (SMOTE) ve GPU Optuna Zirvesi
* **Ne Yaptık?** Sızdırmaz CV mimarisinin üzerine SMOTE sınıf dengeleme algoritmalarını tamamen sızıntısız şekilde (sadece eğitim setine) entegre ettik. Optuna ile yüzlerce denemelik hiperparametre optimizasyonunu GPU üzerinde gerçekleştirdik.
* **Neden Yaptık?** Veri setindeki dengesizliği (imbalance) sentetik verilerle çözerken sızıntı yapmamak çok zordur. Sentetik verileri sadece model eğitimine verip, doğrulama (validation) setini tamamen saf (gerçek dünyadan kopmamış) bırakarak modelin dayanıklılığını test ettik.

### V8: Yanlış Negatif (FN) Otopsisi ve Hata Karakteristiği
* **Ne Yaptık?** Modelin kaçırdığı (Yanlış Negatif / FN) vakaların neden kaçırıldığını tek tek inceledik. Amino asit verisinin eksik olmasının ve "EK_7" biyolojik değerinin çok düşük olmasının (EK_7 < 3) modeli yanılttığını (hastayı eve gönderdiğini) bulduk.
* **Neden Yaptık?** Amacımız skoru sadece kağıt üzerinde yükseltmekten çok tıbbi bir gerçekliği çözmekti. En ölümcül hata türü olan FN'in (False Negative) karakteristiğini çıkartmak için "hata yapmanın örüntüsünü" teşhis ettik.

### V9(L): Nihai Hibrit Karar Destek Sistemi ve Klinik Stres Testi Başarısı (FINAL MODEL)
* **Ne Yaptık?** Grantham mesafe matrisine ek olarak BLOSUM62 evrimsel şiddet matrisini ve popülasyon (CAT_) etkileşim özelliklerini sızıntısız (leak-free) pipeline içerisinde birleştirdik. Karar eşiğini tıbbi güvenlik için 0.25'e sabitledik.
* **Neden Yaptık?** Biyolojik evrimin varyantlar üzerindeki etkisini matematiksel olarak (Grantham+BLOSUM62) modelledik. Test setinin %86 oranında asimetrik (Benign ağırlıklı) olacağı ("Klinik Stres Testi") bilindiğinden, modelin ayrıştırma gücünü ölçen MCC metriğine odaklandık ve V9 ile projenin rekor MCC değerine (0.5594) ulaştık. 0.25 klinik eşik stratejisiyle de en tehlikeli hata olan Yanlış Negatif (FN) oranını en aza indirdik. V9, sadece algoritmik bir model değil, tıp bilimiyle veri biliminin kusursuz bir hibritidir.
