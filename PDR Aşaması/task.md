# TEKNOFEST 2026 - PDR Aşaması Görev Dağılımı ve V6 Stratejisi (Son 13 Gün)

Önümüzde PDR (Proje Tasarım Raporu) teslimi için kritik 13 gün var. Yarışma kurallarını (veri sızıntısı kısıtlamaları) ihlal etmeden, elimizdeki donanımları ve V5'te kurduğumuz tıbbi/mühendislik vizyonunu en üst seviyeye çıkarmak için takımın nihai görev dağılımı aşağıdadır.

---

## 💻 1. Furkan — Derin Öğrenme (Deep Learning) ve Stacking [AĞIR GPU]
Furkan'ın bilgisayarı güçlü olduğu için makine öğreniminin (XGBoost/CatBoost) dışına çıkıp Yapay Sinir Ağlarını deneyecek.
- [ ] Tablo verilerinde son teknoloji olan **TabNet** modelini GPU üzerinde kurup eğitmek.
- [ ] PyTorch/TensorFlow kullanarak **1D-CNN / Multi-Layer Perceptron (MLP)** ağları tasarlamak.
- [ ] 🚨 **KRİTİK KURAL:** Derin öğrenme modelleri çok hızlı overfit olacağı için bu modelleri *sadece* büyük olan MASTER (2900+ satır) veri setinde test etmek. CFTR ve KANSER gibi küçük panellerde bu modellerle zaman kaybetmemek.
- [ ] Modeller başarılı olursa (F1 > 0.88), bunları V5'teki Stacking mimarisine 4. Base Model olarak entegre etmek.

## 💻 2. Funda — Sızdırmaz Boru Hattı (Pipeline) ve Optuna [AĞIR GPU]
Funda'nın bilgisayarı V5 mimarisinin sınırlarını zorlamak ve sınıf dengesizliğini çözmek için kullanılacak.
- [ ] V5'te kurduğumuz XGBoost ve CatBoost mimarisini **Optuna ile 500 ila 1000 deneme (trial)** yaparak sabaha kadar optimizasyona bırakmak (L1/L2 regülarizasyon, max_depth, scale_pos_weight odaklı).
- [ ] Imbalanced (Dengesiz) veri setini çözmek için **SMOTE, ADASYN veya BorderlineSMOTE** algoritmalarını test etmek.
- [ ] 🚨 **KRİTİK KURAL:** Veri sızıntısını (Data Leakage) önlemek için SMOTE işlemini kesinlikle tüm veriye UYGULAMAMAK. SMOTE, `imblearn.pipeline` kullanılarak *sadece* Cross-Validation döngüsünün TRAIN (Eğitim) katmanlarında çalışacak şekilde ayarlanacak. Validation katmanı orijinal kalacak.

## 🧬 3. Özkan — Veri Bükme (Feature Engineering) ve Unsupervised Learning
Dış veri kullanımı (ClinVar/gnomAD) yarışma kuralları gereği riskli olduğundan, tamamen elimizdeki kapalı veriyi matematiksel olarak zenginleştirmeye odaklanacak.
- [x] V3'te yapılan Amino Asit biyokimyasal çevirilerini derinleştirmek (Örneğin: Amino asidin esnekliği, hacmi, hidrofobik indeks skorları).
- [x] Gözetimsiz Öğrenme (Unsupervised Learning) teknikleri kullanmak: K-Means ile hastaları kümeleyip bu küme numaralarını (Cluster ID) modele yeni bir sütun olarak vermek.
- [x] Sayısal (AL_ ve EK_) sütunlar arasında matematiksel etkileşimler (Feature Interactions) yaratmak (Örn: EK_7 x AL_327) veya PCA (Temel Bileşen Analizi) ile yeni boyutlar üretmek.
> **Not:**  tüm bu özellikler "V6" modeli altında rekor kırarak GitHub'a eklenmiştir (F1: 0.8965, MCC: 0.5563).

## 🔍 4. Cansu — "False Negative" Hata Analizi ve Klinik Yorumlama
Sistemin gözden kaçırdığı patojenik (hasta) vakalara odaklanarak modelin kör noktalarını tespit edecek.
- [ ] V5 modelinin "Sağlıklı (Benign)" dediği ama aslında "Patojenik" olan o 81 varyantlık listeyi çıkarıp satır satır incelemek.
- [ ] Bu 81 hatanın ortak bir örüntüsü var mı? (Örneğin; hepsi PAH grubunda mı? Belli bir amino asit dönüşümünde mi takılıyorlar? EK_7 skorları hep düşük mü?)
- [ ] Bulduğu tıbbi/istatistiksel örüntüleri Özkan'a bildirerek, o hataları çözecek (ağırlıklandıracak) özel sütunlar üretilmesini sağlamak.

## 📝 5. Berra — Akademik PDR Yazımı ve Açıklanabilir YZ (XAI)
13 gün sonra teslim edilecek resmi raporun editoryal sorumluluğunu ve jürinin "Kara Kutu" eleştirilerini savuşturma görevini üstlenecek.
- [ ] `Rapor/2026_PDR_Şablon_*.docx` dosyasını açıp yarışma formatına göre başlıkları oluşturmak.
- [ ] V1'den V5'e kadar olan "Tıbbi Gerekçe ve Gelişim Hikayesini" akademik bir Türkçeyle Giriş, Yöntem ve Bulgular kısımlarına yedirmek.
- [ ] **Doktor Ekranı / SHAP Yorumlaması:** Modelin en iyi tahmin yaptığı birkaç örnek üzerinden SHAP grafikleri çıkarıp raporun sonuna eklemek. "Modelimiz şu varyanta patojenik dedi, çünkü EK_7 skoru yüksekti ve polaritesi bozulmuştu" şeklinde tıbbi açıklamalar yazmak.
- [ ] Karar eşiği (Threshold) optimizasyon tablolarını ve Confusion Matrix görsellerini şık bir biçimde rapora yerleştirmek.

---
**🏆 STRATEJİ VE ZAMAN ÇİZELGESİ:** Önümüzdeki **7 gün** herkes kendi görevinde izole olarak çalışacak ve denemelerini yapacak. Son **6 gün** kala kimin modeli (Furkan/Funda) veya verisi (Özkan/Cansu) daha iyi F1/MCC skoru getirdiyse, her şeyi "V6_Final" dalı (branch) altında birleştirip raporu sonlandıracağız.