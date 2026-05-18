# V3 

> Üçüncü versiyon — Biyokimyasal özellik mühendisliği (AA_1 / AA_2), Optuna ile hiperparametre optimizasyonu, SHAP hatasının giderilmesi ve model sadeleştirme.

---

## Özet

V3 aşamasında, V2'de kurulan LightGBM + XGBoost Soft-Voting Ensemble yapısı geliştirilmiştir.
Bu versiyonda amaç, veri setinde bulunan amino asit özelliklerinin (AA_1 ve AA_2) biyokimyasal olarak anlamlandırılarak modele verilmesi ve hiperparametre optimizasyonudur.

Özellikle:
- AA_1 ve AA_2 özelliklerinden polarite, hidropati, moleküler ağırlık ve elektriksel yük çıkarımı,
- Optuna kütüphanesi ile hiperparametre arayışı,
- Missing Indicator kullanımından kaynaklanan SHAP özelliğindeki mapping hatasının düzeltilmesi,
- SHAP değerlerine göre Top-100 özellik seçimi ile modelin sadeleştirilmesi,
- PAH alt grubu için özel performans artırımı

üzerine deneyler yapılmıştır.

V3 sonucunda hem F1 hem MCC değerlerinde V2'nin skorları (F1: 0.8903, MCC: 0.5301) başarıyla aşılmıştır.

---

## EDA Tabanlı Yeni Bulgular

V3 analizlerinden elde edilen biyokimyasal bulgular doğrudan modele uygulanmıştır:

- Patojenik varyantlarda amino asit mutasyonlarının (hidropati, ağırlık ve yük farklılıkları) benign varyantlara kıyasla istatistiksel olarak daha şiddetli olduğu kanıtlandı.
- V2'de eklenen "Missing Indicator" değerlerinin hedefe doğrudan %30-34 oranlarında korele olduğu görüldü (örn: `AL_4_missing`).
- `AA_1` ve `AA_2`'nin tek başına kategorik değişken olarak bırakılması yerine sayısal biyokimyasal farklara dönüştürülmesi varyans açıklayıcılığını artırdı.

---

## Yöntem

### Model
- **LightGBM + XGBoost Soft-Voting Ensemble**
- LightGBM için Optuna ile hiperparametre arayışı yapılmıştır.
- Özellik sayısı Top-100'e düşürülerek karmaşıklık azaltılmıştır.

### Hiperparametreler (Optuna Sonrası)
```python
LightGBM: n_estimators=800, lr=0.024, max_depth=9,
          num_leaves=55, subsample=0.66, colsample_bytree=0.62,
          reg_alpha=0.1, reg_lambda=1.0

XGBoost:  n_estimators=800, lr=0.05, max_depth=5,
          subsample=0.8, colsample_bytree=0.8,
          reg_alpha=0.1, reg_lambda=1.0
```

---

## Veri Ön İşleme Pipeline

| Adım | Yöntem | Gerekçe |
|---|---|---|
| Amino Asit Özellikleri | AA_1/AA_2'den biyokimyasal sözlük eşleştirmesi | Orijinal harflerin arkasındaki biyolojik anlamı modele kazandırmak |
| Fark Değişkenleri | `hydro_diff`, `weight_diff`, `charge_change` vb. | Mutasyonun yaratacağı fiziksel etkiyi sayısal ifade etmek |
| Yüksek eksikli sütunlar | `%80+` missing sütunlar kaldırıldı | Gürültüyü azaltmak |
| Sayısal eksik değer | Median Imputation + Missing Indicator | Biyokimyasal eksikleri de robust şekilde doldurmak |
| Özellik Seçimi | SHAP değerlerine göre Top-100 | Modeli en önemli değişkenlere odaklamak |

---

## MASTER Veri Seti — Sonuçlar

Optuna ve Biyokimyasal özellik mühendisliği sonrası MASTER veri seti üzerindeki Cross-Validation (5-Fold) skorları:

| Metrik | Ortalama | V2'ye Göre Değişim |
|---|---|---|
| **F1 Skoru** | **0.8920** | Artış |
| **MCC** | **0.5367** | Artış |

---

## Top-100 Model Performansı (Sadeleştirilmiş Model)

SHAP hatası giderildikten sonra en çok katkı sağlayan 100 özellik ile model yeniden eğitildiğinde:

| Metrik | Ortalama |
|---|---|
| **F1 Skoru** | **0.8929** |
| **MCC** | **0.5401** |

> Bu sayede çok daha az özellikle daha yüksek genelleme başarısı elde edilmiştir.

## Confusion Matrix (CV üzerinden, Eşik=0.45)

```text
              Tahmin
              Benign  Patojenik
Gerçek Benign   412      370
Gerçek Patojen  109     2040
```

- **FN = 109** (Klinik risk taşıyan kaçırılmış hastalar)
- **FP = 370** (Gereksiz alarm verilen sağlıklı kişiler)

> **Kazanım:** V2'de 120 olan FN (Yanlış Negatif) sayısı, V3'teki yeni özellikler ve Optuna ile **109'a düşürülmüştür**. Klinik açıdan bu çok değerli bir gelişmedir.

---

## Karar Eşiği Analizi

V3 modelinde karar eşiği (threshold) analizi yapılmış ve en yüksek F1/MCC dengesini veren noktanın **0.45** olduğu saptanmıştır. Yeni eklenen biyokimyasal özellikler sayesinde modelin güven aralığı daha keskinleşmiş ve optimal eşik 0.40'tan 0.45'e yükselmiştir.

---

## Alt Grup Sonuçları (PAH Odaklı)

PAH grubu (V2'de MCC: 0.5138) için özel analizler yapılmıştır:

| Grup | V2 MCC | V3 (Geliştirilmiş MASTER Model) MCC | V3 (Özel SMOTE PAH Modeli) MCC |
|---|---|---|---|
| **PAH** | 0.5138 | **0.5437** | 0.4195 |

### Yorumlar
- Geliştirilen MASTER modelin PAH tahmin kapasitesi belirgin şekilde artmıştır.
- PAH grubuna özel uygulanan SMOTE sentetik veri üretiminin overfit yarattığı ve performansı (0.4195) düşürdüğü görülmüştür. Doğal haliyle güçlü bir ensemble model kullanmak PAH için en iyi sonucu vermiştir.

---

## V1 ve V2 ile Karşılaştırma

| Metrik | V1 | V2 | V3 (Top-100) |
|---|---|---|---|
| MASTER F1 | 0.8711 | 0.8903 | **0.8929** |
| MASTER MCC | 0.5054 | 0.5301 | **0.5401** |
| PAH MCC | 0.4542 | 0.5138 | **0.5437** |

### Kazanımlar
- F1 ve MCC'de yeni rekorlar
- Amino asit özelliklerinin başarıyla sayısallaştırılması
- SHAP hatasının tamamen giderilmesi
- 600 özellikten 100 özelliğe inen daha verimli model mimarisi

---

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `analiz_dataset.py` | Veri seti dağılımları ve temel boyut incelemesi |
| `eda.py` | Biyokimyasal özellik korelasyonları ve eksik değer analizi |
| `train.py` | Optuna, Biyokimyasal Extraction ve SHAP Top-100 Model Eğitimi |
| `README.MD` | V3 sonuçları |

---

## GitHub'a Pushlandı mı?

- [x] Evet — V3 kodları ve sonuçları başarıyla eklendi.

### Commit mesajı

```text
[V3] Biochem Features + Optuna + SHAP Top-100 — MASTER F1:0.8929, MCC:0.5401
```
