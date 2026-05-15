# V2 — Cansu — Mayıs 2026

> İkinci versiyon — preprocessing geliştirmeleri, missing indicator yaklaşımı ve karar eşiği optimizasyonu.

---

## Özet

V2 aşamasında V1’de kurulan LightGBM + XGBoost Soft-Voting Ensemble yapısı korunmuştur.
Bu versiyonda amaç model mimarisini tamamen değiştirmek yerine veri ön işleme,
eksik değer yönetimi ve threshold optimizasyonu üzerinden performansı artırmaktır.

Özellikle:
- yüksek eksik değerli sütunların temizlenmesi,
- missing indicator yaklaşımı,
- `CAT_6` sütununun kaldırılması,
- karar eşiğinin hassas optimize edilmesi,
- estimator sayısının artırılması

üzerine deneyler yapılmıştır.

V2 sonucunda hem F1 hem MCC değerlerinde iyileşme sağlanmıştır.

---

## EDA Tabanlı Yeni Bulgular

V1 analizlerinden elde edilen bazı önemli bulgular V2’de doğrudan modele uygulanmıştır:

- `CAT_6` sütunu `%97+` eksik olduğu için kaldırıldı
- `%80+` eksik olan sayısal sütunlar temizlendi
- Eksik değerlerin yalnızca veri problemi değil, potansiyel biyolojik sinyal taşıyabileceği düşünüldü
- `EK_7`, `AA_1`, `AA_2` gibi sütunların SHAP analizinde baskın çıkması dikkat çekti
- Missing pattern’lerin modele ayrıca verilmesi gerektiği düşünüldü

---

## Yöntem

### Model

- **LightGBM + XGBoost Soft-Voting Ensemble**
- Her iki model eşit ağırlıklı kullanıldı
- `scale_pos_weight = 0.364`

---

## Veri Ön İşleme Pipeline

| Adım | Yöntem | Gerekçe |
|---|---|---|
| Yüksek eksikli sütunlar | `%80+` missing sütunlar kaldırıldı | Gürültüyü azaltmak |
| Sayısal eksik değer | Median Imputation | Aykırı değerlere dayanıklı |
| Missing Indicator | Eksik değer maskesi eklendi | Missing pattern bilgisini korumak |
| Ölçekleme | RobustScaler | Aykırı değerlere dayanıklı ölçekleme |
| Sıfır varyans | VarianceThreshold (1e-4) | Sabit sütunları temizlemek |
| Kategorik sütunlar | OrdinalEncoder | Tree-based modellerle uyum |
| `CAT_6` | Tamamen kaldırıldı | `%97+` eksik |

### Özellik Sayısı

| Aşama | Özellik |
|---|---|
| Başlangıç | 351 |
| V2 sonrası | 600 |

> Missing indicator eklendiği için feature sayısı arttı.

---

## Hiperparametreler

```python
LightGBM:
n_estimators=800
learning_rate=0.05
max_depth=6
num_leaves=31
subsample=0.8
colsample_bytree=0.8
reg_alpha=0.1
reg_lambda=1.0

XGBoost:
n_estimators=800
learning_rate=0.05
max_depth=5
subsample=0.8
colsample_bytree=0.8
reg_alpha=0.1
reg_lambda=1.0
```

---

## Yapılan Deneyler

| Deney | Sonuç | Karar |
|---|---|---|
| `%80+` missing sütunları kaldırma | Küçük iyileşme | Korundu |
| Missing indicator ekleme | F1 ve MCC arttı | Korundu |
| `CAT_6` kaldırma | Pozitif etki | Korundu |
| Threshold fine-tuning | En iyi eşik bulundu | Korundu |
| `n_estimators=800` | MCC artışı sağladı | Korundu |
| `max_depth` artırma | Overfitting oluşturdu | Geri alındı |


---

## MASTER Veri Seti — 5-Fold Stratified CV

| Fold | F1 | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|---|
| 1 | 0.8950 | 0.8352 | 0.9190 | 0.5510 |
| 2 | 0.8918 | 0.8473 | 0.9296 | 0.5254 |
| 3 | 0.8928 | 0.8348 | 0.9213 | 0.5464 |
| 4 | 0.8889 | 0.8602 | 0.9294 | 0.5248 |
| 5 | 0.8830 | 0.8183 | 0.9036 | 0.5030 |

| Metrik | Ortalama | Std |
|---|---|---|
| **F1 Skoru** | **0.8903** | ±0.0041 |
| **MCC** | **0.5301** | ±0.0172 |
| **PR-AUC** | **0.9206** | ±0.0095 |
| **ROC-AUC** | **0.8392** | ±0.0140 |
| Dengeli Doğruluk | 0.7291 | ±0.0096 |

---

## Confusion Matrix (CV üzerinden)

```text
              Tahmin
              Benign  Patojenik
Gerçek Benign   402      380
Gerçek Patojen  120     2029
```

- **FN = 120**
- **FP = 380**

> FN değeri V1’e göre ciddi şekilde azalmıştır.
> Klinik açıdan bu önemli bir gelişmedir.

---

## Karar Eşiği Analizi

Threshold aralığı:
`0.20 → 0.70`
arası
`0.01`
adımlarla tarandı.

| Eşik | Precision | Recall | F1 | MCC |
|---|---|---|---|---|
| 0.30 | 0.8208 | 0.9716 | 0.8898 | 0.5076 |
| 0.35 | 0.8275 | 0.9665 | 0.8916 | 0.5206 |
| 0.40 | 0.8327 | 0.9609 | **0.8922** | **0.5274** |
| 0.45 | 0.8351 | 0.9521 | 0.8898 | 0.5200 |
| 0.50 | 0.8423 | 0.9442 | 0.8903 | 0.5297 |

> Optimal threshold:
> **0.40**
>
> Bu eşikte en iyi F1 değeri elde edildi.

---

## Alt Grup Sonuçları (eşik=0.40)

| Grup | N | F1 | MCC | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| **KANSER** | 388 | 0.8978 | 0.6329 | 0.9615 | 0.9270 |
| **PAH** | 372 | 0.9346 | 0.5138 | 0.9424 | 0.8097 |
| **CFTR** | 111 | 0.9508 | 0.7236 | 0.9801 | 0.9312 |

### Yorumlar

- PAH grubunda MCC hâlâ görece düşük
- CFTR grubunda belirgin güçlenme görüldü
- KANSER grubunda ROC-AUC yükseldi

---

## SHAP — Açıklanabilirlik

SHAP feature importance çalıştırıldı ancak:

```text
All arrays must be of the same length
```

hatası nedeniyle feature-name mapping kısmı başarısız oldu.

Sorunun:
- missing indicator sonrası feature sayısının değişmesi,
- SHAP feature isimlerinin yeniden oluşturulamaması

kaynaklı olduğu düşünülüyor.

Bu sorun sonraki versiyonda düzeltilecek.

---

## V1 ile Karşılaştırma

| Metrik | V1 | V2 |
|---|---|---|
| MASTER F1 | 0.8711 | **0.8903** |
| MCC | 0.5054 | **0.5301** |
| PR-AUC | 0.9207 | 0.9206 |
| ROC-AUC | 0.8372 | **0.8392** |

### Kazanımlar

- F1 artışı
- MCC artışı
- FN azalması
- Threshold optimizasyonu
- Daha güçlü CFTR performansı

---

## Sonuç

V2 aşamasında preprocessing ve threshold optimizasyonunun model performansına doğrudan katkı sağladığı görüldü.

Özellikle:
- missing indicator yaklaşımı,
- yüksek eksikli feature temizliği,
- threshold tuning

en faydalı geliştirmeler oldu.

Daha agresif model karmaşıklığı (`max_depth` artırma) ise overfitting oluşturdu ve performansı düşürdü.

Bu nedenle V2 sonunda:
- daha stabil,
- daha dengeli,
- klinik risk açısından daha güvenli

bir ensemble pipeline elde edildi.

---

## Sonraki Adımlar

- [ ] Optuna ile hiperparametre optimizasyonu
- [ ] SHAP feature-name hatasını düzeltme
- [ ] AA_ amino asit feature engineering
- [ ] EK_ feature yorumlama
- [ ] PAH grubuna özel yaklaşım
- [ ] Top-N feature selection

---

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `eda.py` | Keşifsel veri analizi |
| `train.py` | V2 ensemble modeli |
| `README.md` | V2 sonuçları |

---

## GitHub'a Pushlandı mı?

- [x] Evet — V2 başarıyla pushlandı

### Commit mesajı

```text
[V2] Missing indicators + threshold tuning — MASTER F1:0.8903, MCC:0.5301
```