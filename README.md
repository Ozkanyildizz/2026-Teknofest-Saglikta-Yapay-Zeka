# 🧬 V3 — CatBoost & Target Encoding Entegrasyonu

> **Mayıs 2026** · Üçüncü versiyon — Kategorik özellikler için Target Encoding kullanımı, üçlü ensemble (CatBoost entegrasyonu) ve MCC odaklı eşik optimizasyonu.

---

## 📋 Özet

V3 aşamasında, V2'de kurulan başarılı veri temizleme ve missing indicator altyapısı korunmuştur.

Bu versiyonda amaç, yüksek kardinaliteye sahip amino asit (`AA_`) ve kategorik (`CAT_`) sütunlardaki gizli kalıpları daha iyi öğrenebilmek için modelin matematiksel yaklaşımını değiştirmektir.

### Bu versiyonda öne çıkan geliştirmeler:

- `OrdinalEncoder` yerine **Target Encoding** kullanılması
- Ensemble yapısına **CatBoost** algoritmasının eklenmesi
- Sınıflandırma kararlarının üç modelin **(LightGBM + XGBoost + CatBoost)** ağırlıklı oylamasıyla yapılması
- Yüksek donanım kullanımında oluşan trafik sıkışıklığını çözmek için **işlemci (`n_jobs`) ayarlarının stabilize edilmesi**

> **Sonuç:** V3, V2'nin rekorunu kırarak F1 ve özellikle MCC değerlerinde ciddi bir sıçrama sağlamıştır.

---

## 💡 Yeni Bulgular & Strateji Değişikliği

V2 analizleri ve deneyimleri üzerinden V3'te şu stratejiler uygulandı:

- `AA_` (Amino asit) sütunlarının her bir varyantının patojeniklik oranını hesaplamak için rastgele sayı atamak yerine **Target Encoder** kullanıldı.
- Target Encoding sonrası veriyi en iyi okuyan ağaç tabanlı model olan **CatBoost**, sisteme ana karar vericilerden biri olarak eklendi.

---

## ⚙️ Yöntem

| Bileşen | Detay |
|---|---|
| **Model** | LightGBM + XGBoost + CatBoost Soft-Voting Ensemble |
| **Oylama Ağırlıkları** | `[1, 1, 1.2]` — CatBoost'a Target Encoding meyvelerini toplaması için hafif üstünlük |

---

## 🔧 Veri Ön İşleme Pipeline

| Adım | Yöntem | Gerekçe |
|---|---|---|
| Yüksek eksikli sütunlar | %80+ missing sütunlar kaldırıldı | V2'den korundu (18 sütun silindi) |
| Sayısal eksik değer | Median Imputation + Missing Indicator | V2'den korundu |
| `CAT_6` | Tamamen kaldırıldı | V2'den korundu (%97+ eksik) |
| Sıfır varyans | VarianceThreshold (`1e-4`) | V2'den korundu |
| Kategorik sütunlar | **TargetEncoder** (`smoothing=10`) | **[YENİ]** Kategorik verilerin patojeniklik olasılığını modele yansıtmak |

---

## 🎛️ Hiperparametreler (V3)

### LightGBM
```python
n_estimators    = 800
learning_rate   = 0.03
max_depth       = 6
num_leaves      = 31
subsample       = 0.8
colsample_bytree = 0.8
```

### XGBoost
```python
n_estimators    = 800
learning_rate   = 0.03
max_depth       = 5
subsample       = 0.8
colsample_bytree = 0.8
```

### CatBoost
```python
iterations      = 800
learning_rate   = 0.03
depth           = 6
```

---

## 🧪 Yapılan Deneyler

| Deney | Sonuç | Karar |
|---|---|---|
| Target Encoding kullanımı | Kategorik gürültüyü azalttı, MCC arttı | ✅ Korundu |
| CatBoost eklenmesi | Ensemble stabilitesini güçlendirdi | ✅ Korundu |
| CatBoost ağırlığını 1.2 yapma | F1 ve MCC'de optimum denge sağladı | ✅ Korundu |
| `n_jobs=-1` kullanımı | İşlemci sıkışıklığı yarattı (Worker error) | 🔄 `n_jobs=1` olarak güncellendi |

---

## 📊 Karar Eşiği Analizi (V3 Sonuçları)

**Threshold aralığı:** `0.20 → 0.70` arası `0.05` adımlarla tarandı.

| Eşik | Precision | Recall | F1 | MCC |
|:---:|:---:|:---:|:---:|:---:|
| 0.40 | 0.8325 | 0.9623 | 0.8927 | 0.5294 |
| 0.45 | 0.8382 | 0.9572 | **0.8938** | 0.5386 |
| 0.50 | 0.8430 | 0.9470 | 0.8920 | 0.5364 |
| 0.55 | 0.8498 | 0.9372 | 0.8913 | 0.5418 |
| 0.60 | 0.8591 | 0.9307 | 0.8935 | **0.5594** |

> 🎯 **Optimal Threshold (F1 Odaklı):** `0.45` → F1: `0.8938`
>
> 🏆 **Optimal Threshold (MCC Odaklı):** `0.60` → MCC: `0.5594`

> **Not:** Tıbbi teşhis dengesi açısından F1'i çok düşürmeden MCC'yi `0.5594`'e çıkaran `0.60` eşiği, klinik risk açısından büyük bir potansiyel barındırmaktadır.

---

## 📈 V2 ile Karşılaştırma

| Metrik | V2 (Önceki) | V3 (Yeni) | Kazanım |
|---|:---:|:---:|:---:|
| **MASTER F1** | 0.8903 | 0.8938 | `+0.0035` ✅ |
| **MASTER MCC** | 0.5301 | 0.5386 *(Max: 0.5594)* | `+0.0085` *(Max `+0.0293`)* 🚀 |
| **Optimal Eşik** | 0.40 | 0.45 | Daha dengeli tahmin sınırına ulaşıldı ✅ |

---

## 🏅 Kazanımlar

- **Target Encoding** ile modelin kategorikleri anlama kapasitesi arttı.
- V2'deki overfitting riski `learning_rate` (`0.03`) düşürülerek dizginlendi.
- MCC'de ulaşılan yeni zirve (**0.5594**), yarışma metrikleri açısından çok kritik bir eşiğin geçilmesini sağladı.
