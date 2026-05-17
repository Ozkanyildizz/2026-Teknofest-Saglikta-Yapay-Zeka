# V1 

> İlk versiyon — Baseline model. Yarışma veri setinin keşfi ve ilk sonuçlar.

---

## Özet

PDR sürecinin başlangıç versiyonu. Yarışmanın asıl veri seti (`YARISMA_TRAIN_MASTER.csv`)
üzerinde LightGBM + XGBoost Soft-Voting Ensemble ile baseline model kuruldu.
PSR aşamasındaki yaklaşım yeni veri setinin yapısına (şifreli sütunlar, yüksek eksiklik,
sınıf dengesizliği) uyarlandı.

---

## EDA Bulguları (eda.py)

| Veri Seti | Satır | Patojenik | Benign | Eksik % |
|---|---|---|---|---|
| MASTER | 2.931 | %73.3 | %26.7 | %54.9 |
| KANSER | 388 | %69.1 | %30.9 | %57.2 |
| PAH | 372 | %83.3 | %16.7 | %54.3 |
| CFTR | 111 | %81.1 | %18.9 | %30.3 |

**Önemli bulgular:**
- Hiçbir sayısal sütunda eksik değerden tamamen arınmış sütun yok
- AL_27–AL_38 grubu ~%91 eksik (en problemli grup)
- 180 sütun %0-50 arası eksik, 163 sütun %50+ eksik
- `CAT_6` sütunu %97.7 eksik (neredeyse değersiz)
- `AA_1`: 24 farklı amino asit, `AA_2`: 25 farklı amino asit
- En baskın amino asitler: R (Arjinin), G (Glisin), L (Lösin)

---

## Yöntem

### Model
- **LightGBM + XGBoost Soft-Voting Ensemble** (her ikisi eşit ağırlıklı)
- `scale_pos_weight = 0.364` (benign/pathogenic oranı — dengesizliği dengeler)

### Veri Ön İşleme Pipeline
| Adım | Yöntem | Gerekçe |
|---|---|---|
| Sayısal eksik | Median Imputation | Aykırı değere karşı dayanıklı |
| Ölçekleme | RobustScaler | %90+ eksik sütunlarda aykırı değer riski |
| Sıfır varyans | VarianceThreshold (1e-4) | 351 → 294 özellik |
| Kategorik (CAT_) | OrdinalEncoder | LightGBM ile uyumlu |
| Amino asit (AA_) | OrdinalEncoder | Harf → sayısal |

**İşlem sonrası:** 351 → **294 özellik** (57 sabit sütun elendi)

### Hiperparametreler
```python
LightGBM: n_estimators=500, lr=0.05, max_depth=6,
          num_leaves=31, subsample=0.8, colsample_bytree=0.8,
          reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=0.364

XGBoost:  n_estimators=500, lr=0.05, max_depth=5,
          subsample=0.8, colsample_bytree=0.8,
          reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=0.364
```

---

## Sonuçlar

### MASTER Veri Seti — 5-Fold Stratified CV

| Fold | F1 | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|---|
| 1 | 0.8687 | 0.8281 | 0.9150 | 0.4965 |
| 2 | 0.8737 | 0.8377 | 0.9245 | 0.4974 |
| 3 | 0.8707 | 0.8376 | 0.9259 | 0.5049 |
| 4 | 0.8811 | 0.8669 | 0.9330 | 0.5563 |
| 5 | 0.8611 | 0.8156 | 0.9054 | 0.4717 |

| Metrik | Ortalama | Std |
|---|---|---|
| **F1 Skoru** | **0.8711** | ±0.0065 |
| **MCC** | **0.5054** | ±0.0278 |
| **PR-AUC** | **0.9207** | ±0.0096 |
| **ROC-AUC** | **0.8372** | ±0.0169 |
| Dengeli Doğruluk | 0.7488 | ±0.0162 |

### Confusion Matrix (CV üzerinden, eşik=0.30)
```
              Tahmin
              Benign  Patojenik
Gerçek Benign   484      298
       Patojen  261     1888
```
- **FN = 261** (gizli patoloji — klinik risk)
- **FP = 298** (gereksiz alarm)

### Karar Eşiği Analizi

| Eşik | Precision | Recall | F1 | MCC |
|---|---|---|---|---|
| 0.25 | 0.8188 | 0.9670 | 0.8867 | 0.4916 |
| 0.30 | 0.8288 | 0.9600 | **0.8896** | **0.5132** |
| 0.35 | 0.8361 | 0.9497 | 0.8893 | 0.5195 |
| 0.40 | 0.8510 | 0.9116 | 0.8803 | 0.5095 |
| 0.45 | 0.8582 | 0.8874 | 0.8726 | 0.5005 |
| 0.50 | 0.8637 | 0.8785 | 0.8710 | 0.5053 |

> **Optimal eşik: 0.30** (F1 = 0.8896, MCC = 0.5132)  
> Daha düşük eşik → daha yüksek Recall (klinik açıdan önemli: FN azalır)

### Alt Grup Sonuçları (eşik=0.30)

| Grup | N | F1 | MCC | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| **KANSER** | 388 | 0.8990 | 0.6365 | 0.9565 | 0.9182 |
| **PAH** | 372 | 0.9287 | 0.4542 | 0.9320 | 0.7883 |
| **CFTR** | 111 | 0.9451 | 0.6965 | 0.9791 | 0.9228 |

> PAH grubunda MCC düşük (0.45) — sınıf dengesizliği çok şiddetli (%83 patojenik)

### SHAP — En Önemli 15 Özellik

| Özellik | Ort. |SHAP| | Yorum |
|---|---|---|
| EK_7 | 0.643 | En baskın özellik |
| AA_1 | 0.286 | Referans amino asit çok etkili |
| AL_327 | 0.249 | Şifreli skor |
| EK_9 | 0.226 | |
| AL_12 | 0.173 | |
| AA_2 | 0.156 | Alternatif amino asit etkili |
| EK_5 | 0.152 | |
| AL_26 | 0.144 | |
| AL_14 | 0.140 | |
| EK_1 | 0.140 | |

> EK_ ve AA_ sütunları beklenenden çok daha önemli çıktı!

---

## Önceki Versiyonla Karşılaştırma

V1 ilk baseline — referans noktası olarak kullanılacak.

| Metrik | V1 (Bu) | V2 | V3 |
|---|---|---|---|
| MASTER F1 | **0.8711** | — | — |
| MCC | **0.5054** | — | — |
| PR-AUC | **0.9207** | — | — |
| ROC-AUC | **0.8372** | — | — |

---

## Sonraki Versiyona Öneriler

- [ ] **AA_ özellik mühendisliği** — EK_7 ve AA_ sütunları en önemli çıktı. AA_1/AA_2'den
  biyokimyasal özellik türet: polarity, hydrophobicity, molecular weight, pKa, charge
- [ ] **EK_ sütunlarını araştır** — SHAP'ta EK_7 birinci sırada. Bu sütunların ne olduğunu anlamaya çalış
- [ ] **Optuna ile hiperparametre optimizasyonu** — şu an varsayılan değerler kullanıldı
- [ ] **PAH grubunu özel ele al** — MCC=0.45 zayıf, SMOTE veya alt-grup spesifik model dene
- [ ] **CAT_6'yı kaldır** — %97.7 eksik, bilgi taşımıyor olabilir
- [ ] **Feature selection** — 294 özellikten SHAP ile top-N seç, daha az özellikle model dene

---

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `eda.py` | Keşifsel veri analizi |
| `train.py` | Baseline model eğitimi ve değerlendirme |
| `README.md` | Bu dosya — sonuçlar ve notlar |

---

## GitHub'a Pushlandı mı?

- [x] **Evet** — İlk baseline olduğu için push edildi

**Commit mesajı:**
```
[V1] Baseline — MASTER F1:0.8711, MCC:0.5054, PR-AUC:0.9207
```
