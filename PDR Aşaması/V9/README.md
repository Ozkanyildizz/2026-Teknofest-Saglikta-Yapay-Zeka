# V7 — Sızdırmaz Pipeline + Optuna GPU Optimizasyonu

> **V5 MİMARİSİNİN SINIRLARINI ZORLAMAK**
>
> V5'teki LightGBM + XGBoost + CatBoost Stacking modelini Optuna ile 500–1000 trial optimize etmek
> ve sınıf dengesizliğini **imblearn.pipeline** içinde sızdırmaz biçimde çözmek.

---

## ✅ V7 Sonuçları (Eğitim Tamamlandı)

> **V7, V5'i hem F1 hem MCC'de geçti!** `dF1=+0.0018  dMCC=+0.0162`

### MASTER 5-Fold CV (SMOTE=ADASYN, Eşik=0.21)

| Metrik | V5 | **V7** | Fark |
|---|---|---|---|
| **F1 Skoru** | 0.8941 | **0.8959** | +0.0018 ✅ |
| **MCC** | 0.5370 | **0.5532** | +0.0162 ✅ |
| **PR-AUC** | 0.9222 | **0.9268** | +0.0046 ✅ |
| **ROC-AUC** | — | **0.8523** | — |

### Alt Grup Sonuçları

| Grup | F1 | MCC | PR-AUC | ROC-AUC | Eşik |
|---|---|---|---|---|---|
| **KANSER** | 0.9116 | 0.6889 | 0.9557 | 0.9203 | 0.21 |
| **PAH** | 0.9383 | 0.5590 | 0.9345 | 0.8091 | 0.28 |
| **CFTR** | 0.9556 | 0.7651 | 0.9878 | 0.9487 | 0.21 |

### Confusion Matrix (MASTER)

```
                Tahmin
                Benign  Patojenik
Gerçek Benign     407       375
Gerçek Patojen    101      2048

FN=101  FP=375
```

### Oversampling Karşılaştırması

| Yöntem | F1 | MCC | PR-AUC | ROC-AUC | Süre |
|---|---|---|---|---|---|
| SMOTE | 0.8889 | 0.5189 | 0.9272 | 0.8510 | 38.0 dk |
| **ADASYN** | **0.8959** | **0.5532** | 0.9268 | 0.8523 | 37.2 dk ← KAZANAN |
| BorderlineSMOTE | 0.8960 | 0.5478 | 0.9263 | 0.8518 | 36.6 dk |

### Versiyon Karşılaştırması

| Ver | Model | F1 | MCC | PR-AUC |
|---|---|---|---|---|
| V1 | Baseline LGB+XGB | 0.8711 | 0.5054 | 0.9207 |
| V2 | Missing Indicator + Threshold | 0.8903 | 0.5301 | 0.9206 |
| V3 | Biochem AA + Optuna + SHAP Top-100 | 0.8929 | 0.5401 | — |
| V4 | CatBoost Stacking + EK Interact | 0.8958 | 0.5434 | 0.9199 |
| V5 | GPU Stacking (100 trial Optuna) | 0.8941 | 0.5370 | 0.9222 |
| V6 | Leak-Free + Grantham + LGBM Meta | 0.8919 | 0.5450 | 0.9116 |
| **V7** | **V5+Optuna500t+ADASYN** | **0.8959** | **0.5532** | **0.9268** |

---

## Optuna En İyi Parametreler

### XGBoost (700 trial — GPU)

| Parametre | Değer |
|---|---|
| n_estimators | 1100 |
| learning_rate | 0.02587 |
| max_depth | 7 |
| subsample | 0.8664 |
| colsample_bytree | 0.5120 |
| reg_alpha (L1) | 6.651 |
| reg_lambda (L2) | 0.034 |
| min_child_weight | 2 |
| scale_pos_weight | 0.5252 |
| gamma | 0.1063 |
| **Optuna MCC (3-fold)** | **0.5685** |

### CatBoost (500 trial — GPU)

| Parametre | Değer |
|---|---|
| iterations | 1000 |
| learning_rate | 0.006690 |
| depth | 6 |
| l2_leaf_reg | 5.534 |
| scale_pos_weight | 0.5405 |
| bootstrap_type | Bernoulli |
| subsample | 0.9915 |
| **Optuna MCC (3-fold)** | **0.5741** |

---

## Özet

V7, **V5'in doğrudan devamıdır**. V5'te kurulan mimari değiştirilmez — sadece iki yeni güç eklenir:

| V5'te ne vardı? | V7'de ne ekleniyor? |
|---|---|
| LightGBM + XGBoost + CatBoost → LogReg meta | ✅ Aynı mimari korunuyor |
| GPU hızlandırma (CUDA/hist) | ✅ Aynı GPU desteği |
| V3'ten biyokimyasal AA özellikleri (4 özellik) | ✅ Aynen alındı |
| V4'ten EK_i×EK_j etkileşim terimleri | ✅ Aynen alındı |
| V4'ten eksiklik örüntüsü özellikleri | ✅ Aynen alındı |
| Optuna: sadece XGBoost için 100 trial | 🆕 **700 trial (XGB) + 500 trial (CatBoost)** |
| Sınıf dengesizliği: sadece scale_pos_weight | 🆕 **SMOTE / ADASYN / BorderlineSMOTE** |

> **Not:** V6'nın Grantham matrisi, BLOSUM62, PCA, K-Means, LGBM meta gibi eklentileri
> V7'ye dahil **edilmemiştir**. V7'nin görevi V5'i zorlamaktır, V6 üzerine inşa etmek değil.

---

## Neden Bu İki Ekleme?

### 1. Optuna 500–1000 Trial

V5'te yalnızca 100 trial yapılmıştı. V5 README'sinde şöyle yazar:
> *"V5, V4'ün skorlarını (F1:0.8958) birebir yakalayamamış olsa da..."*

Bu farkın sebebi hiperparametre optimizasyonunun yüzeysel kalmasıydı.
700+ trial ile **L1 (`reg_alpha`), L2 (`reg_lambda`), `max_depth` ve `scale_pos_weight`**
parametreleri derinlemesine tarandı.

| Parametre | V5'te | V7'de |
|---|---|---|
| XGBoost Optuna trial | 100 | **700** |
| CatBoost Optuna trial | 0 (hiç yapılmadı) | **500** |
| L1 regülarizasyon aranıyor mu? | Hayır | **Evet** |
| L2 regülarizasyon aranıyor mu? | Hayır | **Evet** |

### 2. SMOTE / ADASYN / BorderlineSMOTE

MASTER veri setinde **%73 patojenik / %27 benign** dengesizliği var.
PAH grubunda bu **5:1** seviyesine ulaşıyor.

`scale_pos_weight` tek başına yeterli değil — model "her şeyi patojenik say" gibi davranabiliyor.
Sentetik benign örnekler üretmek bu eğilimi kırar. **Kazanan: ADASYN** (MCC=0.5532).

---

## 🚨 KRİTİK KURAL — DATA LEAKAGE

**SMOTE tüm veriye UYGULANMAZ. Her fold'da:**

```
❌ YANLIŞ (Leaky):
   1. Tüm X_train'e SMOTE uygula
   2. CV yap → Val seti sentetik veriden "haberdar"

✅ DOĞRU (V7 yaklaşımı — Sızdırmaz):
   CV döngüsü başlar
     Fold açılır: X_train / X_val ayrılır
     → Preprocessor SADECE X_train'e fit edilir
     → SMOTE SADECE X_train_proc'a uygulanır (sentetik üretilir)
     → X_val DOKUNULMAZ — orijinal, sentetik veri görmez
     Fold kapanır
   CV döngüsü biter
```

---

## V5 Mimarisi (Değişmeden Korunuyor)

```
StackingClassifier
├── LightGBM   (V5 Optuna params — GPU)
├── XGBoost    (V7 Optuna: 700 trial — CUDA/hist)
├── CatBoost   (V7 Optuna: 500 trial — GPU) ← V5'te Optuna yoktu!
└── Meta-Model: LogisticRegression(C=0.1)   ← V5 ile aynı
```

---

## Optuna Arama Uzayı (L1/L2 Odaklı)

### XGBoost (700 trial):

| Parametre | Aralık | Neden Kritik? |
|---|---|---|
| `reg_alpha` (L1) | 1e-8 → 10 (log) | Özellik seçimi, sparse veri için |
| `reg_lambda` (L2) | 1e-8 → 10 (log) | Ağırlık büyüklüğü — overfitting önleme |
| `max_depth` | 3 → 10 | Derinlik / overfit kontrolü |
| `scale_pos_weight` | 0.15 → 0.60 | Sınıf dengesizliği ağırlığı |
| `gamma` | 0 → 5 | Dal budama eşiği |
| `learning_rate` | 0.003 → 0.15 | Adım büyüklüğü |
| `n_estimators` | 300 → 1500 | Ağaç sayısı |

### CatBoost (500 trial) — V5'te HİÇ yapılmamıştı:

| Parametre | Aralık | Neden Kritik? |
|---|---|---|
| `l2_leaf_reg` | 1 → 10 | L2 regülarizasyon |
| `depth` | 4 → 8 | Derinlik / overfit kontrolü |
| `scale_pos_weight` | 0.15 → 0.60 | Sınıf dengesizliği ağırlığı |
| `bootstrap_type` | Bayesian / Bernoulli | Örnekleme stratejisi |
| `learning_rate` | 0.003 → 0.15 | Adım büyüklüğü |
| `iterations` | 300 → 1000 | Ağaç sayısı |

---

## Oversampling Algoritmaları

| Algoritma | Nasıl Çalışır? | Avantaj | Ne Zaman İyi? |
|---|---|---|---|
| **SMOTE** | K-NN komşular arası interpolasyon | Stabil, yaygın | Genel kullanım |
| **ADASYN** | Zor örneklere odaklı ağırlıklı SMOTE | Sınır bölgesine yoğunlaşır | Belirsiz sınırlar |
| **BorderlineSMOTE** | Sadece sınıra yakın örnekleri büyütür | Hassas sınır öğrenmesi | Net sınır varsa |

`SMOTE_METHOD = 'auto'` seçilirse üçü de test edilir, **MCC'ye göre** en iyisi seçilir. **V7'de kazanan: ADASYN**.

---

## Çalıştırma

### Kurulum:
```bash
pip install lightgbm xgboost catboost optuna imbalanced-learn scikit-learn pandas numpy
```

### Eğitimi tekrar çalıştırmak için:
```bash
cd "PDR Aşaması\V7"
python train.py
```

### Kontrol parametreleri (train.py başında):
```python
OPTUNA_TRIALS   = 500    # Deneme sayısı
RUN_XGB_OPTUNA  = False  # XGBoost Optuna tamamlandı (700 trial, 91.4 dk)
RUN_CAT_OPTUNA  = False  # CatBoost Optuna tamamlandı (500 trial, 631.6 dk)
RUN_OPTUNA      = False  # False -> JSON'dan yükler, True -> sıfırdan çalıştırır
SMOTE_METHOD    = 'auto' # 'smote' | 'adasyn' | 'borderline' | 'auto'
USE_GPU         = True
```

---

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `train.py` | Ana eğitim: V5 mimari + Optuna (XGB:700t, CAT:500t) + SMOTE pipeline |
| `optuna_study.py` | Optuna'yı train.py'den bağımsız çalıştırmak için |
| `README.md` | Bu dosya |
| `v7_optuna_results.json` | Optuna + CV sonuçları (kaydedildi ✅) |
| `v7_optuna.db` | SQLite yedek — Optuna çalışma geçmişi |

---

## GitHub'a Pushlandı mı?

- [x] Eğitim tamamlandı ✅
- [x] GitHub'a pushlandı ✅

### Commit mesajı:
```
[V7] V5+Optuna(XGB:700t,CAT:500t)+ADASYN_Leak-Free — MASTER F1:0.8959, MCC:0.5532
```

---

**Sonuç: BAŞARILI — V5 geçildi (dF1=+0.0018, dMCC=+0.0162)** ✅
