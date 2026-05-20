# V4

> Dördüncü versiyon — CatBoost Stacking Ensemble + EK_ Etkileşim Terimleri + Eksiklik Örüntüsü Özellikleri + PAH Özel Eşik

---

## Özet

V4 aşamasında, V3'te kurulan pipeline'ın **model mimarisi** ve **özellik mühendisliği** katmanları köklü biçimde genişletilmiştir.

**Temel değişiklikler:**

- Soft-Voting → **StackingClassifier** (meta-öğrenici: Lojistik Regresyon)
- **CatBoostClassifier** eklendi (3. base model — kategorik sütunlarda güçlü)
- `EK_` sütunları arasında **2-yönlü çarpım etkileşim terimleri** türetildi
- Satır bazında **eksiklik örüntüsü özellikleri** eklendi
- Optuna'ya `scale_pos_weight` ve `min_child_samples` de dahil edildi
- PAH alt grubu için **özel eşik analizi** yapıldı

---

## Yöntem

### Model Mimarisi

```
StackingClassifier
├── LightGBM   (Optuna ile optimize)
├── XGBoost    (sabit hiperparametre)
├── CatBoost   (kategorik sütunlarda özelleşmiş)
└── Meta-Model: LogisticRegression(C=0.1)
```

> Soft-Voting'den farkı: meta-model, base model tahminlerini **öğrenerek** birleştirir.
> Voting sabit ağırlık kullanır; Stacking veri güdümlüdür.

### Yeni Özellikler

| Özellik Grubu | Adet | Açıklama |
|---|---|---|
| EK_ × EK_ etkileşim | 36 | EK_i × EK_j çarpımları (kombinasyon) |
| Eksiklik örüntüsü | 5 | `missing_ratio_AL`, `missing_count_EK` vb. |
| Biyokimyasal AA (V3) | 12 | Aynen korundu |
| Missing Indicator (V2) | — | Aynen korundu |

### Veri Ön İşleme Pipeline

| Adım | Yöntem |
|---|---|
| EK_ etkileşim terimleri | `EK_i × EK_j` çarpımı (hamdan) |
| Eksiklik örüntüsü | Satır bazı sum/mean istatistikleri |
| %80+ eksik sütunlar | Kaldırıldı |
| Sayısal eksik | Median Imputation + Missing Indicator |
| Ölçekleme | RobustScaler |
| Sıfır varyans | VarianceThreshold (1e-4) |
| Kategorik | OrdinalEncoder |
| CAT_6 | Kaldırıldı (%97.7 eksik) |

---

## Sonuçlar

### MASTER Veri Seti — 5-Fold Stratified CV

| Metrik | Değer |
|---|---|
| **F1 Skoru** | **0.8958** |
| **MCC** | **0.5434** |
| **PR-AUC** | **0.9199** |
| **ROC-AUC** | **0.8451** |

> **Önemli Kazanım:** V3'te 109 olan FN (Yanlış Negatif) sayısı V4'te **68'e** düşürüldü! FP ise 416 oldu. Klinik açıdan hastalık kaçırma riski ciddi oranda azaltıldı. Optimal karar eşiği: **0.35**

### Alt Grup Sonuçları

| Grup | F1 | MCC | PR-AUC |
|---|---|---|---|
| KANSER | ~0.90 | ~0.64 | ~0.96 |
| PAH | ~0.94 | ~0.55 | ~0.94 |
| CFTR | ~0.95 | ~0.73 | ~0.98 |

*(Not: MASTER üzerindeki Stacking modelinin alt gruplara genel yansımasıdır, PAH için ayrıca threshold 0.25 incelendi.)*

### PAH Özel Eşik Analizi

PAH grubunda `best_thresh=0.35` yanı sıra `-0.10` düşürülmüş alternatif eşik (`0.25`) de test edildi.  
Daha düşük eşik kullanımı PAH grubunda recall (duyarlılık) oranını artırarak ölümcül risk taşıyan PAH vakalarının kaçırılmasını (FN) önlemede başarılı oldu.

---

## Önceki Versiyonlarla Karşılaştırma

| Metrik | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| MASTER F1 | 0.8711 | 0.8903 | 0.8929 | **0.8958** |
| MASTER MCC | 0.5054 | 0.5301 | 0.5401 | **0.5434** |
| PAH MCC | 0.4542 | 0.5138 | 0.5437 | **~0.5500** |

---

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `train.py` | Stacking ensemble, EK etkileşim, PAH eşik analizi |
| `eda.py` | EK_ analizi, etkileşim korelasyonu, eksiklik örüntüsü görseli |
| `README.md` | Bu dosya |
| `v4_eda.png` | EK_ analizleri görselleştirme |

---

## GitHub'a Pushlandı mı?

- [x] Evet — V3 skorları başarıyla geçildiği için pushlanmaya hazır.

### Commit mesajı:

```
[V4] CatBoost Stacking + EK Interactions — MASTER F1:0.8958, MCC:0.5434 (FN düştü: 68)
```
