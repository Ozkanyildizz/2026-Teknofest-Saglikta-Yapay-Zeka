# V5 (GPU Hızlandırılmış Versiyon)

> Beşinci versiyon — V4 mimarisinin (CatBoost Stacking + EK Etkileşim) direkt olarak NVIDIA GPU (CUDA) üzerinden eğitilecek şekilde uyarlanmış hali.

---

## Özet

V5 aşamasında model mimarisinde büyük bir değişikliğe gidilmemiş, ancak eğitim süresini dramatik ölçüde kısaltmak için **tam GPU desteği (Hardware Acceleration)** entegre edilmiştir.

**Temel değişiklikler:**

- LightGBM: `device_type='gpu'` eklendi.
- XGBoost: `device='cuda'`, `tree_method='hist'` parametreleriyle donanımsal hızlandırma aktifleştirildi.
- CatBoost: `task_type='GPU'` ile eğitimin doğrudan ekran kartında gerçekleşmesi sağlandı.
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
| **F1 Skoru** | **0.8941** |
| **MCC** | **0.5370** |
| **PR-AUC** | **0.9222** |
| **ROC-AUC** | **0.8440** |

**Confusion Matrix (Karışıklık Matrisi) (Eşik: 0.38):**
```text
                Tahmin
                Benign  Patojenik
  Gerçek Benign    373        409
  Gerçek Patojen    81       2068
```

> **Önemli Kazanım:** V5'te tam GPU hızlandırması sayesinde süre saniyelere inerken model kalitesi korundu. FN (Yanlış Negatif) sayısı **81** oldu. Optimal karar eşiği: **0.38**

### Alt Grup Sonuçları

| Grup | F1 | MCC | PR-AUC |
|---|---|---|---|
| KANSER | **0.9069** | 0.6704 | 0.9615 |
| PAH | ~0.94 | ~0.55 | ~0.94 |
| CFTR | ~0.95 | ~0.73 | ~0.98 |

*(Not: MASTER üzerindeki Stacking modelinin alt gruplara genel yansımasıdır, PAH için ayrıca threshold 0.25 incelendi.)*

### PAH Özel Eşik Analizi

PAH grubunda `best_thresh=0.35` yanı sıra `-0.10` düşürülmüş alternatif eşik (`0.25`) de test edildi.  
Daha düşük eşik kullanımı PAH grubunda recall (duyarlılık) oranını artırarak ölümcül risk taşıyan PAH vakalarının kaçırılmasını (FN) önlemede başarılı oldu.

---

## Önceki Versiyonlarla Karşılaştırma

| Metrik | V2 | V3 | V4 | V5 |
|---|---|---|---|---|
| MASTER F1 | 0.8903 | 0.8929 | **0.8958** | 0.8941 |
| MASTER MCC | 0.5301 | 0.5401 | **0.5434** | 0.5370 |
| KANSER F1 | - | - | ~0.90 | **0.9069** |

---

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `train.py` | GPU Stacking ensemble, XGBoost Optuna |
| `eda.py` | EK_ analizi |
| `README.md` | Bu dosya |

---

## GitHub'a Pushlanmalı mı?

- [ ] Beklemede — V5, V4'ün skorlarını (F1:0.8958) birebir yakalayamamış olsa da (-0.0017 fark), 15 dakikalık süreyi saniyelere indiren müthiş bir **Hardware Acceleration (Donanım Hızlandırma)** sürümüdür. KANSER alt grubunda ise F1'i **0.9069**'a taşıyarak rekor kırmıştır.

### Commit mesajı önerisi:

```
[V5] GPU Accelerated Stacking (XGB Optuna) — MASTER F1:0.8941, KANSER F1:0.9069
```
