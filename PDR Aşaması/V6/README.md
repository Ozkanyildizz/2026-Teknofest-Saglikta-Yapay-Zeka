# V6 — BLOSUM + Grantham + SHAP Top-100 + LGBM Meta-Learner

> Altıncı versiyon — V5 GPU mimarisinin üzerine domain özellikleri, özellik seçimi ve meta-learner iyileştirmeleri.

---

## Özet

V6, V4'ün MASTER skorunu (F1: 0.8958) geçmek için odaklı iyileştirmeler içerir:

- **BLOSUM62** substitution skoru + **Grantham mesafesi** (yeni AA özellikleri)
- **AL blok eksiklik** özellikleri (AL_27–38 bloğu ayrı izlenir)
- **SHAP Top-100** özellik seçimi (V3'teki başarı geri getirildi)
- **passthrough=True** + sığ **LGBM meta-learner** (LogisticRegression yerine)
- **MCC-optimal** karar eşiği
- Alt gruplar için **ayrı 5-fold CV** değerlendirmesi
- Tam **GPU** desteği (LightGBM + XGBoost + CatBoost)

---

## Yöntem

### Yeni Özellikler (V6)

| Özellik | Açıklama |
|---|---|
| `AA_blosum62` | Referans→alt amino asit BLOSUM62 skoru |
| `AA_is_conservative` | BLOSUM ≥ 0 → konservatif mutasyon |
| `AA_grantham_dist` | Grantham fizikokimyasal mesafe |
| `AL_block1/2/3_missing_ratio` | AL blok bazında eksiklik oranı |

### Model Mimarisi

```
StackingClassifier (passthrough=True)
├── LightGBM   (V4 Optuna params, GPU)
├── XGBoost    (V5 Optuna params, CUDA)
├── CatBoost   (GPU)
└── Meta: LGBM (max_depth=3, num_leaves=8)
    ↑ base tahminleri + ham özellikler
```

### Özellik Seçimi

1. Tüm özellikler ön işlenir (~400+)
2. LightGBM probe modeli ile SHAP hesaplanır
3. Top-100 özellik stacking'e verilir

---

## Sonuçlar

### MASTER Veri Seti

| Metrik | Değer |
|---|---|
| F1 Skoru | **0.8965** |
| MCC | **0.5563** |
| PR-AUC | **0.9257** |
| ROC-AUC | **0.8534** |
| FN | 100 (Eşik: 0.46) |

### Alt Grup Sonuçları (Ayrı CV)

*(Alt grup CV bu turda atlandı)*

| Grup | F1 | MCC | PR-AUC | FN |
|---|---|---|---|---|
| KANSER | — | — | — | — |
| PAH | — | — | — | — |
| CFTR | — | — | — | — |

---

## Önceki Versiyonla Karşılaştırma

| Metrik | V4 | V5 | V6 |
|---|---|---|---|
| MASTER F1 | 0.8958 | 0.8941 | **0.8965** (+0.0007) |
| MASTER MCC | 0.5434 | 0.5370 | **0.5563** (+0.0129) |
| MASTER PR-AUC | 0.9199 | 0.9222 | **0.9257** (+0.0058) |
| MASTER ROC-AUC | 0.8451 | 0.8440 | **0.8534** (+0.0083) |

---

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `train.py` | GPU stacking, SHAP seçimi, PCA, K-Means, Derin AA Bio |
| `README.md` | Bu dosya |

---

## GitHub'a Pushlandı mı?

- [x] Başarıyla tamamlandı ve push edildi! 🚀
