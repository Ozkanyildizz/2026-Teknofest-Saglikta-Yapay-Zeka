# V6(L) — Sızıntısız (Leak-Free) Pipeline + Tam Grantham Matrisi + SHAP Top-100 + LGBM Meta-Learner

> Altıncı versiyon — Kod incelemesi sonrası tespit edilen Veri Sızıntısı (Data Leakage) sorunları giderilmiş, tam Grantham matrisi eklenmiş gerçek dünya performansını yansıtan yasal ve temizlenmiş (Leak-Free) versiyondur.

---

## Özet

V6, V4'ün MASTER skorunu (F1: 0.8958) geçmek için odaklı iyileştirmeler içerir:

- **Veri Sızıntısı Çözüldü**: PCA, K-Means ve SHAP Feature Selection adımları artık 5-Fold CV döngüsünün *içine* gömülerek Validation verisinin Train aşamasına sızması engellendi.
- **Tam Grantham Matrisi**: Tüm 400 kombinasyonu barındıran tam radikallik mesafe matrisi (0-215) entegre edildi.
- **BLOSUM62** substitution skoru + biyokimyasal özellik farkları.
- **AL blok eksiklik** özellikleri eklendi.
- **SHAP Top-100** özellik seçimi (Her fold'un kendi eğitim setine özel).
- **passthrough=True** + sığ **LGBM meta-learner** (LogisticRegression düzeltildi).
- Tam **GPU** desteği (LightGBM + XGBoost + CatBoost).

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

1. 5-Fold Custom CV başlar. Her fold için `X_tr` ve `X_val` ayrılır.
2. `X_tr` üzerinden Imputer ve Scaler `fit` edilir.
3. K-Means ve PCA *sadece* `X_tr` kullanılarak eğitilir.
4. LightGBM probe modeli `X_tr` ile eğitilip SHAP hesaplanır. Top-100 özellik belirlenir.
5. Sadece bu 100 özellik kullanılarak Stacking Ensemble `fit` edilir.
6. `X_val` üzerinde `predict_proba` alınır. Her fold bittiğinde gerçekçi yansıma elde edilir.

---

## Sonuçlar

### MASTER Veri Seti

| Metrik | Değer |
|---|---|
| F1 Skoru | **0.8919** |
| MCC | **0.5450** |
| PR-AUC | **0.9116** |
| ROC-AUC | **0.8377** |
| FN | 135 (Eşik: 0.51) |

> **Not:** Data Leakage düzeltildiği için görünürdeki skorlar bir önceki sızıntılı V6'ya (0.8965) göre doğal olarak düşmüştür. Ancak V4'e kıyasla MCC'de iyileşme sağlanmıştır ve bu skor gerçek, savunulabilir yasal skordur.

### Alt Grup Sonuçları (Ayrı CV)

*(Sızıntı engelleme testleri Master üzerinde yapıldığından alt gruplar bu koşuda atlandı)*

| Grup | F1 | MCC | PR-AUC | FN |
|---|---|---|---|---|
| KANSER | — | — | — | — |
| PAH | — | — | — | — |
| CFTR | — | — | — | — |

---

## Önceki Versiyonla Karşılaştırma

| Metrik | V4 | V5 | V6(Leak-Free) |
|---|---|---|---|
| MASTER F1 | 0.8958 | 0.8941 | **0.8919** (-0.0039) |
| MASTER MCC | 0.5434 | 0.5370 | **0.5450** (+0.0016) |
| MASTER PR-AUC | 0.9199 | 0.9222 | **0.9116** (-0.0083) |
| MASTER ROC-AUC | 0.8451 | 0.8440 | **0.8377** (-0.0074) |

---

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `train.py` | GPU stacking, SHAP seçimi, PCA, K-Means, Derin AA Bio |
| `README.md` | Bu dosya |

---

## GitHub'a Pushlandı mı?

- [x] Başarıyla tamamlandı ve push edildi! 🚀
