# V1 — Özkan — Mayıs 2026

> İlk versiyon. Yarışma veri setinin keşfi, baseline model kurulumu.

---

## Özet

Bu versiyon PDR sürecinin başlangıç noktasıdır. Amaç yarışmanın asıl veri setini
(`YARISMA_TRAIN_MASTER.csv`) anlamak, temel veri mühendisliği adımlarını uygulamak
ve bir baseline model oluşturmaktır. PSR aşamasında LightGBM + XGBoost ensemble
ile iyi sonuçlar alınmıştı; bu versiyon o yaklaşımı yeni veri setine uyarlıyor.

---

## Yöntem

### Kullanılan Model
- **LightGBM + XGBoost Soft-Voting Ensemble** (PSR'den devam)
- Gelecek versiyonlar için baseline referans noktası

### Veri Ön İşleme
- Kategorik sütunlar (`CAT_1..CAT_6`): One-Hot Encoding
- Amino asit sütunları (`AA_1, AA_2`): Label Encoding (26 harf → int)
- Eksik değerler: Median imputation
- Ölçekleme: RobustScaler
- Veri dengesizliği: `class_weight='balanced'`

### Yapılmayan / Sonraki Versiyona Bırakılan
- Amino asit özellik mühendisliği (polarity, hydrophobicity, MW)
- Optuna ile hiperparametre optimizasyonu
- SMOTE ile oversample
- Feature selection (334 AL_ sütunu için)

---

## Sonuçlar

> ⚠️ Sonuçlar henüz doldurulmadı — model eğitimi tamamlandıkça burası güncellenecek.

### MASTER Veri Seti (5-Fold CV)

| Metrik | Değer |
|---|---|
| F1 Skoru | — |
| MCC (Matthews Korelasyon Katsayısı) | — |
| PR-AUC (Precision-Recall AUC) | — |
| ROC-AUC | — |
| Balanced Accuracy | — |

### Confusion Matrix

```
              Tahmin
              Benign  Patojenik
Gerçek Benign   TN      FP
       Patojen  FN      TP
```

| | Değer |
|---|---|
| TN | — |
| FP | — |
| FN | — |
| TP | — |

### Alt Grup Sonuçları

| Grup | Satır | F1 | MCC | PR-AUC |
|---|---|---|---|---|
| KANSER | 388 | — | — | — |
| PAH | 372 | — | — | — |
| CFTR | 111 | — | — | — |

### Karar Eşiği Analizi

| Eşik | Precision | Recall | F1 | MCC |
|---|---|---|---|---|
| 0.30 | — | — | — | — |
| 0.40 | — | — | — | — |
| 0.50 | — | — | — | — |
| 0.60 | — | — | — | — |
| 0.70 | — | — | — | — |

**Seçilen optimal eşik:** —

---

## Önceki Versiyonla Karşılaştırma

Bu V1'dir — referans baseline olarak kullanılacak.  
Sonraki versiyonlar bu tablodaki değerleri geçmeyi hedeflemeli:

| Metrik | V1 (Bu) | V2 | V3 |
|---|---|---|---|
| MASTER F1 | — | — | — |
| MCC | — | — | — |
| PR-AUC | — | — | — |

---

## Öğrendiklerin / Sonraki Versiyona Öneriler

- [ ] `AA_1` ve `AA_2` sütunlarından biyokimyasal özellikler türetilebilir (polarity, hydrophobicity, MW, pKa)  
- [ ] `CAT_6` (bölge kalite filtresi: lcr, segdup) bazı varyantları hariç tutmak için kullanılabilir  
- [ ] 334 AL_ sütunundan **Feature Selection** yapılmalı (VarianceThreshold + Mutual Information)  
- [ ] Optuna ile LightGBM hiperparametre optimizasyonu önemli kazanım sağlayabilir  
- [ ] CFTR ve PAH setleri çok küçük (111 ve 372) — SMOTE ile dengeleme düşünülebilir  

---

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `analiz_dataset.py` | Veri seti yapısını inceleyen ilk keşif scripti |
| `model.py` | *(henüz oluşturulmadı)* Baseline model scripti |
| `eda.py` | *(henüz oluşturulmadı)* Keşifsel veri analizi |

---

## GitHub'a Pushlandı mı?

- [x] **Evet** — Bu versiyon baseline olduğu için push edildi  
- [ ] Hayır

**Commit mesajı:**
```
[V1] Baseline kurulumu — sonuçlar doldurulacak
```
