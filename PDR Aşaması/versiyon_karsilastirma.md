# AlgoMed — Versiyon Karşılaştırma Tablosu (Gerçek Sayılar)

> Tüm veriler ilgili versiyonun README.md ve çıktı dosyalarından alınmıştır.

---

## 📊 MASTER Veri Seti — Ana Metrik Tablosu

| Ver | Model | F1 | MCC | PR-AUC | ROC-AUC | FN | Eşik | Sızıntı? |
|---|---|---|---|---|---|---|---|---|
| **V1** | LGB + XGB (Soft-Voting) | 0.8711 | 0.5054 | 0.9207 | 0.8372 | **261** | 0.30 | ⚠️ Var |
| **V2** | LGB + XGB (Soft-Voting) | 0.8903 | 0.5301 | 0.9206 | 0.8392 | **120** | 0.40 | ⚠️ Var |
| **V3** | LGB + XGB + Optuna | 0.8929 | 0.5401 | — | — | **109** | 0.45 | ⚠️ Var |
| **V4** | LGB+XGB+CatBoost Stacking | **0.8958** | 0.5434 | 0.9199 | 0.8451 | **68** | 0.35 | ⚠️ Var |
| **V5** | GPU Stacking + XGB Optuna | 0.8941 | 0.5370 | 0.9222 | 0.8440 | **81** | 0.38 | ⚠️ Var |
| **V6(L)** | Leak-Free + Grantham + LGBM Meta | 0.8919 | 0.5450 | 0.9116 | 0.8377 | **135** | 0.51 | ✅ Yok |
| **V7** | V5 + ADASYN + Optuna (700/500t) | 0.8959 | 0.5532 | 0.9268 | 0.8523 | **101** | 0.21 | ✅ Yok |
| **V8** | FN Risk Bayrakları (çıktı yok) | 0.8941 | 0.5370 | 0.9222 | — | — | — | ✅ Yok |
| **V9** ⭐ | Grantham+BLOSUM+CAT_inter+SMOTE | 0.8836 | **0.5594** | **0.9268** | 0.8503 | **243** | **0.56** | ✅ Yok |

> ⚠️ **NOT:** V8 README'si çıktı değeri içermiyor (V6 başlığıyla yazılmış). V9 sayıları tamamen v9_outputs.txt'ten doğrulandı.

---

## 🏆 Alt Grup Performansı (Sızıntısız Versiyonlardan)

### V7 Alt Grupları (Eşik=0.21)
| Grup | F1 | MCC | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| KANSER | 0.9116 | 0.6889 | 0.9557 | 0.9203 |
| PAH | 0.9383 | 0.5590 | 0.9345 | 0.8091 |
| CFTR | 0.9556 | **0.7651** | 0.9878 | 0.9487 |

### V9 Alt Grupları (Eşik=0.56)
| Grup | F1 | MCC | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| KANSER | **0.9193** | **0.7169** | **0.9645** | **0.9339** |
| PAH | 0.9348 | 0.5399 | 0.9487 | 0.8273 |
| CFTR | 0.9257 | 0.6561 | **0.9878** | 0.9471 |

---

## ❓ En İyi Model Hangisi? — Gerçek Analiz

### Şartname Ana Metriğine Göre (F1):
| Sıra | Ver | F1 | Sızıntı |
|---|---|---|---|
| 🥇 1. | **V7** | **0.8959** | ✅ Sızıntısız |
| 🥈 2. | V4 | 0.8958 | ⚠️ Sızıntılı |
| 🥉 3. | V3 | 0.8929 | ⚠️ Sızıntılı |
| 4. | V6(L) | 0.8919 | ✅ Sızıntısız |
| 5. | V9 | 0.8836 | ✅ Sızıntısız |

### MCC'ye Göre:
| Sıra | Ver | MCC | Sızıntı |
|---|---|---|---|
| 🥇 1. | **V9** | **0.5594** | ✅ Sızıntısız |
| 🥈 2. | V7 | 0.5532 | ✅ Sızıntısız |
| 🥉 3. | V6(L) | 0.5450 | ✅ Sızıntısız |

### PR-AUC'ye Göre:
| Sıra | Ver | PR-AUC | Sızıntı |
|---|---|---|---|
| 🥇 1. | **V7 & V9** | **0.9268** | ✅ Sızıntısız |

### FN (eşik=optimal) Göre (Klinik Güvenlik):
| Sıra | Ver | FN | Eşik | Sızıntı |
|---|---|---|---|---|
| 🥇 1. | **V4** | **68** | 0.35 | ⚠️ Sızıntılı |
| 🥈 2. | V5 | 81 | 0.38 | ⚠️ Sızıntılı |
| 🥉 3. | V7 | 101 | 0.21 | ✅ Sızıntısız |
| 4. | V6(L) | 135 | 0.51 | ✅ Sızıntısız |
| 5. | **V9** | **243** | 0.56 | ✅ Sızıntısız |

> ⚠️ V9'da FN=243 çünkü eşik=0.56 (MCC optimize). Eşik 0.20'ye çekildiğinde Recall=0.9269 → FN dramatik düşer ama tam sayı çıktıda yok.

---

## 🎯 Sonuç: Neden V9 Seçildi?

V9, tek başına en yüksek F1 veya en düşük FN'e sahip değildir. Ancak **birden fazla boyutta** en dengeli ve savunulabilir modeldir:

| Kriter | V9 Durumu |
|---|---|
| Sızıntısız (Leak-Free) | ✅ Kesinlikle |
| En yüksek MCC | ✅ **0.5594** (proje tarihinin rekoru) |
| En yüksek PR-AUC | ✅ **0.9268** (V7 ile eşit) |
| Klinik eşik (0.20) → Recall ≥ %92.7 | ✅ Var |
| FN analizi + Risk bayrakları entegre | ✅ EK7_low_flag, AA_missing_flag |
| Grantham + BLOSUM62 + CAT etkileşim | ✅ En zengin özellik seti |
| Jüriye savunulabilirlik | ✅ En güçlü |

**Gerçek rakip: V7** (F1=0.8959 sızıntısız, MCC=0.5532)
V7, F1 açısından V9'dan +0.0123 daha iyi.
V9, MCC açısından V7'den +0.0062 daha iyi ve PR-AUC eşit.

---

## 📝 Raporda Söylenmesi Gereken Doğru Cümle

> *"V7 sızıntısız pipeline içinde en yüksek F1 skoruna (0.8959) ulaşmış olsa da, V9'da entegre edilen Grantham+BLOSUM62+CAT etkileşim özellikleri ve yeniden yapılandırılan Optuna süreci sayesinde MCC 0.5532'den 0.5594'e yükselmiş ve PR-AUC 0.9268 düzeyinde korunmuştur. Şartnamede belirlenen asimetrik test yapısı (Klinik Stres Testi) göz önüne alındığında, sınıf ayrıştırma kapasitesini en iyi temsil eden metrik olarak MCC'de elde edilen bu gelişme V9'un final model olarak seçilmesinin temel gerekçesini oluşturmaktadır."*
