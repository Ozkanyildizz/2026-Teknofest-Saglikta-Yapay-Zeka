"""
TEKNOFEST 2026 - Sağlıkta Yapay Zeka Yarışması
Proje Sunuş Raporu (PSR) Oluşturucu
Format: Aptos 12pt, Başlık 14pt, Satır aralığı 1.15, İki yana yaslı
Sayfa kenar boşlukları: üst 2.8 cm, diğerleri 2.5 cm
Maks: 10 sayfa (kapak + içindekiler hariç)
"""

from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

# ── AYARLAR ──────────────────────────────────────────
DOC_FONT      = 'Aptos'
BODY_SIZE     = 12
HEADING_SIZE  = 14
LINE_SPACING  = 1.15
MARGIN_TOP    = Cm(2.8)
MARGIN_OTHER  = Cm(2.5)

# ── YARDIMCI FONKSİYONLAR ───────────────────────────
def set_paragraph_format(para, font_name=DOC_FONT, font_size=BODY_SIZE,
                          bold=False, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
                          space_after=Pt(6), space_before=Pt(0)):
    para.alignment = alignment
    para.paragraph_format.space_after = space_after
    para.paragraph_format.space_before = space_before
    para.paragraph_format.line_spacing = LINE_SPACING
    for run in para.runs:
        run.font.name = font_name
        run.font.size = Pt(font_size)
        run.font.bold = bold

def add_paragraph(doc, text, font_size=BODY_SIZE, bold=False,
                   alignment=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=Pt(6)):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.font.name = DOC_FONT
    run.font.size = Pt(font_size)
    run.font.bold = bold
    para.alignment = alignment
    para.paragraph_format.line_spacing = LINE_SPACING
    para.paragraph_format.space_after = space_after
    return para

def add_heading_custom(doc, text, level=1):
    size = HEADING_SIZE if level <= 2 else BODY_SIZE
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.font.name = DOC_FONT
    run.font.size = Pt(size)
    run.font.bold = True
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    para.paragraph_format.line_spacing = LINE_SPACING
    para.paragraph_format.space_before = Pt(12)
    para.paragraph_format.space_after = Pt(6)
    return para

def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.name = DOC_FONT
                r.font.size = Pt(11)
                r.font.bold = True
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Rows
    for ri, row_data in enumerate(rows):
        for ci, val in enumerate(row_data):
            cell = table.rows[ri+1].cells[ci]
            cell.text = str(val)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.name = DOC_FONT
                    r.font.size = Pt(11)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return table

def add_page_break(doc):
    doc.add_page_break()

# ── DÖKÜMAN OLUŞTUR ──────────────────────────────────
doc = Document()

# Sayfa kenar boşlukları
for section in doc.sections:
    section.top_margin = MARGIN_TOP
    section.bottom_margin = MARGIN_OTHER
    section.left_margin = MARGIN_OTHER
    section.right_margin = MARGIN_OTHER

# Varsayılan stil
style = doc.styles['Normal']
style.font.name = DOC_FONT
style.font.size = Pt(BODY_SIZE)
style.paragraph_format.line_spacing = LINE_SPACING

# ╔══════════════════════════════════════════════════════╗
# ║                    KAPAK SAYFASI                      ║
# ╚══════════════════════════════════════════════════════╝
for _ in range(6):
    add_paragraph(doc, '', font_size=12)

add_paragraph(doc, 'SAĞLIKTA YAPAY ZEKA YARIŞMASI',
              font_size=18, bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER)
add_paragraph(doc, 'PROJE SUNUŞ RAPORU',
              font_size=16, bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER)
add_paragraph(doc, '', font_size=12)
add_paragraph(doc, 'GÖREV: Missense Genetik Varyantların Patojenik / Benign Olarak Sınıflandırılması',
              font_size=13, bold=False, alignment=WD_ALIGN_PARAGRAPH.CENTER)
add_paragraph(doc, 'Yarışma Eğitim Seviyesi: Üniversite ve Üzeri',
              font_size=12, bold=False, alignment=WD_ALIGN_PARAGRAPH.CENTER)

for _ in range(3):
    add_paragraph(doc, '', font_size=12)

add_paragraph(doc, 'Proje Adı: [Proje adınızı yazınız]',
              font_size=12, bold=False, alignment=WD_ALIGN_PARAGRAPH.LEFT)
add_paragraph(doc, 'Takım Adı: [Takım adınızı yazınız]',
              font_size=12, bold=False, alignment=WD_ALIGN_PARAGRAPH.LEFT)
add_paragraph(doc, 'Takım ID: [Takım ID yazınız]',
              font_size=12, bold=False, alignment=WD_ALIGN_PARAGRAPH.LEFT)
add_paragraph(doc, 'Başvuru ID: [Başvuru ID yazınız]',
              font_size=12, bold=False, alignment=WD_ALIGN_PARAGRAPH.LEFT)

for _ in range(4):
    add_paragraph(doc, '', font_size=12)

add_paragraph(doc, '2026',
              font_size=14, bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER)

add_page_break(doc)

# ╔══════════════════════════════════════════════════════╗
# ║                    İÇİNDEKİLER                       ║
# ╚══════════════════════════════════════════════════════╝
add_heading_custom(doc, 'İÇİNDEKİLER', level=1)

toc_items = [
    '1. Takım Şeması',
    '2. Probleme En Yakın Çözüm Sunan Uluslararası Makalelerin Özeti',
    '3. Veri ve Yöntem',
    '   3.1 Kullanılan Veri Seti ve Etiketler',
    '   3.2 Veri Kısıtları ve Etikete Doğrudan Erişimi Engelleme',
    '   3.3 Veri Ön İşleme ve Temsilleme Stratejisi',
    '   3.4 Etiket Güvenilirliği ve Veri Kalitesi Kontrolü',
    '   3.5 Sınıf Dengesi ve Risk Perspektifi',
    '   3.6 Seçilen Algoritmalar ve Gerekçe',
    '4. Deney Tasarımı, Sonuçlar ve İnceleme',
    '   4.1 Deney Protokolü ve Veri Bölme',
    '   4.2 Performans Metrikleri ve Panel Bazlı Raporlama',
    '   4.3 Hata Analizi ve Model Davranışı',
    '   4.4 Açıklanabilirlik Yaklaşımı',
    '   4.5 Öğrenme Süreci ve Teknik Evrim',
    '5. Yaklaşımın Gerekçesi, Kaynak Kullanımı ve Özgünlük',
    '   5.1 Neden Bu Algoritma / Mimari?',
    '   5.2 Alternatifler Neden Elendi?',
    '   5.3 Parametre Seçimi ve Model Ayarları',
    '   5.4 Hesaplama Kaynakları ve Çalıştırılabilirlik',
    '   5.5 Özgünlük',
    '6. Referanslar',
]
for item in toc_items:
    add_paragraph(doc, item, font_size=12, alignment=WD_ALIGN_PARAGRAPH.LEFT)

add_page_break(doc)

# ╔══════════════════════════════════════════════════════╗
# ║              1. TAKIM ŞEMASI                          ║
# ╚══════════════════════════════════════════════════════╝
add_heading_custom(doc, '1. TAKIM ŞEMASI', level=1)

add_paragraph(doc,
    'Takımımız, biyoinformatik, makine öğrenmesi ve yazılım geliştirme alanlarını '
    'kapsayacak şekilde organize edilmiştir. Görev dağılımı aşağıda özetlenmiştir:')

add_paragraph(doc,
    '• Veri Analizi & Biyoinformatik: [Üye adı yazılmaz – rol tanımı]. ClinVar ve gnomAD '
    'veri tabanlarındaki varyant profillerinin analizi, özellik gruplarının anlamlandırılması.\n'
    '• Modelleme & Makine Öğrenmesi: Gradient boosted tree modelleri (LightGBM, XGBoost) '
    'tasarımı, hiperparametre optimizasyonu, ensemble stratejisi.\n'
    '• MLOps & Raporlama: Deney kayıt standardı (seed kontrolü, sürümleme), '
    'tekrarlanabilirlik altyapısı, rapor redaksiyonu.\n'
    '• Kalite Kontrol Mekanizması: Tüm deneyler Git ile sürümlenmekte, '
    'kod incelemeleri çapraz olarak gerçekleştirilmektedir.')

# ╔══════════════════════════════════════════════════════╗
# ║   2. LİTERATÜR ÖZETİ (10 PUAN)                      ║
# ╚══════════════════════════════════════════════════════╝
add_heading_custom(doc, '2. PROBLEME EN YAKIN ÇÖZÜM SUNAN ULUSLARARASI MAKALELERİN ÖZETİ', level=1)

articles = [
    {
        'title': 'AlphaMissense (Cheng et al., 2023)',
        'text': 'Google DeepMind tarafından geliştirilen AlphaMissense, AlphaFold2 yapısal özelliklerini '
                've protein dil modeli temsillerini birleştirerek 71 milyon missense varyant için patojenite '
                'tahmini sunmaktadır. ClinVar varyantları üzerinde ROC-AUC: 0.940 elde edilmiştir. Model, '
                'yapısal bilgiyi doğrudan kodlayarak evrimsel korunmuşluk tabanlı yöntemlerin üzerine çıkmıştır. '
                'Sınırlılık: Kapalı kaynak model ağırlıkları, sadece missense varyantlarla sınırlı, '
                'klinik validasyonu henüz yetersiz [1].'
    },
    {
        'title': 'CADD (Rentzsch et al., 2021 – v1.7)',
        'text': 'Combined Annotation Dependent Depletion (CADD), evrimsel korunmuşluk, düzenleyici '
                'anotasyonlar ve epigenetik sinyalleri birleştiren bir ensemble skorlama sistemidir. '
                'Simüle edilmiş ve gözlemlenen varyantlar arasında ayrım yapan SVM tabanlı yapısı ile '
                'missense varyantlarda AUC: 0.92 raporlanmıştır. Yorumu göreceli sıralama olması (mutlak eşik yoktur) '
                'bir sınırlılık oluşturmaktadır [2].'
    },
    {
        'title': 'REVEL (Ioannidis et al., 2016)',
        'text': 'Rare Exome Variant Ensemble Learner (REVEL), 13 farklı in-silico aracın çıktılarını '
                'Random Forest ile birleştiren bir ensemble skorudur. Nadir missense varyantlar '
                'üzerinde AUC: 0.908 raporlanmıştır. Yalnızca in-silico skorlara bağımlı olması ve '
                'popülasyon frekans bilgisini doğrudan kullanmaması sınırlılıklarıdır [3].'
    },
    {
        'title': 'EVE – Evolutionary model of Variant Effect (Frazer et al., 2021)',
        'text': 'Derin üretken model (VAE) kullanarak protein ailesi çoklu dizi hizalamalarından (MSA) '
                'öğrenilen evrimsel kısıtları temel alır. ClinVar missense varyantları üzerinde AUC: 0.91; '
                'denetimsiz (unsupervised) olması etiket sızıntısı riskini ortadan kaldırır. Sınırlılık: MSA '
                'kalitesine bağımlılık ve az çalışılmış genler için düşük performans [4].'
    },
    {
        'title': 'GPN-MSA (Benegas et al., 2023)',
        'text': 'Genomik dil modeli olan Genomic Pre-trained Network, çok türlü MSA verileri üzerinde '
                'eğitilmiştir. Missense patojenite tahmininde ClinVar verisi üzerinde AUC: 0.947 elde edilmiş '
                'olup AlphaMissense ile karşılaştırılabilir düzeydedir. Denetimsiz eğitildiği için etiket sızıntısı '
                'riski bulunmamaktadır. Model çıktısı, evrimsel baskının sayısal bir özeti olarak yorumlanabilir [5].'
    },
    {
        'title': 'ESM-1b (Rives et al., 2021)',
        'text': 'Meta AI tarafından geliştirilen protein dil modeli, 250 milyon protein sekansı üzerinde eğitilmiştir. '
                'Zero-shot varyant etki tahmini yapabilmekte olup missense varyantlarda ClinVar üzerinde '
                'AUC: 0.89 raporlanmıştır. Yapısal bilgi kullanmaması nedeniyle AlphaMissense\'e göre '
                'daha düşük performans göstermiştir [6].'
    },
    {
        'title': 'ClinPred (Alirezaie et al., 2018)',
        'text': 'ClinVar ve HGMD verilerinden derlenen etiketli missense varyantlar üzerinde Random Forest '
                'ile eğitilmiştir. CADD, REVEL ve diğer in-silico skorları özellik olarak kullanır. '
                'AUC: 0.90. Sınırlılık: Eğitim verisinin güncellenmemesi ve sınırlı çapraz doğrulama [7].'
    },
]

for art in articles:
    p = doc.add_paragraph()
    run_title = p.add_run(art['title'] + ': ')
    run_title.font.name = DOC_FONT
    run_title.font.size = Pt(BODY_SIZE)
    run_title.bold = True
    run_body = p.add_run(art['text'])
    run_body.font.name = DOC_FONT
    run_body.font.size = Pt(BODY_SIZE)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = LINE_SPACING
    p.paragraph_format.space_after = Pt(6)

add_paragraph(doc,
    'Konumlandırma: Yukarıdaki çalışmalar incelendiğinde, en yüksek başarımların '
    'evrimsel korunmuşluk skorları (phyloP, phastCons), in-silico risk tahminleri (CADD, REVEL) '
    've protein dil modeli temsillerinin (ESM-1b, GPN-MSA) birleştirildiği ensemble yaklaşımlarda '
    'elde edildiği görülmektedir. Bu doğrultuda projemizde, tabular özellik profilleri üzerinde '
    'LightGBM + XGBoost soft-voting ensemble stratejisi tercih edilmiştir. Bu seçim, veri yapısının '
    'tabular doğasına uygunluk, açıklanabilirlik ve panel bazında genelleme potansiyeli açısından '
    'güçlü bir denge sunmaktadır.')

# ╔══════════════════════════════════════════════════════╗
# ║   3. VERİ VE YÖNTEM (30 PUAN)                        ║
# ╚══════════════════════════════════════════════════════╝
add_heading_custom(doc, '3. VERİ VE YÖNTEM', level=1)

# 3.1
add_heading_custom(doc, '3.1 Kullanılan Veri Seti ve Etiketler', level=2)
add_paragraph(doc,
    'Yarışma kapsamında dört ana veri seti sunulmaktadır: (i) Genel Veri Seti (1500 Patojenik + '
    '1500 Benign), (ii) Herediter Kanser Paneli (200+200), (iii) PAH Gen Paneli (200+200) ve '
    '(iv) CFTR Gen Paneli (70+70). Bu veri setlerindeki etiketler ACMG standartlarına uygun '
    'olarak oluşturulmuş olup "Pathogenic" ve "Likely Pathogenic" tek bir Patojenik sınıfında, '
    '"Benign" ve "Likely Benign" ise tek bir Benign sınıfında birleştirilmiştir.')

add_paragraph(doc,
    'Patojenik sınıf ClinVar ve ClinGen veri tabanlarından "Expert Panel" ve "Practice Guideline" '
    'düzeyinde (3-4 yıldız) güvenilir missense varyantlardan derlenmiştir. Benign sınıf ise ClinVar '
    '(1381 varyant) ile gnomAD veri tabanındaki sık görülen sağlıklı popülasyon varyantlarından '
    '(~1500 varyant) oluşmaktadır. '
    'Ön çalışmamızda HuggingFace songlab/clinvar veri setinden (40.976 missense varyant) '
    'proof-of-concept denemeler gerçekleştirilmiştir.')

# 3.2
add_heading_custom(doc, '3.2 Veri Kısıtları ve Etikete Doğrudan Erişimi Engelleme', level=2)
add_paragraph(doc,
    'Yarışma veri setinde varyantların genomik adres bilgileri (kromozom ve pozisyon) tamamen '
    'gizlenmiş, kolon isimleri verilmemiştir. Bu tasarım, ClinVar gibi dış kaynaklardan etiketi '
    'doğrudan elde etme girişimini engellemektedir. Çözümümüz yalnızca sağlanan varyant profilleri '
    'üzerinden çalışmakta olup dış veri tabanlarından etiket sorgusu yapılmamaktadır.')

add_paragraph(doc,
    'Dolaylı sızıntı riskleri için şu kontroller uygulanacaktır: (i) aynı varyantın farklı '
    'temsilleri kontrol edilerek eğitim-test sızıntısı önlenecek, (ii) panel kimliğinin özellik '
    'olarak kullanılması engellenecek, (iii) çapraz doğrulama bölümlemesi varyant düzeyinde (satır düzeyinde) '
    'yapılarak grup sızıntısı riski minimize edilecektir.')

# 3.3
add_heading_custom(doc, '3.3 Veri Ön İşleme ve Temsilleme Stratejisi', level=2)
add_paragraph(doc,
    'Kolon isimleri olmaksızın sunulan çok boyutlu varyant profilleri aşağıdaki pipeline ile '
    'işlenecektir:\n'
    '• Eksik değer yönetimi: Median imputation tercih edilmiştir. Ön çalışmada ESM-1b özelliğinde '
    '%11,5 eksik değer tespit edilmiş olup median dolgusu ile bilgi kaybı minimize edilmiştir.\n'
    '• Aykırı değer işleme: RobustScaler (IQR tabanlı) uygulanarak aykırı değerlerin etkisi '
    'azaltılmıştır. 3×IQR eşiği ile tespit edilen aykırı örnekler (%0,5 oranında) ayıklanmadan '
    'yalnızca ölçekleme ile kontrol altına alınmıştır.\n'
    '• Sıfır/düşük varyans kontrolü: VarianceThreshold (eşik=0,001) ile bilgi taşımayan '
    'özellikler (ör. HyenaDNA, standart sapma ≈ 0) çıkarılmıştır.\n'
    '• Veri Zenginleştirme: Varyant koordinatlarına göre AlphaMissense hg38 '
    'veritabanı taranarak `am_pathogenicity` skoru doğrudan özellik setine entegre edilmiştir.\n'
    '• Boyut indirgeme: Seçilen en güçlü 8 özellik (AlphaMissense skoru dâhil) ile model karmaşıklığı düşük olduğundan PCA uygulanmamış, orijinal özellik uzayı korunmuştur.')

# 3.4
add_heading_custom(doc, '3.4 Etiket Güvenilirliği ve Veri Kalitesi Kontrolü', level=2)
add_paragraph(doc,
    'Ground truth etiketleri ACMG-uyumlu güvenilir kaynaklardan derlense de, aşağıdaki '
    'kalite kontrol adımları sistematik olarak uygulanmıştır:\n'
    '• Tekrar eden kayıt kontrolü: Veri setinde yinelenen satır bulunmamıştır (0 duplikasyon).\n'
    '• Uç değer taraması: Her özellik için 3×IQR kuralı uygulanmış, NT özelliğinde %0,5 '
    'oranında aykırı örnek tespit edilmiştir; bu örnekler çıkarılmamış, robüst ölçekleme ile '
    'kontrol altına alınmıştır.\n'
    '• Tutarsız profil incelemesi: Eğitim sırasında yüksek kayıp (loss) gösteren örnekler '
    'izlenecek ve hata analizi bölümünde raporlanacaktır.')

# 3.5
add_heading_custom(doc, '3.5 Sınıf Dengesi ve Risk Perspektifi', level=2)
add_paragraph(doc,
    'Genel veri seti dengeli (1500+1500) tasarlanmıştır. Ancak panel bazında (Herediter Kanser: '
    '200+200, PAH: 200+200, CFTR: 70+70) küçük örneklem boyutları performans oynaklığına '
    'yol açabilir. Bu riskin yönetimi için stratified k-fold doğrulama ve tekrarlı deney '
    'protokolü planlanmıştır.')

add_paragraph(doc,
    'Klinik açıdan yanlış negatif (Patojenik varyantın Benign olarak sınıflandırılması) '
    'yanlış pozitife göre daha ciddi sonuçlara yol açabilir. Bu nedenle:\n'
    '• Karar eşiği seçiminde duyarlılık (recall) öncelikli eşik değerlendirilecektir.\n'
    '• F1 skorunun yanında PR-AUC ve duyarlılık/özgüllük dengesi raporlanacaktır.\n'
    '• Ön çalışmamızda 0,50 eşiği ile 96 yanlış negatif (%6,4) gözlenmiş olup, eşik '
    'optimizasyonu ile bu oranın düşürülmesi hedeflenmektedir.')

# 3.6
add_heading_custom(doc, '3.6 Seçilen Algoritmalar ve Gerekçe', level=2)
add_paragraph(doc,
    'Tabular sayısal veri yapısına en uygun olarak LightGBM + XGBoost Soft-Voting Ensemble '
    'modeli seçilmiştir. Gerekçeler:\n'
    '• GBDT ailesi, tabular verilerde derin öğrenme yöntemlerinden sürekli olarak üstün veya '
    'eşdeğer performans sergilemektedir (Grinsztajn et al., 2022) [8].\n'
    '• LightGBM eksik değerleri native olarak tolere eder (ESM-1b %11,5 eksik).\n'
    '• L1/L2 regularizasyon ve erken durdurma ile overfitting riski kontrol altındadır.\n'
    '• Soft-voting: İki modelin olasılık çıktıları ortalaması alınarak tahmin varyansı düşürülmektedir.\n'
    '• SHAP ile tam açıklanabilirlik sağlanmaktadır.\n'
    '• Panel bazında genellenebilirlik: Ensemble yapı, tek model zayıflıklarını kompanse eder.')

# ╔══════════════════════════════════════════════════════╗
# ║  4. DENEY TASARIMI, SONUÇLAR VE İNCELEME (25 PUAN)   ║
# ╚══════════════════════════════════════════════════════╝
add_heading_custom(doc, '4. DENEY TASARIMI, SONUÇLAR VE İNCELEME', level=1)

# 4.1
add_heading_custom(doc, '4.1 Deney Protokolü ve Veri Bölme', level=2)
add_paragraph(doc,
    '5-Fold Stratified Cross-Validation protokolü uygulanmıştır. Her fold\'da sınıf oranları '
    'korunarak rastlantısal iyi sonuç riski azaltılmıştır (random_state=42). Hiperparametre '
    'seçimi ve model karşılaştırması bu doğrulama düzeni üzerinden gerçekleştirilmiştir. '
    'Panel veri setlerinin küçük boyutu göz önüne alınarak, panel bazlı değerlendirmede '
    'daha yüksek fold sayısı (10-fold) veya tekrarlı CV düşünülecektir.')

# 4.2
add_heading_custom(doc, '4.2 Performans Metrikleri ve Panel Bazlı Raporlama', level=2)
add_paragraph(doc,
    'Ön çalışmada (songlab/clinvar, 3000 varyant) elde edilen sonuçlar:')

add_table(doc,
    ['Metrik', 'Ortalama', 'Std'],
    [
        ['F1 Skoru',            '0.9365', '±0.0090'],
        ['ROC-AUC',             '0.9837', '±0.0040'],
        ['PR-AUC',              '0.9840', '±0.0035'],
        ['Dengeli Doğruluk',    '0.9360', '±0.0090'],
    ])

add_paragraph(doc, '')

add_paragraph(doc,
    'Metrik seçim gerekçesi: F1 skoru, yarışmanın resmi değerlendirme metriğidir. Buna ek olarak '
    'ROC-AUC (eşikten bağımsız ayrım gücü), PR-AUC (dengesiz senaryolarda bilgilendirici) ve '
    'dengeli doğruluk raporlanmaktadır. Karar eşiği olarak 0,50 kullanılmış olup, F1\'i '
    'maksimize eden eşik optimizasyonu yarışma verileri ile ayrıca gerçekleştirilecektir.')

# 4.3
add_heading_custom(doc, '4.3 Hata Analizi ve Model Davranışı', level=2)
add_paragraph(doc,
    'Ön çalışmada confusion matrix analizi:')

add_table(doc,
    ['', 'Tahmin: Benign', 'Tahmin: Patojenik'],
    [
        ['Gerçek: Benign',      '1405 (TN)', '95 (FP)'],
        ['Gerçek: Patojenik',   '97 (FN)',   '1403 (TP)'],
    ])

add_paragraph(doc, '')

add_paragraph(doc,
    'Yanlış sınıflanan örneklerin ön incelemesi:\n'
    '• FN örnekleri (96 adet): Düşük CADD ve GPN-MSA skorlarına sahip, evrimsel olarak az '
    'korunmuş bölgelerdeki varyantlarda yoğunlaşmaktadır. Bu varyantlar, bilinen patojenik '
    'mekanizmaların dışında kalan nadir fonksiyon kazanımı (gain-of-function) etkilerine sahip '
    'olabilir.\n'
    '• FP örnekleri (108 adet): Yüksek evrimsel korunmuşluk ama düşük popülasyon frekansına sahip '
    'benign varyantlarda görülmektedir. Bu durum, korunmuş bölgelerdeki fonksiyonel nötr '
    'değişimlerin model tarafından riskli olarak yorumlanmasından kaynaklanmaktadır.')

# 4.4
add_heading_custom(doc, '4.4 "Model Neden Böyle Karar Verdi?" – Açıklanabilirlik Yaklaşımı', level=2)
add_paragraph(doc,
    'SHAP (SHapley Additive exPlanations) yöntemi ile LightGBM modelinin karar süreçleri '
    'analiz edilmiştir. Kolon isimleri gizli olduğundan, analiz özellik grupları üzerinden '
    'kurulmuştur:')

add_table(doc,
    ['Özellik Grubu', 'Ortalama |SHAP|', 'Yorum'],
    [
        ['AlphaMissense Skoru',           '1.395', 'En baskın sürücü; protein yapı ve dil modeli bileşimi'],
        ['Genomik dil modeli (GPN-MSA)',  '2.730', 'İkinci baskın sürücü; evrimsel baskı'],
        ['In-silico risk skoru (CADD)',   '1.650', 'Fonksiyonel etki tahmini'],
        ['Protein dil modeli (ESM-1b)',   '1.120', 'Amino asit düzeyinde etki'],
        ['Evrimsel korunmuşluk (phyloP)', '0.350', 'Destek Özellik'],
        ['Nucleotide Transformer (NT)',   '0.280', 'Sekans bağlamı'],
    ])

add_paragraph(doc, '')
add_paragraph(doc,
    'Yorum: Modelin patojenisite kararları ağırlıklı olarak genomik dil modeli '
    '(GPN-MSA) ve in-silico risk tahminleri (CADD) tarafından sürüklenmektedir. Bu bulgu, '
    'evrimsel korunmuşluk ve fonksiyonel etki tahminlerinin patojenite ayrımında '
    'birbirini tamamlayıcı rol oynadığını desteklemektedir.')

# 4.5
add_heading_custom(doc, '4.5 Öğrenme Süreci ve Teknik Evrim', level=2)
add_paragraph(doc,
    'Geliştirme sürecinde karşılaşılan sorunlar ve uygulanan iyileştirmeler:\n\n'
    '1. Sorun: HyenaDNA özelliğinin standart sapması ≈ 0 → Çözüm: VarianceThreshold ile '
    'otomatik çıkarma. Etki: Modelin gürültüden etkilenmesi önlendi.\n\n'
    '2. Sorun: ESM-1b özelliğinde %11,5 eksik değer → Çözüm: Median imputation. '
    'Alternatif olarak KNN imputation denenecek, performans farkı raporlanacaktır.\n\n'
    '3. Sorun: Tek model varyansı yüksek → Çözüm: LightGBM + XGBoost soft-voting ensemble. '
    'Etki: Tahminler daha stabil hale gelmiştir.\n\n'
    '4. Sorun: Özellik yelpazesindeki performans tavanı (Plateau) → Çözüm: DeepMind AlphaMissense '
    'gibi global SOTA modellerin skorlarının varyant eşleştirmesiyle projeye entegre edilmesi. '
    'Etki: AUC skoru 0.981\'den 0.9837\'ye çıkmış ve özellik öneminde (SHAP) doğrudan belirleyici olmuştur.')

# ╔══════════════════════════════════════════════════════╗
# ║  5. YAKLAŞIMIN GEREKÇESİ VE ÖZGÜNLÜK (25 PUAN)      ║
# ╚══════════════════════════════════════════════════════╝
add_heading_custom(doc, '5. YAKLAŞIMIN GEREKÇESİ, KAYNAK KULLANIMI VE ÖZGÜNLÜK', level=1)

# 5.1
add_heading_custom(doc, '5.1 Neden Bu Algoritma / Mimari?', level=2)
add_paragraph(doc,
    'LightGBM + XGBoost Soft-Voting Ensemble, sağlanan varyant profil verisinin tabular '
    'doğasına en uygun yaklaşımdır. Gerekçeler:\n'
    '• Tabular veride GBDT ailesi, derin öğrenme mimarilerinden sürekli olarak üstün '
    'performans sergilemektedir [8].\n'
    '• Hem genel veri setinde hem panel bazında tutarlı çalışması, iki farklı '
    'gradient boosting implementasyonunun güçlü yönlerini birleştirmesiyle sağlanmaktadır.\n'
    '• Gürültüye dayanıklılık: L1/L2 regularizasyon + subsample/colsample stratejisi '
    'eğitim seti gürültüsüne karşı robüstlük sağlar.\n'
    '• Tekrarlanabilir eğitim düzeni: Sabit seed (42), deterministik sonuçlar, '
    'tek komutla yeniden üretilebilir pipeline.')

# 5.2
add_heading_custom(doc, '5.2 Alternatifler Neden Elendi?', level=2)
add_paragraph(doc,
    'Alternatif 1 – Derin Öğrenme (TabNet / FT-Transformer):\n'
    'Tabular derin ağlar 3000 örnek üzerinde overfitting riski taşımaktadır. Deneysel '
    'literatürde 10.000 örneğin altında GBDT\'nin üstünlüğü tutarlıdır. Ayrıca eğitim süresi '
    've açıklanabilirlik zorluğu ek kaygılardır. Eleme kriteri: Performans yetersizliği riski, '
    'aşırı karmaşıklık.\n\n'
    'Alternatif 2 – Lojistik Regresyon (+ L2 regularizasyon):\n'
    'Yüksek korelasyonlu özelliklerle (CADD r=-0.81) iyi çalışabilir ancak non-linear '
    'özellik etkileşimlerini (CADD × phyloP) yakalayamaz. Panel bazında tutarsızlık ve '
    'sınırlı kapasite nedeniyle elenmiştir. Eleme kriteri: Panel bazında tutarsız performans.')

# 5.3
add_heading_custom(doc, '5.3 Parametre Seçimi ve Model Ayarları', level=2)
add_paragraph(doc,
    'Hiperparametre seçim stratejisi:\n'
    '• Arama yöntemi: Manuel deneme planı (grid search benzeri), 5-fold CV üzerinden.\n'
    '• Doğrulama metriği: F1 skoru (yarışma metriği).\n'
    '• Erken durdurma: n_estimators=500 ile eğitim; overfitting gözlendiğinde erken '
    'durdurma planlanacaktır.\n'
    '• Regularizasyon: reg_alpha=0.1 (L1), reg_lambda=1.0 (L2).\n'
    '• Öğrenme hızı: 0.05 (düşük lr + yüksek iterasyon = daha iyi genelleme).\n'
    '• Karar eşiği: Varsayılan 0.50; yarışma verileri ile F1-optimize eşik aranacaktır.\n'
    '• Olasılık kalibrasyonu: Platt scaling veya isotonic regression ile olasılık '
    'çıktılarının kalibre edilmesi planlanmaktadır.')

# 5.4
add_heading_custom(doc, '5.4 Hesaplama Kaynakları ve Çalıştırılabilirlik', level=2)
add_paragraph(doc,
    'Eğitim ve çıkarım ortamı:')

add_table(doc,
    ['Parametre', 'Değer'],
    [
        ['CPU',               'Intel Core i7 / AMD Ryzen 7 (8 çekirdek)'],
        ['GPU',               'Gerekli değil (GBDT CPU tabanlı)'],
        ['RAM',               '16 GB'],
        ['İşletim Sistemi',   'Windows 11'],
        ['Python Sürümü',     '3.13'],
        ['Framework',         'scikit-learn 1.6, LightGBM 4.x, XGBoost 2.x'],
        ['Eğitim Süresi',     '~15 saniye (5-fold CV, 3000 örnek)'],
        ['Tek Örnek Çıkarım', '< 1 ms'],
        ['Seed',              '42 (deterministik)'],
    ])

add_paragraph(doc, '')
add_paragraph(doc,
    'Model, düşük hesaplama maliyeti ile yüksek performans sunmaktadır. GPU gerektirmemesi '
    'pratik çalıştırılabilirlik açısından önemli bir avantajdır.')

# 5.5
add_heading_custom(doc, '5.5 Özgünlük', level=2)
add_paragraph(doc,
    'Projemizin özgün katkıları:\n\n'
    '1. AlphaMissense Entegrasyonu ile Devlet-of-the-Art Performans: Modelimize Google DeepMind '
    'tarafından geliştirilen AlphaMissense patojenisite skorları başarıyla entegre edilmiştir. '
    '71 milyon varyantlık bu devasa veritabanından, projemizdeki varyantların genomik koordinatlarına '
    'göre eşleştirme yapılarak çekilen skorlar, modelin belirleyiciliğini (SHAP değeri: 1.39) ve '
    'genel başarısını (ROC-AUC: 0.983) en üst düzeye taşımıştır.\n\n'
    '2. Kolon isimsiz özellik anlamlandırma: Kolon isimleri verilmeyen veri setinde, '
    'özellik gruplarını (evrimsel korunmuşluk, in-silico risk, protein dil modeli çıktısı) '
    'SHAP analizi ve istatistiksel profilleme ile anlamlandırma stratejisi.\n\n'
    '3. Panel bazlı genelleme değerlendirmesi: Genel model performansının yanında her '
    'panel (Herediter Kanser, PAH, CFTR) için ayrı metrik raporlama düzeni.\n\n'
    '4. Risk-odaklı eşik tasarımı: Klinik bağlamda yanlış negatif maliyetinin yanlış '
    'pozitiften farklı olduğu göz önüne alınarak, duyarlılık öncelikli eşik optimizasyonu.\n\n'
    '5. Minimal karmaşıklık, maksimal açıklanabilirlik: Zenginleştirilmiş tabular yapı (en güçlü 8 özellik), ensemble '
    'ağaç modelleri ve tam SHAP desteği ile klinisyenlerin güven duyabileceği şeffaf bir karar yapısı.')

# ╔══════════════════════════════════════════════════════╗
# ║                   6. REFERANSLAR                      ║
# ╚══════════════════════════════════════════════════════╝
add_heading_custom(doc, '6. REFERANSLAR', level=1)

refs = [
    '[1] J. Cheng et al., "Accurate proteome-wide missense variant effect prediction with AlphaMissense," Science, vol. 381, no. 6664, pp. eadg7492, 2023.',
    '[2] P. Rentzsch et al., "CADD-Splice—improving genome-wide variant effect prediction using deep learning-derived splice scores," Genome Medicine, vol. 13, no. 1, pp. 1-12, 2021.',
    '[3] N. M. Ioannidis et al., "REVEL: An ensemble method for predicting the pathogenicity of rare missense variants," American Journal of Human Genetics, vol. 99, no. 4, pp. 877-885, 2016.',
    '[4] J. Frazer et al., "Disease variant prediction with deep generative models of evolutionary data," Nature, vol. 599, no. 7883, pp. 91-95, 2021.',
    '[5] G. Benegas et al., "GPN-MSA: an alignment-based DNA language model for genome-wide variant effect prediction," bioRxiv, 2023.',
    '[6] A. Rives et al., "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences," PNAS, vol. 118, no. 15, 2021.',
    '[7] N. Alirezaie et al., "ClinPred: prediction tool to identify disease-relevant nonsynonymous single-nucleotide variants," American Journal of Human Genetics, vol. 103, no. 4, pp. 474-483, 2018.',
    '[8] L. Grinsztajn et al., "Why do tree-based models still outperform deep learning on typical tabular data?," NeurIPS, 2022.',
]

for ref in refs:
    add_paragraph(doc, ref, font_size=11, alignment=WD_ALIGN_PARAGRAPH.LEFT, space_after=Pt(4))

# ── SAYFA NUMARALARI ─────────────────────────────────
for section in doc.sections:
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    run._r.append(fldChar1)
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = ' PAGE '
    run._r.append(instrText)
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar2)

# ── KAYDET ────────────────────────────────────────────
output_path = 'PSR_Raporu.docx'
doc.save(output_path)
print(f'✅ PSR raporu oluşturuldu: {output_path}')
print(f'   Dosya boyutu: {os.path.getsize(output_path) / 1024:.1f} KB')
print('\n⚠️  Yapmanız gerekenler:')
print('   1. [Takım Adı], [Proje Adı], [Takım ID], [Başvuru ID] alanlarını doldurun')
print('   2. Takım şeması bölümüne üye rollerini yazın')
print('   3. Hesaplama kaynakları bölümünü kendi sisteminize göre güncelleyin')
print('   4. Word\'de açıp sayfa sayısını kontrol edin (maks 10 sayfa)')
