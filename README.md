# IOTN-AC — Benchmarking & Eksperimen Model Skor Aesthetic Component

Prediksi skor **IOTN Aesthetic Component (AC, skala 1–10)** dari foto frontal intraoral,
untuk dipakai on-device di aplikasi iOS **Malokit**. Repo ini menyimpan seluruh riwayat
eksperimen — dari baseline pertama sampai model yang akhirnya dipakai di produksi — plus
beberapa notebook pendukung untuk fitur lain di sekitar AC (segmentasi gigi, penomoran FDI,
validasi foto per-view, captioning).

Dokumen ini merangkum **alur eksperimen dari notebook pertama sampai keputusan akhir
memakai `10_ac_multiannotator_resnet.ipynb`** sebagai satu-satunya notebook yang
menghasilkan model produksi.

---

## 1. Alur Eksperimen — Notebook 1 sampai 10

### Ringkasan singkat

Semua percobaan di notebook 1–9 memakai **arsitektur transfer learning ResNet18** (atau
variannya) sebagai acuan, dan mencoba berbagai perubahan — head ordinal, training dari nol,
domain pretraining, arsitektur multi-task — untuk melihat apakah ada yang mengalahkan
baseline. **Tidak ada satu pun yang mengalahkannya secara arsitektural.** Yang akhirnya
membuat perbedaan besar bukan arsitektur baru, melainkan **perbaikan fundamental pada data
dan metodologi evaluasi** di notebook 10: label dari 3 annotator (bukan 1 sumber label yang
tidak terverifikasi), ceiling kesepakatan manusia sebagai tolok ukur realistis, dan
cross-validation 5-fold untuk estimasi performa yang bisa dipercaya.

> **Catatan penting soal angka.** Notebook 1–9 dievaluasi terhadap label lama
> (`new ac report.xlsx`, hasil pra-anotasi otomatis/single-rater, test set 102 foto dari 73
> pasien). Notebook 10 memakai label baru (`Grade AC by Team.xlsx`, 3 annotator manusia,
> test set 106 foto). **Kedua kelompok angka ini tidak boleh dibandingkan langsung** sebagai
> "sebelum vs sesudah" — keduanya diukur terhadap referensi yang berbeda. Yang bisa
> dibandingkan secara adil adalah *urutan* hasil di dalam masing-masing kelompok.

### Tahap demi tahap

**1) `01_ac_pipeline.ipynb` — Baseline transfer learning.**
Membandingkan 7 backbone pretrained ImageNet (mobilenet_v3_small/large, shufflenet_v2,
efficientnet_b0, resnet18/34/50) dengan head linear satu-keluaran (regresi). ResNet18 menang
di validasi, dan setelah kalibrasi jadi hasil terbaik dari seluruh notebook 1–9 di test set:
**MAE 1.333, akurasi ±1 59.8%, QWK 0.714**. Notebook ini juga menetapkan metodologi evaluasi
yang dipakai konsisten di notebook 2–5: MAE, akurasi tepat, akurasi ±1, dan QWK, dibandingkan
terhadap tebakan konstan sebagai sanity check.

**2) `02_ac_pipeline_corn.ipynb` — Head ordinal CORN.**
Hipotesis: karena skor AC bersifat ordinal (bukan kategori lepas), head yang secara eksplisit
memodelkan urutan (CORN) mungkin lebih baik dari regresi biasa. Hasil: **MAE 1.598, QWK
0.608** — lebih buruk dari baseline. Kesimpulan: untuk data sekecil ini, regresi MSE biasa
tetap lebih baik daripada head ordinal yang lebih kompleks.

**3) `03_lightweight_cnn.ipynb` — CNN ringan dari nol.**
Hipotesis: mungkin arsitektur custom yang lebih kecil kurang rawan overfitting dibanding
transfer learning. Hasil: **MAE 1.637, QWK 0.562** — paling buruk kedua. Melatih dari nol
tanpa fitur ImageNet ternyata tidak cukup dengan data sekecil ini (~555 foto).

**4) `04_mini_resnet.ipynb` — Mini-ResNet (skip connection) dari nol.**
Masih dari nol, tapi menambahkan residual connection untuk melihat apakah itu membantu
stabilitas training. Hasil: **MAE 1.588, QWK 0.628** — sedikit lebih baik dari notebook 3,
mengonfirmasi residual connection membantu, tapi tetap kalah dari transfer learning polos di
notebook 1.

**5) `05_transfer_custom_head.ipynb` — Transfer + custom head + fine-tune 2 tahap.**
Kembali ke transfer learning, tapi dengan head kustom (dense+dropout+scaling), fine-tuning
2 tahap, backbone lebih ringan (mobilenet_v3_small), dan opsi memakai bobot hasil domain
pretraining dari notebook 6. Hasil: **MAE 1.775, QWK 0.476** — hasil terburuk dari semua
notebook 1–9. Head yang lebih rumit dan backbone yang lebih ringan sama-sama tidak
membantu dibanding pendekatan sederhana di notebook 1.

**6) `06_pretrain_gingivitis.ipynb` — Domain pretraining backbone.**
Bukan model AC — tujuannya menyiapkan bobot alternatif untuk notebook 5. Backbone
(MobileNetV3-Small) dilatih mengklasifikasi keparahan gingivitis (0–4) dari 1.096 foto
intraoral lain, dengan asumsi ini mengadaptasi fitur visual (warna, pencahayaan, tekstur)
ke domain foto intraoral secara umum. Task proxy ini sendiri sulit (akurasi val terbaik
hanya 14.8%, wajar karena labelnya bising), dan pada akhirnya tidak membuat notebook 5
mengalahkan baseline — domain pretraining terbukti tidak membantu di jalur ini.

**7) `07_eda_gingivitis.ipynb` — EDA dataset gingivitis.**
Eksplorasi dataset pendukung notebook 6 (1.096 foto, 732/182/182). Mengonfirmasi modalitas
mirip (foto frontal intraoral) tapi labelnya menilai gusi, bukan susunan gigi — menjelaskan
kenapa manfaatnya untuk AC terbatas pada adaptasi domain visual, bukan sinyal tugas.

**8) `08_clustering_roi.ipynb` — Clustering tanpa label.**
Mengelompokkan 555 foto OMNI ke 10 cluster berbasis ROI gigi (heuristik warna + region
growing, lalu embedding MobileNetV3). Sifatnya eksploratif — untuk mendeteksi outlier dan
menilai stratifikasi data, bukan model prediktif. ROI heuristik yang dikembangkan di sini
juga jadi dasar pipeline ROI yang dipakai/dibandingkan di notebook 9 dan 10.

**9) `09_ac_frontview_multitask.ipynb` — Arsitektur multi-task-ready + Grad-CAM.**
Langkah arsitektural menuju pipeline IOTN-DHC penuh: satu backbone dengan kepala AC yang
aktif (dilatih di sini, masih memakai label lama single-source) dan kepala kasus MOCDO
(9 komponen) yang **sudah didefinisikan tapi belum dilatih** karena label komponennya belum
ada. Grad-CAM ditambahkan untuk memverifikasi model benar melihat gigi, bukan retraktor atau
latar. Ini bukan percobaan untuk mengalahkan skor AC — ini persiapan struktur untuk fitur
DHC di masa depan.

**10) `10_ac_multiannotator_resnet.ipynb` — Notebook produksi.**
Titik baliknya bukan arsitektur baru (masih ResNet18 transfer learning, sama seperti
notebook 1), melainkan **data dan metodologi evaluasi yang diperbaiki secara mendasar**:

- Label AC sekarang berasal dari **3 annotator manusia** (Nana, Anin, Nico) untuk 555 foto
  penuh, bukan 1 sumber label yang tidak terverifikasi seperti di notebook 1–9.
- Sebelum melatih apa pun, notebook ini mengukur **kesepakatan antar-annotator** (QWK
  berpasangan rata-rata **0.918**) sebagai **ceiling** — batas atas performa yang realistis
  untuk dituju model, bukan skor sempurna.
- Empat target konsensus (mean, median, majority, soft-label) dibandingkan secara sistematis;
  **mean** menang di semua metrik.
- Ensemble 3 model + Test-Time Augmentation memberi perbaikan kecil tapi konsisten.
- **Cross-validation 5-fold** dipakai untuk estimasi performa yang jauh lebih andal
  daripada satu angka dari test set kecil: **OOF QWK 0.838** (dari 449 foto out-of-fold),
  dikonfirmasi oleh ensemble 5-fold di test holdout independen (QWK 0.822).
- Model produksi akhir diretrain di seluruh data + kalibrasi linear + **gerbang
  out-of-distribution** (menolak foto yang jelas bukan foto gigi valid), lalu diekspor ke
  **Core ML** untuk berjalan on-device di aplikasi iOS.

Lihat bagian 2 di bawah untuk detail lengkap notebook 10.

### Tabel perbandingan (notebook 1–5, label lama, test set 102 foto)

| Notebook | Pendekatan | MAE | Akurasi ±1 | QWK |
|---|---|---|---|---|
| 01 | Transfer ResNet18 + head linear (**baseline**) | **1.333** | **59.8%** | **0.714** |
| 02 | Transfer ResNet18 + head ordinal CORN | 1.598 | 52.9% | 0.608 |
| 03 | CNN ringan, dari nol | 1.637 | 55.9% | 0.562 |
| 04 | Mini-ResNet (skip connection), dari nol | 1.588 | 56.9% | 0.628 |
| 05 | Transfer + custom head, fine-tune 2 tahap, backbone ringan | 1.775 | 52.0% | 0.476 |

### Kenapa akhirnya notebook 10 yang dipakai

1. **Semua variasi arsitektur di notebook 2–5 dan 9 gagal mengalahkan baseline resnet18
   sederhana dari notebook 1.** Ini sinyal kuat bahwa arsitektur bukan lagi pembatas utama.
2. **Label di notebook 1–9 tidak pernah diverifikasi tingkat kesepakatannya** — tidak ada
   cara mengetahui berapa banyak dari error model yang sebetulnya berasal dari label yang
   tidak konsisten. Notebook 10 mengatasi ini dengan mengukur ceiling manusia secara
   eksplisit (QWK 0.918) sebelum menilai model.
3. **Evaluasi di notebook 1–9 memakai satu angka dari test set kecil** (102–106 foto) yang
   rawan berubah tergantung sampel. Notebook 10 memvalidasi lewat cross-validation 5-fold,
   jauh lebih bisa dipercaya (OOF QWK 0.838 ± konsisten di test holdout terpisah).
4. **Notebook 10 satu-satunya yang menghasilkan pipeline produksi lengkap** — kalibrasi,
   gerbang OOD, dan ekspor Core ML 2-keluaran (skor + vektor fitur) yang benar-benar dipakai
   `ACGraderRegressor` di aplikasi iOS Malokit.

Kesimpulan notebook 10 sendiri: model terbaik (ResNet18 + target mean) mendekati tapi masih
di bawah ceiling manusia (OOF QWK 0.838 vs 0.918) — gap yang wajar untuk dataset sekecil 555
foto. Peningkatan lebih lanjut kemungkinan besar butuh **lebih banyak data berlabel**, bukan
arsitektur baru.

---

## 2. Model Produksi (Notebook 10) — Detail

**Arsitektur:** ResNet18 (transfer learning ImageNet), head diganti regresi 1 keluaran (atau
10 keluaran untuk varian soft-label). Dipilih karena data sangat kecil (555 foto), ringan
(~11M parameter, cocok Core ML on-device), dan residual connection membuatnya stabil dilatih.
Regularisasi: dropout 0.4, weight decay 5e-2, MixUp (α=0.2), augmentasi geometris/warna kuat.

**Metrik evaluasi & alasannya:**

| Metrik | Kegunaan |
|---|---|
| MAE | Rata-rata selisih grade — mudah ditafsirkan secara klinis |
| Akurasi ±1 | Toleransi 1 grade — realistis karena disagreement antar-annotator sendiri hampir selalu ±1 |
| QWK (quadratic weighted kappa) | Metrik utama untuk ranking model; menghukum selisih besar lebih berat; bisa dibandingkan langsung dengan ceiling manusia |

**Hasil cross-validation 5-fold:**

| Fold | MAE | ±1 | QWK |
|---|---|---|---|
| 1 | 1.053 | 76.8% | 0.793 |
| 2 | 0.901 | 81.3% | 0.872 |
| 3 | 1.056 | 74.4% | 0.805 |
| 4 | 0.852 | 84.1% | 0.877 |
| 5 | 0.941 | 78.8% | 0.842 |
| **mean±SD** | **0.961±0.081** | **79.1%±3.4%** | **0.838±0.034** |

Model produksi diretrain di seluruh data (target `mean`, 30 epoch, `INPUT_MODE='full'`),
ditambah kalibrasi linear dan gerbang out-of-distribution (centroid k-means k=48 dari
embedding train, threshold dari kuantil-99% skor val ×1.1 margin = 0.109), lalu diekspor
sebagai Core ML 2-keluaran (skor + vektor fitur).

## 3. Data & Pelabelan (EDA Notebook 10)

Dataset: 555 foto intraoral frontal (koleksi OMNI) berlabel lengkap oleh 3 annotator, dibagi
374 train / 75 val / 106 test. Sebaran grade timpang — grade 6–7 paling banyak, grade 9–10
(kasus paling parah) paling jarang; ini terkonfirmasi berdampak ke model (recall grade 5 hanya
9%, grade 10 0%, sementara grade dengan data lebih banyak recall-nya 36–50%). Kesepakatan
antar-annotator: QWK berpasangan 0.909–0.932 (rata-rata 0.918), akurasi tepat berpasangan
hanya 47.2% tapi akurasi ±1 mencapai 89.2% — mengonfirmasi bahwa perbedaan antar-annotator
hampir selalu selisih tipis, sehingga QWK/±1 adalah metrik yang lebih adil daripada akurasi
tepat untuk task ini.

---

## 4. Struktur Folder

```
IOTN-AC/
├── README.md
├── notebooks/
│   ├── 00_fdi_number_detector.ipynb        # deteksi & penomoran FDI per gigi (di luar jalur AC)
│   ├── 01_ac_pipeline.ipynb                # transfer (ImageNet) + head linear — BASELINE
│   ├── 02_ac_pipeline_corn.ipynb           # sama, head ordinal CORN
│   ├── 03_lightweight_cnn.ipynb            # custom CNN ringan, dari nol
│   ├── 04_mini_resnet.ipynb                # mini-ResNet (skip connection), dari nol
│   ├── 05_transfer_custom_head.ipynb       # transfer + custom head, fine-tune 2 tahap
│   ├── 06_pretrain_gingivitis.ipynb        # domain pretraining backbone -> models/
│   ├── 07_eda_gingivitis.ipynb             # EDA dataset gingivitis (pendukung nb 06)
│   ├── 08_clustering_roi.ipynb             # clustering 555 foto OMNI, 10 cluster berbasis ROI
│   ├── 09_ac_frontview_multitask.ipynb     # AC aktif + kepala MOCDO stub + Grad-CAM
│   ├── 10_ac_multiannotator_resnet.ipynb   # ★ NOTEBOOK PRODUKSI — 3 annotator, CV 5-fold, ekspor Core ML
│   ├── 11_download_roboflow_masks.ipynb    # unduh mask segmentasi gigi dari Roboflow
│   ├── 12_teeth_masking.ipynb              # RLE mask -> gambar ter-masking (data/masked, masked_crop)
│   ├── 13_teeth_segmentation_model.ipynb   # segmentasi gigi YOLOv8-seg (instance) + ekspor Core ML
│   ├── 14_teeth_semantic_seg.ipynb         # segmentasi gigi semantic (LR-ASPP) + Core ML (dipakai app)
│   ├── 15_bite2text_eda_proxy_labels.ipynb # EDA + proxy label untuk captioning bite2text
│   ├── 15_clip_retrieval_caption.ipynb     # captioning berbasis retrieval CLIP
│   ├── 16_view_code_verification.ipynb     # verifikasi kode/label per-view (5 view capture)
│   ├── 17_view_classifier.ipynb            # klasifikasi foto termasuk view yang mana
│   ├── 18_view_classifier_others.ipynb     # varian/pendekatan lain view classifier
│   ├── 19_view_validator.ipynb             # validasi kualitas & kebenaran foto per-view
│   ├── 20_export_view_validator_coreml.ipynb # ekspor view validator ke Core ML
│   ├── 21_bite2text_view5.ipynb            # bite2text untuk view ke-5
│   ├── 22_bite2text_lateral_review.ipynb   # review bite2text untuk view lateral
│   └── roboflow_config.json                # API key + detail project Roboflow
├── data/
│   ├── raw/{train,val,test}/               # 555 foto (374/75/106), label lengkap 3 annotator
│   ├── labels/
│   │   ├── new ac report.xlsx              # label lama (single-source, dipakai nb 1–9)
│   │   └── Grade AC by Team.xlsx           # label baru: 3 kolom annotator (Nana/Anin/Nico), dipakai nb 10
│   └── processed/                          # cache hasil olahan
├── models/                                 # bobot + ekspor: .pt, Core ML (.mlpackage), config.json
└── outputs/
    └── runs/YYYY-MM-DD_HHMMSS/             # 1 folder per run (previews/figures/reports)
```

## 5. Notebook Pendukung di Luar Jalur AC

Notebook 00 dan 11–22 bukan bagian dari pencarian model skor AC — masing-masing mendukung
fitur lain di sekitarnya di aplikasi Malokit:

- **00** — deteksi & penomoran gigi berdasarkan sistem FDI dari foto intraoral.
- **11–14** — pipeline segmentasi gigi: unduh mask dari Roboflow, terapkan ke foto, latih
  model segmentasi instance (YOLOv8-seg) dan semantic (LR-ASPP) beserta ekspor Core ML.
  Notebook 14 adalah versi yang dipakai aplikasi.
- **15–22** — pipeline verifikasi & interpretasi foto multi-view: klasifikasi/validasi view
  (memastikan foto yang diambil memang sesuai view yang diminta), serta eksperimen
  captioning otomatis ("bite2text") dari foto intraoral.

## 6. Deployment ke Aplikasi iOS (Malokit)

Skor AC berjalan **on-device** lewat Core ML (bukan lewat server) — alasan utamanya:
privasi pasien (foto medis tidak pernah lewat jaringan), latensi & keandalan di klinik
dengan koneksi tidak stabil, dan ukuran model yang cocok mobile (ResNet18 ~11M parameter,
jalan di Neural Engine). Alur ekspor: `.mlpackage` 2-output (grade + vektor fitur) dari
notebook 10 + config JSON pendamping (centroid, threshold OOD, parameter kalibrasi) →
dibundel ke target Xcode → `ACGraderRegressor` memuat `MLModel` langsung → preprocessing
stretch-resize 224×224 (sama seperti training) → post-processing: deteksi OOD via cosine
similarity ke centroid, kalibrasi, pembulatan ke grade 1–10, confidence dari distribusi
Gaussian.

## 7. Menjalankan

Jalankan notebook dari folder proyek (atau dari dalam `notebooks/`); akar proyek dideteksi
otomatis. Untuk notebook AC (01–10), jalankan cell berurutan dari Tahap 1.

```bash
pip install torch torchvision numpy pillow matplotlib openpyxl
# opsional untuk ekspor iOS:
pip install coremltools
```

Untuk notebook 10 (produksi), label harus berupa **1 file Excel dengan 3 kolom skor
annotator** (`data/labels/Grade AC by Team.xlsx`) — bukan file label lama single-source.

---

*Dokumen ini disusun dari isi notebook aktual (01–10) dan dari eksperimen/EDA yang
terdokumentasi untuk notebook 10 — bukan ringkasan sintetis.*
