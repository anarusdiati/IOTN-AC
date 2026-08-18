# IOTN-AC — Benchmarking Prediksi Skor Aesthetic Component

Prediksi skor **IOTN Aesthetic Component (1–10)** dari foto frontal intraoral, dengan
membandingkan (benchmarking) beberapa arsitektur CNN pretrained. Notebook menyusun alur
lengkap dari pengumpulan data sampai ekspor model untuk aplikasi iPhone.

## Struktur folder

```
IOTN-AC/
├── README.md
├── notebooks/
│   ├── 01_ac_pipeline.ipynb          # transfer (ImageNet) + head Linear — baseline
│   ├── 02_ac_pipeline_corn.ipynb     # sama, tapi head ordinal CORN
│   ├── 03_lightweight_cnn.ipynb      # custom CNN ringan, dilatih dari nol
│   ├── 04_mini_resnet.ipynb          # mini-ResNet (skip connection), dari nol
│   ├── 05_transfer_custom_head.ipynb # transfer + custom head (dense+dropout+scaling), fine-tune 2 tahap
│   ├── 06_pretrain_gingivitis.ipynb  # domain pretraining backbone di dataset gingivitis -> models/
│   ├── 07_eda_gingivitis.ipynb       # eksplorasi dataset gingivitis (sebaran, geometri, contoh anotasi)
│   ├── 08_clustering_roi.ipynb       # clustering foto OMNI ke 10 cluster berbasis ROI gigi
│   ├── 09_ac_frontview_multitask.ipynb # AC front-view: kepala AC aktif + kepala MOCDO stub + Grad-CAM
│   ├── 10_ac_multiannotator_resnet.ipynb # AC dari 3 annotator: kesepakatan + target mean/median/majority/soft (ResNet18)
│   ├── 11_download_roboflow_masks.ipynb # unduh mask segmentasi gigi dari Roboflow (train/val/test)
│   ├── 12_teeth_masking.ipynb          # RLE mask Roboflow -> gambar ter-masking (data/masked & data/masked_crop)
│   ├── 13_teeth_segmentation_model.ipynb # latih segmentasi gigi YOLOv8-seg (instance) + ekspor Core ML
│   ├── 14_teeth_semantic_seg.ipynb     # segmentasi gigi SEMANTIC (LR-ASPP) 1-mask + Core ML (dipakai app iOS)
│   └── roboflow_config.json            # API key + detail project Roboflow (edit sebelum menjalankan nb 11)
├── data/
│   ├── raw/
│   │   ├── train/                # 374 foto (data latih)
│   │   ├── val/                  #  75 foto (validasi)
│   │   ├── test/                 # 106 foto (uji, tersegel s/d Tahap 8)
│   │   ├── inference_teeth/      # gigi luar-distribusi — uji inferensi (Tahap 9)
│   │   └── inference_random/     # bukan foto gigi — uji penolakan validator (Tahap 9)
│   ├── labels/
│   │   └── new ac report.xlsx    # label; dipakai kolom "Nama File" & "Grade AC"
│   └── processed/                # cache hasil olahan (diisi notebook)
├── models/                       # bobot + ekspor: .pt, TorchScript, Core ML, config.json
└── outputs/
    └── runs/
        ├── latest_run.txt        # path run terakhir
        └── YYYY-MM-DD_HHMMSS/     # 1 folder per run (timestamp otomatis)
            ├── previews/          # before/after pra-pemrosesan, per model
            ├── figures/           # semua grafik (kurva latih, confusion matrix, perbandingan performa, dll)
            └── reports/           # tabel & metrik (.csv)
```

## Keluaran tersimpan otomatis tiap run

Setiap kali notebook dijalankan, cell setup (Tahap 1) membuat satu folder baru
`outputs/runs/<timestamp>/` dan mengarahkan semua keluaran ke sana:

- **previews/** — untuk tiap kandidat model, satu gambar `<model>_before_after.png`
  memperlihatkan foto sebelum dan sesudah pra-pemrosesan (resize/crop + normalisasi) sesuai
  spesifikasi input model tersebut.
- **figures/** — grafik yang ingin disimpan bisa ditulis dengan memanggil helper
  `simpan_gambar('nama')` sesudah membuat grafiknya (dinamai dari judul bila kosong). Helper
  ini tidak mengait `plt.show`, jadi aman dijalankan berulang.
- **reports/** — tabel kelayakan model (`kelayakan_model.csv`) dan hasil inferensi
  (`hasil_inferensi.csv`).

Karena tiap run bertimestamp sendiri, hasil antar-run tidak saling menimpa dan mudah
dibandingkan.

## Kandidat model (benchmarking)

Regresi ordinal (grade diperlakukan sebagai angka kontinu). Semua pretrained ImageNet,
lapisan akhir diganti menjadi satu keluaran = Grade AC.

| Model | Alasan |
|---|---|
| `mobilenet_v3_small` | Paling ringan; kandidat utama untuk aplikasi ponsel |
| `shufflenet_v2_x1_0` | Sekelas MobileNet, arsitektur berbeda; pembanding |
| `efficientnet_b0` | Rasio akurasi-terhadap-ukuran terbaik di kelasnya |
| `mobilenet_v3_large` | Kompromi ukuran kecil vs kapasitas |
| `resnet18` | Baku & teruji; baseline |
| `resnet34` | Menguji apakah kedalaman tambahan membantu |
| `resnet50` | Batas atas ukuran; menunjukkan model besar cenderung memburuk pada data kecil |

## Menjalankan

Jalankan notebook dari folder proyek (atau dari dalam `notebooks/`); akar proyek dideteksi
otomatis. Jalankan cell berurutan dari Tahap 1. Ekspor Core ML pada Tahap 9 bersifat opsional
dan hanya aktif bila `coremltools` terpasang.

```
pip install torch torchvision numpy pillow matplotlib openpyxl
# opsional untuk ekspor iOS:
pip install coremltools
```
