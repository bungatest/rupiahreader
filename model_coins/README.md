# Rupiah Coin Classifier (Assistive Technology for Visually Impaired)

Proyek klasifikasi nominal koin Rupiah (Rp 100, Rp 200, Rp 500, Rp 1000) yang dirancang khusus untuk berjalan di perangkat smartphone sebagai alat bantu tunanetra. Fokus utama pengembangan adalah **precision tinggi** dan **estimasi probabilitas yang terkalibrasi**, sehingga sistem lebih memilih menjawab *"Tidak yakin, coba arahkan kamera kembali"* daripada salah menyebutkan nominal uang.

---

## 1. Struktur Direktori Proyek
```text
model_coins/
├── coin_training.ipynb          # Master notebook mencakup 8 Tahapan end-to-end
├── augment_coin.py              # Modul ekstraksi & augmentasi sintetis fisik realistis
├── infer.py                     # Skrip inferensi mandiri (TFLite/Keras, Voting Temporal)
├── requirements.txt             # Dependensi environment Python
├── README.md                    # Dokumentasi komprehensif proyek
├── splits/                      # Manifest pemisahan data (Hold-out Test & 5-Fold CV)
│   ├── test_manifest.csv
│   ├── dev_manifest.csv
│   └── fold_*_train.csv / fold_*_val.csv
├── cache/                       # Dataset sintetis offline per fold
├── models/                      # Artefak model produksi
│   ├── best.keras               # Model Keras format native
│   ├── best.tflite              # Model TFLite Float16 (on-device edge mobile)
│   ├── best_dynamic.tflite      # Model TFLite Dynamic Range Quantization
│   └── labels.json              # Pemetaan indeks ke nominal
└── results/                     # Laporan & visualisasi
    ├── experiments.csv          # Log seluruh eksperimen & metrik
    ├── data_audit.md            # Laporan audit dataset & integritas
    ├── report.md                # Laporan akhir komprehensif & rekomendasi
    ├── coin_training.html       # Ekspor salinan HTML dari notebook
    ├── extraction_verification.png
    ├── aug_samples_overview.png
    ├── model_cv_comparison.png
    ├── test_evaluation_plots.png
    └── gradcam_verification.png
```

---

## 2. Metodologi & Aturan Ketat Pemodelan
1. **Pemisahan Sumber Terisolasi (Anti-Leakage):**
   - Hold-out test set (~20%, minimal 4 citra per kelas) dipisahkan berbasis hash MD5 unik dari citra sumber sebelum augmentasi apa pun.
   - Variasi hasil augmentasi sintetis tidak pernah bocor lintas split.
2. **Augmentasi Fisik Sintetis (`augment_coin.py`):**
   - Rotasi 0–360° (strictly *NO flipping* agar angka nominal tidak terbalik).
   - Penyesuaian perspektif miring, glare spekular, motion blur kamera, oklusi jari, dan komposit latar meja/kain.
   - Kelas ke-5 pengecoh (`bukan_koin`) untuk meminimalkan False Positive Rate.
3. **Perbandingan Tiga Arsitektur Model:**
   - **Custom CNN**: Dilatih dari nol (4 blok conv ganda + BatchNorm + MaxPool, rescaling 1/255).
   - **MobileNetV2**: Preprocessing yang benar `Rescaling(1/127.5, offset=-1)`, transfer learning dua tahap dengan BatchNorm dalam inference mode.
   - **EfficientNetV2-B0**: Preprocessing built-in 0..255 tanpa rescale tambahan.
4. **Verifikasi Shortcut (Grad-CAM):**
   - Memastikan model memperhatikan kontur angka, relief lambang Garuda, dan warna logam koin, bukan artefak latar belakang.

---

## 3. Cara Menjalankan Inferensi
### Uji Citra Tunggal
```bash
python infer.py --model models/best.tflite --input path/to/sample.png --threshold 0.62
```

### Uji Seluruh Folder Citra
```bash
python infer.py --model models/best.tflite --input path/to/folder --threshold 0.62
```

### Menjalankan Ulang Notebook End-to-End
```bash
jupyter nbconvert --to notebook --execute --inplace coin_training.ipynb
jupyter nbconvert --to html coin_training.ipynb --output-dir results --output coin_training.html
```

