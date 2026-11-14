# 💵🪙 Rupiah Reader – Deep Learning & MLOps Mata Uang Rupiah

[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-2ea44f?style=for-the-badge&logo=github)](https://bungatest.github.io/rupiahreader/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving%20API-009688?style=for-the-badge&logo=fastapi)](https://bungatest.github.io/rupiahreader/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-TFLite%20FP16-FF6F00?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)

**Rupiah Reader** adalah sistem asistensi berbasis **Deep Learning** dan **MLOps** yang dirancang untuk membantu penyandang tunanetra dan masyarakat umum dalam mengenali nominal **uang kertas** dan **uang koin** Rupiah secara cepat, akurat, dan aksesibel.

🌐 **Aplikasi Web Live:** [https://bungatest.github.io/rupiahreader/](https://bungatest.github.io/rupiahreader/)

---

## 🚀 Fitur Utama

### 1. 💵 Pengenalan Uang Kertas (7 Nominal)
* **Kelas:** Rp 1.000, Rp 2.000, Rp 5.000, Rp 10.000, Rp 20.000, Rp 50.000, dan Rp 100.000 (Emisi Baru TE 2022).
* **Multi-Model Support:**
  * **Custom CNN (8-Layer ConvNet):** Arsitektur konvolusi kustom dengan Batch Normalization & Dropout.
  * **MobileNetV2 Fine-Tuned:** Transfer learning ringan teroptimasi untuk perangkat mobile.
  * **FNN Baseline:** Pembanding fitur flattened dense layer.
  * **Majority Consensus Ensemble:** Menggabungkan prediksi multi-model untuk akurasi tertinggi.

### 2. 🪙 Pengenalan Uang Koin (5 Kelas)
* **Kelas:** Rp 100, Rp 200, Rp 500, Rp 1.000, dan Sisi Garuda/Pahlawan.
* **Arsitektur:**
  * **Custom CNN / MobileNetV2 (TFLite FP16):** Model terkuantisasi hanya **2.8 MB**.
  * **GoogLeNet Inception Multi-Scale:** Menangkap relief mikro angka koin (1x1, 3x3, 5x5 paralel).
* **Fitur Cerdas:**
  * **Frame Quality Gate:** Filter frame buram (Laplacian blur) dan pantulan kilau logam otomatis.
  * **Dual-Sided Hero Recognition:** Mengenali nominal dari sisi tokoh pahlawan nasional (Prof. Dr. Herman Johannes, Dr. Tjipto Mangoenkoesoemo, Letjen T.B. Simatupang, Mr. I Gusti Ketut Pudja).
  * **Multi-Frame Consensus Session:** Akumulasi bukti antar-frame untuk kepastian prediksi 100%.

### 3. ♿ Aksesibilitas Tunanetra & Low Vision (WCAG AAA)
* **Panggung Kamera Penuh (Fullscreen Viewport):** Ketuk layar di mana saja untuk memindai.
* **Audio Voice Guidance (id-ID):** Suara nominal terbilang otomatis tanpa perlu melihat layar.
* **Visual Kontras Tinggi:** Warna kontras kuning-hitam-putih untuk pengguna low vision.
* **Penjumlahan Total:** Ketuk layar 2 kali berturut-turut untuk mendengar akumulasi belanja.

### 4. 📊 MLOps & Uji Lapangan (Field Testing)
* **Pencatatan Ground Truth:** Tombol verifikasi label sebenarnya langsung di antarmuka web.
* **Tracking Akurasi Real-World:** Monitoring performa dan drift akurasi secara live.
* **Ekspor Laporan CSV:** Unduh rekaman uji lapangan untuk analisis evaluasi skripsi/UAS.
* **Umpan Balik Pengguna:** Pengguna dapat mengirim catatan atau foto kegagalan deteksi.

---

## 🛠️ Arsitektur Sistem

```
┌────────────────────────────────────────────────────────┐
│             Frontend Web (GitHub Pages)                │
│       https://bungatest.github.io/rupiahreader         │
│  - Kamera Layar Penuh                                  │
│  - Audio Synthesizer (id-ID)                           │
│  - Mode Switcher (Kertas / Koin)                       │
│  - Indikator Status & Pengaturan Backend API           │
└───────────────────────────┬────────────────────────────┘
                            │ HTTPS JSON / FormData
                            ▼
┌────────────────────────────────────────────────────────┐
│            Unified Serving API (FastAPI)               │
│  - /api/kertas/prediksi (Inference Uang Kertas)        │
│  - /api/koin/sesi       (Interactive Consensus Koin)   │
│  - /api/lapangan/*      (MLOps, Ground Truth, CSV)     │
│  - /docs                (Dokumentasi Interaktif Swagger│
└────────────────────────────────────────────────────────┘
```

---

## 💻 Panduan Menjalankan Secara Lokal

### 1. Clone Repository & Install Dependensi
```bash
git clone https://github.com/bungatest/rupiahreader.git
cd rupiahreader
pip install -r requirements.txt
```

### 2. Jalankan Server Terpadu
Cukup klik ganda file batch:
```bash
JALANKAN_UNIFIED_SERVER.bat
```
Atau jalankan melalui command line:
```bash
python -m uvicorn serving.server:app --host 0.0.0.0 --port 8001 --reload
```

* Buka Web App: **`http://localhost:8001`**
* Buka Dokumentasi Swagger API: **`http://localhost:8001/docs`**

### 3. Pengujian Langsung dari HP (Cloudflare Tunnel)
Untuk menguji kamera HP tanpa ribet deploy:
```bash
3_BUKA_TUNNEL_HP.bat
```
Gunakan URL HTTPS publik yang muncul di HP Anda.

---

## 📂 Struktur Direktori

```
rupiahreader/
├── index.html                  # Antarmuka Web Utama (GitHub Pages)
├── script.js                   # Logika Uang Kertas, Audio, & MLOps
├── coin-reader.js              # Logika Multi-Frame Sesi Koin
├── style.css                   # Desain Kontras Tinggi Aksesibilitas
├── requirements.txt            # Dependensi Python
├── JALANKAN_UNIFIED_SERVER.bat # Skrip 1-Klik Menjalankan Server
├── 3_BUKA_TUNNEL_HP.bat        # Skrip Tunnel HTTPS untuk HP
├── coin_training.ipynb         # Notebook Pelatihan CNN Uang Koin
├── coin_googlenet.ipynb        # Notebook Pelatihan GoogLeNet Koin
├── serving/                    # Backend FastAPI & MLOps Engine
│   ├── server.py               # Endpoint API Uang Kertas, Koin, MLOps
│   ├── config.py               # Konfigurasi Path & Threshold
│   ├── preprocessing.py        # Frame Quality Gate & Image Processing
│   ├── model_manager.py        # Manajemen Multi-Model Deep Learning
│   └── field_logger.py         # Pencatatan Ground Truth & Rekap CSV
├── model/                      # Notebooks & Bobot Model Kertas
└── model_coins/                # Model TFLite FP16 Koin
```

---

## 🎓 Tim Pengembang
Proyek Asistensi Pengenalan Mata Uang Rupiah Berbasis Deep Learning & MLOps (UAS Deep Learning).
* **Repository:** [https://github.com/bungatest/rupiahreader](https://github.com/bungatest/rupiahreader)
* **Demo Publik:** [https://bungatest.github.io/rupiahreader/](https://bungatest.github.io/rupiahreader/)
