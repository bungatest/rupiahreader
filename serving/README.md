# Rupiah Reader Unified Serving System

Layanan inferensi multi-model terpadu (Unified Serving API) yang mengintegrasikan klasifikasi uang kertas Rupiah (7 nominal) dan uang koin Rupiah (5 kelas), terhubung langsung dengan antarmuka web responsif berstandar aksesibilitas tinggi (WCAG AAA).

---

## Arsitektur & Direktori

- **Kode Serving Terpadu**: `frontend\serving\`
  - `server.py`: FastAPI server terpadu (inferensi uang kertas, inferensi koin *native in-process*, field logging, static UI).
  - `model_manager.py`: Thread-safe inference & majority consensus aggregation untuk uang kertas.
  - `preprocessing.py`: EXIF transpose, pad-resize (224x224), normalisasi skala piksel.
  - `field_logger.py`: Pencatatan hasil uji lapangan & ground truth (SQLite & CSV).
  - `config.py`: Konfigurasi path, kelas koin/kertas, teks suara bahasa Indonesia, ambang keputusan.
- **Frontend Aksesibilitas**: `frontend\rupiah-reader\`
  - `index.html`: Antarmuka layar penuh dengan mode switch (💵 Uang Kertas / 🪙 Koin), model selector, dan panel uji lapangan.
  - `script.js`: Logika kamera, pengiriman frame, speech synthesis id-ID, pencatatan label asli instan.
  - `coin-reader.js`: Modul sesi deteksi koin multi-frame kontinu.
  - `style.css`: Palet kontras tinggi (WCAG AAA >= 7:1), target sentuh besar.
- **Model Deep Learning yang Terintegrasi**:
  - **Uang Kertas (7 Kelas):**
    - `mobilenet`: MobileNetV2 Fine-Tuned (`mobilenetv2_finetuned_final.keras`)
    - `fnn`: FNN / MLP Flatten Baseline (`fnn_selected_final.keras`)
    - `cnn`: Custom CNN (`custom_cnn_selected_final.keras`)
    - `consensus`: Ensemble majority voting antar-arsitektur
  - **Uang Koin (5 Kelas: `100`, `200`, `500`, `1000`, `sisi_garuda`):**
    - `coin_cnn_5class`: Custom CNN / MobileNetV2 Coin TFLite FP16 (`best_5class.tflite`) dengan *Frame Quality Gate* (blur/glare check) dan *Temporal Evidence Consensus*.
    - `coin_googlenet`: Arsitektur Multi-Scale Inception (`Coin_GoogLeNet.ipynb`).

---

## Cara Menjalankan Layanan (Satu Klik)

### Menggunakan Batch Script Satu Pintu:
Cukup klik ganda:
```bash
JALANKAN_UNIFIED_SERVER.bat
```
Atau via terminal:
```bash
python -m uvicorn serving.server:app --host 0.0.0.0 --port 8001 --workers 1
```

Buka browser di laptop atau HP (dalam jaringan Wi-Fi yang sama):
- **Aplikasi Web Terpadu:** `http://localhost:8001/`
- **Dokumentasi API Interaktif (Swagger):** `http://localhost:8001/docs`

> **Catatan:** Mode Uang Kertas dan Mode Koin kini berjalan dalam **satu server yang sama**. Anda tidak perlu lagi menjalankan dua server terminal terpisah!

---

## Daftar Endpoint API Deep Learning

| Metode | Endpoint | Fungsi | Payload / Parameter |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Memeriksa status kesehatan server dan kesiapan engine model | - |
| `GET` | `/models` | Katalog seluruh model Deep Learning (kertas & koin) | - |
| `POST` | `/api/banknote/predict` | Inferensi uang kertas (single/multi-frame multipart) | `file` / `files`, `model` (cnn, mobilenet, fnn, consensus) |
| `POST` | `/api/banknote/predict_json` | Inferensi uang kertas via Base64 data URL (canvas HP) | JSON `{ "images": ["data:image/jpeg;base64,..."], "model": "consensus" }` |
| `POST` | `/predict_coin_session` | Sesi interaktif koin multi-frame kontinu (`coin-reader.js`) | Multipart `session_id`, `frame` |
| `POST` | `/api/coin/predict` | Inferensi koin instan (single/multi-frame) | Multipart `file` / `files` |
| `POST` | `/api/field/update_ground_truth` | Memperbarui label asli hasil uji lapangan | JSON `{ "log_id": 1, "ground_truth": "idr_10000" }` |
| `GET` | `/api/field/report` | Rekap metrik akurasi uji lapangan | - |
| `GET` | `/api/field/export` | Mengunduh log uji lapangan sebagai CSV | - |
| `POST` | `/api/feedback` | Menyimpan umpan balik pengguna | Multipart `rating`, `user_note`, `save_image` |
