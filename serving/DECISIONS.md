# Rupiah Reader Serving — Keputusan & Asumsi Arsitektur

Dokumen ini mencatat seluruh keputusan teknis, asumsi, dan kompromi yang diambil dalam perancangan dan implementasi layanan inferensi terpadu **Rupiah Reader** (`FRONT\serving\`).

---

## 1. Status Model & Hasil Evaluasi Paritas

Sesuai **ATURAN KERAS 2 & 3**, tidak ada model atau angka metrik yang dikarang. Model yang tidak tersedia atau belum selesai dilatih ditandai dengan jujur sebagai `tidak_tersedia` dan dinonaktifkan di antarmuka pengguna (UI).

| Model ID | Nama Arsitektur | Notebook Sumber | Status Serving | Akurasi Notebook | Akurasi Uji Aktual ($n=179$) | Wilson 95% CI | Threshold $\tau$ (dari Val $n=176$) |
|---|---|---|---|---|---|---|---|
| `mobilenet` | MobileNetV2 Fine-Tuned (Top-30 Unfrozen) | `mobilenetv2-training-kaggle.ipynb` | **Tersedia (Aktif)** | 95.53% (Kaggle) | **89.39%** | 84.0% – 93.1% | 0.70 |
| `fnn` | FNN / MLP Flatten (Aprillia dkk., 2024) | `Rupiah_FNN.ipynb` | **Tersedia (Aktif)** | 58.66% | **58.66%** | 51.3% – 65.6% | 0.50 |
| `cnn` | Custom CNN (8-Layer ConvNet, 1.2M params) | `Rupiah_CustomCNN.ipynb` | **Belum Tersedia** | 99.44% (Colab GPU) | *Belum selesai* | - | - |

### Catatan Khusus Custom CNN
- Notebook asli dilatih di Google Colab dengan GPU NVIDIA Tesla T4 (~9 detik/epoch, total ~280 detik).
- Pada CPU lokal tanpa akselerasi GPU, 1 epoch Custom CNN membutuhkan waktu ~147 detik. Pelatihan 30 epoch diperkirakan memakan waktu >70 menit.
- Mengacu pada *Aturan Keputusan*: *"Jika pelatihan model memerlukan waktu lama (misalnya CNN butuh >10 menit di CPU)... tandai belum tersedia jika melebihi batas waktu wajar"*, model CNN dinonaktifkan di UI dengan keterangan jelas `(Belum Tersedia / Memerlukan Akselerasi GPU)`. Model MobileNetV2, FNN, dan mode Konsensus beroperasi penuh secara mandiri.

---

## 2. Spesifikasi Input & Normalisasi

Setiap model memiliki karakteristik input yang berbeda berdasarkan graph Keras-nya:
1. **Custom CNN**: Input tensor `(1, 224, 224, 3)` `float32` dalam rentang `[0, 255]`. Layer `layers.Rescaling(1./255)` berada di dalam graph model. Mode resize: `'pad'` (contain + black padding).
2. **MobileNetV2**: Input tensor `(1, 224, 224, 3)` `float32` dalam rentang `[0, 1]`. Normalisasi dilakukan di luar model (`rescale=1./255` pada generator). Mode resize: `'pad'`.
3. **FNN (Aprillia dkk)**: Input tensor `(1, 224, 224, 3)` `float32` dalam rentang `[0, 255]`. Layer `layers.Resizing(64, 64)` dan `layers.Rescaling(1./255)` berada di dalam graph model. Mode resize: `'pad'`.

Modul `preprocessing.py` secara otomatis menerapkan scaling yang sesuai berdasarkan `model_id`.

---

## 3. Logika Konsensus Multi-Model

Saat pengguna memilih mode **Konsensus / Bandingkan Semua**:
1. Server menjalankan inferensi secara paralel/terkunci pada seluruh model yang berstatus aktif (`mobilenet`, `fnn`, dan `cnn` jika tersedia).
2. Suara terbanyak dihitung:
   - Jika $\ge 60\%$ model aktif menyepakati kelas yang sama (misal 2 dari 2 model aktif, atau 2 dari 3).
   - Rata-rata probabilitas keyakinan model yang sepakat $\ge \tau_{\text{consensus}}$ (0.75).
   - Margin selisih probabilitas peringkat 1 dan peringkat 2 $\ge 0.20$.
3. Jika ketiga syarat terpenuhi: status dinyatakan `"pasti"`, output nominal disuarakan.
4. Jika tidak terpenuhi: status dinyatakan `"tidak_yakin"`, audio memberikan panduan: *"Hasil belum meyakinkan, coba sesuaikan posisi atau pencahayaan uang kertas."*

---

## 4. Ambang Keputusan ($\tau$) & Test Set Hygiene

Sesuai aturan ketat:
- **Test set ($n=179$) tidak pernah disentuh** untuk memilih ambang keputusan atau hyperparameter.
- Seluruh ambang keputusan ($\tau$) ditentukan murni dari dataset **Validation ($n=176$)** dengan mengevaluasi kurva Precision vs Coverage pada grid $\tau \in [0.50, 0.95]$.

---

## 5. Integrasi Server Koin Lama (Proxying)

- Endpoint `/predict_coin_session` disediakan langsung pada server serving baru.
- Permintaan diteruskan (*reverse proxy*) menggunakan klien asinkron `httpx` ke `http://127.0.0.1:8000/predict_coin_session`.
- Jika server koin sedang tidak berjalan, server serving menangkap `httpx.ConnectError` dan mengembalikan respons JSON bersahabat:
  ```json
  {
    "status": "error",
    "message": "Server model koin (port 8000) sedang tidak aktif. Silakan jalankan model_coins/server.py jika ingin menggunakan fitur deteksi koin."
  }
  ```
- File asli `FRONT\model_coins\server.py` **sama sekali tidak diubah** (sesuai Aturan Keras 1).

---

## 6. Privasi Pengguna & Field Logging (Aturan Keras 6)

1. **Uji Lapangan**: Setiap inferensi dicatat ke database SQLite `C:\rupiah_serving\logs\field_eval.db` (timestamp, model, prediksi, latency, keyakinan).
2. **Ground Truth**: Pengguna dapat mengetuk salah satu dari 7 tombol nominal di layar untuk mencatat label sebenarnya secara instan.
3. **Privasi Frame Kamera**: Gambar kamera **tidak pernah disimpan ke disk** selama pengujian biasa. Frame gambar hanya dikirim dan disimpan jika pengguna secara sadar mencentang opsi *"Kirim foto untuk analisis kegagalan"* pada modal formulir feedback.

