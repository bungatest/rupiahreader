# Laporan Audit Dataset Koin Rupiah
**Tanggal Analisis:** 2026-11-01  
**Lokasi Dataset:** `dataset coin/` (subfolder: `100`, `200`, `500`, `1000`)

---

## 1. Karakteristik & Distribusi Citra
- **Total Citra**: 94 file `.png`
- **Distribusi Kelas**:
  - `100`: 23 citra (24.5%)
  - `200`: 18 citra (19.1%)
  - `500`: 30 citra (31.9%)
  - `1000`: 23 citra (24.5%)
- **Format Warna**: 100% citra adalah RGB 3-channel (tidak terdapat channel transparansi Alpha/RGBA).
- **Variasi Resolusi**: Rentang dimensi dari 224x225 piksel hingga 2532x2495 piksel. Seluruh citra berlatar belakang putih studio / katalog e-commerce.

---

## 2. Audit Integritas Biner (Hash MD5)
- **Total Unique MD5 Hashes**: 89 hash unik dari 94 file.
- **Konflik MD5 Lintas Kelas (Cross-class Conflict)**: **0 kasus**. Tidak ada file identik yang memiliki label berbeda.
- **Duplikasi Persis di Dalam Kelas (Intra-class Exact Duplicates)**: **5 pasang file identik** (duplikasi file `image copy.png`):
  1. Kelas `100`: `image copy 4.png` identik dengan `image.png` (MD5: `eb8be8a3...`)
  2. Kelas `1000`: `image copy 2.png` identik dengan `image.png` (MD5: `7f8ea241...`)
  3. Kelas `200`: `image copy 12.png` identik dengan `image copy 13.png` (MD5: `5fdc7c1b...`)
  4. Kelas `500`: `image copy 11.png` identik dengan `image copy 12.png` (MD5: `c1762e38...`)
  5. Kelas `500`: `image copy 16.png` identik dengan `image copy 23.png` (MD5: `2548b501...`)

Semua pasangan duplikat intra-kelas ini ditangani dalam proses pembagian train/val/test agar tidak bocor antar split.

---

## 3. Analisis Kemiripan Visual (dHash & Hamming Distance)
Menggunakan Difference Hash (dHash 8x8, 64 bit per citra) untuk menganalisis kemiripan pola visual:
- **Pasangan dengan Jarak Hamming $\le 10$**: 41 pasangan citra.
- **Temuan Kritis Kemiripan Lintas Kelas (Hamming Distance = 0)**:
  - `[100] image copy 22.png` vs `[1000] image copy 18.png` (Jarak Hamming = 0)
  - `[100] image copy 22.png` vs `[200] image copy 15.png` (Jarak Hamming = 0)
  - `[1000] image copy 18.png` vs `[200] image copy 15.png` (Jarak Hamming = 0)

**Akar Penyebab:**
Ketiga citra tersebut merupakan foto sisi lambang Burung Garuda dengan template katalog yang identik. Pada koin Rupiah asli, sisi lambang Garuda tidak memuat informasi nominal angka, sehingga secara visual hampir indistinguishable antar nominal pada resolusi standar. Hal ini menjelaskan mengapa model berbasis citra tunggal akan mengalami ketidakpastian tinggi jika sisi koin yang menghadap kamera adalah sisi Garuda.

---

## 4. Evaluasi Pipeline Segmentasi Koin (`augment_coin.py`)
- Algoritma ekstraksi koin adaptif (`extract_coin`):
  - Kombinasi filter Euclidean distance latar putih ($d > 28$), threshold Otsu multi-level, dan circularity contour filtering ($\ge 0.55$).
  - **Tingkat Keberhasilan Segmentasi**: **100% (94 dari 94 citra)** berhasil dipisahkan koinnya secara bersih dari latar putih tanpa pemotongan kontur koin.
