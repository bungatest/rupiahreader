import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

report_content = """# Laporan Teknis Rekayasa ML: Arsitektur Klasifikasi Koin Rupiah untuk Tunanetra
**Proyek:** Rupiah Reader Mobile (Assistive Technology)  
**Status Pengembangan:** Prototipe Eksperimental Awal (BELUM AMAN DIGUNAKAN PENGGUNA TUNANETRA)  
**Tanggal Evaluasi:** 2026-11-01  
**Repositori & Model:** [model_coins/](file:///C:/Users/Bunga%20Alyssa/OneDrive/文档/DS/smt%20%207/deep%20l/uas/frontend/model_coins)  

---

## 1. Status Realistis & Target Proyek

> **Peringatan Keselamatan Pengguna:**  
> Pada pengujian simulasi OOF terkini dengan data katalog statis saat ini, model **belum memenuhi syarat keamanan operasional tunanetra**.  
> Aplikasi tunanetra menuntut **False Positive (salah menyebut nominal uang) $\le 2–3\%$**.  
> Menjawab *"tidak yakin / coba cari tempat lebih terang"* jauh lebih aman daripada salah menyebut uang.

### Target Kinerja yang Harus Dibuktikan pada Uji Video Lapangan Independen:
1. **Presisi Nominal (Saat Aplikasi Menyebut Uang)**: $\ge 90–95\%$.
2. **Tingkat False Positive (Salah Sebut Uang)**: $\le 2–3\%$ dari total sesi.
3. **Tingkat Menyerah / Tidak Yakin**: $\le 10–15\%$ dalam batas waktu global 15 detik.
4. **Metrik Terbuka**: Dilaporkan menggunakan **Bootstrap Confidence Interval per koin fisik** (bukan Wilson CI per frame/sesi).

---

## 2. Analisis Kritis: Mengapa Akumulasi Bayes Sederhana Gagal (27.6% Galat Sesi)

Pada pengujian awal simulasi multi-frame dengan akumulasi log-posterior naive ($\log p += 0.5 \log(p_4 + \epsilon)$):
- **Hasil Sesi Salah (False Positive)**: **276 dari 1.000 sesi (27.6%)** — sangat tidak aman!
- **Presisi Sesi**: hanya **57.99%** (Wilson CI: [54.2%, 61.7%]).

### Mengapa Hal Ini Terjadi?
1. **Inflasi Posterior dari Bukti Lemah**: Penjumlahan log-probabilitas tanpa syarat batas per-frame membuat posterior cepat melonjak $\ge 0.85$ meskipun setiap frame hanya memiliki confidence lemah (misal $p \approx 0.50$). Padahal dari data OOF, frame berkeyakinan $\approx 0.50$ salah sekitar 26% dari waktu.
2. **Bias Sistematis pada Koin Sumber**: Karena sampel sintetis berasal dari foto katalog yang sama, kesalahan model pada koin tertentu bersifat seragam (misal koin 1000 dikira 500 atau Garuda). Menumpuk bukti dari koin yang salah sistematis justru **memperkuat keyakinan yang salah (*correlated overconfidence*)**.
3. **Ambang Akumulasi $\neq$ Ambang Frame Tunggal**: Ambang $\tau=0.85$ pada posterior akumulasi bukan ambang yang sama dengan keyakinan $\tau=0.85$ pada citra independen.

---

## 3. Solusi: Aturan Konsensus Multi-Frame Ketat (`decide_strict` di `infer.py`)

Aturan keputusan diperketat dengan menyaratkan bukti per-frame yang konsisten, bukan sekadar total posterior:

```python
def decide_strict(frames, tau_frame=0.75, min_agree=0.85, min_frames=6, min_strong=0.60, min_mean_p=0.70, max_garuda=0.35):
    # frames: probabilitas 5-kelas terkalibrasi (T=0.663) dari frame-frame beragam (dHash >= 10 bit)
    if len(frames) < min_frames: return None, "insufficient_frames"
    
    garuda_mean = float(np.mean([f[4] for f in frames]))
    if garuda_mean >= max_garuda: return "balik_koin", "garuda_detected"
        
    P = np.array([f[:4] / (f[:4].sum() + 1e-12) for f in frames])
    top = P.argmax(axis=1)
    cls = int(np.bincount(top, minlength=4).argmax())
    
    agree  = float((top == cls).mean())             # porsi frame yang sepakat
    strong = float((P[:, cls] >= tau_frame).mean()) # porsi frame yang yakin sendiri
    mean_p = float(P[:, cls].mean())
    
    if agree >= min_agree and strong >= min_strong and mean_p >= min_mean_p:
        return cls, "answered"
    return None, "uncertain"
```

### Hasil Simulasi 1.000 Sesi dengan Aturan Konsensus Ketat (`simulate_sessions.py`):
```text
Total Sesi yang Dijawab Nominal : 194 / 1000 (Coverage: 19.4%)
  - Dijawab BENAR (Nominal Tepat): 166 sesi (16.6% total)
  - Dijawab SALAH (False Positive): 28 sesi (2.8% total) -> MASUK TARGET <= 2-3%!
Sesi Terdeteksi Sisi Garuda (FLIP): 338 sesi (33.8%)
Sesi Menyerah / Tidak Yakin     : 468 sesi (46.8%)

PRESISI NOMINAL SAAT MENJAWAB   : 85.57% (166 / 194)
Wilson 95% Confidence Interval  : [79.9%, 89.8%]
TINGKAT FALSE POSITIVE SALAH    : 2.8% (Target: <= 2-3%)
```
*Ketika aturan diperketat, coverage turun menjadi 19.4% (aplikasi menyerah pada ~46.8% sesi ambigu). Ini sejalan dengan prinsip aksesibilitas: lebih baik 'tidak yakin' daripada salah memberi tahu nominal kepada tunanetra.*

---

## 4. Evaluasi Risk-Coverage Jujur: PREDIKSI NOMINAL SAJA (Pred < 4)

Sebelumnya, tabel risk-coverage 5-kelas (presisi 96.8% pada $\tau=0.70$) terdistorsi oleh tingginya akurasi kelas `sisi_garuda` (26/34 benar). Untuk aplikasi pengguna tunanetra, yang krusial adalah **presisi prediksi nominal saja (100, 200, 500, 1000)** pada data OOF ($N=55$ citra nominal) setelah kalibrasi suhu $T^*=0.663$:

| Threshold $\tau$ | Coverage Dijawab ($n/55$) | Prediksi Benar ($k$) | Presisi Nominal (%) | Wilson 95% Confidence Interval |
| :---: | :---: | :---: | :---: | :---: |
| **0.40** | 52 / 55 (94.5%) | 29 | 55.8% | [42.3%, 68.4%] |
| **0.50** | 37 / 55 (67.3%) | 24 | 64.9% | [48.8%, 78.2%] |
| **0.60** | 25 / 55 (45.5%) | 17 | 68.0% | [48.4%, 82.8%] |
| **0.70** | 15 / 55 (27.3%) | 14 | **93.3%** | [70.2%, 98.8%] |
| **0.80** | 13 / 55 (23.6%) | 13 | **100.0%** | [77.2%, 100.0%] |
| **0.85** | 6 / 55 (10.9%) | 6 | **100.0%** | [61.0%, 100.0%] |
| **0.90** | 3 / 55 (5.5%) | 3 | **100.0%** | [43.9%, 100.0%] |

### Perbandingan: Semua 5-Kelas vs Nominal-Saja (T=0.663):
- Pada $\tau=0.60$: Presisi 5-kelas tampak bagus ($82.6\%$, 38/46), padahal **Nominal-Saja hanya $68.0\%$ (17/25)**!
- Pada $\tau=0.70$: Coverage nominal-saja turun ke **$27.3\%$ (hanya 15 dari 55 citra yang dijawab)**.
- Pada $\tau=0.85$: Coverage nominal-saja tinggal **$10.9\%$ (hanya 6 dari 55 citra)** dengan Wilson CI lebar $[61.0\%, 100.0\%]$.

---

## 5. Investigasi Ulang 9 Galat 1000 $\leftrightarrow$ Garuda dengan Suhu $T^*=0.663$

Probabilitas dihitung ulang menggunakan suhu terkalibrasi $T^*=0.663$ (probabilitas dipertajam):

### A. True sisi_garuda diprediksi 1000 (N=5):
1. `image copy 20.png` (folder 100): Raw Conf 0.558 $\rightarrow$ **T-Conf = 0.731**! ($p_T(1000)=0.731, p_T(\text{garuda})=0.138$).  
   *Penting:* Citra ini **lolos ambang $\tau=0.70$** jika hanya menggunakan frame tunggal! Ini membuktikan bahwa tanpa aturan konsensus multi-frame dan guard Garuda, sistem satu frame akan salah mengumumkan nominal 1000 pada koin Garuda.
2. `image copy 14.png` (folder 1000): Raw Conf 0.534 $\rightarrow$ **T-Conf = 0.638** ($p_T(1000)=0.638$).
3. `image copy 21.png` (folder 1000): Raw Conf 0.471 $\rightarrow$ **T-Conf = 0.613** ($p_T(1000)=0.613$).
4. `image copy 17.png` (folder 1000): Raw Conf 0.379 $\rightarrow$ **T-Conf = 0.443**.
5. `image copy 22.png` (folder 500): Raw Conf 0.304 $\rightarrow$ **T-Conf = 0.363**.

### B. True 1000 diprediksi sisi_garuda (N=4):
- `image copy 10.png`: $p_T(\text{garuda})=0.542, p_T(1000)=0.439$.
- `image copy 11.png`: $p_T(\text{garuda})=0.592, p_T(1000)=0.370$.
- `image copy 13.png`: $p_T(\text{garuda})=0.540, p_T(1000)=0.389$.
- `image copy 19.png`: $p_T(\text{garuda})=0.595, p_T(1000)=0.379$.

### Inspeksi Visual & Koreksi Karakteristik Koin:
Inspeksi visual membuktikan bahwa pada koin Rp 1.000 emisi 2016, sisi angka memang memuat lambang burung Garuda mini tepat di atas tulisan "1000 BANK INDONESIA".  
**Koreksi Rekayasa:** Hal ini **BUKAN label noise**, melainkan karakteristik fisik asli koin Rupiah. Pada pengumpulan data video berikutnya, model harus dilatih secara hierarkis spasial: lambang Garuda besar satu keping penuh = sisi non-angka (perintah balik koin), sedangkan Garuda kecil di atas angka = sisi nominal Rp 1.000.

---

## 6. Protokol Akuisisi Data Video Riil Independen (Wajib Dilakukan)

Data katalog saat ini telah mencapai batas kegunaannya. Langkah yang tidak bisa dilewati adalah mengumpulkan data video lapangan riil:

### A. Persyaratan Keragaman Koin Fisik (Mencegah Sampel Semu)
- Sesi dari koin fisik yang sama **bukan sampel independen**. 150 sesi dari 4 koin fisik hanya memiliki ukuran sampel efektif $\approx 4$.
- **Target Koin Fisik**: Minimal **8–10 keping koin fisik berbeda per nominal** (meminjam dari berbagai sumber/emisi berbeda: baru mengkilap, lecet, kusam teroksidasi).
- **Metode Evaluasi**: Evaluasi menggunakan **Bootstrap Confidence Interval per koin fisik** (cluster bootstrap per `coin_id`).

### B. Format Rekaman Cepat (1 Hari Kerja)
- Struktur penamaan file:  
  `{nominal}_{idKoin}_{sisi}_{latar}_{cahaya}_{hp}.mp4`  
  Contoh: `1000_koin03_angka_kayugelap_redup_samsungA52.mp4`
- Matriks variasi per koin:
  - 2 Sisi (Sisi Angka & Sisi Garuda/Pahlawan)
  - 3 Permukaan Latar (Telapak tangan pengguna, meja kayu, taplak/kain)
  - 2 Kondisi Cahaya (Ruangan redup & cahaya terang)
  - Durasi: 15–20 detik per video, koin diputar dan dimiringkan pelan.
  - Ekstraksi: 1 frame tiap $\approx 0.3$ detik, `group_id = idKoin`.
- Koin untuk Test Set harus dipisahkan sejak awal (*held-out coins*) dan tidak boleh dilihat saat kalibrasi $\tau, \alpha$, atau parameter konsensus.

---

## 7. Integrasi Antarmuka Web ([`rupiah-reader`](file:///C:/Users/Bunga%20Alyssa/OneDrive/文档/DS/smt%20%207/deep%20l/uas/frontend/rupiah-reader))

Arsitektur produksi menggunakan **FastAPI Server** ([`model_coins/server.py`](file:///C:/Users/Bunga%20Alyssa/OneDrive/文档/DS/smt%20%207/deep%20l/uas/frontend/model_coins/server.py)) yang terhubung ke [`rupiah-reader/script.js`](file:///C:/Users/Bunga%20Alyssa/OneDrive/文档/DS/smt%20%207/deep%20l/uas/frontend/rupiah-reader/script.js):
- Endpoint: `POST /predict_coin_session`
- Menerima frame stream kontinu (JPEG blob) dari canvas video browser.
- Menjalankan `FrameQualityGate` (Laplacian blur 224x224), `CoinClassifierTTA` (TFLite), dan `CoinSession` consensus engine.
- Mengembalikan respons status real-time:
  ```json
  {
    "status": "jawab" | "balik_koin" | "lanjut" | "gate_failed" | "menyerah",
    "decision": "1000",
    "confidence": 0.88,
    "audio": "Rp 1.000"
  }
  ```
- Di frontend, audio langsung dimainkan via Web Speech API (`id-ID`) dengan pembatalan antrean suara instan (`speechSynthesis.cancel()`).
"""

with open('model_coins/results/report.md', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("model_coins/results/report.md successfully updated with honest, strict metrics!")

