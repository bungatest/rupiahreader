# Laporan Teknis Rekayasa ML: Arsitektur Klasifikasi Koin Rupiah untuk Tunanetra
**Proyek:** Rupiah Reader Mobile (Assistive Technology)  
**Status Pengembangan:** Prototipe Eksperimental Awal (BELUM AMAN DIGUNAKAN PENGGUNA TUNANETRA)  
**Tanggal Evaluasi:** 2026-11-01  
**Repositori & Model:** [model_coins/](file:///C:/Users/Bunga%20Alyssa/OneDrive/文档/DS/smt%20%207/deep%20l/uas/frontend/model_coins)  

---

## 1. Status Realistis & Target Pengujian Lapangan

> **Peringatan Keselamatan Pengguna:**  
> Pada pengujian simulasi OOF terkini dengan data katalog statis saat ini, model **belum memenuhi syarat keamanan operasional tunanetra**.  
> Target $\ge 90–95\%$ benar saat menjawab **belum tercapai**.  
> Menjawab *"tidak yakin / coba cari tempat lebih terang"* jauh lebih aman daripada salah menyebut uang.

### Target Kinerja yang Harus Dibuktikan pada Uji Video Lapangan Independen:
1. **Presisi Saat Menjawab (Jawaban Benar / Jawaban Diberikan)**: $\ge 90–95\%$.
2. **Salah per Jawaban (1 - Presisi)**: $\le 5–10\%$.
3. **Salah per Sesi (False Positive Rate)**: $\le 2–3\%$ dari total seluruh sesi.
4. **Coverage Sesi (Sesi yang Berani Dijawab)**: $\ge 50–70\%$ (saat ini masih terlalu rendah: $\approx 19.4\%$).
5. **Metrik Berbasis Koin Fisik**: Dilaporkan menggunakan **Bootstrap Confidence Interval per koin fisik** (bukan per frame atau per sesi).

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

## 3. Evaluasi Aturan Konsensus Multi-Frame Ketat (`decide_strict` di `infer.py`)

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
    
    agree  = float((top == cls).mean())             # porsi frame yang sepakat (>= 85%)
    strong = float((P[:, cls] >= tau_frame).mean()) # porsi frame yang yakin sendiri (>= 60%)
    mean_p = float(P[:, cls].mean())                # rata-rata keyakinan (>= 0.70)
    
    if agree >= min_agree and strong >= min_strong and mean_p >= min_mean_p:
        return cls, "answered"
    return None, "uncertain"
```

### Hasil Simulasi 1.000 Sesi & Analisis Kejujuran Target:
- **Coverage Dijawab**: **19.4%** (194 / 1000 sesi).
- **Salah per Sesi (FP rate total)**: **2.8%** (28 / 1000 sesi).
- **Salah per Jawaban (1 - Presisi)**: **14.4%** (28 salah dari 194 jawaban).
- **Presisi Saat Menjawab**: **85.57%** (166 benar dari 194 jawaban).
- **Sesi Menyerah / Balik Koin**: **80.6%** (468 menyerah + 338 balik koin).

> [!WARNING]
> **Status Target:** Target $\ge 90–95\%$ benar saat menjawab **BELUM TERCAPAI** (masih $85.6\%$, salah per jawaban $14.4\%$).  
> Selain itu, coverage $19.4\%$ berarti aplikasi menyerah pada $\approx 80\%$ sesi, sehingga konfigurasi ini **belum layak dipakai pengguna**.

### Estimasi Tidak Bias: Cluster-Bootstrap CI (Resampling per Gambar Sumber):
*Catatan Metodologi: Parameter ($\tau_f=0.75, \text{agree}=0.85$, dll.) dipilih pada data simulasi yang sama (hasil dev). Cluster bootstrap ($B=5.000$) meresample 89 cluster gambar sumber bertingkat (stratified by nominal):*

| Metrik Evaluasi Sesi | Estimasi Titik | Cluster Bootstrap 95% CI | Keterangan |
| :--- | :---: | :---: | :--- |
| **Coverage (Dijawab Nominal / Semua Sesi)** | **19.4%** | **[11.8%, 27.1%]** | Sangat rendah (menyerah pada ~80% sesi) |
| **Benar per Sesi** | 16.6% | [9.6%, 23.8%] | Proporsi sesi sukses |
| **Salah per Sesi (False Positive Rate)** | **2.8%** | **[0.1%, 6.3%]** | CI melewati batas toleransi 3% |
| **Presisi Saat Menjawab** | **85.6%** | **[68.4%, 99.4%]** | Belum mencapai target $\ge 90–95\%$ |
| **Salah per Jawaban ($1 - \text{Presisi}$)** | **14.4%** | **[0.6%, 31.6%]** | Rentang ketidakpastian sangat lebar |

---

## 4. Evaluasi Risk-Coverage Jujur: SELURUH 89 GAMBAR (Termasuk Kebocoran Garuda $\rightarrow$ Nominal)

Di aplikasi nyata, pengguna juga mengarahkan kamera ke sisi Garuda. Kesalahan sisi Garuda yang diprediksi sebagai nominal adalah kegagalan paling berbahaya. Evaluasi dihitung pada **seluruh 89 citra unik** setelah kalibrasi suhu $T^*=0.663$:

| Threshold $\tau$ | Dijawab ($n/89$) | Prediksi Benar ($k$) | Salah / FP ($n - k$) | Presisi Saat Menjawab (%) | Wilson 95% Confidence Interval |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.40** | 52 / 89 (58.4%) | 29 | 23 | 55.8% | [42.3%, 68.4%] |
| **0.50** | 37 / 89 (41.6%) | 24 | 13 | 64.9% | [48.8%, 78.2%] |
| **0.60** | 25 / 89 (28.1%) | 17 | 8 | **68.0%** | [48.4%, 82.8%] |
| **0.70** | 15 / 89 (16.9%) | 14 | 1 | **93.3%** | [70.2%, 98.8%] |
| **0.75** | 13 / 89 (14.6%) | 13 | 0 | **100.0%** | [77.2%, 100.0%] |
| **0.80** | 13 / 89 (14.6%) | 13 | 0 | **100.0%** | [77.2%, 100.0%] |
| **0.85** | 6 / 89 ( 6.7%) | 6 | 0 | **100.0%** | [61.0%, 100.0%] |
| **0.90** | 3 / 89 ( 3.4%) | 3 | 0 | **100.0%** | [43.9%, 100.0%] |

### Fakta Kritis:
- Pada $\tau=0.70$, citra `image copy 20.png` (sisi Garuda dari folder 100) memiliki $T$-confidence $= 0.731$ dan **bocor sebagai prediksi salah Rp 1.000**, menyebabkan 1 kesalahan ($k=14, n=15$, presisi $93.3\%$).
- Pada $\tau=0.85$, model hanya berani menjawab **6 dari 89 citra ($6.7\%$ coverage)** dengan Wilson CI lebar $[61.0\%, 100.0\%]$.

---

## 5. Investigasi Ulang 9 Galat 1000 $\leftrightarrow$ Garuda dengan Suhu $T^*=0.663$

Probabilitas dihitung ulang menggunakan suhu terkalibrasi $T^*=0.663$:

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

### Inspeksi Visual Koin:
Inspeksi visual membuktikan bahwa pada koin Rp 1.000 emisi 2016, sisi angka memang memuat lambang burung Garuda mini tepat di atas tulisan "1000 BANK INDONESIA".  
**Koreksi Rekayasa:** Hal ini **BUKAN label noise**, melainkan karakteristik fisik asli koin Rupiah. Pada pengumpulan data video berikutnya, model harus dilatih secara hierarkis spasial: lambang Garuda besar satu keping penuh = sisi non-angka (perintah balik koin), sedangkan Garuda kecil di atas angka = sisi nominal Rp 1.000.

---

## 6. Persyaratan Statistik untuk Membuktikan Target $\ge 90\%$ di Lapangan

Perhitungan validitas statistik wajib berbasis **koin fisik independen**, bukan frame atau sesi:

### Batas Bawah Wilson 95% untuk $n$ Koin Fisik (Asumsi 0 Kesalahan):
Formula: $\text{Batas Bawah} = \frac{n}{n + 3.84}$
- $n = 12 \text{ koin} \rightarrow 76.0\%$
- $n = 20 \text{ koin} \rightarrow 83.9\%$
- $n = 35 \text{ koin} \rightarrow \mathbf{90.1\%}$
- $n = 40 \text{ koin} \rightarrow \mathbf{91.2\%}$

### Implikasi Desain Data:
1. Untuk membuktikan $\ge 90\%$ dengan keyakinan 95% (bahkan dengan asumsi nol kesalahan), dibutuhkan **minimal 35 keping koin fisik berbeda** di test set ($\approx 9$ keping per nominal).
2. Jika ada beberapa kesalahan di lapangan, dibutuhkan jumlah koin lebih banyak lagi.
3. **Rencana Pengumpulan Koin**: Target minimal **15–20 keping koin fisik per nominal**:
   - $\approx 10$ keping dialokasikan murni untuk Test Set (diisolasi sejak awal).
   - Sisanya untuk pelatihan dan kalibrasi parameter ($\tau_f, \text{agree}, \text{min\_frames}$).

---

## 7. Arsitektur Frontend & API Server

Tersedia implementasi integrasi siap uji:
- **FastAPI Server**: [`server.py`](file:///C:/Users/Bunga%20Alyssa/OneDrive/文档/DS/smt%20%207/deep%20l/uas/frontend/model_coins/server.py)
  - Endpoint: `POST /predict_coin_session` (menerima `session_id` & `frame`)
  - Pembersihan sesi otomatis (TTL 60 detik) & batas payload 5MB.
  - **Prinsip Privasi Ketat**: Frame kamera diolah in-memory dan tidak pernah disimpan ke disk.
- **Client Frontend**: [`rupiah-reader/coin-reader.js`](file:///C:/Users/Bunga%20Alyssa/OneDrive/文档/DS/smt%20%207/deep%20l/uas/frontend/rupiah-reader/coin-reader.js)
  - Terhubung ke [`rupiah-reader/index.html`](file:///C:/Users/Bunga%20Alyssa/OneDrive/文档/DS/smt%20%207/deep%20l/uas/frontend/rupiah-reader/index.html) dan [`rupiah-reader/script.js`](file:///C:/Users/Bunga%20Alyssa/OneDrive/文档/DS/smt%20%207/deep%20l/uas/frontend/rupiah-reader/script.js).
  - Mengirim 1 frame in-flight tiap 300ms, menangani arahan suara dinamis via `speechSynthesis`, dan menghentikan streaming begitu sesi selesai atau timeout.
- **Skrip Baseline Embedding Colab**:
  - [`colab_embedding_baseline.py`](file:///C:/Users/Bunga%20Alyssa/OneDrive/文档/DS/smt%20%207/deep%20l/uas/frontend/model_coins/colab_embedding_baseline.py) & [`colab_embedding_baseline.ipynb`](file:///C:/Users/Bunga%20Alyssa/OneDrive/文档/DS/smt%20%207/deep%20l/uas/frontend/model_coins/colab_embedding_baseline.ipynb) untuk evaluasi komparatif EfficientNetV2-B0 vs DINOv2-small dengan Logistic Regression dan TTA 8 sudut rotasi di GPU Colab.
