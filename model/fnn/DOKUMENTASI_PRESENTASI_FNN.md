# Panduan Presentasi Teknis: Model Feedforward Neural Network (FNN / MLP 7 Kelas)
**Proyek:** Klasifikasi Uang Kertas Rupiah untuk Aplikasi Ramah Tunanetra (*Rupiah Reader*)  
**File Notebook:** [`Rupiah_FNN.ipynb`](file:///c:/Users/Bunga%20Alyssa/OneDrive/文档/DS/smt%20%207/deep%20l/uas/frontend/model/fnn/Rupiah_FNN.ipynb)  
**Target Audiens:** Dosen Pembimbing dan Dosen Penguji Tugas Akhir / Ujian Komprehensif

---

## 1. Ikhtisar & Latar Belakang Eksperimen FNN

### Mengapa Model FNN Diuji dalam Penelitian Ini?
Dalam penelitian klasifikasi citra mata uang Rupiah, model konvolusi (Custom CNN dan MobileNetV2) umumnya menjadi standar utama. Namun, untuk menjamin **keabsahan ilmiah yang komparatif dan menyeluruh (*scientific rigor*)**, kami mengikutsertakan paradigma **Feedforward Neural Network (FNN / Multi-Layer Perceptron pada piksel mentah)** berdasarkan publikasi rujukan:
> **Aprillia, D. P. dkk. (2024)**. *"Rupiah Banknotes Recognition Using Neural Network"*, Jurnal Komtika (KLIK).

Eksperimen ini bertujuan untuk menguji secara empiris batas performa teoretis (*theoretical ceiling*) dari arsitektur tanpa konvolusi saat dihadapkan pada variasi citra dunia nyata.

---

## 2. Struktur Pipeline Kodingan Step-by-Step

Berikut adalah bedah teknis baris per baris kodingan dari sel awal hingga sel akhir notebook:

```
[ Dataset 1.249 Citra ] ──► [ Audit dHash & SHA-256 ] ──► [ Group-Aware Split 70/15/15 ]
                                                                     │
┌────────────────────────────────────────────────────────────────────┴──────────────────────────────────────┐
▼                                                                                                           ▼
[ Train: 894 Citra ]                                                                               [ Test: 179 Citra ]
│                                                                                                  (Segel Hold-out)
├─► Preprocessing: Resize 224x224 (Letterbox Padding Hitam)                                                 │
├─► Pipeline tf.data: Caching & Prefetching (AUTOTUNE)                                                      │
├─► Model 1: Baseline Paper (Dense 128 -> 19,2M Parameter) ──► Val Acc: 37.50%                              │
└─► Model 2: FNN Tuned (Resizing 64x64 + BN + Dropout + L2)                                                  │
     ├── fnn_tuned_with_aug (Dengan Augmentasi) ─────────────► Val Acc: 35.23% (Anjlok!)                   │
     └── fnn_tuned_no_aug (Tanpa Augmentasi) ────────────────► Val Acc: 56.82% (Pemenang Validasi)          │
                                                                       │                                    │
                                                                       └───────────► [ Evaluasi Akhir ] ◄───┘
                                                                                     • Test Acc : 58.66%
                                                                                     • Top-2    : 75.42%
                                                                                     • Gelap    : 17.32% (Runtuh)
```

---

### Tahap 1: Setup Parameter Global & Reproducibility (Cell 2)
* **Kodingan Kunci:**
  ```python
  @dataclass(frozen=True)
  class Config:
      seed: int = 42
      classes: tuple = ('idr_1000', 'idr_2000', 'idr_5000', 'idr_10000', 
                        'idr_20000', 'idr_50000', 'idr_100000')
      img_size: tuple = (224, 224)
      split_ratios: tuple = (0.70, 0.15, 0.15)
      near_dup_threshold: int = 6  # Ambang batas jarak Hamming dHash
  
  # Penguncian Random Seed Deterministik:
  random.seed(SEED)
  np.random.seed(SEED)
  tf.keras.utils.set_random_seed(SEED)
  ```
* **Narasi Presentasi ke Dosen:**
  > *"Pada tahap awal, seluruh konfigurasi dibungkus dalam dataclass `Config` yang bersifat `frozen=True` agar parameter tidak dapat dimanipulasi di tengah proses. Kami juga mengunci *random seed* pada Python, NumPy, dan TensorFlow ke nilai 42 untuk menjamin **reproducibility mutlak**."*

---

### Tahap 2: Audit Kualitas Citra & Deteksi Duplikat (Cell 4 - 7)
* **Kodingan Kunci:**
  ```python
  # 1. Deteksi Blur via Laplacian Variance:
  def laplacian_variance(gray):
      g = gray.astype(np.float32)
      lap = -4*g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
      return float(lap.var())

  # 2. Deteksi Pasangan Serupa via 64-bit dHash:
  def dhash64(gray_img):
      a = np.asarray(gray_img.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
      diff = a[:, 1:] > a[:, :-1]
      return np.packbits(diff).view(np.uint64)[0]
  ```
* **Narasi Presentasi ke Dosen:**
  > *"Pengambilan foto uang di lapangan rentan menghasilkan foto beruntun (*burst-shot*) dari satu lembar uang yang sama. Jika dibiarkan, ini menyebabkan **kebocoran data (*data leakage*)**. Kami menggunakan nilai **dHash 64-bit**. Pasangan foto dengan jarak Hamming $\le 6$ dikelompokkan ke dalam satu `group_id` yang sama menggunakan *connected components* graf."*

---

### Tahap 3: Pembagian Data Group-Aware (Cell 9)
* **Kodingan Kunci:**
  ```python
  from sklearn.model_selection import StratifiedGroupKFold

  sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
  # 14 fold Train (~70%), 3 fold Val (~15%), 3 fold Test (~15%)
  assert (split_df.groupby('group_id').split.nunique() == 1).all()
  ```
* **Hasil Distribusi Data:**
  * **Train Set**: 894 citra ($71.6\%$)
  * **Validation Set**: 176 citra ($14.1\%$)
  * **Test Set**: 179 citra ($14.3\%$) — *Segel Murni (Hold-Out)*
* **Narasi Presentasi ke Dosen:**
  > *"Dengan metode `StratifiedGroupKFold`, kami memastikan tidak ada gambar dari sesi pemotretan yang sama terpecah di antara data latih dan data uji. Assertion `nunique() == 1` memvalidasi integritas ini secara matematis."*

---

### Tahap 4: Preprocessing Aspect-Ratio Padding & Pipeline `tf.data` (Cell 11 & 13)
* **Kodingan Kunci:**
  ```python
  # 1. Padding Hitam (Menjaga Rasio Aspek):
  resized = ImageOps.contain(img, (224, 224), method=Image.Resampling.LANCZOS)
  canvas = Image.new('RGB', (224, 224), (0, 0, 0))
  canvas.paste(resized, ...)

  # 2. Pipeline tf.data:
  def make_ds(split_name, training):
      raw = tf.keras.utils.image_dataset_from_directory(
          OUT_ROOT / split_name, labels='inferred', label_mode='int',
          class_names=CLASSES, image_size=(224, 224), shuffle=False)
      ds = raw.cache()
      if training:
          ds = ds.shuffle(len(paths), seed=SEED)
      return ds.batch(32).prefetch(AUTOTUNE)
  ```
* **Narasi Presentasi ke Dosen:**
  > *"Uang kertas berbentuk persegi panjang, sehingga jika di-*resize* biasa akan terdistorsi (*stretching*). Kami menerapkan **letterbox padding hitam** agar rasio geometris asli terjaga. Pipeline data dioptimalkan menggunakan `.cache()` dan `.prefetch(AUTOTUNE)` untuk throughput maksimal."*

---

### Tahap 5: Arsitektur FNN Baseline — Replikasi Paper (Cell 15)
* **Kodingan Kunci:**
  ```python
  def build_fnn_baseline_paper():
      inp = tf.keras.Input(shape=(224, 224, 3))
      x = layers.Rescaling(1./255)(inp)
      x = layers.Flatten()(x)                       # 224 * 224 * 3 = 150.528 fitur
      x = layers.Dense(128, activation='relu')(x)   # 150.528 * 128 + 128 = 19.267.712 param
      out = layers.Dense(7, activation='softmax')(x)
      model = tf.keras.Model(inp, out)
      model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
      return model
  ```
* **Analisis Parameter:**
  * Total Parameter: **19.268.615 parameter** ($\approx 73.5\text{ MB}$).
  * Akurasi Validasi Maksimal: Hanya **$37.50\%$**.
* **Narasi Presentasi ke Dosen:**
  > *"Baseline paper Aprillia dkk. langsung meratakan gambar $224 \times 224 \times 3$ menjadi vektor 1D sebesar $150.528$ fitur. Layer `Dense(128)` pertama langsung menyedot 19,2 juta parameter, memicu pemborosan memori ekstrem dan menghafal piksel (*overfitting* parah)."*

---

### Tahap 6: Tuning Arsitektur FNN & Regulasi Modern (Cell 17)
* **Kodingan Kunci:**
  ```python
  def build_fnn_tuned(img_resize, hidden_units, dropout_rate, with_aug, name):
      inp = tf.keras.Input(shape=(224, 224, 3))
      x = layers.Resizing(*img_resize, antialias=True)(inp) # Reduksi ke 64x64 atau 96x96
      if with_aug:
          x = build_augmentation()(x)
      x = layers.Rescaling(1./255)(x)
      x = layers.Flatten()(x)
      
      for units in hidden_units:
          x = layers.Dense(units, kernel_regularizer=regularizers.l2(1e-4))(x)
          x = layers.BatchNormalization()(x)
          x = layers.Activation('relu')(x)
          x = layers.Dropout(dropout_rate)(x)
          
      out = layers.Dense(7, activation='softmax')(x)
      return model
  ```
* **Inovasi yang Diterapkan:**
  1. **Reduksi Spasial ($64 \times 64$)**: Memangkas fitur Flatten menjadi $12.288$, memotong parameter hingga $66\%$ menjadi **6,46 juta parameter**.
  2. **Batch Normalization**: Menstabilkan distribusi aktivasi antar-lapisan Dense.
  3. **Penalti Bobot $L_2$ ($10^{-4}$) & Dropout ($0.4$)**: Menahan memorisasi piksel secara agresif.

---

### Tahap 7: Hasil Eksperimen Validasi & Mengapa Tanpa Augmentasi Menang (Cell 19)

| Nama Model FNN | Resolusi | Augmentasi | Val Accuracy | 95% Wilson CI | Val Loss | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`fnn_tuned_no_aug`** | **$64 \times 64$** | **False** | **$56.82\%$** | **$49.4\% - 63.9\%$** | **$1.6018$** | **TERPILIH** |
| `fnn_tuned_96_no_aug` | $96 \times 96$ | False | $38.07\%$ | $31.2\% - 45.4\%$ | $2.3532$ | Gugur |
| `fnn_baseline_paper` | $224 \times 224$ | False | $37.50\%$ | $30.7\% - 44.8\%$ | $1.8758$ | Gugur |
| `fnn_tuned_with_aug` | $64 \times 64$ | **True** | **$35.23\%$** | $28.6\% - 42.5\%$ | $1.7410$ | **Anjlok!** |
| `fnn_tuned_96_with_aug`| $96 \times 96$ | **True** | **$34.09\%$** | $27.5\% - 41.4\%$ | $1.9270$ | **Anjlok!** |

* **Narasi Ilmiah untuk Dosen (Poin Paling Krusial):**
  > *"Bapak/Ibu Dosen, penambahan augmentasi pada FNN justru membuat akurasi validasi anjlok drastis dari **$56.82\%$ menjadi $35.23\%$**.
  > 
  > **Penyebab Teoretis:** FNN tidak memiliki sifat **Translation Invariance**. Setiap neuron di layer pertama terikat absolut pada posisi piksel tertentu. Saat citra dirotasi atau digeser oleh augmentasi, nilai piksel berpindah ke neuron yang berbeda. FNN menganggapnya sebagai citra acak baru, mengalami *underfitting* parah, dan gagal mengenali pola nominal. Oleh karena itu, model yang terpilih murni dari data validasi adalah **`fnn_tuned_no_aug`**."*

---

### Tahap 8: Evaluasi Final pada Test Set Hold-Out ($n=179$) & Uji Cahaya (Cell 22 & 25)
* **Hasil Evaluasi Test Set ($n=179$):**
  * **Test Accuracy**: **$58.66\%$** ($95\%$ Wilson CI: $51.3\% - 65.6\%$)
  * **Top-2 Accuracy**: **$75.42\%$**
  * **Macro F1-Score**: **$0.4852$**
  * **Test Loss**: **$1.3524$**

* **Hasil Uji Ketahanan Pencahayaan (*Lighting Robustness*):**
  * Kondisi Normal ($1.00\times$): Akurasi $58.66\%$
  * Kondisi Terang ($1.25\times$): Akurasi $51.40\%$
  * Kondisi Gelap ($0.75\times$): **Akurasi RUNTUH TOTAL ke $17.32\%$**!

---

## 3. Matriks Perbandingan Lintas Arsitektur pada Test Set Sama ($n=179$)

| Model | Paradigma Arsitektur | Total Parameter | Test Accuracy | 95% Wilson CI | Macro F1 | Uji Gelap ($0.75\times$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN** | 4-Blok Conv2D + BN (*Scratch*) | $\approx 5.9\text{ M}$ | **$99.44\%$** | $96.9\% - 99.9\%$ | **$0.9944$** | **$91.62\%$** |
| **MobileNetV2** | Inverted Residual (*Transfer Learning*) | $\approx 2.4\text{ M}$ | **$95.53\%$** | $91.4\% - 97.7\%$ | **$0.9547$** | **$92.18\%$** |
| **FNN (Tuned)** | Resizing 64x64 + MLP | $\approx 6.5\text{ M}$ | **$58.66\%$** | $51.3\% - 65.6\%$ | **$0.4852$** | **$17.32\%$** |
| *FNN Paper (2024)*| Flatten 224x224 + Dense 128 | $\approx 19.3\text{ M}$ | $53.57\%$ | N/A | N/A | Rentan |

---

## 4. Antisipasi Pertanyaan Dosen (Q&A Defense Guide)

### Pertanyaan 1: *"Mengapa FNN akurasinya hanya 58% padahal parameternya lebih banyak daripada MobileNetV2?"*
**Jawaban:**
> *"Karena parameter pada FNN teralokasi secara tidak efisien pada koneksi *Fully-Connected* ($150.528 \times 128$) yang memperlakukan piksel sebagai data tabular independen. FNN kehilangan **Spatial Inductive Bias**—yaitu pemahaman bahwa piksel-piksel yang bersebelahan membentuk garis, teks nominal, dan tekstur pengaman. Sebaliknya, MobileNetV2 memanfaatkan filter konvolusi berbobot kecil ($3 \times 3$) yang dibagikan ke seluruh bidang citra (*weight sharing*), sehingga jauh lebih cerdas mengenali fitur visual."*

### Pertanyaan 2: *"Kenapa saat diuji di kondisi gelap, akurasi FNN langsung anjlok ke 17%?"*
**Jawaban:**
> *"FNN tidak memiliki sifat **Illumination Invariance**. Neuron FNN mempelajari nilai numerik absolut dari piksel $[0, 255]$. Ketika citra digelapkan $25\%$ ($0.75\times$), nilai input bergeser serempak ke bawah, memicu aktivasi neuron yang salah total. Berbeda dengan CNN yang mendeteksi perubahan gradien lokal (*edges/contrast differences*), sehingga bentuk angka nominal tetap terbaca meskipun ruangan redup."*

### Pertanyaan 3: *"Apakah kodingan FNN Anda sudah bebas dari kebocoran data (*data leakage*)?"*
**Jawaban:**
> *"Sudah 100% bebas kebocoran, Bapak/Ibu. Kami menggunakan `StratifiedGroupKFold` dengan pengelompokan dHash ($d_H \le 6$) pada citra *burst-shot*. Selain itu, seleksi model terbaik murni dilakukan pada set validasi. Test set ($n=179$) berstatus *unseen* dan baru dievaluasi satu kali di akhir eksperimen menggunakan bobot model yang telah dikunci."*

