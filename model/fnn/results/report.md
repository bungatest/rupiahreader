# Laporan Akhir Eksperimen Feedforward Neural Network (FNN) Rupiah



## 1. Ringkasan Eksekutif
- **Tugas**: Klasifikasi nominal uang kertas Rupiah (7 kelas) menggunakan arsitektur FNN / MLP murni.
- **Model Terpilih**: `fnn_tuned_no_aug`
- **Test Accuracy**: **58.66%** (95% Wilson CI: 51.3% - 65.6%, n=179)
- **Top-2 Accuracy**: 75.42%
- **Macro F1-Score**: 0.4852

## 2. Dataset & Split
- Total citra valid: 1249 (1249 unik)
- Group near-duplicate (dHash $\le 6$): 1126
- Pembagian data: Train 894 (71.6%), Val 176 (14.1%), Test 179 (14.3%)
- Status kecocokan dengan split Custom CNN: 100% IDENTIK

## 3. Hasil Validasi Seluruh Eksperimen
```
                 name  val_accuracy  val_ci_low  val_ci_high  val_loss  n_params  train_sec
     fnn_tuned_no_aug        0.5682      0.4943       0.6391    1.6018   6460679    86.3219
  fnn_tuned_96_no_aug        0.3807      0.3122       0.4542    2.3532  14292487    64.6427
   fnn_baseline_paper        0.3750      0.3069       0.4485    1.8758  19268615    97.9754
   fnn_tuned_with_aug        0.3523      0.2855       0.4253    1.7410   6460679    30.5050
fnn_tuned_96_with_aug        0.3409      0.2749       0.4137    1.9270  14292487    65.8645
```

## 4. Ketahanan Pencahayaan
- **gelap (0.75x)**: Accuracy 17.32% (95% CI 12.5% - 23.5%)\n- **normal (1.00x)**: Accuracy 58.66% (95% CI 51.3% - 65.6%)\n- **terang (1.25x)**: Accuracy 16.76% (95% CI 12.0% - 22.9%)\n
## 5. Komparasi Lintas Model (Test Set Sama, n=179)
```
                           Model Test Accuracy        95% CI Macro F1 Top-2 Acc
     Custom CNN (tuned_with_aug)        99.44% 96.9% - 99.9%   0.9944    100.0%
   MobileNetV2 Transfer Learning        95.53% 91.4% - 97.7%   0.9547    98.88%
          FNN (fnn_tuned_no_aug)        58.66% 51.3% - 65.6%   0.4852    75.42%
FNN Paper (Aprillia dkk., 2024)*        53.57%           N/A      N/A       N/A
```

## 6. Kesimpulan Utama
FNN murni memiliki batasan mendasar dalam pengenalan citra karena ketiadaan sifat translation invariance dan penghancuran relasi spasial oleh operasi Flatten. Custom CNN unggul jauh (99.44% vs 58.66%) dengan parameter yang jauh lebih hemat.
