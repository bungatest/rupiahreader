import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
results_dir = BASE_DIR / 'results'
os.environ['MPLCONFIGDIR'] = os.path.join(os.environ.get("TEMP", "/tmp"), "mpl_cache")

oof_probs = np.load(results_dir / 'oof_5class_probs.npy')
y_true = np.load(results_dir / 'oof_5class_targets.npy')
df_manifest = pd.read_csv(BASE_DIR / 'splits' / 'clean_89_manifest.csv', encoding='utf-8')
cm_df = pd.read_csv(results_dir / 'oof_5class_confusion_matrix.csv', index_col=0)

class_names = ['100', '1000', '200', '500', 'sisi_garuda']
N_TOTAL = len(y_true)

def wilson_ci(k, n, confidence=0.95):
    if n == 0: return 0.0, 0.0
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return round(float(max(0.0, center - spread)), 4), round(float(min(1.0, center + spread)), 4)

# Akurasi per sisi
is_angka = (df_manifest['coin_side'] == 'angka').values
is_garuda = (df_manifest['coin_side'] == 'garuda_pahlawan').values

preds_raw = np.argmax(oof_probs, axis=1)
correct_raw = (preds_raw == y_true)

n_angka = int(np.sum(is_angka))
n_garuda = int(np.sum(is_garuda))
corr_angka = int(np.sum(correct_raw[is_angka]))
corr_garuda = int(np.sum(correct_raw[is_garuda]))
corr_total = int(np.sum(correct_raw))

acc_angka = corr_angka / n_angka
acc_garuda = corr_garuda / n_garuda
acc_total = corr_total / N_TOTAL

ci_angka = wilson_ci(corr_angka, n_angka)
ci_garuda = wilson_ci(corr_garuda, n_garuda)
ci_total = wilson_ci(corr_total, N_TOTAL)

# Temperature scaled probs
T_star = 0.663
eps = 1e-12
clipped = np.clip(oof_probs, eps, 1.0 - eps)
scaled_logits = np.log(clipped) / T_star
exp_s = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
probs_cal = exp_s / np.sum(exp_s, axis=1, keepdims=True)

# Reliability binning (10 bins)
def compute_ece_with_counts(probs, y, n_bins=8):
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accs = (preds == y).astype(float)
    bins = np.linspace(0.20, 1.0, n_bins + 1)
    ece = 0.0
    b_confs, b_accs, b_counts, b_mids = [], [], [], []
    for i in range(n_bins):
        low, high = bins[i], bins[i+1]
        m = (confs >= low) & (confs <= high)
        cnt = np.sum(m)
        b_mids.append((low + high) / 2)
        b_counts.append(cnt)
        if cnt > 0:
            avg_c = np.mean(confs[m])
            avg_a = np.mean(accs[m])
            ece += (cnt / len(y)) * np.abs(avg_a - avg_c)
            b_confs.append(avg_c)
            b_accs.append(avg_a)
        else:
            b_confs.append((low + high) / 2)
            b_accs.append(0.0)
    return ece, b_confs, b_accs, b_counts, b_mids

ece_raw, bc_r, ba_r, cnt_r, mids = compute_ece_with_counts(oof_probs, y_true)
ece_cal, bc_c, ba_c, cnt_c, _ = compute_ece_with_counts(probs_cal, y_true)

# Risk coverage data
rc_df = pd.read_csv(results_dir / 'oof_5class_risk_coverage.csv')

# --- Plot 4 Panel Bersih dan Presisi ---
fig = plt.figure(figsize=(15, 12))
plt.rcParams['font.sans-serif'] = 'Arial'

# Panel A: Akurasi OOF Terpisah per Sisi Koin
ax1 = fig.add_subplot(2, 2, 1)
categories = ['Sisi Angka\n(N=55)', 'Sisi Garuda\n(N=34)', 'Total 5-Kelas\n(N=89)']
acc_vals = [acc_angka * 100, acc_garuda * 100, acc_total * 100]
err_low = [acc_vals[0] - ci_angka[0]*100, acc_vals[1] - ci_garuda[0]*100, acc_vals[2] - ci_total[0]*100]
err_high = [ci_angka[1]*100 - acc_vals[0], ci_garuda[1]*100 - acc_vals[1], ci_total[1]*100 - acc_vals[2]]

bars = ax1.bar(categories, acc_vals, color=['#2ca02c', '#9467bd', '#1f77b4'], width=0.45,
               yerr=[err_low, err_high], capsize=6, edgecolor='black', alpha=0.85)
ax1.set_ylim(0, 115)
ax1.set_ylabel('Akurasi OOF (%)', fontsize=11, fontweight='bold')
ax1.set_title('(A) Akurasi OOF per Sisi Koin (Wilson 95% CI)', fontsize=12, fontweight='bold', pad=12)

# Taruh label akurasi DI ATAS ujung error bar agar tidak menimpa garis
sample_k = [34, 26, 60]
sample_n = [55, 34, 89]
for bar, val, e_h, k, n_cnt in zip(bars, acc_vals, err_high, sample_k, sample_n):
    top_pos = val + e_h + 3.0
    ax1.text(bar.get_x() + bar.get_width()/2, top_pos, f'{val:.1f}%\n({k}/{n_cnt})',
             ha='center', va='bottom', fontweight='bold', fontsize=9.5)
ax1.grid(axis='y', linestyle=':', alpha=0.6)

# Panel B: Confusion Matrix 5x5 Heatmap
ax2 = fig.add_subplot(2, 2, 2)
im = ax2.imshow(cm_df.values, cmap='Blues', interpolation='nearest')
ax2.set_xticks(range(5))
ax2.set_yticks(range(5))
ax2.set_xticklabels(class_names, rotation=30, ha='right', fontsize=10)
ax2.set_yticklabels(class_names, fontsize=10)
ax2.set_xlabel('Prediksi Model (OOF)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Label Aktual', fontsize=11, fontweight='bold')
ax2.set_title('(B) Confusion Matrix 5x5 OOF (N=89)', fontsize=12, fontweight='bold', pad=12)
for i in range(5):
    for j in range(5):
        val = cm_df.values[i, j]
        color = 'white' if val > cm_df.values.max() / 2 else 'black'
        ax2.text(j, i, str(val), ha='center', va='center', color=color, fontweight='bold')

# Panel C: Reliability Diagram dengan Histogram Sampel
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot([0.2, 1], [0.2, 1], 'k--', label='Kalibrasi Sempurna', lw=1.5)
ax3.plot(bc_r, ba_r, 's-', color='#d62728', label=f'Mentah (ECE = {ece_raw:.3f})', lw=2)
ax3.plot(bc_c, ba_c, 'o-', color='#1f77b4', label=f'T*={T_star:.3f} (ECE = {ece_cal:.3f})', lw=2)
ax3.set_xlabel('Confidence Rata-rata per Bin', fontsize=11, fontweight='bold')
ax3.set_ylabel('Akurasi Sebenarnya per Bin', fontsize=11, fontweight='bold')
ax3.set_title(f'(C) Reliability Curve (Catatan: Bergerigi akibat N={N_TOTAL})', fontsize=12, fontweight='bold', pad=12)
ax3.set_xlim(0.2, 1.0)
ax3.set_ylim(0.0, 1.05)
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, linestyle=':', alpha=0.6)

# Histogram sample support ditaruh di inset kecil dalam plot C
ax3_inset = ax3.inset_axes([0.62, 0.12, 0.35, 0.30])
ax3_inset.bar(mids, cnt_c, width=0.08, color='#1f77b4', alpha=0.6, edgecolor='black')
ax3_inset.set_title('Distribusi Sampel per Bin', fontsize=8, fontweight='bold')
ax3_inset.set_xlabel('Bin Conf', fontsize=7)
ax3_inset.set_ylabel('Jumlah Citra', fontsize=7)
ax3_inset.tick_params(labelsize=7)

# Panel D: Risk-Coverage Curve (Bersih Tanpa Text Clutter)
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(rc_df['threshold'], rc_df['precision_pct'], 'o-', label='Precision (%)', color='#2ca02c', lw=2.5)
ax4.plot(rc_df['threshold'], rc_df['coverage_pct'], 's--', label='Coverage (%)', color='#ff7f0e', lw=2.5)
ax4.axhline(90.0, color='gray', linestyle=':', alpha=0.7, label='Target Presisi 90%')
ax4.axvline(0.70, color='#d62728', linestyle=':', lw=1.5, label='Ambang Rekomendasi (tau=0.70)')
ax4.set_xlabel('Confidence Threshold (tau, sesudah T*)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Persentase (%)', fontsize=11, fontweight='bold')
ax4.set_title('(D) Kurva Risk-Coverage OOF (Terstandarisasi T*=0.663)', fontsize=12, fontweight='bold', pad=12)
ax4.set_ylim(0, 105)
ax4.legend(loc='lower left', fontsize=9)
ax4.grid(True, linestyle=':', alpha=0.6)

# Kotak tabel ringkas support di bagian kanan bawah plot D (tidak menimpa judul!)
table_data = [
    ["tau=0.50", "n=65/89", "Cov: 73%", "Prec: 73.8%"],
    ["tau=0.70", "n=31/89", "Cov: 35%", "Prec: 96.8% [CI: 83.8-99.4%]"],
    ["tau=0.85", "n=15/89", "Cov: 17%", "Prec: 100% [CI: 79.6-100%]"],
    ["tau=0.90", "n=8/89",  "Cov: 9%",  "Prec: 100% [CI: 67.6-100%]"],
]
table = ax4.table(cellText=table_data, colLabels=["Threshold", "Support (n)", "Coverage", "Precision (Wilson CI)"],
                  cellLoc='center', loc='bottom', bbox=[0.0, -0.42, 1.0, 0.28])
table.auto_set_font_size(False)
table.set_fontsize(8)

plt.subplots_adjust(bottom=0.18, hspace=0.35, wspace=0.25)
out_plot_path = results_dir / 'clean_analysis_plots.png'
plt.savefig(out_plot_path, dpi=200, bbox_inches='tight')
plt.close()

print(f"SUCCESS: Saved updated plot without title collisions to {out_plot_path}")
