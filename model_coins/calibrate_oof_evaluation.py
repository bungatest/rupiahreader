import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
os.environ['MPLCONFIGDIR'] = os.path.join(os.environ.get("TEMP", "/tmp"), "mpl_cache")

# 1. Muat artefak OOF resmi yang baru dihasilkan
results_dir = BASE_DIR / 'results'
oof_probs = np.load(results_dir / 'oof_5class_probs.npy')
y_true = np.load(results_dir / 'oof_5class_targets.npy')
df_manifest = pd.read_csv(BASE_DIR / 'splits' / 'clean_89_manifest.csv', encoding='utf-8')

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

# 2. Akurasi per sisi koin
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

print("="*65)
print("HASIL EVALUASI OOF RESMI (5-KELAS PADA 89 SAMPEL UNIK)")
print("="*65)
print(f"Total Sampel Unik:        {N_TOTAL}")
print(f"Total 5-Kelas:            {corr_total}/{N_TOTAL} ({corr_total/N_TOTAL*100:.2f}%) | Wilson 95% CI: [{ci_total[0]*100:.1f}%, {ci_total[1]*100:.1f}%]")
print(f"Sisi Angka Murni:         {corr_angka}/{n_angka} ({corr_angka/n_angka*100:.2f}%) | Wilson 95% CI: [{ci_angka[0]*100:.1f}%, {ci_angka[1]*100:.1f}%]")
print(f"Sisi Garuda:              {corr_garuda}/{n_garuda} ({corr_garuda/n_garuda*100:.2f}%) | Wilson 95% CI: [{ci_garuda[0]*100:.1f}%, {ci_garuda[1]*100:.1f}%]")
print("="*65)

# 3. Fitting Temperature Scaling T* tunggal via NLL minimization
eps = 1e-12
clipped_probs = np.clip(oof_probs, eps, 1.0 - eps)
raw_logits = np.log(clipped_probs)

def nll_objective(t):
    scaled = raw_logits / t
    exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
    correct_p = probs[np.arange(N_TOTAL), y_true]
    return -np.mean(np.log(np.clip(correct_p, eps, 1.0)))

res = minimize_scalar(nll_objective, bounds=(0.5, 3.0), method='bounded')
T_star = round(float(res.x), 3)

def compute_ece(probs, y, n_bins=10):
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accs = (preds == y).astype(float)
    bins = np.linspace(0.20, 1.0, n_bins + 1)
    ece = 0.0
    bin_confs, bin_accs = [], []
    for i in range(n_bins):
        low, high = bins[i], bins[i+1]
        m = (confs >= low) & (confs <= high)
        cnt = np.sum(m)
        if cnt > 0:
            avg_c = np.mean(confs[m])
            avg_a = np.mean(accs[m])
            ece += (cnt / len(y)) * np.abs(avg_a - avg_c)
            bin_confs.append(avg_c)
            bin_accs.append(avg_a)
    return ece, bin_confs, bin_accs

scaled_logits = raw_logits / T_star
exp_cal = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
probs_cal = exp_cal / np.sum(exp_cal, axis=1, keepdims=True)

ece_raw, bc_raw, ba_raw = compute_ece(oof_probs, y_true)
ece_cal, bc_cal, ba_cal = compute_ece(probs_cal, y_true)

print(f"\nTEMPERATURE SCALING (OOF N={N_TOTAL}):")
print(f"Optimal Temperature T*: {T_star:.3f} (NLL berkurang dari {nll_objective(1.0):.4f} ke {nll_objective(T_star):.4f})")
print(f"ECE Mentah (T=1.000):   {ece_raw:.4f}")
print(f"ECE Terkalibrasi (T*):  {ece_cal:.4f}")

# 4. Kurva Risk-Coverage dengan Sample Support n/N dan Wilson CI
cal_confs = np.max(probs_cal, axis=1)
cal_preds = np.argmax(probs_cal, axis=1)
cal_correct = (cal_preds == y_true)

rc_rows = []
thresholds = [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

for th in thresholds:
    mask = cal_confs >= th
    n_ans = int(np.sum(mask))
    cov = n_ans / N_TOTAL
    if n_ans > 0:
        k_corr = int(np.sum(cal_correct[mask]))
        prec = k_corr / n_ans
        ci_low, ci_high = wilson_ci(k_corr, n_ans)
    else:
        k_corr = 0
        prec = 1.0
        ci_low, ci_high = 0.0, 1.0
        
    rc_rows.append({
        'threshold': th,
        'answered_n': n_ans,
        'total_N': N_TOTAL,
        'coverage_pct': round(cov * 100, 1),
        'correct_k': k_corr,
        'precision_pct': round(prec * 100, 1),
        'wilson_ci_95': f"[{ci_low*100:.1f}%, {ci_high*100:.1f}%]"
    })

rc_df = pd.DataFrame(rc_rows)
rc_df.to_csv(results_dir / 'oof_5class_risk_coverage.csv', index=False)
print("\nTabel Risk-Coverage OOF (Dengan Sample Support & Wilson CI):")
print(rc_df.to_string(index=False))

# 5. Visualisasi 4-Panel Ilmiah
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
plt.rcParams['font.sans-serif'] = 'Arial'

# Panel A: Akurasi OOF Terpisah per Sisi Koin
ax = axes[0, 0]
categories = ['Sisi Angka Murni\n(N=55)', 'Sisi Garuda\n(N=34)', 'Total 5-Kelas\n(N=89)']
acc_vals = [acc_angka * 100, acc_garuda * 100, acc_total * 100]
err_low = [acc_vals[0] - ci_angka[0]*100, acc_vals[1] - ci_garuda[0]*100, acc_vals[2] - ci_total[0]*100]
err_high = [ci_angka[1]*100 - acc_vals[0], ci_garuda[1]*100 - acc_vals[1], ci_total[1]*100 - acc_vals[2]]

bars = ax.bar(categories, acc_vals, color=['#2ca02c', '#9467bd', '#1f77b4'], width=0.45,
              yerr=[err_low, err_high], capsize=5, edgecolor='black', alpha=0.85)
ax.set_ylim(0, 105)
ax.set_ylabel('Akurasi OOF (%)', fontsize=11, fontweight='bold')
ax.set_title('(A) Akurasi OOF Terpisah per Sisi Koin (Wilson 95% CI)', fontsize=12, fontweight='bold')
for bar, val in zip(bars, acc_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 6, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax.grid(axis='y', linestyle=':', alpha=0.6)

# Panel B: Confusion Matrix 5x5 Heatmap
ax = axes[0, 1]
cm = pd.read_csv(results_dir / 'oof_5class_confusion_matrix.csv', index_col=0)
im = ax.imshow(cm.values, cmap='Blues', interpolation='nearest')
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=10)
ax.set_yticklabels(class_names, fontsize=10)
ax.set_xlabel('Prediksi Model (OOF)', fontsize=11, fontweight='bold')
ax.set_ylabel('Label Sebenarnya', fontsize=11, fontweight='bold')
ax.set_title('(B) Confusion Matrix 5x5 OOF (N=89)', fontsize=12, fontweight='bold')
for i in range(5):
    for j in range(5):
        val = cm.values[i, j]
        color = 'white' if val > cm.values.max() / 2 else 'black'
        ax.text(j, i, str(val), ha='center', va='center', color=color, fontweight='bold')

# Panel C: Reliability Curve (Kalibrasi Overconfidence)
ax = axes[1, 0]
ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', lw=1.5)
ax.plot(bc_raw, ba_raw, 's-', color='#d62728', label=f'Mentah (ECE = {ece_raw:.3f})', lw=2)
ax.plot(bc_cal, ba_cal, 'o-', color='#1f77b4', label=f'Temperature Scaled T*={T_star:.3f} (ECE = {ece_cal:.3f})', lw=2)
ax.set_xlabel('Confidence Rata-rata per Bin', fontsize=11, fontweight='bold')
ax.set_ylabel('Akurasi Sebenarnya per Bin', fontsize=11, fontweight='bold')
ax.set_title('(C) Reliability Diagram OOF (N=89)', fontsize=12, fontweight='bold')
ax.set_xlim(0.2, 1.0)
ax.set_ylim(0.0, 1.05)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, linestyle=':', alpha=0.6)

# Panel D: Risk-Coverage Curve dengan Sample Support
ax = axes[1, 1]
ax.plot(rc_df['threshold'], rc_df['precision_pct'], 'o-', label='Precision (%)', color='#2ca02c', lw=2.5)
ax.plot(rc_df['threshold'], rc_df['coverage_pct'], 's--', label='Coverage (%)', color='#ff7f0e', lw=2.5)
ax.axhline(90.0, color='gray', linestyle=':', alpha=0.7, label='Target Presisi 90%')
ax.set_xlabel('Confidence Threshold (tau)', fontsize=11, fontweight='bold')
ax.set_ylabel('Persentase (%)', fontsize=11, fontweight='bold')
ax.set_title('(D) Risk-Coverage Curve OOF dengan Sample Support', fontsize=12, fontweight='bold')
ax.set_ylim(0, 105)
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, linestyle=':', alpha=0.6)

# Anotasi support n pada plot D
for _, r in rc_df.iterrows():
    if r['threshold'] in [0.50, 0.70, 0.85]:
        ax.annotate(f"n={r['answered_n']}/{r['total_N']}\nPrec: {r['precision_pct']}%\nCI: {r['wilson_ci_95']}",
                    (r['threshold'], r['precision_pct']),
                    textcoords="offset points", xytext=(-25, 12), ha='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', alpha=0.85))

plt.tight_layout()
plot_path = results_dir / 'clean_analysis_plots.png'
plt.savefig(plot_path, dpi=200)
plt.close()

print(f"\nSaved updated analysis plot to {plot_path}")

# Simpan ringkasan kalibrasi resmi
cal_summary = {
    "N_total_unique": N_TOTAL,
    "N_sisi_angka": n_angka,
    "N_sisi_garuda": n_garuda,
    "accuracy_total_oof": float(acc_total),
    "ci_total_oof": list(ci_total),
    "accuracy_angka_oof": float(acc_angka),
    "ci_angka_oof": list(ci_angka),
    "accuracy_garuda_oof": float(acc_garuda),
    "ci_garuda_oof": list(ci_garuda),
    "optimal_temperature_T": T_star,
    "ece_uncalibrated": float(ece_raw),
    "ece_calibrated": float(ece_cal),
    "risk_coverage_table": rc_rows
}

with open(results_dir / 'oof_5class_calibration_summary.json', 'w', encoding='utf-8') as f:
    json.dump(cal_summary, f, indent=2)

print(f"Saved calibration summary to {results_dir / 'oof_5class_calibration_summary.json'}")
