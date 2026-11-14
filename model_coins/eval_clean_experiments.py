import os, sys
sys.stdout.reconfigure(encoding='utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
mpl_cache = os.path.join(os.path.dirname(__file__), 'cache', 'mpl')
os.makedirs(mpl_cache, exist_ok=True)
os.environ['MPLCONFIGDIR'] = mpl_cache

print("=== STARTING SCRIPT ===", flush=True)

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("=== 1. MEMUAT INVENTORY BERSIH ===", flush=True)
df = pd.read_csv(r"model_coins\splits\dataset_detailed_inventory.csv")
df_unique = df[~df['is_duplicate']].reset_index(drop=True)
print(f"Total citra unik: {len(df_unique)}", flush=True)
print(df_unique['target_class_5way'].value_counts().to_string(), flush=True)

print("\n=== 2. EKSTRAKSI FITUR EFFICIENTNETV2-B0 (DARI BEST.KERAS) ===", flush=True)
full_model = tf.keras.models.load_model(r"model_coins\models\best.keras")
feature_model = tf.keras.Model(inputs=full_model.input, outputs=full_model.get_layer('global_average_pooling2d').output)

imgs = []
for p in df_unique['path']:
    im = Image.open(p).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
    imgs.append(np.array(im, dtype=np.float32))
X = np.stack(imgs, axis=0)

# Batched feature extraction
feat_batches = []
batch_size = 16
for i in range(0, len(X), batch_size):
    b = X[i:i+batch_size]
    fb = feature_model(b, training=False).numpy()
    feat_batches.append(fb)
feats_all = np.concatenate(feat_batches, axis=0)
print(f"Fitur berhasil diekstrak: shape {feats_all.shape}", flush=True)

# -------------------------------------------------------------
# SKENARIO A: 4 KELAS ANGKA BERSIH (N=55)
# -------------------------------------------------------------
print("\n" + "="*50)
print("SKENARIO A: 4 KELAS ANGKA BERSIH (HANYA SISI ANGKA)")
print("="*50)
mask_angka = (df_unique['coin_side'] == 'angka').values
df_angka = df_unique[mask_angka].reset_index(drop=True)
X_a = feats_all[mask_angka]
y_a = df_angka['original_class'].astype(str).values
labels_a = ['100', '200', '500', '1000']
label_to_idx_a = {l: i for i, l in enumerate(labels_a)}
y_a_idx = np.array([label_to_idx_a[y] for y in y_a])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs_a = np.zeros((len(df_angka), 4), dtype=np.float32)

for train_idx, val_idx in skf.split(X_a, y_a_idx):
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_a[train_idx], y_a_idx[train_idx])
    oof_probs_a[val_idx] = clf.predict_proba(X_a[val_idx])

oof_preds_a = np.argmax(oof_probs_a, axis=1)
acc_a = accuracy_score(y_a_idx, oof_preds_a)
f1_a = f1_score(y_a_idx, oof_preds_a, average='macro')
prec_a = precision_score(y_a_idx, oof_preds_a, average='macro', zero_division=0)
rec_a = recall_score(y_a_idx, oof_preds_a, average='macro')

print(f"OOF Accuracy Skenario A (N={len(df_angka)}): {acc_a*100:.2f}% (Peluang acak: 25.0%)")
print(f"OOF Macro F1: {f1_a:.4f} | Precision: {prec_a:.4f} | Recall: {rec_a:.4f}")
print("Confusion Matrix Skenario A (100, 200, 500, 1000):")
print(confusion_matrix(y_a_idx, oof_preds_a))

# -------------------------------------------------------------
# SKENARIO B: 5 KELAS REFORMULASI (+ SISI_GARUDA) (N=89)
# -------------------------------------------------------------
print("\n" + "="*50)
print("SKENARIO B: 5 KELAS REFORMULASI (+ SISI_GARUDA)")
print("="*50)
labels_b = ['100', '200', '500', '1000', 'sisi_garuda']
label_to_idx_b = {l: i for i, l in enumerate(labels_b)}
y_b = df_unique['target_class_5way'].values
y_b_idx = np.array([label_to_idx_b[y] for y in y_b])

oof_probs_b = np.zeros((len(df_unique), 5), dtype=np.float32)

for train_idx, val_idx in skf.split(feats_all, y_b_idx):
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(feats_all[train_idx], y_b_idx[train_idx])
    oof_probs_b[val_idx] = clf.predict_proba(feats_all[val_idx])

oof_preds_b = np.argmax(oof_probs_b, axis=1)
acc_b = accuracy_score(y_b_idx, oof_preds_b)
f1_b = f1_score(y_b_idx, oof_preds_b, average='macro')
prec_b = precision_score(y_b_idx, oof_preds_b, average='macro', zero_division=0)
rec_b = recall_score(y_b_idx, oof_preds_b, average='macro')

print(f"OOF Accuracy Skenario B (N={len(df_unique)}): {acc_b*100:.2f}% (Peluang acak: 20.0%)")
print(f"OOF Macro F1: {f1_b:.4f} | Precision: {prec_b:.4f} | Recall: {rec_b:.4f}")
cm_b = confusion_matrix(y_b_idx, oof_preds_b)
print("Confusion Matrix Skenario B:")
print(pd.DataFrame(cm_b, index=labels_b, columns=labels_b))

# -------------------------------------------------------------
# 3. TEMPERATURE SCALING & ECE PADA OOF
# -------------------------------------------------------------
print("\n=== 3. TEMPERATURE SCALING & ECE PADA PREDIKSI OOF ===")
# Convert probabilities to unnormalized logits approx: log(p + eps)
eps = 1e-7
oof_logits = np.log(np.clip(oof_probs_b, eps, 1.0 - eps))

def nll_func(T):
    scaled_logits = oof_logits / T
    # Softmax
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    nll = -np.mean(np.log(probs[np.arange(len(y_b_idx)), y_b_idx] + eps))
    return nll

res_opt = minimize(nll_func, x0=[1.2], bounds=[(0.1, 10.0)])
opt_temp = float(res_opt.x[0])
print(f"Optimal Temperature (dipilih murni dari OOF): T = {opt_temp:.3f}")

def compute_ece(probs, true_labels, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == true_labels)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []
    
    for i in range(n_bins):
        b_low, b_high = bin_boundaries[i], bin_boundaries[i+1]
        mask = (confidences > b_low) & (confidences <= b_high) if i > 0 else (confidences >= b_low) & (confidences <= b_high)
        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            ece += (n_in_bin / len(true_labels)) * np.abs(bin_acc - bin_conf)
            bin_stats.append((b_low, b_high, n_in_bin, bin_acc, bin_conf))
        else:
            bin_stats.append((b_low, b_high, 0, 0.0, 0.0))
    return ece, bin_stats

ece_before, stats_before = compute_ece(oof_probs_b, y_b_idx)
# Scaled probs
scaled_logits = oof_logits / opt_temp
exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
scaled_probs_b = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
ece_after, stats_after = compute_ece(scaled_probs_b, y_b_idx)

print(f"ECE Sebelum Temperature Scaling: {ece_before:.4f}")
print(f"ECE Sesudah Temperature Scaling : {ece_after:.4f}")

# -------------------------------------------------------------
# 4. CALIBRATED THRESHOLD SELECTION PADA OOF (RISK-COVERAGE)
# -------------------------------------------------------------
print("\n=== 4. ANALISIS RISK-COVERAGE & PEMILIHAN THRESHOLD DARI OOF ===")
thresholds = np.linspace(0.25, 0.90, 66)
risk_coverage_data = []

conf_scaled = np.max(scaled_probs_b, axis=1)
pred_scaled = np.argmax(scaled_probs_b, axis=1)

for th in thresholds:
    mask = (conf_scaled >= th)
    cov = np.mean(mask)
    if np.sum(mask) > 0:
        prec = accuracy_score(y_b_idx[mask], pred_scaled[mask])
    else:
        prec = 1.0
    risk_coverage_data.append((th, cov, prec))

df_rc = pd.DataFrame(risk_coverage_data, columns=['threshold', 'coverage', 'precision'])

# Target: Presisi >= 90% atau Presisi >= 95%
th_90 = df_rc[df_rc['precision'] >= 0.90]['threshold'].min()
th_95 = df_rc[df_rc['precision'] >= 0.95]['threshold'].min()
cov_90 = df_rc[df_rc['threshold'] == th_90]['coverage'].values[0] if not pd.isna(th_90) else 0.0
cov_95 = df_rc[df_rc['threshold'] == th_95]['coverage'].values[0] if not pd.isna(th_95) else 0.0

print(f"Threshold untuk Presisi >= 90%: tau = {th_90:.2f} (Coverage = {cov_90*100:.1f}%)")
print(f"Threshold untuk Presisi >= 95%: tau = {th_95:.2f} (Coverage = {cov_95*100:.1f}%)")

# Pilih threshold operasional dari OOF (misal 0.55 untuk presisi tinggi dengan coverage memadai)
tau_selected = float(th_90) if not pd.isna(th_90) else 0.60
print(f"\n>> THRESHOLD OPERASIONAL TERPILIH DARI OOF: tau* = {tau_selected:.2f}")

# Simpan Ringkasan CSV
prec_at_tau = df_rc[df_rc['threshold'] == tau_selected]['precision'].values[0] * 100.0
cov_at_tau = df_rc[df_rc['threshold'] == tau_selected]['coverage'].values[0] * 100.0

df_summary = pd.DataFrame([
    {'Metric': 'Skenario A (4 Kelas Bersih) OOF Acc', 'Value': f'{acc_a*100:.2f}%'},
    {'Metric': 'Skenario A OOF Macro F1', 'Value': f'{f1_a:.4f}'},
    {'Metric': 'Skenario B (5 Kelas: +Garuda) OOF Acc', 'Value': f'{acc_b*100:.2f}%'},
    {'Metric': 'Skenario B OOF Macro F1', 'Value': f'{f1_b:.4f}'},
    {'Metric': 'Optimal Temperature (T)', 'Value': f'{opt_temp:.3f}'},
    {'Metric': 'ECE Sebelum Scaling', 'Value': f'{ece_before:.4f}'},
    {'Metric': 'ECE Sesudah Scaling', 'Value': f'{ece_after:.4f}'},
    {'Metric': 'Threshold Terpilih (tau*)', 'Value': f'{tau_selected:.2f}'},
    {'Metric': 'Precision pada tau* (OOF)', 'Value': f'{prec_at_tau:.1f}%'},
    {'Metric': 'Coverage pada tau* (OOF)', 'Value': f'{cov_at_tau:.1f}%'},
])
df_summary.to_csv(r"model_coins\results\clean_audit_metrics.csv", index=False)
print("Metrik disimpan ke model_coins/results/clean_audit_metrics.csv")

# -------------------------------------------------------------
# 5. VISUALISASI DIAGNOSTIK
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Confusion Matrix 5 Kelas
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=labels_b, yticklabels=labels_b)
axes[0].set_title(f"Confusion Matrix 5-Kelas (OOF N={len(df_unique)})\nAkurasi: {acc_b*100:.1f}%", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Prediksi Model")
axes[0].set_ylabel("Label Sebenarnya")

# 2. Kurva Risk-Coverage
axes[1].plot(df_rc['coverage'] * 100, df_rc['precision'] * 100, color='darkgreen', lw=2.5, label='Presisi Jawaban')
axes[1].axvline(cov_90 * 100, color='red', linestyle='--', label=f'Threshold {tau_selected:.2f} (Cov {cov_90*100:.0f}%)')
axes[1].set_xlabel("Coverage (% Sesi Dijawab)", fontsize=10)
axes[1].set_ylabel("Precision pada Jawaban (%)", fontsize=10)
axes[1].set_title("Kurva Risk-Coverage pada OOF", fontsize=11, fontweight='bold')
axes[1].set_ylim(40, 105)
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend(loc='lower left')

# 3. Reliability Diagram (Sebelum vs Sesudah T-Scaling)
confs_b = [s[4] for s in stats_after if s[2] > 0]
accs_b = [s[3] for s in stats_after if s[2] > 0]
counts_b = [s[2] for s in stats_after if s[2] > 0]

axes[2].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Kalibrasi Sempurna')
axes[2].plot(confs_b, accs_b, 'o-', color='purple', lw=2, label=f'Setelah T-Scaling (ECE={ece_after:.3f})')
axes[2].set_xlim(0.0, 1.0)
axes[2].set_ylim(0.0, 1.0)
axes[2].set_xlabel("Rata-rata Keyakinan (Confidence)", fontsize=10)
axes[2].set_ylabel("Akurasi Sebenarnya", fontsize=10)
axes[2].set_title(f"Reliability Diagram (Full [0, 1])\nT = {opt_temp:.2f}", fontsize=11, fontweight='bold')
axes[2].grid(True, linestyle=':', alpha=0.6)
axes[2].legend(loc='upper left')

plt.tight_layout()
plot_path = r"model_coins\results\clean_analysis_plots.png"
plt.savefig(plot_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Visualisasi disimpan ke {plot_path}")
