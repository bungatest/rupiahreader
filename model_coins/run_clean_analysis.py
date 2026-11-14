import os, sys, json, traceback
import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

def main():
    # Ensure running inside model_coins directory
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    os.chdir(script_dir)

    from infer import CoinClassifier

    print("=== 1. MEMUAT INVENTORY DATASET & MODEL INFERENSI ===", flush=True)
    df = pd.read_csv(os.path.join('splits', 'dataset_detailed_inventory.csv'))
    df_unique = df[~df['is_duplicate']].reset_index(drop=True)
    print(f"Total citra unik: {len(df_unique)}", flush=True)
    print(df_unique['target_class_5way'].value_counts().to_string(), flush=True)

    engine = CoinClassifier(
        model_path=os.path.join('models', 'best.tflite'),
        labels_path=os.path.join('models', 'labels.json'),
        conf_threshold=0.0  # Raw predictions first
    )

    print("\n=== 2. MENJALANKAN INFERENSI PADA SELURUH CITRA UNIK ===", flush=True)
    results = []
    for idx, row in df_unique.iterrows():
        res = engine.predict_single(row['path'])
        results.append({
            'path': row['path'],
            'filename': row['filename'],
            'true_class': str(row['original_class']),
            'coin_side': row['coin_side'],
            'target_5way': row['target_class_5way'],
            'pred_label': res['raw_label'],
            'confidence': res['confidence'],
            'prob_100': res['probabilities'].get('100', 0.0),
            'prob_200': res['probabilities'].get('200', 0.0),
            'prob_500': res['probabilities'].get('500', 0.0),
            'prob_1000': res['probabilities'].get('1000', 0.0),
        })

    res_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    res_df.to_csv(os.path.join('results', 'clean_eval_predictions.csv'), index=False)
    print("Hasil prediksi tersimpan ke results/clean_eval_predictions.csv", flush=True)

    # -------------------------------------------------------------------------
    # ANALISIS AKURASI PER SUBSET
    # -------------------------------------------------------------------------
    print("\n=== 3. ANALISIS PERFORMA SUBSET CITRA ===", flush=True)

    # A. Sisi Angka Murni (55 citra)
    df_angka = res_df[res_df['coin_side'] == 'angka']
    acc_angka = np.mean(df_angka['true_class'] == df_angka['pred_label'])
    print(f"[A] AKURASI SISI ANGKA MURNI (N={len(df_angka)}): {acc_angka*100:.2f}%")
    for cls in ['100', '200', '500', '1000']:
        sub = df_angka[df_angka['true_class'] == cls]
        sub_acc = np.mean(sub['true_class'] == sub['pred_label']) if len(sub) > 0 else 0
        print(f"    - Kelas {cls} (N={len(sub)}): {sub_acc*100:.1f}% ({np.sum(sub['true_class'] == sub['pred_label'])}/{len(sub)})")

    # B. Sisi Garuda / Pahlawan (34 citra)
    df_garuda = res_df[res_df['coin_side'] == 'garuda_pahlawan']
    acc_garuda = np.mean(df_garuda['true_class'] == df_garuda['pred_label'])
    print(f"\n[B] AKURASI SISI GARUDA (LABEL NOISE) (N={len(df_garuda)}): {acc_garuda*100:.2f}% (Peluang acak ~25%)")
    print("    Distribusi tebakan model pada sisi Garuda (tanpa angka):")
    print(df_garuda['pred_label'].value_counts().to_string())

    # C. Total Seluruh Data (89 citra)
    acc_total = np.mean(res_df['true_class'] == res_df['pred_label'])
    print(f"\n[C] AKURASI KESELURUHAN (N={len(res_df)}): {acc_total*100:.2f}%")

    # -------------------------------------------------------------------------
    # 4. TEMPERATURE SCALING & CALIBRATED THRESHOLD SELECTION
    # -------------------------------------------------------------------------
    print("\n=== 4. TEMPERATURE SCALING & SAFE THRESHOLDING PADA SISI ANGKA ===", flush=True)

    classes = ['100', '1000', '200', '500']
    cls2idx = {c: i for i, c in enumerate(classes)}
    y_true_idx = np.array([cls2idx[c] for c in df_angka['true_class']])
    probs_matrix = np.array([[row[f'prob_{c}'] for c in classes] for _, row in df_angka.iterrows()])

    # Temperature optimization via Grid Search on NLL
    best_T, best_nll = 1.0, 1e9
    eps = 1e-7
    for T in np.linspace(0.5, 3.0, 51):
        scaled_p = np.exp(np.log(np.clip(probs_matrix, eps, 1.0)) / T)
        scaled_p = scaled_p / np.sum(scaled_p, axis=1, keepdims=True)
        nll = -np.mean(np.log(scaled_p[np.arange(len(y_true_idx)), y_true_idx] + eps))
        if nll < best_nll:
            best_nll = nll
            best_T = T

    print(f"Optimal Temperature (dipilih dari sisi angka): T* = {best_T:.2f}")

    def get_ece(p_mat, y_true, n_bins=10):
        conf = np.max(p_mat, axis=1)
        pred = np.argmax(p_mat, axis=1)
        acc = (pred == y_true)
        ece = 0.0
        for b in np.linspace(0, 1, n_bins + 1)[:-1]:
            mask = (conf >= b) & (conf < b + 0.1)
            if np.sum(mask) > 0:
                ece += (np.sum(mask) / len(y_true)) * np.abs(np.mean(acc[mask]) - np.mean(conf[mask]))
        return ece

    ece_raw = get_ece(probs_matrix, y_true_idx)
    scaled_probs = np.exp(np.log(np.clip(probs_matrix, eps, 1.0)) / best_T)
    scaled_probs = scaled_probs / np.sum(scaled_probs, axis=1, keepdims=True)
    ece_scaled = get_ece(scaled_probs, y_true_idx)
    print(f"ECE Sebelum Temperature Scaling: {ece_raw:.4f}")
    print(f"ECE Sesudah Temperature Scaling : {ece_scaled:.4f}")

    # Threshold Curve
    conf_scaled = np.max(scaled_probs, axis=1)
    pred_scaled = np.argmax(scaled_probs, axis=1)

    rc_list = []
    for tau in np.linspace(0.25, 0.90, 66):
        mask = (conf_scaled >= tau)
        cov = np.mean(mask)
        prec = np.mean(pred_scaled[mask] == y_true_idx[mask]) if np.sum(mask) > 0 else 1.0
        rc_list.append((tau, cov, prec))

    rc_df = pd.DataFrame(rc_list, columns=['tau', 'coverage', 'precision'])
    tau_safe = rc_df[rc_df['precision'] >= 0.90]['tau'].min()
    if pd.isna(tau_safe):
        tau_safe = 0.60
    cov_at_safe = rc_df[rc_df['tau'] == tau_safe]['coverage'].values[0] * 100
    prec_at_safe = rc_df[rc_df['tau'] == tau_safe]['precision'].values[0] * 100

    print(f"\nSafe Threshold Terpilih: tau* = {tau_safe:.2f}")
    print(f"  - Precision pada Jawaban: {prec_at_safe:.1f}%")
    print(f"  - Coverage (% Koin yang Dijawab): {cov_at_safe:.1f}%")

    # -------------------------------------------------------------------------
    # 5. VISUALISASI DIAGNOSTIK
    # -------------------------------------------------------------------------
    print("\n=== 5. MEMBUAT VISUALISASI DIAGNOSTIK ===", flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    subsets = ['Sisi Angka\n(N=55)', 'Sisi Garuda\n(Label Noise N=34)', 'Total Data\n(N=89)']
    accs = [acc_angka * 100, acc_garuda * 100, acc_total * 100]
    colors = ['#2ca02c', '#d62728', '#1f77b4']
    bars = axes[0].bar(subsets, accs, color=colors, width=0.55)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel('Akurasi Top-1 (%)', fontsize=11)
    axes[0].set_title('Akurasi Berdasarkan Sisi Koin\n(Membuktikan Pengaruh Label Noise)', fontsize=12, fontweight='bold')
    axes[0].axhline(25, color='gray', linestyle='--', label='Peluang Tebakan Acak (25%)')
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)

    axes[1].plot(rc_df['coverage']*100, rc_df['precision']*100, color='darkgreen', lw=2.5, label='Presisi pada Prediksi Dijawab')
    axes[1].axvline(cov_at_safe, color='red', linestyle='--', label=f'Safe Threshold {tau_safe:.2f} (Cov {cov_at_safe:.0f}%)')
    axes[1].set_xlabel('Coverage (% Koin yang Terjawab)', fontsize=11)
    axes[1].set_ylabel('Precision (%)', fontsize=11)
    axes[1].set_title(f'Risk-Coverage Curve (Sisi Angka)\nPresisi = {prec_at_safe:.1f}% pada Coverage {cov_at_safe:.0f}%', fontsize=12, fontweight='bold')
    axes[1].set_ylim(40, 105)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='lower left')

    bin_edges = np.linspace(0, 1, 11)
    confs, accs_bin = [], []
    for b in bin_edges[:-1]:
        mask = (conf_scaled >= b) & (conf_scaled < b + 0.1)
        if np.sum(mask) > 0:
            confs.append(np.mean(conf_scaled[mask]))
            accs_bin.append(np.mean(pred_scaled[mask] == y_true_idx[mask]))

    axes[2].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Kalibrasi Sempurna')
    axes[2].plot(confs, accs_bin, 'o-', color='purple', lw=2, label=f'Setelah T-Scaling (T={best_T:.2f}, ECE={ece_scaled:.3f})')
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel('Keyakinan Rata-rata (Confidence)', fontsize=11)
    axes[2].set_ylabel('Akurasi Sebenarnya', fontsize=11)
    axes[2].set_title('Reliability Diagram [0.0 - 1.0]\n(Setelah Temperature Scaling)', fontsize=12, fontweight='bold')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].legend(loc='upper left')

    plt.tight_layout()
    out_plot = os.path.join('results', 'clean_analysis_plots.png')
    plt.savefig(out_plot, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Visualisasi disimpan ke {out_plot}", flush=True)

    summary_data = [
        {'Parameter': 'Total Citra Unik', 'Nilai': len(df_unique)},
        {'Parameter': 'Citra Sisi Angka Murni', 'Nilai': len(df_angka)},
        {'Parameter': 'Citra Sisi Garuda (Label Noise)', 'Nilai': len(df_garuda)},
        {'Parameter': 'Akurasi Sisi Angka Murni', 'Nilai': f'{acc_angka*100:.2f}%'},
        {'Parameter': 'Akurasi Sisi Garuda (Noise)', 'Nilai': f'{acc_garuda*100:.2f}%'},
        {'Parameter': 'Akurasi Keseluruhan (N=89)', 'Nilai': f'{acc_total*100:.2f}%'},
        {'Parameter': 'Optimal Temperature (T*)', 'Nilai': f'{best_T:.2f}'},
        {'Parameter': 'ECE Sebelum T-Scaling', 'Nilai': f'{ece_raw:.4f}'},
        {'Parameter': 'ECE Sesudah T-Scaling', 'Nilai': f'{ece_scaled:.4f}'},
        {'Parameter': 'Safe Threshold (tau*)', 'Nilai': f'{tau_safe:.2f}'},
        {'Parameter': 'Precision pada tau*', 'Nilai': f'{prec_at_safe:.1f}%'},
        {'Parameter': 'Coverage pada tau*', 'Nilai': f'{cov_at_safe:.1f}%'},
    ]
    pd.DataFrame(summary_data).to_csv(os.path.join('results', 'clean_audit_metrics.csv'), index=False)
    print("Metrik audit tersimpan ke results/clean_audit_metrics.csv", flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nCRASH OCCURRED: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
