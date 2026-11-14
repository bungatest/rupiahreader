import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

BASE_DIR = Path(__file__).parent
results_dir = BASE_DIR / 'results'

oof_probs = np.load(results_dir / 'oof_5class_probs.npy')
y_true = np.load(results_dir / 'oof_5class_targets.npy')
df_manifest = pd.read_csv(BASE_DIR / 'splits' / 'clean_89_manifest.csv', encoding='utf-8')

class_names = ['100', '1000', '200', '500', 'sisi_garuda']
N_TOTAL = len(y_true)

def apply_temp_scale(probs, T=0.663):
    eps = 1e-12
    clipped = np.clip(probs, eps, 1.0 - eps)
    scaled = np.log(clipped) / T
    exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)

oof_T = apply_temp_scale(oof_probs, T=0.663)

def wilson_ci(k, n, confidence=0.95):
    if n == 0: return 0.0, 0.0
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    variance = max(0.0, p * (1 - p) + z**2 / (4 * n))
    spread = z * np.sqrt(variance / n) / denom
    return round(float(max(0.0, center - spread)), 4), round(float(min(1.0, center + spread)), 4)

def decide_strict(frames, tau_frame=0.75, min_agree=0.85, min_frames=6, min_strong=0.60, min_mean_p=0.70, max_garuda=0.35):
    if len(frames) < min_frames:
        return None, "insufficient_frames"
    
    garuda_mean = float(np.mean([f[4] for f in frames]))
    if garuda_mean >= max_garuda:
        return "balik_koin", "garuda_detected"
        
    P = np.array([f[:4] / (f[:4].sum() + 1e-12) for f in frames])
    top = P.argmax(axis=1)
    cls = int(np.bincount(top, minlength=4).argmax())
    
    agree = float((top == cls).mean())
    strong = float((P[:, cls] >= tau_frame).mean())
    mean_p = float(P[:, cls].mean())
    
    if agree >= min_agree and strong >= min_strong and mean_p >= min_mean_p:
        return cls, "answered"
    
    return None, "uncertain"

print("="*70)
print("SIMULASI MULTI-FRAME CONSENSUS DECISION RULE PADA OOF DATA (N=1000 SESI)")
print("="*70)

rng = np.random.default_rng(42)
N_SESSIONS = 1000

records = []
for s_idx in range(N_SESSIONS):
    coin_idx = rng.integers(0, len(oof_T))
    base_p = oof_T[coin_idx]
    true_label = y_true[coin_idx]
    
    n_frames = rng.integers(6, 13)
    session_frames = []
    for _ in range(n_frames):
        alpha_vec = np.clip(base_p * 20.0, 0.1, 100.0)
        sample_p = rng.dirichlet(alpha_vec)
        session_frames.append(sample_p)
        
    dec, reason = decide_strict(session_frames)
    
    is_correct = False
    is_fp = False
    if dec == "balik_koin":
        status = "balik_koin"
        is_correct = (true_label == 4)
    elif isinstance(dec, int):
        status = "jawab"
        if dec == true_label:
            is_correct = True
        else:
            is_fp = True
    else:
        status = "menyerah"
        
    records.append({
        'session_id': s_idx,
        'coin_source_idx': coin_idx,
        'true_label': class_names[true_label],
        'status': status,
        'decision': class_names[dec] if isinstance(dec, int) else dec,
        'is_correct': is_correct,
        'is_false_positive': is_fp
    })

df_sim = pd.DataFrame(records)
df_sim.to_csv(results_dir / 'simulated_session_results.csv', index=False)

n_answered = (df_sim['status'] == 'jawab').sum()
n_correct_nominal = ((df_sim['status'] == 'jawab') & df_sim['is_correct']).sum()
n_fp = df_sim['is_false_positive'].sum()
n_flip = (df_sim['status'] == 'balik_koin').sum()
n_correct_flip = ((df_sim['status'] == 'balik_koin') & df_sim['is_correct']).sum()
n_giveup = (df_sim['status'] == 'menyerah').sum()

prec = (n_correct_nominal / n_answered * 100) if n_answered > 0 else 0.0
ci = wilson_ci(n_correct_nominal, n_answered) if n_answered > 0 else (0.0, 0.0)

print(f"\nHASIL SIMULASI SESI (TOTAL N={N_SESSIONS} SESI UJI):")
print(f"Total Sesi yang Dijawab Nominal:   {n_answered}/{N_SESSIONS} ({n_answered/N_SESSIONS*100:.1f}%)")
print(f"  - Dijawab BENAR (Nominal Tepat): {n_correct_nominal} sesi ({n_correct_nominal/N_SESSIONS*100:.1f}%)")
print(f"  - Dijawab SALAH (False Positive):{n_fp} sesi ({n_fp/N_SESSIONS*100:.1f}%)")
print(f"Sesi Terdeteksi Sisi Garuda (FLIP):{n_flip} sesi ({n_flip/N_SESSIONS*100:.1f}%) [Benar Garuda: {n_correct_flip}]")
print(f"Sesi Menyerah / Tidak Yakin:       {n_giveup} sesi ({n_giveup/N_SESSIONS*100:.1f}%)")

print("\n" + "="*55)
print(f"PRESISI NOMINAL (SAAT APLIKASI MENYEBUT NOMINAL): {prec:.2f}% ({n_correct_nominal}/{n_answered})")
print(f"Wilson 95% Confidence Interval:    [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
print(f"TINGKAT FALSE POSITIVE SALAH SEBUT: {n_fp/N_SESSIONS*100:.1f}% (Target: <= 2-3%)")
print("="*55)
