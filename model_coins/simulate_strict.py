import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.stats import norm

oof_probs = np.load('model_coins/results/oof_5class_probs.npy')
y_true = np.load('model_coins/results/oof_5class_targets.npy')
df_manifest = pd.read_csv('model_coins/splits/clean_89_manifest.csv', encoding='utf-8')

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
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return round(float(max(0.0, center - spread)), 4), round(float(min(1.0, center + spread)), 4)

# User's strict decision rule
def decide_strict(frames, tau_frame=0.70, min_agree=0.80, min_frames=6, min_strong=0.50, min_mean_p=0.65, max_garuda=0.35):
    # frames: list of 5-class calibrated probabilities from diverse accepted frames
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

print("="*75)
print("UJI ATURAN KEPUTUSAN KETAT (PER-FRAME EVIDENCE & CONSENSUS) PADA SESI OOF")
print("="*75)

# Simulate 1,000 sessions with genuine OOF probabilities
# For each session: pick a coin from df_manifest (using its strictly OOF probability vector)
# Generate 10-15 frames with physical variation around its OOF vector
rng = np.random.default_rng(42)
N_SESSIONS = 1000

configs = [
    {"name": "Konfigurasi Lama (Posterior Tau=0.85, Bayes)", "type": "legacy"},
    {"name": "Aturan Ketat (tau_f=0.70, agree=0.80, strong=0.50, mean_p=0.65)", "tau_f": 0.70, "agree": 0.80, "strong": 0.50, "mean_p": 0.65},
    {"name": "Aturan Sangat Ketat (tau_f=0.75, agree=0.85, strong=0.60, mean_p=0.70)", "tau_f": 0.75, "agree": 0.85, "strong": 0.60, "mean_p": 0.70},
    {"name": "Aturan Ultra Aman (tau_f=0.80, agree=0.90, strong=0.60, mean_p=0.75)", "tau_f": 0.80, "agree": 0.90, "strong": 0.60, "mean_p": 0.75},
]

for cfg in configs:
    n_answered = 0
    n_correct = 0
    n_wrong = 0
    n_flip = 0
    n_uncertain = 0
    
    for s_idx in range(N_SESSIONS):
        coin_idx = rng.integers(0, len(oof_T))
        base_p = oof_T[coin_idx]
        true_label = y_true[coin_idx] # 0..3 for nominal, 4 for garuda
        
        # Sesi menghasilkan 6 hingga 12 frame dengan variasi rotasi/cahaya
        n_frames = rng.integers(6, 13)
        session_frames = []
        for _ in range(n_frames):
            # Simulasi variasi Dirichlet di sekitar distribusi OOF koin ini
            alpha_vec = np.clip(base_p * 20.0, 0.1, 100.0)
            sample_p = rng.dirichlet(alpha_vec)
            session_frames.append(sample_p)
            
        if cfg.get("type") == "legacy":
            # Log-posterior akumulasi lama
            logp4 = np.zeros(4)
            for f in session_frames:
                p4 = f[:4] / f[:4].sum()
                logp4 += 0.5 * np.log(p4 + 1e-6)
            post = np.exp(logp4 - np.max(logp4))
            post /= post.sum()
            mean_garuda = np.mean([f[4] for f in session_frames])
            if mean_garuda >= 0.35:
                n_flip += 1
            elif np.max(post) >= 0.85:
                pred_c = int(np.argmax(post))
                n_answered += 1
                if pred_c == true_label:
                    n_correct += 1
                else:
                    n_wrong += 1
            else:
                n_uncertain += 1
        else:
            dec, reason = decide_strict(
                session_frames,
                tau_frame=cfg["tau_f"],
                min_agree=cfg["agree"],
                min_strong=cfg["strong"],
                min_mean_p=cfg["mean_p"],
                max_garuda=0.35
            )
            if dec == "balik_koin":
                n_flip += 1
            elif isinstance(dec, int):
                n_answered += 1
                if dec == true_label:
                    n_correct += 1
                else:
                    n_wrong += 1
            else:
                n_uncertain += 1
                
    cov = n_answered / N_SESSIONS * 100
    prec = (n_correct / n_answered * 100) if n_answered > 0 else 0.0
    fp_rate = (n_wrong / N_SESSIONS * 100)
    ci = wilson_ci(n_correct, n_answered) if n_answered > 0 else (0.0, 0.0)
    
    print(f"\n[{cfg['name']}]")
    print(f"  Coverage Dijawab : {n_answered}/{N_SESSIONS} ({cov:.1f}%)")
    print(f"  Sesi Benar       : {n_correct} ({n_correct/N_SESSIONS*100:.1f}%)")
    print(f"  Sesi Salah (FP)  : {n_wrong} ({fp_rate:.1f}%) -> Target: <= 2-3%")
    print(f"  Presisi Sesi     : {prec:.1f}% | Wilson 95% CI: [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
    print(f"  Balik Koin       : {n_flip} ({n_flip/N_SESSIONS*100:.1f}%)")
    print(f"  Menyerah/Uncertain: {n_uncertain} ({n_uncertain/N_SESSIONS*100:.1f}%)")

