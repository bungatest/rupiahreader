import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
results_dir = BASE_DIR / 'results'

oof_probs = np.load(results_dir / 'oof_5class_probs.npy')
y_true = np.load(results_dir / 'oof_5class_targets.npy')
df_manifest = pd.read_csv(BASE_DIR / 'splits' / 'clean_89_manifest.csv', encoding='utf-8')

class_names = ['100', '1000', '200', '500', 'sisi_garuda']

def apply_temp_scale(probs, T=0.663):
    eps = 1e-12
    clipped = np.clip(probs, eps, 1.0 - eps)
    scaled = np.log(clipped) / T
    exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)

oof_T = apply_temp_scale(oof_probs, T=0.663)

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

print("="*75)
print("SIMULASI MULTI-FRAME & CLUSTER BOOTSTRAP RESAMPLING (PER GAMBAR SUMBER)")
print("="*75)

# Simulate 1,000 sessions: pick src_id in 0..88
rng = np.random.default_rng(42)
N_SESSIONS = 1000

rows = []
for s_idx in range(N_SESSIONS):
    src_id = int(rng.integers(0, len(oof_T)))
    base_p = oof_T[src_id]
    y_real = y_true[src_id]
    real_str = class_names[y_real]
    
    n_frames = rng.integers(6, 13)
    session_frames = []
    for _ in range(n_frames):
        alpha_vec = np.clip(base_p * 20.0, 0.1, 100.0)
        session_frames.append(rng.dirichlet(alpha_vec))
        
    dec, reason = decide_strict(session_frames)
    
    # 4 mutually exclusive outcomes:
    # 1. 'benar': nominal dijawab dan benar
    # 2. 'salah': nominal dijawab dan salah (termasuk Garuda dijawab nominal)
    # 3. 'tidak_yakin': tidak menjawab, termasuk balik koin pada sisi angka
    # 4. 'balik_benar': balik koin pada sisi Garuda sebenarnya
    if isinstance(dec, int):
        if dec == y_real:
            outcome = "benar"
        else:
            outcome = "salah"
    elif dec == "balik_koin":
        if y_real == 4:
            outcome = "balik_benar"
        else:
            outcome = "tidak_yakin"
    else:
        outcome = "tidak_yakin"
        
    rows.append({
        "session_id": s_idx,
        "src_id": src_id,
        "src_file": df_manifest.loc[src_id, "filename"],
        "nominal": real_str,
        "outcome": outcome,
        "decision": class_names[dec] if isinstance(dec, int) else dec
    })

df_sim = pd.DataFrame(rows)
df_sim.to_csv(results_dir / 'sim_sessions_cluster.csv', index=False)
print(f"Simulasi {N_SESSIONS} sesi disimpan ke results/sim_sessions_cluster.csv")

OUT = ["benar", "salah", "tidak_yakin", "balik_benar"]

def _metrics(M):
    benar, salah, ty, bb = M.sum(axis=0)
    total, ans = benar + salah + ty + bb, benar + salah
    return {
        "coverage (dijawab nominal / semua sesi)": ans / total,
        "benar per sesi":        benar / total,
        "salah per sesi (FP rate)": salah / total,
        "presisi saat menjawab": benar / ans if ans else np.nan,
        "salah per jawaban (1 - precision)": salah / ans if ans else np.nan,
    }

def cluster_bootstrap(df, cluster_col="src_id", outcome_col="outcome",
                      strata_col=None, n_boot=5000, seed=0):
    counts = (df.groupby(cluster_col)[outcome_col].value_counts()
                .unstack(fill_value=0).reindex(columns=OUT, fill_value=0))
    M = counts.to_numpy(float)
    if strata_col:
        s = (df.groupby(cluster_col)[strata_col].first()
               .reindex(counts.index).to_numpy())
        groups = [np.where(s == v)[0] for v in np.unique(s)]
    else:
        groups = [np.arange(len(M))]

    rng = np.random.default_rng(seed)
    point = _metrics(M)
    draws = {k: np.empty(n_boot) for k in point}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        for k, v in _metrics(M[idx]).items():
            draws[k][b] = v

    rows_res = []
    for k, v in point.items():
        lo, hi = np.nanpercentile(draws[k], [2.5, 97.5])
        rows_res.append((k, v, lo, hi))
    res = pd.DataFrame(rows_res, columns=["metrik", "estimasi", "CI95_bawah", "CI95_atas"])
    print(f"n_cluster = {len(M)}, n_sesi = {int(M.sum())}")
    return res.round(3)

print("\n--- HASIL CLUSTER BOOTSTRAP (B=5.000, Cluster: src_id, Stratified: nominal) ---")
res_boot = cluster_bootstrap(df_sim, cluster_col="src_id", outcome_col="outcome", strata_col="nominal", n_boot=5000, seed=42)
print(res_boot.to_string(index=False))

