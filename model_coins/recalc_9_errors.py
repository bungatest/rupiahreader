import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os
import numpy as np
import pandas as pd
from PIL import Image

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

class_names = ['100', '1000', '200', '500', 'sisi_garuda']

df_manifest['pred_raw'] = [class_names[i] for i in oof_probs.argmax(1)]
df_manifest['conf_raw'] = oof_probs.max(1)
df_manifest['pred_T'] = [class_names[i] for i in oof_T.argmax(1)]
df_manifest['conf_T'] = oof_T.max(1)

for idx, c in enumerate(class_names):
    df_manifest['pT_' + c] = oof_T[:, idx]

print("="*75)
print("9 KASUS SALAH 1000 <-> GARUDA DENGAN TEMPERATURE SCALING T=0.663:")
print("="*75)

c1 = df_manifest[(df_manifest['label_5class'] == '1000') & (df_manifest['pred_T'] == 'sisi_garuda')]
print(f"\n1. True 1000 diprediksi sisi_garuda (N={len(c1)}):")
for _, r in c1.iterrows():
    fn = r['filename']
    print(f"  {fn:18s} | Raw: {r['conf_raw']:.3f} -> T-Conf: {r['conf_T']:.3f} | pT(1000)={r['pT_1000']:.3f} | pT(garuda)={r['pT_sisi_garuda']:.3f}")
    print(f"    Path: {r['path']}")

c2 = df_manifest[(df_manifest['label_5class'] == 'sisi_garuda') & (df_manifest['pred_T'] == '1000')]
print(f"\n2. True sisi_garuda diprediksi 1000 (N={len(c2)}):")
for _, r in c2.iterrows():
    fn = r['filename']
    print(f"  {fn:18s} | Raw: {r['conf_raw']:.3f} -> T-Conf: {r['conf_T']:.3f} | pT(1000)={r['pT_1000']:.3f} | pT(garuda)={r['pT_sisi_garuda']:.3f}")
    print(f"    Path: {r['path']}")

