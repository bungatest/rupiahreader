import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import pandas as pd

p = r'C:\Users\Bunga Alyssa\OneDrive\文档\DS\smt  7\deep l\uas\frontend\model_coins\results\oof_5class_manifest.csv'
df = pd.read_csv(p, encoding='utf-8')

c1 = df[(df['label_5class'] == '1000') & (df['pred_5class_label'] == 'sisi_garuda')]
c2 = df[(df['label_5class'] == 'sisi_garuda') & (df['pred_5class_label'] == '1000')]

print(f"=== 1. True 1000 diprediksi sisi_garuda (N={len(c1)}) ===")
for _, r in c1.iterrows():
    print(f"  File: {r['filename']} | Original Class: {r['original_class']} | Sisi: {r['coin_side']} | Conf: {r['confidence']:.3f} | p(1000)={r['prob_1000']:.3f} | p(garuda)={r['prob_sisi_garuda']:.3f}")

print(f"\n=== 2. True sisi_garuda diprediksi 1000 (N={len(c2)}) ===")
for _, r in c2.iterrows():
    print(f"  File: {r['filename']} | Original Class: {r['original_class']} | Sisi: {r['coin_side']} | Conf: {r['confidence']:.3f} | p(1000)={r['prob_1000']:.3f} | p(garuda)={r['prob_sisi_garuda']:.3f}")

