import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from infer import FrameQualityGate, CoinClassifierTTA
import augment_coin

manifest_path = BASE_DIR / 'splits' / 'clean_89_manifest.csv'
labels_path = BASE_DIR / 'models' / 'labels_5class.json'
tflite_path = BASE_DIR / 'models' / 'best_5class.tflite'

df = pd.read_csv(manifest_path, encoding='utf-8')
with open(labels_path, 'r', encoding='utf-8') as f:
    labels = json.load(f)
label_to_idx = {name: i for i, name in enumerate(labels)}

gate = FrameQualityGate()
clf = CoinClassifierTTA(model_path=str(tflite_path), labels_path=str(labels_path), temperature=0.663)

y_true = np.array([label_to_idx[lbl] for lbl in df['label_5class']])
preds_no_tta = []
preds_tta = []

print(f"Menguji {len(df)} citra: evaluasi komparatif TTA vs Non-TTA...", flush=True)

for idx, (_, row) in enumerate(df.iterrows()):
    bgr = augment_coin.cv2_imread_unicode(row['path'])
    ok, reason, crop = gate.check(bgr)
    if crop is None:
        crop = cv2.resize(bgr, (224, 224))
    
    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    
    # 1. Non-TTA (0 derajat saja)
    p0 = clf._predict_single_rot(pil_crop)
    preds_no_tta.append(np.argmax(p0))
    
    # 2. TTA (4 sudut ortogonal: 0, 90, 180, 270)
    ptta = clf.predict_with_tta(pil_crop)
    preds_tta.append(np.argmax(ptta))

preds_no_tta = np.array(preds_no_tta)
preds_tta = np.array(preds_tta)

acc_no = np.mean(preds_no_tta == y_true)
acc_tta = np.mean(preds_tta == y_true)

print("\n" + "="*60)
print(f"HASIL EVALUASI EMPIRIS DAMPAK TTA (N = {len(y_true)} citra):")
print("="*60)
print(f"Akurasi TANPA TTA (0° tunggal)     : {acc_no*100:.2f}% ({np.sum(preds_no_tta == y_true)}/{len(y_true)})")
print(f"Akurasi DENGAN TTA (4 rotasi ort.) : {acc_tta*100:.2f}% ({np.sum(preds_tta == y_true)}/{len(y_true)})")
delta = (acc_tta - acc_no) * 100
print(f"Perubahan Akurasi Netto (Delta)    : {delta:+.2f}%\n")

print("Breakdown Akurasi per Kelas:")
for i, name in enumerate(labels):
    mask_c = (y_true == i)
    n_c = np.sum(mask_c)
    acc_c_no = np.sum(preds_no_tta[mask_c] == i) / n_c
    acc_c_tta = np.sum(preds_tta[mask_c] == i) / n_c
    print(f"  - {name:12s} (N={n_c:2d}): Tanpa TTA = {acc_c_no*100:.1f}%, Dengan TTA = {acc_c_tta*100:.1f}% (Delta: {(acc_c_tta - acc_c_no)*100:+.1f}%)")
