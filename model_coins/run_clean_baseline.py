import os, sys
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

sys.stdout.reconfigure(encoding='utf-8')

# Load inventory
df = pd.read_csv(r"model_coins\splits\dataset_detailed_inventory.csv")
# Remove exact duplicates
df_unique = df[~df['is_duplicate']].reset_index(drop=True)

# EfficientNetV2B0 feature extractor
base_model = tf.keras.applications.EfficientNetV2B0(
    include_top=False,
    weights='imagenet',
    pooling='avg',
    input_shape=(224, 224, 3)
)

def extract_features(paths):
    feats = []
    for p in paths:
        img = Image.open(p).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        # EfficientNetV2 built-in preprocessing expects [0..255]
        feat = base_model(arr, training=False).numpy()[0]
        feats.append(feat)
    return np.array(feats)

print("Extracting ImageNet features for all unique images...")
features = extract_features(df_unique['path'].tolist())
print(f"Features extracted: shape {features.shape}")

# Skenario A: 4 Kelas Angka Bersih (Hanya sisi angka, label noise Garuda dibuang)
df_angka = df_unique[df_unique['coin_side'] == 'angka'].reset_index(drop=True)
feats_angka = features[df_unique['coin_side'] == 'angka']
y_angka = df_angka['original_class'].astype(str).values

print(f"\n=======================================================")
print(f"SKENARIO A: 4 KELAS ANGKA BERSIH (N={len(df_angka)} citra)")
print(f"Distribusi: {pd.Series(y_angka).value_counts().to_dict()}")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_a = np.empty(len(df_angka), dtype=object)

for train_idx, val_idx in skf.split(feats_angka, y_angka):
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(feats_angka[train_idx], y_angka[train_idx])
    oof_preds_a[val_idx] = clf.predict(feats_angka[val_idx])

acc_a = accuracy_score(y_angka, oof_preds_a)
f1_a = f1_score(y_angka, oof_preds_a, average='macro')
print(f"OOF Accuracy Skenario A: {acc_a * 100:.2f}% (Peluang acak: 25.0%)")
print(f"OOF Macro F1 Skenario A: {f1_a:.4f}")
print("Classification Report:")
print(classification_report(y_angka, oof_preds_a))


# Skenario B: 5 Kelas Lengkap (100, 200, 500, 1000, sisi_garuda) (N=89 citra)
y_5way = df_unique['target_class_5way'].values
print(f"\n=======================================================")
print(f"SKENARIO B: 5 KELAS (+ SISI_GARUDA) (N={len(df_unique)} citra)")
print(f"Distribusi: {pd.Series(y_5way).value_counts().to_dict()}")

oof_preds_b = np.empty(len(df_unique), dtype=object)
oof_probs_b = np.empty((len(df_unique), len(np.unique(y_5way))), dtype=np.float32)

classes_5way = np.unique(y_5way)
class_to_idx = {c: i for i, c in enumerate(classes_5way)}

for train_idx, val_idx in skf.split(features, y_5way):
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(features[train_idx], y_5way[train_idx])
    oof_preds_b[val_idx] = clf.predict(features[val_idx])
    oof_probs_b[val_idx] = clf.predict_proba(features[val_idx])

acc_b = accuracy_score(y_5way, oof_preds_b)
f1_b = f1_score(y_5way, oof_preds_b, average='macro')
print(f"OOF Accuracy Skenario B: {acc_b * 100:.2f}% (Peluang acak: 20.0%)")
print(f"OOF Macro F1 Skenario B: {f1_b:.4f}")
print("Classification Report:")
print(classification_report(y_5way, oof_preds_b))

# Simpan hasil untuk analisis kalibrasi & thresholding
np.save(r"model_coins\results\oof_probs_5way.npy", oof_probs_b)
np.save(r"model_coins\results\oof_true_5way.npy", np.array([class_to_idx[y] for y in y_5way]))
np.save(r"model_coins\results\classes_5way.npy", classes_5way)
print("Data OOF berhasil disimpan untuk analisis Temperature Scaling!")

