import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "/tmp"), "mpl_cache")

print("1. Loading TensorFlow first...", flush=True)
import tensorflow as tf
print("TensorFlow loaded cleanly!", flush=True)

import time, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import norm

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
import augment_coin

def wilson_ci(k, n, confidence=0.95):
    if n == 0: return 0.0, 0.0
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return round(float(max(0.0, center - spread)), 4), round(float(min(1.0, center + spread)), 4)

# 2. Muat manifest bersih 89 citra unik
clean_manifest_path = BASE_DIR / 'splits' / 'clean_89_manifest.csv'
df_clean = pd.read_csv(clean_manifest_path, encoding='utf-8')
N_TOTAL = len(df_clean)

# Label mapping 5 kelas
class_names = ['100', '1000', '200', '500', 'sisi_garuda']
label_to_idx = {name: i for i, name in enumerate(class_names)}
idx_to_label = {i: name for i, name in enumerate(class_names)}

with open(BASE_DIR / 'models' / 'labels_5class.json', 'w', encoding='utf-8') as f:
    json.dump(class_names, f, indent=2)

print(f"\nDataset 5-Kelas: {N_TOTAL} citra unik. Distribusi:")
print(df_clean['label_5class'].value_counts())

# Pre-extract coin crops for speed
print("\n2. Ekstraksi koin & segmentasi awal...", flush=True)
cached_items = {}
for _, row in df_clean.iterrows():
    bgr = augment_coin.cv2_imread_unicode(row['path'])
    mask, bbox, _ = augment_coin.extract_coin(bgr)
    cached_items[row['path']] = (bgr, mask, str(row['label_5class']))

print(f"Ekstraksi selesai untuk {len(cached_items)} citra.", flush=True)

INPUT_SHAPE = (224, 224, 3)

def build_model(num_classes=5):
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
    base = tf.keras.applications.EfficientNetV2B0(
        input_tensor=inputs,
        include_top=False,
        weights='imagenet',
        include_preprocessing=True
    )
    base.trainable = False
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs), base

def prepare_data(df_subset, copies=7, seed=42):
    rng = np.random.default_rng(seed)
    images, labels = [], []
    tw, th = 224, 224
    
    for _, row in df_subset.iterrows():
        bgr, mask, lbl = cached_items[row['path']]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            pad = int(max(w, h) * 0.08)
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(bgr.shape[1], x + w + pad); y2 = min(bgr.shape[0], y + h + pad)
            clean = cv2.resize(bgr[y1:y2, x1:x2], (tw, th))
        else:
            clean = cv2.resize(bgr, (tw, th))
            
        images.append(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
        labels.append(label_to_idx[lbl])
        
        # Augmentasi sintetis
        for _ in range(copies):
            aug = augment_coin.augment_coin_synthetic(
                bgr, mask=mask, target_size=(tw, th), intensity='medium', rng=rng, mix_raw_crop_prob=0.25
            )
            images.append(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB))
            labels.append(label_to_idx[lbl])
            
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=int)

def prepare_val_clean(df_subset):
    images, labels = [], []
    tw, th = 224, 224
    for _, row in df_subset.iterrows():
        bgr, mask, lbl = cached_items[row['path']]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            pad = int(max(w, h) * 0.08)
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(bgr.shape[1], x + w + pad); y2 = min(bgr.shape[0], y + h + pad)
            clean = cv2.resize(bgr[y1:y2, x1:x2], (tw, th))
        else:
            clean = cv2.resize(bgr, (tw, th))
        images.append(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
        labels.append(label_to_idx[lbl])
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=int)

# 3. Stratified 5-Fold Cross-Validation
oof_probs = np.zeros((N_TOTAL, 5), dtype=np.float32)
y_all_true = np.array([label_to_idx[lbl] for lbl in df_clean['label_5class']])

print("\n" + "="*60)
print("3. MEMULAI TRAINING 5-CLASS DENGAN 5-FOLD STRATIFIED GROUP CV")
print("="*60, flush=True)

for fold in range(5):
    print(f"\n--- [FOLD {fold}/4] ---", flush=True)
    df_train = df_clean[df_clean['fold'] != fold]
    df_val = df_clean[df_clean['fold'] == fold]
    val_indices = df_val.index.values
    
    # Hitung class weights seimbang
    y_tr_raw = [label_to_idx[lbl] for lbl in df_train['label_5class']]
    weights = compute_class_weight('balanced', classes=np.arange(5), y=y_tr_raw)
    class_weights_dict = {i: float(w) for i, w in enumerate(weights)}
    
    X_tr, y_tr = prepare_data(df_train, copies=7, seed=200 + fold)
    X_val, y_val = prepare_val_clean(df_val)
    
    Y_tr = tf.keras.utils.to_categorical(y_tr, num_classes=5)
    Y_val = tf.keras.utils.to_categorical(y_val, num_classes=5)
    
    model, base = build_model(num_classes=5)
    
    # Tahap 1: Head warmup (4 epochs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['categorical_accuracy']
    )
    model.fit(
        X_tr, Y_tr,
        validation_data=(X_val, Y_val),
        epochs=4,
        batch_size=32,
        class_weight=class_weights_dict,
        verbose=1,
        shuffle=True
    )
    
    # Tahap 2: Fine-tuning lapisan atas (5 epochs)
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(6e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['categorical_accuracy']
    )
    model.fit(
        X_tr, Y_tr,
        validation_data=(X_val, Y_val),
        epochs=5,
        batch_size=32,
        class_weight=class_weights_dict,
        verbose=1,
        shuffle=True
    )
    
    # Simpan prediksi OOF murni dari model fold
    val_preds = model.predict(X_val, verbose=0)
    for i_local, idx_global in enumerate(val_indices):
        oof_probs[idx_global] = val_preds[i_local]
        
    fold_preds_cls = np.argmax(val_preds, axis=1)
    fold_acc = accuracy_score(y_val, fold_preds_cls)
    print(f"Fold {fold} OOF Accuracy: {fold_acc*100:.2f}% ({np.sum(fold_preds_cls == y_val)}/{len(y_val)})", flush=True)
    
    del model, base, X_tr, Y_tr, X_val, Y_val
    tf.keras.backend.clear_session()
    gc.collect()

# 4. Simpan Artefak OOF Resmi ke Disk
results_dir = BASE_DIR / 'results'
results_dir.mkdir(parents=True, exist_ok=True)

np.save(results_dir / 'oof_5class_probs.npy', oof_probs)
np.save(results_dir / 'oof_5class_targets.npy', y_all_true)

df_clean['pred_5class_idx'] = np.argmax(oof_probs, axis=1)
df_clean['pred_5class_label'] = [idx_to_label[i] for i in df_clean['pred_5class_idx']]
df_clean['confidence'] = np.max(oof_probs, axis=1)
df_clean['is_correct'] = (df_clean['pred_5class_idx'] == y_all_true)

for c_i, c_name in enumerate(class_names):
    df_clean[f'prob_{c_name}'] = oof_probs[:, c_i]

df_clean.to_csv(results_dir / 'oof_5class_manifest.csv', index=False, encoding='utf-8')
print(f"\nBerhasil menyimpan oof_5class_probs.npy ({oof_probs.shape}) dan oof_5class_manifest.csv!")

# 5. Evaluasi Kuantitatif Terpisah per Sisi Koin
is_angka = (df_clean['coin_side'] == 'angka').values
is_garuda = (df_clean['coin_side'] == 'garuda_pahlawan').values

n_angka = int(np.sum(is_angka))
n_garuda = int(np.sum(is_garuda))
corr_angka = int(np.sum(df_clean['is_correct'][is_angka]))
corr_garuda = int(np.sum(df_clean['is_correct'][is_garuda]))
corr_total = int(np.sum(df_clean['is_correct']))

acc_angka = corr_angka / n_angka
acc_garuda = corr_garuda / n_garuda
acc_total = corr_total / N_TOTAL

ci_angka = wilson_ci(corr_angka, n_angka)
ci_garuda = wilson_ci(corr_garuda, n_garuda)
ci_total = wilson_ci(corr_total, N_TOTAL)

macro_f1 = f1_score(y_all_true, df_clean['pred_5class_idx'], average='macro')
macro_prec = precision_score(y_all_true, df_clean['pred_5class_idx'], average='macro', zero_division=0)
macro_rec = recall_score(y_all_true, df_clean['pred_5class_idx'], average='macro', zero_division=0)

print("\n" + "="*70)
print("HASIL EVALUASI OUT-OF-FOLD (OOF) 5-KELAS RESMI (N=89 CITRA UNIK)")
print("="*70)
print(f"Total Sampel Unik:            {N_TOTAL}")
print(f"Akurasi Total 5-Kelas:        {acc_total*100:.2f}% ({corr_total}/{N_TOTAL}) | Wilson 95% CI: [{ci_total[0]*100:.1f}%, {ci_total[1]*100:.1f}%]")
print(f"Akurasi Sisi Angka Murni:     {acc_angka*100:.2f}% ({corr_angka}/{n_angka}) | Wilson 95% CI: [{ci_angka[0]*100:.1f}%, {ci_angka[1]*100:.1f}%]")
print(f"Akurasi Sisi Garuda:          {acc_garuda*100:.2f}% ({corr_garuda}/{n_garuda}) | Wilson 95% CI: [{ci_garuda[0]*100:.1f}%, {ci_garuda[1]*100:.1f}%]")
print(f"Macro F1-Score:               {macro_f1:.4f}")
print(f"Macro Precision:              {macro_prec:.4f}")
print(f"Macro Recall:                 {macro_rec:.4f}")
print("="*70)

# 6. Confusion Matrix 5x5
cm = confusion_matrix(y_all_true, df_clean['pred_5class_idx'], labels=np.arange(5))
cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
cm_df.to_csv(results_dir / 'oof_5class_confusion_matrix.csv')
print("\nConfusion Matrix 5x5:")
print(cm_df)

# 7. Latih Model Final pada Seluruh 89 Citra untuk Ekspor TFLite
print("\n4. Melatih Model Final pada Seluruh 89 Citra & Ekspor TFLite...", flush=True)
X_all, y_all = prepare_data(df_clean, copies=8, seed=999)
Y_all = tf.keras.utils.to_categorical(y_all, num_classes=5)
weights_all = compute_class_weight('balanced', classes=np.arange(5), y=y_all_true)
class_weights_all = {i: float(w) for i, w in enumerate(weights_all)}

final_model, final_base = build_model(num_classes=5)
final_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05))
final_model.fit(X_all, Y_all, epochs=5, batch_size=32, class_weight=class_weights_all, verbose=1, shuffle=True)

final_base.trainable = True
for layer in final_base.layers[:-20]:
    layer.trainable = False
final_model.compile(optimizer=tf.keras.optimizers.Adam(6e-5), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05))
final_model.fit(X_all, Y_all, epochs=5, batch_size=32, class_weight=class_weights_all, verbose=1, shuffle=True)

final_keras_path = BASE_DIR / 'models' / 'best_5class.keras'
final_model.save(final_keras_path)
print(f"Model Keras tersimpan: {final_keras_path}")

# Konversi ke TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = BASE_DIR / 'models' / 'best_5class.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"Model TFLite tersimpan: {tflite_path} ({len(tflite_model)/1024/1024:.2f} MB)")

# Simpan ringkasan evaluasi resmi
eval_summary = {
    "n_unique_total": N_TOTAL,
    "n_sisi_angka": n_angka,
    "n_sisi_garuda": n_garuda,
    "accuracy_5class_total": float(acc_total),
    "ci_5class_total": list(ci_total),
    "accuracy_sisi_angka": float(acc_angka),
    "ci_sisi_angka": list(ci_angka),
    "accuracy_sisi_garuda": float(acc_garuda),
    "ci_sisi_garuda": list(ci_garuda),
    "macro_f1": float(macro_f1),
    "macro_precision": float(macro_prec),
    "macro_recall": float(macro_rec),
    "class_names": class_names
}

with open(results_dir / 'oof_5class_evaluation_summary.json', 'w', encoding='utf-8') as f:
    json.dump(eval_summary, f, indent=2)

print("\nSELESAI DENGAN SUKSES! Seluruh artefak tersimpan di results/ dan models/.", flush=True)
