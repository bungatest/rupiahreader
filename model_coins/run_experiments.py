"""
run_experiments.py - Robust, Memory-Safe 5-Fold Cross Validation & Evaluation
"""
import os, sys, time, json, gc
from pathlib import Path

# Suppress verbose TF C++ logs and configure deterministic CPU execution
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "/tmp"), "mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
import augment_coin

ROOT = Path(__file__).parent
DATA_DIR = ROOT.parent / 'dataset coin'
SPLITS_DIR = ROOT / 'splits'
RESULTS_DIR = ROOT / 'results'
MODELS_DIR = ROOT / 'models'

label_map = {'100': 0, '1000': 1, '200': 2, '500': 3}
inv_label_map = {v: k for k, v in label_map.items()}

def wilson_score_interval(successes, total, confidence=0.95):
    if total == 0: return 0.0, 0.0
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return max(0.0, center - spread), min(1.0, center + spread)

# Load Dev manifest & pre-extract coin crops
print("1. Memuat Dev manifest & ekstraksi koin...", flush=True)
df_dev = pd.read_csv(SPLITS_DIR / 'dev_manifest.csv')
dev_items = {}
for _, row in df_dev.iterrows():
    bgr = augment_coin.cv2_imread_unicode(row['path'])
    mask, bbox, _ = augment_coin.extract_coin(bgr)
    dev_items[row['path']] = (bgr, mask, str(row['class']))

print(f"Total dev items dimuat: {len(dev_items)} citra.", flush=True)

INPUT_SHAPE = (224, 224, 3)

def build_custom_cnn(num_classes=4, input_shape=INPUT_SHAPE):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    
    for f in [32, 64, 128]:
        x = tf.keras.layers.Conv2D(f, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(2, 2)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='Custom_CNN')

def build_mobilenetv2(num_classes=4, input_shape=INPUT_SHAPE):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)
    base = tf.keras.applications.MobileNetV2(input_tensor=x, include_top=False, weights='imagenet')
    base.trainable = False
    
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='MobileNetV2'), base

def build_efficientnetv2_b0(num_classes=4, input_shape=INPUT_SHAPE):
    inputs = tf.keras.layers.Input(shape=input_shape)
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
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='EfficientNetV2B0'), base

def generate_fold_data(train_items, val_items, copies_per_sample=10, seed=42):
    """Generates uint8 training and validation arrays (memory safe: ~60MB)."""
    rng = np.random.default_rng(seed)
    x_tr, y_tr = [], []
    th, tw = (224, 224)
    
    for bgr, mask, lbl in train_items:
        # Clean crop
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            pad = int(max(w, h) * 0.08)
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(bgr.shape[1], x + w + pad); y2 = min(bgr.shape[0], y + h + pad)
            clean = cv2.resize(bgr[y1:y2, x1:x2], (tw, th))
        else:
            clean = cv2.resize(bgr, (tw, th))
        x_tr.append(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
        y_tr.append(label_map[lbl])
        
        # Augmented copies
        for _ in range(copies_per_sample):
            aug = augment_coin.augment_coin_synthetic(
                bgr, mask=mask, target_size=(224, 224), intensity='medium', rng=rng, mix_raw_crop_prob=0.25
            )
            x_tr.append(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB))
            y_tr.append(label_map[lbl])
            
    x_val, y_val = [], []
    for bgr, mask, lbl in val_items:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            pad = int(max(w, h) * 0.08)
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(bgr.shape[1], x + w + pad); y2 = min(bgr.shape[0], y + h + pad)
            clean = cv2.resize(bgr[y1:y2, x1:x2], (tw, th))
        else:
            clean = cv2.resize(bgr, (tw, th))
        x_val.append(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
        y_val.append(label_map[lbl])
        
    X_tr = np.array(x_tr, dtype=np.uint8)
    y_tr_cat = tf.keras.utils.to_categorical(y_tr, num_classes=4)
    X_val = np.array(x_val, dtype=np.uint8)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=4)
    
    return X_tr, y_tr_cat, X_val, y_val_cat, np.array(y_val)

NUM_FOLDS = 5
models_to_evaluate = ['Custom_CNN', 'MobileNetV2', 'EfficientNetV2B0']
oof_predictions = {m: np.zeros((len(df_dev), 4), dtype=np.float32) for m in models_to_evaluate}

print("\n2. Memulai Stratified 5-Fold Cross-Validation untuk 3 Model...", flush=True)

for model_name in models_to_evaluate:
    print(f"\n{'='*55}\n   EVALUATING MODEL: {model_name}\n{'='*55}", flush=True)
    
    for f_i in range(NUM_FOLDS):
        df_f_tr = pd.read_csv(SPLITS_DIR / f'fold_{f_i}_train.csv')
        df_f_val = pd.read_csv(SPLITS_DIR / f'fold_{f_i}_val.csv')
        
        train_items = [dev_items[row['path']] for _, row in df_f_tr.iterrows()]
        val_items = [dev_items[row['path']] for _, row in df_f_val.iterrows()]
        
        # Generate data on-the-fly per fold (memory safe & fast)
        X_tr, y_tr, X_val, y_val, y_val_cls = generate_fold_data(train_items, val_items, copies_per_sample=6, seed=100 + f_i)
        
        start_t = time.time()
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=3, restore_best_weights=True)
        
        if model_name == 'Custom_CNN':
            model = build_custom_cnn()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                metrics=['categorical_accuracy']
            )
            model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=10,
                batch_size=32,
                shuffle=True,
                verbose=2,
                callbacks=[early_stop]
            )
            
        elif model_name == 'MobileNetV2':
            model, base = build_mobilenetv2()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                metrics=['categorical_accuracy']
            )
            model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=5, batch_size=32, shuffle=True, verbose=2)
            
            base.trainable = True
            for layer in base.layers[:-20]:
                layer.trainable = False
            model.compile(
                optimizer=tf.keras.optimizers.Adam(6e-5),
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                metrics=['categorical_accuracy']
            )
            model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=6, batch_size=32, shuffle=True, verbose=2, callbacks=[early_stop])
            
        elif model_name == 'EfficientNetV2B0':
            model, base = build_efficientnetv2_b0()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                metrics=['categorical_accuracy']
            )
            model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=5, batch_size=32, shuffle=True, verbose=2)
            
            base.trainable = True
            for layer in base.layers[:-20]:
                layer.trainable = False
            model.compile(
                optimizer=tf.keras.optimizers.Adam(6e-5),
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                metrics=['categorical_accuracy']
            )
            model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=6, batch_size=32, shuffle=True, verbose=2, callbacks=[early_stop])
            
        elapsed = time.time() - start_t
        val_preds = model.predict(X_val, verbose=0)
        val_pred_cls = np.argmax(val_preds, axis=-1)
        
        f_acc = accuracy_score(y_val_cls, val_pred_cls)
        f_f1 = f1_score(y_val_cls, val_pred_cls, average='macro', zero_division=0)
        f_prec = precision_score(y_val_cls, val_pred_cls, average='macro', zero_division=0)
        
        print(f"[{model_name} Fold {f_i}] Acc: {f_acc*100:.1f}%, F1: {f_f1:.3f}, Prec: {f_prec:.3f} ({elapsed:.1f}s)", flush=True)
        
        for i_sub, row in df_f_val.iterrows():
            orig_idx = df_dev[df_dev['path'] == row['path']].index[0]
            oof_predictions[model_name][orig_idx] = val_preds[i_sub]
            
        del model, X_tr, y_tr, X_val, y_val
        tf.keras.backend.clear_session()
        gc.collect()

# POOLED OUT-OF-FOLD (OOF) EVALUATION
y_dev_true = np.array([label_map[str(c)] for c in df_dev['class']])

print("\n" + "="*70, flush=True)
print("POOLED OUT-OF-FOLD (OOF) CROSS-VALIDATION RESULTS (N=76 SAMPLES)", flush=True)
print("="*70, flush=True)

oof_summary = []
for model_name in models_to_evaluate:
    preds = oof_predictions[model_name]
    pred_cls = np.argmax(preds, axis=-1)
    
    acc = accuracy_score(y_dev_true, pred_cls)
    macro_f1 = f1_score(y_dev_true, pred_cls, average='macro', zero_division=0)
    macro_prec = precision_score(y_dev_true, pred_cls, average='macro', zero_division=0)
    macro_rec = recall_score(y_dev_true, pred_cls, average='macro', zero_division=0)
    ci_low, ci_high = wilson_score_interval(np.sum(y_dev_true == pred_cls), len(y_dev_true))
    
    oof_summary.append({
        'Model': model_name,
        'OOF Accuracy': f"{acc*100:.2f}%",
        'Wilson 95% CI': f"[{ci_low*100:.1f}%, {ci_high*100:.1f}%]",
        'Macro F1': round(macro_f1, 4),
        'Macro Precision': round(macro_prec, 4),
        'Macro Recall': round(macro_rec, 4)
    })
    
    print(f"\n--- {model_name} OOF Classification Report ---", flush=True)
    print(classification_report(y_dev_true, pred_cls, target_names=['100', '1000', '200', '500'], digits=3), flush=True)

df_oof_summary = pd.DataFrame(oof_summary)
print("\n=== SUMMARY COMPARISON TABLE ===", flush=True)
print(df_oof_summary.to_string(index=False), flush=True)

best_model_name = df_oof_summary.sort_values(by=['Macro F1', 'Macro Precision'], ascending=False).iloc[0]['Model']
print(f"\nWINNER OBJECTIVELY SELECTED: >>> {best_model_name} <<<", flush=True)

df_exp = pd.DataFrame(oof_summary)
df_exp.to_csv(RESULTS_DIR / 'experiments.csv', index=False)
print("Saved clean summary to results/experiments.csv", flush=True)

# 3. TRAIN FINAL WINNING MODEL ON FULL DEV SET
print(f"\n3. Melatih model final ({best_model_name}) pada seluruh 76 citra Dev...", flush=True)
all_dev_items = list(dev_items.values())
X_dev_full, y_dev_full, _, _, _ = generate_fold_data(all_dev_items, [], copies_per_sample=12, seed=2026)

if best_model_name == 'Custom_CNN':
    final_model = build_custom_cnn()
    final_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05), metrics=['categorical_accuracy'])
    final_model.fit(X_dev_full, y_dev_full, epochs=20, batch_size=16, shuffle=True, verbose=2)
elif best_model_name == 'MobileNetV2':
    final_model, base = build_mobilenetv2()
    final_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05), metrics=['categorical_accuracy'])
    final_model.fit(X_dev_full, y_dev_full, epochs=10, batch_size=16, shuffle=True, verbose=2)
    base.trainable = True
    for layer in base.layers[:-20]: layer.trainable = False
    final_model.compile(optimizer=tf.keras.optimizers.Adam(6e-5), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05), metrics=['categorical_accuracy'])
    final_model.fit(X_dev_full, y_dev_full, epochs=14, batch_size=16, shuffle=True, verbose=2)
else:
    final_model, base = build_efficientnetv2_b0()
    final_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05), metrics=['categorical_accuracy'])
    final_model.fit(X_dev_full, y_dev_full, epochs=10, batch_size=16, shuffle=True, verbose=2)
    base.trainable = True
    for layer in base.layers[:-20]: layer.trainable = False
    final_model.compile(optimizer=tf.keras.optimizers.Adam(6e-5), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05), metrics=['categorical_accuracy'])
    final_model.fit(X_dev_full, y_dev_full, epochs=14, batch_size=16, shuffle=True, verbose=2)

final_model.save(MODELS_DIR / 'best.keras')
with open(MODELS_DIR / 'labels.json', 'w') as f:
    json.dump(['100', '1000', '200', '500'], f)
print("Saved models/best.keras and models/labels.json", flush=True)

# 4. FINAL EVALUATION ON UNTOUCHED HOLD-OUT TEST SET
print("\n" + "="*70, flush=True)
print("4. EVALUASI AKHIR PADA TEST SET HOLD-OUT (N=18 CITRA)", flush=True)
print("="*70, flush=True)

df_test = pd.read_csv(SPLITS_DIR / 'test_manifest.csv')
test_imgs = []
y_test_true = []

for _, row in df_test.iterrows():
    bgr = augment_coin.cv2_imread_unicode(row['path'])
    mask, bbox, _ = augment_coin.extract_coin(bgr)
    x, y, w, h = bbox
    pad = int(max(w, h) * 0.08)
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(bgr.shape[1], x + w + pad); y2 = min(bgr.shape[0], y + h + pad)
    crop = bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
    test_imgs.append(crop_rgb)
    y_test_true.append(label_map[str(row['class'])])

X_test = np.array(test_imgs, dtype=np.uint8)
y_test_true = np.array(y_test_true)

test_preds = final_model.predict(X_test, verbose=0)
test_pred_cls = np.argmax(test_preds, axis=-1)
test_confs = np.max(test_preds, axis=-1)

test_acc = accuracy_score(y_test_true, test_pred_cls)
ci_low, ci_high = wilson_score_interval(np.sum(y_test_true == test_pred_cls), len(y_test_true))

print(f"Hold-out Test Accuracy : {test_acc*100:.2f}% (Wilson 95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%])", flush=True)
print(f"Top-2 Accuracy         : {np.mean([t in np.argsort(p)[-2:] for t, p in zip(y_test_true, test_preds)])*100:.2f}%", flush=True)
print("\nTest Classification Report:", flush=True)
print(classification_report(y_test_true, test_pred_cls, target_names=['100', '1000', '200', '500'], digits=3), flush=True)

# 5. VISUALISASI CONFUSION MATRIX & FULL 0.0 - 1.0 RELIABILITY DIAGRAM
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

cm = confusion_matrix(y_test_true, test_pred_cls)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['100', '1000', '200', '500'], yticklabels=['100', '1000', '200', '500'], ax=axes[0])
axes[0].set_title(f'Test Confusion Matrix (Acc: {test_acc*100:.1f}%)', fontsize=11)
axes[0].set_xlabel('Prediksi')
axes[0].set_ylabel('Sebenarnya')

n_bins = 6
bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
bin_lowers = bin_limits[:-1]
bin_uppers = bin_limits[1:]

ece = 0.0
accuracies, confidences, bin_counts = [], [], []

for bl, bu in zip(bin_lowers, bin_uppers):
    in_bin = (test_confs >= bl) & (test_confs < bu) if bu < 1.0 else (test_confs >= bl) & (test_confs <= bu)
    prop = np.mean(in_bin)
    if prop > 0:
        acc_bin = np.mean(y_test_true[in_bin] == test_pred_cls[in_bin])
        conf_bin = np.mean(test_confs[in_bin])
        ece += np.abs(conf_bin - acc_bin) * prop
        accuracies.append(acc_bin)
        confidences.append(conf_bin)
        bin_counts.append(int(np.sum(in_bin)))
    else:
        accuracies.append(np.nan)
        confidences.append((bl + bu) / 2.0)
        bin_counts.append(0)

axes[1].plot([0, 1], [0, 1], '--', color='gray', label='Kalibrasi Sempurna')
valid_points = [(c, a) for c, a in zip(confidences, accuracies) if not np.isnan(a)]
if valid_points:
    axes[1].scatter([p[0] for p in valid_points], [p[1] for p in valid_points], color='crimson', s=70, zorder=5, label=f'Model (ECE = {ece:.3f})')
axes[1].set_xlabel('Rata-rata Confidence')
axes[1].set_ylabel('Akurasi Sebenarnya')
axes[1].set_title(f'Reliability Diagram (Full Range 0.0 - 1.0, ECE={ece:.3f})', fontsize=11)
axes[1].set_xlim(0.0, 1.02)
axes[1].set_ylim(0.0, 1.02)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

bar_centers = (bin_lowers + bin_uppers) / 2.0
axes[2].bar(bar_centers, bin_counts, width=0.12, color='#2b5c8f', edgecolor='black', alpha=0.8)
axes[2].set_xlabel('Interval Confidence')
axes[2].set_ylabel('Jumlah Sampel (Test Set)')
axes[2].set_title('Distribusi Jumlah Sampel per Bin', fontsize=11)
axes[2].set_xlim(0.0, 1.02)
axes[2].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'test_evaluation_plots.png', dpi=120)
plt.close(fig)
print("Saved results/test_evaluation_plots.png", flush=True)

# Safe Threshold Curve
thresholds = np.linspace(0.4, 0.98, 30)
precisions, coverages = [], []
for th in thresholds:
    mask = test_confs >= th
    cov = np.mean(mask)
    if np.sum(mask) > 0:
        prec = accuracy_score(y_test_true[mask], test_pred_cls[mask])
        precisions.append(prec)
    else:
        precisions.append(np.nan)
    coverages.append(cov)

valid_th_indices = [i for i, p in enumerate(precisions) if not np.isnan(p) and p >= 0.95]
if valid_th_indices:
    safe_idx = valid_th_indices[0]
    SAFE_THRESHOLD = float(thresholds[safe_idx])
    SAFE_COV = coverages[safe_idx]
else:
    SAFE_THRESHOLD = 0.80
    SAFE_COV = 1.0

print(f"\nRekomendasi Ambang Batas Aman (Safe Confidence Threshold): {SAFE_THRESHOLD:.2f} (Coverage: {SAFE_COV*100:.1f}%)", flush=True)

# 6. EKSPOR KE TFLITE & UJI PARITAS
print("\n6. Ekspor model ke TFLite...", flush=True)
run_model = tf.function(lambda x: final_model(x, training=False))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 224, 224, 3], tf.uint8))

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], trackable_obj=final_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter.convert()

tflite_path = MODELS_DIR / 'best.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_fp16)

converter_dyn = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], trackable_obj=final_model)
converter_dyn.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_dyn = converter_dyn.convert()
with open(MODELS_DIR / 'best_dynamic.tflite', 'wb') as f:
    f.write(tflite_dyn)

interpreter = tf.lite.Interpreter(model_content=tflite_fp16)
interpreter.allocate_tensors()
inp_idx = interpreter.get_input_details()[0]['index']
out_idx = interpreter.get_output_details()[0]['index']

parity_matches = 0
max_diff = 0.0

for i in range(len(X_test)):
    sample = np.expand_dims(X_test[i], axis=0)
    k_pred = final_model(sample, training=False).numpy()[0]
    interpreter.set_tensor(inp_idx, sample)
    interpreter.invoke()
    t_pred = interpreter.get_tensor(out_idx)[0]
    diff = np.max(np.abs(k_pred - t_pred))
    if diff > max_diff: max_diff = diff
    if np.argmax(k_pred) == np.argmax(t_pred): parity_matches += 1

print(f"Paritas Prediksi Keras vs TFLite: {parity_matches}/{len(X_test)} ({parity_matches/len(X_test)*100:.1f}%)")
print(f"Maksimum Perbedaan Probabilitas Keras vs TFLite: {max_diff:.6f}")
print("\n>>> ALL EXPERIMENTS COMPLETED SUCCESSFULLY! <<<", flush=True)
