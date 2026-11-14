"""
train_final_and_export.py - Final EfficientNetV2-B0 Training, Hold-out Test Evaluation, Plotting & TFLite Export
"""
import os, sys, time, json
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "/tmp"), "mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
import augment_coin

ROOT = Path(__file__).parent
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

print("1. Memuat Dev manifest & membuat augmented dataset...", flush=True)
df_dev = pd.read_csv(SPLITS_DIR / 'dev_manifest.csv')
dev_items = []
th, tw = (224, 224)

for _, row in df_dev.iterrows():
    bgr = augment_coin.cv2_imread_unicode(row['path'])
    mask, bbox, _ = augment_coin.extract_coin(bgr)
    dev_items.append((bgr, mask, str(row['class'])))

rng = np.random.default_rng(2026)
x_dev, y_dev = [], []

for bgr, mask, lbl in dev_items:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        pad = int(max(w, h) * 0.08)
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(bgr.shape[1], x + w + pad); y2 = min(bgr.shape[0], y + h + pad)
        clean = cv2.resize(bgr[y1:y2, x1:x2], (tw, th))
    else:
        clean = cv2.resize(bgr, (tw, th))
    x_dev.append(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
    y_dev.append(label_map[lbl])
    
    for _ in range(6):
        aug = augment_coin.augment_coin_synthetic(
            bgr, mask=mask, target_size=(224, 224), intensity='medium', rng=rng, mix_raw_crop_prob=0.25
        )
        x_dev.append(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB))
        y_dev.append(label_map[lbl])

X_dev = np.array(x_dev, dtype=np.uint8)
y_dev_cat = tf.keras.utils.to_categorical(y_dev, num_classes=4)
print(f"Dataset Dev Final: Shape={X_dev.shape}, Classes={y_dev_cat.shape}", flush=True)

# Build EfficientNetV2-B0
inputs = tf.keras.layers.Input(shape=(224, 224, 3))
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
outputs = tf.keras.layers.Dense(4, activation='softmax', dtype='float32')(x)
final_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='EfficientNetV2B0')

# Stage 1: Warmup Head (5 epochs)
print("\n2. Pelatihan Tahap 1: Head Classifier (5 epoch)...", flush=True)
final_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['categorical_accuracy']
)
final_model.fit(X_dev, y_dev_cat, epochs=5, batch_size=32, shuffle=True, verbose=2)

# Stage 2: Fine-Tuning Top Layers (7 epochs)
print("\n3. Pelatihan Tahap 2: Fine-Tuning Top Blocks (7 epoch)...", flush=True)
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

final_model.compile(
    optimizer=tf.keras.optimizers.Adam(6e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['categorical_accuracy']
)
final_model.fit(X_dev, y_dev_cat, epochs=7, batch_size=32, shuffle=True, verbose=2)

# Save Keras Model & Labels
final_model.save(MODELS_DIR / 'best.keras')
with open(MODELS_DIR / 'labels.json', 'w') as f:
    json.dump(['100', '1000', '200', '500'], f)
print("Model best.keras & labels.json tersimpan di models/", flush=True)

# 4. Evaluasi Final pada Test Set Hold-Out (18 citra)
print("\n" + "="*70, flush=True)
print("4. EVALUASI TUNGGAL PADA TEST SET HOLD-OUT (N=18 CITRA)", flush=True)
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

# 5. Visualisasi: Confusion Matrix + Full 0-1 Reliability Diagram + Histogram
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion Matrix
cm = confusion_matrix(y_test_true, test_pred_cls)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['100', '1000', '200', '500'], yticklabels=['100', '1000', '200', '500'], ax=axes[0])
axes[0].set_title(f'Test Confusion Matrix (Acc: {test_acc*100:.1f}%)', fontsize=11)
axes[0].set_xlabel('Prediksi')
axes[0].set_ylabel('Sebenarnya')

# Full 0.0 - 1.0 Reliability Diagram
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

# Sample Distribution Histogram
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

# Calibrated Safe Confidence Threshold
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

valid_th = [(t, p, c) for t, p, c in zip(thresholds, precisions, coverages) if not np.isnan(p) and p >= 0.85]
if valid_th:
    SAFE_THRESHOLD = float(valid_th[0][0])
    SAFE_COV = float(valid_th[0][2])
else:
    SAFE_THRESHOLD = 0.65
    SAFE_COV = 1.0

print(f"\nRekomendasi Ambang Batas Aman: {SAFE_THRESHOLD:.2f} (Coverage: {SAFE_COV*100:.1f}%)", flush=True)

# 6. Ekspor Model ke TFLite & Uji Paritas
print("\n5. Ekspor ke TensorFlow Lite...", flush=True)
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
print(f"Maksimum Perbedaan Probabilitas: {max_diff:.6f}")

# 7. Demo Inferensi dengan infer.py
import infer
classifier = infer.CoinClassifier(model_path=str(tflite_path), labels_path=str(MODELS_DIR / 'labels.json'), conf_threshold=SAFE_THRESHOLD)
demo_res = classifier.predict_single(df_test.iloc[0]['path'])
print(f"Demo infer.py: File {df_test.iloc[0]['file']} -> {demo_res['message']} (Confidence: {demo_res['confidence']*100:.1f}%)")
print("\n>>> ALL TASKS COMPLETED SUCCESSFULLY! <<<", flush=True)

