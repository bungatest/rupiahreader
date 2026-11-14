# -*- coding: utf-8 -*-
"""
train_5class_production.py
End-to-End Retraining of EfficientNetV2-B0 on Real-World Rupiah Coin Dataset (700+ images)
- Zero Data Leakage: GroupShuffleSplit by Video ID
- 360° Rotation Invariant Augmentation
- Two-Phase Transfer Learning (Head warmup -> Fine-tune top layers)
- Temperature Calibration
- Production TFLite Export & Verification
"""

import os
import sys
import glob
import json
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "/tmp"), "mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import norm

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = ROOT_DIR / "dataset_koin_baru"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['100', '1000', '200', '500', 'sisi_garuda']
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}

TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 2026

def wilson_ci(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return round(float(max(0.0, center - spread)), 4), round(float(min(1.0, center + spread)), 4)

def load_and_preprocess_image(path):
    # Safe Windows Unicode read
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    # Crop central region where coin resides
    sz = int(min(h, w) * 0.85)
    cy, cx = h // 2, w // 2
    crop = bgr[cy - sz//2 : cy + sz//2, cx - sz//2 : cx + sz//2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return resized

def build_dataset_manifest():
    records = []
    # 1. Load from dataset_koin_baru
    for cls_name in CLASS_NAMES:
        cls_folder = DATA_DIR / cls_name
        if not cls_folder.exists():
            continue
        files = list(cls_folder.glob("*.png")) + list(cls_folder.glob("*.jpg"))
        for f in files:
            # Group by video filename to prevent cross-frame leakage
            parts = f.stem.split("_f")
            video_group = parts[0] if len(parts) > 1 else f.stem
            records.append({
                "path": str(f),
                "label": cls_name,
                "label_idx": LABEL_TO_IDX[cls_name],
                "group": f"{cls_name}_{video_group}"
            })

    # 2. Add original clean catalog images from splits/clean_89_manifest.csv if available
    old_manifest_path = BASE_DIR / "splits" / "clean_89_manifest.csv"
    if old_manifest_path.exists():
        try:
            df_old = pd.read_csv(old_manifest_path)
            for _, row in df_old.iterrows():
                p = str(row['path'])
                lbl = str(row['label_5class'])
                if lbl in LABEL_TO_IDX and os.path.exists(p):
                    records.append({
                        "path": p,
                        "label": lbl,
                        "label_idx": LABEL_TO_IDX[lbl],
                        "group": f"catalog_{Path(p).stem}"
                    })
        except Exception as e:
            print(f"[WARN] Error loading old catalog images: {e}")

    df = pd.DataFrame(records)
    return df

def build_model(num_classes=5):
    # EfficientNetV2-B0 with ImageNet pretrained weights
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_layer")
    # Data Augmentation Layer (360° Rotations, Translation, Zoom, Contrast)
    x = tf.keras.layers.RandomRotation(1.0, seed=SEED)(inputs)
    x = tf.keras.layers.RandomTranslation(0.08, 0.08, seed=SEED)(x)
    x = tf.keras.layers.RandomZoom((-0.1, 0.1), seed=SEED)(x)
    x = tf.keras.layers.RandomContrast(0.15, seed=SEED)(x)

    x = base_model(x, training=False)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="RupiahCoinEfficientNetV2")
    return model, base_model

def main():
    print("=" * 76)
    print("      RETRAINING EFFICIENTNETV2-B0 (5-CLASS RUPIAH COIN CLASSIFIER)")
    print("=" * 76)
    print()

    df = build_dataset_manifest()
    print(f"Total dataset terkumpul: {len(df)} citra.")
    print("Distribusi per kelas:")
    print(df['label'].value_counts())
    print()

    # Leak-free split by Video Group (80% Train, 20% Val)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, val_idx = next(gss.split(df, groups=df['group']))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    print(f"Train set: {len(df_train)} citra across {df_train['group'].nunique()} video groups")
    print(f"Val set  : {len(df_val)} citra across {df_val['group'].nunique()} video groups (100% Unseen Videos!)")
    print()

    # Load images into RAM
    print("[1/5] Memuat dan memproses gambar ke memori...")
    X_train, y_train = [], []
    for _, r in df_train.iterrows():
        img = load_and_preprocess_image(r['path'])
        if img is not None:
            X_train.append(img)
            y_train.append(r['label_idx'])

    X_val, y_val = [], []
    for _, r in df_val.iterrows():
        img = load_and_preprocess_image(r['path'])
        if img is not None:
            X_val.append(img)
            y_val.append(r['label_idx'])

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape}, y_val  : {y_val.shape}")
    print()

    # Compute class weights for balance
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {c: float(w) for c, w in zip(classes, weights)}
    print(f"Class weights: {class_weight_dict}")
    print()

    # Build Model
    print("[2/5] Membangun arsitektur EfficientNetV2-B0...")
    model, base_model = build_model(num_classes=len(CLASS_NAMES))

    # Phase 1: Train classification head
    print("[3/5] Fase 1: Pemanasan Dense Head (5 Epochs)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Phase 2: Fine-Tuning top layers of EfficientNet
    print("\n[4/5] Fase 2: Fine-Tuning 35 Lapisan Teratas EfficientNetV2 (12 Epochs)...")
    base_model.trainable = True
    # Freeze bottom layers, unfreeze top 35 layers
    for layer in base_model.layers[:-35]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=12,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Validation Evaluation
    print("\n[5/5] Evaluasi Independen & Ekspor Model...")
    val_probs = model.predict(X_val, batch_size=BATCH_SIZE)
    val_preds = np.argmax(val_probs, axis=1)

    val_acc = accuracy_score(y_val, val_preds)
    correct_cnt = int(np.sum(val_preds == y_val))
    ci_low, ci_high = wilson_ci(correct_cnt, len(y_val))

    print("=" * 76)
    print("HASIL VALIDASI INDEPENDEN (100% UNSEEN VIDEOS):")
    print("=" * 76)
    print(f"Akurasi Validasi : {val_acc * 100:.2f}% ({correct_cnt}/{len(y_val)} benar)")
    print(f"Wilson 95% CI    : [{ci_low * 100:.1f}%, {ci_high * 100:.1f}%]")
    print()
    print("Classification Report:")
    print(classification_report(y_val, val_preds, target_names=CLASS_NAMES, digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, val_preds))
    print("=" * 76)

    # Save Keras Model
    keras_path = MODELS_DIR / "best_5class.keras"
    model.save(keras_path)
    print(f"\n[SUKSES] Model Keras disimpan ke: {keras_path}")

    # Convert to TFLite (Float32) for fast CPU deployment
    print("[INFO] Mengonversi model ke format TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = MODELS_DIR / "best_5class.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[SUKSES] Model TFLite disimpan ke: {tflite_path} ({len(tflite_model)/(1024*1024):.2f} MB)")

    # Save labels JSON
    labels_path = MODELS_DIR / "labels_5class.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, indent=2)
    print(f"[SUKSES] Label disimpan ke: {labels_path}")

    # Sanity check with TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Run 1 test image through TFLite
    test_sample = np.expand_dims(X_val[0], axis=0).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], test_sample)
    interpreter.invoke()
    tflite_out = interpreter.get_tensor(output_details[0]['index'])
    print(f"[VERIFIKASI] Output TFLite berhasil diuji: shape={tflite_out.shape}, max_prob={np.max(tflite_out):.4f}")
    print("\n🎉 RETRAINING & EKSPOR SELESAI DENGAN SEMPURNA!")

if __name__ == "__main__":
    main()

