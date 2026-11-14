# colab_embedding_baseline.py
# Baseline Frozen Embedding (EfficientNetV2-B0 & DINOv2-S) + Logistic Regression + TTA Rotasi
# Siap diunggah dan dijalankan langsung di Google Colab / GPU Environment

"""
Panduan Penggunaan di Google Colab:
1. Buat notebook baru di Colab (Pilih Runtime GPU T4 bila tersedia).
2. Upload file berikut ke workspace Colab (/content/):
   - augment_coin.py
   - splits/clean_89_manifest.csv
   - dataset coin/ (seluruh folder koin)
3. Jalankan skrip ini atau file colab_embedding_baseline.ipynb.
"""

# !pip -q install transformers

import os, json, numpy as np, pandas as pd, cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
import augment_coin as ac

# ---------- KONFIGURASI ----------
MANIFEST = "splits/clean_89_manifest.csv"   # kolom: path, label, group
CLASSES  = ["100", "1000", "200", "500", "sisi_garuda"]
ANGLES   = [0, 45, 90, 135, 180, 225, 270, 315]

df = pd.read_csv(MANIFEST)
if "label_5class" in df and "label" not in df:
    df["label"] = df["label_5class"]
df["label"] = df["label"].astype(str)

if "group_id" in df and "group" not in df:
    df["group"] = df["group_id"]
elif "group" not in df:
    df["group"] = np.arange(len(df))

y = df["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
groups = df["group"].to_numpy()

# Fold: prioritaskan kolom 'fold' di manifest bila ada
if "fold" in df:
    folds = [(np.where(df.fold != f)[0], np.where(df.fold == f)[0]) for f in sorted(df.fold.unique())]
else:
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(df, y, groups))

print(f"Dataset dimuat: {len(df)} citra unik, {len(folds)} folds.")

# ---------- Preprocessing: crop koin persegi + rotasi ----------
def coin_crop(path, size=224, margin=1.15):
    img = ac.cv2_imread_unicode(path)
    _, (x, y0, w, h), _ = ac.extract_coin(img)
    s = int(max(w, h) * margin); cx, cy = x + w // 2, y0 + h // 2
    pad = s
    imgp = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = imgp[cy - s // 2 + pad: cy - s // 2 + pad + s, cx - s // 2 + pad: cx - s // 2 + pad + s]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

def rot(img, ang):
    if ang % 90 == 0:
        return np.ascontiguousarray(np.rot90(img, k=(ang // 90) % 4))
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), ang, 1.0)
    return cv2.warpAffine(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

print("Memulai ekstraksi crop koin berpusat...")
crops = np.stack([coin_crop(p) for p in df["path"]])
print("Shape crops:", crops.shape)

# ---------- Backbone Feature Extractors ----------
def make_effnet():
    import tensorflow as tf
    m = tf.keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet",
            pooling="avg", input_shape=(224, 224, 3), include_preprocessing=True)
    return lambda b: m.predict(b.astype("float32"), verbose=0, batch_size=32)

def make_dino():
    import torch
    from transformers import AutoModel, AutoImageProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Memuat DINOv2-small pada device: {dev}")
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    mdl = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    @torch.no_grad()
    def f(batch):
        out = []
        for i in range(0, len(batch), 32):
            inp = proc(images=list(batch[i:i + 32]), return_tensors="pt").to(dev)
            h = mdl(**inp).last_hidden_state
            out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu().numpy())
        return np.concatenate(out)
    return f

def embed_all(fn):
    F = {}
    for a in ANGLES:
        e = fn(np.stack([rot(c, a) for c in crops]))
        F[a] = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    return F

# ---------- CV & Skor ----------
def run_cv(F, train_angles, test_angles, C=1.0):
    oof = np.zeros((len(y), len(CLASSES)))
    for tr, va in folds:
        Xtr = np.concatenate([F[a][tr] for a in train_angles])
        ytr = np.concatenate([y[tr] for _ in train_angles])
        clf = make_pipeline(StandardScaler(),
              LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        clf.fit(Xtr, ytr)
        P = np.mean([clf.predict_proba(F[a][va]) for a in test_angles], axis=0)
        oof[np.ix_(va, clf.classes_)] = P
    return oof

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / d
    return c - h, c + h

def score(name, oof):
    pred = oof.argmax(1); ok = pred == y
    row = {"konfigurasi": name}
    for lab, m in [("total", np.ones(len(y), bool)), ("sisi_angka", y < 4), ("sisi_garuda", y == 4)]:
        k, n = ok[m].sum(), m.sum(); lo, hi = wilson(k, n)
        row[lab] = f"{k}/{n} = {k/n:.1%} [{lo:.0%}-{hi:.0%}]"
    row["macroF1"] = round(f1_score(y, pred, average="macro"), 3)
    return row

# ---------- Eksekusi Eksperimen ----------
if __name__ == "__main__":
    rows, saved = [], {}
    for bb_name, maker in [("EffNetV2B0", make_effnet), ("DINOv2-S", make_dino)]:
        print(f"\n--- Menjalankan Embedding {bb_name} ---")
        try:
            fn = maker()
            F = embed_all(fn)
            configs = {
                "tanpa augmentasi, tanpa TTA":          ([0],      [0]),
                "train 8 rotasi, tanpa TTA":            (ANGLES,   [0]),
                "train 8 rotasi + TTA 4 sudut":         (ANGLES,   [0, 90, 180, 270]),
                "train 8 rotasi + TTA 8 sudut":         (ANGLES,   ANGLES),
            }
            for cname, (tr_a, te_a) in configs.items():
                oof = run_cv(F, tr_a, te_a)
                rows.append(score(f"{bb_name} | {cname}", oof))
                saved[f"{bb_name}|{cname}"] = oof
            print(bb_name, "selesai.")
        except Exception as e:
            print(f"Gagal menjalankan {bb_name}: {e}")

    if rows:
        res = pd.DataFrame(rows)
        pd.set_option("display.max_colwidth", 80, "display.width", 200)
        print("\n" + "="*80)
        print("HASIL EVALUASI BASELINE EMBEDDING + LOGISTIC REGRESSION + TTA")
        print("="*80)
        print(res.to_string(index=False))
        print("\nPembanding (Fine-Tune EfficientNetV2-B0 Anda):")
        print("  - Total       : 60/89 = 67.4% [57%-76%]")
        print("  - Sisi Angka  : 34/55 = 61.8% [49%-74%]")
        print("  - Sisi Garuda : 26/34 = 76.5% [60%-88%]")
        np.savez("oof_embedding_baselines.npz", y=y, **{k.replace("|", "__").replace(" ", "_"): v for k, v in saved.items()})
        print("Hasil OOF disimpan ke oof_embedding_baselines.npz")

