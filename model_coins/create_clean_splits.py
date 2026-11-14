import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os, json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
inv_path = os.path.join(BASE_DIR, 'splits', 'dataset_detailed_inventory.csv')
df = pd.read_csv(inv_path, encoding='utf-8')

# 1. Hapus exact duplicates (hanya 89 citra unik)
df_clean = df[~df['is_duplicate']].copy().reset_index(drop=True)
print(f"Total unique images retained: {len(df_clean)}")

# 2. Definisikan label 5-kelas
# 100, 200, 500, 1000 (sisi angka) dan sisi_garuda (sisi lambang garuda/pahlawan)
df_clean['label_5class'] = df_clean.apply(
    lambda r: 'sisi_garuda' if r['coin_side'] == 'garuda_pahlawan' else str(r['original_class']),
    axis=1
)

# 3. Hitung dHash 8x8 untuk grouping (mencegah citra visual kembar beda fold)
def calc_dhash(img_path):
    img = Image.open(img_path).convert('L').resize((9, 8), Image.Resampling.BILINEAR)
    pixels = np.array(img)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()

hashes = [calc_dhash(p) for p in df_clean['path']]
n = len(df_clean)

# Cluster into groups using Hamming distance <= 6
parent = list(range(n))
def find(i):
    if parent[i] == i: return i
    parent[i] = find(parent[i])
    return parent[i]

def union(i, j):
    root_i = find(i)
    root_j = find(j)
    if root_i != root_j:
        parent[root_i] = root_j

for i in range(n):
    for j in range(i + 1, n):
        h_dist = np.count_nonzero(hashes[i] != hashes[j])
        if h_dist <= 6:
            union(i, j)

group_ids = [find(i) for i in range(n)]
# Re-index groups to 0..G-1
unique_groups = {g: idx for idx, g in enumerate(set(group_ids))}
df_clean['group_id'] = [unique_groups[g] for g in group_ids]

print(f"Grouped {n} samples into {len(unique_groups)} visual clusters using dHash.")

# 4. StratifiedGroupKFold (5 Fold)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fold_assignments = np.zeros(n, dtype=int)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(df_clean, df_clean['label_5class'], df_clean['group_id'])):
    fold_assignments[val_idx] = fold

df_clean['fold'] = fold_assignments

# Simpan manifest bersih
clean_manifest_path = os.path.join(BASE_DIR, 'splits', 'clean_89_manifest.csv')
df_clean.to_csv(clean_manifest_path, index=False, encoding='utf-8')

# Simpan summary fold distribution
splits_summary = {}
for f in range(5):
    val_subset = df_clean[df_clean['fold'] == f]
    counts = val_subset['label_5class'].value_counts().to_dict()
    splits_summary[f"fold_{f}"] = {
        "total_val": len(val_subset),
        "val_indices": val_subset.index.tolist(),
        "class_distribution": counts
    }

splits_json_path = os.path.join(BASE_DIR, 'splits', 'clean_89_5fold_splits.json')
with open(splits_json_path, 'w', encoding='utf-8') as f:
    json.dump(splits_summary, f, indent=2)

print("\nFold Validation Sample Distribution (5-Class):")
for f, info in splits_summary.items():
    print(f"  {f}: Total {info['total_val']} samples | {info['class_distribution']}")

print(f"\nSaved clean manifest to {clean_manifest_path}")
print(f"Saved 5-fold split JSON to {splits_json_path}")

