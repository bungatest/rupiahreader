import os, sys, glob, hashlib
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
data_dir = r"C:\Users\Bunga Alyssa\OneDrive\文档\DS\smt  7\deep l\uas\frontend\dataset coin"

# Exact list of Garuda / Pahlawan images identified from contact sheets:
# Class 100:
garuda_100 = {
    'image copy 10.png', 'image copy 11.png', 'image copy 12.png',
    'image copy 17.png', 'image copy 19.png', 'image copy 20.png',
    'image copy 21.png', 'image copy 22.png'
}

# Class 200:
garuda_200 = {
    'image copy 10.png', 'image copy 11.png', 'image copy 15.png',
    'image copy 7.png', 'image copy 8.png', 'image copy 9.png'
}

# Class 500:
garuda_500 = {
    'image copy 15.png', 'image copy 16.png', 'image copy 17.png',
    'image copy 18.png', 'image copy 19.png', 'image copy 21.png',
    'image copy 22.png', 'image copy 23.png', 'image copy 24.png',
    'image copy 25.png', 'image copy 26.png', 'image copy 27.png',
    'image copy 28.png', 'image copy 29.png'
}

# Class 1000:
garuda_1000 = {
    'image copy 14.png', 'image copy 15.png', 'image copy 17.png',
    'image copy 18.png', 'image copy 21.png', 'image copy 4.png',
    'image copy 8.png'
}

garuda_lookup = {
    '100': garuda_100,
    '200': garuda_200,
    '500': garuda_500,
    '1000': garuda_1000
}

files = glob.glob(os.path.join(data_dir, '*', '*.png'))
records = []
seen_md5 = set()

for f in files:
    cls = os.path.basename(os.path.dirname(f))
    fname = os.path.basename(f)
    with open(f, 'rb') as fp:
        m = hashlib.md5(fp.read()).hexdigest()
    
    is_dupe = (m in seen_md5)
    seen_md5.add(m)
    
    is_garuda = fname in garuda_lookup.get(cls, set())
    coin_side = 'garuda_pahlawan' if is_garuda else 'angka'
    target_5way = 'sisi_garuda' if is_garuda else cls
    
    records.append({
        'path': f,
        'filename': fname,
        'original_class': cls,
        'coin_side': coin_side,
        'target_class_5way': target_5way,
        'is_duplicate': is_dupe,
        'md5': m
    })

df = pd.DataFrame(records)
out_csv = r"model_coins\splits\dataset_detailed_inventory.csv"
df.to_csv(out_csv, index=False)
print(f"Inventory saved to {out_csv}")
print(f"Total images: {len(df)}")
print(f"Coin side breakdown:")
print(df['coin_side'].value_counts())
print(f"\n5-way target breakdown (including duplicates):")
print(df['target_class_5way'].value_counts())

df_unique = df[~df['is_duplicate']]
print(f"\nUnique images: {len(df_unique)}")
print(f"5-way target breakdown (UNIQUE ONLY):")
print(df_unique['target_class_5way'].value_counts())

