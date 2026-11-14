import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import pandas as pd

df = pd.read_csv('model_coins/splits/dataset_detailed_inventory.csv', encoding='utf-8')
target_files = ['image copy 10.png', 'image copy 11.png', 'image copy 13.png', 'image copy 19.png', 'image copy 20.png']

for _, r in df[df['filename'].isin(target_files)].iterrows():
    print(f"File: {r['filename']} | Original Class: {r['original_class']} | Sisi: {r['coin_side']} | Target 5-Way: {r['target_class_5way']}")

