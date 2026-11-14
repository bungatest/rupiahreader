import os, sys, glob, hashlib
import numpy as np
import cv2
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

data_dir = r"C:\Users\Bunga Alyssa\OneDrive\文档\DS\smt  7\deep l\uas\frontend\dataset coin"
files = glob.glob(os.path.join(data_dir, '*', '*.png'))

print(f"Total files: {len(files)}")

# Compute MD5
md5_dict = {}
for f in files:
    with open(f, 'rb') as fp:
        m = hashlib.md5(fp.read()).hexdigest()
    md5_dict.setdefault(m, []).append(f)

duplicates = {k: v for k, v in md5_dict.items() if len(v) > 1}
print(f"Total unique MD5: {len(md5_dict)}")
print(f"Duplicate groups: {len(duplicates)}")
for k, v in duplicates.items():
    rel = [os.path.join(os.path.basename(os.path.dirname(p)), os.path.basename(p)) for p in v]
    print(f"  MD5 {k[:8]}: {rel}")


