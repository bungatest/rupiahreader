import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

target_files = [
    (r"C:\Users\Bunga Alyssa\OneDrive\文档\DS\smt  7\deep l\uas\frontend\dataset coin\1000\image copy 10.png", "1000/image copy 10.png"),
    (r"C:\Users\Bunga Alyssa\OneDrive\文档\DS\smt  7\deep l\uas\frontend\dataset coin\1000\image copy 11.png", "1000/image copy 11.png"),
    (r"C:\Users\Bunga Alyssa\OneDrive\文档\DS\smt  7\deep l\uas\frontend\dataset coin\1000\image copy 13.png", "1000/image copy 13.png"),
    (r"C:\Users\Bunga Alyssa\OneDrive\文档\DS\smt  7\deep l\uas\frontend\dataset coin\1000\image copy 19.png", "1000/image copy 19.png"),
    (r"C:\Users\Bunga Alyssa\OneDrive\文档\DS\smt  7\deep l\uas\frontend\dataset coin\100\image copy 20.png", "100/image copy 20.png"),
]

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, (p, title) in zip(axes, target_files):
    img = Image.open(p)
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

plt.tight_layout()
out_p = r"model_coins/results/inspect_1000_errors.png"
plt.savefig(out_p, dpi=150)
plt.close()
print("Saved inspect_1000_errors.png successfully!")

