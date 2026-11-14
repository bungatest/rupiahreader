import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import cv2
import numpy as np

for fn in ['image copy 10.png', 'image copy 11.png', 'image copy 13.png', 'image copy 19.png']:
    p = f"dataset coin/1000/{fn}"
    with open(p, 'rb') as fp:
        buf = fp.read()
    bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    # Check average color
    b, g, r = bgr[:,:,0].mean(), bgr[:,:,1].mean(), bgr[:,:,2].mean()
    print(f"{p}: shape=({w}x{h}), mean BGR=({b:.1f}, {g:.1f}, {r:.1f})")

