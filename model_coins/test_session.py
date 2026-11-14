"""
test_session.py - Client Test Simulator for Coin Session Reader
Usage:
  python test_session.py "path/to/coin_image.png"

Simulates physical coin rotation across 14 frames, measuring per-request latency (ms)
and verifying the status progression (lanjut -> jawab/menyerah/balik_koin).
"""

import sys
import time
import uuid
from pathlib import Path
import cv2
import numpy as np
import requests

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

if len(sys.argv) < 2:
    print("Penggunaan: python test_session.py \"path/ke/gambar_koin.png\"")
    sys.exit(1)

img_path = sys.argv[1]
with open(img_path, "rb") as f:
    buf_raw = f.read()
img = cv2.imdecode(np.frombuffer(buf_raw, np.uint8), cv2.IMREAD_COLOR)

if img is None:
    print(f"Error: Gagal memuat citra dari {img_path}")
    sys.exit(1)

h, w = img.shape[:2]
sid = str(uuid.uuid4())
URL = "http://127.0.0.1:8000/predict_coin_session"

print("="*75)
print(f"UJI SIMULASI SESI MULTI-FRAME: {Path(img_path).name} (Resolusi: {w}x{h})")
print(f"Target Endpoint: {URL} | Session ID: {sid[:8]}...")
print("="*75)

latencies = []
for i in range(14):
    # Rotasikan citra sebesar 27 derajat tiap frame untuk mensimulasikan gerakan tangan
    M = cv2.getRotationMatrix2D((w / 2, h / 2), i * 27, 1.0)
    f = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    t0 = time.time()
    try:
        r = requests.post(
            URL,
            data={"session_id": sid},
            files={"frame": ("f.jpg", buf.tobytes(), "image/jpeg")},
            timeout=10
        )
        elapsed_ms = 1000 * (time.time() - t0)
        latencies.append(elapsed_ms)
        resp_json = r.json()
        print(f"Frame {i+1:2d} | Latensi: {elapsed_ms:5.1f} ms | Status: {r.status_code} | {resp_json}")
        
        status = resp_json.get("status")
        if status in ("jawab", "menyerah"):
            print(f"\n>>> Sesi BERAKHIR pada frame {i+1} dengan status '{status}'!")
            break
        elif status == "balik_koin":
            print(f"    [Arahan Balik Koin]: {resp_json.get('message')}")
    except requests.exceptions.ConnectionError:
        print(f"Error: Server tidak aktif di {URL}. Pastikan 'python -m uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1' sedang berjalan.")
        sys.exit(1)
    except Exception as e:
        print(f"Frame {i+1:2d} Error: {e}")
        break

if latencies:
    print("\n" + "-"*50)
    print(f"Rata-rata Latensi Server : {np.mean(latencies):.1f} ms per request")
    print(f"Median Latensi           : {np.median(latencies):.1f} ms per request")
    print(f"Max Latensi              : {np.max(latencies):.1f} ms")
    print("-"*50)

