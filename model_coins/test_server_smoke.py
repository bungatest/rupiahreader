import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "/tmp")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import io
import asyncio
from pathlib import Path
import httpx

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

import server
from server import app

async def run_smoke_test():
    print("="*75, flush=True)
    print("SMOKE TEST RESMI: FASTAPI SERVER END-TO-END DENGAN CITRA KOIN ASLI", flush=True)
    print("="*75, flush=True)

    # 1. Trigger startup event to load model & TFLite interpreter
    server.startup_event()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8000") as client:
        # Step A: Test /health endpoint
        res_health = await client.get("/health")
        print(f"1. GET /health: Status {res_health.status_code} | Payload: {res_health.json()}", flush=True)
        assert res_health.status_code == 200, "Health check failed!"

        # Step B: Test Frontend Static Routes
        res_front = await client.get("/")
        print(f"2. GET / (Frontend Index): Status {res_front.status_code} ({len(res_front.text)} chars)", flush=True)
        assert res_front.status_code == 200, "Frontend index failed!"
        assert "Rupiah Reader" in res_front.text, "index.html missing Rupiah Reader!"

        res_cr = await client.get("/coin-reader.js")
        print(f"   GET /coin-reader.js: Status {res_cr.status_code} ({len(res_cr.text)} chars)", flush=True)
        assert res_cr.status_code == 200, "coin-reader.js failed!"

        # Step C: Test POST /predict_coin_session with Real Coin Image
        sample_img_path = Path(r"C:\Users\Bunga Alyssa\OneDrive\文档\DS\smt  7\deep l\uas\frontend\dataset coin\1000\image copy 10.png")
        if not sample_img_path.exists():
            sample_img_path = Path("dataset coin/1000/image copy 10.png")

        with open(sample_img_path, "rb") as f:
            img_bytes = f.read()

        print(f"\n3. Menguji POST /predict_coin_session dengan {sample_img_path.name} ({len(img_bytes)} bytes)...", flush=True)
        session_id = "smoke_session_001"

        files = {"frame": ("frame_1.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        data = {"session_id": session_id}
        
        response = await client.post("/predict_coin_session", data=data, files=files)
        print(f"   Response HTTP {response.status_code}: {response.json()}", flush=True)
        assert response.status_code == 200, f"Request failed: {response.text}"
        
        resp_json = response.json()
        assert "status" in resp_json, "Response missing 'status'!"
        assert "message" in resp_json, "Response missing 'message'!"
        print(f"   Status respons: '{resp_json['status']}' | Arahan suara: '{resp_json['message']}'", flush=True)

    print("\n" + "="*75, flush=True)
    print("HASIL UJI SMOKE: FASTAPI SERVER & FRONTEND STATIS BERHASIL 100%!", flush=True)
    print("="*75, flush=True)

if __name__ == "__main__":
    asyncio.run(run_smoke_test())
