"""
server.py - Production-Ready FastAPI Service for Rupiah Coin Session Classifier

Features:
- Single-worker design with thread-safe TFLite invocation via threading.Lock
- Low latency: Configurable TTA (defaults to single forward pass [0°] to preserve 60-80ms responsiveness)
- Safe binding: Binds strictly to 127.0.0.1 (safe for cloudflared / ngrok tunnels)
- Strict scoping: Only serves rupiah-reader frontend assets (index.html, script.js, coin-reader.js, style.css)
- Strict privacy: NEVER writes camera frames to disk
"""

import os
import sys
import time
import threading
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# Prevent oneDNN/OpenMP thread conflicts
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "/tmp"), "mpl_cache")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from infer import FrameQualityGate, CoinClassifierTTA, CoinSession

app = FastAPI(title="Rupiah Coin Reader API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Thread-safe lock for TFLite Interpreter
_tflite_lock = threading.Lock()

# Global singletons
gate = FrameQualityGate()
classifier = None

# Active in-memory sessions: session_id -> { "session": CoinSession, "last_active": timestamp }
active_sessions = {}
_session_lock = threading.Lock()
SESSION_TTL_SECONDS = 60.0
MAX_FILE_BYTES = 5 * 1024 * 1024 # 5 MB

# Latency Optimization: Set USE_TTA=False by default to avoid 4x CPU latency penalty
# In CPU mode: 1 pass ~70ms vs 4 passes ~300ms
USE_TTA = False

def cleanup_stale_sessions():
    now = time.time()
    with _session_lock:
        stale_keys = [sid for sid, data in active_sessions.items() if now - data["last_active"] > SESSION_TTL_SECONDS]
        for sid in stale_keys:
            del active_sessions[sid]

@app.on_event("startup")
def startup_event():
    global classifier
    if classifier is None:
        classifier = CoinClassifierTTA()
        print(f"API initialized: CoinClassifierTTA loaded (USE_TTA={USE_TTA}).", flush=True)

def _sync_step_session(session_id: str, bgr_img: np.ndarray):
    """Executes Quality Gate, inference, and session consensus inside worker threadpool."""
    now = time.time()
    with _session_lock:
        if session_id not in active_sessions:
            active_sessions[session_id] = {
                "session": CoinSession(
                    tau_frame=0.75,
                    min_agree=0.85,
                    min_strong=0.60,
                    min_mean_p=0.70,
                    max_garuda=0.35,
                    min_frames=6,
                    max_seconds=15.0,
                    max_flips=3
                ),
                "last_active": now
            }
        session_data = active_sessions[session_id]
        session_data["last_active"] = now
        session = session_data["session"]

    # Step A: Quality Gate Filter
    passed, reason, crop = gate.check(bgr_img)
    if not passed:
        msg_map = {
            "terlalu_gelap": "Terlalu gelap, cari tempat lebih terang.",
            "terlalu_silau": "Terlalu silau, miringkan koin sedikit.",
            "gambar_blur": "Kamera kabur, tahan tangan agar stabil."
        }
        return {
            "status": "gate_failed",
            "message": msg_map.get(reason, "Dekatkan koin ke tengah kamera.")
        }

    # Step B: Fast Inference (Thread-Safe)
    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    with _tflite_lock:
        if USE_TTA:
            p5 = classifier.predict_with_tta(pil_crop).copy()
        else:
            p5 = classifier._predict_single_rot(pil_crop).copy()

    # Step C: Step session consensus
    res = session.update(p5, crop)
    status = res.get("status", "lanjut")

    resp = {
        "status": status,
        "message": res.get("audio", "")
    }

    if status == "jawab":
        resp["label"] = str(res.get("decision", ""))
        with _session_lock:
            if session_id in active_sessions:
                del active_sessions[session_id]
    elif status == "menyerah":
        with _session_lock:
            if session_id in active_sessions:
                del active_sessions[session_id]

    return resp

@app.get("/health")
def health():
    with _session_lock:
        count = len(active_sessions)
    return {"status": "ok", "service": "rupiah-coin-reader", "use_tta": USE_TTA, "active_sessions": count}

@app.post("/predict_coin_session")
async def predict_coin_session(
    session_id: str = Form(...),
    frame: UploadFile = File(...)
):
    """
    Contract:
      POST /predict_coin_session (multipart/form-data)
        session_id: string (UUID per session)
        frame: JPEG file
      Response JSON:
        { "status": "jawab" | "balik_koin" | "gate_failed" | "lanjut" | "menyerah",
          "label": "100" | "200" | "500" | "1000",
          "message": "teks arahan" }
    """
    cleanup_stale_sessions()

    # Read bytes into memory (Strict privacy: no disk write)
    contents = await frame.read()
    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Ukuran frame melebihi batas 5MB.")

    nparr = np.frombuffer(contents, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"status": "gate_failed", "message": "Frame kamera tidak terbaca."}

    # Offload blocking computer vision & model inference to threadpool
    return await run_in_threadpool(_sync_step_session, session_id, bgr)

# Strict Scoping: ONLY serve rupiah-reader web assets
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "rupiah-reader"

@app.get("/", response_class=HTMLResponse)
def get_index():
    with open(FRONTEND_DIR / "index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/script.js")
def get_script():
    with open(FRONTEND_DIR / "script.js", "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="application/javascript")

@app.get("/coin-reader.js")
def get_coin_reader():
    with open(FRONTEND_DIR / "coin-reader.js", "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="application/javascript")

@app.get("/style.css")
def get_style():
    with open(FRONTEND_DIR / "style.css", "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="text/css")

if __name__ == "__main__":
    import uvicorn
    # Single worker is mandatory for in-memory session coherence; bind strictly to 127.0.0.1
    uvicorn.run("server:app", host="127.0.0.1", port=8000, workers=1, reload=False)
