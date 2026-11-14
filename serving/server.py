"""
FastAPI Serving Server untuk Rupiah Reader (Unified Deep Learning API).
Mengintegrasikan Model Klasifikasi Uang Kertas (Custom CNN, MobileNetV2, FNN, Consensus)
dan Uang Koin (Custom CNN / MobileNetV2 TFLite FP16, GoogLeNet Inception),
serta Sistem Uji Lapangan MLOps dan Frontend Web Responsif.
"""

import os
import sys
import json
import time
import io
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import httpx
import numpy as np
import cv2
from PIL import Image

from .config import (
    SERVING_PORT,
    RUPIAH_READER_DIR,
    FEEDBACK_DIR,
    FEEDBACK_IMAGES_DIR,
    COIN_SERVER_URL,
    CLASSES,
    CLASS_DISPLAY_NAMES,
    COIN_CLASSES,
    COIN_PRETTY_NOMINAL,
    COIN_SPOKEN_NOMINAL,
)
from .preprocessing import decode_image_bytes
from .model_manager import ModelManager
from .field_logger import FieldLogger

# ---------------------------------------------------------------------------
# Metadata Tag OpenAPI / Swagger UI
# ---------------------------------------------------------------------------
tags_metadata = [
    {
        "name": "Uang Kertas",
        "description": "Endpoint klasifikasi **7 nominal uang kertas Rupiah** (Rp 1.000 s/d Rp 100.000) menggunakan model Custom CNN, MobileNetV2, FNN, dan Consensus Majority Vote.",
    },
    {
        "name": "Uang Koin",
        "description": "Endpoint klasifikasi **5 kelas uang koin Rupiah** (Rp 100, Rp 200, Rp 500, Rp 1.000, dan Sisi Garuda/Pahlawan) dengan Frame Quality Gate, Deteksi Pahlawan, dan Sesi Kontinu Tunanetra.",
    },
    {
        "name": "Uji Lapangan & MLOps",
        "description": "Pencatatan ground truth label sebenarnya, rekapitulasi akurasi real-world, ekspor log CSV, dan penerimaan umpan balik (feedback) pengguna.",
    },
    {
        "name": "Sistem & Status Model",
        "description": "Pemeriksaan kesehatan layanan (health check) dan pemantauan status ketersediaan model Deep Learning.",
    },
]

app = FastAPI(
    title="Rupiah Reader - API Klasifikasi Uang Kertas & Uang Koin Deep Learning",
    description="""
### API Resmi Sistem Asistensi Pengenalan Mata Uang Rupiah (Deep Learning UAS)

Layanan inferensi Deep Learning terpadu untuk penyandang tunanetra dan masyarakat umum:
- **💵 Uang Kertas (7 Nominal):** Didukung arsitektur **Custom CNN (8-Layer ConvNet)**, **MobileNetV2 Fine-Tuned**, **FNN Baseline**, dan **Majority Consensus Ensemble**.
- **🪙 Uang Koin (5 Kelas):** Didukung arsitektur **Custom CNN / MobileNetV2 TFLite FP16** dan **GoogLeNet Inception Multi-Scale**, dilengkapi dengan *Frame Quality Gate* (filter blur & kilau logam) dan *Dual-Sided Hero Recognition*.

Dokumentasi interaktif Swagger ini dapat diuji coba langsung (*Try it out*) untuk setiap endpoint di bawah.
    """,
    version="2.0.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local dev, mobile tunnel, etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi komponen
model_manager = ModelManager()
field_logger = FieldLogger()
START_TIME = time.time()

# ---------------------------------------------------------------------------
# Schema Model Pydantic (Request & Response)
# ---------------------------------------------------------------------------
class CatatLabelRequest(BaseModel):
    log_id: int = Field(..., description="ID log inferensi dari database pengujian lapangan", example=1)
    ground_truth: str = Field(..., description="Label kelas sebenarnya (contoh: 'idr_50000' atau '500')", example="idr_50000")
    notes: Optional[str] = Field(None, description="Catatan tambahan kondisi pencahayaan atau sudut foto")


class KertasJsonRequest(BaseModel):
    images: List[str] = Field(..., description="Daftar frame citra dalam format Base64 data URL (data:image/jpeg;base64,...)")
    model: str = Field("consensus", description="Model yang dipilih: 'consensus', 'cnn', 'mobilenet', 'fnn'", example="consensus")
    ground_truth: Optional[str] = Field(None, description="Label kelas asli opsional untuk logging MLOps")


class KertasPrediksiResponse(BaseModel):
    model_id: str = Field(..., example="consensus")
    predicted_class: str = Field(..., example="idr_50000")
    predicted_display: str = Field(..., example="Rp 50.000")
    raw_class: str = Field(..., example="idr_50000")
    raw_display: str = Field(..., example="Rp 50.000")
    confidence: float = Field(..., example=0.9852)
    margin: float = Field(..., example=0.8521)
    threshold: float = Field(..., example=0.75)
    status: str = Field(..., example="pasti")
    audio_text: str = Field(..., example="Lima puluh ribu rupiah")
    latency_ms: float = Field(..., example=72.4)
    top3: List[Dict[str, Any]]
    all_probabilities: Dict[str, float]
    log_id: Optional[int] = Field(None, example=42)


class KoinPrediksiResponse(BaseModel):
    predicted_class: str = Field(..., example="500")
    predicted_nominal: str = Field(..., example="Rp 500")
    spoken_text: str = Field(..., example="Lima ratus rupiah")
    side_type: str = Field(..., example="Sisi Nominal (Angka Koin)")
    confidence: float = Field(..., example=0.978)
    quality_gate_passed: bool = Field(..., example=True)
    quality_gate_reason: str = Field(..., example="ok")
    latency_ms: float = Field(..., example=64.1)
    all_probabilities: Dict[str, float]


class KoinSesiResponse(BaseModel):
    status: str = Field(..., description="'jawab' | 'balik_koin' | 'gate_failed' | 'lanjut' | 'menyerah'", example="jawab")
    message: str = Field(..., description="Teks audio untuk text-to-speech bahasa Indonesia", example="Lima ratus rupiah")
    label: Optional[str] = Field(None, description="Nominal koin jika status='jawab'", example="500")


# ---------------------------------------------------------------------------
# Native Coin Engine Singleton & Session State
# ---------------------------------------------------------------------------
_coin_gate = None
_coin_classifier = None
_coin_tflite_lock = threading.Lock()
_coin_session_lock = threading.Lock()
_coin_active_sessions: Dict[str, Any] = {}
COIN_SESSION_TTL = 60.0


def get_native_coin_components():
    global _coin_gate, _coin_classifier
    if _coin_classifier is None:
        try:
            root_ws = Path(__file__).resolve().parent.parent
            if str(root_ws) not in sys.path:
                sys.path.insert(0, str(root_ws))
            from model_coins.infer import FrameQualityGate, CoinClassifierTTA
            _coin_gate = FrameQualityGate()
            _coin_classifier = CoinClassifierTTA()
            print("[Server] Native Coin Engine (Quality Gate & TFLite) initialized successfully.", flush=True)
        except Exception as e:
            print(f"[Server] Warning: failed to initialize native coin engine: {e}", flush=True)
    return _coin_gate, _coin_classifier


def _sync_step_coin_session(session_id: str, bgr_img: np.ndarray) -> Dict[str, Any]:
    from model_coins.infer import CoinSession

    gate, classifier = get_native_coin_components()
    if classifier is None:
        return {"status": "error", "message": "Engine model koin tidak dapat dimuat di server."}

    now = time.time()
    with _coin_session_lock:
        stale = [sid for sid, d in _coin_active_sessions.items() if now - d["last_active"] > COIN_SESSION_TTL]
        for sid in stale:
            del _coin_active_sessions[sid]

        if session_id not in _coin_active_sessions:
            _coin_active_sessions[session_id] = {
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
        sdata = _coin_active_sessions[session_id]
        sdata["last_active"] = now
        session = sdata["session"]

    # 1. Quality Gate Filter
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

    # 2. Fast Inference TFLite
    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    with _coin_tflite_lock:
        p5 = classifier._predict_single_rot(pil_crop).copy()

    # 3. Akumulator Bukti Sesi Konsensus
    res = session.update(p5, crop)
    status = res.get("status", "lanjut")

    resp = {
        "status": status,
        "message": res.get("audio", "")
    }

    if status in ("jawab", "menyerah"):
        if status == "jawab":
            resp["label"] = str(res.get("decision", ""))
        with _coin_session_lock:
            if session_id in _coin_active_sessions:
                del _coin_active_sessions[session_id]

    return resp


# ---------------------------------------------------------------------------
# 1. ENDPOINT UANG KERTAS (TAG: Uang Kertas)
# ---------------------------------------------------------------------------

@app.post(
    "/api/kertas/prediksi",
    response_model=KertasPrediksiResponse,
    tags=["Uang Kertas"],
    summary="Klasifikasi Uang Kertas (File / Multi-Frame)",
    description="Mengunggah satu atau beberapa berkas citra uang kertas untuk diklasifikasikan ke dalam 7 nominal Rupiah. Mendukung model: 'consensus' (rekomendasi ensemble), 'cnn', 'mobilenet', atau 'fnn'."
)
async def prediksi_uang_kertas(
    files: Optional[List[UploadFile]] = File(None, description="Daftar berkas gambar uang kertas"),
    file: Optional[UploadFile] = File(None, description="Berkas gambar uang kertas tunggal"),
    model: str = Query("consensus", description="Pilihan model: 'consensus', 'cnn', 'mobilenet', 'fnn'"),
    ground_truth: Optional[str] = Query(None, description="Label nominal sebenarnya (opsional untuk audit uji lapangan)")
):
    upload_list: List[UploadFile] = []
    if files:
        upload_list.extend(files)
    if file:
        upload_list.append(file)

    if not upload_list:
        raise HTTPException(status_code=400, detail="Tidak ada berkas gambar uang kertas yang diunggah.")

    frames: List[Image.Image] = []
    for f in upload_list:
        content = await f.read()
        try:
            img = decode_image_bytes(content)
            frames.append(img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Format citra tidak valid: {e}")

    if not frames:
        raise HTTPException(status_code=400, detail="Gagal mendekode gambar.")

    target_model = model.lower()
    if target_model in ("consensus", "all", "compare"):
        result = model_manager.predict_consensus(frames)
    else:
        if target_model not in model_manager.models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{target_model}' tidak tersedia. Model aktif: {list(model_manager.models.keys())}",
            )
        result = model_manager.predict_multi_frame(target_model, frames)

    pred_class = result.get("predicted_class", "tidak_yakin")
    conf = result.get("confidence", result.get("average_confidence", 0.0))
    lat = result.get("latency_ms", result.get("total_latency_ms", 0.0))
    agreement = result.get("agreement_ratio", None)

    log_id = field_logger.log_prediction(
        model_used=target_model,
        predicted_class=pred_class,
        confidence=conf,
        latency_ms=lat,
        ground_truth=ground_truth,
        agreement=str(agreement) if agreement is not None else None,
    )
    result["log_id"] = log_id
    return result


@app.post(
    "/api/kertas/prediksi_json",
    response_model=KertasPrediksiResponse,
    tags=["Uang Kertas"],
    summary="Klasifikasi Uang Kertas (Base64 Canvas HP)",
    description="Mengirim payload JSON berisi array string citra Base64 (data:image/jpeg;base64,...) langsung dari canvas browser HP/kamera."
)
def prediksi_uang_kertas_json(req: KertasJsonRequest):
    if not req.images:
        raise HTTPException(status_code=400, detail="Daftar gambar Base64 kosong.")

    frames: List[Image.Image] = []
    for data_url in req.images:
        try:
            frames.append(decode_image_bytes(data_url))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Gagal mendekode Base64: {e}")

    target_model = req.model.lower()
    if target_model in ("consensus", "all", "compare"):
        result = model_manager.predict_consensus(frames)
    else:
        if target_model not in model_manager.models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{target_model}' tidak tersedia. Model aktif: {list(model_manager.models.keys())}",
            )
        result = model_manager.predict_multi_frame(target_model, frames)

    pred_class = result.get("predicted_class", "tidak_yakin")
    conf = result.get("confidence", result.get("average_confidence", 0.0))
    lat = result.get("latency_ms", result.get("total_latency_ms", 0.0))
    agreement = result.get("agreement_ratio", None)

    log_id = field_logger.log_prediction(
        model_used=target_model,
        predicted_class=pred_class,
        confidence=conf,
        latency_ms=lat,
        ground_truth=req.ground_truth,
        agreement=str(agreement) if agreement is not None else None,
    )
    result["log_id"] = log_id
    return result


# Alias backward-compatibility untuk frontend lama (tidak ditampilkan di Swagger)
@app.post("/api/banknote/predict", include_in_schema=False)
async def alias_banknote_predict(
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
    model: str = Query("consensus"),
    ground_truth: Optional[str] = Query(None)
):
    return await prediksi_uang_kertas(files=files, file=file, model=model, ground_truth=ground_truth)


@app.post("/api/banknote/predict_json", include_in_schema=False)
def alias_banknote_predict_json(req: KertasJsonRequest):
    return prediksi_uang_kertas_json(req)


# ---------------------------------------------------------------------------
# 2. ENDPOINT UANG KOIN (TAG: Uang Koin)
# ---------------------------------------------------------------------------

@app.post(
    "/api/koin/prediksi",
    response_model=KoinPrediksiResponse,
    tags=["Uang Koin"],
    summary="Klasifikasi Uang Koin Instan (File Citra)",
    description="Mengunggah citra koin tunggal untuk mendeteksi nominal koin (`100`, `200`, `500`, `1000`) atau `sisi_garuda` (lambang pahlawan/Garuda) menggunakan model TFLite FP16 teroptimasi."
)
async def prediksi_uang_koin_instan(
    file: Optional[UploadFile] = File(None, description="Berkas gambar koin"),
    files: Optional[List[UploadFile]] = File(None, description="Daftar berkas gambar koin")
):
    upload_list: List[UploadFile] = []
    if files:
        upload_list.extend(files)
    if file:
        upload_list.append(file)
    if not upload_list:
        raise HTTPException(status_code=400, detail="Tidak ada berkas citra koin yang diunggah.")

    gate, classifier = get_native_coin_components()
    if classifier is None:
        raise HTTPException(status_code=500, detail="Engine model koin tidak dapat dimuat di server.")

    content = await upload_list[0].read()
    nparr = np.frombuffer(content, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Citra koin tidak dapat dibaca.")

    passed, reason, crop = gate.check(bgr)
    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) if (passed and crop is not None) else Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    t0 = time.time()
    with _coin_tflite_lock:
        probs = classifier._predict_single_rot(pil_crop)
    lat = round((time.time() - t0) * 1000, 1)

    pred_idx = int(np.argmax(probs))
    pred_class = COIN_CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    is_garuda = (pred_class == 'sisi_garuda')
    side_type = "Sisi Balik (Garuda/Pahlawan)" if is_garuda else "Sisi Nominal (Angka Koin)"

    return {
        "predicted_class": pred_class,
        "predicted_nominal": COIN_PRETTY_NOMINAL.get(pred_class, pred_class),
        "spoken_text": COIN_SPOKEN_NOMINAL.get(pred_class, pred_class),
        "side_type": side_type,
        "confidence": round(confidence, 4),
        "quality_gate_passed": passed,
        "quality_gate_reason": reason,
        "latency_ms": lat,
        "all_probabilities": {COIN_CLASSES[i]: round(float(probs[i]), 4) for i in range(len(COIN_CLASSES))}
    }


@app.post(
    "/api/koin/sesi",
    response_model=KoinSesiResponse,
    tags=["Uang Koin"],
    summary="Sesi Interaktif Koin Multi-Frame (Audio & Konsensus Tunanetra)",
    description="Endpoint sesi kontinu untuk mendeteksi koin secara dinamis dari aliran kamera. Menerapkan Frame Quality Gate (filter blur/kilau) dan akumulasi bukti probabilitas hingga tercapai konsensus stabil."
)
async def sesi_interaktif_koin(
    session_id: str = Form(..., description="UUID unik per sesi pemindaian koin", example="sess_12345"),
    frame: UploadFile = File(..., description="Frame kamera JPEG berurutan")
):
    contents = await frame.read()
    nparr = np.frombuffer(contents, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse(status_code=400, content={"status": "gate_failed", "message": "Frame kamera rusak."})

    res = await run_in_threadpool(_sync_step_coin_session, session_id, bgr)
    return JSONResponse(content=res)


# Alias untuk coin-reader.js frontend
@app.api_route("/predict_coin_session", methods=["GET", "POST"], include_in_schema=False)
async def alias_predict_coin_session(request: Request):
    if request.method == "GET":
        gate, classifier = get_native_coin_components()
        return {
            "status": "ok",
            "service": "rupiah-coin-reader-unified",
            "native_engine_ready": classifier is not None,
            "active_sessions": len(_coin_active_sessions)
        }

    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)

    # 1. Coba proxy ke port 8000 bila aktif
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.request(
                method=request.method,
                url=f"{COIN_SERVER_URL}/predict_coin_session",
                params=request.query_params,
                content=body,
                headers=headers,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )
    except Exception:
        pass

    # 2. Fallback native
    form = await request.form()
    session_id = str(form.get("session_id", "sess_default"))
    frame_file = form.get("frame")
    if not frame_file:
        return JSONResponse(status_code=400, content={"status": "gate_failed", "message": "Tidak ada data frame."})

    contents = await frame_file.read()
    nparr = np.frombuffer(contents, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse(status_code=400, content={"status": "gate_failed", "message": "Frame kamera rusak."})

    res = await run_in_threadpool(_sync_step_coin_session, session_id, bgr)
    return JSONResponse(content=res)


# ---------------------------------------------------------------------------
# 3. ENDPOINT UJI LAPANGAN & MLOPS (TAG: Uji Lapangan & MLOps)
# ---------------------------------------------------------------------------

@app.post(
    "/api/lapangan/catat_label",
    tags=["Uji Lapangan & MLOps"],
    summary="Catat Label Asli (Ground Truth) Uji Lapangan",
    description="Memperbarui label nominal sebenarnya pada riwayat log inferensi ketika pengguna/penguji menekan tombol konfirmasi nominal."
)
def catat_label_lapangan(req: CatatLabelRequest):
    success = field_logger.update_ground_truth(req.log_id, req.ground_truth)
    if not success:
        raise HTTPException(status_code=404, detail=f"Log ID {req.log_id} tidak ditemukan.")
    return {"status": "success", "log_id": req.log_id, "ground_truth": req.ground_truth}


@app.get(
    "/api/lapangan/laporan",
    tags=["Uji Lapangan & MLOps"],
    summary="Lihat Rekap Akurasi Uji Lapangan",
    description="Mengambil ringkasan metrik performa uji lapangan: total pengujian, akurasi berjalan, distribusi per model, dan tingkat persetujuan konsensus."
)
def laporan_uji_lapangan():
    return field_logger.get_summary_report()


@app.get(
    "/api/lapangan/ekspor_csv",
    tags=["Uji Lapangan & MLOps"],
    summary="Unduh Database Uji Lapangan (CSV)",
    description="Mengekspor seluruh riwayat pengujian lapangan beserta ground truth ke dalam berkas CSV untuk analisis statistik."
)
def ekspor_csv_lapangan():
    csv_data = field_logger.export_csv()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=rupiah_field_logs_{int(time.time())}.csv"},
    )


@app.post(
    "/api/lapangan/feedback",
    tags=["Uji Lapangan & MLOps"],
    summary="Kirim Umpan Balik Pengguna & Rating",
    description="Mencatat masukan subjektif, rating kemudahan (1-5), dan catatan evaluasi pengguna. Citra kamera hanya disimpan jika opsi simpan dicentang."
)
async def kirim_feedback(
    file: Optional[UploadFile] = File(None, description="Citra opsional jika pengguna mengizinkan penyimpanan"),
    log_id: Optional[int] = Form(None, description="ID log pengujian terkait"),
    user_note: str = Form("", description="Catatan atau masukan dari pengguna"),
    rating: Optional[int] = Form(None, description="Skala kepuasan 1 hingga 5"),
    model_used: Optional[str] = Form(None, description="Model yang digunakan saat pengujian"),
    save_image: bool = Form(False, description="Persetujuan penyimpanan citra (Privasi Terjamin)")
):
    saved_img_name = None
    if save_image and file:
        FEEDBACK_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_img_name = f"feedback_{timestamp_str}_{file.filename}"
        img_path = FEEDBACK_IMAGES_DIR / saved_img_name
        content = await file.read()
        with open(img_path, "wb") as f_out:
            f_out.write(content)

    feedback_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "log_id": log_id,
        "rating": rating,
        "user_note": user_note,
        "model_used": model_used,
        "saved_image": saved_img_name,
    }
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_DIR / "feedback.jsonl", "a", encoding="utf-8") as f_fb:
        f_fb.write(json.dumps(feedback_entry) + "\n")

    return {"status": "success", "message": "Umpan balik berhasil dicatat."}


# Aliases for backward compatibility
@app.post("/api/field/update_ground_truth", include_in_schema=False)
def alias_update_ground_truth(req: CatatLabelRequest):
    return catat_label_lapangan(req)

@app.get("/api/field/report", include_in_schema=False)
def alias_get_field_report():
    return laporan_uji_lapangan()

@app.get("/api/field/export", include_in_schema=False)
def alias_export_field_logs():
    return ekspor_csv_lapangan()

@app.post("/api/feedback", include_in_schema=False)
async def alias_submit_feedback(
    file: Optional[UploadFile] = File(None),
    log_id: Optional[int] = Form(None),
    user_note: str = Form(""),
    rating: Optional[int] = Form(None),
    model_used: Optional[str] = Form(None),
    save_image: bool = Form(False),
):
    return await kirim_feedback(file=file, log_id=log_id, user_note=user_note, rating=rating, model_used=model_used, save_image=save_image)


# ---------------------------------------------------------------------------
# 4. ENDPOINT SISTEM & MONITORING (TAG: Sistem & Status Model)
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    tags=["Sistem & Status Model"],
    summary="Cek Status Kesehatan Layanan & Engine",
    description="Memeriksa status aktif server, uptime, daftar model uang kertas aktif, serta kesiapan engine model koin."
)
def health_check():
    uptime_sec = round(time.time() - START_TIME, 1)
    gate, classifier = get_native_coin_components()
    return {
        "status": "healthy",
        "uptime_seconds": uptime_sec,
        "model_kertas_aktif": list(model_manager.models.keys()),
        "total_model_kertas": len(model_manager.models),
        "engine_koin_status": "native_aktif" if classifier is not None else "proxy_fallback",
        "kelas_koin": COIN_CLASSES,
        "kelas_kertas": CLASSES,
    }


@app.get(
    "/models",
    tags=["Sistem & Status Model"],
    summary="Katalog Seluruh Model Deep Learning (Kertas & Koin)",
    description="Menampilkan rincian seluruh model Deep Learning yang tersedia, threshold keyakinan, sumber notebook pelatihan, dan status kesiapannya."
)
def get_model_catalog():
    gate, classifier = get_native_coin_components()
    coin_models = [
        {
            "id": "coin_cnn_5class",
            "name": "Custom CNN / MobileNetV2 Coin (5-Class TFLite FP16)",
            "classes": COIN_CLASSES,
            "class_display_names": COIN_PRETTY_NOMINAL,
            "loaded": classifier is not None,
            "status": "tersedia" if classifier is not None else "tidak_tersedia",
            "threshold": 0.75,
            "description": "Model koin 5-kelas teroptimasi FP16 dengan Frame Quality Gate dan pencegahan kebocoran dHash."
        },
        {
            "id": "coin_googlenet",
            "name": "GoogLeNet Inception Coin (5-Class Multi-Scale)",
            "classes": COIN_CLASSES,
            "class_display_names": COIN_PRETTY_NOMINAL,
            "notebook": "Coin_GoogLeNet.ipynb",
            "status": "arsitektur_tersedia",
            "description": "Arsitektur multi-skala Inception (1x1, 3x3, 5x5 paralel) untuk menangkap relief mikro angka dan batas lingkaran koin."
        }
    ]
    return {
        "model_uang_kertas": model_manager.get_models_status(),
        "model_uang_koin": coin_models,
        "kelas_uang_kertas": CLASSES,
        "tampilan_uang_kertas": CLASS_DISPLAY_NAMES,
        "kelas_uang_koin": COIN_CLASSES,
        "tampilan_uang_koin": COIN_PRETTY_NOMINAL,
    }


@app.get("/api/sistem/health", tags=["Sistem & Status Model"], include_in_schema=False)
def alias_health():
    return health_check()

@app.get("/api/sistem/models", tags=["Sistem & Status Model"], include_in_schema=False)
def alias_models():
    return get_model_catalog()


# ---------------------------------------------------------------------------
# Static UI Mount (Aplikasi Web Tunanetra)
# ---------------------------------------------------------------------------
if RUPIAH_READER_DIR.exists():
    app.mount("/", StaticFiles(directory=str(RUPIAH_READER_DIR), html=True), name="static_reader")
