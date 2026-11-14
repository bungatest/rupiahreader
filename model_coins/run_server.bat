@echo off
chcp 65001 >nul
title Rupiah Reader - FastAPI Server (Port 8000)

echo ===========================================================================
echo MENJALANKAN SERVER RUPIAH COIN READER (FASTAPI + TFLITE)
echo ===========================================================================
echo.

:: Pindah ke direktori skrip ini berada
cd /d "%~dp0"

:: Set variabel lingkungan untuk performa optimal dan bebas thread lock
set MPLCONFIGDIR=%TEMP%\mpl_cache
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=3
set PYTHONIOENCODING=utf-8

:: Periksa apakah Python terpasang
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python tidak ditemukan di PATH. Pastikan Python 3.10+ terpasang!
    echo.
    pause
    exit /b 1
)

echo [INFO] Menyajikan Frontend di : http://127.0.0.1:8000/
echo [INFO] Endpoint Prediksi Sesi : http://127.0.0.1:8000/predict_coin_session
echo [INFO] Worker: 1 (In-Memory Session Coherence)
echo [INFO] Binding: 0.0.0.0 (Tersedia untuk Localhost & Jaringan Wi-Fi Lokal)
echo.
echo Tekan CTRL+C untuk menghentikan server kapan saja.
echo ---------------------------------------------------------------------------
echo.

python -m uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Server berhenti dengan kode error: %errorlevel%
    pause
)

