@echo off
chcp 65001 >nul
title Rupiah Reader - Unified Deep Learning Server (Port 8001)

echo ===========================================================================
echo       RUPIAH READER - UNIFIED DEEP LEARNING SERVING API
echo       (Klasifikasi Terpadu Uang Kertas 7 Kelas ^& Uang Koin 5 Kelas)
echo ===========================================================================
echo.

cd /d "%~dp0"

set MPLCONFIGDIR=%TEMP%\mpl_cache
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2
set PYTHONIOENCODING=utf-8

echo [INFO] Membuka Frontend di : http://localhost:8001/
echo [INFO] Endpoint Uang Kertas: http://localhost:8001/api/kertas/prediksi
echo [INFO] Endpoint Uang Koin  : http://localhost:8001/api/koin/prediksi
echo [INFO] Endpoint Sesi Koin  : http://localhost:8001/api/koin/sesi
echo [INFO] Dokumentasi Swagger : http://localhost:8001/docs
echo.
echo Tekan CTRL+C untuk menghentikan server kapan saja.
echo ---------------------------------------------------------------------------
echo.

python -m uvicorn serving.server:app --host 0.0.0.0 --port 8001 --workers 1

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Server berhenti dengan kode: %errorlevel%
    pause
)
