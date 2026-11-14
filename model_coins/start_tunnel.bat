@echo off
chcp 65001 >nul
title Rupiah Reader - HTTPS Tunnel untuk HP

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

:: Periksa apakah Python terpasang
python --version >nul 2>&1
if %errorlevel% equ 0 (
    python tunnel_launcher.py
    exit /b %errorlevel%
)

:: Fallback jika python tidak terdeteksi di PATH
echo [INFO] Menjalankan tunnel via skrip batch fallback...
if not exist "cloudflared.exe" (
    echo [INFO] Mengunduh cloudflared.exe resmi (otomatis)...
    powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe', 'cloudflared.exe')"
)

if exist "cloudflared.exe" (
    echo [INFO] Membuka tunnel ke http://127.0.0.1:8000 ...
    cloudflared.exe tunnel --url http://127.0.0.1:8000
) else (
    echo [ERROR] Gagal mengunduh cloudflared.exe secara otomatis.
    pause
)
