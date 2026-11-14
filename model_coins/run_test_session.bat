@echo off
chcp 65001 >nul
title Rupiah Reader - Uji Sesi Multi-Frame (Client Tester)

echo ===========================================================================
echo UJI SIMULASI SESI MULTI-FRAME & PENGUKURAN LATENSI (MS)
echo ===========================================================================
echo.
echo Pastikan server lokal sudah berjalan di 'run_server.bat'!
echo.

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

:MENU
echo Pilih skenario pengujian:
echo [1] Uji Koin Rp 1.000 Sisi Angka (dataset coin\1000\image copy 10.png)
echo [2] Uji Koin Rp 500 Sisi Angka   (dataset coin\500\image copy 10.png)
echo [3] Uji Koin Sisi Garuda / Flip  (dataset coin\100\image copy 11.png)
echo [4] Masukkan path gambar koin custom
echo [5] Keluar
echo.

set /p PILIHAN="Masukkan pilihan [1-5]: "

if "%PILIHAN%"=="1" (
    echo.
    echo Menjalankan uji pada Koin Rp 1.000...
    python test_session.py "..\dataset coin\1000\image copy 10.png"
    echo.
    pause
    cls
    goto MENU
)

if "%PILIHAN%"=="2" (
    echo.
    echo Menjalankan uji pada Koin Rp 500...
    python test_session.py "..\dataset coin\500\image copy 10.png"
    echo.
    pause
    cls
    goto MENU
)

if "%PILIHAN%"=="3" (
    echo.
    echo Menjalankan uji pada Sisi Garuda...
    python test_session.py "..\dataset coin\100\image copy 11.png"
    echo.
    pause
    cls
    goto MENU
)

if "%PILIHAN%"=="4" (
    echo.
    set /p CUSTOM_PATH="Masukkan path lengkap gambar koin: "
    echo Menjalankan uji pada: %CUSTOM_PATH%
    python test_session.py "%CUSTOM_PATH%"
    echo.
    pause
    cls
    goto MENU
)

if "%PILIHAN%"=="5" (
    exit /b 0
)

echo Pilihan tidak valid!
timeout /t 2 >nul
cls
goto MENU

