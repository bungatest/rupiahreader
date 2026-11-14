@echo off
chcp 65001 >nul
title Ekstraksi Dataset Video Koin Rupiah

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

python ekstrak_video_dataset.py

echo.
pause

