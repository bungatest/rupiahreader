@echo off
chcp 65001 >nul
title Retraining EfficientNetV2-B0 Koin Rupiah (5 Kelas)

cd /d "%~dp0model_coins"
set PYTHONIOENCODING=utf-8
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2
set MPLCONFIGDIR=%TEMP%\mpl_cache

echo ===========================================================================
echo MULAI RETRAINING EFFICIENTNETV2-B0 DENGAN 700+ FOTO ASLI HP
echo ===========================================================================
echo.

python train_5class_production.py

echo.
pause

