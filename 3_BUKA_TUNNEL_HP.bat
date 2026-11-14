@echo off
chcp 65001 >nul
title Rupiah Reader - HTTPS Tunnel untuk HP

cd /d "%~dp0model_coins"
call start_tunnel.bat
