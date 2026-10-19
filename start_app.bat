@echo off
title Wafer Yield Analytics Studio
cd /d "%~dp0"
echo =========================================================
echo    Wafer Yield Analytics Studio - ResNet-18 Production
echo =========================================================
echo.
echo Starting Web Server on http://127.0.0.1:8000 ...
echo.
start "" powershell -WindowStyle Hidden -NoProfile -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:8000'"
python -m uvicorn deployment.server:app --host 127.0.0.1 --port 8000
pause
