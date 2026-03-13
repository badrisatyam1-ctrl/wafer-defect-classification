@echo off
title Wafer Defect Classifier
cd /d "%~dp0"
echo Starting Wafer Defect Classifier...
echo.
echo Once ready, open your browser at:
echo   http://127.0.0.1:8501
echo.
echo Press Ctrl+C to stop the server.
echo.
streamlit run deployment/streamlit_app.py --server.address=127.0.0.1 --server.port=8501
pause
