@echo off
title YOLO Mask Tracking (Laptop Webcam)
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\activate.bat" call "%~dp0.venv\Scripts\activate.bat"
echo ========================================================
echo   Starting YOLO Mask Tracking on Laptop Webcam
echo   Device: 0
echo   Web Stream: http://localhost:8080
echo ========================================================
python yolo_mask_track.py --source 0 --web-port 8080
pause
