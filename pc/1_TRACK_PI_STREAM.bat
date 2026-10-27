@echo off
title YOLO Mask Tracking (Raspberry Pi Robot Camera)
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\activate.bat" call "%~dp0.venv\Scripts\activate.bat"
echo ========================================================
echo   Connecting to Raspberry Pi Camera Stream (Auto-detecting relay/direct)...
echo ========================================================
python yolo_mask_track.py --source pi --web-port 8080
pause
