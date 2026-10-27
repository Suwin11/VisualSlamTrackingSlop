@echo off
title YOLO Mask Tracking Webcam (Headless - Port 8080)
cd /d "%~dp0.."
echo ========================================================
echo Running Webcam Mask Tracking without GUI window
echo Stream available at http://localhost:8080
echo ========================================================
python yolo_mask_track.py --source 0 --model yolo11n-seg.pt --no-display --web-port 8080
pause
