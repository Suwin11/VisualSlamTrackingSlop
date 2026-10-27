@echo off
title YOLO Mask Tracking (AI PC MediaMTX Relay)
cd /d "%~dp0.."
echo ========================================================
echo Connecting via AI PC MediaMTX Relay (rtsp://10.61.24.75:8554/cam)...
echo ========================================================
python yolo_mask_track.py --source rtsp://10.61.24.75:8554/cam --model yolo11n-seg.pt --web-port 8080
pause
