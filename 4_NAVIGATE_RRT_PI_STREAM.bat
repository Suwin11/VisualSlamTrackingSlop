@echo off
title Autonomous Navigation RRT* (Raspberry Pi Robot RTSP)
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\activate.bat" call "%~dp0.venv\Scripts\activate.bat"
echo ========================================================
echo   FIBO AI Navigation: Visual SLAM + Global Grid + RRT*
echo   Connecting to Raspberry Pi 4 Robot Stream (Auto-detecting relay/direct)...
echo   Controls: Press 'r' to reset map, 'q' to quit
echo ========================================================
python navigate_rrt.py --source pi --web-port 8080
pause
