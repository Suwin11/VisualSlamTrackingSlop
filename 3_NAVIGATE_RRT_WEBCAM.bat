@echo off
title Autonomous Navigation RRT* (Webcam Simulation)
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\activate.bat" call "%~dp0.venv\Scripts\activate.bat"
echo ========================================================
echo   FIBO AI Navigation: Visual SLAM + Global Grid + RRT*
echo   Testing with Laptop Webcam (Pi Offline Simulation)
echo   Web Stream: http://localhost:8080
echo   Controls: Press 'r' to reset map, 'q' to quit
echo ========================================================
python navigate_rrt.py --source 0 --web-port 8080
pause
