@echo off
title 7 - Collect Custom YOLO Training Data (Robot Camera)
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\activate.bat" call "%~dp0.venv\Scripts\activate.bat"
echo ========================================================
echo   FIBO AI: Collect Dataset for Wall and Obstacle Detection
echo   Drive your robot around the room while capturing images.
echo   [SPACE] / [C] : Snapshot current frame
echo   [A]           : Toggle Auto-Capture while driving
echo   [Q]           : Finish and exit
echo ========================================================
python collect_training_data.py --source pi
pause
