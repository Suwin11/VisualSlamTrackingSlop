@echo off
title 8 - Auto-Label Walls and Obstacles
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\activate.bat" call "%~dp0.venv\Scripts\activate.bat"
echo ========================================================
echo   FIBO AI: Auto-Label Dataset for Walls and Obstacles
echo   Generating YOLO label files (.txt) and dataset.yaml...
echo ========================================================
python auto_label_obstacles.py --dataset dataset --preview
pause
