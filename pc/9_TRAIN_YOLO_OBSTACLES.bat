@echo off
title 9 - Train Custom YOLO on NVIDIA GPU
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\activate.bat" call "%~dp0.venv\Scripts\activate.bat"
echo ========================================================
echo   FIBO AI: Train Custom YOLO on NVIDIA GeForce GPU
echo   Fine-tuning wall & obstacle detection...
echo ========================================================
python train_yolo_obstacles.py --epochs 35 --batch 16
pause
