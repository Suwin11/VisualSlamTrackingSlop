@echo off
title Deploy Code to Raspberry Pi
cd /d "%~dp0"
python deploy_to_pi.py
pause
