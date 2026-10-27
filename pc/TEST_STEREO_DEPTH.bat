@echo off
title FIBO - Stereo Camera Metric Depth Perception Test
cd /d "%~dp0"
echo ======================================================================
echo  FIBO Stereo Depth Perception Viewer (Left + Right + Metric Depth Map)
echo  Press 'S' to Swap Left/Right | 'I'/'K' to Nudge Vertical Alignment
echo ======================================================================
python stereo_visual_odometry.py --test-depth
pause
