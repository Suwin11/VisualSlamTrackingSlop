@echo off
title FIBO Map Creator - Interactive Floorplan Editor
cd /d "%~dp0"
echo ====================================================================
echo  FIBO MAP CREATOR: Interactive 2D Floorplan & Wall Editor
echo ====================================================================
echo  * Use mouse to draw walls, boxes, pillars, doorways, or robot start.
echo  * Press 'S' to save map directly to saved_maps/
echo  * Once saved, run: python navigate_rrt.py --load-map latest
echo ====================================================================
python map_creator.py %*
pause
