@echo off
if exist "%~dp0.venv\Scripts\activate.bat" call "%~dp0.venv\Scripts\activate.bat"
:MENU
cls
cd /d "%~dp0"
for /f "delims=" %%i in ('python -c "import config_manager; print(config_manager.resolve_pi_host(save_if_changed=True))" 2^>nul') do set PI_IP=%%i
if "%PI_IP%"=="" set PI_IP=auto
echo ==============================================================
echo       FIBO YOLO MASK ^& AUTONOMOUS NAVIGATION SYSTEM
echo ==============================================================
echo   Active Pi IP: %PI_IP%  (Auto-Discovered / mapping_config.json)
echo ==============================================================
echo.
echo   [1] Track Raspberry Pi Camera (Direct RTSP - Auto IP)
echo   [2] Track Laptop Webcam
echo   [3] Open Live Camera in Browser (WebRTC Feed)
echo   [4] SSH to Raspberry Pi (admin@%PI_IP%)
echo   [5] SSH to AI PC (Terminal)
echo   [6] Set / Override Raspberry Pi IP in mapping_config.json
echo   [7] Navigate RRT* with Webcam (Simulation - Pi Offline)
echo   [8] Navigate RRT* with Raspberry Pi Robot Stream (Auto IP)
echo   [9] Interactive 2D Map Creator (Draw Custom Floorplan)
echo   [10] Deploy / Sync Code to Raspberry Pi (pi/ -> /home/admin/robot/)
echo   [0] Exit
echo.
echo ==============================================================
set /p choice="Select an option [0-10]: "

if "%choice%"=="1" goto TRACK_PI
if "%choice%"=="2" goto TRACK_WEBCAM
if "%choice%"=="3" goto VIEW_BROWSER
if "%choice%"=="4" goto SSH_PI
if "%choice%"=="5" goto SSH_PC
if "%choice%"=="6" goto SET_IP
if "%choice%"=="7" goto NAVIGATE_WEBCAM
if "%choice%"=="8" goto NAVIGATE_PI
if "%choice%"=="9" goto MAP_CREATOR
if "%choice%"=="10" goto DEPLOY_PI
if "%choice%"=="0" exit
goto MENU

:TRACK_PI
cls
call "%~dp01_TRACK_PI_STREAM.bat"
goto MENU

:TRACK_WEBCAM
cls
call "%~dp02_TRACK_WEBCAM.bat"
goto MENU

:VIEW_BROWSER
cls
echo Opening live camera feed at http://%PI_IP%:8889/cam ...
start http://%PI_IP%:8889/cam
timeout /t 2 >nul
goto MENU

:SSH_PI
cls
echo Connecting to Raspberry Pi (admin@%PI_IP%)...
ssh admin@%PI_IP%
pause
goto MENU

:SSH_PC
cls
echo Connecting to AI PC...
ssh pc
pause
goto MENU

:SET_IP
cls
set /p new_ip="Enter new Raspberry Pi IP (or 'auto' for subnet scan): "
python -c "import config_manager; cfg=config_manager.load_config(); cfg['network']['pi_host']='%new_ip%'; cfg['network']['last_known_pi_ip']='%new_ip%' if '%new_ip%'!='auto' else cfg['network']['last_known_pi_ip']; config_manager.save_config(cfg); print('Saved Pi IP:', '%new_ip%')"
timeout /t 2 >nul
goto MENU

:NAVIGATE_WEBCAM
cls
call "%~dp03_NAVIGATE_RRT_WEBCAM.bat"
goto MENU

:NAVIGATE_PI
cls
call "%~dp04_NAVIGATE_RRT_PI_STREAM.bat"
goto MENU

:MAP_CREATOR
cls
call "%~dp0MAP_CREATOR.bat"
goto MENU

:DEPLOY_PI
cls
call "%~dp0deploy_to_pi.bat"
goto MENU
