@echo off
cd /d "%~dp0.."
for /f "delims=" %%i in ('python -c "import config_manager; print(config_manager.resolve_pi_host(save_if_changed=True))" 2^>nul') do set PI_IP=%%i
title SSH to Raspberry Pi (%PI_IP%)
ssh admin@%PI_IP%
pause
