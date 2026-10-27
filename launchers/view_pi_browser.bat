@echo off
cd /d "%~dp0.."
for /f "delims=" %%i in ('python -c "import config_manager; print(config_manager.resolve_pi_host(save_if_changed=True))" 2^>nul') do set PI_IP=%%i
start http://%PI_IP%:8889/cam
