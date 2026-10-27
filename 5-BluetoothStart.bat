@echo off
title Robot Bluetooth Motor Bridge (ESP32)
echo ========================================================
echo   Connecting to Raspberry Pi (192.168.0.170)...
echo   Starting Bluetooth Motor Bridge for ESPino32...
echo ========================================================

ssh -tt admin@192.168.0.170 "cd ~/robot && sudo rfkill unblock bluetooth && sudo hciconfig hci0 up && sudo rfcomm release /dev/rfcomm0 2>/dev/null || true && sudo rfcomm bind /dev/rfcomm0 E0:8C:FE:C9:47:1A 1 && sudo chmod 666 /dev/rfcomm0 && python3 ngamove_bt.py --test"

pause
