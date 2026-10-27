# Raspberry Pi 4 Daemon & Sensor Suite

This folder contains all code and services running directly on the Raspberry Pi 4 and ESP32 motor microcontroller.

## Files & Daemons

| File | Purpose | Network / Port |
| :--- | :--- | :--- |
| `pi_stereo_capture.py` | Synchronized dual Logitech C930e V4L2 capture -> Low-latency length-prefixed JPEG | TCP :8555 |
| `pi_imu_streamer.py` | 100 Hz Bosch BNO055 9-DOF IMU + ST VL53L1X TOF400C Laser Ranging streamer | UDP / TCP :5005 |
| `pi_bt_motor_bridge.py` | Bluetooth SPP bridge receiving PC velocity commands -> ESP32 motor controller | UDP :5006 |
| `ngamove_bt.py` | Direct interactive CLI WASD motor drive tester for ESP32 | Bluetooth SPP |
| `detect_cameras.py` | Diagnostic tool to identify Logitech C930e camera USB device nodes (`/dev/videoX`) | Local CLI |
| `set_camera_exposure.py` | Sets manual exposure and locks white balance on the Logitech C930e cameras | Local CLI |
| `esp32_compass/` | ESP32 firmware for secondary compass / BNO055 telemetry | Arduino / C++ |
| `espinoasd/` | ESPino32 / ESP32 dual Dynamixel motor controller firmware | Arduino / C++ |

## Deployment to Raspberry Pi

To sync all Pi files to the Raspberry Pi over SSH with one command:
```powershell
python deploy_to_pi.py
```
Or manually via SCP:
```bash
scp -r * admin@192.168.0.170:/home/admin/robot/
```
