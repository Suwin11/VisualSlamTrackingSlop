#!/usr/bin/env python3
"""
reorganize_workspace.py
Reorganizes the repository into dedicated 'pi' and 'pc' subdirectories.
"""

import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))

PI_FILES = [
    "pi_stereo_capture.py",
    "pi_imu_streamer.py",
    "pi_bt_motor_bridge.py",
    "pi_mpu6500_streamer.py",
    "bno055_broadcaster.py",
    "ngamove_bt.py",
    "detect_cameras.py",
    "set_camera_exposure.py",
    "mediamtx_pi.yml",
]

PI_DIRS = [
    "esp32_compass",
    "espinoasd",
]

PC_FILES = [
    "navigate_rrt.py",
    "map_creator.py",
    "save_map.py",
    "stereo_visual_odometry.py",
    "visual_odometry.py",
    "capture.py",
    "imu_receiver.py",
    "detectors.py",
    "pointcloud_3d.py",
    "global_map.py",
    "occupancy.py",
    "rrt_star.py",
    "astar.py",
    "motor_control.py",
    "streamer.py",
    "config_manager.py",
    "mapping_config.json",
    "camera_pi.npz",
    "stereo_calibrate.py",
    "yolo_mask_track.py",
    "START.bat",
    "1_TRACK_PI_STREAM.bat",
    "2_TRACK_WEBCAM.bat",
    "3_NAVIGATE_RRT_WEBCAM.bat",
    "4_NAVIGATE_RRT_PI_STREAM.bat",
    "5-BluetoothStart.bat",
    "6_MAP_CREATOR.bat",
    "7_COLLECT_TRAINING_DATA.bat",
    "8_AUTO_LABEL_DATASET.bat",
    "9_TRAIN_YOLO_OBSTACLES.bat",
    "collect_training_data.py",
    "auto_label_obstacles.py",
    "train_yolo_obstacles.py",
    "deploy_to_pi.bat",
    "TEST_STEREO_DEPTH.bat",
]

PC_DIRS = [
    "models",
    "saved_maps",
]


def reorganize():
    pi_dir = os.path.join(ROOT, "pi")
    pc_dir = os.path.join(ROOT, "pc")
    os.makedirs(pi_dir, exist_ok=True)
    os.makedirs(pc_dir, exist_ok=True)

    print("Copying Raspberry Pi & Arduino files to pi/...")
    for f in PI_FILES:
        src = os.path.join(ROOT, f)
        dst = os.path.join(pi_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  -> Copied {f} to pi/")

    for d in PI_DIRS:
        src = os.path.join(ROOT, d)
        dst = os.path.join(pi_dir, d)
        if os.path.exists(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  -> Copied {d}/ to pi/")

    print("\nCopying AI Workstation / PC files to pc/...")
    for f in PC_FILES:
        src = os.path.join(ROOT, f)
        dst = os.path.join(pc_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  -> Copied {f} to pc/")

    for d in PC_DIRS:
        src = os.path.join(ROOT, d)
        dst = os.path.join(pc_dir, d)
        if os.path.exists(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  -> Copied {d}/ to pc/")

    print("\nCreating documentation and helper scripts...")

    # Write pi/README.md
    with open(os.path.join(pi_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("""# Raspberry Pi 4 Daemon & Sensor Suite

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
""")

    # Write pc/README.md
    with open(os.path.join(pc_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("""# AI Workstation (PC) Navigation & Perception Suite

This folder contains all high-performance GPU neural inference, 3D metric perception,
Visual-Inertial-Laser odometry, Bayesian log-odds mapping, RRT* path planning, and GUI tools.

## Core Modules

| Module | Description |
| :--- | :--- |
| `navigate_rrt.py` | Master Application: 4-Quadrant composite HUD, YOLO segmentation, RRT* planning, and teleoperation |
| `map_creator.py` | Interactive Graphical 2D Map Creator & Floorplan Editor (draw walls, doors, pillars) |
| `save_map.py` | Map inspection and export utility |
| `stereo_visual_odometry.py` | 100% BNO055 Rotation + Tri-Modal Motion Fusion (BNO055 + Camera Depth + Commanded Motion) |
| `visual_odometry.py` | Monocular Visual-Inertial Odometry fallback |
| `global_map.py` | Global Bayesian Log-Odds Occupancy Grid with Save/Load and Relocalization support |
| `rrt_star.py` | Asymptotically optimal RRT* path planner with Cubic Bézier smoothing |
| `motor_control.py` | Pure pursuit path tracker, differential kinematics, and 25Hz motor command streamer |
| `capture.py` | Zero-latency TCP :8555 length-prefixed MJPEG frame ingestion |
| `imu_receiver.py` | 100 Hz UDP :5005 BNO055 + ToF receiver with timestamp interpolation and tilt compensation |
| `pointcloud_3d.py` | Metric stereo backprojection, height-band slicing, and 3D voxel viewer |
| `detectors.py` | Ultralytics YOLOv11 instance segmentation and ORB optical flow tracker |
| `config_manager.py` | Zero-hardcoded IP discovery and JSON configuration manager |

## Launching

* **Master Autonomous Navigation:**
  ```powershell
  python navigate_rrt.py --source pi
  ```
* **Interactive Map Creator:**
  ```powershell
  python map_creator.py
  ```
* **Navigation with a Custom Pre-Saved Map:**
  ```powershell
  python navigate_rrt.py --source pi --load-map latest
  ```
""")

    print("\nReorganization complete!")


if __name__ == "__main__":
    reorganize()
