# AI Workstation (PC) Navigation & Perception Suite

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
