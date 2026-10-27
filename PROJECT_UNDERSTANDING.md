# Project Understanding & Visual SLAM Stream Optimization

## 1. System Architecture Overview
The **FIBO Reoptimized Tracking & Autonomous Navigation System** (`Reoptimizedtracking`) is a distributed, real-time Visual-Inertial-Laser SLAM and motion-planning stack for a differential-drive mobile robot:

- **Edge Sensor & Streaming Node (Raspberry Pi 4 @ `10.61.24.72`, `~/robot`)**:
  - **Video Stream (`mediamtx_pi.yml`)**: Uses MediaMTX's native `rpiCamera` source (`640x480 @ 30 FPS`, H.264 Baseline) to stream the CSI camera (`ov5647`) over RTSP (`:8554/cam`) and WebRTC (`:8889/cam`).
  - **100 Hz Telemetry Broadcaster (`pi_imu_streamer.py` / `pi_mpu6500_streamer.py` / `bno055_broadcaster.py`)**: Polls orientation & angular velocity from a Bosch BNO055 (via UART/I2C) or MPU-6500 (I2C) alongside forward laser range from a TOF400C / VL53L1X (`0x29`), streaming 10-field CSV packets over UDP/TCP port `5005`.
- **Host Perception, SLAM & Navigation Node (Laptop @ `10.61.24.45` / AI PC @ `10.61.24.75`)**:
  - **Acquisition (`capture.py`)**: Background thread draining RTSP/Webcam frames and stamping monotonic arrival times (`last_frame_mono`, `frame_dt`).
  - **Dynamic Object Masking (`yolo_mask_track.py`, `navigate_rrt.py`)**: Runs `yolo11n-seg.pt` with ByteTrack to segment dynamic objects and dilate an exclusion mask (`static_mask`) so moving obstacles never corrupt background optical flow.
  - **Optical Flow & Feature Tracking (`detectors.py`)**: Tracks ORB and Shi-Tomasi features using forward-backward Lucas-Kanade (`calcOpticalFlowPyrLK`) on prebuilt image pyramids.
  - **Visual-Inertial-Laser Odometry (`visual_odometry.py`, `stereo_visual_odometry.py` & `imu_receiver.py`)**: Time-aligns camera exposure with 100 Hz IMU yaw/pitch (`frame_mono - video_latency_sec`), maintains an **IMU-dominant motion engine** (`imu_motion_weight = 0.80`) with stationary ZUPT locking, de-couples rotational optical flow from forward translation, and fuses TOF400C laser range deltas.
  - **2.5D/3D Mapping & Path Planning (`pointcloud_3d.py`, `occupancy.py`, `global_map.py`, `rrt_star.py`)**: Unprojects metric depth into a 3D voxel cloud, slices height bands (`4cm - 40cm`) into a local BEV grid, fuses into an 8m x 8m rolling Bayesian log-odds map, and plans Cubic Bezier-smoothed RRT* trajectories tracked by a `PurePursuitController` (UDP port `5006`).

---

## 2. Root Causes of Camera Stream Instability & Sluggishness
1. **Redundant FFmpeg Relay Loop on the Raspberry Pi (`mediamtx_pi.yml`)**:
   - `mediamtx_pi.yml` previously ran `runOnAvailable: ffmpeg ... -i rtsp://127.0.0.1:8554/cam -c copy -f rtsp rtsp://10.61.24.75:8554/cam`. This doubled RTSP socket traffic and saturated the Raspberry Pi's 2.4/5GHz Wi-Fi TX queue.
   - `writeQueueSize: 4096` allowed MediaMTX to buffer seconds of stale packets during Wi-Fi micro-dropouts instead of dropping old frames.
   - Auto-exposure and auto-denoise on the Pi camera caused variable frame exposure times (motion blur on turns) and ISP pipeline lag.
2. **TCP Head-of-Line Blocking & Aggressive Socket Teardowns (`capture.py`)**:
   - Forcing `rtsp_transport;tcp` over Wi-Fi meant a single lost packet stalled the entire video stream waiting for TCP retransmission, followed by a burst of stale frames.
   - `reconnect_timeout=1.2` and `dt > 0.16` were too aggressive: a 165ms Wi-Fi jitter spike flagged `was_interrupted = True`, wiping all ORB optical flow tracks and resetting ByteTrack.
   - Calling `cap.read()` in the worker decoded every buffered frame into BGR rather than draining with `cap.grab()` and decoding only the latest frame.
3. **CPU/GPU Bottlenecks in the SLAM Loop (`visual_odometry.py`, `navigate_rrt.py`, `streamer.py`)**:
   - `cv2.initUndistortRectifyMap(..., cv2.CV_32FC1)` used 32-bit float maps instead of fast fixed-point `cv2.CV_16SC2` maps.
   - `navigate_rrt.py` did not pass a prebuilt LK pyramid (`curr_pyr`) to `orb_tracker.update()`, forcing OpenCV to rebuild image pyramids twice per frame (forward + backward LK passes).
   - `run_yolo()` ran in FP32 at `640` resolution instead of FP16 (`half=True`, `imgsz=480`).
   - `streamer.push_frame()` copied the full `1280x960x3` composite image every frame even when no browser client was connected.

---

## 3. Summary of Fixes Applied
- **`mediamtx_pi.yml`**: Removed the redundant `ffmpeg` relay loop, enabled `udp` transport first, lowered `writeQueueSize: 512`, disabled ISP denoise (`rpiCameraDenoise: "off"`), and locked shutter speed (`rpiCameraShutter: 10000` us) so features stay sharp during fast robot turns.
- **`capture.py`**: Switched to `grab()`/`retrieve()` draining, added `probesize;32768|analyzeduration;0|fflags;nobuffer+discardcorrupt`, preferred `UDP` transport with automatic `TCP` fallback, added H.264 grey-block corruption rejection (`_is_frame_corrupt`), and raised `reconnect_timeout` to `3.5s` (`dt > 0.28s` stutter threshold).
- **`visual_odometry.py` & `mapping_config.json`**: Switched undistortion maps to `cv2.CV_16SC2` (~3x faster `cv2.remap`) and relaxed `max_stutter_dt_sec` to `0.280s` so minor Wi-Fi jitter never wipes SLAM feature tracks.
- **`navigate_rrt.py` & `yolo_mask_track.py`**: Enabled FP16 half-precision (`half=True`, `imgsz=480`) for YOLO11 and shared `build_lk_pyramid(gray)` across `orb_tracker.update()`.
- **`streamer.py`**: Added active client tracking (`_active_clients`) so zero memory copies or thread notifications occur when no web browser is connected.
