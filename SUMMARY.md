# FIBO YOLO Mask Tracking & Robot Streaming - Project Summary

**Date:** October 16, 2026  
**Project Location:** `C:\Users\suwin\Documents\FIBO\MASKINGTRACKING\`  
**Target Application:** Real-time YOLO11 instance segmentation + Lucas-Kanade optical flow background feature tracking for a Raspberry Pi CSI camera mounted on a mobile robot.

---

## 1. Network Architecture & Devices

All three systems operate on the **same local subnet** (`10.61.24.0/24`), enabling direct communication with sub-millisecond network latency and no router firewall restrictions:

| Device | Role | IP Address | OS / Specs | SSH Access |
| :--- | :--- | :--- | :--- | :--- |
| **User Laptop** | Local Dev & Display | `10.61.24.45` | Windows 11 (RTX 4050) | Host Machine |
| **Raspberry Pi** | Robot Camera Streamer | `10.61.24.72` | Debian 13 Bookworm / arm64 | `ssh pi` (Passwordless) |
| **AI PC** | GPU Model Inference | `10.61.24.75` | Windows 11 (RTX 3060) | `ssh pc` (Passwordless) |

---

## 2. Video Streaming Architecture (MediaMTX Native Driver)

### Problems Identified & Solved

1. **`Output file does not contain any stream`**:
   - *Cause*: Piping raw camera video via standard shell pipes (`rpicam-vid ... -o - | ffmpeg ...`) stripped presentation timestamps (PTS) and caused hardware lockups on `/dev/media0`.
   - *Fix*: Switched to MediaMTX's native `source: rpiCamera` hardware encoder, interfacing directly with the Pi's GPU / `libcamera`.
2. **Buffering & Stalling on Wi-Fi**:
   - *Cause*: Default keyframe interval was 60 frames (2.0 seconds). Dropped packets forced the decoder to freeze for up to 2 seconds waiting for an IDR keyframe.
   - *Fix*: Configured `rpiCameraIDRPeriod: 15` (Keyframes every **0.5 seconds** at 30 FPS). Any connection hiccup now recovers in < 500 ms.
3. **Bandwidth & Latency Optimization**:
   - Configured `rpiCameraBitrate: 2000000` (2 Mbps) and `rpiCameraH264Profile: baseline` (no B-frames).
   - Set `writeQueueSize: 4096` to prevent MediaMTX from dropping connections during brief wireless latency spikes.

### Pi Systemd Service
- **Service Name**: `camera-stream.service` (configured to auto-start on boot with automatic restart on failure).
- **Direct RTSP Stream**: `rtsp://10.61.24.72:8554/cam`
- **Browser WebRTC Live Preview**: `http://10.61.24.72:8889/cam`

---

## 3. Integrated YOLO Mask + Optical Flow Pipeline (`yolo_mask_track.py`)

Integrated the optical flow feature tracking logic from `~/robot/track_mask.py` with YOLO11 instance segmentation into a single, clean pipeline:

```
+-------------------------------------------------------------------+
| LAYER 3: REAL-TIME HUD (FPS | Object Count | Pts Count | Motion)  |
+-------------------------------------------------------------------+
| LAYER 2: SHI-TOMASI CORNERS + LUCAS-KANADE OPTICAL FLOW           |
|          - Static background feature points                       |
|          - Motion-fading velocity trails (alpha blend)            |
+-------------------------------------------------------------------+
| LAYER 1: YOLO11 SEGMENTATION & TRACKING                           |
|          - Polygon instance masks overlaid with semi-transparency |
|          - Dilated exclusion mask (prevents points on objects)    |
|          - Bounding boxes + ByteTrack persistent track IDs        |
+-------------------------------------------------------------------+
| CAMERA INPUT (Direct RTSP: 640x480 @ 30 FPS)                      |
+-------------------------------------------------------------------+
```

### Self-Healing & Robotics Resilience Enhancements

- **`LowLatencyCapture` (Threaded Frame Grabber)**:
  - Continuously drains intermediate OpenCV buffers in the background to ensure **zero frame backlog**.
  - Built-in active heartbeat monitor: Automatically tears down and reconnects to the RTSP socket if the stream stalls for > 1.2s or fails consecutively.
  - Configured with `stimeout;2000000` (2s socket timeout) and `nobuffer` so it never hangs indefinitely.
- **Tracker Anomaly Recovery**:
  - Automatically signals a stream discontinuity flag (`interrupted=True`) on connection drops.
  - Automatically resets optical flow point history and wipes `trail_canvas` when a frame jump occurs or median motion exceeds 50px. **This completely eliminates green screen smearing**.
  - Re-seeds fresh Good-Features-to-Track on the clean frame immediately upon recovery.
  - Resets ByteTrack persistence for 1 frame on recovery to discard stale Kalman predictions and re-acquire objects in their new positions.
- **Synchronized Web Server**:
  - Built-in MJPEG HTTP streamer (`http://localhost:8080/`) only transmits when a new frame is generated, preventing browser buffer bloat.

---

## 4. SSH & Automation Setup

Configured dedicated ED25519 authentication keys on the Laptop in `C:\Users\suwin\.ssh\config`:

```ssh
Host pi 10.61.24.72
    HostName 10.61.24.72
    User admin
    IdentityFile C:\Users\suwin\.ssh\id_pi

Host pc aipc 10.61.24.75
    HostName 10.61.24.75
    User computer12
    IdentityFile C:\Users\suwin\.ssh\id_pi
```

- Connect to Raspberry Pi: Type `ssh pi` (Passwordless).
- Connect to AI PC: Type `ssh pc` (Passwordless).

---

## 5. Folder Structure & Launchers

Reorganized `C:\Users\suwin\Documents\FIBO\MASKINGTRACKING\` into a clean layout:

```text
C:\Users\suwin\Documents\FIBO\MASKINGTRACKING\
│
├── START.bat                  # Master Interactive Control Center (Menu [1-6])
├── 1_TRACK_PI_STREAM.bat      # 1-Click: Track Pi robot camera stream (Direct RTSP)
├── 2_TRACK_WEBCAM.bat         # 1-Click: Track laptop webcam
├── yolo_mask_track.py         # Main self-healing tracking engine
├── README.md                  # Project documentation & reference
├── SUMMARY.md                 # Complete project summary report
│
├── models\                    # Model weights directory
│   ├── yolo11n-seg.pt         # Fast nano segmentation model (~6MB)
│   └── yolo11s-seg.pt         # Accurate small segmentation model (~20MB)
│
└── launchers\                 # Secondary utility scripts
    ├── connect_pi.bat         # SSH terminal to Raspberry Pi (Passwordless)
    ├── connect_pc.bat         # SSH terminal to AI PC (Passwordless)
    ├── view_pi_browser.bat    # Open live Pi WebRTC feed in browser
    ├── track_pi_relay.bat     # Track via AI PC MediaMTX relay
    └── track_webcam_headless.bat # Run headless on background web port 8080
```

---

## 6. Verification & Test Results

| Test Case | Method | Result | Notes |
| :--- | :--- | :--- | :--- |
| **Pi Hardware Camera** | MediaMTX native `rpiCamera` | **PASSED** | 640x480 @ 30 FPS, hardware H.264 |
| **Direct Pi RTSP** | OpenCV capture from Laptop | **PASSED** | Zero buffer backlog, < 100ms latency |
| **AI PC RTSP Relay** | MediaMTX on PC (`10.61.24.75`) | **PASSED** | Video stream relayed and accessible |
| **YOLO Tracking** | `yolo_mask_track.py` | **PASSED** | 190+ points tracked, smooth mask layering |
| **Auto-Reconnect** | Forced stream interruption | **PASSED** | Optical flow reset, 0 smears, fast recovery |
| **Passwordless SSH** | Laptop to Pi (`ssh pi`) | **PASSED** | Instant shell access |
| **Passwordless SSH** | Laptop to AI PC (`ssh pc`) | **PASSED** | Instant shell access |
