#!/usr/bin/env python3
"""
stereo_calibrate.py
Interactive & Automatic Checkerboard Stereo Calibrator.
Works with the live Side-by-Side 1280x480 RTSP stream from the Pi (`--source pi`).

Outputs:
  - stereo_calib.npz (K_left, dist_left, K_right, dist_right, R, T, R1, R2, P1, P2, Q)
  - stereo_slam_config.yaml (ORB-SLAM3 Stereo-Inertial compatible YAML)
"""

import argparse
import time
import cv2
import numpy as np

from config_manager import load_config
from capture import LowLatencyCapture


def save_orbslam3_yaml(filepath, K_l, D_l, K_r, D_r, R, T, img_size, fps=24):
    w, h = img_size
    baseline = float(np.linalg.norm(T))
    yaml_text = f"""%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"

# Left Camera (Logitech C930e)
Camera1.fx: {float(K_l[0, 0]):.6f}
Camera1.fy: {float(K_l[1, 1]):.6f}
Camera1.cx: {float(K_l[0, 2]):.6f}
Camera1.cy: {float(K_l[1, 2]):.6f}
Camera1.k1: {float(D_l.ravel()[0]):.6f}
Camera1.k2: {float(D_l.ravel()[1]):.6f}
Camera1.p1: {float(D_l.ravel()[2]):.6f}
Camera1.p2: {float(D_l.ravel()[3]):.6f}

# Right Camera (Logitech C930e)
Camera2.fx: {float(K_r[0, 0]):.6f}
Camera2.fy: {float(K_r[1, 1]):.6f}
Camera2.cx: {float(K_r[0, 2]):.6f}
Camera2.cy: {float(K_r[1, 2]):.6f}
Camera2.k1: {float(D_r.ravel()[0]):.6f}
Camera2.k2: {float(D_r.ravel()[1]):.6f}
Camera2.p1: {float(D_r.ravel()[2]):.6f}
Camera2.p2: {float(D_r.ravel()[3]):.6f}

Camera.width: {w}
Camera.height: {h}
Camera.fps: {fps}
Camera.RGB: 0
Stereo.ThDepth: 40.0
Stereo.b: {baseline:.6f}

# IMU (BNO055 100Hz Raw Mode)
IMU.Frequency: 100.0
IMU.NoiseGyro: 0.005
IMU.NoiseAcc: 0.05
IMU.GyroWalk: 0.0005
IMU.AccWalk: 0.005

ORBextractor.nFeatures: 600
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(yaml_text)


def main():
    ap = argparse.ArgumentParser(description="Checkerboard Stereo Calibration")
    ap.add_argument("--source", default="pi", help="'pi' for RTSP Side-by-Side stream, or RTSP URL")
    ap.add_argument("--board-size", default="9x6", help="Inner corners COLSxROWS (default: 9x6)")
    ap.add_argument("--square-size", type=float, default=25.0, help="Checkerboard square size in mm (default: 25.0)")
    ap.add_argument("--target-pairs", type=int, default=30, help="Number of pairs to collect")
    ap.add_argument("--out", default="stereo_calib.npz", help="Output .npz path")
    ap.add_argument("--yaml-out", default="stereo_slam_config.yaml", help="Output ORB-SLAM3 .yaml path")
    args = ap.parse_args()

    cols, rows = map(int, args.board_size.lower().split("x"))
    board_size = (cols, rows)
    square_m = float(args.square_size) / 1000.0

    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m

    obj_points = []
    img_points_l = []
    img_points_r = []

    cfg = load_config()
    cap = LowLatencyCapture(source=args.source, rtsp_transport="tcp", cfg=cfg)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    last_capture_t = 0.0
    img_shape = (640, 480)

    print("=" * 70)
    print(f" [STEREO CALIBRATION] Board: {cols}x{rows} inner corners | Square: {args.square_size} mm")
    print("  * Hold checkerboard visible in BOTH Left and Right cameras.")
    print("  * Press SPACE to capture a pair (or auto-captures every 1.3s when steady).")
    print("  * Press 'C' once you have >= 12 pairs to compute calibration.")
    print("=" * 70)

    while True:
        ret, frame, _ = cap.read(timeout=0.05)
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        if w < 1000:
            print("[CALIB WARN] Incoming stream is not Side-by-Side (width < 1000). Ensure pi_stereo_capture.py is running!")
            time.sleep(0.5)
            continue

        mid = w // 2
        frame_l = frame[:, :mid].copy()
        frame_r = frame[:, mid:].copy()
        img_shape = (mid, h)

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, board_size, flags)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, board_size, flags)

        vis_l = frame_l.copy()
        vis_r = frame_r.copy()

        both_found = ret_l and ret_r
        if both_found:
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(vis_l, board_size, corners_l, ret_l)
            cv2.drawChessboardCorners(vis_r, board_size, corners_r, ret_r)

        vis = np.hstack([vis_l, vis_r])
        status_col = (0, 255, 0) if both_found else (0, 165, 255)
        cv2.putText(
            vis,
            f"Pairs: {len(obj_points)}/{args.target_pairs} | SPACE=Capture | C=Calibrate | Q=Quit",
            (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2
        )
        cv2.imshow("Stereo Checkerboard Calibration", vis)
        key = cv2.waitKey(1) & 0xFF

        now = time.monotonic()
        if (key == ord(" ") or (both_found and now - last_capture_t > 1.3)) and both_found:
            last_capture_t = now
            obj_points.append(objp)
            img_points_l.append(corners_l)
            img_points_r.append(corners_r)
            print(f"[CALIB] Captured pair #{len(obj_points)} / {args.target_pairs}")
            if len(obj_points) >= args.target_pairs:
                print("[CALIB] Target pairs reached! Computing stereo calibration...")
                break
        elif key in (ord("c"), ord("C")) and len(obj_points) >= 10:
            break
        elif key in (ord("q"), 27):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    print("[CALIB] Calibrating individual intrinsics...")
    _, K_l, dist_l, _, _ = cv2.calibrateCamera(obj_points, img_points_l, img_shape, None, None)
    _, K_r, dist_r, _, _ = cv2.calibrateCamera(obj_points, img_points_r, img_shape, None, None)

    print("[CALIB] Running stereoCalibrate...")
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    rms, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r,
        K_l, dist_l, K_r, dist_r, img_shape,
        criteria=criteria, flags=stereo_flags
    )

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, dist_l, K_r, dist_r, img_shape, R, T, alpha=0
    )

    baseline_m = abs(float(T[0, 0]))
    print("=" * 70)
    print(f"[CALIB SUCCESS] Stereo RMS Reprojection Error : {rms:.4f} px")
    print(f"[CALIB SUCCESS] Measured Stereo Baseline       : {baseline_m * 100.0:.2f} cm ({baseline_m:.4f} m)")
    print(f"[CALIB SUCCESS] Rectified Focal Length (fx)    : {float(P1[0, 0]):.2f} px")
    print("=" * 70)

    np.savez(
        args.out,
        K_left=K_l, dist_left=dist_l,
        K_right=K_r, dist_right=dist_r,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        image_size=np.array(img_shape),
        rms_error=rms,
    )
    save_orbslam3_yaml(args.yaml_out, K_l, dist_l, K_r, dist_r, R, T, img_shape)
    print(f"[CALIB] Saved {args.out} and {args.yaml_out}")


if __name__ == "__main__":
    main()
