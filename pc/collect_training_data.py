#!/usr/bin/env python3
"""
collect_training_data.py
Interactive Dataset Collector for Training Custom YOLO Wall & Obstacle Detection.

Features:
  1. Streams live from Raspberry Pi robot RTSP stream (--source pi) or USB webcam.
  2. Manual Snapshot: Press SPACE or 'C' to capture the current frame.
  3. Auto-Capture Interval: Press 'A' to toggle automated continuous capture (every 0.8s) while driving.
  4. Movement/Difference Filter: Rejects identical frames when the robot is sitting still.
  5. Automatically organizes images into YOLO structure (80% train / 20% val):
     dataset/images/train/
     dataset/images/val/
"""

import argparse
import datetime
import os
import random
import sys
import time

import cv2
import numpy as np

from config_manager import DEFAULT_CONFIG_PATH, load_config, resolve_pi_host
from capture import LowLatencyCapture


def main():
    ap = argparse.ArgumentParser(description="Collect Training Dataset for Custom YOLO Wall/Obstacle Detection")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to mapping_config.json")
    ap.add_argument("--source", default=None, help="Video source ('pi', webcam index '0', or RTSP URL)")
    ap.add_argument("--pi-host", default=None, help="Optional override for Raspberry Pi IP/hostname")
    ap.add_argument("--output-dir", default="dataset", help="Output directory for dataset (default: dataset/)")
    ap.add_argument("--interval", type=float, default=0.8, help="Auto-capture interval in seconds (default: 0.8s)")
    ap.add_argument("--val-split", type=float, default=0.20, help="Fraction of images saved to validation set (default: 0.20)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    source = args.source if args.source is not None else cfg["camera"]["source"]
    resolved_pi_ip = resolve_pi_host(cfg, args.pi_host)

    # Prepare dataset folders
    train_dir = os.path.join(args.output_dir, "images", "train")
    val_dir = os.path.join(args.output_dir, "images", "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Count existing images
    train_count = len([f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.png'))])
    val_count = len([f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.png'))])

    cap = LowLatencyCapture(
        source=source,
        reconnect_timeout=float(cfg["network"].get("reconnect_timeout_sec", 4.0)),
        rtsp_transport=str(cfg["network"].get("rtsp_transport", "tcp")),
        cfg=cfg,
    )

    print("==================================================================")
    print("  FIBO Custom YOLO Dataset Collector")
    print(f"  Target: {train_dir} and {val_dir}")
    print("  Controls:")
    print("    [SPACE] or [C] : Capture current frame snapshot")
    print("    [A]            : Toggle Auto-Capture while driving (every 0.8s)")
    print("    [Q] or [ESC]   : Finish and exit")
    print("==================================================================")

    auto_capture = False
    last_capture_time = 0.0
    last_saved_frame = None
    save_notification = ""
    save_notif_time = 0.0

    window_name = "FIBO YOLO Dataset Collector - Drive & Capture Walls/Obstacles"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    prev_time = time.time()

    try:
        while True:
            ret, frame, _ = cap.read(timeout=0.04)
            now = time.time()
            dt = max(now - prev_time, 0.001)
            prev_time = now
            fps = 1.0 / dt

            if not ret or frame is None:
                ph = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(ph, f"Connecting to {source} ({resolved_pi_ip})...", (30, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.imshow(window_name, ph)
                if (cv2.waitKey(20) & 0xFF) in (ord('q'), 27):
                    break
                continue

            # Split stereo side-by-side if present (use left camera for training)
            if frame.shape[1] >= 1000:
                frame_save = frame[:, :frame.shape[1] // 2].copy()
            else:
                frame_save = frame.copy()

            h, w = frame_save.shape[:2]
            vis = frame_save.copy()

            # Check for movement difference if auto-capturing
            should_auto_save = False
            if auto_capture and (now - last_capture_time >= args.interval):
                if last_saved_frame is None:
                    should_auto_save = True
                else:
                    # Compute frame diff to avoid saving duplicate stationary images
                    diff = cv2.absdiff(cv2.cvtColor(frame_save, cv2.COLOR_BGR2GRAY),
                                       cv2.cvtColor(last_saved_frame, cv2.COLOR_BGR2GRAY))
                    mean_diff = float(np.mean(diff))
                    if mean_diff >= 4.5:  # Sufficient motion detected
                        should_auto_save = True

            key = cv2.waitKey(1) & 0xFF

            save_this_frame = False
            if key in (ord(' '), ord('c'), ord('C')):
                save_this_frame = True
            elif key in (ord('a'), ord('A')):
                auto_capture = not auto_capture
                print(f"[COLLECTOR] Auto-Capture -> {'ENABLED (Capturing while moving)' if auto_capture else 'DISABLED'}")
            elif key in (ord('q'), 27):
                break

            if save_this_frame or should_auto_save:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                is_val = (random.random() < args.val_split)
                target_folder = val_dir if is_val else train_dir
                filename = f"fibo_{ts}.jpg"
                filepath = os.path.join(target_folder, filename)

                cv2.imwrite(filepath, frame_save, [cv2.IMWRITE_JPEG_QUALITY, 96])
                last_capture_time = now
                last_saved_frame = frame_save.copy()

                if is_val:
                    val_count += 1
                else:
                    train_count += 1

                subset_tag = "VAL" if is_val else "TRAIN"
                save_notification = f"SAVED [{subset_tag}] #{train_count + val_count}: {filename}"
                save_notif_time = now
                print(f"[DATASET] {save_notification}")

            # Draw HUD
            hud_bg = np.zeros((70, w, 3), dtype=np.uint8)
            cv2.rectangle(hud_bg, (0, 0), (w, 70), (25, 28, 35), -1)

            status_color = (0, 220, 100) if auto_capture else (180, 180, 180)
            auto_str = "AUTO: ON (0.8s)" if auto_capture else "AUTO: OFF (Press 'A')"
            cv2.putText(vis, auto_str, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

            count_str = f"Train: {train_count} | Val: {val_count} | Total: {train_count + val_count}"
            cv2.putText(vis, count_str, (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            fps_str = f"FPS: {fps:.1f} | Res: {w}x{h}"
            cv2.putText(vis, fps_str, (w - 220, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

            # Show save flash banner
            if now - save_notif_time < 0.8:
                cv2.rectangle(vis, (0, h - 36), (w, h), (0, 180, 60), -1)
                cv2.putText(vis, save_notification, (16, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(vis, "Press [SPACE] Snapshot | [A] Auto-Capture | [Q] Exit", (16, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

            cv2.imshow(window_name, vis)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n==================================================================")
        print(f"  Capture Complete! Saved {train_count} train images and {val_count} val images.")
        print(f"  Total Images: {train_count + val_count} in '{args.output_dir}'")
        print("  Next Step: Run '8_AUTO_LABEL_DATASET.bat' to generate labels!")
        print("==================================================================")


if __name__ == "__main__":
    main()
