"""
yolo_mask_track.py
Ultra-Low Latency, Self-Healing YOLO Segmentation + Multi-Detector Optical
Flow + Binary Occupancy Grid — rendered as a single window with 4 sub-panels.
Uses centralized `config_manager.py` (Zero Hardcoded IPs & Configurable Ignored Classes).
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config_manager import DEFAULT_CONFIG_PATH, load_config, resolve_pi_host
from capture import LowLatencyCapture
import streamer
from detectors import ShiTomasiTracker, ORBTracker, build_lk_pyramid
import occupancy
from visual_odometry import VisualOdometry


_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))


def _colour_for_id(track_id):
    rng = np.random.default_rng(int(track_id) * 42 + 17)
    return tuple(int(c) for c in rng.integers(50, 255, size=3))


def _put_panel_label(img, text, bg=(20, 20, 20)):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 26), bg, -1)
    cv2.putText(img, text, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)


def run_yolo(model, frame, persist, conf, device, ignored_classes=None):
    h, w = frame.shape[:2]
    results = model.track(frame, persist=persist, tracker="bytetrack.yaml",
                          conf=conf, device=device, imgsz=480, verbose=False)

    ignore_set = {c.lower().strip() for c in (ignored_classes or [])}
    static_mask = np.full((h, w), 255, dtype=np.uint8)
    mask_overlay = np.zeros_like(frame)
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    num_objects = 0

    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            track_ids = (r.boxes.id.int().cpu().numpy()
                         if r.boxes.id is not None else [0] * len(boxes))
            has_masks = (r.masks is not None)
            masks = r.masks.data.cpu().numpy() if has_masks else None

            for i, (box, cls_idx, conf_val, tid) in enumerate(zip(
                    boxes, classes, confs, track_ids)):
                cls_name = str(model.names[int(cls_idx)]).lower().strip()
                if cls_name in ignore_set:
                    continue

                if "wall" in cls_name:
                    continue

                num_objects += 1
                color = _colour_for_id(tid)

                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))

                if has_masks and masks is not None and i < len(masks):
                    mr = cv2.resize((masks[i] > 0.5).astype(np.uint8), (w, h),
                                    interpolation=cv2.INTER_NEAREST)
                else:
                    # Guard against huge bounding boxes turning the map into a red blob
                    box_w = x2 - x1
                    box_h = y2 - y1
                    if (box_w * box_h) > (w * h * 0.40):
                        continue
                    mr = np.zeros((h, w), dtype=np.uint8)
                    mr[y1:y2, x1:x2] = 1

                mask_overlay[mr > 0] = color
                combined_mask[mr > 0] = 255

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{tid} {model.names[int(cls_idx)]} {conf_val:.2f}"
                cv2.putText(frame, label, (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            if num_objects > 0:
                dilated = cv2.dilate(combined_mask, _DILATE_KERNEL)
                static_mask[dilated > 0] = 0

    annotated = cv2.addWeighted(frame, 1.0, mask_overlay, 0.35, 0)
    return annotated, static_mask, combined_mask, num_objects


def main():
    ap = argparse.ArgumentParser(
        description="YOLO Mask + Multi-Detector Optical Flow + Occupancy Grid (4-panel)")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to mapping_config.json")
    ap.add_argument("--source", default=None, help="Webcam index, video file, or RTSP URL")
    ap.add_argument("--pi-host", default=None, help="Optional override for Raspberry Pi IP")
    ap.add_argument("--model", default=None, help="YOLO segmentation model")
    ap.add_argument("--conf", type=float, default=None, help="YOLO confidence threshold")
    ap.add_argument("--web-port", type=int, default=None, help="MJPEG stream port (0 to disable)")
    ap.add_argument("--no-display", action="store_true", help="Disable local GUI window")
    ap.add_argument("--calib", default=None, help="Camera calibration file")
    args = ap.parse_args()

    cfg = load_config(os.path.abspath(args.config))
    if args.pi_host is not None:
        cfg["network"]["pi_host"] = args.pi_host
    if args.source is not None:
        cfg["camera"]["source"] = args.source
    if args.model is not None:
        cfg["yolo_and_planner"]["yolo_model"] = args.model
    if args.conf is not None:
        cfg["yolo_and_planner"]["yolo_conf"] = args.conf
    if args.web_port is not None:
        cfg["network"]["web_stream_port"] = args.web_port
    if args.calib is not None:
        cfg["camera"]["calib_file"] = args.calib

    source = str(cfg["camera"].get("source", "pi"))
    is_webcam = source.isdigit()
    if not is_webcam:
        pi_ip = resolve_pi_host(cfg, config_path=os.path.abspath(args.config), save_if_changed=True)
        print(f"[NETWORK] Active Raspberry Pi Host -> {pi_ip}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device} ({torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'})")

    model_name = cfg["yolo_and_planner"]["yolo_model"]
    cand_paths = [
        model_name,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", model_name),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", model_name),
        os.path.join("models", model_name),
        os.path.join("pc", "models", model_name),
    ]
    model_path = next((p for p in cand_paths if os.path.exists(p)), model_name)
    print(f"[INFO] Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    print(f"[INFO] Active Detection Classes: {model.names}")

    cap = LowLatencyCapture(source=source, cfg=cfg)
    web_port = int(cfg["network"].get("web_stream_port", 8080))
    if web_port > 0:
        streamer.start(web_port)

    undistorter = VisualOdometry(calib_file=cfg["camera"]["calib_file"], enable_undistort=not is_webcam)

    st_tracker  = ShiTomasiTracker(max_points=250, min_points=120)
    orb_tracker = ORBTracker(max_points=200, min_points=80)
    orb_trail   = None
    H_ipm = None

    window_name = "YOLO + ShiTomasi + ORB + Occupancy"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

    prev_time = time.time()
    last_valid = time.time()
    ignored_classes = cfg["yolo_and_planner"].get("ignored_classes", [])

    dynamic_marking = True
    dynamic_masking = True

    print("=" * 68)
    print(" [YOLO MASK + DYNAMIC FEATURE TRACKING PIPELINE READY]")
    print("  * 'D' or 'M' : Toggle Dynamic Marking (Red dots on moving objects ON/OFF)")
    print("  * 'K'        : Toggle Dynamic Masking (Mask exclusion ON/OFF)")
    print("  * 'Q' or ESC : Quit")
    print("=" * 68)
    try:
        while True:
            ret, frame, interrupted = cap.read(timeout=0.04)

            if not ret or frame is None:
                if time.time() - last_valid > 1.0:
                    ph = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(ph, "[RECONNECTING...]", (50, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    streamer.push_frame(ph)
                if not args.no_display:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        break
                    elif key in (ord('d'), ord('D'), ord('m'), ord('M')):
                        dynamic_marking = not dynamic_marking
                        st_desc = "ENABLED (Red dots mark moving objects)" if dynamic_marking else "DISABLED (Static dots only)"
                        print(f"[TEST] Dynamic Marking -> {st_desc}")
                    elif key in (ord('k'), ord('K')):
                        dynamic_masking = not dynamic_masking
                        mask_desc = "ENABLED (Mask exclusion active)" if dynamic_masking else "DISABLED (Tracking everywhere)"
                        print(f"[TEST] Dynamic Masking -> {mask_desc}")
                time.sleep(0.01)
                continue

            last_valid = time.time()
            if frame.shape[1] >= 1000:
                frame = frame[:, :frame.shape[1] // 2]
            frame = undistorter.undistort(frame)
            h, w = frame.shape[:2]

            if interrupted or orb_trail is None or orb_trail.shape[:2] != (h, w):
                st_tracker.reset()
                orb_tracker.reset()
                orb_trail = np.zeros((h, w, 3), dtype=np.uint8)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_pyr = build_lk_pyramid(gray)

            yolo_vis, static_mask, combined_mask, n_obj = run_yolo(
                model, frame.copy(), persist=not interrupted,
                conf=float(cfg["yolo_and_planner"]["yolo_conf"]),
                device=device,
                ignored_classes=ignored_classes,
            )

            active_static_mask = static_mask if dynamic_masking else None

            st_old, st_new = st_tracker.update(gray, active_static_mask, curr_pyr=curr_pyr)
            st_vis = frame.copy()
            ShiTomasiTracker.draw(st_vis, st_old, st_new, dynamic_marking=dynamic_marking, object_mask=combined_mask)

            orb_old, orb_new = orb_tracker.update(gray, active_static_mask, curr_pyr=curr_pyr)
            orb_vis = frame.copy()
            ORBTracker.draw(orb_vis, orb_old, orb_new, orb_trail, dynamic_marking=dynamic_marking, object_mask=combined_mask)

            if H_ipm is None:
                H_ipm = occupancy.default_ipm(img_w=w, img_h=h)

            grid = occupancy.compute(combined_mask, orb_new, (h, w), H_ipm, frame_gray=gray)

            now = time.time()
            fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
            prev_time = now

            st_motion = 0.0
            if len(st_new) > 0 and len(st_old) == len(st_new):
                st_motion = float(np.median(np.linalg.norm(st_new - st_old, axis=1)))
            orb_motion = 0.0
            if len(orb_new) > 0 and len(orb_old) == len(orb_new):
                orb_motion = float(np.median(np.linalg.norm(orb_new - orb_old, axis=1)))

            status = "LIVE" if cap.is_connected else "RECONN"
            _put_panel_label(yolo_vis, f"[{status}] {fps:.0f}FPS | YOLO Objects: {n_obj}")
            dm_txt = "ON (Red Dots)" if dynamic_marking else "OFF (Static)"
            mask_txt = "ON" if dynamic_masking else "OFF"
            _put_panel_label(st_vis, f"Shi-Tomasi | DynMark: {dm_txt} [D/M] | Mask: {mask_txt} [K] | Pts: {len(st_new)}")
            _put_panel_label(orb_vis, f"ORB Features | DynMark: {dm_txt} | Pts: {len(orb_new)} | Motion: {orb_motion:.1f}px")

            occ_vis = occupancy.render(grid, out_w=w, out_h=h)
            free_pct = int(100 * np.sum(grid == occupancy.FREE) / grid.size)
            occ_pct  = int(100 * np.sum(grid == occupancy.OCCUPIED) / grid.size)
            _put_panel_label(occ_vis, f"Top-Down BEV (ORB) | Free: {free_pct}% | Occ: {occ_pct}%")

            top = np.hstack([yolo_vis, st_vis])
            bot = np.hstack([orb_vis, occ_vis])
            composite = np.vstack([top, bot])

            streamer.push_frame(composite)

            if not args.no_display:
                try:
                    cv2.imshow(window_name, composite)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        break
                    elif key in (ord('d'), ord('D'), ord('m'), ord('M')):
                        dynamic_marking = not dynamic_marking
                        st_desc = "ENABLED (Red dots mark moving objects)" if dynamic_marking else "DISABLED (Static dots only)"
                        print(f"[TEST] Dynamic Marking -> {st_desc}")
                    elif key in (ord('k'), ord('K')):
                        dynamic_masking = not dynamic_masking
                        mask_desc = "ENABLED (Mask exclusion active)" if dynamic_masking else "DISABLED (Tracking everywhere)"
                        print(f"[TEST] Dynamic Masking -> {mask_desc}")
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
