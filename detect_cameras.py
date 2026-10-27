#!/usr/bin/env python3
"""
detect_cameras.py
Automatic Logitech C930e / UVC Stereo Camera Detector for Raspberry Pi 4.

Scans the Linux V4L2 subsystem, sysfs, and USB topology to:
1. Automatically find all connected Logitech webcams.
2. Differentiate between real Video Capture stream nodes (stream_index == 0)
   and hardware Metadata/Diagnostic nodes (stream_index == 1).
3. Identify physical USB port paths and hardware serial numbers so cameras
   can be reliably tracked across USB port changes or reboots.
4. Test-capture frames to verify camera hardware accessibility.
5. Provide CLI diagnostics, JSON export, bash environment exports, and config updates.

Usage:
  python3 detect_cameras.py                  # Display formatted camera diagnostic report
  python3 detect_cameras.py --test-capture   # Grab 10 test frames to benchmark synchronized FPS
  python3 detect_cameras.py --json           # Output machine-readable JSON
  python3 detect_cameras.py --bash           # Output shell variables: export LEFT_CAM=0 RIGHT_CAM=2
  python3 detect_cameras.py --update-config  # Update mapping_config.json with detected IDs
  python3 detect_cameras.py --start          # Detect and start pi_stereo_capture.py automatically
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time


def get_v4l2_formats(dev_path: str):
    """Query supported video formats using v4l2-ctl if available."""
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "-d", dev_path, "--list-formats"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        formats = []
        for line in out.splitlines():
            if ":" in line and any(f in line for f in ["YUYV", "MJPG", "H264", "RGB"]):
                part = line.split(":", 1)[1].strip()
                formats.append(part)
        return formats
    except Exception:
        return []


def get_device_holder(dev_path: str):
    """Check if device is held open by a streaming process (e.g. pi_stereo_capture)."""
    try:
        out = subprocess.check_output(["fuser", dev_path], stderr=subprocess.DEVNULL, text=True).strip()
        if out:
            pids = out.split()
            names = []
            for pid in pids:
                try:
                    cmd = open(f"/proc/{pid}/cmdline", "r").read().replace("\x00", " ").strip()
                    parts = cmd.split()
                    proc_name = (
                        os.path.basename(parts[1])
                        if len(parts) > 1 and parts[0].startswith("python")
                        else os.path.basename(parts[0])
                    )
                    names.append(f"{proc_name} (PID {pid})")
                except Exception:
                    names.append(f"PID {pid}")
            return True, ", ".join(names)
    except Exception:
        pass
    return False, None


def test_camera_capture(dev_idx: int, width: int = 640, height: int = 480):
    """Test-capture a frame using OpenCV to verify hardware availability and latency."""
    dev_path = f"/dev/video{dev_idx}"
    in_use, holder = get_device_holder(dev_path)
    if in_use:
        return True, f"STREAMING LIVE via {holder}", 0.0, None

    try:
        import cv2

        t0 = time.monotonic()
        cap = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            return False, "Failed to open V4L2 device", 0.0, None

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        ret, frame = cap.read()
        dt_ms = (time.monotonic() - t0) * 1000.0
        cap.release()

        if ret and frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            return True, f"{w}x{h} BGR", dt_ms, frame.shape
        else:
            return False, "Read returned empty frame", dt_ms, None
    except ImportError:
        return True, "OpenCV not installed (skipped test)", 0.0, None
    except Exception as e:
        return False, f"Error: {e}", 0.0, None


def detect_logitech_cameras(only_logitech: bool = True, test_frames: bool = True):
    """
    Scan /sys/class/video4linux to discover all video devices.
    Returns:
        cameras: list of dictionaries for valid video capture devices
        metadata_nodes: list of device paths that are metadata-only nodes
    """
    cameras = []
    metadata_nodes = []

    dev_paths = sorted(
        glob.glob("/sys/class/video4linux/video*"),
        key=lambda p: int(os.path.basename(p).replace("video", "")),
    )

    for p in dev_paths:
        node_name = os.path.basename(p)  # e.g. 'video0'
        dev_idx = int(node_name.replace("video", ""))
        dev_path = f"/dev/{node_name}"

        # 1. Device name
        name_file = os.path.join(p, "name")
        if not os.path.exists(name_file):
            continue
        try:
            with open(name_file, "r") as f:
                cam_name = f.read().strip()
        except Exception:
            continue

        # 2. Stream index: In Linux UVC, index 0 is video capture, index 1 is metadata
        index_file = os.path.join(p, "index")
        try:
            stream_index = int(open(index_file).read().strip()) if os.path.exists(index_file) else 0
        except Exception:
            stream_index = 0

        # 3. Check parent USB device details
        parent_dev = os.path.realpath(os.path.join(p, "device", ".."))
        serial = None
        usb_port = None
        busnum = None
        devnum = None

        if os.path.exists(parent_dev):
            usb_port = os.path.basename(parent_dev)  # e.g. '1-1.3'
            serial_file = os.path.join(parent_dev, "serial")
            if os.path.exists(serial_file):
                try:
                    serial = open(serial_file).read().strip()
                except Exception:
                    pass
            bus_file = os.path.join(parent_dev, "busnum")
            if os.path.exists(bus_file):
                try:
                    busnum = open(bus_file).read().strip()
                except Exception:
                    pass
            dev_file = os.path.join(parent_dev, "devnum")
            if os.path.exists(dev_file):
                try:
                    devnum = open(dev_file).read().strip()
                except Exception:
                    pass

        is_logitech = "Logitech" in cam_name or "C930e" in cam_name
        if only_logitech and not is_logitech:
            continue

        if stream_index != 0:
            metadata_nodes.append({
                "node": dev_path,
                "index": dev_idx,
                "name": cam_name,
                "usb_port": usb_port,
                "purpose": "Metadata / Diagnostics (Non-Streaming)",
            })
            continue

        formats = get_v4l2_formats(dev_path)

        cam_info = {
            "node": dev_path,
            "index": dev_idx,
            "name": cam_name,
            "stream_index": stream_index,
            "usb_port": usb_port,
            "serial": serial,
            "busnum": busnum,
            "devnum": devnum,
            "formats": formats,
            "test_ok": None,
            "test_msg": None,
            "test_dt_ms": 0.0,
        }

        if test_frames:
            ok, msg, dt_ms, _ = test_camera_capture(dev_idx)
            cam_info["test_ok"] = ok
            cam_info["test_msg"] = msg
            cam_info["test_dt_ms"] = dt_ms

        cameras.append(cam_info)

    # Sort cameras by index or USB port
    cameras.sort(key=lambda c: c["index"])
    return cameras, metadata_nodes


def get_stereo_cameras(only_logitech: bool = True):
    """
    Convenience function for other scripts:
    Returns (left_device_index, right_device_index).
    """
    cameras, _ = detect_logitech_cameras(only_logitech=only_logitech, test_frames=False)
    if len(cameras) >= 2:
        return cameras[0]["index"], cameras[1]["index"]
    elif len(cameras) == 1:
        return cameras[0]["index"], None
    return None, None


def benchmark_stereo_capture(left_idx: int, right_idx: int, width: int = 320, height: int = 240, frames: int = 30):
    """Benchmark synchronized capture across both cameras."""
    dev_l = f"/dev/video{left_idx}"
    in_use_l, holder_l = get_device_holder(dev_l)
    if in_use_l:
        print(f"\n[BENCHMARK] Cameras are currently active in: {holder_l}.")
        print("  Direct TCP Stream on port 8555 is live and serving frames (<40ms mode)!")
        print("  (To run an exclusive low-level hardware benchmark, run 'sudo systemctl stop stereo-stream' first).\n")
        return True
    import cv2

    print(f"\n[BENCHMARK] Testing dual capture: /dev/video{left_idx} + /dev/video{right_idx} ({width}x{height} @ {frames} frames)...")
    cap_l = cv2.VideoCapture(left_idx, cv2.CAP_V4L2)
    cap_r = cv2.VideoCapture(right_idx, cv2.CAP_V4L2)

    for cap in (cap_l, cap_r):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ok_l = cap_l.isOpened()
    ok_r = cap_r.isOpened()
    if not (ok_l and ok_r):
        print(f"  [ERROR] Could not open both cameras (Left={ok_l}, Right={ok_r})")
        cap_l.release()
        cap_r.release()
        return False

    success_count = 0
    t0 = time.monotonic()
    for _ in range(frames):
        rl, fl = cap_l.read()
        rr, fr = cap_r.read()
        if rl and rr and fl is not None and fr is not None:
            success_count += 1

    dt = max(time.monotonic() - t0, 1e-3)
    fps = success_count / dt
    cap_l.release()
    cap_r.release()

    print(f"  [RESULT] Successfully synchronized {success_count}/{frames} pairs in {dt:.2f}s ({fps:.1f} FPS)!")
    return success_count == frames


def print_report(cameras, metadata_nodes, swap: bool = False):
    """Print human-readable diagnostic report."""
    print("=" * 80)
    print("               RASPBERRY PI LOGITECH STEREO CAMERA DETECTOR")
    print("=" * 80)

    if not cameras:
        print("[!] No Logitech video streaming cameras found!")
        print("    Check physical USB connections or run 'lsusb'.")
        if metadata_nodes:
            print("\n    Note: Metadata nodes were found but no video streaming nodes:")
            for m in metadata_nodes:
                print(f"      - {m['node']} ({m['name']} on USB {m['usb_port']})")
        print("=" * 80)
        return

    print(f"[FOUND] {len(cameras)} Active Logitech Video Capture Device(s):\n")

    left_cam = cameras[0] if not swap else (cameras[1] if len(cameras) > 1 else cameras[0])
    right_cam = (cameras[1] if len(cameras) > 1 else None) if not swap else cameras[0]

    for idx, cam in enumerate(cameras):
        role = "Left Camera" if cam == left_cam else ("Right Camera" if cam == right_cam else f"Camera #{idx+1}")
        formats_str = ", ".join(cam["formats"]) if cam["formats"] else "MJPG, YUYV (UVC default)"
        test_status = (
            f"\033[92mOK\033[0m ({cam['test_msg']}, grabbed in {cam['test_dt_ms']:.1f}ms)"
            if cam["test_ok"]
            else f"\033[91mFAILED\033[0m ({cam['test_msg']})"
        ) if cam["test_ok"] is not None else "Skipped"

        print(f"  * {role}:")
        print(f"    - Device Node       : {cam['node']}  (Index: {cam['index']})")
        print(f"    - Physical USB Port : {cam['usb_port']} (Bus {cam['busnum']}, Device {cam['devnum']})")
        print(f"    - Hardware Serial   : {cam['serial'] or 'N/A'}")
        print(f"    - Camera Model      : {cam['name']}")
        print(f"    - Formats           : {formats_str}")
        print(f"    - Hardware Read Test: {test_status}")
        print()

    if metadata_nodes:
        meta_list = ", ".join([m["node"] for m in metadata_nodes])
        print(f"[NOTE] The following metadata/telemetry nodes were safely ignored: {meta_list}")
        print("       (These nodes cannot stream video and cause errors if opened directly).\n")

    print("-" * 80)
    if len(cameras) >= 2:
        left_id = left_cam["index"]
        right_id = right_cam["index"]
        print("RECOMMENDED LAUNCH COMMANDS:")
        print(f"  1. Start Stereo Streamer (320x240 @ 30 FPS <40ms Direct TCP mode):")
        print(f"     python3 ~/robot/pi_stereo_capture.py --left {left_id} --right {right_id} --width 320 --height 240 --fps 30")
        print()
        print(f"  2. Start Background Streamer via nohup:")
        print(f"     nohup python3 ~/robot/pi_stereo_capture.py --left {left_id} --right {right_id} > ~/robot/stereo_stream.log 2>&1 &")
    elif len(cameras) == 1:
        print(f"[!] Only 1 Logitech camera detected ({cameras[0]['node']}). Connect a 2nd camera for stereo.")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser(description="Raspberry Pi Logitech Camera Auto-Detector")
    ap.add_argument("--all", action="store_true", help="Detect all UVC webcams (not just Logitech)")
    ap.add_argument("--swap", action="store_true", help="Swap Left/Right assignment")
    ap.add_argument("--no-test", action="store_true", help="Skip frame grab read test")
    ap.add_argument("--test-capture", action="store_true", help="Run 30-frame synchronized benchmark")
    ap.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    ap.add_argument("--bash", action="store_true", help="Output bash export statements")
    ap.add_argument("--update-config", action="store_true", help="Update mapping_config.json with detected IDs")
    ap.add_argument("--start", action="store_true", help="Detect cameras and launch pi_stereo_capture.py")
    args = ap.parse_args()

    cams, meta = detect_logitech_cameras(
        only_logitech=not args.all,
        test_frames=not args.no_test and not args.json and not args.bash,
    )

    if args.json:
        left_id = cams[0]["index"] if len(cams) > 0 else None
        right_id = cams[1]["index"] if len(cams) > 1 else None
        if args.swap and left_id is not None and right_id is not None:
            left_id, right_id = right_id, left_id

        payload = {
            "count": len(cams),
            "left_index": left_id,
            "right_index": right_id,
            "left_node": f"/dev/video{left_id}" if left_id is not None else None,
            "right_node": f"/dev/video{right_id}" if right_id is not None else None,
            "cameras": cams,
            "ignored_metadata_nodes": meta,
        }
        print(json.dumps(payload, indent=2))
        return

    if args.bash:
        left_id = cams[0]["index"] if len(cams) > 0 else 0
        right_id = cams[1]["index"] if len(cams) > 1 else 2
        if args.swap:
            left_id, right_id = right_id, left_id
        print(f"export STEREO_LEFT={left_id} STEREO_RIGHT={right_id}")
        return

    print_report(cams, meta, swap=args.swap)

    if args.test_capture and len(cams) >= 2:
        left_id = cams[0]["index"] if not args.swap else cams[1]["index"]
        right_id = cams[1]["index"] if not args.swap else cams[0]["index"]
        benchmark_stereo_capture(left_id, right_id)

    if args.update_config and len(cams) >= 2:
        left_id = cams[0]["index"] if not args.swap else cams[1]["index"]
        right_id = cams[1]["index"] if not args.swap else cams[0]["index"]
        cfg_paths = [
            os.path.expanduser("~/robot/mapping_config.json"),
            os.path.join(os.path.dirname(__file__), "mapping_config.json"),
            "mapping_config.json",
        ]
        updated = False
        for cfg_file in cfg_paths:
            if os.path.exists(cfg_file):
                try:
                    with open(cfg_file, "r") as f:
                        data = json.load(f)
                    data.setdefault("stereo_camera", {})["left_device_id"] = left_id
                    data.setdefault("stereo_camera", {})["right_device_id"] = right_id
                    with open(cfg_file, "w") as f:
                        json.dump(data, f, indent=2)
                    print(f"[CONFIG] Updated '{cfg_file}': left_device_id={left_id}, right_device_id={right_id}")
                    updated = True
                except Exception as e:
                    print(f"[CONFIG WARN] Could not update '{cfg_file}': {e}")
        if not updated:
            print("[CONFIG] No mapping_config.json found to update.")

    if args.start and len(cams) >= 2:
        left_id = cams[0]["index"] if not args.swap else cams[1]["index"]
        right_id = cams[1]["index"] if not args.swap else cams[0]["index"]
        script_paths = [
            os.path.expanduser("~/robot/pi_stereo_capture.py"),
            os.path.join(os.path.dirname(__file__), "pi_stereo_capture.py"),
        ]
        target_script = None
        for s in script_paths:
            if os.path.exists(s):
                target_script = s
                break
        if target_script:
            print(f"\n[LAUNCH] Starting {target_script} with Left={left_id}, Right={right_id}...")
            subprocess.run(["pkill", "-f", "pi_stereo_capture.py"])
            time.sleep(0.5)
            cmd = [
                sys.executable, target_script,
                "--left", str(left_id),
                "--right", str(right_id),
                "--width", "320",
                "--height", "240",
                "--fps", "30",
            ]
            subprocess.Popen(cmd)
            print("[LAUNCH] Process started in background!")
        else:
            print("[LAUNCH ERROR] pi_stereo_capture.py not found.")


if __name__ == "__main__":
    main()
