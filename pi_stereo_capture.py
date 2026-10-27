#!/usr/bin/env python3
"""
pi_stereo_capture.py
Sub-40ms Ultra-Low-Latency Dual Logitech C930e Stereo Streamer for Raspberry Pi 4.
- Captures 320x240 @ 30 FPS per camera (640x240 Side-by-Side) with V4L2 ring-buffer drain threads.
- Streams length-prefixed JPEG frames over a raw TCP_NODELAY socket on port 8555 (~1.5ms encode, zero queue lag).
"""

import argparse
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import cv2
import numpy as np


def auto_detect_stereo_devices():
    """Detect real V4L2 video capture nodes for Logitech C930e cameras."""
    import re
    try:
        out = subprocess.check_output(["v4l2-ctl", "--list-devices"], text=True, stderr=subprocess.DEVNULL)
        blocks = out.strip().split("\n\n")
        logi_nodes = []
        for block in blocks:
            if "Logitech" in block or "C930e" in block:
                m = re.search(r"/dev/video(\d+)", block)
                if m:
                    logi_nodes.append(int(m.group(1)))
        if len(logi_nodes) >= 2:
            return logi_nodes
    except Exception:
        pass

    valid = []
    for idx in range(12):
        dev = f"/dev/video{idx}"
        if not os.path.exists(dev):
            continue
        try:
            out = subprocess.check_output(
                ["v4l2-ctl", "-d", dev, "--list-formats"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if "MJPG" in out:
                valid.append(idx)
        except Exception:
            pass
    return valid


def lock_camera_settings(
    device_id: int,
    focus_val: int = 0,
    auto_exposure: bool = True,
    exposure_val: int = 250,
    brightness_val: int = 128,
):
    dev = f"/dev/video{device_id}"
    cmds = [
        ["v4l2-ctl", "-d", dev, f"--set-ctrl=brightness={brightness_val}"],
        ["v4l2-ctl", "-d", dev, "--set-ctrl=focus_automatic_continuous=0"],
        ["v4l2-ctl", "-d", dev, "--set-ctrl=focus_auto=0"],
        ["v4l2-ctl", "-d", dev, f"--set-ctrl=focus_absolute={focus_val}"],
        ["v4l2-ctl", "-d", dev, "--set-ctrl=exposure_dynamic_framerate=0"],
    ]
    if auto_exposure:
        # V4L2 menu item 3: Aperture Priority Mode (Full Auto Exposure on UVC webcams)
        cmds.extend([
            ["v4l2-ctl", "-d", dev, "--set-ctrl=auto_exposure=3"],
            ["v4l2-ctl", "-d", dev, "--set-ctrl=exposure_auto=3"],
        ])
    else:
        # V4L2 menu item 1: Manual Mode
        cmds.extend([
            ["v4l2-ctl", "-d", dev, "--set-ctrl=auto_exposure=1"],
            ["v4l2-ctl", "-d", dev, "--set-ctrl=exposure_auto=1"],
            ["v4l2-ctl", "-d", dev, f"--set-ctrl=exposure_time_absolute={exposure_val}"],
            ["v4l2-ctl", "-d", dev, f"--set-ctrl=exposure_absolute={exposure_val}"],
        ])

    cmds.extend([
        ["v4l2-ctl", "-d", dev, "--set-ctrl=white_balance_automatic=0"],
        ["v4l2-ctl", "-d", dev, "--set-ctrl=white_balance_temperature=4600"],
    ])
    for cmd in cmds:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


class _CameraDrainWorker:
    def __init__(self, dev_id: int, width: int, height: int, fps: int):
        self.dev_id = int(dev_id)
        self.cap = cv2.VideoCapture(self.dev_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.frame = None
        self.ts = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            ret, frm = self.cap.read()
            if ret and frm is not None:
                now = time.monotonic()
                with self.lock:
                    self.frame = frm
                    self.ts = now
            else:
                time.sleep(0.001)

    def get_latest(self):
        with self.lock:
            return self.frame, self.ts

    def release(self):
        self.running = False
        try:
            self.thread.join(timeout=0.3)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass


class StereoCapture:
    def __init__(
        self,
        left_id=1,
        right_id=3,
        width=320,
        height=240,
        fps=30,
        lock_focus=True,
        auto_exposure=True,
        exposure_val=250,
        brightness_val=128,
    ):
        self.left_id = int(left_id)
        self.right_id = int(right_id)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.auto_exposure = bool(auto_exposure)
        self.exposure_val = int(exposure_val)
        self.brightness_val = int(brightness_val)

        if lock_focus:
            lock_camera_settings(
                self.left_id,
                auto_exposure=self.auto_exposure,
                exposure_val=self.exposure_val,
                brightness_val=self.brightness_val,
            )
            lock_camera_settings(
                self.right_id,
                auto_exposure=self.auto_exposure,
                exposure_val=self.exposure_val,
                brightness_val=self.brightness_val,
            )

        self.cam_l = _CameraDrainWorker(self.left_id, self.width, self.height, self.fps)
        self.cam_r = _CameraDrainWorker(self.right_id, self.width, self.height, self.fps)
        self._last_ts_l = 0.0
        self._last_ts_r = 0.0

        if lock_focus:
            lock_camera_settings(
                self.left_id,
                auto_exposure=self.auto_exposure,
                exposure_val=self.exposure_val,
                brightness_val=self.brightness_val,
            )
            lock_camera_settings(
                self.right_id,
                auto_exposure=self.auto_exposure,
                exposure_val=self.exposure_val,
                brightness_val=self.brightness_val,
            )

    def read(self, max_wait_sec: float = 0.035):
        t0 = time.monotonic()
        while time.monotonic() - t0 < max_wait_sec:
            fl, ts_l = self.cam_l.get_latest()
            fr, ts_r = self.cam_r.get_latest()
            if fl is None or fr is None:
                time.sleep(0.001)
                continue
            new_l = ts_l > self._last_ts_l
            new_r = ts_r > self._last_ts_r
            if new_l and new_r:
                self._last_ts_l = ts_l
                self._last_ts_r = ts_r
                return True, fl, fr, max(ts_l, ts_r)
            # If one camera hasn't delivered in > 65ms, don't stall the streamer
            if (new_l or new_r) and (time.monotonic() - min(ts_l, ts_r) > 0.065):
                self._last_ts_l = ts_l
                self._last_ts_r = ts_r
                return True, fl, fr, max(ts_l, ts_r)
            time.sleep(0.001)
        return False, None, None, time.monotonic()

    def release(self):
        self.cam_l.release()
        self.cam_r.release()


def main():
    ap = argparse.ArgumentParser(description="Pi Dual C930e <40ms Direct TCP Stereo Streamer")
    ap.add_argument("--left", type=int, default=None, help="Left camera /dev/videoX index (default: auto-detect)")
    ap.add_argument("--right", type=int, default=None, help="Right camera /dev/videoX index (default: auto-detect)")
    ap.add_argument("--swap", action="store_true", help="Swap Left and Right camera indices")
    ap.add_argument("--width", type=int, default=320, help="Per-eye width (default 320 -> 640x240 SBS)")
    ap.add_argument("--height", type=int, default=240, help="Per-eye height (default 240)")
    ap.add_argument("--fps", type=int, default=24, help="Stereo FPS (default: 24)")
    ap.add_argument("--port", type=int, default=8555, help="Direct TCP_NODELAY JPEG port (default: 8555)")
    ap.add_argument("--jpeg-quality", type=int, default=65, help="JPEG quality 1-100 (default: 65 for low Wi-Fi bandwidth)")
    ap.add_argument("--auto-exposure", action="store_true", default=True, help="Enable automatic exposure (default: True)")
    ap.add_argument("--manual-exposure", action="store_true", help="Force manual exposure mode")
    ap.add_argument("--exposure", type=int, default=250, help="Manual exposure shutter time (default: 250, used if --manual-exposure)")
    ap.add_argument("--brightness", type=int, default=128, help="Camera brightness 0-255 (default: 128)")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    detected = auto_detect_stereo_devices()
    left_id = args.left if args.left is not None else (detected[0] if len(detected) > 0 else 0)
    right_id = args.right if args.right is not None else (detected[1] if len(detected) > 1 else 2)

    if args.swap:
        left_id, right_id = right_id, left_id

    auto_exp = not args.manual_exposure
    print(f"[STEREO INIT] Opening /dev/video{left_id} (L) and /dev/video{right_id} (R)... (detected: {detected}, auto_exposure={auto_exp})", flush=True)
    stereo = StereoCapture(
        left_id,
        right_id,
        args.width,
        args.height,
        args.fps,
        auto_exposure=auto_exp,
        exposure_val=args.exposure,
        brightness_val=args.brightness,
    )
    time.sleep(0.3)

    if args.test:
        ok_count = 0
        t0 = time.monotonic()
        while ok_count < 60 and (time.monotonic() - t0) < 5.0:
            ok, fl, fr, _ = stereo.read()
            if ok:
                ok_count += 1
            else:
                time.sleep(0.002)
        dt = max(time.monotonic() - t0, 1e-3)
        print(f"[STEREO TEST] Captured {ok_count}/60 pairs ({args.width}x{args.height}) at {ok_count / dt:.1f} FPS.", flush=True)
        stereo.release()
        sys.exit(0 if ok_count > 0 else 1)

    latest_jpeg = None
    latest_ts = 0.0
    latest_seq = 0
    cond = threading.Condition()

    def _client_handler(conn, addr):
        import select
        print(f"[STEREO TCP] Low-latency client connected from {addr[0]}:{addr[1]}!", flush=True)
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)
            except Exception:
                pass
            conn.setblocking(False)
            sent_seq = -1
            while True:
                with cond:
                    cond.wait_for(lambda: latest_seq > sent_seq, timeout=0.25)
                    if latest_jpeg is None or latest_seq <= sent_seq:
                        continue
                    payload = latest_jpeg
                    sent_seq = latest_seq
                    ts = latest_ts

                # Non-blocking write check: if TCP socket is not ready within 2ms, drop this frame!
                _, writable, _ = select.select([], [conn], [], 0.002)
                if not writable:
                    continue  # Wi-Fi link busy: skip frame to maintain zero latency backlog

                # Header: 4-byte Magic (0x5354524D 'STRM') + 4-byte len + 8-byte float64 timestamp
                header = struct.pack("!IId", 0x5354524D, len(payload), ts)
                try:
                    conn.sendall(header + payload)
                except (BlockingIOError, socket.error):
                    continue
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
            print(f"[STEREO TCP] Client {addr[0]} disconnected.", flush=True)

    def _accept_loop():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", args.port))
        srv.listen(4)
        print(f"[STEREO TCP] Listening on 0.0.0.0:{args.port} ({args.width*2}x{args.height} @ {args.fps}FPS, <40ms mode)", flush=True)
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=_client_handler, args=(conn, addr), daemon=True).start()

    threading.Thread(target=_accept_loop, daemon=True).start()

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
    try:
        while True:
            ok, fl, fr, ts_cap = stereo.read()
            if not ok:
                time.sleep(0.002)
                continue
            sbs = np.hstack([fl, fr])
            ret_enc, buf = cv2.imencode(".jpg", sbs, encode_params)
            if ret_enc:
                with cond:
                    latest_jpeg = buf.tobytes()
                    latest_ts = time.time()
                    latest_seq += 1
                    cond.notify_all()
    except KeyboardInterrupt:
        print("\n[STEREO TCP] Stopped.", flush=True)
    finally:
        stereo.release()


if __name__ == "__main__":
    main()
