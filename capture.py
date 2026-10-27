"""
capture.py
Ultra-Low-Latency, Jitter-Stabilized RTSP/Webcam Frame Grabber for Visual SLAM.

Zero Hardcoded IPs:
- Dynamically resolves the Raspberry Pi RTSP endpoint via `config_manager.build_rtsp_candidates()`
  and `config_manager.resolve_pi_host()`, automatically recovering if the Pi changes its IP over DHCP!
"""

import os
import time
import threading
import socket
import urllib.parse

import cv2
import numpy as np

from config_manager import load_config, build_rtsp_candidates, resolve_pi_host


class LowLatencyCapture:
    """
    Background worker thread that continuously drains camera buffers and provides
    corruption-checked, timestamp-aligned frames for Visual SLAM.
    """

    def __init__(self, source="pi", reconnect_timeout=None, rtsp_transport=None, cfg=None, on_ip_change=None):
        self.cfg = cfg if cfg is not None else load_config()
        net_cfg = self.cfg.get("network", {})

        self._initial_source = source if source is not None else self.cfg.get("camera", {}).get("source", "pi")
        self.source = str(self._initial_source)
        self.on_ip_change = on_ip_change

        if reconnect_timeout is None:
            reconnect_timeout = float(net_cfg.get("reconnect_timeout_sec", 3.5))
        self.reconnect_timeout = max(float(reconnect_timeout), 2.0)

        if rtsp_transport is None:
            rtsp_transport = str(net_cfg.get("rtsp_transport", "udp"))
        self.rtsp_transport = str(rtsp_transport).lower().strip()
        if self.rtsp_transport not in ("udp", "tcp"):
            self.rtsp_transport = "udp"

        src_str = str(self._initial_source)
        self.is_webcam = src_str.isdigit()
        self.is_file = (not self.is_webcam) and os.path.isfile(src_str)
        self.is_network = (not self.is_webcam) and (not self.is_file)

        if self.is_network:
            self._apply_ffmpeg_env(self.rtsp_transport)

        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.new_frame_event = threading.Event()
        self.last_frame_time = time.time()
        self.last_frame_mono = time.monotonic()
        self.prev_frame_mono = self.last_frame_mono
        self.frame_dt = 1.0 / 30.0
        self.smooth_dt = 1.0 / 30.0
        self.inst_fps = 30.0
        self.frame_seq = 0
        self.dropped_corrupt = 0
        self.was_interrupted = True
        self.is_connected = False
        self.active_host = None

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

        print(f"[CAPTURE] Initializing low-latency capture (source={self.source}, transport={self.rtsp_transport.upper()})")
        t0 = time.time()
        while self.running and not self.is_connected and (time.time() - t0 < 5.0):
            time.sleep(0.05)

        if not self.is_connected:
            print(f"[CAPTURE WARN] Stream {self.source} pending... background auto-discovery & reconnect active.")

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_ffmpeg_env(transport="udp"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{transport}"
            "|fflags;nobuffer+discardcorrupt"
            "|flags;low_delay"
            "|max_delay;0"
            "|probesize;32768"
            "|analyzeduration;0"
            "|stimeout;2500000"
            "|reorder_queue_size;0"
            "|threads;2"
        )

    @staticmethod
    def _probe_rtsp(url, timeout=0.15):
        try:
            p = urllib.parse.urlparse(url)
            host = p.hostname or "127.0.0.1"
            port = p.port or 8554
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect((host, port))
                return True
        except Exception:
            return False

    @staticmethod
    def _is_frame_corrupt(frame):
        if frame is None or frame.size == 0:
            return True
        h, w = frame.shape[:2]
        if h < 64 or w < 64:
            return True
        bottom_strip = frame[h - 24:h, ::4]
        if float(np.std(bottom_strip)) < 1.2:
            return True
        return False

    def _open_stream(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        try:
            if self.is_webcam:
                src_idx = int(self._initial_source)
                cap = None
                if hasattr(cv2, "CAP_DSHOW"):
                    cap = cv2.VideoCapture(src_idx, cv2.CAP_DSHOW)
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(src_idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap = cap
                    return True
            elif self.is_file:
                cap = cv2.VideoCapture(str(self._initial_source))
                if cap.isOpened():
                    self.cap = cap
                    return True
            else:
                candidates = build_rtsp_candidates(self.cfg, source_override=str(self._initial_source))
                valid_candidates = [c for c in candidates if self._probe_rtsp(c)]
                if not valid_candidates:
                    return False

                transports_to_try = [self.rtsp_transport]
                alt = "tcp" if self.rtsp_transport == "udp" else "udp"
                transports_to_try.append(alt)

                for trans in transports_to_try:
                    self._apply_ffmpeg_env(trans)
                    for cand in valid_candidates:
                        cap = cv2.VideoCapture(cand, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        if cap.isOpened():
                            ret, test_frame = cap.read()
                            if ret and not self._is_frame_corrupt(test_frame):
                                self.cap = cap
                                self.source = cand
                                parsed_host = urllib.parse.urlparse(cand).hostname
                                if parsed_host and parsed_host != self.active_host:
                                    self.active_host = parsed_host
                                    print(f"[CAPTURE] Locked onto RTSP stream: {cand} ({trans.upper()})")
                                    if callable(self.on_ip_change):
                                        try:
                                            self.on_ip_change(parsed_host)
                                        except Exception:
                                            pass
                                return True
                            cap.release()
        except Exception:
            pass
        return False

    def _try_direct_tcp_stream(self, host: str, port: int = 8555) -> bool:
        """Connect to pi_stereo_capture.py's <40ms TCP_NODELAY JPEG socket on port 8555."""
        import select
        import socket
        import struct

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.5)
        try:
            sock.connect((host, int(port)))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32768)
            except Exception:
                pass
            sock.setblocking(False)
            print(f"[CAPTURE] Connected to Ultra-Low-Latency Direct TCP Stereo Stream ({host}:{port})!", flush=True)
            if host != self.active_host:
                self.active_host = host
                if callable(self.on_ip_change):
                    try:
                        self.on_ip_change(host)
                    except Exception:
                        pass
        except Exception:
            try:
                sock.close()
            except Exception:
                pass
            return False

        was_ever_connected = self.is_connected
        received_any = False
        buf = bytearray()
        last_data_time = time.time()

        try:
            while self.running:
                # 1. Wait for incoming socket data with a tight select timeout
                readable, _, _ = select.select([sock], [], [], 0.05)
                if not readable:
                    if time.time() - last_data_time > 2.5:
                        break  # Stream stalled
                    continue

                # 2. Aggressively drain ALL available chunks from kernel socket buffer to prevent any bufferbloat
                while True:
                    try:
                        chunk = sock.recv(65536)
                        if not chunk:
                            return received_any
                        buf.extend(chunk)
                        last_data_time = time.time()
                    except (BlockingIOError, socket.error):
                        break

                # 3. Extract the NEWEST complete frame, instantly skipping any older frames
                latest_jpg = None
                latest_sender_ts = 0.0

                while True:
                    if len(buf) < 4:
                        break

                    magic = struct.unpack("!I", buf[:4])[0]
                    if magic == 0x5354524D:  # "STRM" 16-byte header
                        if len(buf) < 16:
                            break
                        _, frame_len, frame_ts = struct.unpack("!IId", buf[:16])
                        if frame_len <= 0 or frame_len > 2000000:
                            buf.clear()
                            break
                        if len(buf) < 16 + frame_len:
                            break
                        latest_jpg = buf[16 : 16 + frame_len]
                        latest_sender_ts = frame_ts
                        del buf[: 16 + frame_len]
                    else:
                        # Legacy 4-byte header
                        frame_len = magic
                        if frame_len <= 0 or frame_len > 2000000:
                            idx = buf.find(b"STRM")
                            if idx > 0:
                                del buf[:idx]
                            else:
                                buf.clear()
                            break
                        if len(buf) < 4 + frame_len:
                            break
                        latest_jpg = buf[4 : 4 + frame_len]
                        latest_sender_ts = 0.0
                        del buf[: 4 + frame_len]

                if latest_jpg is None:
                    continue

                t_grab = time.monotonic()
                arr = np.frombuffer(latest_jpg, dtype=np.uint8)
                frm = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frm is None:
                    continue

                # If downscaled 640x240 SBS (320x240 per eye), upscale to 1280x480 so all 640x480 intrinsics match
                if frm.shape[1] == 640 and frm.shape[0] == 240:
                    frm = cv2.resize(frm, (1280, 480), interpolation=cv2.INTER_LINEAR)

                now = time.time()
                with self.lock:
                    if self.last_frame_mono > 0.0:
                        self.frame_dt = max(0.001, min(0.5, t_grab - self.last_frame_mono))
                    self.smooth_dt = 0.25 * self.frame_dt + 0.75 * self.smooth_dt
                    self.inst_fps = 1.0 / max(self.smooth_dt, 1e-3)
                    self.frame_seq += 1
                    self.last_frame_mono = t_grab
                    self.last_frame_time = now
                    self.latest_frame = frm
                    self.ret = True
                    received_any = True
                    if not self.is_connected:
                        if was_ever_connected:
                            self.was_interrupted = True
                        self.is_connected = True
                        was_ever_connected = True
                self.new_frame_event.set()
            return received_any
        except Exception:
            return received_any
        finally:
            self.is_connected = False
            try:
                sock.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _worker(self):
        consecutive_fails = 0
        while self.running:
            if not self.is_webcam and not self.is_file:
                try:
                    pi_ip = resolve_pi_host(self.cfg, save_if_changed=True)
                    if pi_ip and self._try_direct_tcp_stream(pi_ip, port=8555):
                        continue
                except Exception:
                    pass
            if self.cap is None or not self.cap.isOpened():
                if not self._open_stream():
                    time.sleep(0.25)
                    continue
                else:
                    with self.lock:
                        self.was_interrupted = True
                        self.last_frame_time = time.time()
                        self.last_frame_mono = time.monotonic()
                    consecutive_fails = 0

            grabbed = self.cap.grab()
            if grabbed:
                ret, frame = self.cap.retrieve()
            else:
                ret, frame = False, None

            now = time.time()
            mono_now = time.monotonic()

            if ret and frame is not None and frame.size > 0:
                consecutive_fails = 0
                if self.is_network and self._is_frame_corrupt(frame):
                    self.dropped_corrupt += 1
                    continue

                with self.lock:
                    dt = mono_now - self.last_frame_mono
                    if dt > 0.28 and self.frame_seq > 0:
                        self.was_interrupted = True
                    self.prev_frame_mono = self.last_frame_mono
                    self.last_frame_mono = mono_now
                    clamped_dt = max(min(dt, 0.35), 0.005)
                    self.frame_dt = clamped_dt
                    self.smooth_dt = 0.25 * clamped_dt + 0.75 * self.smooth_dt
                    self.inst_fps = 1.0 / max(self.smooth_dt, 1e-3)
                    self.frame_seq += 1
                    self.latest_frame = frame
                    self.last_frame_time = now
                    self.is_connected = True
                self.new_frame_event.set()
                if self.is_file:
                    time.sleep(1.0 / 30.0)
            else:
                if self.is_file and self.cap is not None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    with self.lock:
                        self.was_interrupted = True
                    continue

                consecutive_fails += 1
                if consecutive_fails >= 45 or (now - self.last_frame_time > self.reconnect_timeout):
                    print(f"[CAPTURE WARN] Stream stalled ({now - self.last_frame_time:.1f}s). Re-probing Pi IP & reconnecting...")
                    with self.lock:
                        self.is_connected = False
                        self.was_interrupted = True
                    self._open_stream()
                    self.last_frame_time = time.time()
                    self.last_frame_mono = time.monotonic()
                    consecutive_fails = 0
                    time.sleep(0.1)
                else:
                    time.sleep(0.004)

    # ------------------------------------------------------------------
    def read(self, timeout=0.025):
        if self.new_frame_event.wait(timeout=timeout):
            with self.lock:
                self.new_frame_event.clear()
                if self.latest_frame is None:
                    return False, None, False
                frame_out = self.latest_frame.copy()
                interrupted = self.was_interrupted
                self.was_interrupted = False
                return True, frame_out, interrupted
        return False, None, False

    def release(self):
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
