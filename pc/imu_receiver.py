"""
imu_receiver.py
Timestamp-Interpolated UDP + TCP Receiver for the Raspberry Pi 4 IMU + TOF400C Stream
(supports 16-field Raw + Fused BNO055/MPU-6500 telemetry and TOF400C laser rangefinder).
"""

import collections
import math
import socket
import threading
import time


def _angle_diff(a, b):
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def _interp_angle(a0, a1, alpha):
    return (a0 + alpha * _angle_diff(a1, a0) + math.pi) % (2.0 * math.pi) - math.pi


class BNO055Receiver:
    def __init__(self, udp_port=5005, pi_host="auto",
                 mount_pitch_deg=15.0, video_latency_sec=0.080,
                 imu_mounted_on_camera=False, on_ip_discovered=None,
                 imu_mount_yaw_deg=0.0, imu_mount_pitch_deg=-4.4,
                 imu_mount_roll_deg=5.7, auto_calibrate_mount=True):
        self.udp_port = int(udp_port)
        self.pi_host = str(pi_host)
        self.mount_pitch_rad = math.radians(mount_pitch_deg)
        self.video_latency_sec = float(video_latency_sec)
        self.imu_mounted_on_camera = bool(imu_mounted_on_camera)
        self.on_ip_discovered = on_ip_discovered

        # Configurable mounting angle compensation for BNO055 installed at an angle
        self.imu_mount_yaw_deg = float(imu_mount_yaw_deg)
        self.imu_mount_pitch_deg = float(imu_mount_pitch_deg)
        self.imu_mount_roll_deg = float(imu_mount_roll_deg)
        self.auto_calibrate_mount = bool(auto_calibrate_mount)
        self._mount_calibrated = False
        self._calib_samples = []

        self.lock = threading.Lock()
        self.buffer = collections.deque(maxlen=600)
        self.running = True
        self.last_recv_time = 0.0
        self._last_seq = -1
        self._synth_seq = 0
        self._pending_reset = False
        self._logged_first_packet = False
        self._logged_first_tof = False
        self._last_target_time = None

        if self.udp_port > 0:
            self.udp_thread = threading.Thread(target=self._udp_worker, daemon=True)
            self.udp_thread.start()
            self.tcp_thread = threading.Thread(target=self._tcp_fallback_worker, daemon=True)
            self.tcp_thread.start()

    def adjust_video_latency(self, delta_sec):
        self.video_latency_sec = float(max(0.0, min(0.50, self.video_latency_sec + delta_sec)))
        print(f"[IMU-SYNC] Camera-to-IMU latency offset -> {self.video_latency_sec * 1000.0:.0f} ms")
        return self.video_latency_sec

    def _parse_packet(self, text: str):
        try:
            for line in text.strip().splitlines():
                parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
                tof_m = -1.0
                raw_accel = (0.0, 0.0, 9.81)
                raw_gyro = (0.0, 0.0, 0.0)
                has_raw_imu = False

                if len(parts) >= 16:
                    seq = int(parts[0])
                    pi_ts = float(parts[1])
                    yaw_deg = float(parts[2])
                    pitch_deg = float(parts[3])
                    roll_deg = float(parts[4])
                    gz_rads = float(parts[5])
                    gy_rads = float(parts[6])
                    lin_acc = float(parts[7])
                    calib = int(parts[8])
                    tof_m = float(parts[9])
                    raw_accel = (float(parts[10]), float(parts[11]), float(parts[12]))
                    raw_gyro = (float(parts[13]), float(parts[14]), float(parts[15]))
                    has_raw_imu = True
                elif len(parts) >= 10:
                    seq = int(parts[0])
                    pi_ts = float(parts[1])
                    yaw_deg = float(parts[2])
                    pitch_deg = float(parts[3])
                    roll_deg = float(parts[4])
                    gz_rads = float(parts[5])
                    gy_rads = float(parts[6])
                    lin_acc = float(parts[7])
                    calib = int(parts[8])
                    tof_m = float(parts[9])
                elif len(parts) >= 9:
                    seq = int(parts[0])
                    pi_ts = float(parts[1])
                    yaw_deg = float(parts[2])
                    pitch_deg = float(parts[3])
                    roll_deg = float(parts[4])
                    gz_rads = float(parts[5])
                    gy_rads = float(parts[6])
                    lin_acc = float(parts[7])
                    calib = int(parts[8])
                elif len(parts) >= 6:
                    self._synth_seq += 1
                    seq = self._synth_seq
                    pi_ts = 0.0
                    yaw_deg = float(parts[0])
                    pitch_deg = float(parts[1])
                    roll_deg = float(parts[2])
                    gz_rads = float(parts[3])
                    gy_rads = 0.0
                    lin_acc = float(parts[4])
                    calib = int(parts[5])
                else:
                    continue

                seq_reset = False
                if seq <= self._last_seq:
                    if seq <= 5 and self._last_seq > 50:
                        seq_reset = True
                        self._last_seq = seq
                    else:
                        continue
                else:
                    self._last_seq = seq

                # Auto-calibrate stationary mounting angles on startup (first 35 resting samples)
                if self.auto_calibrate_mount and not self._mount_calibrated:
                    if abs(gz_rads) < 0.02 and abs(gy_rads) < 0.02 and lin_acc < 0.25:
                        self._calib_samples.append((pitch_deg, roll_deg, raw_accel))
                        if len(self._calib_samples) >= 35:
                            avg_p = float(sum(s[0] for s in self._calib_samples) / len(self._calib_samples))
                            avg_r = float(sum(s[1] for s in self._calib_samples) / len(self._calib_samples))
                            self.imu_mount_pitch_deg = avg_p
                            self.imu_mount_roll_deg = avg_r
                            self._mount_calibrated = True
                            print(f"[IMU-MOUNT] Auto-calibrated BNO055 rest tilt: pitch={avg_p:+.2f}deg, roll={avg_r:+.2f}deg (Chassis Level Zeroed)", flush=True)

                # Compensate for BNO055 installed at an angle
                adj_yaw_deg = (yaw_deg - self.imu_mount_yaw_deg) % 360.0
                yaw_rad = math.radians(-adj_yaw_deg)
                yaw_rad = (yaw_rad + math.pi) % (2.0 * math.pi) - math.pi

                chassis_pitch_deg = pitch_deg - self.imu_mount_pitch_deg
                chassis_roll_deg = roll_deg - self.imu_mount_roll_deg

                if self.imu_mounted_on_camera:
                    cam_pitch_rad = math.radians(chassis_pitch_deg)
                else:
                    cam_pitch_rad = self.mount_pitch_rad + math.radians(chassis_pitch_deg)

                gyro_vec_mag = (
                    math.sqrt(raw_gyro[0] ** 2 + raw_gyro[1] ** 2 + raw_gyro[2] ** 2)
                    if has_raw_imu
                    else math.hypot(gz_rads, gy_rads)
                )

                now = time.monotonic()
                with self.lock:
                    if seq_reset:
                        self._pending_reset = True
                    sample = {
                        "recv_mono": now,
                        "pi_ts": pi_ts,
                        "seq": seq,
                        "yaw_rad": yaw_rad,
                        "pitch_rad": cam_pitch_rad,
                        "roll_rad": math.radians(chassis_roll_deg),
                        "raw_pitch_deg": pitch_deg,
                        "raw_roll_deg": roll_deg,
                        "mount_pitch_deg": self.imu_mount_pitch_deg,
                        "mount_roll_deg": self.imu_mount_roll_deg,
                        "gz_rads": gz_rads,
                        "gy_rads": gy_rads,
                        "gyro_vec_mag": gyro_vec_mag,
                        "lin_acc": lin_acc,
                        "calib": calib,
                        "tof_m": tof_m,
                        "raw_accel": raw_accel,
                        "raw_gyro": raw_gyro,
                        "has_raw_imu": has_raw_imu,
                        "reset": self._pending_reset,
                    }
                    self.buffer.append(sample)
                    self.last_recv_time = now

                if not self._logged_first_packet:
                    self._logged_first_packet = True
                    mode_tag = "RAW+FUSED 16-Field" if has_raw_imu else "10-Field"
                    print(f"[IMU] LOCKED! Receiving live 100Hz IMU ({mode_tag}, Yaw={yaw_deg:.1f} deg)", flush=True)
                if not self._logged_first_tof and tof_m > 0.0:
                    self._logged_first_tof = True
                    print(f"[TOF400C] LOCKED! Receiving live laser ranging telemetry (ToF={tof_m:.2f} m)", flush=True)
        except Exception:
            pass

    def _udp_worker(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32768)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        try:
            sock.bind(("0.0.0.0", self.udp_port))
            sock.settimeout(0.1)
            print(f"[IMU] Listening on UDP :{self.udp_port} (with auto-PING to {self.pi_host}:{self.udp_port})")
        except Exception as e:
            print(f"[IMU] UDP bind warning on port {self.udp_port}: {e} (TCP fallback active)")
            sock.close()
            return

        last_ping = 0.0
        while self.running:
            now = time.monotonic()
            if self.pi_host and self.pi_host.lower() not in ("auto", "pi", "") and (now - last_ping >= 1.0):
                last_ping = now
                try:
                    sock.sendto(b"PING", (self.pi_host, self.udp_port))
                except Exception:
                    pass

            try:
                data, addr = sock.recvfrom(512)
                if data and data != b"PING":
                    sender_ip = addr[0]
                    if sender_ip not in ("127.0.0.1", "0.0.0.0") and sender_ip != self.pi_host:
                        self.pi_host = sender_ip
                        if callable(self.on_ip_discovered):
                            try:
                                self.on_ip_discovered(sender_ip)
                            except Exception:
                                pass
                    self._parse_packet(data.decode("ascii", errors="ignore"))
            except socket.timeout:
                continue
            except Exception:
                time.sleep(0.005)
        sock.close()

    def _tcp_fallback_worker(self):
        time.sleep(1.2)
        while self.running:
            if (time.monotonic() - self.last_recv_time) < 0.8:
                time.sleep(0.5)
                continue

            if not self.pi_host or self.pi_host.lower() in ("auto", "pi", ""):
                time.sleep(1.0)
                continue

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.5)
            try:
                sock.connect((self.pi_host, self.udp_port))
                sock.settimeout(0.5)
                print(f"[IMU] Connected to Pi IMU+ToF via TCP fallback ({self.pi_host}:{self.udp_port})!", flush=True)
                buf = ""
                while self.running:
                    chunk = sock.recv(512)
                    if not chunk:
                        break
                    buf += chunk.decode("ascii", errors="ignore")
                    if "\n" in buf:
                        lines = buf.split("\n")
                        buf = lines[-1]
                        for ln in lines[:-1]:
                            self._parse_packet(ln)
            except Exception:
                time.sleep(1.5)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass

    def _interpolate_at_locked(self, target_time):
        n = len(self.buffer)
        if n == 1 or target_time <= self.buffer[0]["recv_mono"]:
            return dict(self.buffer[0])
        if target_time >= self.buffer[-1]["recv_mono"]:
            return dict(self.buffer[-1])

        s1 = self.buffer[-1]
        s0 = self.buffer[0]
        for i in range(n - 1, 0, -1):
            if self.buffer[i - 1]["recv_mono"] <= target_time <= self.buffer[i]["recv_mono"]:
                s0 = self.buffer[i - 1]
                s1 = self.buffer[i]
                break

        dt_span = s1["recv_mono"] - s0["recv_mono"]
        if dt_span <= 1e-5:
            return dict(s1)

        alpha = max(0.0, min(1.0, (target_time - s0["recv_mono"]) / dt_span))
        out = dict(s1)
        out["recv_mono"] = target_time
        out["yaw_rad"] = _interp_angle(s0["yaw_rad"], s1["yaw_rad"], alpha)
        out["pitch_rad"] = s0["pitch_rad"] + alpha * (s1["pitch_rad"] - s0["pitch_rad"])
        out["roll_rad"] = s0["roll_rad"] + alpha * (s1["roll_rad"] - s0["roll_rad"])
        out["gz_rads"] = s0["gz_rads"] + alpha * (s1["gz_rads"] - s0["gz_rads"])
        out["gy_rads"] = s0["gy_rads"] + alpha * (s1["gy_rads"] - s0["gy_rads"])
        out["lin_acc"] = s0["lin_acc"] + alpha * (s1["lin_acc"] - s0["lin_acc"])
        out["raw_accel"] = tuple(
            s0["raw_accel"][k] + alpha * (s1["raw_accel"][k] - s0["raw_accel"][k]) for k in range(3)
        )
        out["raw_gyro"] = tuple(
            s0["raw_gyro"][k] + alpha * (s1["raw_gyro"][k] - s0["raw_gyro"][k]) for k in range(3)
        )
        if s0["tof_m"] > 0.0 and s1["tof_m"] > 0.0:
            out["tof_m"] = s0["tof_m"] + alpha * (s1["tof_m"] - s0["tof_m"])
        else:
            out["tof_m"] = s1["tof_m"] if s1["tof_m"] > 0.0 else s0["tof_m"]
        return out

    def get_synced_sample(self, frame_mono=None, max_age_sec=0.60):
        now = time.monotonic()
        with self.lock:
            if not self.buffer or (now - self.last_recv_time > max_age_sec):
                return None

            ref_mono = float(frame_mono) if frame_mono is not None else now
            target_time = ref_mono - self.video_latency_sec

            aligned = self._interpolate_at_locked(target_time)
            latest = self.buffer[-1]

            prev_t = self._last_target_time if self._last_target_time is not None else (target_time - 0.033)
            self._last_target_time = target_time

            window_max_gz = max(abs(aligned["gz_rads"]), float(aligned.get("gyro_vec_mag", 0.0)))
            window_max_acc = abs(aligned["lin_acc"])
            for s in reversed(self.buffer):
                t_s = s["recv_mono"]
                if t_s < prev_t - 0.015:
                    break
                if t_s <= target_time + 0.015:
                    g_cand = max(abs(s["gz_rads"]), float(s.get("gyro_vec_mag", 0.0)))
                    if g_cand > window_max_gz:
                        window_max_gz = g_cand
                    if abs(s["lin_acc"]) > window_max_acc:
                        window_max_acc = abs(s["lin_acc"])

            aligned["live_yaw_rad"] = latest["yaw_rad"]
            aligned["live_pitch_rad"] = latest["pitch_rad"]
            aligned["live_gz_rads"] = latest["gz_rads"]
            aligned["live_tof_m"] = latest["tof_m"]
            aligned["window_max_gz"] = window_max_gz
            aligned["window_max_acc"] = window_max_acc
            aligned["reset"] = self._pending_reset
            self._pending_reset = False
            return aligned

    def get_latest_live(self, max_age_sec=0.60):
        now = time.monotonic()
        with self.lock:
            if not self.buffer or (now - self.last_recv_time > max_age_sec):
                return None
            out = dict(self.buffer[-1])
            window_max_gz = max(abs(out["gz_rads"]), float(out.get("gyro_vec_mag", 0.0)))
            window_max_acc = abs(out["lin_acc"])
            for s in reversed(self.buffer):
                if (now - s["recv_mono"]) > 0.15:
                    break
                g_cand = max(abs(s["gz_rads"]), float(s.get("gyro_vec_mag", 0.0)))
                if g_cand > window_max_gz:
                    window_max_gz = g_cand
                if abs(s["lin_acc"]) > window_max_acc:
                    window_max_acc = abs(s["lin_acc"])
            out["window_max_gz"] = window_max_gz
            out["window_max_acc"] = window_max_acc
            return out

    def get_latest(self, max_age_sec=0.60):
        return self.get_synced_sample(max_age_sec=max_age_sec)

    def release(self):
        self.running = False
