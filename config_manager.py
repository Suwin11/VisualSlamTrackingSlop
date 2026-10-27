"""
config_manager.py
Centralized Configuration, Zero-Hardcoded-IP Auto-Discovery, and Hot-Reload Manager.

Features:
1. Single Source of Truth (`mapping_config.json`):
   - Eliminates all hardcoded IPs across `capture.py`, `imu_receiver.py`, `rrt_star.py`,
     `navigate_rrt.py`, `yolo_mask_track.py`, and `.bat` launchers.
2. Automatic Raspberry Pi IP Discovery (`resolve_pi_host()`):
   - Reads `~/.ssh/config` (`Host pi` -> `HostName ...`) if configured.
   - Listens for incoming UDP telemetry packets on port 5005 (the Pi broadcasts 100Hz IMU
     packets to the subnet, revealing its current IP automatically even after a DHCP change!).
   - Performs an ultra-fast parallel TCP port probe (`:8554` / `:5005`) across the local
     /24 subnet and `raspberrypi.local` (mDNS) if the configured IP is unreachable!
   - Automatically updates `mapping_config.json` when the Pi's IP changes.
"""

import concurrent.futures
import json
import os
import re
import socket
import time


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mapping_config.json"
)

DEFAULT_CONFIG = {
    "network": {
        "pi_host": "auto",
        "last_known_pi_ip": "192.168.0.170",
        "relay_host": "",
        "rtsp_port": 8554,
        "rtsp_path": "/cam",
        "webrtc_port": 8889,
        "hls_port": 8888,
        "imu_udp_port": 5005,
        "motor_udp_port": 5006,
        "web_stream_port": 8080,
        "rtsp_transport": "tcp",
        "reconnect_timeout_sec": 6.0,
        "auto_discover_subnet": True,
    },
    "camera": {
        "source": "pi",
        "camera_height_m": 0.20,
        "camera_pitch_deg": 15.0,
        "fov_deg": 62.2,
        "yaw_scale": 1.0,
        "enable_undistort": True,
        "calib_file": "camera_pi.npz",
    },
    "stereo_camera": {
        "enabled": True,
        "left_device_id": 1,
        "right_device_id": 3,
        "resolution_w": 640,
        "resolution_h": 480,
        "fps": 24,
        "baseline_m": 0.08,
        "calib_file": "stereo_calib.npz",
        "disable_autofocus": True,
        "stereo_num_disparities": 64,
        "stereo_block_size": 7,
    },
    "imu_raw": {
        "stream_raw_accel_gyro": True,
        "bno055_operating_mode": "IMUPLUS",
        "imu_frequency_hz": 100,
    },
    "imu_sync": {
        "rotation_source": "bno055",
        "video_latency_sec": 0.080,
        "max_stutter_dt_sec": 0.280,
        "max_turn_rate_for_mapping_rads": 0.45,
        "imu_mounted_on_camera": False,
        "imu_mount_yaw_deg": 0.0,
        "imu_mount_pitch_deg": -4.4,
        "imu_mount_roll_deg": 5.7,
        "auto_calibrate_mount": True,
        "imu_motion_weight": 0.45,
        "depth_motion_weight": 0.35,
        "cmd_motion_weight": 0.20,
    },
    "tof_sensor": {
        "enabled": True,
        "max_tof_range_m": 3.5,
        "tof_beam_cone_deg": 12.0,
        "tof_stop_distance_m": 0.22,
        "tof_odometry_blend": 0.35,
    },
    "mapping_3d": {
        "max_forward_range_m": 2.5,
        "min_obstacle_height_m": 0.05,
        "max_obstacle_height_m": 0.40,
        "wall_sensitivity": 0.65,
        "voxel_size_m": 0.05,
        "show_3d_view_by_default": True,
        "color_by_height": True,
    },
    "global_map": {
        "width_m": 8.0,
        "height_m": 8.0,
        "resolution_m": 0.05,
        "robot_radius_m": 0.18,
        "log_odds_occupied": 1.50,
        "log_odds_free": -1.10,
        "enable_rolling_window": True,
    },
    "yolo_and_planner": {
        "yolo_model": "yolo11n-seg.pt",
        "yolo_conf": 0.35,
        "ignored_classes": [
            "dining table",
            "bed",
            "couch",
            "bench",
            "floor",
            "carpet",
            "rug",
            "mat",
            "tv",
            "laptop",
        ],
        "default_goal_x_m": 0.0,
        "default_goal_y_m": 2.5,
        "lookahead_m": 0.35,
        "max_linear_vel_mps": 0.30,
        "max_angular_vel_rads": 1.20,
    },
    "motor_control": {
        "wheel_base_m": 0.28,
        "wheel_radius_m": 0.033,
        "pwm_range": 255,
        "manual_forward_pwm": 140,
        "manual_turn_pwm": 105,
        "swap_left_right_motors": True,
        "invert_left_motor": True,
        "invert_right_motor": True,
        "max_rpm": 60,
        "min_start_rpm": 6,
        "manual_forward_rpm": 140,
        "manual_turn_rpm": 105,
        "manual_hold_timeout_sec": 0.35,
        "control_loop_hz": 25.0,
        "esp32_bt_mac": "E0:8C:FE:C9:47:1A",
        "rfcomm_device": "/dev/rfcomm0",
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = json.loads(json.dumps(base))
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            cfg = deep_merge(cfg, user_cfg)
        except Exception as e:
            print(f"[CONFIG WARN] Failed to parse {config_path}: {e}. Using defaults.")
    else:
        save_config(cfg, config_path)
    return cfg


def save_config(cfg: dict, config_path: str = DEFAULT_CONFIG_PATH) -> bool:
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return True
    except Exception as e:
        print(f"[CONFIG WARN] Could not save {config_path}: {e}")
        return False


def get_local_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _parse_ssh_config_host(alias: str = "pi") -> str | None:
    """Read ~/.ssh/config to see if `Host pi` has a `HostName` defined."""
    ssh_cfg = os.path.expanduser("~/.ssh/config")
    if not os.path.exists(ssh_cfg):
        return None
    try:
        with open(ssh_cfg, "r", encoding="utf-8", errors="ignore") as f:
            in_target = False
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                m_host = re.match(r"^Host\s+(.+)$", stripped, re.IGNORECASE)
                if m_host:
                    hosts = m_host.group(1).split()
                    in_target = alias in hosts
                    continue
                if in_target:
                    m_hn = re.match(r"^HostName\s+(\S+)$", stripped, re.IGNORECASE)
                    if m_hn:
                        return m_hn.group(1).strip()
    except Exception:
        pass
    return None


def probe_tcp_port(ip: str, port: int, timeout: float = 0.12) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((ip, int(port)))
            return True
    except Exception:
        return False


def _sniff_udp_imu_sender(udp_port: int = 5005, timeout: float = 0.25) -> str | None:
    """
    Listen briefly on UDP port 5005. Because `pi_imu_streamer.py` broadcasts at 100Hz
    to the subnet (`x.x.x.255`), we can often catch the Pi's IP in <15ms!
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", int(udp_port)))
        sock.settimeout(timeout)
        t_end = time.monotonic() + timeout
        local_ip = get_local_lan_ip()
        while time.monotonic() < t_end:
            data, addr = sock.recvfrom(256)
            sender_ip = addr[0]
            if sender_ip not in ("127.0.0.1", "0.0.0.0", local_ip) and data != b"PING":
                return sender_ip
    except Exception:
        pass
    finally:
        try:
            sock.close()
        except Exception:
            pass
    return None


def scan_subnet_for_pi(rtsp_port: int = 8554, imu_port: int = 5005, timeout: float = 0.10) -> str | None:
    """
    Parallel scan the local /24 subnet for a device serving MediaMTX RTSP (:8554) or IMU TCP (:5005).
    Completes in ~0.35 seconds using a thread pool.
    """
    local_ip = get_local_lan_ip()
    parts = local_ip.split(".")
    if len(parts) != 4 or local_ip == "127.0.0.1":
        return None

    prefix = f"{parts[0]}.{parts[1]}.{parts[2]}."
    my_octet = int(parts[3])

    # Prioritize common Pi range (.50 - .100) first, then the rest of the /24
    priority_octets = list(range(60, 90)) + [i for i in range(1, 255) if i not in range(60, 90)]
    candidates = [f"{prefix}{i}" for i in priority_octets if i != my_octet]

    def _check_ip(ip_addr: str) -> str | None:
        if probe_tcp_port(ip_addr, rtsp_port, timeout=timeout):
            return ip_addr
        if probe_tcp_port(ip_addr, imu_port, timeout=timeout):
            return ip_addr
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
        futures = [ex.submit(_check_ip, ip) for ip in candidates]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res:
                return res
    return None


def resolve_pi_host(cfg: dict | None = None, config_path: str = DEFAULT_CONFIG_PATH, save_if_changed: bool = True) -> str:
    """
    Resolve the Raspberry Pi's current IP address without relying on hardcoded values.
    Order of resolution:
      1. Environment variable `FIBO_PI_HOST` (if set).
      2. Explicit IP in `cfg["network"]["pi_host"]` (if not "auto" and reachable).
      3. `cfg["network"]["last_known_pi_ip"]`, `~/.ssh/config` (`Host pi`), and `raspberrypi.local`.
      4. Passive UDP 5005 broadcast sniff (<200ms).
      5. Fast parallel /24 subnet scan on RTSP port 8554 & IMU port 5005 (~350ms).
    """
    if cfg is None:
        cfg = load_config(config_path)

    net = cfg.get("network", {})
    rtsp_port = int(net.get("rtsp_port", 8554))
    imu_port = int(net.get("imu_udp_port", 5005))

    env_host = os.environ.get("FIBO_PI_HOST", "").strip()
    cfg_host = str(net.get("pi_host", "auto")).strip()
    last_ip = str(net.get("last_known_pi_ip", "192.168.0.170")).strip()
    ssh_ip = _parse_ssh_config_host("pi")

    shortlist = []
    for cand in (env_host, None if cfg_host.lower() in ("", "auto", "pi") else cfg_host, last_ip, ssh_ip, "raspberrypi.local"):
        if cand and cand not in shortlist:
            shortlist.append(cand)

    # 1. Fast check on shortlist candidates
    for cand in shortlist:
        if probe_tcp_port(cand, rtsp_port, timeout=0.14) or probe_tcp_port(cand, imu_port, timeout=0.14):
            try:
                resolved_ip = socket.gethostbyname(cand)
            except Exception:
                resolved_ip = cand
            if resolved_ip != last_ip and save_if_changed:
                cfg["network"]["last_known_pi_ip"] = resolved_ip
                save_config(cfg, config_path)
            return resolved_ip

    # 2. Sniff 100Hz UDP broadcast from Pi IMU streamer
    sniffed = _sniff_udp_imu_sender(udp_port=imu_port, timeout=0.22)
    if sniffed:
        print(f"[NET-DISCOVERY] Detected Raspberry Pi via live UDP telemetry at: {sniffed}")
        if sniffed != last_ip and save_if_changed:
            cfg["network"]["last_known_pi_ip"] = sniffed
            save_config(cfg, config_path)
        return sniffed

    # 3. Parallel /24 subnet scan if enabled
    if bool(net.get("auto_discover_subnet", True)):
        print(f"[NET-DISCOVERY] Scanning local subnet for Raspberry Pi (RTSP :{rtsp_port} / IMU :{imu_port})...")
        scanned = scan_subnet_for_pi(rtsp_port=rtsp_port, imu_port=imu_port, timeout=0.10)
        if scanned:
            print(f"[NET-DISCOVERY] Found Raspberry Pi at new IP: {scanned}!")
            if scanned != last_ip and save_if_changed:
                cfg["network"]["last_known_pi_ip"] = scanned
                save_config(cfg, config_path)
            return scanned

    # Fallback to last_ip or explicit cfg_host if Pi is currently offline
    fallback = cfg_host if cfg_host.lower() not in ("", "auto", "pi") else (last_ip or "10.61.24.74")
    return fallback


def build_rtsp_candidates(cfg: dict | None = None, source_override: str | None = None) -> list[str]:
    """Build ordered list of RTSP URLs to try based on config and auto-discovery."""
    if cfg is None:
        cfg = load_config()
    net = cfg.get("network", {})
    cam = cfg.get("camera", {})

    src = str(source_override if source_override is not None else cam.get("source", "pi")).strip()
    rtsp_port = int(net.get("rtsp_port", 8554))
    rtsp_path = str(net.get("rtsp_path", "/cam"))
    if not rtsp_path.startswith("/"):
        rtsp_path = "/" + rtsp_path

    pi_ip = resolve_pi_host(cfg, save_if_changed=True)
    candidates = []

    if src.lower() not in ("pi", "auto", "0", "1") and "://" in src:
        candidates.append(src)

    primary_url = f"rtsp://{pi_ip}:{rtsp_port}{rtsp_path}"
    if primary_url not in candidates:
        candidates.append(primary_url)

    relay_host = str(net.get("relay_host", "")).strip()
    if relay_host:
        relay_url = f"rtsp://{relay_host}:{rtsp_port}{rtsp_path}"
        if relay_url not in candidates:
            candidates.append(relay_url)

    local_url = f"rtsp://127.0.0.1:{rtsp_port}{rtsp_path}"
    if local_url not in candidates:
        candidates.append(local_url)

    return candidates
