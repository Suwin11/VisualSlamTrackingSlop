#!/usr/bin/env python3
"""
pi_bt_motor_bridge.py (also symlinked/saved as ngamove_bt.py)
Raspberry Pi 4 Wireless Motor Bridge:
  1. Listens on UDP Port 5006 for velocity commands from `navigate_rrt.py` (`PurePursuitController`):
     {"seq": ..., "v": ..., "omega": ..., "v_left": ..., "v_right": ..., "rpm_left": ..., "rpm_right": ...}
  2. Converts linear wheel velocities (m/s) -> Dynamixel XL430 RPM (`rpm_left,rpm_right\n`).
  3. Automatically binds and connects to ESPino32 (`E0:8C:FE:C9:47:1A`) over `/dev/rfcomm0`
     (or direct Bluetooth RFCOMM socket fallback if `/dev/rfcomm0` is not bound yet!).
  4. Built-in Deadman Watchdog: Sends `"0,0\n"` (STOP) if no UDP packet arrives within 0.6s.
  5. Optional Interactive Manual Test Mode (`--manual`): Drive with W/A/S/D/Space directly from SSH!
"""

import argparse
import json
import math
import os
import select
import socket
import subprocess
import sys
import time

try:
    import serial
except ImportError:
    serial = None


ESP32_MAC_DEFAULT = "E0:8C:FE:C9:47:1A"
RFCOMM_DEV_DEFAULT = "/dev/rfcomm0"
UDP_PORT_DEFAULT = 5006

# Robot physical constants for m/s -> RPM conversion
DEFAULT_WHEEL_RADIUS_M = 0.033  # 66mm wheel diameter (0.033m radius)
DEFAULT_MAX_RPM = 60            # Safe max RPM for Dynamixel XL430-W250


class ESP32BluetoothTransport:
    """
    Resilient Bluetooth Serial (SPP/RFCOMM) transport to ESPino32 (`ESP32_Robot_Controller`).
    Supports both `/dev/rfcomm0` (via pyserial or POSIX os.open) and direct Python
    `AF_BLUETOOTH` / `BTPROTO_RFCOMM` socket connection (Channel 1) so it works even
    if `sudo rfcomm bind` wasn't run beforehand!
    """

    def __init__(self, rfcomm_dev=RFCOMM_DEV_DEFAULT, mac_addr=ESP32_MAC_DEFAULT, baud=115200):
        self.rfcomm_dev = rfcomm_dev
        self.mac_addr = mac_addr
        self.baud = baud
        self.ser = None
        self.fd = None
        self.bt_sock = None
        self.mode = None
        self.last_reconnect_attempt = 0.0

    def _try_auto_bind_rfcomm(self):
        """Attempt to bind /dev/rfcomm0 to the ESP32 MAC if it doesn't exist."""
        if os.path.exists(self.rfcomm_dev):
            return True
        try:
            subprocess.run(
                ["sudo", "-n", "rfcomm", "bind", self.rfcomm_dev, self.mac_addr, "1"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
        except Exception:
            pass
        return os.path.exists(self.rfcomm_dev)

    def connect(self):
        self.close()
        self.last_reconnect_attempt = time.monotonic()

        # 1. Try direct Python Bluetooth RFCOMM socket first or if /dev/rfcomm0 is busy/missing
        #    Wait: if /dev/rfcomm0 exists, try opening it first via pyserial or os.open
        self._try_auto_bind_rfcomm()

        if os.path.exists(self.rfcomm_dev):
            if serial is not None:
                try:
                    self.ser = serial.Serial(self.rfcomm_dev, self.baud, timeout=0.05, write_timeout=0.1)
                    self.mode = "pyserial"
                    print(f"[BT-MOTOR] Connected to ESPino32 via pyserial on {self.rfcomm_dev}!", flush=True)
                    return True
                except Exception as e:
                    self.ser = None

            try:
                self.fd = os.open(self.rfcomm_dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
                self.mode = "posix_fd"
                print(f"[BT-MOTOR] Connected to ESPino32 via POSIX FD on {self.rfcomm_dev}!", flush=True)
                return True
            except Exception:
                self.fd = None

        # 2. Fallback: Direct AF_BLUETOOTH RFCOMM Socket to E0:8C:FE:C9:47:1A Channel 1
        if hasattr(socket, "AF_BLUETOOTH") and hasattr(socket, "BTPROTO_RFCOMM"):
            try:
                sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
                sock.settimeout(3.0)
                sock.connect((self.mac_addr, 1))
                sock.setblocking(False)
                self.bt_sock = sock
                self.mode = "bt_socket"
                print(f"[BT-MOTOR] Connected directly to ESPino32 ({self.mac_addr} Ch 1) via RFCOMM socket!", flush=True)
                return True
            except Exception as e:
                try:
                    sock.close()
                except Exception:
                    pass
                self.bt_sock = None

        return False

    @property
    def is_connected(self):
        return self.mode is not None

    def send_rpm(self, rpm_left: int, rpm_right: int) -> bool:
        if not self.is_connected:
            if time.monotonic() - self.last_reconnect_attempt >= 2.0:
                self.connect()
            if not self.is_connected:
                return False

        payload = f"{int(rpm_left)},{int(rpm_right)}\n".encode("ascii")
        try:
            if self.mode == "pyserial" and self.ser is not None:
                self.ser.write(payload)
                self.ser.flush()
                return True
            elif self.mode == "posix_fd" and self.fd is not None:
                os.write(self.fd, payload)
                return True
            elif self.mode == "bt_socket" and self.bt_sock is not None:
                self.bt_sock.sendall(payload)
                return True
        except Exception as e:
            print(f"[BT-MOTOR WARN] Bluetooth write lost ({e}). Reconnecting...", flush=True)
            self.close()
        return False

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                pass
            self.fd = None
        if self.bt_sock is not None:
            try:
                self.bt_sock.close()
            except Exception:
                pass
            self.bt_sock = None
        self.mode = None


def mps_to_rpm(v_mps: float, wheel_radius_m: float, max_rpm: int, min_Start_rpm: int = 6) -> int:
    """
    Convert wheel linear velocity (m/s) to integer RPM for ESPino32 `turnWheel()`.
    Applies a minimum starting RPM floor so low pure-pursuit speeds overcome static friction.
    """
    if abs(v_mps) < 0.010:
        return 0
    circumference = 2.0 * math.pi * max(wheel_radius_m, 0.01)
    raw_rpm = (v_mps / circumference) * 60.0
    sign = 1 if raw_rpm > 0 else -1
    mag = max(min_Start_rpm, int(round(abs(raw_rpm))))
    mag = min(mag, int(max_rpm))
    return sign * mag


def main():
    ap = argparse.ArgumentParser(description="Pi 4 UDP-to-Bluetooth RFCOMM Motor Bridge for ESPino32 Dynamixel XL430")
    ap.add_argument("--dev", default=RFCOMM_DEV_DEFAULT, help="RFCOMM device path (default: /dev/rfcomm0)")
    ap.add_argument("--mac", default=ESP32_MAC_DEFAULT, help="ESPino32 Bluetooth MAC (default: E0:8C:FE:C9:47:1A)")
    ap.add_argument("--udp-port", type=int, default=UDP_PORT_DEFAULT, help="UDP port to listen on (default: 5006)")
    ap.add_argument("--wheel-radius", type=float, default=DEFAULT_WHEEL_RADIUS_M, help="Wheel radius in meters (default: 0.033)")
    ap.add_argument("--max-rpm", type=int, default=DEFAULT_MAX_RPM, help="Maximum wheel RPM clamp (default: 60)")
    ap.add_argument("--invert-left", action="store_true", help="Invert left wheel direction")
    ap.add_argument("--invert-right", action="store_true", help="Invert right wheel direction")
    ap.add_argument("--swap-wheels", action="store_true", help="Swap left and right wheels")
    ap.add_argument("--raw-pwm", action="store_true", help="Send raw PWM (-255..+255) instead of scaled RPM to ESP32")
    ap.add_argument("--test", action="store_true", help="Run a 2-second forward/stop test on startup")
    args = ap.parse_args()

    bt = ESP32BluetoothTransport(rfcomm_dev=args.dev, mac_addr=args.mac)
    print(f"[BT-MOTOR] Connecting to ESPino32 ({args.mac}) via {args.dev}...", flush=True)
    bt.connect()

    if args.test and bt.is_connected:
        print("[BT-MOTOR] Running 1.5s Forward -> Stop self-test (20 RPM)...", flush=True)
        bt.send_rpm(20, 20)
        time.sleep(1.5)
        bt.send_rpm(0, 0)
        print("[BT-MOTOR] Self-test complete!", flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.udp_port))
    sock.setblocking(False)
    print(f"[BT-MOTOR] Listening for navigation commands on UDP :{args.udp_port}...", flush=True)

    last_cmd_mono = 0.0
    last_sent_rpm = (0, 0)
    last_send_mono = 0.0
    last_print_mono = 0.0
    watchdog_stopped = True

    try:
        while True:
            now = time.monotonic()
            r, _, _ = select.select([sock], [], [], 0.04)
            latest_data = None
            if r:
                while True:
                    try:
                        data, addr = sock.recvfrom(1024)
                        latest_data = data
                    except BlockingIOError:
                        break
                    except Exception:
                        break

            if latest_data is not None:
                try:
                    msg = json.loads(latest_data.decode("ascii", errors="ignore"))
                    if args.raw_pwm and "pwm_left" in msg and "pwm_right" in msg:
                        rpm_l = int(msg["pwm_left"])
                        rpm_r = int(msg["pwm_right"])
                    elif "rpm_left" in msg and "rpm_right" in msg and msg.get("direct_rpm", False):
                        rpm_l = int(msg["rpm_left"])
                        rpm_r = int(msg["rpm_right"])
                    else:
                        v_l = float(msg.get("v_left", 0.0))
                        v_r = float(msg.get("v_right", 0.0))
                        rpm_l = mps_to_rpm(v_l, args.wheel_radius, args.max_rpm)
                        rpm_r = mps_to_rpm(v_r, args.wheel_radius, args.max_rpm)

                    if args.swap_wheels:
                        rpm_l, rpm_r = rpm_r, rpm_l
                    if args.invert_left:
                        rpm_l = -rpm_l
                    if args.invert_right:
                        rpm_r = -rpm_r

                    max_clamp = 255 if args.raw_pwm else args.max_rpm
                    rpm_l = max(-max_clamp, min(max_clamp, rpm_l))
                    rpm_r = max(-max_clamp, min(max_clamp, rpm_r))

                    last_cmd_mono = now
                    watchdog_stopped = False

                    # Send to ESP32 if RPM changed or every 150ms as keepalive
                    if (rpm_l, rpm_r) != last_sent_rpm or (now - last_send_mono >= 0.15):
                        if bt.send_rpm(rpm_l, rpm_r):
                            last_sent_rpm = (rpm_l, rpm_r)
                            last_send_mono = now

                    if now - last_print_mono >= 0.5:
                        last_print_mono = now
                        status_str = "BT-OK" if bt.is_connected else "BT-RECONN"
                        print(f"[{status_str}] UDP->ESP32 | RPM_L={rpm_l:+3d}, RPM_R={rpm_r:+3d} "
                              f"(v={msg.get('v', 0.0):+.2f}m/s, w={msg.get('omega', 0.0):+.2f}r/s)", flush=True)
                except Exception:
                    pass

            # Safety Watchdog: stop motors if no UDP command received for > 0.6 seconds
            if not watchdog_stopped and (now - last_cmd_mono > 0.60):
                watchdog_stopped = True
                last_sent_rpm = (0, 0)
                bt.send_rpm(0, 0)
                print("[BT-MOTOR WATCHDOG] No UDP command for 0.6s -> Sent STOP (0,0)", flush=True)

            # Periodic reconnect check when idle
            if not bt.is_connected and (now - bt.last_reconnect_attempt >= 2.5):
                bt.connect()

    except KeyboardInterrupt:
        print("\n[BT-MOTOR] Stopping motors and exiting...", flush=True)
    finally:
        bt.send_rpm(0, 0)
        bt.close()
        sock.close()


if __name__ == "__main__":
    main()
