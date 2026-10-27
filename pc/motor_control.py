"""
motor_control.py
Dedicated, Self-Contained PWM-Style Motor Control & Trajectory Tracking Module.

All motor logic lives exclusively in this file!
Whenever you want to tweak motor directions, swap left/right wheels, adjust PWM power
gears, change WASD teleop mixing, or tune autonomous steering, you ONLY edit this file.

================================================================================
QUICK HARDWARE ORIENTATION & PWM TUNING CONSTANTS (EDIT HERE!)
================================================================================
"""

import ctypes
import json
import math
import socket
import sys
import threading
import time
import numpy as np


# ── 1. WHEELBASE ORIENTATION & POLARITY (180-Degree Backwards Fix) ───────────
# For a chassis mounted 180° backwards relative to the cameras:
#   - Both motors must be inverted (True, True) AND Left/Right swapped (True)
#   - That maps (pwm_left, pwm_right) -> (-pwm_right, -pwm_left).
SWAP_LEFT_RIGHT_MOTORS = True
INVERT_LEFT_MOTOR      = True
INVERT_RIGHT_MOTOR     = True

# ── 2. PWM-STYLE DUTY CYCLE SETTINGS (-255 to +255, or % -100 to +100) ───────
PWM_RANGE              = 255    # Standard 8-bit PWM scale (-255 .. +255)
MANUAL_FORWARD_PWM     = 140    # Default manual W/S forward/reverse PWM (0..255)
MANUAL_TURN_PWM        = 105    # Default manual A/D pivot/turn PWM (0..255)
MIN_EFFECTIVE_PWM      = 28     # Deadband floor to overcome static gearbox friction
PWM_STEP_SIZE          = 20     # Step size when pressing '9' (-PWM) or '0' (+PWM)

# ── 3. HARDWARE BRIDGE SCALING (Maps PWM -255..+255 -> ESP32 Hardware Units) ─
# Note: The ESPino32 firmware (`espinoasd.ino`) parses `left,right\n` and passes
# the integer to `turnWheel()` (which maxes out around 60 on XL430 at 12V).
# Set `BRIDGE_OUTPUT_MODE = "ESP32_RPM"` so you control everything in smooth PWM
# (-255..+255) on the PC while the bridge automatically scales it for ESPino32,
# OR set `BRIDGE_OUTPUT_MODE = "RAW_PWM"` if you flash PWM firmware on the ESP32!
BRIDGE_OUTPUT_MODE     = "ESP32_RPM"  # "ESP32_RPM" (scales 255 -> 60) or "RAW_PWM" (sends -255..255 directly)
HARDWARE_MAX_LIMIT     = 60           # Max value sent to ESP32 when BRIDGE_OUTPUT_MODE == "ESP32_RPM"


def apply_motor_orientation(pwm_left: float, pwm_right: float,
                            swap_lr: bool = SWAP_LEFT_RIGHT_MOTORS,
                            invert_l: bool = INVERT_LEFT_MOTOR,
                            invert_r: bool = INVERT_RIGHT_MOTOR) -> tuple[int, int]:
    """
    Transform logical (pwm_left, pwm_right) in [-255, +255] into physical motor PWM
    after applying L/R swap and polarity inversion.
    """
    l_val = int(round(np.clip(pwm_left, -PWM_RANGE, PWM_RANGE)))
    r_val = int(round(np.clip(pwm_right, -PWM_RANGE, PWM_RANGE)))

    if swap_lr:
        l_val, r_val = r_val, l_val
    if invert_l:
        l_val = -l_val
    if invert_r:
        r_val = -r_val

    return l_val, r_val


def pwm_to_hardware_output(pwm_val: int,
                           pwm_range: int = PWM_RANGE,
                           hw_max: int = HARDWARE_MAX_LIMIT,
                           min_hw: int = 6) -> int:
    """
    Convert a signed PWM value (-255 .. +255) to the integer command expected by the
    ESP32 motor driver.
    """
    if abs(pwm_val) < 5:
        return 0
    if BRIDGE_OUTPUT_MODE.upper() == "RAW_PWM":
        return int(np.clip(pwm_val, -pwm_range, pwm_range))

    sign = 1 if pwm_val > 0 else -1
    duty = min(1.0, abs(pwm_val) / float(max(pwm_range, 1)))
    scaled = int(round(duty * hw_max))
    return sign * max(min_hw, min(scaled, hw_max))


class MotorController:
    """
    Unified PWM-Style Motor Controller + Autonomous Pure Pursuit Tracker.

    Direct API for custom scripts or interactive control:
      - `set_pwm(left_pwm, right_pwm)`        : Set raw left/right PWM in [-255, +255]
      - `set_duty_cycle(left_pct, right_pct)` : Set left/right duty cycle in [-100%, +100%]
      - `arcade_drive(throttle_pwm, turn_pwm)`: Mix forward throttle + steering into L/R PWM
      - `stop_motors()`                       : Immediate 0 PWM stop
    """

    # Windows Virtual-Key Codes for zero-delay hardware key state polling
    VK_W = 0x57
    VK_A = 0x41
    VK_S = 0x53
    VK_D = 0x44
    VK_UP = 0x26
    VK_DOWN = 0x28
    VK_LEFT = 0x25
    VK_RIGHT = 0x27

    def __init__(
        self,
        lookahead_m: float = 0.35,
        max_v: float = 0.30,
        max_omega: float = 1.20,
        wheel_base_m: float = 0.28,
        wheel_radius_m: float = 0.033,
        max_rpm: int = HARDWARE_MAX_LIMIT,
        min_start_rpm: int = 6,
        manual_forward_rpm: int = 32,
        manual_turn_rpm: int = 24,
        pwm_range: int = PWM_RANGE,
        manual_forward_pwm: int = MANUAL_FORWARD_PWM,
        manual_turn_pwm: int = MANUAL_TURN_PWM,
        swap_left_right: bool = SWAP_LEFT_RIGHT_MOTORS,
        invert_left: bool = INVERT_LEFT_MOTOR,
        invert_right: bool = INVERT_RIGHT_MOTOR,
        manual_hold_timeout_sec: float = 0.35,
        control_loop_hz: float = 25.0,
        goal_tolerance_m: float = 0.15,
        udp_host: str = "auto",
        udp_port: int = 5006,
    ):
        self.lookahead_m = float(lookahead_m)
        self.max_v = float(max_v)
        self.max_omega = float(max_omega)
        self.wheel_base_m = float(wheel_base_m)
        self.wheel_radius_m = float(wheel_radius_m)

        # Hardware & PWM limits
        self.pwm_range = int(pwm_range)
        self.max_rpm = int(max_rpm)
        self.min_start_rpm = int(min_start_rpm)

        # Manual PWM gears (-255 .. +255)
        self.manual_forward_pwm = int(manual_forward_pwm)
        self.manual_turn_pwm = int(manual_turn_pwm)

        # Orientation flags (fixes 180-deg backwards wheelbase)
        self.swap_left_right = bool(swap_left_right)
        self.invert_left = bool(invert_left)
        self.invert_right = bool(invert_right)

        self.manual_hold_timeout_sec = float(manual_hold_timeout_sec)
        self.control_loop_hz = float(max(control_loop_hz, 10.0))
        self.goal_tolerance_m = float(goal_tolerance_m)

        self.udp_port = int(udp_port)
        self.udp_host = str(udp_host).strip()

        self._lock = threading.Lock()
        self.control_mode = "AUTO"  # "AUTO", "MANUAL", or "PAUSED"
        self.auto_drive_enabled = True
        self._manual_hold_until = 0.0
        self._manual_hold_pwm = (0, 0)
        self._latched_pwm = (0, 0)
        self._emergency_stop = False
        self._escape_active_until = 0.0

        self._latest_auto_cmd = self._make_cmd_dict(0, 0, mode="AUTO")
        self.last_cmd = dict(self._latest_auto_cmd)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except Exception:
            pass
        self.sock.setblocking(False)
        self.seq = 0

        self._is_windows = sys.platform.startswith("win") and hasattr(ctypes, "windll")
        self._running = True
        self._worker_thread = threading.Thread(target=self._control_worker, daemon=True)
        self._worker_thread.start()

    # ── Compatibility properties for navigate_rrt.py ─────────────────
    @property
    def wheel_base(self):
        return self.wheel_base_m

    @property
    def lookahead(self):
        return self.lookahead_m

    @lookahead.setter
    def lookahead(self, val):
        self.lookahead_m = float(val)

    @property
    def manual_forward_rpm(self) -> int:
        """Exposes current manual forward PWM so existing HUD & hotkeys work seamlessly."""
        return self.manual_forward_pwm

    @manual_forward_rpm.setter
    def manual_forward_rpm(self, val: int):
        # If config passes legacy RPM (<=60), scale to PWM; if >60, treat as direct PWM
        v = int(val)
        if 0 < v <= 60 and v < MIN_EFFECTIVE_PWM:
            self.manual_forward_pwm = int(round((v / 60.0) * self.pwm_range))
        else:
            self.manual_forward_pwm = int(np.clip(v, MIN_EFFECTIVE_PWM, self.pwm_range))

    @property
    def manual_turn_rpm(self) -> int:
        return self.manual_turn_pwm

    @manual_turn_rpm.setter
    def manual_turn_rpm(self, val: int):
        v = int(val)
        if 0 < v <= 60 and v < MIN_EFFECTIVE_PWM:
            self.manual_turn_pwm = int(round((v / 60.0) * self.pwm_range))
        else:
            self.manual_turn_pwm = int(np.clip(v, MIN_EFFECTIVE_PWM, self.pwm_range))

    def get_linear_velocity(self) -> float:
        """
        Return the instantaneous commanded forward linear velocity (m/s).
        Thread-safe, accounts for manual hold, latched cruise, auto drive, and emergency stop.
        Manual keyboard/teleop commands take precedence over pause/idle states!
        """
        now = time.monotonic()
        with self._lock:
            # Active manual commands (keyboard hold / cruise) ALWAYS provide linear velocity!
            if now < self._manual_hold_until:
                pl, pr = self._manual_hold_pwm
                avg_pwm = 0.5 * (pl + pr)
                return (avg_pwm / float(max(self.pwm_range, 1))) * self.max_v
            if self._latched_pwm != (0, 0):
                pl, pr = self._latched_pwm
                avg_pwm = 0.5 * (pl + pr)
                return (avg_pwm / float(max(self.pwm_range, 1))) * self.max_v
            if self._emergency_stop or self.control_mode == "PAUSED":
                return 0.0
            cmd = self.last_cmd
            if "pwm_left" in cmd and "pwm_right" in cmd and (cmd["pwm_left"] != 0 or cmd["pwm_right"] != 0):
                avg_pwm = 0.5 * (cmd["pwm_left"] + cmd["pwm_right"])
                return (avg_pwm / float(max(self.pwm_range, 1))) * self.max_v
            if "rpm_left" in cmd and "rpm_right" in cmd and (cmd["rpm_left"] != 0 or cmd["rpm_right"] != 0):
                avg_rpm = 0.5 * (cmd["rpm_left"] + cmd["rpm_right"])
                circumference = 2.0 * math.pi * max(self.wheel_radius_m, 0.01)
                return (avg_rpm / 60.0) * circumference
            return float(cmd.get("v", 0.0))

    def get_angular_velocity(self) -> float:
        """
        Return the instantaneous commanded angular steering rate (rad/s).
        Positive = Turning Left (counter-clockwise), Negative = Turning Right.
        """
        now = time.monotonic()
        with self._lock:
            if now < self._manual_hold_until:
                pl, pr = self._manual_hold_pwm
                diff_pwm = float(pr - pl)  # right > left turns left
                return (diff_pwm / float(max(self.pwm_range, 1))) * self.max_omega
            if self._latched_pwm != (0, 0):
                pl, pr = self._latched_pwm
                diff_pwm = float(pr - pl)
                return (diff_pwm / float(max(self.pwm_range, 1))) * self.max_omega
            if self._emergency_stop or self.control_mode == "PAUSED":
                return 0.0
            cmd = self.last_cmd
            return float(cmd.get("omega", 0.0))

    # ── Direct PWM & Duty-Cycle Control API ──────────────────────────
    def set_pwm(self, left_pwm: float, right_pwm: float, duration_sec: float | None = None):
        """
        Directly command Left and Right motors using signed PWM (-255 to +255).
        Positive = Robot Forward, Negative = Robot Backward (orientation is auto-corrected!).
        """
        if duration_sec is None:
            duration_sec = self.manual_hold_timeout_sec
        pl = int(round(np.clip(left_pwm, -self.pwm_range, self.pwm_range)))
        pr = int(round(np.clip(right_pwm, -self.pwm_range, self.pwm_range)))
        with self._lock:
            self._manual_hold_pwm = (pl, pr)
            self._manual_hold_until = time.monotonic() + float(duration_sec)
            if pl == 0 and pr == 0:
                self._latched_pwm = (0, 0)
            avg_pwm = 0.5 * (pl + pr)
            diff_pwm = float(pr - pl)
            v = (avg_pwm / float(max(self.pwm_range, 1))) * self.max_v
            omega = (diff_pwm / float(max(self.pwm_range, 1))) * self.max_omega
            self.last_cmd = self._make_cmd_dict(pl, pr, mode="MANUAL_HOLD", v=v, omega=omega)

    def set_duty_cycle(self, left_pct: float, right_pct: float, duration_sec: float | None = None):
        """
        Command Left and Right motors using percentage duty cycle (-100.0% to +100.0%).
        """
        pl = (float(np.clip(left_pct, -100.0, 100.0)) / 100.0) * self.pwm_range
        pr = (float(np.clip(right_pct, -100.0, 100.0)) / 100.0) * self.pwm_range
        self.set_pwm(pl, pr, duration_sec=duration_sec)

    def trigger_escape_pivot(self, direction="right", duration_sec: float = 0.65, pwm: int | None = None):
        """
        Command an autonomous in-place pivot to swing sensors away from obstacles.
        direction: 'right' (or +1) -> right turn (+turn_pwm, -turn_pwm)
                   'left'  (or -1) -> left turn (-turn_pwm, +turn_pwm)
        """
        turn_val = pwm if pwm is not None else int(self.manual_turn_pwm * 0.85)
        dir_sign = 1 if str(direction).lower() in ("right", "1", "+1") else -1
        pl = int(round(+turn_val * dir_sign))
        pr = int(round(-turn_val * dir_sign))
        now = time.monotonic()
        with self._lock:
            self._manual_hold_pwm = (pl, pr)
            self._manual_hold_until = now + float(duration_sec)
            self._escape_active_until = now + float(duration_sec)
            self._latched_pwm = (0, 0)

    def cancel_escape_pivot(self):
        """Immediately terminate an active escape pivot."""
        with self._lock:
            self._escape_active_until = 0.0
            self._manual_hold_until = 0.0
            self._manual_hold_pwm = (0, 0)

    def arcade_drive(self, throttle_pwm: float, turn_pwm: float, duration_sec: float | None = None):
        """
        PWM Arcade Mixing: combines forward/reverse throttle (-255..+255) and
        left/right steering (-255..+255, positive = turn left).
        """
        t = float(np.clip(throttle_pwm, -self.pwm_range, self.pwm_range))
        w = float(np.clip(turn_pwm, -self.pwm_range, self.pwm_range))
        if abs(t) > 1.0 and abs(w) > 1.0:
            pl = t - w * 0.55
            pr = t + w * 0.55
        else:
            pl = t - w
            pr = t + w
        self.set_pwm(pl, pr, duration_sec=duration_sec)

    def stop_motors(self):
        """Immediate hard stop (0 PWM to both wheels)."""
        self.set_pwm(0, 0, duration_sec=0.25)

    def set_latched_pwm(self, left_pwm: int, right_pwm: int):
        """Set continuous cruise PWM until stopped with Space/X."""
        pl = int(np.clip(left_pwm, -self.pwm_range, self.pwm_range))
        pr = int(np.clip(right_pwm, -self.pwm_range, self.pwm_range))
        with self._lock:
            self._latched_pwm = (pl, pr)
            if pl != 0 or pr != 0:
                self.control_mode = "MANUAL"
                self.auto_drive_enabled = False
        print(f"[PWM CRUISE] Latched PWM -> L={pl:+d}, R={pr:+d} (range +/-{self.pwm_range})")

    # Legacy aliases so navigate_rrt.py works without breaking changes
    def send_manual_rpm(self, rpm_left: int, rpm_right: int, duration_sec: float | None = None):
        self.set_pwm(rpm_left, rpm_right, duration_sec=duration_sec)

    def set_latched_rpm(self, rpm_left: int, rpm_right: int):
        self.set_latched_pwm(rpm_left, rpm_right)

    def adjust_manual_speed(self, delta_step: int):
        step = PWM_STEP_SIZE if abs(delta_step) <= 6 else int(delta_step)
        if delta_step < 0:
            step = -abs(step)
        with self._lock:
            self.manual_forward_pwm = int(np.clip(self.manual_forward_pwm + step, MIN_EFFECTIVE_PWM, self.pwm_range))
            self.manual_turn_pwm = int(np.clip(round(self.manual_forward_pwm * 0.75), MIN_EFFECTIVE_PWM, self.pwm_range))
            pct = int(round(100.0 * self.manual_forward_pwm / self.pwm_range))
            print(f"[PWM GEAR] Forward PWM={self.manual_forward_pwm}/{self.pwm_range} ({pct}%) | Turn PWM={self.manual_turn_pwm}")
            return self.manual_forward_pwm, self.manual_turn_pwm

    def update_target_host(self, new_host: str):
        if new_host and new_host.lower() not in ("auto", "pi", "") and new_host != self.udp_host:
            self.udp_host = new_host
            print(f"[MOTOR PWM] Updated Pi UDP target -> {self.udp_host}:{self.udp_port}")

    def toggle_mode(self):
        with self._lock:
            if self.control_mode == "AUTO":
                self.control_mode = "MANUAL"
                self.auto_drive_enabled = False
                self._latched_pwm = (0, 0)
            elif self.control_mode == "MANUAL":
                self.control_mode = "PAUSED"
                self.auto_drive_enabled = False
                self._latched_pwm = (0, 0)
                self._manual_hold_pwm = (0, 0)
            else:
                self.control_mode = "AUTO"
                self.auto_drive_enabled = True
                self._latched_pwm = (0, 0)
            mode = self.control_mode
        print(f"[MOTOR PWM] Switched Drive Mode -> {mode}")
        return mode

    def toggle_pause(self):
        with self._lock:
            if self.control_mode == "PAUSED":
                self.control_mode = "AUTO"
                self.auto_drive_enabled = True
            else:
                self.control_mode = "PAUSED"
                self.auto_drive_enabled = False
            self._latched_pwm = (0, 0)
            self._manual_hold_pwm = (0, 0)
            self._manual_hold_until = 0.0
            mode = self.control_mode
        print(f"[MOTOR PWM] Drive State -> {mode}")
        return mode

    def _make_cmd_dict(self, logical_pwm_l: int, logical_pwm_r: int, mode: str = "AUTO",
                       v: float = 0.0, omega: float = 0.0,
                       lookahead=None, goal_reached: bool = False) -> dict:
        # Apply 180-degree wheelbase orientation fix (swap + invert)
        phys_pwm_l, phys_pwm_r = apply_motor_orientation(
            logical_pwm_l,
            logical_pwm_r,
            swap_lr=self.swap_left_right,
            invert_l=self.invert_left,
            invert_r=self.invert_right,
        )
        hw_l = pwm_to_hardware_output(phys_pwm_l, self.pwm_range, self.max_rpm, self.min_start_rpm)
        hw_r = pwm_to_hardware_output(phys_pwm_r, self.pwm_range, self.max_rpm, self.min_start_rpm)
        return {
            "v": round(float(v), 3),
            "omega": round(float(omega), 3),
            "v_left": 0.0,
            "v_right": 0.0,
            "pwm_left": int(logical_pwm_l),
            "pwm_right": int(logical_pwm_r),
            "hw_pwm_left": int(phys_pwm_l),
            "hw_pwm_right": int(phys_pwm_r),
            "rpm_left": int(hw_l),
            "rpm_right": int(hw_r),
            "direct_rpm": True,
            "direct_pwm": True,
            "mode": mode,
            "lookahead": lookahead,
            "goal_reached": bool(goal_reached),
        }

    def _poll_windows_keys(self) -> tuple[int, int] | None:
        if not self._is_windows:
            return None
        try:
            user32 = ctypes.windll.user32
            fg_hwnd = user32.GetForegroundWindow()
            if not fg_hwnd:
                return None
            length = user32.GetWindowTextLengthW(fg_hwnd)
            if length <= 0:
                return None
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(fg_hwnd, buf, length + 1)
            title = buf.value
            allowed_keywords = ("FIBO", "YOLO", "Navigation", "Stereo", "Tracking", "Robot", "Map", "Point Cloud", "RRT")
            if not any(k.lower() in title.lower() for k in allowed_keywords):
                return None

            w_down = (user32.GetAsyncKeyState(self.VK_W) & 0x8000) != 0 or (user32.GetAsyncKeyState(self.VK_UP) & 0x8000) != 0
            s_down = (user32.GetAsyncKeyState(self.VK_S) & 0x8000) != 0 or (user32.GetAsyncKeyState(self.VK_DOWN) & 0x8000) != 0
            a_down = (user32.GetAsyncKeyState(self.VK_A) & 0x8000) != 0 or (user32.GetAsyncKeyState(self.VK_LEFT) & 0x8000) != 0
            d_down = (user32.GetAsyncKeyState(self.VK_D) & 0x8000) != 0 or (user32.GetAsyncKeyState(self.VK_RIGHT) & 0x8000) != 0

            if not (w_down or s_down or a_down or d_down):
                return None

            fwd = 0
            if w_down and not s_down:
                fwd = self.manual_forward_pwm
            elif s_down and not w_down:
                fwd = -self.manual_forward_pwm

            turn = 0
            if a_down and not d_down:
                turn = self.manual_turn_pwm
            elif d_down and not a_down:
                turn = -self.manual_turn_pwm

            if fwd != 0 and turn != 0:
                pl = int(np.clip(fwd - int(turn * 0.55), -self.pwm_range, self.pwm_range))
                pr = int(np.clip(fwd + int(turn * 0.55), -self.pwm_range, self.pwm_range))
            else:
                pl = int(np.clip(fwd - turn, -self.pwm_range, self.pwm_range))
                pr = int(np.clip(fwd + turn, -self.pwm_range, self.pwm_range))
            return (pl, pr)
        except Exception:
            return None

    def _control_worker(self):
        interval = 1.0 / self.control_loop_hz
        while self._running:
            t0 = time.monotonic()
            hw_keys = self._poll_windows_keys()

            with self._lock:
                if hw_keys is not None:
                    self._manual_hold_pwm = hw_keys
                    self._manual_hold_until = t0 + 0.18
                    self._latched_pwm = (0, 0)

                is_escape = t0 < getattr(self, "_escape_active_until", 0.0)
                if self._emergency_stop and hw_keys is None and t0 >= self._manual_hold_until and not is_escape:
                    active_cmd = self._make_cmd_dict(0, 0, mode="EMERGENCY_STOP")
                elif t0 < self._manual_hold_until:
                    pl, pr = self._manual_hold_pwm
                    mode_name = "EMERGENCY_PIVOT" if is_escape else "MANUAL_HOLD"
                    active_cmd = self._make_cmd_dict(pl, pr, mode=mode_name)
                elif self._latched_pwm != (0, 0):
                    pl, pr = self._latched_pwm
                    active_cmd = self._make_cmd_dict(pl, pr, mode="MANUAL_CRUISE")
                elif self.control_mode in ("PAUSED", "MANUAL") or not self.auto_drive_enabled:
                    active_cmd = self._make_cmd_dict(0, 0, mode=self.control_mode)
                else:
                    active_cmd = dict(self._latest_auto_cmd)

                self.last_cmd = active_cmd

            self._broadcast(active_cmd)

            elapsed = time.monotonic() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _mps_to_pwm(self, v_mps: float) -> int:
        """Convert linear wheel velocity (m/s) into signed PWM (-255 .. +255)."""
        if abs(v_mps) < 0.010:
            return 0
        duty = float(np.clip(v_mps / max(self.max_v, 0.05), -1.0, 1.0))
        sign = 1 if duty > 0 else -1
        mag = int(round(abs(duty) * self.pwm_range))
        return sign * max(MIN_EFFECTIVE_PWM, min(mag, self.pwm_range))

    def compute_command(self, robot_pose, path, goal=None, emergency_stop=False):
        rx, ry, rtheta = robot_pose

        if goal is not None and math.hypot(goal[0] - rx, goal[1] - ry) <= self.goal_tolerance_m:
            auto_cmd = self._make_cmd_dict(0, 0, mode="AUTO", lookahead=goal, goal_reached=True)
        elif not path or len(path) < 2:
            auto_cmd = self._make_cmd_dict(0, 0, mode="AUTO", lookahead=None, goal_reached=False)
        else:
            closest_idx = min(
                range(len(path)),
                key=lambda i: math.hypot(path[i][0] - rx, path[i][1] - ry)
            )

            target_pt = path[-1]
            for i in range(closest_idx, len(path)):
                if math.hypot(path[i][0] - rx, path[i][1] - ry) >= self.lookahead_m:
                    target_pt = path[i]
                    break

            dx = target_pt[0] - rx
            dy = target_pt[1] - ry
            dist_to_target = max(math.hypot(dx, dy), 1e-3)

            target_angle = math.atan2(dy, dx)
            alpha = (target_angle - rtheta + math.pi) % (2.0 * math.pi) - math.pi

            if abs(alpha) > math.radians(55.0):
                v = 0.0
                omega = float(max(-self.max_omega, min(self.max_omega, 1.8 * alpha)))
            else:
                goal_dist = math.hypot(path[-1][0] - rx, path[-1][1] - ry)
                turn_scale = max(0.25, math.cos(alpha))
                approach_scale = min(1.0, goal_dist / (self.lookahead_m * 1.5))
                v = self.max_v * turn_scale * max(0.3, approach_scale)
                curvature = (2.0 * math.sin(alpha)) / dist_to_target
                omega = float(max(-self.max_omega, min(self.max_omega, v * curvature)))

            half_L = 0.5 * self.wheel_base_m
            v_left = float(v - omega * half_L)
            v_right = float(v + omega * half_L)
            pwm_l = self._mps_to_pwm(v_left)
            pwm_r = self._mps_to_pwm(v_right)

            auto_cmd = self._make_cmd_dict(
                pwm_l, pwm_r, mode="AUTO", v=v, omega=omega, lookahead=target_pt, goal_reached=False
            )

        with self._lock:
            self._emergency_stop = bool(emergency_stop)
            self._latest_auto_cmd = auto_cmd
            res = dict(self.last_cmd)
            if res.get("lookahead") is None:
                res["lookahead"] = auto_cmd.get("lookahead")
            return res

    def _broadcast(self, cmd: dict):
        if self.udp_port <= 0:
            return
        self.seq += 1
        payload = json.dumps({
            "seq": self.seq,
            "ts": round(time.time(), 4),
            "v": cmd["v"],
            "omega": cmd["omega"],
            "pwm_left": cmd.get("hw_pwm_left", 0),
            "pwm_right": cmd.get("hw_pwm_right", 0),
            "rpm_left": cmd.get("rpm_left", 0),
            "rpm_right": cmd.get("rpm_right", 0),
            "direct_rpm": True,
            "direct_pwm": True,
            "mode": cmd.get("mode", "AUTO"),
            "goal_reached": cmd["goal_reached"],
        }).encode("ascii")

        targets = {("255.255.255.255", self.udp_port)}
        if self.udp_host and self.udp_host.lower() not in ("auto", "pi", ""):
            targets.add((self.udp_host, self.udp_port))

        for tgt in targets:
            try:
                self.sock.sendto(payload, tgt)
            except Exception:
                pass

    def stop(self):
        self._running = False
        if hasattr(self, "_worker_thread") and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)
        stop_cmd = self._make_cmd_dict(0, 0, mode="STOP")
        self._broadcast(stop_cmd)
        try:
            self.sock.close()
        except Exception:
            pass


# Alias for backward compatibility with existing imports
PurePursuitController = MotorController


if __name__ == "__main__":
    import argparse
    from config_manager import load_config, resolve_pi_host

    ap = argparse.ArgumentParser(description="Standalone PWM Motor Test CLI")
    ap.add_argument("--left", type=int, default=0, help="Left wheel PWM (-255..+255)")
    ap.add_argument("--right", type=int, default=0, help="Right wheel PWM (-255..+255)")
    ap.add_argument("--duration", type=float, default=1.5, help="Duration in seconds")
    args = ap.parse_args()

    cfg = load_config()
    pi_ip = resolve_pi_host(cfg, save_if_changed=False)
    mc = MotorController(udp_host=pi_ip, udp_port=int(cfg["network"].get("motor_udp_port", 5006)))
    print(f"[PWM CLI] Sending Left={args.left:+d}, Right={args.right:+d} PWM to {pi_ip} for {args.duration}s...")
    mc.set_pwm(args.left, args.right, duration_sec=args.duration)
    time.sleep(args.duration)
    mc.stop()
    print("[PWM CLI] Done (Motors Stopped).")
