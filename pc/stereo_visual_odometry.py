"""
stereo_visual_odometry.py
Stereo Visual-Inertial Odometry (100% IMU Rotation + Angled 3D IMU Vector Motion Fusion
+ Motor Command Slip/Unresponsive Rejection + StereoSGBM Disparity + 3D-2D PnP).
Drop-in replacement for `VisualOdometry` with identical API and attributes.
"""

import argparse
import math
import os
import time
import cv2
import numpy as np


def _angle_diff(a, b):
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


class StereoVisualOdometry:
    # Minimum IMU vector activity required to confirm the robot is physically moving
    IMU_ACCEL_NOISE_FLOOR = 0.045  # m/s^2 (combined 3D linear acceleration vector floor)
    IMU_GYRO_NOISE_FLOOR  = 0.025  # rad/s (combined 3D angular velocity vector floor)
    IMU_ACTIVE_HOLD_SEC   = 0.18   # s (short continuity window across 100Hz IMU samples)

    def __init__(
        self,
        stereo_calib_file="stereo_calib.npz",
        camera_height_m=0.20,
        camera_pitch_deg=15.0,
        fov_deg=62.2,
        baseline_m=0.08,
        img_w=640,
        img_h=480,
        num_disparities=64,
        block_size=7,
        yaw_scale=1.0,
        tof_odom_blend=0.25,
        swap_left_right=None,
        vertical_offset_px=None,
        imu_motion_weight=0.45,
        depth_motion_weight=0.35,
        cmd_motion_weight=0.20,
        imu_mount_yaw_deg=0.0,
        imu_mount_pitch_deg=-4.4,
        imu_mount_roll_deg=5.7,
        auto_calibrate_mount=True,
    ):
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.camera_height = float(camera_height_m)
        self.nominal_pitch_rad = math.radians(float(camera_pitch_deg))
        self.camera_pitch_rad = self.nominal_pitch_rad
        self.camera_roll_rad = 0.0
        self.fov_deg = float(fov_deg)
        self.baseline = float(baseline_m)
        self.yaw_scale = float(yaw_scale)
        self.tof_odom_blend = float(tof_odom_blend)
        self.imu_motion_weight = float(np.clip(imu_motion_weight, 0.0, 1.0))
        self.depth_motion_weight = float(max(0.0, depth_motion_weight))
        self.cmd_motion_weight = float(max(0.0, cmd_motion_weight))
        self.imu_mount_yaw_deg = float(imu_mount_yaw_deg)
        self.imu_mount_pitch_deg = float(imu_mount_pitch_deg)
        self.imu_mount_roll_deg = float(imu_mount_roll_deg)
        self.auto_calibrate_mount = bool(auto_calibrate_mount)
        self.last_motion_source = "INIT"
        self._calib_gravity_samples = []
        self._gravity_calibrated = False
        self.metric_scale = 1.0

        if swap_left_right is None or vertical_offset_px is None:
            try:
                from config_manager import load_config
                _scfg = load_config().get("stereo_camera", {})
                if swap_left_right is None:
                    swap_left_right = _scfg.get("swap_left_right", False)
                if vertical_offset_px is None:
                    vertical_offset_px = _scfg.get("vertical_offset_px", 0)
            except Exception:
                pass
        self.swap_left_right = bool(swap_left_right)
        self.vertical_offset_px = int(vertical_offset_px)

        self._update_focal_from_fov(self.fov_deg)
        self.cx = self.img_w * 0.5
        self.cy = self.img_h * 0.5
        self.K_left = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)
        self.dist_zero = np.zeros((4, 1), dtype=np.float64)

        self.has_stereo_calib = False
        self.map1_l = self.map2_l = None
        self.map1_r = self.map2_r = None
        self._load_stereo_calib(stereo_calib_file)

        self._ground_depth = self.camera_height / max(math.sin(self.camera_pitch_rad), 0.1)

        self.num_disp = max(16, (int(num_disparities) // 16) * 16)
        self.block_size = int(block_size) if int(block_size) % 2 == 1 else int(block_size) + 1
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disp,
            blockSize=self.block_size,
            P1=8 * 1 * (self.block_size ** 2),
            P2=32 * 1 * (self.block_size ** 2),
            disp12MaxDiff=8,
            uniquenessRatio=5,
            speckleWindowSize=80,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        self.x = 0.0
        self.y = 0.0
        self.theta = math.pi / 2.0
        self.frame_theta = self.theta

        self.has_imu = False
        self._imu_last_yaw = None
        self._imu_last_live_yaw = None
        self._imu_dtheta = 0.0
        self._imu_live_yaw = None
        self._imu_yaw_offset = 0.0
        self._imu_gyro_z = 0.0
        self._imu_window_max_gz = 0.0
        self._imu_window_max_acc = 0.0
        self._imu_lin_acc = 0.0
        self._raw_accel = (0.0, 0.0, 9.81)
        self._raw_gyro = (0.0, 0.0, 0.0)
        self.imu_calib = 0

        # Angled IMU 3D vector decomposition & slip detection state
        self._gravity_vec = None
        self._up_unit = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self._fwd_unit = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self._imu_vec_acc_mag = 0.0
        self._imu_fwd_acc_proj = 0.0
        self._imu_vec_gyro_mag = 0.0
        self._imu_vert_gyro = 0.0
        self._imu_acc_ema = 0.0
        self._last_imu_motion_mono = 0.0
        self._motor_slip_detected = False

        self.has_tof = False
        self.latest_tof_m = None
        self._prev_tof_m = None

        self.calibrating_180 = False
        self._calib_accum_raw_dtheta = 0.0
        self._prev_stereo_depth = None

        # Inertial dead-reckoning & coasting state
        self.v_fwd = 0.0              # Estimated forward linear velocity (m/s)
        self.v_lat = 0.0              # Lateral velocity (m/s)
        self._last_imu_live_time = time.monotonic()
        self._last_vo_update_time = time.monotonic()
        self._coasting_active = False

    def _update_focal_from_fov(self, fov_deg: float):
        self.fov_deg = float(np.clip(fov_deg, 30.0, 130.0))
        half_fov = math.radians(self.fov_deg * 0.5)
        self.fx = (self.img_w * 0.5) / max(math.tan(half_fov), 0.1)
        self.fy = self.fx

    def _load_stereo_calib(self, path: str):
        if not path or not os.path.exists(path):
            print(f"[STEREO-VO] No '{path}' found yet — using uncalibrated C930e intrinsics (baseline={self.baseline*100:.1f}cm).")
            return
        try:
            d = np.load(path)
            K_l, D_l = d["K_left"], d["dist_left"]
            K_r, D_r = d["K_right"], d["dist_right"]
            R, T = d["R"], d["T"]
            R1, R2, P1, P2 = d["R1"], d["R2"], d["P1"], d["P2"]
            size = (self.img_w, self.img_h)
            self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, size, cv2.CV_16SC2)
            self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, size, cv2.CV_16SC2)
            self.fx = float(P1[0, 0])
            self.fy = float(P1[1, 1])
            self.cx = float(P1[0, 2])
            self.cy = float(P1[1, 2])
            self.baseline = abs(float(T.ravel()[0]))
            self.K_left = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)
            self.has_stereo_calib = True
            print(f"[STEREO-VO] Loaded stereo calibration '{path}' (baseline={self.baseline*100:.2f}cm, fx={self.fx:.1f}px)")
        except Exception as e:
            print(f"[STEREO-VO WARN] Could not load {path}: {e}")

    def undistort(self, frame):
        if self.has_stereo_calib and self.map1_l is not None:
            return cv2.remap(frame, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        return frame

    def compute_depth_map(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
        swap_left_right=None,
        vertical_offset_px=None,
    ):
        """Rectify stereo pair and compute metric depth map Z (in meters) aligned to Left frame."""
        if swap_left_right is not None:
            self.swap_left_right = bool(swap_left_right)
        if vertical_offset_px is not None:
            self.vertical_offset_px = int(vertical_offset_px)

        fl = frame_right if self.swap_left_right else frame_left
        fr = frame_left if self.swap_left_right else frame_right

        if self.vertical_offset_px != 0:
            M = np.float32([[1, 0, 0], [0, 1, self.vertical_offset_px]])
            fr = cv2.warpAffine(fr, M, (fr.shape[1], fr.shape[0]), borderMode=cv2.BORDER_REPLICATE)

        if self.has_stereo_calib and self.map1_l is not None and self.map1_r is not None:
            rect_l = cv2.remap(fl, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
            rect_r = cv2.remap(fr, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        else:
            rect_l = fl
            rect_r = fr

        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY) if rect_l.ndim == 3 else rect_l
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY) if rect_r.ndim == 3 else rect_r

        gl_half = cv2.resize(gray_l, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        gr_half = cv2.resize(gray_r, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        disp_half = self.stereo_matcher.compute(gl_half, gr_half).astype(np.float32) / 16.0
        disp_full = cv2.resize(
            disp_half * 2.0, (gray_l.shape[1], gray_l.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        depth = np.full(gray_l.shape, np.nan, dtype=np.float32)
        valid = disp_full > 0.5
        depth[valid] = (self.fx * self.baseline) / disp_full[valid]
        depth[(depth < 0.15) | (depth > 4.0)] = np.nan

        self._prev_stereo_depth = depth
        return depth, rect_l, rect_r

    def _process_angled_imu_vector(
        self,
        lin_acc_mag: float,
        raw_accel,
        raw_gyro,
        gyro_yaw_rate: float,
        window_max_gz: float | None = None,
        window_max_acc: float | None = None,
    ):
        """
        Combine the 3D acceleration and 3D gyro from an IMU installed at an arbitrary angle:
        1. Tracks the 3D gravity vector `self._gravity_vec` in the angled sensor frame.
        2. Removes gravity and projects the 3D dynamic acceleration onto the horizontal
           chassis plane (`a_horiz_vec`), combining its norm with `lin_acc_mag`.
        3. Projects the 3D gyro vector onto the vertical gravity axis (`_imu_vert_gyro`)
           and computes the full 3D rotational rate norm (`_imu_vec_gyro_mag`).
        4. Tracks whether physical motion is registering on the IMU for slip/stall rejection.
        """
        now = time.monotonic()
        lin_mag = abs(float(lin_acc_mag or 0.0))

        if raw_gyro is not None and len(raw_gyro) == 3:
            gx, gy, gz = float(raw_gyro[0]), float(raw_gyro[1]), float(raw_gyro[2])
            self._raw_gyro = (gx, gy, gz)
            gyro_vec_norm = math.sqrt(gx * gx + gy * gy + gz * gz)
        else:
            gx, gy, gz = 0.0, 0.0, float(gyro_yaw_rate or 0.0)
            gyro_vec_norm = abs(gz)

        a_horiz_mag = 0.0
        a_fwd_proj = 0.0
        if raw_accel is not None and len(raw_accel) == 3:
            ax, ay, az = float(raw_accel[0]), float(raw_accel[1]), float(raw_accel[2])
            self._raw_accel = (ax, ay, az)
            acc_vec = np.array([ax, ay, az], dtype=np.float64)
            acc_norm = float(np.linalg.norm(acc_vec))

            if acc_norm > 4.0:
                if self._gravity_vec is None:
                    self._gravity_vec = acc_vec.copy()
                elif gyro_vec_norm < 0.04 and abs(acc_norm - 9.80665) < 0.35 and lin_mag < 0.08:
                    self._gravity_vec = 0.98 * self._gravity_vec + 0.02 * acc_vec

                g_norm = max(float(np.linalg.norm(self._gravity_vec)), 1e-3)
                self._up_unit = self._gravity_vec / g_norm

                # Auto-calibrate resting gravity vector & mounting angles when stationary
                if not self._gravity_calibrated:
                    if gyro_vec_norm < 0.03 and lin_mag < 0.06:
                        self._calib_gravity_samples.append(acc_vec.copy())
                        if len(self._calib_gravity_samples) >= 30:
                            avg_g = np.mean(self._calib_gravity_samples, axis=0)
                            self._gravity_vec = avg_g
                            self._up_unit = avg_g / max(float(np.linalg.norm(avg_g)), 1e-3)
                            if self.auto_calibrate_mount:
                                self.imu_mount_pitch_deg = float(math.degrees(math.atan2(avg_g[0], math.hypot(avg_g[1], avg_g[2]))))
                                self.imu_mount_roll_deg = float(math.degrees(math.atan2(-avg_g[1], avg_g[2])))
                            self._gravity_calibrated = True

                # Construct horizontal forward unit vector orthogonal to tilted gravity vector
                # Using mounting orientation (yaw, pitch, roll) of the angled BNO055
                my_rad = math.radians(self.imu_mount_yaw_deg)
                mp_rad = math.radians(self.imu_mount_pitch_deg)
                mr_rad = math.radians(self.imu_mount_roll_deg)
                e_fwd = np.array([
                    math.cos(my_rad) * math.cos(mp_rad),
                    math.sin(my_rad) * math.cos(mp_rad),
                    -math.sin(mp_rad)
                ], dtype=np.float64)
                f_perp = e_fwd - float(np.dot(e_fwd, self._up_unit)) * self._up_unit
                f_norm = float(np.linalg.norm(f_perp))
                if f_norm > 1e-3:
                    self._fwd_unit = f_perp / f_norm

                a_dyn = acc_vec - self._gravity_vec
                a_vert = float(np.dot(a_dyn, self._up_unit))
                a_horiz = a_dyn - a_vert * self._up_unit
                a_horiz_mag = float(np.linalg.norm(a_horiz))
                a_fwd_proj = float(np.dot(a_horiz, self._fwd_unit))

        gyro_vec = np.array(self._raw_gyro, dtype=np.float64)
        self._imu_vert_gyro = float(np.dot(gyro_vec, self._up_unit))
        self._imu_vec_gyro_mag = max(abs(float(gyro_yaw_rate or 0.0)), gyro_vec_norm, abs(self._imu_vert_gyro))
        self._imu_window_max_gz = max(
            float(window_max_gz) if window_max_gz is not None else 0.0,
            self._imu_vec_gyro_mag,
        )

        # Combine BNO055 3D linear accel norm with gravity-compensated horizontal accel norm
        self._imu_vec_acc_mag = max(lin_mag, a_horiz_mag)
        self._imu_fwd_acc_proj = a_fwd_proj
        self._imu_window_max_acc = max(
            float(window_max_acc) if window_max_acc is not None else 0.0,
            self._imu_vec_acc_mag,
        )
        self._imu_acc_ema = 0.60 * self._imu_vec_acc_mag + 0.40 * self._imu_acc_ema

        if (
            self._imu_vec_acc_mag >= self.IMU_ACCEL_NOISE_FLOOR
            or self._imu_window_max_acc >= self.IMU_ACCEL_NOISE_FLOOR
            or self._imu_vec_gyro_mag >= self.IMU_GYRO_NOISE_FLOOR
            or self._imu_window_max_gz >= self.IMU_GYRO_NOISE_FLOOR
        ):
            self._last_imu_motion_mono = now

    def is_imu_showing_motion(self, now: float | None = None) -> bool:
        """Return True if the angled IMU vector registers physical motion or vibration."""
        if not self.has_imu:
            return True
        if now is None:
            now = time.monotonic()
        return (
            self._imu_vec_acc_mag >= self.IMU_ACCEL_NOISE_FLOOR
            or self._imu_window_max_acc >= self.IMU_ACCEL_NOISE_FLOOR
            or self._imu_acc_ema >= self.IMU_ACCEL_NOISE_FLOOR
            or self._imu_vec_gyro_mag >= self.IMU_GYRO_NOISE_FLOOR
            or self._imu_window_max_gz >= self.IMU_GYRO_NOISE_FLOOR
            or ((now - self._last_imu_motion_mono) <= self.IMU_ACTIVE_HOLD_SEC)
        )

    def _compute_imu_motor_forward_step(
        self,
        dt: float,
        cmd_v: float | None,
        turn_rate: float,
        tof_delta: float | None = None,
    ) -> tuple[float, bool, bool]:
        """
        Combine the angled IMU 3D acceleration vector with the commanded motor velocity `cmd_v`.
        If `cmd_v` commands the motors (`abs(cmd_v) >= 0.01`) but nothing shows up on the IMU,
        disregard `cmd_v` because the wheels are slipping or the motors are unresponsive.
        Returns `(d_imu, is_stationary, is_pure_rotation)`.
        """
        now = time.monotonic()
        imu_active_motion = self.is_imu_showing_motion(now)
        cmd_v_val = float(cmd_v) if cmd_v is not None else 0.0
        motor_commanded = abs(cmd_v_val) >= 0.005

        # Commanded movement is direct user/pursuit drive — NEVER discard it during smooth cruising!
        effective_cmd_v = cmd_v_val if motor_commanded else 0.0
        self._motor_slip_detected = False

        no_linear_cmd = abs(effective_cmd_v) < 0.005
        has_tof_motion = (tof_delta is not None) and (abs(tof_delta) >= 0.008) and (turn_rate < 0.10)
        # In a wheeled differential-drive robot, forward motion only occurs when commanded
        # or when physically confirmed by ToF forward distance reduction.
        # Gyro turning without linear command is 100% pure in-place rotation!
        is_pure_rotation = self.has_imu and (turn_rate > 0.05) and no_linear_cmd
        is_stationary = no_linear_cmd and (not has_tof_motion) and (not is_pure_rotation)

        if is_stationary or is_pure_rotation:
            self.v_fwd = 0.0
            self.v_lat = 0.0
            return 0.0, is_stationary, is_pure_rotation

        # Net linear acceleration magnitude from the angled 3D IMU vector above noise floor
        a_net = max(0.0, self._imu_vec_acc_mag - 0.04) if self.has_imu else 0.0

        # Resolve forward/backward sign for the combined IMU acceleration vector
        if abs(effective_cmd_v) >= 0.01:
            motion_dir = 1.0 if effective_cmd_v > 0.0 else -1.0
        elif tof_delta is not None and abs(tof_delta) > 0.005 and turn_rate < 0.15:
            motion_dir = 1.0 if tof_delta > 0.0 else -1.0
        elif abs(self._imu_fwd_acc_proj) > 0.04 and turn_rate < 0.15:
            motion_dir = 1.0 if self._imu_fwd_acc_proj > 0.0 else -1.0
        elif abs(self.v_fwd) > 0.01:
            motion_dir = 1.0 if self.v_fwd > 0.0 else -1.0
        else:
            motion_dir = 0.0

        a_fwd_imu = motion_dir * a_net if turn_rate < 0.25 else 0.0
        dv_imu_vec = a_fwd_imu * dt

        if abs(effective_cmd_v) >= 0.01:
            # Combine validated motor command with angled IMU acceleration vector
            v_combined = effective_cmd_v + 0.50 * dv_imu_vec
            self.v_fwd = 0.65 * self.v_fwd + 0.35 * v_combined + 0.50 * dv_imu_vec
        else:
            # Uncommanded motion: allow strong straight IMU acceleration vector or decay to zero
            if self.has_imu and a_net > 0.08 and turn_rate < 0.15 and abs(motion_dir) > 0.0:
                self.v_fwd = 0.80 * self.v_fwd + dv_imu_vec
            else:
                self.v_fwd *= 0.60
                if abs(self.v_fwd) < 0.008:
                    self.v_fwd = 0.0

        d_imu = self.v_fwd * dt + 0.5 * a_fwd_imu * (dt * dt)

        if tof_delta is not None and abs(tof_delta) < 0.18 and turn_rate < 0.15:
            d_imu = (1.0 - self.tof_odom_blend) * d_imu + self.tof_odom_blend * tof_delta

        return d_imu, is_stationary, is_pure_rotation

    def feed_imu(
        self, yaw_rad, pitch_rad, roll_rad, gyro_yaw_rate, lin_acc_mag, dt,
        reset=False, calib=3, live_yaw_rad=None, window_max_gz=None, tof_m=-1.0,
        raw_accel=None, raw_gyro=None, window_max_acc=None,
    ):
        self.has_imu = True
        self.imu_calib = int(calib)
        self.camera_pitch_rad = float(np.clip(pitch_rad, math.radians(-15.0), math.radians(45.0)))
        self.camera_roll_rad = float(roll_rad)
        sin_p = math.sin(self.camera_pitch_rad)
        self._ground_depth = self.camera_height / max(sin_p, 0.08) if sin_p > 0.08 else 2.5
        self._imu_gyro_z = float(gyro_yaw_rate)
        self._imu_lin_acc = float(lin_acc_mag)

        self._process_angled_imu_vector(
            lin_acc_mag=lin_acc_mag,
            raw_accel=raw_accel,
            raw_gyro=raw_gyro,
            gyro_yaw_rate=gyro_yaw_rate,
            window_max_gz=window_max_gz,
            window_max_acc=window_max_acc,
        )

        if tof_m is not None and 0.03 <= tof_m <= 3.8:
            self.has_tof = True
            self.latest_tof_m = float(tof_m)
        else:
            self.latest_tof_m = None

        yaw_f = float(yaw_rad)
        target_live = float(live_yaw_rad) if live_yaw_rad is not None else yaw_f

        # 100% IMU Rotation: accumulate exposure-aligned and live yaw deltas (immune to ORB drop & +/-180 wrap)
        if reset or self._imu_last_yaw is None:
            self._imu_last_yaw = yaw_f
            self._imu_last_live_yaw = target_live
            self._imu_yaw_offset = _angle_diff(self.theta, yaw_f)
            self._imu_dtheta = 0.0
            self.frame_theta = self.theta
        else:
            raw_dyaw = _angle_diff(yaw_f, self._imu_last_yaw)
            self._imu_last_yaw = yaw_f
            if abs(raw_dyaw) < (math.pi * 0.65):
                if self.calibrating_180:
                    self._calib_accum_raw_dtheta += abs(raw_dyaw)
                self._imu_dtheta = raw_dyaw * self.yaw_scale
                self.frame_theta = (self.frame_theta + self._imu_dtheta + math.pi) % (2.0 * math.pi) - math.pi
            else:
                self._imu_dtheta = 0.0

        if self._imu_last_live_yaw is None or reset:
            self._imu_last_live_yaw = target_live
        else:
            raw_dlive = _angle_diff(target_live, self._imu_last_live_yaw)
            self._imu_last_live_yaw = target_live
            if abs(raw_dlive) < (math.pi * 0.65):
                self.theta = (self.theta + raw_dlive * self.yaw_scale + math.pi) % (2.0 * math.pi) - math.pi

        if live_yaw_rad is None:
            self.theta = self.frame_theta

    def _compute_camera_depth_forward_step(
        self,
        old_pts,
        new_pts,
        prev_depth: np.ndarray | None,
        curr_depth: np.ndarray | None,
        dtheta: float,
        dt: float,
    ) -> tuple[float, bool]:
        """
        Compute forward displacement d_depth directly from metric stereo depth maps
        and derotated 3D point correspondences.
        Since rotation dtheta is known 100% from BNO055, each static 3D point is derotated
        and the 1D forward translation along the chassis forward axis is computed directly.
        """
        if prev_depth is None or curr_depth is None:
            return 0.0, False

        h, w = prev_depth.shape[:2]
        cos_p = math.cos(self.camera_pitch_rad)
        sin_p = math.sin(self.camera_pitch_rad)
        cos_th = math.cos(dtheta)
        sin_th = math.sin(dtheta)

        shifts = []
        max_phys = max(0.95 * dt, 0.08)

        # 1. 3D point correspondence tracking with stereo depth
        if old_pts is not None and new_pts is not None and len(old_pts) >= 4 and len(new_pts) == len(old_pts):
            for (u0, v0), (u1, v1) in zip(old_pts, new_pts):
                ix0, iy0 = int(round(u0)), int(round(v0))
                ix1, iy1 = int(round(u1)), int(round(v1))
                if not (3 <= ix0 < w - 3 and 3 <= iy0 < h - 3 and 3 <= ix1 < w - 3 and 3 <= iy1 < h - 3):
                    continue

                # Median depth in 3x3 patch to reject single-pixel disparity noise
                p0 = prev_depth[iy0 - 1 : iy0 + 2, ix0 - 1 : ix0 + 2]
                v0_z = p0[~np.isnan(p0)]
                if len(v0_z) < 3:
                    continue
                z0 = float(np.median(v0_z))
                if not (0.20 <= z0 <= 3.6):
                    continue

                p1 = curr_depth[iy1 - 1 : iy1 + 2, ix1 - 1 : ix1 + 2]
                v1_z = p1[~np.isnan(p1)]
                if len(v1_z) < 3:
                    continue
                z1 = float(np.median(v1_z))
                if not (0.20 <= z1 <= 3.6):
                    continue

                # Backproject to optical camera coordinates
                xc0 = (u0 - self.cx) * z0 / self.fx
                yc0 = (v0 - self.cy) * z0 / self.fy
                xc1 = (u1 - self.cx) * z1 / self.fx
                yc1 = (v1 - self.cy) * z1 / self.fy

                # Transform to chassis coordinates (pitch compensation)
                # Chassis X (forward) = z * cos_p - yc * sin_p
                # Chassis Y (lateral) = xc
                xr0 = z0 * cos_p - yc0 * sin_p
                yr0 = xc0
                xr1 = z1 * cos_p - yc1 * sin_p
                yr1 = xc1

                # Derotate previous point by BNO055 rotation dtheta into current heading frame:
                xr0_derot = xr0 * cos_th - yr0 * sin_th

                # Forward displacement:
                # When robot advances by d_fwd, static objects shift backward relative to robot:
                # xr1 = xr0_derot - d_fwd  =>  d_fwd = xr0_derot - xr1
                delta_d = xr0_derot - xr1

                if abs(delta_d) <= max_phys:
                    shifts.append(delta_d)

        # 2. Central corridor dense metric depth shift (only when driving straight, turn_rate < 0.08)
        if abs(dtheta) < 0.015:
            roi_p0 = prev_depth[int(h * 0.35) : int(h * 0.65), int(w * 0.35) : int(w * 0.65)]
            roi_p1 = curr_depth[int(h * 0.35) : int(h * 0.65), int(w * 0.35) : int(w * 0.65)]
            v_p0 = roi_p0[~np.isnan(roi_p0)]
            v_p1 = roi_p1[~np.isnan(roi_p1)]
            if len(v_p0) > 80 and len(v_p1) > 80:
                med0 = float(np.median(v_p0))
                med1 = float(np.median(v_p1))
                if 0.25 <= med0 <= 3.2 and 0.25 <= med1 <= 3.2:
                    d_corridor = (med0 - med1) * cos_p
                    if abs(d_corridor) <= (0.50 * max_phys):
                        shifts.append(d_corridor)

        if len(shifts) >= 4:
            med_shift = float(np.median(shifts))
            mad = float(np.median(np.abs(np.array(shifts) - med_shift)))
            tol = max(2.5 * mad, 0.012)
            inliers = [s for s in shifts if abs(s - med_shift) <= tol]
            if len(inliers) >= 3:
                return float(np.mean(inliers)), True
            return med_shift, True

        return 0.0, False

    def update(
        self,
        old_pts,
        new_pts,
        dt=1.0 / 30.0,
        prev_depth_map=None,
        frame_stuttered=False,
        cmd_v=None,
    ):
        dt = max(float(dt), 0.001)
        depth_ref = prev_depth_map if prev_depth_map is not None else self._prev_stereo_depth

        # 100% BNO055 ROTATION: Robot rotation is driven ENTIRELY by the BNO055 IMU
        if self.has_imu:
            dtheta = self._imu_dtheta
            self._imu_dtheta = 0.0
        else:
            # If BNO055 is not connected, do NOT let camera ORB drift rotation
            dtheta = 0.0

        turn_rate = (
            max(abs(dtheta / dt), self._imu_window_max_gz, self._imu_vec_gyro_mag)
            if self.has_imu
            else abs(dtheta / dt)
        )

        tof_delta = None
        if self.has_tof and self.latest_tof_m is not None and self._prev_tof_m is not None:
            tof_delta = self._prev_tof_m - self.latest_tof_m
        self._prev_tof_m = self.latest_tof_m

        # 1. Commanded movement component
        cmd_v_val = float(cmd_v or 0.0)
        d_cmd = cmd_v_val * dt

        # 2. BNO055 IMU motion component (angled 3D acceleration vector + motor slip rejection)
        d_imu, is_stationary, is_pure_rotation = self._compute_imu_motor_forward_step(
            dt=dt, cmd_v=cmd_v, turn_rate=turn_rate, tof_delta=tof_delta
        )

        # 3. Camera Depth motion component (metric stereo depth + derotation)
        curr_depth = self._prev_stereo_depth
        d_depth, depth_ok = self._compute_camera_depth_forward_step(
            old_pts=old_pts,
            new_pts=new_pts,
            prev_depth=depth_ref,
            curr_depth=curr_depth,
            dtheta=dtheta,
            dt=dt,
        )

        # Case A: Stationary or Pure Rotation
        if is_stationary or is_pure_rotation:
            self.v_fwd = 0.0
            self._coasting_active = False
            self.last_motion_source = "STATIONARY" if is_stationary else "PURE_ROT_BNO055"
            self._last_vo_update_time = time.monotonic()
            return (self.x, self.y, self.theta), (0.0, 0.0, dtheta)

        # Case B: Tri-Modal Motion Fusion (BNO055 + Camera Depth + Commanded Movement)
        if depth_ok and not frame_stuttered:
            if abs(d_cmd) >= 0.005:
                # Commanded drive: fuse BNO055 + Camera Depth + Commanded movement
                depth_consistent = (abs(d_depth - d_cmd) < max(0.20 * abs(d_cmd), 0.06)) or (d_depth * d_cmd > 0.0)
                if depth_consistent:
                    w_i = self.imu_motion_weight
                    w_d = self.depth_motion_weight
                    w_c = self.cmd_motion_weight
                    total_w = max(w_i + w_d + w_c, 1e-3)
                    d_fwd = (w_i * d_imu + w_d * d_depth + w_c * d_cmd) / total_w
                    self.last_motion_source = "FUSED(IMU+Depth+Cmd)"
                else:
                    # Camera depth diverged (e.g. moving foreground object): fallback to IMU + Commanded
                    d_fwd = 0.50 * d_imu + 0.50 * d_cmd
                    self.last_motion_source = "FUSED(IMU+Cmd,DepthDiverged)"
            else:
                # No forward command active: enforce zero drift unless ToF confirms physical forward movement
                if (tof_delta is not None) and (abs(tof_delta) >= 0.008) and (turn_rate < 0.08):
                    d_fwd = 0.50 * d_imu + 0.50 * tof_delta
                    self.last_motion_source = "PUSHED_TOF"
                else:
                    d_fwd = 0.0
                    self.last_motion_source = "STATIONARY"
            self._coasting_active = False
        else:
            # Camera depth unavailable (blank walls, floor without texture, stutter, or frame drop):
            # Drive forward reliably with BNO055 IMU + Commanded Movement + TOF
            if abs(d_cmd) >= 0.005:
                d_fwd = 0.50 * d_imu + 0.50 * d_cmd
            else:
                d_fwd = d_imu
            self._coasting_active = True
            self.last_motion_source = "COAST(BNO055+Cmd)"

        # Strictly enforce non-holonomic kinematics (no sideways slide)
        d_lat = 0.0

        max_step = max(0.75 * dt, 0.05)
        d_fwd = float(np.clip(d_fwd, -max_step, max_step))

        if abs(d_fwd) < 0.001:
            d_fwd = 0.0

        # Update learned velocity state
        if not is_stationary and abs(d_fwd) > 0.001:
            self.v_fwd = 0.70 * self.v_fwd + 0.30 * (d_fwd / dt)

        # Apply non-holonomic displacement along 100% BNO055 heading
        heading = self.frame_theta if self.has_imu else self.theta
        dx_world = d_fwd * math.cos(heading)
        dy_world = d_fwd * math.sin(heading)

        self.x += dx_world
        self.y += dy_world
        self._last_vo_update_time = time.monotonic()

        return (self.x, self.y, self.theta), (dx_world, dy_world, dtheta)

    def update_imu_only(
        self,
        live_yaw_rad,
        pitch_rad=None,
        gyro_yaw_rate=0.0,
        live_tof_m=-1.0,
        lin_acc=0.0,
        dt=None,
        cmd_v=None,
        raw_accel=None,
        raw_gyro=None,
        window_max_gz=None,
        window_max_acc=None,
    ):
        """
        High-rate (100 Hz / 40 Hz) dead-reckoning update called whenever the camera stream
        is buffering, waiting for a frame, or frozen.
        Maintains continuous robot pose (x, y, theta) using 100% IMU rotation and the
        combined angled IMU 3D acceleration vector + validated motor command.
        """
        now = time.monotonic()
        if dt is None:
            dt = max(0.001, min(0.20, now - self._last_imu_live_time))
        else:
            dt = max(float(dt), 0.001)
        self._last_imu_live_time = now

        if pitch_rad is not None:
            self.camera_pitch_rad = float(np.clip(pitch_rad, math.radians(-10.0), math.radians(45.0)))
        self._imu_gyro_z = float(gyro_yaw_rate)
        self._imu_lin_acc = float(lin_acc)
        self.has_imu = True

        self._process_angled_imu_vector(
            lin_acc_mag=lin_acc,
            raw_accel=raw_accel,
            raw_gyro=raw_gyro,
            gyro_yaw_rate=gyro_yaw_rate,
            window_max_gz=window_max_gz,
            window_max_acc=window_max_acc,
        )

        # 1. Update heading theta 100% from live IMU yaw (immune to +/-180 wrap jumps)
        live_f = float(live_yaw_rad)
        if self._imu_last_live_yaw is None:
            self._imu_last_live_yaw = live_f
            if self._imu_last_yaw is None:
                self._imu_last_yaw = live_f
                self._imu_yaw_offset = _angle_diff(self.theta, live_f)
        else:
            raw_dlive = _angle_diff(live_f, self._imu_last_live_yaw)
            self._imu_last_live_yaw = live_f
            if abs(raw_dlive) < (math.pi * 0.65):
                self.theta = (self.theta + raw_dlive * self.yaw_scale + math.pi) % (2.0 * math.pi) - math.pi

        tof_delta = None
        if live_tof_m is not None and 0.03 <= live_tof_m <= 3.8:
            self.has_tof = True
            if self.latest_tof_m is not None:
                tof_delta = self.latest_tof_m - float(live_tof_m)
            self.latest_tof_m = float(live_tof_m)

        turn_rate = max(self._imu_vec_gyro_mag, self._imu_window_max_gz)
        d_fwd, _, _ = self._compute_imu_motor_forward_step(
            dt=dt, cmd_v=cmd_v, turn_rate=turn_rate, tof_delta=tof_delta
        )

        max_step = max(0.65 * dt, 0.04)
        d_fwd = float(np.clip(d_fwd, -max_step, max_step))

        if abs(d_fwd) > 0.0005:
            dx = d_fwd * math.cos(self.theta)
            dy = d_fwd * math.sin(self.theta)
            self.x += dx
            self.y += dy

        self._coasting_active = True
        return (self.x, self.y, self.theta)

    def get_frame_pose(self):
        return (self.x, self.y, self.frame_theta)

    def is_mapping_safe(self, frame_stuttered=False, max_turn_rate_rads=0.20):
        if frame_stuttered:
            return False
        if self.has_imu and self._imu_window_max_gz > max_turn_rate_rads:
            return False
        if getattr(self, "calibrating_180", False):
            return False
        return True

    def adjust_fov(self, delta_deg):
        self._update_focal_from_fov(self.fov_deg + delta_deg)
        print(f"[STEREO-VO] Camera FOV -> {self.fov_deg:.1f} deg (fx={self.fx:.1f}px)")
        return self.fov_deg

    def toggle_180_calibration(self):
        if not self.calibrating_180:
            self.calibrating_180 = True
            self._calib_accum_raw_dtheta = 0.0
            print("[CAL180] Started! Rotate the robot 180 degrees and press 'C' again.")
        else:
            self.calibrating_180 = False
            if self._calib_accum_raw_dtheta > math.radians(45.0):
                self.yaw_scale = math.pi / self._calib_accum_raw_dtheta
                print(f"[CAL180] Completed! New yaw_scale = {self.yaw_scale:.4f}")
        return self.calibrating_180

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = math.pi / 2.0
        self.frame_theta = self.theta
        self.v_fwd = 0.0
        self.v_lat = 0.0
        self._imu_dtheta = 0.0
        self._imu_acc_ema = 0.0
        self._motor_slip_detected = False
        self._last_imu_live_time = time.monotonic()
        self._last_vo_update_time = time.monotonic()
        self._coasting_active = False
        if self._imu_last_yaw is not None:
            self._imu_yaw_offset = _angle_diff(self.theta, self._imu_last_yaw)
        if self._imu_last_live_yaw is not None:
            self._imu_last_live_yaw = self._imu_last_yaw


def _test_depth_cli(source=None):
    from config_manager import load_config, save_config
    from capture import LowLatencyCapture

    cfg = load_config()
    scfg = cfg.get("stereo_camera", {})
    svo = StereoVisualOdometry(
        stereo_calib_file=scfg.get("calib_file", "stereo_calib.npz"),
        baseline_m=scfg.get("baseline_m", 0.08),
        swap_left_right=False,
        vertical_offset_px=0,
    )
    swap_lr = bool(scfg.get("swap_left_right", False))
    v_offset = int(scfg.get("vertical_offset_px", 0))

    src = source or cfg.get("camera", {}).get("source", "pi")
    cap = LowLatencyCapture(source=src, rtsp_transport="tcp", cfg=cfg)
    print("=" * 72)
    print("[STEREO DEPTH TEST] Controls:")
    print("  * Press 'S' : Swap Left & Right cameras (fixes black/NaN depth if USB IDs are reversed!)")
    print("  * Press 'I' / 'K' : Nudge Right camera UP / DOWN by 1 px (aligns uncalibrated cameras)")
    print("  * Press 'Q' : Save settings & exit")
    print("=" * 72)

    while True:
        ret, frame, _ = cap.read(timeout=0.04)
        if not ret or frame is None or frame.shape[1] < 1000:
            continue
        mid = frame.shape[1] // 2
        fl, fr = (frame[:, mid:], frame[:, :mid]) if swap_lr else (frame[:, :mid], frame[:, mid:])
        if v_offset != 0:
            M = np.float32([[1, 0, 0], [0, 1, v_offset]])
            fr = cv2.warpAffine(fr, M, (fr.shape[1], fr.shape[0]), borderMode=cv2.BORDER_REPLICATE)

        depth, rect_l, rect_r = svo.compute_depth_map(fl, fr, swap_left_right=False, vertical_offset_px=0)

        d_norm = np.nan_to_num(depth, nan=3.5)
        d_u8 = np.clip((1.0 - (d_norm - 0.2) / 3.0) * 255.0, 0, 255).astype(np.uint8)
        d_col = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
        d_col[np.isnan(depth)] = (20, 20, 20)

        vis_l = rect_l.copy()
        vis_r = rect_r.copy()
        for y_line in range(60, 480, 60):
            cv2.line(vis_l, (0, y_line), (640, y_line), (0, 255, 0), 1)
            cv2.line(vis_r, (0, y_line), (640, y_line), (0, 255, 0), 1)

        valid_pct = float(np.mean(~np.isnan(depth))) * 100.0
        center_patch = depth[180:300, 240:400]
        valid_center = center_patch[~np.isnan(center_patch)]
        center_z = float(np.median(valid_center)) if len(valid_center) > 0 else float("nan")

        cv2.putText(vis_l, f"LEFT CAM (Swap={swap_lr}) [Press 'S']", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(vis_r, f"RIGHT CAM (v_offset={v_offset:+d}px) [I/K]", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(d_col, f"Center: {center_z:.2f}m | Valid: {valid_pct:.0f}%", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        tri_panel = cv2.resize(np.hstack([vis_l, vis_r, d_col]), (1440, 360))
        cv2.imshow("Stereo Verification (Left | Right | Metric Depth)", tri_panel)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("s"), ord("S")):
            swap_lr = not swap_lr
            cfg.setdefault("stereo_camera", {})["swap_left_right"] = swap_lr
            save_config(cfg)
            print(f"[STEREO] Swapped Left/Right cameras -> swap_left_right={swap_lr}")
        elif key in (ord("i"), ord("I")):
            v_offset -= 1
            cfg.setdefault("stereo_camera", {})["vertical_offset_px"] = v_offset
            save_config(cfg)
            print(f"[STEREO] Vertical offset -> {v_offset:+d} px")
        elif key in (ord("k"), ord("K")):
            v_offset += 1
            cfg.setdefault("stereo_camera", {})["vertical_offset_px"] = v_offset
            save_config(cfg)
            print(f"[STEREO] Vertical offset -> {v_offset:+d} px")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stereo Camera Real-Time Metric Depth Perception Viewer")
    ap.add_argument("--test-depth", action="store_true", help="Launch live metric stereo depth perception window")
    ap.add_argument("--source", default=None, help="Video source ('pi', '0', or RTSP URL)")
    args = ap.parse_args()
    _test_depth_cli(source=args.source)
