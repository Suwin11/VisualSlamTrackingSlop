"""
visual_odometry.py
Visual-Inertial-Laser Odometry (VO) & Robot Pose Tracking for a Mobile Robot.

Key Features:
1. 100% IMU Rotation for Localization:
   - Relies entirely on time-aligned BNO055/MPU-6500 IMU yaw and 3D gyro vector for
     robot rotation (`theta`, `frame_theta`, `dtheta`) so ORB feature dropouts never
     cause heading loss or rotational drift.
2. Angled IMU 3D Acceleration Vector + Motor Command Fusion + Slip Rejection:
   - Tracks the 3D gravity vector in the angled IMU frame, removes gravity, projects
     dynamic acceleration onto the horizontal plane, and combines it with `lin_acc_mag`.
   - Integrates the combined IMU acceleration vector into forward motion alongside
     the commanded motor velocity (`cmd_v`).
   - If the program commands the motors (`abs(cmd_v) >= 0.01`) but nothing shows up on
     the IMU, disregards `cmd_v` because the wheels are slipping or motors are not responding.
3. Non-Holonomic Differential Drive Constraint:
   - Strictly enforces `raw_dx = 0.0` (forward/backward translation + yaw rotation only).
"""

import math
import os
import time
import cv2
import numpy as np


def _wrap_pi(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


class VisualOdometry:
    """
    Monocular, 3D-PnP & ToF-Laser Visual-Inertial Odometry tracking robot pose (x, y, theta).
    """

    IMU_ACCEL_NOISE_FLOOR = 0.045  # m/s^2 (combined 3D linear acceleration vector floor)
    IMU_GYRO_NOISE_FLOOR  = 0.025  # rad/s (combined 3D angular velocity vector floor)
    IMU_ACTIVE_HOLD_SEC   = 0.18   # s (short continuity window across 100Hz IMU samples)

    def __init__(self, camera_height_m=0.20, camera_pitch_deg=15.0,
                 fov_deg=None, focal_length_px=None, img_w=640, img_h=480,
                 calib_file="camera_pi.npz", enable_undistort=True,
                 yaw_scale=1.0, tof_odom_blend=0.35, imu_motion_weight=0.80):
        self.camera_height = float(camera_height_m)
        self.camera_pitch_rad = math.radians(camera_pitch_deg)
        self.nominal_pitch_rad = self.camera_pitch_rad
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.enable_undistort = bool(enable_undistort)
        self.yaw_scale = float(yaw_scale)
        self.tof_odom_blend = float(np.clip(tof_odom_blend, 0.0, 0.85))
        self.imu_motion_weight = float(np.clip(imu_motion_weight, 0.0, 1.0))
        self.dist = None
        self.map1 = None
        self.map2 = None
        self._map_size = (self.img_w, self.img_h)

        if fov_deg is None:
            self.fov_deg = 62.2 if self.enable_undistort else 70.0
        else:
            self.fov_deg = float(fov_deg)

        resolved_calib = None
        if calib_file:
            if os.path.exists(calib_file):
                resolved_calib = calib_file
            else:
                candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), calib_file)
                if os.path.exists(candidate):
                    resolved_calib = candidate

        if resolved_calib and self.enable_undistort:
            data = np.load(resolved_calib)
            self.K_calib = data["K"].astype(np.float64)
            self.dist = data["dist"].astype(np.float64)
            self.cx = float(self.K_calib[0, 2])
            self.cy = float(self.K_calib[1, 2])
            self.new_K, _ = cv2.getOptimalNewCameraMatrix(
                self.K_calib, self.dist, (self.img_w, self.img_h), alpha=0.0, newImgSize=(self.img_w, self.img_h)
            )
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.K_calib, self.dist, None, self.new_K, (self.img_w, self.img_h), cv2.CV_16SC2
            )
        else:
            self.K_calib = None
            self.new_K = None
            self.cx = self.img_w / 2.0
            self.cy = self.img_h / 2.0

        if focal_length_px is not None:
            self.fx = float(focal_length_px)
            self.fy = float(focal_length_px)
            self.fov_deg = math.degrees(2.0 * math.atan(self.img_w / (2.0 * self.fx)))
        else:
            self._update_focal_from_fov(self.fov_deg)

        self._rebuild_K()
        print(f"[VO] Initialized: FOV={self.fov_deg:.1f}deg | fx={self.fx:.1f}px | "
              f"pitch={camera_pitch_deg:.1f}deg | undistort={self.enable_undistort} | "
              f"imu_motion_weight={self.imu_motion_weight:.2f}")

        self._ground_depth = self.camera_height / max(math.sin(self.camera_pitch_rad), 0.1)

        # Global pose: start at (0, 0) facing +Y (North = pi/2 rad = +90 deg)
        self.x = 0.0
        self.y = 0.0
        self.theta = math.pi / 2.0
        self.frame_theta = math.pi / 2.0

        self.metric_scale = 1.0
        self.calibrating_180 = False
        self._calib_accum_raw_dtheta = 0.0

        # IMU & TOF400C state
        self._imu_yaw = None
        self._prev_imu_yaw = None
        self._prev_live_imu_yaw = None
        self._imu_yaw_offset = None
        self._pending_imu_dtheta = 0.0
        self._pending_exposure_dtheta = 0.0
        self._last_imu_time = 0.0
        self._prev_imu_pitch = self.camera_pitch_rad
        self._imu_gyro_z = 0.0
        self._window_max_gz = 0.0
        self._window_max_acc = 0.0
        self._imu_stationary = False
        self._imu_linear_accel = 0.0
        self._raw_accel = (0.0, 0.0, 9.81)
        self._raw_gyro = (0.0, 0.0, 0.0)
        self._use_imu_yaw = False
        self._last_frame_stuttered = False
        self._coasting_active = False

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

        # TOF400C laser range state
        self.latest_tof_m = -1.0
        self._prev_tof_m = -1.0
        self._pending_tof_dy = None
        self._last_tof_time = 0.0

        # Linear velocity EMA filter
        self._ema_dx = 0.0
        self._ema_dy = 0.0
        self._ema_dtheta = 0.0
        self._ema_alpha = 0.55
        self.v_fwd = 0.0
        self._last_imu_live_time = time.monotonic()

    def _rebuild_K(self):
        self.K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def _update_focal_from_fov(self, fov_deg):
        self.fov_deg = float(np.clip(fov_deg, 30.0, 150.0))
        half_fov_rad = math.radians(self.fov_deg * 0.5)
        self.fx = (self.img_w * 0.5) / max(math.tan(half_fov_rad), 0.1)
        self.fy = self.fx
        self._rebuild_K()

    def adjust_fov(self, delta_deg):
        self._update_focal_from_fov(self.fov_deg + delta_deg)
        print(f"[VO] Adjusted Camera Horizontal FOV -> {self.fov_deg:.1f} deg (fx={self.fx:.1f} px)")

    def toggle_180_calibration(self):
        if not self.calibrating_180:
            self.calibrating_180 = True
            self._calib_accum_raw_dtheta = 0.0
            print("[VO-CALIB] 180-Deg Calibration STARTED! Rotate robot/camera 180 deg, then press 'C' again.")
            return True
        else:
            self.calibrating_180 = False
            measured_rad = abs(self._calib_accum_raw_dtheta)
            measured_deg = math.degrees(measured_rad)
            if measured_deg > 25.0:
                correction = 180.0 / measured_deg
                self.yaw_scale = float(np.clip(self.yaw_scale * correction, 0.25, 4.0))
                eff_fov = math.degrees(2.0 * math.atan((self.img_w * 0.5) / (self.fx / self.yaw_scale)))
                print(f"[VO-CALIB] COMPLETE! Measured {measured_deg:.1f} deg on 180 deg turn -> "
                      f"New yaw_scale={self.yaw_scale:.3f} (Effective FOV={eff_fov:.1f} deg)")
            else:
                print(f"[VO-CALIB] Cancelled (only {measured_deg:.1f} deg of rotation detected).")
            return False

    def undistort(self, frame):
        h, w = frame.shape[:2]
        if (w, h) != (self.img_w, self.img_h):
            sx = w / float(self.img_w)
            sy = h / float(self.img_h)
            self.img_w, self.img_h = w, h
            self.cx *= sx
            self.cy *= sy
            self._update_focal_from_fov(self.fov_deg)
            if self.enable_undistort and self.K_calib is not None and self.dist is not None:
                K_s = self.K_calib.copy()
                K_s[0, 0] *= sx
                K_s[0, 2] *= sx
                K_s[1, 1] *= sy
                K_s[1, 2] *= sy
                new_K_s, _ = cv2.getOptimalNewCameraMatrix(K_s, self.dist, (w, h), alpha=0.0, newImgSize=(w, h))
                self.map1, self.map2 = cv2.initUndistortRectifyMap(
                    K_s, self.dist, None, new_K_s, (w, h), cv2.CV_16SC2
                )
                self._map_size = (w, h)

        if self.enable_undistort and self.map1 is not None and self.map2 is not None:
            return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        return frame

    def reset(self, init_x=0.0, init_y=0.0, init_theta=math.pi / 2.0):
        self.x = float(init_x)
        self.y = float(init_y)
        self.theta = float(init_theta)
        self.frame_theta = float(init_theta)
        self.v_fwd = 0.0
        self._last_imu_live_time = time.monotonic()
        self._ema_dx = 0.0
        self._ema_dy = 0.0
        self._ema_dtheta = 0.0
        self._pending_imu_dtheta = 0.0
        self._pending_exposure_dtheta = 0.0
        self._pending_tof_dy = None
        self._prev_tof_m = -1.0
        self._prev_imu_pitch = self.camera_pitch_rad
        self._imu_stationary = False
        self._coasting_active = False
        self._imu_linear_accel = 0.0
        self._imu_acc_ema = 0.0
        self._motor_slip_detected = False
        if self._imu_yaw is not None:
            self._prev_imu_yaw = self._imu_yaw
            self._prev_live_imu_yaw = self._imu_yaw
            self._imu_yaw_offset = self._imu_yaw - init_theta

    def _process_angled_imu_vector(
        self,
        lin_acc_mag: float | None,
        raw_accel,
        raw_gyro,
        gyro_yaw_rate: float | None,
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

                cp = math.cos(self.nominal_pitch_rad)
                sp = math.sin(self.nominal_pitch_rad)
                e_fwd = np.array([cp, 0.0, -sp], dtype=np.float64)
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
        self._window_max_gz = max(
            float(window_max_gz) if window_max_gz is not None else 0.0,
            self._imu_vec_gyro_mag,
        )

        self._imu_vec_acc_mag = max(lin_mag, a_horiz_mag)
        self._imu_fwd_acc_proj = a_fwd_proj
        self._window_max_acc = max(
            float(window_max_acc) if window_max_acc is not None else 0.0,
            self._imu_vec_acc_mag,
        )
        self._imu_acc_ema = 0.60 * self._imu_vec_acc_mag + 0.40 * self._imu_acc_ema

        if (
            self._imu_vec_acc_mag >= self.IMU_ACCEL_NOISE_FLOOR
            or self._window_max_acc >= self.IMU_ACCEL_NOISE_FLOOR
            or self._imu_vec_gyro_mag >= self.IMU_GYRO_NOISE_FLOOR
            or self._window_max_gz >= self.IMU_GYRO_NOISE_FLOOR
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
            or self._window_max_acc >= self.IMU_ACCEL_NOISE_FLOOR
            or self._imu_acc_ema >= self.IMU_ACCEL_NOISE_FLOOR
            or self._imu_vec_gyro_mag >= self.IMU_GYRO_NOISE_FLOOR
            or self._window_max_gz >= self.IMU_GYRO_NOISE_FLOOR
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
        imu_active = self.has_imu
        imu_active_motion = self.is_imu_showing_motion(now)
        cmd_v_val = float(cmd_v) if cmd_v is not None else 0.0
        motor_commanded = abs(cmd_v_val) >= 0.005

        # Commanded movement is direct user/pursuit drive — NEVER discard it during smooth cruising!
        effective_cmd_v = cmd_v_val if motor_commanded else 0.0
        self._motor_slip_detected = False

        no_linear_cmd = abs(effective_cmd_v) < 0.005
        has_tof_motion = (tof_delta is not None) and (abs(tof_delta) >= 0.008) and (turn_rate < 0.10)
        is_pure_rotation = imu_active and (turn_rate > 0.05) and no_linear_cmd
        is_stationary = no_linear_cmd and (not has_tof_motion) and (not is_pure_rotation)

        if is_stationary or is_pure_rotation:
            self.v_fwd = 0.0
            return 0.0, is_stationary, is_pure_rotation

        a_net = max(0.0, self._imu_vec_acc_mag - 0.04) if imu_active else 0.0

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
            v_combined = effective_cmd_v + 0.50 * dv_imu_vec
            self.v_fwd = 0.65 * self.v_fwd + 0.35 * v_combined + 0.50 * dv_imu_vec
        else:
            if imu_active and a_net > 0.08 and turn_rate < 0.15 and abs(motion_dir) > 0.0:
                self.v_fwd = 0.80 * self.v_fwd + dv_imu_vec
            else:
                self.v_fwd *= 0.60
                if abs(self.v_fwd) < 0.008:
                    self.v_fwd = 0.0

        d_imu = self.v_fwd * dt + 0.5 * a_fwd_imu * (dt * dt)

        if tof_delta is not None and abs(tof_delta) < 0.18 and turn_rate < 0.15:
            w_tof = self.tof_odom_blend
            d_imu = (1.0 - w_tof) * d_imu + w_tof * tof_delta

        return d_imu, is_stationary, is_pure_rotation

    def _project_pixels_to_ground(self, pts, pitch_rad):
        xn = (pts[:, 0] - self.cx) / self.fx
        yn = (pts[:, 1] - self.cy) / self.fy

        sin_p = math.sin(pitch_rad)
        cos_p = math.cos(pitch_rad)

        y_down = yn * cos_p + sin_p
        z_fwd = cos_p - yn * sin_p

        valid = y_down > 0.06
        denom = np.where(valid, y_down, 1.0)
        Y_R = self.camera_height * (z_fwd / denom)
        X_R = self.camera_height * (xn / denom)
        valid &= (Y_R >= 0.15) & (Y_R <= 4.0) & (np.abs(X_R) <= 2.5)
        return X_R, Y_R, valid

    def _estimate_visual_yaw(self, old_pts, new_pts, inlier_mask):
        p0 = old_pts[inlier_mask]
        p1 = new_pts[inlier_mask]

        xn0 = (p0[:, 0] - self.cx) / self.fx
        xn1 = (p1[:, 0] - self.cx) / self.fx
        yn_mid = (0.5 * (p0[:, 1] + p1[:, 1]) - self.cy) / self.fy
        xn_mid = 0.5 * (xn0 + xn1)

        d_alpha = np.arctan(xn1) - np.arctan(xn0)

        cos_p = math.cos(self.camera_pitch_rad)
        sin_p = math.sin(self.camera_pitch_rad)
        pitch_factor = cos_p - (yn_mid / (1.0 + xn_mid * xn_mid)) * sin_p
        pitch_factor = np.clip(pitch_factor, 0.40, 1.25)

        dtheta_per_pt = (d_alpha / pitch_factor) * self.yaw_scale
        return float(np.median(dtheta_per_pt))

    def _solve_pnp_motion(self, old_pts, new_pts, prev_depth_map):
        if prev_depth_map is None or len(old_pts) < 8:
            return 0.0, 0.0, False

        h, w = prev_depth_map.shape[:2]
        u0 = np.clip(old_pts[:, 0].astype(np.int32), 0, w - 1)
        v0 = np.clip(old_pts[:, 1].astype(np.int32), 0, h - 1)
        z0 = prev_depth_map[v0, u0]

        valid = (z0 >= 0.18) & (z0 <= 3.5)
        if np.sum(valid) < 8:
            return 0.0, 0.0, False

        p0 = old_pts[valid]
        p1 = new_pts[valid]
        z_v = z0[valid]

        X0 = ((p0[:, 0] - self.cx) / self.fx) * z_v
        Y0 = ((p0[:, 1] - self.cy) / self.fy) * z_v
        pts_3d = np.column_stack([X0, Y0, z_v]).astype(np.float64)
        pts_2d = p1.reshape(-1, 1, 2).astype(np.float64)

        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d, pts_2d, self.K, None,
                iterationsCount=60, reprojectionError=3.5, confidence=0.98,
                flags=cv2.SOLVEPNP_EPNP
            )
            if not ok or inliers is None or len(inliers) < 6:
                return 0.0, 0.0, False

            R, _ = cv2.Rodrigues(rvec)
            cam_motion = -R.T @ tvec.ravel()
            cos_p = math.cos(self.camera_pitch_rad)
            sin_p = math.sin(self.camera_pitch_rad)
            pnp_dy = float(cam_motion[2] * cos_p - cam_motion[1] * sin_p) * self.metric_scale
            pnp_dtheta = float(rvec[1, 0] * cos_p + rvec[2, 0] * sin_p) * self.yaw_scale
            return pnp_dy, pnp_dtheta, True
        except Exception:
            return 0.0, 0.0, False

    def update_imu_only(
        self,
        live_yaw_rad,
        pitch_rad=None,
        gyro_yaw_rate=None,
        live_tof_m=None,
        lin_acc=0.0,
        dt=None,
        cmd_v=None,
        raw_accel=None,
        raw_gyro=None,
        window_max_gz=None,
        window_max_acc=None,
    ):
        """
        Update robot heading (`self.theta`) 100% from live IMU packets and forward translation
        (`self.x, self.y`) from the combined angled IMU 3D acceleration vector + validated
        motor command when the camera stream is stuttering, buffering, or between frames.
        """
        now = time.monotonic()
        if dt is None:
            dt = max(0.001, min(0.20, now - self._last_imu_live_time))
        else:
            dt = max(float(dt), 0.001)
        self._last_imu_live_time = now

        if live_tof_m is not None and live_tof_m > 0.0:
            self.latest_tof_m = float(live_tof_m)
            self._last_tof_time = now

        if live_yaw_rad is None:
            return (self.x, self.y, self.theta)

        self._last_imu_time = now
        self._use_imu_yaw = True
        if gyro_yaw_rate is not None:
            self._imu_gyro_z = float(gyro_yaw_rate)
        self._imu_linear_accel = float(lin_acc)

        self._process_angled_imu_vector(
            lin_acc_mag=lin_acc,
            raw_accel=raw_accel,
            raw_gyro=raw_gyro,
            gyro_yaw_rate=gyro_yaw_rate,
            window_max_gz=window_max_gz,
            window_max_acc=window_max_acc,
        )

        # 1. Update heading 100% from IMU with yaw_scale
        if self._prev_live_imu_yaw is None:
            self._prev_live_imu_yaw = float(live_yaw_rad)
        else:
            dlive = _wrap_pi(float(live_yaw_rad) - self._prev_live_imu_yaw)
            self._prev_live_imu_yaw = float(live_yaw_rad)
            if abs(dlive) < (math.pi * 0.5):
                self.theta = _wrap_pi(self.theta + dlive * self.yaw_scale)

        if pitch_rad is not None:
            self.camera_pitch_rad = float(np.clip(pitch_rad, math.radians(-15.0), math.radians(45.0)))
            sin_p = math.sin(self.camera_pitch_rad)
            self._ground_depth = self.camera_height / max(sin_p, 0.08) if sin_p > 0.08 else 2.5

        tof_delta = None
        if live_tof_m is not None and 0.03 <= live_tof_m <= 3.8:
            if self._prev_tof_m > 0.0:
                tof_delta = self._prev_tof_m - float(live_tof_m)
            self._prev_tof_m = float(live_tof_m)

        turn_rate = max(self._imu_vec_gyro_mag, self._window_max_gz)
        d_fwd, _, _ = self._compute_imu_motor_forward_step(
            dt=dt, cmd_v=cmd_v, turn_rate=turn_rate, tof_delta=tof_delta
        )

        max_step = max(0.65 * dt, 0.04)
        d_fwd = float(np.clip(d_fwd, -max_step, max_step))

        if abs(d_fwd) > 0.0005:
            gx = d_fwd * math.cos(self.theta)
            gy = d_fwd * math.sin(self.theta)
            self.x += gx
            self.y += gy

        return (self.x, self.y, self.theta)

    def update(self, old_pts, new_pts, dt=1.0 / 30.0, prev_depth_map=None,
               frame_stuttered=False, cmd_v=None):
        """
        Estimate pose delta relying 100% on IMU for rotation and fusing the angled IMU
        3D acceleration vector + validated motor command with TOF400C and ORB optical flow.
        """
        dt = max(float(dt), 0.001)
        imu_active = self.has_imu
        self._last_frame_stuttered = bool(frame_stuttered or dt > 0.28)

        tof_dy = self._pending_tof_dy
        self._pending_tof_dy = None

        if imu_active:
            raw_dtheta = self._pending_imu_dtheta
            exposure_dtheta = self._pending_exposure_dtheta
            self._pending_imu_dtheta = 0.0
            self._pending_exposure_dtheta = 0.0
            dpitch = self.camera_pitch_rad - self._prev_imu_pitch
            prev_pitch = self._prev_imu_pitch
            self._prev_imu_pitch = self.camera_pitch_rad
            self.frame_theta = _wrap_pi(self.frame_theta + exposure_dtheta)
            turn_dtheta = exposure_dtheta if abs(exposure_dtheta) > abs(raw_dtheta) else raw_dtheta
        else:
            raw_dtheta = 0.0
            exposure_dtheta = 0.0
            turn_dtheta = 0.0
            dpitch = 0.0
            prev_pitch = self.camera_pitch_rad

        turn_rate = (
            max(abs(turn_dtheta / dt), self._window_max_gz, self._imu_vec_gyro_mag)
            if imu_active
            else 0.0
        )

        # ── 1. Calculate Primary Angled-IMU Vector + Validated Motor Motion Backbone ──
        d_imu, is_stationary, is_pure_rotation = self._compute_imu_motor_forward_step(
            dt=dt, cmd_v=cmd_v, turn_rate=turn_rate, tof_delta=tof_dy
        )

        # ── 2. Handle Stationary / Pure Rotation using 100% IMU ──
        if imu_active and (is_stationary or is_pure_rotation):
            self.theta = _wrap_pi(self.theta + raw_dtheta)
            self._ema_dx = 0.0
            self._ema_dy = 0.0
            self._ema_dtheta = turn_dtheta
            self._coasting_active = False
            return (self.x, self.y, self.theta), (0.0, 0.0, turn_dtheta)

        if self._last_frame_stuttered and imu_active:
            self.theta = _wrap_pi(self.theta + raw_dtheta)
            self._ema_dtheta = turn_dtheta
            max_d = 0.85 * dt
            d_fwd = float(np.clip(d_imu, -max_d, max_d))
            self._ema_dy = d_fwd if abs(d_fwd) > 0.001 else 0.5 * self._ema_dy
            self.x += self._ema_dy * math.cos(self.theta)
            self.y += self._ema_dy * math.sin(self.theta)
            self._coasting_active = True
            return (self.x, self.y, self.theta), (0.0, self._ema_dy, turn_dtheta)

        if old_pts is None or new_pts is None or len(old_pts) < 6 or len(new_pts) < 6:
            if imu_active:
                self.theta = _wrap_pi(self.theta + raw_dtheta)
                self._ema_dtheta = turn_dtheta
            max_d = 0.85 * dt
            d_fwd = float(np.clip(d_imu, -max_d, max_d))
            self._ema_dy = d_fwd if abs(d_fwd) > 0.001 else 0.5 * self._ema_dy
            self.x += self._ema_dy * math.cos(self.theta)
            self.y += self._ema_dy * math.sin(self.theta)
            self._coasting_active = True
            return (self.x, self.y, self.theta), (0.0, self._ema_dy, turn_dtheta)

        # ── 3. Visual Feature Tracking & Ground Plane Projection ──
        X0, Y0, valid0 = self._project_pixels_to_ground(old_pts, prev_pitch)
        X1, Y1, valid1 = self._project_pixels_to_ground(new_pts, self.camera_pitch_rad)
        ground_valid = valid0 & valid1

        if imu_active:
            self._coasting_active = False
            if turn_rate > 0.20:
                vis_dy = 0.0
                w_motion = 1.0
            elif np.sum(ground_valid) >= 5:
                cos_dt = math.cos(exposure_dtheta)
                sin_dt = math.sin(exposure_dtheta)
                Y0_rot = -X0[ground_valid] * sin_dt + Y0[ground_valid] * cos_dt
                dy_samples = -(Y1[ground_valid] - Y0_rot)
                med_dy = float(np.median(dy_samples))
                mad_dy = float(np.median(np.abs(dy_samples - med_dy)))
                if mad_dy > 0.08:
                    vis_dy = 0.0
                    w_motion = 1.0
                else:
                    vis_dy = med_dy * self.metric_scale
                    w_motion = self.imu_motion_weight
            else:
                center = np.array([[self.cx, self.cy]], dtype=np.float32)
                old_c = old_pts - center
                new_c = new_pts - center
                flow = new_c - old_c
                rot_flow_y = -self.fy * dpitch + old_c[:, 0] * math.sin(self.camera_pitch_rad) * exposure_dtheta
                scene_ty_px = float(np.median(flow[:, 1] - rot_flow_y))
                vis_dy = -(scene_ty_px / self.fy) * self._ground_depth * self.metric_scale
                w_motion = self.imu_motion_weight

            sign_conflict = (d_imu > 0.001 and vis_dy < -0.001) or (d_imu < -0.001 and vis_dy > 0.001)
            mag_divergence = abs(vis_dy - d_imu) > max(0.015, 2.0 * abs(d_imu))
            if sign_conflict or mag_divergence:
                raw_dy = d_imu
            else:
                raw_dy = w_motion * d_imu + (1.0 - w_motion) * vis_dy

            if abs(raw_dy) > 0.004 and abs(vis_dy) > 0.004 and (raw_dy * vis_dy > 0) and turn_rate < 0.15:
                unscaled_vis = abs(vis_dy) / max(self.metric_scale, 0.1)
                self.calibrate_scale_from_tof(abs(d_imu), unscaled_vis)
        else:
            center = np.array([[self.cx, self.cy]], dtype=np.float32)
            old_c = old_pts - center
            new_c = new_pts - center
            M, inliers = cv2.estimateAffinePartial2D(
                old_c, new_c, method=cv2.RANSAC, ransacReprojThreshold=4.5
            )
            if M is not None and inliers is not None and np.sum(inliers) >= 5:
                inlier_mask = inliers.ravel().astype(bool)
            else:
                flow_x = new_c[:, 0] - old_c[:, 0]
                med_fx = np.median(flow_x)
                inlier_mask = np.abs(flow_x - med_fx) < 25.0
                if np.sum(inlier_mask) < 5:
                    return (self.x, self.y, self.theta), (0.0, 0.0, 0.0)

            # Rotation is handled ENTIRELY by BNO055 IMU - visual rotation disabled
            raw_dtheta = 0.0
            pnp_dy, _, pnp_ok = self._solve_pnp_motion(old_pts[inlier_mask], new_pts[inlier_mask], prev_depth_map)

            if self.calibrating_180:
                self._calib_accum_raw_dtheta += abs(raw_dtheta) / max(self.yaw_scale, 1e-3)

            comb_mask = ground_valid & inlier_mask
            if np.sum(comb_mask) >= 5:
                cos_dt = math.cos(raw_dtheta)
                sin_dt = math.sin(raw_dtheta)
                Y0_rot = -X0[comb_mask] * sin_dt + Y0[comb_mask] * cos_dt
                raw_dy = float(np.median(-(Y1[comb_mask] - Y0_rot))) * self.metric_scale
            elif pnp_ok:
                raw_dy = pnp_dy
            else:
                flow_y = new_c[inlier_mask, 1] - old_c[inlier_mask, 1]
                rot_flow_y = old_c[inlier_mask, 0] * math.sin(self.camera_pitch_rad) * raw_dtheta
                scene_ty_px = float(np.median(flow_y - rot_flow_y))
                raw_dy = -(scene_ty_px / self.fy) * self._ground_depth * self.metric_scale

            if abs(raw_dtheta) > 0.035 and abs(raw_dy) < 0.025:
                raw_dy *= 0.25

            max_dtheta = 5.5 * dt
            raw_dtheta = float(np.clip(raw_dtheta, -max_dtheta, max_dtheta))
            turn_dtheta = raw_dtheta
            turn_rate = abs(turn_dtheta / dt)
            self.frame_theta = _wrap_pi(self.theta + raw_dtheta)

            if tof_dy is not None and turn_rate < 0.15:
                w_tof = self.tof_odom_blend
                raw_dy = (1.0 - w_tof) * raw_dy + w_tof * tof_dy

        # Strictly enforce non-holonomic differential drive constraint (no sideways motion)
        raw_dx = 0.0
        max_d = 0.85 * dt
        raw_dy = float(np.clip(raw_dy, -max_d, max_d))

        a = self._ema_alpha
        self._ema_dx = a * raw_dx + (1.0 - a) * self._ema_dx
        self._ema_dy = a * raw_dy + (1.0 - a) * self._ema_dy
        dtheta = raw_dtheta
        self._ema_dtheta = turn_dtheta

        mid_theta = self.theta + 0.5 * dtheta
        self.theta = _wrap_pi(self.theta + dtheta)

        gx = self._ema_dy * math.cos(mid_theta) + self._ema_dx * math.sin(mid_theta)
        gy = self._ema_dy * math.sin(mid_theta) - self._ema_dx * math.cos(mid_theta)

        self.x += gx
        self.y += gy
        if turn_rate < 0.15 and self.is_imu_showing_motion():
            self.v_fwd = 0.70 * self.v_fwd + 0.30 * (self._ema_dy / dt)

        return (self.x, self.y, self.theta), (self._ema_dx, self._ema_dy, turn_dtheta)

    def feed_imu(self, yaw_rad=None, pitch_rad=None, roll_rad=None,
                 gyro_yaw_rate=None, lin_acc_mag=None, dt=None,
                 reset=False, calib=3, live_yaw_rad=None,
                 window_max_gz=None, tof_m=None,
                 raw_accel=None, raw_gyro=None, window_max_acc=None):
        """
        Feed time-synced IMU orientation, angled 3D raw accel/gyro vectors, and TOF400C laser distance.
        """
        now = time.monotonic()
        self._last_imu_time = now

        if gyro_yaw_rate is not None:
            self._imu_gyro_z = float(gyro_yaw_rate)
        if lin_acc_mag is not None:
            self._imu_linear_accel = float(lin_acc_mag)

        self._process_angled_imu_vector(
            lin_acc_mag=lin_acc_mag,
            raw_accel=raw_accel,
            raw_gyro=raw_gyro,
            gyro_yaw_rate=gyro_yaw_rate,
            window_max_gz=window_max_gz,
            window_max_acc=window_max_acc,
        )

        # 1. Track exact camera-exposure yaw delta (100% IMU rotation)
        if yaw_rad is not None:
            yaw_f = float(yaw_rad)
            if reset or self._prev_imu_yaw is None:
                self._prev_imu_yaw = yaw_f
                self._imu_yaw_offset = _wrap_pi(yaw_f - self.theta)
                self.frame_theta = self.theta
                print(f"[VO-IMU] IMU Yaw aligned: IMU={math.degrees(yaw_f):.1f} deg "
                      f"-> Robot theta={math.degrees(self.theta):.1f} deg")
            else:
                delta_exp = _wrap_pi(yaw_f - self._prev_imu_yaw)
                self._prev_imu_yaw = yaw_f
                if abs(delta_exp) < (math.pi * 0.65):
                    if self.calibrating_180:
                        self._calib_accum_raw_dtheta += abs(delta_exp)
                    self._pending_exposure_dtheta += delta_exp * self.yaw_scale
            self._imu_yaw = yaw_f
            self._use_imu_yaw = True

        # 2. Track live IMU yaw delta
        target_live = float(live_yaw_rad) if live_yaw_rad is not None else (float(yaw_rad) if yaw_rad is not None else None)
        if target_live is not None:
            if reset or self._prev_live_imu_yaw is None:
                self._prev_live_imu_yaw = target_live
            else:
                delta_live = _wrap_pi(target_live - self._prev_live_imu_yaw)
                self._prev_live_imu_yaw = target_live
                if abs(delta_live) < (math.pi * 0.65):
                    self._pending_imu_dtheta += delta_live * self.yaw_scale
            self._use_imu_yaw = True
        elif gyro_yaw_rate is not None and dt is not None and dt > 0:
            step_gz = float(gyro_yaw_rate) * float(dt) * self.yaw_scale
            self._pending_imu_dtheta += step_gz
            self._pending_exposure_dtheta += step_gz
            self._use_imu_yaw = True

        # 3. Track TOF400C laser range & frame-to-frame forward range delta
        if tof_m is not None and 0.03 <= float(tof_m) <= 4.0:
            tof_val = float(tof_m)
            self.latest_tof_m = tof_val
            self._last_tof_time = now
            if self._prev_tof_m > 0.03:
                d_tof = -(tof_val - self._prev_tof_m)
                max_step = max(0.15, 1.5 * float(dt or 0.033))
                if abs(d_tof) <= max_step:
                    self._pending_tof_dy = d_tof
            self._prev_tof_m = tof_val
        elif tof_m is not None and float(tof_m) < 0.0:
            self._prev_tof_m = -1.0
            if (now - self._last_tof_time) > 0.40:
                self.latest_tof_m = -1.0

        self._imu_stationary = not self.is_imu_showing_motion(now)

        if pitch_rad is not None:
            self.camera_pitch_rad = float(np.clip(pitch_rad, math.radians(-15.0), math.radians(45.0)))
            sin_p = math.sin(self.camera_pitch_rad)
            self._ground_depth = self.camera_height / max(sin_p, 0.08) if sin_p > 0.08 else 2.5

    def get_frame_pose(self):
        return (self.x, self.y, self.frame_theta if self.has_imu else self.theta)

    def is_mapping_safe(self, frame_stuttered=False, max_turn_rate_rads=0.45):
        if frame_stuttered or self._last_frame_stuttered:
            return False
        if self.has_imu:
            if max(abs(self._imu_gyro_z), self._window_max_gz, self._imu_vec_gyro_mag) > max_turn_rate_rads:
                return False
        return True

    @property
    def has_imu(self):
        return self._use_imu_yaw and ((time.monotonic() - self._last_imu_time) < 0.60)

    @property
    def has_tof(self):
        return (self.latest_tof_m > 0.03) and ((time.monotonic() - self._last_tof_time) < 0.60)

    def calibrate_scale_from_tof(self, tof_distance_m, visual_distance_est):
        if visual_distance_est > 0.003 and tof_distance_m > 0.003:
            ratio = float(np.clip(tof_distance_m / visual_distance_est, 0.4, 2.5))
            self.metric_scale = float(np.clip(0.95 * self.metric_scale + 0.05 * ratio, 0.4, 2.5))
