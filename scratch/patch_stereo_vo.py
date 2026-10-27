with open('stereo_visual_odometry.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update _compute_imu_motor_forward_step to prevent uncommanded drift
old_step = '''        no_linear_cmd = abs(effective_cmd_v) < 0.005
        has_tof_motion = (tof_delta is not None) and (abs(tof_delta) >= 0.005)
        is_stationary = self.has_imu and (not imu_active_motion) and no_linear_cmd and (not has_tof_motion)
        is_pure_rotation = self.has_imu and (turn_rate > 0.15) and no_linear_cmd

        if is_stationary or is_pure_rotation:
            self.v_fwd = 0.0
            self.v_lat = 0.0
            return 0.0, is_stationary, is_pure_rotation'''

new_step = '''        no_linear_cmd = abs(effective_cmd_v) < 0.005
        has_tof_motion = (tof_delta is not None) and (abs(tof_delta) >= 0.008) and (turn_rate < 0.10)
        # In a wheeled differential-drive robot, forward motion only occurs when commanded
        # or when physically confirmed by ToF forward distance reduction.
        # Gyro turning without linear command is 100% pure in-place rotation!
        is_pure_rotation = self.has_imu and (turn_rate > 0.05) and no_linear_cmd
        is_stationary = no_linear_cmd and (not has_tof_motion) and (not is_pure_rotation)

        if is_stationary or is_pure_rotation:
            self.v_fwd = 0.0
            self.v_lat = 0.0
            return 0.0, is_stationary, is_pure_rotation'''

assert old_step in content, 'old_step not found in stereo_visual_odometry.py'
content = content.replace(old_step, new_step, 1)

# 2. In _compute_camera_depth_forward_step: disable corridor depth difference during turns
old_corridor = '''        # 2. Central corridor dense metric depth shift (reinforces when few ORB points)
        roi_p0 = prev_depth[int(h * 0.30) : int(h * 0.70), int(w * 0.30) : int(w * 0.70)]
        roi_p1 = curr_depth[int(h * 0.30) : int(h * 0.70), int(w * 0.30) : int(w * 0.70)]
        v_p0 = roi_p0[~np.isnan(roi_p0)]
        v_p1 = roi_p1[~np.isnan(roi_p1)]
        if len(v_p0) > 80 and len(v_p1) > 80:
            med0 = float(np.median(v_p0))
            med1 = float(np.median(v_p1))
            if 0.25 <= med0 <= 3.2 and 0.25 <= med1 <= 3.2:
                d_corridor = (med0 - med1) * cos_p
                if abs(d_corridor) <= max_phys:
                    shifts.append(d_corridor)'''

new_corridor = '''        # 2. Central corridor dense metric depth shift (only when driving straight, turn_rate < 0.08)
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
                        shifts.append(d_corridor)'''

assert old_corridor in content, 'old_corridor not found in stereo_visual_odometry.py'
content = content.replace(old_corridor, new_corridor, 1)

# 3. In update(): when no linear command is active, do not drift forward
old_uncmd_fuse = '''            else:
                # Uncommanded movement (e.g. pushed or smooth cruising): fuse IMU + Camera Depth
                d_fwd = 0.55 * d_imu + 0.45 * d_depth
                self.last_motion_source = "FUSED(IMU+Depth,Uncmd)"'''

new_uncmd_fuse = '''            else:
                # No forward command active: enforce zero drift unless ToF confirms physical forward movement
                if (tof_delta is not None) and (abs(tof_delta) >= 0.008) and (turn_rate < 0.08):
                    d_fwd = 0.50 * d_imu + 0.50 * tof_delta
                    self.last_motion_source = "PUSHED_TOF"
                else:
                    d_fwd = 0.0
                    self.last_motion_source = "STATIONARY"'''

assert old_uncmd_fuse in content, 'old_uncmd_fuse not found in stereo_visual_odometry.py'
content = content.replace(old_uncmd_fuse, new_uncmd_fuse, 1)

# 4. In is_mapping_safe: reduce max_turn_rate_rads to 0.20
old_safe = '''    def is_mapping_safe(self, frame_stuttered=False, max_turn_rate_rads=0.45):
        if frame_stuttered:
            return False
        if self.has_imu and self._imu_window_max_gz > max_turn_rate_rads:
            return False
        return True'''

new_safe = '''    def is_mapping_safe(self, frame_stuttered=False, max_turn_rate_rads=0.20):
        if frame_stuttered:
            return False
        if self.has_imu and self._imu_window_max_gz > max_turn_rate_rads:
            return False
        if getattr(self, "calibrating_180", False):
            return False
        return True'''

assert old_safe in content, 'old_safe not found in stereo_visual_odometry.py'
content = content.replace(old_safe, new_safe, 1)

with open('stereo_visual_odometry.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('stereo_visual_odometry.py patched successfully.')
