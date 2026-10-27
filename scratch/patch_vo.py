with open('visual_odometry.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update _compute_imu_motor_forward_step
old_vo_step = '''        no_linear_cmd = abs(effective_cmd_v) < 0.005
        has_tof_motion = (tof_delta is not None) and (abs(tof_delta) >= 0.005)
        is_stationary = imu_active and (not imu_active_motion) and no_linear_cmd and (not has_tof_motion)
        is_pure_rotation = imu_active and (turn_rate > 0.15) and no_linear_cmd

        if is_stationary or is_pure_rotation:
            self.v_fwd = 0.0
            return 0.0, is_stationary, is_pure_rotation'''

new_vo_step = '''        no_linear_cmd = abs(effective_cmd_v) < 0.005
        has_tof_motion = (tof_delta is not None) and (abs(tof_delta) >= 0.008) and (turn_rate < 0.10)
        is_pure_rotation = imu_active and (turn_rate > 0.05) and no_linear_cmd
        is_stationary = no_linear_cmd and (not has_tof_motion) and (not is_pure_rotation)

        if is_stationary or is_pure_rotation:
            self.v_fwd = 0.0
            return 0.0, is_stationary, is_pure_rotation'''

assert old_vo_step in content, 'old_vo_step not found in visual_odometry.py'
content = content.replace(old_vo_step, new_vo_step, 1)

with open('visual_odometry.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('visual_odometry.py patched successfully.')
