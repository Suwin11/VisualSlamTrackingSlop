with open('navigate_rrt.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update line 756: pass max_tof_range and beam_deg
old_idle_tof = '''                    live_t = live_imu.get("tof_m", -1.0)
                    if 0.04 <= live_t <= 4.0:
                        global_grid.update_from_tof(robot_pose, live_t, max_range_m=4.0)'''

new_idle_tof = '''                    live_t = live_imu.get("tof_m", -1.0)
                    if 0.04 <= live_t <= 4.0:
                        global_grid.update_from_tof(robot_pose, live_t, max_range_m=max_tof_range, beam_deg=tof_beam_deg)'''
assert old_idle_tof in content, 'old_idle_tof not found'
content = content.replace(old_idle_tof, new_idle_tof, 1)

# 2. Update line 779: pass max_range_m=max_tof_range
old_panel2_idle = '''                else:
                    panel2_vis = occupancy.render(last_local_bev, out_w=w, out_h=h, tof_distance_m=active_tof)'''

new_panel2_idle = '''                else:
                    panel2_vis = occupancy.render(last_local_bev, out_w=w, out_h=h, tof_distance_m=active_tof, max_range_m=max_tof_range)'''
assert old_panel2_idle in content, 'old_panel2_idle not found'
content = content.replace(old_panel2_idle, new_panel2_idle, 1)

# 3. Update line 921 default from 2.5 to 4.0
old_max_fwd = '''            max_fwd_range = float(cfg["mapping_3d"].get("max_forward_range_m", 2.5))'''
new_max_fwd = '''            max_fwd_range = float(cfg["mapping_3d"].get("max_forward_range_m", 4.0))'''
assert old_max_fwd in content, 'old_max_fwd not found'
content = content.replace(old_max_fwd, new_max_fwd, 1)

with open('navigate_rrt.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('navigate_rrt.py patched successfully.')
