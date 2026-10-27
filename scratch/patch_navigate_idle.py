with open('navigate_rrt.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_idle_vis = '''                global_vis = global_grid.render_map(
                    out_w=w, out_h=h, show_trajectory=True,
                    path=current_smooth_path, goal=current_goal,
                    robot_pose=robot_pose, lookahead_pt=cmd["lookahead"]
                )'''

new_idle_vis = '''                global_vis = global_grid.render_map(
                    out_w=w, out_h=h, show_trajectory=True,
                    path=current_smooth_path, goal=current_goal,
                    robot_pose=robot_pose, lookahead_pt=cmd["lookahead"]
                )
                pwm_gear = getattr(controller, "manual_forward_pwm", getattr(controller, "manual_forward_rpm", 140))
                lock_badge = "[MAP: LOCKED (Protected)]" if map_locked else "[MAP: LIVE]"
                badge_bg = (15, 30, 60) if map_locked else (20, 20, 20)
                badge_fg = (0, 220, 255) if map_locked else (0, 230, 255)
                if map_status_banner and (time.time() - map_status_time) < 4.0:
                    _put_panel_label(global_vis, f"*** {map_status_banner} *** | Drag:Pan | Wheel:Zoom", bg=(15, 60, 25), fg=(0, 255, 130))
                else:
                    _put_panel_label(global_vis, f"Global 2D Map {lock_badge} (Drag:Pan | Wheel:Zoom | R-Clk:Reloc | 'K':LockMap | 'L':Load | 'O':Save) | Gear:{pwm_gear}PWM", bg=badge_bg, fg=badge_fg)'''

assert old_idle_vis in content, 'old_idle_vis not found in navigate_rrt.py'
content = content.replace(old_idle_vis, new_idle_vis, 1)

with open('navigate_rrt.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Idle label in navigate_rrt.py updated successfully.')
