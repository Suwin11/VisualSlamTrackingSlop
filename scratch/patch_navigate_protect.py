with open('navigate_rrt.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Initialize map_locked and protect_explored_walls in main()
old_map_init = '''    if args.load_map is not None:
        if global_grid.load_map(args.load_map, keep_current_pose=False):
            vo.x, vo.y, vo.theta = global_grid.current_pose
            vo.frame_theta = vo.theta
            print(f"[MAP] Anchored odometry origin to loaded map at ({vo.x:.2f}m, {vo.y:.2f}m, {math.degrees(vo.theta):.1f}°)")'''

new_map_init = '''    map_locked = bool(gmap_cfg.get("lock_map_by_default", False))
    if args.load_map is not None:
        if global_grid.load_map(args.load_map, keep_current_pose=False):
            vo.x, vo.y, vo.theta = global_grid.current_pose
            vo.frame_theta = vo.theta
            # Automatically lock map when loading a pre-saved map so it isn't overwritten!
            map_locked = True
            print(f"[MAP] Anchored odometry origin to loaded map at ({vo.x:.2f}m, {vo.y:.2f}m, {math.degrees(vo.theta):.1f}°)")
            print("[MAP] Map is LOCKED (Read-Only). Press 'K' to toggle live mapping.")

    global_grid.map_locked = map_locked
    global_grid.protect_explored_walls = bool(gmap_cfg.get("protect_explored_walls", True))'''

assert old_map_init in content, 'old_map_init not found in navigate_rrt.py'
content = content.replace(old_map_init, new_map_init, 1)

# 2. Add nonlocal map_locked to handle_key
old_hk_head = '''    def handle_key(key: int) -> bool:
        """
        Unified keyboard handler called BOTH during normal frames AND during camera stutters!
        Returns False if the user pressed Q/ESC to quit.
        """
        nonlocal show_3d_panel, current_goal, current_raw_path, current_smooth_path, last_plan_time'''

new_hk_head = '''    def handle_key(key: int) -> bool:
        """
        Unified keyboard handler called BOTH during normal frames AND during camera stutters!
        Returns False if the user pressed Q/ESC to quit.
        """
        nonlocal show_3d_panel, current_goal, current_raw_path, current_smooth_path, last_plan_time
        nonlocal map_locked, map_status_banner, map_status_time'''

assert old_hk_head in content, 'old_hk_head not found in navigate_rrt.py'
content = content.replace(old_hk_head, new_hk_head, 1)

# 3. Add Keybind 'K' for Map Lock / Freeze Mode
# And update 'L' and 'U' for loading maps
old_u_key = '''        elif key in (ord('u'), ord('U')):
            map_path = None
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                save_dir = os.path.abspath("saved_maps")
                map_path = filedialog.askopenfilename(
                    initialdir=save_dir if os.path.exists(save_dir) else ".",
                    title="Open Occupancy Map for Navigation - FIBO",
                    filetypes=[("Occupancy Maps (*.npz)", "*.npz"), ("All Files (*.*)", "*.*")]
                )
                root.destroy()
            except Exception:
                map_path = None

            target = map_path if (map_path and os.path.exists(map_path)) else 'latest'
            if global_grid.load_map(target):
                vo.x, vo.y, vo.theta = global_grid.current_pose
                vo.frame_theta = vo.theta
                if vo.has_imu and getattr(vo, "_imu_last_yaw", None) is not None:
                    vo._imu_yaw_offset = (vo.theta - vo._imu_last_yaw + math.pi) % (2.0 * math.pi) - math.pi
                    if getattr(vo, "_imu_last_live_yaw", None) is not None:
                        vo._imu_last_live_yaw = vo._imu_last_yaw
                current_smooth_path = None
                last_plan_time = 0.0
                fname = os.path.basename(target) if target != 'latest' else 'latest_map.npz'
                map_status_banner = f"LOADED: {fname} ({global_grid.width_m:.1f}x{global_grid.height_m:.1f}m)"
                map_status_time = time.time()
                print(f"[MAP] Loaded map '{target}' and aligned robot pose to ({vo.x:.2f}m, {vo.y:.2f}m, {math.degrees(vo.theta):.0f}°)")'''

new_u_key = '''        elif key in (ord('k'), ord('K')):
            map_locked = not map_locked
            global_grid.map_locked = map_locked
            lock_label = "LOCKED (Explored Map Protected - Read-Only)" if map_locked else "UNLOCKED (Live Exploring)"
            map_status_banner = f"MAP: {lock_label}"
            map_status_time = time.time()
            print(f"[MAP] Map Protection -> {lock_label}")
        elif key in (ord('u'), ord('U'), ord('l'), ord('L')):
            map_path = None
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                save_dir = os.path.abspath("saved_maps")
                map_path = filedialog.askopenfilename(
                    initialdir=save_dir if os.path.exists(save_dir) else ".",
                    title="Open Occupancy Map for Navigation - FIBO",
                    filetypes=[
                        ("All Supported Maps (*.npz; *.json)", "*.npz;*.json"),
                        ("Occupancy Maps (*.npz)", "*.npz"),
                        ("JSON Floorplans (*.json)", "*.json"),
                        ("All Files (*.*)", "*.*")
                    ]
                )
                root.destroy()
            except Exception as e:
                print(f"[MAP-LOAD] File dialog error: {e}")
                map_path = None

            target = map_path if (map_path and os.path.exists(map_path)) else 'latest'
            if global_grid.load_map(target, keep_current_pose=False):
                vo.x, vo.y, vo.theta = global_grid.current_pose
                vo.frame_theta = vo.theta
                if vo.has_imu and getattr(vo, "_imu_last_yaw", None) is not None:
                    vo._imu_yaw_offset = (vo.theta - vo._imu_last_yaw + math.pi) % (2.0 * math.pi) - math.pi
                    if getattr(vo, "_imu_last_live_yaw", None) is not None:
                        vo._imu_last_live_yaw = vo._imu_last_yaw
                current_smooth_path = None
                last_plan_time = 0.0
                # Automatically lock map when loaded so live camera does not overwrite explored geometry!
                map_locked = True
                global_grid.map_locked = True
                fname = os.path.basename(target) if target != 'latest' else 'latest_map.npz'
                map_status_banner = f"LOADED: {fname} [MAP LOCKED - Press 'K' to Unlock]"
                map_status_time = time.time()
                print(f"[MAP] Loaded map '{target}' and aligned robot pose to ({vo.x:.2f}m, {vo.y:.2f}m, {math.degrees(vo.theta):.0f}°)")
                print("[MAP] Explored map is now LOCKED (Read-Only). Press 'K' to unlock live mapping.")'''

assert old_u_key in content, 'old_u_key not found in navigate_rrt.py'
content = content.replace(old_u_key, new_u_key, 1)

# 4. In active navigation loop (line 943): check not map_locked
old_map_update = '''            # 5. Global Map Integration
            if mapping_safe:
                global_grid.update_from_local_bev(
                    local_bev, frame_pose, local_range_y=local_range_y
                )
                if aligned_tof_m is not None:
                    global_grid.update_from_tof(
                        frame_pose, aligned_tof_m, max_range_m=max_tof_range, beam_deg=tof_beam_deg
                    )'''

new_map_update = '''            # 5. Global Map Integration
            if mapping_safe and (not map_locked):
                global_grid.update_from_local_bev(
                    local_bev, frame_pose, local_range_y=local_range_y
                )
                if aligned_tof_m is not None:
                    global_grid.update_from_tof(
                        frame_pose, aligned_tof_m, max_range_m=max_tof_range, beam_deg=tof_beam_deg
                    )'''

assert old_map_update in content, 'old_map_update not found in navigate_rrt.py'
content = content.replace(old_map_update, new_map_update, 1)

# 5. In idle/stutter loop: check not map_locked
old_idle_tof = '''                    live_t = live_imu.get("tof_m", -1.0)
                    if 0.04 <= live_t <= 4.0:
                        global_grid.update_from_tof(robot_pose, live_t, max_range_m=max_tof_range, beam_deg=tof_beam_deg)'''

new_idle_tof = '''                    live_t = live_imu.get("tof_m", -1.0)
                    if (not map_locked) and 0.04 <= live_t <= 4.0:
                        global_grid.update_from_tof(robot_pose, live_t, max_range_m=max_tof_range, beam_deg=tof_beam_deg)'''

assert old_idle_tof in content, 'old_idle_tof not found in navigate_rrt.py'
content = content.replace(old_idle_tof, new_idle_tof, 1)

# 6. Update HUD label to show map_locked state and keybinds
old_hud_label = '''            if map_status_banner and (time.time() - map_status_time) < 4.0:
                _put_panel_label(global_vis, f"*** {map_status_banner} *** | Drag/Mid:Pan | Wheel:Zoom", bg=(15, 60, 25), fg=(0, 255, 130))
            else:
                _put_panel_label(global_vis, f"Global 2D Map (Drag/Mid:Pan | Wheel:Zoom | L:Goal | R:Pose | F:Follow | 'O':Save | 'U':Load) | Gear:{pwm_gear}PWM")'''

new_hud_label = '''            lock_badge = "[MAP: LOCKED (Protected)]" if map_locked else "[MAP: LIVE]"
            badge_bg = (15, 30, 60) if map_locked else (20, 20, 20)
            badge_fg = (0, 220, 255) if map_locked else (0, 230, 255)
            if map_status_banner and (time.time() - map_status_time) < 4.0:
                _put_panel_label(global_vis, f"*** {map_status_banner} *** | Drag:Pan | Wheel:Zoom", bg=(15, 60, 25), fg=(0, 255, 130))
            else:
                _put_panel_label(global_vis, f"Global 2D Map {lock_badge} (Drag:Pan | Wheel:Zoom | R-Clk:Reloc | 'K':LockMap | 'L':Load | 'O':Save) | Gear:{pwm_gear}PWM", bg=badge_bg, fg=badge_fg)'''

assert old_hud_label in content, 'old_hud_label not found in navigate_rrt.py'
content = content.replace(old_hud_label, new_hud_label, 1)

with open('navigate_rrt.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('navigate_rrt.py successfully patched with map lock and file loading!')
