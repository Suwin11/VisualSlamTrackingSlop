"""
scratch/patch_auto_escape.py
Implement autonomous TOF obstacle escape / unstuck feature.
"""

import json
import os
import re

# ─────────────────────────────────────────────────────────────────────────────
# 1. Update mapping_config.json
# ─────────────────────────────────────────────────────────────────────────────
cfg_path = "mapping_config.json"
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

tof_sec = cfg.setdefault("tof_sensor", {})
tof_sec["auto_escape_enabled"] = True
tof_sec["escape_trigger_delay_sec"] = 0.35
tof_sec["escape_pivot_duration_sec"] = 0.65
tof_sec["escape_clear_margin_m"] = 0.12
tof_sec["escape_direction"] = "auto"

with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print("[PASS] Updated mapping_config.json with auto_escape parameters.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Update motor_control.py
# ─────────────────────────────────────────────────────────────────────────────
mc_path = "motor_control.py"
with open(mc_path, "r", encoding="utf-8") as f:
    mc_text = f.read()

# Add _escape_active_until to __init__
old_init = "        self._emergency_stop = False"
new_init = "        self._emergency_stop = False\n        self._escape_active_until = 0.0"
assert old_init in mc_text, "Init location not found in motor_control.py"
mc_text = mc_text.replace(old_init, new_init, 1)

# Add trigger_escape_pivot and cancel_escape_pivot
old_pwm_methods = "    def arcade_drive(self, throttle_pwm: float, turn_pwm: float, duration_sec: float | None = None):"
new_pwm_methods = """    def trigger_escape_pivot(self, direction="right", duration_sec: float = 0.65, pwm: int | None = None):
        \"\"\"
        Command an autonomous in-place pivot to swing sensors away from obstacles.
        direction: 'right' (or +1) -> right turn (+turn_pwm, -turn_pwm)
                   'left'  (or -1) -> left turn (-turn_pwm, +turn_pwm)
        \"\"\"
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
        \"\"\"Immediately terminate an active escape pivot.\"\"\"
        with self._lock:
            self._escape_active_until = 0.0
            self._manual_hold_until = 0.0
            self._manual_hold_pwm = (0, 0)

    def arcade_drive(self, throttle_pwm: float, turn_pwm: float, duration_sec: float | None = None):"""
assert old_pwm_methods in mc_text, "arcade_drive not found in motor_control.py"
mc_text = mc_text.replace(old_pwm_methods, new_pwm_methods, 1)

# Update _loop to handle EMERGENCY_PIVOT
old_loop_check = """                if self._emergency_stop and hw_keys is None and t0 >= self._manual_hold_until:
                    active_cmd = self._make_cmd_dict(0, 0, mode="EMERGENCY_STOP")
                elif t0 < self._manual_hold_until:
                    pl, pr = self._manual_hold_pwm
                    active_cmd = self._make_cmd_dict(pl, pr, mode="MANUAL_HOLD")"""

new_loop_check = """                is_escape = t0 < getattr(self, "_escape_active_until", 0.0)
                if self._emergency_stop and hw_keys is None and t0 >= self._manual_hold_until and not is_escape:
                    active_cmd = self._make_cmd_dict(0, 0, mode="EMERGENCY_STOP")
                elif t0 < self._manual_hold_until:
                    pl, pr = self._manual_hold_pwm
                    mode_name = "EMERGENCY_PIVOT" if is_escape else "MANUAL_HOLD"
                    active_cmd = self._make_cmd_dict(pl, pr, mode=mode_name)"""
assert old_loop_check in mc_text, "loop check not found in motor_control.py"
mc_text = mc_text.replace(old_loop_check, new_loop_check, 1)

with open(mc_path, "w", encoding="utf-8") as f:
    f.write(mc_text)
print("[PASS] Patched motor_control.py with escape pivot capabilities.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Update navigate_rrt.py
# ─────────────────────────────────────────────────────────────────────────────
nav_path = "navigate_rrt.py"
with open(nav_path, "r", encoding="utf-8") as f:
    nav_text = f.read()

# Add tracking variables before main loop
old_state_init = """    current_raw_path = None
    current_smooth_path = None
    current_tree_edges = []"""

new_state_init = """    current_raw_path = None
    current_smooth_path = None
    current_tree_edges = []

    # Autonomous TOF escape / unstuck state tracking
    tof_stuck_start = None
    escape_active = False
    escape_until = 0.0
    escape_dir = "right"
    escape_count = 0"""
assert old_state_init in nav_text, "old_state_init not found in navigate_rrt.py"
nav_text = nav_text.replace(old_state_init, new_state_init, 1)

# In stutter loop, respect escape_active
old_stutter_cmd = """                tof_emerg = (active_tof is not None) and (active_tof < tof_stop_dist)
                cmd = controller.compute_command(
                    robot_pose, current_smooth_path, goal=current_goal, emergency_stop=tof_emerg
                )"""

new_stutter_cmd = """                tof_emerg = (active_tof is not None) and (active_tof < tof_stop_dist)
                cmd = controller.compute_command(
                    robot_pose, current_smooth_path, goal=current_goal,
                    emergency_stop=(tof_emerg and not escape_active)
                )"""
assert old_stutter_cmd in nav_text, "old_stutter_cmd not found in navigate_rrt.py"
nav_text = nav_text.replace(old_stutter_cmd, new_stutter_cmd, 1)

# In main loop: replace planning & emergency section with auto-escape state machine
old_planning = """            # 6. Event-Driven RRT* Path Planning
            tof_emergency = (live_tof_m is not None) and (live_tof_m < tof_stop_dist)
            if (now - last_plan_time > plan_interval) or tof_emergency:
                last_plan_time = now
                if tof_emergency or not planner.is_path_valid(robot_pose, current_smooth_path, current_goal):
                    rx, ry, _ = robot_pose
                    raw_p, smooth_p, tree_e = planner.plan(
                        (rx, ry), current_goal, seed_waypoints=current_raw_path
                    )
                    current_tree_edges = tree_e
                    if smooth_p is not None:
                        current_raw_path = raw_p
                        current_smooth_path = smooth_p
                    else:
                        current_raw_path = None
                        current_smooth_path = None

            # 7. Closed-Loop Pure Pursuit Control Command + ToF Emergency Brake
            cmd = controller.compute_command(
                robot_pose, current_smooth_path, goal=current_goal, emergency_stop=tof_emergency
            )"""

new_planning = """            # 6. Event-Driven Path Planning & Autonomous TOF Escape Maneuver
            tof_cfg = cfg.get("tof_sensor", {})
            auto_escape_on = bool(tof_cfg.get("auto_escape_enabled", True))
            escape_delay = float(tof_cfg.get("escape_trigger_delay_sec", 0.35))
            escape_duration = float(tof_cfg.get("escape_pivot_duration_sec", 0.65))
            escape_margin = float(tof_cfg.get("escape_clear_margin_m", 0.12))
            escape_pref = str(tof_cfg.get("escape_direction", "auto")).lower().strip()

            tof_emergency = (live_tof_m is not None) and (live_tof_m < tof_stop_dist)

            # Autonomous Escape State Machine: pivot to swing TOF laser away if stuck facing wall
            active_ctrl_mode = cmd.get("mode", controller.control_mode) if 'cmd' in locals() else controller.control_mode
            if auto_escape_on and active_ctrl_mode in ("AUTO", "EMERGENCY_PIVOT"):
                if tof_emergency:
                    if tof_stuck_start is None:
                        tof_stuck_start = now
                    elif (now - tof_stuck_start >= escape_delay) and not escape_active:
                        # Determine pivot direction (check free clearance in 2D occupancy grid)
                        if escape_pref == "right":
                            chosen_dir = "right"
                        elif escape_pref == "left":
                            chosen_dir = "left"
                        else:
                            rx, ry, rtheta = robot_pose
                            r_th = rtheta - math.radians(50.0)
                            l_th = rtheta + math.radians(50.0)
                            pt_r = (rx + 0.65 * math.cos(r_th), ry + 0.65 * math.sin(r_th))
                            pt_l = (rx + 0.65 * math.cos(l_th), ry + 0.65 * math.sin(l_th))
                            r_free = global_grid.is_collision_free(pt_r[0], pt_r[1])
                            l_free = global_grid.is_collision_free(pt_l[0], pt_l[1])
                            if r_free and not l_free:
                                chosen_dir = "right"
                            elif l_free and not r_free:
                                chosen_dir = "left"
                            else:
                                chosen_dir = "right" if (escape_count % 2 == 0) else "left"

                        escape_active = True
                        escape_dir = chosen_dir
                        escape_until = now + escape_duration
                        escape_count += 1
                        controller.trigger_escape_pivot(direction=escape_dir, duration_sec=escape_duration)
                        map_status_banner = f"TOF UNSTUCK: Pivoting {escape_dir.upper()} ~45deg..."
                        map_status_time = time.time()
                        print(f"[UNSTUCK] Obstacle at {live_tof_m:.2f}m (<{tof_stop_dist:.2f}m). Auto-pivoting {escape_dir.upper()} to clear...")
                else:
                    if not escape_active:
                        tof_stuck_start = None

                # Check if escape maneuver completed or cleared
                if escape_active:
                    laser_cleared = (live_tof_m is not None) and (live_tof_m >= tof_stop_dist + escape_margin)
                    if laser_cleared or (now >= escape_until):
                        escape_active = False
                        tof_stuck_start = None
                        controller.cancel_escape_pivot()
                        current_smooth_path = None  # Invalidate path to replan immediately from new heading
                        last_plan_time = 0.0
                        tof_emergency = False
                        c_txt = f"Laser Cleared ({live_tof_m:.2f}m)" if laser_cleared else "Pivot Finished"
                        map_status_banner = f"TOF UNSTUCK: {c_txt}! Replanning..."
                        map_status_time = time.time()
                        print(f"[UNSTUCK] Escape {c_txt}! Replanning from new robot heading.")

            if (now - last_plan_time > plan_interval) or (tof_emergency and not escape_active):
                last_plan_time = now
                if (tof_emergency and not escape_active) or not planner.is_path_valid(robot_pose, current_smooth_path, current_goal):
                    rx, ry, _ = robot_pose
                    raw_p, smooth_p, tree_e = planner.plan(
                        (rx, ry), current_goal, seed_waypoints=current_raw_path
                    )
                    current_tree_edges = tree_e
                    if smooth_p is not None:
                        current_raw_path = raw_p
                        current_smooth_path = smooth_p
                    else:
                        current_raw_path = None
                        current_smooth_path = None

            # 7. Closed-Loop Pure Pursuit Control Command + ToF Emergency Brake
            cmd = controller.compute_command(
                robot_pose, current_smooth_path, goal=current_goal,
                emergency_stop=(tof_emergency and not escape_active)
            )"""
assert old_planning in nav_text, "old_planning not found in navigate_rrt.py"
nav_text = nav_text.replace(old_planning, new_planning, 1)

# In Panel 4 rendering, show [AUTO-UNSTUCK] status
old_panel4_hud = """            if active_mode in ("MANUAL_HOLD", "MANUAL_CRUISE"):
                path_status = f"[{active_mode}] PWM: L={cmd.get('pwm_left', 0):+d}, R={cmd.get('pwm_right', 0):+d} (HW:{cmd.get('rpm_left', 0):+d},{cmd.get('rpm_right', 0):+d}) (X/Space=Stop)"
            elif tof_emergency:
                path_status = f"TOF EMERGENCY BRAKE ({live_tof_m:.2f}m < {tof_stop_dist:.2f}m)! (WASD to override)" """

new_panel4_hud = """            if escape_active:
                path_status = f"[AUTO-UNSTUCK] Pivoting {escape_dir.upper()} ~45deg to clear obstacle ({live_tof_m:.2f}m)..."
            elif active_mode in ("MANUAL_HOLD", "MANUAL_CRUISE"):
                path_status = f"[{active_mode}] PWM: L={cmd.get('pwm_left', 0):+d}, R={cmd.get('pwm_right', 0):+d} (HW:{cmd.get('rpm_left', 0):+d},{cmd.get('rpm_right', 0):+d}) (X/Space=Stop)"
            elif tof_emergency:
                path_status = f"TOF EMERGENCY BRAKE ({live_tof_m:.2f}m < {tof_stop_dist:.2f}m)! (Auto-escape active)" """

if old_panel4_hud.strip() in nav_text:
    nav_text = nav_text.replace(old_panel4_hud.strip(), new_panel4_hud.strip(), 1)
else:
    # Try normalized
    n_nav = nav_text.replace("\r\n", "\n")
    n_old = old_panel4_hud.strip().replace("\r\n", "\n")
    n_new = new_panel4_hud.strip().replace("\r\n", "\n")
    if n_old in n_nav:
        nav_text = n_nav.replace(n_old, n_new, 1)
    else:
        print("[WARN] Could not replace panel4 HUD string, checking manually...")

with open(nav_path, "w", encoding="utf-8") as f:
    f.write(nav_text)
print("[PASS] Patched navigate_rrt.py with auto-escape state machine.")
