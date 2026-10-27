"""
scratch/patch_navigate_astar.py
Patch navigate_rrt.py to add A* planner support.
"""

import os

path = "navigate_rrt.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Add AStarPlanner import
old_import = "from rrt_star import RRTStarPlanner"
new_import = "from rrt_star import RRTStarPlanner\nfrom astar import AStarPlanner"
assert old_import in text, "Import not found"
text = text.replace(old_import, new_import, 1)

# 2. Add --planner CLI arg
old_arg = 'ap.add_argument("--load-map", default=None, help="Path to pre-saved map .npz (or \'latest\' for newest map)")'
new_arg = (
    'ap.add_argument("--load-map", default=None, help="Path to pre-saved map .npz (or \'latest\' for newest map)")\n'
    '    ap.add_argument("--planner", default=None, choices=["astar", "rrt", "rrt_star"],\n'
    '                    help="Path planner: \'astar\' (deterministic optimal) or \'rrt\' (sampling tree)")'
)
assert old_arg in text, "CLI arg not found"
text = text.replace(old_arg, new_arg, 1)

# 3. Read planner_type
old_yp = 'yp_cfg = cfg["yolo_and_planner"]'
new_yp = (
    'yp_cfg = cfg["yolo_and_planner"]\n'
    '    planner_type = str(args.planner or yp_cfg.get("planner_type", "astar")).lower().strip()\n'
    '    if planner_type in ("rrt", "rrt_star", "rrt*"):\n'
    '        planner_type = "rrt"\n'
    '    else:\n'
    '        planner_type = "astar"'
)
assert old_yp in text, "yp_cfg not found"
text = text.replace(old_yp, new_yp, 1)

# 4. Initialize planners
old_plan_init = 'planner = RRTStarPlanner(global_grid, step_size=0.25, search_radius=0.55, max_iter=300)'
new_plan_init = (
    'astar_planner = AStarPlanner(\n'
    '        global_grid,\n'
    '        clearance_weight=float(yp_cfg.get("astar_clearance_weight", 0.20)),\n'
    '        corner_cut_prevention=True\n'
    '    )\n'
    '    rrt_planner = RRTStarPlanner(global_grid, step_size=0.25, search_radius=0.55, max_iter=300)\n'
    '    planner = astar_planner if planner_type == "astar" else rrt_planner'
)
assert old_plan_init in text, "planner init not found"
text = text.replace(old_plan_init, new_plan_init, 1)

# 5. Window title
old_win = 'window_name = "FIBO AI Navigation: 3D Point Cloud + TOF400C Laser + Global Grid + RRT*"'
new_win = 'window_name = "FIBO AI Navigation: 3D Point Cloud + TOF400C Laser + Global Grid + A*/RRT*"'
assert old_win in text, "window title not found"
text = text.replace(old_win, new_win, 1)

# 6. In apply_hot_reloaded_config
old_hot_end = 'controller.manual_turn_rpm = int(c_mot.get("manual_turn_rpm", 24))'
new_hot_end = (
    'controller.manual_turn_rpm = int(c_mot.get("manual_turn_rpm", 24))\n'
    '        nonlocal planner_type, planner\n'
    '        new_p_type = str(c_yp.get("planner_type", planner_type)).lower().strip()\n'
    '        if new_p_type in ("rrt", "rrt_star", "rrt*"):\n'
    '            new_p_type = "rrt"\n'
    '        else:\n'
    '            new_p_type = "astar"\n'
    '        if new_p_type != planner_type:\n'
    '            planner_type = new_p_type\n'
    '            planner = astar_planner if planner_type == "astar" else rrt_planner\n'
    '            current_smooth_path = None\n'
    '            current_raw_path = None\n'
    '            last_plan_time = 0.0\n'
    '            print(f"[CONFIG] Switched planner from hot-reload to: {planner_type.upper()}")'
)
assert old_hot_end in text, "hot reload end not found"
text = text.replace(old_hot_end, new_hot_end, 1)

# 7. In handle_key nonlocal
old_nonlocal = 'nonlocal map_locked, map_status_banner, map_status_time'
new_nonlocal = 'nonlocal map_locked, map_status_banner, map_status_time, planner_type, planner'
assert old_nonlocal in text, "nonlocal not found"
text = text.replace(old_nonlocal, new_nonlocal, 1)

# 8. In handle_key add key 'n' / 'N'
old_key_loc = "        elif key in (ord('v'), ord('V')):"
new_key_loc = (
    "        elif key in (ord('n'), ord('N')):\n"
    "            planner_type = 'rrt' if planner_type == 'astar' else 'astar'\n"
    "            planner = astar_planner if planner_type == 'astar' else rrt_planner\n"
    "            current_smooth_path = None\n"
    "            current_raw_path = None\n"
    "            last_plan_time = 0.0\n"
    "            cfg['yolo_and_planner']['planner_type'] = planner_type\n"
    "            p_desc = 'A* (Optimal Shortest)' if planner_type == 'astar' else 'RRT* (Sampling Tree)'\n"
    "            map_status_banner = f'PLANNER: {p_desc}'\n"
    "            map_status_time = time.time()\n"
    "            print(f'[PLANNER] Switched active path planner to: {p_desc}')\n"
    "        elif key in (ord('v'), ord('V')):"
)
assert old_key_loc in text, "key location not found"
text = text.replace(old_key_loc, new_key_loc, 1)

# 9. Startup banner
old_banner = '    print(f"  * Pi IP Address    : {resolved_pi_ip} (Auto-Discovery Active)")'
new_banner = (
    '    p_name = "A* (Optimal Shortest)" if planner_type == "astar" else "RRT* (Sampling Tree)"\n'
    '    print(f"  * Pi IP Address    : {resolved_pi_ip} (Auto-Discovery Active)")\n'
    '    print(f"  * Path Planner     : {p_name} (Tap \'N\' to switch A* <-> RRT*)")'
)
assert old_banner in text, "startup banner not found"
text = text.replace(old_banner, new_banner, 1)

# 10. HUD labels
old_hud_label = 'f"Global 2D Map {lock_badge} (Drag:Pan | Wheel:Zoom | R-Clk:Reloc | \'K\':LockMap | \'L\':Load | \'O\':Save) | Gear:{pwm_gear}PWM"'
new_hud_label = 'f"Global 2D Map {lock_badge} (Drag:Pan | Wheel:Zoom | R-Clk:Reloc | \'N\':A*/RRT* | \'K\':Lock | \'L\':Load | \'O\':Save) | Gear:{pwm_gear}PWM"'
assert old_hud_label in text, "global hud label not found"
text = text.replace(old_hud_label, new_hud_label, 1)

old_plan_hud = '_put_panel_label(plan_vis, f"RRT* + Teleop | {path_status}")'
new_plan_hud = (
    'p_badge = "A*" if planner_type == "astar" else "RRT*"\n'
    '            _put_panel_label(plan_vis, f"{p_badge} + Teleop | {path_status}")'
)
assert old_plan_hud in text, "plan hud label not found"
text = text.replace(old_plan_hud, new_plan_hud, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Successfully patched navigate_rrt.py with A* planner support!")
