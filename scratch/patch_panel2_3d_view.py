"""
scratch/patch_panel2_3d_view.py
Add feedback and clear toggle labels for Panel 2 (3D Point Cloud vs 2.5D Radar View).
"""

path = "navigate_rrt.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update 'v' key handler with banner
old_v = """        elif key in (ord('v'), ord('V')):
            show_3d_panel = not show_3d_panel
            sync_and_save_json()"""

new_v = """        elif key in (ord('v'), ord('V')):
            show_3d_panel = not show_3d_panel
            sync_and_save_json()
            mode_str = "3D Point Cloud View" if show_3d_panel else "2.5D Radar BEV View"
            map_status_banner = f"PANEL 2: {mode_str}"
            map_status_time = time.time()
            print(f"[PANEL] Panel 2 view switched to: {mode_str}")"""

text = text.replace(old_v, new_v, 1)

# 2. Update Panel 2 labels
old_lbl_3d = 'f"3D Cloud ({n_vox} vox) | Scale:{vo.metric_scale:.2f}x | WallSens:{pc_map.wall_sensitivity:.1f}x [-/=]"'
new_lbl_3d = 'f"3D Cloud ({n_vox} vox) | Orbit:IJKL | WallSens:{pc_map.wall_sensitivity:.1f}x [-/=] | \'V\':2D Radar"'
text = text.replace(old_lbl_3d, new_lbl_3d, 1)

old_lbl_2d = 'f"2.5D Radar+ToF | Free:{free_pct}% Occ:{occ_pct}% | WallSens:{pc_map.wall_sensitivity:.1f}x [-/=]"'
new_lbl_2d = 'f"2.5D Radar+ToF | Free:{free_pct}% Occ:{occ_pct}% | \'V\':3D Point Cloud"'
text = text.replace(old_lbl_2d, new_lbl_2d, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Successfully updated Panel 2 3D Point Cloud toggle labels!")
