with open('pointcloud_3d.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Initialize obstacle_mask in estimate_metric_depth
old_depth_init = '''        z_floor_row = np.clip(z_floor_row, 0.15, self.max_range_m).astype(np.float32)
        depth_map = np.broadcast_to(z_floor_row[:, np.newaxis], (h, w)).copy()'''
new_depth_init = '''        z_floor_row = np.clip(z_floor_row, 0.15, self.max_range_m).astype(np.float32)
        depth_map = np.broadcast_to(z_floor_row[:, np.newaxis], (h, w)).copy()
        obstacle_mask = np.zeros((h, w), dtype=bool)'''
assert old_depth_init in content, 'old_depth_init not found'
content = content.replace(old_depth_init, new_depth_init, 1)

# 2. In wall sensitivity block
old_wall_extrude = '''                        if z_wall < self.max_range_m:
                            v_top = max(0, v_base - int((1.40 * fy) / max(z_wall, 0.25)))
                            depth_map[v_top:v_base + 1, c0:c1] = np.minimum(
                                depth_map[v_top:v_base + 1, c0:c1], z_wall
                            )'''
new_wall_extrude = '''                        if z_wall < self.max_range_m:
                            v_top = max(0, v_base - int((1.40 * fy) / max(z_wall, 0.25)))
                            depth_map[v_top:v_base + 1, c0:c1] = np.minimum(
                                depth_map[v_top:v_base + 1, c0:c1], z_wall
                            )
                            obstacle_mask[v_top:v_base + 1, c0:c1] = True'''
assert old_wall_extrude in content, 'old_wall_extrude not found'
content = content.replace(old_wall_extrude, new_wall_extrude, 1)

# 3. In YOLO block
old_yolo_extrude = '''                depth_map[mask_hw > 0] = z_obj'''
new_yolo_extrude = '''                depth_map[mask_hw > 0] = z_obj
                obstacle_mask[mask_hw > 0] = True'''
assert old_yolo_extrude in content, 'old_yolo_extrude not found'
content = content.replace(old_yolo_extrude, new_yolo_extrude, 1)

# 4. Save latest_obstacle_mask
old_depth_return = '''        self.latest_depth_map = depth_map
        return depth_map'''
new_depth_return = '''        self.latest_depth_map = depth_map
        self.latest_obstacle_mask = obstacle_mask
        return depth_map'''
assert old_depth_return in content, 'old_depth_return not found'
content = content.replace(old_depth_return, new_depth_return, 1)

# 5. Update sem assignment in update_and_slice_bev
old_sem = '''        sem = np.zeros(len(Z_R), dtype=np.uint8)
        obs_mask = (Z_R >= self.min_obs_z) & (Z_R <= self.max_obs_z)
        over_mask = Z_R > self.max_obs_z
        sem[obs_mask] = 1
        sem[over_mask] = 2'''

new_sem = '''        obs_mask_sub = getattr(self, "latest_obstacle_mask", None)
        if obs_mask_sub is not None and obs_mask_sub.shape == (h, w):
            is_obs_px = obs_mask_sub[::s, ::s].ravel()[valid]
        else:
            is_obs_px = np.zeros(len(Z_R), dtype=bool)

        sem = np.zeros(len(Z_R), dtype=np.uint8)
        # In close range (0-4m), only confirmed obstacles (YOLO detections, ToF hits)
        # are marked OCCUPIED. Distant background/horizon pixels remain non-obstacle (sem=2).
        obs_mask = is_obs_px & (Z_R >= self.min_obs_z) & (Z_R <= self.max_obs_z)
        over_mask = (Z_R > self.max_obs_z) | (~is_obs_px & (Z_R >= self.min_obs_z))
        sem[obs_mask] = 1
        sem[over_mask] = 2'''
assert old_sem in content, 'old_sem not found'
content = content.replace(old_sem, new_sem, 1)

# 6. Remove duplicate line in ToF BEV stamping
old_tof_dup = '''            if hit_dist <= max_y:
                half_width_m = max(0.08, hit_dist * math.tan(math.radians(tof_beam_deg * 0.5)))
                half_cells = max(2, int(round((half_width_m / span_x) * grid_w)))
                c0 = max(0, cx_cell - half_cells)
                c1 = min(grid_w, cx_cell + half_cells + 1)
                bev_grid[max(0, hit_row - 1):min(grid_h, hit_row + 2), c0:c1] = OCCUPIED
                c1 = min(grid_w, cx_cell + half_cells + 1)
                bev_grid[max(0, hit_row - 1):min(grid_h, hit_row + 1), c0:c1] = OCCUPIED'''

new_tof_dup = '''            if hit_dist <= max_y:
                half_width_m = max(0.08, hit_dist * math.tan(math.radians(tof_beam_deg * 0.5)))
                half_cells = max(2, int(round((half_width_m / span_x) * grid_w)))
                c0 = max(0, cx_cell - half_cells)
                c1 = min(grid_w, cx_cell + half_cells + 1)
                bev_grid[max(0, hit_row - 1):min(grid_h, hit_row + 2), c0:c1] = OCCUPIED'''
assert old_tof_dup in content, 'old_tof_dup not found'
content = content.replace(old_tof_dup, new_tof_dup, 1)

with open('pointcloud_3d.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Phase 2 of pointcloud_3d.py applied successfully.')
