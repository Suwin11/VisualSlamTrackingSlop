with open('occupancy.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update _detect_non_coco_obstacles to mask out upper horizon
old_detect = '''    non_floor = (valid_bev > 0) & (grad_y > grad_thresh) & (diff > diff_thresh)
    non_floor[grid_h - 4:, :] = False
    return (non_floor.astype(np.uint8) * 255)'''

new_detect = '''    non_floor = (valid_bev > 0) & (grad_y > grad_thresh) & (diff > diff_thresh)
    # The top 45% of IPM grid corresponds to distant horizon/infinity; never mark as close-range obstacle
    non_floor[:int(grid_h * 0.45), :] = False
    non_floor[grid_h - 4:, :] = False
    return (non_floor.astype(np.uint8) * 255)'''
assert old_detect in content, 'old_detect not found'
content = content.replace(old_detect, new_detect, 1)

# 2. Update compute default wall_sensitivity and mask top rows of occ
old_compute = '''def compute(yolo_combined_mask, orb_points, frame_shape, H_ipm,
            grid_w=GRID_W, grid_h=GRID_H, frame_gray=None, wall_sensitivity=1.0,
            tof_distance_m=None, max_range_m=4.0, tof_beam_deg=12.0):
    """Build a top-down occupancy grid for one frame, fused with TOF400C laser ranging (up to 4.0m)."""
    grid = np.full((grid_h, grid_w), UNKNOWN, dtype=np.uint8)

    occ = cv2.warpPerspective(yolo_combined_mask, H_ipm,
                              (grid_w, grid_h), flags=cv2.INTER_NEAREST)
    if frame_gray is not None:
        non_coco_occ = _detect_non_coco_obstacles(
            frame_gray, H_ipm, grid_w, grid_h, wall_sensitivity=wall_sensitivity
        )
        occ = cv2.bitwise_or(occ, non_coco_occ)

    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, _WARP_CLOSE_KERNEL)'''

new_compute = '''def compute(yolo_combined_mask, orb_points, frame_shape, H_ipm,
            grid_w=GRID_W, grid_h=GRID_H, frame_gray=None, wall_sensitivity=0.0,
            tof_distance_m=None, max_range_m=4.0, tof_beam_deg=12.0):
    """Build a top-down occupancy grid for one frame, fused with TOF400C laser ranging (up to 4.0m)."""
    grid = np.full((grid_h, grid_w), UNKNOWN, dtype=np.uint8)

    occ = cv2.warpPerspective(yolo_combined_mask, H_ipm,
                              (grid_w, grid_h), flags=cv2.INTER_NEAREST)
    if frame_gray is not None and wall_sensitivity > 0.05:
        non_coco_occ = _detect_non_coco_obstacles(
            frame_gray, H_ipm, grid_w, grid_h, wall_sensitivity=wall_sensitivity
        )
        occ = cv2.bitwise_or(occ, non_coco_occ)

    # In close range (0-4m), top horizon cells must remain UNKNOWN, never marked OCCUPIED
    occ[:int(grid_h * 0.35), :] = 0
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, _WARP_CLOSE_KERNEL)'''
assert old_compute in content, 'old_compute not found'
content = content.replace(old_compute, new_compute, 1)

# 3. Update ToF ray carving in compute
old_tof_carve = '''    # --- Fuse TOF400C Laser Rangefinder Ray & Obstacle Hit (up to 4.0m) ---
    if tof_distance_m is not None and tof_distance_m > 0.04:
        cx_cell = grid_w // 2
        hit_dist = float(tof_distance_m)
        hit_row = int(round(((max_range_m - min(hit_dist, max_range_m)) / max(max_range_m, 0.5)) * (grid_h - 1)))
        hit_row = int(np.clip(hit_row, 0, grid_h - 1))

        # Carve FREE space along the laser beam between the robot (bottom) and the laser hit
        for r in range(min(grid_h - 1, hit_row + 2), grid_h):
            grid[r, max(0, cx_cell - 2):min(grid_w, cx_cell + 3)] = FREE

        # If obstacle is within the map's forward range, stamp OCCUPIED arc across the laser cone
        if hit_dist <= max_range_m:
            half_width_m = max(0.08, hit_dist * math.tan(math.radians(tof_beam_deg * 0.5)))
            half_cells = max(2, int(round((half_width_m / 2.0) * grid_w)))
            c0 = max(0, cx_cell - half_cells)
            c1 = min(grid_w, cx_cell + half_cells + 1)
            r0 = max(0, hit_row - 1)
            r1 = min(grid_h, hit_row + 2)
            grid[r0:r1, c0:c1] = OCCUPIED'''

new_tof_carve = '''    # --- Fuse TOF400C Laser Rangefinder Ray & Obstacle Hit (up to 4.0m) ---
    if tof_distance_m is not None and tof_distance_m > 0.04:
        cx_cell = grid_w // 2
        hit_dist = float(tof_distance_m)
        effective_dist = min(hit_dist, float(max_range_m))
        hit_row = int(round(((max_range_m - effective_dist) / max(max_range_m, 0.5)) * (grid_h - 1)))
        hit_row = int(np.clip(hit_row, 0, grid_h - 1))

        # Carve FREE space along the laser beam between the robot (bottom) and the laser hit
        free_start_r = min(grid_h - 1, hit_row + 2) if hit_dist <= max_range_m else 0
        for r in range(free_start_r, grid_h):
            grid[r, max(0, cx_cell - 2):min(grid_w, cx_cell + 3)] = FREE

        # If obstacle is within the map's forward range, stamp OCCUPIED arc across the laser cone
        if hit_dist <= max_range_m:
            half_width_m = max(0.08, hit_dist * math.tan(math.radians(tof_beam_deg * 0.5)))
            half_cells = max(2, int(round((half_width_m / 2.0) * grid_w)))
            c0 = max(0, cx_cell - half_cells)
            c1 = min(grid_w, cx_cell + half_cells + 1)
            r0 = max(0, hit_row - 1)
            r1 = min(grid_h, hit_row + 2)
            grid[r0:r1, c0:c1] = OCCUPIED'''
assert old_tof_carve in content, 'old_tof_carve not found'
content = content.replace(old_tof_carve, new_tof_carve, 1)

# 4. Update render() signature and range rings
old_render = '''def render(grid, out_w=640, out_h=480, tof_distance_m=None, max_range_m=2.5):
    gh, gw = grid.shape
    vis = np.zeros((gh, gw, 3), dtype=np.uint8)

    vis[grid == UNKNOWN]  = _PAL_UNKNOWN
    vis[grid == FREE]     = _PAL_FREE
    vis[grid == OCCUPIED] = _PAL_OCCUPIED

    vis = cv2.resize(vis, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    cx = out_w // 2
    cy = out_h - 10

    px_per_m = (out_h - 20) / max(max_range_m, 0.5)
    for m in (1.0, 2.0):'''

new_render = '''def render(grid, out_w=640, out_h=480, tof_distance_m=None, max_range_m=4.0):
    gh, gw = grid.shape
    vis = np.zeros((gh, gw, 3), dtype=np.uint8)

    vis[grid == UNKNOWN]  = _PAL_UNKNOWN
    vis[grid == FREE]     = _PAL_FREE
    vis[grid == OCCUPIED] = _PAL_OCCUPIED

    vis = cv2.resize(vis, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    cx = out_w // 2
    cy = out_h - 10

    px_per_m = (out_h - 20) / max(max_range_m, 0.5)
    for m in (1.0, 2.0, 3.0, 4.0):'''
assert old_render in content, 'old_render not found'
content = content.replace(old_render, new_render, 1)

with open('occupancy.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('occupancy.py patched successfully.')
