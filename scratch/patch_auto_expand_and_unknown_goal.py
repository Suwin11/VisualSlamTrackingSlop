"""
scratch/patch_auto_expand_and_unknown_goal.py
Patch global_map.py, astar.py, rrt_star.py, and navigate_rrt.py to:
1. Dynamically auto-expand the global occupancy grid so it NEVER stops loading/updating.
2. Treat unexplored/unknown space as FREE so the robot can navigate anywhere the user picks.
"""

import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# 1. Patch global_map.py
# ─────────────────────────────────────────────────────────────────────────────
gm_path = "global_map.py"
with open(gm_path, "r", encoding="utf-8") as f:
    gm_text = f.read()

# Replace _shift_grid_if_needed with ensure_bounds_range + ensure_bounds_include
old_shift = '''    def _shift_grid_if_needed(self, rx, ry):
        if not self.enable_rolling:
            return
        center_x = self.origin_x + 0.5 * self.width_m
        center_y = self.origin_y + 0.5 * self.height_m
        dx_m = rx - center_x
        dy_m = ry - center_y

        # Only shift when robot approaches within 0.8m of the outer map boundary
        max_dist_x = max(0.5 * self.width_m - 0.8, 2.8)
        max_dist_y = max(0.5 * self.height_m - 0.8, 2.8)

        shift_cols = int(round(dx_m / self.resolution)) if abs(dx_m) > max_dist_x else 0
        shift_rows = int(round(dy_m / self.resolution)) if abs(dy_m) > max_dist_y else 0

        if shift_cols == 0 and shift_rows == 0:
            return

        if abs(shift_cols) >= self.cols or abs(shift_rows) >= self.rows:
            self.log_odds.fill(0.0)
            self.binary_grid.fill(CELL_UNKNOWN)
            self.origin_x = rx - 0.5 * self.width_m
            self.origin_y = ry - 0.5 * self.height_m
            return

        if shift_cols != 0:
            self.log_odds = np.roll(self.log_odds, -shift_cols, axis=1)
            self.binary_grid = np.roll(self.binary_grid, -shift_cols, axis=1)
            if shift_cols > 0:
                self.log_odds[:, -shift_cols:] = 0.0
                self.binary_grid[:, -shift_cols:] = CELL_UNKNOWN
            else:
                self.log_odds[:, :-shift_cols] = 0.0
                self.binary_grid[:, :-shift_cols] = CELL_UNKNOWN
            self.origin_x += shift_cols * self.resolution

        if shift_rows != 0:
            self.log_odds = np.roll(self.log_odds, -shift_rows, axis=0)
            self.binary_grid = np.roll(self.binary_grid, -shift_rows, axis=0)
            if shift_rows > 0:
                self.log_odds[-shift_rows:, :] = 0.0
                self.binary_grid[-shift_rows:, :] = CELL_UNKNOWN
            else:
                self.log_odds[:-shift_rows, :] = 0.0
                self.binary_grid[:-shift_rows, :] = CELL_UNKNOWN
            self.origin_y += shift_rows * self.resolution'''

new_shift = '''    def ensure_bounds_range(self, min_x, max_x, min_y, max_y, margin_m=2.5):
        """
        Dynamically expands the occupancy grid canvas in any direction when coordinates
        approach or exceed the outer boundaries.
        
        - 100% of explored geometry, log-odds, and obstacles are preserved without shift or data loss.
        - Unobserved space is seamlessly padded with CELL_UNKNOWN (128).
        - Expand in chunks of 50 cells (2.5m) to avoid frequent re-allocation.
        """
        cur_min_x = self.origin_x
        cur_max_x = self.origin_x + self.width_m
        cur_min_y = self.origin_y
        cur_max_y = self.origin_y + self.height_m

        pad_left = 0
        pad_right = 0
        pad_bottom = 0
        pad_top = 0

        chunk_cells = max(10, int(round(2.5 / self.resolution)))

        if min_x < cur_min_x + margin_m:
            dx = (cur_min_x + margin_m) - min_x
            n_cells = int(math.ceil(dx / self.resolution))
            pad_left = ((n_cells // chunk_cells) + 1) * chunk_cells

        if max_x > cur_max_x - margin_m:
            dx = max_x - (cur_max_x - margin_m)
            n_cells = int(math.ceil(dx / self.resolution))
            pad_right = ((n_cells // chunk_cells) + 1) * chunk_cells

        if min_y < cur_min_y + margin_m:
            dy = (cur_min_y + margin_m) - min_y
            n_cells = int(math.ceil(dy / self.resolution))
            pad_bottom = ((n_cells // chunk_cells) + 1) * chunk_cells

        if max_y > cur_max_y - margin_m:
            dy = max_y - (cur_max_y - margin_m)
            n_cells = int(math.ceil(dy / self.resolution))
            pad_top = ((n_cells // chunk_cells) + 1) * chunk_cells

        if pad_left == 0 and pad_right == 0 and pad_bottom == 0 and pad_top == 0:
            return False

        # Apply padding to arrays
        self.log_odds = np.pad(
            self.log_odds,
            ((pad_bottom, pad_top), (pad_left, pad_right)),
            mode='constant',
            constant_values=0.0
        )
        self.binary_grid = np.pad(
            self.binary_grid,
            ((pad_bottom, pad_top), (pad_left, pad_right)),
            mode='constant',
            constant_values=CELL_UNKNOWN
        )
        self.inflated_grid = np.pad(
            self.inflated_grid,
            ((pad_bottom, pad_top), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )

        self.origin_x -= pad_left * self.resolution
        self.origin_y -= pad_bottom * self.resolution
        self.cols += pad_left + pad_right
        self.rows += pad_bottom + pad_top
        self.width_m = self.cols * self.resolution
        self.height_m = self.rows * self.resolution
        return True

    def ensure_bounds_include(self, wx, wy, margin_m=2.5):
        """Ensure the grid canvas contains the point (wx, wy) with margin_m padding."""
        return self.ensure_bounds_range(float(wx), float(wx), float(wy), float(wy), margin_m=margin_m)

    def _shift_grid_if_needed(self, rx, ry):
        """Auto-expand bounds around robot pose so the map never stops updating."""
        self.ensure_bounds_include(rx, ry, margin_m=3.5)'''

assert old_shift in gm_text, "old_shift not found in global_map.py"
gm_text = gm_text.replace(old_shift, new_shift, 1)

# Update update_from_local_bev to ensure bounds include all projected WX, WY points
old_bev_mesh = '''        WX = rx + LY * cos_t + LX * sin_t
        WY = ry + LY * sin_t - LX * cos_t

        GX = np.floor((WX - self.origin_x) / self.resolution).astype(np.int32)'''

new_bev_mesh = '''        WX = rx + LY * cos_t + LX * sin_t
        WY = ry + LY * sin_t - LX * cos_t

        # Auto-expand grid canvas to guarantee camera rays are always within bounds
        min_wx = float(np.min(WX))
        max_wx = float(np.max(WX))
        min_wy = float(np.min(WY))
        max_wy = float(np.max(WY))
        self.ensure_bounds_range(min_wx, max_wx, min_wy, max_wy, margin_m=1.5)

        GX = np.floor((WX - self.origin_x) / self.resolution).astype(np.int32)'''

assert old_bev_mesh in gm_text, "old_bev_mesh not found in global_map.py"
gm_text = gm_text.replace(old_bev_mesh, new_bev_mesh, 1)

# Update update_from_tof to ensure bounds include full laser reach
old_tof_init = '''        rx, ry, rtheta = robot_pose
        d_tof = float(tof_distance_m)
        max_m = float(max_range_m)
        self._shift_grid_if_needed(rx, ry)'''

new_tof_init = '''        rx, ry, rtheta = robot_pose
        d_tof = float(tof_distance_m)
        max_m = float(max_range_m)
        reach_m = min(d_tof, max_m) + 1.2
        self.ensure_bounds_range(rx - reach_m, rx + reach_m, ry - reach_m, ry + reach_m, margin_m=1.0)'''

assert old_tof_init in gm_text, "old_tof_init not found in global_map.py"
gm_text = gm_text.replace(old_tof_init, new_tof_init, 1)

# Update is_collision_free and is_line_free to treat unexplored/outside space as free
old_col_free = '''    def is_collision_free(self, wx, wy):
        pt = self.world_to_grid(wx, wy)
        if pt is None:
            return False
        gx, gy = pt
        if self.inflated_grid[gy, gx] > 0:
            return False
        return self.binary_grid[gy, gx] != CELL_OCCUPIED

    def is_line_free(self, x1, y1, x2, y2, step_size=0.025):
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1e-4:
            return self.is_collision_free(x1, y1)

        steps = max(2, int(math.ceil(dist / step_size)) + 1)
        xs = np.linspace(x1, x2, steps)
        ys = np.linspace(y1, y2, steps)

        gx = np.floor((xs - self.origin_x) / self.resolution).astype(np.int32)
        gy = np.floor((ys - self.origin_y) / self.resolution).astype(np.int32)

        if np.any((gx < 0) | (gx >= self.cols) | (gy < 0) | (gy >= self.rows)):
            return False

        if np.any(self.inflated_grid[gy, gx] > 0):
            return False

        return not np.any(self.binary_grid[gy, gx] == CELL_OCCUPIED)'''

new_col_free = '''    def is_collision_free(self, wx, wy):
        self.ensure_bounds_include(wx, wy, margin_m=1.0)
        pt = self.world_to_grid(wx, wy)
        if pt is None:
            return True  # Unexplored space is assumed FREE
        gx, gy = pt
        if self.inflated_grid[gy, gx] > 0:
            return False
        return self.binary_grid[gy, gx] != CELL_OCCUPIED

    def is_line_free(self, x1, y1, x2, y2, step_size=0.025):
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1e-4:
            return self.is_collision_free(x1, y1)

        self.ensure_bounds_range(min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2), margin_m=1.0)

        steps = max(2, int(math.ceil(dist / step_size)) + 1)
        xs = np.linspace(x1, x2, steps)
        ys = np.linspace(y1, y2, steps)

        gx = np.floor((xs - self.origin_x) / self.resolution).astype(np.int32)
        gy = np.floor((ys - self.origin_y) / self.resolution).astype(np.int32)

        in_bounds = (gx >= 0) & (gx < self.cols) & (gy >= 0) & (gy < self.rows)
        if not np.all(in_bounds):
            gx = gx[in_bounds]
            gy = gy[in_bounds]

        if gx.size == 0:
            return True  # Unexplored space is assumed FREE

        if np.any(self.inflated_grid[gy, gx] > 0):
            return False

        return not np.any(self.binary_grid[gy, gx] == CELL_OCCUPIED)'''

assert old_col_free in gm_text, "old_col_free not found in global_map.py"
gm_text = gm_text.replace(old_col_free, new_col_free, 1)

with open(gm_path, "w", encoding="utf-8") as f:
    f.write(gm_text)
print("[PASS] Patched global_map.py with auto-expanding canvas & unknown-as-free navigation!")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Patch astar.py
# ─────────────────────────────────────────────────────────────────────────────
astar_path = "astar.py"
with open(astar_path, "r", encoding="utf-8") as f:
    astar_text = f.read()

old_astar_plan = '''        sx, sy = float(start_pos[0]), float(start_pos[1])
        gx, gy = float(goal_pos[0]), float(goal_pos[1])

        # 1. Validate / snap start and goal to walkable space'''

new_astar_plan = '''        sx, sy = float(start_pos[0]), float(start_pos[1])
        gx, gy = float(goal_pos[0]), float(goal_pos[1])

        # Ensure grid canvas dynamically encompasses both start and goal
        self.grid.ensure_bounds_include(sx, sy, margin_m=2.0)
        self.grid.ensure_bounds_include(gx, gy, margin_m=2.0)

        # 1. Validate / snap start and goal to walkable space'''

assert old_astar_plan in astar_text, "old_astar_plan not found in astar.py"
astar_text = astar_text.replace(old_astar_plan, new_astar_plan, 1)

with open(astar_path, "w", encoding="utf-8") as f:
    f.write(astar_text)
print("[PASS] Patched astar.py to auto-expand bounds on start/goal!")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Patch rrt_star.py
# ─────────────────────────────────────────────────────────────────────────────
rrt_path = "rrt_star.py"
with open(rrt_path, "r", encoding="utf-8") as f:
    rrt_text = f.read()

old_rrt_plan = '''        self._update_bounds()
        sx, sy = start_pos
        gx, gy = goal_pos'''

new_rrt_plan = '''        sx, sy = float(start_pos[0]), float(start_pos[1])
        gx, gy = float(goal_pos[0]), float(goal_pos[1])
        if hasattr(self.grid, "ensure_bounds_include"):
            self.grid.ensure_bounds_include(sx, sy, margin_m=2.0)
            self.grid.ensure_bounds_include(gx, gy, margin_m=2.0)
        self._update_bounds()'''

assert old_rrt_plan in rrt_text, "old_rrt_plan not found in rrt_star.py"
rrt_text = rrt_text.replace(old_rrt_plan, new_rrt_plan, 1)

with open(rrt_path, "w", encoding="utf-8") as f:
    f.write(rrt_text)
print("[PASS] Patched rrt_star.py to auto-expand bounds on start/goal!")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Patch navigate_rrt.py
# ─────────────────────────────────────────────────────────────────────────────
nav_path = "navigate_rrt.py"
with open(nav_path, "r", encoding="utf-8") as f:
    nav_text = f.read()

old_click = '''                        # Stationary Click: Set Target Goal for RRT*
                        wx, wy = global_grid.screen_to_world(panel_x, panel_y, out_w=w, out_h=h)
                        current_goal = (round(wx, 2), round(wy, 2))'''

new_click = '''                        # Stationary Click: Set Target Goal (Auto-expand canvas if clicked in unexplored space)
                        wx, wy = global_grid.screen_to_world(panel_x, panel_y, out_w=w, out_h=h)
                        current_goal = (round(wx, 2), round(wy, 2))
                        global_grid.ensure_bounds_include(current_goal[0], current_goal[1], margin_m=3.0)'''

assert old_click in nav_text, "old_click not found in navigate_rrt.py"
nav_text = nav_text.replace(old_click, new_click, 1)

with open(nav_path, "w", encoding="utf-8") as f:
    f.write(nav_text)
print("[PASS] Patched navigate_rrt.py to auto-expand canvas on click in unexplored space!")
