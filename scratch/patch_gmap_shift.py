with open('global_map.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_shift = '''    def _shift_grid_if_needed(self, rx, ry):
        if not self.enable_rolling:
            return
        center_x = self.origin_x + 0.5 * self.width_m
        center_y = self.origin_y + 0.5 * self.height_m
        dx_m = rx - center_x
        dy_m = ry - center_y

        shift_cols = int(round(dx_m / self.resolution)) if abs(dx_m) > 1.5 else 0
        shift_rows = int(round(dy_m / self.resolution)) if abs(dy_m) > 1.5 else 0

        if shift_cols == 0 and shift_rows == 0:
            return

        if abs(shift_cols) >= self.cols or abs(shift_rows) >= self.rows:
            self.log_odds.fill(0.0)
            self.origin_x = rx - 0.5 * self.width_m
            self.origin_y = ry - 0.5 * self.height_m
            return

        if shift_cols != 0:
            self.log_odds = np.roll(self.log_odds, -shift_cols, axis=1)
            if shift_cols > 0:
                self.log_odds[:, -shift_cols:] = 0.0
            else:
                self.log_odds[:, :-shift_cols] = 0.0
            self.origin_x += shift_cols * self.resolution

        if shift_rows != 0:
            self.log_odds = np.roll(self.log_odds, -shift_rows, axis=0)
            if shift_rows > 0:
                self.log_odds[-shift_rows:, :] = 0.0
            else:
                self.log_odds[:-shift_rows, :] = 0.0
            self.origin_y += shift_rows * self.resolution'''

new_shift = '''    def _shift_grid_if_needed(self, rx, ry):
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

assert old_shift in content, 'old_shift not found in global_map.py'
content = content.replace(old_shift, new_shift, 1)

with open('global_map.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('global_map.py patched successfully.')
