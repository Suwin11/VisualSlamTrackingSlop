with open('global_map.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add map_locked and protect_explored_walls to __init__
old_init = '''        self.log_odds = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.binary_grid = np.full((self.rows, self.cols), CELL_UNKNOWN, dtype=np.uint8)
        self.inflated_grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        self._rebuild_inflate_kernel()'''

new_init = '''        self.log_odds = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.binary_grid = np.full((self.rows, self.cols), CELL_UNKNOWN, dtype=np.uint8)
        self.inflated_grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # Map protection & lock controls
        self.map_locked = False
        self.protect_explored_walls = True

        self._rebuild_inflate_kernel()'''

assert old_init in content, 'old_init not found in global_map.py'
content = content.replace(old_init, new_init, 1)

# 2. Add .json resolution in load_map
old_load_head = '''        if not os.path.exists(filepath):
            if not filepath.endswith(".npz"):
                filepath += ".npz"
            if not os.path.exists(filepath):
                print(f"[MAP-LOAD ERROR] Map file not found: {filepath}")
                return False'''

new_load_head = '''        if filepath.endswith(".json"):
            npz_cand = os.path.splitext(filepath)[0] + ".npz"
            if os.path.exists(npz_cand):
                filepath = npz_cand

        if not os.path.exists(filepath):
            if not filepath.endswith(".npz"):
                if os.path.exists(filepath + ".npz"):
                    filepath += ".npz"
            if not os.path.exists(filepath):
                print(f"[MAP-LOAD ERROR] Map file not found: {filepath}")
                return False'''

assert old_load_head in content, 'old_load_head not found in global_map.py'
content = content.replace(old_load_head, new_load_head, 1)

# 3. In load_map: ensure protect_explored_walls is True
old_load_end = '''            # When loading a pre-built static map, disable rolling window by default
            # so the coordinate system remains absolute and anchored to the user's floorplan
            self.enable_rolling = False
            self._last_update_pose = None'''

new_load_end = '''            # When loading a pre-built static map, disable rolling window by default
            # so the coordinate system remains absolute and anchored to the user's floorplan
            self.enable_rolling = False
            self.protect_explored_walls = True
            self._last_update_pose = None'''

assert old_load_end in content, 'old_load_end not found in global_map.py'
content = content.replace(old_load_end, new_load_end, 1)

# 4. In update_from_local_bev: check map_locked and protect_explored_walls
old_bev_start = '''    def update_from_local_bev(self, local_grid, robot_pose,
                              local_range_x=(-1.0, 1.0), local_range_y=(0.0, 2.5)):
        """
        Integrate a single-frame local BEV grid into the persistent global map.
        """
        rx, ry, rtheta = robot_pose'''

new_bev_start = '''    def update_from_local_bev(self, local_grid, robot_pose,
                              local_range_x=(-1.0, 1.0), local_range_y=(0.0, 2.5)):
        """
        Integrate a single-frame local BEV grid into the persistent global map.
        """
        if getattr(self, "map_locked", False):
            return
        rx, ry, rtheta = robot_pose'''

assert old_bev_start in content, 'old_bev_start not found in global_map.py'
content = content.replace(old_bev_start, new_bev_start, 1)

# 5. In update_from_local_bev: protect explored walls from free_cells wipeout
old_free_cells = '''        if occ_cells.size > 0 and free_cells.size > 0:
            free_cells = np.setdiff1d(free_cells, occ_cells, assume_unique=True)

        scale = 1.0 if moved else 0.45'''

new_free_cells = '''        if occ_cells.size > 0 and free_cells.size > 0:
            free_cells = np.setdiff1d(free_cells, occ_cells, assume_unique=True)

        # Do not override / erase explored walls!
        if getattr(self, "protect_explored_walls", True) and free_cells.size > 0:
            is_already_occ = (self.binary_grid.ravel()[free_cells] == CELL_OCCUPIED)
            if np.any(is_already_occ):
                free_cells = free_cells[~is_already_occ]

        scale = 1.0 if moved else 0.45'''

assert old_free_cells in content, 'old_free_cells not found in global_map.py'
content = content.replace(old_free_cells, new_free_cells, 1)

# 6. In update_from_tof: check map_locked and protect_explored_walls
old_tof_start = '''    def update_from_tof(self, robot_pose, tof_distance_m, max_range_m=4.0, beam_deg=12.0):
        """
        Direct high-rate laser rangefinder raycast into the persistent global map.'''

new_tof_start = '''    def update_from_tof(self, robot_pose, tof_distance_m, max_range_m=4.0, beam_deg=12.0):
        """
        Direct high-rate laser rangefinder raycast into the persistent global map.'''

# Check where tof checks map_locked
old_tof_body = '''        if tof_distance_m is None or float(tof_distance_m) < 0.04:
            return

        rx, ry, rtheta = robot_pose'''

new_tof_body = '''        if getattr(self, "map_locked", False):
            return
        if tof_distance_m is None or float(tof_distance_m) < 0.04:
            return

        rx, ry, rtheta = robot_pose'''

assert old_tof_body in content, 'old_tof_body not found in global_map.py'
content = content.replace(old_tof_body, new_tof_body, 1)

# In update_from_tof: protect explored walls from laser clearing
old_tof_free = '''        if all_free.size > 0:
            # Actively reset any false camera obstacles along the verified laser line-of-sight
            occ_in_free = all_free[self.log_odds.ravel()[all_free] > 0]
            if occ_in_free.size > 0:
                self.log_odds.ravel()[occ_in_free] = self.l_free * 1.5
            self.log_odds.ravel()[all_free] += self.l_free * 0.75'''

new_tof_free = '''        if all_free.size > 0:
            if getattr(self, "protect_explored_walls", True):
                is_already_occ = (self.binary_grid.ravel()[all_free] == CELL_OCCUPIED)
                if np.any(is_already_occ):
                    all_free = all_free[~is_already_occ]
            if all_free.size > 0:
                occ_in_free = all_free[self.log_odds.ravel()[all_free] > 0]
                if occ_in_free.size > 0:
                    self.log_odds.ravel()[occ_in_free] = self.l_free * 1.5
                self.log_odds.ravel()[all_free] += self.l_free * 0.75'''

assert old_tof_free in content, 'old_tof_free not found in global_map.py'
content = content.replace(old_tof_free, new_tof_free, 1)

with open('global_map.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('global_map.py successfully patched with map_locked and wall protection!')
