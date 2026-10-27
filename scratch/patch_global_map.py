with open('global_map.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_tof = '''    def update_from_tof(self, robot_pose, tof_distance_m, max_range_m=4.0, beam_deg=12.0):
        """
        Direct high-rate laser rangefinder raycast into the persistent global map.
        Treats the ST TOF400C (VL53L1X) as the primary, high-precision distance sensor (up to 4.0m):
        - Carves 100% verified FREE space along the beam between robot and target (d_tof - 0.08m).
        - If d_tof <= max_range_m, stamps high-confidence OCCUPIED cells across the beam arc.
        """
        if tof_distance_m is None or not (0.04 <= float(tof_distance_m) <= float(max_range_m)):
            return

        rx, ry, rtheta = robot_pose
        d_tof = float(tof_distance_m)
        self._shift_grid_if_needed(rx, ry)

        half_beam = math.radians(beam_deg * 0.5)
        angles = [rtheta, rtheta - half_beam * 0.65, rtheta + half_beam * 0.65]

        free_indices = []
        occ_indices = []

        for ang in angles:
            cos_a = math.cos(ang)
            sin_a = math.sin(ang)

            # 1. Carve FREE space along the laser beam
            free_dist = max(0.0, d_tof - 0.08)
            num_steps = max(2, int(round(free_dist / (self.resolution * 0.5))))
            step_ds = np.linspace(0.04, free_dist, num_steps)
            fx = rx + step_ds * cos_a
            fy = ry + step_ds * sin_a

            fgx = np.floor((fx - self.origin_x) / self.resolution).astype(np.int32)
            fgy = np.floor((fy - self.origin_y) / self.resolution).astype(np.int32)
            valid_f = (fgx >= 0) & (fgx < self.cols) & (fgy >= 0) & (fgy < self.rows)
            if np.any(valid_f):
                free_indices.append(fgy[valid_f] * self.cols + fgx[valid_f])

            # 2. Stamp OCCUPIED at target distance
            if d_tof <= max_range_m:
                ox = rx + d_tof * cos_a
                oy = ry + d_tof * sin_a
                ogx = int(round((ox - self.origin_x) / self.resolution))
                ogy = int(round((oy - self.origin_y) / self.resolution))
                if 0 <= ogx < self.cols and 0 <= ogy < self.rows:
                    occ_indices.append(ogy * self.cols + ogx)

        if free_indices:
            all_free = np.unique(np.concatenate(free_indices))
        else:
            all_free = np.empty(0, dtype=np.int32)

        if occ_indices:
            all_occ = np.unique(np.array(occ_indices, dtype=np.int32))
            if all_free.size > 0:
                all_free = np.setdiff1d(all_free, all_occ, assume_unique=True)
        else:
            all_occ = np.empty(0, dtype=np.int32)

        if all_free.size > 0:
            self.log_odds.ravel()[all_free] += self.l_free * 0.55
        if all_occ.size > 0:
            # Laser hit has strong confidence
            self.log_odds.ravel()[all_occ] += self.l_occ * 1.50'''

new_tof = '''    def update_from_tof(self, robot_pose, tof_distance_m, max_range_m=4.0, beam_deg=12.0):
        """
        Direct high-rate laser rangefinder raycast into the persistent global map.
        Treats the ST TOF400C (VL53L1X) as the primary, high-precision distance sensor (up to 4.0m):
        - Carves 100% verified FREE space along the beam between robot and target (d_tof - 0.08m).
        - If d_tof <= max_range_m, stamps high-confidence OCCUPIED cells across the beam arc.
        - Actively clears false camera ghost obstacles along the verified laser line-of-sight.
        """
        if tof_distance_m is None or float(tof_distance_m) < 0.04:
            return

        rx, ry, rtheta = robot_pose
        d_tof = float(tof_distance_m)
        max_m = float(max_range_m)
        self._shift_grid_if_needed(rx, ry)

        half_beam = math.radians(beam_deg * 0.5)
        angles = [rtheta, rtheta - half_beam * 0.65, rtheta + half_beam * 0.65]

        free_indices = []
        occ_indices = []

        for ang in angles:
            cos_a = math.cos(ang)
            sin_a = math.sin(ang)

            # 1. Carve FREE space along the laser beam (out to hit or max_m)
            free_dist = max(0.0, min(d_tof, max_m) - 0.08) if d_tof <= max_m else max_m
            num_steps = max(2, int(round(free_dist / (self.resolution * 0.5))))
            step_ds = np.linspace(0.04, free_dist, num_steps)
            fx = rx + step_ds * cos_a
            fy = ry + step_ds * sin_a

            fgx = np.floor((fx - self.origin_x) / self.resolution).astype(np.int32)
            fgy = np.floor((fy - self.origin_y) / self.resolution).astype(np.int32)
            valid_f = (fgx >= 0) & (fgx < self.cols) & (fgy >= 0) & (fgy < self.rows)
            if np.any(valid_f):
                free_indices.append(fgy[valid_f] * self.cols + fgx[valid_f])

            # 2. Stamp OCCUPIED at target distance if within range
            if d_tof <= max_m:
                ox = rx + d_tof * cos_a
                oy = ry + d_tof * sin_a
                ogx = int(round((ox - self.origin_x) / self.resolution))
                ogy = int(round((oy - self.origin_y) / self.resolution))
                if 0 <= ogx < self.cols and 0 <= ogy < self.rows:
                    occ_indices.append(ogy * self.cols + ogx)

        if free_indices:
            all_free = np.unique(np.concatenate(free_indices))
        else:
            all_free = np.empty(0, dtype=np.int32)

        if occ_indices:
            all_occ = np.unique(np.array(occ_indices, dtype=np.int32))
            if all_free.size > 0:
                all_free = np.setdiff1d(all_free, all_occ, assume_unique=True)
        else:
            all_occ = np.empty(0, dtype=np.int32)

        if all_free.size > 0:
            # Actively reset any false camera obstacles along the verified laser line-of-sight
            occ_in_free = all_free[self.log_odds.ravel()[all_free] > 0]
            if occ_in_free.size > 0:
                self.log_odds.ravel()[occ_in_free] = self.l_free * 1.5
            self.log_odds.ravel()[all_free] += self.l_free * 0.75

        if all_occ.size > 0:
            # Laser hit has strong confidence
            self.log_odds.ravel()[all_occ] += self.l_occ * 1.50'''

assert old_tof in content, 'old_tof not found'
content = content.replace(old_tof, new_tof, 1)

with open('global_map.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('global_map.py patched successfully.')
