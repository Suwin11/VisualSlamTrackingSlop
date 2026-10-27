"""
pointcloud_3d.py
Real-Time 3D Point Cloud Reconstruction, Height-Band Obstacle Slicing (2.5D),
TOF400C Laser Rangefinder Depth Fusion, Persistent 3D Voxel Map, and 3D Renderer.
"""

import math
import os
import cv2
import numpy as np

from occupancy import FREE, OCCUPIED, UNKNOWN, GRID_W, GRID_H


class PointCloudMap3D:
    """
    Hybrid 3D Point Cloud Mapper, TOF400C Laser Depth Anchor, 2.5D Height-Slice
    Occupancy Generator, and Interactive 3D Perspective Renderer.
    """

    def __init__(self, camera_height_m=0.20, min_obs_height_m=0.04,
                 max_obs_height_m=1.60, max_range_m=5.0, voxel_size_m=0.05,
                 max_global_points=80000, stride=3, wall_sensitivity=0.0,
                 close_range_m=4.0, show_ceiling=False):
        self.camera_height = float(camera_height_m)
        self.min_obs_z = float(min_obs_height_m)
        self.max_obs_z = float(max_obs_height_m)
        self.max_range_m = float(max_range_m)
        self.voxel_size = float(voxel_size_m)
        self.max_global_points = int(max_global_points)
        self.stride = int(max(2, stride))
        self.wall_sensitivity = float(wall_sensitivity)
        self.close_range_m = float(close_range_m)
        self.show_ceiling = bool(show_ceiling)

        self.global_xyz = np.empty((0, 3), dtype=np.float32)
        self.global_bgr = np.empty((0, 3), dtype=np.uint8)
        self.global_sem = np.empty((0,), dtype=np.uint8)

        self.local_xyz = np.empty((0, 3), dtype=np.float32)
        self.local_bgr = np.empty((0, 3), dtype=np.uint8)
        self.local_sem = np.empty((0,), dtype=np.uint8)
        self.latest_depth_map = None
        self.latest_obstacle_mask = None

        self.cam_yaw_offset_deg = 0.0
        self.cam_Pitch_deg = 28.0
        self.cam_dist_m = 2.2
        self.color_by_height = True
        self.color_mode = 1  # 0: RGB Texture, 1: Height Elevation, 2: Turbo Distance Heatmap

        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    @property
    def color_mode_name(self):
        modes = {0: "RGB Texture", 1: "Height / Obstacle", 2: "Turbo Depth Heatmap"}
        return modes.get(self.color_mode, "Height")

    def cycle_color_mode(self):
        self.color_mode = (self.color_mode + 1) % 3
        self.color_by_height = (self.color_mode == 1)
        return self.color_mode_name

    def toggle_ceiling(self):
        self.show_ceiling = not self.show_ceiling
        status = "VISIBLE" if self.show_ceiling else "HIDDEN (Floor+Obstacles Only)"
        print(f"[3D VIEW] Overhead / Ceiling points -> {status}")
        return self.show_ceiling

    def reset(self):
        self.global_xyz = np.empty((0, 3), dtype=np.float32)
        self.global_bgr = np.empty((0, 3), dtype=np.uint8)
        self.global_sem = np.empty((0,), dtype=np.uint8)
        self.local_xyz = np.empty((0, 3), dtype=np.float32)
        self.local_bgr = np.empty((0, 3), dtype=np.uint8)
        self.local_sem = np.empty((0,), dtype=np.uint8)
        self.latest_depth_map = None
        self.latest_obstacle_mask = None

    def adjust_orbit(self, dyaw_deg=0.0, dpitch_deg=0.0, ddist_m=0.0):
        self.cam_yaw_offset_deg = (self.cam_yaw_offset_deg + dyaw_deg + 180.0) % 360.0 - 180.0
        self.cam_Pitch_deg = float(np.clip(self.cam_Pitch_deg + dpitch_deg, 8.0, 80.0))
        self.cam_dist_m = float(np.clip(self.cam_dist_m + ddist_m, 0.8, 6.0))

    def adjust_wall_sensitivity(self, delta):
        self.wall_sensitivity = float(np.clip(self.wall_sensitivity + delta, 0.0, 3.0))
        print(f"[MAP-TUNE] Wall/Non-COCO Obstacle Sensitivity -> {self.wall_sensitivity:.2f}x "
              f"(0.0 = YOLO+ToF only, 1.0 = default, >1.3 = detect faint walls)")
        return self.wall_sensitivity

    def estimate_metric_depth(self, frame_bgr, pitch_rad, fx, fy, cx, cy,
                              yolo_detections=None, tof_distance_m=None):
        """
        Build a dense metric depth map `Z_cam (H, W)` (in meters along the camera optical axis)
        combining ground-plane geometry, vertical extrusion of YOLO instance masks,
        non-COCO wall detection, and direct TOF400C laser rangefinder anchoring.
        """
        h, w = frame_bgr.shape[:2]
        sin_p = math.sin(pitch_rad)
        cos_p = math.cos(pitch_rad)

        v_coords = np.arange(h, dtype=np.float32)
        yn_row = (v_coords - cy) / fy
        y_down_row = yn_row * cos_p + sin_p

        valid_floor_row = y_down_row > 0.018
        z_floor_row = np.where(
            valid_floor_row,
            np.clip(self.camera_height / np.maximum(y_down_row, 0.018), 0.15, self.max_range_m),
            np.nan
        ).astype(np.float32)
        depth_map = np.broadcast_to(z_floor_row[:, np.newaxis], (h, w)).copy()
        obstacle_mask = np.zeros((h, w), dtype=bool)

        cx_i = int(round(cx))
        cy_i = int(round(cy))
        tof_valid = (tof_distance_m is not None) and (0.04 <= float(tof_distance_m) <= self.max_range_m)

        # 2. Non-COCO Wall / Box / Vertical Barrier Extrusion
        if self.wall_sensitivity > 0.05:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            ref_y1, ref_y2 = int(h * 0.78), int(h * 0.96)
            ref_x1, ref_x2 = int(w * 0.35), int(w * 0.65)
            ref_patch = gray[ref_y1:ref_y2, ref_x1:ref_x2]
            if ref_patch.size > 0:
                ref_med = float(np.median(ref_patch))
                ref_std = max(float(np.std(ref_patch)), 10.0)

                small_w = w // 4
                small_gray = cv2.resize(gray, (small_w, h), interpolation=cv2.INTER_AREA)
                grad_y = np.abs(cv2.Sobel(small_gray, cv2.CV_16S, 0, 1, ksize=3)).astype(np.float32)
                diff = np.abs(small_gray.astype(np.float32) - ref_med)

                horizon_v = int(np.argmax(valid_floor_row)) if np.any(valid_floor_row) else int(h * 0.45)
                search_start = max(horizon_v + 4, int(h * 0.35))
                search_end = int(h * 0.92)

                inv_sens = 1.0 / max(self.wall_sensitivity, 0.25)
                grad_thresh = 26.0 * inv_sens
                diff_thresh = max(15.0 * inv_sens, (1.6 * inv_sens) * ref_std)

                if search_end > search_start:
                    sub_gray = small_gray[search_start:search_end, :]
                    seam = cv2.Canny(sub_gray, int(25 * inv_sens), int(70 * inv_sens))
                    edge_band = (
                        (grad_y[search_start:search_end, :] > grad_thresh)
                        | (diff[search_start:search_end, :] > diff_thresh)
                        | (seam > 0)
                    )
                    has_wall = np.any(edge_band, axis=0)
                    wall_rel_idx = (search_end - search_start - 1) - np.argmax(edge_band[::-1, :], axis=0)
                    wall_base_v = search_start + wall_rel_idx

                    for col_s in np.flatnonzero(has_wall):
                        v_base = int(wall_base_v[col_s])
                        z_wall = float(z_floor_row[v_base])
                        c0 = col_s * 4
                        c1 = min(w, c0 + 4)
                        if tof_valid and (w * 0.38 <= (c0 + c1) * 0.5 <= w * 0.62):
                            z_wall = min(z_wall, float(tof_distance_m) / max(cos_p, 0.5))

                        if z_wall < self.max_range_m:
                            v_top = max(0, v_base - int((1.40 * fy) / max(z_wall, 0.25)))
                            depth_map[v_top:v_base + 1, c0:c1] = np.minimum(
                                depth_map[v_top:v_base + 1, c0:c1], z_wall
                            )
                            obstacle_mask[v_top:v_base + 1, c0:c1] = True

        # 3. Extrude YOLO Segmented Objects Vertically (and anchor to TOF400C if in center beam!)
        cx_i = int(round(cx))
        cy_i = int(round(cy))
        tof_valid = (tof_distance_m is not None) and (0.04 <= float(tof_distance_m) <= self.max_range_m)

        if yolo_detections:
            for box, mask_hw, _ in yolo_detections:
                x1, y1, x2, y2 = box
                v_base = int(np.clip(y2, 0, h - 1))
                # If the TOF400C laser beam hits inside this YOLO object's bounding box,
                # use the exact laser distance!
                if tof_valid and (x1 <= cx_i <= x2) and (y1 <= cy_i <= y2):
                    z_obj = float(tof_distance_m) / max(cos_p, 0.5)
                elif valid_floor_row[v_base]:
                    z_obj = float(z_floor_row[v_base])
                else:
                    box_h = max(y2 - y1, 12)
                    z_obj = float(np.clip((0.50 * fy) / box_h, 0.25, self.max_range_m))
                depth_map[mask_hw > 0] = z_obj
                obstacle_mask[mask_hw > 0] = True

        self.latest_depth_map = depth_map
        self.latest_obstacle_mask = obstacle_mask
        return depth_map

    def update_and_slice_bev(self, frame_bgr, depth_map, robot_pose, pitch_rad,
                             fx, fy, cx, cy, grid_w=GRID_W, grid_h=GRID_H,
                             local_range_x=(-1.0, 1.0), local_range_y=(0.0, 4.0),
                             accumulate_global=True, tof_distance_m=None,
                             tof_beam_deg=12.0):
        """
        Unproject the metric depth map into a 3D Point Cloud in the robot frame,
        inject TOF400C 3D laser curtain points, accumulate into the global 3D voxel map,
        and slice points by height `Z_R` to produce the 2D local BEV occupancy grid.
        """
        h, w = depth_map.shape[:2]
        s = self.stride

        us = np.arange(0, w, s, dtype=np.float32)
        vs = np.arange(0, h, s, dtype=np.float32)
        U, V = np.meshgrid(us, vs)

        Z_c = depth_map[::s, ::s]
        bgr_sub = frame_bgr[::s, ::s].reshape(-1, 3)

        xn = (U - cx) / fx
        yn = (V - cy) / fy

        sin_p = math.sin(pitch_rad)
        cos_p = math.cos(pitch_rad)

        X_c = xn * Z_c
        Y_c = yn * Z_c

        X_R = X_c.ravel()
        Y_R = (Z_c * cos_p - Y_c * sin_p).ravel()
        Z_R = (self.camera_height - (Y_c * cos_p + Z_c * sin_p)).ravel()

        valid = (~np.isnan(Z_R)) & (Y_R >= 0.10) & (Y_R <= self.max_range_m) & (np.abs(X_R) <= 3.8) & (Z_R >= -0.28) & (Z_R <= 2.40)
        X_R = X_R[valid]
        Y_R = Y_R[valid]
        Z_R = Z_R[valid]
        bgr_valid = bgr_sub[valid]

        obs_mask_sub = getattr(self, "latest_obstacle_mask", None)
        if obs_mask_sub is not None and obs_mask_sub.shape == (h, w):
            is_obs_px = obs_mask_sub[::s, ::s].ravel()[valid]
        else:
            is_obs_px = np.zeros(len(Z_R), dtype=bool)

        sem = np.zeros(len(Z_R), dtype=np.uint8)
        # Confirmed obstacles: YOLO detections or detected obstacles within height band
        obs_mask = is_obs_px & (Z_R >= self.min_obs_z) & (Z_R <= self.max_obs_z)
        over_mask = (Z_R > self.max_obs_z)
        sem[obs_mask] = 1
        sem[over_mask] = 2

        # --- Inject 3D TOF400C Laser Hit Curtain (captures featureless walls & dark obstacles!) ---
        if tof_distance_m is not None and 0.04 <= float(tof_distance_m) <= self.max_range_m:
            d_tof = float(tof_distance_m)
            half_w = max(0.06, d_tof * math.tan(math.radians(tof_beam_deg * 0.5)))
            xs_tof = np.linspace(-half_w, half_w, 11, dtype=np.float32)
            zs_tof = np.linspace(self.min_obs_z + 0.02, min(self.max_obs_z - 0.05, 1.20), 8, dtype=np.float32)
            XT, ZT = np.meshgrid(xs_tof, zs_tof)
            XT = XT.ravel()
            ZT = ZT.ravel()
            YT = np.full_like(XT, d_tof)
            bgr_tof = np.tile(np.array([[255, 40, 255]], dtype=np.uint8), (len(XT), 1))
            sem_tof = np.ones(len(XT), dtype=np.uint8)

            X_R = np.concatenate([X_R, XT])
            Y_R = np.concatenate([Y_R, YT])
            Z_R = np.concatenate([Z_R, ZT])
            bgr_valid = np.vstack([bgr_valid, bgr_tof]) if len(bgr_valid) > 0 else bgr_tof
            sem = np.concatenate([sem, sem_tof])

        self.local_xyz = np.column_stack([X_R, Y_R, Z_R]).astype(np.float32)
        self.local_bgr = bgr_valid
        self.local_sem = sem

        # --- Build 2.5D Height-Sliced Local BEV Occupancy Grid ---
        bev_grid = np.full((grid_h, grid_w), UNKNOWN, dtype=np.uint8)
        span_x = local_range_x[1] - local_range_x[0]
        span_y = local_range_y[1] - local_range_y[0]

        gx = np.floor(((X_R - local_range_x[0]) / span_x) * grid_w).astype(np.int32)
        gy = np.floor(((local_range_y[1] - Y_R) / span_y) * grid_h).astype(np.int32)
        in_bev = (gx >= 0) & (gx < grid_w) & (gy >= 0) & (gy < grid_h)

        floor_pts = in_bev & (sem == 0)
        bev_grid[gy[floor_pts], gx[floor_pts]] = FREE

        occ_pts = in_bev & (sem == 1)
        occ_layer = np.zeros((grid_h, grid_w), dtype=np.uint8)
        occ_layer[gy[occ_pts], gx[occ_pts]] = 255
        occ_layer = cv2.morphologyEx(occ_layer, cv2.MORPH_CLOSE, self._close_kernel)

        has_obs = np.any(occ_layer > 0, axis=0)
        lowest_obs = (grid_h - 1) - np.argmax(occ_layer[::-1, :] > 0, axis=0)
        horizon_row = int(grid_h * 0.35)
        stop_row = np.where(has_obs, lowest_obs, horizon_row)
        rows = np.arange(grid_h).reshape(-1, 1)
        cols = np.arange(grid_w).reshape(1, -1)
        cone_half_w = np.maximum(6.0, ((grid_h - 1 - rows) / float(grid_h)) * (grid_w * 0.48) + 8.0)
        in_cone = np.abs(cols - (grid_w * 0.5)) <= cone_half_w
        bev_grid[(rows > stop_row[np.newaxis, :]) & in_cone] = FREE
        bev_grid[occ_layer > 0] = OCCUPIED

        # --- Carve FREE Ray & Stamp Obstacle for TOF400C Laser Rangefinder ---
        if tof_distance_m is not None and tof_distance_m > 0.04:
            cx_cell = grid_w // 2
            max_y = local_range_y[1]
            hit_dist = float(tof_distance_m)
            hit_row = int(round(((max_y - min(hit_dist, max_y)) / max(max_y, 0.5)) * (grid_h - 1)))
            hit_row = int(np.clip(hit_row, 0, grid_h - 1))

            for r in range(min(grid_h - 1, hit_row + 2), grid_h):
                bev_grid[r, max(0, cx_cell - 2):min(grid_w, cx_cell + 3)] = FREE

            if hit_dist <= max_y:
                half_width_m = max(0.08, hit_dist * math.tan(math.radians(tof_beam_deg * 0.5)))
                half_cells = max(2, int(round((half_width_m / span_x) * grid_w)))
                c0 = max(0, cx_cell - half_cells)
                c1 = min(grid_w, cx_cell + half_cells + 1)
                bev_grid[max(0, hit_row - 1):min(grid_h, hit_row + 2), c0:c1] = OCCUPIED

        # --- Accumulate into Persistent Global 3D Voxel Point Cloud ---
        if accumulate_global and len(X_R) > 0:
            step_g = 3
            xr_s = X_R[::step_g]
            yr_s = Y_R[::step_g]
            zr_s = Z_R[::step_g]
            bgr_s = bgr_valid[::step_g]
            sem_s = sem[::step_g]

            rx, ry, rtheta = robot_pose
            cos_t = math.cos(rtheta)
            sin_t = math.sin(rtheta)

            XW = rx + yr_s * cos_t + xr_s * sin_t
            YW = ry + yr_s * sin_t - xr_s * cos_t
            new_xyz = np.column_stack([XW, YW, zr_s]).astype(np.float32)

            if len(self.global_xyz) == 0:
                comb_xyz = new_xyz
                comb_bgr = bgr_s
                comb_sem = sem_s
            else:
                comb_xyz = np.vstack([self.global_xyz, new_xyz])
                comb_bgr = np.vstack([self.global_bgr, bgr_s])
                comb_sem = np.concatenate([self.global_sem, sem_s])

            vox_coords = np.floor(comb_xyz / self.voxel_size).astype(np.int32)
            vox_pos = (vox_coords + 20000).astype(np.int64)
            keys = (
                (vox_pos[:, 0] & 0xFFFFF)
                | ((vox_pos[:, 1] & 0xFFFFF) << 20)
                | ((vox_pos[:, 2] & 0xFFFFF) << 40)
            )
            _, rev_idx = np.unique(keys[::-1], return_index=True)
            u_idx = (len(keys) - 1) - rev_idx
            if len(u_idx) > self.max_global_points:
                keep_idx = np.sort(u_idx)[-self.max_global_points:]
            else:
                keep_idx = u_idx

            self.global_xyz = comb_xyz[keep_idx]
            self.global_bgr = comb_bgr[keep_idx]
            self.global_sem = comb_sem[keep_idx]

        return bev_grid

    def render_3d_view(self, robot_pose, path=None, goal=None, out_w=640, out_h=480,
                       tof_distance_m=None):
        """
        Render an interactive 3D Perspective Point Cloud, TOF400C Laser Ray & Trajectory Viewport.
        """
        vis = np.full((out_h, out_w, 3), (18, 22, 30), dtype=np.uint8)

        rx, ry, rtheta = robot_pose
        cam_yaw = rtheta + math.radians(self.cam_yaw_offset_deg)
        cam_pitch = math.radians(self.cam_Pitch_deg)

        look_at = np.array([
            rx + 0.65 * math.cos(rtheta),
            ry + 0.65 * math.sin(rtheta),
            0.15
        ], dtype=np.float32)

        horiz_d = self.cam_dist_m * math.cos(cam_pitch)
        cam_pos = np.array([
            look_at[0] - horiz_d * math.cos(cam_yaw),
            look_at[1] - horiz_d * math.sin(cam_yaw),
            look_at[2] + self.cam_dist_m * math.sin(cam_pitch)
        ], dtype=np.float32)

        fwd = look_at - cam_pos
        fwd_norm = float(np.linalg.norm(fwd))
        if fwd_norm < 1e-4:
            return vis
        fwd /= fwd_norm

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(fwd, world_up)
        r_norm = float(np.linalg.norm(right))
        if r_norm < 1e-4:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= r_norm
        up = np.cross(right, fwd)

        f_px = out_w * 0.82
        cx_p = out_w * 0.50
        cy_p = out_h * 0.52

        def project_pts(pts_xyz):
            rel = pts_xyz - cam_pos
            x_c = rel @ right
            y_c = -(rel @ up)
            z_c = rel @ fwd
            valid_z = z_c > 0.15
            z_safe = np.where(valid_z, z_c, 1.0)
            u = (cx_p + (x_c / z_safe) * f_px).astype(np.int32)
            v = (cy_p + (y_c / z_safe) * f_px).astype(np.int32)
            in_scr = valid_z & (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
            return u, v, z_c, in_scr

        gx_min = int(math.floor(rx - 8.0))
        gx_max = int(math.ceil(rx + 8.0))
        gy_min = int(math.floor(ry - 8.0))
        gy_max = int(math.ceil(ry + 8.0))

        for xm in range(gx_min, gx_max + 1):
            ys = np.linspace(gy_min, gy_max, 40, dtype=np.float32)
            line_xyz = np.column_stack([np.full_like(ys, xm), ys, np.zeros_like(ys)])
            u, v, _, ok = project_pts(line_xyz)
            if np.sum(ok) >= 2:
                pts_2d = np.column_stack([u[ok], v[ok]]).reshape(-1, 1, 2)
                col = (55, 70, 90) if xm == 0 else (32, 38, 50)
                cv2.polylines(vis, [pts_2d], False, col, 1, cv2.LINE_AA)

        for ym in range(gy_min, gy_max + 1):
            xs = np.linspace(gx_min, gx_max, 40, dtype=np.float32)
            line_xyz = np.column_stack([xs, np.full_like(xs, ym), np.zeros_like(xs)])
            u, v, _, ok = project_pts(line_xyz)
            if np.sum(ok) >= 2:
                pts_2d = np.column_stack([u[ok], v[ok]]).reshape(-1, 1, 2)
                col = (55, 70, 90) if ym == 0 else (32, 38, 50)
                cv2.polylines(vis, [pts_2d], False, col, 1, cv2.LINE_AA)

        pts_list = []
        bgr_list = []
        sem_list = []

        if len(self.global_xyz) > 0:
            if not self.show_ceiling:
                keep = (self.global_sem != 2) & (self.global_xyz[:, 2] <= 1.25)
                if np.any(keep):
                    pts_list.append(self.global_xyz[keep])
                    bgr_list.append(self.global_bgr[keep])
                    sem_list.append(self.global_sem[keep])
            else:
                pts_list.append(self.global_xyz)
                bgr_list.append(self.global_bgr)
                sem_list.append(self.global_sem)

        # Always render live camera points in the 3D viewport (never drive into an empty void)
        if len(self.local_xyz) > 0:
            cos_t = math.cos(rtheta)
            sin_t = math.sin(rtheta)
            xl = self.local_xyz[:, 0]
            yl = self.local_xyz[:, 1]
            zl = self.local_xyz[:, 2]
            xw_loc = rx + yl * cos_t + xl * sin_t
            yw_loc = ry + yl * sin_t - xl * cos_t
            loc_world = np.column_stack([xw_loc, yw_loc, zl]).astype(np.float32)
            if not self.show_ceiling:
                keep_l = (self.local_sem != 2) & (zl <= 1.25)
                if np.any(keep_l):
                    pts_list.append(loc_world[keep_l])
                    bgr_list.append(self.local_bgr[keep_l])
                    sem_list.append(self.local_sem[keep_l])
            else:
                pts_list.append(loc_world)
                bgr_list.append(self.local_bgr)
                sem_list.append(self.local_sem)

        if len(pts_list) > 0:
            render_xyz = np.vstack(pts_list)
            render_bgr = np.vstack(bgr_list)
            render_sem = np.concatenate(sem_list)
            u, v, z_c, ok = project_pts(render_xyz)
            if np.any(ok):
                idx = np.flatnonzero(ok)
                order = idx[np.argsort(-z_c[idx])]
                u_o = u[order]
                v_o = v[order]

                if self.color_mode == 2:
                    # Mode 2: Metric Distance / Depth Heatmap (TURBO colormap: Near=Red -> Far=Blue)
                    p_xyz = render_xyz[order]
                    dists = np.hypot(p_xyz[:, 0] - rx, p_xyz[:, 1] - ry)
                    d_norm = np.clip((dists - 0.25) / 3.25, 0.0, 1.0)
                    d_u8 = (255.0 * (1.0 - d_norm)).astype(np.uint8)
                    colors = cv2.applyColorMap(d_u8.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
                elif self.color_mode == 1 or self.color_by_height:
                    # Mode 1: Height / Obstacle Elevation (Green Floor -> Red Obstacle)
                    sem_o = render_sem[order]
                    z_o = render_xyz[order, 2]
                    colors = np.zeros((len(order), 3), dtype=np.uint8)
                    colors[sem_o == 0] = (95, 165, 95)
                    obs_m = sem_o == 1
                    if np.any(obs_m):
                        t_h = np.clip((z_o[obs_m] - self.min_obs_z) / max(self.max_obs_z - self.min_obs_z, 0.05), 0.0, 1.0)
                        colors[obs_m, 0] = (30 * (1.0 - t_h)).astype(np.uint8)
                        colors[obs_m, 1] = (180 * (1.0 - t_h) + 40 * t_h).astype(np.uint8)
                        colors[obs_m, 2] = 255
                    colors[sem_o == 2] = (230, 190, 50)
                else:
                    # Mode 0: Photorealistic Camera RGB Texture
                    colors = render_bgr[order]

                vis[v_o, u_o] = colors
                u_p1 = np.minimum(u_o + 1, out_w - 1)
                v_p1 = np.minimum(v_o + 1, out_h - 1)
                vis[v_o, u_p1] = colors
                vis[v_p1, u_o] = colors

        # Draw 3D TOF400C Laser Ray in 3D View
        if tof_distance_m is not None and tof_distance_m > 0.03:
            d_tof = min(float(tof_distance_m), 4.0)
            ray_xyz = np.array([
                [rx, ry, 0.14],
                [rx + d_tof * math.cos(rtheta), ry + d_tof * math.sin(rtheta), 0.14]
            ], dtype=np.float32)
            ur, vr, _, okr = project_pts(ray_xyz)
            if np.all(okr):
                cv2.line(vis, (int(ur[0]), int(vr[0])), (int(ur[1]), int(vr[1])), (255, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(vis, (int(ur[1]), int(vr[1])), 5, (255, 80, 255), -1, cv2.LINE_AA)

        if path is not None and len(path) > 1:
            path_arr = np.asarray(path, dtype=np.float32)
            path_xyz = np.column_stack([path_arr[:, 0], path_arr[:, 1], np.full(len(path_arr), 0.02, dtype=np.float32)])
            u, v, _, ok = project_pts(path_xyz)
            if np.sum(ok) >= 2:
                pts_2d = np.column_stack([u[ok], v[ok]]).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts_2d], False, (0, 140, 255), 4, cv2.LINE_AA)
                cv2.polylines(vis, [pts_2d], False, (0, 245, 255), 2, cv2.LINE_AA)

        if goal is not None:
            gx, gy = goal
            pillar = np.array([[gx, gy, 0.0], [gx, gy, 0.35]], dtype=np.float32)
            u, v, _, ok = project_pts(pillar)
            if np.all(ok):
                cv2.line(vis, (int(u[0]), int(v[0])), (int(u[1]), int(v[1])), (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(vis, (int(u[1]), int(v[1])), 6, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(vis, (int(u[0]), int(v[0])), 8, (255, 255, 255), 1, cv2.LINE_AA)

        ang = np.linspace(0, 2 * math.pi, 20, dtype=np.float32)
        r_rad = 0.16
        ring_base = np.column_stack([rx + r_rad * np.cos(ang), ry + r_rad * np.sin(ang), np.zeros_like(ang)])
        ring_top  = np.column_stack([rx + r_rad * np.cos(ang), ry + r_rad * np.sin(ang), np.full_like(ang, 0.14)])
        ub, vb, _, okb = project_pts(ring_base)
        ut, vt, _, okt = project_pts(ring_top)
        if np.sum(okb) >= 3:
            cv2.polylines(vis, [np.column_stack([ub[okb], vb[okb]]).reshape(-1, 1, 2)], True, (255, 190, 20), 2, cv2.LINE_AA)
        if np.sum(okt) >= 3:
            cv2.polylines(vis, [np.column_stack([ut[okt], vt[okt]]).reshape(-1, 1, 2)], True, (255, 240, 120), 2, cv2.LINE_AA)

        arrow_xyz = np.array([
            [rx, ry, 0.14],
            [rx + 0.35 * math.cos(rtheta), ry + 0.35 * math.sin(rtheta), 0.14]
        ], dtype=np.float32)
        ua, va, _, oka = project_pts(arrow_xyz)
        if np.all(oka):
            cv2.arrowedLine(vis, (int(ua[0]), int(va[0])), (int(ua[1]), int(va[1])),
                            (0, 60, 255), 2, tipLength=0.3, line_type=cv2.LINE_AA)

        ceil_state = "Ceiling:ON" if self.show_ceiling else "Ceiling:OFF('H')"
        cv2.putText(
            vis,
            f"3D: Green=Floor | Red=Obs | Pink=TOF | {ceil_state} | 'M':Color | 'H':Ceiling | J/L:Orbit",
            (8, out_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (185, 200, 220), 1, cv2.LINE_AA
        )
        return vis
