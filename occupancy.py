"""
occupancy.py
Real-time top-down binary occupancy grid for mobile-robot route planning.

Uses Inverse Perspective Mapping (IPM) + TOF400C Laser Raycasting to warp the
forward-facing camera view and laser beam into a bird's-eye-view (BEV) grid.
"""

import math
import cv2
import numpy as np

# ── Grid cell states ─────────────────────────────────────────
UNKNOWN  = 128
FREE     = 0
OCCUPIED = 255

# ── Default grid size ────────────────────────────────────────
GRID_W = 64   # cells wide  (lateral: -1.0m to +1.0m by default)
GRID_H = 80   # cells tall  (forward range: 0.0m to 2.5m by default)

# ── Tunable Non-COCO Wall/Obstacle Sensitivity ───────────────
WALL_GRAD_THRESH = 28.0
WALL_DIFF_THRESH = 16.0

_WARP_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def default_ipm(img_w=640, img_h=480, grid_w=GRID_W, grid_h=GRID_H):
    src = np.float32([
        [int(img_w * 0.10), img_h],
        [int(img_w * 0.90), img_h],
        [int(img_w * 0.38), int(img_h * 0.55)],
        [int(img_w * 0.62), int(img_h * 0.55)],
    ])
    dst = np.float32([
        [0,          grid_h],
        [grid_w,     grid_h],
        [0,          0],
        [grid_w,     0],
    ])
    return cv2.getPerspectiveTransform(src, dst)


def dynamic_ipm(img_w=640, img_h=480, grid_w=GRID_W, grid_h=GRID_H,
                pitch_rad=0.0, nominal_pitch_rad=0.0, fy=530.0):
    dpitch = pitch_rad - nominal_pitch_rad
    dv_px = int(round(fy * math.tan(dpitch)))
    # Horizon shifts down when pitched up (negative pitch), and up when pitched down (positive pitch)
    base_horizon = int(img_h * (0.55 + 0.40 * math.sin(-nominal_pitch_rad)))
    top_y = int(np.clip(base_horizon - dv_px, int(img_h * 0.25), int(img_h * 0.90)))

    src = np.float32([
        [int(img_w * 0.10), img_h],
        [int(img_w * 0.90), img_h],
        [int(img_w * 0.38), top_y],
        [int(img_w * 0.62), top_y],
    ])
    dst = np.float32([
        [0,      grid_h],
        [grid_w, grid_h],
        [0,      0],
        [grid_w, 0],
    ])
    return cv2.getPerspectiveTransform(src, dst)


def calibrate_ipm(floor_pixels, floor_grid_coords):
    src = np.float32(floor_pixels)
    dst = np.float32(floor_grid_coords)
    return cv2.getPerspectiveTransform(src, dst)


def _detect_non_coco_obstacles(gray_frame, H_ipm, grid_w, grid_h, wall_sensitivity=1.0):
    if gray_frame is None or gray_frame.size == 0 or wall_sensitivity <= 0.05:
        return np.zeros((grid_h, grid_w), dtype=np.uint8)

    gray = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY) if gray_frame.ndim == 3 else gray_frame
    bev_gray = cv2.warpPerspective(gray, H_ipm, (grid_w, grid_h), flags=cv2.INTER_LINEAR)
    valid_bev = cv2.warpPerspective(
        np.full_like(gray, 255, dtype=np.uint8), H_ipm, (grid_w, grid_h), flags=cv2.INTER_NEAREST
    )

    ref_patch = bev_gray[max(0, grid_h - 16):grid_h - 2, max(0, grid_w // 2 - 12):min(grid_w, grid_w // 2 + 12)]
    if ref_patch.size == 0:
        return np.zeros((grid_h, grid_w), dtype=np.uint8)

    ref_med = float(np.median(ref_patch))
    ref_std = max(float(np.std(ref_patch)), 12.0)

    grad_y = np.abs(cv2.Sobel(bev_gray, cv2.CV_16S, 0, 1, ksize=3)).astype(np.float32)
    diff = np.abs(bev_gray.astype(np.float32) - ref_med)

    inv_sens = 1.0 / max(float(wall_sensitivity), 0.25)
    grad_thresh = WALL_GRAD_THRESH * inv_sens
    diff_thresh = max(WALL_DIFF_THRESH * inv_sens, (2.8 * inv_sens) * ref_std)

    non_floor = (valid_bev > 0) & (grad_y > grad_thresh) & (diff > diff_thresh)
    # The top 45% of IPM grid corresponds to distant horizon/infinity; never mark as close-range obstacle
    non_floor[:int(grid_h * 0.45), :] = False
    non_floor[grid_h - 4:, :] = False
    return (non_floor.astype(np.uint8) * 255)


def compute(yolo_combined_mask, orb_points, frame_shape, H_ipm,
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
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, _WARP_CLOSE_KERNEL)

    h_img, w_img = frame_shape[:2]
    fov_mask = cv2.warpPerspective(
        np.full((h_img, w_img), 255, dtype=np.uint8), H_ipm,
        (grid_w, grid_h), flags=cv2.INTER_NEAREST
    )

    has_obs = np.any(occ > 0, axis=0)
    lowest_obs = (grid_h - 1) - np.argmax(occ[::-1, :] > 0, axis=0)
    horizon = int(grid_h * 0.40)
    stop_row = np.where(has_obs, lowest_obs, horizon)

    rows = np.arange(grid_h).reshape(-1, 1)
    free_sweep = (rows > stop_row[np.newaxis, :]) & (fov_mask > 0)
    grid[free_sweep] = FREE

    if orb_points is not None and len(orb_points) > 0:
        pts_h = np.hstack([orb_points, np.ones((len(orb_points), 1), dtype=np.float32)])
        wp = (H_ipm @ pts_h.T).T
        w_div = wp[:, 2]

        in_front = w_div > 1e-3
        if np.any(in_front):
            wp_valid = wp[in_front, :2] / w_div[in_front, np.newaxis]
            gx = np.floor(wp_valid[:, 0]).astype(np.int32)
            gy = np.floor(wp_valid[:, 1]).astype(np.int32)

            in_bounds = (gx >= 0) & (gx < grid_w) & (gy >= 0) & (gy < grid_h)
            gx = gx[in_bounds]
            gy = gy[in_bounds]

            below_obs = (gy > stop_row[gx]) & (occ[gy, gx] == 0)
            grid[gy[below_obs], gx[below_obs]] = FREE

    grid[occ > 0] = OCCUPIED

    # --- Fuse TOF400C Laser Rangefinder Ray & Obstacle Hit (up to 4.0m) ---
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
            grid[r0:r1, c0:c1] = OCCUPIED

    return grid


_PAL_FREE     = (215, 235, 220)
_PAL_OCCUPIED = (45, 50, 225)
_PAL_UNKNOWN  = (26, 30, 38)


def render(grid, out_w=640, out_h=480, tof_distance_m=None, max_range_m=4.0):
    gh, gw = grid.shape
    vis = np.zeros((gh, gw, 3), dtype=np.uint8)

    vis[grid == UNKNOWN]  = _PAL_UNKNOWN
    vis[grid == FREE]     = _PAL_FREE
    vis[grid == OCCUPIED] = _PAL_OCCUPIED

    vis = cv2.resize(vis, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    cx = out_w // 2
    cy = out_h - 10

    px_per_m = (out_h - 20) / max(max_range_m, 0.5)
    for m in (1.0, 2.0, 3.0, 4.0):
        if m <= max_range_m:
            r_px = int(m * px_per_m)
            cv2.ellipse(vis, (cx, cy), (r_px, r_px), 0, 180, 360, (65, 80, 100), 1, cv2.LINE_AA)
            cv2.putText(vis, f"{m:.0f}m", (cx + 8, cy - r_px + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 160, 180), 1, cv2.LINE_AA)

    cv2.line(vis, (cx, cy), (cx, 28), (65, 80, 100), 1, cv2.LINE_AA)

    # Draw live TOF400C laser beam on the 2D radar
    if tof_distance_m is not None and tof_distance_m > 0.03:
        beam_m = min(float(tof_distance_m), max_range_m)
        hit_y = int(np.clip(cy - beam_m * px_per_m, 28, cy))
        cv2.line(vis, (cx, cy), (cx, hit_y), (255, 0, 255), 2, cv2.LINE_AA)
        if tof_distance_m <= max_range_m:
            cv2.circle(vis, (cx, hit_y), 6, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(vis, f"ToF {tof_distance_m:.2f}m", (cx + 10, max(40, hit_y + 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 100, 255), 1, cv2.LINE_AA)

    for lat_m in (-0.5, 0.5):
        lx = int(cx + lat_m * px_per_m)
        if 0 <= lx < out_w:
            cv2.line(vis, (lx, cy), (lx, 28), (40, 48, 60), 1, cv2.LINE_AA)

    r_radius = max(8, int(0.15 * px_per_m))
    cv2.circle(vis, (cx, cy), r_radius, (255, 190, 20), -1, cv2.LINE_AA)
    cv2.circle(vis, (cx, cy), r_radius, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.arrowedLine(vis, (cx, cy), (cx, cy - r_radius - 12), (0, 50, 230), 2,
                    tipLength=0.4, line_type=cv2.LINE_AA)

    cv2.rectangle(vis, (0, 0), (out_w - 1, out_h - 1), (60, 75, 95), 1)
    return vis
