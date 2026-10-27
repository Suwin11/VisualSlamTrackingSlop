"""
astar.py
Deterministic 2D Grid A* Path Planner with Distance-Transform Corridor Centering,
Diagonal Corner-Cutting Prevention, Greedy Line-of-Sight String-Pulling,
and Cubic Bézier Trajectory Smoothing.

Designed as a drop-in replacement/alternative to RRT* in `navigate_rrt.py` for
guaranteed optimal shortest paths in narrow corridors and doorways.
"""

import heapq
import math
import time
import numpy as np
import cv2

from global_map import GlobalOccupancyGrid, CELL_OCCUPIED, CELL_FREE


class AStarPlanner:
    """
    Deterministic 2D A* Planner operating on `GlobalOccupancyGrid.inflated_grid`.

    Features:
    - 8-connected grid search with admissible Euclidean heuristic.
    - Diagonal corner-cutting prevention (blocks diagonal moves if both orthogonal neighbors are obstacles).
    - Distance-transform clearance penalty: softly centers the path in corridors and doors rather than hugging wall edges.
    - Greedy line-of-sight shortcutting (string-pulling raycasting) to eliminate grid staircases.
    - Adaptive cubic Bézier curve smoothing for continuous velocity tracking with Pure Pursuit.
    - Exploration edge extraction for real-time visualization on the navigation HUD.
    """

    # 8-connected direction deltas: (dx, dy, step_cost)
    _SQRT2 = math.sqrt(2.0)
    _DIRECTIONS = (
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, _SQRT2),
        (1, -1, _SQRT2),
        (-1, 1, _SQRT2),
        (-1, -1, _SQRT2),
    )

    def __init__(self, occ_grid: GlobalOccupancyGrid,
                 clearance_weight: float = 0.20,
                 corner_cut_prevention: bool = True,
                 max_iterations: int = 40000):
        self.grid = occ_grid
        self.clearance_weight = float(clearance_weight)
        self.corner_cut_prevention = bool(corner_cut_prevention)
        self.max_iterations = int(max_iterations)

    def is_path_valid(self, robot_pose, smooth_path, goal_pos, max_cross_track_m: float = 0.35) -> bool:
        """
        Check whether an existing smoothed path is still collision-free and close to the
        robot and goal, avoiding unnecessary re-planning when tracking the desired trajectory.
        """
        if not smooth_path or len(smooth_path) < 2:
            return False

        rx, ry = robot_pose[0], robot_pose[1]
        gx, gy = goal_pos

        # 1. Goal must still match path terminus
        if math.hypot(smooth_path[-1][0] - gx, smooth_path[-1][1] - gy) > 0.25:
            return False

        # 2. Robot must be within max_cross_track_m of the path
        pts = np.asarray(smooth_path, dtype=np.float64)
        dists = np.hypot(pts[:, 0] - rx, pts[:, 1] - ry)
        closest_idx = int(np.argmin(dists))
        if dists[closest_idx] > max_cross_track_m:
            return False

        # 3. Remaining path ahead of the robot must remain collision-free
        for i in range(closest_idx, len(smooth_path) - 1):
            p0 = smooth_path[i]
            p1 = smooth_path[i + 1]
            if not self.grid.is_line_free(p0[0], p0[1], p1[0], p1[1]):
                return False
        return True

    def _find_nearest_free(self, x: float, y: float, search_dist: float = 0.6, step: float = 0.05):
        """Find the closest collision-free world coordinate if start or goal is slightly inside an inflated cell."""
        r_steps = int(search_dist / step)
        for r in range(1, r_steps + 1):
            radius = r * step
            for angle in np.linspace(0, 2.0 * math.pi, 16, endpoint=False):
                cand_x = x + radius * math.cos(angle)
                cand_y = y + radius * math.sin(angle)
                if self.grid.is_collision_free(cand_x, cand_y):
                    return cand_x, cand_y
        return None

    def plan(self, start_pos, goal_pos, seed_waypoints=None):
        """
        Compute an optimal collision-free path from start_pos to goal_pos.

        Returns:
            (raw_path, smooth_path, tree_edges)
            raw_path: List of (x, y) pruned line-of-sight waypoints in meters.
            smooth_path: Dense list of (x, y) cubic Bézier smoothed waypoints for Pure Pursuit.
            tree_edges: Sampled parent-child search edges for HUD display [((x1, y1), (x2, y2)), ...].
        """
        sx, sy = float(start_pos[0]), float(start_pos[1])
        gx, gy = float(goal_pos[0]), float(goal_pos[1])

        # Ensure grid canvas dynamically encompasses both start and goal
        self.grid.ensure_bounds_include(sx, sy, margin_m=2.0)
        self.grid.ensure_bounds_include(gx, gy, margin_m=2.0)

        # 1. Validate / snap start and goal to walkable space
        if not self.grid.is_collision_free(sx, sy):
            free_start = self._find_nearest_free(sx, sy)
            if free_start is None:
                return None, None, []
            sx, sy = free_start

        if not self.grid.is_collision_free(gx, gy):
            free_goal = self._find_nearest_free(gx, gy)
            if free_goal is None:
                return None, None, []
            gx, gy = free_goal

        # Trivial line-of-sight case: straight path is immediately clear
        if self.grid.is_line_free(sx, sy, gx, gy):
            raw_path = [(sx, sy), (gx, gy)]
            smooth_path = self._interpolate_linear(raw_path, steps=25)
            return raw_path, smooth_path, [((sx, sy), (gx, gy))]

        # 2. Map coordinates to grid cells
        start_cell = self.grid.world_to_grid(sx, sy)
        goal_cell = self.grid.world_to_grid(gx, gy)
        if start_cell is None or goal_cell is None:
            return None, None, []

        start_gx, start_gy = start_cell
        goal_gx, goal_gy = goal_cell

        rows = self.grid.rows
        cols = self.grid.cols
        res = self.grid.resolution

        # 3. Build unwalkable mask
        unwalkable = (self.grid.inflated_grid > 0) | (self.grid.binary_grid == CELL_OCCUPIED)
        if unwalkable[start_gy, start_gx]:
            unwalkable[start_gy, start_gx] = False
        if unwalkable[goal_gy, goal_gx]:
            unwalkable[goal_gy, goal_gx] = False

        # 4. Compute Distance-Transform Clearance Map (soft corridor centering)
        if self.clearance_weight > 1e-4:
            walkable_u8 = (~unwalkable).astype(np.uint8)
            dist_map = cv2.distanceTransform(walkable_u8, cv2.DIST_L2, 5) * res
            # Invert distance so cells close to walls have higher cost penalty:
            # At d = 0.05m -> penalty ~ clearance_weight / 0.10
            # At d = 0.50m -> penalty ~ clearance_weight / 0.55
            clearance_penalty = (self.clearance_weight / (dist_map + 0.05)).astype(np.float32)
        else:
            clearance_penalty = None

        # 5. A* Search Data Structures (Contiguous NumPy buffers for max performance)
        g_cost = np.full((rows, cols), np.inf, dtype=np.float32)
        parent_x = np.full((rows, cols), -1, dtype=np.int32)
        parent_y = np.full((rows, cols), -1, dtype=np.int32)
        closed = np.zeros((rows, cols), dtype=bool)

        g_cost[start_gy, start_gx] = 0.0
        h_start = math.hypot(start_gx - goal_gx, start_gy - goal_gy) * res

        # Priority queue entries: (f_score, g_cost, gx, gy)
        open_heap = [(h_start, 0.0, start_gx, start_gy)]

        tree_edges = []
        found_goal = False
        iterations = 0
        max_iter = self.max_iterations

        corner_check = self.corner_cut_prevention

        while open_heap and iterations < max_iter:
            iterations += 1
            f_curr, g_curr, cx, cy = heapq.heappop(open_heap)

            if closed[cy, cx]:
                continue
            closed[cy, cx] = True

            # Record sampled exploration edges for HUD visualization (every 8th node, capped at 300)
            if len(tree_edges) < 300 and (iterations % 8 == 0):
                px, py = parent_x[cy, cx], parent_y[cy, cx]
                if px >= 0 and py >= 0:
                    pw = self.grid.grid_to_world(px, py)
                    cw = self.grid.grid_to_world(cx, cy)
                    tree_edges.append((pw, cw))

            # Goal check
            if cx == goal_gx and cy == goal_gy:
                found_goal = True
                break

            # 8-Connected Neighbor Expansion
            for dx, dy, step_dist in self._DIRECTIONS:
                nx = cx + dx
                ny = cy + dy

                if not (0 <= nx < cols and 0 <= ny < rows):
                    continue
                if closed[ny, nx] or unwalkable[ny, nx]:
                    continue

                # Diagonal corner-cutting check: do not squeeze between two touching diagonal obstacles
                if corner_check and dx != 0 and dy != 0:
                    if unwalkable[cy, nx] and unwalkable[ny, cx]:
                        continue

                # Base distance cost + optional clearance penalty
                cost_increment = step_dist * res
                if clearance_penalty is not None:
                    cost_increment += clearance_penalty[ny, nx] * (step_dist * res)

                cand_g = g_curr + cost_increment
                if cand_g < g_cost[ny, nx]:
                    g_cost[ny, nx] = cand_g
                    parent_x[ny, nx] = cx
                    parent_y[ny, nx] = cy
                    # Admissible Euclidean heuristic
                    h_cand = math.hypot(nx - goal_gx, ny - goal_gy) * res
                    heapq.heappush(open_heap, (cand_g + h_cand, cand_g, nx, ny))

        if not found_goal:
            # Fallback: check if we reached within 0.25m of goal with clear line-of-sight
            d_to_goal = math.hypot(cx - goal_gx, cy - goal_gy) * res
            if d_to_goal <= 0.40:
                cw = self.grid.grid_to_world(cx, cy)
                if self.grid.is_line_free(cw[0], cw[1], gx, gy):
                    parent_x[goal_gy, goal_gx] = cx
                    parent_y[goal_gy, goal_gx] = cy
                    found_goal = True

        if not found_goal:
            return None, None, tree_edges

        # 6. Backtrack Path
        grid_path = []
        curr_x, curr_y = goal_gx, goal_gy
        visited = set()
        while curr_x >= 0 and curr_y >= 0:
            pt = (curr_x, curr_y)
            if pt in visited:
                break
            visited.add(pt)
            grid_path.append(pt)
            px = parent_x[curr_y, curr_x]
            py = parent_y[curr_y, curr_x]
            curr_x, curr_y = px, py

        grid_path.reverse()

        # Convert to metric world coordinates, ensuring exact start and goal endpoints
        world_pts = [(sx, sy)]
        for gx_node, gy_node in grid_path[1:-1]:
            world_pts.append(self.grid.grid_to_world(gx_node, gy_node))
        world_pts.append((gx, gy))

        # 7. Greedy Line-of-Sight String-Pulling Shortcut Pruner
        pruned_path = self._prune_line_of_sight(world_pts)

        # 8. Cubic Bézier Trajectory Smoothing
        smooth_path = self.smooth_bezier(pruned_path)

        return pruned_path, smooth_path, tree_edges

    def _prune_line_of_sight(self, waypoints: list) -> list:
        """
        Greedy raycast string-pulling shortcutting.
        Collapses staircased grid waypoints into minimal straight line segments.
        """
        if len(waypoints) < 3:
            return list(waypoints)

        pruned = [waypoints[0]]
        curr_idx = 0
        n = len(waypoints)

        while curr_idx < n - 1:
            furthest_idx = curr_idx + 1
            for cand_idx in range(n - 1, curr_idx, -1):
                p0 = pruned[-1]
                p1 = waypoints[cand_idx]
                if self.grid.is_line_free(p0[0], p0[1], p1[0], p1[1]):
                    furthest_idx = cand_idx
                    break
            pruned.append(waypoints[furthest_idx])
            curr_idx = furthest_idx

        return pruned

    def _is_curve_safe(self, seg) -> bool:
        """Verify that every segment of an interpolated/smoothed curve is collision free."""
        if not seg:
            return False
        for i in range(len(seg) - 1):
            if not self.grid.is_line_free(seg[i][0], seg[i][1], seg[i + 1][0], seg[i + 1][1]):
                return False
        return True

    @staticmethod
    def _subdivide_long_segments(waypoints, max_seg_len: float = 0.50):
        """Subdivide segments longer than max_seg_len so Bézier corners maintain uniform curvature."""
        if len(waypoints) < 2:
            return waypoints
        refined = [waypoints[0]]
        for i in range(len(waypoints) - 1):
            p0 = waypoints[i]
            p1 = waypoints[i + 1]
            dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if dist > max_seg_len:
                n_sub = int(math.ceil(dist / max_seg_len))
                for k in range(1, n_sub):
                    alpha = k / float(n_sub)
                    refined.append((p0[0] + alpha * (p1[0] - p0[0]),
                                    p0[1] + alpha * (p1[1] - p0[1])))
            refined.append(p1)
        return refined

    def smooth_bezier(self, waypoints, points_per_seg: int = 15):
        """
        Produce a continuous, smooth C1-continuous trajectory using cubic Bézier curves,
        with strict collision safety fallback to linear segments if a curve cuts an obstacle.
        """
        if len(waypoints) < 3:
            return self._interpolate_linear(waypoints, points_per_seg)

        # Ensure segments are moderately spaced
        pruned = self._subdivide_long_segments(waypoints, max_seg_len=0.50)
        if len(pruned) < 3:
            return self._interpolate_linear(pruned, points_per_seg * 2)

        smooth_pts = []
        n = len(pruned)
        midpoints = [((pruned[i][0] + pruned[i + 1][0]) * 0.5,
                      (pruned[i][1] + pruned[i + 1][1]) * 0.5) for i in range(n - 1)]

        # First segment: start to midpoint 0
        first_seg = self._cubic_bezier_segment(
            pruned[0],
            (pruned[0][0] + 0.5 * (pruned[1][0] - pruned[0][0]),
             pruned[0][1] + 0.5 * (pruned[1][0] - pruned[0][0])),
            (pruned[1][0] - 0.25 * (pruned[1][0] - pruned[0][0]),
             pruned[1][1] - 0.25 * (pruned[1][0] - pruned[0][0])),
            midpoints[0],
            points_per_seg
        )
        if self._is_curve_safe(first_seg):
            smooth_pts.extend(first_seg)
        else:
            smooth_pts.extend(self._interpolate_linear([pruned[0], midpoints[0]], points_per_seg))

        # Middle segments connecting midpoints around corners
        for i in range(len(midpoints) - 1):
            p0 = midpoints[i]
            p3 = midpoints[i + 1]
            corner = pruned[i + 1]

            p1 = (corner[0] - 0.4 * (corner[0] - p0[0]), corner[1] - 0.4 * (corner[1] - p0[1]))
            p2 = (corner[0] + 0.4 * (p3[0] - corner[0]), corner[1] + 0.4 * (p3[1] - corner[1]))

            seg = self._cubic_bezier_segment(p0, p1, p2, p3, points_per_seg)
            if self._is_curve_safe(seg):
                smooth_pts.extend(seg)
            else:
                smooth_pts.extend(self._interpolate_linear([p0, corner, p3], points_per_seg))

        # Last segment: midpoint -1 to goal
        last_seg = self._cubic_bezier_segment(
            midpoints[-1],
            (pruned[-1][0] - 0.5 * (pruned[-1][0] - pruned[-2][0]),
             pruned[-1][1] - 0.5 * (pruned[-1][1] - pruned[-2][1])),
            (pruned[-1][0] - 0.25 * (pruned[-1][0] - pruned[-2][0]),
             pruned[-1][1] - 0.25 * (pruned[-1][1] - pruned[-2][1])),
            pruned[-1],
            points_per_seg
        )
        if self._is_curve_safe(last_seg):
            smooth_pts.extend(last_seg)
        else:
            smooth_pts.extend(self._interpolate_linear([midpoints[-1], pruned[-1]], points_per_seg))

        return smooth_pts

    @staticmethod
    def _cubic_bezier_segment(p0, p1, p2, p3, steps: int = 15):
        t = np.linspace(0.0, 1.0, steps).reshape(-1, 1)
        b0 = (1.0 - t) ** 3
        b1 = 3.0 * (1.0 - t) ** 2 * t
        b2 = 3.0 * (1.0 - t) * (t ** 2)
        b3 = t ** 3
        p_arr = np.array([p0, p1, p2, p3], dtype=np.float64)
        curve = b0 * p_arr[0] + b1 * p_arr[1] + b2 * p_arr[2] + b3 * p_arr[3]
        return [(float(pt[0]), float(pt[1])) for pt in curve]

    @staticmethod
    def _interpolate_linear(pts, steps: int = 20):
        if len(pts) < 2:
            return list(pts)
        res = []
        seg_steps = max(2, steps // (len(pts) - 1))
        for i in range(len(pts) - 1):
            xs = np.linspace(pts[i][0], pts[i + 1][0], seg_steps, endpoint=(i == len(pts) - 2))
            ys = np.linspace(pts[i][1], pts[i + 1][1], seg_steps, endpoint=(i == len(pts) - 2))
            res.extend((float(x), float(y)) for x, y in zip(xs, ys))
        return res


from motor_control import MotorController, PurePursuitController
