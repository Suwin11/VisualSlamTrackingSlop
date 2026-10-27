"""
rrt_star.py
Vectorized RRT* Path Planner with Cubic Bézier Curve Smoothing, Path Stability
Validation, and Closed-Loop Pure Pursuit Trajectory Controller.
"""

import json
import math
import random
import socket
import time
import numpy as np


class Node:
    """RRT* tree node with child tracking for recursive cost propagation."""
    __slots__ = ("x", "y", "cost", "parent", "children")

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.cost = 0.0
        self.parent = None
        self.children = []


class RRTStarPlanner:
    """
    Vectorized RRT* Planner with Rewiring, Path Hysteresis, and Cubic Bézier Smoothing.
    """

    def __init__(self, occ_grid, step_size=0.25, search_radius=0.55,
                 goal_sample_rate=0.12, max_iter=300):
        self.grid = occ_grid
        self.step_size = float(step_size)
        self.search_radius = float(search_radius)
        self.goal_sample_rate = float(goal_sample_rate)
        self.max_iter = int(max_iter)

    def _update_bounds(self):
        self.x_min = self.grid.origin_x
        self.x_max = self.grid.origin_x + self.grid.width_m
        self.y_min = self.grid.origin_y
        self.y_max = self.grid.origin_y + self.grid.height_m

    @staticmethod
    def _propagate_cost_delta(node, delta):
        stack = list(node.children)
        while stack:
            child = stack.pop()
            child.cost += delta
            stack.extend(child.children)

    def is_path_valid(self, robot_pose, smooth_path, goal_pos, max_cross_track_m=0.35):
        """
        Check whether an existing smoothed path is still collision-free and close to the
        robot and goal, avoiding unnecessary stochastic RRT* replanning jitter.
        """
        if not smooth_path or len(smooth_path) < 2:
            return False

        rx, ry = robot_pose[0], robot_pose[1]
        gx, gy = goal_pos

        # 1. Goal must match path end
        if math.hypot(smooth_path[-1][0] - gx, smooth_path[-1][1] - gy) > 0.25:
            return False

        # 2. Robot must be within max_cross_track_m of the path
        pts = np.asarray(smooth_path, dtype=np.float64)
        dists = np.hypot(pts[:, 0] - rx, pts[:, 1] - ry)
        closest_idx = int(np.argmin(dists))
        if dists[closest_idx] > max_cross_track_m:
            return False

        # 3. Remaining path ahead of the robot must be collision-free
        for i in range(closest_idx, len(smooth_path) - 1):
            p0 = smooth_path[i]
            p1 = smooth_path[i + 1]
            if not self.grid.is_line_free(p0[0], p0[1], p1[0], p1[1]):
                return False
        return True

    def plan(self, start_pos, goal_pos, seed_waypoints=None):
        """
        Compute an optimal collision-free path from start to goal using vectorized
        NumPy nearest-neighbor and radius searches (~10x faster than Python loops).
        """
        sx, sy = float(start_pos[0]), float(start_pos[1])
        gx, gy = float(goal_pos[0]), float(goal_pos[1])
        if hasattr(self.grid, "ensure_bounds_include"):
            self.grid.ensure_bounds_include(sx, sy, margin_m=2.0)
            self.grid.ensure_bounds_include(gx, gy, margin_m=2.0)
        self._update_bounds()

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

        if math.hypot(gx - sx, gy - sy) <= self.step_size and self.grid.is_line_free(sx, sy, gx, gy):
            raw_path = [(sx, sy), (gx, gy)]
            return raw_path, self._interpolate_linear(raw_path, 15), [((sx, sy), (gx, gy))]

        start_node = Node(sx, sy)
        nodes = [start_node]

        # Pre-allocated contiguous coordinate buffer for fast vectorized Euclidean queries
        cap = self.max_iter + 32
        coords = np.empty((cap, 2), dtype=np.float64)
        coords[0, 0] = sx
        coords[0, 1] = sy
        n_nodes = 1

        best_goal_node = None
        min_goal_cost = float("inf")
        r_sq = self.search_radius * self.search_radius

        for it in range(self.max_iter):
            # 1. Sample random point (with goal bias and optional seed bias)
            if seed_waypoints and it < len(seed_waypoints) and random.random() < 0.5:
                rnd_x, rnd_y = seed_waypoints[it]
            elif random.random() < self.goal_sample_rate:
                rnd_x, rnd_y = gx, gy
            else:
                rnd_x = random.uniform(self.x_min, self.x_max)
                rnd_y = random.uniform(self.y_min, self.y_max)

            # 2. Vectorized Nearest Neighbor
            dx_arr = coords[:n_nodes, 0] - rnd_x
            dy_arr = coords[:n_nodes, 1] - rnd_y
            dist_sq_arr = dx_arr * dx_arr + dy_arr * dy_arr
            nearest_idx = int(np.argmin(dist_sq_arr))
            nearest = nodes[nearest_idx]

            # 3. Steer towards (rnd_x, rnd_y)
            dist = math.sqrt(float(dist_sq_arr[nearest_idx]))
            if dist < 1e-6:
                continue
            d = min(self.step_size, dist)
            inv_d = d / dist
            new_x = nearest.x + (rnd_x - nearest.x) * inv_d
            new_y = nearest.y + (rnd_y - nearest.y) * inv_d

            # 4. Collision check
            if not self.grid.is_line_free(nearest.x, nearest.y, new_x, new_y):
                continue

            new_node = Node(new_x, new_y)
            new_node.cost = nearest.cost + d
            new_node.parent = nearest

            # 5. Vectorized RRT* Neighbor Search & Parent Selection
            nb_dx = coords[:n_nodes, 0] - new_x
            nb_dy = coords[:n_nodes, 1] - new_y
            nb_dist_sq = nb_dx * nb_dx + nb_dy * nb_dy
            nb_indices = np.flatnonzero(nb_dist_sq <= r_sq)
            nb_dists = np.sqrt(nb_dist_sq[nb_indices])

            min_cost = new_node.cost
            best_parent = nearest

            for idx_pos, nb_idx in enumerate(nb_indices):
                nb = nodes[int(nb_idx)]
                d_nb = float(nb_dists[idx_pos])
                cand_cost = nb.cost + d_nb
                if cand_cost < min_cost:
                    if self.grid.is_line_free(nb.x, nb.y, new_x, new_y):
                        min_cost = cand_cost
                        best_parent = nb

            new_node.cost = min_cost
            new_node.parent = best_parent
            best_parent.children.append(new_node)

            nodes.append(new_node)
            if n_nodes >= len(coords):
                coords = np.vstack([coords, np.empty((128, 2), dtype=np.float64)])
            coords[n_nodes, 0] = new_x
            coords[n_nodes, 1] = new_y
            n_nodes += 1

            # 6. RRT* Rewire Neighbors (with recursive descendant cost propagation)
            for idx_pos, nb_idx in enumerate(nb_indices):
                nb = nodes[int(nb_idx)]
                if nb is best_parent:
                    continue
                d_nb = float(nb_dists[idx_pos])
                new_nb_cost = new_node.cost + d_nb
                if new_nb_cost < nb.cost:
                    if self.grid.is_line_free(new_x, new_y, nb.x, nb.y):
                        if nb.parent is not None and nb in nb.parent.children:
                            nb.parent.children.remove(nb)
                        nb.parent = new_node
                        new_node.children.append(nb)
                        cost_delta = new_nb_cost - nb.cost
                        nb.cost = new_nb_cost
                        self._propagate_cost_delta(nb, cost_delta)

            # 7. Check if goal reached
            d_to_goal = math.hypot(new_x - gx, new_y - gy)
            if d_to_goal <= self.step_size:
                if self.grid.is_line_free(new_x, new_y, gx, gy):
                    cand_goal_cost = new_node.cost + d_to_goal
                    if cand_goal_cost < min_goal_cost:
                        min_goal_cost = cand_goal_cost
                        best_goal_node = Node(gx, gy)
                        best_goal_node.parent = new_node
                        best_goal_node.cost = cand_goal_cost

        tree_edges = [((n.parent.x, n.parent.y), (n.x, n.y)) for n in nodes if n.parent is not None]

        if best_goal_node is None:
            g_dx = coords[:n_nodes, 0] - gx
            g_dy = coords[:n_nodes, 1] - gy
            g_dists = np.hypot(g_dx, g_dy)
            closest_idx = int(np.argmin(g_dists))
            closest = nodes[closest_idx]
            if (float(g_dists[closest_idx]) < self.step_size * 2.0
                    and self.grid.is_line_free(closest.x, closest.y, gx, gy)):
                best_goal_node = Node(gx, gy)
                best_goal_node.parent = closest
                best_goal_node.cost = closest.cost + float(g_dists[closest_idx])
            else:
                return None, None, tree_edges

        raw_path = []
        curr = best_goal_node
        visited = set()
        while curr is not None and id(curr) not in visited:
            visited.add(id(curr))
            raw_path.append((curr.x, curr.y))
            curr = curr.parent
        raw_path.reverse()

        smooth_path = self.smooth_bezier(raw_path)
        return raw_path, smooth_path, tree_edges

    def _find_nearest_free(self, x, y, search_dist=0.6, step=0.05):
        r_steps = int(search_dist / step)
        for r in range(1, r_steps + 1):
            radius = r * step
            for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                cand_x = x + radius * math.cos(angle)
                cand_y = y + radius * math.sin(angle)
                if self.grid.is_collision_free(cand_x, cand_y):
                    return cand_x, cand_y
        return None

    def _is_curve_safe(self, seg):
        if not seg:
            return False
        for i in range(len(seg) - 1):
            if not self.grid.is_line_free(seg[i][0], seg[i][1], seg[i + 1][0], seg[i + 1][1]):
                return False
        return True

    @staticmethod
    def _subdivide_long_segments(waypoints, max_seg_len=0.60):
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

    def smooth_bezier(self, waypoints, points_per_seg=15):
        if len(waypoints) < 3:
            return self._interpolate_linear(waypoints, points_per_seg)

        pruned = [waypoints[0]]
        for i in range(1, len(waypoints) - 1):
            prev = pruned[-1]
            curr = waypoints[i]
            nxt = waypoints[i + 1]
            if self.grid.is_line_free(prev[0], prev[1], nxt[0], nxt[1]):
                continue
            pruned.append(curr)
        pruned.append(waypoints[-1])

        pruned = self._subdivide_long_segments(pruned, max_seg_len=0.60)

        if len(pruned) < 3:
            return self._interpolate_linear(pruned, points_per_seg * 2)

        smooth_pts = []
        n = len(pruned)
        midpoints = [((pruned[i][0] + pruned[i + 1][0]) * 0.5,
                      (pruned[i][1] + pruned[i + 1][1]) * 0.5) for i in range(n - 1)]

        first_seg = self._cubic_bezier_segment(
            pruned[0],
            (pruned[0][0] + 0.5 * (pruned[1][0] - pruned[0][0]),
             pruned[0][1] + 0.5 * (pruned[1][1] - pruned[0][1])),
            (pruned[1][0] - 0.25 * (pruned[1][0] - pruned[0][0]),
             pruned[1][1] - 0.25 * (pruned[1][1] - pruned[0][1])),
            midpoints[0],
            points_per_seg
        )
        if self._is_curve_safe(first_seg):
            smooth_pts.extend(first_seg)
        else:
            smooth_pts.extend(self._interpolate_linear([pruned[0], midpoints[0]], points_per_seg))

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
    def _cubic_bezier_segment(p0, p1, p2, p3, steps=15):
        t = np.linspace(0.0, 1.0, steps).reshape(-1, 1)
        b0 = (1.0 - t) ** 3
        b1 = 3.0 * (1.0 - t) ** 2 * t
        b2 = 3.0 * (1.0 - t) * (t ** 2)
        b3 = t ** 3
        P = np.array([p0, p1, p2, p3], dtype=np.float64)
        curve = b0 * P[0] + b1 * P[1] + b2 * P[2] + b3 * P[3]
        return [(float(pt[0]), float(pt[1])) for pt in curve]

    @staticmethod
    def _interpolate_linear(pts, steps=20):
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
