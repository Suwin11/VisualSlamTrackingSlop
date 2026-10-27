"""
scratch/test_planner_comparison.py
Compare A* and RRT* on the same realistic occupancy grid.
"""

import sys
import os
import time
import math
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from global_map import GlobalOccupancyGrid, CELL_OCCUPIED, CELL_FREE
from astar import AStarPlanner
from rrt_star import RRTStarPlanner
import cv2


def compare_planners():
    print("=" * 65)
    print("COMPARISON: A* vs RRT* PATH PLANNERS")
    print("=" * 65)

    grid = GlobalOccupancyGrid(width_m=8.0, height_m=8.0, resolution=0.05, robot_radius_m=0.18)

    # Wall at Y = 1.0m with a narrow 0.50m doorway at X = -0.5m
    wall_y = 1.0
    door_x = -0.5
    door_width = 0.50

    for wx in np.arange(-3.5, 3.5, 0.04):
        if abs(wx - door_x) < (door_width / 2.0):
            continue  # Doorway
        c = grid.world_to_grid(wx, wall_y)
        if c:
            gx, gy = c
            grid.binary_grid[gy, gx] = CELL_OCCUPIED
            grid.log_odds[gy, gx] = 5.0

    occ_binary = (grid.binary_grid == CELL_OCCUPIED).astype(np.uint8) * 255
    grid.inflated_grid = cv2.dilate(occ_binary, grid.inflate_kernel)

    start = (0.5, 0.0)
    goal = (-0.5, 2.5)

    astar = AStarPlanner(grid, clearance_weight=0.20, corner_cut_prevention=True)
    rrt = RRTStarPlanner(grid, step_size=0.25, search_radius=0.55, max_iter=300)

    # 1. Benchmark A*
    t0 = time.perf_counter()
    a_raw, a_smooth, a_edges = astar.plan(start, goal)
    a_time = (time.perf_counter() - t0) * 1000.0

    # 2. Benchmark RRT*
    t0 = time.perf_counter()
    r_raw, r_smooth, r_edges = rrt.plan(start, goal)
    r_time = (time.perf_counter() - t0) * 1000.0

    print(f"A* Planner  : Found={a_smooth is not None} | Latency={a_time:.2f}ms | RawPts={len(a_raw) if a_raw else 0} | SmoothPts={len(a_smooth) if a_smooth else 0}")
    print(f"RRT* Planner: Found={r_smooth is not None} | Latency={r_time:.2f}ms | RawPts={len(r_raw) if r_raw else 0} | SmoothPts={len(r_smooth) if r_smooth else 0}")

    if a_smooth:
        # Calculate path length
        pts = np.array(a_smooth)
        diffs = np.diff(pts, axis=0)
        a_length = np.sum(np.hypot(diffs[:, 0], diffs[:, 1]))
        print(f"A* Smooth Path Length: {a_length:.3f} m")

    if r_smooth:
        pts = np.array(r_smooth)
        diffs = np.diff(pts, axis=0)
        r_length = np.sum(np.hypot(diffs[:, 0], diffs[:, 1]))
        print(f"RRT* Smooth Path Length: {r_length:.3f} m")

    print("=" * 65)


if __name__ == "__main__":
    compare_planners()
