"""
scratch/test_auto_expand_and_unknown_goal.py
Test dynamic auto-expanding occupancy grid and navigation into unexplored territory.
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from global_map import GlobalOccupancyGrid, CELL_OCCUPIED, CELL_FREE, CELL_UNKNOWN
from astar import AStarPlanner
from rrt_star import RRTStarPlanner


def test_dynamic_expansion():
    print("=" * 60)
    print("TESTING DYNAMIC GRID EXPANSION & UNEXPLORED GOAL NAVIGATION")
    print("=" * 60)

    # Start with a small 6x6m grid (origin at -3.0, -3.0)
    grid = GlobalOccupancyGrid(width_m=6.0, height_m=6.0, resolution=0.05, robot_radius_m=0.18)
    initial_cols, initial_rows = grid.cols, grid.rows

    # Mark a wall at (0, 1.0)
    c = grid.world_to_grid(0.0, 1.0)
    grid.binary_grid[c[1], c[0]] = CELL_OCCUPIED
    grid.log_odds[c[1], c[0]] = 5.0
    w_before = grid.grid_to_world(c[0], c[1])

    # 1. Test expanding to include a point far to the left: (-5.5, 3.5)
    # This was the exact user scenario: robot at (-5.27, 3.43)
    target_x, target_y = -5.5, 3.5
    assert grid.world_to_grid(target_x, target_y) is None, "Should initially be outside grid"

    # Add ensure_bounds_range logic
    
    # Call ensure_bounds_include
    grid.ensure_bounds_include(target_x, target_y, margin_m=2.0)

    new_c = grid.world_to_grid(target_x, target_y)
    assert new_c is not None, "Target should now be comfortably inside grid"
    print(f"[PASS] Grid expanded from ({initial_cols}x{initial_rows}) to ({grid.cols}x{grid.rows}) cells ({grid.width_m:.1f}x{grid.height_m:.1f}m)")

    # Verify existing wall did not move
    new_wall_c = grid.world_to_grid(w_before[0], w_before[1])
    assert grid.binary_grid[new_wall_c[1], new_wall_c[0]] == CELL_OCCUPIED, "Wall cell corrupted after expansion!"
    w_after = grid.grid_to_world(new_wall_c[0], new_wall_c[1])
    assert math.hypot(w_before[0] - w_after[0], w_before[1] - w_after[1]) < 1e-4, "World coords shifted!"
    print("[PASS] Existing map geometry preserved with zero physical coordinate shift.")

    # 2. Test navigation to an unexplored goal
    start_pos = (0.0, 0.0)
    unexplored_goal = (-5.0, 3.0)

    # Goal is in unexplored space (CELL_UNKNOWN)
    astar = AStarPlanner(grid, clearance_weight=0.20)
    raw_p, smooth_p, _ = astar.plan(start_pos, unexplored_goal)
    assert smooth_p is not None, "A* failed to plan to unexplored goal!"
    print(f"[PASS] A* successfully planned to unexplored goal ({unexplored_goal[0]}m, {unexplored_goal[1]}m) with {len(smooth_p)} smooth points!")

    # 3. Test RRT* navigation to unexplored goal
    rrt = RRTStarPlanner(grid, step_size=0.25, search_radius=0.55, max_iter=300)
    r_raw, r_smooth, _ = rrt.plan(start_pos, unexplored_goal)
    assert r_smooth is not None, "RRT* failed to plan to unexplored goal!"
    print(f"[PASS] RRT* successfully planned to unexplored goal with {len(r_smooth)} smooth points!")

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_dynamic_expansion()
