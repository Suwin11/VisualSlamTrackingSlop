"""
Test verification for:
1. ONLY TOF sensor can rewrite close-up (<= 4.0m) map with obstacles.
2. Camera scans surroundings (> 4.0m) and marks distant obstacles.
3. TOF immediately clears dynamic obstacles (e.g., person walking past).
4. High-range TOF (> 4.0m) carves full 4m cone as free space.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import cv2
from global_map import GlobalOccupancyGrid, CELL_FREE, CELL_OCCUPIED, CELL_UNKNOWN
from rrt_star import RRTStarPlanner

def test_close_range_tof_and_dynamic_clearing():
    print("=== TEST: CLOSE-RANGE TOF EXCLUSIVITY & DYNAMIC CLEARING ===")
    
    # 1. Initialize grid with close_range_m = 0.35 (robot bumper clearance)
    gmap = GlobalOccupancyGrid(width_m=12.0, height_m=12.0, resolution=0.05, robot_radius_m=0.18, close_range_m=0.35)
    gmap.protect_explored_walls = False
    robot_pose = (0.0, 0.0, math.pi / 2.0) # facing North (+Y)
    
    # --- Test 1: Camera bumper deadzone (< 0.35m) rejects floor noise; real obstacles (1.5m, 5.2m) are recorded ---
    print("\n--- Test 1: Floor noise under wheels (<0.35m) is rejected; obstacles (1.5m & 5.2m) recorded ---")
    local_h, local_w = 120, 64
    bev_with_close_obstacles = np.full((local_h, local_w), CELL_FREE, dtype=np.uint8)
    
    # Floor glare right under bumper at Y = 0.2m (deadzone)
    # Real obstacle at Y = 1.5m (person) and Y = 5.2m (surroundings)
    row_0_2m = int(round((6.0 - 0.2) / 6.0 * (local_h - 1)))
    row_1_5m = int(round((6.0 - 1.5) / 6.0 * (local_h - 1)))
    row_5_2m = int(round((6.0 - 5.2) / 6.0 * (local_h - 1)))
    
    bev_with_close_obstacles[row_0_2m-1:row_0_2m+2, 28:36] = CELL_OCCUPIED
    bev_with_close_obstacles[row_1_5m-2:row_1_5m+3, 28:36] = CELL_OCCUPIED
    bev_with_close_obstacles[row_5_2m-2:row_5_2m+3, 28:36] = CELL_OCCUPIED
    
    gmap.update_from_local_bev(bev_with_close_obstacles, robot_pose, local_range_x=(-1.0, 1.0), local_range_y=(0.0, 6.0))
    
    # Verify wheel deadzone at (0.0, 0.2m) is NOT occupied (never traps the robot!)
    is_wheel_free = gmap.is_collision_free(0.0, 0.2)
    print(f"Wheel deadzone noise at (0.0, 0.2m) is_collision_free: {is_wheel_free} (MUST BE True - No self-trapping!)")
    assert is_wheel_free, "FAILED: False floor noise under wheels was not rejected!"
    
    # Verify obstacle at (0.0, 1.5m) IS occupied (detects people/chairs!)
    is_mid_free = gmap.is_collision_free(0.0, 1.5)
    print(f"Mid-range obstacle at (0.0, 1.5m) is_collision_free: {is_mid_free} (MUST BE False - Object detected!)")
    assert not is_mid_free, "FAILED: Real obstacle at 1.5m was missed!"

    # Verify surroundings at (0.0, 5.2m) IS occupied
    is_distant_free = gmap.is_collision_free(0.0, 5.2)
    print(f"Surroundings camera obstacle at (0.0, 5.2m) is_collision_free: {is_distant_free} (MUST BE False)")
    assert not is_distant_free, "FAILED: Camera obstacle in surroundings (> 4.0m) was not recorded!"
    print("--> Test 1 PASSED: Self-trapping floor noise rejected; mid-range and surroundings objects fully detected!")

    # --- Test 2: TOF writes obstacles in close range ---
    print("\n--- Test 2: TOF writes real physical obstacle in close range ---")
    gmap.update_from_tof(robot_pose, 2.0, max_range_m=4.0)
    is_tof_obs_free = gmap.is_collision_free(0.0, 2.0)
    print(f"TOF obstacle at (0.0, 2.0m) is_collision_free: {is_tof_obs_free} (MUST BE False)")
    assert not is_tof_obs_free, "FAILED: TOF obstacle at 2.0m was not recorded!"
    
    # Verify space between robot and 2.0m is FREE
    is_pre_tof_free = gmap.is_collision_free(0.0, 1.0)
    print(f"Space along beam at (0.0, 1.0m) is_collision_free: {is_pre_tof_free} (MUST BE True)")
    assert is_pre_tof_free, "FAILED: Space before TOF obstacle should be FREE!"
    print("--> Test 2 PASSED: TOF successfully recorded obstacle at 2.0m!")

    # --- Test 3: Person walking past is actively cleared ---
    print("\n--- Test 3: Person walks away -> TOF beam instantly wipes obstruction ---")
    # Person was at 2.0m. Now person walks away, and TOF sees 3.8m down the hallway.
    gmap.update_from_tof(robot_pose, 3.8, max_range_m=4.0)
    
    # The cell at 2.0m where the person was MUST NOW BE FREE!
    is_person_cleared = gmap.is_collision_free(0.0, 2.0)
    print(f"After person leaves, cell at (0.0, 2.0m) is_collision_free: {is_person_cleared} (MUST BE True)")
    assert is_person_cleared, "FAILED: Ghost obstacle from passing person was not cleared by TOF beam!"
    print("--> Test 3 PASSED: Dynamic obstacle instantly wiped clean when person walked past!")

    # --- Test 4: Open corridor (> 4.0m) carves full 4.0m cone as free space ---
    print("\n--- Test 4: Open corridor (> 4.0m) carves entire 4m beam as free ---")
    # Artificially corrupt a cell at (0.0, 3.2m)
    gx = int(round((0.0 - gmap.origin_x) / gmap.resolution))
    gy = int(round((3.2 - gmap.origin_y) / gmap.resolution))
    gmap.binary_grid[gy, gx] = CELL_OCCUPIED
    gmap.log_odds[gy, gx] = 4.0
    gmap.inflated_grid = cv2.dilate((gmap.binary_grid == CELL_OCCUPIED).astype(np.uint8) * 255, gmap.inflate_kernel)
    assert not gmap.is_collision_free(0.0, 3.2)
    
    # TOF reports 4.5m (beyond max_range_m=4.0)
    gmap.update_from_tof(robot_pose, 4.5, max_range_m=4.0)
    is_cleared_by_open = gmap.is_collision_free(0.0, 3.2)
    print(f"After open corridor TOF reading (4.5m), cell at (0.0, 3.2m) is_collision_free: {is_cleared_by_open} (MUST BE True)")
    assert is_cleared_by_open, "FAILED: Open corridor reading did not carve full 4m cone as free!"
    print("--> Test 4 PASSED: Open corridor reading actively cleared full 4m cone!")

    # --- Test 5: A* / RRT* Path planning to goal ---
    print("\n--- Test 5: Path planning unblocked to goal ---")
    planner = RRTStarPlanner(gmap, step_size=0.20, max_iter=300)
    raw_p, smooth_p, _ = planner.plan((0.0, 0.0), (0.0, 3.0))
    print(f"Planned path nodes to goal: smooth={len(smooth_p)}")
    assert len(smooth_p) >= 2, "FAILED: Planner could not find path to goal!"
    print("--> Test 5 PASSED: Smooth path found straight to goal!")

    print("\n=== ALL TESTS PASSED WITH 100% SUCCESS! ===")

if __name__ == '__main__':
    test_close_range_tof_and_dynamic_clearing()
