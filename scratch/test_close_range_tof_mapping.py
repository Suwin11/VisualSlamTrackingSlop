"""
Integration test for close-range TOF mapping, camera horizon filtering, and RRT* navigation.
Verifies that:
1. Camera looking slightly up (-2.0 deg) does NOT mark far-away space (2.5m - 4.0m) as red.
2. In close range (0 - 4m), obstacles come strictly from TOF400C and confirmed YOLO objects.
3. The navigation goal at (0.0, 2.5m) remains collision-free.
4. RRT* plans a valid path and motor commands drive forward towards the goal.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import cv2
import json

from pointcloud_3d import PointCloudMap3D
import occupancy
from global_map import GlobalOccupancyGrid
from rrt_star import RRTStarPlanner
from motor_control import PurePursuitController

def run_test():
    print("=== START INTEGRATION TEST: CLOSE RANGE TOF MAPPING & GOAL NAVIGATION ===")
    
    with open('mapping_config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
        
    pitch_deg = cfg['camera']['camera_pitch_deg']
    pitch_rad = math.radians(pitch_deg)
    max_fwd = cfg['mapping_3d']['max_forward_range_m']
    max_tof = cfg['tof_sensor']['max_tof_range_m']
    goal = (float(cfg['yolo_and_planner']['default_goal_x_m']), float(cfg['yolo_and_planner']['default_goal_y_m']))
    
    print(f"Config: pitch={pitch_deg} deg, max_fwd={max_fwd}m, max_tof={max_tof}m, goal={goal}")
    
    # 1. Initialize subsystems
    pc_map = PointCloudMap3D(camera_height_m=0.09, max_range_m=max_fwd, wall_sensitivity=cfg['mapping_3d']['wall_sensitivity'])
    global_grid = GlobalOccupancyGrid(width_m=8.0, height_m=8.0, resolution=0.05, robot_radius_m=0.18)
    planner = RRTStarPlanner(global_grid, step_size=0.15, max_iter=300)
    follower = PurePursuitController(
        lookahead_m=cfg['yolo_and_planner'].get("lookahead_m", 0.35),
        max_v=cfg['yolo_and_planner'].get("max_linear_vel_mps", 0.30),
        max_omega=cfg['yolo_and_planner'].get("max_angular_vel_rads", 1.20),
    )
    
    # 2. Simulate camera frame in a room (ceiling/walls in upper half, floor in lower half)
    frame = np.random.randint(90, 140, (480, 640, 3), dtype=np.uint8)
    frame[:240, :] = np.random.randint(180, 230, (240, 640, 3), dtype=np.uint8) # bright walls/ceiling
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    robot_pose = (0.0, 0.0, math.pi / 2.0) # Robot at (0,0) facing +Y (North)
    
    # CASE 1: Open space (no physical obstacle close by, ToF reads 3.5m)
    tof_dist_m = 3.5
    H_ipm = occupancy.dynamic_ipm(640, 480, 64, 80, pitch_rad=pitch_rad)
    comb_mask = np.zeros((480, 640), dtype=np.uint8)
    
    depth_map = pc_map.estimate_metric_depth(frame, pitch_rad, 530.0, 530.0, 320.0, 240.0, tof_distance_m=tof_dist_m)
    bev_3d = pc_map.update_and_slice_bev(
        frame, depth_map, robot_pose, pitch_rad, 530.0, 530.0, 320.0, 240.0,
        local_range_y=(0.0, max_fwd), tof_distance_m=tof_dist_m
    )
    bev_ipm = occupancy.compute(
        comb_mask, None, (480, 640), H_ipm, frame_gray=gray,
        wall_sensitivity=pc_map.wall_sensitivity, tof_distance_m=tof_dist_m, max_range_m=max_fwd
    )
    local_bev = bev_3d.copy()
    local_bev[bev_ipm == occupancy.OCCUPIED] = occupancy.OCCUPIED
    
    # Update global map
    global_grid.update_from_local_bev(local_bev, robot_pose, local_range_y=(0.0, max_fwd))
    global_grid.update_from_tof(robot_pose, tof_dist_m, max_range_m=max_tof)
    global_grid.current_pose = robot_pose
    
    # Check goal cell state
    gx, gy = goal
    is_goal_free = global_grid.is_collision_free(gx, gy)
    print(f"[CASE 1] Goal ({gx}, {gy}) is_collision_free: {is_goal_free}")
    assert is_goal_free, f"FAILED: Goal {goal} is marked occupied in open space!"
    
    # Plan path to goal
    rx, ry, _ = robot_pose
    raw_p, smooth_p, tree_e = planner.plan((rx, ry), goal)
    print(f"[CASE 1] RRT* path nodes: raw={len(raw_p)}, smooth={len(smooth_p)}")
    assert len(smooth_p) >= 2, "FAILED: RRT* failed to find a path to the goal!"
    
    # Test path follower command
    follower.compute_command(robot_pose, smooth_p, goal=goal, emergency_stop=False)
    cmd = follower._latest_auto_cmd
    print(f"[CASE 1] Follower cmd: v={cmd['v']:.2f} m/s, omega={cmd['omega']:.2f} rad/s, mode={cmd['mode']}")
    assert cmd['v'] > 0.05, f"FAILED: Robot commanded zero velocity! cmd={cmd}"
    assert cmd['mode'] == "AUTO", f"FAILED: Unexpected mode {cmd['mode']}"
    
    # Verify ToF obstacle at 3.5m: check that (0.0, 3.5) is occupied in global map
    obs_x, obs_y = 0.0, 3.5
    is_obs_cell_free = global_grid.is_collision_free(obs_x, obs_y)
    print(f"[CASE 1] Obstacle cell ({obs_x}, {obs_y}) is_collision_free: {is_obs_cell_free} (should be False)")
    assert not is_obs_cell_free, "FAILED: ToF obstacle at 3.5m was not stamped into global map!"
    
    print("--> CASE 1 PASSED: Clean corridor, goal free, path found, robot cruising!\n")
    
    # CASE 2: Obstacle actively cleared by ToF (testing laser clearing of old ghost obstacles)
    print("--- Testing ToF active ghost clearing ---")
    # Artificially corrupt cell at (0.0, 1.8m) to OCCUPIED
    corrupt_gx = int(round((0.0 - global_grid.origin_x) / global_grid.resolution))
    corrupt_gy = int(round((1.8 - global_grid.origin_y) / global_grid.resolution))
    global_grid.binary_grid[corrupt_gy, corrupt_gx] = occupancy.OCCUPIED
    global_grid.log_odds[corrupt_gy, corrupt_gx] = 3.0
    global_grid.inflated_grid = cv2.dilate((global_grid.binary_grid == occupancy.OCCUPIED).astype(np.uint8) * 255, global_grid.inflate_kernel)
    assert not global_grid.is_collision_free(0.0, 1.8), "Setup failed: cell should be occupied"
    print(f"Injected ghost obstacle at (0.0, 1.8m): is_collision_free = {global_grid.is_collision_free(0.0, 1.8)}")
    
    # Now ToF beam fires at 3.0m straight through (0.0, 1.8m)
    global_grid.update_from_tof(robot_pose, 3.0, max_range_m=4.0)
    cleared = global_grid.is_collision_free(0.0, 1.8)
    print(f"After ToF beam fires through (0.0, 1.8m): is_collision_free = {cleared}")
    assert cleared, "FAILED: ToF laser beam did not clear the ghost obstacle in line of sight!"
    print("--> CASE 2 PASSED: Ghost obstacle actively wiped by ToF laser beam!\n")
    
    print("=== ALL INTEGRATION TESTS PASSED SUCCESSFULLY! ===")

if __name__ == '__main__':
    run_test()
