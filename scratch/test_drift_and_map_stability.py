import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import json
from stereo_visual_odometry import StereoVisualOdometry
from global_map import GlobalOccupancyGrid
import occupancy

def test_drift_and_stability():
    print("=== TESTING DRIFT AND MAP STABILITY ===")
    vo = StereoVisualOdometry(baseline_m=0.08)
    # Feed initial IMU
    vo.feed_imu(yaw_rad=0.0, pitch_rad=0.0, roll_rad=0.0, gyro_yaw_rate=0.0, lin_acc_mag=0.05, dt=0.04)
    
    # 1. TEST STATIONARY IMMUNITY (Robot sitting still for 100 frames = 4 seconds)
    print("\n--- Test 1: Stationary Zero-Drift ---")
    start_pose = (vo.x, vo.y, vo.theta)
    for _ in range(100):
        # Simulate small accelerometer noise
        vo.feed_imu(yaw_rad=0.0, pitch_rad=0.0, roll_rad=0.0, gyro_yaw_rate=0.01, lin_acc_mag=0.06, dt=0.04)
        vo.update(None, None, dt=0.04, cmd_v=0.0)
    
    end_pose = (vo.x, vo.y, vo.theta)
    drift = math.hypot(end_pose[0] - start_pose[0], end_pose[1] - start_pose[1])
    print(f"Start Pose: {start_pose}")
    print(f"End Pose after 100 idle frames: {end_pose}")
    print(f"Total Drift: {drift:.4f} meters")
    assert drift == 0.0, f"FAILED: Stationary robot drifted by {drift:.4f}m!"
    print("--> Test 1 PASSED: Zero drift when stopped!")

    # 2. TEST PURE IN-PLACE ROTATION IMMUNITY (Robot spinning in place for 50 frames)
    print("\n--- Test 2: In-Place Pure Rotation Stability ---")
    start_xy = (vo.x, vo.y)
    for i in range(50):
        cur_yaw = math.radians(i * 2.0) # rotating 2 deg per frame
        vo.feed_imu(yaw_rad=cur_yaw, pitch_rad=0.0, roll_rad=0.0, gyro_yaw_rate=0.87, lin_acc_mag=0.05, dt=0.04)
        vo.update(None, None, dt=0.04, cmd_v=0.0)
    
    end_xy = (vo.x, vo.y)
    rot_drift = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
    print(f"Position before rotation: {start_xy}")
    print(f"Position after 100 deg turn: {end_xy}")
    print(f"Robot heading: {math.degrees(vo.theta):.1f} deg")
    print(f"Translational Drift during turn: {rot_drift:.4f} meters")
    assert rot_drift == 0.0, f"FAILED: Robot translated by {rot_drift:.4f}m while turning in place!"
    print("--> Test 2 PASSED: Pure rotation has zero translational drift!")

    # 3. TEST MAP STABILITY & NO PREMATURE ROLLING
    print("\n--- Test 3: Global Map Stability (8m x 8m) ---")
    gmap = GlobalOccupancyGrid(width_m=8.0, height_m=8.0, resolution=0.05, enable_rolling=False)
    orig_ox, orig_oy = gmap.origin_x, gmap.origin_y
    print(f"Map origin: ({orig_ox:.2f}, {orig_oy:.2f})")
    
    # Simulate robot moving within 2.5m of origin
    bev = np.full((80, 64), occupancy.FREE, dtype=np.uint8)
    for step in range(20):
        rx = step * 0.10
        ry = step * 0.05
        gmap.update_from_local_bev(bev, (rx, ry, math.pi/2.0), local_range_y=(0.0, 4.0))
    
    print(f"Map origin after travel: ({gmap.origin_x:.2f}, {gmap.origin_y:.2f})")
    assert (gmap.origin_x == orig_ox) and (gmap.origin_y == orig_oy), "FAILED: Map origin shifted unexpectedly!"
    print("--> Test 3 PASSED: Map origin remained rock-solid without rolling or degenerating!")
    
    print("\n=== ALL DRIFT & STABILITY TESTS PASSED! ===")

if __name__ == '__main__':
    test_drift_and_stability()
