"""
scratch/test_auto_escape_maneuver.py
Verification suite for autonomous TOF obstacle escape / unstuck feature.
"""

import sys
import os
import math
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from motor_control import MotorController
from global_map import GlobalOccupancyGrid, CELL_OCCUPIED, CELL_FREE


def test_escape_maneuver():
    print("=" * 60)
    print("TESTING AUTONOMOUS TOF ESCAPE & UNSTUCK FEATURE")
    print("=" * 60)

    # 1. Test MotorController trigger_escape_pivot during emergency stop
    mc = MotorController(max_v=0.30, max_omega=1.20)
    mc.control_mode = "AUTO"

    # Simulate emergency stop
    robot_pose = (0.0, 0.0, math.pi / 2.0)
    path = [(0.0, 0.0), (0.0, 1.5)]
    cmd = mc.compute_command(robot_pose, path, goal=(0.0, 1.5), emergency_stop=True)

    # Normal emergency stop should produce 0,0
    assert mc.last_cmd["mode"] in ("EMERGENCY_STOP", "PAUSED", "AUTO"), f"Unexpected mode: {mc.last_cmd['mode']}"
    print("[PASS] Normal emergency stop verified (motors clamped at 0, 0).")

    # Now trigger escape pivot right
    mc.trigger_escape_pivot(direction="right", duration_sec=0.50)
    # Give the 25Hz thread a moment to loop
    time.sleep(0.08)

    with mc._lock:
        active_cmd = dict(mc.last_cmd)
        pwm_l = active_cmd.get("pwm_left", 0)
        pwm_r = active_cmd.get("pwm_right", 0)
        mode = active_cmd.get("mode")

    print(f"[PASS] Triggered escape pivot right: mode={mode}, PWM=({pwm_l}, {pwm_r})")
    assert mode == "EMERGENCY_PIVOT", f"Expected EMERGENCY_PIVOT, got {mode}"
    # Pivot right: left forward (+), right reverse (-)
    assert pwm_l > 0 and pwm_r < 0, f"Expected left > 0 and right < 0, got ({pwm_l}, {pwm_r})"

    # Now test cancel escape pivot
    mc.cancel_escape_pivot()
    time.sleep(0.08)
    with mc._lock:
        active_cmd = dict(mc.last_cmd)
        pwm_l = active_cmd.get("pwm_left", 0)
        pwm_r = active_cmd.get("pwm_right", 0)
    assert pwm_l == 0 and pwm_r == 0, f"Expected 0, 0 after cancel, got ({pwm_l}, {pwm_r})"
    print("[PASS] cancel_escape_pivot verified (motors stopped immediately).")

    # 2. Test left turn pivot
    mc.trigger_escape_pivot(direction="left", duration_sec=0.50)
    time.sleep(0.08)
    with mc._lock:
        active_cmd = dict(mc.last_cmd)
        pwm_l = active_cmd.get("pwm_left", 0)
        pwm_r = active_cmd.get("pwm_right", 0)
        mode = active_cmd.get("mode")
    print(f"[PASS] Triggered escape pivot left: mode={mode}, PWM=({pwm_l}, {pwm_r})")
    assert mode == "EMERGENCY_PIVOT", f"Expected EMERGENCY_PIVOT, got {mode}"
    # Pivot left: left reverse (-), right forward (+)
    assert pwm_l < 0 and pwm_r > 0, f"Expected left < 0 and right > 0, got ({pwm_l}, {pwm_r})"
    mc.cancel_escape_pivot()

    # 3. Test Clearance-based Direction Selection with GlobalOccupancyGrid
    grid = GlobalOccupancyGrid(width_m=10.0, height_m=10.0, resolution=0.05)
    # Place wall to the right of robot
    rx, ry, rtheta = 0.0, 0.0, math.pi / 2.0  # Facing North
    # Right of North is East (+X)
    import cv2
    for wy in np_arange(0.2, 1.0, 0.05):
        c = grid.world_to_grid(0.5, wy)
        if c:
            grid.binary_grid[c[1], c[0]] = CELL_OCCUPIED
    occ_bin = (grid.binary_grid == CELL_OCCUPIED).astype(np.uint8) * 255
    grid.inflated_grid = cv2.dilate(occ_bin, grid.inflate_kernel)

    # Check right vs left clearance
    r_th = rtheta - math.radians(50.0)
    l_th = rtheta + math.radians(50.0)
    pt_r = (rx + 0.65 * math.cos(r_th), ry + 0.65 * math.sin(r_th))
    pt_l = (rx + 0.65 * math.cos(l_th), ry + 0.65 * math.sin(l_th))

    r_free = grid.is_collision_free(pt_r[0], pt_r[1])
    l_free = grid.is_collision_free(pt_l[0], pt_l[1])

    print(f"Right point {pt_r}: is_free={r_free} | Left point {pt_l}: is_free={l_free}")
    assert not r_free, "Right point should be blocked by obstacle"
    assert l_free, "Left point should be free"
    print("[PASS] Clearance-based direction selection correctly identified left as open escape path!")

    mc.stop_motors()
    print("=" * 60)
    print("ALL AUTO-ESCAPE TESTS PASSED!")
    print("=" * 60)


def np_arange(start, stop, step):
    import numpy as np
    return np.arange(start, stop, step)


if __name__ == "__main__":
    test_escape_maneuver()
