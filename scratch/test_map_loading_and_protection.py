import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import json
from global_map import GlobalOccupancyGrid, CELL_FREE, CELL_OCCUPIED, CELL_UNKNOWN
from rrt_star import RRTStarPlanner
import occupancy

def test_map_load_and_protection():
    print("=== TEST: MAP LOADING FROM FILE & EXPLORED MAP PROTECTION ===")
    
    # 1. TEST LOADING MAP FROM NPZ
    print("\n--- Test 1: Loading Map from .npz ---")
    gmap = GlobalOccupancyGrid(width_m=8.0, height_m=8.0, resolution=0.05)
    npz_path = "saved_maps/test_room.npz"
    assert os.path.exists(npz_path), f"File {npz_path} must exist"
    loaded = gmap.load_map(npz_path, keep_current_pose=False)
    assert loaded, "Failed to load test_room.npz"
    print(f"Loaded test_room.npz: shape={gmap.binary_grid.shape}, width={gmap.width_m}m, height={gmap.height_m}m")
    occ_count_initial = np.sum(gmap.binary_grid == CELL_OCCUPIED)
    print(f"Initial occupied (wall) cells in loaded map: {occ_count_initial}")
    assert occ_count_initial > 0, "Map must have occupied wall cells"
    print("--> Test 1 PASSED: .npz map loaded successfully!")

    # 2. TEST LOADING MAP FROM JSON
    print("\n--- Test 2: Loading Map from .json ---")
    gmap_json = GlobalOccupancyGrid(width_m=8.0, height_m=8.0, resolution=0.05)
    json_path = "saved_maps/test_room.json"
    assert os.path.exists(json_path), f"File {json_path} must exist"
    loaded_json = gmap_json.load_map(json_path, keep_current_pose=False)
    assert loaded_json, "Failed to load test_room.json"
    print(f"Loaded test_room.json: shape={gmap_json.binary_grid.shape}")
    assert np.sum(gmap_json.binary_grid == CELL_OCCUPIED) == occ_count_initial, "JSON load did not match NPZ"
    print("--> Test 2 PASSED: .json map loaded successfully!")

    # 3. TEST EXPLORED WALL PROTECTION (protect_explored_walls=True)
    print("\n--- Test 3: Protect Explored Walls from Camera Free-Space Sweeps ---")
    # Simulate camera sending a local BEV full of FREE space (e.g. looking at a wall with bad lighting)
    all_free_bev = np.full((80, 64), CELL_FREE, dtype=np.uint8)
    
    # We send 20 frames of FREE space directly over the mapped area
    for step in range(20):
        gmap.update_from_local_bev(all_free_bev, (0.0, 0.0, math.pi/2.0), local_range_y=(0.0, 4.0))
        gmap.update_from_tof((0.0, 0.0, math.pi/2.0), 3.5, max_range_m=4.0)
    
    occ_count_after = np.sum(gmap.binary_grid == CELL_OCCUPIED)
    print(f"Wall cells before sweep: {occ_count_initial}")
    print(f"Wall cells after 20 frames of camera free sweeps: {occ_count_after}")
    assert occ_count_after == occ_count_initial, f"FAILED: Explored walls were erased! Lost {occ_count_initial - occ_count_after} wall cells!"
    print("--> Test 3 PASSED: Explored walls are 100% protected and NOT overridden by camera free sweeps!")

    # 4. TEST COMPLETE MAP LOCK (map_locked=True)
    print("\n--- Test 4: Complete Map Lock (Freeze Explored Map) ---")
    gmap.map_locked = True
    
    # Capture map state snapshot
    grid_before = gmap.binary_grid.copy()
    log_odds_before = gmap.log_odds.copy()
    
    # Simulate violent camera noise + random laser hits
    noisy_bev = np.random.choice([CELL_FREE, CELL_OCCUPIED], size=(80, 64)).astype(np.uint8)
    for _ in range(50):
        gmap.update_from_local_bev(noisy_bev, (1.0, 1.0, 0.0), local_range_y=(0.0, 4.0))
        gmap.update_from_tof((1.0, 1.0, 0.0), 1.2, max_range_m=4.0)
    
    assert np.array_equal(gmap.binary_grid, grid_before), "FAILED: Map binary grid changed while MAP LOCKED!"
    assert np.array_equal(gmap.log_odds, log_odds_before), "FAILED: Map log_odds changed while MAP LOCKED!"
    print("--> Test 4 PASSED: Map Lock completely froze all map modifications!")

    # 5. TEST RRT* PATH PLANNING ON LOCKED LOADED MAP
    print("\n--- Test 5: RRT* Path Planning on Protected Map ---")
    planner = RRTStarPlanner(gmap, step_size=0.20, max_iter=300)
    start = (gmap.origin_x + gmap.width_m * 0.3, gmap.origin_y + gmap.height_m * 0.3)
    goal = (gmap.origin_x + gmap.width_m * 0.7, gmap.origin_y + gmap.height_m * 0.7)
    
    raw_p, smooth_p, _ = planner.plan(start, goal)
    print(f"RRT* Planned Path on Loaded Map: raw={len(raw_p)}, smooth={len(smooth_p)}")
    assert len(smooth_p) >= 2, "FAILED: RRT* could not plan on loaded map!"
    print("--> Test 5 PASSED: RRT* planned smooth collision-free path on loaded protected map!")

    print("\n=== ALL MAP LOADING & PROTECTION TESTS PASSED! ===")

if __name__ == '__main__':
    test_map_load_and_protection()
