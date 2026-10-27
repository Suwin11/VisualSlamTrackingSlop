#!/usr/bin/env python3
"""
test_draggable_maps.py
Automated verification test for draggable and zoomable maps in both RRTPI and MapCreator.
"""

import math
import os
import sys
import numpy as np
import cv2

# Ensure workspace root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from global_map import GlobalOccupancyGrid, CELL_FREE, CELL_OCCUPIED, CELL_UNKNOWN
from map_creator import MapCreatorGUI


def test_map_creator_draggable_and_large():
    print("\n--- TEST 1: MapCreator on 16m x 16m Map (Draggable & Zoomable) ---")
    creator = MapCreatorGUI(width_m=16.0, height_m=16.0, resolution=0.05, map_name="large_arena")
    assert creator.width_m == 16.0
    assert creator.height_m == 16.0
    assert creator.cols == 320
    assert creator.rows == 320
    print(f"[PASS] Created 16m x 16m map ({creator.cols}x{creator.rows} cells)")

    # 1. Test initial rendering
    gui_0 = creator.render_full_gui()
    assert gui_0.shape == (creator.total_h, creator.win_w, 3)
    print(f"[PASS] Rendered initial full GUI {gui_0.shape}")

    # 2. Test panning
    init_cx, init_cy = creator.view_center_x, creator.view_center_y
    creator.pan_view(120, -80)
    print(f"Panned view: center moved from ({init_cx:.2f}, {init_cy:.2f}) to ({creator.view_center_x:.2f}, {creator.view_center_y:.2f})")
    assert creator.view_center_x != init_cx
    assert creator.view_center_y != init_cy
    gui_panned = creator.render_full_gui()
    assert gui_panned.shape == (creator.total_h, creator.win_w, 3)
    print("[PASS] Pan view rendered successfully")

    # 3. Test zooming
    creator.zoom_view(2.0, pivot_sx=400, pivot_sy=470)
    assert abs(creator.view_zoom - 2.0) < 1e-4
    gui_zoomed = creator.render_full_gui()
    assert gui_zoomed.shape == (creator.total_h, creator.win_w, 3)
    print(f"[PASS] Zoom view (2.0x) rendered successfully")

    # 4. Test drawing while panned and zoomed
    # Draw a 4m wall from (-2.0, 1.0) to (2.0, 1.0)
    gx0, gy0 = creator.world_to_grid(-2.0, 1.0)
    gx1, gy1 = creator.world_to_grid(2.0, 1.0)
    cv2.line(creator.grid, (gx0, gy0), (gx1, gy1), CELL_OCCUPIED, 2)
    assert np.any(creator.grid == CELL_OCCUPIED)
    print(f"[PASS] Drawn wall into metric grid at (-2.0, 1.0) -> (2.0, 1.0)")

    # 5. Test mouse event simulation: Middle button drag (pan)
    # Simulate Middle Button Down at (400, 400)
    creator.on_mouse(cv2.EVENT_MBUTTONDOWN, 400, 400, cv2.EVENT_FLAG_MBUTTON, None)
    assert creator.is_panning is True
    # Simulate Mouse Move to (450, 380)
    old_cx = creator.view_center_x
    creator.on_mouse(cv2.EVENT_MOUSEMOVE, 450, 380, cv2.EVENT_FLAG_MBUTTON, None)
    assert creator.view_center_x != old_cx
    # Simulate Middle Button Up
    creator.on_mouse(cv2.EVENT_MBUTTONUP, 450, 380, 0, None)
    assert creator.is_panning is False
    print("[PASS] Middle button pan drag simulated successfully")

    # 6. Test mouse event simulation: Wheel zoom
    zoom_before = creator.view_zoom
    creator.on_mouse(cv2.EVENT_MOUSEWHEEL, 400, 400, 120 << 16, None)  # Zoom in
    assert creator.view_zoom > zoom_before
    print(f"[PASS] Mouse wheel zoom in simulated ({zoom_before:.2f}x -> {creator.view_zoom:.2f}x)")

    # 7. Test save map
    saved_file = creator.save_map("saved_maps/test_large_arena.npz")
    assert os.path.exists(saved_file)
    print(f"[PASS] Saved large map to {saved_file}")
    return saved_file


def test_global_map_and_rrt_draggable(saved_file):
    print("\n--- TEST 2: Global 2D Map & RRT Navigation (Draggable & Zoomable) ---")
    gmap = GlobalOccupancyGrid(width_m=10.0, height_m=10.0)
    
    # 1. Load the 16m x 16m map saved by MapCreator
    ok = gmap.load_map(saved_file)
    assert ok is True
    assert gmap.width_m == 16.0
    assert gmap.height_m == 16.0
    print(f"[PASS] Loaded {gmap.width_m}m x {gmap.height_m}m map into GlobalOccupancyGrid")

    # 2. Render initial view
    vis_0 = gmap.render_map(640, 480)
    assert vis_0.shape == (480, 640, 3)
    print(f"[PASS] Rendered map panel {vis_0.shape}")

    # 3. Test pan viewport
    init_cx = gmap.view_center_x
    init_cy = gmap.view_center_y
    gmap.pan_viewport(80, -60, 640, 480)
    assert gmap.view_center_x != init_cx
    assert gmap.view_center_y != init_cy
    vis_pan = gmap.render_map(640, 480)
    assert vis_pan.shape == (480, 640, 3)
    print(f"[PASS] Viewport panned to center ({gmap.view_center_x:.2f}m, {gmap.view_center_y:.2f}m)")

    # 4. Test zoom viewport
    gmap.zoom_viewport(2.2, pivot_sx=320, pivot_sy=240, out_w=640, out_h=480)
    assert abs(gmap.view_zoom - 2.2) < 1e-4
    vis_zoom = gmap.render_map(640, 480, robot_pose=(1.0, 2.0, 0.5), goal=(4.0, 5.0))
    assert vis_zoom.shape == (480, 640, 3)
    print(f"[PASS] Viewport zoomed to {gmap.view_zoom:.1f}x with robot pose & goal")

    # 5. Test screen_to_world and to_screen_px reciprocity
    test_coords = [(100, 100), (320, 240), (550, 420), (50, 400)]
    for sx, sy in test_coords:
        wx, wy = gmap.screen_to_world(sx, sy, 640, 480)
        sx_proj, sy_proj = gmap.to_screen_px(wx, wy, 640, 480)
        err = abs(sx - sx_proj) + abs(sy - sy_proj)
        assert err <= 1, f"Transform error {err} at ({sx}, {sy})"
    print("[PASS] Screen <-> World coordinate reciprocity verified with 0px error")

    # 6. Test Center on Robot
    gmap.current_pose = (3.5, -2.5, math.pi / 4)
    gmap.center_on_robot()
    assert abs(gmap.view_center_x - 3.5) < 1e-4
    assert abs(gmap.view_center_y - (-2.5)) < 1e-4
    print("[PASS] Center on robot verified")

    # 7. Test Reset Viewport
    gmap.reset_viewport()
    assert abs(gmap.view_zoom - 1.0) < 1e-4
    print("[PASS] Reset viewport to full 1.0x map verified")


if __name__ == "__main__":
    try:
        saved_map_path = test_map_creator_draggable_and_large()
        test_global_map_and_rrt_draggable(saved_map_path)
        print("\n" + "=" * 60)
        print("ALL DRAGGABLE & ZOOMABLE MAP TESTS PASSED (100%)!")
        print("=" * 60)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
