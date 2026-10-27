#!/usr/bin/env python3
"""
map_creator.py
Interactive Graphical Map Creator & Floorplan Editor for FIBO Autonomous Robot.

Allows creating custom 2D metric occupancy grids of ANY size (no 3x3 limitation).
The canvas dynamically auto-expands as you draw or pan into new areas.
Saved maps (.npz) can be directly loaded into `navigate_rrt.py` using `--load-map`.

Controls:
  * [0 / P / H] PAN TOOL : Drag canvas with Left Click, Middle Click, or Shift+Drag
  * Number Keys 1-6 : Switch Drawing Tools
      [1] WALL (Straight Line)
      [2] BOX / RECTANGLE (Tables, Furniture)
      [3] FREEHAND PEN (Custom Obstacles)
      [4] CIRCLE (Round Pillars)
      [5] ERASER (Create Doorways / Openings)
      [6] ROBOT START POSE (Place Robot & Heading)
  * [7] ROOM PERIMETER : Generate outer perimeter walls around current map
  * [E] EXPAND CANVAS : Expand map bounds by +5 meters in all directions
  * Mouse Wheel / [+/-] : Zoom in / out centered at cursor position
  * [F / Home] : Center view on robot start pose
  * [R] : Reset view to fit the full map overview
  * [O / L] OPEN MAP : Open native file picker to select and edit any saved map
  * [S] SAVE MAP : Quick save to saved_maps/<name>.npz
  * [Shift+S] SAVE AS : Save with custom name via file dialog
  * [Z] UNDO : Undo last drawing action
  * [C] CLEAR : Clear map to empty floor space
  * [I] INFLATE : Toggle collision clearance inflation preview
  * [Q / ESC] QUIT
"""

import argparse
import datetime
import math
import os
import sys
import cv2
import numpy as np

# Ensure workspace root in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from global_map import CELL_FREE, CELL_OCCUPIED, CELL_UNKNOWN


class MapCreatorGUI:
    def __init__(self, width_m=20.0, height_m=20.0, resolution=0.05, robot_radius_m=0.18, map_name="custom_map"):
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.resolution = float(resolution)
        self.robot_radius_m = float(robot_radius_m)
        self.map_name = str(map_name)

        self.cols = int(round(self.width_m / self.resolution))
        self.rows = int(round(self.height_m / self.resolution))
        self.origin_x = -self.width_m / 2.0
        self.origin_y = -self.height_m / 2.0

        # Reference scale: 60 pixels per meter at 1.0x zoom
        self.pixels_per_meter = 60.0

        # Viewport state for interactive dragging (panning) & zooming
        self.view_center_x = 0.0
        self.view_center_y = 0.0
        # Auto-fit initial zoom to comfortable size
        self.view_zoom = 0.70
        self.is_panning = False
        self.pan_last_pt = None

        # Display window dimensions
        self.win_w = 900
        self.win_h = 800
        self.header_h = 75
        self.footer_h = 42
        self.total_h = self.win_h + self.header_h + self.footer_h

        # Internal grid: CELL_FREE (0) is walkable floor, CELL_OCCUPIED (255) is wall
        self.grid = np.full((self.rows, self.cols), CELL_FREE, dtype=np.uint8)

        # Robot starting pose [x, y, theta]
        self.robot_pose = [0.0, 0.0, math.pi / 2.0]

        # Active tool state: "PAN", "WALL", "BOX", "PEN", "CIRCLE", "ERASER", "ROBOT"
        self.active_tool = "WALL"
        self.wall_thickness_m = 0.10  # 10 cm walls
        self.show_inflation = True
        self.status_msg = "Ready. Draw walls or drag canvas to explore (no 3x3 limit - auto-expands everywhere!)."
        self.status_time = 0.0

        # Mouse interaction state
        self.mouse_down = False
        self.start_pt = None
        self.curr_pt = None
        self.temp_preview = None

        # Undo history
        self.history = []
        self._save_undo_state()

        self.window_name = "FIBO Map Creator - Custom Floorplan Editor"

    def _save_undo_state(self):
        """Save a snapshot of the grid, bounds, and robot pose for undo."""
        if len(self.history) >= 20:
            self.history.pop(0)
        self.history.append((
            self.grid.copy(),
            list(self.robot_pose),
            float(self.origin_x),
            float(self.origin_y),
            float(self.width_m),
            float(self.height_m),
            int(self.cols),
            int(self.rows)
        ))

    def undo(self):
        """Revert the most recent drawing action."""
        if len(self.history) > 1:
            self.history.pop()  # Discard current state
            prev = self.history[-1]
            self.grid = prev[0].copy()
            self.robot_pose = list(prev[1])
            self.origin_x = prev[2]
            self.origin_y = prev[3]
            self.width_m = prev[4]
            self.height_m = prev[5]
            self.cols = prev[6]
            self.rows = prev[7]
            self.set_status("Undo successful.")
        else:
            self.set_status("Nothing more to undo.")

    def set_status(self, msg):
        self.status_msg = msg
        import time
        self.status_time = time.time()

    def get_scale(self):
        """Compute pixels per meter based on current zoom."""
        return max(1.0, self.pixels_per_meter * self.view_zoom)

    def world_to_grid(self, wx, wy):
        """Convert world coordinates (meters) to grid indices (col, row)."""
        gx = int(math.floor((wx - self.origin_x) / self.resolution))
        gy = int(math.floor((wy - self.origin_y) / self.resolution))
        return gx, gy

    def grid_to_world(self, gx, gy):
        """Convert grid indices (col, row) to world coordinates (meters)."""
        wx = self.origin_x + (gx + 0.5) * self.resolution
        wy = self.origin_y + (gy + 0.5) * self.resolution
        return wx, wy

    def screen_to_world(self, sx, sy):
        """Maps full-window pixel coordinates to metric world coordinates (wx, wy)."""
        sy_canvas = sy - self.header_h
        if not (0 <= sx < self.win_w and 0 <= sy_canvas < self.win_h):
            return None, None
        scale = self.get_scale()
        wx = self.view_center_x + (sx - self.win_w / 2.0) / scale
        wy = self.view_center_y - (sy_canvas - self.win_h / 2.0) / scale
        return wx, wy

    def world_to_canvas(self, wx, wy):
        """Convert world coordinates to canvas pixel (0..win_w, 0..win_h)."""
        scale = self.get_scale()
        sx = int(round(self.win_w / 2.0 + (wx - self.view_center_x) * scale))
        sy_c = int(round(self.win_h / 2.0 - (wy - self.view_center_y) * scale))
        return sx, sy_c

    def world_to_screen(self, wx, wy):
        """Convert world coordinates to full window pixel."""
        sx, sy_c = self.world_to_canvas(wx, wy)
        return sx, sy_c + self.header_h

    def pan_view(self, delta_sx, delta_sy):
        """Pan viewport by screen pixel delta."""
        scale = self.get_scale()
        self.view_center_x -= float(delta_sx) / scale
        self.view_center_y += float(delta_sy) / scale

    def zoom_view(self, factor, pivot_sx=None, pivot_sy=None):
        """Zoom viewport centered around pivot screen position."""
        if pivot_sx is None or pivot_sy is None or pivot_sy < self.header_h or pivot_sy > (self.header_h + self.win_h):
            pivot_sx = self.win_w / 2.0
            pivot_sy = self.header_h + (self.win_h / 2.0)

        sy_c = pivot_sy - self.header_h
        wx, wy = self.screen_to_world(pivot_sx, pivot_sy)
        if wx is None:
            wx, wy = self.view_center_x, self.view_center_y

        self.view_zoom = float(np.clip(self.view_zoom * factor, 0.10, 50.0))
        new_scale = self.get_scale()

        self.view_center_x = wx - (pivot_sx - self.win_w / 2.0) / new_scale
        self.view_center_y = wy + (sy_c - self.win_h / 2.0) / new_scale

    def center_on_robot(self):
        """Center view on robot starting pose."""
        self.view_center_x = float(self.robot_pose[0])
        self.view_center_y = float(self.robot_pose[1])
        self.set_status(f"Centered view on Robot ({self.robot_pose[0]:.2f}m, {self.robot_pose[1]:.2f}m)")

    def reset_view(self):
        """Fit the entire active map nicely into the window view."""
        self.view_center_x = self.origin_x + self.width_m / 2.0
        self.view_center_y = self.origin_y + self.height_m / 2.0
        target_scale = min(self.win_w / max(1.0, self.width_m), self.win_h / max(1.0, self.height_m)) * 0.85
        self.view_zoom = float(np.clip(target_scale / self.pixels_per_meter, 0.15, 6.0))
        self.set_status(f"Fit view to full map ({self.width_m:.1f}m x {self.height_m:.1f}m)")

    def ensure_grid_contains(self, min_wx, min_wy, max_wx, max_wy, padding_m=3.0):
        """
        Dynamically expands self.grid to ensure [min_wx, max_wx] x [min_wy, max_wy]
        is fully contained within the grid with padding.
        Expands the NumPy array and shifts the origin seamlessly without altering world coordinates.
        This removes ANY 3x3m or fixed-size building limitation!
        """
        curr_min_x = self.origin_x
        curr_max_x = self.origin_x + self.width_m
        curr_min_y = self.origin_y
        curr_max_y = self.origin_y + self.height_m

        need_min_x = min_wx - padding_m
        need_max_x = max_wx + padding_m
        need_min_y = min_wy - padding_m
        need_max_y = max_wy + padding_m

        expand_left = need_min_x < curr_min_x
        expand_right = need_max_x > curr_max_x
        expand_bottom = need_min_y < curr_min_y
        expand_top = need_max_y > curr_max_y

        if not (expand_left or expand_right or expand_bottom or expand_top):
            return

        new_min_x = math.floor(min(curr_min_x, need_min_x))
        new_max_x = math.ceil(max(curr_max_x, need_max_x))
        new_min_y = math.floor(min(curr_min_y, need_min_y))
        new_max_y = math.ceil(max(curr_max_y, need_max_y))

        new_width_m = float(new_max_x - new_min_x)
        new_height_m = float(new_max_y - new_min_y)

        # Cap max bounds to 150m x 150m to prevent out-of-memory
        new_width_m = min(150.0, new_width_m)
        new_height_m = min(150.0, new_height_m)

        new_cols = int(round(new_width_m / self.resolution))
        new_rows = int(round(new_height_m / self.resolution))

        if new_cols <= self.cols and new_rows <= self.rows and new_min_x >= self.origin_x and new_min_y >= self.origin_y:
            return

        new_grid = np.full((new_rows, new_cols), CELL_FREE, dtype=np.uint8)

        col_offset = int(round((self.origin_x - new_min_x) / self.resolution))
        row_offset = int(round((self.origin_y - new_min_y) / self.resolution))

        col_offset = max(0, min(new_cols - self.cols, col_offset))
        row_offset = max(0, min(new_rows - self.rows, row_offset))

        new_grid[row_offset : row_offset + self.rows, col_offset : col_offset + self.cols] = self.grid

        self.grid = new_grid
        self.origin_x = new_min_x
        self.origin_y = new_min_y
        self.width_m = new_width_m
        self.height_m = new_height_m
        self.cols = new_cols
        self.rows = new_rows

    def expand_canvas(self, padding_m=5.0):
        """Expands the canvas boundaries by padding_m meters in all four directions."""
        self._save_undo_state()
        self.ensure_grid_contains(
            self.origin_x - padding_m,
            self.origin_y - padding_m,
            self.origin_x + self.width_m + padding_m,
            self.origin_y + self.height_m + padding_m,
            padding_m=0.0
        )
        self.set_status(f"Canvas expanded to {self.width_m:.1f}m x {self.height_m:.1f}m ({self.cols}x{self.rows} cells)")

    def build_room_perimeter(self, margin_m=0.40):
        """Build outer rectangular perimeter walls around the current map area."""
        self._save_undo_state()
        thick_px = max(1, int(round(self.wall_thickness_m / self.resolution)))
        margin_cols = max(1, int(round(margin_m / self.resolution)))
        margin_rows = max(1, int(round(margin_m / self.resolution)))

        c0, c1 = margin_cols, self.cols - margin_cols - 1
        r0, r1 = margin_rows, self.rows - margin_rows - 1

        cv2.line(self.grid, (c0, r0), (c1, r0), CELL_OCCUPIED, thick_px)
        cv2.line(self.grid, (c1, r0), (c1, r1), CELL_OCCUPIED, thick_px)
        cv2.line(self.grid, (c1, r1), (c0, r1), CELL_OCCUPIED, thick_px)
        cv2.line(self.grid, (c0, r1), (c0, r0), CELL_OCCUPIED, thick_px)
        self.set_status("Generated outer perimeter walls around canvas.")

    def clear_map(self):
        """Clear all walls and obstacles to empty floor."""
        self._save_undo_state()
        self.grid.fill(CELL_FREE)
        self.set_status("Map cleared to empty floor.")

    def save_map(self, filepath=None):
        """Save map to .npz, .json metadata, and .png preview."""
        save_dir = "saved_maps"
        os.makedirs(save_dir, exist_ok=True)
        if filepath is None:
            filepath = os.path.join(save_dir, f"{self.map_name}.npz")
        elif not filepath.endswith(".npz"):
            filepath += ".npz"

        # Compute log-odds and inflated grid
        log_odds = np.zeros((self.rows, self.cols), dtype=np.float32)
        log_odds[self.grid == CELL_FREE] = -2.0
        log_odds[self.grid == CELL_OCCUPIED] = +3.5

        kernel_cells = max(3, int(round((self.robot_radius_m / self.resolution) * 2)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_cells, kernel_cells))
        inflated = cv2.dilate((self.grid == CELL_OCCUPIED).astype(np.uint8) * 255, kernel)

        # Save NPZ
        np.savez_compressed(
            filepath,
            binary_grid=self.grid,
            log_odds=log_odds,
            inflated_grid=inflated,
            width_m=float(self.width_m),
            height_m=float(self.height_m),
            resolution=float(self.resolution),
            origin_x=float(self.origin_x),
            origin_y=float(self.origin_y),
            robot_radius_m=float(self.robot_radius_m),
            current_pose=np.array(self.robot_pose, dtype=np.float64),
            trajectory=np.array([self.robot_pose], dtype=np.float64),
            timestamp=datetime.datetime.now().timestamp(),
        )

        # Save metadata JSON
        json_path = os.path.splitext(filepath)[0] + ".json"
        meta = {
            "name": self.map_name,
            "width_m": self.width_m,
            "height_m": self.height_m,
            "resolution": self.resolution,
            "cols": self.cols,
            "rows": self.rows,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "robot_radius_m": self.robot_radius_m,
            "robot_start_pose": list(self.robot_pose),
            "wall_thickness_m": self.wall_thickness_m,
            "created_at": datetime.datetime.now().isoformat(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            import json
            json.dump(meta, f, indent=2)

        # Save visual PNG image preview (full overview)
        png_path = os.path.splitext(filepath)[0] + ".png"
        vis = self.render_canvas_only(full_overview=True)
        cv2.imwrite(png_path, vis)

        # Also copy to latest_map.npz
        latest_path = os.path.join(save_dir, "latest_map.npz")
        import shutil
        if os.path.abspath(filepath) != os.path.abspath(latest_path):
            shutil.copy2(filepath, latest_path)
            shutil.copy2(json_path, os.path.splitext(latest_path)[0] + ".json")
            shutil.copy2(png_path, os.path.splitext(latest_path)[0] + ".png")

        print(f"[MAP-CREATOR] Saved map to: {os.path.abspath(filepath)}")
        self.set_status(f"Saved: {os.path.basename(filepath)} (+ latest_map.npz)")
        return filepath

    def open_file_dialog(self):
        """Open native Windows file dialog to select any saved map (.npz or .json) to edit."""
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            save_dir = os.path.abspath("saved_maps")
            if not os.path.exists(save_dir):
                save_dir = os.path.abspath(os.path.dirname(__file__))

            selected = filedialog.askopenfilename(
                initialdir=save_dir,
                title="Open Map for Editing - FIBO Map Creator",
                filetypes=[
                    ("Occupancy Grid Maps (*.npz)", "*.npz"),
                    ("JSON Map Metadata (*.json)", "*.json"),
                    ("All Files (*.*)", "*.*")
                ]
            )
            root.destroy()

            if selected:
                if selected.endswith(".json"):
                    candidate_npz = os.path.splitext(selected)[0] + ".npz"
                    if os.path.exists(candidate_npz):
                        selected = candidate_npz
                return self.load_map(selected)
        except Exception as e:
            self.set_status(f"File dialog error: {e}")
            print(f"[MAP-CREATOR] File dialog error: {e}")
        return False

    def save_as_dialog(self):
        """Open native Windows Save As dialog to save the map under a new name."""
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            save_dir = os.path.abspath("saved_maps")
            os.makedirs(save_dir, exist_ok=True)

            selected = filedialog.asksaveasfilename(
                initialdir=save_dir,
                initialfile=f"{self.map_name}.npz",
                defaultextension=".npz",
                title="Save Map As - FIBO Map Creator",
                filetypes=[
                    ("Occupancy Grid Maps (*.npz)", "*.npz"),
                    ("All Files (*.*)", "*.*")
                ]
            )
            root.destroy()

            if selected:
                base_name = os.path.splitext(os.path.basename(selected))[0]
                self.map_name = base_name
                return self.save_map(selected)
        except Exception as e:
            self.set_status(f"Save As dialog error: {e}")
            print(f"[MAP-CREATOR] Save As dialog error: {e}")
        return None

    def load_map(self, filepath=None):
        """Load an existing map file into the editor."""
        save_dir = "saved_maps"
        if filepath is None:
            latest = os.path.join(save_dir, "latest_map.npz")
            if os.path.exists(latest):
                filepath = latest
            else:
                npzs = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith(".npz")]
                if npzs:
                    filepath = max(npzs, key=os.path.getmtime)
                else:
                    self.set_status("No saved maps found to load.")
                    return False

        if not os.path.exists(filepath):
            self.set_status(f"File not found: {filepath}")
            return False

        try:
            self._save_undo_state()
            data = np.load(filepath, allow_pickle=True)
            loaded_grid = data["binary_grid"].astype(np.uint8)
            self.rows, self.cols = loaded_grid.shape
            self.grid = loaded_grid
            self.width_m = float(data["width_m"]) if "width_m" in data else self.cols * self.resolution
            self.height_m = float(data["height_m"]) if "height_m" in data else self.rows * self.resolution
            self.resolution = float(data["resolution"]) if "resolution" in data else self.resolution
            self.origin_x = float(data["origin_x"]) if "origin_x" in data else -self.width_m / 2.0
            self.origin_y = float(data["origin_y"]) if "origin_y" in data else -self.height_m / 2.0
            if "current_pose" in data:
                self.robot_pose = list(float(x) for x in data["current_pose"].ravel()[:3])

            self.map_name = os.path.splitext(os.path.basename(filepath))[0]

            # Center view and fit zoom
            self.reset_view()
            self.set_status(f"Loaded: {os.path.basename(filepath)} ({self.width_m:.1f}m x {self.height_m:.1f}m)")
            print(f"[MAP-CREATOR] Successfully loaded: {filepath} ({self.cols}x{self.rows} cells, {self.width_m:.1f}m x {self.height_m:.1f}m)")
            return True
        except Exception as e:
            self.set_status(f"Load error: {e}")
            print(f"[MAP-CREATOR] Load error: {e}")
            return False

    def on_mouse(self, event, x, y, flags, param):
        # 1. Header Area Click (Tools & Action Buttons)
        if y < self.header_h:
            if event == cv2.EVENT_LBUTTONDOWN:
                if 6 <= y <= 36:
                    # Row 1: Tool Selection Buttons
                    tools = [
                        ("0: Pan", "PAN"),
                        ("1: Wall", "WALL"),
                        ("2: Box", "BOX"),
                        ("3: Pen", "PEN"),
                        ("4: Circle", "CIRCLE"),
                        ("5: Eraser", "ERASER"),
                        ("6: Robot", "ROBOT"),
                        ("7: Room", "ROOM"),
                        ("+5m Expand", "EXPAND"),
                    ]
                    btn_w = self.win_w // len(tools)
                    idx = min(len(tools) - 1, max(0, x // btn_w))
                    tool_id = tools[idx][1]
                    if tool_id == "ROOM":
                        self.build_room_perimeter()
                    elif tool_id == "EXPAND":
                        self.expand_canvas(5.0)
                    else:
                        self.active_tool = tool_id
                        self.set_status(f"Active Tool: {tools[idx][0]}")

                elif 42 <= y <= 70:
                    # Row 2: Action Buttons
                    if 8 <= x <= 110:
                        self.open_file_dialog()
                    elif 115 <= x <= 195:
                        self.save_map()
                    elif 200 <= x <= 290:
                        self.save_as_dialog()
                    elif 295 <= x <= 365:
                        self.undo()
                    elif 370 <= x <= 440:
                        self.clear_map()
                    elif 445 <= x <= 555:
                        self.show_inflation = not self.show_inflation
                        self.set_status(f"Collision Clearance Inflation -> {'ON' if self.show_inflation else 'OFF'}")
                    elif 560 <= x <= 680:
                        if x < 620:
                            self.wall_thickness_m = max(0.04, round(self.wall_thickness_m - 0.02, 2))
                        else:
                            self.wall_thickness_m = min(0.35, round(self.wall_thickness_m + 0.02, 2))
                        self.set_status(f"Wall Thickness -> {int(self.wall_thickness_m * 100)} cm")
                    elif 685 <= x <= 785:
                        self.reset_view()
                    elif 790 <= x <= 892:
                        self.center_on_robot()
            return

        # 2. Canvas Area Interactions
        # Reset panning if any pan button was released
        if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            if self.is_panning:
                self.is_panning = False
                self.pan_last_pt = None

        # Mouse Wheel Zoom
        if event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.15 if flags > 0 else (1.0 / 1.15)
            self.zoom_view(factor, pivot_sx=x, pivot_sy=y)
            self.set_status(f"Zoom: {self.view_zoom:.2f}x (Scroll wheel / Drag to pan)")
            return

        # Middle Button Drag (Universal Pan)
        if event == cv2.EVENT_MBUTTONDOWN:
            self.is_panning = True
            self.pan_last_pt = (x, y)
            return

        # Right Button Drag (Secondary Pan)
        if event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning = True
            self.pan_last_pt = (x, y)
            return

        # Shift + Left Click: Explicit Pan for laptops
        if (flags & cv2.EVENT_FLAG_SHIFTKEY) and event == cv2.EVENT_LBUTTONDOWN:
            self.is_panning = True
            self.pan_last_pt = (x, y)
            return

        # PAN Tool active: Left click and drag pans the view
        if self.active_tool == "PAN":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.is_panning = True
                self.pan_last_pt = (x, y)
                return
            elif event == cv2.EVENT_LBUTTONUP:
                self.is_panning = False
                self.pan_last_pt = None
                return

        # Active Panning Move
        if self.is_panning:
            if event == cv2.EVENT_MOUSEMOVE:
                # Cancel pan if no buttons are held
                if not (flags & (cv2.EVENT_FLAG_MBUTTON | cv2.EVENT_FLAG_RBUTTON | cv2.EVENT_FLAG_LBUTTON)):
                    self.is_panning = False
                    self.pan_last_pt = None
                    return
                if self.pan_last_pt is not None:
                    dx = x - self.pan_last_pt[0]
                    dy = y - self.pan_last_pt[1]
                    self.pan_view(dx, dy)
                    self.pan_last_pt = (x, y)
            return

        # 3. Drawing Tools (Wall, Box, Pen, Circle, Eraser, Robot)
        wx, wy = self.screen_to_world(x, y)
        if wx is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = True
            self.start_pt = (wx, wy)
            self.curr_pt = (wx, wy)
            self._save_undo_state()

            if self.active_tool == "PEN":
                self.ensure_grid_contains(wx, wy, wx, wy)
                gx, gy = self.world_to_grid(wx, wy)
                thick_px = max(1, int(round(self.wall_thickness_m / self.resolution)))
                cv2.circle(self.grid, (gx, gy), max(1, thick_px // 2), CELL_OCCUPIED, -1)
            elif self.active_tool == "ERASER":
                self.ensure_grid_contains(wx, wy, wx, wy)
                gx, gy = self.world_to_grid(wx, wy)
                thick_px = max(2, int(round((self.wall_thickness_m * 1.5) / self.resolution)))
                cv2.circle(self.grid, (gx, gy), thick_px, CELL_FREE, -1)

        elif event == cv2.EVENT_MOUSEMOVE:
            self.curr_pt = (wx, wy)
            if self.mouse_down:
                if self.active_tool == "PEN":
                    self.ensure_grid_contains(wx, wy, wx, wy)
                    gx, gy = self.world_to_grid(wx, wy)
                    thick_px = max(1, int(round(self.wall_thickness_m / self.resolution)))
                    cv2.circle(self.grid, (gx, gy), max(1, thick_px // 2), CELL_OCCUPIED, -1)
                elif self.active_tool == "ERASER":
                    self.ensure_grid_contains(wx, wy, wx, wy)
                    gx, gy = self.world_to_grid(wx, wy)
                    thick_px = max(2, int(round((self.wall_thickness_m * 1.5) / self.resolution)))
                    cv2.circle(self.grid, (gx, gy), thick_px, CELL_FREE, -1)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.mouse_down and self.start_pt is not None:
                sx_w, sy_w = self.start_pt
                ex_w, ey_w = wx, wy
                thick_px = max(1, int(round(self.wall_thickness_m / self.resolution)))

                if self.active_tool == "WALL":
                    self.ensure_grid_contains(min(sx_w, ex_w), min(sy_w, ey_w), max(sx_w, ex_w), max(sy_w, ey_w))
                    gx0, gy0 = self.world_to_grid(sx_w, sy_w)
                    gx1, gy1 = self.world_to_grid(ex_w, ey_w)
                    cv2.line(self.grid, (gx0, gy0), (gx1, gy1), CELL_OCCUPIED, thick_px)
                    self.set_status(f"Drawn Wall: ({sx_w:.2f}m, {sy_w:.2f}m) -> ({ex_w:.2f}m, {ey_w:.2f}m)")

                elif self.active_tool == "BOX":
                    self.ensure_grid_contains(min(sx_w, ex_w), min(sy_w, ey_w), max(sx_w, ex_w), max(sy_w, ey_w))
                    gx0, gy0 = self.world_to_grid(sx_w, sy_w)
                    gx1, gy1 = self.world_to_grid(ex_w, ey_w)
                    c_min, c_max = min(gx0, gx1), max(gx0, gx1)
                    r_min, r_max = min(gy0, gy1), max(gy0, gy1)
                    cv2.rectangle(self.grid, (c_min, r_min), (c_max, r_max), CELL_OCCUPIED, -1)
                    box_w = abs(ex_w - sx_w)
                    box_h = abs(ey_w - sy_w)
                    self.set_status(f"Drawn Box: {box_w:.2f}m x {box_h:.2f}m")

                elif self.active_tool == "CIRCLE":
                    r_m = math.hypot(ex_w - sx_w, ey_w - sy_w)
                    self.ensure_grid_contains(sx_w - r_m, sy_w - r_m, sx_w + r_m, sy_w + r_m)
                    gx0, gy0 = self.world_to_grid(sx_w, sy_w)
                    r_px = int(round(r_m / self.resolution))
                    cv2.circle(self.grid, (gx0, gy0), max(1, r_px), CELL_OCCUPIED, -1)
                    self.set_status(f"Drawn Circular Pillar: radius={r_m:.2f}m")

                elif self.active_tool == "ROBOT":
                    self.ensure_grid_contains(sx_w, sy_w, sx_w, sy_w)
                    dx_drag = ex_w - sx_w
                    dy_drag = ey_w - sy_w
                    rtheta = math.atan2(dy_drag, dx_drag) if math.hypot(dx_drag, dy_drag) > 0.15 else self.robot_pose[2]
                    self.robot_pose = [round(sx_w, 2), round(sy_w, 2), rtheta]
                    self.set_status(f"Set Robot Start Pose: ({sx_w:.2f}m, {sy_w:.2f}m, {math.degrees(rtheta):.0f}°)")

            self.mouse_down = False
            self.start_pt = None

    def render_canvas_only(self, full_overview=False):
        """Render the map canvas with metric grid lines, active map border, robot pose, and scale bar."""
        canvas = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        # Background: Free space floor
        canvas[self.grid == CELL_FREE] = (220, 235, 225)

        # Show inflated collision zone
        if self.show_inflation:
            kernel_cells = max(3, int(round((self.robot_radius_m / self.resolution) * 2)) | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_cells, kernel_cells))
            inflated = cv2.dilate((self.grid == CELL_OCCUPIED).astype(np.uint8) * 255, kernel)
            canvas[inflated > 0] = (160, 185, 230)

        # Draw walls/obstacles in bold dark blue
        canvas[self.grid == CELL_OCCUPIED] = (40, 45, 180)
        canvas = cv2.flip(canvas, 0)

        if full_overview:
            scale = min(self.win_w / max(1.0, self.width_m), self.win_h / max(1.0, self.height_m)) * 0.95
            c_x = self.origin_x + self.width_m / 2.0
            c_y = self.origin_y + self.height_m / 2.0
        else:
            scale = self.get_scale()
            c_x = self.view_center_x
            c_y = self.view_center_y

        s = self.resolution * scale
        tx = (self.win_w / 2.0) + (self.origin_x - c_x) * scale
        ty = (self.win_h / 2.0) - (self.origin_y + self.height_m - c_y) * scale

        M = np.array([
            [s, 0.0, tx],
            [0.0, s, ty]
        ], dtype=np.float32)

        # Subtle light blueprint drafting background outside current grid boundaries
        vis = cv2.warpAffine(canvas, M, (self.win_w, self.win_h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(235, 240, 238))

        # Draw active map boundary box so user clearly sees current grid bounds
        p0_x = int(round((self.win_w / 2.0) + (self.origin_x - c_x) * scale))
        p0_y = int(round((self.win_h / 2.0) - (self.origin_y + self.height_m - c_y) * scale))
        p1_x = int(round((self.win_w / 2.0) + (self.origin_x + self.width_m - c_x) * scale))
        p1_y = int(round((self.win_h / 2.0) - (self.origin_y - c_y) * scale))
        cv2.rectangle(vis, (p0_x, p0_y), (p1_x, p1_y), (140, 160, 150), 1, cv2.LINE_AA)

        # Dynamic continuous metric grid lines across the visible window
        half_w_m = (self.win_w / 2.0) / scale
        half_h_m = (self.win_h / 2.0) / scale
        wx_min = c_x - half_w_m
        wx_max = c_x + half_w_m
        wy_min = c_y - half_h_m
        wy_max = c_y + half_h_m

        if scale >= 80.0:
            step = 0.5
            label_interval = 1.0
        elif scale >= 30.0:
            step = 1.0
            label_interval = 2.0
        elif scale >= 12.0:
            step = 2.0
            label_interval = 4.0
        else:
            step = 5.0
            label_interval = 10.0

        x_start = int(math.floor(wx_min / step)) * step
        x_end = int(math.ceil(wx_max / step)) * step
        y_start = int(math.floor(wy_min / step)) * step
        y_end = int(math.ceil(wy_max / step)) * step

        max_lines = 150
        num_x = int(round((x_end - x_start) / step)) + 1
        num_y = int(round((y_end - y_start) / step)) + 1

        if num_x <= max_lines:
            cur_x = x_start
            while cur_x <= x_end + 1e-4:
                sx = int(round((self.win_w / 2.0) + (cur_x - c_x) * scale))
                if 0 <= sx < self.win_w:
                    is_zero = abs(cur_x) < 1e-4
                    col = (90, 110, 135) if is_zero else (185, 195, 205)
                    th = 2 if is_zero else 1
                    cv2.line(vis, (sx, 0), (sx, self.win_h), col, th)
                    if abs(cur_x) > 1e-4 and abs(round(cur_x / label_interval) * label_interval - cur_x) < 1e-4:
                        lbl = f"{int(round(cur_x))}m" if abs(cur_x - round(cur_x)) < 1e-3 else f"{cur_x:.1f}m"
                        cv2.putText(vis, lbl, (sx + 3, self.win_h - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 90, 100), 1, cv2.LINE_AA)
                cur_x += step

        if num_y <= max_lines:
            cur_y = y_start
            while cur_y <= y_end + 1e-4:
                sy = int(round((self.win_h / 2.0) - (cur_y - c_y) * scale))
                if 0 <= sy < self.win_h:
                    is_zero = abs(cur_y) < 1e-4
                    col = (90, 110, 135) if is_zero else (185, 195, 205)
                    th = 2 if is_zero else 1
                    cv2.line(vis, (0, sy), (self.win_w, sy), col, th)
                    if abs(cur_y) > 1e-4 and abs(round(cur_y / label_interval) * label_interval - cur_y) < 1e-4:
                        lbl = f"{int(round(cur_y))}m" if abs(cur_y - round(cur_y)) < 1e-3 else f"{cur_y:.1f}m"
                        cv2.putText(vis, lbl, (6, sy - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 90, 100), 1, cv2.LINE_AA)
                cur_y += step

        # Origin Marker (0, 0)
        ox = int(round((self.win_w / 2.0) + (0.0 - c_x) * scale))
        oy = int(round((self.win_h / 2.0) - (0.0 - c_y) * scale))
        if 0 <= ox < self.win_w and 0 <= oy < self.win_h:
            cv2.drawMarker(vis, (ox, oy), (20, 20, 200), cv2.MARKER_CROSS, 16, 2)

        # Robot Pose
        rx, ry, rtheta = self.robot_pose
        rcx = int(round((self.win_w / 2.0) + (rx - c_x) * scale))
        rcy = int(round((self.win_h / 2.0) - (ry - c_y) * scale))
        if -80 <= rcx < self.win_w + 80 and -80 <= rcy < self.win_h + 80:
            r_rad_px = max(10, int(round(self.robot_radius_m * scale)))
            cv2.circle(vis, (rcx, rcy), r_rad_px, (0, 165, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, (rcx, rcy), r_rad_px, (255, 255, 255), 2, cv2.LINE_AA)
            tip_x = int(rcx + r_rad_px * 2.0 * math.cos(rtheta))
            tip_y = int(rcy - r_rad_px * 2.0 * math.sin(rtheta))
            cv2.arrowedLine(vis, (rcx, rcy), (tip_x, tip_y), (0, 0, 230), 3, tipLength=0.35, line_type=cv2.LINE_AA)
            cv2.putText(vis, f"ROBOT START ({rx:+.2f}m, {ry:+.2f}m)", (rcx + 16, rcy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 100, 220), 1, cv2.LINE_AA)

        # Dynamic Scale bar at bottom-right
        bar_m = 1.0 if scale >= 25.0 else (5.0 if scale < 10.0 else 2.0)
        bar_len_px = int(bar_m * scale)
        if bar_len_px > 10:
            bx2 = self.win_w - 20
            bx1 = bx2 - bar_len_px
            by = self.win_h - 18
            if bx1 > 20:
                cv2.line(vis, (bx1, by), (bx2, by), (50, 60, 70), 2)
                cv2.line(vis, (bx1, by - 4), (bx1, by + 4), (50, 60, 70), 2)
                cv2.line(vis, (bx2, by - 4), (bx2, by + 4), (50, 60, 70), 2)
                cv2.putText(vis, f"{bar_m:.1f} m", (bx1 + 6, by - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 60, 70), 1, cv2.LINE_AA)

        # Badges in top-right
        if not full_overview:
            info_badge = f"{self.width_m:.0f}x{self.height_m:.0f}m | {self.view_zoom:.2f}x"
            cv2.putText(vis, info_badge, (self.win_w - 140, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 115, 130), 1, cv2.LINE_AA)

        return vis

    def render_full_gui(self):
        full = np.zeros((self.total_h, self.win_w, 3), dtype=np.uint8)

        # Header background
        cv2.rectangle(full, (0, 0), (self.win_w, self.header_h), (35, 40, 48), -1)

        # Row 1: Drawing & Navigation Tools
        tools = [
            ("0: Pan", "PAN"),
            ("1: Wall", "WALL"),
            ("2: Box", "BOX"),
            ("3: Pen", "PEN"),
            ("4: Circle", "CIRCLE"),
            ("5: Eraser", "ERASER"),
            ("6: Robot", "ROBOT"),
            ("7: Room", "ROOM"),
            ("+5m Expand", "EXPAND"),
        ]
        btn_w = self.win_w // len(tools)
        for i, (label, tool_id) in enumerate(tools):
            bx0 = i * btn_w + 2
            bx1 = (i + 1) * btn_w - 2
            is_active = (self.active_tool == tool_id)
            btn_bg = (0, 150, 230) if is_active else (50, 58, 70)
            text_fg = (255, 255, 255) if is_active else (200, 205, 215)
            cv2.rectangle(full, (bx0, 6), (bx1, 36), btn_bg, -1, cv2.LINE_AA)
            cv2.rectangle(full, (bx0, 6), (bx1, 36), (80, 95, 115), 1, cv2.LINE_AA)
            cv2.putText(full, label, (bx0 + 6, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_fg, 1, cv2.LINE_AA)

        # Row 2: File Actions, Undo, Thickness, View controls
        actions = [
            ("Open [O]", 8, 110, (40, 120, 180)),
            ("Save [S]", 115, 195, (40, 140, 60)),
            ("Save As...", 200, 290, (50, 60, 75)),
            ("Undo [Z]", 295, 365, (50, 60, 75)),
            ("Clear [C]", 370, 440, (140, 40, 40)),
            (f"Inflate: {'ON' if self.show_inflation else 'OFF'}", 445, 555, (0, 110, 170) if self.show_inflation else (55, 60, 70)),
            (f"Thick: {int(self.wall_thickness_m*100)}cm [-/+]", 560, 680, (50, 60, 75)),
            ("Fit View [R]", 685, 785, (50, 60, 75)),
            ("Center Robot [F]", 790, 892, (50, 60, 75)),
        ]
        for label, ax0, ax1, abg in actions:
            cv2.rectangle(full, (ax0, 42), (ax1, 68), abg, -1, cv2.LINE_AA)
            cv2.rectangle(full, (ax0, 42), (ax1, 68), (80, 95, 115), 1, cv2.LINE_AA)
            cv2.putText(full, label, (ax0 + 6, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (230, 235, 245), 1, cv2.LINE_AA)

        # Render canvas
        canvas_vis = self.render_canvas_only()

        # Render in-progress mouse drag preview
        if self.mouse_down and self.start_pt is not None and self.curr_pt is not None:
            sx0, sy0_c = self.world_to_canvas(*self.start_pt)
            sx1, sy1_c = self.world_to_canvas(*self.curr_pt)

            if self.active_tool == "WALL":
                cv2.line(canvas_vis, (sx0, sy0_c), (sx1, sy1_c), (0, 0, 255), 2, cv2.LINE_AA)
            elif self.active_tool == "BOX":
                cv2.rectangle(canvas_vis, (min(sx0, sx1), min(sy0_c, sy1_c)),
                              (max(sx0, sx1), max(sy0_c, sy1_c)), (0, 0, 255), 2, cv2.LINE_AA)
            elif self.active_tool == "CIRCLE":
                r_c = int(math.hypot(sx1 - sx0, sy1_c - sy0_c))
                cv2.circle(canvas_vis, (sx0, sy0_c), r_c, (0, 0, 255), 2, cv2.LINE_AA)
            elif self.active_tool == "ROBOT":
                cv2.arrowedLine(canvas_vis, (sx0, sy0_c), (sx1, sy1_c), (0, 255, 0), 2, tipLength=0.3, line_type=cv2.LINE_AA)

        full[self.header_h : self.header_h + self.win_h, :] = canvas_vis

        # Footer bar
        foot_y = self.header_h + self.win_h
        cv2.rectangle(full, (0, foot_y), (self.win_w, self.total_h), (25, 28, 35), -1)

        curs_str = f"Cursor: ({self.curr_pt[0]:+.2f}m, {self.curr_pt[1]:+.2f}m) | " if self.curr_pt else ""
        hint_str = f"Map: {self.width_m:.1f}m x {self.height_m:.1f}m | Mid/Right-Drag: Pan | Wheel: Zoom"
        full_status = f"{self.status_msg}  [{curs_str}{hint_str}]"
        cv2.putText(full, full_status, (12, foot_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 235, 245), 1, cv2.LINE_AA)

        return full

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.win_w, self.total_h)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        print("=" * 76)
        print("  FIBO Map Creator & Custom Floorplan Editor (Infinite Canvas / Auto-Expand)")
        print("=" * 76)
        print("  * DRAW ANYWHERE: No 3x3m limitation! Canvas automatically expands as you build.")
        print("  * PAN CANVAS   : Middle Mouse Button, Right Click, Shift+Drag, or [0] PAN Tool")
        print("  * ZOOM VIEW    : Mouse Scroll Wheel or [+] / [-] keys")
        print("  * OPEN / EDIT  : Click [Open Map] button or press [O] / [L] to browse files")
        print("  * SAVE MAP     : Click [Save] button or press [S] | [Shift+S] for Save As")
        print("  * CENTER / FIT : Press [F] to center on Robot | [R] to fit full map")
        print("  * TOOLS        : '1' Wall | '2' Box | '3' Pen | '4' Circle | '5' Eraser | '6' Robot")
        print("=" * 76)

        while True:
            gui = self.render_full_gui()
            cv2.imshow(self.window_name, gui)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key in (ord('0'), ord('p'), ord('P'), ord('h'), ord('H')):
                self.active_tool = "PAN"
                self.set_status("Tool: PAN / MOVE VIEW (Drag canvas with Left Click, Middle Click, or Shift+Drag)")
            elif key == ord('1'):
                self.active_tool = "WALL"
                self.set_status("Tool: STRAIGHT WALL (Click & drag line)")
            elif key == ord('2'):
                self.active_tool = "BOX"
                self.set_status("Tool: RECTANGULAR BOX (Click & drag box)")
            elif key == ord('3'):
                self.active_tool = "PEN"
                self.set_status("Tool: FREEHAND PEN (Drag to sketch)")
            elif key == ord('4'):
                self.active_tool = "CIRCLE"
                self.set_status("Tool: CIRCULAR PILLAR (Click & drag radius)")
            elif key == ord('5'):
                self.active_tool = "ERASER"
                self.set_status("Tool: ERASER (Drag to erase / make doorways)")
            elif key == ord('6'):
                self.active_tool = "ROBOT"
                self.set_status("Tool: ROBOT START POSE (Click position & drag heading)")
            elif key == ord('7'):
                self.build_room_perimeter()
            elif key in (ord('e'), ord('E')):
                self.expand_canvas(5.0)
            elif key in (ord('o'), ord('O'), ord('l'), ord('L')):
                self.open_file_dialog()
            elif key == ord('S'):  # Capital S / Shift+S
                self.save_as_dialog()
            elif key == ord('s'):
                self.save_map()
            elif key in (ord('f'), ord('F')):
                self.center_on_robot()
            elif key in (ord('r'), ord('R')):
                self.reset_view()
            elif key in (ord('+'), ord('=')):
                self.zoom_view(1.25)
                self.set_status(f"Zoom -> {self.view_zoom:.2f}x")
            elif key in (ord('-'), ord('_')):
                self.zoom_view(0.80)
                self.set_status(f"Zoom -> {self.view_zoom:.2f}x")
            elif key in (ord('['), ord('{')):
                self.wall_thickness_m = max(0.04, round(self.wall_thickness_m - 0.02, 2))
                self.set_status(f"Wall Thickness -> {int(self.wall_thickness_m * 100)} cm")
            elif key in (ord(']'), ord('}')):
                self.wall_thickness_m = min(0.35, round(self.wall_thickness_m + 0.02, 2))
                self.set_status(f"Wall Thickness -> {int(self.wall_thickness_m * 100)} cm")
            elif key in (ord('i'), ord('I')):
                self.show_inflation = not self.show_inflation
                self.set_status(f"Collision clearance inflation -> {'ON' if self.show_inflation else 'OFF'}")
            elif key in (ord('z'), ord('Z')):
                self.undo()
            elif key in (ord('c'), ord('C')):
                self.clear_map()

        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="FIBO Interactive 2D Map Creator & Floorplan Editor")
    ap.add_argument("--width", type=float, default=20.0, help="Initial map width in meters (default: 20.0)")
    ap.add_argument("--height", type=float, default=20.0, help="Initial map height in meters (default: 20.0)")
    ap.add_argument("--res", type=float, default=0.05, help="Grid resolution in meters/cell (default: 0.05)")
    ap.add_argument("--name", default="custom_map", help="Map name (default: custom_map)")
    ap.add_argument("--load", default=None, help="Existing map file to load and edit (.npz or .json)")
    args = ap.parse_args()

    app = MapCreatorGUI(
        width_m=args.width,
        height_m=args.height,
        resolution=args.res,
        map_name=args.name,
    )
    if args.load is not None:
        app.load_map(args.load)

    app.run()


if __name__ == "__main__":
    main()
