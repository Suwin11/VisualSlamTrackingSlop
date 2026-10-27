#!/usr/bin/env python3
"""
save_map.py
CLI tool to inspect, export, list, and convert saved occupancy maps.
Usage:
  python save_map.py --list                 # List all saved maps
  python save_map.py --info <map.npz>       # Show detailed map metadata
  python save_map.py --export-png <map.npz> # Render map to high-resolution PNG
"""

import argparse
import datetime
import glob
import json
import os
import sys
import numpy as np

# Ensure workspace root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from global_map import GlobalOccupancyGrid, CELL_FREE, CELL_OCCUPIED, CELL_UNKNOWN


def list_maps(map_dir="saved_maps"):
    if not os.path.exists(map_dir):
        print(f"No saved maps directory '{map_dir}' found.")
        return
    files = sorted(glob.glob(os.path.join(map_dir, "*.npz")), key=os.path.getmtime, reverse=True)
    if not files:
        print(f"No .npz maps found in '{map_dir}'.")
        return
    print(f"\nSaved Maps in '{map_dir}':")
    print("-" * 75)
    print(f"{'Filename':<32} {'Size':<12} {'Resolution':<12} {'Modified'}")
    print("-" * 75)
    for f in files:
        fname = os.path.basename(f)
        try:
            d = np.load(f, allow_pickle=True)
            w = float(d["width_m"]) if "width_m" in d else 0.0
            h = float(d["height_m"]) if "height_m" in d else 0.0
            res = float(d["resolution"]) if "resolution" in d else 0.05
            size_str = f"{w:.1f}m x {h:.1f}m"
            res_str = f"{res*100:.0f}cm"
        except Exception:
            size_str = "Unknown"
            res_str = "-"
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")
        print(f"{fname:<32} {size_str:<12} {res_str:<12} {mtime}")
    print("-" * 75)


def show_map_info(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Map file '{filepath}' not found.")
        return
    try:
        data = np.load(filepath, allow_pickle=True)
        bg = data["binary_grid"]
        w = float(data["width_m"]) if "width_m" in data else 0.0
        h = float(data["height_m"]) if "height_m" in data else 0.0
        res = float(data["resolution"]) if "resolution" in data else 0.05
        ox = float(data["origin_x"]) if "origin_x" in data else 0.0
        oy = float(data["origin_y"]) if "origin_y" in data else 0.0
        pose = tuple(data["current_pose"]) if "current_pose" in data else (0, 0, 0)
        n_free = int(np.sum(bg == CELL_FREE))
        n_occ = int(np.sum(bg == CELL_OCCUPIED))
        n_unk = int(np.sum(bg == CELL_UNKNOWN))
        total = bg.size

        print("\n" + "=" * 50)
        print(f"Map File: {os.path.abspath(filepath)}")
        print("=" * 50)
        print(f"Dimensions      : {w:.2f} m x {h:.2f} m ({bg.shape[1]} cols x {bg.shape[0]} rows)")
        print(f"Resolution      : {res:.3f} m/cell ({res*100:.1f} cm)")
        print(f"Origin (SW cor) : ({ox:.2f}m, {oy:.2f}m)")
        print(f"Saved Robot Pose: ({pose[0]:.2f}m, {pose[1]:.2f}m, {math.degrees(pose[2]):.1f}°)")
        print(f"Cell Statistics : Free={n_free} ({n_free*100/total:.1f}%), Occ={n_occ} ({n_occ*100/total:.1f}%), Unk={n_unk} ({n_unk*100/total:.1f}%)")
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")


def export_png(filepath, out_png=None):
    gmap = GlobalOccupancyGrid()
    if not gmap.load_map(filepath):
        return
    if out_png is None:
        out_png = os.path.splitext(filepath)[0] + ".png"
    vis = gmap.render_map(out_w=1024, out_h=1024, show_trajectory=True)
    import cv2
    cv2.imwrite(out_png, vis)
    print(f"[EXPORT] Saved high-resolution PNG to: {os.path.abspath(out_png)}")


def main():
    ap = argparse.ArgumentParser(description="Save, inspect, and export robot occupancy maps")
    ap.add_argument("--list", action="store_true", help="List all saved maps in saved_maps/")
    ap.add_argument("--info", metavar="MAP_FILE", help="Display detailed metadata for a map file")
    ap.add_argument("--export-png", metavar="MAP_FILE", help="Export map to a high-resolution PNG image")
    ap.add_argument("--dir", default="saved_maps", help="Maps directory (default: saved_maps)")
    args = ap.parse_args()

    if args.list or len(sys.argv) == 1:
        list_maps(args.dir)
    elif args.info:
        show_map_info(args.info)
    elif args.export_png:
        export_png(args.export_png)


if __name__ == "__main__":
    main()
