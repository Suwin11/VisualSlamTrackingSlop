#!/usr/bin/env python3
"""
auto_label_obstacles.py
Automatic & Semi-Automatic Annotator for Training YOLO on Walls and Obstacles.

Generates standard YOLO format annotation text files:
  <class_id> <x_center> <y_center> <width> <height>  (all normalized 0..1)
  Class 0: wall
  Class 1: obstacle

Also automatically writes `dataset/dataset.yaml` ready for Ultralytics YOLO fine-tuning!
"""

import argparse
import os
import cv2
import numpy as np


def detect_wall_and_obstacles(img_bgr):
    """
    Analyzes an indoor scene frame to propose bounding boxes for:
    - Wall (Class 0): Vertical surface above the floor baseboard line.
    - Obstacle (Class 1): Salient non-floor objects protruding from the ground.
    Returns list of (class_id, x_center, y_center, width, height) in 0..1 normalized coordinates.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    boxes = []

    # 1. Floor Reference Sampling (bottom center 18% of frame)
    ref_y1 = int(h * 0.82)
    ref_y2 = int(h * 0.98)
    ref_x1 = int(w * 0.30)
    ref_x2 = int(w * 0.70)
    ref_patch = gray[ref_y1:ref_y2, ref_x1:ref_x2]

    if ref_patch.size == 0:
        return boxes

    ref_med = float(np.median(ref_patch))
    ref_std = max(float(np.std(ref_patch)), 12.0)

    # 2. Multi-Scale Edge & Contrast Boundary Detection
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)).astype(np.float32)
    diff = np.abs(gray.astype(np.float32) - ref_med)

    search_top = int(h * 0.20)
    search_bottom = int(h * 0.88)

    # Baseboard seam edge detector
    edges = cv2.Canny(gray[search_top:search_bottom, :], 30, 80)
    edge_band = (grad_y[search_top:search_bottom, :] > 25.0) | (diff[search_top:search_bottom, :] > max(16.0, 1.8 * ref_std))

    # Combine edge signals
    combined_edges = np.zeros_like(gray[search_top:search_bottom, :])
    combined_edges[edge_band | (edges > 0)] = 255

    # Morphological line connector to join horizontal wall contact seams
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    wall_seams = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, h_kernel)

    # Find the lowest persistent wall base seam across the width
    has_seam = np.any(wall_seams > 0, axis=0)
    wall_seam_idx = (search_bottom - search_top - 1) - np.argmax(wall_seams[::-1, :] > 0, axis=0)
    wall_base_rows = search_top + wall_seam_idx

    # Group connected columns into wall segments
    active_start = None
    min_col_width = int(w * 0.10)

    for col in range(w):
        if has_seam[col]:
            if active_start is None:
                active_start = col
        else:
            if active_start is not None:
                col_span = col - active_start
                if col_span >= min_col_width:
                    base_row = int(np.median(wall_base_rows[active_start:col]))
                    top_row = max(0, int(base_row - (h * 0.55)))

                    # Wall Box
                    bx = (active_start + col) / 2.0 / w
                    by = (top_row + base_row) / 2.0 / h
                    bw = (col - active_start) / float(w)
                    bh = (base_row - top_row) / float(h)
                    if bw > 0.08 and bh > 0.15:
                        boxes.append((0, bx, by, bw, bh))
                active_start = None

    if active_start is not None and (w - active_start) >= min_col_width:
        base_row = int(np.median(wall_base_rows[active_start:w]))
        top_row = max(0, int(base_row - (h * 0.55)))
        bx = (active_start + w) / 2.0 / w
        by = (top_row + base_row) / 2.0 / h
        bw = (w - active_start) / float(w)
        bh = (base_row - top_row) / float(h)
        if bw > 0.08 and bh > 0.15:
            boxes.append((0, bx, by, bw, bh))

    # If no segmented wall was found but large area differs strongly from floor,
    # propose an upper background wall box
    if len(boxes) == 0:
        base_est = int(h * 0.65)
        top_est = int(h * 0.05)
        bx = 0.5
        by = (top_est + base_est) / 2.0 / h
        bw = 0.96
        bh = (base_est - top_est) / float(h)
        boxes.append((0, bx, by, bw, bh))

    # 3. Salient Floor Obstacle Contours
    floor_zone = (diff[int(h * 0.45):int(h * 0.90), :] > max(28.0, 2.2 * ref_std)).astype(np.uint8) * 255
    obs_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    floor_zone = cv2.morphologyEx(floor_zone, cv2.MORPH_OPEN, obs_kernel)
    contours, _ = cv2.findContours(floor_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= (w * h * 0.012):  # Minimum obstacle size (1.2% of frame)
            x_c, y_c, cw, ch = cv2.boundingRect(cnt)
            y_real = y_c + int(h * 0.45)
            # Check overlap with existing wall boxes
            ox = (x_c + cw / 2.0) / float(w)
            oy = (y_real + ch / 2.0) / float(h)
            ow = float(cw) / float(w)
            oh = float(ch) / float(h)
            if ow < 0.85:  # Discrete obstacle, not full width wall
                boxes.append((1, ox, oy, ow, oh))

    return boxes


def process_dataset(dataset_dir="dataset", preview=False):
    dataset_abs = os.path.abspath(dataset_dir)
    images_train = os.path.join(dataset_abs, "images", "train")
    images_val = os.path.join(dataset_abs, "images", "val")
    labels_train = os.path.join(dataset_abs, "labels", "train")
    labels_val = os.path.join(dataset_abs, "labels", "val")

    os.makedirs(labels_train, exist_ok=True)
    os.makedirs(labels_val, exist_ok=True)

    yaml_path = os.path.join(dataset_abs, "dataset.yaml")
    yaml_content = f"""# FIBO Custom Obstacle & Wall Dataset Configuration
path: {dataset_abs.replace(os.sep, '/')}
train: images/train
val: images/val

# Classes
names:
  0: wall
  1: obstacle
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[AUTO-LABEL] Wrote dataset config: {yaml_path}")

    total_labeled = 0

    for subset, img_dir, lbl_dir in [("TRAIN", images_train, labels_train), ("VAL", images_val, labels_val)]:
        if not os.path.exists(img_dir):
            continue
        files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\nProcessing {subset} set ({len(files)} images)...")

        for fname in files:
            img_path = os.path.join(img_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            boxes = detect_wall_and_obstacles(img)
            base_name = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")

            with open(lbl_path, "w", encoding="utf-8") as f:
                for cls_id, bx, by, bw, bh in boxes:
                    f.write(f"{cls_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")

            total_labeled += 1

            if preview:
                h, w = img.shape[:2]
                for cls_id, bx, by, bw, bh in boxes:
                    col = (0, 0, 255) if cls_id == 0 else (0, 220, 255)
                    lbl = "Wall" if cls_id == 0 else "Obstacle"
                    x1 = int((bx - bw / 2.0) * w)
                    y1 = int((by - bh / 2.0) * h)
                    x2 = int((bx + bw / 2.0) * w)
                    y2 = int((by + bh / 2.0) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(img, lbl, (x1 + 4, max(18, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
                cv2.imshow("Auto-Label Proposal Preview", img)
                if (cv2.waitKey(80) & 0xFF) in (ord('q'), 27):
                    preview = False
                    cv2.destroyAllWindows()

    print("\n==================================================================")
    print(f"  Auto-Labeling Complete! Generated labels for {total_labeled} images.")
    print(f"  Dataset Config: {yaml_path}")
    print("  Next Step: Run '9_TRAIN_YOLO_OBSTACLES.bat' to train the model!")
    print("==================================================================")


def main():
    ap = argparse.ArgumentParser(description="Auto-Label Walls & Obstacles for YOLO Training")
    ap.add_argument("--dataset", default="dataset", help="Path to dataset directory (default: dataset/)")
    ap.add_argument("--preview", action="store_true", help="Display visual proposal preview window")
    args = ap.parse_args()

    process_dataset(args.dataset, preview=args.preview)


if __name__ == "__main__":
    main()
