"""
scratch/patch_revert_yolo.py
Revert YOLO handling in navigate_rrt.py and yolo_mask_track.py to standard segmentation
with tracking colors, ignore 'wall' classes, and prevent giant bounding box blobs.
"""

import os

def patch_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Old block in run_yolo
    old_target = """                num_objects += 1
                color = (0, 0, 255) if "wall" in cls_name or "obstacle" in cls_name else _colour_for_id(tid)

                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))

                if has_masks and masks is not None and i < len(masks):
                    mr = cv2.resize((masks[i] > 0.5).astype(np.uint8), (w, h),
                                    interpolation=cv2.INTER_NEAREST)
                else:
                    mr = np.zeros((h, w), dtype=np.uint8)
                    mr[y1:y2, x1:x2] = 1"""

    new_target = """                if "wall" in cls_name:
                    continue

                num_objects += 1
                color = _colour_for_id(tid)

                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))

                if has_masks and masks is not None and i < len(masks):
                    mr = cv2.resize((masks[i] > 0.5).astype(np.uint8), (w, h),
                                    interpolation=cv2.INTER_NEAREST)
                else:
                    # Guard against huge bounding boxes turning the map into a red blob
                    box_w = x2 - x1
                    box_h = y2 - y1
                    if (box_w * box_h) > (w * h * 0.40):
                        continue
                    mr = np.zeros((h, w), dtype=np.uint8)
                    mr[y1:y2, x1:x2] = 1"""

    # Also fix label to ID:{tid}
    old_label = 'label = f"{model.names[int(cls_idx)]} {conf_val:.2f}"'
    new_label = 'label = f"ID:{tid} {model.names[int(cls_idx)]} {conf_val:.2f}"'

    if old_target in text:
        text = text.replace(old_target, new_target, 1)
        print(f"Patched run_yolo in {filepath}")
    else:
        print(f"Target not matched in {filepath}, trying line-ending normalized check...")
        text_norm = text.replace("\r\n", "\n")
        old_norm = old_target.replace("\r\n", "\n")
        new_norm = new_target.replace("\r\n", "\n")
        if old_norm in text_norm:
            text = text_norm.replace(old_norm, new_norm, 1)
            print(f"Patched (normalized) run_yolo in {filepath}")
        else:
            print(f"ERROR: Could not find target in {filepath}")

    if old_label in text:
        text = text.replace(old_label, new_label, 1)
        print(f"Patched label in {filepath}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

patch_file("navigate_rrt.py")
patch_file("yolo_mask_track.py")
