#!/usr/bin/env python3
"""
train_yolo_obstacles.py
Fine-tune YOLO on Custom Environment Walls & Obstacles using NVIDIA GPU (CUDA).

Outputs:
  - Best fine-tuned weights saved to: `models/yolo_fibo_walls.pt`
  - Automatic validation metrics (mAP50, mAP50-95)
  - Updates `mapping_config.json` to seamlessly use the newly trained model in navigation!
"""

import argparse
import os
import shutil
import sys
import torch
from ultralytics import YOLO

from config_manager import DEFAULT_CONFIG_PATH, load_config, save_config


def main():
    ap = argparse.ArgumentParser(description="Train Custom YOLO for Robot Wall & Obstacle Detection")
    ap.add_argument("--data", default="dataset/dataset.yaml", help="Path to dataset.yaml")
    ap.add_argument("--base-model", default="yolo11n.pt", help="Base model to fine-tune from (e.g. yolo11n.pt, yolov8n.pt)")
    ap.add_argument("--epochs", type=int, default=35, help="Number of training epochs (default: 35)")
    ap.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    ap.add_argument("--imgsz", type=int, default=640, help="Image resolution for training (default: 640)")
    ap.add_argument("--output-model", default="models/yolo_fibo_walls.pt", help="Path to save best model weights")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to mapping_config.json to update")
    args = ap.parse_args()

    data_yaml = os.path.abspath(args.data)
    if not os.path.exists(data_yaml):
        print(f"[TRAIN ERROR] Dataset configuration file not found at: {data_yaml}")
        print("Please run '7_COLLECT_TRAINING_DATA.bat' and '8_AUTO_LABEL_DATASET.bat' first!")
        sys.exit(1)

    # Check CUDA GPU
    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("==================================================================")
    print("  FIBO AI: Custom YOLO Wall & Obstacle Training")
    print(f"  Dataset Config : {data_yaml}")
    print(f"  Base Model     : {args.base_model}")
    print(f"  Hardware Device: {device} ({device_name})")
    print(f"  Epochs / Batch : {args.epochs} epochs | Batch size {args.batch}")
    print(f"  Output Model   : {args.output_model}")
    print("==================================================================")

    # Load Base Pretrained Model
    print(f"\n[1/3] Loading base model '{args.base_model}'...")
    model = YOLO(args.base_model)

    # Train Model with Transfer Learning
    print(f"\n[2/3] Starting Training on {device_name}...")
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project="runs/train_fibo",
        name="obstacle_wall",
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
        flipud=0.0,     # Walls don't appear upside down
        fliplr=0.5,     # Left-right symmetry is fine
        mosaic=1.0,     # Mosaic augmentation for small obstacles
        patience=15,    # Early stopping patience
    )

    # Check and Copy Best Model
    cand_paths = [
        os.path.join(str(getattr(results, "save_dir", "")), "weights", "best.pt"),
        os.path.join("runs", "train_fibo", "obstacle_wall", "weights", "best.pt"),
        os.path.join(os.path.dirname(os.path.dirname(data_yaml)), "runs", "detect", "runs", "train_fibo", "obstacle_wall", "weights", "best.pt"),
        r"C:\Users\suwin\Documents\FIBO\YOLO\runs\detect\runs\train_fibo\obstacle_wall\weights\best.pt",
    ]
    best_pt = next((p for p in cand_paths if os.path.exists(p)), None)
    if best_pt is not None:
        out_dir = os.path.dirname(os.path.abspath(args.output_model))
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(best_pt, args.output_model)
        print(f"\n[3/3] Successfully exported fine-tuned model to: {args.output_model}")

        # Auto-update mapping_config.json so navigate_rrt.py uses the new model
        if os.path.exists(args.config):
            cfg = load_config(args.config)
            cfg["yolo_and_planner"]["yolo_model"] = os.path.basename(args.output_model)
            save_config(cfg, args.config)
            print(f"[CONFIG] Updated '{args.config}' -> yolo_model='{os.path.basename(args.output_model)}'")

        print("\n==================================================================")
        print("  Training Completed Successfully!")
        print(f"  Saved Weights : {args.output_model}")
        print("  Classes Learned: ['wall', 'obstacle']")
        print("  You can now run '4_NAVIGATE_RRT_PI_STREAM.bat' to navigate with your custom model!")
        print("==================================================================")
    else:
        print("[TRAIN WARN] Could not locate best.pt in runs/ directory.")


if __name__ == "__main__":
    main()
