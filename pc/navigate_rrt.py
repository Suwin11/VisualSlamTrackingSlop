"""
navigate_rrt.py
Modular, Zero-Hardcoded-IP Autonomous Navigation & Stutter-Proof Manual Teleop Dashboard:
YOLO Dynamic Masking (with Configurable Class Filtering) + TOF400C Laser Rangefinder
+ 3D Point Cloud Height-Slicing + 3D-2D PnP Visual Odometry + Persistent 2D/3D Global Map
+ RRT* Path Planning + 25Hz Independent Manual/Auto Motor Control.
"""

import argparse
import math
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config_manager import DEFAULT_CONFIG_PATH, load_config, save_config, resolve_pi_host
from capture import LowLatencyCapture
import streamer
from detectors import ORBTracker, build_lk_pyramid
import occupancy
from visual_odometry import VisualOdometry
from stereo_visual_odometry import StereoVisualOdometry
from global_map import GlobalOccupancyGrid
from rrt_star import RRTStarPlanner
from astar import AStarPlanner
from motor_control import MotorController, PurePursuitController
from imu_receiver import BNO055Receiver
from pointcloud_3d import PointCloudMap3D


_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))


def _put_panel_label(img, text, bg=(20, 20, 20), fg=(0, 230, 255)):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 26), bg, -1)
    cv2.putText(img, text, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, fg, 1, cv2.LINE_AA)


def _colour_for_id(track_id):
    rng = np.random.default_rng(int(track_id) * 42 + 17)
    return tuple(int(c) for c in rng.integers(50, 255, size=3))


def run_yolo(model, frame, persist, conf, device, ignored_classes=None):
    h, w = frame.shape[:2]
    results = model.track(frame, persist=persist, tracker="bytetrack.yaml",
                          conf=conf, device=device, imgsz=480, verbose=False)

    ignore_set = {c.lower().strip() for c in (ignored_classes or [])}
    static_mask = np.full((h, w), 255, dtype=np.uint8)
    mask_overlay = np.zeros_like(frame)
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    yolo_detections = []
    num_objects = 0

    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            track_ids = (r.boxes.id.int().cpu().numpy()
                         if r.boxes.id is not None else [0] * len(boxes))
            has_masks = (r.masks is not None)
            masks = r.masks.data.cpu().numpy() if has_masks else None

            for i, (box, cls_idx, conf_val, tid) in enumerate(zip(
                    boxes, classes, confs, track_ids)):
                cls_name = str(model.names[int(cls_idx)]).lower().strip()
                if cls_name in ignore_set:
                    continue

                if "wall" in cls_name:
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
                    mr[y1:y2, x1:x2] = 1

                mask_overlay[mr > 0] = color
                combined_mask[mr > 0] = 255
                yolo_detections.append(((x1, y1, x2, y2), mr, int(cls_idx)))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{tid} {model.names[int(cls_idx)]} {conf_val:.2f}"
                cv2.putText(frame, label, (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            if num_objects > 0:
                dilated = cv2.dilate(combined_mask, _DILATE_KERNEL)
                static_mask[dilated > 0] = 0

    annotated = cv2.addWeighted(frame, 1.0, mask_overlay, 0.35, 0)
    return annotated, static_mask, combined_mask, num_objects, yolo_detections


def main():
    ap = argparse.ArgumentParser(description="Full 3D Point Cloud + TOF400C + RRT* Autonomous Navigation Pipeline")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to mapping_config.json")
    ap.add_argument("--source", default=None, help="Optional override for video source ('pi', '0', or RTSP URL)")
    ap.add_argument("--pi-host", default=None, help="Optional override for Raspberry Pi IP/hostname")
    ap.add_argument("--model", default=None, help="Optional override for YOLO model")
    ap.add_argument("--conf", type=float, default=None, help="Optional override for YOLO confidence")
    ap.add_argument("--fov", type=float, default=None, help="Optional override for camera FOV")
    ap.add_argument("--pitch", type=float, default=None, help="Optional override for camera pitch")
    ap.add_argument("--goal-x", type=float, default=None, help="Optional override for Goal X")
    ap.add_argument("--goal-y", type=float, default=None, help="Optional override for Goal Y")
    ap.add_argument("--web-port", type=int, default=None, help="MJPEG stream port")
    ap.add_argument("--no-display", action="store_true", help="Disable local GUI window")
    ap.add_argument("--calib", default=None, help="Optional override for camera calibration .npz file")
    ap.add_argument("--mode-2d", action="store_true", help="Start Panel 2 in 2D BEV Radar view")
    ap.add_argument("--load-map", default=None, help="Path to pre-saved map .npz (or 'latest' for newest map)")
    ap.add_argument("--planner", default=None, choices=["astar", "rrt", "rrt_star"],
                    help="Path planner: 'astar' (deterministic optimal) or 'rrt' (sampling tree)")
    args = ap.parse_args()

    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path)
    config_mtime = os.path.getmtime(config_path) if os.path.exists(config_path) else 0.0
    last_cfg_check = time.monotonic()

    if args.pi_host is not None:
        cfg["network"]["pi_host"] = args.pi_host
    if args.source is not None:
        cfg["camera"]["source"] = args.source
    if args.model is not None:
        cfg["yolo_and_planner"]["yolo_model"] = args.model
    if args.conf is not None:
        cfg["yolo_and_planner"]["yolo_conf"] = args.conf
    if args.fov is not None:
        cfg["camera"]["fov_deg"] = args.fov
    if args.pitch is not None:
        cfg["camera"]["camera_pitch_deg"] = args.pitch
    if args.calib is not None:
        cfg["camera"]["calib_file"] = args.calib
    if args.goal_x is not None:
        cfg["yolo_and_planner"]["default_goal_x_m"] = args.goal_x
    if args.goal_y is not None:
        cfg["yolo_and_planner"]["default_goal_y_m"] = args.goal_y
    if args.web_port is not None:
        cfg["network"]["web_stream_port"] = args.web_port

    net_cfg = cfg["network"]
    cam_cfg = cfg["camera"]
    imu_cfg = cfg["imu_sync"]
    tof_cfg = cfg["tof_sensor"]
    map3d_cfg = cfg["mapping_3d"]
    gmap_cfg = cfg["global_map"]
    yp_cfg = cfg["yolo_and_planner"]
    planner_type = str(args.planner or yp_cfg.get("planner_type", "astar")).lower().strip()
    if planner_type in ("rrt", "rrt_star", "rrt*"):
        planner_type = "rrt"
    else:
        planner_type = "astar"
    mot_cfg = cfg["motor_control"]

    print(f"[CONFIG] Loaded centralized settings from: {config_path}")

    source = str(cam_cfg.get("source", "pi"))
    is_webcam = source.isdigit()

    # Dynamically resolve Raspberry Pi IP (no hardcoded IP needed!)
    if not is_webcam:
        resolved_pi_ip = resolve_pi_host(cfg, config_path=config_path, save_if_changed=True)
        print(f"[NETWORK] Active Raspberry Pi Host -> {resolved_pi_ip}")
    else:
        resolved_pi_ip = str(net_cfg.get("last_known_pi_ip", "127.0.0.1"))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Compute Device: {device} ({torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'})")

    model_name = yp_cfg["yolo_model"]
    cand_paths = [
        model_name,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", model_name),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", model_name),
        os.path.join("models", model_name),
        os.path.join("pc", "models", model_name),
    ]
    model_path = next((p for p in cand_paths if os.path.exists(p)), model_name)
    print(f"[INFO] Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    print(f"[INFO] Active Detection Classes: {model.names}")

    controller = PurePursuitController(
        lookahead_m=yp_cfg.get("lookahead_m", 0.35),
        max_v=yp_cfg.get("max_linear_vel_mps", 0.30),
        max_omega=yp_cfg.get("max_angular_vel_rads", 1.20),
        wheel_base_m=mot_cfg.get("wheel_base_m", 0.28),
        wheel_radius_m=mot_cfg.get("wheel_radius_m", 0.033),
        max_rpm=mot_cfg.get("max_rpm", 60),
        min_start_rpm=mot_cfg.get("min_start_rpm", 6),
        manual_forward_rpm=mot_cfg.get("manual_forward_rpm", 32),
        manual_turn_rpm=mot_cfg.get("manual_turn_rpm", 24),
        manual_hold_timeout_sec=mot_cfg.get("manual_hold_timeout_sec", 0.35),
        control_loop_hz=mot_cfg.get("control_loop_hz", 25.0),
        udp_host=resolved_pi_ip,
        udp_port=int(net_cfg.get("motor_udp_port", 5006)),
    )

    def on_pi_ip_updated(new_ip: str):
        nonlocal resolved_pi_ip, config_mtime
        if new_ip and new_ip not in ("127.0.0.1", "0.0.0.0") and new_ip != resolved_pi_ip:
            resolved_pi_ip = new_ip
            cfg["network"]["last_known_pi_ip"] = new_ip
            controller.update_target_host(new_ip)
            imu.pi_host = new_ip
            if save_config(cfg, config_path) and os.path.exists(config_path):
                config_mtime = os.path.getmtime(config_path)
            print(f"[NETWORK] Auto-updated Raspberry Pi IP to {new_ip} across all modules!")

    imu = BNO055Receiver(
        udp_port=int(net_cfg.get("imu_udp_port", 5005)),
        pi_host=resolved_pi_ip,
        mount_pitch_deg=cam_cfg["camera_pitch_deg"],
        video_latency_sec=imu_cfg["video_latency_sec"],
        imu_mounted_on_camera=imu_cfg.get("imu_mounted_on_camera", False),
        on_ip_discovered=on_pi_ip_updated,
        imu_mount_yaw_deg=imu_cfg.get("imu_mount_yaw_deg", 0.0),
        imu_mount_pitch_deg=imu_cfg.get("imu_mount_pitch_deg", -4.4),
        imu_mount_roll_deg=imu_cfg.get("imu_mount_roll_deg", 5.7),
        auto_calibrate_mount=imu_cfg.get("auto_calibrate_mount", True),
    )

    cap = LowLatencyCapture(
        source=source,
        reconnect_timeout=float(net_cfg.get("reconnect_timeout_sec", 3.5)),
        rtsp_transport=str(net_cfg.get("rtsp_transport", "udp")),
        cfg=cfg,
        on_ip_change=on_pi_ip_updated,
    )

    web_port = int(net_cfg.get("web_stream_port", 8080))
    if web_port > 0:
        streamer.start(web_port)

    orb_tracker = ORBTracker(max_points=200, min_points=80)
    orb_trail   = None

    stereo_cfg = cfg.get("stereo_camera", {})
    enable_undistort = bool(cam_cfg.get("enable_undistort", True)) and (not is_webcam)
    imu_motion_weight = float(imu_cfg.get("imu_motion_weight", 0.45))
    depth_motion_weight = float(imu_cfg.get("depth_motion_weight", 0.35))
    cmd_motion_weight = float(imu_cfg.get("cmd_motion_weight", 0.20))
    imu_mount_yaw = float(imu_cfg.get("imu_mount_yaw_deg", 0.0))
    imu_mount_pitch = float(imu_cfg.get("imu_mount_pitch_deg", -4.4))
    imu_mount_roll = float(imu_cfg.get("imu_mount_roll_deg", 5.7))
    auto_calib_mount = bool(imu_cfg.get("auto_calibrate_mount", True))

    if bool(stereo_cfg.get("enabled", True)) and (not is_webcam):
        vo = StereoVisualOdometry(
            stereo_calib_file=stereo_cfg.get("calib_file", "stereo_calib.npz"),
            camera_height_m=cam_cfg["camera_height_m"],
            camera_pitch_deg=cam_cfg["camera_pitch_deg"],
            fov_deg=cam_cfg["fov_deg"],
            baseline_m=stereo_cfg.get("baseline_m", 0.08),
            num_disparities=stereo_cfg.get("stereo_num_disparities", 64),
            block_size=stereo_cfg.get("stereo_block_size", 7),
            yaw_scale=cam_cfg.get("yaw_scale", 1.0),
            tof_odom_blend=tof_cfg.get("tof_odometry_blend", 0.25),
            swap_left_right=stereo_cfg.get("swap_left_right", False),
            vertical_offset_px=stereo_cfg.get("vertical_offset_px", 0),
            imu_motion_weight=imu_motion_weight,
            depth_motion_weight=depth_motion_weight,
            cmd_motion_weight=cmd_motion_weight,
            imu_mount_yaw_deg=imu_mount_yaw,
            imu_mount_pitch_deg=imu_mount_pitch,
            imu_mount_roll_deg=imu_mount_roll,
            auto_calibrate_mount=auto_calib_mount,
        )
    else:
        vo = VisualOdometry(
            camera_height_m=cam_cfg["camera_height_m"],
            camera_pitch_deg=cam_cfg["camera_pitch_deg"],
            fov_deg=cam_cfg["fov_deg"],
            calib_file=cam_cfg["calib_file"],
            enable_undistort=enable_undistort,
            yaw_scale=cam_cfg.get("yaw_scale", 1.0),
            tof_odom_blend=tof_cfg.get("tof_odometry_blend", 0.35),
            imu_motion_weight=imu_motion_weight,
        )

    close_range_tof_m = float(cfg.get("tof_sensor", {}).get("max_tof_range_m", 4.0))

    pc_map = PointCloudMap3D(
        camera_height_m=cam_cfg["camera_height_m"],
        min_obs_height_m=map3d_cfg["min_obstacle_height_m"],
        max_obs_height_m=map3d_cfg["max_obstacle_height_m"],
        max_range_m=max(float(map3d_cfg.get("max_forward_range_m", 5.0)) + 0.5, 4.0),
        voxel_size_m=map3d_cfg.get("voxel_size_m", 0.05),
        max_global_points=map3d_cfg.get("max_global_points", 80000),
        wall_sensitivity=map3d_cfg["wall_sensitivity"],
        close_range_m=close_range_tof_m,
        show_ceiling=map3d_cfg.get("show_ceiling", False),
    )
    pc_map.color_by_height = bool(map3d_cfg.get("color_by_height", True))
    show_3d_panel = False if args.mode_2d else bool(map3d_cfg.get("show_3d_view_by_default", True))
    prev_depth_map = None

    global_grid = GlobalOccupancyGrid(
        width_m=gmap_cfg["width_m"],
        height_m=gmap_cfg["height_m"],
        resolution=gmap_cfg["resolution_m"],
        robot_radius_m=gmap_cfg["robot_radius_m"],
        enable_rolling=gmap_cfg.get("enable_rolling_window", True),
        l_occ=gmap_cfg.get("log_odds_occupied", 1.50),
        l_free=gmap_cfg.get("log_odds_free", -1.10),
        close_range_m=close_range_tof_m,
    )
    map_locked = bool(gmap_cfg.get("lock_map_by_default", False))
    if args.load_map is not None:
        if global_grid.load_map(args.load_map, keep_current_pose=False):
            vo.x, vo.y, vo.theta = global_grid.current_pose
            vo.frame_theta = vo.theta
            # Automatically lock map when loading a pre-saved map so it isn't overwritten!
            map_locked = True
            print(f"[MAP] Anchored odometry origin to loaded map at ({vo.x:.2f}m, {vo.y:.2f}m, {math.degrees(vo.theta):.1f}°)")
            print("[MAP] Map is LOCKED (Read-Only). Press 'K' to toggle live mapping.")

    global_grid.map_locked = map_locked
    global_grid.protect_explored_walls = bool(gmap_cfg.get("protect_explored_walls", False))
    global_grid.close_range_m = close_range_tof_m
    pc_map.close_range_m = close_range_tof_m

    astar_planner = AStarPlanner(
        global_grid,
        clearance_weight=float(yp_cfg.get("astar_clearance_weight", 0.20)),
        corner_cut_prevention=True
    )
    rrt_planner = RRTStarPlanner(global_grid, step_size=0.25, search_radius=0.55, max_iter=300)
    planner = astar_planner if planner_type == "astar" else rrt_planner

    H_ipm = None
    window_name = "FIBO AI Navigation: 3D Point Cloud + TOF400C Laser + Global Grid + A*/RRT*"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

    w, h = 640, 480
    current_goal = (float(yp_cfg["default_goal_x_m"]), float(yp_cfg["default_goal_y_m"]))
    last_plan_time = 0.0
    plan_interval = 0.5
    current_raw_path = None
    current_smooth_path = None
    current_tree_edges = []

    # Autonomous TOF escape / unstuck state tracking
    tof_stuck_start = None
    escape_active = False
    escape_until = 0.0
    escape_dir = "right"
    escape_count = 0

    last_yolo_vis = np.zeros((h, w, 3), dtype=np.uint8)
    last_local_bev = np.full((occupancy.GRID_H, occupancy.GRID_W), occupancy.UNKNOWN, dtype=np.uint8)

    prev_time = time.time()
    last_valid = time.time()

    def sync_and_save_json():
        cfg["camera"]["fov_deg"] = round(float(vo.fov_deg), 2)
        cfg["camera"]["yaw_scale"] = round(float(vo.yaw_scale), 4)
        cfg["imu_sync"]["video_latency_sec"] = round(float(imu.video_latency_sec), 3)
        cfg["mapping_3d"]["wall_sensitivity"] = round(float(pc_map.wall_sensitivity), 2)
        cfg["mapping_3d"]["show_3d_view_by_default"] = bool(show_3d_panel)
        cfg["mapping_3d"]["color_by_height"] = bool(pc_map.color_by_height)
        cfg["mapping_3d"]["show_ceiling"] = bool(pc_map.show_ceiling)
        cfg["motor_control"]["manual_forward_rpm"] = int(controller.manual_forward_rpm)
        cfg["motor_control"]["manual_turn_rpm"] = int(controller.manual_turn_rpm)
        if save_config(cfg, config_path):
            nonlocal config_mtime
            if os.path.exists(config_path):
                config_mtime = os.path.getmtime(config_path)
            print(f"[CONFIG] Saved updated parameters to {os.path.basename(config_path)}")

    def apply_hot_reloaded_config(new_cfg):
        nonlocal cfg
        cfg = new_cfg
        c_net = cfg["network"]
        c_cam = cfg["camera"]
        c_imu = cfg["imu_sync"]
        c_tof = cfg["tof_sensor"]
        c_m3d = cfg["mapping_3d"]
        c_gmp = cfg["global_map"]
        c_yp  = cfg["yolo_and_planner"]
        c_mot = cfg["motor_control"]

        new_pi_host = str(c_net.get("pi_host", "auto")).strip()
        if new_pi_host.lower() not in ("", "auto", "pi"):
            on_pi_ip_updated(new_pi_host)

        vo.camera_height = float(c_cam["camera_height_m"])
        vo.nominal_pitch_rad = math.radians(float(c_cam["camera_pitch_deg"]))
        if not vo.has_imu:
            vo.camera_pitch_rad = vo.nominal_pitch_rad
        sin_p = math.sin(vo.camera_pitch_rad)
        vo._ground_depth = vo.camera_height / max(sin_p, 0.08) if sin_p > 0.08 else 2.5
        vo.yaw_scale = float(c_cam.get("yaw_scale", 1.0))
        vo.tof_odom_blend = float(c_tof.get("tof_odometry_blend", 0.35))
        vo.imu_motion_weight = float(c_imu.get("imu_motion_weight", 0.45))
        if hasattr(vo, "depth_motion_weight"):
            vo.depth_motion_weight = float(c_imu.get("depth_motion_weight", 0.35))
            vo.cmd_motion_weight = float(c_imu.get("cmd_motion_weight", 0.20))
            vo.imu_mount_yaw_deg = float(c_imu.get("imu_mount_yaw_deg", 0.0))
            vo.imu_mount_pitch_deg = float(c_imu.get("imu_mount_pitch_deg", -4.4))
            vo.imu_mount_roll_deg = float(c_imu.get("imu_mount_roll_deg", 5.7))

        vo._update_focal_from_fov(float(c_cam["fov_deg"]))

        imu.mount_pitch_rad = math.radians(float(c_cam["camera_pitch_deg"]))
        imu.video_latency_sec = float(c_imu["video_latency_sec"])
        imu.imu_mounted_on_camera = bool(c_imu.get("imu_mounted_on_camera", False))
        imu.imu_mount_yaw_deg = float(c_imu.get("imu_mount_yaw_deg", 0.0))
        imu.imu_mount_pitch_deg = float(c_imu.get("imu_mount_pitch_deg", -4.4))
        imu.imu_mount_roll_deg = float(c_imu.get("imu_mount_roll_deg", 5.7))

        pc_map.camera_height = float(c_cam["camera_height_m"])
        pc_map.min_obs_z = float(c_m3d["min_obstacle_height_m"])
        pc_map.max_obs_z = float(c_m3d["max_obstacle_height_m"])
        pc_map.max_range_m = max(float(c_m3d["max_forward_range_m"]) + 0.5, 3.5)
        pc_map.voxel_size = float(c_m3d.get("voxel_size_m", 0.05))
        pc_map.wall_sensitivity = float(c_m3d["wall_sensitivity"])

        global_grid.l_occ = float(c_gmp.get("log_odds_occupied", 1.50))
        global_grid.l_free = float(c_gmp.get("log_odds_free", -1.10))
        global_grid.enable_rolling = bool(c_gmp.get("enable_rolling_window", True))
        global_grid.set_robot_radius(float(c_gmp["robot_radius_m"]))

        controller.lookahead_m = float(c_yp.get("lookahead_m", 0.35))
        controller.max_v = float(c_yp.get("max_linear_vel_mps", 0.30))
        controller.max_omega = float(c_yp.get("max_angular_vel_rads", 1.20))
        controller.wheel_base_m = float(c_mot.get("wheel_base_m", 0.28))
        controller.wheel_radius_m = float(c_mot.get("wheel_radius_m", 0.033))
        controller.max_rpm = int(c_mot.get("max_rpm", 60))
        controller.manual_forward_rpm = int(c_mot.get("manual_forward_rpm", 32))
        controller.manual_turn_rpm = int(c_mot.get("manual_turn_rpm", 24))
        nonlocal planner_type, planner
        new_p_type = str(c_yp.get("planner_type", planner_type)).lower().strip()
        if new_p_type in ("rrt", "rrt_star", "rrt*"):
            new_p_type = "rrt"
        else:
            new_p_type = "astar"
        if new_p_type != planner_type:
            planner_type = new_p_type
            planner = astar_planner if planner_type == "astar" else rrt_planner
            current_smooth_path = None
            current_raw_path = None
            last_plan_time = 0.0
            print(f"[CONFIG] Switched planner from hot-reload to: {planner_type.upper()}")
        print(f"[CONFIG] Hot-reloaded {os.path.basename(config_path)}!")

    is_panning = False
    pan_last_pt = None
    lbutton_down = False
    lbutton_start_pt = None
    lbutton_dragged = False
    reloc_start_pt = None
    map_status_banner = None
    map_status_time = 0.0
    dynamic_marking = True

    def on_mouse(event, x, y, flags, param):
        nonlocal current_goal, last_plan_time, current_smooth_path, reloc_start_pt, map_status_banner, map_status_time
        nonlocal is_panning, pan_last_pt, lbutton_down, lbutton_start_pt, lbutton_dragged

        # Reset panning if any pan button was released
        if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            if is_panning:
                is_panning = False
                pan_last_pt = None

        # Only process mouse interactions within the bottom two map panels (y >= h)
        if y >= h and w > 0 and h > 0:
            panel_x = x % w
            panel_y = y - h

            # 1. Mouse Wheel Zoom (centered at mouse cursor)
            if event == cv2.EVENT_MOUSEWHEEL:
                factor = 1.15 if flags > 0 else (1.0 / 1.15)
                global_grid.zoom_viewport(factor, pivot_sx=panel_x, pivot_sy=panel_y, out_w=w, out_h=h)
                map_status_banner = f"ZOOM: {global_grid.view_zoom:.2f}x (Scroll wheel / Drag to Pan)"
                map_status_time = time.time()
                return

            # 2. Middle Button Drag: Universal CAD/GIS Pan
            if event == cv2.EVENT_MBUTTONDOWN:
                is_panning = True
                pan_last_pt = (x, y)
                return

            # 3. Shift + Left Click: Explicit Pan for laptops
            if (flags & cv2.EVENT_FLAG_SHIFTKEY) and event == cv2.EVENT_LBUTTONDOWN:
                is_panning = True
                pan_last_pt = (x, y)
                return

            # 4. Normal Left Button Down: Could be Click (Goal) or Drag (Pan)
            if event == cv2.EVENT_LBUTTONDOWN:
                lbutton_down = True
                lbutton_start_pt = (x, y)
                pan_last_pt = (x, y)
                lbutton_dragged = False
                return

            # 5. Mouse Move
            elif event == cv2.EVENT_MOUSEMOVE:
                if is_panning:
                    # Cancel pan if no buttons are held
                    if not (flags & (cv2.EVENT_FLAG_MBUTTON | cv2.EVENT_FLAG_RBUTTON | cv2.EVENT_FLAG_LBUTTON)):
                        is_panning = False
                        pan_last_pt = None
                        return
                    if pan_last_pt is not None:
                        dx = x - pan_last_pt[0]
                        dy = y - pan_last_pt[1]
                        global_grid.pan_viewport(dx, dy, out_w=w, out_h=h)
                        pan_last_pt = (x, y)
                    return
                elif lbutton_down and lbutton_start_pt is not None and pan_last_pt is not None:
                    dist = math.hypot(x - lbutton_start_pt[0], y - lbutton_start_pt[1])
                    if dist > 6 or lbutton_dragged:
                        lbutton_dragged = True
                        dx = x - pan_last_pt[0]
                        dy = y - pan_last_pt[1]
                        global_grid.pan_viewport(dx, dy, out_w=w, out_h=h)
                        pan_last_pt = (x, y)
                    return

            # 6. Left Button Up
            elif event == cv2.EVENT_LBUTTONUP:
                if is_panning:
                    is_panning = False
                    pan_last_pt = None
                    return
                if lbutton_down:
                    lbutton_down = False
                    pan_last_pt = None
                    if not lbutton_dragged:
                        # Stationary Click: Set Target Goal (Auto-expand canvas if clicked in unexplored space)
                        wx, wy = global_grid.screen_to_world(panel_x, panel_y, out_w=w, out_h=h)
                        current_goal = (round(wx, 2), round(wy, 2))
                        global_grid.ensure_bounds_include(current_goal[0], current_goal[1], margin_m=3.0)
                        current_smooth_path = None
                        last_plan_time = 0.0
                        map_status_banner = f"GOAL: ({current_goal[0]:.2f}m, {current_goal[1]:.2f}m)"
                        map_status_time = time.time()
                        print(f"[GOAL] New target set by click: ({current_goal[0]}m, {current_goal[1]}m)")
                    lbutton_dragged = False
                    return

            # 7. Right Click Down: Start Robot Relocalization (pick position)
            elif event == cv2.EVENT_RBUTTONDOWN:
                wx, wy = global_grid.screen_to_world(panel_x, panel_y, out_w=w, out_h=h)
                reloc_start_pt = (round(wx, 2), round(wy, 2))

            # 8. Right Click Up: Finish Relocalization (if dragged, set heading)
            elif event == cv2.EVENT_RBUTTONUP:
                if reloc_start_pt is not None:
                    wx, wy = global_grid.screen_to_world(panel_x, panel_y, out_w=w, out_h=h)
                    rx, ry = reloc_start_pt
                    dx = wx - rx
                    dy = wy - ry
                    if math.hypot(dx, dy) > 0.15:
                        rtheta = math.atan2(dy, dx)
                    else:
                        rtheta = vo.theta  # Keep current orientation

                    vo.x = float(rx)
                    vo.y = float(ry)
                    vo.theta = float(rtheta)
                    vo.frame_theta = float(rtheta)
                    if vo.has_imu and getattr(vo, "_imu_last_yaw", None) is not None:
                        vo._imu_yaw_offset = (rtheta - vo._imu_last_yaw + math.pi) % (2.0 * math.pi) - math.pi
                        if getattr(vo, "_imu_last_live_yaw", None) is not None:
                            vo._imu_last_live_yaw = vo._imu_last_yaw

                    global_grid.set_robot_pose(vo.x, vo.y, vo.theta)
                    current_smooth_path = None
                    last_plan_time = 0.0
                    map_status_banner = f"RELOCALIZED: ({vo.x:.2f}m, {vo.y:.2f}m, {math.degrees(vo.theta):.0f}deg)"
                    map_status_time = time.time()
                    print(f"[RELOCALIZE] Robot relocalized to: ({vo.x:.2f}m, {vo.y:.2f}m, {math.degrees(vo.theta):.1f}°)")
                    reloc_start_pt = None

    if not args.no_display:
        cv2.setMouseCallback(window_name, on_mouse)

    def handle_key(key: int) -> bool:
        """
        Unified keyboard handler called BOTH during normal frames AND during camera stutters!
        Returns False if the user pressed Q/ESC to quit.
        """
        nonlocal show_3d_panel, current_goal, current_raw_path, current_smooth_path, last_plan_time
        nonlocal map_locked, map_status_banner, map_status_time, planner_type, planner, dynamic_marking
        if key == 255 or key < 0:
            return True
        if key in (ord('q'), 27):
            return False
        elif key in (ord('b'), ord('B')):
            dynamic_marking = not dynamic_marking
            dm_label = "ENABLED (Red dots mark moving objects)" if dynamic_marking else "DISABLED (Static points)"
            map_status_banner = f"DYN MARKING: {'ON (Red Dots)' if dynamic_marking else 'OFF'}"
            map_status_time = time.time()
            print(f"[TEST] Dynamic Marking -> {dm_label}")
        elif key in (ord(' '), ord('p'), ord('P')):
            controller.toggle_pause()
        elif key in (ord('t'), ord('T')):
            controller.toggle_mode()
        elif key in (ord('x'), ord('X')):
            controller.send_manual_rpm(0, 0, duration_sec=0.2)
            print("[MANUAL TELEOP] Emergency / Manual STOP (0, 0)")
        elif key == ord('w'):
            f_rpm = controller.manual_forward_rpm
            controller.send_manual_rpm(+f_rpm, +f_rpm)
            if is_webcam:
                vo.x += 0.12 * math.cos(vo.theta)
                vo.y += 0.12 * math.sin(vo.theta)
                last_plan_time = 0.0
        elif key == ord('s'):
            f_rpm = controller.manual_forward_rpm
            controller.send_manual_rpm(-f_rpm, -f_rpm)
            if is_webcam:
                vo.x -= 0.10 * math.cos(vo.theta)
                vo.y -= 0.10 * math.sin(vo.theta)
                last_plan_time = 0.0
        elif key == ord('a'):
            t_rpm = controller.manual_turn_rpm
            controller.send_manual_rpm(-t_rpm, +t_rpm)
            if is_webcam:
                vo.theta = (vo.theta + 0.15 + math.pi) % (2.0 * math.pi) - math.pi
                vo.frame_theta = vo.theta
                last_plan_time = 0.0
        elif key == ord('d'):
            t_rpm = controller.manual_turn_rpm
            controller.send_manual_rpm(+t_rpm, -t_rpm)
            if is_webcam:
                vo.theta = (vo.theta - 0.15 + math.pi) % (2.0 * math.pi) - math.pi
                vo.frame_theta = vo.theta
                last_plan_time = 0.0
        # Sticky Cruise Latch Keys (1/2/3/4: tap once to cruise without holding key!)
        elif key == ord('1'):
            f_rpm = controller.manual_forward_rpm
            controller.set_latched_rpm(+f_rpm, +f_rpm)
        elif key == ord('2'):
            f_rpm = controller.manual_forward_rpm
            controller.set_latched_rpm(-f_rpm, -f_rpm)
        elif key == ord('3'):
            t_rpm = controller.manual_turn_rpm
            controller.set_latched_rpm(-t_rpm, +t_rpm)
        elif key == ord('4'):
            t_rpm = controller.manual_turn_rpm
            controller.set_latched_rpm(+t_rpm, -t_rpm)
        elif key == ord('9'):
            controller.adjust_manual_speed(-5)
            sync_and_save_json()
        elif key == ord('0'):
            controller.adjust_manual_speed(+5)
            sync_and_save_json()
        elif key in (ord('n'), ord('N')):
            planner_type = 'rrt' if planner_type == 'astar' else 'astar'
            planner = astar_planner if planner_type == 'astar' else rrt_planner
            current_smooth_path = None
            current_raw_path = None
            last_plan_time = 0.0
            cfg['yolo_and_planner']['planner_type'] = planner_type
            p_desc = 'A* (Optimal Shortest)' if planner_type == 'astar' else 'RRT* (Sampling Tree)'
            map_status_banner = f'PLANNER: {p_desc}'
            map_status_time = time.time()
            print(f'[PLANNER] Switched active path planner to: {p_desc}')
        elif key in (ord('v'), ord('V')):
            show_3d_panel = not show_3d_panel
            sync_and_save_json()
            mode_str = "3D Point Cloud View" if show_3d_panel else "2.5D Radar BEV View"
            map_status_banner = f"PANEL 2: {mode_str}"
            map_status_time = time.time()
            print(f"[PANEL] Panel 2 view switched to: {mode_str}")
        elif key in (ord('m'), ord('M')):
            mode_name = pc_map.cycle_color_mode()
            sync_and_save_json()
            map_status_banner = f"3D COLOR: {mode_name}"
            map_status_time = time.time()
            print(f"[3D VIEW] Color mode switched to: {mode_name}")
        elif key in (ord('h'), ord('H')):
            c_vis = pc_map.toggle_ceiling()
            sync_and_save_json()
            c_txt = "SHOWN" if c_vis else "HIDDEN (Floor+Obs Only)"
            map_status_banner = f"3D CEILING: {c_txt}"
            map_status_time = time.time()
        elif key == ord('j'):
            pc_map.adjust_orbit(dyaw_deg=-8.0)
        elif key == ord('l'):
            pc_map.adjust_orbit(dyaw_deg=+8.0)
        elif key == ord('i'):
            pc_map.adjust_orbit(dpitch_deg=+5.0)
        elif key == ord('k'):
            pc_map.adjust_orbit(dpitch_deg=-5.0)
        elif key in (ord(','), ord('<')):
            imu.adjust_video_latency(-0.010)
            sync_and_save_json()
        elif key in (ord('.'), ord('>')):
            imu.adjust_video_latency(+0.010)
            sync_and_save_json()
        elif key in (ord('-'), ord('_')):
            pc_map.adjust_wall_sensitivity(-0.15)
            sync_and_save_json()
        elif key in (ord('='), ord('+')):
            pc_map.adjust_wall_sensitivity(+0.15)
            sync_and_save_json()
        elif key == ord('r'):
            global_grid.reset()
            pc_map.reset()
            vo.reset()
            current_raw_path = None
            current_smooth_path = None
            last_plan_time = 0.0
            print("[INFO] Global 2D Map, 3D Point Cloud & Odometry Reset.")
        elif key == ord('R'):
            global_grid.reset_viewport()
            map_status_banner = "MAP VIEWPORT RESET (1.0x Full Map)"
            map_status_time = time.time()
            print("[VIEWPORT] Map viewport reset to full overview.")
        elif key in (ord('f'), ord('F')):
            global_grid.follow_robot = not global_grid.follow_robot
            global_grid.center_on_robot()
            f_str = "ON" if global_grid.follow_robot else "OFF"
            map_status_banner = f"FOLLOW ROBOT: {f_str} ({vo.x:.2f}m, {vo.y:.2f}m)"
            map_status_time = time.time()
            print(f"[VIEWPORT] Centered on robot ({vo.x:.2f}m, {vo.y:.2f}m). Auto-follow: {f_str}")
        elif key in (ord('c'), ord('C')):
            still_cal = vo.toggle_180_calibration()
            if not still_cal:
                sync_and_save_json()
        elif key == ord('['):
            vo.adjust_fov(-2.0)
            sync_and_save_json()
        elif key == ord(']'):
            vo.adjust_fov(+2.0)
            sync_and_save_json()
        elif key in (ord('o'), ord('O')):
            saved_f = global_grid.save_map()
            map_status_banner = f"SAVED: {os.path.basename(saved_f)}"
            map_status_time = time.time()
            print(f"[MAP] Saved live global grid to '{saved_f}'")
        elif key in (ord('k'), ord('K')):
            map_locked = not map_locked
            global_grid.map_locked = map_locked
            lock_label = "LOCKED (Explored Map Protected - Read-Only)" if map_locked else "UNLOCKED (Live Exploring)"
            map_status_banner = f"MAP: {lock_label}"
            map_status_time = time.time()
            print(f"[MAP] Map Protection -> {lock_label}")
        elif key in (ord('u'), ord('U'), ord('l'), ord('L')):
            map_path = None
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                save_dir = os.path.abspath("saved_maps")
                map_path = filedialog.askopenfilename(
                    initialdir=save_dir if os.path.exists(save_dir) else ".",
                    title="Open Occupancy Map for Navigation - FIBO",
                    filetypes=[
                        ("All Supported Maps (*.npz; *.json)", "*.npz;*.json"),
                        ("Occupancy Maps (*.npz)", "*.npz"),
                        ("JSON Floorplans (*.json)", "*.json"),
                        ("All Files (*.*)", "*.*")
                    ]
                )
                root.destroy()
            except Exception as e:
                print(f"[MAP-LOAD] File dialog error: {e}")
                map_path = None

            target = map_path if (map_path and os.path.exists(map_path)) else 'latest'
            if global_grid.load_map(target, keep_current_pose=False):
                vo.x, vo.y, vo.theta = global_grid.current_pose
                vo.frame_theta = vo.theta
                if vo.has_imu and getattr(vo, "_imu_last_yaw", None) is not None:
                    vo._imu_yaw_offset = (vo.theta - vo._imu_last_yaw + math.pi) % (2.0 * math.pi) - math.pi
                    if getattr(vo, "_imu_last_live_yaw", None) is not None:
                        vo._imu_last_live_yaw = vo._imu_last_yaw
                current_smooth_path = None
                last_plan_time = 0.0
                # Automatically lock map when loaded so live camera does not overwrite explored geometry!
                map_locked = True
                global_grid.map_locked = True
                fname = os.path.basename(target) if target != 'latest' else 'latest_map.npz'
                map_status_banner = f"LOADED: {fname} [MAP LOCKED - Press 'K' to Unlock]"
                map_status_time = time.time()
                print(f"[MAP] Loaded map '{target}' and aligned robot pose to ({vo.x:.2f}m, {vo.y:.2f}m, {math.degrees(vo.theta):.0f}°)")
                print("[MAP] Explored map is now LOCKED (Read-Only). Press 'K' to unlock live mapping.")
        elif key == ord('g'):
            goals = [(0.0, 2.5), (1.5, 2.0), (-1.5, 2.0), (0.0, 3.5)]
            idx = (goals.index(current_goal) + 1) % len(goals) if current_goal in goals else 0
            current_goal = goals[idx]
            current_smooth_path = None
            last_plan_time = 0.0
            print(f"[INFO] Goal switched to: {current_goal}")
        return True

    print("=" * 74)
    print(" [FIBO AUTONOMOUS & TELEOP NAVIGATION READY]")
    p_name = "A* (Optimal Shortest)" if planner_type == "astar" else "RRT* (Sampling Tree)"
    print(f"  * Pi IP Address    : {resolved_pi_ip} (Auto-Discovery Active)")
    print(f"  * Path Planner     : {p_name} (Tap 'N' to switch A* <-> RRT*)")
    print("  * Hold Teleop      : Hold W/A/S/D or Arrow Keys (Zero-Stutter 25Hz Thread!)")
    print("  * Cruise Teleop    : Tap 1(Fwd), 2(Back), 3(Left), 4(Right), X(Stop)")
    print("  * Speed Gear       : Tap 9 (-5 RPM) / 0 (+5 RPM)")
    print("  * Mode / Pause     : SPACE/P = Pause/Resume | T = Cycle AUTO/MANUAL/PAUSED")
    print("  * Dynamic Marking  : Tap B = Toggle Dynamic Marking (Red dots on moving objects)")
    print("=" * 74)

    try:
        while True:
            mono_now = time.monotonic()
            if mono_now - last_cfg_check >= 1.0:
                last_cfg_check = mono_now
                if os.path.exists(config_path):
                    mtime = os.path.getmtime(config_path)
                    if mtime > config_mtime + 0.05:
                        config_mtime = mtime
                        apply_hot_reloaded_config(load_config(config_path))

            ret, frame, interrupted = cap.read(timeout=0.025)
            now = time.time()

            # --- Live IMU + ToF + Stutter-Proof Manual Teleop Coasting ---
            if not ret or frame is None:
                live_imu = imu.get_latest_live()
                cmd_v = controller.get_linear_velocity()
                if live_imu is not None:
                    robot_pose = vo.update_imu_only(
                        live_yaw_rad=live_imu["yaw_rad"],
                        pitch_rad=live_imu["pitch_rad"],
                        gyro_yaw_rate=live_imu["gz_rads"],
                        live_tof_m=live_imu.get("tof_m", -1.0),
                        lin_acc=live_imu.get("lin_acc", 0.0),
                        cmd_v=cmd_v,
                        raw_accel=live_imu.get("raw_accel"),
                        raw_gyro=live_imu.get("raw_gyro"),
                        window_max_gz=live_imu.get("window_max_gz"),
                        window_max_acc=live_imu.get("window_max_acc"),
                    )
                    global_grid.current_pose = robot_pose
                    live_t = live_imu.get("tof_m", -1.0)
                    if (not map_locked) and 0.04 <= live_t <= 4.0:
                        global_grid.update_from_tof(robot_pose, live_t, max_range_m=max_tof_range, beam_deg=tof_beam_deg)
                else:
                    robot_pose = (vo.x, vo.y, vo.theta)

                if now - last_valid > 1.0:
                    ph = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(ph, f"[CAMERA STUTTER - 25Hz TELEOP & IMU ACTIVE ({resolved_pi_ip})]", (18, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 2)
                    last_yolo_vis = ph

                tof_stop_dist = float(cfg["tof_sensor"].get("tof_stop_distance_m", 0.22))
                active_tof = vo.latest_tof_m if vo.has_tof else None
                tof_emerg = (active_tof is not None) and (active_tof < tof_stop_dist)
                cmd = controller.compute_command(
                    robot_pose, current_smooth_path, goal=current_goal,
                    emergency_stop=(tof_emerg and not escape_active)
                )

                if show_3d_panel:
                    panel2_vis = pc_map.render_3d_view(
                        robot_pose, path=current_smooth_path, goal=current_goal, out_w=w, out_h=h,
                        tof_distance_m=active_tof
                    )
                else:
                    panel2_vis = occupancy.render(last_local_bev, out_w=w, out_h=h, tof_distance_m=active_tof, max_range_m=max_tof_range)
                global_vis = global_grid.render_map(
                    out_w=w, out_h=h, show_trajectory=True,
                    path=current_smooth_path, goal=current_goal,
                    robot_pose=robot_pose, lookahead_pt=cmd["lookahead"]
                )
                pwm_gear = getattr(controller, "manual_forward_pwm", getattr(controller, "manual_forward_rpm", 140))
                lock_badge = "[MAP: LOCKED (Protected)]" if map_locked else "[MAP: LIVE]"
                badge_bg = (15, 30, 60) if map_locked else (20, 20, 20)
                badge_fg = (0, 220, 255) if map_locked else (0, 230, 255)
                if map_status_banner and (time.time() - map_status_time) < 4.0:
                    _put_panel_label(global_vis, f"*** {map_status_banner} *** | Drag:Pan | Wheel:Zoom", bg=(15, 60, 25), fg=(0, 255, 130))
                else:
                    _put_panel_label(global_vis, f"Global 2D Map {lock_badge} (Drag:Pan | Wheel:Zoom | R-Clk:Reloc | 'N':A*/RRT* | 'K':Lock | 'L':Load | 'O':Save) | Gear:{pwm_gear}PWM", bg=badge_bg, fg=badge_fg)
                plan_vis = global_grid.render_map(
                    out_w=w, out_h=h, show_trajectory=False,
                    path=current_smooth_path, tree_edges=current_tree_edges,
                    goal=current_goal, robot_pose=robot_pose, lookahead_pt=cmd["lookahead"]
                )
                mode_str = cmd.get("mode", controller.control_mode)
                pwm_info = f"PWM: L={cmd.get('pwm_left', 0):+d} R={cmd.get('pwm_right', 0):+d}"
                _put_panel_label(
                    plan_vis,
                    f"[{mode_str}] {pwm_info} (HW:{cmd.get('rpm_left', 0):+d},{cmd.get('rpm_right', 0):+d}) (WASD Active!)"
                )
                composite = np.vstack([np.hstack([last_yolo_vis, panel2_vis]), np.hstack([global_vis, plan_vis])])
                streamer.push_frame(composite)
                if not args.no_display:
                    try:
                        cv2.imshow(window_name, composite)
                        key = cv2.waitKey(1) & 0xFF
                        if not handle_key(key):
                            break
                    except Exception:
                        pass
                continue

            # --- Camera Frame Arrived: Align to Exact Hardware Capture Timestamp ---
            last_valid = now
            frame_mono = cap.last_frame_mono
            frame_dt = cap.frame_dt
            max_stutter_dt = float(cfg["imu_sync"].get("max_stutter_dt_sec", 0.280))
            max_turn_rads = float(cfg["imu_sync"].get("max_turn_rate_for_mapping_rads", 0.45))
            frame_stuttered = bool(interrupted or frame_dt > max_stutter_dt)

            imu_sample = imu.get_synced_sample(frame_mono=frame_mono)

            # Split Side-by-Side Stereo Frame (1280x480 -> Left 640x480 + Right 640x480)
            stereo_depth_map = None
            if frame.shape[1] >= 1000 and hasattr(vo, "compute_depth_map"):
                mid = frame.shape[1] // 2
                frame_left_raw = frame[:, :mid]
                frame_right_raw = frame[:, mid:]
                stereo_depth_map, frame, _ = vo.compute_depth_map(frame_left_raw, frame_right_raw)
            else:
                frame = vo.undistort(frame)
            h, w = frame.shape[:2]

            if interrupted or orb_trail is None or orb_trail.shape[:2] != (h, w):
                orb_tracker.reset()
                orb_trail = np.zeros((h, w, 3), dtype=np.uint8)
                prev_depth_map = None

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_pyr = build_lk_pyramid(gray)

            # 1. YOLO Segmentation & Object Tracking (with Ignored Classes Filter)
            yolo_vis, static_mask, combined_mask, n_obj, yolo_dets = run_yolo(
                model, frame.copy(), persist=not interrupted,
                conf=float(cfg["yolo_and_planner"]["yolo_conf"]),
                device=device,
                ignored_classes=cfg["yolo_and_planner"].get("ignored_classes", []),
            )

            # 2. Track Background Features (ORB)
            orb_old, orb_new = orb_tracker.update(gray, static_mask, curr_pyr=curr_pyr)
            ORBTracker.draw(yolo_vis, orb_old, orb_new, orb_trail, dynamic_marking=dynamic_marking, object_mask=combined_mask)

            # 3. Synchronized IMU + TOF400C + 3D-PnP Visual Odometry update
            dt = max(now - prev_time, 0.001)
            prev_time = now
            fps = 1.0 / dt

            tof_enabled = bool(cfg["tof_sensor"].get("enabled", True))
            max_tof_range = float(cfg["tof_sensor"].get("max_tof_range_m", 4.0))
            tof_beam_deg = float(cfg["tof_sensor"].get("tof_beam_cone_deg", 15.0))
            tof_stop_dist = float(cfg["tof_sensor"].get("tof_stop_distance_m", 0.22))

            aligned_tof_m = None
            live_tof_m = None
            if imu_sample is not None:
                raw_tof = imu_sample.get("tof_m", -1.0)
                raw_live_tof = imu_sample.get("live_tof_m", raw_tof)
                if tof_enabled and raw_tof >= 0.03:
                    aligned_tof_m = float(raw_tof)
                if tof_enabled and raw_live_tof >= 0.03:
                    live_tof_m = float(raw_live_tof)

                vo.feed_imu(
                    yaw_rad=imu_sample["yaw_rad"],
                    pitch_rad=imu_sample["pitch_rad"],
                    roll_rad=imu_sample["roll_rad"],
                    gyro_yaw_rate=imu_sample["gz_rads"],
                    lin_acc_mag=imu_sample["lin_acc"],
                    dt=frame_dt,
                    reset=imu_sample.get("reset", False),
                    calib=imu_sample.get("calib", 3),
                    live_yaw_rad=imu_sample.get("live_yaw_rad"),
                    window_max_gz=imu_sample.get("window_max_gz"),
                    tof_m=aligned_tof_m if aligned_tof_m is not None else -1.0,
                    raw_accel=imu_sample.get("raw_accel"),
                    raw_gyro=imu_sample.get("raw_gyro"),
                    window_max_acc=imu_sample.get("window_max_acc"),
                )
                H_ipm = occupancy.dynamic_ipm(
                    img_w=w, img_h=h, pitch_rad=vo.camera_pitch_rad,
                    nominal_pitch_rad=getattr(vo, "nominal_pitch_rad", 0.0), fy=vo.fy
                )
            elif H_ipm is None:
                H_ipm = occupancy.default_ipm(img_w=w, img_h=h)

            cmd_v = controller.get_linear_velocity()
            robot_pose, (dx, dy, dtheta) = vo.update(
                orb_old, orb_new, dt=frame_dt, prev_depth_map=prev_depth_map,
                frame_stuttered=frame_stuttered, cmd_v=cmd_v
            )
            frame_pose = vo.get_frame_pose()
            mapping_safe = vo.is_mapping_safe(
                frame_stuttered=frame_stuttered, max_turn_rate_rads=max_turn_rads
            )

            if aligned_tof_m is not None:
                cx_i, cy_i = int(vo.cx), int(vo.cy)
                t_col = (0, 0, 255) if aligned_tof_m < tof_stop_dist else (255, 0, 255)
                cv2.circle(yolo_vis, (cx_i, cy_i), 10, t_col, 2, cv2.LINE_AA)
                cv2.line(yolo_vis, (cx_i - 16, cy_i), (cx_i + 16, cy_i), t_col, 1, cv2.LINE_AA)
                cv2.line(yolo_vis, (cx_i, cy_i - 16), (cx_i, cy_i + 16), t_col, 1, cv2.LINE_AA)
                cv2.putText(yolo_vis, f"ToF: {aligned_tof_m:.2f}m", (cx_i + 14, cy_i - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, t_col, 2, cv2.LINE_AA)

            # 4. 3D Point Cloud Reconstruction & Height-Band + Stereo + TOF400C Slicing
            depth_map = pc_map.estimate_metric_depth(
                frame, vo.camera_pitch_rad, vo.fx, vo.fy, vo.cx, vo.cy,
                yolo_detections=yolo_dets, tof_distance_m=aligned_tof_m
            )
            if stereo_depth_map is not None:
                valid_s = ~np.isnan(stereo_depth_map)
                depth_map[valid_s] = stereo_depth_map[valid_s]
            prev_depth_map = depth_map

            max_fwd_range = float(cfg["mapping_3d"].get("max_forward_range_m", 4.0))
            local_range_y = (0.0, max_fwd_range)
            bev_3d_slice = pc_map.update_and_slice_bev(
                frame, depth_map, frame_pose, vo.camera_pitch_rad,
                vo.fx, vo.fy, vo.cx, vo.cy,
                local_range_y=local_range_y,
                accumulate_global=mapping_safe,
                tof_distance_m=aligned_tof_m,
                tof_beam_deg=tof_beam_deg,
            )
            bev_ipm = occupancy.compute(
                combined_mask, orb_new, (h, w), H_ipm, frame_gray=gray,
                wall_sensitivity=pc_map.wall_sensitivity,
                tof_distance_m=aligned_tof_m,
                max_range_m=max_fwd_range,
                tof_beam_deg=tof_beam_deg,
            )
            local_bev = bev_3d_slice.copy()
            local_bev[bev_ipm == occupancy.OCCUPIED] = occupancy.OCCUPIED
            last_local_bev = local_bev

            # 5. Global Map Integration: Camera maps objects & walls; TOF carves path and stamps laser hits
            if mapping_safe and (not map_locked):
                global_grid.update_from_local_bev(
                    local_bev, frame_pose, local_range_y=local_range_y
                )
                if aligned_tof_m is not None:
                    global_grid.update_from_tof(
                        frame_pose, aligned_tof_m, max_range_m=max_tof_range, beam_deg=tof_beam_deg
                    )
            global_grid.current_pose = robot_pose

            # 6. Event-Driven Path Planning & Autonomous TOF Escape Maneuver
            tof_cfg = cfg.get("tof_sensor", {})
            auto_escape_on = bool(tof_cfg.get("auto_escape_enabled", True))
            escape_delay = float(tof_cfg.get("escape_trigger_delay_sec", 0.35))
            escape_duration = float(tof_cfg.get("escape_pivot_duration_sec", 0.65))
            escape_margin = float(tof_cfg.get("escape_clear_margin_m", 0.12))
            escape_pref = str(tof_cfg.get("escape_direction", "auto")).lower().strip()

            tof_emergency = (live_tof_m is not None) and (live_tof_m < tof_stop_dist)

            # Autonomous Escape State Machine: pivot to swing TOF laser away if stuck facing wall
            active_ctrl_mode = cmd.get("mode", controller.control_mode) if 'cmd' in locals() else controller.control_mode
            if auto_escape_on and active_ctrl_mode in ("AUTO", "EMERGENCY_PIVOT"):
                if tof_emergency:
                    if tof_stuck_start is None:
                        tof_stuck_start = now
                    elif (now - tof_stuck_start >= escape_delay) and not escape_active:
                        # Determine pivot direction (check free clearance in 2D occupancy grid)
                        if escape_pref == "right":
                            chosen_dir = "right"
                        elif escape_pref == "left":
                            chosen_dir = "left"
                        else:
                            rx, ry, rtheta = robot_pose
                            r_th = rtheta - math.radians(50.0)
                            l_th = rtheta + math.radians(50.0)
                            pt_r = (rx + 0.65 * math.cos(r_th), ry + 0.65 * math.sin(r_th))
                            pt_l = (rx + 0.65 * math.cos(l_th), ry + 0.65 * math.sin(l_th))
                            r_free = global_grid.is_collision_free(pt_r[0], pt_r[1])
                            l_free = global_grid.is_collision_free(pt_l[0], pt_l[1])
                            if r_free and not l_free:
                                chosen_dir = "right"
                            elif l_free and not r_free:
                                chosen_dir = "left"
                            else:
                                chosen_dir = "right" if (escape_count % 2 == 0) else "left"

                        escape_active = True
                        escape_dir = chosen_dir
                        escape_until = now + escape_duration
                        escape_count += 1
                        controller.trigger_escape_pivot(direction=escape_dir, duration_sec=escape_duration)
                        map_status_banner = f"TOF UNSTUCK: Pivoting {escape_dir.upper()} ~45deg..."
                        map_status_time = time.time()
                        print(f"[UNSTUCK] Obstacle at {live_tof_m:.2f}m (<{tof_stop_dist:.2f}m). Auto-pivoting {escape_dir.upper()} to clear...")
                else:
                    if not escape_active:
                        tof_stuck_start = None

                # Check if escape maneuver completed or cleared
                if escape_active:
                    laser_cleared = (live_tof_m is not None) and (live_tof_m >= tof_stop_dist + escape_margin)
                    if laser_cleared or (now >= escape_until):
                        escape_active = False
                        tof_stuck_start = None
                        controller.cancel_escape_pivot()
                        current_smooth_path = None  # Invalidate path to replan immediately from new heading
                        last_plan_time = 0.0
                        tof_emergency = False
                        c_txt = f"Laser Cleared ({live_tof_m:.2f}m)" if laser_cleared else "Pivot Finished"
                        map_status_banner = f"TOF UNSTUCK: {c_txt}! Replanning..."
                        map_status_time = time.time()
                        print(f"[UNSTUCK] Escape {c_txt}! Replanning from new robot heading.")

            if (now - last_plan_time > plan_interval) or (tof_emergency and not escape_active):
                last_plan_time = now
                if (tof_emergency and not escape_active) or not planner.is_path_valid(robot_pose, current_smooth_path, current_goal):
                    rx, ry, _ = robot_pose
                    raw_p, smooth_p, tree_e = planner.plan(
                        (rx, ry), current_goal, seed_waypoints=current_raw_path
                    )
                    current_tree_edges = tree_e
                    if smooth_p is not None:
                        current_raw_path = raw_p
                        current_smooth_path = smooth_p
                    else:
                        current_raw_path = None
                        current_smooth_path = None

            # 7. Closed-Loop Pure Pursuit Control Command + ToF Emergency Brake
            cmd = controller.compute_command(
                robot_pose, current_smooth_path, goal=current_goal,
                emergency_stop=(tof_emergency and not escape_active)
            )

            # 8. Render Panels
            heading_deg = math.degrees(vo.theta)
            delay_ms = int(round(imu.video_latency_sec * 1000.0))
            tof_hud = f" | ToF:{live_tof_m:.2f}m" if live_tof_m is not None else ""
            if vo.calibrating_180:
                imu_tag = f"CAL180:{math.degrees(vo._calib_accum_raw_dtheta):.0f}d"
            elif vo.has_imu:
                motion_src = getattr(vo, "last_motion_source", "FUSED")
                mount_tilt = f"P{math.degrees(vo.nominal_pitch_rad - vo.camera_pitch_rad):+.0f}deg"
                imu_tag = f"BNO055-ROT:{heading_deg:.0f}deg | {motion_src}{tof_hud}"
            else:
                imu_tag = f"WAIT-BNO055!:{heading_deg:.0f}deg{tof_hud}"

            _put_panel_label(
                yolo_vis,
                f"[{resolved_pi_ip}] {fps:.0f}FPS | {imu_tag} | Pose:({robot_pose[0]:.2f},{robot_pose[1]:.2f})"
            )
            last_yolo_vis = yolo_vis

            if show_3d_panel:
                panel2_vis = pc_map.render_3d_view(
                    robot_pose, path=current_smooth_path, goal=current_goal, out_w=w, out_h=h,
                    tof_distance_m=aligned_tof_m
                )
                n_vox = len(pc_map.global_xyz)
                ceil_lbl = "Ceil:ON" if pc_map.show_ceiling else "Ceil:OFF"
                _put_panel_label(
                    panel2_vis,
                    f"3D Cloud ({n_vox} vox, {ceil_lbl}) | [{pc_map.color_mode_name}] 'M':Color | 'H':Ceil | 'V':Radar"
                )
            else:
                panel2_vis = occupancy.render(
                    local_bev, out_w=w, out_h=h, tof_distance_m=aligned_tof_m, max_range_m=max_fwd_range
                )
                free_pct = int(100 * np.sum(local_bev == occupancy.FREE) / local_bev.size)
                occ_pct  = int(100 * np.sum(local_bev == occupancy.OCCUPIED) / local_bev.size)
                _put_panel_label(
                    panel2_vis,
                    f"2.5D Radar+ToF | Free:{free_pct}% Occ:{occ_pct}% | 'V':3D Point Cloud"
                )

            global_vis = global_grid.render_map(
                out_w=w, out_h=h, show_trajectory=True,
                path=current_smooth_path, goal=current_goal,
                robot_pose=robot_pose, lookahead_pt=cmd["lookahead"]
            )
            pwm_gear = getattr(controller, "manual_forward_pwm", getattr(controller, "manual_forward_rpm", 140))
            lock_badge = "[MAP: LOCKED (Protected)]" if map_locked else "[MAP: LIVE]"
            badge_bg = (15, 30, 60) if map_locked else (20, 20, 20)
            badge_fg = (0, 220, 255) if map_locked else (0, 230, 255)
            if map_status_banner and (time.time() - map_status_time) < 4.0:
                _put_panel_label(global_vis, f"*** {map_status_banner} *** | Drag:Pan | Wheel:Zoom", bg=(15, 60, 25), fg=(0, 255, 130))
            else:
                _put_panel_label(global_vis, f"Global 2D Map {lock_badge} (Drag:Pan | Wheel:Zoom | R-Clk:Reloc | 'K':LockMap | 'L':Load | 'O':Save) | Gear:{pwm_gear}PWM", bg=badge_bg, fg=badge_fg)

            plan_vis = global_grid.render_map(
                out_w=w, out_h=h, show_trajectory=False,
                path=current_smooth_path, tree_edges=current_tree_edges,
                goal=current_goal, robot_pose=robot_pose, lookahead_pt=cmd["lookahead"]
            )
            active_mode = cmd.get("mode", controller.control_mode)
            if escape_active:
                path_status = f"[AUTO-UNSTUCK] Pivoting {escape_dir.upper()} ~45deg to clear obstacle ({live_tof_m:.2f}m)..."
            elif active_mode in ("MANUAL_HOLD", "MANUAL_CRUISE"):
                path_status = f"[{active_mode}] PWM: L={cmd.get('pwm_left', 0):+d}, R={cmd.get('pwm_right', 0):+d} (HW:{cmd.get('rpm_left', 0):+d},{cmd.get('rpm_right', 0):+d}) (X/Space=Stop)"
            elif tof_emergency:
                path_status = f"TOF EMERGENCY BRAKE ({live_tof_m:.2f}m < {tof_stop_dist:.2f}m)! (Auto-escape active)"
            elif active_mode in ("PAUSED", "MANUAL"):
                path_status = f"[{active_mode}] Motors: 0,0 (Hold WASD | 1/2/3/4 Cruise | Space=Auto)"
            elif cmd["goal_reached"]:
                path_status = "GOAL REACHED | Motors: STOP (0, 0)"
            elif current_smooth_path:
                path_status = f"[AUTO] PWM: L={cmd.get('pwm_left', 0):+d} R={cmd.get('pwm_right', 0):+d} | v={cmd['v']:.2f}m/s"
            else:
                path_status = "[AUTO] Searching Collision-Free Path..."
            p_badge = "A*" if planner_type == "astar" else "RRT*"
            _put_panel_label(plan_vis, f"{p_badge} + Teleop | {path_status}")

            top = np.hstack([yolo_vis, panel2_vis])
            bot = np.hstack([global_vis, plan_vis])
            composite = np.vstack([top, bot])

            streamer.push_frame(composite)

            if not args.no_display:
                try:
                    cv2.imshow(window_name, composite)
                    key = cv2.waitKey(1) & 0xFF
                    if not handle_key(key):
                        break
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        controller.stop()
        imu.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
