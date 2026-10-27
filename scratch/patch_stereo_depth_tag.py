"""
scratch/patch_stereo_depth_tag.py
Enhance stereo depth perception test CLI.
"""

path = "stereo_visual_odometry.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

old_def = """def _test_depth_cli():
    from config_manager import load_config, save_config
    from capture import LowLatencyCapture

    cfg = load_config()
    scfg = cfg.get("stereo_camera", {})
    svo = StereoVisualOdometry(
        stereo_calib_file=scfg.get("calib_file", "stereo_calib.npz"),
        baseline_m=scfg.get("baseline_m", 0.08),
        swap_left_right=False,
        vertical_offset_px=0,
    )
    swap_lr = bool(scfg.get("swap_left_right", False))
    v_offset = int(scfg.get("vertical_offset_px", 0))

    cap = LowLatencyCapture(source="pi", rtsp_transport="tcp", cfg=cfg)"""

new_def = """def _test_depth_cli(source=None):
    from config_manager import load_config, save_config
    from capture import LowLatencyCapture

    cfg = load_config()
    scfg = cfg.get("stereo_camera", {})
    svo = StereoVisualOdometry(
        stereo_calib_file=scfg.get("calib_file", "stereo_calib.npz"),
        baseline_m=scfg.get("baseline_m", 0.08),
        swap_left_right=False,
        vertical_offset_px=0,
    )
    swap_lr = bool(scfg.get("swap_left_right", False))
    v_offset = int(scfg.get("vertical_offset_px", 0))

    src = source or cfg.get("camera", {}).get("source", "pi")
    cap = LowLatencyCapture(source=src, rtsp_transport="tcp", cfg=cfg)"""

assert old_def in text, "old_def not found in stereo_visual_odometry.py"
text = text.replace(old_def, new_def, 1)

old_main = """if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-depth", action="store_true")
    args = ap.parse_args()
    if args.test_depth:
        _test_depth_cli()"""

new_main = """if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stereo Camera Real-Time Metric Depth Perception Viewer")
    ap.add_argument("--test-depth", action="store_true", help="Launch live metric stereo depth perception window")
    ap.add_argument("--source", default=None, help="Video source ('pi', '0', or RTSP URL)")
    args = ap.parse_args()
    _test_depth_cli(source=args.source)"""

assert old_main in text, "old_main not found in stereo_visual_odometry.py"
text = text.replace(old_main, new_main, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Successfully updated stereo_visual_odometry.py!")
