#!/usr/bin/env python3
"""
deploy_to_pi.py
One-click deployment script to copy all files from pi/ to the Raspberry Pi over SSH.
Uses SCP or SFTP to sync files to /home/admin/robot/.
"""

import os
import subprocess
import sys

# Ensure workspace root in path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from config_manager import load_config, resolve_pi_host

cfg = load_config()
pi_host = resolve_pi_host(cfg)

print("=" * 60)
print(f" Deploying 'pi/' files to Raspberry Pi ({pi_host})")
print(" Target: admin@{pi_host}:/home/admin/robot/")
print("=" * 60)

pi_dir = os.path.join(ROOT, "pi")
if not os.path.exists(pi_dir):
    print("Error: 'pi/' directory not found!")
    sys.exit(1)

# Files to transfer
files = [
    "pi_stereo_capture.py",
    "pi_imu_streamer.py",
    "pi_bt_motor_bridge.py",
    "pi_mpu6500_streamer.py",
    "bno055_broadcaster.py",
    "ngamove_bt.py",
    "detect_cameras.py",
    "set_camera_exposure.py",
    "mediamtx_pi.yml",
]

success_count = 0
for f in files:
    local_path = os.path.join(pi_dir, f)
    if os.path.exists(local_path):
        cmd = ["scp", local_path, f"admin@{pi_host}:/home/admin/robot/{f}"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                print(f"  [OK] Deployed {f}")
                success_count += 1
            else:
                print(f"  [WARN] Failed to deploy {f}: {res.stderr.strip()}")
        except Exception as e:
            print(f"  [ERROR] {f}: {e}")

print("=" * 60)
print(f"Deployment finished: {success_count}/{len(files)} files updated on Pi.")
print("To restart services on Pi, run: ssh pi 'sudo systemctl restart imu-stream stereo-stream'")
print("=" * 60)
