#!/usr/bin/env python3
"""
set_camera_exposure.py
Control exposure, brightness, and gain on Logitech C930e cameras (local or remote via SSH).

Usage Examples:
  # Check current exposure and brightness status of both cameras:
  python set_camera_exposure.py --status

  # Turn Auto Exposure ON (Aperture Priority mode) with balanced brightness:
  python set_camera_exposure.py --auto

  # Set Manual Exposure with specific shutter value (e.g., 250):
  python set_camera_exposure.py --manual 250

  # Adjust brightness specifically on the left camera:
  python set_camera_exposure.py --cam left --brightness 135
"""

import argparse
import os
import subprocess
import sys


def run_cmd(cmd_list, is_remote=False, ssh_client=None):
    if not is_remote:
        res = subprocess.run(cmd_list, capture_output=True, text=True)
        return res.stdout.strip(), res.stderr.strip()
    else:
        cmd_str = " ".join(cmd_list)
        stdin, stdout, stderr = ssh_client.exec_command(cmd_str)
        return stdout.read().decode().strip(), stderr.read().decode().strip()


def main():
    parser = argparse.ArgumentParser(description="Logitech C930e Camera Exposure Controller")
    parser.add_argument("--status", action="store_true", help="Print current settings for both cameras")
    parser.add_argument("--auto", action="store_true", help="Enable Auto Exposure (Aperture Priority mode)")
    parser.add_argument("--manual", type=int, default=None, metavar="SHUTTER", help="Set manual exposure shutter time (e.g. 150-500)")
    parser.add_argument("--brightness", type=int, default=None, metavar="0-255", help="Set camera brightness (default 128)")
    parser.add_argument("--gain", type=int, default=None, metavar="0-255", help="Set camera sensor gain")
    parser.add_argument("--cam", choices=["all", "left", "right"], default="all", help="Target camera (default: all)")
    parser.add_argument("--ip", default="192.168.0.170", help="Raspberry Pi IP (default: 192.168.0.170)")
    parser.add_argument("--user", default="admin", help="SSH username (default: admin)")
    parser.add_argument("--password", default="1234", help="SSH password (default: 1234)")
    parser.add_argument("--local", action="store_true", help="Run locally using v4l2-ctl without SSH")
    args = parser.parse_args()

    # Determine if running locally on Pi or remotely from PC
    is_remote = not args.local and not (os.path.exists("/dev/video0") and os.path.exists("/etc/rpi-issue"))

    ssh_client = None
    if is_remote:
        import paramiko
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh_client.connect(args.ip, username=args.user, password=args.password, timeout=5)
        except Exception as e:
            print(f"[ERROR] Failed to connect to Pi at {args.user}@{args.ip}: {e}")
            sys.exit(1)

    devices = []
    if args.cam in ("all", "left"):
        devices.append(("/dev/video0", "Left Camera"))
    if args.cam in ("all", "right"):
        devices.append(("/dev/video2", "Right Camera"))

    # Apply adjustments if requested
    if args.auto:
        for dev, label in devices:
            print(f"[{label} ({dev})] Enabling Auto Exposure (auto_exposure=3)...")
            run_cmd(["v4l2-ctl", "-d", dev, "--set-ctrl=auto_exposure=3"], is_remote, ssh_client)
            run_cmd(["v4l2-ctl", "-d", dev, "--set-ctrl=exposure_auto=3"], is_remote, ssh_client)
            run_cmd(["v4l2-ctl", "-d", dev, "--set-ctrl=exposure_dynamic_framerate=0"], is_remote, ssh_client)
            if args.brightness is None:
                run_cmd(["v4l2-ctl", "-d", dev, "--set-ctrl=brightness=128"], is_remote, ssh_client)

    if args.manual is not None:
        for dev, label in devices:
            print(f"[{label} ({dev})] Setting Manual Exposure to {args.manual}...")
            run_cmd(["v4l2-ctl", "-d", dev, "--set-ctrl=auto_exposure=1"], is_remote, ssh_client)
            run_cmd(["v4l2-ctl", "-d", dev, "--set-ctrl=exposure_auto=1"], is_remote, ssh_client)
            run_cmd(["v4l2-ctl", "-d", dev, f"--set-ctrl=exposure_time_absolute={args.manual}"], is_remote, ssh_client)
            run_cmd(["v4l2-ctl", "-d", dev, f"--set-ctrl=exposure_absolute={args.manual}"], is_remote, ssh_client)

    if args.brightness is not None:
        for dev, label in devices:
            print(f"[{label} ({dev})] Setting Brightness to {args.brightness}...")
            run_cmd(["v4l2-ctl", "-d", dev, f"--set-ctrl=brightness={args.brightness}"], is_remote, ssh_client)

    if args.gain is not None:
        for dev, label in devices:
            print(f"[{label} ({dev})] Setting Gain to {args.gain}...")
            run_cmd(["v4l2-ctl", "-d", dev, f"--set-ctrl=gain={args.gain}"], is_remote, ssh_client)

    # Print status
    print("\n" + "=" * 60)
    print(f"CURRENT CAMERA STATUS ({'REMOTE ' + args.ip if is_remote else 'LOCAL'}):")
    print("=" * 60)
    for dev, label in [("/dev/video0", "Left Camera"), ("/dev/video2", "Right Camera")]:
        out, _ = run_cmd(["v4l2-ctl", "-d", dev, "-C", "auto_exposure,exposure_time_absolute,brightness,gain"], is_remote, ssh_client)
        print(f"\n{label} ({dev}):")
        for line in out.splitlines():
            print(f"  {line}")
    print("=" * 60)

    if ssh_client:
        ssh_client.close()


if __name__ == "__main__":
    main()
