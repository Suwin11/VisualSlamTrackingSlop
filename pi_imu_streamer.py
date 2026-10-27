#!/usr/bin/env python3
"""
pi_mpu6500_streamer.py
Unified Raspberry Pi 4 100 Hz Telemetry Streamer for:
  1. Bosch BNO055 9-DOF Absolute Orientation IMU (over Hardware UART or Software/Hardware I2C)
  2. ST TOF400C / VL53L1X 4.0m Laser Ranging Sensor (over Dedicated Hardware I2C Bus 1 @ 0x29)
  3. Automatic Fallback to MPU-6500 / MPU-6050 / MPU-9250 (if BNO055 is not plugged in)

Recommended Wiring (Zero Bus Contention & Immune to Pi I2C Clock-Stretching Bug):
  [A] BNO055 on Raspberry Pi UART (/dev/serial0, 115200 8N1):
      - BNO055 VIN      -> Pi 3.3V (Pin 1 or 17)
      - BNO055 GND      -> Pi GND  (Pin 6, 9, or 14)
      - BNO055 PS1      -> Pi 3.3V (HIGH selects UART protocol; keep PS0 LOW/GND)
      - BNO055 SDA / TX -> Pi Pin 10 (GPIO 15 / RXD)
      - BNO055 SCL / RX -> Pi Pin 8  (GPIO 14 / TXD)
      * Alternative if your BNO055 board has no PS1 pin: wire BNO055 SDA->Pin 16 (GPIO 23)
        and SCL->Pin 18 (GPIO 24) on Software I2C Bus 3 (/dev/i2c-3 @ 0x28).

  [B] TOF400C (VL53L1X) on Dedicated Hardware I2C Bus 1 (/dev/i2c-1 @ 0x29):
      - TOF400C VIN     -> Pi 3.3V (Pin 1 or 17)
      - TOF400C GND     -> Pi GND  (Pin 6, 9, or 14)
      - TOF400C SDA     -> Pi Pin 3 (GPIO 2 / SDA1)
      - TOF400C SCL     -> Pi Pin 5 (GPIO 3 / SCL1)

Streams 10-field CSV packets over UDP & TCP port 5005:
  seq,timestamp,yaw_deg,pitch_deg,roll_deg,gz_rads,gy_rads,lin_acc,calib,tof_m
"""

import argparse
import glob
import math
import os
import select
import socket
import struct
import sys
import termios
import threading
import time

try:
    from smbus2 import SMBus, i2c_msg
except ImportError:
    SMBus = None
    i2c_msg = None

STATIC_TARGETS = [
    ("255.255.255.255", 5005),
]

# ── Bosch BNO055 Registers & Modes ───────────────────────────────────────────
BNO055_CHIP_ID      = 0xA0
BNO055_REG_CHIP_ID  = 0x00
BNO055_REG_PAGE_ID  = 0x07
BNO055_REG_GYR_Y    = 0x16  # 24-byte contiguous block: GYR_Y,Z(4) + EUL(6) + QUA(8) + LIA(6)
BNO055_REG_CALIB    = 0x35
BNO055_REG_UNIT_SEL = 0x3B
BNO055_REG_OPR_MODE = 0x3D
BNO055_REG_PWR_MODE = 0x3E
BNO055_OPR_CONFIG   = 0x00
BNO055_OPR_IMUPLUS  = 0x08  # 100 Hz 6-DOF Accel+Gyro fusion (immune to indoor magnetic distortion)
BNO055_OPR_NDOF     = 0x0C  # 100 Hz 9-DOF fusion with Magnetometer

# ── MPU-6500 / MPU-6050 / MPU-9250 Fallback Registers ────────────────────────
REG_SMPLRT_DIV   = 0x19
REG_CONFIG       = 0x1A
REG_GYRO_CONFIG  = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_ACCEL_CFG2   = 0x1D
REG_ACCEL_XOUT_H = 0x3B
REG_PWR_MGMT_1   = 0x6B
REG_PWR_MGMT_2   = 0x6C
REG_WHO_AM_I     = 0x75

VALID_WHO_AM_I = {
    0x70: "MPU-6500",
    0x71: "MPU-9250",
    0x73: "MPU-9255",
    0x68: "MPU-6050",
    0x72: "MPU-6555",
}

GYRO_LSB_PER_DPS = 65.5
ACCEL_LSB_PER_G  = 8192.0
G_MPS2           = 9.80665

# ── ST VL53L1X (TOF400C) 16-bit Registers & ULD Default Configuration ────────
VL53L1X_ADDR = 0x29
_VL53L1X_VHV_LOOP_BOUND  = 0x0008
_VL53L1X_GPIO_HV_MUX     = 0x0030
_VL53L1X_GPIO_TIO_HV     = 0x0031
_VL53L1X_PHASECAL_MACROP = 0x004B
_VL53L1X_TIMEOUT_A_HI    = 0x005E
_VL53L1X_VCSEL_PERIOD_A  = 0x0060
_VL53L1X_TIMEOUT_B_HI    = 0x0061
_VL53L1X_VCSEL_PERIOD_B  = 0x0063
_VL53L1X_VALID_PHASE_HI  = 0x0069
_VL53L1X_SD_WOI_SD0      = 0x0078
_VL53L1X_SD_INIT_PHASE0  = 0x007A
_VL53L1X_SYSTEM_INT_CLR  = 0x0086
_VL53L1X_SYSTEM_START    = 0x0087
_VL53L1X_RESULT_STATUS   = 0x0089
_VL53L1X_RESULT_DIST_MM  = 0x0096
_VL53L1X_MODEL_ID        = 0x010F

_VL53L1X_DEFAULT_CFG = bytes([
    0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x02, 0x08, 0x00, 0x08, 0x10, 0x01, 0x01, 0x00, 0x00, 0x00,
    0x00, 0xFF, 0x00, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x0B, 0x00, 0x00, 0x02, 0x0A, 0x21,
    0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0xC8, 0x00, 0x00, 0x38, 0xFF, 0x01, 0x00, 0x08, 0x00,
    0x00, 0x01, 0xCC, 0x0F, 0x01, 0xF1, 0x0D, 0x01, 0x68, 0x00, 0x80, 0x08, 0xB8, 0x00, 0x00, 0x00,
    0x00, 0x0F, 0x89, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x0F, 0x0D, 0x0E, 0x0E, 0x00,
    0x00, 0x02, 0xC7, 0xFF, 0x9B, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00
])


def discover_pc_targets(port=5005):
    """Dynamically find all active PC IPs and subnet broadcasts without hardcoded IPs."""
    targets = {("255.255.255.255", int(port))}
    ssh_client = os.environ.get("SSH_CLIENT", "") or os.environ.get("SSH_CONNECTION", "")
    if ssh_client:
        ssh_ip = ssh_client.split()[0].strip()
        if ssh_ip:
            targets.add((ssh_ip, int(port)))

    # Add local interface /24 broadcast address
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        my_ip = s.getsockname()[0]
        s.close()
        octets = my_ip.split(".")
        if len(octets) == 4:
            targets.add((f"{octets[0]}.{octets[1]}.{octets[2]}.255", int(port)))
    except Exception:
        pass

    try:
        with open("/proc/net/arp", "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                parts = line.split()
                if len(parts) >= 4 and parts[2] != "0x0" and parts[3] != "00:00:00:00:00:00":
                    ip = parts[0].strip()
                    targets.add((ip, int(port)))
                    octets = ip.split(".")
                    if len(octets) == 4:
                        targets.add((f"{octets[0]}.{octets[1]}.{octets[2]}.255", int(port)))
    except Exception:
        pass
    return targets


# ==============================================================================
# 1. BOSCH BNO055 DRIVER (Supports BOTH Hardware UART and Software/Hardware I2C)
# ==============================================================================
class BNO055Driver:
    """
    High-reliability BNO055 driver supporting:
      - Native UART mode (`115200 8N1` via POSIX `termios` — zero pip dependencies!)
      - I2C mode (`/dev/i2c-3`, `/dev/i2c-1` @ `0x28` or `0x29`) with single-burst reads
        and clock-stretching glitch rejection.
    """

    def __init__(self, uart_ports=("/dev/serial0", "/dev/ttyAMA0", "/dev/ttyS0"),
                 i2c_buses=(3, 1, 0, 2), i2c_addrs=(0x28, 0x29),
                 fusion_mode=BNO055_OPR_IMUPLUS, invert_yaw=False):
        self.uart_ports = list(uart_ports) + sorted(glob.glob("/dev/ttyUSB*"))
        self.i2c_buses = list(i2c_buses)
        self.i2c_addrs = list(i2c_addrs)
        self.fusion_mode = int(fusion_mode)
        self.invert_yaw = bool(invert_yaw)

        self.transport = None  # "uart" or "i2c"
        self.uart_fd = None
        self.uart_dev = None
        self.bus = None
        self.bus_num = None
        self.addr = None
        self.chip_name = "BNO055"
        self.last_calib = 3
        self._last_calib_poll = 0.0
        self._prev_valid = None

    def close(self):
        if self.uart_fd is not None:
            try:
                os.close(self.uart_fd)
            except Exception:
                pass
            self.uart_fd = None
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:
                pass
            self.bus = None
        self.transport = None

    # ── POSIX termios UART helpers (no pyserial needed!) ─────────────
    def _open_uart_port(self, dev_path):
        fd = os.open(dev_path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(fd)
        # 115200 8N1 raw mode
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL | termios.B115200
        attrs[3] = 0
        attrs[4] = termios.B115200
        attrs[5] = termios.B115200
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        termios.tcflush(fd, termios.TCIOFLUSH)
        return fd

    def _uart_read_exact(self, length, timeout=0.04):
        buf = bytearray()
        t_end = time.monotonic() + timeout
        while len(buf) < length and time.monotonic() < t_end:
            rem = max(0.001, t_end - time.monotonic())
            r, _, _ = select.select([self.uart_fd], [], [], rem)
            if r:
                chunk = os.read(self.uart_fd, length - len(buf))
                if chunk:
                    buf.extend(chunk)
        return bytes(buf)

    def _uart_write_reg(self, reg, val_byte, retries=3):
        cmd = bytes([0xAA, 0x00, reg & 0xFF, 1, val_byte & 0xFF])
        for _ in range(retries):
            termios.tcflush(self.uart_fd, termios.TCIFLUSH)
            os.write(self.uart_fd, cmd)
            resp = self._uart_read_exact(2, timeout=0.05)
            if len(resp) == 2 and resp[0] == 0xEE and resp[1] == 0x01:
                return True
            time.sleep(0.005)
        return False

    def _uart_read_regs(self, reg, length, retries=2):
        cmd = bytes([0xAA, 0x01, reg & 0xFF, length & 0xFF])
        for _ in range(retries):
            termios.tcflush(self.uart_fd, termios.TCIFLUSH)
            os.write(self.uart_fd, cmd)
            hdr = self._uart_read_exact(2, timeout=0.025)
            if len(hdr) < 2:
                continue
            if hdr[0] == 0xBB and hdr[1] == length:
                payload = self._uart_read_exact(length, timeout=0.03)
                if len(payload) == length:
                    return payload
            elif hdr[0] == 0xEE:
                # 0x07 = BUS_OVER_RUN_ERROR / busy; brief pause and retry
                time.sleep(0.002)
        return None

    # ── Unified Register Access (UART or I2C) ────────────────────────
    def _write_byte(self, reg, val):
        if self.transport == "uart":
            return self._uart_write_reg(reg, val)
        else:
            self.bus.write_byte_data(self.addr, reg, val)
            return True

    def _read_bytes(self, reg, length):
        if self.transport == "uart":
            return self._uart_read_regs(reg, length)
        else:
            if length <= 32:
                return bytes(self.bus.read_i2c_block_data(self.addr, reg, length))
            # SMBus block read limit is 32 bytes; chunk larger reads
            buf = bytearray()
            cur_reg = reg
            remaining = length
            while remaining > 0:
                chunk = min(remaining, 32)
                buf.extend(self.bus.read_i2c_block_data(self.addr, cur_reg, chunk))
                cur_reg += chunk
                remaining -= chunk
            return bytes(buf)

    def _configure_bno055(self):
        # Select Page 0 and enter CONFIG mode
        self._write_byte(BNO055_REG_PAGE_ID, 0x00)
        self._write_byte(BNO055_REG_OPR_MODE, BNO055_OPR_CONFIG)
        time.sleep(0.03)
        # Normal power mode, standard units (deg, dps, m/s^2, Windows orientation)
        self._write_byte(BNO055_REG_PWR_MODE, 0x00)
        time.sleep(0.01)
        self._write_byte(BNO055_REG_UNIT_SEL, 0x00)
        time.sleep(0.01)
        # Enter 100 Hz fusion mode (IMUPLUS 0x08 by default)
        self._write_byte(BNO055_REG_OPR_MODE, self.fusion_mode)
        time.sleep(0.05)

    def discover_and_init(self, tof_bus_num=None, tof_addr=0x29):
        self.close()
        mode_str = "IMUPLUS (6-DOF)" if self.fusion_mode == BNO055_OPR_IMUPLUS else "NDOF (9-DOF)"

        # 1. Try Hardware UART first (/dev/serial0, /dev/ttyAMA0, /dev/ttyS0, /dev/ttyUSB*)
        seen_realpaths = set()
        for port in self.uart_ports:
            if not os.path.exists(port):
                continue
            rp = os.path.realpath(port)
            if rp in seen_realpaths:
                continue
            seen_realpaths.add(rp)
            try:
                fd = self._open_uart_port(port)
                self.uart_fd = fd
                self.uart_dev = port
                self.transport = "uart"
                chip = self._uart_read_regs(BNO055_REG_CHIP_ID, 1, retries=3)
                if chip and chip[0] == BNO055_CHIP_ID:
                    self._configure_bno055()
                    self.chip_name = f"BNO055-UART({port})"
                    print(f"[BNO055] Connected via UART on {port} (115200 8N1) | Mode: {mode_str}", flush=True)
                    return True
                os.close(fd)
                self.uart_fd = None
                self.transport = None
            except Exception:
                if self.uart_fd is not None:
                    try:
                        os.close(self.uart_fd)
                    except Exception:
                        pass
                    self.uart_fd = None
                self.transport = None

        # 2. Try I2C buses (prioritizing Bus 3 software I2C, then Bus 1 @ 0x28)
        for b_num in self.i2c_buses:
            try:
                bus = SMBus(b_num)
            except Exception:
                continue

            for addr in self.i2c_addrs:
                # Never clobber an active TOF400C on the same bus & address (0x29)
                if b_num == tof_bus_num and addr == tof_addr:
                    continue
                try:
                    chip_id = bus.read_byte_data(addr, BNO055_REG_CHIP_ID)
                    if chip_id != BNO055_CHIP_ID:
                        continue
                    self.bus = bus
                    self.bus_num = b_num
                    self.addr = addr
                    self.transport = "i2c"
                    self._configure_bno055()
                    self.chip_name = f"BNO055-I2C(/dev/i2c-{b_num}@0x{addr:02X})"
                    print(f"[BNO055] Connected via I2C on /dev/i2c-{b_num} @ 0x{addr:02X} | Mode: {mode_str}", flush=True)
                    return True
                except Exception:
                    continue

            try:
                bus.close()
            except Exception:
                pass

        return False

    def update(self, dt):
        """
        Read a single 38-byte contiguous burst (0x08..0x2D) from the BNO055:
          - 0x08..0x0D: ACC_X, ACC_Y, ACC_Z (6 bytes, raw specific force WITH gravity)
          - 0x0E..0x13: MAG_X, MAG_Y, MAG_Z (6 bytes)
          - 0x14..0x19: GYR_X, GYR_Y, GYR_Z (6 bytes, raw angular velocity)
          - 0x1A..0x1F: EUL_HEADING, EUL_ROLL, EUL_PITCH (6 bytes)
          - 0x20..0x27: Quaternion (8 bytes)
          - 0x28..0x2D: LIA_X, LIA_Y, LIA_Z (6 bytes, linear accel without gravity)
        """
        raw = self._read_bytes(0x08, 38)
        if raw is None or len(raw) != 38:
            if self._prev_valid is not None:
                return self._prev_valid
            raise RuntimeError("BNO055 38-byte burst read incomplete")

        ax_raw, ay_raw, az_raw = struct.unpack("<hhh", raw[0:6])
        gx_raw, gy_raw, gz_raw = struct.unpack("<hhh", raw[12:18])
        yaw_raw, roll_raw, pitch_raw = struct.unpack("<hhh", raw[18:24])
        lax_raw, lay_raw, laz_raw = struct.unpack("<hhh", raw[32:38])

        raw_ax = ax_raw / 100.0
        raw_ay = ay_raw / 100.0
        raw_az = az_raw / 100.0
        raw_gx = math.radians(gx_raw / 16.0)
        raw_gy = math.radians(gy_raw / 16.0)
        raw_gz = math.radians(gz_raw / 16.0)

        yaw_deg = yaw_raw / 16.0
        roll_deg = roll_raw / 16.0
        pitch_deg = pitch_raw / 16.0
        gy_rads = raw_gy
        gz_rads = raw_gz

        # Reject BNO055 I2C clock-stretch glitch spikes
        if not (0.0 <= yaw_deg <= 360.0 and abs(pitch_deg) <= 180.0 and abs(roll_deg) <= 180.0 and abs(gz_rads) <= 25.0):
            if self._prev_valid is not None:
                return self._prev_valid
            raise RuntimeError("BNO055 glitch sample rejected")

        if self.invert_yaw:
            yaw_deg = (360.0 - yaw_deg) % 360.0
            gz_rads = -gz_rads
            raw_gz = -raw_gz

        lin_acc = math.sqrt((lax_raw / 100.0) ** 2 + (lay_raw / 100.0) ** 2 + (laz_raw / 100.0) ** 2)
        if lin_acc > 40.0:
            lin_acc = 0.0

        now = time.monotonic()
        if now - self._last_calib_poll >= 1.0:
            self._last_calib_poll = now
            try:
                c_raw = self._read_bytes(BNO055_REG_CALIB, 1)
                if c_raw:
                    c_byte = c_raw[0]
                    self.last_calib = max((c_byte >> 6) & 0x03, (c_byte >> 4) & 0x03, 1)
            except Exception:
                pass

        res = (
            yaw_deg, pitch_deg, roll_deg, gz_rads, gy_rads, lin_acc, self.last_calib,
            raw_ax, raw_ay, raw_az, raw_gx, raw_gy, raw_gz
        )
        self._prev_valid = res
        return res


# ==============================================================================
# 2. ST TOF400C (VL53L1X) 4.0m LASER RANGING DRIVER (Dedicated I2C Bus 1 @ 0x29)
# ==============================================================================
class TOF400CDriver:
    """
    Non-blocking pure-smbus2 driver for the TOF400C (VL53L1X) 4m Laser Ranging Sensor @ 0x29.
    """

    def __init__(self, bus_candidates=(1, 3, 0, 2), addr=VL53L1X_ADDR):
        self.bus_candidates = list(bus_candidates)
        self.addr = int(addr)
        self.bus = None
        self.bus_num = None
        self.connected = False
        self._int_pol = 1
        self.latest_distance_m = -1.0
        self.last_valid_mono = 0.0
        self._error_count = 0

    def _write_reg16(self, reg16, data_bytes):
        payload = [((reg16 >> 8) & 0xFF), (reg16 & 0xFF)] + list(data_bytes)
        msg = i2c_msg.write(self.addr, payload)
        self.bus.i2c_rdwr(msg)

    def _read_reg16(self, reg16, length=1):
        w = i2c_msg.write(self.addr, [((reg16 >> 8) & 0xFF), (reg16 & 0xFF)])
        r = i2c_msg.read(self.addr, length)
        self.bus.i2c_rdwr(w, r)
        return bytes(r)

    def discover_and_init(self, shared_bus=None, shared_bus_num=None):
        candidates = []
        if shared_bus is not None:
            candidates.append((shared_bus_num, shared_bus, True))
        for b_num in self.bus_candidates:
            if b_num != shared_bus_num:
                candidates.append((b_num, None, False))

        for b_num, bus_obj, is_shared in candidates:
            try:
                bus = bus_obj if is_shared else SMBus(b_num)
                self.bus = bus
                self.bus_num = b_num
                info = self._read_reg16(_VL53L1X_MODEL_ID, 3)
                model_id, module_type, mask_rev = info[0], info[1], info[2]
                # Ensure we only initialize a genuine VL53L1X (0xEA) and never a BNO055 at 0x29
                if model_id != 0xEA:
                    if not is_shared:
                        bus.close()
                    continue

                for offset in range(0, len(_VL53L1X_DEFAULT_CFG), 16):
                    chunk = _VL53L1X_DEFAULT_CFG[offset:offset + 16]
                    self._write_reg16(0x002D + offset, chunk)

                int_pol_raw = self._read_reg16(_VL53L1X_GPIO_HV_MUX, 1)[0] & 0x10
                self._int_pol = 0 if ((int_pol_raw >> 4) & 0x01) else 1

                self._write_reg16(_VL53L1X_SYSTEM_START, b"\x40")
                t0 = time.monotonic()
                while (time.monotonic() - t0) < 0.25:
                    if (self._read_reg16(_VL53L1X_GPIO_TIO_HV, 1)[0] & 0x01) == self._int_pol:
                        break
                    time.sleep(0.005)
                self._write_reg16(_VL53L1X_SYSTEM_INT_CLR, b"\x01")
                self._write_reg16(_VL53L1X_SYSTEM_START, b"\x00")
                self._write_reg16(_VL53L1X_VHV_LOOP_BOUND, b"\x09")
                self._write_reg16(0x000B, b"\x00")

                # Long Distance Mode (up to 400cm / 4.0m) + 33ms timing budget (~30Hz)
                self._write_reg16(_VL53L1X_PHASECAL_MACROP, b"\x0a")
                self._write_reg16(_VL53L1X_VCSEL_PERIOD_A, b"\x0f")
                self._write_reg16(_VL53L1X_VCSEL_PERIOD_B, b"\x0d")
                self._write_reg16(_VL53L1X_VALID_PHASE_HI, b"\xb8")
                self._write_reg16(_VL53L1X_SD_WOI_SD0, b"\x0f\x0d")
                self._write_reg16(_VL53L1X_SD_INIT_PHASE0, b"\x0e\x0e")
                self._write_reg16(_VL53L1X_TIMEOUT_A_HI, b"\x00\x60")
                self._write_reg16(_VL53L1X_TIMEOUT_B_HI, b"\x00\x6e")
                self._write_reg16(0x006C, b"\x00\x00\x05\x40")

                self._write_reg16(_VL53L1X_SYSTEM_START, b"\x40")
                self.connected = True
                self._error_count = 0
                print(f"[TOF400C] Connected to VL53L1X (ID=0x{model_id:02X}:0x{module_type:02X}:0x{mask_rev:02X}) "
                      f"on /dev/i2c-{b_num} @ 0x{self.addr:02X} (4.0m Long-Range Mode, 30Hz)", flush=True)
                return True
            except Exception:
                if not is_shared and bus_obj is None:
                    try:
                        bus.close()
                    except Exception:
                        pass
                continue

        self.bus = None
        self.bus_num = None
        self.connected = False
        return False

    def poll_distance(self):
        if not self.connected or self.bus is None:
            return -1.0

        now = time.monotonic()
        try:
            rdy = (self._read_reg16(_VL53L1X_GPIO_TIO_HV, 1)[0] & 0x01) == self._int_pol
            if rdy:
                status = self._read_reg16(_VL53L1X_RESULT_STATUS, 1)[0]
                raw_mm = struct.unpack(">H", self._read_reg16(_VL53L1X_RESULT_DIST_MM, 2))[0]
                self._write_reg16(_VL53L1X_SYSTEM_INT_CLR, b"\x01")
                self._error_count = 0

                if (status & 0x1F) == 0x09 and 20 <= raw_mm <= 4200:
                    dist_m = raw_mm / 1000.0
                    if self.latest_distance_m > 0.0:
                        self.latest_distance_m = 0.65 * dist_m + 0.35 * self.latest_distance_m
                    else:
                        self.latest_distance_m = dist_m
                    self.last_valid_mono = now
                else:
                    if (now - self.last_valid_mono) > 0.20:
                        self.latest_distance_m = -1.0
        except Exception:
            self._error_count += 1
            if self._error_count > 10:
                print("[TOF400C WARN] I2C read lost; will auto-reconnect...", flush=True)
                self.connected = False
                self.latest_distance_m = -1.0

        if (now - self.last_valid_mono) > 0.35:
            return -1.0
        return self.latest_distance_m


# ==============================================================================
# 3. FALLBACK MPU-6500 / MPU-6050 DRIVER (Used if BNO055 is not connected)
# ==============================================================================
class MPU6500Driver:
    def __init__(self, bus_candidates=(1, 3, 0, 2), addr_candidates=(0x68, 0x69),
                 calib_samples=150, invert_yaw=False):
        self.bus_candidates = list(bus_candidates)
        self.addr_candidates = list(addr_candidates)
        self.calib_samples = int(calib_samples)
        self.invert_yaw = bool(invert_yaw)

        self.bus = None
        self.bus_num = None
        self.addr = None
        self.chip_name = "MPU-6500"

        self.bg_x = 0.0
        self.bg_y = 0.0
        self.bg_z = 0.0
        self.acc_scale = 1.0
        self.up_x = 0.0
        self.up_y = 0.0
        self.up_z = 1.0
        self.is_calibrated = False
        self.yaw_deg = 0.0
        self.pitch_rad = 0.0
        self.roll_rad = 0.0
        self.prev_yaw_rate_dps = 0.0
        self.stationary_count = 0

    def discover_and_init(self):
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:
                pass
            self.bus = None

        for b_num in self.bus_candidates:
            try:
                bus = SMBus(b_num)
            except Exception:
                continue

            for addr in self.addr_candidates:
                try:
                    bus.read_byte_data(addr, REG_PWR_MGMT_1)
                    who_id = bus.read_byte_data(addr, REG_WHO_AM_I)
                    self.chip_name = VALID_WHO_AM_I.get(who_id, f"MPU-0x{who_id:02X}")
                    self.bus = bus
                    self.bus_num = b_num
                    self.addr = addr
                    self._configure_registers()
                    print(f"[MPU-6500] Connected to {self.chip_name} (WHO_AM_I=0x{who_id:02X}) "
                          f"on /dev/i2c-{b_num} @ 0x{addr:02X}", flush=True)
                    self.calibrate_stationary()
                    return True
                except Exception:
                    continue

            try:
                bus.close()
            except Exception:
                pass
        return False

    def _configure_registers(self):
        self.bus.write_byte_data(self.addr, REG_PWR_MGMT_1, 0x80)
        time.sleep(0.05)
        self.bus.write_byte_data(self.addr, REG_PWR_MGMT_1, 0x01)
        self.bus.write_byte_data(self.addr, REG_PWR_MGMT_2, 0x00)
        time.sleep(0.02)
        self.bus.write_byte_data(self.addr, REG_CONFIG, 0x03)
        self.bus.write_byte_data(self.addr, REG_SMPLRT_DIV, 0x09)
        self.bus.write_byte_data(self.addr, REG_GYRO_CONFIG, 0x08)
        self.bus.write_byte_data(self.addr, REG_ACCEL_CONFIG, 0x08)
        try:
            self.bus.write_byte_data(self.addr, REG_ACCEL_CFG2, 0x03)
        except Exception:
            pass
        time.sleep(0.02)

    def read_raw_ag(self):
        raw = self.bus.read_i2c_block_data(self.addr, REG_ACCEL_XOUT_H, 14)
        ax_r, ay_r, az_r, _, gx_r, gy_r, gz_r = struct.unpack(">hhhhhhh", bytes(raw))
        ax = (ax_r / ACCEL_LSB_PER_G) * G_MPS2
        ay = (ay_r / ACCEL_LSB_PER_G) * G_MPS2
        az = (az_r / ACCEL_LSB_PER_G) * G_MPS2
        gx = math.radians(gx_r / GYRO_LSB_PER_DPS)
        gy = math.radians(gy_r / GYRO_LSB_PER_DPS)
        gz = math.radians(gz_r / GYRO_LSB_PER_DPS)
        return ax, ay, az, gx, gy, gz

    def calibrate_stationary(self):
        print(f"[MPU-6500] Calibrating gyro bias ({self.calib_samples} samples, ~1.5s)... Keep robot still!", flush=True)
        sum_gx = sum_gy = sum_gz = 0.0
        sum_ax = sum_ay = sum_az = 0.0
        valid = 0
        for _ in range(self.calib_samples):
            try:
                ax, ay, az, gx, gy, gz = self.read_raw_ag()
                sum_gx += gx
                sum_gy += gy
                sum_gz += gz
                sum_ax += ax
                sum_ay += ay
                sum_az += az
                valid += 1
            except Exception:
                pass
            time.sleep(0.008)
        if valid < 20:
            return
        self.bg_x = sum_gx / valid
        self.bg_y = sum_gy / valid
        self.bg_z = sum_gz / valid
        mean_ax = sum_ax / valid
        mean_ay = sum_ay / valid
        mean_az = sum_az / valid
        grav_norm = math.sqrt(mean_ax ** 2 + mean_ay ** 2 + mean_az ** 2)
        if grav_norm > 2.0:
            self.acc_scale = G_MPS2 / grav_norm
            self.up_x = mean_ax / grav_norm
            self.up_y = mean_ay / grav_norm
            self.up_z = mean_az / grav_norm
        self.roll_rad = math.atan2(mean_ay, max(math.hypot(mean_ax, mean_az), 1e-3))
        self.pitch_rad = math.atan2(-mean_ax, max(math.hypot(mean_ay, mean_az), 1e-3))
        self.yaw_deg = 0.0
        self.prev_yaw_rate_dps = 0.0
        self.is_calibrated = True

    def update(self, dt):
        ax, ay, az, gx_raw, gy_raw, gz_raw = self.read_raw_ag()
        ax *= self.acc_scale
        ay *= self.acc_scale
        az *= self.acc_scale
        gx = gx_raw - self.bg_x
        gy = gy_raw - self.bg_y
        gz = gz_raw - self.bg_z
        acc_norm = math.sqrt(ax * ax + ay * ay + az * az)
        acc_dev = abs(acc_norm - G_MPS2)
        gyro_norm = math.sqrt(gx * gx + gy * gy + gz * gz)
        if gyro_norm < 0.040 and acc_dev < 0.30:
            self.stationary_count += 1
            if self.stationary_count > 15:
                alpha_b = 0.008
                self.bg_x = (1.0 - alpha_b) * self.bg_x + alpha_b * gx_raw
                self.bg_y = (1.0 - alpha_b) * self.bg_y + alpha_b * gy_raw
                self.bg_z = (1.0 - alpha_b) * self.bg_z + alpha_b * gz_raw
                gx = gy = gz = 0.0
        else:
            self.stationary_count = 0

        yaw_rate_rads = gx * self.up_x + gy * self.up_y + gz * self.up_z
        if self.invert_yaw:
            yaw_rate_rads = -yaw_rate_rads
        if abs(yaw_rate_rads) < 0.0050:
            yaw_rate_rads = 0.0

        yaw_rate_dps = math.degrees(yaw_rate_rads)
        dyaw_cw_deg = -0.5 * (self.prev_yaw_rate_dps + yaw_rate_dps) * dt
        self.prev_yaw_rate_dps = yaw_rate_dps
        self.yaw_deg = (self.yaw_deg + dyaw_cw_deg) % 360.0

        roll_acc = math.atan2(ay, max(math.hypot(ax, az), 1e-3))
        pitch_acc = math.atan2(-ax, max(math.hypot(ay, az), 1e-3))
        alpha_cf = 0.98 if acc_dev < 1.5 else 0.999
        self.roll_rad = alpha_cf * (self.roll_rad + gx * dt) + (1.0 - alpha_cf) * roll_acc
        self.pitch_rad = alpha_cf * (self.pitch_rad + gy * dt) + (1.0 - alpha_cf) * pitch_acc
        lin_acc = abs(acc_norm - G_MPS2) if self.stationary_count <= 10 else 0.0
        calib_level = 3 if self.is_calibrated else 1
        return (
            self.yaw_deg,
            math.degrees(self.pitch_rad),
            math.degrees(self.roll_rad),
            yaw_rate_rads,
            gy,
            lin_acc,
            calib_level,
            ax,
            ay,
            az,
            gx,
            gy,
            gz,
        )


def main():
    ap = argparse.ArgumentParser(description="Unified BNO055 (UART/I2C) + TOF400C (I2C) + MPU-6500 100Hz Streamer")
    ap.add_argument("--uart", default=None, help="UART device for BNO055 (default: auto-scan /dev/serial0, /dev/ttyAMA0, /dev/ttyUSB*)")
    ap.add_argument("--bus", type=int, default=None, help="I2C bus number (default: auto-scan 3, 1, 0, 2)")
    ap.add_argument("--tof-addr", type=lambda x: int(x, 0), default=0x29, help="TOF400C I2C address (default: 0x29)")
    ap.add_argument("--mode", choices=["imuplus", "ndof"], default="imuplus", help="BNO055 fusion mode (default: imuplus)")
    ap.add_argument("--port", type=int, default=5005, help="UDP/TCP port (default: 5005)")
    ap.add_argument("--rate", type=float, default=100.0, help="Stream rate in Hz (default: 100 Hz)")
    ap.add_argument("--invert-z", action="store_true", help="Invert Yaw direction")
    args = ap.parse_args()

    uart_ports = [args.uart] if args.uart else ["/dev/serial0", "/dev/ttyAMA0", "/dev/ttyS0"]
    i2c_buses = [args.bus] if args.bus is not None else [3, 1, 0, 2]
    tof_buses = [args.bus] if args.bus is not None else [1, 3, 0, 2]
    fusion_mode = BNO055_OPR_NDOF if args.mode == "ndof" else BNO055_OPR_IMUPLUS

    # 1. Initialize TOF400C first on Dedicated Hardware I2C Bus 1 (@ 0x29)
    tof = TOF400CDriver(bus_candidates=tof_buses, addr=args.tof_addr)
    if not tof.discover_and_init():
        print(f"[TOF400C] Not detected yet @ 0x{args.tof_addr:02X} (will auto-connect when plugged in).", flush=True)

    # 2. Initialize BNO055 (UART first, then I2C) with automatic fallback to MPU-6500
    bno_driver = BNO055Driver(
        uart_ports=uart_ports,
        i2c_buses=i2c_buses,
        fusion_mode=fusion_mode,
        invert_yaw=args.invert_z,
    )
    mpu_driver = MPU6500Driver(bus_candidates=tof_buses, invert_yaw=args.invert_z)
    active_imu = None

    def connect_imu():
        nonlocal active_imu
        if bno_driver.discover_and_init(tof_bus_num=tof.bus_num, tof_addr=tof.addr):
            active_imu = bno_driver
            return True
        if mpu_driver.discover_and_init():
            active_imu = mpu_driver
            return True
        return False

    print("[IMU] Scanning for BNO055 (UART /dev/serial0, /dev/ttyAMA0, or I2C 0x28/0x29) or MPU-6500...", flush=True)
    while not connect_imu():
        time.sleep(1.0)

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    try:
        udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp_sock.bind(("0.0.0.0", args.port))
    except Exception:
        pass
    udp_sock.setblocking(False)

    dynamic_targets = discover_pc_targets(args.port)
    print(f"[STREAMER] Active IMU: {active_imu.chip_name} | Target UDP receivers: {sorted(ip for ip, _ in dynamic_targets)}", flush=True)

    tcp_clients = []
    tcp_lock = threading.Lock()

    def _tcp_accept_worker():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("0.0.0.0", args.port))
            srv.listen(4)
            while True:
                conn, caddr = srv.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.setblocking(False)
                with tcp_lock:
                    tcp_clients.append(conn)
                print(f"[STREAMER] PC connected via TCP from {caddr[0]}:{caddr[1]}!", flush=True)
        except Exception:
            pass

    threading.Thread(target=_tcp_accept_worker, daemon=True).start()

    interval = 1.0 / max(args.rate, 10.0)
    seq = 0
    consecutive_errors = 0
    prev_t = time.monotonic()
    last_print_t = time.monotonic() - 0.9
    last_arp_refresh = time.monotonic()
    last_tof_poll = time.monotonic()
    last_hotplug_probe = time.monotonic()
    tof_dist_m = -1.0

    try:
        while True:
            t0 = time.monotonic()
            dt = max(min(t0 - prev_t, 0.1), 0.001)
            prev_t = t0

            if t0 - last_arp_refresh >= 5.0:
                last_arp_refresh = t0
                dynamic_targets.update(discover_pc_targets(args.port))

            # Hot-plug probe every 3.0s:
            # - Connect TOF400C if plugged in
            # - Upgrade from MPU-6500 to BNO055 automatically if BNO055 is plugged in!
            if t0 - last_hotplug_probe >= 3.0:
                last_hotplug_probe = t0
                if not tof.connected:
                    shared_b = active_imu.bus if getattr(active_imu, "transport", None) == "i2c" else None
                    shared_n = active_imu.bus_num if shared_b is not None else None
                    tof.discover_and_init(shared_bus=shared_b, shared_bus_num=shared_n)
                if active_imu is mpu_driver:
                    if bno_driver.discover_and_init(tof_bus_num=tof.bus_num, tof_addr=tof.addr):
                        active_imu = bno_driver
                        print(f"[STREAMER] Hot-swapped IMU to {active_imu.chip_name}!", flush=True)

            # Poll TOF400C at 40 Hz on I2C
            if tof.connected and (t0 - last_tof_poll >= 0.025):
                last_tof_poll = t0
                tof_dist_m = tof.poll_distance()

            try:
                while True:
                    _, sender_addr = udp_sock.recvfrom(64)
                    if sender_addr[0] not in ("127.0.0.1", "0.0.0.0"):
                        if sender_addr not in dynamic_targets:
                            dynamic_targets.add(sender_addr)
                            print(f"[STREAMER] Auto-discovered PC UDP receiver at {sender_addr[0]}:{sender_addr[1]}!", flush=True)
            except Exception:
                pass

            try:
                (yaw_deg, pitch_deg, roll_deg, gz_rads, gy_rads, lin_acc, calib,
                 raw_ax, raw_ay, raw_az, raw_gx, raw_gy, raw_gz) = active_imu.update(dt)
                consecutive_errors = 0
                seq += 1

                line = (
                    f"{seq},{time.time():.4f},{yaw_deg:.2f},{pitch_deg:.2f},"
                    f"{roll_deg:.2f},{gz_rads:.4f},{gy_rads:.4f},{lin_acc:.3f},{calib},{tof_dist_m:.3f},"
                    f"{raw_ax:.4f},{raw_ay:.4f},{raw_az:.4f},{raw_gx:.5f},{raw_gy:.5f},{raw_gz:.5f}\n"
                ).encode("ascii")

                for addr in list(dynamic_targets):
                    try:
                        udp_sock.sendto(line, addr)
                    except Exception:
                        pass

                with tcp_lock:
                    dead = []
                    for c in tcp_clients:
                        try:
                            c.sendall(line)
                        except Exception:
                            dead.append(c)
                    for c in dead:
                        tcp_clients.remove(c)
                        try:
                            c.close()
                        except Exception:
                            pass

                if t0 - last_print_t >= 1.0:
                    last_print_t = t0
                    tof_str = f"{tof_dist_m:5.2f}m" if tof_dist_m > 0.0 else " --- "
                    print(f"[{active_imu.chip_name} LIVE] Yaw={yaw_deg:6.1f} deg | Pitch={pitch_deg:+5.1f} deg | "
                          f"YawRate={math.degrees(gz_rads):+6.1f} deg/s | Calib={calib}/3 | ToF={tof_str}", flush=True)

            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= 20:
                    print(f"[STREAMER WARN] IMU read errors ({consecutive_errors}): {e}. Re-connecting...", flush=True)
                    time.sleep(0.05)
                    if connect_imu():
                        consecutive_errors = 0

            elapsed = time.monotonic() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    except KeyboardInterrupt:
        print("\n[STREAMER] Stopped by user.")
    finally:
        bno_driver.close()
        udp_sock.close()


if __name__ == "__main__":
    main()