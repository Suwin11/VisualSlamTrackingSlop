/*
  ========================================================================
  ESP32 Dedicated Bosch BNO055 Sensor Fusion Driver & Broadcaster
  ========================================================================
  Features:
  - Connects to Bosch BNO055 on standard ESP32 I2C pins (GPIO 21 SDA, 22 SCL)
  - ESP32 hardware I2C supports BNO055 clock stretching natively
  - Mode: IMUPLUS (6-DOF Accel + Gyro fusion, immune to motor magnetic bias)
          or NDOF (9-DOF full absolute orientation)
  - Dual Output:
      1. High-speed USB Serial (115200 baud)
      2. High-speed UDP Broadcast over Wi-Fi (optional, port 5005)
  - Unified 9-field CSV Telemetry format (@ 100 Hz, compatible with imu_receiver.py):
      "seq,ts_sec,yaw_deg,pitch_deg,roll_deg,gz_rads,gy_rads,lin_acc_mag,calib_stat\n"
  ========================================================================
*/

#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>

// --- Wi-Fi Configuration (Optional: Set true to enable direct UDP broadcast) ---
const bool ENABLE_WIFI_UDP = false;
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASS     = "YOUR_WIFI_PASSWORD";
const IPAddress UDP_TARGET_IP(255, 255, 255, 255); // Broadcast to entire subnet
const uint16_t UDP_PORT   = 5005;

WiFiUDP udp;

// --- BNO055 Configuration ---
#define BNO055_ADDR 0x28 // Default address (0x29 if COM3 pin high)

// Registers
#define BNO_PAGE_ID_REG      0x07
#define BNO_CHIP_ID_REG      0x00
#define BNO_OPR_MODE_REG     0x3D
#define BNO_PWR_MODE_REG     0x3E
#define BNO_SYS_TRIGGER_REG  0x3F
#define BNO_UNIT_SEL_REG     0x3B
#define BNO_CALIB_STAT_REG   0x35

// Data registers
#define BNO_EUL_DATA_X_LSB   0x1A // Yaw, Roll, Pitch (each 2 bytes signed)
#define BNO_GYR_DATA_Y_LSB   0x16 // Gyro Y & Z (each 2 bytes signed)
#define BNO_LIA_DATA_X_LSB   0x28 // Linear Accel X, Y, Z (each 2 bytes signed)

// Modes
#define BNO_MODE_CONFIG      0x00
#define BNO_MODE_IMUPLUS     0x08 // Recommended for indoor SLAM (Accel + Gyro fusion)
#define BNO_MODE_NDOF        0x0C // 9-DOF fusion with Magnetometer

uint8_t current_mode = BNO_MODE_IMUPLUS;

unsigned long last_send_time = 0;
const unsigned long SEND_INTERVAL_MS = 10; // 100 Hz
uint32_t packet_seq = 0;

bool writeRegister(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(BNO055_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return (Wire.endTransmission() == 0);
}

uint8_t readRegister(uint8_t reg) {
  Wire.beginTransmission(BNO055_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)BNO055_ADDR, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0;
}

bool initBNO055() {
  Wire.begin(21, 22, 400000); // 400kHz fast I2C
  delay(100);

  uint8_t id = readRegister(BNO_CHIP_ID_REG);
  if (id != 0xA0) {
    Serial.printf("[BNO055] Device not found at 0x%02X (ID: 0x%02X)\n", BNO055_ADDR, id);
    return false;
  }

  writeRegister(BNO_OPR_MODE_REG, BNO_MODE_CONFIG);
  delay(30);

  writeRegister(BNO_PWR_MODE_REG, 0x00);
  delay(10);

  // Units: Deg, dps, m/s^2, Celsius, Windows orientation
  writeRegister(BNO_UNIT_SEL_REG, 0x00);
  delay(10);

  writeRegister(BNO_OPR_MODE_REG, current_mode);
  delay(50);

  Serial.printf("[BNO055] Initialized successfully in %s mode!\n",
                (current_mode == BNO_MODE_IMUPLUS) ? "IMUPLUS" : "NDOF");
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== FIBO ESP32 BNO055 Telemetry Driver ===");

  if (ENABLE_WIFI_UDP) {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.print("[WiFi] Connecting");
    int retries = 0;
    while (WiFi.status() != WL_CONNECTED && retries < 20) {
      delay(500);
      Serial.print(".");
      retries++;
    }
    if (WiFi.status() == WL_CONNECTED) {
      Serial.printf("\n[WiFi] Connected! IP: %s\n", WiFi.localIP().toString().c_str());
      udp.begin(UDP_PORT);
    } else {
      Serial.println("\n[WiFi] Connection timeout. Continuing with Serial only.");
    }
  }

  while (!initBNO055()) {
    Serial.println("[BNO055] Retrying in 1 second...");
    delay(1000);
  }
}

void loop() {
  unsigned long now = millis();
  if (now - last_send_time >= SEND_INTERVAL_MS) {
    last_send_time = now;

    // 1. Read Euler Angles (Yaw, Roll, Pitch)
    Wire.beginTransmission(BNO055_ADDR);
    Wire.write(BNO_EUL_DATA_X_LSB);
    if (Wire.endTransmission(false) != 0) return;

    if (Wire.requestFrom((uint8_t)BNO055_ADDR, (uint8_t)6) != 6) return;
    int16_t raw_yaw   = Wire.read() | (Wire.read() << 8);
    int16_t raw_roll  = Wire.read() | (Wire.read() << 8);
    int16_t raw_pitch = Wire.read() | (Wire.read() << 8);

    float yaw_deg   = raw_yaw / 16.0f;
    float roll_deg  = raw_roll / 16.0f;
    float pitch_deg = raw_pitch / 16.0f;

    // 2. Read Gyro Y & Z (Angular Velocity)
    Wire.beginTransmission(BNO055_ADDR);
    Wire.write(BNO_GYR_DATA_Y_LSB);
    if (Wire.endTransmission(false) != 0) return;
    if (Wire.requestFrom((uint8_t)BNO055_ADDR, (uint8_t)4) != 4) return;
    int16_t raw_gy = Wire.read() | (Wire.read() << 8);
    int16_t raw_gz = Wire.read() | (Wire.read() << 8);
    float gy_rads = (raw_gy / 16.0f) * (PI / 180.0f);
    float gz_rads = (raw_gz / 16.0f) * (PI / 180.0f);

    // 3. Read Linear Acceleration
    Wire.beginTransmission(BNO055_ADDR);
    Wire.write(BNO_LIA_DATA_X_LSB);
    if (Wire.endTransmission(false) != 0) return;
    if (Wire.requestFrom((uint8_t)BNO055_ADDR, (uint8_t)6) != 6) return;
    int16_t raw_lax = Wire.read() | (Wire.read() << 8);
    int16_t raw_lay = Wire.read() | (Wire.read() << 8);
    int16_t raw_laz = Wire.read() | (Wire.read() << 8);

    float lax = raw_lax / 100.0f;
    float lay = raw_lay / 100.0f;
    float laz = raw_laz / 100.0f;
    float lin_acc_mag = sqrtf(lax * lax + lay * lay + laz * laz);

    // 4. Read Calibration Status
    uint8_t calib_byte = readRegister(BNO_CALIB_STAT_REG);
    uint8_t sys_calib = (calib_byte >> 6) & 0x03;
    uint8_t gyr_calib = (calib_byte >> 4) & 0x03;
    uint8_t cal_level = (sys_calib > gyr_calib) ? sys_calib : gyr_calib;

    packet_seq++;
    float ts_sec = now / 1000.0f;

    // Unified 9-field CSV format:
    // "seq,ts,yaw_deg,pitch_deg,roll_deg,gz_rads,gy_rads,lin_acc_mag,calib_stat"
    char buffer[160];
    snprintf(buffer, sizeof(buffer), "%lu,%.4f,%.2f,%.2f,%.2f,%.4f,%.4f,%.3f,%u\n",
             (unsigned long)packet_seq, ts_sec, yaw_deg, pitch_deg, roll_deg,
             gz_rads, gy_rads, lin_acc_mag, cal_level);

    Serial.print(buffer);

    if (ENABLE_WIFI_UDP && WiFi.status() == WL_CONNECTED) {
      udp.beginPacket(UDP_TARGET_IP, UDP_PORT);
      udp.write((const uint8_t*)buffer, strlen(buffer));
      udp.endPacket();
    }
  }
}
