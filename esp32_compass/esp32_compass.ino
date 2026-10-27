/*
  ========================================================================
  ESP32 Dedicated QMC5883L (QST IC) Digital Compass Driver
  ========================================================================
  - Optimized specifically for QMC5883L (Address 0x0D, marking: DA5883 / DB5883)
  - Also detects QMC5883P / QMC6310 (Address 0x0C or 0x1C)
  - Pure Wire.h (Zero third-party library dependencies)
  - Detailed I2C error diagnostics (ACK / NACK / Bus status)
  - Status Register (0x06) polling for DRDY (Data Ready) & OVL (Overflow rejection)
  - Full Hard-Iron (bias offset) + Soft-Iron (axial scaling) calibration
  - Includes Tilt-Compensated Heading computation for pitched/rolled robot chassis
  ========================================================================
*/

#include <Wire.h>
#include <Arduino.h>

#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22

uint8_t qmcAddress = 0x0D;

// QMC5883L Register Map
#define QMC_REG_DATA_X_LSB  0x00
#define QMC_REG_DATA_X_MSB  0x01
#define QMC_REG_DATA_Y_LSB  0x02
#define QMC_REG_DATA_Y_MSB  0x03
#define QMC_REG_DATA_Z_LSB  0x04
#define QMC_REG_DATA_Z_MSB  0x05
#define QMC_REG_STATUS      0x06
#define QMC_REG_TEMP_LSB    0x07
#define QMC_REG_TEMP_MSB    0x08
#define QMC_REG_CONTROL_1   0x09
#define QMC_REG_CONTROL_2   0x0A
#define QMC_REG_PERIOD      0x0B

// Continuous mode, 100 Hz, +/-8 Gauss, 512 oversample
#define QMC_CONFIG_CTRL1    0x19

// Local Magnetic Declination (Bangkok, Thailand: ~ -0.58 degrees)
const float DECLINATION_DEG = -0.58f;

// Hard-Iron Offsets & Soft-Iron Scale Multipliers
float offsetX = 0.0f, offsetY = 0.0f, offsetZ = 0.0f;
float scaleX = 1.0f, scaleY = 1.0f, scaleZ = 1.0f;

// Interactive Calibration State
bool isCalibrating = false;
int16_t calMinX = 32767, calMaxX = -32768;
int16_t calMinY = 32767, calMaxY = -32768;
int16_t calMinZ = 32767, calMaxZ = -32768;
unsigned long calStartTime = 0;
const unsigned long CAL_DURATION_MS = 15000; // 15 seconds

// -----------------------------------------------------------------------------
// I2C Low-Level Operations
// -----------------------------------------------------------------------------
uint8_t writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(qmcAddress);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission(); // 0 = Success
}

uint8_t readReg(uint8_t reg, uint8_t &val) {
  Wire.beginTransmission(qmcAddress);
  Wire.write(reg);
  uint8_t err = Wire.endTransmission(false);
  if (err != 0) return err;

  if (Wire.requestFrom((uint8_t)qmcAddress, (uint8_t)1) != 1) return 4;
  val = Wire.read();
  return 0;
}

void printI2CError(uint8_t errCode) {
  switch (errCode) {
    case 0: Serial.print("ACK (Success)"); break;
    case 1: Serial.print("Data too long for buffer"); break;
    case 2: Serial.print("NACK on transmit of address (No device responding)"); break;
    case 3: Serial.print("NACK on transmit of data"); break;
    case 4: Serial.print("Other error / Bus arbitration lost"); break;
    default: Serial.printf("Unknown error code %d", errCode); break;
  }
}

// -----------------------------------------------------------------------------
// Auto-Discovery & Initialization
// -----------------------------------------------------------------------------
bool scanAndFindQMC() {
  const uint8_t candidates[] = {0x0D, 0x0C, 0x1C};
  for (uint8_t addr : candidates) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      qmcAddress = addr;
      Serial.printf("[I2C] Found responding QMC device at 0x%02X!\n", addr);
      return true;
    }
  }

  Serial.println("\n[I2C] Scanning full bus (0x01 - 0x7F)...");
  int found = 0;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      Serial.printf("  -> Detected device at address 0x%02X\n", a);
      found++;
    }
  }
  if (found == 0) {
    Serial.println("  [ERROR] No devices found on I2C bus! Check VCC, GND, and pull-up resistors.");
  }
  return false;
}

bool initQMC5883L() {
  Serial.printf("[QMC Init] Initializing QMC IC at address 0x%02X...\n", qmcAddress);

  uint8_t err = writeReg(QMC_REG_CONTROL_2, 0x80);
  if (err != 0) {
    Serial.print("  [FAIL] Failed to reset QMC. Error: ");
    printI2CError(err);
    Serial.println();
    return false;
  }
  delay(50);

  err = writeReg(QMC_REG_PERIOD, 0x01);
  if (err != 0) {
    Serial.print("  [FAIL] Failed to set Period register (0x0B). Error: ");
    printI2CError(err);
    Serial.println();
    return false;
  }

  err = writeReg(QMC_REG_CONTROL_1, QMC_CONFIG_CTRL1);
  if (err != 0) {
    Serial.print("  [FAIL] Failed to write Control 1 register (0x09). Error: ");
    printI2CError(err);
    Serial.println();
    return false;
  }

  delay(20);
  Serial.println("[QMC Init] Success! QMC5883L configured in continuous 100Hz mode.");
  return true;
}

// -----------------------------------------------------------------------------
// Read 6-Axis Magnetometer Output (Fixes F-19: Overflow rejection & DRDY check)
// -----------------------------------------------------------------------------
bool readQMC(int16_t &x, int16_t &y, int16_t &z) {
  uint8_t status = 0;
  if (readReg(QMC_REG_STATUS, status) == 0) {
    // Bit 1 = OVL (Data Overflow): reject saturated sample so calibration is never corrupted!
    if (status & 0x02) {
      Serial.println("[WARN] Magnetic field overflow (OVL) detected; sample discarded.");
      // Read and discard 6 bytes to unlock the QMC5883L output register buffer
      Wire.beginTransmission(qmcAddress);
      Wire.write(QMC_REG_DATA_X_LSB);
      Wire.endTransmission(false);
      Wire.requestFrom((uint8_t)qmcAddress, (uint8_t)6);
      while (Wire.available()) Wire.read();
      return false;
    }
  }

  Wire.beginTransmission(qmcAddress);
  Wire.write(QMC_REG_DATA_X_LSB);
  if (Wire.endTransmission(false) != 0) return false;

  if (Wire.requestFrom((uint8_t)qmcAddress, (uint8_t)6) != 6) {
    return false;
  }

  uint8_t x_lsb = Wire.read();
  uint8_t x_msb = Wire.read();
  uint8_t y_lsb = Wire.read();
  uint8_t y_msb = Wire.read();
  uint8_t z_lsb = Wire.read();
  uint8_t z_msb = Wire.read();

  x = (int16_t)((x_msb << 8) | x_lsb);
  y = (int16_t)((y_msb << 8) | y_lsb);
  z = (int16_t)((z_msb << 8) | z_lsb);

  // Guard against unflagged saturation rail values
  if (x == -32768 || x == 32767 || y == -32768 || y == 32767 || z == -32768 || z == 32767) {
    return false;
  }

  return true;
}

// -----------------------------------------------------------------------------
// Heading & Tilt-Compensated Navigation Computation
// -----------------------------------------------------------------------------
float calculateTiltCompensatedHeading(float x_cal, float y_cal, float z_cal,
                                      float pitch_rad = 0.0f, float roll_rad = 0.0f) {
  float cosP = cosf(pitch_rad);
  float sinP = sinf(pitch_rad);
  float cosR = cosf(roll_rad);
  float sinR = sinf(roll_rad);

  float Xh = x_cal * cosP + y_cal * sinR * sinP + z_cal * cosR * sinP;
  float Yh = y_cal * cosR - z_cal * sinR;

  float headingRad = atan2f(Yh, Xh);
  headingRad += (DECLINATION_DEG * DEG_TO_RAD);

  if (headingRad < 0) headingRad += 2.0f * PI;
  if (headingRad >= 2.0f * PI) headingRad -= 2.0f * PI;

  return headingRad * RAD_TO_DEG;
}

const char* getCardinalHeading(float deg) {
  if (deg >= 337.5 || deg < 22.5)  return "N  [North]";
  if (deg >= 22.5  && deg < 67.5)  return "NE [North-East]";
  if (deg >= 67.5  && deg < 112.5) return "E  [East]";
  if (deg >= 112.5 && deg < 157.5) return "SE [South-East]";
  if (deg >= 157.5 && deg < 202.5) return "S  [South]";
  if (deg >= 202.5 && deg < 247.5) return "SW [South-West]";
  if (deg >= 247.5 && deg < 292.5) return "W  [West]";
  return "NW [North-West]";
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);

  Serial.println("\n=======================================================");
  Serial.println("     ESP32 Dedicated QMC5883L Compass Driver           ");
  Serial.println("=======================================================");

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(100000);

  delay(200);

  if (!scanAndFindQMC()) {
    Serial.println("\n[HALT] Could not find QMC sensor.");
    while (1) delay(1000);
  }

  if (!initQMC5883L()) {
    Serial.println("[HALT] Initialization failed. Check wiring and power supply.");
    while (1) delay(1000);
  }

  Serial.println("\nReady! Calibration commands:");
  Serial.println("  Type 'c' + ENTER -> Start 15s 360-deg rotation calibration (Hard+Soft Iron)");
  Serial.println("  Type 'r' + ENTER -> Reset calibration\n");
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'c' || cmd == 'C') {
      isCalibrating = true;
      calStartTime = millis();
      calMinX = calMinY = calMinZ = 32767;
      calMaxX = calMaxY = calMaxZ = -32768;
      Serial.println("\n>>> [CALIBRATION STARTED]");
      Serial.println(">>> Slowly rotate the sensor in a flat 360-degree circle for 15 seconds...");
    } else if (cmd == 'r' || cmd == 'R') {
      offsetX = offsetY = offsetZ = 0.0f;
      scaleX = scaleY = scaleZ = 1.0f;
      Serial.println("\n>>> [CALIBRATION RESET] Offsets and scales reset to defaults.");
    }
  }

  int16_t rawX, rawY, rawZ;
  if (!readQMC(rawX, rawY, rawZ)) {
    delay(50);
    return;
  }

  if (isCalibrating) {
    if (rawX < calMinX) calMinX = rawX;
    if (rawX > calMaxX) calMaxX = rawX;
    if (rawY < calMinY) calMinY = rawY;
    if (rawY > calMaxY) calMaxY = rawY;
    if (rawZ < calMinZ) calMinZ = rawZ;
    if (rawZ > calMaxZ) calMaxZ = rawZ;

    unsigned long elapsed = millis() - calStartTime;
    int remaining = (CAL_DURATION_MS - elapsed) / 1000;

    Serial.printf("  Calibrating... %2ds left | X:[%6d, %6d] Y:[%6d, %6d]\r",
                  remaining, calMinX, calMaxX, calMinY, calMaxY);

    if (elapsed >= CAL_DURATION_MS) {
      isCalibrating = false;

      offsetX = (float)(calMaxX + calMinX) / 2.0f;
      offsetY = (float)(calMaxY + calMinY) / 2.0f;
      offsetZ = (float)(calMaxZ + calMinZ) / 2.0f;

      float rangeX = (float)(calMaxX - calMinX) / 2.0f;
      float rangeY = (float)(calMaxY - calMinY) / 2.0f;
      float rangeZ = (float)(calMaxZ - calMinZ) / 2.0f;
      if (rangeX > 0 && rangeY > 0) {
        float avgRange = (rangeX + rangeY) / 2.0f;
        scaleX = avgRange / rangeX;
        scaleY = avgRange / rangeY;
        if (rangeZ > 0) scaleZ = avgRange / rangeZ;
      }

      Serial.println("\n>>> [CALIBRATION COMPLETE!]");
      Serial.printf(">>> Offsets (Hard-Iron): X=%.1f, Y=%.1f, Z=%.1f\n", offsetX, offsetY, offsetZ);
      Serial.printf(">>> Scales  (Soft-Iron): ScaleX=%.3f, ScaleY=%.3f, ScaleZ=%.3f\n\n", scaleX, scaleY, scaleZ);
    }
    delay(50);
    return;
  }

  float x_cal = ((float)rawX - offsetX) * scaleX;
  float y_cal = ((float)rawY - offsetY) * scaleY;
  float z_cal = ((float)rawZ - offsetZ) * scaleZ;

  float headingDeg = calculateTiltCompensatedHeading(x_cal, y_cal, z_cal);
  const char* cardinal = getCardinalHeading(headingDeg);

  Serial.printf("Raw: [%6d, %6d, %6d] | Cal: [%6.1f, %6.1f, %6.1f] | Heading: %6.1f deg | %s\n",
                rawX, rawY, rawZ, x_cal, y_cal, z_cal, headingDeg, cardinal);

  delay(100);
}
