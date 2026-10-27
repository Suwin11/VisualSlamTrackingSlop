// #include <Arduino.h>
// #include "MotorV2.h"

// #define MOTOR_ID_LEFT  1
// #define MOTOR_ID_RIGHT 2

// void setup() {
//   Serial.begin(115200);

//   // 1. ตั้งค่า Baud Rate สื่อสารกับ Dynamixel (115200 bps)
//   Serial2.begin(115200, SERIAL_8N1, 16, 17);
//   pinMode(4, OUTPUT);
//   digitalWrite(4, LOW); // เริ่มต้นโหมดรับข้อมูล

//   delay(1000);
//   Serial.println("--- Starting Motor Test (Baudrate: 115200) ---");

//   // 2. ลำดับขั้นตอนบังคับตามคำแนะนำของรุ่นพี่และเพื่อน
//   // Step A: ปิด Torque ก่อนเปลี่ยนโหมด
//   noTorque(MOTOR_ID_LEFT);
//   noTorque(MOTOR_ID_RIGHT);
//   delay(100);

//   // Step B: ตั้งโหมดเป็น Velocity Control (หมุนต่อเนื่อง)
//   setMode(MOTOR_ID_LEFT);
//   setMode(MOTOR_ID_RIGHT);
//   delay(100);

//   // Step C: เปิด Torque ล็อกตำแหน่ง
//   enableTorque(MOTOR_ID_LEFT);
//   enableTorque(MOTOR_ID_RIGHT);
//   delay(100);

//   Serial.println("Motors Initialized Successfully!");
// }

// void loop() {
//   // --- 1. สั่งเดินหน้า (30 RPM) 5 วินาที ---
//   Serial.println("Action: Moving Forward...");
//   turnWheel(MOTOR_ID_LEFT,  LEFT,  30);
//   turnWheel(MOTOR_ID_RIGHT, RIGHT, 30);
//   delay(5000);

//   // --- 2. หยุดหุ่นยนต์ 2 วินาที ---
//   Serial.println("Action: Stopping...");
//   STOP(MOTOR_ID_LEFT);
//   STOP(MOTOR_ID_RIGHT);
//   delay(2000);

//   // --- 3. ถอยหลัง (30 RPM) 5 วินาที ---
//   Serial.println("Action: Moving Backward...");
//   turnWheel(MOTOR_ID_LEFT,  RIGHT, 30); // สลับทิศ
//   turnWheel(MOTOR_ID_RIGHT, LEFT,  30); // สลับทิศ
//   delay(5000);

//   // --- 4. หยุดหุ่นยนต์ 2 วินาที ---
//   Serial.println("Action: Stopping...");
//   STOP(MOTOR_ID_LEFT);
//   STOP(MOTOR_ID_RIGHT);
//   delay(2000);
// }