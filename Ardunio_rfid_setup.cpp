#include <SPI.h>
#include <MFRC522.h>

#define RST_PIN 9
#define SS_PIN 10

#define FLAME_SENSOR_PIN A0
#define BUZZER_PIN 8+

MFRC522 mfrc522(SS_PIN, RST_PIN);

const int FLAME_THRESHOLD = 300;
const unsigned long BUZZ_DURATION_MS = 1000;

unsigned long buzzStopTime = 0;
bool flame_active = false;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  SPI.begin();
  mfrc522.PCD_Init();
  pinMode(FLAME_SENSOR_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.println("System Ready. Scan your card.");
}

void loop() {
  checkRFID();
  checkSerialInput();
  checkFlameSensor();
  manageBuzzer();
  delay(50);
}

void checkRFID() {
  if (mfrc522.PICC_IsNewCardPresent() && mfrc522.PICC_ReadCardSerial()) {
    String cardId = "";
    for (byte i = 0; i < mfrc522.uid.size; i++) {
      if (mfrc522.uid.uidByte[i] < 0x10) {
         cardId += "0";
      }
      cardId += String(mfrc522.uid.uidByte[i], HEX);
    }
    cardId.toUpperCase();
    Serial.print("Card ID: ");
    Serial.println(cardId);
    mfrc522.PICC_HaltA();
    mfrc522.PCD_StopCrypto1();
    delay(200);
  }
}

void checkSerialInput() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command == "BUZZ") {
      digitalWrite(BUZZER_PIN, HIGH);
      buzzStopTime = millis() + BUZZ_DURATION_MS;
    }
  }
}

void checkFlameSensor() {
  int flameState = analogRead(FLAME_SENSOR_PIN);
  if (flameState < FLAME_THRESHOLD) {
    if (!flame_active) {
      Serial.println("Flame detected! Buzzer activated.");
      flame_active = true;
      digitalWrite(BUZZER_PIN, HIGH);
    }
    digitalWrite(BUZZER_PIN, HIGH);
  } else {
    if (flame_active) {
      Serial.println("Flame cleared.");
      flame_active = false;
    }
  }
}

void manageBuzzer() {
  unsigned long currentTime = millis();
  if (buzzStopTime > 0 && currentTime >= buzzStopTime) {
    buzzStopTime = 0;
    if (!flame_active) {
      digitalWrite(BUZZER_PIN, LOW);
    }
  }
  if (!flame_active && buzzStopTime == 0) {
     digitalWrite(BUZZER_PIN, LOW);
  }
}
