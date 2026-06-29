#include <Servo.h>

Servo thumb, index, middle, ring, little, wrist;

const int pins[6] = {3, 5, 6, 9, 12, 11}; // T,I,M,R,L,W
const int relaxAngles[5] = {180, 180, 180, 180, 0};
const int closeAngles[5] = {100, 70, 55, 80, 100};

// Wrist
const int wristRelax = 90;
int wristPos = wristRelax;       // Current wrist position
int wristLastPos = wristRelax;   // Last written position to prevent jitter

void setup() {
  Serial.begin(9600);

  thumb.attach(pins[0]);
  index.attach(pins[1]);
  middle.attach(pins[2]);
  ring.attach(pins[3]);
  little.attach(pins[4]);
  wrist.attach(pins[5]);

  thumb.write(relaxAngles[0]);
  index.write(relaxAngles[1]);
  middle.write(relaxAngles[2]);
  ring.write(relaxAngles[3]);
  little.write(relaxAngles[4]);
  wrist.write(wristPos);
}

int parseValue(const String &s, char key, int defaultVal) {
  int idx = s.indexOf(key);
  if (idx == -1) return defaultVal;
  int start = idx + 1;
  int end = s.indexOf(',', start);
  if (end == -1) end = s.length();
  String numStr = s.substring(start, end);
  numStr.trim();
  return numStr.toInt();
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    int rec[5];
    rec[0] = parseValue(cmd, 'T', relaxAngles[0]);
    rec[1] = parseValue(cmd, 'I', relaxAngles[1]);
    rec[2] = parseValue(cmd, 'M', relaxAngles[2]);
    rec[3] = parseValue(cmd, 'R', relaxAngles[3]);
    rec[4] = parseValue(cmd, 'L', relaxAngles[4]);

    // Wrist update
    int wristCmdIdx = cmd.indexOf('W');
    if (wristCmdIdx != -1) {
      int start = wristCmdIdx + 1;
      int end = cmd.indexOf(',', start);
      if (end == -1) end = cmd.length();
      wristPos = cmd.substring(start, end).toInt(); // only update if command exists
    }

    // Single finger invert logic
    int closeCount = 0;
    int lastCloseIdx = -1;
    for (int i = 0; i < 5; ++i) {
      int dClose = abs(rec[i] - closeAngles[i]);
      int dRelax = abs(rec[i] - relaxAngles[i]);
      if (dClose <= dRelax) {
        closeCount++;
        lastCloseIdx = i;
      }
    }

    int target[5];
    if (closeCount == 1) {
      for (int i = 0; i < 5; ++i)
        target[i] = (i == lastCloseIdx) ? relaxAngles[i] : closeAngles[i];
    } else {
      for (int i = 0; i < 5; ++i)
        target[i] = rec[i];
    }

    // Write fingers
    thumb.write(target[0]);
    index.write(target[1]);
    middle.write(target[2]);
    ring.write(target[3]);
    little.write(target[4]);

    // ===== Write wrist ONLY if it changed =====
    if (wristPos != wristLastPos) {
      wrist.write(wristPos);
      wristLastPos = wristPos;
    }
  }
}
