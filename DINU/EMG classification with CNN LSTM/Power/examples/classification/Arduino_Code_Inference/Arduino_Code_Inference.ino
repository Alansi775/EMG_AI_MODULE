#include <Servo.h>

// ====== Servo setup ======
Servo thumb, indexF, middle, ring, little, wrist;
const int pins[6] = {3, 5, 6, 9, 12, 11}; // T,I,M,R,L,W

// ====== Angle presets ======
const int RELAX[5] = {180, 180, 180, 180, 0};
const int CLOSE_[5]= {100, 70, 55, 80, 100};
const int WRIST_RELAX = 90;

// ====== Motion control ======
int currentPos[6];      // current servo positions
int targetPos[6];       // goal positions
unsigned long lastStepMs = 0;

const int STEP_DEG = 3;        // degrees per step (smaller = smoother)
const unsigned long STEP_DT = 10; // ms between steps (~300°/s)

// ====== Helpers ======
int clamp(int v, int a, int b){ return v < a ? a : (v > b ? b : v); }

int parseVal(const String &s, char key, int fallback) {
  int i = s.indexOf(key);
  if (i == -1) return fallback;
  int start = i + 1;
  int end = s.indexOf(',', start);
  if (end == -1) end = s.length();
  String num = s.substring(start, end);
  num.trim();
  return num.toInt();
}

void writeAll() {
  thumb.write(currentPos[0]);
  indexF.write(currentPos[1]);
  middle.write(currentPos[2]);
  ring.write(currentPos[3]);
  little.write(currentPos[4]);
  wrist.write(currentPos[5]);
}

void stepTowardTargets() {
  unsigned long now = millis();
  if (now - lastStepMs < STEP_DT) return;
  lastStepMs = now;

  bool anyChange = false;
  for (int i = 0; i < 6; ++i) {
    int cur = currentPos[i], tgt = targetPos[i];
    if (cur == tgt) continue;
    int dir = (tgt > cur) ? 1 : -1;
    cur += dir * STEP_DEG;
    if ((dir > 0 && cur > tgt) || (dir < 0 && cur < tgt)) cur = tgt;
    currentPos[i] = cur;
    anyChange = true;
  }
  if (anyChange) writeAll();
}

// ====== Command handling ======
void setTargets(const String& cmd) {
  int rec[5];
  for (int i = 0; i < 5; ++i)
    rec[i] = parseVal(cmd, "TIMRL"[i], targetPos[i]);

  // Wrist
  targetPos[5] = clamp(parseVal(cmd, 'W', targetPos[5]), 0, 180);

  // Single-finger invert logic
  int closeCount = 0, lastClose = -1;
  for (int i = 0; i < 5; ++i) {
    int dClose = abs(rec[i] - CLOSE_[i]);
    int dRelax = abs(rec[i] - RELAX[i]);
    if (dClose <= dRelax) { closeCount++; lastClose = i; }
  }
  if (closeCount == 1) {
    for (int i = 0; i < 5; ++i)
      targetPos[i] = (i == lastClose) ? RELAX[i] : CLOSE_[i];
  } else {
    for (int i = 0; i < 5; ++i)
      targetPos[i] = clamp(rec[i], 0, 180);
  }
}

// ====== Setup ======
void setup() {
  Serial.begin(9600);
  thumb.attach(pins[0]);
  indexF.attach(pins[1]);
  middle.attach(pins[2]);
  ring.attach(pins[3]);
  little.attach(pins[4]);
  wrist.attach(pins[5]);

  for (int i = 0; i < 5; ++i)
    currentPos[i] = targetPos[i] = RELAX[i];
  currentPos[5] = targetPos[5] = WRIST_RELAX;
  writeAll();

  delay(300);
  Serial.println("READY");
}

// ====== Main loop ======
void loop() {
  stepTowardTargets();

  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    if (line == "PING") {
      Serial.println("PONG");
      return;
    }

    setTargets(line);
  }
}
