#include <Servo.h>

Servo thumb, indexF, middle, ring, little, wristS;
const int pins[6] = {3, 5, 6, 9, 12, 11};  // T, I, M, R, L, W

const int RELAX[5] = {180, 180, 180, 180, 0};
const int CLOSE_[5] = {100, 70, 55, 80, 100};
const int WRIST_RELAX = 90;
const int WRIST_LEFT  = 180;
const int WRIST_RIGHT = 0;

long currentCmdId = -1;

void attachAll() {
  thumb.attach(pins[0]);
  indexF.attach(pins[1]);
  middle.attach(pins[2]);
  ring.attach(pins[3]);
  little.attach(pins[4]);
  wristS.attach(pins[5]);
}

int clampVal(int v, int a, int b) { return v < a ? a : (v > b ? b : v); }

int parseKV(const String &s, char key, int fallback) {
  int k = s.indexOf(key);
  if (k == -1) return fallback;
  int start = k + 1;
  int end = s.indexOf(',', start);
  if (end == -1) end = s.length();
  return s.substring(start, end).toInt();
}

long parseCmdId(const String &s) {
  int c = s.indexOf('C');
  if (c == -1) return -1;
  int semi = s.indexOf(';', c + 1);
  return s.substring(c + 1, semi).toInt();
}

void applyAngles(int T, int I, int M, int R, int L, int W) {
  thumb.write(clampVal(T, 0, 180));
  indexF.write(clampVal(I, 0, 180));
  middle.write(clampVal(M, 0, 180));
  ring.write(clampVal(R, 0, 180));
  little.write(clampVal(L, 0, 180));
  wristS.write(clampVal(W, 0, 180));
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(5);
  attachAll();

  // Start relaxed
  applyAngles(RELAX[0], RELAX[1], RELAX[2], RELAX[3], RELAX[4], WRIST_RELAX);
  delay(200);
  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    long cid = parseCmdId(line);
    int T = parseKV(line, 'T', RELAX[0]);
    int I = parseKV(line, 'I', RELAX[1]);
    int M = parseKV(line, 'M', RELAX[2]);
    int R = parseKV(line, 'R', RELAX[3]);
    int L = parseKV(line, 'L', RELAX[4]);
    int W = parseKV(line, 'W', WRIST_RELAX);

    // Apply immediately
    applyAngles(T, I, M, R, L, W);

    // Instant feedback
    if (cid >= 0) {
      Serial.print("DONE,");
      Serial.println(cid);
    }
  }
}
