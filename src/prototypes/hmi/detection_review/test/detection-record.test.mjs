import test from "node:test";
import assert from "node:assert/strict";

import {
  DetectionValidationError,
  validateDetectionRecord,
} from "../src/detection-record.mjs";
import { makeDetection } from "../test-support/helpers.mjs";

test("normalizes and freezes a valid detection fixture", () => {
  const rawRecord = makeDetection();
  const record = validateDetectionRecord(rawRecord);

  assert.equal(record.id, "det-echo-001");
  assert.equal(record.species, "Powerful Owl");
  assert.notEqual(record, rawRecord);
  assert.notEqual(record.animalEstLLA, rawRecord.animalEstLLA);
  assert.equal(Object.isFrozen(record), true);
  assert.equal(Object.isFrozen(record.microphoneLLA), true);
  assert.equal(Object.isFrozen(record.animalEstLLA), true);
});

test("rejects every malformed required detection field", () => {
  const malformedRecords = [
    null,
    makeDetection({ id: "" }),
    makeDetection({ timestamp: "not-a-timestamp" }),
    makeDetection({ sensorId: " " }),
    makeDetection({ species: "" }),
    makeDetection({ confidence: 0 }),
    makeDetection({ confidence: 100 }),
    makeDetection({ microphoneLLA: [-37.8, 144.9] }),
    makeDetection({ animalEstLLA: [-37.8, "east", 12] }),
    makeDetection({ animalLLAUncertainty: -1 }),
    makeDetection({ audioAvailable: "yes" }),
    makeDetection({ sampleRate: 0 }),
    makeDetection({ queueStatus: "resolved" }),
    makeDetection({ version: 0 }),
  ];

  for (const malformedRecord of malformedRecords) {
    assert.throws(
      () => validateDetectionRecord(malformedRecord),
      DetectionValidationError,
    );
  }
});

test("rejects non-date-time and calendar-invalid ISO timestamps", () => {
  const malformedTimestamps = [
    "2026-05-22",
    "05/22/2026 04:12",
    "2026-02-31T04:12:00.000Z",
  ];

  for (const timestamp of malformedTimestamps) {
    assert.throws(
      () => validateDetectionRecord(makeDetection({ timestamp })),
      DetectionValidationError,
    );
  }
});

test("accepts backend-compatible ISO microseconds and UTC offsets", () => {
  const timestamp = "2026-05-22T04:12:00.123456+10:00";
  const record = validateDetectionRecord(makeDetection({ timestamp }));

  assert.equal(record.timestamp, timestamp);
});

test("accepts backend-compatible ISO years below 100", () => {
  const timestamp = "0001-01-01T00:00:00Z";
  const record = validateDetectionRecord(makeDetection({ timestamp }));

  assert.equal(record.timestamp, timestamp);
});
