const QUEUE_STATUSES = new Set(["pending_review"]);
const ISO_DATE_TIME_PATTERN = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?(Z|[+-]\d{2}:\d{2})$/;

export class DetectionValidationError extends Error {
  constructor(field, message) {
    super(`${field}: ${message}`);
    this.name = "DetectionValidationError";
    this.field = field;
  }
}

function fail(field, message) {
  throw new DetectionValidationError(field, message);
}

function assertRecordObject(record) {
  if (record === null || typeof record !== "object" || Array.isArray(record)) {
    fail("record", "must be an object");
  }
}

function assertNonEmptyString(value, field) {
  if (typeof value !== "string" || value.trim().length === 0) {
    fail(field, "must be a non-empty string");
  }
}

function assertIsoTimestamp(value) {
  assertNonEmptyString(value, "timestamp");

  const match = ISO_DATE_TIME_PATTERN.exec(value);

  if (!match) {
    fail("timestamp", "must be a valid ISO 8601 timestamp");
  }

  const [, yearText, monthText, dayText, hourText, minuteText, secondText, , zone] = match;
  const [year, month, day, hour, minute, second] = [
    yearText,
    monthText,
    dayText,
    hourText,
    minuteText,
    secondText,
  ].map(Number);
  const calendarProbe = new Date(0);
  calendarProbe.setUTCHours(hour, minute, second, 0);
  calendarProbe.setUTCFullYear(year, month - 1, day);
  const calendarIsValid = year >= 1
    && hour <= 23
    && minute <= 59
    && second <= 59
    && calendarProbe.getUTCFullYear() === year
    && calendarProbe.getUTCMonth() === month - 1
    && calendarProbe.getUTCDate() === day
    && calendarProbe.getUTCHours() === hour
    && calendarProbe.getUTCMinutes() === minute
    && calendarProbe.getUTCSeconds() === second;
  const [, offsetHourText = "0", offsetMinuteText = "0"] = zone.match(/[+-](\d{2}):(\d{2})/) ?? [];
  const offsetHours = Number(offsetHourText);
  const offsetMinutes = Number(offsetMinuteText);
  const zoneIsValid = zone === "Z"
    || (offsetHours <= 14 && offsetMinutes <= 59 && (offsetHours < 14 || offsetMinutes === 0));

  if (!calendarIsValid || !zoneIsValid || Number.isNaN(Date.parse(value))) {
    fail("timestamp", "must be a valid ISO 8601 timestamp");
  }
}

function assertNumberInRange(value, field, minimum, maximum) {
  if (!Number.isFinite(value) || value <= minimum || value >= maximum) {
    fail(field, `must be greater than ${minimum} and less than ${maximum}`);
  }
}

function assertLla(value, field) {
  if (!Array.isArray(value)
    || value.length !== 3
    || value.some(coordinate => !Number.isFinite(coordinate))) {
    fail(field, "must contain exactly three finite numbers");
  }
}

function assertNonNegativeNumber(value, field) {
  if (!Number.isFinite(value) || value < 0) {
    fail(field, "must be a non-negative number");
  }
}

function assertBoolean(value, field) {
  if (typeof value !== "boolean") {
    fail(field, "must be a boolean");
  }
}

function assertPositiveInteger(value, field) {
  if (!Number.isInteger(value) || value < 1) {
    fail(field, "must be a positive integer");
  }
}

function assertQueueStatus(value) {
  if (!QUEUE_STATUSES.has(value)) {
    fail("queueStatus", "must be pending_review in the Week 1 prototype");
  }
}

function freezeDetection(rawRecord) {
  return Object.freeze({
    id: rawRecord.id.trim(),
    timestamp: rawRecord.timestamp,
    sensorId: rawRecord.sensorId.trim(),
    species: rawRecord.species.trim(),
    confidence: rawRecord.confidence,
    microphoneLLA: Object.freeze([...rawRecord.microphoneLLA]),
    animalEstLLA: Object.freeze([...rawRecord.animalEstLLA]),
    animalLLAUncertainty: rawRecord.animalLLAUncertainty,
    audioAvailable: rawRecord.audioAvailable,
    sampleRate: rawRecord.sampleRate,
    queueStatus: rawRecord.queueStatus,
    version: rawRecord.version,
  });
}

export function validateDetectionRecord(rawRecord) {
  assertRecordObject(rawRecord);
  assertNonEmptyString(rawRecord.id, "id");
  assertIsoTimestamp(rawRecord.timestamp);
  assertNonEmptyString(rawRecord.sensorId, "sensorId");
  assertNonEmptyString(rawRecord.species, "species");
  assertNumberInRange(rawRecord.confidence, "confidence", 0, 100);
  assertLla(rawRecord.microphoneLLA, "microphoneLLA");
  assertLla(rawRecord.animalEstLLA, "animalEstLLA");
  assertNonNegativeNumber(rawRecord.animalLLAUncertainty, "animalLLAUncertainty");
  assertBoolean(rawRecord.audioAvailable, "audioAvailable");
  assertPositiveInteger(rawRecord.sampleRate, "sampleRate");
  assertQueueStatus(rawRecord.queueStatus);
  assertPositiveInteger(rawRecord.version, "version");

  return freezeDetection(rawRecord);
}
