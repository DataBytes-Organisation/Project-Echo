export const BASE_DETECTION = Object.freeze({
  id: "det-echo-001",
  timestamp: "2026-05-22T04:12:00.000Z",
  sensorId: "ECHO-07",
  species: "Powerful Owl",
  confidence: 94.6,
  microphoneLLA: Object.freeze([-37.8142, 144.9628, 18]),
  animalEstLLA: Object.freeze([-37.8136, 144.9631, 24]),
  animalLLAUncertainty: 18,
  audioAvailable: true,
  sampleRate: 48000,
  queueStatus: "pending_review",
  version: 1,
});

export function makeDetection(overrides = {}) {
  const record = { ...BASE_DETECTION, ...overrides };

  if (Array.isArray(record.microphoneLLA)) {
    record.microphoneLLA = [...record.microphoneLLA];
  }

  if (Array.isArray(record.animalEstLLA)) {
    record.animalEstLLA = [...record.animalEstLLA];
  }

  return record;
}

export function createDeferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });

  return { promise, resolve, reject };
}

export function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function createRepositoryDouble(reviewCase, options = {}) {
  let currentCase = reviewCase;
  const saveCalls = [];

  return {
    saveCalls,
    async loadCase(detectionId) {
      return currentCase?.detectionId === detectionId ? currentCase : null;
    },
    async saveCase(nextCase) {
      saveCalls.push(nextCase);

      if (options.saveError) {
        throw options.saveError;
      }

      currentCase = nextCase;
      return nextCase;
    },
    async listAdjudication() {
      return currentCase?.status === "awaiting_adjudication"
        ? [currentCase]
        : [];
    },
  };
}
