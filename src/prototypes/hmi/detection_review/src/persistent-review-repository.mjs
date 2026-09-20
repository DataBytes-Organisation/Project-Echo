import {
  recordReviewConflict,
  restoreReviewCase,
} from "./review-domain.mjs";
import { StaleReviewVersionError } from "./review-repository.mjs";

export const REVIEW_STORAGE_KEY = "echo-detection-review:cases:v1";
export const REVIEW_SCHEMA_VERSION = 1;
export const REVIEW_STORAGE_ERROR_MESSAGE =
  "Saved review progress could not be accessed in this browser.";

export class PersistentReviewStorageError extends Error {
  constructor(options) {
    super(REVIEW_STORAGE_ERROR_MESSAGE, options);
    this.name = "PersistentReviewStorageError";
    this.userMessage = REVIEW_STORAGE_ERROR_MESSAGE;
  }
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function storageError(error) {
  return error instanceof PersistentReviewStorageError
    ? error
    : new PersistentReviewStorageError({ cause: error });
}

function restoreCases(cases) {
  if (!Array.isArray(cases)) {
    throw new Error("Saved review payload cases are invalid.");
  }

  const restoredCases = cases.map(restoreReviewCase);
  const detectionIds = new Set();
  for (const reviewCase of restoredCases) {
    if (detectionIds.has(reviewCase.detectionId)) {
      throw new Error("Saved review payload contains duplicate detection IDs.");
    }
    detectionIds.add(reviewCase.detectionId);
  }

  return restoredCases;
}

function restoreSeeds(seeds) {
  try {
    return restoreCases(seeds);
  } catch (error) {
    throw storageError(error);
  }
}

export class PersistentReviewWorkflowRepository {
  #storage;
  #seeds;
  #conflictOnNextSave;
  #initialConflictOnNextSave;
  #now;

  constructor(storage, seeds, {
    conflictOnNextSave = false,
    now = () => new Date().toISOString(),
  } = {}) {
    this.#storage = storage;
    this.#seeds = restoreSeeds(seeds);
    this.#conflictOnNextSave = conflictOnNextSave;
    this.#initialConflictOnNextSave = conflictOnNextSave;
    this.#now = now;
    this.#ensurePayload();
  }

  #getItem() {
    try {
      return this.#storage.getItem(REVIEW_STORAGE_KEY);
    } catch (error) {
      throw storageError(error);
    }
  }

  #setItem(payload) {
    try {
      this.#storage.setItem(REVIEW_STORAGE_KEY, JSON.stringify(payload));
    } catch (error) {
      throw storageError(error);
    }
  }

  #readPayload() {
    const rawPayload = this.#getItem();
    if (rawPayload === null) {
      return null;
    }

    try {
      const payload = JSON.parse(rawPayload);
      if (!isRecord(payload) || payload.schemaVersion !== REVIEW_SCHEMA_VERSION) {
        throw new Error("Saved review payload schema is unsupported.");
      }

      return {
        schemaVersion: REVIEW_SCHEMA_VERSION,
        cases: restoreCases(payload.cases),
      };
    } catch (error) {
      throw storageError(error);
    }
  }

  #writeCases(cases) {
    let restoredCases;
    try {
      restoredCases = restoreCases(cases);
    } catch (error) {
      throw storageError(error);
    }

    this.#setItem({
      schemaVersion: REVIEW_SCHEMA_VERSION,
      cases: restoredCases,
    });
  }

  #ensurePayload() {
    if (this.#readPayload() === null) {
      this.#writeCases(this.#seeds);
    }
  }

  #latestPayload() {
    const payload = this.#readPayload();
    if (payload === null) {
      this.#writeCases(this.#seeds);
      return {
        schemaVersion: REVIEW_SCHEMA_VERSION,
        cases: this.#seeds,
      };
    }
    return payload;
  }

  async loadCase(detectionId) {
    const payload = this.#latestPayload();
    const reviewCase = payload.cases.find(item => item.detectionId === detectionId);
    return reviewCase ?? null;
  }

  async saveCase(reviewCase, expectedVersion) {
    const payload = this.#latestPayload();
    const caseIndex = payload.cases.findIndex(item => item.detectionId === reviewCase?.detectionId);
    if (caseIndex === -1) {
      throw new Error("Cannot save a review case that is not in this persistent repository.");
    }

    let nextCase;
    try {
      nextCase = restoreReviewCase(reviewCase);
    } catch (error) {
      throw storageError(error);
    }
    if (nextCase.version !== expectedVersion + 1) {
      throw new Error("A saved review case must advance the expected version exactly once.");
    }

    let currentCase = payload.cases[caseIndex];
    if (this.#conflictOnNextSave) {
      this.#conflictOnNextSave = false;
      currentCase = recordReviewConflict(
        currentCase,
        "persistent-concurrent-review",
        this.#now(),
      );
      const conflictedCases = [...payload.cases];
      conflictedCases[caseIndex] = currentCase;
      this.#writeCases(conflictedCases);
      throw new StaleReviewVersionError(
        expectedVersion,
        currentCase.version,
        currentCase,
      );
    }

    if (expectedVersion !== currentCase.version) {
      throw new StaleReviewVersionError(
        expectedVersion,
        currentCase.version,
        currentCase,
      );
    }

    const savedCases = [...payload.cases];
    savedCases[caseIndex] = nextCase;
    this.#writeCases(savedCases);
    return nextCase;
  }

  async listAdjudication() {
    return this.#latestPayload().cases.filter(
      reviewCase => reviewCase.status === "awaiting_adjudication",
    );
  }

  async reset() {
    this.#latestPayload();
    this.#writeCases(this.#seeds);
    this.#conflictOnNextSave = this.#initialConflictOnNextSave;
  }
}

export function createBrowserPersistentReviewRepository(windowObject, seeds, options) {
  let storage;
  try {
    storage = windowObject.localStorage;
  } catch (error) {
    throw storageError(error);
  }

  try {
    return new PersistentReviewWorkflowRepository(storage, seeds, options);
  } catch (error) {
    throw storageError(error);
  }
}
