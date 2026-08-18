import { recordReviewConflict } from "./review-domain.mjs";

export class ReviewWorkflowRepository {
  async loadCase(_detectionId) {
    throw new Error("loadCase() must be implemented");
  }

  async saveCase(_reviewCase, _expectedVersion) {
    throw new Error("saveCase() must be implemented");
  }

  async listAdjudication() {
    throw new Error("listAdjudication() must be implemented");
  }

  async reset() {
    throw new Error("reset() must be implemented");
  }
}

export class StaleReviewVersionError extends Error {
  constructor(expectedVersion, actualVersion, latestCase) {
    super(`Expected review version ${expectedVersion}, but found ${actualVersion}.`);
    this.name = "StaleReviewVersionError";
    this.expectedVersion = expectedVersion;
    this.actualVersion = actualVersion;
    this.latestCase = immutableSnapshot(latestCase);
  }
}

function immutableSnapshot(reviewCase) {
  const submissions = Object.fromEntries(
    Object.entries(reviewCase.submissions ?? {}).map(([actor, submission]) => [
      actor,
      Object.freeze({ ...submission }),
    ]),
  );
  const history = (reviewCase.history ?? []).map(entry => Object.freeze({ ...entry }));

  return Object.freeze({
    ...reviewCase,
    submissions: Object.freeze(submissions),
    consensus: reviewCase.consensus
      ? Object.freeze({ ...reviewCase.consensus })
      : null,
    adjudication: reviewCase.adjudication
      ? Object.freeze({ ...reviewCase.adjudication })
      : null,
    history: Object.freeze(history),
  });
}

export class FixtureReviewWorkflowRepository extends ReviewWorkflowRepository {
  #cases;
  #conflictOnNextSave;
  #initialCases;
  #initialConflictOnNextSave;
  #now;

  constructor(reviewCases, {
    conflictOnNextSave = false,
    now = () => new Date().toISOString(),
  } = {}) {
    super();
    this.#initialCases = reviewCases.map(immutableSnapshot);
    this.#cases = new Map(this.#initialCases.map(reviewCase => [
      reviewCase.detectionId,
      immutableSnapshot(reviewCase),
    ]));
    this.#conflictOnNextSave = conflictOnNextSave;
    this.#initialConflictOnNextSave = conflictOnNextSave;
    this.#now = now;
  }

  async loadCase(detectionId) {
    const reviewCase = this.#cases.get(detectionId);
    return reviewCase ? immutableSnapshot(reviewCase) : null;
  }

  async saveCase(reviewCase, expectedVersion) {
    if (!this.#cases.has(reviewCase.detectionId)) {
      throw new Error("Cannot save a review case that is not in this fixture repository.");
    }

    let currentCase = this.#cases.get(reviewCase.detectionId);

    if (this.#conflictOnNextSave) {
      this.#conflictOnNextSave = false;
      currentCase = immutableSnapshot(recordReviewConflict(
        currentCase,
        "fixture-concurrent-review",
        this.#now(),
      ));
      this.#cases.set(reviewCase.detectionId, currentCase);
    }

    if (expectedVersion !== currentCase.version) {
      throw new StaleReviewVersionError(
        expectedVersion,
        currentCase.version,
        currentCase,
      );
    }

    if (reviewCase.version !== expectedVersion + 1) {
      throw new Error("A saved review case must advance the expected version exactly once.");
    }

    const snapshot = immutableSnapshot(reviewCase);
    this.#cases.set(reviewCase.detectionId, snapshot);
    return immutableSnapshot(snapshot);
  }

  async listAdjudication() {
    return [...this.#cases.values()]
      .filter(reviewCase => reviewCase.status === "awaiting_adjudication")
      .map(immutableSnapshot);
  }

  async reset() {
    this.#cases = new Map(this.#initialCases.map(reviewCase => [
      reviewCase.detectionId,
      immutableSnapshot(reviewCase),
    ]));
    this.#conflictOnNextSave = this.#initialConflictOnNextSave;
  }
}
