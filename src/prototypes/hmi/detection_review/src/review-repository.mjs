export class ReviewWorkflowRepository {
  async loadCase(_detectionId) {
    throw new Error("loadCase() must be implemented");
  }

  async saveCase(_reviewCase) {
    throw new Error("saveCase() must be implemented");
  }

  async listAdjudication() {
    throw new Error("listAdjudication() must be implemented");
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

  constructor(reviewCases) {
    super();
    this.#cases = new Map(
      reviewCases.map(reviewCase => [
        reviewCase.detectionId,
        immutableSnapshot(reviewCase),
      ]),
    );
  }

  async loadCase(detectionId) {
    const reviewCase = this.#cases.get(detectionId);
    return reviewCase ? immutableSnapshot(reviewCase) : null;
  }

  async saveCase(reviewCase) {
    if (!this.#cases.has(reviewCase.detectionId)) {
      throw new Error("Cannot save a review case that is not in this fixture repository.");
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
}
