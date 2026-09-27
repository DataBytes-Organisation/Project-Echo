import {
  finalizeAdjudication,
  ReviewValidationError,
  submitIndependentReview,
} from "./review-domain.mjs";
import { StaleReviewVersionError } from "./review-repository.mjs";

const REVIEW_SAVE_ERROR = "The review could not be saved. Your submission was not retried.";
const ADJUDICATION_SAVE_ERROR = "The adjudication could not be saved. Your submission was not retried.";
const REVIEW_LOAD_ERROR = "The review workflow could not be loaded. Try again manually.";
const ADJUDICATION_LOAD_ERROR = "The adjudication queue could not be loaded. Try again manually.";

export class ReviewLoadError extends Error {
  constructor(message, options) {
    super(message, options);
    this.name = "ReviewLoadError";
    this.userMessage = message;
  }
}

export class ReviewSubmissionError extends Error {
  constructor(message, options) {
    super(message, options);
    this.name = "ReviewSubmissionError";
    this.userMessage = message;
  }
}

function attemptedValues(command) {
  return Object.freeze({
    decision: typeof command?.decision === "string" ? command.decision : "",
    correctedSpecies: typeof command?.correctedSpecies === "string"
      ? command.correctedSpecies
      : "",
    reason: typeof command?.reason === "string" ? command.reason : "",
    resolutionReason: typeof command?.resolutionReason === "string"
      ? command.resolutionReason
      : "",
  });
}

export class ReviewConflictError extends Error {
  constructor(expectedVersion, actualVersion, latestSession, command, options) {
    super("The review changed before your save completed.", options);
    this.name = "ReviewConflictError";
    this.userMessage = "This review changed before your save completed. Your entered data is preserved.";
    this.expectedVersion = expectedVersion;
    this.actualVersion = actualVersion;
    this.latestSession = latestSession;
    this.attemptedValues = attemptedValues(command);
  }
}

function freezeProjection(projection) {
  return Object.freeze({
    ...projection,
    submissions: Object.freeze({ ...projection.submissions }),
    history: Object.freeze([...projection.history]),
  });
}

export function projectReviewCase(reviewCase, actor) {
  const secondReviewerIsBlind = actor === "reviewer-2"
    && reviewCase.status === "awaiting_second_review";
  const independentReviewsComplete = [
    "consensus",
    "awaiting_adjudication",
    "finalized",
  ].includes(reviewCase.status);
  let submissions = {};
  let history = reviewCase.history;

  if (independentReviewsComplete) {
    submissions = reviewCase.submissions;
  } else if (actor === "reviewer-1" && reviewCase.submissions["reviewer-1"]) {
    submissions = { "reviewer-1": reviewCase.submissions["reviewer-1"] };
  }

  if (secondReviewerIsBlind) {
    history = [];
  }

  return freezeProjection({
    detectionId: reviewCase.detectionId,
    actor,
    status: reviewCase.status,
    version: reviewCase.version,
    firstReviewComplete: Boolean(reviewCase.submissions["reviewer-1"]),
    submissions,
    consensus: reviewCase.consensus,
    adjudication: reviewCase.adjudication,
    history,
  });
}

async function loadRequiredCase(repository, detectionId) {
  let reviewCase;

  try {
    reviewCase = await repository.loadCase(detectionId);
  } catch (error) {
    throw new ReviewLoadError(REVIEW_LOAD_ERROR, { cause: error });
  }

  if (!reviewCase) {
    throw new ReviewLoadError("The requested review case is unavailable.");
  }

  return reviewCase;
}

function expectedVersionFor(_reviewCase, command) {
  if (!Number.isInteger(command?.expectedVersion) || command.expectedVersion < 1) {
    throw new ReviewValidationError(
      "A positive expected review version is required.",
      "expectedVersion",
    );
  }

  return command.expectedVersion;
}

function conflictFromLatest(expectedVersion, reviewCase, actor, command, options) {
  return new ReviewConflictError(
    expectedVersion,
    reviewCase.version,
    projectReviewCase(reviewCase, actor),
    command,
    options,
  );
}

async function saveOnce(repository, nextCase, expectedVersion, failureMessage, actor, command) {
  try {
    return await repository.saveCase(nextCase, expectedVersion);
  } catch (error) {
    if (error instanceof StaleReviewVersionError) {
      throw conflictFromLatest(
        error.expectedVersion,
        error.latestCase,
        actor,
        command,
        { cause: error },
      );
    }

    throw new ReviewSubmissionError(failureMessage, { cause: error });
  }
}

export function createReviewWorkflow(repository, { now = () => new Date().toISOString() } = {}) {
  return Object.freeze({
    async getSession(detectionId, actor) {
      const reviewCase = await loadRequiredCase(repository, detectionId);
      return projectReviewCase(reviewCase, actor);
    },

    async submitReview(detectionId, command) {
      const reviewCase = await loadRequiredCase(repository, detectionId);
      const expectedVersion = expectedVersionFor(reviewCase, command);

      if (expectedVersion !== reviewCase.version) {
        throw conflictFromLatest(expectedVersion, reviewCase, command.actor, command);
      }

      const nextCase = submitIndependentReview(reviewCase, command, now());
      const savedCase = await saveOnce(
        repository,
        nextCase,
        expectedVersion,
        REVIEW_SAVE_ERROR,
        command.actor,
        command,
      );
      return projectReviewCase(savedCase, command.actor);
    },

    async listAdjudication() {
      let reviewCases;

      try {
        reviewCases = await repository.listAdjudication();
      } catch (error) {
        throw new ReviewLoadError(ADJUDICATION_LOAD_ERROR, { cause: error });
      }

      return reviewCases.map(reviewCase => projectReviewCase(reviewCase, "adjudicator"));
    },

    async finalize(detectionId, command) {
      const reviewCase = await loadRequiredCase(repository, detectionId);
      const expectedVersion = expectedVersionFor(reviewCase, command);

      if (expectedVersion !== reviewCase.version) {
        throw conflictFromLatest(expectedVersion, reviewCase, "adjudicator", command);
      }

      const nextCase = finalizeAdjudication(reviewCase, command, now());
      const savedCase = await saveOnce(
        repository,
        nextCase,
        expectedVersion,
        ADJUDICATION_SAVE_ERROR,
        "adjudicator",
        command,
      );
      return projectReviewCase(savedCase, "adjudicator");
    },
  });
}
