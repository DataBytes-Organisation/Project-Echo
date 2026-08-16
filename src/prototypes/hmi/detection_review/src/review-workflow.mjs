import {
  finalizeAdjudication,
  submitIndependentReview,
} from "./review-domain.mjs";

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

async function saveOnce(repository, nextCase, failureMessage) {
  try {
    await repository.saveCase(nextCase);
  } catch (error) {
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
      const nextCase = submitIndependentReview(reviewCase, command, now());
      await saveOnce(repository, nextCase, REVIEW_SAVE_ERROR);
      return projectReviewCase(nextCase, command.actor);
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
      const nextCase = finalizeAdjudication(reviewCase, command, now());
      await saveOnce(repository, nextCase, ADJUDICATION_SAVE_ERROR);
      return projectReviewCase(nextCase, "adjudicator");
    },
  });
}
