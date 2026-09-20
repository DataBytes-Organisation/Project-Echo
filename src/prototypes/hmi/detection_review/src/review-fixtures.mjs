import {
  createReviewCase,
  finalizeAdjudication,
  submitIndependentReview,
} from "./review-domain.mjs";

const FIRST_REVIEW_AT = "2026-08-16T01:00:00.000Z";
const SECOND_REVIEW_AT = "2026-08-16T02:00:00.000Z";
const FINALIZED_AT = "2026-08-16T03:00:00.000Z";

const DETECTION_IDS = Object.freeze([
  "det-echo-001",
  "det-echo-002",
  "det-echo-003",
]);

function afterFirstReview(detectionId) {
  return submitIndependentReview(createReviewCase(detectionId), {
    actor: "reviewer-1",
    decision: "confirmed",
  }, FIRST_REVIEW_AT);
}

function consensusCase(detectionId) {
  return submitIndependentReview(afterFirstReview(detectionId), {
    actor: "reviewer-2",
    decision: "confirmed",
  }, SECOND_REVIEW_AT);
}

function disagreementCase(detectionId) {
  return submitIndependentReview(afterFirstReview(detectionId), {
    actor: "reviewer-2",
    decision: "rejected",
    reason: "The call structure does not match the proposed species.",
  }, SECOND_REVIEW_AT);
}

function finalizedCase(detectionId) {
  return finalizeAdjudication(disagreementCase(detectionId), {
    actor: "adjudicator",
    decision: "rejected",
    resolutionReason: "The repeated call structure supports rejection after comparing both reviews.",
  }, FINALIZED_AT);
}

function casesFor(buildCase) {
  return Object.freeze(DETECTION_IDS.map(buildCase));
}

export const reviewScenarioFixtures = Object.freeze({
  "first-review": casesFor(createReviewCase),
  "second-review": casesFor(afterFirstReview),
  consensus: casesFor(consensusCase),
  adjudication: casesFor(disagreementCase),
  finalized: casesFor(finalizedCase),
  "draft-restored": casesFor(createReviewCase),
  conflict: casesFor(createReviewCase),
  success: casesFor(consensusCase),
});
