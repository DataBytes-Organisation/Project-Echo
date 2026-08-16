export const DECISIONS = Object.freeze([
  "confirmed",
  "rejected",
  "corrected_species",
  "insufficient_evidence",
]);

const DECISION_SET = new Set(DECISIONS);
const REASON_REQUIRED = new Set([
  "rejected",
  "corrected_species",
  "insufficient_evidence",
]);

export class ReviewValidationError extends Error {
  constructor(message, field) {
    super(message);
    this.name = "ReviewValidationError";
    this.field = field;
  }
}

function requiredText(value, field, message) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new ReviewValidationError(message, field);
  }

  return value.trim().replace(/\s+/g, " ");
}

function optionalText(value) {
  return typeof value === "string" && value.trim() !== ""
    ? value.trim().replace(/\s+/g, " ")
    : null;
}

function validateDecision(value) {
  if (!DECISION_SET.has(value)) {
    throw new ReviewValidationError("Choose a supported decision.", "decision");
  }

  return value;
}

function normalizeReviewCommand(command) {
  const decision = validateDecision(command?.decision);
  const correctedSpecies = decision === "corrected_species"
    ? requiredText(
      command?.correctedSpecies,
      "correctedSpecies",
      "Enter the corrected species.",
    )
    : null;
  const reason = REASON_REQUIRED.has(decision)
    ? requiredText(command?.reason, "reason", "Give a reason for this decision.")
    : optionalText(command?.reason);

  return Object.freeze({ decision, reason, correctedSpecies });
}

function freezeReviewCase(reviewCase) {
  const submissions = Object.fromEntries(
    Object.entries(reviewCase.submissions).map(([actor, submission]) => [
      actor,
      Object.freeze({ ...submission }),
    ]),
  );
  const history = reviewCase.history.map(entry => Object.freeze({ ...entry }));

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

function appendHistory(reviewCase, entry) {
  return [...reviewCase.history, entry];
}

function historyEntry(actor, action, timestamp, previousStatus, resultingStatus) {
  return {
    actor,
    action,
    timestamp: requiredText(timestamp, "timestamp", "A transition timestamp is required."),
    previousStatus,
    resultingStatus,
  };
}

function normalizeSpeciesForMatch(value) {
  return typeof value === "string"
    ? value.trim().replace(/\s+/g, " ").toLocaleLowerCase("en")
    : "";
}

export function createReviewCase(detectionId) {
  return freezeReviewCase({
    detectionId: requiredText(
      detectionId,
      "detectionId",
      "A detection ID is required.",
    ),
    status: "awaiting_first_review",
    submissions: {},
    consensus: null,
    adjudication: null,
    history: [],
  });
}

export function decisionsMatch(first, second) {
  if (first?.decision !== second?.decision || !DECISION_SET.has(first?.decision)) {
    return false;
  }

  return first.decision !== "corrected_species"
    || normalizeSpeciesForMatch(first.correctedSpecies)
      === normalizeSpeciesForMatch(second.correctedSpecies);
}

export function submitIndependentReview(reviewCase, command, timestamp) {
  const actor = command?.actor;
  const isFirstReview = reviewCase.status === "awaiting_first_review"
    && actor === "reviewer-1";
  const isSecondReview = reviewCase.status === "awaiting_second_review"
    && actor === "reviewer-2";

  if (!isFirstReview && !isSecondReview) {
    throw new ReviewValidationError(
      "This reviewer cannot submit a decision in the current state.",
      "actor",
    );
  }

  const normalized = normalizeReviewCommand(command);
  const submittedAt = requiredText(
    timestamp,
    "timestamp",
    "A review timestamp is required.",
  );
  const submission = Object.freeze({
    actor,
    ...normalized,
    submittedAt,
  });
  const submissions = {
    ...reviewCase.submissions,
    [actor]: submission,
  };

  if (isFirstReview) {
    const resultingStatus = "awaiting_second_review";
    return freezeReviewCase({
      ...reviewCase,
      status: resultingStatus,
      submissions,
      history: appendHistory(reviewCase, historyEntry(
        actor,
        "first_review_submitted",
        submittedAt,
        reviewCase.status,
        resultingStatus,
      )),
    });
  }

  const matched = decisionsMatch(submissions["reviewer-1"], submission);
  const resultingStatus = matched ? "consensus" : "awaiting_adjudication";
  const action = matched ? "consensus_reached" : "disagreement_routed";

  return freezeReviewCase({
    ...reviewCase,
    status: resultingStatus,
    submissions,
    consensus: matched
      ? {
        decision: submission.decision,
        correctedSpecies: submission.correctedSpecies,
        reachedAt: submittedAt,
      }
      : null,
    history: appendHistory(reviewCase, historyEntry(
      actor,
      action,
      submittedAt,
      reviewCase.status,
      resultingStatus,
    )),
  });
}

export function finalizeAdjudication(reviewCase, command, timestamp) {
  if (reviewCase.status !== "awaiting_adjudication"
    || command?.actor !== "adjudicator") {
    throw new ReviewValidationError(
      "This case cannot be finalized in the current state.",
      "actor",
    );
  }

  const decision = validateDecision(command?.decision);
  const correctedSpecies = decision === "corrected_species"
    ? requiredText(
      command?.correctedSpecies,
      "correctedSpecies",
      "Enter the corrected species.",
    )
    : null;
  const resolutionReason = requiredText(
    command?.resolutionReason,
    "resolutionReason",
    "Give a reason for the adjudication result.",
  );
  const submittedAt = requiredText(
    timestamp,
    "timestamp",
    "An adjudication timestamp is required.",
  );
  const resultingStatus = "finalized";

  return freezeReviewCase({
    ...reviewCase,
    status: resultingStatus,
    adjudication: {
      actor: "adjudicator",
      decision,
      correctedSpecies,
      resolutionReason,
      submittedAt,
    },
    history: appendHistory(reviewCase, historyEntry(
      "adjudicator",
      "adjudication_finalized",
      submittedAt,
      reviewCase.status,
      resultingStatus,
    )),
  });
}
