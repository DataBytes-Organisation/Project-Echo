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
const REVIEW_STATUSES = new Set([
  "awaiting_first_review",
  "awaiting_second_review",
  "consensus",
  "awaiting_adjudication",
  "finalized",
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

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function storedTimestamp(value) {
  const timestamp = requiredText(
    value,
    "timestamp",
    "A stored transition timestamp is required.",
  );
  if (!Number.isFinite(Date.parse(timestamp))) {
    throw new Error("A stored transition timestamp must be valid.");
  }
  return timestamp;
}

function normalizeStoredSubmission(actorKey, candidate) {
  if (!isRecord(candidate)) {
    throw new Error("A stored submission must be an object.");
  }

  const actor = requiredText(
    candidate.actor,
    "actor",
    "A stored submission actor is required.",
  );
  if (actor !== actorKey) {
    throw new Error("A stored submission actor does not match its key.");
  }
  if (candidate.reason !== undefined
    && candidate.reason !== null
    && typeof candidate.reason !== "string") {
    throw new Error("A stored submission reason must be text.");
  }
  if (candidate.correctedSpecies !== undefined
    && candidate.correctedSpecies !== null
    && typeof candidate.correctedSpecies !== "string") {
    throw new Error("A stored corrected species must be text.");
  }

  const normalized = normalizeReviewCommand(candidate);
  return {
    actor,
    ...normalized,
    submittedAt: storedTimestamp(candidate.submittedAt),
  };
}

function normalizeStoredConsensus(candidate) {
  if (!isRecord(candidate)) {
    throw new Error("Stored consensus data must be an object.");
  }
  const decision = validateDecision(candidate.decision);
  if (candidate.correctedSpecies !== undefined
    && candidate.correctedSpecies !== null
    && typeof candidate.correctedSpecies !== "string") {
    throw new Error("A stored consensus species must be text.");
  }
  const correctedSpecies = decision === "corrected_species"
    ? requiredText(
      candidate.correctedSpecies,
      "correctedSpecies",
      "A stored consensus species is required.",
    )
    : null;

  return {
    decision,
    correctedSpecies,
    reachedAt: storedTimestamp(candidate.reachedAt),
  };
}

function normalizeStoredAdjudication(candidate) {
  if (!isRecord(candidate)) {
    throw new Error("Stored adjudication data must be an object.");
  }
  const actor = requiredText(
    candidate.actor,
    "actor",
    "A stored adjudication actor is required.",
  );
  if (actor !== "adjudicator") {
    throw new Error("A stored adjudication actor is invalid.");
  }
  const resolutionReason = requiredText(
    candidate.resolutionReason,
    "resolutionReason",
    "A stored adjudication reason is required.",
  );
  if (candidate.correctedSpecies !== undefined
    && candidate.correctedSpecies !== null
    && typeof candidate.correctedSpecies !== "string") {
    throw new Error("A stored adjudication species must be text.");
  }
  const normalized = normalizeReviewCommand({
    decision: candidate.decision,
    correctedSpecies: candidate.correctedSpecies,
    reason: resolutionReason,
  });

  return {
    actor,
    decision: normalized.decision,
    correctedSpecies: normalized.correctedSpecies,
    resolutionReason,
    submittedAt: storedTimestamp(candidate.submittedAt),
  };
}

function expectedStoredHistory(status) {
  const first = {
    actor: "reviewer-1",
    action: "first_review_submitted",
    previousStatus: "awaiting_first_review",
    resultingStatus: "awaiting_second_review",
  };
  const second = {
    actor: "reviewer-2",
    action: "second_review_submitted",
    previousStatus: "awaiting_second_review",
    resultingStatus: "awaiting_second_review",
  };
  const normal = {
    consensus: [first, second, {
      actor: "reviewer-2",
      action: "consensus_reached",
      previousStatus: "awaiting_second_review",
      resultingStatus: "consensus",
    }],
    awaiting_adjudication: [first, second, {
      actor: "reviewer-2",
      action: "disagreement_routed",
      previousStatus: "awaiting_second_review",
      resultingStatus: "awaiting_adjudication",
    }],
    finalized: [first, second, {
      actor: "reviewer-2",
      action: "disagreement_routed",
      previousStatus: "awaiting_second_review",
      resultingStatus: "awaiting_adjudication",
    }, {
      actor: "adjudicator",
      action: "adjudication_finalized",
      previousStatus: "awaiting_adjudication",
      resultingStatus: "finalized",
    }],
  };

  return normal[status] ?? (status === "awaiting_second_review" ? [first] : []);
}

function normalizeStoredHistory(candidate, status) {
  if (!Array.isArray(candidate)) {
    throw new Error("Stored history must be an array.");
  }
  const expected = expectedStoredHistory(status);
  const normalized = [];
  let currentStatus = "awaiting_first_review";
  let expectedIndex = 0;
  let previousTimestamp = null;
  let staleCount = 0;

  for (const entry of candidate) {
    if (!isRecord(entry)) {
      throw new Error("A stored history entry must be an object.");
    }
    const actor = requiredText(
      entry.actor,
      "actor",
      "A stored history actor is required.",
    );
    const action = requiredText(
      entry.action,
      "action",
      "A stored history action is required.",
    );
    const timestamp = storedTimestamp(entry.timestamp);
    if (previousTimestamp !== null && Date.parse(timestamp) < Date.parse(previousTimestamp)) {
      throw new Error("Stored history must be ordered by timestamp.");
    }
    previousTimestamp = timestamp;

    if (action === "stale_write_conflict") {
      if (entry.previousStatus !== currentStatus
        || entry.resultingStatus !== currentStatus) {
        throw new Error("A stored conflict history entry has invalid statuses.");
      }
      staleCount += 1;
      normalized.push(historyEntry(
        actor,
        action,
        timestamp,
        currentStatus,
        currentStatus,
      ));
      continue;
    }

    const expectedEntry = expected[expectedIndex];
    if (!expectedEntry
      || actor !== expectedEntry.actor
      || action !== expectedEntry.action
      || entry.previousStatus !== expectedEntry.previousStatus
      || entry.resultingStatus !== expectedEntry.resultingStatus
      || entry.previousStatus !== currentStatus) {
      throw new Error("A stored history transition is invalid.");
    }
    normalized.push(historyEntry(
      actor,
      action,
      timestamp,
      expectedEntry.previousStatus,
      expectedEntry.resultingStatus,
    ));
    currentStatus = expectedEntry.resultingStatus;
    expectedIndex += 1;
  }

  if (expectedIndex !== expected.length || currentStatus !== status) {
    throw new Error("Stored history does not describe the case status.");
  }

  return { history: normalized, staleCount, normalCount: expected.length };
}

function validateStoredSubmissions(candidate, status) {
  if (!isRecord(candidate)) {
    throw new Error("Stored submissions must be an object.");
  }
  const expectedActors = status === "awaiting_first_review"
    ? []
    : status === "awaiting_second_review"
      ? ["reviewer-1"]
      : ["reviewer-1", "reviewer-2"];
  const actualActors = Object.keys(candidate);
  if (actualActors.some(actor => !["reviewer-1", "reviewer-2"].includes(actor))
    || actualActors.length !== expectedActors.length
    || expectedActors.some(actor => !Object.hasOwn(candidate, actor))) {
    throw new Error("Stored submissions do not match the case status.");
  }
  return Object.fromEntries(expectedActors.map(actor => [
    actor,
    normalizeStoredSubmission(actor, candidate[actor]),
  ]));
}

export function restoreReviewCase(candidate) {
  try {
    if (!isRecord(candidate)) {
      throw new Error("A stored review case must be an object.");
    }
    const detectionId = requiredText(
      candidate.detectionId,
      "detectionId",
      "A stored detection ID is required.",
    );
    const version = candidate.version;
    if (!Number.isInteger(version) || version <= 0) {
      throw new Error("A stored review version must be positive.");
    }
    const status = candidate.status;
    if (!REVIEW_STATUSES.has(status)) {
      throw new Error("A stored review status is invalid.");
    }

    const submissions = validateStoredSubmissions(candidate.submissions, status);
    const consensus = candidate.consensus === null
      ? null
      : normalizeStoredConsensus(candidate.consensus);
    const adjudication = candidate.adjudication === null
      ? null
      : normalizeStoredAdjudication(candidate.adjudication);
    const restoredHistory = normalizeStoredHistory(candidate.history, status);

    const expectedConsensus = status === "consensus";
    const expectedAdjudication = status === "finalized";
    if ((status === "awaiting_first_review" || status === "awaiting_second_review")
      && (consensus !== null || adjudication !== null)) {
      throw new Error("An in-progress stored case cannot contain a result.");
    }
    if (expectedConsensus !== (consensus !== null)
      || expectedAdjudication !== (adjudication !== null)
      || (status !== "finalized" && adjudication !== null)
      || (status !== "consensus" && consensus !== null)) {
      throw new Error("Stored result data does not match the case status.");
    }

    if (status === "consensus"
      && (!decisionsMatch(submissions["reviewer-1"], submissions["reviewer-2"])
        || consensus.decision !== submissions["reviewer-2"].decision
        || consensus.correctedSpecies !== submissions["reviewer-2"].correctedSpecies)) {
      throw new Error("Stored consensus does not match the submissions.");
    }
    if (status === "finalized" && adjudication.actor !== "adjudicator") {
      throw new Error("Stored adjudication actor is invalid.");
    }

    const normalMinimum = (status === "awaiting_first_review"
      ? 1
      : status === "awaiting_second_review"
        ? 2
        : status === "finalized" || status === "consensus" || status === "awaiting_adjudication"
          ? 3
          : 1) + restoredHistory.staleCount;
    if (version < normalMinimum) {
      throw new Error("A stored review version is behind its history.");
    }

    const normalHistory = restoredHistory.history.filter(
      entry => entry.action !== "stale_write_conflict",
    );
    const firstHistory = normalHistory.find(entry => entry.action === "first_review_submitted");
    if (firstHistory && firstHistory.timestamp !== submissions["reviewer-1"]?.submittedAt) {
      throw new Error("Stored first-review history does not match its submission.");
    }
    const secondHistory = normalHistory.find(entry => entry.action === "second_review_submitted");
    if (secondHistory && secondHistory.timestamp !== submissions["reviewer-2"]?.submittedAt) {
      throw new Error("Stored second-review history does not match its submission.");
    }
    const resultHistory = normalHistory.find(entry => [
      "consensus_reached",
      "disagreement_routed",
    ].includes(entry.action));
    if (resultHistory && resultHistory.timestamp !== submissions["reviewer-2"]?.submittedAt) {
      throw new Error("Stored result history does not match its submission.");
    }
    const adjudicationHistory = normalHistory.find(
      entry => entry.action === "adjudication_finalized",
    );
    if (adjudicationHistory && adjudicationHistory.timestamp !== adjudication?.submittedAt) {
      throw new Error("Stored adjudication history does not match its result.");
    }
    if (consensus && consensus.reachedAt !== submissions["reviewer-2"]?.submittedAt) {
      throw new Error("Stored consensus timestamp does not match its submission.");
    }

    return freezeReviewCase({
      detectionId,
      version,
      status,
      submissions,
      consensus,
      adjudication,
      history: restoredHistory.history,
    });
  } catch (_error) {
    throw new ReviewValidationError("Saved review data is invalid.", "storedCase");
  }
}

export function createReviewCase(detectionId) {
  return freezeReviewCase({
    detectionId: requiredText(
      detectionId,
      "detectionId",
      "A detection ID is required.",
    ),
    version: 1,
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
      version: reviewCase.version + 1,
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
    version: reviewCase.version + 1,
    status: resultingStatus,
    submissions,
    consensus: matched
      ? {
        decision: submission.decision,
        correctedSpecies: submission.correctedSpecies,
        reachedAt: submittedAt,
      }
      : null,
    history: [
      ...reviewCase.history,
      historyEntry(
        actor,
        "second_review_submitted",
        submittedAt,
        reviewCase.status,
        reviewCase.status,
      ),
      historyEntry(
        actor,
        action,
        submittedAt,
        reviewCase.status,
        resultingStatus,
      ),
    ],
  });
}

export function recordReviewConflict(reviewCase, actor, timestamp) {
  const recordedAt = requiredText(
    timestamp,
    "timestamp",
    "A conflict timestamp is required.",
  );

  return freezeReviewCase({
    ...reviewCase,
    version: reviewCase.version + 1,
    history: appendHistory(reviewCase, historyEntry(
      requiredText(actor, "actor", "A conflict actor is required."),
      "stale_write_conflict",
      recordedAt,
      reviewCase.status,
      reviewCase.status,
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
    version: reviewCase.version + 1,
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
