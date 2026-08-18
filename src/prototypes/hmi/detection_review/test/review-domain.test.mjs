import test from "node:test";
import assert from "node:assert/strict";

import {
  ReviewValidationError,
  createReviewCase,
  decisionsMatch,
  finalizeAdjudication,
  recordReviewConflict,
  submitIndependentReview,
} from "../src/review-domain.mjs";

const T1 = "2026-08-16T01:00:00.000Z";
const T2 = "2026-08-16T02:00:00.000Z";
const T3 = "2026-08-16T03:00:00.000Z";

function firstReview(command = { decision: "confirmed" }) {
  return submitIndependentReview(
    createReviewCase("det-echo-001"),
    { actor: "reviewer-1", ...command },
    T1,
  );
}

function disagreementCase() {
  const afterFirst = firstReview();
  return submitIndependentReview(afterFirst, {
    actor: "reviewer-2",
    decision: "rejected",
    reason: "Call structure does not match the proposed species.",
  }, T2);
}

test("rejects a decision outside the four supported review outcomes", () => {
  assert.throws(
    () => firstReview({ decision: "maybe" }),
    error => error instanceof ReviewValidationError
      && error.field === "decision"
      && /supported decision/i.test(error.message),
  );
});

test("starts review cases at version 1 and advances the version for each transition", () => {
  const initial = createReviewCase("det-echo-001");
  const afterFirst = submitIndependentReview(initial, {
    actor: "reviewer-1",
    decision: "confirmed",
  }, T1);
  const afterSecond = submitIndependentReview(afterFirst, {
    actor: "reviewer-2",
    decision: "confirmed",
  }, T2);

  assert.equal(initial.version, 1);
  assert.equal(afterFirst.version, 2);
  assert.equal(afterSecond.version, 3);
});

test("requires a reason for rejection, species correction, and insufficient evidence", () => {
  for (const decision of ["rejected", "corrected_species", "insufficient_evidence"]) {
    assert.throws(
      () => firstReview({ decision, correctedSpecies: decision === "corrected_species" ? "Boobook" : undefined }),
      error => error instanceof ReviewValidationError
        && error.field === "reason",
      `${decision} should require a reason`,
    );
  }
});

test("requires and normalizes a corrected species for a correction decision", () => {
  assert.throws(
    () => firstReview({ decision: "corrected_species", reason: "A different call pattern is audible." }),
    error => error instanceof ReviewValidationError
      && error.field === "correctedSpecies",
  );

  const reviewCase = firstReview({
    decision: "corrected_species",
    reason: "  A different call pattern is audible.  ",
    correctedSpecies: "  Southern Boobook  ",
  });

  assert.equal(reviewCase.submissions["reviewer-1"].reason, "A different call pattern is audible.");
  assert.equal(reviewCase.submissions["reviewer-1"].correctedSpecies, "Southern Boobook");
});

test("reports review validation fields in form order", () => {
  assert.throws(
    () => firstReview({ decision: "corrected_species" }),
    error => error instanceof ReviewValidationError
      && error.field === "correctedSpecies",
  );
});

test("rejects wrong reviewers and repeated submissions for the current state", () => {
  const initial = createReviewCase("det-echo-001");
  assert.throws(
    () => submitIndependentReview(initial, {
      actor: "reviewer-2",
      decision: "confirmed",
    }, T1),
    error => error instanceof ReviewValidationError && error.field === "actor",
  );

  const afterFirst = submitIndependentReview(initial, {
    actor: "reviewer-1",
    decision: "confirmed",
  }, T1);
  assert.throws(
    () => submitIndependentReview(afterFirst, {
      actor: "reviewer-1",
      decision: "confirmed",
    }, T2),
    error => error instanceof ReviewValidationError && error.field === "actor",
  );
});

test("matches decisions by outcome and normalized corrected species, not reason text", () => {
  assert.equal(decisionsMatch(
    { decision: "confirmed", reason: "First note" },
    { decision: "confirmed", reason: "Second note" },
  ), true);
  assert.equal(decisionsMatch(
    { decision: "corrected_species", correctedSpecies: "Southern Boobook" },
    { decision: "corrected_species", correctedSpecies: "  southern   boobook " },
  ), true);
  assert.equal(decisionsMatch(
    { decision: "corrected_species", correctedSpecies: "Southern Boobook" },
    { decision: "corrected_species", correctedSpecies: "Powerful Owl" },
  ), false);
});

test("records consensus when the second independent decision matches", () => {
  const afterFirst = firstReview();
  const consensusCase = submitIndependentReview(afterFirst, {
    actor: "reviewer-2",
    decision: "confirmed",
  }, T2);

  assert.equal(consensusCase.status, "consensus");
  assert.deepEqual(consensusCase.consensus, {
    decision: "confirmed",
    correctedSpecies: null,
    reachedAt: T2,
  });
  assert.equal(consensusCase.history.at(-1).action, "consensus_reached");
});

test("routes mismatched decisions to adjudication", () => {
  const reviewCase = disagreementCase();

  assert.equal(reviewCase.status, "awaiting_adjudication");
  assert.equal(reviewCase.consensus, null);
  assert.equal(reviewCase.history.at(-1).action, "disagreement_routed");
});

test("appends immutable history entries for every status transition", () => {
  const initial = createReviewCase("det-echo-001");
  const afterFirst = submitIndependentReview(initial, {
    actor: "reviewer-1",
    decision: "confirmed",
  }, T1);
  const afterSecond = submitIndependentReview(afterFirst, {
    actor: "reviewer-2",
    decision: "rejected",
    reason: "The evidence contradicts the prediction.",
  }, T2);

  assert.equal(initial.history.length, 0);
  assert.deepEqual(afterSecond.history, [
    {
      actor: "reviewer-1",
      action: "first_review_submitted",
      timestamp: T1,
      previousStatus: "awaiting_first_review",
      resultingStatus: "awaiting_second_review",
    },
    {
      actor: "reviewer-2",
      action: "second_review_submitted",
      timestamp: T2,
      previousStatus: "awaiting_second_review",
      resultingStatus: "awaiting_second_review",
    },
    {
      actor: "reviewer-2",
      action: "disagreement_routed",
      timestamp: T2,
      previousStatus: "awaiting_second_review",
      resultingStatus: "awaiting_adjudication",
    },
  ]);
  assert.equal(Object.isFrozen(afterSecond.history), true);
  assert.equal(Object.isFrozen(afterSecond.history[0]), true);
});

test("records a versioned immutable conflict event without changing workflow status", () => {
  const initial = createReviewCase("det-echo-001");
  const conflicted = recordReviewConflict(
    initial,
    "fixture-concurrent-review",
    "2026-08-17T02:00:00.000Z",
  );

  assert.equal(conflicted.version, 2);
  assert.equal(conflicted.status, "awaiting_first_review");
  assert.deepEqual(conflicted.history.at(-1), {
    actor: "fixture-concurrent-review",
    action: "stale_write_conflict",
    timestamp: "2026-08-17T02:00:00.000Z",
    previousStatus: "awaiting_first_review",
    resultingStatus: "awaiting_first_review",
  });
  assert.equal(Object.isFrozen(conflicted.history.at(-1)), true);
});

test("adjudicator finalization requires a resolution reason and records the final result", () => {
  const reviewCase = disagreementCase();

  assert.throws(
    () => finalizeAdjudication(reviewCase, {
      actor: "adjudicator",
      resolutionReason: " ",
    }, T3),
    error => error instanceof ReviewValidationError
      && error.field === "decision",
  );

  assert.throws(
    () => finalizeAdjudication(reviewCase, {
      actor: "adjudicator",
      decision: "corrected_species",
      resolutionReason: " ",
    }, T3),
    error => error instanceof ReviewValidationError
      && error.field === "correctedSpecies",
  );

  for (const decision of ["confirmed", "rejected", "insufficient_evidence"]) {
    assert.throws(
      () => finalizeAdjudication(reviewCase, {
        actor: "adjudicator",
        decision,
        resolutionReason: " ",
      }, T3),
      error => error instanceof ReviewValidationError
        && error.field === "resolutionReason",
      `${decision} should report a missing adjudication explanation on resolutionReason`,
    );
  }

  const finalized = finalizeAdjudication(reviewCase, {
    actor: "adjudicator",
    decision: "corrected_species",
    correctedSpecies: "Southern Boobook",
    resolutionReason: "The call cadence supports the corrected species.",
  }, T3);

  assert.equal(finalized.status, "finalized");
  assert.deepEqual(finalized.adjudication, {
    actor: "adjudicator",
    decision: "corrected_species",
    correctedSpecies: "Southern Boobook",
    resolutionReason: "The call cadence supports the corrected species.",
    submittedAt: T3,
  });
  assert.deepEqual(finalized.history.at(-1), {
    actor: "adjudicator",
    action: "adjudication_finalized",
    timestamp: T3,
    previousStatus: "awaiting_adjudication",
    resultingStatus: "finalized",
  });

  assert.throws(
    () => finalizeAdjudication(finalized, {
      actor: "adjudicator",
      decision: "confirmed",
      resolutionReason: "Attempted duplicate finalization.",
    }, T3),
    error => error instanceof ReviewValidationError && error.field === "actor",
  );
});
