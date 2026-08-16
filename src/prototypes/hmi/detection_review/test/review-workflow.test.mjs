import test from "node:test";
import assert from "node:assert/strict";

import {
  createReviewCase,
  submitIndependentReview,
} from "../src/review-domain.mjs";
import {
  ReviewLoadError,
  ReviewSubmissionError,
  createReviewWorkflow,
} from "../src/review-workflow.mjs";
import { createRepositoryDouble } from "../test-support/helpers.mjs";

const T1 = "2026-08-16T01:00:00.000Z";
const T2 = "2026-08-16T02:00:00.000Z";
const T3 = "2026-08-16T03:00:00.000Z";

function afterFirstReview(command = { decision: "confirmed" }) {
  return submitIndependentReview(createReviewCase("det-echo-001"), {
    actor: "reviewer-1",
    ...command,
  }, T1);
}

function awaitingAdjudication() {
  return submitIndependentReview(afterFirstReview(), {
    actor: "reviewer-2",
    decision: "rejected",
    reason: "The evidence contradicts the prediction.",
  }, T2);
}

test("second reviewer session is blind to reviewer 1 decision content before submission", async () => {
  const repository = createRepositoryDouble(afterFirstReview({
    decision: "corrected_species",
    reason: "The call has a shorter repeated cadence.",
    correctedSpecies: "Southern Boobook",
  }));
  const workflow = createReviewWorkflow(repository, { now: () => T2 });

  const session = await workflow.getSession("det-echo-001", "reviewer-2");
  const serialized = JSON.stringify(session);

  assert.equal(session.firstReviewComplete, true);
  assert.deepEqual(session.submissions, {});
  assert.doesNotMatch(serialized, /corrected_species|Southern Boobook|shorter repeated cadence/);
});

test("matching second review saves once and returns a consensus projection", async () => {
  const repository = createRepositoryDouble(afterFirstReview());
  const workflow = createReviewWorkflow(repository, { now: () => T2 });

  const session = await workflow.submitReview("det-echo-001", {
    actor: "reviewer-2",
    decision: "confirmed",
  });

  assert.equal(repository.saveCalls.length, 1);
  assert.equal(session.status, "consensus");
  assert.equal(session.consensus.decision, "confirmed");
  assert.deepEqual(Object.keys(session.submissions), ["reviewer-1", "reviewer-2"]);
});

test("mismatched second review is returned by the adjudication queue", async () => {
  const repository = createRepositoryDouble(afterFirstReview());
  const workflow = createReviewWorkflow(repository, { now: () => T2 });

  await workflow.submitReview("det-echo-001", {
    actor: "reviewer-2",
    decision: "insufficient_evidence",
    reason: "The audio evidence is unavailable.",
  });
  const queue = await workflow.listAdjudication();

  assert.equal(queue.length, 1);
  assert.equal(queue[0].status, "awaiting_adjudication");
  assert.equal(queue[0].submissions["reviewer-1"].decision, "confirmed");
  assert.equal(queue[0].submissions["reviewer-2"].decision, "insufficient_evidence");
});

test("adjudicator finalization saves once and exposes the final resolution", async () => {
  const repository = createRepositoryDouble(awaitingAdjudication());
  const workflow = createReviewWorkflow(repository, { now: () => T3 });

  const session = await workflow.finalize("det-echo-001", {
    actor: "adjudicator",
    decision: "rejected",
    resolutionReason: "Two independent evidence concerns support rejection.",
  });

  assert.equal(repository.saveCalls.length, 1);
  assert.equal(session.status, "finalized");
  assert.equal(session.adjudication.decision, "rejected");
  assert.equal(session.adjudication.resolutionReason, "Two independent evidence concerns support rejection.");
  assert.equal(session.history.at(-1).action, "adjudication_finalized");
});

test("stateful review submission does not retry a failed repository write", async () => {
  const repository = createRepositoryDouble(afterFirstReview(), {
    saveError: new Error("fixture write failed"),
  });
  const workflow = createReviewWorkflow(repository, { now: () => T2 });

  await assert.rejects(
    () => workflow.submitReview("det-echo-001", {
      actor: "reviewer-2",
      decision: "confirmed",
    }),
    error => error instanceof ReviewSubmissionError
      && error.userMessage === "The review could not be saved. Your submission was not retried.",
  );
  assert.equal(repository.saveCalls.length, 1);
  assert.equal((await workflow.getSession("det-echo-001", "reviewer-2")).status, "awaiting_second_review");
});

test("stateful adjudication finalization does not retry a failed repository write", async () => {
  const repository = createRepositoryDouble(awaitingAdjudication(), {
    saveError: new Error("fixture write failed"),
  });
  const workflow = createReviewWorkflow(repository, { now: () => T3 });

  await assert.rejects(
    () => workflow.finalize("det-echo-001", {
      actor: "adjudicator",
      decision: "confirmed",
      resolutionReason: "The prediction is supported after evidence comparison.",
    }),
    error => error instanceof ReviewSubmissionError,
  );
  assert.equal(repository.saveCalls.length, 1);
  assert.equal((await workflow.getSession("det-echo-001", "adjudicator")).status, "awaiting_adjudication");
});

test("converts repository load failures to a controlled workflow error", async () => {
  const workflow = createReviewWorkflow({
    async loadCase() {
      throw new Error("raw adapter detail");
    },
  });

  await assert.rejects(
    () => workflow.getSession("det-echo-001", "reviewer-1"),
    error => error instanceof ReviewLoadError
      && error.userMessage === "The review workflow could not be loaded. Try again manually."
      && !error.userMessage.includes("raw adapter detail"),
  );
});

test("converts adjudication-list failures to a controlled workflow error", async () => {
  const workflow = createReviewWorkflow({
    async listAdjudication() {
      throw new Error("raw adapter detail");
    },
  });

  await assert.rejects(
    () => workflow.listAdjudication(),
    error => error instanceof ReviewLoadError
      && error.userMessage === "The adjudication queue could not be loaded. Try again manually.",
  );
});
