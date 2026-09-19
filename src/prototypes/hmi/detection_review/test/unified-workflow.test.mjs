import test from "node:test";
import assert from "node:assert/strict";

import { createReviewCase } from "../src/review-domain.mjs";
import { PersistentReviewWorkflowRepository } from "../src/persistent-review-repository.mjs";
import { createReviewWorkflow } from "../src/review-workflow.mjs";
import { createMemoryStorage } from "../test-support/helpers.mjs";

function workflowFor(storage, now) {
  return createReviewWorkflow(
    new PersistentReviewWorkflowRepository(storage, [createReviewCase("det-echo-001")]),
    { now: () => now },
  );
}

test("matching independent reviews persist consensus across repository refreshes", async () => {
  const storage = createMemoryStorage();
  const reviewerOne = workflowFor(storage, "2026-08-17T01:00:00.000Z");

  const firstSession = await reviewerOne.submitReview("det-echo-001", {
    actor: "reviewer-1",
    decision: "confirmed",
    expectedVersion: 1,
  });
  assert.equal(firstSession.status, "awaiting_second_review");
  assert.equal(firstSession.version, 2);

  const reviewerTwo = workflowFor(storage, "2026-08-17T02:00:00.000Z");
  const blindSession = await reviewerTwo.getSession("det-echo-001", "reviewer-2");
  assert.equal(blindSession.firstReviewComplete, true);
  assert.deepEqual(blindSession.submissions, {});
  assert.deepEqual(blindSession.history, []);

  const consensus = await reviewerTwo.submitReview("det-echo-001", {
    actor: "reviewer-2",
    decision: "confirmed",
    expectedVersion: 2,
  });
  assert.equal(consensus.status, "consensus");
  assert.equal(consensus.version, 3);

  const restored = workflowFor(storage, "2026-08-17T03:00:00.000Z");
  const persisted = await restored.getSession("det-echo-001", "reviewer-1");
  assert.equal(persisted.status, "consensus");
  assert.equal(persisted.version, 3);
  assert.deepEqual(persisted.history.map(entry => entry.action), [
    "first_review_submitted",
    "second_review_submitted",
    "consensus_reached",
  ]);
});

test("disagreement routes to adjudication and finalization persists across repository refreshes", async () => {
  const storage = createMemoryStorage();
  const reviewerOne = workflowFor(storage, "2026-08-17T01:00:00.000Z");
  await reviewerOne.submitReview("det-echo-001", {
    actor: "reviewer-1",
    decision: "confirmed",
    expectedVersion: 1,
  });

  const reviewerTwo = workflowFor(storage, "2026-08-17T02:00:00.000Z");
  const blindSession = await reviewerTwo.getSession("det-echo-001", "reviewer-2");
  assert.deepEqual(blindSession.submissions, {});
  assert.deepEqual(blindSession.history, []);
  const disagreement = await reviewerTwo.submitReview("det-echo-001", {
    actor: "reviewer-2",
    decision: "rejected",
    reason: "The evidence contradicts the predicted species.",
    expectedVersion: 2,
  });
  assert.equal(disagreement.status, "awaiting_adjudication");
  assert.equal(disagreement.version, 3);

  const adjudicator = workflowFor(storage, "2026-08-17T03:00:00.000Z");
  const queue = await adjudicator.listAdjudication();
  assert.deepEqual(queue.map(item => item.detectionId), ["det-echo-001"]);
  assert.deepEqual(queue[0].history.map(entry => entry.action), [
    "first_review_submitted",
    "second_review_submitted",
    "disagreement_routed",
  ]);

  const finalized = await adjudicator.finalize("det-echo-001", {
    actor: "adjudicator",
    decision: "rejected",
    resolutionReason: "The two independent reviews justify rejection.",
    expectedVersion: 3,
  });
  assert.equal(finalized.status, "finalized");
  assert.equal(finalized.version, 4);

  const restored = workflowFor(storage, "2026-08-17T04:00:00.000Z");
  const persisted = await restored.getSession("det-echo-001", "adjudicator");
  assert.equal(persisted.status, "finalized");
  assert.equal(persisted.version, 4);
  assert.deepEqual(persisted.history.map(entry => entry.action), [
    "first_review_submitted",
    "second_review_submitted",
    "disagreement_routed",
    "adjudication_finalized",
  ]);
});
