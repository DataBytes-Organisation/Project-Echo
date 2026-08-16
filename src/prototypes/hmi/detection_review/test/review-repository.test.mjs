import test from "node:test";
import assert from "node:assert/strict";

import {
  createReviewCase,
  submitIndependentReview,
} from "../src/review-domain.mjs";
import {
  FixtureReviewWorkflowRepository,
  ReviewWorkflowRepository,
} from "../src/review-repository.mjs";

test("review workflow repository declares load, save, and adjudication list operations", async () => {
  const repository = new ReviewWorkflowRepository();

  await assert.rejects(() => repository.loadCase("det-echo-001"), /must be implemented/);
  await assert.rejects(() => repository.saveCase(createReviewCase("det-echo-001")), /must be implemented/);
  await assert.rejects(() => repository.listAdjudication(), /must be implemented/);
});

test("fixture repository loads and saves deterministic immutable review cases", async () => {
  const initial = createReviewCase("det-echo-001");
  const repository = new FixtureReviewWorkflowRepository([initial]);
  const afterFirst = submitIndependentReview(initial, {
    actor: "reviewer-1",
    decision: "confirmed",
  }, "2026-08-16T01:00:00.000Z");

  assert.notEqual(await repository.loadCase("det-echo-001"), initial);
  await repository.saveCase(afterFirst);

  assert.notEqual(await repository.loadCase("det-echo-001"), afterFirst);
  assert.equal(Object.isFrozen((await repository.loadCase("det-echo-001")).submissions), true);
});

test("fixture repository isolates stored cases from caller-owned mutation", async () => {
  const callerOwned = structuredClone(createReviewCase("det-echo-001"));
  const repository = new FixtureReviewWorkflowRepository([callerOwned]);
  callerOwned.status = "tampered";

  const loaded = await repository.loadCase("det-echo-001");
  assert.equal(loaded.status, "awaiting_first_review");
  assert.equal(Object.isFrozen(loaded), true);
  assert.throws(() => {
    loaded.status = "tampered";
  }, TypeError);
});

test("fixture repository isolates saved cases from later caller mutation", async () => {
  const repository = new FixtureReviewWorkflowRepository([
    createReviewCase("det-echo-001"),
  ]);
  const callerOwned = structuredClone(submitIndependentReview(
    createReviewCase("det-echo-001"),
    { actor: "reviewer-1", decision: "confirmed" },
    "2026-08-16T01:00:00.000Z",
  ));

  await repository.saveCase(callerOwned);
  callerOwned.status = "tampered";

  assert.equal((await repository.loadCase("det-echo-001")).status, "awaiting_second_review");
});

test("fixture repository lists only cases awaiting adjudication", async () => {
  const untouched = createReviewCase("det-echo-001");
  const first = submitIndependentReview(createReviewCase("det-echo-002"), {
    actor: "reviewer-1",
    decision: "confirmed",
  }, "2026-08-16T01:00:00.000Z");
  const disagreement = submitIndependentReview(first, {
    actor: "reviewer-2",
    decision: "rejected",
    reason: "The evidence contradicts the prediction.",
  }, "2026-08-16T02:00:00.000Z");
  const repository = new FixtureReviewWorkflowRepository([untouched, disagreement]);

  const queued = await repository.listAdjudication();

  assert.deepEqual(queued.map(reviewCase => reviewCase.detectionId), ["det-echo-002"]);
});
