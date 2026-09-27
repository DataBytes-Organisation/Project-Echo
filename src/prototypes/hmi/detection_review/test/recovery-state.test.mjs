import test from "node:test";
import assert from "node:assert/strict";

import {
  createRecoveryState,
  createConflictState,
  discardConflictDraft,
  discardRecoveredDraft,
  keepConflictDraft,
  offerRecoveredDraft,
  restoreRecoveredDraft,
} from "../src/recovery-state.mjs";

const draft = Object.freeze({
  detectionId: "det-echo-001",
  actor: "reviewer-1",
  kind: "review",
  version: 1,
  savedAt: "2026-08-17T01:00:00.000Z",
  values: Object.freeze({
    decision: "corrected_species",
    correctedSpecies: "Southern Boobook",
    reason: "The cadence supports a different species.",
    resolutionReason: "",
  }),
});

test("conflict recovery rebases preserved input to the latest version without submitting", () => {
  const conflict = createConflictState({
    expectedVersion: 1,
    latestSession: { detectionId: "det-echo-001", actor: "reviewer-1", version: 2 },
    attemptedValues: draft.values,
  });

  const rebased = keepConflictDraft(conflict);

  assert.equal(rebased.status, "rebased");
  assert.equal(rebased.expectedVersion, 2);
  assert.deepEqual(rebased.values, draft.values);
  assert.equal(Object.isFrozen(rebased.values), true);
  assert.equal(rebased.shouldSubmit, false);
});

test("discarding a conflict draft keeps the latest session and removes entered values", () => {
  const latestSession = { detectionId: "det-echo-001", actor: "reviewer-1", version: 2 };
  const conflict = createConflictState({
    expectedVersion: 1,
    latestSession,
    attemptedValues: draft.values,
  });

  const discarded = discardConflictDraft(conflict);

  assert.equal(discarded.status, "discarded");
  assert.equal(discarded.latestSession, latestSession);
  assert.equal(discarded.values, null);
  assert.equal(discarded.shouldSubmit, false);
});

test("a recovered draft is offered without populating the form until Restore", () => {
  const offered = offerRecoveredDraft(createRecoveryState(), draft);

  assert.equal(offered.status, "available");
  assert.equal(offered.values, null);

  const restored = restoreRecoveredDraft(offered);
  assert.equal(restored.status, "restored");
  assert.deepEqual(restored.values, draft.values);
  assert.equal(Object.isFrozen(restored.values), true);
});

test("Discard clears recovered values and records the explicit choice", () => {
  const offered = offerRecoveredDraft(createRecoveryState(), draft);
  const discarded = discardRecoveredDraft(offered);

  assert.deepEqual(discarded, {
    status: "discarded",
    draft: null,
    values: null,
    errorMessage: null,
  });
});

test("a controlled storage failure can be represented without a draft", () => {
  const state = createRecoveryState("Draft recovery is unavailable in this browser.");

  assert.equal(state.status, "failed");
  assert.equal(state.draft, null);
  assert.match(state.errorMessage, /unavailable/);
});
