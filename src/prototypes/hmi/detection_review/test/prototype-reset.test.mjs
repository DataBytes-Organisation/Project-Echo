import test from "node:test";
import assert from "node:assert/strict";

import { createReviewCase, submitIndependentReview } from "../src/review-domain.mjs";
import { PersistentReviewWorkflowRepository } from "../src/persistent-review-repository.mjs";
import { createReviewDraftStore } from "../src/draft-store.mjs";
import { createPrototypePreferenceStore } from "../src/prototype-preferences.mjs";
import { resetPrototype } from "../src/prototype-reset.mjs";

function createMemoryStorage(initial = {}) {
  const entries = new Map(Object.entries(initial));
  return {
    entries,
    get length() {
      return entries.size;
    },
    key(index) {
      return [...entries.keys()][index] ?? null;
    },
    getItem(key) {
      return entries.has(key) ? entries.get(key) : null;
    },
    setItem(key, value) {
      entries.set(key, String(value));
    },
    removeItem(key) {
      entries.delete(key);
    },
  };
}

test("confirmed reset clears only prototype state and reseeds Reviewer 1", async () => {
  const storage = createMemoryStorage({ unrelated: "keep" });
  const reviewRepository = new PersistentReviewWorkflowRepository(storage, [
    createReviewCase("det-echo-001"),
  ]);
  const changedCase = submitIndependentReview(await reviewRepository.loadCase("det-echo-001"), {
    actor: "reviewer-1",
    decision: "confirmed",
  }, "2026-08-18T01:00:00.000Z");
  await reviewRepository.saveCase(changedCase, 1);
  const draftStore = createReviewDraftStore(storage);
  draftStore.save({
    detectionId: "det-echo-001",
    actor: "reviewer-1",
    kind: "review",
  }, {
    decision: "confirmed",
    correctedSpecies: "",
    reason: "",
    resolutionReason: "",
  }, 2);
  const preferenceStore = createPrototypePreferenceStore(storage);
  preferenceStore.saveRole("reviewer-2");

  const result = await resetPrototype({
    isPending: false,
    confirmReset: () => true,
    reviewRepository,
    draftStore,
    preferenceStore,
  });

  assert.deepEqual(result, {
    status: "reset",
    role: "reviewer-1",
    message: "Prototype data reset.",
  });
  assert.equal(Object.isFrozen(result), true);
  assert.equal(preferenceStore.loadRole(), "reviewer-1");
  assert.equal((await reviewRepository.loadCase("det-echo-001")).version, 1);
  assert.equal(draftStore.load({
    detectionId: "det-echo-001",
    actor: "reviewer-1",
    kind: "review",
  }), null);
  assert.equal(storage.getItem("unrelated"), "keep");
});

test("pending save blocks reset without asking for confirmation", async () => {
  let confirmationCount = 0;
  const result = await resetPrototype({
    isPending: true,
    confirmReset: () => {
      confirmationCount += 1;
      return true;
    },
  });

  assert.deepEqual(result, {
    status: "blocked",
    message: "Wait for the current save to finish.",
  });
  assert.equal(confirmationCount, 0);
});

test("cancelled confirmation leaves prototype state untouched", async () => {
  let operations = 0;
  const result = await resetPrototype({
    isPending: false,
    confirmReset: () => false,
    draftStore: { discardAll() { operations += 1; } },
    preferenceStore: { reset() { operations += 1; } },
    reviewRepository: { async reset() { operations += 1; } },
  });

  assert.deepEqual(result, { status: "cancelled", message: null });
  assert.equal(operations, 0);
});

test("confirmed reset calls every targeted cleanup exactly once", async () => {
  const calls = [];
  const result = await resetPrototype({
    isPending: false,
    confirmReset: () => true,
    draftStore: { discardAll() { calls.push("drafts"); } },
    preferenceStore: { reset() { calls.push("preference"); } },
    reviewRepository: { async reset() { calls.push("reviews"); } },
  });

  assert.equal(result.status, "reset");
  assert.deepEqual(calls, ["drafts", "preference", "reviews"]);
});

test("a failed cleanup returns controlled copy without retrying the operation", async () => {
  let discardCount = 0;
  const result = await resetPrototype({
    isPending: false,
    confirmReset: () => true,
    draftStore: {
      discardAll() {
        discardCount += 1;
        throw new Error("storage profile and key details");
      },
    },
    preferenceStore: { reset() {} },
    reviewRepository: { async reset() {} },
  });

  assert.deepEqual(result, {
    status: "failed",
    message: "Prototype data could not be reset.",
  });
  assert.equal(discardCount, 1);
});
