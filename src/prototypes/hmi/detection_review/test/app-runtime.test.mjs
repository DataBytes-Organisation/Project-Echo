import test from "node:test";
import assert from "node:assert/strict";

import {
  beginRecognizedWorkflowSubmission,
  createAppRuntimeController,
} from "../src/app-runtime.mjs";
import { createReviewDraftStore } from "../src/draft-store.mjs";
import { PersistentReviewWorkflowRepository } from "../src/persistent-review-repository.mjs";
import { createPrototypePreferenceStore } from "../src/prototype-preferences.mjs";
import { createReviewCase, submitIndependentReview } from "../src/review-domain.mjs";

function createMemoryStorage(initial = {}) {
  const entries = new Map(Object.entries(initial));
  return {
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

function createRuntime({
  storage = createMemoryStorage(),
  initialRole = "reviewer-1",
  getDetectionId = () => "det-echo-001",
} = {}) {
  return createAppRuntimeController({
    initialRole,
    getDetectionId,
    preferenceStore: createPrototypePreferenceStore(storage),
  });
}

test("pending submission blocks queue and role navigation without invalidating its view", () => {
  const storage = createMemoryStorage();
  let detectionId = "det-echo-001";
  let navigationCount = 0;
  const runtime = createRuntime({ storage, getDetectionId: () => detectionId });
  const submission = runtime.beginSubmission();

  const navigation = runtime.beginNavigation(() => {
    navigationCount += 1;
    detectionId = "det-echo-002";
  });
  const roleSwitch = runtime.switchRole("reviewer-2");

  assert.deepEqual(navigation, { status: "blocked", token: null });
  assert.deepEqual(roleSwitch, { status: "blocked", role: "reviewer-1" });
  assert.equal(navigationCount, 0);
  assert.equal(runtime.activeRole, "reviewer-1");
  assert.equal(runtime.isCurrent(submission.token), true);
  assert.equal(createPrototypePreferenceStore(storage).loadRole(), "reviewer-1");
});

test("recognized workflow submits prevent browser default behavior while already pending", () => {
  const runtime = createRuntime();
  runtime.beginSubmission();
  let preventDefaultCount = 0;

  const result = beginRecognizedWorkflowSubmission({
    preventDefault() {
      preventDefaultCount += 1;
    },
  }, runtime, { canSubmit: true });

  assert.equal(preventDefaultCount, 1);
  assert.deepEqual(result, { status: "blocked", token: null });
});

test("role switching persists the role and invalidates the previous projection", () => {
  const storage = createMemoryStorage();
  const runtime = createRuntime({ storage });
  const reviewerOneView = runtime.beginView();

  const result = runtime.switchRole("reviewer-2");

  assert.deepEqual(result, { status: "switched", role: "reviewer-2" });
  assert.equal(runtime.activeRole, "reviewer-2");
  assert.equal(runtime.isCurrent(reviewerOneView), false);
  assert.equal(createPrototypePreferenceStore(storage).loadRole(), "reviewer-2");
});

test("successful stale completion still removes the exact submitted draft", () => {
  const storage = createMemoryStorage();
  const draftStore = createReviewDraftStore(storage, {
    now: () => "2026-08-19T01:00:00.000Z",
  });
  const context = {
    detectionId: "det-echo-001",
    actor: "reviewer-1",
    kind: "review",
  };
  const submittedDraft = draftStore.save(context, { decision: "confirmed" }, 1);
  const runtime = createRuntime({ storage });
  const submission = runtime.beginSubmission();
  runtime.beginView();

  const completion = runtime.completeSuccessfulSubmission(
    submission.token,
    () => draftStore.discardIfUnchanged(context, submittedDraft),
  );

  assert.deepEqual(completion, { status: "stale", cleanupResult: true });
  assert.equal(runtime.isPending, false);
  assert.equal(draftStore.load(context), null);
});

test("exact cleanup preserves a newer draft after a stale successful completion", () => {
  const storage = createMemoryStorage();
  let savedAt = "2026-08-19T01:00:00.000Z";
  const draftStore = createReviewDraftStore(storage, { now: () => savedAt });
  const context = {
    detectionId: "det-echo-001",
    actor: "reviewer-1",
    kind: "review",
  };
  const submittedDraft = draftStore.save(context, { decision: "confirmed" }, 1);
  const runtime = createRuntime({ storage });
  const submission = runtime.beginSubmission();
  savedAt = "2026-08-19T02:00:00.000Z";
  const newerDraft = draftStore.save(context, {
    decision: "rejected",
    reason: "New evidence entered after the save began.",
  }, 1);
  runtime.beginView();

  const completion = runtime.completeSuccessfulSubmission(
    submission.token,
    () => draftStore.discardIfUnchanged(context, submittedDraft),
  );

  assert.deepEqual(completion, { status: "stale", cleanupResult: false });
  assert.deepEqual(draftStore.load(context), newerDraft);
});

test("confirmed runtime reset targets prototype state and restores Reviewer 1", async () => {
  const storage = createMemoryStorage({ unrelated: "keep" });
  const reviewRepository = new PersistentReviewWorkflowRepository(storage, [
    createReviewCase("det-echo-001"),
  ]);
  const firstReview = submitIndependentReview(
    await reviewRepository.loadCase("det-echo-001"),
    { actor: "reviewer-1", decision: "confirmed" },
    "2026-08-19T01:00:00.000Z",
  );
  await reviewRepository.saveCase(firstReview, 1);
  const draftStore = createReviewDraftStore(storage);
  draftStore.save({
    detectionId: "det-echo-001",
    actor: "reviewer-2",
    kind: "review",
  }, { decision: "confirmed" }, 2);
  const preferenceStore = createPrototypePreferenceStore(storage);
  preferenceStore.saveRole("reviewer-2");
  const runtime = createAppRuntimeController({
    initialRole: "reviewer-2",
    getDetectionId: () => "det-echo-001",
    preferenceStore,
  });

  const result = await runtime.reset({
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
  assert.equal(runtime.activeRole, "reviewer-1");
  assert.equal((await reviewRepository.loadCase("det-echo-001")).version, 1);
  assert.equal(draftStore.load({
    detectionId: "det-echo-001",
    actor: "reviewer-2",
    kind: "review",
  }), null);
  assert.equal(storage.getItem("unrelated"), "keep");
});
