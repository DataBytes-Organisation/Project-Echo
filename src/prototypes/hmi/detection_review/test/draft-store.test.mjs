import test from "node:test";
import assert from "node:assert/strict";

import {
  DraftStorageError,
  createBrowserReviewDraftStore,
  createReviewDraftStore,
} from "../src/draft-store.mjs";
import { createMemoryStorage } from "../test-support/helpers.mjs";

const context = Object.freeze({
  detectionId: "det-echo-001",
  actor: "reviewer-1",
  kind: "review",
});

test("blocked access to the localStorage property becomes a controlled error", () => {
  const blockedWindow = {};
  Object.defineProperty(blockedWindow, "localStorage", {
    get() {
      throw new Error("SecurityError with browser profile details");
    },
  });

  assert.throws(
    () => createBrowserReviewDraftStore(blockedWindow),
    error => error instanceof DraftStorageError
      && /could not be accessed/i.test(error.userMessage)
      && !error.userMessage.includes("profile"),
  );
});

const values = Object.freeze({
  decision: "rejected",
  correctedSpecies: "",
  reason: "The call pattern does not support the prediction.",
  resolutionReason: "",
});

test("saves a versioned draft and restores it from a new store after navigation", () => {
  const storage = createMemoryStorage();
  const firstStore = createReviewDraftStore(storage, {
    now: () => "2026-08-17T01:00:00.000Z",
  });

  const saved = firstStore.save(context, values, 3);
  const restored = createReviewDraftStore(storage).load(context);

  assert.deepEqual(saved, {
    ...context,
    values,
    version: 3,
    savedAt: "2026-08-17T01:00:00.000Z",
  });
  assert.deepEqual(restored, saved);
  assert.equal(Object.isFrozen(restored), true);
  assert.equal(Object.isFrozen(restored.values), true);
});

test("discard removes only the matching actor and workflow draft", () => {
  const storage = createMemoryStorage();
  const store = createReviewDraftStore(storage);
  store.save(context, values, 1);
  store.save({ ...context, actor: "reviewer-2" }, values, 2);

  assert.equal(store.discard(context), true);
  assert.equal(store.load(context), null);
  assert.notEqual(store.load({ ...context, actor: "reviewer-2" }), null);
});

test("discardAll removes only detection-review drafts", () => {
  const storage = createMemoryStorage({ unrelated: "keep" });
  const store = createReviewDraftStore(storage);
  store.save(context, values, 1);
  store.save({ ...context, actor: "reviewer-2" }, values, 1);

  assert.equal(store.discardAll(), 2);
  assert.equal(storage.getItem("unrelated"), "keep");
});

test("an older submission completion cannot discard a newer saved draft", () => {
  let now = "2026-08-17T01:00:00.000Z";
  const storage = createMemoryStorage();
  const store = createReviewDraftStore(storage, { now: () => now });
  const submittedDraft = store.save(context, values, 1);

  now = "2026-08-17T01:01:00.000Z";
  const newerDraft = store.save(context, {
    ...values,
    reason: "This edit was made after the earlier submission started.",
  }, 1);

  assert.equal(store.discardIfUnchanged(context, submittedDraft), false);
  assert.deepEqual(store.load(context), newerDraft);
  assert.equal(store.discardIfUnchanged(context, newerDraft), true);
  assert.equal(store.load(context), null);
});

test("an empty form clears its matching draft instead of preserving stale input", () => {
  const storage = createMemoryStorage();
  const store = createReviewDraftStore(storage);
  store.save(context, values, 1);

  const result = store.save(context, {
    decision: "",
    correctedSpecies: " ",
    reason: "",
    resolutionReason: "",
  }, 1);

  assert.equal(result, null);
  assert.equal(store.load(context), null);
});

test("invalid stored data is removed and never presented as a recovered draft", () => {
  const storage = createMemoryStorage({
    "echo-review-draft:review:reviewer-1:det-echo-001": "{not-json",
  });
  const store = createReviewDraftStore(storage);

  assert.equal(store.load(context), null);
  assert.equal(storage.entries.size, 0);
});

test("storage failures expose controlled copy without raw browser details", () => {
  const store = createReviewDraftStore({
    getItem() {
      throw new Error("Quota internals and browser profile path");
    },
    setItem() {},
    removeItem() {},
  });

  assert.throws(
    () => store.load(context),
    error => error instanceof DraftStorageError
      && /could not be accessed/i.test(error.userMessage)
      && !error.userMessage.includes("Quota internals"),
  );
});
