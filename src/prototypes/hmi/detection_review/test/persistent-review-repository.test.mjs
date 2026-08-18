import test from "node:test";
import assert from "node:assert/strict";

import {
  createReviewCase,
  submitIndependentReview,
} from "../src/review-domain.mjs";
import {
  PersistentReviewStorageError,
  PersistentReviewWorkflowRepository,
  REVIEW_SCHEMA_VERSION,
  REVIEW_STORAGE_ERROR_MESSAGE,
  REVIEW_STORAGE_KEY,
  createBrowserPersistentReviewRepository,
} from "../src/persistent-review-repository.mjs";
import { StaleReviewVersionError } from "../src/review-repository.mjs";
import { createReviewWorkflow } from "../src/review-workflow.mjs";
import { createMemoryStorage } from "../test-support/helpers.mjs";

function storageError(operation) {
  return error => error instanceof PersistentReviewStorageError
    && error.userMessage === REVIEW_STORAGE_ERROR_MESSAGE
    && error.cause?.message === operation;
}

function afterFirstReview(detectionId = "det-echo-001") {
  return submitIndependentReview(createReviewCase(detectionId), {
    actor: "reviewer-1",
    decision: "confirmed",
  }, "2026-08-17T01:00:00.000Z");
}

function awaitingAdjudication(detectionId = "det-echo-001") {
  return submitIndependentReview(afterFirstReview(detectionId), {
    actor: "reviewer-2",
    decision: "rejected",
    reason: "The evidence contradicts the prediction.",
  }, "2026-08-17T02:00:00.000Z");
}

test("seeds once and restores submitted review state from a new repository", async () => {
  const storage = createMemoryStorage();
  const seeds = [createReviewCase("det-echo-001")];
  const firstRepository = new PersistentReviewWorkflowRepository(storage, seeds);
  const workflow = createReviewWorkflow(firstRepository, {
    now: () => "2026-08-17T01:00:00.000Z",
  });

  await workflow.submitReview("det-echo-001", {
    actor: "reviewer-1",
    decision: "confirmed",
    expectedVersion: 1,
  });

  const restoredRepository = new PersistentReviewWorkflowRepository(storage, seeds);
  const restored = await restoredRepository.loadCase("det-echo-001");

  assert.equal(restored.status, "awaiting_second_review");
  assert.equal(restored.version, 2);
  assert.equal(restored.history[0].action, "first_review_submitted");
});

test("first run writes the versioned payload and a valid payload is not reseeded", async () => {
  const storage = createMemoryStorage();
  const initialSeed = createReviewCase("det-echo-001");
  new PersistentReviewWorkflowRepository(storage, [initialSeed]);

  assert.deepEqual(JSON.parse(storage.getItem(REVIEW_STORAGE_KEY)), {
    schemaVersion: REVIEW_SCHEMA_VERSION,
    cases: [initialSeed],
  });

  const savedPayload = storage.getItem(REVIEW_STORAGE_KEY);
  new PersistentReviewWorkflowRepository(storage, [createReviewCase("det-echo-099")]);

  assert.equal(storage.getItem(REVIEW_STORAGE_KEY), savedPayload);
});

test("save checks the latest payload across repositories and returns an immutable latest case", async () => {
  const storage = createMemoryStorage();
  const seeds = [createReviewCase("det-echo-001")];
  const firstRepository = new PersistentReviewWorkflowRepository(storage, seeds);
  const secondRepository = new PersistentReviewWorkflowRepository(storage, seeds);
  const firstWrite = afterFirstReview();

  await firstRepository.saveCase(firstWrite, 1);

  await assert.rejects(
    () => secondRepository.saveCase(firstWrite, 1),
    error => error instanceof StaleReviewVersionError
      && error.expectedVersion === 1
      && error.actualVersion === 2
      && error.latestCase.version === 2
      && Object.isFrozen(error.latestCase)
      && Object.isFrozen(error.latestCase.history),
  );
});

test("each successful save writes the complete payload exactly once", async () => {
  const storage = createMemoryStorage();
  const originalSetItem = storage.setItem;
  let writeCount = 0;
  storage.setItem = (key, value) => {
    writeCount += 1;
    originalSetItem.call(storage, key, value);
  };
  const repository = new PersistentReviewWorkflowRepository(storage, [
    createReviewCase("det-echo-001"),
  ]);
  const writesBeforeSave = writeCount;

  await repository.saveCase(afterFirstReview(), 1);

  assert.equal(writeCount - writesBeforeSave, 1);
});

test("persistent repository records its debug conflict before returning the stale error", async () => {
  const storage = createMemoryStorage();
  const repository = new PersistentReviewWorkflowRepository(storage, [
    createReviewCase("det-echo-001"),
  ], {
    conflictOnNextSave: true,
    now: () => "2026-08-17T02:00:00.000Z",
  });

  await assert.rejects(
    () => repository.saveCase(afterFirstReview(), 1),
    error => error instanceof StaleReviewVersionError
      && error.actualVersion === 2
      && error.latestCase.history.at(-1).action === "stale_write_conflict",
  );
  const latest = await repository.loadCase("det-echo-001");
  assert.equal(latest.version, 2);
  assert.equal(latest.history.at(-1).action, "stale_write_conflict");
});

test("lists restored cases awaiting adjudication", async () => {
  const repository = new PersistentReviewWorkflowRepository(createMemoryStorage(), [
    createReviewCase("det-echo-001"),
    awaitingAdjudication("det-echo-002"),
  ]);

  const queued = await repository.listAdjudication();

  assert.deepEqual(queued.map(reviewCase => reviewCase.detectionId), ["det-echo-002"]);
  assert.equal(Object.isFrozen(queued[0]), true);
});

test("reset deterministically restores original seeds after saved work", async () => {
  const seed = createReviewCase("det-echo-001");
  const repository = new PersistentReviewWorkflowRepository(createMemoryStorage(), [seed]);
  await repository.saveCase(afterFirstReview(), 1);

  await repository.reset();

  assert.deepEqual(await repository.loadCase("det-echo-001"), seed);
});

test("rejects an unsupported persisted schema without replacing it", () => {
  const rawPayload = JSON.stringify({ schemaVersion: 2, cases: [] });
  const storage = createMemoryStorage({ [REVIEW_STORAGE_KEY]: rawPayload });

  assert.throws(
    () => new PersistentReviewWorkflowRepository(storage, [createReviewCase("det-echo-001")]),
    storageError("Saved review payload schema is unsupported."),
  );
  assert.equal(storage.getItem(REVIEW_STORAGE_KEY), rawPayload);
});

test("rejects corrupt JSON, invalid cases, and duplicate detection IDs", () => {
  for (const payload of [
    "{not json",
    JSON.stringify({ schemaVersion: 1, cases: [{ detectionId: "det-echo-001" }] }),
    JSON.stringify({
      schemaVersion: 1,
      cases: [createReviewCase("det-echo-001"), createReviewCase("det-echo-001")],
    }),
  ]) {
    const storage = createMemoryStorage({ [REVIEW_STORAGE_KEY]: payload });
    assert.throws(
      () => new PersistentReviewWorkflowRepository(storage, [createReviewCase("det-echo-099")]),
      error => error instanceof PersistentReviewStorageError
        && error.userMessage === REVIEW_STORAGE_ERROR_MESSAGE,
    );
  }
});

test("wraps failing Web Storage reads and writes with controlled storage errors", async () => {
  const readStorage = createMemoryStorage();
  readStorage.getItem = () => {
    throw new Error("getItem failed");
  };
  assert.throws(
    () => new PersistentReviewWorkflowRepository(readStorage, [createReviewCase("det-echo-001")]),
    storageError("getItem failed"),
  );

  const writeStorage = createMemoryStorage();
  writeStorage.setItem = () => {
    throw new Error("setItem failed");
  };
  assert.throws(
    () => new PersistentReviewWorkflowRepository(writeStorage, [createReviewCase("det-echo-001")]),
    storageError("setItem failed"),
  );
});

test("browser factory wraps a throwing localStorage getter", () => {
  const windowObject = {};
  Object.defineProperty(windowObject, "localStorage", {
    get() {
      throw new Error("storage access failed");
    },
  });

  assert.throws(
    () => createBrowserPersistentReviewRepository(windowObject, [createReviewCase("det-echo-001")]),
    storageError("storage access failed"),
  );
});
