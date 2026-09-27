import test from "node:test";
import assert from "node:assert/strict";

import {
  createInitialReviewState,
  createReviewWorkbench,
} from "../src/workbench-state.mjs";
import {
  DetectionRepositoryError,
  INVALID_FIXTURE_MESSAGE,
} from "../src/repository.mjs";
import { createDeferred, makeDetection } from "../test-support/helpers.mjs";

test("starts in the loading queue state", () => {
  assert.deepEqual(createInitialReviewState(), {
    status: "loading",
    records: [],
    selectedId: null,
    errorMessage: null,
  });
});

test("exposes loading while the repository request is pending", async () => {
  const pending = createDeferred();
  const workbench = createReviewWorkbench({ list: () => pending.promise });

  const load = workbench.load();

  assert.equal(workbench.getState().status, "loading");
  pending.resolve([]);
  await load;
});

test("selects the first detection in a populated queue", async () => {
  const records = [
    makeDetection(),
    makeDetection({ id: "det-echo-002", species: "Koala" }),
  ];
  const workbench = createReviewWorkbench({ list: async () => records });

  const state = await workbench.load();

  assert.equal(state.status, "populated");
  assert.equal(state.records.length, 2);
  assert.equal(state.selectedId, "det-echo-001");
});

test("uses an explicit empty state when no records are available", async () => {
  const workbench = createReviewWorkbench({ list: async () => [] });

  const state = await workbench.load();

  assert.equal(state.status, "empty");
  assert.deepEqual(state.records, []);
  assert.equal(state.selectedId, null);
});

test("uses a controlled failed state when repository loading fails", async () => {
  const rawMessage = "confidence must be below 100 at fixture[0]";
  const error = new DetectionRepositoryError(INVALID_FIXTURE_MESSAGE, {
    cause: new Error(rawMessage),
  });
  const workbench = createReviewWorkbench({
    list: async () => { throw error; },
  });

  const state = await workbench.load();

  assert.equal(state.status, "failed");
  assert.equal(state.errorMessage, INVALID_FIXTURE_MESSAGE);
  assert.equal(state.errorMessage.includes(rawMessage), false);
});

test("changes the selected record without changing queue data", async () => {
  const records = [
    makeDetection(),
    makeDetection({ id: "det-echo-002", species: "Koala" }),
  ];
  const workbench = createReviewWorkbench({ list: async () => records });
  await workbench.load();

  const state = workbench.select("det-echo-002");

  assert.equal(state.selectedId, "det-echo-002");
  assert.equal(state.records, records);
});
