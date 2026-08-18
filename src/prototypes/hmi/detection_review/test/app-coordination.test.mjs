import test from "node:test";
import assert from "node:assert/strict";

import {
  createAsyncViewGuard,
  getSubmissionVersion,
  lockSubmissionForm,
  persistDraftAtBoundary,
} from "../src/app-coordination.mjs";

test("restored drafts submit against their stored base version until explicitly rebased", () => {
  const session = { detectionId: "det-echo-001", actor: "reviewer-1", version: 4 };
  const restored = {
    status: "restored",
    draft: { detectionId: "det-echo-001", actor: "reviewer-1", version: 2 },
  };
  const rebased = {
    status: "restored",
    draft: { detectionId: "det-echo-001", actor: "reviewer-1", version: 4 },
  };

  assert.equal(getSubmissionVersion(session, restored), 2);
  assert.equal(getSubmissionVersion(session, rebased), 4);
  assert.equal(getSubmissionVersion(session, { status: "idle", draft: null }), 4);
});

test("a restored draft from another actor or detection cannot supply the version token", () => {
  const session = { detectionId: "det-echo-001", actor: "reviewer-1", version: 4 };

  assert.equal(getSubmissionVersion(session, {
    status: "restored",
    draft: { detectionId: "det-echo-002", actor: "reviewer-1", version: 1 },
  }), 4);
  assert.equal(getSubmissionVersion(session, {
    status: "restored",
    draft: { detectionId: "det-echo-001", actor: "reviewer-2", version: 1 },
  }), 4);
});

test("draft persistence returns a controlled discriminated failure", () => {
  const result = persistDraftAtBoundary({
    save() {
      throw new Error("raw browser profile and quota details");
    },
  }, {}, {}, 1);

  assert.deepEqual(result, {
    status: "failed",
    draft: null,
    errorMessage: "Saved review drafts could not be accessed in this browser.",
  });
  assert.doesNotMatch(result.errorMessage, /profile|quota/i);
});

test("draft persistence distinguishes saved drafts from empty-form removal", () => {
  const savedDraft = { version: 1, values: { decision: "confirmed" } };
  assert.deepEqual(persistDraftAtBoundary({
    save() {
      return savedDraft;
    },
  }, {}, {}, 1), {
    status: "saved",
    draft: savedDraft,
    errorMessage: null,
  });
  assert.deepEqual(persistDraftAtBoundary({
    save() {
      return null;
    },
  }, {}, {}, 1), {
    status: "discarded",
    draft: null,
    errorMessage: null,
  });
});

test("async view guard rejects older loads and submit completions after navigation", () => {
  let selectedId = "det-echo-001";
  const guard = createAsyncViewGuard(() => selectedId);
  const firstLoad = guard.begin("det-echo-001");
  const firstSubmit = guard.capture("det-echo-001");

  selectedId = "det-echo-002";
  const secondLoad = guard.begin("det-echo-002");

  assert.equal(guard.isCurrent(firstLoad), false);
  assert.equal(guard.isCurrent(firstSubmit), false);
  assert.equal(guard.isCurrent(secondLoad), true);
});

test("submission lock prevents edits while a stateful write is pending", () => {
  const attributes = new Map();
  const controls = [{ disabled: false }, { disabled: false }, { disabled: false }];
  const form = {
    elements: controls,
    setAttribute(name, value) {
      attributes.set(name, value);
    },
  };

  lockSubmissionForm(form);

  assert.deepEqual(controls.map(control => control.disabled), [true, true, true]);
  assert.equal(attributes.get("aria-busy"), "true");
});
