import test from "node:test";
import assert from "node:assert/strict";

import { getQueueNavigationTarget } from "../src/keyboard.mjs";

const records = Object.freeze([
  { id: "det-echo-001" },
  { id: "det-echo-002" },
  { id: "det-echo-003" },
]);

test("ArrowDown and ArrowUp move through the queue and wrap at boundaries", () => {
  assert.equal(getQueueNavigationTarget(records, "det-echo-001", "ArrowDown"), "det-echo-002");
  assert.equal(getQueueNavigationTarget(records, "det-echo-003", "ArrowDown"), "det-echo-001");
  assert.equal(getQueueNavigationTarget(records, "det-echo-003", "ArrowUp"), "det-echo-002");
  assert.equal(getQueueNavigationTarget(records, "det-echo-001", "ArrowUp"), "det-echo-003");
});

test("ArrowRight and ArrowLeft follow the horizontal queue and wrap at boundaries", () => {
  assert.equal(getQueueNavigationTarget(records, "det-echo-001", "ArrowRight"), "det-echo-002");
  assert.equal(getQueueNavigationTarget(records, "det-echo-003", "ArrowRight"), "det-echo-001");
  assert.equal(getQueueNavigationTarget(records, "det-echo-003", "ArrowLeft"), "det-echo-002");
  assert.equal(getQueueNavigationTarget(records, "det-echo-001", "ArrowLeft"), "det-echo-003");
});

test("Home and End select queue boundaries", () => {
  assert.equal(getQueueNavigationTarget(records, "det-echo-002", "Home"), "det-echo-001");
  assert.equal(getQueueNavigationTarget(records, "det-echo-002", "End"), "det-echo-003");
});

test("unhandled keys and an empty queue do not request a selection", () => {
  assert.equal(getQueueNavigationTarget(records, "det-echo-001", "Enter"), null);
  assert.equal(getQueueNavigationTarget([], null, "ArrowDown"), null);
});
