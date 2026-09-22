import test from "node:test";
import assert from "node:assert/strict";

import { makeDetection } from "../test-support/helpers.mjs";
import {
  ROLES,
  firstActionableDetectionId,
  isActionableForRole,
  orderRecordsForRole,
} from "../src/role-workspace.mjs";

function record(id) {
  return makeDetection({ id });
}

test("orders actionable cases first for each role without hiding completed cases", () => {
  const records = [record("det-1"), record("det-2"), record("det-3")];
  const sessions = new Map([
    ["det-1", { detectionId: "det-1", status: "consensus" }],
    ["det-2", { detectionId: "det-2", status: "awaiting_second_review" }],
    ["det-3", { detectionId: "det-3", status: "awaiting_first_review" }],
  ]);

  const reviewerTwo = orderRecordsForRole(records, sessions, "reviewer-2");

  assert.deepEqual(reviewerTwo.map(item => item.id), ["det-2", "det-1", "det-3"]);
  assert.equal(reviewerTwo[0].isActionable, true);
  assert.equal(reviewerTwo[0].reviewStatus, "awaiting_second_review");
  assert.equal(Object.isFrozen(reviewerTwo[0]), true);
  assert.notEqual(reviewerTwo[0], records[1]);
});

test("matches each role to only its actionable review status", () => {
  assert.deepEqual(ROLES, ["reviewer-1", "reviewer-2", "adjudicator"]);
  assert.equal(isActionableForRole("awaiting_first_review", "reviewer-1"), true);
  assert.equal(isActionableForRole("awaiting_second_review", "reviewer-2"), true);
  assert.equal(isActionableForRole("awaiting_adjudication", "adjudicator"), true);

  for (const role of ROLES) {
    for (const status of [
      "awaiting_first_review",
      "awaiting_second_review",
      "consensus",
      "awaiting_adjudication",
      "finalized",
    ]) {
      const expected = (role === "reviewer-1" && status === "awaiting_first_review")
        || (role === "reviewer-2" && status === "awaiting_second_review")
        || (role === "adjudicator" && status === "awaiting_adjudication");
      assert.equal(isActionableForRole(status, role), expected);
    }
  }
});

test("rejects invalid roles and retains input order inside each partition", () => {
  const records = [record("det-1"), record("det-2"), record("det-3"), record("det-4")];
  const sessions = new Map([
    ["det-1", { detectionId: "det-1", status: "consensus" }],
    ["det-2", { detectionId: "det-2", status: "awaiting_second_review" }],
    ["det-3", { detectionId: "det-3", status: "awaiting_second_review" }],
    ["det-4", { detectionId: "det-4", status: "finalized" }],
  ]);

  const ordered = orderRecordsForRole(records, sessions, "reviewer-2");

  assert.deepEqual(ordered.map(item => item.id), ["det-2", "det-3", "det-1", "det-4"]);
  assert.throws(() => isActionableForRole("awaiting_first_review", "observer"), TypeError);
  assert.throws(() => orderRecordsForRole(records, sessions, "observer"), TypeError);
  assert.throws(() => firstActionableDetectionId(records, sessions, "observer"), TypeError);
});

test("returns the first actionable detection ID or null without changing inputs", () => {
  const records = [record("det-1"), record("det-2")];
  const sessions = new Map([
    ["det-1", { detectionId: "det-1", status: "consensus" }],
    ["det-2", { detectionId: "det-2", status: "awaiting_second_review" }],
  ]);

  assert.equal(firstActionableDetectionId(records, sessions, "reviewer-2"), "det-2");
  assert.equal(firstActionableDetectionId(records, sessions, "adjudicator"), null);
  assert.equal(records[1].reviewStatus, undefined);
  assert.equal(sessions.get("det-2").isActionable, undefined);
});
