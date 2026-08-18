import test from "node:test";
import assert from "node:assert/strict";

import { reviewScenarioFixtures } from "../src/review-fixtures.mjs";

test("deterministic scenario fixtures contain the expected states and ordered histories", () => {
  const expectations = {
    "first-review": ["awaiting_first_review", 0],
    "second-review": ["awaiting_second_review", 1],
    consensus: ["consensus", 3],
    adjudication: ["awaiting_adjudication", 3],
    finalized: ["finalized", 4],
    "draft-restored": ["awaiting_first_review", 0],
    conflict: ["awaiting_first_review", 0],
    success: ["consensus", 3],
  };

  for (const [scenario, [status, historyLength]] of Object.entries(expectations)) {
    const cases = reviewScenarioFixtures[scenario];
    assert.deepEqual(cases.map(reviewCase => reviewCase.detectionId), [
      "det-echo-001",
      "det-echo-002",
      "det-echo-003",
    ]);
    assert.equal(cases.every(reviewCase => reviewCase.status === status), true);
    assert.equal(cases.every(reviewCase => reviewCase.history.length === historyLength), true);
    assert.equal(cases.every(reviewCase => Number.isInteger(reviewCase.version)), true);
    assert.equal(cases.every(reviewCase => reviewCase.history.every((entry, index, history) => (
      index === 0 || Date.parse(history[index - 1].timestamp) <= Date.parse(entry.timestamp)
    ))), true);
  }
});
