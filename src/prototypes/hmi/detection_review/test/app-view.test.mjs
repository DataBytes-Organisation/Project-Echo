import test from "node:test";
import assert from "node:assert/strict";

import { renderApplicationView } from "../src/app-view.mjs";

const detectionState = Object.freeze({
  status: "empty",
  records: [],
  selectedId: null,
  errorMessage: null,
});

const staleAdjudicationSession = Object.freeze({
  detectionId: "det-stale",
  actor: "adjudicator",
  status: "awaiting_adjudication",
  firstReviewComplete: true,
  submissions: Object.freeze({}),
  consensus: null,
  adjudication: null,
  history: Object.freeze([]),
});

test("adjudication load failure hides stale or false-empty queue data", () => {
  const html = renderApplicationView({
    detectionState,
    workflowActor: "adjudicator",
    workflowSession: staleAdjudicationSession,
    workflowLoadError: "The adjudication queue could not be loaded. Try again manually.",
    adjudicationSessions: [staleAdjudicationSession],
  });

  assert.match(html, /Review state could not be loaded/);
  assert.match(html, /data-workflow-retry/);
  assert.doesNotMatch(html, /Disagreement queue|0 cases|det-stale/);
});
