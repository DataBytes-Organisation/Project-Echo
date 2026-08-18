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

test("renders controls before the workbench and keeps completed cases visible when a role has no actionable work", () => {
  const html = renderApplicationView({
    detectionState: {
      status: "populated",
      records: [{
        id: "det-finalized",
        timestamp: "2026-05-22T04:12:00.000Z",
        sensorId: "ECHO-07",
        species: "Powerful Owl",
        confidence: 94.6,
        animalEstLLA: [-37.8136, 144.9631, 24],
        animalLLAUncertainty: 18,
        audioAvailable: true,
        reviewStatus: "finalized",
        isActionable: false,
      }],
      selectedId: "det-finalized",
      errorMessage: null,
    },
    workflowActor: "reviewer-1",
    activeRole: "reviewer-1",
    isPending: false,
    resetStatus: null,
    workflowSession: null,
    workflowLoadError: null,
  });

  assert.ok(html.indexOf("prototype-controls") < html.indexOf("workbench"));
  assert.match(html, /No cases need Reviewer 1 right now/);
  assert.match(html, /Finalized/);
});
