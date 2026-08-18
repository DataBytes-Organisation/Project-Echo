import test from "node:test";
import assert from "node:assert/strict";

import { renderApplicationView } from "../src/app-view.mjs";

const record = Object.freeze({
  id: "det-echo-001",
  timestamp: "2026-05-22T04:12:00.000Z",
  sensorId: "ECHO-07",
  species: "Powerful Owl",
  confidence: 94.6,
  microphoneLLA: Object.freeze([-37.8142, 144.9628, 18]),
  animalEstLLA: Object.freeze([-37.8136, 144.9631, 24]),
  animalLLAUncertainty: 18,
  audioAvailable: true,
  sampleRate: 48000,
  queueStatus: "pending_review",
  version: 1,
});

function detectionState(status) {
  return {
    status,
    records: status === "populated" ? [record] : [],
    selectedId: status === "populated" ? record.id : null,
    errorMessage: status === "failed" ? "Controlled queue failure." : null,
  };
}

function session(overrides = {}) {
  return {
    detectionId: record.id,
    actor: "reviewer-1",
    status: "awaiting_first_review",
    version: 1,
    firstReviewComplete: false,
    submissions: {},
    consensus: null,
    adjudication: null,
    history: [],
    ...overrides,
  };
}

test("application markup names every queue page state", () => {
  for (const status of ["loading", "empty", "failed", "populated"]) {
    const html = renderApplicationView({
      detectionState: detectionState(status),
      workflowActor: null,
      workflowSession: null,
      workflowLoadError: null,
    });
    assert.match(html, new RegExp(`data-page-state="${status}"`));
    assert.match(html, /class="prototype-controls"/);
  }
});

test("draft-restored state populates the form only after explicit recovery", () => {
  const html = renderApplicationView({
    detectionState: detectionState("populated"),
    workflowActor: "reviewer-1",
    workflowSession: session(),
    workflowLoadError: null,
    draftRecovery: {
      status: "restored",
      draft: { version: 1, savedAt: "2026-08-17T01:00:00.000Z" },
      values: {
        decision: "corrected_species",
        correctedSpecies: "Southern Boobook",
        reason: "The cadence supports a different species.",
        resolutionReason: "",
      },
      errorMessage: null,
    },
  });

  assert.match(html, /data-page-state="draft-restored"/);
  assert.match(html, /Draft restored/);
  assert.match(html, /value="corrected_species"[\s\S]*checked/);
  assert.match(html, /value="Southern Boobook"/);
});

test("conflict state compares versions and preserves unsaved input without a submit retry", () => {
  const latestSession = session({
    version: 2,
    history: [{
      actor: "fixture-concurrent-review",
      action: "stale_write_conflict",
      timestamp: "2026-08-17T02:00:00.000Z",
      previousStatus: "awaiting_first_review",
      resultingStatus: "awaiting_first_review",
    }],
  });
  const html = renderApplicationView({
    detectionState: detectionState("populated"),
    workflowActor: "reviewer-1",
    workflowSession: latestSession,
    workflowLoadError: null,
    workflowConflict: {
      expectedVersion: 1,
      latestSession,
      attemptedValues: {
        decision: "rejected",
        correctedSpecies: "",
        reason: "Entered evidence remains visible.",
        resolutionReason: "",
      },
    },
  });

  assert.match(html, /data-page-state="conflict"/);
  assert.match(html, /Version 1/);
  assert.match(html, /Version 2/);
  assert.match(html, /Entered evidence remains visible/);
  assert.match(html, /data-conflict-keep-draft/);
  assert.match(html, /data-conflict-discard-draft/);
  assert.doesNotMatch(html, /type="submit"/);
});

test("consensus and finalized sessions expose the success page state", () => {
  for (const workflowSession of [
    session({
      actor: "reviewer-2",
      status: "consensus",
      version: 3,
      submissions: {
        "reviewer-1": { actor: "reviewer-1", decision: "confirmed", correctedSpecies: null, reason: null },
        "reviewer-2": { actor: "reviewer-2", decision: "confirmed", correctedSpecies: null, reason: null },
      },
      consensus: { decision: "confirmed", correctedSpecies: null },
    }),
    session({
      actor: "adjudicator",
      status: "finalized",
      version: 4,
      submissions: {
        "reviewer-1": { actor: "reviewer-1", decision: "confirmed", correctedSpecies: null, reason: null },
        "reviewer-2": { actor: "reviewer-2", decision: "rejected", correctedSpecies: null, reason: "Mismatch" },
      },
      adjudication: {
        decision: "rejected",
        correctedSpecies: null,
        resolutionReason: "Evidence comparison supports rejection.",
      },
    }),
  ]) {
    const html = renderApplicationView({
      detectionState: detectionState("populated"),
      workflowActor: workflowSession.actor,
      workflowSession,
      workflowLoadError: null,
    });
    assert.match(html, /data-page-state="success"/);
    assert.match(html, /role="status"/);
  }
});

test("adjudication has a named page state and labelled final-decision form", () => {
  const workflowSession = session({
    actor: "adjudicator",
    status: "awaiting_adjudication",
    version: 3,
    firstReviewComplete: true,
    submissions: {
      "reviewer-1": { actor: "reviewer-1", decision: "confirmed", correctedSpecies: null, reason: null },
      "reviewer-2": { actor: "reviewer-2", decision: "rejected", correctedSpecies: null, reason: "Mismatch" },
    },
  });
  const html = renderApplicationView({
    detectionState: detectionState("populated"),
    workflowActor: "adjudicator",
    workflowSession,
    workflowLoadError: null,
  });

  assert.match(html, /data-page-state="adjudication"/);
  assert.match(html, /data-adjudication-form[\s\S]*aria-describedby=/);
  assert.match(html, /Case audit history/);
});
