import test from "node:test";
import assert from "node:assert/strict";

import { validateDetectionRecord } from "../src/detection-record.mjs";
import {
  renderEvidence,
  renderQueue,
  renderWorkbench,
} from "../src/render.mjs";
import { escapeRegExp, makeDetection } from "../test-support/helpers.mjs";

test("renders the queue loading state", () => {
  const html = renderQueue({
    status: "loading",
    records: [],
    selectedId: null,
    errorMessage: null,
  });

  assert.match(html, /role="status"/);
  assert.match(html, /Loading detection queue/);
  assert.match(html, />Loading<\/span>/);
  assert.doesNotMatch(html, />0 records<\/span>/);
});

test("renders the queue empty state", () => {
  const html = renderQueue({
    status: "empty",
    records: [],
    selectedId: null,
    errorMessage: null,
  });

  assert.match(html, /No detections waiting for review/);
  assert.match(html, />0 records<\/span>/);
});

test("renders a controlled queue error without internal details", () => {
  const rawMessage = "fixture index 0 has confidence 101";
  const controlledMessage = "Detection records could not be loaded because the fixture data is invalid.";
  const html = renderQueue({
    status: "failed",
    records: [],
    selectedId: null,
    errorMessage: controlledMessage,
  });

  assert.match(html, /role="alert"/);
  assert.match(html, new RegExp(escapeRegExp(controlledMessage)));
  assert.doesNotMatch(html, new RegExp(escapeRegExp(rawMessage)));
  assert.match(html, />Unavailable<\/span>/);
  assert.doesNotMatch(html, />0 records<\/span>/);
});

test("renders a populated queue with an explicit selected detection", () => {
  const records = [
    validateDetectionRecord(makeDetection()),
    validateDetectionRecord(makeDetection({ id: "det-echo-002", species: "Koala" })),
  ];
  const html = renderQueue({
    status: "populated",
    records,
    selectedId: "det-echo-002",
    errorMessage: null,
  });

  assert.match(html, /Powerful Owl/);
  assert.match(html, /Koala/);
  assert.match(html, /data-detection-id="det-echo-002"[^>]*aria-pressed="true"/);
  assert.match(html, /Pending review/);
  const queueButton = html.match(/<button[\s\S]*?>/)[0];
  assert.doesNotMatch(queueButton, /aria-label=/);
});

test("renders every required evidence field and available-audio state", () => {
  const html = renderEvidence(validateDetectionRecord(makeDetection()));
  const requiredValues = [
    "Powerful Owl",
    "94.6%",
    "22 May 2026, 04:12 UTC",
    "Sensor ECHO-07",
    "-37.8136",
    "144.9631",
    "24 m",
    "± 18 m",
    "Audio evidence available",
  ];

  for (const value of requiredValues) {
    assert.match(html, new RegExp(escapeRegExp(value)));
  }
});

test("renders the unavailable-audio evidence state", () => {
  const record = validateDetectionRecord(makeDetection({ audioAvailable: false }));

  assert.match(renderEvidence(record), /No audio evidence attached/);
});

test("prompts for a selection when standalone evidence has no record", () => {
  assert.match(renderEvidence(null), /Select a detection/);
});

test("escapes fixture text before rendering evidence", () => {
  const record = validateDetectionRecord(makeDetection({
    species: "<script>alert('field')</script>",
  }));
  const html = renderEvidence(record);

  assert.doesNotMatch(html, /<script>/);
  assert.match(html, /&lt;script&gt;/);
});

test("renders selected evidence with the populated workbench", () => {
  const records = [
    validateDetectionRecord(makeDetection()),
    validateDetectionRecord(makeDetection({ id: "det-echo-002", species: "Koala" })),
  ];
  const html = renderWorkbench({
    status: "populated",
    records,
    selectedId: "det-echo-002",
    errorMessage: null,
  });

  assert.match(html, /Detection queue/);
  assert.match(html, /Evidence record/);
  assert.match(html, /Koala/);
});

test("renders evidence guidance that matches each non-populated queue state", () => {
  const expectations = [
    ["loading", "Evidence will appear after loading"],
    ["empty", "No evidence to inspect"],
    ["failed", "Evidence unavailable"],
  ];

  for (const [status, expectedCopy] of expectations) {
    const html = renderWorkbench({
      status,
      records: [],
      selectedId: null,
      errorMessage: status === "failed" ? "Controlled fixture error" : null,
    });

    assert.match(html, new RegExp(expectedCopy));
    assert.doesNotMatch(html, /Select a detection/);
  }
});

test("composes the Week 2 workflow and adjudication queue with Week 1 evidence", () => {
  const records = [validateDetectionRecord(makeDetection())];
  const html = renderWorkbench({
    status: "populated",
    records,
    selectedId: "det-echo-001",
    errorMessage: null,
  }, {
    workflowHtml: '<section data-test-workflow="true">Review workflow</section>',
    adjudicationQueueHtml: '<aside data-test-adjudication="true">Disagreement queue</aside>',
  });

  assert.match(html, /data-test-workflow="true"/);
  assert.match(html, /data-test-adjudication="true"/);
  assert.match(html, /Evidence record/);
});
