import test from "node:test";
import assert from "node:assert/strict";

import {
  renderAdjudicationQueue,
  renderReviewWorkflowFailure,
  renderReviewWorkflow,
} from "../src/workflow-render.mjs";

const reviewerOneSubmission = Object.freeze({
  actor: "reviewer-1",
  decision: "confirmed",
  reason: null,
  correctedSpecies: null,
  submittedAt: "2026-08-16T01:00:00.000Z",
});

const reviewerTwoSubmission = Object.freeze({
  actor: "reviewer-2",
  decision: "rejected",
  reason: "The evidence contradicts the prediction.",
  correctedSpecies: null,
  submittedAt: "2026-08-16T02:00:00.000Z",
});

function makeSession(overrides = {}) {
  return {
    detectionId: "det-echo-001",
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

test("renders all four outcomes in the first-review decision form", () => {
  const html = renderReviewWorkflow(makeSession());

  assert.match(html, /First independent review/);
  assert.match(html, /value="confirmed"/);
  assert.match(html, /value="rejected"/);
  assert.match(html, /value="corrected_species"/);
  assert.match(html, /value="insufficient_evidence"/);
  assert.match(html, /name="reason"/);
  assert.match(html, /name="correctedSpecies"/);
  assert.equal((html.match(/name="decision"[\s\S]*?required/g) ?? []).length, 4);
  assert.match(html, /data-review-form[\s\S]*aria-describedby=/);
});

test("offers explicit Restore and Discard actions before applying a recovered draft", () => {
  const html = renderReviewWorkflow(makeSession(), {
    recoveryState: {
      status: "available",
      draft: { version: 1, savedAt: "2026-08-17T01:00:00.000Z" },
      values: null,
      errorMessage: null,
    },
  });

  assert.match(html, /Recovered draft available/);
  assert.match(html, /data-draft-restore/);
  assert.match(html, /data-draft-discard/);
  assert.doesNotMatch(html, /checked/);
});

test("renders audit history as an always-visible labelled ordered list", () => {
  const html = renderReviewWorkflow(makeSession({
    history: [{
      actor: "reviewer-1",
      action: "first_review_submitted",
      timestamp: "2026-08-16T01:00:00.000Z",
      previousStatus: "awaiting_first_review",
      resultingStatus: "awaiting_second_review",
    }],
  }));

  assert.match(html, /<section[^>]*class="history-panel"/);
  assert.match(html, /Case audit history/);
  assert.match(html, /<ol>/);
  assert.doesNotMatch(html, /<details|<summary/);
});

test("renders native required state and associated field errors in form order", () => {
  const html = renderReviewWorkflow(makeSession(), {
    values: {
      decision: "corrected_species",
      correctedSpecies: "",
      reason: "",
    },
    errors: {
      correctedSpecies: "Enter the corrected species.",
    },
  });

  assert.match(html, /name="correctedSpecies"[\s\S]*?required[\s\S]*?aria-invalid="true"[\s\S]*?aria-describedby="correctedSpecies-error"/);
  assert.match(html, /name="reason"[\s\S]*?required/);
  assert.match(html, /id="correctedSpecies-error" data-error-for="correctedSpecies"/);
});

test("renders a blind second-review form without first-review decision content", () => {
  const html = renderReviewWorkflow(makeSession({
    actor: "reviewer-2",
    status: "awaiting_second_review",
    firstReviewComplete: true,
  }));

  assert.match(html, /Second independent review/);
  assert.match(html, /first review is complete/i);
  assert.doesNotMatch(html, /Independent review comparison|submission-card|Reviewer 1<\/p>/i);
});

test("renders consensus success after matching independent reviews", () => {
  const html = renderReviewWorkflow(makeSession({
    actor: "reviewer-2",
    status: "consensus",
    firstReviewComplete: true,
    submissions: {
      "reviewer-1": reviewerOneSubmission,
      "reviewer-2": { ...reviewerOneSubmission, actor: "reviewer-2", submittedAt: "2026-08-16T02:00:00.000Z" },
    },
    consensus: {
      decision: "confirmed",
      correctedSpecies: null,
      reachedAt: "2026-08-16T02:00:00.000Z",
    },
  }));

  assert.match(html, /Consensus reached/);
  assert.match(html, /Confirmed/);
  assert.doesNotMatch(html, /name="decision"/);
});

test("renders disagreement context and adjudication controls", () => {
  const session = makeSession({
    actor: "adjudicator",
    status: "awaiting_adjudication",
    firstReviewComplete: true,
    submissions: {
      "reviewer-1": reviewerOneSubmission,
      "reviewer-2": reviewerTwoSubmission,
    },
  });
  const html = renderReviewWorkflow(session);
  const queueHtml = renderAdjudicationQueue([session]);

  assert.match(html, /Adjudication required/);
  assert.match(html, /Reviewer 1/);
  assert.match(html, /Reviewer 2/);
  assert.match(html, /name="resolutionReason"/);
  assert.match(html, /name="resolutionReason"[\s\S]*?required/);
  assert.match(queueHtml, /1 case/);
  assert.match(queueHtml, /det-echo-001/);
});

test("escapes reviewer and workflow error content before rendering", () => {
  const session = makeSession({
    actor: "adjudicator",
    status: "awaiting_adjudication",
    firstReviewComplete: true,
    submissions: {
      "reviewer-1": reviewerOneSubmission,
      "reviewer-2": {
        ...reviewerTwoSubmission,
        reason: '<img src=x onerror="alert(1)">',
      },
    },
  });
  const html = renderReviewWorkflow(session, {
    submissionError: "<script>internal()</script>",
  });
  const failureHtml = renderReviewWorkflowFailure("<script>raw()</script>");

  assert.doesNotMatch(html, /<img|<script>/);
  assert.match(html, /&lt;img/);
  assert.doesNotMatch(failureHtml, /<script>/);
  assert.match(failureHtml, /&lt;script&gt;/);
  assert.match(failureHtml, /data-workflow-retry/);
});

test("renders the finalized outcome and resolution reason without another form", () => {
  const html = renderReviewWorkflow(makeSession({
    actor: "adjudicator",
    status: "finalized",
    firstReviewComplete: true,
    submissions: {
      "reviewer-1": reviewerOneSubmission,
      "reviewer-2": reviewerTwoSubmission,
    },
    adjudication: {
      actor: "adjudicator",
      decision: "rejected",
      correctedSpecies: null,
      resolutionReason: "The disagreement was resolved against the prediction.",
      submittedAt: "2026-08-16T03:00:00.000Z",
    },
  }));

  assert.match(html, /Adjudication finalized/);
  assert.match(html, /The disagreement was resolved against the prediction/);
  assert.doesNotMatch(html, /<form/);
});
