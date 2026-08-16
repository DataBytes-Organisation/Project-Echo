import {
  detectionFixtures,
  malformedDetectionFixtures,
} from "./fixtures.mjs";
import { renderApplicationView } from "./app-view.mjs";
import { restoreQueueItemFocus } from "./focus.mjs";
import { FixtureDetectionReviewRepository } from "./repository.mjs";
import {
  reviewScenarioFixtures,
  scenarioActors,
} from "./review-fixtures.mjs";
import { ReviewValidationError } from "./review-domain.mjs";
import { FixtureReviewWorkflowRepository } from "./review-repository.mjs";
import {
  ReviewLoadError,
  ReviewSubmissionError,
  createReviewWorkflow,
} from "./review-workflow.mjs";
import { createReviewWorkbench } from "./workbench-state.mjs";

const QUEUE_SCENARIOS = new Set(["populated", "loading", "empty", "error"]);
const WORKFLOW_SCENARIOS = new Set(Object.keys(reviewScenarioFixtures));
const ALLOWED_SCENARIOS = new Set([...QUEUE_SCENARIOS, ...WORKFLOW_SCENARIOS]);
const requestedParameters = new URLSearchParams(window.location.search);
const requestedScenario = requestedParameters.get("scenario");
const requestedDetection = requestedParameters.get("detection");
const scenario = ALLOWED_SCENARIOS.has(requestedScenario)
  ? requestedScenario
  : "first-review";
const fixtures = scenario === "empty"
  ? []
  : scenario === "error"
    ? malformedDetectionFixtures
    : detectionFixtures;
const detectionRepository = new FixtureDetectionReviewRepository(fixtures);
const root = document.querySelector("#app");
const workflowActor = scenarioActors[scenario] ?? null;
const workflowRepository = workflowActor
  ? new FixtureReviewWorkflowRepository(reviewScenarioFixtures[scenario])
  : null;
const workflow = workflowRepository
  ? createReviewWorkflow(workflowRepository, {
    now: () => "2026-08-16T04:00:00.000Z",
  })
  : null;

let workflowSession = null;
let adjudicationSessions = [];
let workflowRenderOptions = {};
let workflowLoadError = null;
let workbench;

function render() {
  root.innerHTML = renderApplicationView({
    detectionState: workbench.getState(),
    workflowActor,
    workflowSession,
    workflowLoadError,
    adjudicationSessions,
    workflowRenderOptions,
  });
}

workbench = createReviewWorkbench(detectionRepository, render);

for (const link of document.querySelectorAll("[data-scenario-link]")) {
  if (link.dataset.scenarioLink === scenario) {
    link.setAttribute("aria-current", "page");
  }
}

async function refreshWorkflow(detectionId) {
  if (!workflow) {
    return;
  }

  try {
    workflowSession = await workflow.getSession(detectionId, workflowActor);
    adjudicationSessions = workflowActor === "adjudicator"
      ? await workflow.listAdjudication()
      : [];
    workflowRenderOptions = {};
    workflowLoadError = null;
  } catch (error) {
    workflowSession = null;
    adjudicationSessions = [];
    workflowLoadError = error instanceof ReviewLoadError
      ? error.userMessage
      : "The review workflow could not be loaded. Try again manually.";
  }

  render();
}

function formValues(form) {
  const formData = new FormData(form);
  return {
    decision: String(formData.get("decision") ?? ""),
    reason: String(formData.get("reason") ?? ""),
    correctedSpecies: String(formData.get("correctedSpecies") ?? ""),
    resolutionReason: String(formData.get("resolutionReason") ?? ""),
  };
}

function syncConditionalFields(form) {
  const decision = form.querySelector('[name="decision"]:checked')?.value ?? "";
  const correctedField = form.querySelector("[data-corrected-species-field]");
  const correctedInput = form.querySelector('[name="correctedSpecies"]');
  const reasonField = form.querySelector("[data-reason-field]");
  const reasonInput = form.querySelector('[name="reason"], [name="resolutionReason"]');

  if (correctedField) {
    correctedField.hidden = decision !== "corrected_species";
    correctedInput.required = decision === "corrected_species";
  }

  if (reasonField && reasonInput) {
    const isAdjudication = reasonInput.name === "resolutionReason";
    reasonField.hidden = !isAdjudication
      && (decision === "" || decision === "confirmed");
    reasonInput.required = isAdjudication
      || (decision !== "" && decision !== "confirmed");
  }
}

root.addEventListener("change", event => {
  if (event.target.matches('[name="decision"]')) {
    syncConditionalFields(event.target.form);
  }
});

root.addEventListener("click", async event => {
  const retryButton = event.target.closest("[data-workflow-retry]");

  if (retryButton) {
    const state = workbench.getState();
    if (state.status === "populated") {
      await refreshWorkflow(state.selectedId);
    }
    return;
  }

  const queueItem = event.target.closest("[data-detection-id]");

  if (!queueItem) {
    return;
  }

  const detectionId = queueItem.dataset.detectionId;
  workbench.select(detectionId);

  if (workflow) {
    await refreshWorkflow(detectionId);
  }

  restoreQueueItemFocus(root, detectionId);
});

root.addEventListener("submit", async event => {
  const form = event.target.closest("[data-review-form], [data-adjudication-form]");

  if (!form || !workflowSession) {
    return;
  }

  event.preventDefault();
  const values = formValues(form);
  const submitButton = form.querySelector('[type="submit"]');
  submitButton.disabled = true;
  submitButton.setAttribute("aria-busy", "true");

  try {
    workflowSession = form.matches("[data-adjudication-form]")
      ? await workflow.finalize(workflowSession.detectionId, {
        actor: "adjudicator",
        decision: values.decision,
        correctedSpecies: values.correctedSpecies,
        resolutionReason: values.resolutionReason,
      })
      : await workflow.submitReview(workflowSession.detectionId, {
        actor: form.dataset.actor,
        decision: values.decision,
        correctedSpecies: values.correctedSpecies,
        reason: values.reason,
      });
    adjudicationSessions = workflowActor === "adjudicator"
      ? await workflow.listAdjudication()
      : [];
    workflowRenderOptions = {};
    render();
  } catch (error) {
    if (error instanceof ReviewLoadError) {
      workflowSession = null;
      workflowLoadError = error.userMessage;
      render();
      return;
    }

    const errors = error instanceof ReviewValidationError
      ? { [error.field]: error.message }
      : {};
    workflowRenderOptions = {
      errors,
      values,
      submissionError: error instanceof ReviewSubmissionError
        ? error.userMessage
        : error instanceof ReviewValidationError
          ? ""
          : "The submission could not be processed.",
    };
    render();

    if (error instanceof ReviewValidationError) {
      root.querySelector(`[name="${error.field}"]`)?.focus();
    }
  }
});

render();

if (scenario !== "loading") {
  let state = await workbench.load();

  if (requestedDetection && state.status === "populated") {
    state = workbench.select(requestedDetection);
  }

  if (workflow && state.status === "populated") {
    await refreshWorkflow(state.selectedId);
  }
}
