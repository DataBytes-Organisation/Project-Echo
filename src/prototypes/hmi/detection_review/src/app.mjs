import {
  detectionFixtures,
  malformedDetectionFixtures,
} from "./fixtures.mjs";
import { renderApplicationView } from "./app-view.mjs";
import {
  DRAFT_STORAGE_ERROR_MESSAGE,
  DraftStorageError,
  createBrowserReviewDraftStore,
} from "./draft-store.mjs";
import {
  createAsyncViewGuard,
  getSubmissionVersion,
  lockSubmissionForm,
  persistDraftAtBoundary,
} from "./app-coordination.mjs";
import { restoreQueueItemFocus } from "./focus.mjs";
import { getQueueNavigationTarget } from "./keyboard.mjs";
import {
  createConflictState,
  createRecoveryState,
  discardConflictDraft,
  discardRecoveredDraft,
  keepConflictDraft,
  offerRecoveredDraft,
  restoreRecoveredDraft,
} from "./recovery-state.mjs";
import { FixtureDetectionReviewRepository } from "./repository.mjs";
import {
  reviewScenarioFixtures,
  scenarioActors,
} from "./review-fixtures.mjs";
import { ReviewValidationError } from "./review-domain.mjs";
import { FixtureReviewWorkflowRepository } from "./review-repository.mjs";
import {
  ReviewLoadError,
  ReviewConflictError,
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
  ? new FixtureReviewWorkflowRepository(reviewScenarioFixtures[scenario], {
    conflictOnNextSave: scenario === "conflict",
    now: () => "2026-08-16T04:00:00.000Z",
  })
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
let draftRecovery = createRecoveryState();
let workflowConflict = null;
let workbench;
let viewGuard;
let draftStore = null;

try {
  draftStore = createBrowserReviewDraftStore(window, {
    now: () => "2026-08-16T04:00:00.000Z",
  });
} catch (error) {
  draftRecovery = createRecoveryState(error instanceof DraftStorageError
    ? error.userMessage
    : DRAFT_STORAGE_ERROR_MESSAGE);
}

const DEMO_DRAFT_VALUES = Object.freeze({
  decision: "corrected_species",
  correctedSpecies: "Southern Boobook",
  reason: "The shorter repeated cadence supports a different species.",
  resolutionReason: "",
});

function render() {
  root.innerHTML = renderApplicationView({
    detectionState: workbench.getState(),
    workflowActor,
    workflowSession,
    workflowLoadError,
    adjudicationSessions,
    workflowRenderOptions,
    draftRecovery,
    workflowConflict,
  });
}

workbench = createReviewWorkbench(detectionRepository, render);
viewGuard = createAsyncViewGuard(() => workbench.getState().selectedId);

for (const link of document.querySelectorAll("[data-scenario-link]")) {
  if (link.dataset.scenarioLink === scenario) {
    link.setAttribute("aria-current", "page");
  }
}

async function refreshWorkflow(
  detectionId,
  viewToken = viewGuard.begin(detectionId),
) {
  if (!workflow) {
    return false;
  }

  try {
    const loadedSession = await workflow.getSession(detectionId, workflowActor);
    const loadedAdjudicationSessions = workflowActor === "adjudicator"
      ? await workflow.listAdjudication()
      : [];

    if (!viewGuard.isCurrent(viewToken)) {
      return false;
    }

    workflowSession = loadedSession;
    adjudicationSessions = loadedAdjudicationSessions;
    workflowRenderOptions = {};
    workflowLoadError = null;
    workflowConflict = null;
    loadRecoveredDraft();
  } catch (error) {
    if (!viewGuard.isCurrent(viewToken)) {
      return false;
    }

    workflowSession = null;
    adjudicationSessions = [];
    workflowLoadError = error instanceof ReviewLoadError
      ? error.userMessage
      : "The review workflow could not be loaded. Try again manually.";
    draftRecovery = createRecoveryState();
    workflowConflict = null;
  }

  render();
  return true;
}

function draftContext(session = workflowSession) {
  if (!session) {
    return null;
  }

  return {
    detectionId: session.detectionId,
    actor: session.actor,
    kind: session.actor === "adjudicator" ? "adjudication" : "review",
  };
}

function loadRecoveredDraft() {
  const context = draftContext();

  if (!context) {
    draftRecovery = createRecoveryState();
    return;
  }

  if (!draftStore) {
    draftRecovery = createRecoveryState(DRAFT_STORAGE_ERROR_MESSAGE);
    return;
  }

  try {
    let draft = draftStore.load(context);

    if (scenario === "draft-restored" && !draft) {
      draft = draftStore.save(context, DEMO_DRAFT_VALUES, workflowSession.version);
    }

    draftRecovery = offerRecoveredDraft(createRecoveryState(), draft);

    if (scenario === "draft-restored" && draft) {
      draftRecovery = restoreRecoveredDraft(draftRecovery);
      workflowRenderOptions = { values: draftRecovery.values };
    }
  } catch (error) {
    draftRecovery = error instanceof DraftStorageError
      ? createRecoveryState(error.userMessage)
      : createRecoveryState("Draft recovery is unavailable in this browser.");
  }
}

function persistDraftValues(
  values,
  session = workflowSession,
  version = getSubmissionVersion(session, draftRecovery),
) {
  const context = draftContext(session);

  if (!context) {
    return Object.freeze({
      status: "discarded",
      draft: null,
      errorMessage: null,
    });
  }

  const result = persistDraftAtBoundary(draftStore, context, values, version);

  if (result.status === "failed") {
    draftRecovery = createRecoveryState(result.errorMessage);
  }

  return result;
}

function discardStoredDraft(session = workflowSession, expectedDraft = null) {
  const context = draftContext(session);

  if (!context) {
    return Object.freeze({ status: "discarded", errorMessage: null });
  }

  try {
    if (expectedDraft) {
      draftStore.discardIfUnchanged(context, expectedDraft);
    } else {
      draftStore.discard(context);
    }
    return Object.freeze({ status: "discarded", errorMessage: null });
  } catch (_error) {
    return Object.freeze({
      status: "failed",
      errorMessage: DRAFT_STORAGE_ERROR_MESSAGE,
    });
  }
}

function discardDraft(session = workflowSession) {
  const result = discardStoredDraft(session);
  draftRecovery = result.status === "failed"
    ? createRecoveryState(result.errorMessage)
    : discardRecoveredDraft(draftRecovery);
  return result;
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

root.addEventListener("input", event => {
  const form = event.target.closest("[data-review-form], [data-adjudication-form]");

  if (!form || !workflowSession
    || ["available", "failed"].includes(draftRecovery.status)) {
    return;
  }

  const values = formValues(form);
  const fieldName = event.target.name;
  const result = persistDraftValues(values);

  if (result.status === "failed") {
    workflowRenderOptions = { ...workflowRenderOptions, values };
    render();
    root.querySelector(`[name="${fieldName}"]`)?.focus();
  }
});

async function selectDetection(detectionId) {
  workflowSession = null;
  adjudicationSessions = [];
  workflowRenderOptions = {};
  workflowLoadError = null;
  draftRecovery = createRecoveryState();
  workflowConflict = null;
  const state = workbench.select(detectionId);

  if (state.selectedId !== detectionId) {
    return;
  }

  const viewToken = viewGuard.begin(detectionId);

  if (workflow) {
    await refreshWorkflow(detectionId, viewToken);
  }

  if (viewGuard.isCurrent(viewToken)) {
    restoreQueueItemFocus(root, detectionId);
  }
}

root.addEventListener("keydown", async event => {
  const queueItem = event.target.closest("[data-detection-id]");

  if (!queueItem || event.altKey || event.ctrlKey || event.metaKey) {
    return;
  }

  const state = workbench.getState();
  const targetId = getQueueNavigationTarget(
    state.records,
    queueItem.dataset.detectionId,
    event.key,
  );

  if (targetId) {
    event.preventDefault();
    await selectDetection(targetId);
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

  if (event.target.closest("[data-draft-restore]")) {
    draftRecovery = restoreRecoveredDraft(draftRecovery);
    workflowRenderOptions = { values: draftRecovery.values ?? {} };
    render();
    root.querySelector('[name="decision"]:checked, [name="decision"]')?.focus();
    return;
  }

  if (event.target.closest("[data-draft-discard]")) {
    discardDraft();
    workflowRenderOptions = {};
    render();
    root.querySelector('[name="decision"]')?.focus();
    return;
  }

  if (event.target.closest("[data-conflict-keep-draft]") && workflowConflict) {
    const rebased = keepConflictDraft(workflowConflict);
    workflowSession = rebased.latestSession;
    const persistence = persistDraftValues(
      rebased.values,
      workflowSession,
      rebased.expectedVersion,
    );

    if (persistence.status === "saved") {
      draftRecovery = restoreRecoveredDraft(offerRecoveredDraft(
        createRecoveryState(),
        persistence.draft,
      ));
    } else if (persistence.status === "discarded") {
      draftRecovery = discardRecoveredDraft(createRecoveryState());
    }

    workflowRenderOptions = { values: rebased.values };
    workflowConflict = null;
    render();
    root.querySelector('[name="decision"]:checked, [name="decision"]')?.focus();
    return;
  }

  if (event.target.closest("[data-conflict-discard-draft]") && workflowConflict) {
    const discarded = discardConflictDraft(workflowConflict);
    workflowSession = discarded.latestSession;
    discardDraft(workflowSession);
    workflowRenderOptions = {};
    workflowConflict = null;
    render();
    root.querySelector('[name="decision"]')?.focus();
    return;
  }

  const queueItem = event.target.closest("[data-detection-id]");

  if (!queueItem) {
    return;
  }

  const detectionId = queueItem.dataset.detectionId;
  await selectDetection(detectionId);
});

root.addEventListener("submit", async event => {
  const form = event.target.closest("[data-review-form], [data-adjudication-form]");

  if (!form || !workflowSession) {
    return;
  }

  event.preventDefault();
  const values = formValues(form);
  const submittedSession = workflowSession;
  const expectedVersion = getSubmissionVersion(submittedSession, draftRecovery);
  const submissionToken = viewGuard.capture(submittedSession.detectionId);
  const submittedDraftResult = persistDraftValues(
    values,
    submittedSession,
    expectedVersion,
  );
  lockSubmissionForm(form);

  try {
    const nextSession = form.matches("[data-adjudication-form]")
      ? await workflow.finalize(submittedSession.detectionId, {
        actor: "adjudicator",
        decision: values.decision,
        correctedSpecies: values.correctedSpecies,
        resolutionReason: values.resolutionReason,
        expectedVersion,
      })
      : await workflow.submitReview(submittedSession.detectionId, {
        actor: form.dataset.actor,
        decision: values.decision,
        correctedSpecies: values.correctedSpecies,
        reason: values.reason,
        expectedVersion,
      });
    const nextAdjudicationSessions = workflowActor === "adjudicator"
      ? await workflow.listAdjudication()
      : [];

    if (!viewGuard.isCurrent(submissionToken)) {
      return;
    }

    const discardResult = submittedDraftResult.status === "saved"
      ? discardStoredDraft(submittedSession, submittedDraftResult.draft)
      : submittedDraftResult;

    workflowSession = nextSession;
    adjudicationSessions = nextAdjudicationSessions;
    workflowRenderOptions = {};
    draftRecovery = submittedDraftResult.status === "failed"
      ? createRecoveryState(submittedDraftResult.errorMessage)
      : discardResult.status === "failed"
      ? createRecoveryState(discardResult.errorMessage)
      : discardRecoveredDraft(draftRecovery);
    workflowConflict = null;
    workflowLoadError = null;
    render();
  } catch (error) {
    if (!viewGuard.isCurrent(submissionToken)) {
      return;
    }

    if (error instanceof ReviewLoadError) {
      workflowSession = null;
      workflowLoadError = error.userMessage;
      render();
      return;
    }

    if (error instanceof ReviewConflictError) {
      workflowSession = error.latestSession;
      workflowConflict = createConflictState(error);
      workflowRenderOptions = { values: error.attemptedValues };
      render();
      root.querySelector("#conflict-heading")?.focus();
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
    await refreshWorkflow(state.selectedId, viewGuard.begin(state.selectedId));
  }
}
