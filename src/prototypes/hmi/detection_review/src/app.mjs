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
  getSubmissionVersion,
  lockSubmissionForm,
  persistDraftAtBoundary,
} from "./app-coordination.mjs";
import {
  beginRecognizedWorkflowSubmission,
  createAppRuntimeController,
} from "./app-runtime.mjs";
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
import { createReviewCase, ReviewValidationError } from "./review-domain.mjs";
import { createBrowserPersistentReviewRepository } from "./persistent-review-repository.mjs";
import { createBrowserPrototypePreferenceStore } from "./prototype-preferences.mjs";
import {
  firstActionableDetectionId,
  isActionableForRole,
  orderRecordsForRole,
} from "./role-workspace.mjs";
import {
  ReviewLoadError,
  ReviewConflictError,
  ReviewSubmissionError,
  createReviewWorkflow,
} from "./review-workflow.mjs";
import { createReviewWorkbench } from "./workbench-state.mjs";

const now = () => "2026-08-16T04:00:00.000Z";
const requestedParameters = new URLSearchParams(window.location.search);
const fixtureState = requestedParameters.get("fixtureState");
const conflictOnNextSave = requestedParameters.get("simulateConflict") === "1";
const simulateStorageFailure = requestedParameters.get("simulateStorageFailure") === "1";
const detectionRepository = new FixtureDetectionReviewRepository(detectionFixtures);
const emptyDetectionRepository = new FixtureDetectionReviewRepository([]);
const failedDetectionRepository = new FixtureDetectionReviewRepository(
  malformedDetectionFixtures,
);
const selectedDetectionRepository = fixtureState === "empty"
  ? emptyDetectionRepository
  : fixtureState === "error"
    ? failedDetectionRepository
    : detectionRepository;
const reviewSeeds = detectionFixtures.map(record => createReviewCase(record.id));
const root = document.querySelector("#app");

function createStorageFailureAdapter() {
  const windowAdapter = {};
  Object.defineProperty(windowAdapter, "localStorage", {
    get() {
      throw new Error("Deterministic blocked browser storage fixture.");
    },
  });
  return windowAdapter;
}

function controlledMessage(error) {
  return typeof error?.userMessage === "string"
    ? error.userMessage
    : "Prototype data could not be accessed in this browser.";
}

function failedRepository(message) {
  return Object.freeze({
    async list() {
      const error = new Error(message);
      error.userMessage = message;
      throw error;
    },
  });
}

const browserWindow = simulateStorageFailure
  ? createStorageFailureAdapter()
  : window;
let workflowRepository = null;
let workflow = null;
let draftStore = null;
let preferenceStore = null;
let initialRole = "reviewer-1";
let startupErrorMessage = null;

try {
  workflowRepository = createBrowserPersistentReviewRepository(
    browserWindow,
    reviewSeeds,
    { conflictOnNextSave },
  );
  workflow = createReviewWorkflow(workflowRepository, { now });
  draftStore = createBrowserReviewDraftStore(browserWindow, { now });
  preferenceStore = createBrowserPrototypePreferenceStore(browserWindow);
  initialRole = preferenceStore.loadRole();
} catch (error) {
  startupErrorMessage = controlledMessage(error);
}

let baseRecords = [];
let roleSessions = new Map();
let workflowSession = null;
let adjudicationSessions = [];
let workflowRenderOptions = {};
let workflowLoadError = null;
let draftRecovery = createRecoveryState();
let workflowConflict = null;
let resetStatus = null;
let workbench;
let runtime;

function sessionForRender() {
  if (workflowSession?.status === "awaiting_first_review"
    && !isActionableForRole(workflowSession.status, runtime.activeRole)) {
    return null;
  }

  return workflowSession;
}

function render() {
  root.innerHTML = renderApplicationView({
    detectionState: workbench.getState(),
    workflowActor: runtime.activeRole,
    activeRole: runtime.activeRole,
    isPending: runtime.isPending,
    resetStatus,
    workflowSession: sessionForRender(),
    workflowLoadError,
    adjudicationSessions,
    workflowRenderOptions,
    draftRecovery,
    workflowConflict,
  });
}

function clearWorkflowView() {
  workflowSession = null;
  adjudicationSessions = [];
  workflowRenderOptions = {};
  workflowLoadError = null;
  draftRecovery = createRecoveryState();
  workflowConflict = null;
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
    const draft = draftStore.load(context);
    draftRecovery = offerRecoveredDraft(createRecoveryState(), draft);
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

async function buildWorkbench(records, selectedId) {
  const nextWorkbench = createReviewWorkbench({
    async list() {
      return records;
    },
  });
  await nextWorkbench.load();

  if (selectedId) {
    nextWorkbench.select(selectedId);
  }

  return nextWorkbench;
}

async function refreshRoleWorkspace({
  forceFirstActionable = false,
  selectedId = workbench.getState().selectedId,
  viewToken = runtime.beginView(),
  focusSelector = null,
} = {}) {
  if (!workflow || baseRecords.length === 0) {
    return false;
  }

  try {
    const entries = await Promise.all(baseRecords.map(async record => [
      record.id,
      await workflow.getSession(record.id, runtime.activeRole),
    ]));

    if (!runtime.isCurrent(viewToken)) {
      return false;
    }

    const nextSessions = new Map(entries);
    const orderedRecords = orderRecordsForRole(
      baseRecords,
      nextSessions,
      runtime.activeRole,
    );
    const firstActionableId = firstActionableDetectionId(
      baseRecords,
      nextSessions,
      runtime.activeRole,
    );
    const selectedSession = nextSessions.get(selectedId);
    const selectedIsActionable = selectedSession
      ? isActionableForRole(selectedSession.status, runtime.activeRole)
      : false;
    let nextSelectedId = orderedRecords.some(record => record.id === selectedId)
      ? selectedId
      : orderedRecords[0]?.id ?? null;

    if (firstActionableId && (forceFirstActionable || !selectedIsActionable)) {
      nextSelectedId = firstActionableId;
    }

    const nextWorkbench = await buildWorkbench(orderedRecords, nextSelectedId);

    if (!runtime.isCurrent(viewToken)) {
      return false;
    }

    roleSessions = nextSessions;
    workbench = nextWorkbench;
    workflowSession = roleSessions.get(nextSelectedId) ?? null;
    adjudicationSessions = runtime.activeRole === "adjudicator"
      ? [...roleSessions.values()].filter(
        session => session.status === "awaiting_adjudication",
      )
      : [];
    workflowRenderOptions = {};
    workflowLoadError = null;
    workflowConflict = null;
    loadRecoveredDraft();
    render();
    root.querySelector(focusSelector)?.focus();
    return true;
  } catch (error) {
    if (!runtime.isCurrent(viewToken)) {
      return false;
    }

    workflowSession = null;
    adjudicationSessions = [];
    workflowLoadError = error instanceof ReviewLoadError
      ? error.userMessage
      : controlledMessage(error);
    workflowRenderOptions = {};
    draftRecovery = createRecoveryState();
    workflowConflict = null;
    render();
    root.querySelector(focusSelector)?.focus();
    return false;
  }
}

async function selectDetection(detectionId) {
  let state;
  const navigation = runtime.beginNavigation(() => {
    clearWorkflowView();
    resetStatus = null;
    state = workbench.select(detectionId);
  });

  if (navigation.status === "blocked") {
    return false;
  }

  if (state.selectedId !== detectionId) {
    return false;
  }

  await refreshRoleWorkspace({
    selectedId: detectionId,
    viewToken: navigation.token,
  });

  if (workbench.getState().selectedId === detectionId) {
    restoreQueueItemFocus(root, detectionId);
  }
  return true;
}

workbench = createReviewWorkbench(
  startupErrorMessage
    ? failedRepository(startupErrorMessage)
    : selectedDetectionRepository,
  render,
);
runtime = createAppRuntimeController({
  initialRole,
  getDetectionId: () => workbench.getState().selectedId,
  preferenceStore,
});

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
  const roleButton = event.target.closest("[data-role]");

  if (roleButton) {
    const nextRole = roleButton.dataset.role;

    if (nextRole === runtime.activeRole) {
      roleButton.focus();
      return;
    }

    let switchResult;

    try {
      switchResult = runtime.switchRole(nextRole);
    } catch (error) {
      workflowSession = null;
      workflowLoadError = controlledMessage(error);
      render();
      root.querySelector(`[data-role="${runtime.activeRole}"]`)?.focus();
      return;
    }

    if (switchResult.status === "blocked") {
      roleButton.focus();
      return;
    }

    clearWorkflowView();
    resetStatus = null;
    render();
    const viewToken = runtime.beginView();
    await refreshRoleWorkspace({
      forceFirstActionable: true,
      viewToken,
      focusSelector: `[data-role="${runtime.activeRole}"]`,
    });
    return;
  }

  if (event.target.closest("[data-prototype-reset]")) {
    const resetResult = await runtime.reset({
      confirmReset: () => window.confirm(
        "Reset submitted reviews, final decisions, audit history, and drafts for this prototype?",
      ),
      reviewRepository: workflowRepository,
      draftStore,
      preferenceStore,
    });

    if (resetResult.status === "cancelled") {
      return;
    }

    resetStatus = resetResult;

    if (resetResult.status !== "reset") {
      render();
      return;
    }

    clearWorkflowView();
    workbench = createReviewWorkbench(selectedDetectionRepository, render);
    const state = await workbench.load();
    baseRecords = state.status === "populated" ? state.records : [];

    if (state.status !== "populated") {
      render();
      return;
    }

    const firstDetectionId = baseRecords[0]?.id ?? null;
    if (firstDetectionId) {
      workbench.select(firstDetectionId);
    }
    const viewToken = runtime.beginView();
    await refreshRoleWorkspace({
      forceFirstActionable: true,
      selectedId: firstDetectionId,
      viewToken,
    });
    restoreQueueItemFocus(root, workbench.getState().selectedId);
    return;
  }

  const retryButton = event.target.closest("[data-workflow-retry]");

  if (retryButton) {
    if (runtime.isPending) {
      return;
    }

    clearWorkflowView();
    const viewToken = runtime.beginView();
    await refreshRoleWorkspace({ viewToken });
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
    roleSessions.set(workflowSession.detectionId, workflowSession);
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
    roleSessions.set(workflowSession.detectionId, workflowSession);
    discardDraft(workflowSession);
    workflowRenderOptions = {};
    workflowConflict = null;
    render();
    root.querySelector('[name="decision"]')?.focus();
    return;
  }

  const queueItem = event.target.closest("[data-detection-id]");

  if (queueItem) {
    await selectDetection(queueItem.dataset.detectionId);
  }
});

root.addEventListener("submit", async event => {
  const form = event.target.closest("[data-review-form], [data-adjudication-form]");

  if (!form) {
    return;
  }

  const submission = beginRecognizedWorkflowSubmission(event, runtime, {
    canSubmit: Boolean(workflowSession)
      && isActionableForRole(workflowSession.status, runtime.activeRole),
  });

  if (submission.status === "blocked") {
    return;
  }

  const values = formValues(form);
  const submittedSession = workflowSession;
  const expectedVersion = getSubmissionVersion(submittedSession, draftRecovery);
  const submissionToken = submission.token;
  const submittedDraftResult = persistDraftValues(
    values,
    submittedSession,
    expectedVersion,
  );
  resetStatus = null;
  workflowRenderOptions = { ...workflowRenderOptions, values };
  render();
  const pendingForm = root.querySelector(
    form.matches("[data-adjudication-form]")
      ? "[data-adjudication-form]"
      : "[data-review-form]",
  );
  if (pendingForm) {
    lockSubmissionForm(pendingForm);
  }

  try {
    if (form.matches("[data-adjudication-form]")) {
      await workflow.finalize(submittedSession.detectionId, {
        actor: runtime.activeRole,
        decision: values.decision,
        correctedSpecies: values.correctedSpecies,
        resolutionReason: values.resolutionReason,
        expectedVersion,
      });
    } else {
      await workflow.submitReview(submittedSession.detectionId, {
        actor: runtime.activeRole,
        decision: values.decision,
        correctedSpecies: values.correctedSpecies,
        reason: values.reason,
        expectedVersion,
      });
    }

    const completion = runtime.completeSuccessfulSubmission(
      submissionToken,
      () => submittedDraftResult.status === "saved"
        ? discardStoredDraft(submittedSession, submittedDraftResult.draft)
        : submittedDraftResult,
    );

    if (completion.status === "stale") {
      render();
      return;
    }

    const discardResult = completion.cleanupResult;
    const recoveryError = submittedDraftResult.status === "failed"
      ? submittedDraftResult.errorMessage
      : discardResult.status === "failed"
        ? discardResult.errorMessage
        : null;
    workflowRenderOptions = {};
    draftRecovery = recoveryError
      ? createRecoveryState(recoveryError)
      : discardRecoveredDraft(draftRecovery);
    workflowConflict = null;
    workflowLoadError = null;
    await refreshRoleWorkspace({
      selectedId: submittedSession.detectionId,
      viewToken: submissionToken,
    });

    if (recoveryError) {
      draftRecovery = createRecoveryState(recoveryError);
      render();
    }
  } catch (error) {
    const completion = runtime.completeFailedSubmission(submissionToken);

    if (completion.status === "stale") {
      render();
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
      roleSessions.set(error.latestSession.detectionId, error.latestSession);
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
const initialState = await workbench.load();
baseRecords = initialState.status === "populated" ? initialState.records : [];

if (initialState.status === "populated") {
  const viewToken = runtime.beginView();
  await refreshRoleWorkspace({ viewToken });
}
