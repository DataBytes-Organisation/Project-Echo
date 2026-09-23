function freezeValues(values) {
  return values ? Object.freeze({ ...values }) : null;
}

function freezeState(state) {
  return Object.freeze({
    ...state,
    values: freezeValues(state.values),
  });
}

export function createRecoveryState(errorMessage = null) {
  return freezeState({
    status: errorMessage ? "failed" : "idle",
    draft: null,
    values: null,
    errorMessage,
  });
}

export function offerRecoveredDraft(_state, draft) {
  if (!draft) {
    return createRecoveryState();
  }

  return freezeState({
    status: "available",
    draft,
    values: null,
    errorMessage: null,
  });
}

export function restoreRecoveredDraft(state) {
  if (!state?.draft) {
    return state ?? createRecoveryState();
  }

  return freezeState({
    status: "restored",
    draft: state.draft,
    values: state.draft.values,
    errorMessage: null,
  });
}

export function discardRecoveredDraft(_state) {
  return freezeState({
    status: "discarded",
    draft: null,
    values: null,
    errorMessage: null,
  });
}

export function createConflictState({
  expectedVersion,
  latestSession,
  attemptedValues,
}) {
  return Object.freeze({
    status: "conflict",
    expectedVersion,
    latestSession,
    attemptedValues: freezeValues(attemptedValues),
    shouldSubmit: false,
  });
}

export function keepConflictDraft(conflictState) {
  return Object.freeze({
    status: "rebased",
    expectedVersion: conflictState.latestSession.version,
    latestSession: conflictState.latestSession,
    values: freezeValues(conflictState.attemptedValues),
    shouldSubmit: false,
  });
}

export function discardConflictDraft(conflictState) {
  return Object.freeze({
    status: "discarded",
    expectedVersion: conflictState.latestSession.version,
    latestSession: conflictState.latestSession,
    values: null,
    shouldSubmit: false,
  });
}
