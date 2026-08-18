import { DRAFT_STORAGE_ERROR_MESSAGE } from "./draft-store.mjs";

function isMatchingRestoredDraft(session, recoveryState) {
  const draft = recoveryState?.status === "restored"
    ? recoveryState.draft
    : null;

  return draft?.detectionId === session.detectionId
    && draft?.actor === session.actor
    && Number.isInteger(draft?.version)
    && draft.version > 0;
}

export function getSubmissionVersion(session, recoveryState) {
  return isMatchingRestoredDraft(session, recoveryState)
    ? recoveryState.draft.version
    : session.version;
}

export function persistDraftAtBoundary(store, context, values, version) {
  try {
    const draft = store.save(context, values, version);
    return Object.freeze({
      status: draft ? "saved" : "discarded",
      draft,
      errorMessage: null,
    });
  } catch (_error) {
    return Object.freeze({
      status: "failed",
      draft: null,
      errorMessage: DRAFT_STORAGE_ERROR_MESSAGE,
    });
  }
}

export function lockSubmissionForm(form) {
  form.setAttribute("aria-busy", "true");

  for (const control of form.elements) {
    control.disabled = true;
  }
}

export function createAsyncViewGuard(getSelectedId) {
  let revision = 0;

  function tokenFor(detectionId) {
    return Object.freeze({ detectionId, revision });
  }

  return Object.freeze({
    begin(detectionId) {
      revision += 1;
      return tokenFor(detectionId);
    },
    capture(detectionId) {
      return tokenFor(detectionId);
    },
    isCurrent(token) {
      return token?.revision === revision
        && token.detectionId === getSelectedId();
    },
  });
}
