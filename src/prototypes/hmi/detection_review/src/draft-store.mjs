const DEFAULT_PREFIX = "echo-review-draft";
export const DRAFT_STORAGE_ERROR_MESSAGE = "Saved review drafts could not be accessed in this browser.";
const VALUE_FIELDS = Object.freeze([
  "decision",
  "correctedSpecies",
  "reason",
  "resolutionReason",
]);

export class DraftStorageError extends Error {
  constructor(options) {
    super(DRAFT_STORAGE_ERROR_MESSAGE, options);
    this.name = "DraftStorageError";
    this.userMessage = DRAFT_STORAGE_ERROR_MESSAGE;
  }
}

export function createBrowserReviewDraftStore(windowObject, options) {
  let storage;

  try {
    storage = windowObject.localStorage;
  } catch (error) {
    throw new DraftStorageError({ cause: error });
  }

  return createReviewDraftStore(storage, options);
}

function requiredIdentityText(value, field) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new TypeError(`${field} is required to identify a review draft.`);
  }

  return value.trim();
}

function normalizeContext(context) {
  return Object.freeze({
    detectionId: requiredIdentityText(context?.detectionId, "detectionId"),
    actor: requiredIdentityText(context?.actor, "actor"),
    kind: requiredIdentityText(context?.kind, "kind"),
  });
}

function normalizeValues(values = {}) {
  return Object.freeze(Object.fromEntries(
    VALUE_FIELDS.map(field => [
      field,
      typeof values[field] === "string" ? values[field] : "",
    ]),
  ));
}

function hasMeaningfulValues(values) {
  return VALUE_FIELDS.some(field => values[field].trim() !== "");
}

function draftKey(prefix, context) {
  return [prefix, context.kind, context.actor, context.detectionId]
    .map(encodeURIComponent)
    .join(":");
}

function freezeDraft(draft) {
  return Object.freeze({
    detectionId: draft.detectionId,
    actor: draft.actor,
    kind: draft.kind,
    values: normalizeValues(draft.values),
    version: draft.version,
    savedAt: draft.savedAt,
  });
}

function isValidDraft(candidate, context) {
  return candidate !== null
    && typeof candidate === "object"
    && candidate.detectionId === context.detectionId
    && candidate.actor === context.actor
    && candidate.kind === context.kind
    && Number.isInteger(candidate.version)
    && candidate.version > 0
    && typeof candidate.savedAt === "string"
    && candidate.savedAt.trim() !== ""
    && candidate.values !== null
    && typeof candidate.values === "object"
    && VALUE_FIELDS.every(field => typeof candidate.values[field] === "string");
}

function storageOperation(operation) {
  try {
    return operation();
  } catch (error) {
    throw new DraftStorageError({ cause: error });
  }
}

export function createReviewDraftStore(storage, {
  now = () => new Date().toISOString(),
  prefix = DEFAULT_PREFIX,
} = {}) {
  return Object.freeze({
    save(contextInput, valuesInput, version) {
      const context = normalizeContext(contextInput);
      const values = normalizeValues(valuesInput);
      const key = draftKey(prefix, context);

      if (!hasMeaningfulValues(values)) {
        storageOperation(() => storage.removeItem(key));
        return null;
      }

      if (!Number.isInteger(version) || version < 1) {
        throw new TypeError("A positive case version is required to save a review draft.");
      }

      const draft = freezeDraft({
        ...context,
        values,
        version,
        savedAt: now(),
      });
      storageOperation(() => storage.setItem(key, JSON.stringify(draft)));
      return draft;
    },

    load(contextInput) {
      const context = normalizeContext(contextInput);
      const key = draftKey(prefix, context);
      const raw = storageOperation(() => storage.getItem(key));

      if (raw === null) {
        return null;
      }

      let candidate;
      try {
        candidate = JSON.parse(raw);
      } catch (_error) {
        storageOperation(() => storage.removeItem(key));
        return null;
      }

      if (!isValidDraft(candidate, context)) {
        storageOperation(() => storage.removeItem(key));
        return null;
      }

      return freezeDraft(candidate);
    },

    discard(contextInput) {
      const context = normalizeContext(contextInput);
      const key = draftKey(prefix, context);
      const existed = storageOperation(() => storage.getItem(key)) !== null;
      storageOperation(() => storage.removeItem(key));
      return existed;
    },

    discardAll() {
      const encodedPrefix = `${encodeURIComponent(prefix)}:`;
      const keys = storageOperation(() => Array.from(
        { length: storage.length },
        (_value, index) => storage.key(index),
      ));
      const draftKeys = keys.filter(key => key?.startsWith(encodedPrefix));

      for (const key of draftKeys) {
        storageOperation(() => storage.removeItem(key));
      }

      return draftKeys.length;
    },

    discardIfUnchanged(contextInput, expectedDraft) {
      const context = normalizeContext(contextInput);
      const key = draftKey(prefix, context);
      const raw = storageOperation(() => storage.getItem(key));

      if (raw === null || !isValidDraft(expectedDraft, context)) {
        return false;
      }

      let currentDraft;
      try {
        currentDraft = JSON.parse(raw);
      } catch (_error) {
        return false;
      }

      if (!isValidDraft(currentDraft, context)
        || JSON.stringify(freezeDraft(currentDraft))
          !== JSON.stringify(freezeDraft(expectedDraft))) {
        return false;
      }

      storageOperation(() => storage.removeItem(key));
      return true;
    },
  });
}
