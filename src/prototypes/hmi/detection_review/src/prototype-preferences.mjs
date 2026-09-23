import { ROLES } from "./role-workspace.mjs";

export const PROTOTYPE_ROLE_STORAGE_KEY = "echo-detection-review:active-role:v1";
export const PROTOTYPE_PREFERENCE_STORAGE_ERROR_MESSAGE =
  "Active review role preferences could not be accessed in this browser.";

export class PrototypePreferenceStorageError extends Error {
  constructor(options) {
    super(PROTOTYPE_PREFERENCE_STORAGE_ERROR_MESSAGE, options);
    this.name = "PrototypePreferenceStorageError";
    this.userMessage = PROTOTYPE_PREFERENCE_STORAGE_ERROR_MESSAGE;
  }
}

function storageOperation(operation) {
  try {
    return operation();
  } catch (error) {
    throw new PrototypePreferenceStorageError({ cause: error });
  }
}

function assertRole(role) {
  if (!ROLES.includes(role)) {
    throw new TypeError("Choose a supported review role.");
  }
}

export function createPrototypePreferenceStore(storage) {
  return Object.freeze({
    loadRole() {
      const savedRole = storageOperation(() => storage.getItem(PROTOTYPE_ROLE_STORAGE_KEY));
      return ROLES.includes(savedRole) ? savedRole : "reviewer-1";
    },

    saveRole(role) {
      assertRole(role);
      storageOperation(() => storage.setItem(PROTOTYPE_ROLE_STORAGE_KEY, role));
      return role;
    },

    reset() {
      storageOperation(() => storage.removeItem(PROTOTYPE_ROLE_STORAGE_KEY));
    },
  });
}

export function createBrowserPrototypePreferenceStore(windowObject) {
  let storage;
  try {
    storage = windowObject.localStorage;
  } catch (error) {
    throw new PrototypePreferenceStorageError({ cause: error });
  }

  return createPrototypePreferenceStore(storage);
}
