import test from "node:test";
import assert from "node:assert/strict";

import {
  PrototypePreferenceStorageError,
  createBrowserPrototypePreferenceStore,
  createPrototypePreferenceStore,
} from "../src/prototype-preferences.mjs";

const ROLE_KEY = "echo-detection-review:active-role:v1";

function createMemoryStorage(initial = {}) {
  const entries = new Map(Object.entries(initial));
  return {
    entries,
    getItem(key) {
      return entries.has(key) ? entries.get(key) : null;
    },
    setItem(key, value) {
      entries.set(key, String(value));
    },
    removeItem(key) {
      entries.delete(key);
    },
  };
}

test("defaults to Reviewer 1 and persists only supported roles across store instances", () => {
  const storage = createMemoryStorage();
  const firstStore = createPrototypePreferenceStore(storage);

  assert.equal(firstStore.loadRole(), "reviewer-1");
  assert.equal(firstStore.saveRole("reviewer-2"), "reviewer-2");
  assert.equal(createPrototypePreferenceStore(storage).loadRole(), "reviewer-2");
  assert.throws(() => firstStore.saveRole("observer"), TypeError);
});

test("falls back to Reviewer 1 for an invalid saved role and resets only the role preference", () => {
  const storage = createMemoryStorage({
    [ROLE_KEY]: "observer",
    unrelated: "keep",
  });
  const store = createPrototypePreferenceStore(storage);

  assert.equal(store.loadRole(), "reviewer-1");
  store.saveRole("adjudicator");
  store.reset();

  assert.equal(storage.getItem(ROLE_KEY), null);
  assert.equal(store.loadRole(), "reviewer-1");
  assert.equal(storage.getItem("unrelated"), "keep");
});

test("wraps storage failures with controlled preference copy", () => {
  const store = createPrototypePreferenceStore({
    getItem() {
      throw new Error("browser profile quota details");
    },
    setItem() {},
    removeItem() {},
  });

  assert.throws(
    () => store.loadRole(),
    error => error instanceof PrototypePreferenceStorageError
      && /could not be accessed/i.test(error.userMessage)
      && !error.userMessage.includes("quota"),
  );
});

test("browser preference factory catches a blocked localStorage getter", () => {
  const blockedWindow = {};
  Object.defineProperty(blockedWindow, "localStorage", {
    get() {
      throw new Error("SecurityError browser profile details");
    },
  });

  assert.throws(
    () => createBrowserPrototypePreferenceStore(blockedWindow),
    error => error instanceof PrototypePreferenceStorageError
      && /could not be accessed/i.test(error.userMessage)
      && !error.userMessage.includes("profile"),
  );
});
