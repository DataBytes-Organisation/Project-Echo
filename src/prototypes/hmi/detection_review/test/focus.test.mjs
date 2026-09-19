import test from "node:test";
import assert from "node:assert/strict";

import { restoreQueueItemFocus } from "../src/focus.mjs";

test("restores focus to the re-rendered selected queue item", () => {
  const focusCalls = [];
  const queueItems = ["det-echo-001", "det-echo-002"].map(id => ({
    dataset: { detectionId: id },
    focus() {
      focusCalls.push(id);
    },
  }));
  const root = {
    querySelectorAll(selector) {
      assert.equal(selector, "[data-detection-id]");
      return queueItems;
    },
  };

  assert.equal(restoreQueueItemFocus(root, "det-echo-002"), true);
  assert.deepEqual(focusCalls, ["det-echo-002"]);
});

test("does not move focus when the selected queue item is absent", () => {
  const root = {
    querySelectorAll() {
      return [];
    },
  };

  assert.equal(restoreQueueItemFocus(root, "det-missing"), false);
});
