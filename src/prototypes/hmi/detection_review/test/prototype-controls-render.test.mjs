import test from "node:test";
import assert from "node:assert/strict";

import { renderPrototypeControls } from "../src/prototype-controls-render.mjs";

test("renders one labelled role switcher with the active role pressed", () => {
  const html = renderPrototypeControls({
    activeRole: "reviewer-2",
    isPending: false,
    resetStatus: null,
  });

  assert.match(html, /role="group"[^>]*aria-label="Active review role"/);
  assert.match(html, /data-role="reviewer-1"[^>]*aria-pressed="false"/);
  assert.match(html, /data-role="reviewer-2"[^>]*aria-pressed="true"/);
  assert.match(html, /data-role="adjudicator"[^>]*aria-pressed="false"/);
  assert.match(html, /data-prototype-reset/);
  assert.match(html, /class="prototype-controls__roles"/);
  assert.match(html, /class="prototype-controls__reset"/);
  assert.doesNotMatch(html, /Reset submitted reviews, final decisions/);
});

test("disables role and reset controls while a stateful operation is pending", () => {
  const html = renderPrototypeControls({
    activeRole: "reviewer-1",
    isPending: true,
    resetStatus: null,
  });

  assert.equal((html.match(/data-role="[^"]+"[^>]*disabled/g) ?? []).length, 3);
  assert.match(html, /data-prototype-reset[^>]*disabled/);
});

test("announces a completed prototype reset politely", () => {
  const html = renderPrototypeControls({
    activeRole: "reviewer-1",
    isPending: false,
    resetStatus: { status: "reset", message: "Prototype data reset." },
  });

  assert.match(html, /role="status"[^>]*aria-live="polite"/);
  assert.match(html, /Prototype data reset\./);
});
