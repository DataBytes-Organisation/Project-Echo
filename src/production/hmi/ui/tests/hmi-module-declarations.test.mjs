import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const hmiPath = path.join(here, "..", "public", "features", "map", "hmi-map.js");

// The browser loads HMI.js as an ES module, where a duplicate top-level
// function/lexical declaration is a SyntaxError that kills the whole module:
// no initialiseHMI (no map, no spinner) and no window.retrieveUserProfile
// (profile shows "undefined"). node --check does not catch this, so scan the
// column-0 declarations directly. `var` is exempt: redeclaring var is legal.
test("HMI.js has no duplicate top-level function/lexical declarations", async () => {
  const source = await readFile(hmiPath, "utf8");
  const seen = new Set();
  const dupes = new Set();
  for (const line of source.split("\n")) {
    const match = line.match(/^(?:export\s+)?(?:async\s+function\s+|function\s+|const\s+|let\s+|class\s+)([A-Za-z_$][A-Za-z0-9_$]*)/);
    if (!match) continue;
    if (seen.has(match[1])) dupes.add(match[1]);
    seen.add(match[1]);
  }
  assert.deepEqual([...dupes], [], `duplicate top-level declarations: ${[...dupes].join(", ")}`);
});
