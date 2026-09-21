import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const indexPath = path.join(here, "..", "public", "pages", "map", "index.html");
const hmiPath = path.join(here, "..", "public", "features", "map", "hmi-map.js");

function extractHmiImport(source) {
  const match = source.match(/import\s*\{([\s\S]*?)\}\s*from\s*["']\/features\/map\/hmi-map\.js["']\s*;/);
  assert.ok(match, 'map entry must import its HMI module from /features/map/hmi-map.js');
  return match[1];
}

function importNames(body) {
  return body.split(",").map((entry) => entry.trim()).filter(Boolean);
}

test("map entry import block parses: each entry is one identifier (no missing comma), no duplicates", async () => {
  const html = await readFile(indexPath, "utf8");
  const names = importNames(extractHmiImport(html));
  assert.ok(names.length > 0, "expected at least one HMI.js import");
  for (const entry of names) {
    assert.match(
      entry,
      /^[A-Za-z_$][A-Za-z0-9_$]*$/,
      `import entry must be a single identifier (missing comma?), got: ${JSON.stringify(entry)}`
    );
  }
  const seen = new Set();
  const dupes = new Set();
  for (const name of names) {
    if (seen.has(name)) dupes.add(name);
    seen.add(name);
  }
  assert.deepEqual([...dupes], [], `duplicate HMI.js imports: ${[...dupes].join(", ")}`);
});

test("map entry imports all resolve to HMI.js exports", async () => {
  const html = await readFile(indexPath, "utf8");
  const hmi = await readFile(hmiPath, "utf8");
  const names = importNames(extractHmiImport(html));
  const exported = new Set(
    [...hmi.matchAll(/export\s+(?:async\s+)?(?:function|const|let|var|class)\s+([A-Za-z_$][A-Za-z0-9_$]*)/g)].map(
      (match) => match[1]
    )
  );
  for (const name of names) {
    assert.ok(exported.has(name), `${name} is imported by index.html but not exported from HMI.js`);
  }
});
