// Verifies sensor_health/script.js can execute in the same global lexical scope
// as an admin page's inline `const pageState = createAdminPageState()` block.
// A duplicate top-level `const` in classic scripts aborts the entire file, which
// silently breaks the Sensor Health table and the device detail workflow.

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const scriptPath = path.join(here, "script.js");

function createDomStub() {
  const noopElement = () => ({
    classList: { add() {}, remove() {}, contains: () => false },
    addEventListener() {},
    removeEventListener() {},
    setAttribute() {},
    removeAttribute() {},
    appendChild() {},
    querySelector: () => null,
    querySelectorAll: () => [],
    style: {},
    innerHTML: "",
    textContent: "",
    value: "",
    hidden: false,
    focus() {},
  });

  const document = {
    readyState: "loading",
    documentElement: {
      getAttribute: () => null,
      setAttribute() {},
      removeAttribute() {},
    },
    body: { classList: { add() {}, remove() {}, toggle() {} } },
    title: "",
    scripts: [],
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
    createElement: noopElement,
    addEventListener() {},
  };

  const context = {
    document,
    localStorage: { getItem: () => null, setItem() {} },
    location: { search: "", href: "" },
    setTimeout,
    clearTimeout,
    setInterval: () => 0,
    clearInterval() {},
    fetch: async () => ({
      ok: true,
      status: 200,
      headers: { get: () => "application/json" },
      json: async () => ({ items: [] }),
      text: async () => "",
    }),
    AbortController,
    URLSearchParams,
    console: { log() {}, warn() {}, error() {} },
    confirm: () => false,
  };

  context.window = context;
  context.globalThis = context;
  return context;
}

test("script.js runs after a page already declared const pageState", async () => {
  const source = await readFile(scriptPath, "utf8");
  const context = vm.createContext(createDomStub());

  // Mirrors the inline block present on every sensor_health admin page.
  vm.runInContext("const pageState = { resetPageState() {} };", context);

  assert.doesNotThrow(() => {
    vm.runInContext(source, context, { filename: "script.js" });
  });

  // The dashboard and device detail entry points must be reachable.
  assert.equal(
    vm.runInContext("typeof loadSensorHealthPage", context),
    "function"
  );
  assert.equal(
    vm.runInContext("typeof loadDeviceDetailPage", context),
    "function"
  );
});

// `Number(null)` is 0, so a missing reading previously rendered as a real value:
// "0%" battery, "0 °C", and a "Just now" heartbeat on a device that never reported.
test("missing readings render as em dash, not zero", async () => {
  const source = await readFile(scriptPath, "utf8");
  const context = vm.createContext(createDomStub());
  vm.runInContext(source, context, { filename: "script.js" });

  const evaluate = (expression) => vm.runInContext(expression, context);

  for (const missing of ["null", "undefined", '""']) {
    assert.equal(evaluate(`formatPercent(${missing})`), "—", `formatPercent(${missing})`);
    assert.equal(evaluate(`formatUptime(${missing})`), "—", `formatUptime(${missing})`);
    assert.equal(evaluate(`formatTemperature(${missing})`), "—", `formatTemperature(${missing})`);
    assert.match(evaluate(`formatBattery(${missing})`), /—/, `formatBattery(${missing})`);
  }

  // A real zero reading must still be shown rather than hidden.
  assert.equal(evaluate("formatPercent(0)"), "0%");
  assert.equal(evaluate("formatTemperature(0)"), "0 °C");
  assert.equal(evaluate("formatBattery(75)").includes("75%"), true);
});

test("a device that never reported is Unknown, not offline or 'Just now'", async () => {
  const source = await readFile(scriptPath, "utf8");
  const context = vm.createContext(createDomStub());
  vm.runInContext(source, context, { filename: "script.js" });

  const evaluate = (expression) => vm.runInContext(expression, context);

  assert.equal(
    evaluate("formatLastSeen({ lastSeen: null, lastSeenMinutesAgo: null })"),
    "Never reported"
  );
  assert.equal(evaluate("formatLastAudio({ lastAudioMinutesAgo: null, lastAudioTs: null })"), "—");
  assert.match(evaluate('pillHtml("Unknown", null)'), /pill-muted/);
  assert.match(evaluate('pillHtml("Offline", null)'), /pill-danger/);

  // A genuinely fresh heartbeat still reads as current.
  assert.equal(evaluate("formatLastSeen({ lastSeenMinutesAgo: 0 })"), "Just now");
});

test("device location uses the bundled map, never a cross-origin iframe", async () => {
  const source = await readFile(scriptPath, "utf8");
  assert.ok(
    !/<iframe/i.test(source),
    "the HMI Content-Security-Policy restricts frames to 'self', so an embedded map iframe is blocked"
  );
  assert.ok(source.includes("ol.source.OSM"), "expected the bundled OpenLayers map");
});

test("script.js does not redeclare pageState at top level", async () => {
  const source = await readFile(scriptPath, "utf8");
  assert.ok(
    !/^\s*(const|let|var)\s+pageState\b/m.test(source),
    "top-level `pageState` declaration would collide with the page's inline script"
  );
});
