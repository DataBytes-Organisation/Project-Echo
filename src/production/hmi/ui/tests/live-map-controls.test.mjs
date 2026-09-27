import test from "node:test";
import assert from "node:assert/strict";

// Minimal DOM/fetch/ol stubs mirroring tests/real-detections.test.mjs.
class Element {
  constructor(tagName = "div") {
    this.tagName = tagName;
    this.children = [];
    this.style = {};
    this.listeners = {};
    this.dataset = {};
    this.attributes = {};
    this.textContent = "";
    this.innerHTML = "";
    this.checked = false;
    this.value = "";
    this.scrollHeight = 120;
    this.classList = { add() {}, remove() {} };
  }
  appendChild(el) { this.children.push(el); return el; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  getAttribute(key) { return this.attributes[key] ?? null; }
  addEventListener(type, fn) { this.listeners[type] = fn; }
  click() { if (typeof this.listeners.click === "function") this.listeners.click(); }
  change() { if (typeof this.listeners.change === "function") this.listeners.change(); }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  remove() {}
  insertAdjacentElement() {}
}

const registry = {};
for (const id of ["pause-resume-btn", "manual-refresh-btn", "last-update-label", "real-detection-status", "liveModeToggle", "event-type-filter", "species-filter"]) {
  registry[id] = new Element(id === "manual-refresh-btn" || id === "pause-resume-btn" ? "button" : "div");
  registry[id].id = id;
}
registry["pause-resume-btn"].textContent = "Pause";
registry["last-update-label"].textContent = "Last update: —";
registry["liveModeToggle"].checked = true;
const pastDataContainer = new Element("div");
pastDataContainer.style.maxHeight = "0";

globalThis.document = {
  readyState: "loading",
  addEventListener() {},
  dispatchEvent() {},
  head: new Element("head"),
  body: new Element("body"),
  createElement: (tag) => new Element(tag),
  getElementById: (id) => registry[id] || null,
  querySelector: (selector) => selector === ".past-data-container" ? pastDataContainer : null,
};
globalThis.requestAnimationFrame = () => {};
globalThis.getComputedStyle = () => ({ position: "relative" });

let fetchUrls = [];
globalThis.fetch = async (url) => {
  fetchUrls.push(String(url));
  if (String(url).includes("/mqtt/latest-events")) {
    return { ok: true, json: async () => ({ events: [] }) };
  }
  if (String(url).includes("/mqtt/connection-state")) {
    return { ok: true, json: async () => ({ state: "connected" }) };
  }
  return { ok: true, json: async () => ({}) };
};

let axiosGets = [];
globalThis.window = {
  axios: { create() {
    return { async get(url, options) {
      axiosGets.push(url);
      return { data: [] };
    } };
  } },
  addEventListener() {},
};

class Source {
  constructor() { this.features = []; }
  clear() { this.features = []; }
  addFeature(feature) { this.features.push(feature); }
  changed() {}
  getFeatures() { return this.features; }
  getExtent() { return [0, 0, 0, 0]; }
}
class Layer {
  constructor(options) { this.source = options.source; this.style = options.style; }
  getSource() { return this.source; }
  set() {}
  setZIndex(z) { this.zIndex = z; }
  setVisible() {}
  changed() {}
}
class Feature {
  constructor(properties) { this.properties = properties; }
  setId(id) { this.id = id; }
}
globalThis.ol = {
  source: { Vector: Source },
  layer: { Vector: Layer },
  Feature,
  geom: { Point: class { constructor(coords) { this.coords = coords; } } },
  proj: { fromLonLat: (coords) => coords },
  style: { Style: class {}, Circle: class {}, Fill: class {}, Stroke: class {}, Icon: class {}, Text: class {} },
  control: { Control: class { constructor(options) { this.element = options.element; } } },
};

const hmi = await import("../public/js/HMI.js");
const detections = await import("../public/js/real-detections.js");

// One shared state: setupLiveMapControls binds once (as in production),
// so every button-driven test acts on this object.
const shared = {
  liveMode: true,
  liveWindow: 60000,
  simUpdateDelay: 10000,
  timeOffset: 0,
  currentTime: 1750000000,
  liveEventCutoff: 1750000000,
  movementEvents: {},
  vocalizationEvents: [],
  layers: {},
  basemap: {
    layers: [],
    controls: [],
    interactions: [],
    fits: [],
    getView() { return { fit: (...args) => this.fits.push(args) }; },
    addLayer(layer) { this.layers.push(layer); },
    addControl(control) { this.controls.push(control); },
    addInteraction(interaction) { this.interactions.push(interaction); },
  },
};
hmi.setupLiveMapControls(shared);

function resetLive() {
  fetchUrls = [];
  axiosGets = [];
  hmi.setLivePaused(shared, false);
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 30));

test("frozen predicate covers paused flag and history mode", () => {
  assert.equal(typeof hmi.isLiveUpdatesFrozen, "function");
  resetLive();
  assert.equal(hmi.isLiveUpdatesFrozen(shared), false);
  hmi.setLivePaused(shared, true);
  assert.equal(hmi.isLiveUpdatesFrozen(shared), true);
  hmi.setLivePaused(shared, false);
  shared.liveMode = false;
  assert.equal(hmi.isLiveUpdatesFrozen(shared), true);
  resetLive();
});

test("pause toggles to Resume with aria-pressed and a Paused label", () => {
  resetLive();
  assert.equal(registry["pause-resume-btn"].textContent, "Pause");

  registry["pause-resume-btn"].click();
  assert.equal(hmi.isMqttPollingPaused(), true);
  assert.equal(registry["pause-resume-btn"].textContent, "Resume");
  assert.equal(registry["pause-resume-btn"].getAttribute("aria-pressed"), "true");
  assert.match(registry["last-update-label"].textContent, /paused/i);

  registry["pause-resume-btn"].click();
  assert.equal(hmi.isMqttPollingPaused(), false);
  assert.equal(registry["pause-resume-btn"].textContent, "Pause");
  assert.equal(registry["pause-resume-btn"].getAttribute("aria-pressed"), "false");
});

test("pause unchecks the toggle but leaves the picker closed", () => {
  resetLive();
  assert.equal(registry["liveModeToggle"].checked, true);

  registry["pause-resume-btn"].click();
  assert.equal(registry["liveModeToggle"].checked, false);
  assert.equal(pastDataContainer.style.maxHeight, "0");
  assert.equal(registry["pause-resume-btn"].textContent, "Resume");

  resetLive();
  assert.equal(registry["liveModeToggle"].checked, true);
});

test("checking the toggle while paused resumes live", () => {
  resetLive();
  registry["pause-resume-btn"].click();
  assert.equal(hmi.isLiveUpdatesFrozen(shared), true);

  registry["liveModeToggle"].checked = true;
  registry["liveModeToggle"].change();
  assert.equal(hmi.isLiveUpdatesFrozen(shared), false);
  assert.equal(registry["pause-resume-btn"].textContent, "Pause");
  assert.equal(registry["liveModeToggle"].checked, true);
  resetLive();
});

test("pause freezes the sim-window loop until resumed", async () => {
  resetLive();
  registry["pause-resume-btn"].click();
  await hmi.runLiveWindowUpdate(shared);
  assert.ok(!axiosGets.some((url) => url.includes("/movement_time")), "no sim movement fetch while paused");
  assert.ok(!axiosGets.some((url) => url.includes("/events_time")), "no sim vocalization fetch while paused");

  registry["pause-resume-btn"].click();
  await hmi.runLiveWindowUpdate(shared);
  assert.ok(axiosGets.some((url) => url.includes("/movement_time")), "sim movement fetch resumes");
  assert.ok(axiosGets.some((url) => url.includes("/events_time")), "sim vocalization fetch resumes");
  resetLive();
});

test("unchecking Live Mode freezes the MQTT feed and opens past data", async () => {
  resetLive();
  registry["liveModeToggle"].checked = false;
  registry["liveModeToggle"].change();
  assert.equal(shared.liveMode, false);
  assert.equal(hmi.isLiveUpdatesFrozen(shared), true);
  assert.equal(registry["pause-resume-btn"].textContent, "Resume");
  assert.notEqual(pastDataContainer.style.maxHeight, "0");

  fetchUrls = [];
  await hmi.pollMqttLatestEvents(shared);
  assert.ok(!fetchUrls.some((url) => url.includes("/mqtt/latest-events")), "no MQTT fetch in history mode");
  resetLive();
});

test("checking Live Mode again clears a stale paused flag", () => {
  resetLive();
  registry["pause-resume-btn"].click();
  registry["liveModeToggle"].checked = false;
  registry["liveModeToggle"].change();
  registry["liveModeToggle"].checked = true;
  registry["liveModeToggle"].change();
  assert.equal(hmi.isMqttPollingPaused(), false);
  assert.equal(shared.liveMode, true);
  assert.equal(hmi.isLiveUpdatesFrozen(shared), false);
  assert.equal(registry["pause-resume-btn"].textContent, "Pause");
  resetLive();
});

test("resume from history via the Pause button returns to live", () => {
  resetLive();
  registry["liveModeToggle"].checked = false;
  registry["liveModeToggle"].change();
  assert.equal(registry["pause-resume-btn"].textContent, "Resume");

  registry["pause-resume-btn"].click();
  assert.equal(shared.liveMode, true);
  assert.equal(registry["liveModeToggle"].checked, true);
  assert.equal(pastDataContainer.style.maxHeight, "0");
  assert.equal(hmi.isLiveUpdatesFrozen(shared), false);
  resetLive();
});

test("manual Refresh while paused reloads every live source", async () => {
  resetLive();
  registry["pause-resume-btn"].click();
  fetchUrls = [];
  axiosGets = [];
  registry["manual-refresh-btn"].click();
  await tick();

  assert.ok(fetchUrls.some((url) => url.includes("/mqtt/latest-events")), "paused refresh still polls the live feed");
  assert.ok(fetchUrls.some((url) => url.includes("/mqtt/connection-state")), "paused refresh still checks connection");
  assert.ok(axiosGets.some((url) => url.includes("/movement_time")), "paused refresh one-shots the sim movement window");
  assert.ok(axiosGets.some((url) => url.includes("/events_time")), "paused refresh one-shots the sim vocalization window");
  assert.ok(axiosGets.includes("/api/detections"), "single Refresh also reloads real-device detections");
  assert.match(registry["last-update-label"].textContent, /last update:/i);
  resetLive();
});

test("manual Refresh in history mode leaves live sources alone", async () => {
  resetLive();
  registry["liveModeToggle"].checked = false;
  registry["liveModeToggle"].change();
  fetchUrls = [];
  axiosGets = [];
  registry["manual-refresh-btn"].click();
  await tick();

  assert.ok(!fetchUrls.some((url) => url.includes("/mqtt/latest-events")), "no live feed fetch in history mode");
  assert.ok(!axiosGets.some((url) => url.includes("/movement_time")), "no sim movement fetch in history mode");
  assert.ok(!axiosGets.some((url) => url.includes("/events_time")), "no sim vocalization fetch in history mode");
  assert.ok(axiosGets.includes("/api/detections"), "real-device detections still reload in history mode");
  resetLive();
});

test("real-device status renders in the top panel with no corner map control", async () => {
  resetLive();
  const layersBefore = shared.basemap.layers.length;
  const controlsBefore = shared.basemap.controls.length;
  await detections.loadRealDetections(shared);
  await detections.loadRealDetections(shared);
  assert.equal(shared.basemap.controls.length, controlsBefore);
  assert.equal(shared.realDetectionStatus, registry["real-detection-status"]);
  assert.match(registry["real-detection-status"].textContent, /no real-device detections/i);
  assert.ok(shared.basemap.layers.length >= layersBefore);
  resetLive();
});
