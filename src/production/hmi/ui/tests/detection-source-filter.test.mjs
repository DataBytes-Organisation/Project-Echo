import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

class Element {
  constructor() {
    this.children = [];
    this.style = {};
    this.listeners = {};
    this.classList = { add() {}, remove() {} };
    this.textContent = "";
  }
  appendChild(el) { this.children.push(el); return el; }
  setAttribute(key, value) { this[key] = value; }
  addEventListener(key, fn) { this.listeners[key] = fn; }
  querySelector() { return null; }
  remove() {}
}

globalThis.window = { axios: { create() { return { async get() { return { data: [] }; } }; } }, addEventListener() {} };
globalThis.document = {
  readyState: "loading", addEventListener() {}, head: new Element(), body: new Element(),
  createElement: () => new Element(), getElementById: () => null, querySelector: () => null,
};
globalThis.requestAnimationFrame = () => {};
globalThis.getComputedStyle = () => ({ position: "relative" });

class Source {
  constructor() { this.features = []; }
  clear() { this.features = []; }
  addFeature(feature) { this.features.push(feature); }
  getFeatures() { return this.features; }
  getExtent() { return [0, 0, 0, 0]; }
}
class Layer {
  constructor(options = {}) {
    this.source = options.source || new Source();
    this.visible = true;
    this.style = options.style;
  }
  getSource() { return this.source; }
  set(key, value) { this[key] = value; }
  setZIndex(zIndex) { this.zIndex = zIndex; }
  setVisible(visible) { this.visible = visible; }
}
class Feature {
  constructor(properties) { this.properties = properties; this.style = null; }
  setId(id) { this.id = id; }
  get(key) { return this.properties[key]; }
  setStyle(style) { this.style = style; }
}
class Style {
  constructor(options) { this.options = options; }
}

globalThis.ol = {
  source: { Vector: Source },
  layer: { Vector: Layer },
  Feature,
  geom: { Point: class { constructor(coords) { this.coords = coords; } } },
  proj: { fromLonLat: (coords) => coords },
  style: {
    Style,
    Circle: Style,
    Fill: Style,
    Stroke: Style,
    Icon: Style,
    Text: Style,
  },
  control: { Control: class { constructor(options) { this.element = options.element; } } },
};

const filter = await import("../public/js/detection-source-filter.js");
const detections = await import("../public/js/real-detections.js");

function makeHmi() {
  const status = new Element();
  const layers = {
    normal_bird: new Layer(),
    endangered_mammal: new Layer(),
    normal_bird_truth: new Layer(),
    endangered_mammal_truth: new Layer(),
  };
  const real = new Layer();
  return {
    layers,
    realDetectionLayer: real,
    realDetectionStatus: new Element(),
    detectionSourceStatus: status,
    vocalizationEvents: [
      { sourceType: "simulator", eventId: "sim-1" },
      { sourceType: "simulator", eventId: "sim-2" },
    ],
    basemap: {
      layers: [],
      controls: [],
      fits: [],
      getView() { return { fit: (...args) => this.fits.push(args) }; },
      addLayer(layer) { this.layers.push(layer); },
      addControl(control) { this.controls.push(control); },
    },
  };
}

test("source filter labels keep simulator contract and human-readable names", () => {
  assert.equal(filter.formatDetectionSourceLabel("simulator"), "Simulated");
  assert.equal(filter.formatDetectionSourceLabel("real"), "Real-device");
  assert.equal(filter.formatDetectionSourceLabel(undefined), "Simulated");
  assert.equal(filter.formatDetectionSourceLabel("bogus"), "Unknown source");
  assert.equal(filter.normalizeDetectionSource("simulated"), "unknown");
});

test("unknown sourceType never matches All/Simulated/Real filters", () => {
  for (const mode of ["all", "simulator", "real"]) {
    assert.equal(filter.detectionMatchesSourceFilter("bogus", mode), false);
    assert.equal(filter.detectionMatchesSourceFilter("simulated", mode), false);
  }
  assert.equal(filter.detectionMatchesSourceFilter("simulator", "simulator"), true);
  assert.equal(filter.detectionMatchesSourceFilter("real", "real"), true);
  assert.equal(filter.detectionMatchesSourceFilter(null, "all"), true);
});

test("each source filter toggles existing layers without recreating them", () => {
  const hmi = makeHmi();
  layersSeed(hmi);

  const first = filter.applyDetectionSourceFilter(hmi, "simulator");
  assert.equal(first.filter, "simulator");
  assert.equal(hmi.layers.normal_bird.visible, true);
  assert.equal(hmi.layers.normal_bird_truth.visible, true);
  assert.equal(hmi.realDetectionLayer.visible, false);
  assert.match(hmi.detectionSourceStatus.textContent, /simulated/i);

  const second = filter.applyDetectionSourceFilter(hmi, "real");
  assert.equal(second.filter, "real");
  assert.equal(hmi.layers.normal_bird.visible, false);
  assert.equal(hmi.layers.normal_bird_truth.visible, false);
  assert.equal(hmi.realDetectionLayer.visible, true);

  filter.applyDetectionSourceFilter(hmi, "all");
  assert.equal(hmi.layers.normal_bird.visible, true);
  assert.equal(hmi.layers.normal_bird_truth.visible, true);
  assert.equal(hmi.realDetectionLayer.visible, true);

  // Repeated filtering reuses the same layer objects.
  assert.equal(hmi.basemap.layers.length, 0);
  assert.equal(Object.keys(hmi.layers).length, 4);
});

function layersSeed(hmi) {
  hmi.layers.normal_bird.getSource().addFeature(new Feature({ sourceType: "simulator" }));
  hmi.layers.normal_bird.getSource().addFeature(new Feature({ sourceType: "simulator" }));
  hmi.realDetectionLayer.getSource().addFeature(new Feature({ sourceType: "real" }));
}

test("repeated filtering does not duplicate markers or invent demo data", () => {
  const hmi = makeHmi();
  layersSeed(hmi);
  for (let i = 0; i < 5; i++) {
    filter.applyDetectionSourceFilter(hmi, "simulator");
    filter.applyDetectionSourceFilter(hmi, "real");
    filter.applyDetectionSourceFilter(hmi, "all");
  }
  assert.equal(hmi.layers.normal_bird.getSource().getFeatures().length, 2);
  assert.equal(hmi.realDetectionLayer.getSource().getFeatures().length, 1);
  assert.doesNotMatch(hmi.detectionSourceStatus.textContent, /demo|sample|hardcoded|fallback/i);
});

test("empty filter results show a clear empty state", () => {
  const hmi = makeHmi();
  hmi.vocalizationEvents = [];
  const result = filter.applyDetectionSourceFilter(hmi, "simulator");
  assert.match(result.message, /No simulated detections/i);
  assert.doesNotMatch(result.message, /demo|sample|Magpie|hardcoded/i);

  const realEmpty = filter.applyDetectionSourceFilter(hmi, "real");
  assert.match(realEmpty.message, /No real-device detections/i);
});

test("loading and backend failure states remain visible while filtering", () => {
  const hmi = makeHmi();
  layersSeed(hmi);
  const loading = filter.applyDetectionSourceFilter(hmi, "all", { loading: true });
  assert.match(loading.message, /Loading detections/i);
  const failed = filter.applyDetectionSourceFilter(hmi, "real", {
    errorMessage: "Unable to load detections. Please try again.",
  });
  assert.match(failed.message, /Unable to load detections/i);
  const ok = filter.applyDetectionSourceFilter(hmi, "all");
  assert.match(ok.message, /detection/i);
});

test("real and simulated marker styles differ by more than colour", () => {
  const realStyle = filter.buildRealDetectionStyle();
  const simStyle = filter.buildSimulatorVocalizationStyle("./icon.png");
  assert.ok(Array.isArray(realStyle) && realStyle.length >= 2);
  assert.ok(Array.isArray(simStyle) && simStyle.length >= 2);
  const realText = realStyle.find((style) => style.options?.text)?.options.text.options.text;
  const simText = simStyle.find((style) => style.options?.text)?.options.text.options.text;
  assert.equal(realText, "REAL");
  assert.equal(simText, "SIM");
  assert.ok(realStyle.some((style) => style.options?.image));
  assert.ok(simStyle.some((style) => style.options?.image));
});

test("real-device filter hides simulated layers while keeping the real layer", () => {
  const hmi = makeHmi();
  layersSeed(hmi);
  hmi.detectionSourceFilter = "real";
  const applied = filter.applyDetectionSourceFilter(hmi, "real");
  assert.equal(hmi.realDetectionLayer.visible, true);
  assert.equal(hmi.layers.normal_bird.visible, false);
  assert.equal(hmi.layers.normal_bird_truth.visible, false);
  assert.match(applied.message, /real-device/i);
  assert.equal(
    filter.buildRealDetectionStyle()[1].options.text.options.text,
    "REAL",
  );
  assert.equal(typeof detections.loadRealDetections, "function");
});

test("species filter state still gates simulated layers under All", () => {
  const hmi = makeHmi();
  layersSeed(hmi);
  hmi.speciesFilterState = ["_normal", "_bird", "normal", "bird"];
  filter.applyDetectionSourceFilter(hmi, "all");
  assert.equal(hmi.layers.normal_bird.visible, true);
  assert.equal(hmi.layers.endangered_mammal.visible, false);
  assert.equal(hmi.layers.normal_bird_truth.visible, true);
  assert.equal(hmi.layers.endangered_mammal_truth.visible, false);
  assert.equal(hmi.realDetectionLayer.visible, true);
});

test("detail labels map contract values to Simulated and Real-device", () => {
  assert.equal(filter.formatDetectionSourceLabel("simulator"), "Simulated");
  assert.equal(filter.formatDetectionSourceLabel("real"), "Real-device");
});

test("truth layers follow production checkbox state (underscore ids)", () => {
  // Production speciesFilterState holds checkbox ids (_normal, _bird, ...),
  // not bare status/type names. Truth layers must follow it like vocalization
  // layers do, or movement markers never render and the toggle looks dead.
  const hmi = makeHmi();
  layersSeed(hmi);
  hmi.speciesFilterState = [
    "_endangered", "_vulnerable", "_near-threatened", "_normal", "_invasive",
    "_mammal", "_bird", "_amphibian", "_reptile", "_insect",
  ];
  filter.applyDetectionSourceFilter(hmi, "simulator");
  assert.equal(hmi.layers.normal_bird_truth.visible, true);
  assert.equal(hmi.layers.normal_bird.visible, true);
  filter.applyDetectionSourceFilter(hmi, "real");
  assert.equal(hmi.layers.normal_bird_truth.visible, false);
  assert.equal(hmi.layers.normal_bird.visible, false);
  filter.applyDetectionSourceFilter(hmi, "all");
  assert.equal(hmi.layers.normal_bird_truth.visible, true);
});

test("/map offers All/Simulated/Real-device as segmented buttons", () => {
  const html = fs.readFileSync(new URL("../public/index.html", import.meta.url), "utf8");
  for (const value of ["all", "simulator", "real"]) {
    assert.match(html, new RegExp(`data-detection-source="${value}"`));
  }
  assert.match(html, /data-detection-source="all"[^>]*aria-pressed="true"/);
  assert.doesNotMatch(html, /<input[^>]*name="detectionSource"/);
  assert.match(html, /id="detection-source-status"/);
});
