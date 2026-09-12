import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

let result = { data: [] };
let calls = [];
let config;
let holdAncillary = false;
globalThis.window = { axios: { create(options) {
  config = options;
  return { async get(url) {
    calls.push(url);
    if (holdAncillary && url !== "/api/detections") return new Promise(() => {});
    return await result;
  } };
} }, addEventListener() {} };
class Element {
  constructor() { this.children = []; this.style = {}; this.listeners = {}; this.classList = { add() {}, remove() {} }; }
  appendChild(el) { this.children.push(el); return el; }
  setAttribute(key, value) { this[key] = value; }
  addEventListener(key, fn) { this.listeners[key] = fn; }
  querySelector() { return null; }
  remove() {}
}
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
  getExtent() { return [144.9631, -37.8136, 144.9631, -37.8136]; }
}
class Layer {
  constructor(options) { this.source = options.source; this.visible = true; }
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
class Style { constructor(options) { this.options = options; } }
globalThis.ol = {
  source: { Vector: Source }, layer: { Vector: Layer }, Feature,
  geom: { Point: class { constructor(coords) { this.coords = coords; } } },
  proj: { fromLonLat: coords => coords },
  style: { Style, Circle: Style, Fill: Style, Stroke: Style, Icon: Style, Text: Style },
  control: { Control: class { constructor(options) { this.element = options.element; } } },
};
const routes = await import("../public/js/routes.js");
// The production module may not exist yet during RED.
let detections;
try { detections = await import("../public/js/real-detections.js"); } catch (error) {
  if (error.code !== "ERR_MODULE_NOT_FOUND") throw error;
}
const record = {
  _id: "esp32-detection", sourceType: "real", species: "Magpie", confidence: 91.5,
  timestamp: "2026-08-06T10:30:00Z", sensorId: "esp32-001",
  microphoneLLA: [-37.8136, 144.9631, 0], animalTrueLLA: [10, 20, 0], animalEstLLA: [30, 40, 0],
};
function state() {
  return { basemap: { layers: [], controls: [], fits: [],
    getView() { return { fit: (...args) => this.fits.push(args) }; },
    addLayer(layer) { this.layers.push(layer); },
    addControl(control) { this.controls.push(control); } } };
}

test("shared detection client uses same-origin endpoint and timeout", async () => {
  calls = []; result = { data: [record] };
  assert.equal(typeof routes.retrieveDetections, "function");
  assert.deepEqual(await routes.retrieveDetections(), result);
  assert.deepEqual(calls, ["/api/detections"]);
  assert.equal(config.timeout, 10000);
});

test("real marker uses microphone coordinates and refresh reuses its layer/control", async () => {
  assert.ok(detections, "real detection loader exists");
  const hmi = state(); result = { data: [record, record] };
  await detections.loadRealDetections(hmi);
  await detections.loadRealDetections(hmi);
  assert.equal(hmi.basemap.layers.length, 1);
  assert.equal(hmi.basemap.controls.length, 1);
  assert.equal(hmi.basemap.fits.length, 1, "first successful load brings real markers into view only once");
  const features = hmi.realDetectionLayer.getSource().getFeatures();
  assert.equal(features.length, 1);
  assert.deepEqual(features[0].get("geometry").coords, [144.9631, -37.8136]);
  assert.equal(features[0].get("sourceType"), "real");
  assert.equal(features[0].get("sensorId"), "esp32-001");
  assert.match(hmi.realDetectionStatus.textContent, /1 real-device detection/);
});

test("real layer sits above microphone layers sharing the same coordinate", async () => {
  assert.ok(detections, "real detection loader exists");
  const hmi = { ...state(), layerPool: 1000 };
  result = { data: [record] };
  await detections.loadRealDetections(hmi);
  assert.equal(hmi.realDetectionLayer.zIndex, 1000);
  assert.equal(hmi.layerPool, 999);
  // A microphone layer allocated afterwards from the same pool renders below.
  assert.ok(hmi.realDetectionLayer.zIndex > hmi.layerPool);
});

test("real layer still renders on top without a layer pool", async () => {
  const hmi = state(); result = { data: [record] };
  await detections.loadRealDetections(hmi);
  assert.ok(hmi.realDetectionLayer.zIndex > 1000);
});

test("invalid real coordinates never create a marker and show a safe data error", async () => {
  assert.ok(detections, "real detection loader exists");
  for (const microphoneLLA of [undefined, null, [], [1, 2], [1, 2, 3, 4],
    ["1", 2, 3], [true, 2, 3], [NaN, 2, 3], [1, Infinity, 3], [1, 2, Infinity],
    [-91, 2, 3], [91, 2, 3], [1, -181, 3], [1, 181, 3]]) {
    const hmi = state(); result = { data: [{ ...record, microphoneLLA }] };
    await detections.loadRealDetections(hmi);
    assert.equal(hmi.realDetectionLayer.getSource().getFeatures().length, 0);
    assert.match(hmi.realDetectionStatus.textContent, /invalid microphone coordinates/i);
    assert.doesNotMatch(hmi.realDetectionStatus.textContent, /NaN|Infinity|undefined|http/);
  }
});

const objectRecord = {
  _id: "esp32-object", sourceType: "real", species: "Magpie", confidence: 91.5,
  timestamp: "2026-08-06T10:30:00Z", sensorId: "esp32-002",
  microphoneLLA: { latitude: -37.8136, longitude: 144.9631, altitude: 0 },
  animalTrueLLA: null, animalEstLLA: null, animalLLAUncertainty: null,
};

test("real marker uses object microphone coordinates and tolerates null animal fields", async () => {
  assert.ok(detections, "real detection loader exists");
  const hmi = state(); result = { data: [objectRecord] };
  await detections.loadRealDetections(hmi);
  const features = hmi.realDetectionLayer.getSource().getFeatures();
  assert.equal(features.length, 1);
  assert.deepEqual(features[0].get("geometry").coords, [144.9631, -37.8136]);
  assert.equal(features[0].get("sourceType"), "real");
  assert.equal(features[0].get("sensorId"), "esp32-002");
  assert.match(hmi.realDetectionStatus.textContent, /1 real-device detection/);
});

test("simulator records never create real markers", async () => {
  assert.ok(detections, "real detection loader exists");
  const hmi = state();
  result = { data: [{ ...objectRecord, _id: "sim-1", sourceType: "simulator" }] };
  await detections.loadRealDetections(hmi);
  assert.equal(hmi.realDetectionLayer.getSource().getFeatures().length, 0);
  assert.match(hmi.realDetectionStatus.textContent, /No real-device detections/);
});

test("invalid object microphone coordinates never create a marker and show a safe data error", async () => {
  assert.ok(detections, "real detection loader exists");
  for (const microphoneLLA of [{ longitude: 0, altitude: 0 }, { latitude: 0, altitude: 0 },
    { latitude: -91, longitude: 0, altitude: 0 }, { latitude: 0, longitude: 181, altitude: 0 },
    { latitude: "-37.8", longitude: 144.9, altitude: 0 }, { latitude: true, longitude: 0, altitude: 0 },
    { latitude: NaN, longitude: 0, altitude: 0 }, { latitude: 0, longitude: Infinity, altitude: 0 },
    null, "[-37.8, 144.9, 0]"]) {
    const hmi = state(); result = { data: [{ ...objectRecord, microphoneLLA }] };
    await detections.loadRealDetections(hmi);
    assert.equal(hmi.realDetectionLayer.getSource().getFeatures().length, 0);
    assert.match(hmi.realDetectionStatus.textContent, /invalid microphone coordinates/i);
    assert.doesNotMatch(hmi.realDetectionStatus.textContent, /NaN|Infinity|undefined|http/);
  }
});

test("vocalization converter reads object LLAs and preserves null animal locations", async () => {
  const hmiModule = await import("../public/js/HMI.js");
  const simRecord = { _id: "sim-1", sourceType: "simulator", species: "Magpie", confidence: 88,
    commonName: "Magpie", type: "Bird", status: "Least Concern", diet: "Omnivore",
    timestamp: "2026-08-06T10:30:00Z", sensorId: "sim-001",
    microphoneLLA: { latitude: -37.8, longitude: 144.9, altitude: 0 },
    animalEstLLA: { latitude: 10, longitude: 20, altitude: 0 },
    animalTrueLLA: { latitude: 30, longitude: 40, altitude: 0 },
    animalLLAUncertainty: 5 };
  const event = hmiModule.convertJSONtoAnimalVocalizationEvent({}, simRecord);
  assert.equal(event.sensorLat, -37.8);
  assert.equal(event.sensorLon, 144.9);
  assert.equal(event.estLat, 10);
  assert.equal(event.estLon, 20);
  assert.equal(event.locationLat, 30);
  assert.equal(event.locationLon, 40);
  assert.equal(event.locationConfidence, 95);
  assert.equal(event.sourceType, "simulator");
  const nulls = hmiModule.convertJSONtoAnimalVocalizationEvent(
    {}, { ...simRecord, animalEstLLA: null, animalTrueLLA: null, animalLLAUncertainty: null });
  assert.equal(nulls.estLat, null);
  assert.equal(nulls.locationLat, null);
  assert.equal(nulls.locationLon, null);
  assert.equal(nulls.locationConfidence, null);
  assert.equal(nulls.sensorLat, -37.8);
  assert.equal(nulls.sensorLon, 144.9);
  assert.equal(nulls.sourceType, "simulator");
  const unknown = hmiModule.convertJSONtoAnimalVocalizationEvent(
    {}, { ...simRecord, sourceType: "bogus" });
  assert.equal(unknown.sourceType, "bogus");
});

test("vocalization plot location falls back to the microphone only at render time", async () => {
  const hmiModule = await import("../public/js/HMI.js");
  assert.deepEqual(
    hmiModule.resolveVocalizationPlotLocation({ locationLat: 30, locationLon: 40, sensorLat: -37.8, sensorLon: 144.9 }),
    { lat: 30, lon: 40, isFallback: false });
  assert.deepEqual(
    hmiModule.resolveVocalizationPlotLocation({ locationLat: null, locationLon: null, sensorLat: -37.8, sensorLon: 144.9 }),
    { lat: -37.8, lon: 144.9, isFallback: true });
  assert.equal(
    hmiModule.resolveVocalizationPlotLocation({ locationLat: null, locationLon: null, sensorLat: null, sensorLon: null }),
    null);
});

test("vocalization detail values show unavailable instead of null or null%", async () => {
  const hmiModule = await import("../public/js/HMI.js");
  assert.equal(hmiModule.formatVocalizationDetailValue(null), "unavailable");
  assert.equal(hmiModule.formatVocalizationDetailValue(undefined), "unavailable");
  assert.equal(hmiModule.formatVocalizationDetailValue(NaN), "unavailable");
  assert.equal(hmiModule.formatVocalizationDetailValue(null, "%"), "unavailable");
  assert.equal(hmiModule.formatVocalizationDetailValue(95, "%"), "95%");
  assert.equal(hmiModule.formatVocalizationDetailValue(-37.8), "-37.8");
});

test("loading, empty, malformed and failed reads never substitute sample records", async (t) => {
  assert.ok(detections, "real detection loader exists");
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const hmi = state();
  let resolve;
  result = new Promise(done => { resolve = done; });
  const pending = detections.loadRealDetections(hmi);
  assert.match(hmi.realDetectionStatus.textContent, /Loading/);
  resolve({ data: [record] }); await pending;
  result = { data: [] }; await detections.loadRealDetections(hmi);
  assert.equal(hmi.realDetectionLayer.getSource().getFeatures().length, 0);
  assert.match(hmi.realDetectionStatus.textContent, /No real-device detections/);
  result = { data: { bad: "http://private-service/secret" } };
  await detections.loadRealDetections(hmi);
  assert.match(hmi.realDetectionStatus.textContent, /invalid detection data/i);
  result = { then(resolve, reject) { reject(new Error("http://private-service/secret")); } };
  calls = [];
  const failed = detections.loadRealDetections(hmi);
  for (let i = 0; i < 10; i++) { await Promise.resolve(); t.mock.timers.tick(2000); }
  await failed;
  assert.equal(calls.length, 3);
  assert.equal(hmi.realDetectionLayer.getSource().getFeatures().length, 0);
  assert.doesNotMatch(hmi.realDetectionStatus.textContent, /private-service|secret/);
  assert.match(hmi.realDetectionStatus.textContent, /Unable to load detections/);
});

test("failure statuses show distinct safe guidance", async (t) => {
  assert.ok(detections, "real detection loader exists");
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const cases = [
    [401, /log in again/i],
    [403, /administrator/i],
    [429, /too many|wait/i],
    [503, /try again later/i],
  ];
  for (const [status, pattern] of cases) {
    const hmi = state();
    result = { then(_resolve, reject) { reject({ response: { status } }); } };
    const failed = detections.loadRealDetections(hmi);
    for (let i = 0; i < 10; i++) { await Promise.resolve(); t.mock.timers.tick(2000); }
    await failed;
    assert.equal(hmi.realDetectionLayer.getSource().getFeatures().length, 0);
    assert.match(hmi.realDetectionStatus.textContent, pattern);
    assert.doesNotMatch(hmi.realDetectionStatus.textContent, /http|undefined/i);
  }
});

test("a late old response cannot overwrite newer live data", async () => {
  assert.ok(detections, "real detection loader exists");
  const hmi = state(); let resolve;
  result = new Promise(done => { resolve = done; });
  const old = detections.loadRealDetections(hmi);
  result = { data: [record] }; await detections.loadRealDetections(hmi);
  resolve({ data: [] }); await old;
  assert.equal(hmi.realDetectionLayer.getSource().getFeatures().length, 1);
});

test("production startup has no embedded detection assignments", async () => {
  const html = await readFile(new URL("../public/index.html", import.meta.url), "utf8");
  assert.doesNotMatch(html, /hmiState\.vocalizationEvents\s*=\s*vocalizationEvents/);
  assert.doesNotMatch(html, /const vocalizationEvents\s*=\s*\[\s*\{/);
});

test("HMI initialization loads real detections without waiting for microphone or simulator services", async () => {
  globalThis.fetch = async () => ({ json: async () => ({ data: [] }) });
  const hmiModule = await import("../public/js/HMI.js");
  const hmi = state();
  holdAncillary = true; calls = []; result = { data: [record] };
  try {
    await hmiModule.initialiseHMI(hmi);
    assert.equal(hmi.realDetectionLayer.getSource().getFeatures().length, 1);
    assert.ok(calls.includes("/api/detections"));
    await hmiModule.initialiseHMI(hmi);
    assert.equal(hmi.basemap.layers.length, 1);
    // The simulator/history path must not produce a second copy of a real marker.
    hmi.vocalizationEvents = [];
    hmiModule.updateVocalizationLayerFromLiveData(hmi, [record]);
    assert.equal(hmi.vocalizationEvents.length, 0);
  } finally { holdAncillary = false; }
});
