import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

let result = { data: [] };
let calls = [];
let callArgs = [];
let config;
let holdAncillary = false;
globalThis.window = { axios: { create(options) {
  config = options;
  return { async get(url, options) {
    calls.push(url);
    callArgs.push([url, options]);
    if (holdAncillary && url !== "/api/detections") return new Promise(() => {});
    return await result;
  } };
} }, addEventListener() {} };
class Element {
  constructor(tagName = "div") { this.tagName = tagName; this.children = []; this.style = {}; this.listeners = {}; this.textContent = ""; this.innerText = ""; this._innerHTML = ""; this.classList = { add() {}, remove() {} }; }
  appendChild(el) { this.children.push(el); return el; }
  replaceChildren(...children) {
  this.children = [...children];
}
  get innerHTML() { return this._innerHTML; }
  set innerHTML(value) { this._innerHTML = String(value); if (String(value) === "") this.children = []; }

focus() {
  this.focused = true;
}
  setAttribute(key, value) { this[key] = value; }
  addEventListener(key, fn) { this.listeners[key] = fn; }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  insertAdjacentElement() {}
  remove() {}
}
const SIDEBAR_IDS = [
  "desc_name", "desc_confidence", "desc_species", "desc_summary", "desc_details",
  "desc_img", "markup_details", "markup_loc_lat", "markup_loc_lon", "markup_source",
  "markup_date", "markup_confidence", "markup_location_metric_label",
  "animal_weather_section", "animalAudioHeader", "animalAudioControl",
  "animal-spectrogram", "request-edit-button", "animal-popup-content", "basemap",
];
const sidebarRegistry = {};
function resetSidebarDOM() {
  for (const id of SIDEBAR_IDS) sidebarRegistry[id] = new Element();
  return sidebarRegistry;
}
resetSidebarDOM();
function sidebarText(el) {
  if (!el) return "";
  const own = el.textContent || el.innerText || "";
  const kids = (el.children || []).map(sidebarText).join(" ");
  return [own, kids].filter(Boolean).join(" ");
}
globalThis.document = {
  readyState: "loading", addEventListener() {}, dispatchEvent() {}, head: new Element(), body: new Element(),
  createElement: (tag) => new Element(tag), getElementById: (id) => sidebarRegistry[id] || null, querySelector: () => null,
};
globalThis.requestAnimationFrame = () => {};
globalThis.getComputedStyle = () => ({ position: "relative" });
class Source {
  constructor() { this.features = []; }
  clear() { this.features = []; }
  addFeature(feature) { this.features.push(feature); }
  changed() {}
  getFeatures() { return this.features; }
  getExtent() { return [144.9631, -37.8136, 144.9631, -37.8136]; }
}
class Layer {
  constructor(options) { this.source = options.source; this.style = options.style; this.visible = true; }
  getSource() { return this.source; }
  set(key, value) { this[key] = value; }
  setZIndex(zIndex) { this.zIndex = zIndex; }
  setVisible(visible) { this.visible = visible; }
  changed() {}
}
class Feature {
  constructor(properties) { this.properties = properties; }
  setStyle(style) { this.style = style; }
  setId(id) { this.id = id; }
  get(key) { return this.properties[key]; }
  getProperties() { return this.properties; }
}
class Style { constructor(options) { this.options = options; } }

globalThis.ol = {
  source: { Vector: Source }, layer: { Vector: Layer }, Feature,
  geom: { Point: class { constructor(coords) { this.coords = coords; } } },
  proj: { fromLonLat: coords => coords },
  style: { Style, Circle: Style, Fill: Style, Stroke: Style, Icon: Style, Text: Style },
  control: { Control: class { constructor(options) { this.element = options.element; } } },
};
const routes = await import("../../public/shared/http/routes.js");
// The production module may not exist yet during RED.
let detections;
try { detections = await import("../../public/features/detections/real-detections.js"); } catch (error) {
  if (error.code !== "ERR_MODULE_NOT_FOUND") throw error;
}
const record = {
  _id: "esp32-detection", sourceType: "real", species: "Magpie", confidence: 91.5,
  timestamp: "2026-08-06T10:30:00Z", sensorId: "esp32-001",
  microphoneLLA: [-37.8136, 144.9631, 0], animalTrueLLA: [10, 20, 0], animalEstLLA: [30, 40, 0],
};

function state() {
  return {
    basemap: {
      layers: [],
      controls: [],
      interactions: [],
      fits: [],

      getView() {
        return {
          fit: (...args) => this.fits.push(args)
        };
      },

      addLayer(layer) {
        this.layers.push(layer);
      },

      addControl(control) {
        this.controls.push(control);
      },

      addInteraction(interaction) {
        this.interactions.push(interaction);
      }
    }
  };
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

test("repeated loads keep one layer and one status/refresh control with no extra interaction", async () => {
  assert.ok(detections, "real detection loader exists");
  const hmi = state(); result = { data: [record] };
  await detections.loadRealDetections(hmi);
  await detections.loadRealDetections(hmi);
  assert.equal(hmi.basemap.layers.length, 1);
  assert.equal(hmi.basemap.controls.length, 1);
  assert.equal(hmi.basemap.interactions.length, 0);
  const panel = hmi.basemap.controls[0].element;
  assert.equal(panel.children.length, 2);
  assert.equal(panel.children[1].textContent, "Refresh detections");
  assert.equal(hmi.realDetectionSelect, undefined);
  assert.equal(hmi.realDetectionDetails, undefined);
});

test("loading real detections creates no Select interaction", async () => {
  assert.ok(detections, "real detection loader exists");
  const hmi = state(); result = { data: [record] };
  await detections.loadRealDetections(hmi);
  assert.equal(hmi.basemap.interactions.length, 0);
  assert.equal(hmi.realDetectionSelect, undefined);
  assert.equal(typeof ol.interaction, "undefined");
  const source = await readFile(new URL("../../public/features/detections/real-detections.js", import.meta.url), "utf8");
  assert.doesNotMatch(source, /ol\.interaction\.Select/);
  assert.doesNotMatch(source, /realDetectionSelect/);
  assert.doesNotMatch(source, /realDetectionDetails/);
  assert.doesNotMatch(source, /document\.addEventListener/);
});

test("real layer style comes from buildRealDetectionStyle and includes REAL label", async () => {
  assert.ok(detections, "real detection loader exists");
  const { buildRealDetectionStyle } = await import("../../public/features/detections/source-filter.js");
  assert.match(JSON.stringify(buildRealDetectionStyle()), /REAL/);
  const hmi = state(); result = { data: [record] };
  await detections.loadRealDetections(hmi);
  assert.match(JSON.stringify(hmi.realDetectionLayer.style), /REAL/);
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
  const hmiModule = await import("../../public/features/map/hmi-map.js");
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
  const { detectionMatchesSourceFilter, formatDetectionSourceLabel } =
    await import("../../public/features/detections/source-filter.js");
  for (const missing of [undefined, ""]) {
    const sourceless = hmiModule.convertJSONtoAnimalVocalizationEvent(
      {}, { ...simRecord, sourceType: missing });
    assert.equal(detectionMatchesSourceFilter(sourceless.sourceType, "all"), true);
    assert.equal(detectionMatchesSourceFilter(sourceless.sourceType, "simulator"), false);
    assert.equal(detectionMatchesSourceFilter(sourceless.sourceType, "real"), false);
    assert.equal(formatDetectionSourceLabel(sourceless.sourceType), "Unknown source");
  }
});

test("vocalization plot location falls back to the microphone only at render time", async () => {
  const hmiModule = await import("../../public/features/map/hmi-map.js");
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
  const hmiModule = await import("../../public/features/map/hmi-map.js");
  assert.equal(hmiModule.formatVocalizationDetailValue(null), "unavailable");
  assert.equal(hmiModule.formatVocalizationDetailValue(undefined), "unavailable");
  assert.equal(hmiModule.formatVocalizationDetailValue(NaN), "unavailable");
  assert.equal(hmiModule.formatVocalizationDetailValue(null, "%"), "unavailable");
  assert.equal(hmiModule.formatVocalizationDetailValue("", "%"), "unavailable");
  assert.equal(hmiModule.formatVocalizationDetailValue(95, "%"), "95%");
  assert.equal(hmiModule.formatVocalizationDetailValue(-37.8), "-37.8");
});

test("simulator details use the detection timestamp and preserve zero confidence", async () => {
  const hmiModule = await import("../../public/features/map/hmi-map.js");
  const event = hmiModule.convertJSONtoAnimalVocalizationEvent(
    { currentTime: 999999 },
    { ...record, sourceType: "simulator", confidence: 0 },
  );

  assert.equal(event.timestamp, 999999, "session time remains available for live expiry");
  assert.equal(event.eventTimestamp, record.timestamp);
  assert.equal(
    hmiModule.formatDetectionTimestamp(event.eventTimestamp),
    new Date(record.timestamp).toUTCString(),
  );
  assert.equal(hmiModule.formatVocalizationDetailValue(event.speciesIdentificationConfidence, "%"), "0%");
  assert.equal(hmiModule.formatVocalizationDetailValue(null, "%"), "unavailable");
});

test("simulator details show unavailable for a missing detection timestamp", async () => {
  const hmiModule = await import("../../public/features/map/hmi-map.js");
  assert.equal(hmiModule.formatDetectionTimestamp(undefined), "unavailable");
  assert.equal(hmiModule.formatDetectionTimestamp("not-a-date"), "unavailable");
});

test("simulator features retain detection time separately from session time", async () => {
  const hmiModule = await import("../../public/features/map/hmi-map.js");
  const layer = new Layer({ source: new Source() });
  const hmi = {
    ...state(),
    currentTime: 999999,
    basemap: {},
    layers: { normal_mammal: layer },
    detectionSourceFilter: "all",
    vocalizationEvents: [],
  };

  hmiModule.updateVocalizationLayerFromPastData(hmi, [{
    ...record,
    sourceType: "simulator",
    type: undefined,
    status: undefined,
    diet: undefined,
  }]);

  const feature = layer.getSource().getFeatures()[0];
  assert.equal(feature.get("animalRecordDate"), 999999);
  assert.equal(feature.get("eventTimestamp"), record.timestamp);
});

test("selecting a simulator detection renders its event timestamp and safe confidence", async () => {
  const hmiModule = await import("../../public/features/map/hmi-map.js");
  resetSidebarDOM();
  const previousJQuery = globalThis.window.$;
  const previousResult = result;
  globalThis.window.$ = () => ({ show() {}, hide() {} });
  calls = [];
  result = {
    data: {
      Date: { 0: "2026-08-06" },
      "Min Temperature (°C)": { 0: 0 },
      "Max Temperature (°C)": { 0: 0 },
      "Rainfall (mm)": { 0: 0 },
      "Wind Speed (m/sec)": { 0: 0 },
      "Max Humidity (%)": { 0: 0 },
      "Min Humidity (%)": { 0: 0 },
    },
  };
  let clickHandler;
  const feature = new Feature({
    animalSpecies: "magpie", animalType: "mammal", animalDiet: "herbivore",
    animalStatus: "normal", animalIcon: "", animalLat: 0, animalLon: 0,
    animalConfidence: null, animalLocConfidence: null, animalRecordDate: 999999,
    eventTimestamp: record.timestamp, eventId: null, isAnimalMovement: 0,
    sourceType: "simulator",
  });
  const hmi = { basemap: {
    on(_event, handler) { clickHandler = handler; },
    forEachFeatureAtPixel() { return feature; },
  } };

  try {
    hmiModule.createMapClickEvent(hmi);
    clickHandler({ pixel: [] });
    await Promise.resolve();
    assert.equal(sidebarRegistry.markup_date.innerText, new Date(record.timestamp).toUTCString());
    assert.equal(sidebarRegistry.desc_confidence.innerText, "unavailable");
    assert.ok(calls.includes(
      `/api/weather?timestamp=${Math.floor(new Date(record.timestamp).getTime() / 1000)}&lat=0&lon=0`,
    ));
  } finally {
    globalThis.window.$ = previousJQuery;
    result = previousResult;
  }
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
  const html = await readFile(new URL("../../public/pages/map/index.html", import.meta.url), "utf8");
  assert.doesNotMatch(html, /hmiState\.vocalizationEvents\s*=\s*vocalizationEvents/);
  assert.doesNotMatch(html, /const vocalizationEvents\s*=\s*\[\s*\{/);
});

test("map and detail reads use the shared request boundary, not page-local backend calls", async () => {
  assert.equal(typeof routes.retrieveDetections, "function");
  assert.equal(typeof routes.retrieveVocalizationEventsInTimeRange, "function");
  assert.equal(typeof routes.retrieveTruthEventsInTimeRange, "function");
  assert.equal(typeof routes.retrieveMicrophones, "function");
  assert.equal(typeof routes.retrieveAudio, "function");
  assert.equal(typeof routes.retrieveWeatherData, "function");
  calls = []; result = { data: [] };
  await routes.retrieveWeatherData(1721997541, -38.8, 143.5);
  assert.deepEqual(calls, ["/api/weather?timestamp=1721997541&lat=-38.8&lon=143.5"]);
  assert.equal(config.timeout, 10000);
  const hmiSource = await readFile(new URL("../../public/features/map/hmi-map.js", import.meta.url), "utf8");
  assert.doesNotMatch(hmiSource, /localhost:9000/);
  assert.doesNotMatch(hmiSource, /fetch\(\s*[`'"]http/);
  const loaderSource = await readFile(new URL("../../public/features/detections/real-detections.js", import.meta.url), "utf8");
  assert.doesNotMatch(loaderSource, /localhost:9000/);
  assert.doesNotMatch(loaderSource, /fetch\(/);
});

test("no embedded, localStorage, or seeded records can populate the live success path", async () => {
  const html = await readFile(new URL("../../public/pages/map/index.html", import.meta.url), "utf8");
  assert.doesNotMatch(html, /sampleJSONmovementEvents/);
  const hmiSource = await readFile(new URL("../../public/features/map/hmi-map.js", import.meta.url), "utf8");
  assert.doesNotMatch(hmiSource, /localStorage/);
  const loaderSource = await readFile(new URL("../../public/features/detections/real-detections.js", import.meta.url), "utf8");
  assert.doesNotMatch(loaderSource, /localStorage/);
  assert.doesNotMatch(loaderSource, /sample_data/);
});

test("HMI initialization loads real detections without waiting for microphone or simulator services", async () => {
  globalThis.fetch = async () => ({ json: async () => ({ data: [] }) });
  const hmiModule = await import("../../public/features/map/hmi-map.js");
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

const completeReal = {
  _id: "esp32-001", sourceType: "real", species: "Magpie", confidence: 91.5,
  timestamp: "2026-08-06T10:30:00Z", sensorId: "esp32-001",
  microphoneLLA: [-37.8136, 144.9631, 10],
  animalEstLLA: [-37.82, 144.97, 5],
  animalTrueLLA: null, animalLLAUncertainty: 12,
};
const missingFieldReal = {
  _id: "esp32-002", sourceType: "real", species: "Magpie", confidence: null,
  timestamp: "not-a-date", sensorId: "",
  microphoneLLA: [-37.8136, 144.9631, 10],
  animalEstLLA: null, animalTrueLLA: null, animalLLAUncertainty: null,
};
const unsafeReal = {
  ...completeReal, _id: "esp32-unsafe",
  species: '<img src=x onerror="alert(1)">',
  sensorId: '<script>alert("xss")</script>',
};

test("real sidebar shows species, confidence, sensor, source, timestamp, mic and estimated location", async () => {
  const hmiModule = await import("../../public/features/map/hmi-map.js");
  resetSidebarDOM();
  hmiModule.showRealDetectionDetails(completeReal);
  assert.equal(sidebarRegistry.desc_name.textContent, "Magpie");
  assert.equal(sidebarRegistry.desc_species.textContent, "Magpie");
  assert.equal(sidebarRegistry.desc_confidence.textContent, "91.5%");
  assert.doesNotMatch(sidebarText(sidebarRegistry.markup_details), /%/);
  assert.doesNotMatch(sidebarText(sidebarRegistry.desc_details), /91\.5/);
  assert.match(sidebarText(sidebarRegistry.markup_details), /esp32-001/);
  assert.equal(sidebarRegistry.markup_source.textContent, "Real-device");
  assert.equal(sidebarRegistry.markup_date.textContent, new Date("2026-08-06T10:30:00Z").toUTCString());
  assert.equal(sidebarRegistry.markup_loc_lat.textContent, "-37.8136");
  assert.equal(sidebarRegistry.markup_loc_lon.textContent, "144.9631");
  assert.match(sidebarText(sidebarRegistry.desc_details), /Estimated animal location/);
  assert.match(sidebarText(sidebarRegistry.desc_details), /-37\.82/);
  assert.match(sidebarText(sidebarRegistry.desc_details), /144\.97/);
  assert.equal(sidebarRegistry.markup_confidence.textContent, "12");
});

test("real sidebar shows unavailable for invalid timestamp, missing estimate and null uncertainty", async () => {
  const hmiModule = await import("../../public/features/map/hmi-map.js");
  resetSidebarDOM();
  hmiModule.showRealDetectionDetails({ ...missingFieldReal, timestamp: "not-a-date" });
  assert.equal(sidebarRegistry.markup_date.textContent, "unavailable");
  assert.doesNotMatch(sidebarRegistry.markup_date.textContent, /Invalid Date/);
  assert.match(sidebarText(sidebarRegistry.desc_details), /unavailable/);
  assert.equal(sidebarRegistry.markup_confidence.textContent, "unavailable");
  resetSidebarDOM();
  hmiModule.showRealDetectionDetails({ ...missingFieldReal, timestamp: undefined });
  assert.equal(sidebarRegistry.markup_date.textContent, "unavailable");
  assert.doesNotMatch(sidebarRegistry.markup_date.textContent, /Invalid Date/);
  assert.match(sidebarText(sidebarRegistry.desc_details), /unavailable/);
  assert.equal(sidebarRegistry.markup_confidence.textContent, "unavailable");
});

test("real sidebar keeps unsafe species and sensor text without child markup", async () => {
  const hmiModule = await import("../../public/features/map/hmi-map.js");
  resetSidebarDOM();
  hmiModule.showRealDetectionDetails(unsafeReal);
  assert.match(sidebarText(sidebarRegistry.desc_name), /<img/);
  assert.match(sidebarText(sidebarRegistry.markup_details), /<script>/);
  assert.equal(sidebarRegistry.desc_name.children.length, 0);
  assert.equal(sidebarRegistry.desc_species.children.length, 0);
  assert.equal(sidebarRegistry.markup_details.children.length, 0);
  assert.doesNotMatch(sidebarRegistry.desc_name.innerHTML, /</);
  assert.doesNotMatch(sidebarRegistry.markup_details.innerHTML, /</);
});

test("real sidebar mode hides unsupported sections and restores simulator state", async () => {
  const hmiModule = await import("../../public/features/map/hmi-map.js");
  resetSidebarDOM();
  hmiModule.setRealDetectionSidebarMode(true);
  assert.equal(sidebarRegistry.animal_weather_section.style.display, "none");
  assert.equal(sidebarRegistry.desc_img.style.display, "none");
  assert.equal(sidebarRegistry["request-edit-button"].style.display, "none");
  assert.equal(sidebarRegistry.animalAudioHeader.style.display, "none");
  assert.equal(sidebarRegistry.animalAudioControl.style.display, "none");
  assert.equal(sidebarRegistry["animal-spectrogram"].style.display, "none");
  assert.equal(sidebarRegistry.markup_location_metric_label.textContent, "Location uncertainty");
  hmiModule.setRealDetectionSidebarMode(false);
  assert.notEqual(sidebarRegistry.animal_weather_section.style.display, "none");
  assert.notEqual(sidebarRegistry.desc_img.style.display, "none");
  assert.notEqual(sidebarRegistry["request-edit-button"].style.display, "none");
  assert.equal(sidebarRegistry.markup_location_metric_label.textContent, "Location Confidence");
  assert.equal(sidebarRegistry.animalAudioHeader.style.display, "none");
});

test("detection map has keyboard focus and accessible label", async () => {
  const html = await readFile(new URL("../../public/pages/map/index.html", import.meta.url), "utf8");
  const basemap = html.match(/<div[^>]*id="basemap"[^>]*>/);
  assert.ok(basemap, "#basemap exists");
  assert.match(basemap[0], /tabindex="0"/);
  assert.match(basemap[0], /aria-label="Detection map"/);
});

test("detection details region has accessible name and focus target", async () => {
  const html = await readFile(new URL("../../public/pages/map/index.html", import.meta.url), "utf8");
  const region = html.match(/<div[^>]*id="animal-popup-content"[^>]*>/);
  assert.ok(region, "#animal-popup-content exists");
  assert.match(region[0], /role="region"/);
  assert.match(region[0], /aria-label="Detection details"/);
  assert.match(region[0], /tabindex="-1"/);
});

test("animalToggled handler focuses detail region after opening sidebar", async () => {
  const html = await readFile(new URL("../../public/pages/map/index.html", import.meta.url), "utf8");
  const toggledIndex = html.indexOf("animalToggled");
  assert.notEqual(toggledIndex, -1, "animalToggled handler exists");
  const handlerSlice = html.slice(toggledIndex, toggledIndex + 800);
  const openDouble = handlerSlice.indexOf('openNav("animal-popup")');
  const openSingle = handlerSlice.indexOf("openNav('animal-popup')");
  const openIndex = openDouble !== -1 ? openDouble : openSingle;
  assert.notEqual(openIndex, -1, "openNav animal-popup call exists");
  const focusDouble = handlerSlice.indexOf('getElementById("animal-popup-content").focus');
  const focusSingle = handlerSlice.indexOf("getElementById('animal-popup-content').focus");
  const focusIndex = focusDouble !== -1 ? focusDouble : focusSingle;
  assert.notEqual(focusIndex, -1, "detail region focus call exists");
  assert.ok(focusIndex > openIndex, "focus runs after openNav");
});

test("detail-region Escape handler closes menu before returning focus to map", async () => {
  const html = await readFile(new URL("../../public/pages/map/index.html", import.meta.url), "utf8");
  const keydownMatches = html.match(/getElementById\(["']animal-popup-content["']\)\.addEventListener\(["']keydown["']/g) || [];
  assert.equal(keydownMatches.length, 1, "one detail-region keydown handler");
  const anchorDouble = html.indexOf('getElementById("animal-popup-content").addEventListener("keydown"');
  const anchorSingle = html.indexOf("getElementById('animal-popup-content').addEventListener('keydown'");
  const anchor = anchorDouble !== -1 ? anchorDouble : anchorSingle;
  assert.notEqual(anchor, -1, "detail-region keydown registration exists");
  const handlerSlice = html.slice(anchor, anchor + 800);
  const escapeDouble = handlerSlice.indexOf('event.key === "Escape"');
  const escapeSingle = handlerSlice.indexOf("event.key === 'Escape'");
  assert.ok(escapeDouble !== -1 || escapeSingle !== -1, "Escape key check exists");
  const closeIndex = handlerSlice.indexOf("closeMenu()");
  assert.notEqual(closeIndex, -1, "closeMenu call exists");
  const mapFocusDouble = handlerSlice.indexOf('getElementById("basemap").focus');
  const mapFocusSingle = handlerSlice.indexOf("getElementById('basemap').focus");
  const mapFocusIndex = mapFocusDouble !== -1 ? mapFocusDouble : mapFocusSingle;
  assert.notEqual(mapFocusIndex, -1, "basemap focus call exists");
  assert.ok(closeIndex < mapFocusIndex, "closeMenu runs before basemap focus");
});

test("detection client forwards the selected source and never sends sourceType=all", async () => {
  calls = []; callArgs = []; result = { data: [] };
  await routes.retrieveDetections("real");
  assert.deepEqual(callArgs, [["/api/detections", { params: { sourceType: "real" } }]]);
  calls = []; callArgs = [];
  await routes.retrieveDetections("simulator");
  assert.deepEqual(callArgs, [["/api/detections", { params: { sourceType: "simulator" } }]]);
  for (const source of ["all", undefined, "simulated", ""]) {
    calls = []; callArgs = [];
    await routes.retrieveDetections(source);
    assert.deepEqual(callArgs, [["/api/detections", undefined]]);
    assert.doesNotMatch(JSON.stringify(callArgs), /sourceType/);
  }
});

test("real loader fetches with the selected source filter", async () => {
  assert.ok(detections, "real detection loader exists");
  const hmi = { ...state(), detectionSourceFilter: "real" };
  calls = []; callArgs = []; result = { data: [record] };
  await detections.loadRealDetections(hmi);
  assert.deepEqual(callArgs, [["/api/detections", { params: { sourceType: "real" } }]]);
  assert.equal(hmi.realDetectionLayer.getSource().getFeatures().length, 1);
});

test("switching the source filter refetches without duplicating layers", async () => {
  const hmiModule = await import("../../public/features/map/hmi-map.js");
  const hmi = { ...state(), detectionSourceFilter: "all" };
  calls = []; callArgs = []; result = { data: [record] };
  await detections.loadRealDetections(hmi);
  assert.equal(hmi.basemap.layers.length, 1);
  calls = []; callArgs = [];
  const applied = hmiModule.setDetectionSourceFilter(hmi, "real");
  assert.equal(applied.filter, "real");
  for (let i = 0; i < 20; i++) await Promise.resolve();
  assert.deepEqual(callArgs, [["/api/detections", { params: { sourceType: "real" } }]]);
  assert.equal(hmi.basemap.layers.length, 1);
  assert.equal(hmi.basemap.controls.length, 1);
});
