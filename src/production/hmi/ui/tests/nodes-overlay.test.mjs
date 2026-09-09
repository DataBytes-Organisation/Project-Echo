"use strict";

/**
 * Regression tests for FR-A1 (nodes-overlay.js) — imports and exercises the
 * REAL exported functions (addIoTNodesToMap, hasValidCoordinates,
 * addUniqueConnectionFeatures), not a re-implementation of their logic.
 *
 * nodes-overlay.js is a browser ES module: it references the global `ol`
 * (OpenLayers) and does `require("axios")` when `window` is undefined (see
 * routes.js). This file is .mjs (native ESM, not the CommonJS the rest of
 * this test setup uses) so it can `import` that module directly, and it
 * shims just the small slice of globals nodes-overlay.js's call path
 * actually touches — `ol`, `document`, `require("axios")` — rather than
 * pulling in a real browser/DOM testing framework. That keeps this on
 * Node's built-in test runner exercising the production code, without
 * adding jsdom or any other new dependency for what is a fairly small
 * amount of DOM/OL surface.
 *
 * HMI.js itself is NOT imported here: it runs DOM setup as a side effect of
 * being imported at all (see initializeStaticDOMHooks(), called at module
 * top level), which would need a much larger DOM shim (effectively jsdom)
 * to import safely. Its FR-A1 changes (idempotent createBasemap, awaiting
 * addIoTNodesToMap before showing success) are covered by inspection/manual
 * verification instead — see the PR description.
 */

import test from "node:test";
import assert from "node:assert/strict";

let axiosGetResponse = { data: [] };
let axiosGetError = null;

globalThis.require = (specifier) => {
  if (specifier === "axios") {
    return {
      create() {
        return {
          get: async () => {
            if (axiosGetError) throw axiosGetError;
            return axiosGetResponse;
          },
        };
      },
    };
  }
  throw new Error(`Unexpected require("${specifier}") in test`);
};

// ---- minimal OpenLayers stub: just enough surface for nodes-overlay.js ----
class FakeSource {
  constructor() { this.features = []; }
  addFeature(f) { this.features.push(f); }
  clear() { this.features = []; }
  getFeatures() { return this.features; }
}
class FakeVectorLayer {
  constructor(opts) { this._source = opts.source; this._props = {}; }
  getSource() { return this._source; }
  set(k, v) { this._props[k] = v; }
  get(k) { return this._props[k]; }
}
class FakeFeature {
  constructor(props = {}) { this._props = { ...props }; }
  get(k) { return this._props[k]; }
  set(k, v) { this._props[k] = v; }
  setId(id) { this._id = id; }
  getId() { return this._id; }
  setStyle(s) { this._style = s; }
  getGeometry() { return this._props.geometry; }
}
class FakeOverlay {
  constructor(opts) { this._opts = opts; }
  getElement() { return this._opts.element; }
  setPosition() {}
}

globalThis.ol = {
  style: {
    Style: class { constructor(o) { this._o = o; } },
    Circle: class { constructor(o) { this._o = o; } },
    Fill: class { constructor(o) { this._o = o; } },
    Icon: class { constructor(o) { this._o = o; } },
    Stroke: class { constructor(o) { this._o = o; } },
  },
  layer: { Vector: FakeVectorLayer },
  source: { Vector: FakeSource },
  Feature: FakeFeature,
  geom: {
    Point: class { constructor(c) { this.coords = c; } },
    LineString: class { constructor(c) { this.coords = c; } },
  },
  proj: { fromLonLat: (c) => c },
  Overlay: FakeOverlay,
};

// minimal fake basemap: addLayer/addOverlay/on/forEachFeatureAtPixel
function makeFakeBasemap() {
  return {
    layers: [],
    overlays: [],
    handlers: {},
    addLayer(l) { this.layers.push(l); },
    addOverlay(o) { this.overlays.push(o); },
    on(evt, fn) { this.handlers[evt] = this.handlers[evt] || []; this.handlers[evt].push(fn); },
    forEachFeatureAtPixel() { return null; },
  };
}

// document stub — needed for popup element creation and because retrieveIotNodes()
// (via withRetry) calls showToast()/HMI-utils on retry attempts.
class FakeElement {
  constructor() {
    this.className = "";
    this.style = {};
    this.textContent = "";
    this.innerHTML = "";
    this.children = [];
    this.classList = { add() {}, remove() {}, toggle() {}, contains: () => false };
  }
  appendChild(child) { this.children.push(child); return child; }
  insertAdjacentElement() {}
  setAttribute() {}
  removeAttribute() {}
  addEventListener() {}
  removeEventListener() {}
  remove() {}
  focus() {}
  querySelector() { return new FakeElement(); }
}
if (typeof document === "undefined") {
  globalThis.document = {
    head: new FakeElement(),
    body: new FakeElement(),
    createElement: () => new FakeElement(),
    getElementById: () => null,
    addEventListener() {},
    removeEventListener() {},
  };
}
if (typeof getComputedStyle === "undefined") {
  globalThis.getComputedStyle = () => ({ position: "static" });
}
if (typeof requestAnimationFrame === "undefined") {
  globalThis.requestAnimationFrame = (cb) => setTimeout(cb, 0);
}

const mod = await import("../public/js/nodes-overlay.js");
const { addIoTNodesToMap, hasValidCoordinates, addUniqueConnectionFeatures } = mod;

// ── hasValidCoordinates ──────────────────────────────────────────────
test("hasValidCoordinates: valid, including 0,0", () => {
  assert.equal(hasValidCoordinates({ location: { latitude: 0, longitude: 0 } }), true);
  assert.equal(hasValidCoordinates({ location: { latitude: -38.7, longitude: 143.5 } }), true);
});

test("hasValidCoordinates: rejects out-of-range / missing", () => {
  assert.equal(hasValidCoordinates({ location: { latitude: 200, longitude: 0 } }), false);
  assert.equal(hasValidCoordinates({ location: { latitude: 0, longitude: 200 } }), false);
  assert.equal(hasValidCoordinates({}), false);
  assert.equal(hasValidCoordinates(null), false);
  assert.equal(hasValidCoordinates({ location: { latitude: "abc", longitude: 0 } }), false);
});

// ── addUniqueConnectionFeatures (real production function) ──────────
test("addUniqueConnectionFeatures: bidirectional A<->B renders once", () => {
  const source = new FakeSource();
  const nodes = [
    { _id: "A", location: { latitude: 0, longitude: 0 }, connectedNodes: ["B"] },
    { _id: "B", location: { latitude: 1, longitude: 1 }, connectedNodes: ["A"] },
  ];
  addUniqueConnectionFeatures(source, nodes);
  assert.equal(source.features.length, 1);
});

test("addUniqueConnectionFeatures: duplicate connection ids collapse to one line", () => {
  const source = new FakeSource();
  const nodes = [
    { _id: "A", location: { latitude: 0, longitude: 0 }, connectedNodes: ["B", "B", "B"] },
    { _id: "B", location: { latitude: 1, longitude: 1 }, connectedNodes: [] },
  ];
  addUniqueConnectionFeatures(source, nodes);
  assert.equal(source.features.length, 1);
});

test("addUniqueConnectionFeatures: self-connection ignored", () => {
  const source = new FakeSource();
  const nodes = [{ _id: "A", location: { latitude: 0, longitude: 0 }, connectedNodes: ["A"] }];
  addUniqueConnectionFeatures(source, nodes);
  assert.equal(source.features.length, 0);
});

test("addUniqueConnectionFeatures: missing connected node handled safely", () => {
  const source = new FakeSource();
  const nodes = [{ _id: "A", location: { latitude: 0, longitude: 0 }, connectedNodes: ["MISSING"] }];
  addUniqueConnectionFeatures(source, nodes);
  assert.equal(source.features.length, 0);
});

test("addUniqueConnectionFeatures: multiple logical connections rendered once each", () => {
  const source = new FakeSource();
  const nodes = [
    { _id: "A", location: { latitude: 0, longitude: 0 }, connectedNodes: ["B", "C"] },
    { _id: "B", location: { latitude: 1, longitude: 1 }, connectedNodes: ["A", "C"] },
    { _id: "C", location: { latitude: 2, longitude: 2 }, connectedNodes: ["A", "B"] },
  ];
  addUniqueConnectionFeatures(source, nodes);
  assert.equal(source.features.length, 3);
});

// ── addIoTNodesToMap (real production function, end to end) ─────────
test("addIoTNodesToMap: invalid coordinates are filtered out", async () => {
  axiosGetError = null;
  axiosGetResponse = {
    data: [
      { _id: "A", location: { latitude: 0, longitude: 0 }, connectedNodes: [] },
      { _id: "B", location: { latitude: 999, longitude: 0 }, connectedNodes: [] },
    ],
  };
  const hmiState = { basemap: makeFakeBasemap() };
  await addIoTNodesToMap(hmiState);
  const source = hmiState.iotNodeLayer.getSource();
  assert.equal(source.features.length, 1);
});

test("addIoTNodesToMap: empty node response does not crash", async () => {
  axiosGetError = null;
  axiosGetResponse = { data: [] };
  const hmiState = { basemap: makeFakeBasemap() };
  await assert.doesNotReject(() => addIoTNodesToMap(hmiState));
  assert.equal(hmiState.iotNodeLayer.getSource().features.length, 0);
});

test("addIoTNodesToMap: repeated refresh reuses the layer and does not accumulate", async () => {
  axiosGetError = null;
  const hmiState = { basemap: makeFakeBasemap() };

  axiosGetResponse = {
    data: [
      { _id: "A", location: { latitude: 0, longitude: 0 }, connectedNodes: ["B"] },
      { _id: "B", location: { latitude: 1, longitude: 1 }, connectedNodes: ["A"] },
    ],
  };
  await addIoTNodesToMap(hmiState);
  assert.equal(hmiState.basemap.layers.length, 1, "one layer added on first load");
  assert.equal(hmiState.iotNodeLayer.getSource().features.length, 3, "2 nodes + 1 connection");

  // second refresh with a different (smaller) node set
  axiosGetResponse = {
    data: [{ _id: "A", location: { latitude: 0, longitude: 0 }, connectedNodes: [] }],
  };
  await addIoTNodesToMap(hmiState);

  assert.equal(hmiState.basemap.layers.length, 1, "layer is reused, not duplicated");
  assert.equal(
    hmiState.iotNodeLayer.getSource().features.length,
    1,
    "old features cleared, only latest response present"
  );
});

test("addIoTNodesToMap: popup overlay and pointer handler are only registered once", async () => {
  axiosGetError = null;
  axiosGetResponse = { data: [] };
  const hmiState = { basemap: makeFakeBasemap() };

  await addIoTNodesToMap(hmiState);
  await addIoTNodesToMap(hmiState);

  assert.equal(hmiState.basemap.overlays.length, 1, "popup overlay added exactly once");
  assert.equal(
    hmiState.basemap.handlers.pointermove.length,
    1,
    "pointermove handler registered exactly once"
  );
});

test("addIoTNodesToMap: API failure rejects instead of failing silently", async () => {
  axiosGetError = new Error("network down");
  axiosGetResponse = { data: [] };
  const hmiState = { basemap: makeFakeBasemap() };

  await assert.rejects(() => addIoTNodesToMap(hmiState), /network down/);
});
