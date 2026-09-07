// Called by Backend integration tests: only HTTP, session storage and DOM are faked.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import vm from "node:vm";
import VectorLayer from "ol/layer/Vector.js";
import VectorSource from "ol/source/Vector.js";
import Feature from "ol/Feature.js";
import Point from "ol/geom/Point.js";
import { fromLonLat, toLonLat } from "ol/proj.js";
import Style from "ol/style/Style.js";
import Circle from "ol/style/Circle.js";
import Fill from "ol/style/Fill.js";
import Stroke from "ol/style/Stroke.js";

const require = createRequire(import.meta.url);
const { createCheckUserSession } = require("../middleware/session");
const data = JSON.parse(readFileSync(0, "utf8"));
const handlers = new Map();
const module = { exports: {} };
vm.runInNewContext(readFileSync(new URL("../routes/map.routes.js", import.meta.url), "utf8"), {
  module, process: { env: { API_HOST: "backend.test" } }, console,
  require(name) {
    if (name === "dotenv") return { config() {} };
    if (name === "../middleware") return { checkUserSession: createCheckUserSession({
      isOpen: true, async get() { return "session-jwt"; },
    }) };
    if (name === "axios") return { async get(url, options) {
      assert.equal(url, "http://backend.test:9000/hmi/detections");
      assert.equal(options.headers.Authorization, "Bearer session-jwt");
      return { data };
    } };
    throw new Error(name);
  },
});
module.exports({ use() {}, get(path, ...chain) { handlers.set(path, chain); }, post() {}, put() {} });
globalThis.window = { axios: { create() { return { async get(path) {
  assert.equal(path, "/api/detections");
  const req = { path, session: { token: "session-jwt" } };
  let body;
  const res = { json(value) { body = value; } };
  let index = 0;
  const next = async () => { const handler = handlers.get(path)[index++]; if (handler) await handler(req, res, next); };
  await next(); return { data: body };
} }; } } };
globalThis.document = { createElement() { return { style: {}, setAttribute() {}, appendChild() {}, addEventListener() {} }; } };
globalThis.ol = {
  layer: { Vector: VectorLayer }, source: { Vector: VectorSource }, Feature, geom: { Point },
  proj: { fromLonLat }, style: { Style, Circle, Fill, Stroke }, control: { Control: class {} },
};
const { loadRealDetections } = await import("../public/js/real-detections.js");
const state = { basemap: { addLayer() {}, addControl() {}, getView() { return { fit() {} }; } } };
await loadRealDetections(state);
const features = state.realDetectionLayer.getSource().getFeatures();
assert.equal(features.length, 1);
assert.equal(features[0].get("sourceType"), "real");
assert.equal(features[0].get("sensorId"), "esp32-001");
const [lon, lat] = toLonLat(features[0].getGeometry().getCoordinates());
assert.ok(Math.abs(lat - (-37.8136)) < 1e-8);
assert.ok(Math.abs(lon - 144.9631) < 1e-8);
process.stdout.write(JSON.stringify({ sourceType: features[0].get("sourceType"), lat, lon }));
