// Local-only UI check: actual Express detection route, shared client and OpenLayers.
// All Backend/session data below is an explicitly labelled deterministic fixture.
const express = require("express");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { createCheckUserSession } = require("../middleware/session");
const app = express();
let mode = "success";
const fixture = { _id: "esp32-fixture", sourceType: "real", sensorId: "esp32-001",
  species: "Magpie", confidence: 91.5, timestamp: "2026-08-06T10:30:00Z",
  microphoneLLA: [-37.8136, 144.9631, 0], animalTrueLLA: [10, 20, 0] };
app.use((req, _res, next) => { req.session = { token: "fixture-session" }; next(); });
const routeModule = { exports: {} };
vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../routes/map.routes.js"), "utf8"), {
  module: routeModule, process: { env: { API_HOST: "fixture.invalid" } }, console,
  require(name) {
    if (name === "dotenv") return { config() {} };
    if (name === "../middleware") return { checkUserSession: createCheckUserSession({
      isOpen: true, async get() { return "fixture-session"; },
    }) };
    if (name === "axios") return { async get() {
      if (mode === "loading") await new Promise(resolve => setTimeout(resolve, 15000));
      if (mode === "error") throw { response: { status: 503 } };
      return { data: mode === "empty" ? [] : [{ ...fixture,
        microphoneLLA: mode === "invalid" ? [91, 0, 0] : fixture.microphoneLLA }] };
    } };
    throw new Error(name);
  },
});
routeModule.exports(app);
app.get("/scenario/:mode", (req, res) => {
  if (!["success", "empty", "error", "invalid", "loading"].includes(req.params.mode)) return res.sendStatus(400);
  mode = req.params.mode;
  res.redirect("/");
});
app.get("/", (_req, res) => res.send(`<!doctype html><html lang="en"><head>
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ESP32 map verification</title>
<link rel="stylesheet" href="/css/ol.css"><style>
body{margin:0;font:16px system-ui;color:#222}header{padding:16px}nav{display:flex;flex-wrap:wrap;gap:18px}
#basemap{height:calc(100vh - 140px);min-height:300px;background:#e6ece8}output{display:block;padding:8px}
</style><script src="/js/ol.js"></script><script src="/axios.js"></script></head><body>
<header><strong>ESP32 map verification — deterministic fixture, no live services</strong>
<p>Production detection control and rendering. Scenario: ${mode}.</p><nav>
${["success", "empty", "error", "invalid", "loading"].map(value => `<a href="/scenario/${value}">${value}</a>`).join("")}
</nav></header><div id="basemap"></div><output id="counts"></output>
<script type="module">
import { loadRealDetections } from '/js/real-detections.js';
const state = { basemap: new ol.Map({ target: 'basemap', layers: [],
  view: new ol.View({ center: ol.proj.fromLonLat([144.9631,-37.8136]), zoom: 10 }) }) };
await loadRealDetections(state);
function counts() {
  const features = state.realDetectionLayer.getSource().getFeatures();
  document.getElementById('counts').textContent = 'Detection layers: ' + state.basemap.getLayers().getLength() +
    ' | Markers: ' + features.length + ' | Map controls: ' + state.basemap.getControls().getLength() +
    (features.length ? ' | Coordinates: ' + ol.proj.toLonLat(features[0].getGeometry().getCoordinates()).map(n => n.toFixed(4)).join(', ') : '');
}
counts(); state.realDetectionLayer.getSource().on('change', counts);
</script></body></html>`));
app.get("/axios.js", (_req, res) => res.sendFile(path.resolve(path.dirname(require.resolve("axios")), "../axios.min.js")));
app.use(express.static(path.join(__dirname, "../public")));
app.listen(3101, "127.0.0.1", () => console.log("Fixture UI: http://127.0.0.1:3101"));
