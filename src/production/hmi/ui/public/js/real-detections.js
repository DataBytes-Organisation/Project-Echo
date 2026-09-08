import { retrieveDetections, getApiErrorMessage } from "./routes.js";

function validMicrophoneLLA(lla) {
  return Array.isArray(lla) && lla.length === 3 && lla.every(Number.isFinite) &&
    Math.abs(lla[0]) <= 90 && Math.abs(lla[1]) <= 180;
}

export async function loadRealDetections(hmiState) {
  if (!hmiState.realDetectionLayer) {
    const layer = new ol.layer.Vector({
      source: new ol.source.Vector(),
      style: new ol.style.Style({ image: new ol.style.Circle({
        radius: 8, fill: new ol.style.Fill({ color: "#007f73" }),
        stroke: new ol.style.Stroke({ color: "#fff", width: 2 }),
      }) }),
    });
    layer.set("name", "real_detections");
    // Microphone layers take z-indices near 1000 from the shared pool, and
    // this layer is created before they exist: claim the pool top so a
    // microphone icon never covers the detection circle at its coordinate.
    if (typeof layer.setZIndex === "function") {
      layer.setZIndex(Number.isFinite(hmiState.layerPool) ? hmiState.layerPool : 1001);
      if (Number.isFinite(hmiState.layerPool)) hmiState.layerPool -= 1;
    }
    hmiState.basemap.addLayer(layer);
    hmiState.realDetectionLayer = layer;

    const panel = document.createElement("div");
    panel.className = "ol-unselectable ol-control";
    Object.assign(panel.style, {
      bottom: "12px", left: "12px", maxWidth: "calc(100% - 24px)",
      padding: "8px", background: "#fff", color: "#222", fontSize: "14px",
    });
    const status = document.createElement("span");
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");
    panel.appendChild(status);
    hmiState.realDetectionStatus = status;
    const refresh = document.createElement("button");
    refresh.type = "button";
    refresh.textContent = "Refresh detections";
    Object.assign(refresh.style, { width: "auto", padding: "0 8px" });
    refresh.addEventListener("click", () => { void loadRealDetections(hmiState); });
    panel.appendChild(refresh);
    hmiState.basemap.addControl(new ol.control.Control({ element: panel }));
  }

  const request = (hmiState.realDetectionRequest || 0) + 1;
  hmiState.realDetectionRequest = request;
  const source = hmiState.realDetectionLayer.getSource();
  const status = hmiState.realDetectionStatus;
  status.textContent = "Loading real-device detections…";
  try {
    const response = await retrieveDetections();
    if (request !== hmiState.realDetectionRequest) return;
    source.clear();
    if (!Array.isArray(response.data)) {
      status.textContent = "The server returned invalid detection data.";
      return;
    }
    let invalid = 0;
    const ids = new Set();
    for (const detection of response.data) {
      if (detection?.sourceType !== "real") continue;
      if (!validMicrophoneLLA(detection.microphoneLLA)) { invalid++; continue; }
      if (ids.has(detection._id)) continue;
      ids.add(detection._id);
      const [lat, lon] = detection.microphoneLLA;
      const feature = new ol.Feature({
        ...detection, geometry: new ol.geom.Point(ol.proj.fromLonLat([lon, lat])),
      });
      feature.setId(detection._id);
      source.addFeature(feature);
    }
    if (ids.size && !hmiState.realDetectionFitted) {
      hmiState.basemap.getView().fit(source.getExtent(), { padding: [60, 60, 60, 60], maxZoom: 14 });
      hmiState.realDetectionFitted = true;
    }
    status.textContent = invalid
      ? "Some detections have invalid microphone coordinates and cannot be shown."
      : ids.size ? `${ids.size} real-device detection${ids.size === 1 ? "" : "s"} loaded.`
        : "No real-device detections found.";
  } catch (error) {
    if (request !== hmiState.realDetectionRequest) return;
    source.clear();
    // The shared formatter receives only status/code; provider messages stay private.
    status.textContent = getApiErrorMessage(
      { code: error.code, response: error.response && { status: error.response.status } },
      "Unable to load detections. Please try again."
    );
  }
}
