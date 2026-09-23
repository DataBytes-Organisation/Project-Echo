import { retrieveDetections, getApiErrorMessage } from "./routes.js";
import {
  DETECTION_SOURCE_FILTERS,
  applyDetectionSourceFilter,
  buildRealDetectionStyle,
  normalizeDetectionSource,
} from "./detection-source-filter.js";

function getMicrophoneCoordinates(lla) {
  if (!Array.isArray(lla) && (!lla || typeof lla !== "object")) return null;
  if (Array.isArray(lla) && lla.length !== 3) return null;

  const latitude = Array.isArray(lla) ? lla[0] : lla?.latitude;
  const longitude = Array.isArray(lla) ? lla[1] : lla?.longitude;
  const altitude = Array.isArray(lla) ? lla[2] : lla?.altitude;

  if (![latitude, longitude, altitude].every(Number.isFinite)) return null;
  if (Math.abs(latitude) > 90 || Math.abs(longitude) > 180) return null;
  return [longitude, latitude];
}

export async function loadRealDetections(hmiState) {
  if (!hmiState.realDetectionLayer) {
    const layer = new ol.layer.Vector({
      source: new ol.source.Vector(),
      style: buildRealDetectionStyle(),
    });

    layer.set("name", "real_detections");

    if (typeof layer.setZIndex === "function") {
      layer.setZIndex(
        Number.isFinite(hmiState.layerPool)
          ? hmiState.layerPool
          : 1001
      );

      if (Number.isFinite(hmiState.layerPool)) {
        hmiState.layerPool -= 1;
      }
    }

    hmiState.basemap.addLayer(layer);
    hmiState.realDetectionLayer = layer;

    // Status lives in the top #live-map-controls panel
    // (#real-detection-status), so no corner ol-control is created.
    hmiState.realDetectionStatus = document.getElementById("real-detection-status");
  }

  const request =
    (hmiState.realDetectionRequest || 0) + 1;

  hmiState.realDetectionRequest = request;
  const source = hmiState.realDetectionLayer.getSource();
  const status = hmiState.realDetectionStatus;
  status.textContent = "Loading real-device detections…";
  applyDetectionSourceFilter(
    hmiState,
    hmiState.detectionSourceFilter || DETECTION_SOURCE_FILTERS.ALL,
    { loading: true },
  );
  try {
    const response =
      await retrieveDetections(hmiState.detectionSourceFilter || DETECTION_SOURCE_FILTERS.ALL);

    if (
      request !== hmiState.realDetectionRequest
    ) {
      return;
    }

    source.clear();

    if (!Array.isArray(response.data)) {
      status.textContent = "The server returned invalid detection data.";
      applyDetectionSourceFilter(
        hmiState,
        hmiState.detectionSourceFilter || DETECTION_SOURCE_FILTERS.ALL,
        { errorMessage: "Unable to filter detections: the server returned invalid data." },
      );
      return;
    }

    let invalid = 0;
    const ids = new Set();

    for (const detection of response.data) {
      // All is a composite: sim markers render via the /events_time vocalization
      // layers; this layer keeps real records only.
      if (normalizeDetectionSource(detection?.sourceType) !== DETECTION_SOURCE_FILTERS.REAL) continue;
      const coordinates = getMicrophoneCoordinates(detection.microphoneLLA);
      if (!coordinates) { invalid++; continue; }
      if (ids.has(detection._id)) continue;
      ids.add(detection._id);

      const feature = new ol.Feature({
        ...detection,
        sourceType: "real",
        geometry: new ol.geom.Point(ol.proj.fromLonLat(coordinates)),
      });

      feature.setId(detection._id);
      source.addFeature(feature);
    }

    if (
      ids.size &&
      !hmiState.realDetectionFitted
    ) {
      hmiState.basemap.getView().fit(
        source.getExtent(),
        {
          padding: [60, 60, 60, 60],
          maxZoom: 14,
        }
      );

      hmiState.realDetectionFitted = true;
    }

    status.textContent = invalid
      ? "Some detections have invalid microphone coordinates and cannot be shown."
      : ids.size
        ? `${ids.size} real-device detection${ids.size === 1 ? "" : "s"} loaded.`
        : "No real-device detections found.";
    applyDetectionSourceFilter(
      hmiState,
      hmiState.detectionSourceFilter || DETECTION_SOURCE_FILTERS.ALL,
    );
  } catch (error) {
    if (
      request !== hmiState.realDetectionRequest
    ) {
      return;
    }

    source.clear();
    // The shared formatter receives only status/code; provider messages stay private.
    status.textContent = getApiErrorMessage(
      { code: error.code, response: error.response && { status: error.response.status } },
      "Unable to load detections. Please try again."
    );
    applyDetectionSourceFilter(
      hmiState,
      hmiState.detectionSourceFilter || DETECTION_SOURCE_FILTERS.ALL,
      { errorMessage: status.textContent },
    );
  }
}