"use strict";

/**
 * Ticket 03: filter and distinguish simulated vs real-device detections.
 * Keeps the Backend/Engine contract value "simulator" while UI labels say
 * "Simulated". Does not fetch new APIs — only toggles existing layers.
 */

export const DETECTION_SOURCE_FILTERS = Object.freeze({
  ALL: "all",
  SIMULATOR: "simulator",
  REAL: "real",
});

const VOCALIZATION_STATUSES = ["endangered", "vulnerable", "near-threatened", "normal", "invasive"];
const VOCALIZATION_TYPES = ["mammal", "bird", "amphibian", "reptile", "insect"];

export function normalizeDetectionSource(sourceType) {
  if (sourceType === "real") return DETECTION_SOURCE_FILTERS.REAL;
  if (sourceType === "simulator" || sourceType == null || sourceType === "") {
    return DETECTION_SOURCE_FILTERS.SIMULATOR;
  }
  return "unknown";
}

export function formatDetectionSourceLabel(sourceType) {
  const normalized = normalizeDetectionSource(sourceType);
  if (normalized === DETECTION_SOURCE_FILTERS.REAL) return "Real-device";
  if (normalized === DETECTION_SOURCE_FILTERS.SIMULATOR) return "Simulated";
  return "Unknown source";
}

export function detectionMatchesSourceFilter(sourceType, filter) {
  // All is fail-open: unrecognised values (e.g. the spec typo "simulated")
  // stay visible under All so a filter can never silently drop records.
  // Explicit Simulated/Real filters still exclude them.
  if (filter === DETECTION_SOURCE_FILTERS.ALL) return true;
  const normalized = normalizeDetectionSource(sourceType);
  if (normalized === "unknown") return false;
  return normalized === filter;
}

export function buildRealDetectionStyle() {
  return [
    new ol.style.Style({
      image: new ol.style.Circle({
        radius: 8,
        fill: new ol.style.Fill({ color: "#007f73" }),
        stroke: new ol.style.Stroke({ color: "#fff", width: 2 }),
      }),
    }),
    new ol.style.Style({
      text: new ol.style.Text({
        text: "REAL",
        offsetY: -16,
        font: "bold 11px sans-serif",
        fill: new ol.style.Fill({ color: "#003d38" }),
        stroke: new ol.style.Stroke({ color: "#fff", width: 3 }),
      }),
    }),
  ];
}

export function buildSimulatorVocalizationStyle(iconPath) {
  return [
    new ol.style.Style({
      image: new ol.style.Icon({
        src: iconPath,
        anchor: [0.5, 1],
        scale: 0.75,
        className: "vocalization-icon",
      }),
    }),
    new ol.style.Style({
      text: new ol.style.Text({
        text: "SIM",
        offsetY: -28,
        font: "bold 11px sans-serif",
        fill: new ol.style.Fill({ color: "#5a3d00" }),
        stroke: new ol.style.Stroke({ color: "#fff", width: 3 }),
      }),
    }),
  ];
}

function speciesAllowsVocalization(filterState, status, animalType) {
  if (!Array.isArray(filterState)) return true;
  return filterState.includes(`_${status}`) && filterState.includes(`_${animalType}`);
}

function speciesAllowsTruth(filterState, status, animalType) {
  if (!Array.isArray(filterState)) return true;
  // Production speciesFilterState holds checkbox ids (_normal, _bird, ...).
  const statusOk = filterState.includes(status) || filterState.includes(`_${status}`);
  const typeOk = filterState.includes(animalType) || filterState.includes(`_${animalType}`);
  return statusOk && typeOk;
}

function setVocalizationLayersVisible(hmiState, visible, speciesFilterState) {
  for (const status of VOCALIZATION_STATUSES) {
    for (const animalType of VOCALIZATION_TYPES) {
      const layer = hmiState.layers?.[`${status}_${animalType}`];
      if (!layer || typeof layer.setVisible !== "function") continue;
      layer.setVisible(Boolean(
        visible && speciesAllowsVocalization(speciesFilterState, status, animalType)
      ));
    }
  }
}

function setTruthLayersVisible(hmiState, visible, speciesFilterState) {
  // Deliberate: animal-simulation movement markers are simulator-side context,
  // so Real-device hides them. Ticket 03 handoff only asked for real/sim
  // vocalization layers — this extra hiding is intentional, locked by tests.
  for (const status of VOCALIZATION_STATUSES) {
    for (const animalType of VOCALIZATION_TYPES) {
      const layer = hmiState.layers?.[`${status}_${animalType}_truth`];
      if (!layer || typeof layer.setVisible !== "function") continue;
      layer.setVisible(Boolean(
        visible && speciesAllowsTruth(speciesFilterState, status, animalType)
      ));
    }
  }
}

function countLayerFeatures(layer) {
  if (!layer || typeof layer.getSource !== "function") return 0;
  const source = layer.getSource();
  if (!source || typeof source.getFeatures !== "function") return 0;
  return source.getFeatures().length;
}

function countSimulatorMarkers(hmiState) {
  let total = 0;
  let sawLayer = false;
  for (const status of VOCALIZATION_STATUSES) {
    for (const animalType of VOCALIZATION_TYPES) {
      const layer = hmiState.layers?.[`${status}_${animalType}`];
      if (!layer) continue;
      sawLayer = true;
      total += countLayerFeatures(layer);
    }
  }
  if (sawLayer) return total;
  if (Array.isArray(hmiState.vocalizationEvents)) {
    return hmiState.vocalizationEvents.filter((entry) =>
      detectionMatchesSourceFilter(entry.sourceType, DETECTION_SOURCE_FILTERS.SIMULATOR)
    ).length;
  }
  return 0;
}

function countRealMarkers(hmiState) {
  return countLayerFeatures(hmiState.realDetectionLayer);
}

export function describeDetectionSourceFilterState(hmiState, filter, options = {}) {
  const { loading = false, errorMessage = null } = options;
  if (loading) return "Loading detections for the selected source…";
  if (errorMessage) return errorMessage;

  const showSim = filter === DETECTION_SOURCE_FILTERS.ALL
    || filter === DETECTION_SOURCE_FILTERS.SIMULATOR;
  const showReal = filter === DETECTION_SOURCE_FILTERS.ALL
    || filter === DETECTION_SOURCE_FILTERS.REAL;

  const simCount = showSim ? countSimulatorMarkers(hmiState) : 0;
  const realCount = showReal ? countRealMarkers(hmiState) : 0;

  if (filter === DETECTION_SOURCE_FILTERS.SIMULATOR) {
    return simCount
      ? `${simCount} simulated detection${simCount === 1 ? "" : "s"} shown.`
      : "No simulated detections to show.";
  }
  if (filter === DETECTION_SOURCE_FILTERS.REAL) {
    return realCount
      ? `${realCount} real-device detection${realCount === 1 ? "" : "s"} shown.`
      : "No real-device detections to show.";
  }
  const total = simCount + realCount;
  if (!total) return "No detections to show for the current source filter.";
  return `${total} detection${total === 1 ? "" : "s"} shown (${simCount} simulated, ${realCount} real-device).`;
}

/**
 * Apply All / Simulated / Real-device visibility without recreating the map,
 * layers, markers, or click listeners.
 */
export function applyDetectionSourceFilter(hmiState, filter, options = {}) {
  const next = Object.values(DETECTION_SOURCE_FILTERS).includes(filter)
    ? filter
    : DETECTION_SOURCE_FILTERS.ALL;

  hmiState.detectionSourceFilter = next;

  const showSim = next === DETECTION_SOURCE_FILTERS.ALL
    || next === DETECTION_SOURCE_FILTERS.SIMULATOR;
  const showReal = next === DETECTION_SOURCE_FILTERS.ALL
    || next === DETECTION_SOURCE_FILTERS.REAL;

  setVocalizationLayersVisible(hmiState, showSim, hmiState.speciesFilterState);
  setTruthLayersVisible(hmiState, showSim, hmiState.speciesFilterState);

  if (hmiState.realDetectionLayer && typeof hmiState.realDetectionLayer.setVisible === "function") {
    hmiState.realDetectionLayer.setVisible(showReal);
  }

  const message = describeDetectionSourceFilterState(hmiState, next, options);
  if (hmiState.detectionSourceStatus) {
    hmiState.detectionSourceStatus.textContent = message;
  }
  if (
    hmiState.realDetectionStatus
    && !options.loading
    && !options.errorMessage
    && showReal
    && next === DETECTION_SOURCE_FILTERS.REAL
    && countRealMarkers(hmiState) === 0
  ) {
    hmiState.realDetectionStatus.textContent = "No real-device detections to show.";
  }

  return { filter: next, message, showSim, showReal };
}
