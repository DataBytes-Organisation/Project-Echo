"use strict";

/**
 * map.js
 * Canonical Leaflet map implementation for the EchoNet production HMI.
 *
 * FR-A1 architecture:
 * - Leaflet is the only production map-rendering library used on this page.
 * - A module-level singleton guard prevents duplicate Leaflet map instances.
 * - IoT nodes are retrieved through retrieveIotNodes() from routes.js.
 * - The frontend does not communicate with MongoDB directly.
 * - API configuration, environment URLs, timeouts, and request handling are
 *   managed by the shared API layer.
 * - A shared Leaflet layer group is cleared before each render, preventing
 *   markers and connection lines from accumulating during live updates.
 */

import {
  showToast,
  getApiErrorMessage,
  showElementLoading,
  hideElementLoading,
  showRetryState,
  hideRetryState,
  escapeHtml,
} from "./HMI-utils.js";

import { retrieveIotNodes } from "./routes.js";

let map = null;
let nodeLayerGroup = null;
let loadRequestId = 0;

const masterIcon = L.icon({
  iconUrl: "/images/nodes/master-node.png",
  iconSize: [32, 32],
  iconAnchor: [16, 32],
  popupAnchor: [0, -32],
});

const childIcon = L.icon({
  iconUrl: "/images/nodes/child-node.jpg",
  iconSize: [24, 24],
  iconAnchor: [12, 24],
  popupAnchor: [0, -24],
});

/**
 * Initialise the canonical Leaflet map once.
 */
async function initMap() {
  const mapContainer = document.getElementById("map");

  if (!mapContainer) {
    console.error("Unable to initialise map: #map container was not found.");
    return;
  }

  if (map) {
    console.debug("Leaflet map is already initialised.");
    return;
  }

  /*
   * Leaflet stores an internal ID on an initialised container.
   * This protects against another script having already created the map.
   */
  if (mapContainer._leaflet_id) {
    console.warn(
      "Map container is already associated with a Leaflet instance."
    );
    return;
  }

  map = L.map(mapContainer).setView([-38.7789, 143.5705], 14);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "© OpenStreetMap contributors",
  }).addTo(map);

  nodeLayerGroup = L.layerGroup().addTo(map);

  await loadNodes();
}

/**
 * Retrieve and render IoT nodes using the shared frontend API helper.
 */
async function loadNodes() {
  if (!map || !nodeLayerGroup) {
    console.error("Cannot load nodes before the map has been initialised.");
    return;
  }

  const currentRequestId = ++loadRequestId;

  showElementLoading("map", "Loading IoT nodes...");
  hideRetryState("map");

  try {
    const response = await retrieveIotNodes();

    /*
     * Ignore an older response if another refresh completed after it.
     * This prevents stale requests from overwriting newer map data.
     */
    if (currentRequestId !== loadRequestId) {
      return;
    }

    const nodes = Array.isArray(response?.data) ? response.data : [];

    nodeLayerGroup.clearLayers();

    const validNodes = nodes.filter(hasValidCoordinates);
    const nodesById = new Map(
      validNodes.map((node) => [String(node._id), node])
    );

    validNodes.forEach((node) => {
      renderNodeMarker(node);
    });

    renderConnectionLines(validNodes, nodesById);

    showToast("IoT nodes loaded successfully", "success");
  } catch (error) {
    if (currentRequestId !== loadRequestId) {
      return;
    }

    console.error("Error loading nodes:", error);

    const message = getApiErrorMessage(
      error,
      "Unable to load nodes. Please try again."
    );

    showRetryState("map", message, loadNodes);
    showToast("Failed to load IoT nodes", "error");
  } finally {
    if (currentRequestId === loadRequestId) {
      hideElementLoading("map");
    }
  }
}

/**
 * Add one IoT node marker to the shared map layer.
 */
function renderNodeMarker(node) {
  const latitude = Number(node.location.latitude);
  const longitude = Number(node.location.longitude);
  const icon = node.type === "master" ? masterIcon : childIcon;

  const marker = L.marker([latitude, longitude], {
    icon,
    title: String(node.name ?? "Unnamed node"),
  });

  const popupContent = `
    <div class="node-popup">
      <h3>${escapeHtml(node.name ?? "Unnamed node")}</h3>
      <p>Type: ${escapeHtml(node.type ?? "Unknown")}</p>
      <p>Model: ${escapeHtml(node.model ?? "Unknown")}</p>
      ${
        node.parentNode
          ? `<p>Parent: ${escapeHtml(node.parentNode)}</p>`
          : ""
      }
      <p>Components: ${
        Array.isArray(node.components) ? node.components.length : 0
      }</p>
    </div>
  `;

  marker.bindPopup(popupContent);
  marker.addTo(nodeLayerGroup);
}

/**
 * Add each logical node connection once.
 */
function renderConnectionLines(nodes, nodesById) {
  const renderedConnections = new Set();

  nodes.forEach((node) => {
    if (!Array.isArray(node.connectedNodes)) {
      return;
    }

    node.connectedNodes.forEach((connectedId) => {
      const connectedNode = nodesById.get(String(connectedId));

      if (!connectedNode || String(node._id) === String(connectedNode._id)) {
        return;
      }

      /*
       * Sorting produces the same key for A → B and B → A,
       * preventing duplicate bidirectional lines.
       */
      const connectionKey = [
        String(node._id),
        String(connectedNode._id),
      ]
        .sort()
        .join(":");

      if (renderedConnections.has(connectionKey)) {
        return;
      }

      renderedConnections.add(connectionKey);

      L.polyline(
        [
          [
            Number(node.location.latitude),
            Number(node.location.longitude),
          ],
          [
            Number(connectedNode.location.latitude),
            Number(connectedNode.location.longitude),
          ],
        ],
        {
          color: "#3388ff",
          weight: 2,
          opacity: 0.6,
          dashArray: "5, 10",
        }
      ).addTo(nodeLayerGroup);
    });
  });
}

/**
 * Confirm that a node contains usable numeric coordinates.
 */
function hasValidCoordinates(node) {
  if (!node?.location) {
    return false;
  }

  const latitude = Number(node.location.latitude);
  const longitude = Number(node.location.longitude);

  return (
    Number.isFinite(latitude) &&
    Number.isFinite(longitude) &&
    latitude >= -90 &&
    latitude <= 90 &&
    longitude >= -180 &&
    longitude <= 180
  );
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initMap, { once: true });
} else {
  initMap();
}

export { initMap, loadNodes };