"use strict";

/**
 * map.js
 * Leaflet map initialisation and IoT node rendering for the EchoNet HMI.
 *
 * Sprint 1/2 : Map init, node markers, polylines, loading/error overlays.
 * Task 7     : Removed local duplicates — showToastMessage, getFriendlyFetchError,
 *              escapeHtml, safeFetchJson, and the inline loading/error/retry UI —
 *              all replaced with imports from HMI-utils.js and routes.js.
 *              loadNodes now calls retrieveIotNodes() (routes.js) so all API
 *              traffic goes through the shared axios instance with its timeout.
 */

import {
  showToast,
  getApiErrorMessage,
  showElementLoading,
  hideElementLoading,
  showRetryState,
  hideRetryState,
} from "./HMI-utils.js";

import { retrieveIotNodes } from "./routes.js";

// ─────────────────────────────────────────────────────────────────────────────
// Map state
// ─────────────────────────────────────────────────────────────────────────────

let map;
let markers = [];

// ─────────────────────────────────────────────────────────────────────────────
// Leaflet icons
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// Map initialisation
// ─────────────────────────────────────────────────────────────────────────────

async function initMap() {
  map = L.map("map").setView([-38.7789, 143.5705], 14);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "© OpenStreetMap contributors",
  }).addTo(map);

  await loadNodes();
}

// ─────────────────────────────────────────────────────────────────────────────
// Node loading
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fetch IoT nodes from the API and render them on the map.
 * Uses showElementLoading / hideElementLoading for the loading state and
 * showRetryState for the error state — both from HMI-utils.js.
 * The fetch goes through retrieveIotNodes() in routes.js so it uses the
 * shared axios instance (timeout, future interceptors, consistent base URL).
 */
async function loadNodes() {
  // "map" is the id of the map container div
  showElementLoading("map", "Loading IoT nodes...");
  hideRetryState("map");

  try {
    const response = await retrieveIotNodes();
    const nodes = response.data;

    // Clear existing markers and lines
    markers.forEach((marker) => marker.remove());
    markers = [];

    nodes.forEach((node) => {
      if (
        !node.location ||
        !node.location.latitude ||
        !node.location.longitude
      ) {
        return;
      }

      const icon = node.type === "master" ? masterIcon : childIcon;

      const marker = L.marker(
        [node.location.latitude, node.location.longitude],
        { icon, title: node.name }
      );

      const popupContent = `
        <div class="node-popup">
          <h3>${_escapeHtml(node.name)}</h3>
          <p>Type: ${_escapeHtml(node.type)}</p>
          <p>Model: ${_escapeHtml(node.model)}</p>
          ${node.parentNode ? `<p>Parent: ${_escapeHtml(node.parentNode)}</p>` : ""}
          <p>Components: ${Array.isArray(node.components) ? node.components.length : 0}</p>
        </div>
      `;

      marker.bindPopup(popupContent);
      marker.addTo(map);
      markers.push(marker);

      // Draw polylines to connected nodes
      if (node.connectedNodes && node.connectedNodes.length > 0) {
        node.connectedNodes.forEach((connectedId) => {
          const connectedNode = nodes.find((n) => n._id === connectedId);

          if (
            connectedNode &&
            connectedNode.location &&
            connectedNode.location.latitude &&
            connectedNode.location.longitude
          ) {
            const line = L.polyline(
              [
                [node.location.latitude, node.location.longitude],
                [
                  connectedNode.location.latitude,
                  connectedNode.location.longitude,
                ],
              ],
              {
                color: "#3388ff",
                weight: 2,
                opacity: 0.6,
                dashArray: "5, 10",
              }
            ).addTo(map);

            markers.push(line);
          }
        });
      }
    });

    showToast("IoT nodes loaded successfully", "success");
  } catch (error) {
    console.error("Error loading nodes:", error);

    const message = getApiErrorMessage(
      error,
      "Unable to load nodes. Please try again."
    );

    // showRetryState renders the error message and a Retry button inside
    // the map container.  When the user clicks Retry, loadNodes is called again.
    showRetryState("map", message, loadNodes);
    showToast("Failed to load IoT nodes", "error");
  } finally {
    hideElementLoading("map");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Escape HTML special characters.
 * Used only for Leaflet popup content built in this file.
 * (HMI-utils.js handles escaping internally for toast/banner/retry content.)
 */
function _escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", initMap);
