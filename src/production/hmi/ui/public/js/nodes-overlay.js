"use strict";

/**
 * nodes-overlay.js
 *
 * FR-A1: Unified Live Map Rendering
 *
 * OpenLayers remains the single production map renderer used by EchoNet.
 *
 * IoT node data is loaded through retrieveIotNodes() from routes.js so the
 * map uses the shared frontend API client rather than making page-local
 * Axios/fetch requests.
 *
 * The IoT vector layer, popup overlay and pointer listener are created once
 * and reused. The vector source is cleared before every node refresh so
 * repeated updates cannot accumulate duplicate markers or connection lines.
 */

import { retrieveIotNodes } from "./routes.js";

const IOT_LAYER_NAME = "iot_nodes";

function hasValidCoordinates(node) {
  if (!node || !node.location) {
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

function createIoTNodeStyle(feature) {
  const type = feature.get("type");

  const iconSrc =
    type === "master"
      ? "images/nodes/master-node.svg"
      : "images/nodes/microchip-solid.svg";

  const circleColor = type === "master" ? "#ff4444" : "#4CAF50";

  return [
    new ol.style.Style({
      image: new ol.style.Circle({
        radius: 23,
        fill: new ol.style.Fill({
          color: circleColor,
        }),
      }),
    }),

    new ol.style.Style({
      image: new ol.style.Icon({
        src: iconSrc,
        scale: 0.05,
        anchor: [0.5, 0.5],
        anchorXUnits: "fraction",
        anchorYUnits: "fraction",
      }),
    }),
  ];
}

function createIoTNodeLayer() {
  const layer = new ol.layer.Vector({
    source: new ol.source.Vector(),
    style: createIoTNodeStyle,
  });

  layer.set("name", IOT_LAYER_NAME);

  return layer;
}

function initialiseNodePopup(hmiState) {
  if (hmiState.iotNodePopup) {
    return;
  }

  const popupElement = document.createElement("div");
  popupElement.className = "node-popup-container";

  const popup = new ol.Overlay({
    element: popupElement,
    positioning: "bottom-center",
    stopEvent: false,
  });

  hmiState.iotNodePopup = popup;
  hmiState.basemap.addOverlay(popup);

  if (!hmiState.iotNodePointerMoveHandler) {
    hmiState.iotNodePointerMoveHandler = function (evt) {
      const feature = hmiState.basemap.forEachFeatureAtPixel(
        evt.pixel,
        function (candidate) {
          return candidate.get("isNode") ? candidate : null;
        }
      );

      const element = popup.getElement();

      if (
        feature &&
        feature.get("name") &&
        feature.get("type") &&
        feature.get("model")
      ) {
        popup.setPosition(feature.getGeometry().getCoordinates());

        element.textContent = "";

        const container = document.createElement("div");
        container.className = "node-popup";

        const title = document.createElement("strong");
        title.textContent = feature.get("name");

        const type = document.createElement("div");
        type.textContent = `Type: ${feature.get("type")}`;

        const model = document.createElement("div");
        model.textContent = `Model: ${feature.get("model")}`;

        container.appendChild(title);
        container.appendChild(type);
        container.appendChild(model);

        element.appendChild(container);
        element.style.display = "block";
      } else {
        element.style.display = "none";
      }
    };

    hmiState.basemap.on(
      "pointermove",
      hmiState.iotNodePointerMoveHandler
    );
  }
}

function addNodeFeatures(source, nodes) {
  nodes.forEach((node) => {
    const latitude = Number(node.location.latitude);
    const longitude = Number(node.location.longitude);

    const feature = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([longitude, latitude])
      ),
      lat: latitude,
      lon: longitude,
      type: node.type,
      name: node.name,
      model: node.model,
      isNode: true,
    });

    feature.setId(node._id);

    source.addFeature(feature);
  });
}

function addUniqueConnectionFeatures(source, nodes) {
  const nodesById = new Map(
    nodes.map((node) => [String(node._id), node])
  );

  const drawnConnections = new Set();

  nodes.forEach((node) => {
    if (!Array.isArray(node.connectedNodes)) {
      return;
    }

    node.connectedNodes.forEach((connectedId) => {
      const connectedNode = nodesById.get(String(connectedId));

      if (!connectedNode) {
        return;
      }

      if (String(node._id) === String(connectedNode._id)) {
        return;
      }

      const pairKey = [
        String(node._id),
        String(connectedNode._id),
      ]
        .sort()
        .join(":");

      if (drawnConnections.has(pairKey)) {
        return;
      }

      drawnConnections.add(pairKey);

      const lineFeature = new ol.Feature({
        geometry: new ol.geom.LineString([
          ol.proj.fromLonLat([
            Number(node.location.longitude),
            Number(node.location.latitude),
          ]),
          ol.proj.fromLonLat([
            Number(connectedNode.location.longitude),
            Number(connectedNode.location.latitude),
          ]),
        ]),
        isNodeConnection: true,
      });

      lineFeature.setId(`connection:${pairKey}`);

      lineFeature.setStyle(
        new ol.style.Style({
          stroke: new ol.style.Stroke({
            color: "#3388ff",
            width: 2,
            lineDash: [5, 10],
          }),
        })
      );

      source.addFeature(lineFeature);
    });
  });
}

/**
 * Load IoT nodes into the existing production OpenLayers map.
 *
 * Repeated calls reuse one layer and clear its source before rerendering,
 * preventing duplicate markers and logical connections.
 *
 * @param {object} hmiState
 */
async function addIoTNodesToMap(hmiState) {
  if (!hmiState || !hmiState.basemap) {
    console.error("Basemap not initialized");
    return;
  }

  try {
    const response = await retrieveIotNodes();

    const nodes = Array.isArray(response?.data)
      ? response.data
      : [];

    const validNodes = nodes.filter(hasValidCoordinates);

    if (!hmiState.iotNodeLayer) {
      hmiState.iotNodeLayer = createIoTNodeLayer();
      hmiState.basemap.addLayer(hmiState.iotNodeLayer);
    }

    const source = hmiState.iotNodeLayer.getSource();

    /*
     * Idempotent refresh:
     * remove the previous node/connection features before rendering
     * the latest API response.
     */
    source.clear();

    addNodeFeatures(source, validNodes);
    addUniqueConnectionFeatures(source, validNodes);

    initialiseNodePopup(hmiState);
  } catch (error) {
    console.error("Error loading IoT nodes:", error);
    throw error;
  }
}

export {
  addIoTNodesToMap,
  hasValidCoordinates,
  addUniqueConnectionFeatures,
};