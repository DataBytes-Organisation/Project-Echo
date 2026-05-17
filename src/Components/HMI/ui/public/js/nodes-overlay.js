


let axios;

if (typeof window === "undefined") {
  axios = require("axios");
} else {
  axios = window.axios;
}

let detectionSimulationInterval = null;
let currentHighlightedFeature = null;
let currentHighlightResetTimeout = null;
let nodeFeaturesCache = [];

/* ---------------- NOTIFICATION ---------------- */

function showNotification(message, type = "info") {
  const notification = document.getElementById("notification");
  const text = document.getElementById("notification-text");
  const closeBtn = document.getElementById("notification-close");

  if (!notification || !text) {
    console.error("Notification elements not found");
    return;
  }

  text.innerText = message;

  notification.classList.remove(
    "notification-success",
    "notification-error",
    "notification-info"
  );

  notification.classList.add(`notification-${type}`);
  notification.style.display = "block";

  const hide = () => {
    notification.style.display = "none";
  };

  if (closeBtn) {
    closeBtn.onclick = hide;
  }

  setTimeout(hide, 3500);
}

window.showNotification = showNotification;

/* ---------------- NODE STYLING ---------------- */

function getNodeStyles(feature, state = "idle") {
  const type = feature.get("type");

  const iconSrc =
    type === "master"
      ? "images/nodes/master-node.svg"
      : "images/nodes/microchip-solid.svg";

  const baseColor = type === "master" ? "#ff4444" : "#4CAF50";

  if (state === "detecting") {
    return [
      new ol.style.Style({
        image: new ol.style.Circle({
          radius: 34,
          fill: new ol.style.Fill({ color: "rgba(255, 215, 0, 0.25)" }),
          stroke: new ol.style.Stroke({
            color: "#FFD700",
            width: 4
          })
        })
      }),
      new ol.style.Style({
        image: new ol.style.Circle({
          radius: 24,
          fill: new ol.style.Fill({ color: baseColor }),
          stroke: new ol.style.Stroke({
            color: "#ffffff",
            width: 2
          })
        })
      }),
      new ol.style.Style({
        image: new ol.style.Icon({
          src: iconSrc,
          scale: 0.05,
          anchor: [0.5, 0.5],
          anchorXUnits: "fraction",
          anchorYUnits: "fraction"
        })
      })
    ];
  }

  if (state === "detected") {
    return [
      new ol.style.Style({
        image: new ol.style.Circle({
          radius: 30,
          fill: new ol.style.Fill({ color: "#FFD700" }),
          stroke: new ol.style.Stroke({
            color: "#ffffff",
            width: 4
          })
        })
      }),
      new ol.style.Style({
        image: new ol.style.Icon({
          src: iconSrc,
          scale: 0.055,
          anchor: [0.5, 0.5],
          anchorXUnits: "fraction",
          anchorYUnits: "fraction"
        })
      })
    ];
  }

  return [
    new ol.style.Style({
      image: new ol.style.Circle({
        radius: 23,
        fill: new ol.style.Fill({ color: baseColor }),
        stroke: new ol.style.Stroke({
          color: "#ffffff",
          width: 2
        })
      })
    }),
    new ol.style.Style({
      image: new ol.style.Icon({
        src: iconSrc,
        scale: 0.05,
        anchor: [0.5, 0.5],
        anchorXUnits: "fraction",
        anchorYUnits: "fraction"
      })
    })
  ];
}

/* ---------------- DETECTION HANDLER ---------------- */

function resetCurrentHighlightedNode() {
  if (currentHighlightResetTimeout) {
    clearTimeout(currentHighlightResetTimeout);
    currentHighlightResetTimeout = null;
  }

  if (currentHighlightedFeature) {
    currentHighlightedFeature.setStyle(
      getNodeStyles(currentHighlightedFeature, "idle")
    );
    currentHighlightedFeature = null;
  }
}

function handleDetectionEvent(detection) {
  if (!detection || !detection.nodeId) {
    console.warn("Invalid detection event received:", detection);
    return;
  }

  const feature = nodeFeaturesCache.find((f) => f.getId() === detection.nodeId);

  if (!feature) {
    console.warn("Detection node not found:", detection.nodeId);
    showNotification(
      `Detection received, but node ${detection.nodeId} was not found`,
      "error"
    );
    return;
  }

  resetCurrentHighlightedNode();

  const nodeName = feature.get("name") || "Unknown Node";
  const nodeType = feature.get("type") || "unknown";

  feature.setStyle(getNodeStyles(feature, "detecting"));
  currentHighlightedFeature = feature;

  setTimeout(() => {
    feature.setStyle(getNodeStyles(feature, "detected"));

    showNotification(
      `Detection Alert: ${nodeName} (${nodeType}) detected activity`,
      "success"
    );

    flashSidebarDetection(nodeName, nodeType);
    updateDetectionBadge();

    currentHighlightResetTimeout = setTimeout(() => {
      feature.setStyle(getNodeStyles(feature, "idle"));
      currentHighlightedFeature = null;
    }, 2500);
  }, 700);

  console.log("Detection event handled:", detection);
}

window.handleDetectionEvent = handleDetectionEvent;

/* ---------------- SIDEBAR + BADGE ---------------- */

function flashSidebarDetection(nodeName, nodeType) {
  let panel = document.getElementById("latest-detection-panel");

  if (!panel) {
    panel = document.createElement("div");
    panel.id = "latest-detection-panel";
    panel.className = "latest-detection-panel";

    const menu = document.getElementById("menu") || document.body;
    menu.appendChild(panel);
  }

  const time = new Date().toLocaleTimeString();

  panel.innerHTML = `
    <h4>Latest Detection</h4>
    <p><strong>Node:</strong> ${nodeName}</p>
    <p><strong>Type:</strong> ${nodeType}</p>
    <p><strong>Time:</strong> ${time}</p>
  `;

  panel.classList.remove("detection-flash");
  void panel.offsetWidth;
  panel.classList.add("detection-flash");

  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function updateDetectionBadge() {
  let badge = document.getElementById("detection-badge");

  if (!badge) {
    const notificationBtn =
      document.getElementById("notification-menu-button") ||
      document.getElementById("node-popup-menu-btn");

    if (!notificationBtn) return;

    badge = document.createElement("span");
    badge.id = "detection-badge";
    badge.className = "detection-badge";
    badge.innerText = "0";

    notificationBtn.appendChild(badge);
  }

  const currentValue = parseInt(badge.innerText || "0", 10);
  badge.innerText = currentValue + 1;
}

/* ---------------- SIMULATION FALLBACK ---------------- */

function triggerSimulatedDetection(feature) {
  if (!feature) return;

  const fakeDetection = {
    nodeId: feature.getId(),
    nodeName: feature.get("name"),
    nodeType: feature.get("type"),
    timestamp: new Date().toISOString()
  };

  handleDetectionEvent(fakeDetection);
}

function triggerRandomSimulatedDetection() {
  if (!nodeFeaturesCache || nodeFeaturesCache.length === 0) {
    console.warn("No node features available for fallback simulation.");
    return;
  }

  const randomIndex = Math.floor(Math.random() * nodeFeaturesCache.length);
  const randomNodeFeature = nodeFeaturesCache[randomIndex];

  triggerSimulatedDetection(randomNodeFeature);
}

/* ---------------- SIMULATION MODE ---------------- */

function startDetectionSimulation(intervalMs = 3000) {
  stopDetectionSimulation();

  if (!nodeFeaturesCache || nodeFeaturesCache.length === 0) {
    console.warn("No node features available for detection simulation.");
    return;
  }

  detectionSimulationInterval = setInterval(() => {
    triggerRandomSimulatedDetection();
  }, intervalMs);

  console.log(`Detection simulation started with interval ${intervalMs}ms`);
}

/* ---------------- LIVE POLLING MODE + FALLBACK ---------------- */

function startLiveDetectionPolling(intervalMs = 3000) {
  stopDetectionSimulation();

  detectionSimulationInterval = setInterval(async () => {
    try {
      const response = await axios.get("http://localhost:8080/latest_detection");
      const detection = response.data;

      if (detection && detection.nodeId) {
        handleDetectionEvent(detection);
      } else {
        console.warn("No valid live detection received. Using simulation fallback.");
        triggerRandomSimulatedDetection();
      }
    } catch (error) {
      console.warn("Live detection backend not available. Using simulation fallback.");
      triggerRandomSimulatedDetection();
    }
  }, intervalMs);

  console.log(`Live detection polling started with interval ${intervalMs}ms`);
}

/* ---------------- STOP ALL DETECTION ---------------- */

function stopDetectionSimulation() {
  if (detectionSimulationInterval) {
    clearInterval(detectionSimulationInterval);
    detectionSimulationInterval = null;
  }

  resetCurrentHighlightedNode();
  console.log("Detection simulation stopped");
}

window.startDetectionSimulation = startDetectionSimulation;
window.startLiveDetectionPolling = startLiveDetectionPolling;
window.stopDetectionSimulation = stopDetectionSimulation;

/* ---------------- FALLBACK NODES ---------------- */

function getFallbackNodes() {
  return [
    {
      _id: "node-1",
      name: "Node A",
      type: "master",
      model: "Echo Master",
      location: { longitude: 143.509, latitude: -38.673 }
    },
    {
      _id: "node-2",
      name: "Node B",
      type: "sensor",
      model: "Echo Sensor",
      location: { longitude: 143.515, latitude: -38.678 }
    },
    {
      _id: "node-3",
      name: "Node C",
      type: "sensor",
      model: "Echo Sensor",
      location: { longitude: 143.52, latitude: -38.684 }
    },
    {
      _id: "node-4",
      name: "Node D",
      type: "sensor",
      model: "Echo Sensor",
      location: { longitude: 143.525, latitude: -38.679 }
    },
    {
      _id: "node-5",
      name: "Node E",
      type: "sensor",
      model: "Echo Sensor",
      location: { longitude: 143.512, latitude: -38.688 }
    }
  ];
}

/* ---------------- MAP NODE LOADING ---------------- */

async function addIoTNodesToMap(hmiState) {
  if (!hmiState.basemap) {
    console.error("Basemap not initialized");
    return;
  }

  try {
    let nodes = [];

    try {
      const response = await axios.get("http://127.0.0.1:8000/iot/nodes");
      nodes = response.data;
      console.log("Loaded nodes from backend API");
    } catch (apiError) {
      console.warn("Backend /iot/nodes not available. Using fallback nodes instead.");
      nodes = getFallbackNodes();
    }

    const iotLayer = new ol.layer.Vector({
      source: new ol.source.Vector(),
      style: function (feature) {
        if (feature.get("isNode")) {
          return getNodeStyles(feature, "idle");
        }
        return feature.getStyle();
      }
    });

    nodeFeaturesCache = [];

    nodes.forEach((node) => {
      const feature = new ol.Feature({
        geometry: new ol.geom.Point(
          ol.proj.fromLonLat([
            node.location.longitude,
            node.location.latitude
          ])
        ),
        lat: node.location.latitude,
        lon: node.location.longitude,
        type: node.type,
        name: node.name,
        model: node.model,
        isNode: true
      });

      feature.setId(node._id);
      feature.setStyle(getNodeStyles(feature, "idle"));

      iotLayer.getSource().addFeature(feature);
      nodeFeaturesCache.push(feature);
    });

    hmiState.basemap.addLayer(iotLayer);

    const popupElement = document.createElement("div");
    popupElement.className = "node-popup";

    const popup = new ol.Overlay({
      element: popupElement,
      positioning: "bottom-center",
      stopEvent: false
    });

    hmiState.basemap.addOverlay(popup);

    hmiState.basemap.on("pointermove", function (evt) {
      const feature = hmiState.basemap.forEachFeatureAtPixel(
        evt.pixel,
        function (feature) {
          return feature;
        }
      );

      const element = popup.getElement();

      if (
        feature &&
        feature.get("isNode") &&
        feature.get("name") &&
        feature.get("type") &&
        feature.get("model")
      ) {
        const coordinates = feature.getGeometry().getCoordinates();
        popup.setPosition(coordinates);

        element.innerHTML = `
          <div class="node-popup">
            <strong>${feature.get("name")}</strong><br>
            Type: ${feature.get("type")}<br>
            Model: ${feature.get("model")}
          </div>
        `;

        element.style.display = "block";
      } else {
        element.style.display = "none";
      }
    });

    hmiState.basemap.on("singleclick", function (evt) {
      const feature = hmiState.basemap.forEachFeatureAtPixel(
        evt.pixel,
        function (feature) {
          return feature;
        }
      );

      if (feature && feature.get("isNode")) {
        triggerSimulatedDetection(feature);
      }
    });

    // Sprint 2 HD upgrade:
    // First tries live backend. If unavailable, automatically uses simulation fallback.
    startLiveDetectionPolling(3000);

  } catch (error) {
    console.error("Error loading IoT nodes:", error);
    showNotification("Error loading IoT nodes", "error");
  }
}

export {
  addIoTNodesToMap,
  startDetectionSimulation,
  startLiveDetectionPolling,
  stopDetectionSimulation,
  handleDetectionEvent
};