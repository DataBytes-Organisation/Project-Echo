import { retrieveDetections, getApiErrorMessage } from "./routes.js";

function validMicrophoneLLA(lla) {
  if (Array.isArray(lla)) {
    return lla.length === 3 &&
      lla.every(Number.isFinite) &&
      Math.abs(lla[0]) <= 90 &&
      Math.abs(lla[1]) <= 180;
  }

  if (lla && typeof lla === "object") {
    return Number.isFinite(lla.latitude) &&
      Number.isFinite(lla.longitude) &&
      Math.abs(lla.latitude) <= 90 &&
      Math.abs(lla.longitude) <= 180;
  }

  return false;
}

function getSourceType(event) {
  return event?.source_type ??
    event?.sourceType ??
    "simulator";
}

function formatDetectionValue(value) {
  if (value === null || value === undefined || value === "") {
    return "unavailable";
  }

  return String(value);
}

function formatConfidence(value) {
  if (!Number.isFinite(value)) {
    return "unavailable";
  }

  return `${value}%`;
}

function formatTimestamp(value) {
  if (!value) {
    return "unavailable";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return "unavailable";
  }

  return date.toLocaleString();
}

function formatLocation(lla) {
  if (!lla) {
    return "unavailable";
  }

  if (Array.isArray(lla)) {
    if (lla.length < 2) {
      return "unavailable";
    }

    return `${lla[0]}, ${lla[1]}`;
  }

  if (
    typeof lla === "object" &&
    Number.isFinite(lla.latitude) &&
    Number.isFinite(lla.longitude)
  ) {
    return `${lla.latitude}, ${lla.longitude}`;
  }

  return "unavailable";
}

function addDetailRow(container, label, value) {
  const row = document.createElement("div");

  const labelElement = document.createElement("strong");
  labelElement.textContent = `${label}: `;

  const valueElement = document.createElement("span");
  valueElement.textContent = formatDetectionValue(value);

  row.appendChild(labelElement);
  row.appendChild(valueElement);
  container.appendChild(row);
}

function showDetectionDetails(hmiState, feature) {
  const details = hmiState.realDetectionDetails;

  if (!details || !feature) return;

  details.replaceChildren();

  const title = document.createElement("strong");
  title.textContent = "Detection details";
  details.appendChild(title);

  addDetailRow(
    details,
    "Species",
    feature.get("species")
  );

  addDetailRow(
    details,
    "Confidence",
    formatConfidence(feature.get("confidence"))
  );

  addDetailRow(
    details,
    "Timestamp",
    formatTimestamp(feature.get("timestamp"))
  );

  addDetailRow(
    details,
    "Sensor ID",
    feature.get("sensorId")
  );

  addDetailRow(
    details,
    "Source",
    feature.get("sourceType")
  );

  addDetailRow(
    details,
    "Microphone location",
    formatLocation(feature.get("microphoneLLA"))
  );

  addDetailRow(
    details,
    "Estimated animal location",
    formatLocation(feature.get("animalEstLLA"))
  );

  addDetailRow(
    details,
    "Animal location uncertainty",
    feature.get("animalLLAUncertainty")
  );

  details.hidden = false;
  details.focus();
}

export async function loadRealDetections(hmiState) {
  if (!hmiState.realDetectionLayer) {
    const layer = new ol.layer.Vector({
      source: new ol.source.Vector(),
      style: new ol.style.Style({
        image: new ol.style.Circle({
          radius: 8,
          fill: new ol.style.Fill({
            color: "#007f73",
          }),
          stroke: new ol.style.Stroke({
            color: "#fff",
            width: 2,
          }),
        }),
      }),
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

    const panel = document.createElement("div");
    panel.className = "ol-unselectable ol-control";

    Object.assign(panel.style, {
      bottom: "12px",
      left: "12px",
      maxWidth: "calc(100% - 24px)",
      padding: "8px",
      background: "#fff",
      color: "#222",
      fontSize: "14px",
    });

    const status = document.createElement("span");
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");

    panel.appendChild(status);
    hmiState.realDetectionStatus = status;

    const refresh = document.createElement("button");
    refresh.type = "button";
    refresh.textContent = "Refresh detections";

    Object.assign(refresh.style, {
      width: "auto",
      padding: "0 8px",
    });

    refresh.addEventListener("click", () => {
      void loadRealDetections(hmiState);
    });

    panel.appendChild(refresh);

    const details = document.createElement("div");
    details.setAttribute("role", "region");
    details.setAttribute(
      "aria-label",
      "Real detection details"
    );

    details.tabIndex = -1;
    details.hidden = true;
    details.style.marginTop = "8px";

    panel.appendChild(details);
    hmiState.realDetectionDetails = details;

    const select = new ol.interaction.Select({
      layers: [layer],
    });

    select.on("select", (event) => {
      const selectedFeature = event.selected?.[0];

      if (selectedFeature) {
        showDetectionDetails(
          hmiState,
          selectedFeature
        );
      } else if (hmiState.realDetectionDetails) {
        hmiState.realDetectionDetails.hidden = true;
      }
    });

    hmiState.basemap.addInteraction(select);
    hmiState.realDetectionSelect = select;

    document.addEventListener("keydown", (event) => {
      if (
        event.key === "Escape" &&
        hmiState.realDetectionDetails
      ) {
        hmiState.realDetectionDetails.hidden = true;

        if (hmiState.realDetectionSelect) {
          hmiState.realDetectionSelect
            .getFeatures()
            .clear();
        }
      }
    });

    hmiState.basemap.addControl(
      new ol.control.Control({
        element: panel,
      })
    );
  }

  const request =
    (hmiState.realDetectionRequest || 0) + 1;

  hmiState.realDetectionRequest = request;

  const source =
    hmiState.realDetectionLayer.getSource();

  const status =
    hmiState.realDetectionStatus;

  status.textContent =
    "Loading real-device detections…";

  try {
    const response =
      await retrieveDetections();

    if (
      request !== hmiState.realDetectionRequest
    ) {
      return;
    }

    source.clear();

    if (!Array.isArray(response.data)) {
      status.textContent =
        "The server returned invalid detection data.";

      return;
    }

    let invalid = 0;
    const ids = new Set();

    for (const detection of response.data) {
      const sourceType = getSourceType(detection);

      if (sourceType !== "real") {
        continue;
      }

      if (
        !validMicrophoneLLA(
          detection.microphoneLLA
        )
      ) {
        invalid++;
        continue;
      }

      if (ids.has(detection._id)) {
        continue;
      }

      ids.add(detection._id);

      const lat = Array.isArray(
        detection.microphoneLLA
      )
        ? detection.microphoneLLA[0]
        : detection.microphoneLLA.latitude;

      const lon = Array.isArray(
        detection.microphoneLLA
      )
        ? detection.microphoneLLA[1]
        : detection.microphoneLLA.longitude;

      const feature = new ol.Feature({
        ...detection,

        // Normalize backend field name for the HMI.
        sourceType,

        geometry: new ol.geom.Point(
          ol.proj.fromLonLat([lon, lat])
        ),
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

  } catch (error) {
    if (
      request !== hmiState.realDetectionRequest
    ) {
      return;
    }

    source.clear();

    status.textContent =
      getApiErrorMessage(
        {
          code: error.code,
          response:
            error.response && {
              status: error.response.status,
            },
        },
        "Unable to load detections. Please try again."
      );
  }
}