"use strict";

import { showToast } from "./HMI-utils.js";
import { getAudioRecorder } from "./audio_recorder.js";
import {
  retrieveTruthEventsInTimeRange,
  retrieveVocalizationEventsInTimeRange,
  retrieveMicrophones,
  retrieveAudio,
  retrieveSimTime,
  postRecording,
  setSimModeAnimal,
  setSimModeRecording,
  setSimModeRecordingV2,
  stopSimulator
} from "./routes.js";
import { addIoTNodesToMap } from "./nodes-overlay.js";

const EARTH_RADIUS = 6371000;
const MIC_DETECTION_RANGE = 300;
const MAX_RECORDING_TIME_S = "20";
const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;

var audioRecorder = getAudioRecorder();

var statuses = [
  "endangered",
  "vulnerable",
  "near-threatened",
  "normal",
  "invasive"
];

var animalTypes = ["mammal", "bird", "amphibian", "reptile", "insect"];

var statusPrintLookup = {
  endangered: "endangered",
  vulnerable: "vulnerable",
  "near-threatened": "near-threatened",
  normal: "least concern",
  invasive: "invasive"
};

var statusIconLookup = {
  endangered: "1",
  vulnerable: "2",
  "near-threatened": "3",
  normal: "4",
  invasive: "5"
};

var animalTypeIconLookup = {
  mammal: "Mammals",
  bird: "Bird",
  amphibian: "Amphibians",
  insect: "Insects",
  reptile: "Reptiles"
};

var selectedVocalizationEventId = null;
var sample_data = [];
var animal_data = [];
var latestSimAnimals = {};
var current_mic_lat = 0.0;
var current_mic_lon = 0.0;
var current_mic_id = "";

var activeAudioNode = null;
var audioAnimTimeout = null;
var playNextTrack = false;

var micAnimFrameIndex = 1;
var animTimeout = null;
let simUpdateTimeout = null;

var audioRecordStartTime = null;
var durationTimer = null;

var playNextRecordedTrack = false;
var recordingPlaybackAnimTimeout = null;
var audioRecordingElement = null;

var audioContext = null;
var a_source = null;
var decodedAudioStore = null;
var fileContent = null;

export var animal_toggled = false;

/* ----------------------------- DOM HELPERS ----------------------------- */

function getDurationTag() {
  return document.getElementById("recording_duration");
}

function getAudioElement() {
  return document.getElementsByClassName("audio-element")[0] || null;
}

function getAudioElementSource() {
  const audioElement = getAudioElement();
  if (!audioElement) return null;
  return audioElement.getElementsByTagName("source")[0] || null;
}

function getPlaybackIndicator() {
  return document.getElementsByClassName("playback_indicator")[0] || null;
}

function getRecordButton() {
  return document.getElementById("record_audio_button");
}

function getRecordingControls() {
  return document.getElementsByClassName("recording_controls")[0] || null;
}

function getFileInput() {
  return document.getElementById("fileInput");
}

function getAudioElemForRecordedPlayback() {
  return document.getElementById("audioElem");
}

function isJQueryAvailable() {
  return typeof window !== "undefined" && typeof window.$ === "function";
}

function safeShowJQuery(selector) {
  if (isJQueryAvailable()) window.$(selector).show();
}

function safeHideJQuery(selector) {
  if (isJQueryAvailable()) window.$(selector).hide();
}

/* ---------------------- TASK 6.2 SPINNER / ERROR ----------------------- */

let _mapOverlayStylesInjected = false;

function _injectMapOverlayStyles() {
  if (_mapOverlayStylesInjected) return;
  _mapOverlayStylesInjected = true;

  const style = document.createElement("style");
  style.id = "hmi-map-overlay-styles";
  style.textContent = `
    #hmi-map-spinner {
      position: absolute;
      inset: 0;
      z-index: 1000;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      background: rgba(0, 0, 0, 0.45);
      border-radius: inherit;
      pointer-events: all;
    }

    #hmi-map-spinner .hmi-spinner__wheel {
      width: 52px;
      height: 52px;
      border: 5px solid rgba(255,255,255,0.25);
      border-top-color: #17a2b8;
      border-radius: 50%;
      animation: hmi-spin 0.8s linear infinite;
    }

    #hmi-map-spinner .hmi-spinner__label {
      margin-top: 14px;
      color: #fff;
      font-family: 'Segoe UI', Arial, sans-serif;
      font-size: 14px;
      letter-spacing: 0.03em;
    }

    @keyframes hmi-spin {
      to { transform: rotate(360deg); }
    }

    #hmi-map-error {
      position: absolute;
      top: 16px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 1001;
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 20px;
      border-radius: 8px;
      background: #7b1624;
      border: 1px solid #dc3545;
      box-shadow: 0 4px 20px rgba(0,0,0,0.45);
      font-family: 'Segoe UI', Arial, sans-serif;
      font-size: 14px;
      color: #fff;
      max-width: 480px;
      width: calc(100% - 48px);
      pointer-events: all;
    }

    #hmi-map-error .hmi-error__icon {
      font-size: 20px;
      flex-shrink: 0;
    }

    #hmi-map-error .hmi-error__msg {
      flex: 1;
      line-height: 1.4;
    }

    #hmi-map-error .hmi-error__retry {
      flex-shrink: 0;
      padding: 6px 14px;
      background: #dc3545;
      border: none;
      border-radius: 5px;
      color: #fff;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s;
    }

    #hmi-map-error .hmi-error__retry:hover {
      background: #a71d2a;
    }
  `;
  document.head.appendChild(style);
}

function _getMapContainer() {
  return (
    document.getElementById("map") ||
    document.getElementById("mapPanel") ||
    document.querySelector(".map-container") ||
    document.getElementById("basemap") ||
    document.body
  );
}

function _escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function showMapSpinner(label = "Loading map data…") {
  _injectMapOverlayStyles();
  hideMapSpinner();

  const container = _getMapContainer();
  if (getComputedStyle(container).position === "static") {
    container.style.position = "relative";
  }

  const el = document.createElement("div");
  el.id = "hmi-map-spinner";
  el.innerHTML = `
    <div class="hmi-spinner__wheel" aria-hidden="true"></div>
    <p class="hmi-spinner__label">${_escapeHtml(label)}</p>
  `;
  el.setAttribute("role", "status");
  el.setAttribute("aria-live", "polite");
  container.appendChild(el);
}

export function hideMapSpinner() {
  const el = document.getElementById("hmi-map-spinner");
  if (el) el.remove();
}

export function showMapError(message, onRetry) {
  _injectMapOverlayStyles();
  hideMapError();

  const container = _getMapContainer();
  if (getComputedStyle(container).position === "static") {
    container.style.position = "relative";
  }

  const el = document.createElement("div");
  el.id = "hmi-map-error";
  el.setAttribute("role", "alert");
  el.innerHTML = `
    <span class="hmi-error__icon" aria-hidden="true">⚠</span>
    <span class="hmi-error__msg">${_escapeHtml(message)}</span>
    <button class="hmi-error__retry">Retry</button>
  `;
  container.appendChild(el);

  const retryBtn = el.querySelector(".hmi-error__retry");
  if (retryBtn) {
    retryBtn.addEventListener("click", () => {
      hideMapError();
      if (typeof onRetry === "function") onRetry();
    });
  }
}

export function hideMapError() {
  const el = document.getElementById("hmi-map-error");
  if (el) el.remove();
}

/* ------------------------------ UTILITIES ------------------------------ */

function matchStatus(status) {
  return status === "least concern" ? "normal" : status;
}

function getIconName(status, type) {
  return animalTypeIconLookup[type] + statusIconLookup[status] + "-01.png";
}

export function convertCSV(json) {
  if (json == null) return null;
  if (typeof json === "undefined" || json.length === 0) return null;

  let data = json;
  let fields = Object.keys(data[0]);
  let replacer = function (key, value) {
    return value === null ? "N/A" : value;
  };

  let csv = data.map(function (row) {
    return fields
      .map(function (fieldName) {
        return JSON.stringify(row[fieldName], replacer);
      })
      .join(",");
  });

  csv.unshift(fields.join(","));
  return csv.join("\r\n");
}

export function getUTC() {
  const now = new Date();
  return Date.UTC(
    now.getUTCFullYear(),
    now.getUTCMonth(),
    now.getUTCDate(),
    now.getUTCHours(),
    now.getUTCMinutes(),
    now.getUTCSeconds(),
    now.getUTCMilliseconds()
  );
}

function initializeStaticDOMHooks() {
  const audioElement = getAudioElement();
  if (audioElement && !audioElement.dataset.hmiBound) {
    audioElement.onended = hidePlaybackIndicator;
    audioElement.dataset.hmiBound = "true";
  }

  const fileInput = getFileInput();
  if (fileInput && !fileInput.dataset.hmiBound) {
    fileInput.dataset.hmiBound = "true";
    fileInput.addEventListener("change", function (event) {
      const selectedFile = event.target.files[0];
      audioContext = new (window.AudioContext || window.webkitAudioContext)();

      if (selectedFile) {
        const reader = new FileReader();

        reader.onload = function (loadEvent) {
          fileContent = loadEvent.target.result;
          audioContext.decodeAudioData(fileContent.slice(0), function (decodedAudio) {
            decodedAudioStore = decodedAudio;
            a_source = null;

            const audioElementRef = getAudioElement();
            if (audioElementRef) {
              audioElementRef.src = URL.createObjectURL(selectedFile);
            }
          });
        };

        reader.readAsArrayBuffer(selectedFile);
      }
    });
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeStaticDOMHooks, { once: true });
} else {
  initializeStaticDOMHooks();
}

/* ---------------------------- DATA LOADERS ----------------------------- */

fetch("./js/sample_data.json")
  .then((res) => res.json())
  .then((data) => {
    sample_data = data.data;
  })
  .catch((error) => {
    console.error("Failed to load sample_data.json:", error);
    showToast("Failed to load sample data", "error");
  });

/* ---------------------------- INITIALISATION --------------------------- */

export function initialiseHMI(hmiState) {
  console.log("initialising");

  showMapSpinner("Loading map data…");
  hideMapError();

  createBasemap(hmiState);
  addVocalisationLayers(hmiState);
  addTruthLayers(hmiState);

  for (let i = 1; i < 26; i++) {
    addVectorLayerTopDown(hmiState, `mic_layer_${i}`);
    console.log(`Adding microphone layer:${i}`);
  }
  addVectorLayerTopDown(hmiState, "mic_layer");

  addAllTruthFeatures(hmiState);
  addAllVocalizationFeatures(hmiState);
  createMapClickEvent(hmiState);

  retrieveMicrophones()
    .then((res) => {
      hideMapSpinner();
      updateMicrophoneLayer(hmiState, res.data);
      stepMicAnimation(hmiState);
      addIoTNodesToMap(hmiState);
      queueSimUpdate(hmiState);
      showToast("Map data loaded successfully", "success");
    })
    .catch((error) => {
      hideMapSpinner();
      console.error("Error loading microphones:", error);

      const isTimeout = error && error.code === "ECONNABORTED";
      const userMsg = isTimeout
        ? "Request timed out. Check your connection and try again."
        : "Failed to load map data. The server may be unavailable.";

      showMapError(userMsg, () => initialiseHMI(hmiState));
      showToast(userMsg, "error");
    });
}

/* ----------------------------- EMAIL CHECK ----------------------------- */

const validEmailRegex =
  /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$/;

export function emailValidation(inp) {
  let email_id = inp + "-email-inp";
  let error_id = inp + "-email-error";
  let btn_id = inp + "-button";

  let input_ele = document.getElementById(email_id);
  let error_ele = document.getElementById(error_id);
  let btn_ele = document.getElementById(btn_id);

  if (!input_ele || !error_ele || !btn_ele) return;

  console.log("Toggle email: ", input_ele.value);

  if (input_ele.value.match(validEmailRegex)) {
    error_ele.style.display = "none";
    error_ele.innerHTML = "";
    btn_ele.disabled = false;
  } else {
    error_ele.style.display = "block";
    error_ele.innerHTML = "Please insert a valid email address";
    btn_ele.disabled = true;
  }
}

/* ----------------------------- LAYER RESET ----------------------------- */

export function resetWildlifeLayers(hmiState) {
  hmiState.vocalizationEvents = [];
  hmiState.movementEvents = {};
  clearAllVocalizationLayers(hmiState);
  clearAllTruthLayers(hmiState);
}

export function clearAllVocalizationLayers(hmiState) {
  for (let stat of statuses) {
    for (let animalType of animalTypes) {
      let nextName = stat + "_" + animalType;
      let layer = findMapLayerWithName(hmiState, nextName);
      if (layer) {
        layer.getSource().clear();
      }
    }
  }
}

export function clearAllTruthLayers(hmiState) {
  for (let stat of statuses) {
    for (let animalType of animalTypes) {
      let nextName = stat + "_" + animalType + "_truth";
      let layer = findMapLayerWithName(hmiState, nextName);
      if (layer) {
        layer.getSource().clear();
      }
    }
  }
}

export function clearMicrophoneLayer(hmiState) {
  let layer = findMapLayerWithName(hmiState, "mic_layer");
  if (layer) {
    layer.getSource().clear();
  }
}

/* --------------------------- DATA CONVERTERS --------------------------- */

export function convertJSONtoAnimalMovementEvent(hmiState, data) {
  let movementEvent = {};

  movementEvent.animalId = data.animalId;
  movementEvent.eventId = data._id;
  movementEvent.timestamp = Math.floor(
    (getUTC() - hmiState.timeOffset - hmiState.simUpdateDelay) / 1000
  );
  movementEvent.eventTimestamp = data.timestamp;
  movementEvent.speciesScientificName = data.species.toLowerCase();
  movementEvent.speciesIdentificationConfidence = 100.0;
  movementEvent.locationLat = data.animalTrueLLA[0];
  movementEvent.locationLon = data.animalTrueLLA[1];
  movementEvent.locationConfidence = 100.0;
  movementEvent.animalType = data.type.toLowerCase();
  movementEvent.animalStatus = matchStatus(data.status.toLowerCase());
  movementEvent.animalDiet = data.diet.toLowerCase();

  return movementEvent;
}

export function convertJSONtoAnimalVocalizationEvent(hmiState, data) {
  let vocalizationEvent = {};

  vocalizationEvent.timestamp = hmiState.currentTime;
  vocalizationEvent.eventTimestamp = data.timestamp;
  vocalizationEvent.eventId = data._id;
  vocalizationEvent.speciesIdentificationConfidence = data.confidence;
  vocalizationEvent.speciesScientificName = data.species.toLowerCase();
  vocalizationEvent.commonName = data.commonName.toLowerCase();
  vocalizationEvent.animalType = data.type.toLowerCase();
  vocalizationEvent.animalStatus = matchStatus(data.status.toLowerCase());
  vocalizationEvent.animalDiet = data.diet.toLowerCase();
  vocalizationEvent.locationConfidence = 100 - data.animalLLAUncertainty;
  vocalizationEvent.estLat = data.animalEstLLA[0];
  vocalizationEvent.estLon = data.animalEstLLA[1];
  vocalizationEvent.locationLat = data.animalTrueLLA[0];
  vocalizationEvent.locationLon = data.animalTrueLLA[1];
  vocalizationEvent.sensorId = data.sensorId;
  vocalizationEvent.sensorLat = data.microphoneLLA[0];
  vocalizationEvent.sensorLon = data.microphoneLLA[1];

  return vocalizationEvent;
}

export function convertJSONtoMicrophone(hmiState, data) {
  let mic = {};

  if (data.microphoneLLA !== null) {
    mic.id = data._id;
    mic.lat = data.microphoneLLA[0];
    mic.lon = data.microphoneLLA[1];
    return mic;
  }
  return null;
}

/* --------------------------- UPDATE FUNCTIONS -------------------------- */

export function updateAnimalMovementLayerFromPastData(hmiState, results) {
  clearAllTruthLayers(hmiState);
  hmiState.movementEvents = {};
  latestSimAnimals = {};

  let updateDict = {};

  for (let data of results) {
    if (latestSimAnimals.hasOwnProperty(data.animalId)) {
      let entry = latestSimAnimals[data.animalId];
      if (entry.timestamp < data.timestamp) {
        latestSimAnimals[data.animalId] = data;
        updateDict[data.animalId] = true;
      }
    } else {
      latestSimAnimals[data.animalId] = data;
      let event = convertJSONtoAnimalMovementEvent(hmiState, data);
      hmiState.movementEvents[event.animalId] = event;
    }
  }

  for (const key in updateDict) {
    let event = convertJSONtoAnimalMovementEvent(hmiState, latestSimAnimals[key]);
    hmiState.movementEvents[event.animalId] = event;
  }

  addAllTruthFeatures(hmiState);
}

export function updateVocalizationLayerFromPastData(hmiState, results) {
  clearAllVocalizationLayers(hmiState);
  hmiState.vocalizationEvents = [];

  for (let data of results) {
    let event = convertJSONtoAnimalVocalizationEvent(hmiState, data);
    hmiState.vocalizationEvents.push(event);
  }

  addAllVocalizationFeatures(hmiState);
}

export function updateMicrophoneLayer(hmiState, results) {
  clearMicrophoneLayer(hmiState);
  hmiState.microphoneLocations = [];

  for (let data of results) {
    let location = convertJSONtoMicrophone(hmiState, data);
    if (location !== null) {
      hmiState.microphoneLocations.push(location);
    }
  }

  addmicrophones(hmiState);
}

export function updateAnimalMovementLayerFromLiveData(hmiState, results) {
  let newMovementEvents = [];
  let updatedMovementEvents = [];
  let updateDict = {};

  for (let data of results) {
    if (latestSimAnimals.hasOwnProperty(data.animalId)) {
      let entry = latestSimAnimals[data.animalId];
      if (entry.timestamp < data.timestamp) {
        latestSimAnimals[data.animalId] = data;
        updateDict[data.animalId] = true;
      }
    } else {
      latestSimAnimals[data.animalId] = data;
      let event = convertJSONtoAnimalMovementEvent(hmiState, data);
      hmiState.movementEvents[event.animalId] = event;
      newMovementEvents.push(event);
    }
  }

  for (const key in updateDict) {
    let event = convertJSONtoAnimalMovementEvent(hmiState, latestSimAnimals[key]);
    hmiState.movementEvents[event.animalId] = event;
    updatedMovementEvents.push(event);
  }

  for (let evt of updatedMovementEvents) {
    let layerName = deriveTruthLayerName(evt.animalStatus, evt.animalType);
    let layer = findMapLayerWithName(hmiState, layerName);
    if (layer) {
      const featureToPurge = layer.getSource().getFeatureById(evt.animalId);
      if (featureToPurge) {
        layer.getSource().removeFeature(featureToPurge);
      }
    }
  }

  addNewTruthFeatures(hmiState, updatedMovementEvents);
  addNewTruthFeatures(hmiState, newMovementEvents);
}

export function updateVocalizationLayerFromLiveData(hmiState, results) {
  let newVocalizationEvents = [];

  for (let data of results) {
    let event = convertJSONtoAnimalVocalizationEvent(hmiState, data);
    hmiState.vocalizationEvents.push(event);
    newVocalizationEvents.push(event);
  }

  addNewVocalizationFeatures(hmiState, newVocalizationEvents);
}

/* ----------------------------- AUDIO EVENTS ---------------------------- */

export function muteAudioAnimation() {
  const mute_audio = new CustomEvent("muteAnimation", {
    detail: { message: "mute animation" }
  });
  document.dispatchEvent(mute_audio);
}

export function muteRecordingPlaybackAnimation() {
  const mute_audio = new CustomEvent("muteRecordingAnimation", {
    detail: { message: "mute animation" }
  });
  document.dispatchEvent(mute_audio);
}

export function stopAudioPlayback() {
  muteAudioAnimation();

  if (audioAnimTimeout) {
    clearTimeout(audioAnimTimeout);
  }
  if (activeAudioNode != null) {
    console.log("calling stop");
    activeAudioNode.stop();
  }

  activeAudioNode = null;
}

function playAudioString(audioDataString, sampleRate) {
  const audioData = new Uint8Array(
    atob(audioDataString)
      .split("")
      .map((char) => char.charCodeAt(0))
  );

  const localAudioContext = new AudioContext();
  const numChannels = 1;
  const bufferLength = audioData.length / 2;

  const audioBuffer = localAudioContext.createBuffer(
    numChannels,
    bufferLength,
    sampleRate
  );

  audioBuffer.copyToChannel(new Float32Array(audioData.buffer), 0);

  let duration = audioBuffer.duration;

  activeAudioNode = localAudioContext.createBufferSource();
  activeAudioNode.buffer = audioBuffer;
  activeAudioNode.connect(localAudioContext.destination);

  if (playNextTrack) {
    activeAudioNode.start();
    audioAnimTimeout = setTimeout(muteAudioAnimation, duration * 1000);
  }
}

document.addEventListener("playAudio", function () {
  playNextTrack = true;
  if (selectedVocalizationEventId != null) {
    retrieveAudio(selectedVocalizationEventId)
      .then((res) => {
        playAudioString(res.data.audioClip, res.data.sampleRate);
      })
      .catch((error) => {
        console.error("Error loading audio:", error);
        showToast("Failed to load audio clip", "error");
      });
  }
});

document.addEventListener("stopAudio", function () {
  playNextTrack = false;
  stopAudioPlayback();
});

/* --------------------------- MAP LAYER VISIBILITY ---------------------- */

export function updateLayers(hmiState, filterState) {
  for (let stat of statuses) {
    for (let animalType of animalTypes) {
      let layerName = deriveLayerName(stat, animalType);
      let layer = findMapLayerWithName(hmiState, layerName);

      if (layer) {
        if (
          filterState.includes("_" + stat) &&
          filterState.includes("_" + animalType)
        ) {
          layer.setVisible(true);
        } else {
          layer.setVisible(false);
        }
      }
    }
  }

  for (let stat of statuses) {
    for (let animalType of animalTypes) {
      let layerName = deriveTruthLayerName(stat, animalType);
      let layer = findMapLayerWithName(hmiState, layerName);

      if (layer) {
        if (filterState.includes(stat) && filterState.includes(animalType)) {
          layer.setVisible(true);
        } else {
          layer.setVisible(false);
        }
      }
    }
  }
}

/* ------------------------------ MAP FEATURES --------------------------- */

function addAllTruthFeatures(hmiState) {
  for (const key in hmiState.movementEvents) {
    let entry = hmiState.movementEvents[key];

    var iconPath = "";
    if (
      entry.animalDiet === "insectivore" ||
      entry.animalDiet === "omnivore" ||
      entry.animalDiet === "carnivore"
    ) {
      iconPath = "./../images/Predator/sim/" + getIconName(entry.animalStatus, entry.animalType);
    } else {
      iconPath = "./../images/sim/" + getIconName(entry.animalStatus, entry.animalType);
    }

    var trueLocation = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([entry.locationLon, entry.locationLat])
      ),
      name: "trueLocation_" + entry.speciesScientificName,
      animalType: entry.animalType,
      animalStatus: entry.animalStatus,
      animalSpecies: entry.speciesScientificName,
      animalLon: entry.locationLon,
      animalLat: entry.locationLat,
      animalDiet: entry.animalDiet,
      animalConfidence: entry.speciesIdentificationConfidence,
      animalLocConfidence: entry.locationConfidence,
      animalIcon: iconPath,
      animalRecordDate: entry.timestamp,
      isAnimalMovement: 1
    });

    var trueIcon = new ol.style.Style({
      image: new ol.style.Icon({
        src: iconPath,
        anchor: [0.5, 1],
        scale: 0.75,
        className: "true-icon"
      })
    });

    trueLocation.setStyle(trueIcon);
    trueLocation.setId(entry.animalId);

    let layerName = deriveTruthLayerName(entry.animalStatus, entry.animalType);
    let layer = findMapLayerWithName(hmiState, layerName);

    if (layer) {
      layer.getSource().addFeature(trueLocation);
      layer.getSource().changed();
      layer.changed();
    }
  }
}

function addNewTruthFeatures(hmiState, events) {
  for (let entry of events) {
    var iconPath = "";
    if (
      entry.animalDiet === "omnivore" ||
      entry.animalDiet === "carnivore" ||
      entry.animalDiet === "insectivore"
    ) {
      iconPath = "./../images/Predator/sim/" + getIconName(entry.animalStatus, entry.animalType);
    } else {
      iconPath = "./../images/sim/" + getIconName(entry.animalStatus, entry.animalType);
    }

    var trueLocation = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([entry.locationLon, entry.locationLat])
      ),
      name: "trueLocation_" + entry.speciesScientificName,
      animalType: entry.animalType,
      animalStatus: entry.animalStatus,
      animalSpecies: entry.speciesScientificName,
      animalLon: entry.locationLon,
      animalLat: entry.locationLat,
      animalDiet: entry.animalDiet,
      animalConfidence: entry.speciesIdentificationConfidence,
      animalLocConfidence: entry.locationConfidence,
      animalIcon: iconPath,
      animalRecordDate: entry.timestamp,
      isAnimalMovement: 1
    });

    var trueIcon = new ol.style.Style({
      image: new ol.style.Icon({
        src: iconPath,
        anchor: [0.5, 1],
        scale: 0.75,
        className: "true-icon"
      })
    });

    trueLocation.setStyle(trueIcon);
    trueLocation.setId(entry.animalId);

    let layerName = deriveTruthLayerName(entry.animalStatus, entry.animalType);
    let layer = findMapLayerWithName(hmiState, layerName);

    if (layer) {
      layer.getSource().addFeature(trueLocation);
      layer.getSource().changed();
      layer.changed();
    }
  }
}

function addAllVocalizationFeatures(hmiState) {
  for (let entry of hmiState.vocalizationEvents) {
    var iconPath = "";
    if (entry.animalDiet === "herbivore" || entry.animalDiet === "frugivore") {
      iconPath = "./../images/vocalization/" + getIconName(entry.animalStatus, entry.animalType);
    } else {
      iconPath = "./../images/Predator/vocalization/" + getIconName(entry.animalStatus, entry.animalType);
    }

    var evtLocation = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([entry.locationLon, entry.locationLat])
      ),
      name: "vocalisation_" + entry.speciesScientificName,
      animalType: entry.animalType,
      animalStatus: entry.animalStatus,
      animalSpecies: entry.speciesScientificName,
      animalLon: entry.locationLon,
      animalLat: entry.locationLat,
      animalConfidence: entry.speciesIdentificationConfidence,
      animalLocConfidence: entry.locationConfidence,
      animalDiet: entry.animalDiet,
      animalIcon: iconPath,
      animalRecordDate: entry.timestamp,
      eventId: entry.eventId,
      isAnimalMovement: 0
    });

    var icon = new ol.style.Style({
      image: new ol.style.Icon({
        src: iconPath,
        anchor: [0.5, 1],
        scale: 0.75,
        className: "vocalization-icon"
      })
    });

    evtLocation.setStyle(icon);
    evtLocation.setId(entry.eventId);

    let layerName = deriveLayerName(entry.animalStatus, entry.animalType);
    let layer = findMapLayerWithName(hmiState, layerName);

    if (layer) {
      layer.getSource().addFeature(evtLocation);
      layer.getSource().changed();
      layer.changed();
    }
  }
}

function addNewVocalizationFeatures(hmiState, events) {
  for (let entry of events) {
    var iconPath = "";
    if (entry.animalDiet === "herbivore" || entry.animalDiet === "frugivore") {
      iconPath = "./../images/vocalization/" + getIconName(entry.animalStatus, entry.animalType);
    } else {
      iconPath = "./../images/Predator/vocalization/" + getIconName(entry.animalStatus, entry.animalType);
    }

    var evtLocation = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([entry.locationLon, entry.locationLat])
      ),
      name: "vocalisation_" + entry.speciesScientificName,
      animalType: entry.animalType,
      animalStatus: entry.animalStatus,
      animalSpecies: entry.speciesScientificName,
      animalLon: entry.locationLon,
      animalLat: entry.locationLat,
      animalConfidence: entry.speciesIdentificationConfidence,
      animalLocConfidence: entry.locationConfidence,
      animalDiet: entry.animalDiet,
      animalIcon: iconPath,
      animalRecordDate: entry.timestamp,
      eventId: entry.eventId,
      isAnimalMovement: 0
    });

    var icon = new ol.style.Style({
      image: new ol.style.Icon({
        src: iconPath,
        anchor: [0.5, 1],
        scale: 0.75,
        className: "vocalization-icon"
      })
    });

    evtLocation.setStyle(icon);
    evtLocation.setId(entry.eventId);

    let layerName = deriveLayerName(entry.animalStatus, entry.animalType);
    let layer = findMapLayerWithName(hmiState, layerName);

    if (layer) {
      layer.getSource().addFeature(evtLocation);
      layer.getSource().changed();
      layer.changed();
    }
  }
}

/* --------------------------- MICROPHONE LAYERS ------------------------- */

function addMicrophonesByLayer(hmiState, layerName, iconPath) {
  var mics = [];

  hmiState.microphoneLocations.forEach((location) => {
    var mic = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([location.lon, location.lat])
      ),
      name: "mic",
      micLat: location.lat,
      micLon: location.lon,
      micIcon: iconPath,
      id: location.id,
      isMic: 1
    });

    var icon = new ol.style.Style({
      image: new ol.style.Icon({
        src: iconPath,
        anchor: [0.5, 1],
        scale: 0.175
      })
    });

    mic.setStyle(icon);
    mics.push(mic);
  });

  let layer = findMapLayerWithName(hmiState, layerName);
  if (layer) {
    layer.getSource().addFeatures(mics);
    layer.getSource().changed();
    layer.changed();
  }
}

function addMicrophonesByHiddenLayer(hmiState, layerName, iconPath) {
  var mics = [];

  hmiState.microphoneLocations.forEach((location) => {
    var mic = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([location.lon, location.lat])
      ),
      name: "mic"
    });

    var icon = new ol.style.Style({
      image: new ol.style.Icon({
        src: iconPath,
        anchor: [0.5, 1],
        scale: 0.175
      })
    });

    mic.setStyle(icon);
    mics.push(mic);
  });

  let layer = findMapLayerWithName(hmiState, layerName);
  if (layer) {
    layer.getSource().addFeatures(mics);
    layer.getSource().changed();
    layer.changed();
    layer.setVisible(false);
  }
}

function addmicrophones(hmiState) {
  for (let i = 25; i > 0; i--) {
    addMicrophonesByLayer(hmiState, `mic_layer_${i}`, `./../images/${i}-01.png`);
  }
  addMicrophonesByHiddenLayer(hmiState, "mic_layer", "./../images/1-01.png");
}

export function enableMicAnimation(hmiState) {
  var staticLayer = findMapLayerWithName(hmiState, "mic_layer");
  if (staticLayer) staticLayer.setVisible(false);

  for (var i = 1; i <= 25; i++) {
    var nextLayer = findMapLayerWithName(hmiState, "mic_layer_" + i);
    if (nextLayer) nextLayer.setVisible(true);
  }

  stepMicAnimation(hmiState);
}

export function disableMicAnimation(hmiState) {
  if (animTimeout) clearTimeout(animTimeout);

  for (var i = 1; i <= 25; i++) {
    var nextLayer = findMapLayerWithName(hmiState, "mic_layer_" + i);
    if (nextLayer) nextLayer.setVisible(false);
  }
}

function stepMicAnimation(hmiState) {
  var currentIndex = micAnimFrameIndex;
  micAnimFrameIndex = (micAnimFrameIndex % 25) + 1;

  var nextLayer = findMapLayerWithName(hmiState, "mic_layer_" + micAnimFrameIndex);
  var currentLayer = findMapLayerWithName(hmiState, "mic_layer_" + currentIndex);

  if (nextLayer) nextLayer.setVisible(true);
  if (currentLayer) currentLayer.setVisible(false);

  if (animTimeout) clearTimeout(animTimeout);
  animTimeout = setTimeout(stepMicAnimation, 100, hmiState);
}

export function showMics(hmiState) {
  let layer = findMapLayerWithName(hmiState, "mic_layer");
  if (layer) layer.setVisible(true);
}

export function hideMics(hmiState) {
  let layer = findMapLayerWithName(hmiState, "mic_layer");
  if (layer) layer.setVisible(false);
}

/* ----------------------------- MAP HELPERS ----------------------------- */

function findMapLayerWithName(hmiState, name) {
  if (!hmiState.basemap) {
    console.log("findMapLayerWithName: invalid basemap");
    return null;
  }

  if (hmiState.layers && hmiState.layers.hasOwnProperty(name)) {
    return hmiState.layers[name];
  }

  console.log("findMapLayerWithName: layer not found: " + name);
  return null;
}

function addVectorLayerTopDown(hmiState, layerName) {
  addVectorLayerToBasemap(hmiState, layerName, hmiState.layerPool);
  hmiState.layerPool = hmiState.layerPool - 1;
}

function addVectorLayerToBasemap(hmiState, layerName, zIndex) {
  if (!hmiState.basemap) {
    console.log("findMapLayerWithName: invalid basemap");
    return null;
  }

  let layer = new ol.layer.Vector({
    name: layerName,
    source: new ol.source.Vector(),
    visible: true
  });

  if (zIndex != 0) {
    layer.setZIndex(zIndex);
  }

  hmiState.basemap.addLayer(layer);
  hmiState.layers[layerName] = layer;
}

function addVocalisationLayers(hmiState) {
  for (let stat of statuses) {
    for (let animalType of animalTypes) {
      addVectorLayerTopDown(hmiState, stat + "_" + animalType);
    }
  }
}

function addTruthLayers(hmiState) {
  for (let stat of statuses) {
    for (let animalType of animalTypes) {
      addVectorLayerTopDown(hmiState, stat + "_" + animalType + "_truth");
    }
  }
}

function deriveLayerName(status, animalType) {
  return status + "_" + animalType;
}

function deriveTruthLayerName(status, animalType) {
  return status + "_" + animalType + "_truth";
}

function getOlDefaultControls(options) {
  try {
    if (ol && ol.control) {
      if (typeof ol.control.defaults === "function") {
        return ol.control.defaults(options);
      }
      if (
        ol.control.defaults &&
        typeof ol.control.defaults.defaults === "function"
      ) {
        return ol.control.defaults.defaults(options);
      }
    }
  } catch (error) {
    console.error("Failed to resolve OpenLayers controls defaults:", error);
  }
  return undefined;
}

function getOlDefaultInteractions(options) {
  try {
    if (ol && ol.interaction) {
      if (typeof ol.interaction.defaults === "function") {
        return ol.interaction.defaults(options);
      }
      if (
        ol.interaction.defaults &&
        typeof ol.interaction.defaults.defaults === "function"
      ) {
        return ol.interaction.defaults.defaults(options);
      }
    }
  } catch (error) {
    console.error("Failed to resolve OpenLayers interactions defaults:", error);
  }
  return undefined;
}

function createBasemap(hmiState) {
  const controls = getOlDefaultControls({ zoom: false });
  const interactions = getOlDefaultInteractions({ constrainResolution: false });

  var basemap = new ol.Map({
    target: "basemap",
    featureEvents: true,
    controls: controls,
    interactions: interactions,
    layers: [
      new ol.layer.Tile({
        name: "mapTileLayer",
        source: new ol.source.XYZ({
          url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
          maxZoom: 18
        })
      })
    ],
    view: new ol.View({
      center: ol.proj.fromLonLat([hmiState.originLon, hmiState.originLat]),
      zoom: hmiState.defaultZoom
    })
  });

  hmiState.basemap = basemap;
  return basemap;
}

/* ----------------------------- WEATHER DATA ---------------------------- */

async function fetchWeatherData(timestamp, lat, lon) {
  try {
    const response = await fetch(
      `http://localhost:9000/hmi/weather?timestamp=${timestamp}&lat=${lat}&lon=${lon}`
    );
    if (!response.ok) {
      throw new Error("Failed to fetch weather data");
    }
    return await response.json();
  } catch (error) {
    console.error("Error fetching weather data:", error);
    throw error;
  }
}

/* ----------------------------- MAP CLICK UI ---------------------------- */

function createMapClickEvent(hmiState) {
  hmiState.basemap.on("click", function (evt) {
    const feature = hmiState.basemap.forEachFeatureAtPixel(evt.pixel, function (feature) {
      return feature;
    });

    if (!feature) {
      safeHideJQuery("#animal-popup-content");
      safeHideJQuery("#mic-popup-content");
      safeHideJQuery("#node-popup-content");
      safeShowJQuery("#animal-default-content");
      safeShowJQuery("#mic-default-content");
      safeShowJQuery("#node-default-content");
      return;
    }

    if (!isJQueryAvailable()) {
      console.error("jQuery is not loaded.");
      showToast("UI dependency missing: jQuery is not loaded", "error");
      return;
    }

    const active_content = window.$("#animal-popup-content");
    const default_content = window.$("#animal-default-content");
    const active_mic_content = window.$("#mic-popup-content");
    const default_mic_content = window.$("#mic-default-content");
    const default_node_content = window.$("#node-default-content");
    const active_node_content = window.$("#node-popup-content");

    let values = feature.getProperties();

    if (values.hasOwnProperty("animalRecordDate")) {
      fetchWeatherData(values.animalRecordDate, values.animalLat, values.animalLon)
        .then((weatherData) => {
          const key = Object.keys(weatherData.Date)[0];
          const date_ele = document.getElementById("weather_date");
          const mintemp_ele = document.getElementById("weather_mintemp");
          const maxtemp_ele = document.getElementById("weather_maxtemp");
          const rainfall_ele = document.getElementById("weather_rainfall");
          const windspeed_ele = document.getElementById("weather_windspeed");
          const maxhumidity_ele = document.getElementById("weather_maxhumidity");
          const minhumidity_ele = document.getElementById("weather_minhumidity");

          if (date_ele) date_ele.innerHTML = weatherData["Date"][key];
          if (mintemp_ele) mintemp_ele.innerHTML = weatherData["Min Temperature (°C)"][key] + " (°C)";
          if (maxtemp_ele) maxtemp_ele.innerHTML = weatherData["Max Temperature (°C)"][key] + " (°C)";
          if (rainfall_ele) rainfall_ele.innerHTML = weatherData["Rainfall (mm)"][key] + " (mm)";
          if (windspeed_ele) windspeed_ele.innerHTML = weatherData["Wind Speed (m/sec)"][key] + " (m/sec)";
          if (maxhumidity_ele) maxhumidity_ele.innerHTML = weatherData["Max Humidity (%)"][key] + " (%)";
          if (minhumidity_ele) minhumidity_ele.innerHTML = weatherData["Min Humidity (%)"][key] + " (%)";
        })
        .catch((error) => {
          console.error("Error fetching weather data:", error);
          showToast("Weather data failed to load", "error");
        });
    }

    if (values.isMic) {
      active_mic_content.show();
      default_mic_content.hide();
      active_node_content.hide();
      default_node_content.show();
      active_content.hide();
      default_content.show();

      const img = new Image();
      let dice = Math.floor(Math.random() * 4) + 1;
      img.onload = function () {
        const micImg = document.getElementById("mic_desc_img");
        if (micImg) micImg.src = "../../images/bio/mic_bio_" + dice + ".png";
      };
      img.onerror = function () {
        console.log("Mic image does not exist!");
      };
      img.src = "../../images/bio/mic_bio_" + dice + ".png";

      const descId = document.getElementById("mic_desc_id");
      if (descId) descId.innerText = values.id;

      let dateFormat = new Date();
      const micMarkupImg = document.getElementById("mic_markup_img");
      const micMarkupDetails = document.getElementById("mic_markup_details");
      const micMarkupLat = document.getElementById("mic_markup_loc_lat");
      const micMarkupLon = document.getElementById("mic_markup_loc_lon");
      const micMarkupDate = document.getElementById("mic_markup_date");

      if (micMarkupImg) micMarkupImg.src = values.micIcon;
      if (micMarkupDetails) micMarkupDetails.innerHTML = "Microphone";
      if (micMarkupLat) micMarkupLat.innerHTML = values.micLat;
      if (micMarkupLon) micMarkupLon.innerHTML = values.micLon;
      if (micMarkupDate) micMarkupDate.innerHTML = dateFormat.toUTCString();

      current_mic_lat = values.micLat;
      current_mic_lon = values.micLon;
      current_mic_id = values.id;

      animal_toggled = true;
      document.dispatchEvent(
        new CustomEvent("micToggled", {
          detail: { message: "Mic toggled:" }
        })
      );
    } else if (values.isNode) {
      active_node_content.show();
      default_node_content.hide();
      active_mic_content.hide();
      default_mic_content.show();
      active_content.hide();
      default_content.show();

      const nodeLat = document.getElementById("node_markup_loc_lat");
      const nodeLon = document.getElementById("node_markup_loc_lon");
      const nodeName = document.getElementById("node_markup_name");
      const nodeType = document.getElementById("node_markup_type");
      const nodeModel = document.getElementById("node_markup_model");

      if (nodeLat) nodeLat.innerHTML = values.lat;
      if (nodeLon) nodeLon.innerHTML = values.lon;
      if (nodeName) nodeName.innerHTML = values.name;
      if (nodeType) nodeType.innerHTML = values.type;
      if (nodeModel) nodeModel.innerHTML = values.model;

      animal_toggled = true;
      document.dispatchEvent(
        new CustomEvent("nodeToggled", {
          detail: { message: "Node toggled:" }
        })
      );
    } else {
      stopAudioPlayback();

      active_content.show();
      default_content.hide();
      active_mic_content.hide();
      default_mic_content.show();
      active_node_content.hide();
      default_node_content.show();

      if (values.eventId) {
        selectedVocalizationEventId = values.eventId;
      }

      const audioHeader = document.getElementById("audioHeader");
      const audioControl = document.getElementById("audioControl");

      if (values.isAnimalMovement) {
        if (audioHeader) audioHeader.style.display = "none";
        if (audioControl) audioControl.style.display = "none";
      } else {
        if (audioHeader) audioHeader.style.display = "flex";
        if (audioControl) audioControl.style.display = "flex";
      }

      if (values.animalSpecies) {
        var result = sample_data.find(
          ({ species }) => species.toLowerCase() === values.animalSpecies.toLowerCase()
        );

        if (result) {
          const img = new Image();
          img.onload = function () {
            const descImg = document.getElementById("desc_img");
            if (descImg) {
              descImg.src = "../../images/bio/" + result.common.toLowerCase() + "-bio.png";
            }
          };
          img.onerror = function () {
            let dice = Math.floor(Math.random() * 5) + 1;
            const descImg = document.getElementById("desc_img");
            if (descImg) {
              descImg.src = "../../images/bio/not_available_" + dice + "-bio.png";
            }
          };
          img.src = "../../images/bio/" + result.common.toLowerCase() + "-bio.png";

          animal_data = result;

          const descName = document.getElementById("desc_name");
          const descConfidence = document.getElementById("desc_confidence");
          const descSpecies = document.getElementById("desc_species");
          const descSummary = document.getElementById("desc_summary");
          const summary = document.getElementById("desc_details");

          if (descName) descName.innerText = result.common;
          if (descConfidence) descConfidence.innerText = values.animalConfidence + "%";
          if (descSpecies) descSpecies.innerText = result.species;
          if (descSummary) descSummary.innerText = result.summary;

          if (summary) {
            summary.innerHTML = "";
            result.description.forEach((content) => {
              if (content) {
                var p = document.createElement("p");
                p.className = "desc_ul";
                p.innerText = content;
                summary.appendChild(p);
              }
            });
          }
        } else {
          let dice = Math.floor(Math.random() * 5) + 1;
          const descImg = document.getElementById("desc_img");
          const descName = document.getElementById("desc_name");
          const descConfidence = document.getElementById("desc_confidence");
          const descSpecies = document.getElementById("desc_species");
          const descSummary = document.getElementById("desc_summary");
          const summary = document.getElementById("desc_details");

          if (descImg) descImg.src = "../../images/bio/not_available_" + dice + "-bio.png";
          if (descName) descName.innerText = values.animalSpecies;
          if (descConfidence) descConfidence.innerText = values.animalConfidence + "%";
          if (descSpecies) descSpecies.innerText = values.animalSpecies;
          if (descSummary) descSummary.innerText = "Bio data coming soon.";
          if (summary) summary.innerHTML = "";
        }

        let dateFormat = new Date(values.animalRecordDate);
        const markupImg = document.getElementById("markup_img");
        const markupDetails = document.getElementById("markup_details");
        const markupLon = document.getElementById("markup_loc_lon");
        const markupLat = document.getElementById("markup_loc_lat");
        const markupConfidence = document.getElementById("markup_confidence");
        const markupDate = document.getElementById("markup_date");

        if (markupImg) markupImg.src = values.animalIcon;
        if (markupDetails) {
          markupDetails.innerHTML =
            values.animalType + " | " + values.animalDiet + " | " + statusPrintLookup[values.animalStatus];
        }
        if (markupLon) markupLon.innerHTML = values.animalLon;
        if (markupLat) markupLat.innerHTML = values.animalLat;
        if (markupConfidence) markupConfidence.innerHTML = values.animalLocConfidence + "%";
        if (markupDate) markupDate.innerHTML = dateFormat.toUTCString();

        animal_toggled = true;
        document.dispatchEvent(
          new CustomEvent("animalToggled", {
            detail: { message: "Animal toggled:" }
          })
        );
      } else {
        console.log(values);
      }
    }
  });
}

export function MapOpenNav() {
  if (animal_toggled) {
    const menuPanel = document.getElementById("menuPanel");
    if (menuPanel) menuPanel.style.width = "30%";
  }
}

export function getAnimalToggled() {
  return animal_toggled;
}

export function MapCloseNav() {
  const menuPanel = document.getElementById("menuPanel");
  if (menuPanel) menuPanel.style.width = "0";
  animal_toggled = false;
}

/* ------------------------------ LIVE UPDATES --------------------------- */

function updateTruthEvents(hmiState) {
  retrieveTruthEventsInTimeRange(hmiState.currentTime - 5, hmiState.currentTime)
    .then((res) => {
      updateAnimalMovementLayerFromLiveData(hmiState, res.data);
    })
    .catch((error) => {
      console.error("Error loading truth events:", error);
      showToast("Failed to load movement events", "error");
    });
}

function updateVocalizationEvents(hmiState) {
  retrieveVocalizationEventsInTimeRange(hmiState.currentTime - 5, hmiState.currentTime)
    .then((res) => {
      updateVocalizationLayerFromLiveData(hmiState, res.data);
    })
    .catch((error) => {
      console.error("Error loading vocalization events:", error);
      showToast("Failed to load vocalization events", "error");
    });
}

function purgeTruthEvents(hmiState) {
  let persistEvents = {};

  for (const key in hmiState.movementEvents) {
    let event = hmiState.movementEvents[key];

    if (hmiState.liveEventCutoff > event.timestamp) {
      let layerName = deriveTruthLayerName(event.animalStatus, event.animalType);
      let layer = findMapLayerWithName(hmiState, layerName);
      if (layer) {
        const featureToPurge = layer.getSource().getFeatureById(event.animalId);
        if (featureToPurge) {
          layer.getSource().removeFeature(featureToPurge);
        }
      }
    } else {
      persistEvents[event.animalId] = event;
    }
  }

  hmiState.movementEvents = persistEvents;
}

function purgeVocalizationEvents(hmiState) {
  let persistEvents = [];

  for (let event of hmiState.vocalizationEvents) {
    if (hmiState.liveEventCutoff > event.timestamp) {
      let layerName = deriveLayerName(event.animalStatus, event.animalType);
      let layer = findMapLayerWithName(hmiState, layerName);
      if (layer) {
        const featureToPurge = layer.getSource().getFeatureById(event.eventId);
        if (featureToPurge) {
          layer.getSource().removeFeature(featureToPurge);
        }
      }
    } else {
      persistEvents.push(event);
    }
  }

  hmiState.vocalizationEvents = persistEvents;
}

function simulateData(hmiState) {
  queueSimUpdate(hmiState);
}

export function updateTimeOffset(hmiState) {
  return retrieveSimTime()
    .then((res) => {
      let unixMs = Date.parse(res.data.timestamp);
      let newDelay = getUTC() - unixMs + 1000;

      if (isNaN(newDelay)) {
        hmiState.simUpdateDelay = 10000;
      } else {
        hmiState.simUpdateDelay = newDelay;
      }
    })
    .catch((error) => {
      console.error("Failed to update simulation time:", error);
      showToast("Failed to update simulation time", "error");
      hmiState.simUpdateDelay = 10000;
    });
}

function queueSimUpdate(hmiState) {
  updateTimeOffset(hmiState).finally(() => {
    try {
      if (hmiState.liveMode) {
        hmiState.currentTime = Math.floor(
          (getUTC() - hmiState.timeOffset - hmiState.simUpdateDelay) / 1000
        );
        hmiState.liveEventCutoff = Math.floor(
          (getUTC() - hmiState.timeOffset - hmiState.simUpdateDelay - hmiState.liveWindow) / 1000
        );

        purgeTruthEvents(hmiState);
        purgeVocalizationEvents(hmiState);
        updateTruthEvents(hmiState);
        updateVocalizationEvents(hmiState);
        hmiState.previousUpdateTime = hmiState.currentTime;
      }

      if (simUpdateTimeout) {
        clearTimeout(simUpdateTimeout);
      }

      for (let stat of statuses) {
        for (let animalType of animalTypes) {
          let layerName = deriveTruthLayerName(stat, animalType);
          let layer = findMapLayerWithName(hmiState, layerName);
          if (layer) {
            layer.changed();
            layer.getSource().changed();
          }
        }
      }

      simUpdateTimeout = setTimeout(simulateData, hmiState.requestInterval, hmiState);
    } catch (error) {
      console.error("queueSimUpdate failed:", error);
      showToast("Live update failed", "error");
      simUpdateTimeout = setTimeout(simulateData, hmiState.requestInterval, hmiState);
    }
  });
}

/* ---------------------------- RECORDING UI ----------------------------- */

export function showRecordingControls() {
  console.log("showing controls");
  const recordButton = getRecordButton();
  const recordingControls = getRecordingControls();
  if (!recordButton || !recordingControls) return;

  recordButton.style.display = "none";
  recordingControls.classList.remove("hide");
  initializeRecordingDuration();
}

export function hideRecordingControls() {
  console.log("hiding controls");
  const recordButton = getRecordButton();
  const recordingControls = getRecordingControls();
  if (!recordButton || !recordingControls) return;

  recordButton.style.display = "block";
  recordingControls.classList.add("hide");
  clearInterval(durationTimer);
}

export function showRecordingNotSupportedOverlay() {
  // implement if needed
}

function hideRecordingNotSupportedOverlay() {
  // implement if needed
}

export function createSourceForAudioElement() {
  const audioElement = getAudioElement();
  if (!audioElement) return;

  let sourceElement = document.createElement("source");
  audioElement.appendChild(sourceElement);
}

export function showPlaybackIndicator() {
  const indicator = getPlaybackIndicator();
  if (indicator) indicator.classList.remove("hide");
}

export function hidePlaybackIndicator() {
  const indicator = getPlaybackIndicator();
  if (indicator) indicator.classList.add("hide");
}

export function testFunct() {
  console.log("Recording started 1");
}

export function startAudioRecording() {
  console.log("Recording started 2");

  const audioElement = getAudioElement();
  const audioElementSource = getAudioElementSource();

  if (!audioElementSource) {
    // no source yet
  } else if (audioElement && !audioElement.paused) {
    console.log("Paused playback");
    audioElement.pause();
    hidePlaybackIndicator();
  }

  audioRecorder
    .start()
    .then(() => {
      audioRecordStartTime = new Date();
      showRecordingControls();
    })
    .catch((error) => {
      console.log(error.message);
      showToast("Audio recording failed: " + error.message, "error");

      if (
        error.message.includes(
          "mediaDevices API or getUserMedia method is not supported in this browser."
        )
      ) {
        console.log("To record audio, use browsers like Chrome and Firefox.");
        showRecordingNotSupportedOverlay();
      }

      switch (error.name) {
        case "AbortError":
          console.log("An AbortError has occurred.");
          break;
        case "NotAllowedError":
          console.log("User denied microphone permission.");
          showToast("Microphone permission was denied", "error");
          break;
        case "NotFoundError":
          console.log("No microphone device was found.");
          showToast("No microphone device found", "error");
          break;
        case "NotReadableError":
          console.log("Microphone is already in use or not readable.");
          showToast("Microphone is not available right now", "error");
          break;
        case "SecurityError":
          console.log("A SecurityError has occurred.");
          showToast("Microphone access blocked for security reasons", "error");
          break;
        case "TypeError":
          console.log("A TypeError has occurred.");
          break;
        case "InvalidStateError":
          console.log("An InvalidStateError has occurred.");
          break;
        case "UnknownError":
          console.log("An UnknownError has occurred.");
          showToast("Unknown audio recording error", "error");
          break;
        default:
          console.log("An error occurred with the error name " + error.name);
      }
    });
}

export function stopAudioRecording() {
  console.log("Stopped recording...");

  audioRecorder
    .stop()
    .then(() => {
      playAudio();
      hideRecordingControls();
    })
    .catch((error) => {
      showToast("Error stopping recording", "error");
      switch (error.name) {
        case "InvalidStateError":
          console.log("An InvalidStateError has occured.");
          break;
        default:
          console.log("ERROR: " + error.name);
      }
    });
}

export function cancelAudioRecording() {
  console.log("Cancelled recording");
  audioRecorder.cancel();
  hideRecordingControls();
}

document.addEventListener("saveRecording", function () {
  save();
});

document.addEventListener("simulateRecording", function () {
  simulateRecording(window.hmiState);
});

/* -------------------------- RECORDING HELPERS -------------------------- */

function generateRandomCoordinate(latitude, longitude) {
  const latRad = latitude * DEG_TO_RAD;
  const lonRad = longitude * DEG_TO_RAD;

  const dist = Math.random() * MIC_DETECTION_RANGE + 50;
  const theta = Math.random() * 2 * Math.PI;

  const newLatRad = latRad + (dist / EARTH_RADIUS) * Math.cos(theta);
  const newLonRad = lonRad + (dist / EARTH_RADIUS) * Math.sin(theta);

  return { lat: newLatRad * RAD_TO_DEG, lon: newLonRad * RAD_TO_DEG };
}

function arrayBufferToBase64(buffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function simulateRecording(hmiState) {
  if (!decodedAudioStore) {
    console.log("No recording available.");
    showToast("No recording available to simulate", "error");
    return;
  }

  if (!fileContent) {
    console.log("No audio file content available.");
    showToast("No audio file loaded", "error");
    return;
  }

  const base64String = arrayBufferToBase64(fileContent);
  let coords = generateRandomCoordinate(current_mic_lat, current_mic_lon);

  const recordingData = {
    timestamp: Math.floor(
      (getUTC() - hmiState.timeOffset - hmiState.simUpdateDelay) / 1000
    ),
    sensorId: current_mic_id,
    microphoneLLA: [current_mic_lat, current_mic_lon, 0.0],
    animalEstLLA: [coords.lat, coords.lon, 0.0],
    animalTrueLLA: [coords.lat, coords.lon, 0.0],
    animalLLAUncertainty: 50.0,
    audioClip: base64String,
    mode: hmiState.simMode,
    audioFile: hmiState.simMode
  };

  postRecording(recordingData).catch((error) => {
    console.error("Failed to post recording:", error);
    showToast("Failed to submit recording", "error");
  });
}

function save() {
  if (audioRecorder.audioBlobs.length != 0) {
    const base64AudioDataArray = audioRecorder.audioBlobs.map((blob) => {
      const reader = new FileReader();
      return new Promise((resolve) => {
        reader.onloadend = () => {
          resolve(reader.result.split(",")[1]);
        };
        reader.readAsDataURL(blob);
      });
    });

    Promise.all(base64AudioDataArray)
      .then((audioDataArray) => {
        const jsonData = {
          audioBlobs: audioDataArray
        };

        const jsonDataStr = JSON.stringify(jsonData);
        const filename = prompt("Enter a filename for the JSON file:", "data.json");

        if (filename) {
          const blob = new Blob([jsonDataStr], { type: "application/json" });
          const blobURL = URL.createObjectURL(blob);

          const downloadLink = document.createElement("a");
          downloadLink.href = blobURL;
          downloadLink.download = filename;
          downloadLink.textContent = "Download JSON";

          const downloadTarget = document.getElementById("downloadLink");
          if (downloadTarget) {
            downloadTarget.innerHTML = "";
            downloadTarget.appendChild(downloadLink);
          }

          console.log(jsonDataStr);
          downloadLink.click();
        }
      })
      .catch((err) => {
        console.error("Error processing audio blobs: ", err);
        showToast("Audio processing failed", "error");
      });
  }
}

/* -------------------------- RECORDED PLAYBACK -------------------------- */

document.addEventListener("playRecordedAudio", function () {
  playNextRecordedTrack = true;
  playAudio();
});

document.addEventListener("stopRecordedAudio", function () {
  playNextRecordedTrack = false;
  stopRecordingPlayback();
});

function playRecording(recordedChunks) {
  if (recordedChunks.length === 0) {
    console.log("No recording available.");
    return;
  }

  const blob = new Blob(recordedChunks, { type: "audio/wav; codecs=MS_PCM" });

  if (blob.size > 0) {
    const url = URL.createObjectURL(blob);
    audioRecordingElement = getAudioElemForRecordedPlayback();
    if (!audioRecordingElement) return;

    audioRecordingElement.src = url;
    audioRecordingElement.load();

    if (playNextRecordedTrack) {
      recordingPlaybackAnimTimeout = setTimeout(
        muteRecordingPlaybackAnimation,
        10000
      );
      audioRecordingElement.play();
    }
  } else {
    console.log("Recording failed check microphone configuration settings.");
  }
}

function stopRecordingPlayback() {
  muteRecordingPlaybackAnimation();

  if (recordingPlaybackAnimTimeout) {
    clearTimeout(recordingPlaybackAnimTimeout);
  }

  if (a_source == null) {
    if (audioRecordingElement != null) {
      audioRecordingElement.pause();
      audioRecordingElement.currentTime = 0;
    }
  } else {
    a_source.stop();
    a_source = null;
  }
}

export function playAudio() {
  console.log("play");

  if (!decodedAudioStore) {
    playRecording(audioRecorder.audioBlobs);
    return;
  }

  audioContext = new (window.AudioContext || window.webkitAudioContext)();

  if (playNextRecordedTrack) {
    recordingPlaybackAnimTimeout = setTimeout(
      muteRecordingPlaybackAnimation,
      10000
    );

    a_source = audioContext.createBufferSource();
    a_source.buffer = decodedAudioStore;
    a_source.connect(audioContext.destination);
    a_source.start();
  }
}

/* ---------------------------- DURATION TIMER --------------------------- */

export function initializeRecordingDuration() {
  showRecordingDuration("00:00:00");

  durationTimer = setInterval(() => {
    let duration = computeRecordingDuration(audioRecordStartTime);
    console.log("Start time" + audioRecordStartTime);
    console.log("Recording " + duration);
    showRecordingDuration(duration);
  }, 1000);
}

export function showRecordingDuration(duration) {
  const durationTag = getDurationTag();
  if (!durationTag) return;

  durationTag.innerHTML = duration;

  if (checkAudioDurationThreshold(duration)) {
    stopAudioRecording();
  }
}

export function checkAudioDurationThreshold(duration) {
  let timers = duration.split(":");
  return timers[2] === MAX_RECORDING_TIME_S;
}

export function computeRecordingDuration(startTime) {
  let currentTime = new Date();
  let timeDelta = currentTime - startTime;
  let timeDeltaS = timeDelta / 1000;

  let seconds = Math.floor(timeDeltaS % 60);
  seconds = seconds < 10 ? "0" + seconds : seconds;

  let timeDeltaM = Math.floor(timeDeltaS / 60);
  let minutes = timeDeltaM % 60;
  minutes = minutes < 10 ? "0" + minutes : minutes;

  return "00:" + minutes + ":" + seconds;
}

/* ------------------------------ MODE SWITCH ---------------------------- */

document.addEventListener("modeSwitch", function (event) {
  window.hmiState.simMode = event.detail.message;

  if (window.hmiState.simMode == "Animal_Mode") {
    setSimModeAnimal();
  } else if (window.hmiState.simMode == "Recording_Mode") {
    setSimModeRecording();
  } else if (window.hmiState.simMode == "Recording_Mode_V2") {
    setSimModeRecordingV2();
  } else if (window.hmiState.simMode == "Stop") {
    stopSimulator();
  }
});