"use strict";

/**
 * HMI.js
 * OpenLayers map, layer management, live data updates, audio recording.
 *
 * Sprint 1/2 : Core HMI functionality.
 * Task 7     : Imported getApiErrorMessage from HMI-utils.js so all error
 *              messages are formatted consistently through the shared helper.
 *              Imported withRetry for the microphone load in initialiseHMI.
 *              Fixed four == comparisons to === in the modeSwitch listener.
 *              No changes to map logic, layer management, or audio handling.
 */

import { showToast, getApiErrorMessage, withRetry } from "./HMI-utils.js";
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
  stopSimulator,
} from "./routes.js";
import { addIoTNodesToMap } from "./nodes-overlay.js";

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const EARTH_RADIUS        = 6371000;
const MIC_DETECTION_RANGE = 300;
const MAX_RECORDING_TIME_S = "20";
const DEG_TO_RAD          = Math.PI / 180;
const RAD_TO_DEG          = 180 / Math.PI;

// ─────────────────────────────────────────────────────────────────────────────
// Module-level state
// ─────────────────────────────────────────────────────────────────────────────

var audioRecorder = getAudioRecorder();

var statuses    = ["endangered", "vulnerable", "near-threatened", "normal", "invasive"];
var animalTypes = ["mammal", "bird", "amphibian", "reptile", "insect"];

var statusPrintLookup = {
  endangered:       "endangered",
  vulnerable:       "vulnerable",
  "near-threatened":"near-threatened",
  normal:           "least concern",
  invasive:         "invasive",
};

var statusIconLookup = {
  endangered:       "1",
  vulnerable:       "2",
  "near-threatened":"3",
  normal:           "4",
  invasive:         "5",
};

var animalTypeIconLookup = {
  mammal:    "Mammals",
  bird:      "Bird",
  amphibian: "Amphibians",
  insect:    "Insects",
  reptile:   "Reptiles",
};

var selectedVocalizationEventId = null;
var sample_data    = [];
var animal_data    = [];
var latestSimAnimals = {};
var current_mic_lat  = 0.0;
var current_mic_lon  = 0.0;
var current_mic_id   = "";

var activeAudioNode     = null;
var audioAnimTimeout    = null;
var playNextTrack       = false;

var micAnimFrameIndex = 1;
var animTimeout       = null;
let simUpdateTimeout  = null;

var audioRecordStartTime = null;
var durationTimer        = null;

var playNextRecordedTrack         = false;
var recordingPlaybackAnimTimeout  = null;
var audioRecordingElement         = null;

var audioContext    = null;
var a_source        = null;
var decodedAudioStore = null;
var fileContent     = null;

export var animal_toggled = false;

// ─────────────────────────────────────────────────────────────────────────────
// DOM helpers
// ─────────────────────────────────────────────────────────────────────────────

function getDurationTag()       { return document.getElementById("recording_duration"); }
function getAudioElement()      { return document.getElementsByClassName("audio-element")[0] || null; }
function getAudioElementSource(){ const a = getAudioElement(); return a ? a.getElementsByTagName("source")[0] || null : null; }
function getPlaybackIndicator() { return document.getElementsByClassName("playback_indicator")[0] || null; }
function getRecordButton()      { return document.getElementById("record_audio_button"); }
function getRecordingControls() { return document.getElementsByClassName("recording_controls")[0] || null; }
function getFileInput()         { return document.getElementById("fileInput"); }
function getAudioElemForRecordedPlayback() { return document.getElementById("audioElem"); }

function isJQueryAvailable() {
  return typeof window !== "undefined" && typeof window.$ === "function";
}
function safeShowJQuery(selector) { if (isJQueryAvailable()) window.$(selector).show(); }
function safeHideJQuery(selector) { if (isJQueryAvailable()) window.$(selector).hide(); }

// ─────────────────────────────────────────────────────────────────────────────
// Map overlay — spinner and error banner
// (Sprint 1/2 — kept as-is; these are positioned overlays specific to the
//  OpenLayers basemap and are intentionally different from showRetryState,
//  which replaces container content.  getApiErrorMessage is now used to
//  format the error string passed in from call sites.)
// ─────────────────────────────────────────────────────────────────────────────

let _mapOverlayStylesInjected = false;

function _injectMapOverlayStyles() {
  if (_mapOverlayStylesInjected) return;
  _mapOverlayStylesInjected = true;

  const style = document.createElement("style");
  style.id = "hmi-map-overlay-styles";
  style.textContent = `
    #hmi-map-spinner {
      position: absolute; inset: 0; z-index: 1000;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      background: rgba(0,0,0,0.45); border-radius: inherit; pointer-events: all;
    }
    #hmi-map-spinner .hmi-spinner__wheel {
      width: 52px; height: 52px;
      border: 5px solid rgba(255,255,255,0.25); border-top-color: #17a2b8;
      border-radius: 50%; animation: hmi-spin 0.8s linear infinite;
    }
    #hmi-map-spinner .hmi-spinner__label {
      margin-top: 14px; color: #fff;
      font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; letter-spacing: 0.03em;
    }
    @keyframes hmi-spin { to { transform: rotate(360deg); } }

    #hmi-map-error {
      position: absolute; top: 16px; left: 50%; transform: translateX(-50%);
      z-index: 1001; display: flex; align-items: center; gap: 12px;
      padding: 12px 20px; border-radius: 8px;
      background: #7b1624; border: 1px solid #dc3545;
      box-shadow: 0 4px 20px rgba(0,0,0,0.45);
      font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; color: #fff;
      max-width: 480px; width: calc(100% - 48px); pointer-events: all;
    }
    #hmi-map-error .hmi-error__icon  { font-size: 20px; flex-shrink: 0; }
    #hmi-map-error .hmi-error__msg   { flex: 1; line-height: 1.4; }
    #hmi-map-error .hmi-error__retry {
      flex-shrink: 0; padding: 6px 14px; background: #dc3545;
      border: none; border-radius: 5px; color: #fff;
      font-size: 13px; font-weight: 600; cursor: pointer; transition: background 0.15s;
    }
    #hmi-map-error .hmi-error__retry:hover { background: #a71d2a; }
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
  el.setAttribute("role", "status");
  el.setAttribute("aria-live", "polite");
  el.innerHTML = `
    <div class="hmi-spinner__wheel" aria-hidden="true"></div>
    <p class="hmi-spinner__label">${_escapeHtml(label)}</p>
  `;
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

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

function matchStatus(status) {
  return status === "least concern" ? "normal" : status;
}

function getIconName(status, type) {
  return animalTypeIconLookup[type] + statusIconLookup[status] + "-01.png";
}

export function convertCSV(json) {
  if (json == null || typeof json === "undefined" || json.length === 0) return null;

  const fields   = Object.keys(json[0]);
  const replacer = (key, value) => (value === null ? "N/A" : value);

  const csv = json.map((row) =>
    fields.map((field) => JSON.stringify(row[field], replacer)).join(",")
  );
  csv.unshift(fields.join(","));
  return csv.join("\r\n");
}

export function getUTC() {
  const now = new Date();
  return Date.UTC(
    now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(),
    now.getUTCHours(), now.getUTCMinutes(), now.getUTCSeconds(), now.getUTCMilliseconds()
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

// ─────────────────────────────────────────────────────────────────────────────
// Sample data loader
// ─────────────────────────────────────────────────────────────────────────────

fetch("./js/sample_data.json")
  .then((res) => res.json())
  .then((data) => { sample_data = data.data; })
  .catch((error) => {
    console.error("Failed to load sample_data.json:", error);
    showToast("Failed to load sample data", "error");
  });

// ─────────────────────────────────────────────────────────────────────────────
// Initialisation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Initialise the HMI: create the basemap, add all layers, then load
 * microphone data.  The microphone fetch is wrapped in withRetry (imported
 * from HMI-utils) so transient network failures retry automatically before
 * falling back to the map error banner.
 *
 * Task 7: error message now routed through getApiErrorMessage so the wording
 * is consistent with every other error surface in the application.
 */
export function initialiseHMI(hmiState) {
  console.log("initialising");

  showMapSpinner("Loading map data…");
  hideMapError();

  createBasemap(hmiState);
  addVocalisationLayers(hmiState);
  addTruthLayers(hmiState);

  for (let i = 1; i < 26; i++) {
    addVectorLayerTopDown(hmiState, `mic_layer_${i}`);
  }
  addVectorLayerTopDown(hmiState, "mic_layer");

  addAllTruthFeatures(hmiState);
  addAllVocalizationFeatures(hmiState);
  createMapClickEvent(hmiState);

  // Task 7: withRetry wraps the microphone fetch.  routes.js already retries
  // the axios call; this outer withRetry handles cases where the call itself
  // throws before reaching the network (e.g. axios not yet initialised).
  withRetry(() => retrieveMicrophones(), {
    attempts: 3,
    delayMs: 2000,
    retryMessage: "Microphone load failed, retrying",
  })
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

      // Task 7: use getApiErrorMessage instead of manual isTimeout check
      const userMsg = getApiErrorMessage(
        error,
        "Failed to load map data. The server may be unavailable."
      );

      showMapError(userMsg, () => initialiseHMI(hmiState));
      showToast(userMsg, "error");
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Email validation
// ─────────────────────────────────────────────────────────────────────────────

const validEmailRegex =
  /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$/;

export function emailValidation(inp) {
  const email_id = inp + "-email-inp";
  const error_id = inp + "-email-error";
  const btn_id   = inp + "-button";

  const input_ele = document.getElementById(email_id);
  const error_ele = document.getElementById(error_id);
  const btn_ele   = document.getElementById(btn_id);

  if (!input_ele || !error_ele || !btn_ele) return;

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

// ─────────────────────────────────────────────────────────────────────────────
// Layer reset
// ─────────────────────────────────────────────────────────────────────────────

export function resetWildlifeLayers(hmiState) {
  hmiState.vocalizationEvents = [];
  hmiState.movementEvents = {};
  clearAllVocalizationLayers(hmiState);
  clearAllTruthLayers(hmiState);
}

export function clearAllVocalizationLayers(hmiState) {
  for (let stat of statuses)
    for (let animalType of animalTypes) {
      const layer = findMapLayerWithName(hmiState, stat + "_" + animalType);
      if (layer) layer.getSource().clear();
    }
}

export function clearAllTruthLayers(hmiState) {
  for (let stat of statuses)
    for (let animalType of animalTypes) {
      const layer = findMapLayerWithName(hmiState, stat + "_" + animalType + "_truth");
      if (layer) layer.getSource().clear();
    }
}

export function clearMicrophoneLayer(hmiState) {
  const layer = findMapLayerWithName(hmiState, "mic_layer");
  if (layer) layer.getSource().clear();
}

// ─────────────────────────────────────────────────────────────────────────────
// Data converters
// ─────────────────────────────────────────────────────────────────────────────

export function convertJSONtoAnimalMovementEvent(hmiState, data) {
  return {
    animalId:                      data.animalId,
    eventId:                       data._id,
    timestamp:                     Math.floor((getUTC() - hmiState.timeOffset - hmiState.simUpdateDelay) / 1000),
    eventTimestamp:                data.timestamp,
    speciesScientificName:         data.species.toLowerCase(),
    speciesIdentificationConfidence: 100.0,
    locationLat:                   data.animalTrueLLA[0],
    locationLon:                   data.animalTrueLLA[1],
    locationConfidence:            100.0,
    animalType:                    data.type.toLowerCase(),
    animalStatus:                  matchStatus(data.status.toLowerCase()),
    animalDiet:                    data.diet.toLowerCase(),
  };
}

export function convertJSONtoAnimalVocalizationEvent(hmiState, data) {
  return {
    timestamp:                      hmiState.currentTime,
    eventTimestamp:                 data.timestamp,
    eventId:                        data._id,
    speciesIdentificationConfidence:data.confidence,
    speciesScientificName:          data.species.toLowerCase(),
    commonName:                     data.commonName.toLowerCase(),
    animalType:                     data.type.toLowerCase(),
    animalStatus:                   matchStatus(data.status.toLowerCase()),
    animalDiet:                     data.diet.toLowerCase(),
    locationConfidence:             100 - data.animalLLAUncertainty,
    estLat:                         data.animalEstLLA[0],
    estLon:                         data.animalEstLLA[1],
    locationLat:                    data.animalTrueLLA[0],
    locationLon:                    data.animalTrueLLA[1],
    sensorId:                       data.sensorId,
    sensorLat:                      data.microphoneLLA[0],
    sensorLon:                      data.microphoneLLA[1],
  };
}

export function convertJSONtoMicrophone(hmiState, data) {
  if (data.microphoneLLA !== null) {
    return { id: data._id, lat: data.microphoneLLA[0], lon: data.microphoneLLA[1] };
  }
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer update functions
// ─────────────────────────────────────────────────────────────────────────────

export function updateAnimalMovementLayerFromPastData(hmiState, results) {
  clearAllTruthLayers(hmiState);
  hmiState.movementEvents = {};
  latestSimAnimals = {};

  const updateDict = {};

  for (let data of results) {
    if (latestSimAnimals.hasOwnProperty(data.animalId)) {
      if (latestSimAnimals[data.animalId].timestamp < data.timestamp) {
        latestSimAnimals[data.animalId] = data;
        updateDict[data.animalId] = true;
      }
    } else {
      latestSimAnimals[data.animalId] = data;
      const event = convertJSONtoAnimalMovementEvent(hmiState, data);
      hmiState.movementEvents[event.animalId] = event;
    }
  }

  for (const key in updateDict) {
    const event = convertJSONtoAnimalMovementEvent(hmiState, latestSimAnimals[key]);
    hmiState.movementEvents[event.animalId] = event;
  }

  addAllTruthFeatures(hmiState);
}

export function updateVocalizationLayerFromPastData(hmiState, results) {
  clearAllVocalizationLayers(hmiState);
  hmiState.vocalizationEvents = [];

  for (let data of results) {
    hmiState.vocalizationEvents.push(convertJSONtoAnimalVocalizationEvent(hmiState, data));
  }

  addAllVocalizationFeatures(hmiState);
}

export function updateMicrophoneLayer(hmiState, results) {
  clearMicrophoneLayer(hmiState);
  hmiState.microphoneLocations = [];

  for (let data of results) {
    const location = convertJSONtoMicrophone(hmiState, data);
    if (location !== null) hmiState.microphoneLocations.push(location);
  }

  addmicrophones(hmiState);
}

export function updateAnimalMovementLayerFromLiveData(hmiState, results) {
  const newMovementEvents     = [];
  const updatedMovementEvents = [];
  const updateDict            = {};

  for (let data of results) {
    if (latestSimAnimals.hasOwnProperty(data.animalId)) {
      if (latestSimAnimals[data.animalId].timestamp < data.timestamp) {
        latestSimAnimals[data.animalId] = data;
        updateDict[data.animalId] = true;
      }
    } else {
      latestSimAnimals[data.animalId] = data;
      const event = convertJSONtoAnimalMovementEvent(hmiState, data);
      hmiState.movementEvents[event.animalId] = event;
      newMovementEvents.push(event);
    }
  }

  for (const key in updateDict) {
    const event = convertJSONtoAnimalMovementEvent(hmiState, latestSimAnimals[key]);
    hmiState.movementEvents[event.animalId] = event;
    updatedMovementEvents.push(event);
  }

  for (let evt of updatedMovementEvents) {
    const layer = findMapLayerWithName(hmiState, deriveTruthLayerName(evt.animalStatus, evt.animalType));
    if (layer) {
      const f = layer.getSource().getFeatureById(evt.animalId);
      if (f) layer.getSource().removeFeature(f);
    }
  }

  addNewTruthFeatures(hmiState, updatedMovementEvents);
  addNewTruthFeatures(hmiState, newMovementEvents);
}

export function updateVocalizationLayerFromLiveData(hmiState, results) {
  const newEvents = [];
  for (let data of results) {
    const event = convertJSONtoAnimalVocalizationEvent(hmiState, data);
    hmiState.vocalizationEvents.push(event);
    newEvents.push(event);
  }
  addNewVocalizationFeatures(hmiState, newEvents);
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio events
// ─────────────────────────────────────────────────────────────────────────────

export function muteAudioAnimation() {
  document.dispatchEvent(new CustomEvent("muteAnimation", { detail: { message: "mute animation" } }));
}

export function muteRecordingPlaybackAnimation() {
  document.dispatchEvent(new CustomEvent("muteRecordingAnimation", { detail: { message: "mute animation" } }));
}

export function stopAudioPlayback() {
  muteAudioAnimation();
  if (audioAnimTimeout) clearTimeout(audioAnimTimeout);
  if (activeAudioNode !== null) activeAudioNode.stop();
  activeAudioNode = null;
}

function playAudioString(audioDataString, sampleRate) {
  const audioData = new Uint8Array(
    atob(audioDataString).split("").map((char) => char.charCodeAt(0))
  );

  const localAudioContext = new AudioContext();
  const audioBuffer       = localAudioContext.createBuffer(1, audioData.length / 2, sampleRate);
  audioBuffer.copyToChannel(new Float32Array(audioData.buffer), 0);

  activeAudioNode = localAudioContext.createBufferSource();
  activeAudioNode.buffer = audioBuffer;
  activeAudioNode.connect(localAudioContext.destination);

  if (playNextTrack) {
    activeAudioNode.start();
    audioAnimTimeout = setTimeout(muteAudioAnimation, audioBuffer.duration * 1000);
  }
}

document.addEventListener("playAudio", function () {
  playNextTrack = true;
  if (selectedVocalizationEventId !== null) {
    retrieveAudio(selectedVocalizationEventId)
      .then((res) => { playAudioString(res.data.audioClip, res.data.sampleRate); })
      .catch((error) => {
        console.error("Error loading audio:", error);
        showToast(getApiErrorMessage(error, "Failed to load audio clip"), "error");
      });
  }
});

document.addEventListener("stopAudio", function () {
  playNextTrack = false;
  stopAudioPlayback();
});

// ─────────────────────────────────────────────────────────────────────────────
// Layer visibility
// ─────────────────────────────────────────────────────────────────────────────

export function updateLayers(hmiState, filterState) {
  for (let stat of statuses) {
    for (let animalType of animalTypes) {
      const layer = findMapLayerWithName(hmiState, deriveLayerName(stat, animalType));
      if (layer) {
        layer.setVisible(
          filterState.includes("_" + stat) && filterState.includes("_" + animalType)
        );
      }
    }
  }

  for (let stat of statuses) {
    for (let animalType of animalTypes) {
      const layer = findMapLayerWithName(hmiState, deriveTruthLayerName(stat, animalType));
      if (layer) {
        layer.setVisible(
          filterState.includes(stat) && filterState.includes(animalType)
        );
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Map features
// ─────────────────────────────────────────────────────────────────────────────

function _makeAnimalIcon(iconPath) {
  return new ol.style.Style({
    image: new ol.style.Icon({ src: iconPath, anchor: [0.5, 1], scale: 0.75, className: "true-icon" }),
  });
}

function _makeVocalizationIcon(iconPath) {
  return new ol.style.Style({
    image: new ol.style.Icon({ src: iconPath, anchor: [0.5, 1], scale: 0.75, className: "vocalization-icon" }),
  });
}

function _resolveSimIconPath(entry) {
  const isPredator = ["insectivore", "omnivore", "carnivore"].includes(entry.animalDiet);
  return (isPredator ? "./../images/Predator/sim/" : "./../images/sim/") +
    getIconName(entry.animalStatus, entry.animalType);
}

function _resolveVocalizationIconPath(entry) {
  const isHerbivore = ["herbivore", "frugivore"].includes(entry.animalDiet);
  return (isHerbivore ? "./../images/vocalization/" : "./../images/Predator/vocalization/") +
    getIconName(entry.animalStatus, entry.animalType);
}

function _addTruthFeature(hmiState, entry) {
  const iconPath = _resolveSimIconPath(entry);

  const feature = new ol.Feature({
    geometry:          new ol.geom.Point(ol.proj.fromLonLat([entry.locationLon, entry.locationLat])),
    name:              "trueLocation_" + entry.speciesScientificName,
    animalType:        entry.animalType,
    animalStatus:      entry.animalStatus,
    animalSpecies:     entry.speciesScientificName,
    animalLon:         entry.locationLon,
    animalLat:         entry.locationLat,
    animalDiet:        entry.animalDiet,
    animalConfidence:  entry.speciesIdentificationConfidence,
    animalLocConfidence: entry.locationConfidence,
    animalIcon:        iconPath,
    animalRecordDate:  entry.timestamp,
    isAnimalMovement:  1,
  });

  feature.setStyle(_makeAnimalIcon(iconPath));
  feature.setId(entry.animalId);

  const layer = findMapLayerWithName(hmiState, deriveTruthLayerName(entry.animalStatus, entry.animalType));
  if (layer) { layer.getSource().addFeature(feature); layer.getSource().changed(); layer.changed(); }
}

function _addVocalizationFeature(hmiState, entry) {
  const iconPath = _resolveVocalizationIconPath(entry);

  const feature = new ol.Feature({
    geometry:          new ol.geom.Point(ol.proj.fromLonLat([entry.locationLon, entry.locationLat])),
    name:              "vocalisation_" + entry.speciesScientificName,
    animalType:        entry.animalType,
    animalStatus:      entry.animalStatus,
    animalSpecies:     entry.speciesScientificName,
    animalLon:         entry.locationLon,
    animalLat:         entry.locationLat,
    animalConfidence:  entry.speciesIdentificationConfidence,
    animalLocConfidence: entry.locationConfidence,
    animalDiet:        entry.animalDiet,
    animalIcon:        iconPath,
    animalRecordDate:  entry.timestamp,
    eventId:           entry.eventId,
    isAnimalMovement:  0,
  });

  feature.setStyle(_makeVocalizationIcon(iconPath));
  feature.setId(entry.eventId);

  const layer = findMapLayerWithName(hmiState, deriveLayerName(entry.animalStatus, entry.animalType));
  if (layer) { layer.getSource().addFeature(feature); layer.getSource().changed(); layer.changed(); }
}

function addAllTruthFeatures(hmiState) {
  for (const key in hmiState.movementEvents) _addTruthFeature(hmiState, hmiState.movementEvents[key]);
}

function addNewTruthFeatures(hmiState, events) {
  for (let entry of events) _addTruthFeature(hmiState, entry);
}

function addAllVocalizationFeatures(hmiState) {
  for (let entry of hmiState.vocalizationEvents) _addVocalizationFeature(hmiState, entry);
}

function addNewVocalizationFeatures(hmiState, events) {
  for (let entry of events) _addVocalizationFeature(hmiState, entry);
}

// ─────────────────────────────────────────────────────────────────────────────
// Microphone layers
// ─────────────────────────────────────────────────────────────────────────────

function addMicrophonesByLayer(hmiState, layerName, iconPath) {
  const mics = hmiState.microphoneLocations.map((location) => {
    const mic = new ol.Feature({
      geometry: new ol.geom.Point(ol.proj.fromLonLat([location.lon, location.lat])),
      name: "mic", micLat: location.lat, micLon: location.lon,
      micIcon: iconPath, id: location.id, isMic: 1,
    });
    mic.setStyle(new ol.style.Style({
      image: new ol.style.Icon({ src: iconPath, anchor: [0.5, 1], scale: 0.175 }),
    }));
    return mic;
  });

  const layer = findMapLayerWithName(hmiState, layerName);
  if (layer) { layer.getSource().addFeatures(mics); layer.getSource().changed(); layer.changed(); }
}

function addMicrophonesByHiddenLayer(hmiState, layerName, iconPath) {
  const mics = hmiState.microphoneLocations.map((location) => {
    const mic = new ol.Feature({
      geometry: new ol.geom.Point(ol.proj.fromLonLat([location.lon, location.lat])),
      name: "mic",
    });
    mic.setStyle(new ol.style.Style({
      image: new ol.style.Icon({ src: iconPath, anchor: [0.5, 1], scale: 0.175 }),
    }));
    return mic;
  });

  const layer = findMapLayerWithName(hmiState, layerName);
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
  const staticLayer = findMapLayerWithName(hmiState, "mic_layer");
  if (staticLayer) staticLayer.setVisible(false);
  for (let i = 1; i <= 25; i++) {
    const layer = findMapLayerWithName(hmiState, "mic_layer_" + i);
    if (layer) layer.setVisible(true);
  }
  stepMicAnimation(hmiState);
}

export function disableMicAnimation(hmiState) {
  if (animTimeout) clearTimeout(animTimeout);
  for (let i = 1; i <= 25; i++) {
    const layer = findMapLayerWithName(hmiState, "mic_layer_" + i);
    if (layer) layer.setVisible(false);
  }
}

function stepMicAnimation(hmiState) {
  const currentIndex = micAnimFrameIndex;
  micAnimFrameIndex  = (micAnimFrameIndex % 25) + 1;

  const nextLayer    = findMapLayerWithName(hmiState, "mic_layer_" + micAnimFrameIndex);
  const currentLayer = findMapLayerWithName(hmiState, "mic_layer_" + currentIndex);

  if (nextLayer)    nextLayer.setVisible(true);
  if (currentLayer) currentLayer.setVisible(false);

  if (animTimeout) clearTimeout(animTimeout);
  animTimeout = setTimeout(stepMicAnimation, 100, hmiState);
}

export function showMics(hmiState) {
  const layer = findMapLayerWithName(hmiState, "mic_layer");
  if (layer) layer.setVisible(true);
}

export function hideMics(hmiState) {
  const layer = findMapLayerWithName(hmiState, "mic_layer");
  if (layer) layer.setVisible(false);
}

// ─────────────────────────────────────────────────────────────────────────────
// Map layer helpers
// ─────────────────────────────────────────────────────────────────────────────

function findMapLayerWithName(hmiState, name) {
  if (!hmiState.basemap) { console.log("findMapLayerWithName: invalid basemap"); return null; }
  if (hmiState.layers && hmiState.layers.hasOwnProperty(name)) return hmiState.layers[name];
  console.log("findMapLayerWithName: layer not found: " + name);
  return null;
}

function addVectorLayerTopDown(hmiState, layerName) {
  addVectorLayerToBasemap(hmiState, layerName, hmiState.layerPool);
  hmiState.layerPool = hmiState.layerPool - 1;
}

function addVectorLayerToBasemap(hmiState, layerName, zIndex) {
  if (!hmiState.basemap) { console.log("addVectorLayerToBasemap: invalid basemap"); return null; }

  const layer = new ol.layer.Vector({ name: layerName, source: new ol.source.Vector(), visible: true });
  if (zIndex !== 0) layer.setZIndex(zIndex);
  hmiState.basemap.addLayer(layer);
  hmiState.layers[layerName] = layer;
}

function addVocalisationLayers(hmiState) {
  for (let stat of statuses)
    for (let animalType of animalTypes)
      addVectorLayerTopDown(hmiState, stat + "_" + animalType);
}

function addTruthLayers(hmiState) {
  for (let stat of statuses)
    for (let animalType of animalTypes)
      addVectorLayerTopDown(hmiState, stat + "_" + animalType + "_truth");
}

function deriveLayerName(status, animalType)      { return status + "_" + animalType; }
function deriveTruthLayerName(status, animalType) { return status + "_" + animalType + "_truth"; }

function getOlDefaultControls(options) {
  try {
    if (ol && ol.control) {
      if (typeof ol.control.defaults === "function") return ol.control.defaults(options);
      if (ol.control.defaults && typeof ol.control.defaults.defaults === "function")
        return ol.control.defaults.defaults(options);
    }
  } catch (error) { console.error("Failed to resolve OpenLayers controls defaults:", error); }
  return undefined;
}

function getOlDefaultInteractions(options) {
  try {
    if (ol && ol.interaction) {
      if (typeof ol.interaction.defaults === "function") return ol.interaction.defaults(options);
      if (ol.interaction.defaults && typeof ol.interaction.defaults.defaults === "function")
        return ol.interaction.defaults.defaults(options);
    }
  } catch (error) { console.error("Failed to resolve OpenLayers interactions defaults:", error); }
  return undefined;
}

function createBasemap(hmiState) {
  const basemap = new ol.Map({
    target: "basemap",
    featureEvents: true,
    controls:     getOlDefaultControls({ zoom: false }),
    interactions: getOlDefaultInteractions({ constrainResolution: false }),
    layers: [
      new ol.layer.Tile({
        name: "mapTileLayer",
        source: new ol.source.XYZ({
          url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
          maxZoom: 18,
        }),
      }),
    ],
    view: new ol.View({
      center: ol.proj.fromLonLat([hmiState.originLon, hmiState.originLat]),
      zoom:   hmiState.defaultZoom,
    }),
  });

  hmiState.basemap = basemap;
  return basemap;
}

// ─────────────────────────────────────────────────────────────────────────────
// Weather data
// ─────────────────────────────────────────────────────────────────────────────

async function fetchWeatherData(timestamp, lat, lon) {
  const response = await fetch(
    `http://localhost:9001/hmi/weather?timestamp=${timestamp}&lat=${lat}&lon=${lon}`
  );
  if (!response.ok) throw new Error("Failed to fetch weather data");
  return response.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// Map click handler
// ─────────────────────────────────────────────────────────────────────────────

function createMapClickEvent(hmiState) {
  hmiState.basemap.on("click", function (evt) {
    const feature = hmiState.basemap.forEachFeatureAtPixel(evt.pixel, (f) => f);

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

    const active_content      = window.$("#animal-popup-content");
    const default_content     = window.$("#animal-default-content");
    const active_mic_content  = window.$("#mic-popup-content");
    const default_mic_content = window.$("#mic-default-content");
    const default_node_content = window.$("#node-default-content");
    const active_node_content  = window.$("#node-popup-content");

    const values = feature.getProperties();

    if (values.hasOwnProperty("animalRecordDate")) {
      fetchWeatherData(values.animalRecordDate, values.animalLat, values.animalLon)
        .then((weatherData) => {
          const key = Object.keys(weatherData.Date)[0];
          const fields = {
            weather_date:         weatherData["Date"][key],
            weather_mintemp:      weatherData["Min Temperature (°C)"][key] + " (°C)",
            weather_maxtemp:      weatherData["Max Temperature (°C)"][key] + " (°C)",
            weather_rainfall:     weatherData["Rainfall (mm)"][key] + " (mm)",
            weather_windspeed:    weatherData["Wind Speed (m/sec)"][key] + " (m/sec)",
            weather_maxhumidity:  weatherData["Max Humidity (%)"][key] + " (%)",
            weather_minhumidity:  weatherData["Min Humidity (%)"][key] + " (%)",
          };
          for (const [id, val] of Object.entries(fields)) {
            const el = document.getElementById(id);
            if (el) el.innerHTML = val;
          }
        })
        .catch((error) => {
          console.error("Error fetching weather data:", error);
          showToast(getApiErrorMessage(error, "Weather data failed to load"), "error");
        });
    }

    if (values.isMic) {
      active_mic_content.show();  default_mic_content.hide();
      active_node_content.hide(); default_node_content.show();
      active_content.hide();      default_content.show();

      const img  = new Image();
      const dice = Math.floor(Math.random() * 4) + 1;
      img.onload  = () => { const el = document.getElementById("mic_desc_img"); if (el) el.src = "../../images/bio/mic_bio_" + dice + ".png"; };
      img.onerror = () => console.log("Mic image does not exist!");
      img.src = "../../images/bio/mic_bio_" + dice + ".png";

      const descId = document.getElementById("mic_desc_id");
      if (descId) descId.innerText = values.id;

      const dateFormat = new Date();
      const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.innerHTML = val; };
      setEl("mic_markup_img",      ""); // src set via DOM
      const micImg = document.getElementById("mic_markup_img"); if (micImg) micImg.src = values.micIcon;
      setEl("mic_markup_details",  "Microphone");
      setEl("mic_markup_loc_lat",  values.micLat);
      setEl("mic_markup_loc_lon",  values.micLon);
      setEl("mic_markup_date",     dateFormat.toUTCString());

      current_mic_lat = values.micLat;
      current_mic_lon = values.micLon;
      current_mic_id  = values.id;

      animal_toggled = true;
      document.dispatchEvent(new CustomEvent("micToggled", { detail: { message: "Mic toggled:" } }));

    } else if (values.isNode) {
      active_node_content.show();  default_node_content.hide();
      active_mic_content.hide();   default_mic_content.show();
      active_content.hide();       default_content.show();

      const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.innerHTML = val; };
      setEl("node_markup_loc_lat", values.lat);
      setEl("node_markup_loc_lon", values.lon);
      setEl("node_markup_name",    values.name);
      setEl("node_markup_type",    values.type);
      setEl("node_markup_model",   values.model);

      animal_toggled = true;
      document.dispatchEvent(new CustomEvent("nodeToggled", { detail: { message: "Node toggled:" } }));

    } else {
      stopAudioPlayback();

      active_content.show();       default_content.hide();
      active_mic_content.hide();   default_mic_content.show();
      active_node_content.hide();  default_node_content.show();

      if (values.eventId) selectedVocalizationEventId = values.eventId;

      const audioHeader  = document.getElementById("audioHeader");
      const audioControl = document.getElementById("audioControl");
      if (values.isAnimalMovement) {
        if (audioHeader)  audioHeader.style.display  = "none";
        if (audioControl) audioControl.style.display = "none";
      } else {
        if (audioHeader)  audioHeader.style.display  = "flex";
        if (audioControl) audioControl.style.display = "flex";
      }

      if (values.animalSpecies) {
        const result = sample_data.find(
          ({ species }) => species.toLowerCase() === values.animalSpecies.toLowerCase()
        );

        const dice = Math.floor(Math.random() * 5) + 1;

        if (result) {
          const img = new Image();
          img.onload  = () => { const el = document.getElementById("desc_img"); if (el) el.src = "../../images/bio/" + result.common.toLowerCase() + "-bio.png"; };
          img.onerror = () => { const el = document.getElementById("desc_img"); if (el) el.src = "../../images/bio/not_available_" + dice + "-bio.png"; };
          img.src = "../../images/bio/" + result.common.toLowerCase() + "-bio.png";

          animal_data = result;

          const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.innerText = val; };
          setEl("desc_name",       result.common);
          setEl("desc_confidence", values.animalConfidence + "%");
          setEl("desc_species",    result.species);
          setEl("desc_summary",    result.summary);

          const summary = document.getElementById("desc_details");
          if (summary) {
            summary.innerHTML = "";
            result.description.forEach((content) => {
              if (content) {
                const p = document.createElement("p");
                p.className = "desc_ul";
                p.innerText = content;
                summary.appendChild(p);
              }
            });
          }
        } else {
          const descImg = document.getElementById("desc_img");
          if (descImg) descImg.src = "../../images/bio/not_available_" + dice + "-bio.png";
          const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.innerText = val; };
          setEl("desc_name",       values.animalSpecies);
          setEl("desc_confidence", values.animalConfidence + "%");
          setEl("desc_species",    values.animalSpecies);
          setEl("desc_summary",    "Bio data coming soon.");
          const summary = document.getElementById("desc_details");
          if (summary) summary.innerHTML = "";
        }

        const dateFormat = new Date(values.animalRecordDate);
        const markupImg  = document.getElementById("markup_img");
        if (markupImg) markupImg.src = values.animalIcon;
        const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.innerHTML = val; };
        setEl("markup_details",   values.animalType + " | " + values.animalDiet + " | " + statusPrintLookup[values.animalStatus]);
        setEl("markup_loc_lon",   values.animalLon);
        setEl("markup_loc_lat",   values.animalLat);
        setEl("markup_confidence",values.animalLocConfidence + "%");
        setEl("markup_date",      dateFormat.toUTCString());

        animal_toggled = true;
        document.dispatchEvent(new CustomEvent("animalToggled", { detail: { message: "Animal toggled:" } }));
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

export function getAnimalToggled() { return animal_toggled; }

export function MapCloseNav() {
  const menuPanel = document.getElementById("menuPanel");
  if (menuPanel) menuPanel.style.width = "0";
  animal_toggled = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Live updates
// ─────────────────────────────────────────────────────────────────────────────

function updateTruthEvents(hmiState) {
  retrieveTruthEventsInTimeRange(hmiState.currentTime - 5, hmiState.currentTime)
    .then((res) => { updateAnimalMovementLayerFromLiveData(hmiState, res.data); })
    .catch((error) => {
      console.error("Error loading truth events:", error);
      showToast(getApiErrorMessage(error, "Failed to load movement events"), "error");
    });
}

function updateVocalizationEvents(hmiState) {
  retrieveVocalizationEventsInTimeRange(hmiState.currentTime - 5, hmiState.currentTime)
    .then((res) => { updateVocalizationLayerFromLiveData(hmiState, res.data); })
    .catch((error) => {
      console.error("Error loading vocalization events:", error);
      showToast(getApiErrorMessage(error, "Failed to load vocalization events"), "error");
    });
}

function purgeTruthEvents(hmiState) {
  const persistEvents = {};

  for (const key in hmiState.movementEvents) {
    const event = hmiState.movementEvents[key];
    if (hmiState.liveEventCutoff > event.timestamp) {
      const layer = findMapLayerWithName(hmiState, deriveTruthLayerName(event.animalStatus, event.animalType));
      if (layer) {
        const f = layer.getSource().getFeatureById(event.animalId);
        if (f) layer.getSource().removeFeature(f);
      }
    } else {
      persistEvents[event.animalId] = event;
    }
  }

  hmiState.movementEvents = persistEvents;
}

function purgeVocalizationEvents(hmiState) {
  const persistEvents = [];

  for (let event of hmiState.vocalizationEvents) {
    if (hmiState.liveEventCutoff > event.timestamp) {
      const layer = findMapLayerWithName(hmiState, deriveLayerName(event.animalStatus, event.animalType));
      if (layer) {
        const f = layer.getSource().getFeatureById(event.eventId);
        if (f) layer.getSource().removeFeature(f);
      }
    } else {
      persistEvents.push(event);
    }
  }

  hmiState.vocalizationEvents = persistEvents;
}

function simulateData(hmiState) { queueSimUpdate(hmiState); }

export function updateTimeOffset(hmiState) {
  return retrieveSimTime()
    .then((res) => {
      const unixMs   = Date.parse(res.data.timestamp);
      const newDelay = getUTC() - unixMs + 1000;
      hmiState.simUpdateDelay = isNaN(newDelay) ? 10000 : newDelay;
    })
    .catch((error) => {
      console.error("Failed to update simulation time:", error);
      showToast(getApiErrorMessage(error, "Failed to update simulation time"), "error");
      hmiState.simUpdateDelay = 10000;
    });
}

function queueSimUpdate(hmiState) {
  updateTimeOffset(hmiState).finally(() => {
    try {
      if (hmiState.liveMode) {
        hmiState.currentTime     = Math.floor((getUTC() - hmiState.timeOffset - hmiState.simUpdateDelay) / 1000);
        hmiState.liveEventCutoff = Math.floor((getUTC() - hmiState.timeOffset - hmiState.simUpdateDelay - hmiState.liveWindow) / 1000);

        purgeTruthEvents(hmiState);
        purgeVocalizationEvents(hmiState);
        updateTruthEvents(hmiState);
        updateVocalizationEvents(hmiState);
        hmiState.previousUpdateTime = hmiState.currentTime;
      }

      if (simUpdateTimeout) clearTimeout(simUpdateTimeout);

      for (let stat of statuses) {
        for (let animalType of animalTypes) {
          const layer = findMapLayerWithName(hmiState, deriveTruthLayerName(stat, animalType));
          if (layer) { layer.changed(); layer.getSource().changed(); }
        }
      }

      simUpdateTimeout = setTimeout(simulateData, hmiState.requestInterval, hmiState);
    } catch (error) {
      console.error("queueSimUpdate failed:", error);
      showToast(getApiErrorMessage(error, "Live update failed"), "error");
      simUpdateTimeout = setTimeout(simulateData, hmiState.requestInterval, hmiState);
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Recording UI
// ─────────────────────────────────────────────────────────────────────────────

export function showRecordingControls() {
  const recordButton     = getRecordButton();
  const recordingControls = getRecordingControls();
  if (!recordButton || !recordingControls) return;
  recordButton.style.display = "none";
  recordingControls.classList.remove("hide");
  initializeRecordingDuration();
}

export function hideRecordingControls() {
  const recordButton     = getRecordButton();
  const recordingControls = getRecordingControls();
  if (!recordButton || !recordingControls) return;
  recordButton.style.display = "block";
  recordingControls.classList.add("hide");
  clearInterval(durationTimer);
}

export function showRecordingNotSupportedOverlay() { /* implement if needed */ }

export function createSourceForAudioElement() {
  const audioElement = getAudioElement();
  if (!audioElement) return;
  audioElement.appendChild(document.createElement("source"));
}

export function showPlaybackIndicator() {
  const indicator = getPlaybackIndicator();
  if (indicator) indicator.classList.remove("hide");
}

export function hidePlaybackIndicator() {
  const indicator = getPlaybackIndicator();
  if (indicator) indicator.classList.add("hide");
}

export function testFunct() { console.log("Recording started 1"); }

export function startAudioRecording() {
  const audioElement       = getAudioElement();
  const audioElementSource = getAudioElementSource();

  if (audioElementSource && audioElement && !audioElement.paused) {
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
      showToast("Audio recording failed: " + error.message, "error");

      if (error.message.includes("mediaDevices API or getUserMedia method is not supported in this browser.")) {
        showRecordingNotSupportedOverlay();
      }

      const toastMap = {
        NotAllowedError:  "Microphone permission was denied",
        NotFoundError:    "No microphone device found",
        NotReadableError: "Microphone is not available right now",
        SecurityError:    "Microphone access blocked for security reasons",
        UnknownError:     "Unknown audio recording error",
      };

      if (toastMap[error.name]) showToast(toastMap[error.name], "error");
    });
}

export function stopAudioRecording() {
  audioRecorder
    .stop()
    .then(() => { playAudio(); hideRecordingControls(); })
    .catch((error) => {
      showToast("Error stopping recording", "error");
      console.log("Stop recording error:", error.name);
    });
}

export function cancelAudioRecording() {
  audioRecorder.cancel();
  hideRecordingControls();
}

document.addEventListener("saveRecording",     function () { save(); });
document.addEventListener("simulateRecording", function () { simulateRecording(window.hmiState); });

// ─────────────────────────────────────────────────────────────────────────────
// Recording helpers
// ─────────────────────────────────────────────────────────────────────────────

function generateRandomCoordinate(latitude, longitude) {
  const latRad = latitude  * DEG_TO_RAD;
  const lonRad = longitude * DEG_TO_RAD;
  const dist   = Math.random() * MIC_DETECTION_RANGE + 50;
  const theta  = Math.random() * 2 * Math.PI;
  return {
    lat: (latRad + (dist / EARTH_RADIUS) * Math.cos(theta)) * RAD_TO_DEG,
    lon: (lonRad + (dist / EARTH_RADIUS) * Math.sin(theta)) * RAD_TO_DEG,
  };
}

function arrayBufferToBase64(buffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function simulateRecording(hmiState) {
  if (!decodedAudioStore) {
    showToast("No recording available to simulate", "error");
    return;
  }
  if (!fileContent) {
    showToast("No audio file loaded", "error");
    return;
  }

  const coords = generateRandomCoordinate(current_mic_lat, current_mic_lon);
  const recordingData = {
    timestamp:           Math.floor((getUTC() - hmiState.timeOffset - hmiState.simUpdateDelay) / 1000),
    sensorId:            current_mic_id,
    microphoneLLA:       [current_mic_lat, current_mic_lon, 0.0],
    animalEstLLA:        [coords.lat, coords.lon, 0.0],
    animalTrueLLA:       [coords.lat, coords.lon, 0.0],
    animalLLAUncertainty:50.0,
    audioClip:           arrayBufferToBase64(fileContent),
    mode:                hmiState.simMode,
    audioFile:           hmiState.simMode,
  };

  postRecording(recordingData).catch((error) => {
    console.error("Failed to post recording:", error);
    showToast(getApiErrorMessage(error, "Failed to submit recording"), "error");
  });
}

function save() {
  if (audioRecorder.audioBlobs.length === 0) return;

  Promise.all(
    audioRecorder.audioBlobs.map((blob) =>
      new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result.split(",")[1]);
        reader.readAsDataURL(blob);
      })
    )
  )
    .then((audioDataArray) => {
      const jsonDataStr = JSON.stringify({ audioBlobs: audioDataArray });
      const filename    = prompt("Enter a filename for the JSON file:", "data.json");

      if (filename) {
        const blobURL      = URL.createObjectURL(new Blob([jsonDataStr], { type: "application/json" }));
        const downloadLink = document.createElement("a");
        downloadLink.href     = blobURL;
        downloadLink.download = filename;
        downloadLink.textContent = "Download JSON";

        const downloadTarget = document.getElementById("downloadLink");
        if (downloadTarget) { downloadTarget.innerHTML = ""; downloadTarget.appendChild(downloadLink); }
        downloadLink.click();
      }
    })
    .catch((err) => {
      console.error("Error processing audio blobs:", err);
      showToast("Audio processing failed", "error");
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Recorded audio playback
// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener("playRecordedAudio",  function () { playNextRecordedTrack = true;  playAudio(); });
document.addEventListener("stopRecordedAudio",  function () { playNextRecordedTrack = false; stopRecordingPlayback(); });

function playRecording(recordedChunks) {
  if (recordedChunks.length === 0) return;

  const blob = new Blob(recordedChunks, { type: "audio/wav; codecs=MS_PCM" });
  if (blob.size === 0) return;

  audioRecordingElement = getAudioElemForRecordedPlayback();
  if (!audioRecordingElement) return;

  audioRecordingElement.src = URL.createObjectURL(blob);
  audioRecordingElement.load();

  if (playNextRecordedTrack) {
    recordingPlaybackAnimTimeout = setTimeout(muteRecordingPlaybackAnimation, 10000);
    audioRecordingElement.play();
  }
}

function stopRecordingPlayback() {
  muteRecordingPlaybackAnimation();
  if (recordingPlaybackAnimTimeout) clearTimeout(recordingPlaybackAnimTimeout);

  if (a_source === null) {
    if (audioRecordingElement !== null) {
      audioRecordingElement.pause();
      audioRecordingElement.currentTime = 0;
    }
  } else {
    a_source.stop();
    a_source = null;
  }
}

export function playAudio() {
  if (!decodedAudioStore) { playRecording(audioRecorder.audioBlobs); return; }

  audioContext = new (window.AudioContext || window.webkitAudioContext)();

  if (playNextRecordedTrack) {
    recordingPlaybackAnimTimeout = setTimeout(muteRecordingPlaybackAnimation, 10000);
    a_source = audioContext.createBufferSource();
    a_source.buffer = decodedAudioStore;
    a_source.connect(audioContext.destination);
    a_source.start();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Recording duration timer
// ─────────────────────────────────────────────────────────────────────────────

export function initializeRecordingDuration() {
  showRecordingDuration("00:00:00");
  durationTimer = setInterval(() => {
    showRecordingDuration(computeRecordingDuration(audioRecordStartTime));
  }, 1000);
}

export function showRecordingDuration(duration) {
  const tag = getDurationTag();
  if (!tag) return;
  tag.innerHTML = duration;
  if (checkAudioDurationThreshold(duration)) stopAudioRecording();
}

export function checkAudioDurationThreshold(duration) {
  return duration.split(":")[2] === MAX_RECORDING_TIME_S;
}

export function computeRecordingDuration(startTime) {
  const delta = (new Date() - startTime) / 1000;
  const secs  = Math.floor(delta % 60);
  const mins  = Math.floor(delta / 60) % 60;
  return "00:" + String(mins).padStart(2, "0") + ":" + String(secs).padStart(2, "0");
}

// ─────────────────────────────────────────────────────────────────────────────
// Simulator mode switch
// Task 7: fixed == to === in all four comparisons
// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener("modeSwitch", function (event) {
  window.hmiState.simMode = event.detail.message;

  if (window.hmiState.simMode === "Animal_Mode")      setSimModeAnimal();
  else if (window.hmiState.simMode === "Recording_Mode")   setSimModeRecording();
  else if (window.hmiState.simMode === "Recording_Mode_V2") setSimModeRecordingV2();
  else if (window.hmiState.simMode === "Stop")         stopSimulator();
});
