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

import { showToast, getApiErrorMessage, withRetry, showPageBanner, hidePageBanner } from "./HMI-utils.js";
import { getAudioRecorder } from "./audio_recorder.js";
import {
  AudioDecoder,
  SpectrogramView,
  decodeFloat32PcmBase64,
} from "./spectrogram.js";
import {
  AnimalSpectrogramWorkflow,
  LatestSourceGuard,
  MicrophoneSpectrogramWorkflow,
} from "./spectrogram-workflow.js";
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

const EARTH_RADIUS         = 6371000;
const MIC_DETECTION_RANGE  = 300;
const MAX_RECORDING_SECONDS = 20;
const DEG_TO_RAD           = Math.PI / 180;
const RAD_TO_DEG           = 180 / Math.PI;

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
var activeAudioContext  = null;
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
var stopRecordingInProgress       = false;
var lastBrowserRecordingBlob      = null;
var lastBrowserRecordingUrl       = null;
var lastRecordingDurationLabel    = null;
var lastRecordingSamples          = null;
var lastRecordingSampleRate       = 44100;
var playbackSourceNode            = null;
var playbackAudioContext          = null;

var recordingPlaybackContext = null;
var a_source        = null;
var decodedAudioStore = null;
var fileContent     = null;
var microphoneSourceGuard = new LatestSourceGuard();
var selectedFileObjectUrl = null;
var recordedAudioObjectUrl = null;

var animalSpectrogramWorkflow = null;
var microphoneSpectrogramWorkflow = null;

export var animal_toggled = false;

// ─────────────────────────────────────────────────────────────────────────────
// DOM helpers
// ─────────────────────────────────────────────────────────────────────────────

function getDurationTag()       { return document.getElementById("recording_duration"); }
function getAudioElement()      { return document.getElementById("audioElem"); }
function getAudioElementSource(){ const a = getAudioElement(); return a ? a.getElementsByTagName("source")[0] || null : null; }
function getPlaybackIndicator() { return document.getElementById("audio_playback_indicator"); }
function getRecordButton()      { return document.getElementById("record_audio_button"); }
function getStopButton()        { return document.getElementById("stop_recording_button"); }
function getCancelButton()      { return document.getElementById("cancel_recording_button"); }
function getPlayButton()        { return document.getElementById("frb2-play-button"); }
function getPlayLabel()         { return document.getElementById("frb2-play-label"); }
function getLiveStatus()        { return document.getElementById("frb2-live-status"); }
function getStatusLabel()       { return document.getElementById("frb2-status-label"); }
function getFileInput()         { return document.getElementById("fileInput"); }
function getAudioElemForRecordedPlayback() { return document.getElementById("audioElem"); }
function getRecordingPlaybackStatus() { return document.getElementById("recording_playback_status"); }

function setRecordingPlaybackStatus(message) {
  const status = getRecordingPlaybackStatus();
  if (status) status.textContent = message;
}

function setLiveRecordingState(state, label) {
  const live = getLiveStatus();
  const statusLabel = getStatusLabel();
  if (live) live.setAttribute("data-state", state);
  if (statusLabel && label) statusLabel.textContent = label;
}

function setRecordingActionState(isRecording) {
  const recordBtn = getRecordButton();
  const stopBtn = getStopButton();
  const cancelBtn = getCancelButton();
  if (recordBtn) recordBtn.disabled = !!isRecording;
  if (stopBtn) stopBtn.disabled = !isRecording;
  if (cancelBtn) cancelBtn.disabled = !isRecording;
}

function setPlayEnabled(enabled) {
  const playBtn = getPlayButton();
  if (playBtn) playBtn.disabled = !enabled;
}

function setPlayButtonPlaying(isPlaying) {
  const playBtn = getPlayButton();
  const playLabel = getPlayLabel();
  if (playBtn) playBtn.classList.toggle("is-playing", !!isPlaying);
  if (playLabel) playLabel.textContent = isPlaying ? "Pause" : "Play recording";
}

function clampRecordingSamples(samples, sampleRate) {
  if (!samples || !samples.length) return samples;
  const maxSamples = Math.floor(sampleRate * MAX_RECORDING_SECONDS);
  if (samples.length <= maxSamples) return samples;
  return samples.slice(0, maxSamples);
}

function clearBrowserRecordingPlayback() {
  stopPcmPlayback();

  if (lastBrowserRecordingUrl) {
    URL.revokeObjectURL(lastBrowserRecordingUrl);
    lastBrowserRecordingUrl = null;
  }
  lastBrowserRecordingBlob = null;
  lastRecordingDurationLabel = null;
  lastRecordingSamples = null;
  lastRecordingSampleRate = 44100;

  const audioEl = getAudioElemForRecordedPlayback();
  if (audioEl) {
    audioEl.pause();
    audioEl.removeAttribute("src");
    while (audioEl.firstChild) audioEl.removeChild(audioEl.firstChild);
  }

  setPlayEnabled(false);
  setPlayButtonPlaying(false);
  setRecordingPlaybackStatus("No recording yet. Press Record, then Stop.");
}

function prepareBrowserRecordingPlayback(result, durationLabel) {
  lastRecordingDurationLabel = durationLabel || null;
  const rawSamples = result && result.samples ? result.samples : null;
  lastRecordingSampleRate = (result && result.sampleRate) || 44100;
  lastRecordingSamples = clampRecordingSamples(rawSamples, lastRecordingSampleRate);
  lastBrowserRecordingBlob = (result && result.blob) ? result.blob : (result instanceof Blob ? result : null);

  // Keep a blob URL around for optional native player, but replay uses PCM.
  if (lastBrowserRecordingBlob) {
    if (lastBrowserRecordingUrl) URL.revokeObjectURL(lastBrowserRecordingUrl);
    lastBrowserRecordingUrl = URL.createObjectURL(lastBrowserRecordingBlob);
    const audioEl = getAudioElemForRecordedPlayback();
    if (audioEl) {
      audioEl.src = lastBrowserRecordingUrl;
    }
  }

  const hasClip = !!(lastRecordingSamples && lastRecordingSamples.length);
  setPlayEnabled(hasClip);
  setPlayButtonPlaying(false);

  const durationText = durationLabel ? ` (${durationLabel})` : "";
  setRecordingPlaybackStatus(
    hasClip
      ? "Recording ready" + durationText + ". Press the green play button to listen."
      : "Recording finished, but no audio was captured. Try again."
  );
}

function stopPcmPlayback() {
  try {
    if (playbackSourceNode) {
      playbackSourceNode.onended = null;
      playbackSourceNode.stop();
    }
  } catch (_err) { /* ignore */ }
  playbackSourceNode = null;

  if (playbackAudioContext) {
    playbackAudioContext.close().catch(() => {});
    playbackAudioContext = null;
  }
}

function playPcmRecording() {
  if (!lastRecordingSamples || !lastRecordingSamples.length) {
    return Promise.reject(new Error("No PCM samples available"));
  }

  stopPcmPlayback();

  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioCtx();
  playbackAudioContext = ctx;

  const buffer = ctx.createBuffer(1, lastRecordingSamples.length, lastRecordingSampleRate);
  buffer.getChannelData(0).set(lastRecordingSamples);

  const source = ctx.createBufferSource();
  playbackSourceNode = source;
  source.buffer = buffer;
  source.connect(ctx.destination);

  return ctx.resume().then(() => {
    source.start(0);
    return new Promise((resolve) => {
      source.onended = () => {
        playbackSourceNode = null;
        resolve();
      };
    });
  });
}

function revokeObjectUrl(url) {
  if (url) URL.revokeObjectURL(url);
}

function initializeSpectrogramWorkflows() {
  if (!animalSpectrogramWorkflow) {
    const animalRoot = document.getElementById("animal-spectrogram");
    if (animalRoot) {
      animalSpectrogramWorkflow = new AnimalSpectrogramWorkflow({
        decodePcm: decodeFloat32PcmBase64,
        retrieveAudio,
        view: new SpectrogramView(animalRoot),
      });
    }
  }

  if (!microphoneSpectrogramWorkflow) {
    const microphoneRoot = document.getElementById("microphone-spectrogram");
    if (microphoneRoot) {
      microphoneSpectrogramWorkflow = new MicrophoneSpectrogramWorkflow({
        decoder: new AudioDecoder(),
        view: new SpectrogramView(microphoneRoot),
      });
    }
  }
}

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
  initializeSpectrogramWorkflows();

  const audioElement = getAudioElement();
  if (audioElement && !audioElement.dataset.hmiBound) {
    audioElement.onended = hidePlaybackIndicator;
    audioElement.dataset.hmiBound = "true";
  }

  const fileInput = getFileInput();
  if (fileInput && !fileInput.dataset.hmiBound) {
    fileInput.dataset.hmiBound = "true";
    fileInput.addEventListener("change", async function (event) {
      const selection = microphoneSourceGuard.begin();
      const selectedFile = event.target.files?.[0] || null;

      if (!selectedFile) {
        fileContent = null;
        decodedAudioStore = null;
        microphoneSpectrogramWorkflow?.clear();
        revokeObjectUrl(selectedFileObjectUrl);
        selectedFileObjectUrl = null;
        return;
      }

      stopRecordingPlayback();
      audioRecorder.audioBlobs = [];
      revokeObjectUrl(selectedFileObjectUrl);
      revokeObjectUrl(recordedAudioObjectUrl);
      selectedFileObjectUrl = null;
      recordedAudioObjectUrl = null;

      const [encodedData, decodedAudio] = await Promise.all([
        selectedFile.arrayBuffer().catch(() => null),
        microphoneSpectrogramWorkflow?.load(selectedFile) ?? Promise.resolve(null),
      ]);
      if (!microphoneSourceGuard.isCurrent(selection)) return;

      if (!encodedData || !decodedAudio) {
        fileContent = null;
        decodedAudioStore = null;
        return;
      }

      fileContent = encodedData;
      decodedAudioStore = decodedAudio;
      a_source = null;
      const audioElementRef = getAudioElement();
      if (audioElementRef) {
        revokeObjectUrl(selectedFileObjectUrl);
        selectedFileObjectUrl = URL.createObjectURL(selectedFile);
        audioElementRef.src = selectedFileObjectUrl;
      }
    });
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeStaticDOMHooks, { once: true });
} else {
  initializeStaticDOMHooks();
}

async function destroyAudioFeatures() {
  microphoneSourceGuard.invalidate();
  playNextTrack = false;
  playNextRecordedTrack = false;
  stopAudioPlayback();
  stopRecordingPlayback();
  revokeObjectUrl(selectedFileObjectUrl);
  revokeObjectUrl(recordedAudioObjectUrl);
  selectedFileObjectUrl = null;
  recordedAudioObjectUrl = null;
  if (audioRecorder.mediaRecorder) audioRecorder.cancel();
  await Promise.allSettled([
    animalSpectrogramWorkflow?.destroy(),
    microphoneSpectrogramWorkflow?.destroy(),
  ]);
  animalSpectrogramWorkflow = null;
  microphoneSpectrogramWorkflow = null;
}

window.addEventListener("beforeunload", () => { void destroyAudioFeatures(); }, { once: true });

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
// ─────────────────────────────────────────────────────────────────────────────
// MQTT connection state polling (FR-A2)
// ─────────────────────────────────────────────────────────────────────────────

let _lastMqttState = null;

async function pollMqttConnectionState() {
  try {
    const response = await fetch("http://localhost:9000/mqtt/connection-state");
    if (!response.ok) throw new Error("Failed to fetch connection state");
    const data = await response.json();
    const state = data.state;

    if (state !== _lastMqttState) {
      if (state === "connected") {
        hidePageBanner("warning");
        hidePageBanner("error");
        if (_lastMqttState !== null) {
          showToast("Live data connection restored", "success");
        }
      } else if (state === "reconnecting") {
        showPageBanner("Live data connection lost — reconnecting…", "warning", false);
        showToast("Live data connection lost, reconnecting…", "warning");
      } else if (state === "disconnected") {
        showPageBanner("Live data unavailable", "error", false);
      }
      _lastMqttState = state;
    }
  } catch (err) {
    console.error("Error polling MQTT connection state:", err);

    // The status check itself failed (backend unreachable) — treat this
    // as unavailable rather than silently keeping the last-known state.
    if (_lastMqttState !== "unavailable") {
      showPageBanner("Live data unavailable — unable to check connection status", "error", false);
      _lastMqttState = "unavailable";
    }
  }
}

function startMqttConnectionPolling() {
  pollMqttConnectionState();
  setInterval(pollMqttConnectionState, 5000);
}

// ─────────────────────────────────────────────────────────────────────────────
// Live MQTT events → map layers (FR-A2)
// ─────────────────────────────────────────────────────────────────────────────

const _seenMqttEventIds = new Set();

async function pollMqttLatestEvents(hmiState) {
  try {
    const response = await fetch("http://localhost:9000/mqtt/latest-events");
    if (!response.ok) throw new Error("Failed to fetch latest events");
    const data = await response.json();
    const events = data.events || [];


    const newVocalizationEvents = [];
    const newMovementEvents = [];

    for (const event of events) {
      if (_seenMqttEventIds.has(event._id)) continue;
      _seenMqttEventIds.add(event._id);

      if (event.eventType === "vocalization") {
        newVocalizationEvents.push(event);
      } else if (event.eventType === "movement") {
        newMovementEvents.push(event);
      }
      // sensor_health / iot_node: normalized on the backend, but there is
      // no existing HMI render function for these yet, so they are not
      // routed to the map here. Flagged for a follow-up ticket.
    }

    if (newVocalizationEvents.length > 0) {
      updateVocalizationLayerFromLiveData(hmiState, newVocalizationEvents);
    }
    if (newMovementEvents.length > 0) {
      updateAnimalMovementLayerFromLiveData(hmiState, newMovementEvents);
    }


  } catch (err) {
    console.error("Error polling MQTT latest events:", err);
  }
}

function startMqttEventPolling(hmiState) {
  pollMqttLatestEvents(hmiState);
  setInterval(() => pollMqttLatestEvents(hmiState), 5000);
}







export function initialiseHMI(hmiState) {
  console.log("initialising");
  startMqttConnectionPolling();
  startMqttEventPolling(hmiState);

  showMapSpinner("Loading map data…");
  hideMapError();

  // FR-A1: initialiseHMI() is re-entered by the map-error retry button
  // (see showMapError(userMsg, () => initialiseHMI(hmiState)) below), on
  // the same hmiState. createBasemap() itself is idempotent (see above),
  // but everything in this block is one-time setup: it adds a brand new
  // vector layer for every vocalisation/truth/mic slot and registers a new
  // map click listener. None of that is guarded against being called
  // twice, so without this check a retry would silently stack a second
  // copy of every layer and a second click handler on top of the reused
  // map. Only run it the first time hmiState has no basemap yet.
  const isFirstInit = !hmiState.basemap;

  createBasemap(hmiState);

  if (isFirstInit) {
    addVocalisationLayers(hmiState);
    addTruthLayers(hmiState);

    for (let i = 1; i < 26; i++) {
      addVectorLayerTopDown(hmiState, `mic_layer_${i}`);
    }
    addVectorLayerTopDown(hmiState, "mic_layer");

    addAllTruthFeatures(hmiState);
    addAllVocalizationFeatures(hmiState);
    createMapClickEvent(hmiState);
  }

  // Task 7: withRetry wraps the microphone fetch.  routes.js already retries
  // the axios call; this outer withRetry handles cases where the call itself
  // throws before reaching the network (e.g. axios not yet initialised).
  withRetry(() => retrieveMicrophones(), {
    attempts: 3,
    delayMs: 2000,
    retryMessage: "Microphone load failed, retrying",
  })
    .then(async (res) => {
      hideMapSpinner();
      updateMicrophoneLayer(hmiState, res.data);
      stepMicAnimation(hmiState);

      // FR-A1: await this instead of firing-and-forgetting it. Before,
      // a failed node load rejected with nothing awaiting it (an unhandled
      // rejection) while this .then() carried on straight to the "Map data
      // loaded successfully" toast regardless. Awaiting it here means a
      // node-load failure is caught by the .catch() below instead, so the
      // success toast only fires once nodes have actually loaded.
      await addIoTNodesToMap(hmiState);

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

export function stopAudioPlayback(updateAnimation = true) {
  if (updateAnimation) muteAudioAnimation();
  if (audioAnimTimeout) clearTimeout(audioAnimTimeout);
  audioAnimTimeout = null;
  if (activeAudioNode !== null) {
    try { activeAudioNode.stop(); } catch (_error) { /* Source may already have ended. */ }
    activeAudioNode.disconnect();
  }
  activeAudioNode = null;
  if (activeAudioContext !== null) {
    void activeAudioContext.close().catch(() => {});
  }
  activeAudioContext = null;
}

function playDecodedAudio(decodedAudio) {
  if (!decodedAudio || !playNextTrack) return;
  stopAudioPlayback(false);
  playNextTrack = true;

  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const context = new AudioContextClass();
  const channelCount = Math.max(1, decodedAudio.numberOfChannels || 1);
  const audioBuffer = context.createBuffer(channelCount, decodedAudio.length, decodedAudio.sampleRate);
  for (let channel = 0; channel < channelCount; channel += 1) {
    audioBuffer.copyToChannel(decodedAudio.getChannelData(channel), channel);
  }

  const source = context.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(context.destination);
  source.onended = () => {
    if (activeAudioNode !== source) return;
    if (audioAnimTimeout) clearTimeout(audioAnimTimeout);
    audioAnimTimeout = null;
    activeAudioNode = null;
    activeAudioContext = null;
    source.disconnect();
    void context.close().catch(() => {});
    muteAudioAnimation();
  };
  activeAudioContext = context;
  activeAudioNode = source;
  source.start();
  audioAnimTimeout = setTimeout(muteAudioAnimation, audioBuffer.duration * 1000);
}

document.addEventListener("playAudio", function () {
  playNextTrack = true;
  if (!animalSpectrogramWorkflow || selectedVocalizationEventId === null) {
    muteAudioAnimation();
    return;
  }
  animalSpectrogramWorkflow.getSelectedAudio().then((decodedAudio) => {
    if (!decodedAudio || !playNextTrack) {
      muteAudioAnimation();
      return;
    }
    playDecodedAudio(decodedAudio);
  });
});

document.addEventListener("stopAudio", function () {
  playNextTrack = false;
  stopAudioPlayback();
});

function clearAnimalAudioSelection() {
  selectedVocalizationEventId = null;
  animalSpectrogramWorkflow?.clear();
  stopAudioPlayback();
}

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
  return (isPredator ? "./../images/predator/sim/" : "./../images/sim/") +
    getIconName(entry.animalStatus, entry.animalType);
}

function _resolveVocalizationIconPath(entry) {
  const isHerbivore = ["herbivore", "frugivore"].includes(entry.animalDiet);
  return (isHerbivore ? "./../images/vocalization/" : "./../images/predator/vocalization/") +
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
  // FR-A1: initialiseHMI() can run more than once (the map-error retry
  // button calls it again on the same hmiState). Reuse the existing
  // ol.Map instead of constructing a second one on top of the same
  // #basemap target — a second ol.Map would leave the first one's canvas,
  // render loop and event listeners still attached underneath it.
  if (hmiState.basemap) {
    return hmiState.basemap;
  }

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
    `http://localhost:9000/hmi/weather?timestamp=${timestamp}&lat=${lat}&lon=${lon}`
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
      clearAnimalAudioSelection();
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
      clearAnimalAudioSelection();
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
      clearAnimalAudioSelection();
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

      const audioHeader  = document.getElementById("animalAudioHeader");
      const audioControl = document.getElementById("animalAudioControl");
      const spectrogram  = document.getElementById("animal-spectrogram");
      if (values.isAnimalMovement) {
        clearAnimalAudioSelection();
        if (audioHeader)  audioHeader.style.display  = "none";
        if (audioControl) audioControl.style.display = "none";
        if (spectrogram)  spectrogram.style.display  = "none";
      } else {
        if (audioHeader)  audioHeader.style.display  = "flex";
        if (audioControl) audioControl.style.display = "flex";
        if (spectrogram)  spectrogram.style.display  = "block";
        selectedVocalizationEventId = values.eventId || null;
        if (selectedVocalizationEventId !== null) {
          void animalSpectrogramWorkflow?.select(selectedVocalizationEventId);
        } else {
          animalSpectrogramWorkflow?.clear();
        }
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
  setRecordingActionState(true);
  setLiveRecordingState("recording", "Recording…");
  initializeRecordingDuration();
}

export function hideRecordingControls() {
  setRecordingActionState(false);
  if (durationTimer) {
    clearInterval(durationTimer);
    durationTimer = null;
  }
}

export function showRecordingNotSupportedOverlay() {
  showToast("Audio recording is not supported in this browser.", "error");
}

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
  const sourceToken = microphoneSourceGuard.begin();
  const audioElement       = getAudioElement();
  const audioElementSource = getAudioElementSource();

  if (audioElementSource && audioElement && !audioElement.paused) {
    audioElement.pause();
    hidePlaybackIndicator();
  }

  stopRecordingPlayback();
  playNextRecordedTrack = false;
  // Lock controls immediately to prevent rapid concurrent starts while
  // permission/getUserMedia is still pending.
  const recordBtn = getRecordButton();
  const stopBtn = getStopButton();
  const cancelBtn = getCancelButton();
  if (recordBtn) recordBtn.disabled = true;
  if (stopBtn) stopBtn.disabled = true;
  if (cancelBtn) cancelBtn.disabled = true;
  setLiveRecordingState("idle", "Requesting microphone permission…");

  audioRecorder
    .start()
    .then(() => {
      if (!microphoneSourceGuard.isCurrent(sourceToken)) {
        audioRecorder.cancel();
        return;
      }
      fileContent = null;
      decodedAudioStore = null;
      microphoneSpectrogramWorkflow?.clear();
      revokeObjectUrl(selectedFileObjectUrl);
      revokeObjectUrl(recordedAudioObjectUrl);
      selectedFileObjectUrl = null;
      recordedAudioObjectUrl = null;
      stopRecordingPlayback();
      audioRecordStartTime = new Date();
      showRecordingControls();
      showToast("Recording started", "info");
    })
    .catch((error) => {
      setLiveRecordingState("idle", "Ready to record");
      setRecordingActionState(false);

      if (error.message.includes("mediaDevices API or getUserMedia method is not supported in this browser.")) {
        showRecordingNotSupportedOverlay();
      }

      const toastMap = {
        NotAllowedError:  "Microphone permission was denied. Allow mic access in your browser settings and try again.",
        NotFoundError:    "No microphone device found",
        NotReadableError: "Microphone is not available right now",
        SecurityError:    "Microphone access blocked for security reasons",
        AbortError:       "Microphone access was interrupted",
        UnknownError:     "Unknown audio recording error",
      };

      showToast(
        toastMap[error.name] || ("Audio recording failed: " + error.message),
        "error"
      );
    });
}

export function stopAudioRecording() {
  const sourceToken = microphoneSourceGuard.begin();
  audioRecorder
    .stop()
    .then(async (audioBlob) => {
      hideRecordingControls();
      if (!microphoneSourceGuard.isCurrent(sourceToken)) return;
      fileContent = null;
      const decodedAudio = await microphoneSpectrogramWorkflow?.load(audioBlob) || null;
      if (!microphoneSourceGuard.isCurrent(sourceToken)) return;
      decodedAudioStore = decodedAudio;
      revokeObjectUrl(recordedAudioObjectUrl);
      recordedAudioObjectUrl = URL.createObjectURL(audioBlob);
      const audioElement = getAudioElemForRecordedPlayback();
      if (audioElement) audioElement.src = recordedAudioObjectUrl;
      if (!decodedAudioStore) {
        showToast("Recorded audio could not be decoded", "error");
      }
    })
    .catch((error) => {
      if (!microphoneSourceGuard.isCurrent(sourceToken)) return;
      showToast("Error stopping recording", "error");
      console.log("Stop recording error:", error.name);
    });
}

export function cancelAudioRecording() {
  microphoneSourceGuard.invalidate();
  audioRecorder.cancel();
  decodedAudioStore = null;
  microphoneSpectrogramWorkflow?.clear();
  hideRecordingControls();
  setLiveRecordingState("idle", "Recording cancelled");
  showToast("Recording cancelled", "info");
}

/** FR-B2 replay via green play button (PCM / Web Audio). */
export function playRecordingClip() {
  if (!lastRecordingSamples || !lastRecordingSamples.length) {
    showToast("No recording available to play. Record audio first.", "error");
    return;
  }

  // Toggle pause
  if (playbackSourceNode) {
    stopPcmPlayback();
    setPlayButtonPlaying(false);
    setLiveRecordingState("idle", "Playback paused");
    return;
  }

  setPlayButtonPlaying(true);
  setLiveRecordingState("playing", "Playing recording");

  playPcmRecording()
    .then(() => {
      setPlayButtonPlaying(false);
      setLiveRecordingState("idle", "Playback finished");
    })
    .catch((err) => {
      setPlayButtonPlaying(false);
      setLiveRecordingState("idle", "Ready to record");
      showToast("Unable to play recording: " + (err.message || "unknown error"), "error");
    });
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

function playRecording(recordedChunksOrBlob) {
  let blob = null;

  const mimeType = recordedChunks[0]?.type || "audio/webm";
  const blob = new Blob(recordedChunks, { type: mimeType });
  if (blob.size === 0) return;

  audioRecordingElement = getAudioElemForRecordedPlayback();
  if (!audioRecordingElement) return;

  revokeObjectUrl(recordedAudioObjectUrl);
  recordedAudioObjectUrl = URL.createObjectURL(blob);
  audioRecordingElement.src = recordedAudioObjectUrl;
  audioRecordingElement.load();

  if (playNextRecordedTrack) {
    recordingPlaybackAnimTimeout = setTimeout(muteRecordingPlaybackAnimation, 10000);
    audioRecordingElement.onended = () => {
      if (recordingPlaybackAnimTimeout) clearTimeout(recordingPlaybackAnimTimeout);
      recordingPlaybackAnimTimeout = null;
      muteRecordingPlaybackAnimation();
    };
    audioRecordingElement.play();
  }
}

function stopRecordingPlayback(updateAnimation = true) {
  if (updateAnimation) muteRecordingPlaybackAnimation();
  if (recordingPlaybackAnimTimeout) clearTimeout(recordingPlaybackAnimTimeout);
  recordingPlaybackAnimTimeout = null;

  if (a_source === null) {
    if (audioRecordingElement !== null) {
      audioRecordingElement.pause();
      audioRecordingElement.currentTime = 0;
    }
  } else {
    try { a_source.stop(); } catch (_error) { /* Source may already have ended. */ }
    a_source.disconnect();
    a_source = null;
  }
  if (recordingPlaybackContext !== null) {
    void recordingPlaybackContext.close().catch(() => {});
    recordingPlaybackContext = null;
  }
}

export function playAudio() {
  // Mic-panel browser recordings take priority for FR-B2 replay.
  if (lastBrowserRecordingBlob || (audioRecorder.audioBlobs && audioRecorder.audioBlobs.length > 0)) {
    playRecording(lastBrowserRecordingBlob || audioRecorder.audioBlobs);
    return;
  }

  if (!decodedAudioStore) {
    showToast("No recording available to play. Record audio first.", "error");
    return;
  }

  if (playNextRecordedTrack) {
    stopRecordingPlayback(false);
    playNextRecordedTrack = true;
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    recordingPlaybackContext = new AudioContextClass();
    const context = recordingPlaybackContext;
    const channelCount = Math.max(1, decodedAudioStore.numberOfChannels || 1);
    const audioBuffer = context.createBuffer(
      channelCount,
      decodedAudioStore.length,
      decodedAudioStore.sampleRate
    );
    for (let channel = 0; channel < channelCount; channel += 1) {
      audioBuffer.copyToChannel(decodedAudioStore.getChannelData(channel), channel);
    }
    recordingPlaybackAnimTimeout = setTimeout(muteRecordingPlaybackAnimation, audioBuffer.duration * 1000);
    const source = context.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(context.destination);
    source.onended = () => {
      if (a_source !== source) return;
      if (recordingPlaybackAnimTimeout) clearTimeout(recordingPlaybackAnimTimeout);
      recordingPlaybackAnimTimeout = null;
      source.disconnect();
      a_source = null;
      recordingPlaybackContext = null;
      void context.close().catch(() => {});
      muteRecordingPlaybackAnimation();
    };
    a_source = source;
    source.start();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Recording duration timer
// ─────────────────────────────────────────────────────────────────────────────

export function initializeRecordingDuration() {
  showRecordingDuration("00:00");
  durationTimer = setInterval(() => {
    showRecordingDuration(computeRecordingDuration(audioRecordStartTime));
  }, 250);
}

export function showRecordingDuration(duration) {
  const tag = getDurationTag();
  if (!tag) return;
  tag.textContent = duration;
  if (checkAudioDurationThreshold(duration)) stopAudioRecording();
}

export function checkAudioDurationThreshold(duration) {
  const parts = String(duration).split(":");
  if (parts.length !== 2) return false;
  const mins = Number(parts[0]);
  const secs = Number(parts[1]);
  if (Number.isNaN(mins) || Number.isNaN(secs)) return false;
  return mins * 60 + secs >= MAX_RECORDING_SECONDS;
}

export function computeRecordingDuration(startTime) {
  const delta = Math.max(0, (new Date() - startTime) / 1000);
  const mins  = Math.floor(delta / 60) % 60;
  const secs  = Math.floor(delta % 60);
  return String(mins).padStart(2, "0") + ":" + String(secs).padStart(2, "0");
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
