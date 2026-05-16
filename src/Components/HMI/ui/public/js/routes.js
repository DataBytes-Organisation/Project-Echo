"use strict";

/**
 * routes.js
 * All API calls for the EchoNet HMI frontend.
 *
 * Sprint 1/2 : axios instance, all API functions.
 * Task 7     : Removed duplicate getApiErrorMessage (now imported from
 *              HMI-utils.js — single source of truth).
 *              Added withRetry wrappers to all read-only GET functions.
 *              Added retrieveIotNodes() so map.js uses the shared axios
 *              instance instead of a raw fetch() call.
 */

import { getApiErrorMessage, withRetry } from "./HMI-utils.js";

// ─────────────────────────────────────────────────────────────────────────────
// Axios instance
// ─────────────────────────────────────────────────────────────────────────────

let axios;

if (typeof window === "undefined") {
  axios = require("axios");
} else {
  axios = window.axios;
}

const API_TIMEOUT_MS = 10000;

const api = axios.create({
  timeout: API_TIMEOUT_MS,
});

// ─────────────────────────────────────────────────────────────────────────────
// Re-export getApiErrorMessage so existing callers that import it from
// routes.js do not need to change their import path.
// ─────────────────────────────────────────────────────────────────────────────

export { getApiErrorMessage };

// ─────────────────────────────────────────────────────────────────────────────
// Retry defaults
//
// Applied to all read-only GET functions.
// POST / sim-control functions are NOT retried — see individual comments.
// ─────────────────────────────────────────────────────────────────────────────

const RETRY_OPTS = {
  attempts: 3,
  delayMs: 1500,
  retryMessage: "Request failed, retrying",
};

// ─────────────────────────────────────────────────────────────────────────────
// IoT map  (Task 7 — new)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fetch all IoT nodes for the map page.
 * Added in Task 7 — previously map.js called /iot/nodes directly with fetch().
 * Bringing it here means it uses the shared axios instance (timeout, future
 * interceptors) and is consistent with every other API call in the project.
 *
 * @returns {Promise<AxiosResponse>}  response.data is the array of node objects.
 */
export function retrieveIotNodes() {
  return withRetry(() => api.get("/iot/nodes"), RETRY_OPTS);
}

// ─────────────────────────────────────────────────────────────────────────────
// Movement / truth events
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @param {number|string} from  - Start timestamp.
 * @param {number|string} to    - End timestamp.
 * @returns {Promise<AxiosResponse>}
 */
export function retrieveTruthEventsInTimeRange(from, to) {
  const start = parseInt(from);
  const end = parseInt(to);
  return withRetry(() => api.get(`/movement_time/${start}/${end}`), RETRY_OPTS);
}

/**
 * @param {number|string} from
 * @param {number|string} to
 * @returns {Promise<AxiosResponse>}
 */
export function retrieveVocalizationEventsInTimeRange(from, to) {
  const start = parseInt(from);
  const end = parseInt(to);
  return withRetry(() => api.get(`/events_time/${start}/${end}`), RETRY_OPTS);
}

// ─────────────────────────────────────────────────────────────────────────────
// Microphones / Audio
// ─────────────────────────────────────────────────────────────────────────────

/** @returns {Promise<AxiosResponse>} */
export function retrieveMicrophones() {
  return withRetry(() => api.get("/microphones"), RETRY_OPTS);
}

/** @param {string|number} id @returns {Promise<AxiosResponse>} */
export function retrieveAudio(id) {
  return withRetry(() => api.get(`/audio/${id}`), RETRY_OPTS);
}

// ─────────────────────────────────────────────────────────────────────────────
// Recordings  (POST — not retried: could create duplicate records)
// ─────────────────────────────────────────────────────────────────────────────

/** @param {object} recordingData @returns {Promise<AxiosResponse>} */
export function postRecording(recordingData) {
  return api.post("/post_recording", recordingData);
}

// ─────────────────────────────────────────────────────────────────────────────
// Simulator control  (POST — not retried: commands are stateful)
// ─────────────────────────────────────────────────────────────────────────────

export function setSimModeAnimal()     { return api.post("/sim_control/Animal_Mode"); }
export function setSimModeRecording()  { return api.post("/sim_control/Recording_Mode"); }
export function setSimModeRecordingV2(){ return api.post("/sim_control/Recording_Mode_V2"); }
export function stopSimulator()        { return api.post("/sim_control/Stop"); }

// ─────────────────────────────────────────────────────────────────────────────
// Simulator time
// ─────────────────────────────────────────────────────────────────────────────

/** @returns {Promise<AxiosResponse>} */
export function retrieveSimTime() {
  return withRetry(() => api.get("/latest_movement"), RETRY_OPTS);
}
