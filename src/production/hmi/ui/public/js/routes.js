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
 * @param {object} [opts]  Overrides merged over RETRY_OPTS. Pass { silent: true }
 *                         for background pollers so the retry toasts stay quiet.
 * @returns {Promise<AxiosResponse>}  response.data is the array of node objects.
 */
export function retrieveIotNodes(opts = {}) {
  return withRetry(() => api.get("/iot/nodes"), { ...RETRY_OPTS, ...opts });
}

/**
 * Fetch a single IoT node's detail.
 * Goes through the Node proxy route rather than the browser hitting the Python
 * API on :9000 directly, which is what admin-nodes.html used to do.
 *
 * @param {string} nodeId
 * @param {object} [opts]  Overrides merged over RETRY_OPTS.
 * @returns {Promise<AxiosResponse>}  response.data is the node object.
 */
export function retrieveIotNode(nodeId, opts = {}) {
  return withRetry(
    () => api.get(`/iot/nodes/${encodeURIComponent(nodeId)}`),
    { ...RETRY_OPTS, ...opts }
  );
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

// ─────────────────────────────────────────────────────────────────────────────
// Auth  (FR-D1 follow-up — new)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Sign in against the Node auth route (which proxies to /hmi/signin).
 *
 * POST — deliberately NOT retried. Blindly re-posting credentials would burn
 * attempts against any lockout/rate-limit on the auth backend, and a failed
 * login is something the user should decide to retry themselves.
 *
 * Throws on any non-2xx (axios default). The route replies with a JSON
 * `{ message }` body on 401/502/500, so callers should prefer
 * `err.response.data.message` before falling back to getApiErrorMessage().
 *
 * @param {{username?: string, email?: string, password: string}} credentials
 * @returns {Promise<AxiosResponse>}  response.data is { message, token, userId }.
 */
export function signIn({ username, email, password }) {
  return api.post("/api/auth/signin", { username, email, password });
}

// ─────────────────────────────────────────────────────────────────────────────
// Admin - edit requests  (FR-D1 follow-up — new)
// ─────────────────────────────────────────────────────────────────────────────

/** @returns {Promise<AxiosResponse>} */
export function retrieveAdminRequests() {
  return withRetry(() => api.get("/api/requests"), RETRY_OPTS);
}

/**
 * PATCH — not retried, this is changing a request's status, not reading it.
 * @param {string} requestId
 * @param {string} status
 * @returns {Promise<AxiosResponse>}
 */
export function updateRequestStatus(requestId, status) {
  return api.patch(`/api/requests/${requestId}`, { status });
}

/**
 * PATCH — not retried, same reasoning as above.
 * @param {string} animal
 * @param {string} status
 * @returns {Promise<AxiosResponse>}
 */
export function updateConservationStatus(animal, status) {
  return api.patch(`/api/updateConservationStatus/${animal}`, { status });
}

// ─────────────────────────────────────────────────────────────────────────────
// Admin - sensor health  (FR-D1 follow-up — new)
// ─────────────────────────────────────────────────────────────────────────────

/** @returns {Promise<AxiosResponse>} */
export function retrieveSensorUpdates() {
  return withRetry(() => api.get("/sensors/updates"), RETRY_OPTS);
}

/**
 * @param {object} [opts]  Overrides merged over RETRY_OPTS. charts.js passes
 *                         { silent: true } - the chart is decorative and falls
 *                         back to zeroes, so it shouldn't toast on the way down.
 * @returns {Promise<AxiosResponse>}
 */
export function retrieveSensorAlerts(opts = {}) {
  return withRetry(() => api.get("/sensors/alerts"), { ...RETRY_OPTS, ...opts });
}

/** @param {number} [limit=50] @returns {Promise<AxiosResponse>} */
export function retrieveRecentReboots(limit = 50) {
  return withRetry(() => api.get("/sensors/reboots/recent", { params: { limit } }), RETRY_OPTS);
}

/** @returns {Promise<AxiosResponse>} */
export function retrieveSensorSettings() {
  return withRetry(() => api.get("/sensors/__default__/settings"), RETRY_OPTS);
}

/**
 * PUT — not retried, this is a save, not a read.
 * @param {object} settings
 * @returns {Promise<AxiosResponse>}
 */
export function updateSensorSettings(settings) {
  return api.put("/sensors/__default__/settings", { settings });
}

/**
 * POST — not retried, don't want to accidentally queue a reboot twice.
 * @param {string} sensorId
 * @param {string|null} [reason]
 * @returns {Promise<AxiosResponse>}
 */
export function rebootSensor(sensorId, reason = null) {
  return api.post(`/sensors/${encodeURIComponent(sensorId)}/reboot`, { reason });
}

// ─────────────────────────────────────────────────────────────────────────────
// Notifications  (FR-D1 follow-up — new)
//
// These went to http://localhost:9000 directly, which only ever worked on a dev
// machine. server.js already proxies /hmi/* through to the API, so the relative
// path works everywhere and picks up the shared timeout and error handling.
// ─────────────────────────────────────────────────────────────────────────────

/** Bearer header for the routes the API expects a token on. */
function authHeaders(token) {
  return token ? { headers: { Authorization: `Bearer ${token}` } } : {};
}

/**
 * @param {string} userId
 * @param {string} token
 * @returns {Promise<AxiosResponse>}  response.data is the saved species list.
 */
export function retrieveUserNotifications(userId, token) {
  return withRetry(
    () => api.get(`/hmi/users/${encodeURIComponent(userId)}/notifications`, authHeaders(token)),
    RETRY_OPTS
  );
}

/**
 * POST — not retried, this adds a subscription rather than reading one.
 * @param {string} userId
 * @param {{species: string, common: string}} animal
 * @param {string} token
 * @returns {Promise<AxiosResponse>}
 */
export function addUserNotification(userId, animal, token) {
  return api.post(
    `/hmi/users/${encodeURIComponent(userId)}/notifications`,
    { species: animal.species, common: animal.common },
    authHeaders(token)
  );
}

/**
 * DELETE — not retried, same reasoning as the POST above.
 * @param {string} userId
 * @param {string} species
 * @param {string} token
 * @returns {Promise<AxiosResponse>}
 */
export function removeUserNotification(userId, species, token) {
  return api.delete(
    `/hmi/users/${encodeURIComponent(userId)}/notifications/${encodeURIComponent(species)}`,
    authHeaders(token)
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Submissions, profile, donations  (FR-D1 follow-up — new)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * POST — not retried, a resubmit would create a second review request.
 * @param {object} requestData
 * @returns {Promise<AxiosResponse>}
 */
export function submitAnimalRequest(requestData) {
  return api.post("/api/submit", requestData);
}

/** @returns {Promise<AxiosResponse>}  response.data is { username, email, ... }. */
export function retrieveUserProfile() {
  return withRetry(() => api.get("/user_profile"), RETRY_OPTS);
}

/** @returns {Promise<AxiosResponse>}  response.data is { cumulativeTotal }. */
export function retrieveCumulativeDonations() {
  return withRetry(() => api.get("/cumulativeDonations"), RETRY_OPTS);
}

/** @returns {Promise<AxiosResponse>}  response.data is the donations array. */
export function retrieveDonations() {
  return withRetry(() => api.get("/donations"), RETRY_OPTS);
}

/**
 * Weather for a detection, used by the map popups.
 * Was another direct localhost:9000 call - /hmi/* proxies through server.js.
 *
 * @param {number|string} timestamp
 * @param {number} lat
 * @param {number} lon
 * @returns {Promise<AxiosResponse>}
 */
export function retrieveWeather(timestamp, lat, lon) {
  return withRetry(
    () => api.get("/hmi/weather", { params: { timestamp, lat, lon } }),
    RETRY_OPTS
  );
}
