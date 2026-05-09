"use strict";

/*var requirejs = require('./r.js')

requirejs.config({
    //Pass the top-level main.js/index.js require
    //function to requirejs so that node modules
    //are loaded relative to the top-level JS file.
    nodeRequire: require
});*/

let axios;

if (typeof window === 'undefined') {
  axios = require('axios');
} else {
  axios = window.axios;
}



//
// All requests share a single axios instance configured with a timeout.
// If the server does not respond within API_TIMEOUT_MS the request rejects
// with an error whose `code` is "ECONNABORTED", giving the caller a clear
// signal to show the error banner rather than hanging indefinitely.

const API_TIMEOUT_MS = 10000; // 10 seconds – adjust to taste

const api = axios.create({
  timeout: API_TIMEOUT_MS,
});


// Route functions  (all unchanged in signature – only `axios` → `api`)


export function retrieveTruthEventsInTimeRange(from, to) {
  const start = parseInt(from);
  const end   = parseInt(to);
  return api.get(`/movement_time/${start}/${end}`);
}

export function retrieveVocalizationEventsInTimeRange(from, to) {
  const start = parseInt(from);
  const end   = parseInt(to);
  return api.get(`/events_time/${start}/${end}`);
}

export function retrieveMicrophones() {
  return api.get(`/microphones`);
}

export function retrieveAudio(id) {
  return api.get(`/audio/${id}`);
}

export function postRecording(recordingData) {
  return api.post(`/post_recording`, recordingData);
}

export function setSimModeAnimal() {
  return api.post(`/sim_control/Animal_Mode`);
}

export function setSimModeRecording() {
  return api.post(`/sim_control/Recording_Mode`);
}

export function setSimModeRecordingV2() {
  return api.post(`/sim_control/Recording_Mode_V2`);
}

export function stopSimulator() {
  return api.post(`/sim_control/Stop`);
}

export function retrieveSimTime() {
  return api.get(`/latest_movement`);
}
