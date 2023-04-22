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

const MESSAGE_API_URL = 'http://localhost:8000';

export function retrieveTruthEventsInTimeRange(from, to) {
  var start = parseInt(from);
  var end = parseInt(to);
  return axios.get(`${MESSAGE_API_URL}/movement_time?start=${start}&end=${end}`);
}

/*
export function retrieveAllMessages() {
  return axios.get(`${MESSAGE_API_URL}/messages`);
}

export function deleteMessage(id) {
  return axios.delete(`${MESSAGE_API_URL}/messages/${id}`);
}*/
