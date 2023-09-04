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

const MESSAGE_API_URL = 'http://localhost:9000/hmi';
//const MESSAGE_API_URL = 'http://ts-api-cont:9000/hmi';

export function retrieveTruthEventsInTimeRange(from, to) {
  var start = parseInt(from);
  var end = parseInt(to);
  return axios.get(`${MESSAGE_API_URL}/movement_time?start=${start}&end=${end}`);
}

export function retrieveVocalizationEventsInTimeRange(from, to) {
  var start = parseInt(from);
  var end = parseInt(to);
  return axios.get(`${MESSAGE_API_URL}/events_time?start=${start}&end=${end}`);
}

export function retrieveMicrophones() {
  return axios.get(`${MESSAGE_API_URL}/microphones`);
}

export function retrieveAudio(id){
  //console.log(`${MESSAGE_API_URL}/audio?id=${id}`);
  return axios.get(`${MESSAGE_API_URL}/audio?id=${id}`);
}

export function postRecording(recordingData){
  axios.post(`${MESSAGE_API_URL}/post_recording`, recordingData)
  .then(response => {
    console.log('Response:', response.data);
  })
  .catch(error => {
    console.error('Error:', error);
  });
  //axios.post(`${MESSAGE_API_URL}/post_recording?data=${recordingData}`);
}

export function startSimulator(){
  axios.post(`${MESSAGE_API_URL}/sim_control?control=Start`);
}

export function stopSimulator(){
  axios.post(`${MESSAGE_API_URL}/sim_control?control=Stop`);
}

export function retrieveSimTime(){
  return axios.get(`${MESSAGE_API_URL}/latest_movement`);
}

// export function signup(){
//   return axios.post(`${MESSAGE_API_URL}/signup`)
// }

// export function signin(username, password){
//   return axios.post(`${MESSAGE_API_URL}/signin`, {
//     username,
//     password
//   }).then(res => console.log(res))
//   .catch(err => console.log(err))
// }

exports.abc = (req, res) => {
  return axios.get(`${MESSAGE_API_URL}/abc`);
}

