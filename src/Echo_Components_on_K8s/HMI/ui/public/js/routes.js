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

//const MESSAGE_API_URL = 'http://ts-api-cont:9000/hmi';

export function retrieveTruthEventsInTimeRange(from, to) {
  var start = parseInt(from);
  var end = parseInt(to);
  return axios.get(`/movement_time/${start}/${end}`);
}

export function retrieveVocalizationEventsInTimeRange(from, to) {
  var start = parseInt(from);
  var end = parseInt(to);
  return axios.get(`/events_time/${start}/${end}`);
}

export function retrieveMicrophones() {
  return axios.get(`/microphones`);
}

export function retrieveAudio(id){
  //console.log(`/audio?id=${id}`);
  return axios.get(`/audio/${id}`);
}

export function postRecording(recordingData){
  axios.post(`/post_recording`, recordingData)
}

export function setSimModeAnimal(){
  axios.post(`/sim_control/Animal_Mode`);
}

export function setSimModeRecording(){
  axios.post(`/sim_control/Recording_Mode`);
}

export function setSimModeRecordingV2(){
  axios.post(`/sim_control/Recording_Mode_V2`);
}

export function stopSimulator(){
  axios.post(`/sim_control/Stop`);
}

export function retrieveSimTime(){
  return axios.get(`/latest_movement`);
}

// export function signup(){
//   return axios.post(`/signup`)
// }

// export function signin(username, password){
//   return axios.post(`/signin`, {
//     username,
//     password
//   }).then(res => console.log(res))
//   .catch(err => console.log(err))
// }