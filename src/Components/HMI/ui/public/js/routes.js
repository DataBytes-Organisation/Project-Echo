let axios;

if (typeof window === 'undefined') {
  axios = require('axios');
} else {
  axios = window.axios;
}

// Use this for all API calls
const API = axios.create({
  baseURL: 'http://localhost:9000'
});

export function retrieveTruthEventsInTimeRange(from, to) {
  var start = parseInt(from);
  var end = parseInt(to);
  return API.get(`/movement_time/${start}/${end}`);
}

export function retrieveVocalizationEventsInTimeRange(from, to) {
  var start = parseInt(from);
  var end = parseInt(to);
  return API.get(`/events_time/${start}/${end}`);
}

export function retrieveMicrophones() {
  return API.get(`/microphones`);
}

export function retrieveAudio(id){
  return API.get(`/audio/${id}`);
}

export function postRecording(recordingData){
  return API.post(`/post_recording`, recordingData);
}

export function setSimModeAnimal(){
  return API.post(`/sim_control/Animal_Mode`);
}

export function setSimModeRecording(){
  return API.post(`/sim_control/Recording_Mode`);
}

export function setSimModeRecordingV2(){
  return API.post(`/sim_control/Recording_Mode_V2`);
}

export function stopSimulator(){
  return API.post(`/sim_control/Stop`);
}

export function retrieveSimTime(){
  return API.get(`/latest_movement`);
}

// Uncomment these if you're using signup/signin
// export function signup(){
//   return API.post(`/signup`);
// }

// export function signin(username, password){
//   return API.post(`/signin`, {
//     username,
//     password
//   }).then(res => console.log(res))
//     .catch(err => console.log(err));
// }