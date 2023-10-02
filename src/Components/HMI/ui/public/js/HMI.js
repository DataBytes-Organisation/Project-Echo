"use strict";

import { getAudioTestString } from "./HMI-utils.js";
import { getAudioRecorder } from "./audio_recorder.js";
import { retrieveTruthEventsInTimeRange, retrieveVocalizationEventsInTimeRange, 
  retrieveMicrophones, retrieveAudio, retrieveSimTime, postRecording, 
  setSimModeAnimal, setSimModeRecording, setSimModeRecordingV2, stopSimulator } from "./routes.js";

  //import data from "./sample_data.json" assert { type: 'json' }; Browser assertions not yet supported in all browsers, alternative method used instead.


// import { parse } from 'json2csv';


var markups = ["elephant.png", "monkey.png", "tiger.png"];

const EARTH_RADIUS = 6371000;
const MIC_DETECTION_RANGE = 300;
const MAX_RECORDING_TIME_S = "20";
const DEG_TO_RAD = (Math.PI / 180);
const RAD_TO_DEG = (180 / Math.PI);

var audioRecorder = getAudioRecorder();

var statuses = [
  "endangered",
  "vulnerable",
  "near-threatened",
  "normal",
  "invasive",
];

function matchStatus(status){
  if (status == "least concern"){
    return "normal";
  }
  else{
    return status;
  }
}

export function convertCSV(json) {
  if (json == null) return null
  if (typeof json === undefined | json.length === 0){
    return null
  }
  let data = json;
  let fields = Object.keys(data[0]);
  let replacer = function(key, value) { return value === null ? 'N/A' : value } ;

  let csv = null;
  csv = data.map(function(row){
    return fields.map(function(fieldName){
      return JSON.stringify(row[fieldName], replacer)
    }).join(',')
  })
  csv.unshift(fields.join(',')) // add header column
  csv = csv.join('\r\n');
  return csv
}

var statusPrintLookup = {
  "endangered" : "endangered",
  "vulnerable": "vulnerable",
  "near-threatened" : "near-threatened",
  "normal" : "least concern",
  "invasive" : "invasive"
};

var vocalizedLayers = []

var animalTypes = ["mammal", "bird", "amphibian", "reptile", "insect"];

var statusIconLookup = {
  "endangered" : "1",
  "vulnerable": "2",
  "near-threatened" : "3",
  "normal" : "4",
  "invasive" : "5"
};

var animalTypeIconLookup = {
  "mammal" : "Mammals",
  "bird" : "Bird",
  "amphibian" : "Amphibians",
  "insect" : "Insects",
  "reptile" : "Reptiles"
};

var selectedVocalizationEventId = null;

function getIconName(status, type){
  return animalTypeIconLookup[type] + statusIconLookup[status] + "-01.png";
}

//var sample_data = data.data; For assertion method
var sample_data = []; // For the universal method
fetch("./js/sample_data.json").then(res => res.json()).then(data => sample_data = data.data);
// Went from 4 lines to 1. This method works on all browsers according to previous tests.
var animal_data = [];

export var animal_toggled = false;

//For Demo purposes only
//This plays an awful quailty audio test string
document.addEventListener('click', function() {
  //playAudioString(getAudioTestString());  
});


export function initialiseHMI(hmiState) {
  console.log(`initialising`);
  createBasemap(hmiState);
  //console.log("Get sample element from document: ", document.getElementById("menuPanel"))
  addVocalisationLayers(hmiState);
  addTruthLayers(hmiState);
  
  
  for(let i=1; i<26; i++){
    addVectorLayerTopDown(hmiState, `mic_layer_${i}`);
    console.log(`Adding microphone layer:${i}`);
  }
  
  addVectorLayerTopDown(hmiState, "mic_layer");
  /*
  addVectorLayerTopDown(hmiState, "mic_layer");
  addVectorLayerTopDown(hmiState, "mic_layer_1");
  addVectorLayerTopDown(hmiState, "mic_layer_2");
  addVectorLayerTopDown(hmiState, "mic_layer_3");
  addVectorLayerTopDown(hmiState, "mic_layer_4");
  addVectorLayerTopDown(hmiState, "mic_layer_5");
  addVectorLayerTopDown(hmiState, "mic_layer_6");
  addVectorLayerTopDown(hmiState, "mic_layer_7");
  addVectorLayerTopDown(hmiState, "mic_layer_8");
  addVectorLayerTopDown(hmiState, "mic_layer_9");
  */
  addAllTruthFeatures(hmiState);
  addAllVocalizationFeatures(hmiState);

  createMapClickEvent(hmiState);

  retrieveMicrophones().then((res) => {
    
    updateMicrophoneLayer(hmiState, res.data);
    stepMicAnimation(hmiState);
  })
  addmicrophones(hmiState);
  stepMicAnimation(hmiState);
  queueSimUpdate(hmiState);
  //simulateData(hmiState);
}

function updateFilters(){
  
}

const validEmailRegex = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$/;
export function emailValidation(inp){
  let email_id = inp + "-email-inp";
  let error_id = inp + "-email-error";
  let btn_id = inp + "-button"
  let input_ele = document.getElementById(email_id);
  let error_ele = document.getElementById(error_id);
  let btn_ele = document.getElementById(btn_id);
  console.log("Toggle email: ", input_ele.value)
  if (input_ele.value.match(validEmailRegex)){
    error_ele.style.display = 'none';
    error_ele.innerHTML = '';
    btn_ele.disabled = false;
  } else {
    error_ele.style.display = 'block';
    error_ele.innerHTML = 'Please insert a valid email address';
    btn_ele.disabled = true;
  }
  
}

let latestSimAnimals = {};

export function updateAnimalMovementLayerFromPastData(hmiState, results){

  clearAllTruthLayers();

  hmiState.movementEvents = {};
  latestSimAnimals = {};

  let updateDict = {};

  //ensure events are unique per id
  for (let data of results) {
    if(latestSimAnimals.hasOwnProperty(data.animalId)){
      let entry = latestSimAnimals[data.animalId];
      if(entry.timestamp < data.timestamp){
        latestSimAnimals[data.animalId] = data;
        updateDict[data.animalId] = true;
      }
    }
    else{
      latestSimAnimals[data.animalId] = data;
      let event = convertJSONtoAnimalMovementEvent(hmiState, data);
      hmiState.movementEvents[event.animalId] = event;
    }
  }

  for(const key in updateDict){
    let event = convertJSONtoAnimalMovementEvent(hmiState, latestSimAnimals[key]);
    hmiState.movementEvents[event.animalId] = event;
  }

  addAllTruthFeatures(hmiState);
}

export function updateVocalizationLayerFromPastData(hmiState, results){

  clearAllVocalizationLayers();

  hmiState.vocalizationEvents = [];
  //console.log(results);

  for (let data of results) {
    let event = convertJSONtoAnimalVocalizationEvent(hmiState, data);
    hmiState.vocalizationEvents.push(event);
  }

  addAllVocalizationFeatures(hmiState);
}

export function updateMicrophoneLayer(hmiState, results){

  clearMicrophoneLayer();

  hmiState.microphoneLocations = [];
  //console.log(results);

  for (let data of results) {
    let location = convertJSONtoMicrophone(hmiState, data);
    if(location !== null){
      hmiState.microphoneLocations.push(location);
    }
  }

  addmicrophones(hmiState);
}


export function updateAnimalMovementLayerFromLiveData(hmiState, results){
  let newMovementEvents = [];
  let updatedMovementEvents = [];
  let updateDict = {};

  //ensure events are unique per id
  for (let data of results) {
    
    //console.log(data);

    if(latestSimAnimals.hasOwnProperty(data.animalId)){
      let entry = latestSimAnimals[data.animalId];
      if(entry.timestamp < data.timestamp){
        latestSimAnimals[data.animalId] = data;
        updateDict[data.animalId] = true;
      }
    }
    else{
      latestSimAnimals[data.animalId] = data;
      let event = convertJSONtoAnimalMovementEvent(hmiState, data);
      hmiState.movementEvents[event.animalId] = event;
      newMovementEvents.push(event); 
    }
  }

  for(const key in updateDict){
    let event = convertJSONtoAnimalMovementEvent(hmiState, latestSimAnimals[key]);
    hmiState.movementEvents[event.animalId] = event;
    updatedMovementEvents.push(event); 
  }

  //purge old events per id
  for(let evt of updatedMovementEvents){
    let layerName = deriveTruthLayerName(evt.animalStatus, evt.animalType);
    let layer = findMapLayerWithName(hmiState, layerName);
    if(layer){
      const featureToPurge = layer.getSource().getFeatureById(evt.animalId);
      //console.log(featureToPurge);
      if(featureToPurge){
        //console.log("purge lat: " + evt.locationLat + " lon: " + evt.locationLon);
        layer.getSource().removeFeature(featureToPurge);
      }
      else{
        console.log(layerName);
      }
    }
  }

  addNewTruthFeatures(hmiState, updatedMovementEvents);
  addNewTruthFeatures(hmiState, newMovementEvents);
}

export function updateVocalizationLayerFromLiveData(hmiState, results){

  let newVocalizationEvents = [];

  for (let data of results) {

    //console.log(data);

    let event = convertJSONtoAnimalVocalizationEvent(hmiState, data);
    hmiState.vocalizationEvents.push(event);
    newVocalizationEvents.push(event);
  }

  addNewVocalizationFeatures(hmiState, newVocalizationEvents);
}

export function resetWildlifeLayers(hmiState){
  hmiState.vocalizationEvents = [];
  hmiState.movementEvents = {};
  clearAllVocalizationLayers();
  clearAllTruthLayers();
}

export function clearAllVocalizationLayers(){
  for (let stat of statuses) {
    for (let animalType of animalTypes) {
      let nextName = stat + "_" + animalType;
      let layer = findMapLayerWithName(hmiState, nextName);
      layer.getSource().clear();
    }
  }
}

export function clearAllTruthLayers(){
  for (let stat of statuses) {
    for (let animalType of animalTypes) {
      let nextName = stat + "_" + animalType + "_truth";
      let layer = findMapLayerWithName(hmiState, nextName);
      layer.getSource().clear();
    }
  }
}

export function clearMicrophoneLayer(){
  let layer = findMapLayerWithName(hmiState, "mic_layer");
  layer.getSource().clear();
}

/*function commonNameFix(common){
  if (common == "orange footed scrubfowl"){
    return "orange-footed scrubfowl"
  }
  else{
    return common;
  }
}*/

export function convertJSONtoAnimalMovementEvent(hmiState, data){
  let movementEvent = {};

  //console.log(data);

  movementEvent.animalId = data.animalId;
  movementEvent.eventId = data._id;
  movementEvent.timestamp = Math.floor((getUTC() - hmiState.timeOffset - hmiState.simUpdateDelay) / 1000);
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

//TODO update external data properties
export function convertJSONtoAnimalVocalizationEvent(hmiState, data){
  let vocalizationEvent = {};

  //console.log(data);

  vocalizationEvent.timestamp = hmiState.currentTime;
  vocalizationEvent.eventTimestamp = data.timestamp;
  vocalizationEvent.eventId = data._id;

  //console.log(vocalizationEvent.eventId);
  
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

export function convertJSONtoMicrophone(hmiState, data){
  let mic = {};

  if(data.microphoneLLA !== null){
    //console.log(data);
    mic.id = data._id;
    mic.lat = data.microphoneLLA[0];
    mic.lon = data.microphoneLLA[1];
    return mic;
  }
  else{
    return null;
  }
}

export function muteAudioAnimation(){
  const mute_audio = new CustomEvent('muteAnimation',{
    detail: {
      message: "mute animation"
    }
  })

  document.dispatchEvent(mute_audio);
}

export function muteRecordingPlaybackAnimation(){
  const mute_audio = new CustomEvent('muteRecordingAnimation',{
    detail: {
      message: "mute animation"
    }
  })

  document.dispatchEvent(mute_audio);
}

var activeAudioNode = null;
var audioAnimTimeout = null;
var playNextTrack = false;

export function stopAudioPlayback(){
  muteAudioAnimation();

  if(audioAnimTimeout){
    clearTimeout(audioAnimTimeout);
  }
  if(activeAudioNode != null){
    console.log("calling stop");
    activeAudioNode.stop();
  }

  activeAudioNode = null;
}

function playAudioString(audioDataString, sampleRate) {
  // Convert the binary audio data string into a typed array
  const audioData = new Uint8Array(
    atob(audioDataString)
      .split("")
      .map((char) => char.charCodeAt(0))
  );
         
  // Create an audio context
  const audioContext = new AudioContext();
  
  const el = document.getElementById("player");
  // Create a new audio buffer
  const numChannels = 1;
  const bufferLength = audioData.length / 2;

  const audioBuffer = audioContext.createBuffer(
    numChannels,
    bufferLength,
    sampleRate
  );

  // Copy the binary audio data into the audio buffer
  audioBuffer.copyToChannel(new Float32Array(audioData.buffer), 0);

  let duration = audioBuffer.duration;
  //console.log(duration);

  // Create a new audio buffer source node and set its buffer
  activeAudioNode = audioContext.createBufferSource();
  activeAudioNode.buffer = audioBuffer;

  // Connect the source node to the destination node of the audio context
  activeAudioNode.connect(audioContext.destination);

  if(playNextTrack){
    // Start playing the audio
    activeAudioNode.start();

    audioAnimTimeout = setTimeout(
      muteAudioAnimation,
      duration*1000,
      hmiState
    );
  }
}

export function updateLayers(filterState)  {

  for (let stat of statuses) {
    for (let animalType of animalTypes) {
      let layerName = deriveLayerName(stat, animalType);
      let layer = findMapLayerWithName(hmiState, layerName);
      
      if (filterState.includes("_" + stat) && filterState.includes("_" + animalType))
      {
        layer.setVisible(true);
      }
      else{
        layer.setVisible(false);
      }
    }
  }

  for (let stat of statuses) {
    for (let animalType of animalTypes) {
      let layerName = deriveTruthLayerName(stat, animalType);
      let layer = findMapLayerWithName(hmiState, layerName);
      if (filterState.includes(stat) && filterState.includes(animalType))
      {
        layer.setVisible(true);
      }
      else{
        layer.setVisible(false);
      }
    }
  }
}

function addAllTruthFeatures(hmiState) {
  //console.log("addTruthLayers called.")
  //console.log("Truth locs", hmiState.movementEvents);
  for (const key in hmiState.movementEvents) {
    //console.log("True location found!:  ")

    let entry = hmiState.movementEvents[key];

    var iconPath = "";
    if(entry.animalDiet === "insectivore" || entry.animalDiet === "omnivore" || entry.animalDiet === "carnivore"){
      iconPath = './../images/Predator/sim/' + getIconName(entry.animalStatus, entry.animalType);
    }
    else{
      iconPath ='./../images/sim/' + getIconName(entry.animalStatus, entry.animalType);
    }
    
    var trueLocation = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([entry.locationLon,entry.locationLat])
      ),
        name: 'trueLocation_' + entry.speciesScientificName,
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
        isAnimalMovement: 1,
    });
      //console.log(entry.locationLon, " ", entry.locationLat)
    
    var trueIcon = new ol.style.Style({
      image: new ol.style.Icon({
        src: iconPath,
        anchor: [0.5, 1],
        scale: 0.75,
        className: 'true-icon'
      }),
    })
  
    trueLocation.setStyle(trueIcon);
    trueLocation.setId(entry.animalId);

    let layerName = deriveTruthLayerName(entry.animalStatus,entry.animalType);
    let layer = findMapLayerWithName(hmiState, layerName);
    let layerSource = layer.getSource();

    //console.log(layerName);
    //console.log("animal type: ", entry.speciesScientificName);

    layerSource.addFeature(trueLocation);
    layer.getSource().changed();
    layer.changed();
  }
}

function addNewTruthFeatures(hmiState, events) {
  //console.log("addTruthLayers called.")
  //console.log("Truth locs", hmiState.movementEvents);
  for (let entry of events) {
    //console.log("True location found!:  ")
    var iconPath = "";
    if(entry.animalDiet ==="omnivore" || entry.animalDiet === "carnivore" || entry.animalDiet === "insectivore" ){
      iconPath = './../images/Predator/sim/' + getIconName(entry.animalStatus, entry.animalType);
    }
    else{
      iconPath ='./../images/sim/' + getIconName(entry.animalStatus, entry.animalType);
    }

    var trueLocation = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([entry.locationLon, entry.locationLat])
      ),
        name: 'trueLocation_' + entry.speciesScientificName,
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
        isAnimalMovement: 1,
    });
      //console.log(entry.locationLon, " ", entry.locationLat)

    var trueIcon = new ol.style.Style({
      image: new ol.style.Icon({
        src: iconPath,
        anchor: [0.5, 1],
        scale: 0.75,
        className: 'true-icon'
      }),
    })

    trueLocation.setStyle(trueIcon);
    trueLocation.setId(entry.animalId);

    let layerName = deriveTruthLayerName(entry.animalStatus,entry.animalType);
    let layer = findMapLayerWithName(hmiState, layerName);
    let layerSource = layer.getSource();

    //console.log(layerName);

    layerSource.addFeature(trueLocation);
    layer.getSource().changed();
    layer.changed();
  }
}

function addAllVocalizationFeatures(hmiState) {
  //console.log("addTruthLayers called.")
  //console.log("Truth locs", hmiState.movementEvents);
  for (let entry of hmiState.vocalizationEvents) {
    //console.log("True location found!:  ")
    var iconPath = "";
    if(entry.animalDiet === "herbavore" || entry.animalDiet === "frugivore"){
      iconPath = './../images/vocalization/' + getIconName(entry.animalStatus, entry.animalType);
    }
    else{
      iconPath ='./../images/Predator/vocalization/' + getIconName(entry.animalStatus, entry.animalType);
    }

    var evtLocation = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([entry.locationLon,entry.locationLat])
      ),
        name: 'vocalisation_' + entry.speciesScientificName,
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
        isAnimalMovement: 0,
    });
      //console.log(entry.locationLon, " ", entry.locationLat)
    
    var icon = new ol.style.Style({
      image: new ol.style.Icon({
        src: iconPath,
        anchor: [0.5, 1],
        scale: 0.75,
        className: 'vocalization-icon'
      }),
    })
    evtLocation.setStyle(icon);
    evtLocation.setId(entry.animalId);

    let layerName = deriveLayerName(entry.animalStatus,entry.animalType);
    let layer = findMapLayerWithName(hmiState, layerName);
    if(layer){
      let layerSource = layer.getSource();

      //console.log(layerName);
      //console.log("animal type: ", entry.speciesScientificName);
  
      layerSource.addFeature(evtLocation);
      layer.getSource().changed();
      layer.changed();
    }
    else{
      console.log(layerName);
    }

  }
}

function addNewVocalizationFeatures(hmiState, events) {
  //console.log("addTruthLayers called.")
  //console.log("Truth locs", hmiState.movementEvents);
  for (let entry of events) {
    //console.log("True location found!:  ")
    var iconPath = "";
    if(entry.animalDiet === "herbavore" || entry.animalDiet === "frugivore"){
      iconPath = './../images/vocalization/' + getIconName(entry.animalStatus, entry.animalType);
    }
    else{
      iconPath ='./../images/Predator/vocalization/' + getIconName(entry.animalStatus, entry.animalType);
    }

    var evtLocation = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([entry.locationLon,entry.locationLat])
      ),
        name: 'vocalisation_' + entry.speciesScientificName,
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
        isAnimalMovement: 0,
    });
      //console.log(entry.locationLon, " ", entry.locationLat)

    var icon = new ol.style.Style({
      image: new ol.style.Icon({
        src: iconPath,
        anchor: [0.5, 1],
        scale: 0.75,
        className: 'vocalization-icon'
      }),
    })
    
    evtLocation.setStyle(icon);
    evtLocation.setId(entry.eventId);

    let layerName = deriveLayerName(entry.animalStatus,entry.animalType);
    let layer = findMapLayerWithName(hmiState, layerName);
    if(layer){
      let layerSource = layer.getSource();

      //console.log(layerName);
      //console.log("animal type: ", entry.speciesScientificName);
  
      layerSource.addFeature(evtLocation);
      layer.getSource().changed();
      layer.changed();
    }
    else{
      console.log(layerName);
    }
  }
}

function addMicrophonesByLayer(hmiState, layerName, iconPath){
  var mics = [];

  //console.log("locs", hmiState.microphoneLocations);
  hmiState.microphoneLocations.forEach((location) => {
    //console.log(location);
    // Add the marker into the array
    var mic = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([location.lon, location.lat])
      ),
      name: "mic",
      micLat: location.lat,
      micLon: location.lon,
      micIcon: iconPath,
      id: location.id,
      isMic: 1,
    });
    var icon = new ol.style.Style({
      image: new ol.style.Icon({
        src: iconPath,
        anchor: [0.5, 1],
        scale: 0.175,
      }),
    });
    mic.setStyle(icon);
    mics.push(mic);
  });

  //console.log("mics: ", mics);
  let layer = findMapLayerWithName(hmiState, layerName);
  let layerSource = layer.getSource();
  layerSource.addFeatures(mics);
  layer.getSource().changed();
  layer.changed();
}


function addMicrophonesByHiddenLayer(hmiState, layerName, iconPath){
  var mics = [];

  //console.log("locs", hmiState.microphoneLocations);
  hmiState.microphoneLocations.forEach((location) => {
    //console.log(location);
    // Add the marker into the array
    var mic = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([location.lon, location.lat])
      ),
      name: "mic",
    });
    var icon = new ol.style.Style({
      image: new ol.style.Icon({
        src: iconPath,
        anchor: [0.5, 1],
        scale: 0.175,
      }),
    });
    mic.setStyle(icon);
    mics.push(mic);
  });

  //console.log("mics: ", mics);
  let layer = findMapLayerWithName(hmiState, layerName);
  let layerSource = layer.getSource();
  layerSource.addFeatures(mics);
  layer.getSource().changed();
  layer.changed();
  layer.setVisible(false);
}

var micAnimFrameIndex = 1;
var animTimeout = null;

function addmicrophones(hmiState) {
  
  //Loop function to add all the frames that compose the microphone animation.
  for(let i=25; i>0;i--){
    addMicrophonesByLayer(hmiState,`mic_layer_${i}`, `./../images/${i}-01.png`);
  }
  //Add the single white dot as the microphone icon for when animation is disabled.
  addMicrophonesByHiddenLayer(hmiState, "mic_layer", "./../images/1-01.png");
}

export function enableMicAnimation(hmiState){
  var staticLayer = findMapLayerWithName(hmiState, "mic_layer");
  staticLayer.setVisible(false);

  for(var i = 1; i <= 25; i++){
    var nextLayer = findMapLayerWithName(hmiState, "mic_layer_" + i);
    nextLayer.setVisible(true);
  }

  stepMicAnimation(hmiState);
}

export function disableMicAnimation(hmiState){
  if(animTimeout){
    clearTimeout(animTimeout);
  }

  for(var i = 1; i <= 25; i++){
    var nextLayer = findMapLayerWithName(hmiState, "mic_layer_" + i);
    nextLayer.setVisible(false);
  }
}

function stepMicAnimation(hmiState) {
  var currentIndex = micAnimFrameIndex;
  micAnimFrameIndex = (micAnimFrameIndex % 25) + 1;

  var nextLayer = findMapLayerWithName(hmiState, "mic_layer_" + micAnimFrameIndex);
  nextLayer.setVisible(true);

  var currentLayer = findMapLayerWithName(hmiState, "mic_layer_" + currentIndex);
  currentLayer.setVisible(false);

  if(animTimeout){
    clearTimeout(animTimeout);
  }

  animTimeout = setTimeout(
    stepMicAnimation,
    100,
    hmiState
  );
}

export function showMics(hmiState){
  let layer = findMapLayerWithName(hmiState, "mic_layer");
  let layerSource = layer.getSource();
  layer.setVisible(true);
}

export function hideMics(hmiState){
  let layer = findMapLayerWithName(hmiState, "mic_layer");
  let layerSource = layer.getSource();
  layer.setVisible(false);
}

function findMapLayerWithName(hmiState, name) {
  if (!hmiState.basemap) {
    console.log(`findMapLayerWithName: invalid basemap`);
    return null;
  } else {
    if(hmiState.layers.hasOwnProperty(name)){
      return hmiState.layers[name];
    }
  }

  console.log(`findMapLayerWithName: layer not found: ` + name);
  return null;
}

function findMapLayerWithName_Deprecated(hmiState, name) {
  if (!hmiState.basemap) {
    console.log(`findMapLayerWithName: invalid basemap`);
    return null;
  } else {
    let mapLayers = hmiState.basemap.getLayers();

    for (let i = 0; i < mapLayers.getLength(); ++i) {
      let currentLayer = mapLayers.item(i);
      if (currentLayer.get("name") == name.toLowerCase()) {
        return currentLayer;
      }
    }
  }

  console.log(`findMapLayerWithName: layer not found: ` + name);
  return null;
}

function addVectorLayerTopDown(hmiState, layerName) {
  addVectorLayerToBasemap(hmiState, layerName, hmiState.layerPool);
  hmiState.layerPool = hmiState.layerPool - 1;
}

function addVectorLayerToBasemap(hmiState, layerName, zIndex) {
  if (!hmiState.basemap) {
    console.log(`findMapLayerWithName: invalid basemap`);
    return null;
  } else {
    let layer = new ol.layer.Vector({
      name: layerName,
      source: new ol.source.Vector(),
      visible: true,
    });

    if (zIndex != 0) {
      layer.setZIndex(zIndex);
    }

    hmiState.basemap.addLayer(layer);
    hmiState.layers[layerName] = layer;
  }
}

function addVocalisationLayers(hmiState) {
  // Wildlife layers
  // Invasive, Normal, Near-Threatened, Vulnerable, Endangered
  // Mammal, Reptile, Predator, Bird, Amphibian
  for (let stat of statuses) {
    for (let animalType of animalTypes) {
      let nextName = stat + "_" + animalType;

      addVectorLayerTopDown(hmiState, nextName);
    }
  }
}

function addTruthLayers(hmiState) {
  // Wildlife layers
  // Invasive, Normal, Near-Threatened, Vulnerable, Endangered
  // Mammal, Reptile, Predator, Bird, Amphibian
  for (let stat of statuses) {
    for (let animalType of animalTypes) {
      let nextName = stat + "_" + animalType + "_truth";

      addVectorLayerTopDown(hmiState, nextName);
    }
  }
}

function deriveLayerName(status, animalType) {
  return status + "_" + animalType;
}

function deriveTruthLayerName(status, animalType) {
  return status + "_" + animalType + "_truth";
}

function createBasemap(hmiState) {

  var basemap = new ol.Map({
    target: "basemap",
    featureEvents: true,
    controls: ol.control.defaults({
      zoom: false,
    }),
    interactions: ol.interaction.defaults({
      constrainResolution: false,
    }),
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
      zoom: hmiState.defaultZoom,
    }),
  });

  hmiState.basemap = basemap;
  return basemap;
}

var current_mic_lat = 0.0;
var current_mic_lon = 0.0;
var current_mic_id = "";


function createMapClickEvent(hmiState){
  hmiState.basemap.on("click", function (evt) {
    const feature = hmiState.basemap.forEachFeatureAtPixel(evt.pixel, function (feature) {
      return feature;
    });

    let active_content = $("#animal-popup-content");
    let default_content = $("#animal-default-content");
    let active_mic_content = $("#mic-popup-content");
    let default_mic_content = $("mic-default-content");
    if (feature){

      let values = feature.getProperties();
      if (values.isMic){
        active_mic_content.show();
        default_mic_content.hide();

        const img = new Image();
        let dice = Math.floor(Math.random() * 4) + 1;
        img.onload = function() {
          //console.log('Image exists!');
          document.getElementById("mic_desc_img").src = "../../images/bio/mic_bio_" + dice + ".png";
        }
        img.onerror = function() {
          console.log('Mic image does not exist!');
        }
        img.src = "../../images/bio/mic_bio_" + dice + ".png";
        //document.getElementById("mic_desc_name").innerText = result.common;
        document.getElementById("mic_desc_id").innerText = values.id;
        //document.getElementById("desc_summary").innerText = result.summary;

        //Markup details specific session
        let dateFormat = new Date();
        document.getElementById("mic_markup_img").src = values.micIcon;
        document.getElementById("mic_markup_details").innerHTML = "Microphone";
        document.getElementById("mic_markup_loc_lat").innerHTML = values.micLat;
        document.getElementById("mic_markup_loc_lon").innerHTML = values.micLon;
        document.getElementById("mic_markup_date").innerHTML = dateFormat.toUTCString()

        current_mic_lat = values.micLat;
        current_mic_lon = values.micLon;
        current_mic_id = values.id;


        animal_toggled = true;
        const toggled_mic = new CustomEvent('micToggled',{
        detail: {
          message: "Mic toggled:",
          }
        })
        
        document.dispatchEvent(toggled_mic);
      }
      else{
        // console.log('feature: ', feature);
        stopAudioPlayback();

        active_content.show();
        default_content.hide();
        if (values.eventId){
          //console.log("saving " + values.eventId);
          selectedVocalizationEventId = values.eventId;
        }
        if(values.isAnimalMovement){
          document.getElementById("audioHeader").style.display = "none"
          document.getElementById("audioControl").style.display = "none"
        }
        else{
          document.getElementById("audioHeader").style.display = "flex"
          document.getElementById("audioControl").style.display = "flex"
        }
        if (values.animalSpecies){
          //console.log(values.animalSpecies)
          var result = sample_data.find(({ species }) => species.toLowerCase() === values.animalSpecies.toLowerCase())
          if (result) {
            const img = new Image();
            img.onload = function() {
              //console.log('Image exists!');
              // Set the source of the img tag
              document.getElementById("desc_img").src = "../../images/bio/" + result.common.toLowerCase() + "-bio.png";
            }
            img.onerror = function() {
              let dice = Math.floor(Math.random() * 5) + 1;
              document.getElementById("desc_img").src = "../../images/bio/not_available_" + dice + "-bio.png";
            }
            img.src = "../../images/bio/" + result.common.toLowerCase() + "-bio.png";
          
            //console.log("found")
            //Animal Bio specific session
            animal_data = result;
            document.getElementById("desc_name").innerText = result.common;
            //document.getElementById("markup_img_2").src = values.animalIcon;
            document.getElementById("desc_confidence").innerText = values.animalConfidence + "%";
            document.getElementById("desc_species").innerText = result.species;
            document.getElementById("desc_summary").innerText = result.summary;

            let summary = document.getElementById("desc_details");
            summary.innerHTML = '';
            result.description.forEach(content => {
              if (content){
                var p = document.createElement('p');
                p.className = "desc_ul";
                p.innerText = content;
                summary.appendChild(p);
              }
            })
          }
          else{
            let dice = Math.floor(Math.random() * 5) + 1;
            document.getElementById("desc_img").src = "../../images/bio/not_available_" + dice + "-bio.png";

            document.getElementById("desc_name").innerText = values.animalSpecies;
            //document.getElementById("markup_img_2").src = values.animalIcon;
            document.getElementById("desc_confidence").innerText = values.animalConfidence + "%";
            document.getElementById("desc_species").innerText = values.animalSpecies;
            document.getElementById("desc_summary").innerText = "Bio data coming soon.";
            let summary = document.getElementById("desc_details");
            summary.innerHTML = '';
          }


            //Markup details specific session
            let dateFormat = new Date(values.animalRecordDate);
            document.getElementById("markup_img").src = values.animalIcon;
            document.getElementById("markup_details").innerHTML = values.animalType + " | " + values.animalDiet + " | " + statusPrintLookup[values.animalStatus];
            document.getElementById("markup_loc_lon").innerHTML = values.animalLon;
            document.getElementById("markup_loc_lat").innerHTML = values.animalLat;
            document.getElementById("markup_confidence").innerHTML = values.animalLocConfidence + "%";
            document.getElementById("markup_date").innerHTML = dateFormat.toUTCString()

            animal_toggled = true;
            const toggled_animal = new CustomEvent('animalToggled',{
              detail: {
                message: "Animal toggled:",
              }
            })

            document.dispatchEvent(toggled_animal);
          

        }
        else{
          console.log(values);
        }
      }
    } else {
      active_content.hide();
      active_mic_content.hide();
      default_content.show();
      default_mic_content.show();
    }
  });
}

export function MapOpenNav() {
  if (animal_toggled){
    document.getElementById("menuPanel").style.width = "30%";
  }
}

export function getAnimalToggled(){
  return animal_toggled;
}

export function MapCloseNav() {
  document.getElementById("menuPanel").style.width = "0";
  animal_toggled = false;
}

function updateTruthEvents(hmiState){
  retrieveTruthEventsInTimeRange(hmiState.currentTime-5, hmiState.currentTime).then((res) => {
    updateAnimalMovementLayerFromLiveData(hmiState, res.data);
  })
}

function updateVocalizationEvents(hmiState){
  retrieveVocalizationEventsInTimeRange(hmiState.currentTime-5, hmiState.currentTime).then((res) => {
    updateVocalizationLayerFromLiveData(hmiState, res.data);
  })
}

function purgeTruthEvents(hmiState){
  let persistEvents = {};
  //console.log("purging");

  //console.log(hmiState.liveEventCutoff);
  for(const key in hmiState.movementEvents){
    let event = hmiState.movementEvents[key];
    //console.log(event);

    if(hmiState.liveEventCutoff > event.timestamp){
      let layerName = deriveTruthLayerName(event.animalStatus, event.animalType);
      let layer = findMapLayerWithName(hmiState, layerName);
      if(layer){
        const featureToPurge = layer.getSource().getFeatureById(event.animalId);
        //console.log(featureToPurge);
        layer.getSource().removeFeature(featureToPurge);
      }
      else{
        console.log(layerName);
      }
    }
    else{
      persistEvents[event.animalId] = event;
    }
  }

  hmiState.movementEvents = persistEvents;
}

function purgeVocalizationEvents(hmiState){
  let persistEvents = [];
  //console.log("purging");

  //console.log(hmiState.liveEventCutoff);
  for(let event of hmiState.vocalizationEvents){
    //console.log(event);
    //console.log(hmiState.liveEventCutoff);
    //console.log(event.timestamp); 

    if(hmiState.liveEventCutoff > event.timestamp){
      let layerName = deriveLayerName(event.animalStatus, event.animalType);
      let layer = findMapLayerWithName(hmiState, layerName);
      if(layer){
        //console.log(event.eventId);
        const featureToPurge = layer.getSource().getFeatureById(event.eventId);
        //console.log(featureToPurge);
        if(featureToPurge !== null){
          layer.getSource().removeFeature(featureToPurge);
        }
      }
      else{
        console.log(layerName);
      }
    }
    else{
      persistEvents.push(event);
    }
  }

  hmiState.vocalizationEvents = persistEvents;
}

function simulateData(hmiState) {

  queueSimUpdate(hmiState);
}

export function getUTC(){
  const now = new Date();
  const utcTimestamp = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(),
    now.getUTCHours(), now.getUTCMinutes(), now.getUTCSeconds(), now.getUTCMilliseconds());
  return utcTimestamp;
}

export function updateTimeOffset(hmiState){
  try{
  retrieveSimTime().then((res) => {
    //console.log(res.data);
    let unix = Date.parse(res.data.timestamp) / 1000;
    let newDelay = (getUTC() - new Date((unix + (10*60*60)) * 1000)) + 1000;
    if(isNaN(newDelay)){
      hmiState.simUpdateDelay = 10000;
    }
    else{
      hmiState.simUpdateDelay = newDelay;
    }
    // Multiply by 1000 to convert to milliseconds
    /*const utcDate = new Date(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate(),
      date.getUTCHours(), date.getUTCMinutes(), date.getUTCSeconds());
    hmiState.simTime = utcDate;*/
    //console.log(hmiState.simUpdateDelay);
  });
  }catch(error){
    console.log(failed);
    hmiState.simUpdateDelay = 10000;
  }
}

let simUpdateTimeout = null;

function queueSimUpdate(hmiState) {
  if(hmiState.liveMode){
    updateTimeOffset(hmiState);

    hmiState.currentTime = Math.floor((getUTC() - hmiState.timeOffset - hmiState.simUpdateDelay) / 1000);
    hmiState.liveEventCutoff = Math.floor((getUTC() - hmiState.timeOffset - hmiState.simUpdateDelay - hmiState.liveWindow) / 1000);
                
    purgeTruthEvents(hmiState);
    purgeVocalizationEvents(hmiState);
    updateTruthEvents(hmiState);
    updateVocalizationEvents(hmiState);
    hmiState.previousUpdateTime = hmiState.currentTime;
  }

  if (simUpdateTimeout) {
    clearTimeout(simUpdateTimeout);
  }

  //refresh layers
  for (let stat of statuses) {
    for (let animalType of animalTypes) {
      let layerName = deriveTruthLayerName(stat, animalType);
      let layer = findMapLayerWithName(hmiState, layerName);
      layer.changed();
      layer.getSource().changed();
    }
  }

  simUpdateTimeout = setTimeout(
    simulateData,
    hmiState.requestInterval,
    hmiState
  );
}



document.addEventListener('playAudio', function(event){
  //console.log("play audio");
  playNextTrack = true;
  if(selectedVocalizationEventId != null){
    retrieveAudio(selectedVocalizationEventId).then((res) => {
      playAudioString(res.data.audioClip, res.data.sampleRate);
    })
  }
})

document.addEventListener('stopAudio', function(event){
  //console.log("stop audio");
  playNextTrack = false;
  stopAudioPlayback();
})

var durationTag = document.getElementById("recording_duration");


var audioElement = document.getElementsByClassName("audio-element")[0];
var audioElementSource = document.getElementsByClassName("audio-element")[0]
    .getElementsByTagName("source")[0];
audioElement.onended = hidePlaybackIndicator;
var textIndicatorOfAudiPlaying = document.getElementsByClassName("playback_indicator")[0];

var recordButton = document.getElementById("record_audio_button");
//recordButton.onclick = startAudioRecording;

var recordingControls = document.getElementsByClassName("recording_controls")[0];

export function showRecordingControls() {
  console.log("showing controls")
  recordButton.style.display = "none";
  recordingControls.classList.remove("hide");
  initializeRecordingDuration();
}

export function hideRecordingControls() {
  console.log("hiding controls")
  recordButton.style.display = "block";
  recordingControls.classList.add("hide");
  clearInterval(durationTimer);
}

//var overlay = document.getElementsByClassName("overlay")[0];
//var acknowledgeButton = document.getElementById("acknowledge_button");
//acknowledgeButton.onclick = hideRecordingNotSupportedOverlay;

export function showRecordingNotSupportedOverlay() {
    //overlay.classList.remove("hide");
}

function hideRecordingNotSupportedOverlay() {
    //overlay.classList.add("hide");
}

export function createSourceForAudioElement() {
    let sourceElement = document.createElement("source");
    audioElement.appendChild(sourceElement);

    audioElementSource = sourceElement;
}

export function showPlaybackIndicator() {
    textIndicatorOfAudiPlaying.classList.remove("hide");
}

export function hidePlaybackIndicator() {
    textIndicatorOfAudiPlaying.classList.add("hide");
}

var audioRecordStartTime = null;
var durationTimer = null;

export function testFunct(){
  console.log("Recording started 1");
}

export function startAudioRecording() {
  console.log("Recording started 2");

  if (!audioElementSource){}
  else if (!audioElement.paused) {
    console.log("Paused playback");
    audioElement.pause();
    hidePlaybackIndicator();
  }

  audioRecorder.start()
    .then(() => {
      audioRecordStartTime = new Date();
        showRecordingControls();
      })
    .catch(error => {
      console.log(error.message);

      if (error.message.includes("mediaDevices API or getUserMedia method is not supported in this browser.")) {
        console.log("To record audio, use browsers like Chrome and Firefox.");
        showRecordingNotSupportedOverlay();
      }

      switch (error.name) {
        case 'AbortError': 
          console.log("An AbortError has occured.");
          break;
        case 'NotAllowedError': 
          console.log("A NotAllowedError has occured. User might have denied permission.");
          break;
        case 'NotFoundError': 
          console.log("A NotFoundError has occured.");
          break;
        case 'NotReadableError': 
          console.log("A NotReadableError has occured.");
          break;
        case 'SecurityError': 
          console.log("A SecurityError has occured.");
          break;
        case 'TypeError': 
          console.log("A TypeError has occured.");
          break;
        case 'InvalidStateError': 
          console.log("An InvalidStateError has occured.");
          break;
        case 'UnknownError': 
          console.log("An UnknownError has occured.");
          break;
        default:
          console.log("An error occured with the error name " + error.name);
        };
      });
}


export function stopAudioRecording() {
  console.log("Stopped recording...");

  audioRecorder.stop()
    .then(audioAsblob => {
      playAudio();
      hideRecordingControls();
    })
    .catch(error => {
      switch (error.name) {
        case 'InvalidStateError':
          console.log("An InvalidStateError has occured.");
          break;
        default:
          console.log("ERROR: " + error.name);
      };
    });
}

export function cancelAudioRecording() {
  console.log("Cancelled recording");

  audioRecorder.cancel();
  hideRecordingControls();
}

document.addEventListener('saveRecording', function(event){
  save();
})

document.addEventListener('loadRecording', function(event){
  //console.log("stop audio");
  //playNextTrack = false;
  //stopAudioPlayback();
})

document.addEventListener('simulateRecording', function(event){
  //simulateRecording(audioRecorder.audioBlobs, window.hmiState);
  simulateRecording(window.hmiState);
})

function generateRandomCoordinate(latitude, longitude) {
  const latRad = latitude * DEG_TO_RAD;
  const lonRad = longitude * DEG_TO_RAD;

  const dist = (Math.random() * MIC_DETECTION_RANGE) + 50;

  const theta = Math.random() * 2 * Math.PI;

  const newLatRad = latRad + (dist / EARTH_RADIUS) * Math.cos(theta);
  const newLonRad = lonRad + (dist / EARTH_RADIUS) * Math.sin(theta);

  const newLat = newLatRad * RAD_TO_DEG;
  const newLon = newLonRad * RAD_TO_DEG;

  return { lat: newLat, lon: newLon };
}

function arrayBufferToBase64(buffer) {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function simulateRecording(hmiState){
  if (decodedAudioStore.length === 0) {
    console.log("No recording available.");
    return;
  }else{

    const base64String = arrayBufferToBase64(fileContent);
    
    let coords = generateRandomCoordinate(current_mic_lat, current_mic_lon);

    const recordingData = {
      timestamp: Math.floor((getUTC() - hmiState.timeOffset - hmiState.simUpdateDelay) / 1000),
      sensorId: current_mic_id,
      microphoneLLA: [current_mic_lat, current_mic_lon, 0.0],
      animalEstLLA: [coords.lat, coords.lon, 0.0],
      animalTrueLLA: [coords.lat, coords.lon, 0.0],
      animalLLAUncertainty: 50.0,
      audioClip: base64String,
      mode: hmiState.simMode,
      audioFile: hmiState.simMode
    }
  
    postRecording(recordingData);
      
    //let decodedAudio = stringToAudio(base64String);

    //playAudioFromUint8Array(decodedAudio);
    //testEncoding(base64String);
  }
}

function testEncoding(base64String){
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const binaryData = atob(base64String);

  const float32Array = new Float32Array(binaryData.length / Float32Array.BYTES_PER_ELEMENT);
  const view = new DataView(float32Array.buffer);
  for (let i = 0; i < binaryData.length; i++) {
    view.setUint8(i, binaryData.charCodeAt(i));
  }

  const newBuffer = audioContext.createBuffer(1, float32Array.length, audioContext.sampleRate);
  newBuffer.copyToChannel(float32Array, 0);

  const audioSource = audioContext.createBufferSource();
  audioSource.buffer = newBuffer;
  audioSource.connect(audioContext.destination);
  audioSource.start();
}

function stringToAudio(base64String) {
  const binaryString = atob(base64String);

  const uint8Array = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    uint8Array[i] = binaryString.charCodeAt(i);
  }

  return uint8Array;
}

function playAudioFromUint8Array(uint8Array) {
  const blob = new Blob([uint8Array], { type: 'audio/wav' }); // Change the MIME type if needed

  const audioUrl = URL.createObjectURL(blob);

  const audioElement = new Audio(audioUrl);

  audioElement.play();
}


function save() {
  if(audioRecorder.audioBlobs.length != 0){
    let exportData = {};
    exportData.audioBlobs = [];

    audioRecorder.audioBlobs.forEach(item => {
      exportData.audioBlobs.push(btoa(item));
    });

    const base64AudioDataArray = audioRecorder.audioBlobs.map(blob => {
      const reader = new FileReader();
      return new Promise((resolve, reject) => {
          reader.onloadend = () => {
              resolve(reader.result.split(',')[1]); // Extract Base64 data
          };
          reader.readAsDataURL(blob);
      });
    });

    var jsonDataStr = "";

    Promise.all(base64AudioDataArray)
      .then(audioDataArray => {
          const jsonData = {
              audioBlobs: audioDataArray,
          };

          jsonDataStr = JSON.stringify(jsonData);
          

          const filename = prompt("Enter a filename for the JSON file:", "data.json");

          if (filename) {
      
            const jsonDataString = jsonDataStr;
            const blob = new Blob([jsonDataString], { type: 'application/json' });
      
            const blobURL = URL.createObjectURL(blob);
      
            const downloadLink = document.createElement('a');
            downloadLink.href = blobURL;
            downloadLink.download = filename;
            downloadLink.textContent = 'Download JSON';
      
            document.getElementById("downloadLink").innerHTML = "";
            document.getElementById("downloadLink").appendChild(downloadLink);
            console.log(jsonDataString);
            downloadLink.click();
          }
      })
      .catch(err => console.error("Error processing audio blobs: ", err));
  }
}

var fileInput = document.getElementById("fileInput");

var playNextRecordedTrack = false;
var audioAnimTimeout = null;

document.addEventListener('playRecordedAudio', function(event){
  //console.log("play audio");
  playNextRecordedTrack = true;
  playAudio()
})

document.addEventListener('stopRecordedAudio', function(event){
  //console.log("stop audio");
  playNextRecordedTrack = false;
  stopRecordingPlayback();
})

var audioRecordingElement = null;
var recordingPlaybackAnimTimeout = null;

function playRecording(recordedChunks) {
    if (recordedChunks.length === 0) {
        console.log("No recording available.");
        return;
    }else{

      const blob = new Blob(recordedChunks, { type: 'audio/wav; codecs=MS_PCM' });

      if(blob.size > 0){
        const url = URL.createObjectURL(blob);
        audioRecordingElement = document.getElementById("audioElem");
  
        audioRecordingElement.src = url;
        audioRecordingElement.load();
        if(playNextRecordedTrack){
          recordingPlaybackAnimTimeout = setTimeout(
            muteRecordingPlaybackAnimation,
            10000,
            hmiState
          );
          audioRecordingElement.play();
        }
      }
      else{
        console.log("Recording failed check microphone configuration settings.");
      }
    }
}


var audioContext = null;
var a_source = null;
var decodedAudioStore = null;
var fileContent = null;

fileInput.addEventListener("change", function(event) {
  const selectedFile = event.target.files[0];
  audioContext = new (window.AudioContext || window.webkitAudioContext)();

  if (selectedFile) {
    const reader = new FileReader();

    reader.onload = function(event) {
      fileContent = event.target.result;
      audioContext.decodeAudioData(fileContent.slice(0), function(decodedAudio) {

          decodedAudioStore = decodedAudio;
          a_source = audioContext.createBufferSource();
          a_source.buffer = decodedAudioStore;

          a_source.connect(audioContext.destination);

          audioElement.src = URL.createObjectURL(selectedFile);
      });
    };

    reader.readAsArrayBuffer(selectedFile);
  }
});

function stopRecordingPlayback(){
  muteRecordingPlaybackAnimation();

  if(recordingPlaybackAnimTimeout){
    clearTimeout(recordingPlaybackAnimTimeout);
  }

  if(a_source == null){
    if(audioRecordingElement != null){
      audioRecordingElement.pause();
      audioRecordingElement.currentTime = 0; 
    }
  }
  else{
    a_source.stop();
  }
}

export function playAudio() {
  console.log("play");
  if(a_source == null){
    playRecording(audioRecorder.audioBlobs);
  }
  else{
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    if(playNextRecordedTrack){
      recordingPlaybackAnimTimeout = setTimeout(
        muteRecordingPlaybackAnimation,
        10000,
        hmiState
      );
      a_source = audioContext.createBufferSource();
      a_source.buffer = decodedAudioStore;

      a_source.connect(audioContext.destination);
      a_source.start();
    }
  }
}

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
    durationTag.innerHTML = duration;

    if (checkAudioDurationThreshold(duration)) {
        stopAudioRecording();
    }
}

export function checkAudioDurationThreshold(duration) {
    let timers = duration.split(":");

    if (timers[2] === MAX_RECORDING_TIME_S)
        return true;
    else 
        return false;
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

    return  "00:" + minutes + ":" + seconds;
}

document.addEventListener('modeSwitch', function(event){
  window.hmiState.simMode = event.detail.message;
  if(window.hmiState.simMode == "Animal_Mode"){
    setSimModeAnimal();
  }
  else if(window.hmiState.simMode == "Recording_Mode"){
    setSimModeRecording();
  }
  else if(window.hmiState.simMode == "Recording_Mode_V2"){
    setSimModeRecordingV2();
  }
  else if(window.hmiState.simMode == "Stop"){
    stopSimulator();
  }
});