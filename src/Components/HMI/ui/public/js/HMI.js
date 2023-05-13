"use strict";

import { getAudioTestString } from "./HMI-utils.js";
import { retrieveTruthEventsInTimeRange, retrieveVocalizationEventsInTimeRange, 
  retrieveMicrophones, retrieveAudio, retrieveSimTime } from "./routes.js";
import data from "./sample_data.json" assert { type: 'json' };

var markups = ["elephant.png", "monkey.png", "tiger.png"];

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

var sample_data = data.data;

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
  addTruthLayers(hmiState);
  addVocalisationLayers(hmiState);
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

  addAllTruthFeatures(hmiState);
  addAllVocalizationFeatures(hmiState);

  createMapClickEvent(hmiState);

  retrieveMicrophones().then((res) => {
    
    updateMicrophoneLayer(hmiState, res.data);
    stepMicAnimation(hmiState);
  })
  
  queueSimUpdate(hmiState);
  //simulateData(hmiState);
}

function updateFilters(){
  
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
    const featureToPurge = layer.getSource().getFeatureById(evt.animalId);
    //console.log(featureToPurge);
    if(featureToPurge){
      //console.log("purge lat: " + evt.locationLat + " lon: " + evt.locationLon);
      layer.getSource().removeFeature(featureToPurge);
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
    if(entry.animalDiet === "herbavore" || entry.animalDiet === "frugivore"){
      iconPath = './../images/sim/' + getIconName(entry.animalStatus, entry.animalType);
    }
    else{
      iconPath ='./../images/Predator/sim/' + getIconName(entry.animalStatus, entry.animalType);
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
    if(entry.animalDiet === "herbavore" || entry.animalDiet === "frugivore"){
      iconPath = './../images/sim/' + getIconName(entry.animalStatus, entry.animalType);
    }
    else{
      iconPath ='./../images/Predator/sim/' + getIconName(entry.animalStatus, entry.animalType);
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
    let layerSource = layer.getSource();

    //console.log(layerName);
    //console.log("animal type: ", entry.speciesScientificName);

    layerSource.addFeature(evtLocation);
    layer.getSource().changed();
    layer.changed();
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
    evtLocation.setId(entry.animalId);

    let layerName = deriveLayerName(entry.animalStatus,entry.animalType);
    let layer = findMapLayerWithName(hmiState, layerName);
    let layerSource = layer.getSource();

    //console.log(layerName);
    //console.log("animal type: ", entry.speciesScientificName);

    layerSource.addFeature(evtLocation);
    layer.getSource().changed();
    layer.changed();
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
    });
    var icon = new ol.style.Style({
      image: new ol.style.Icon({
        src: iconPath,
        anchor: [0.5, 1],
        scale: 0.75,
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
        scale: 1.0,
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

  addMicrophonesByLayer(hmiState, "mic_layer_9", "./../images/Microphone - 3-ai-9.png");
  addMicrophonesByLayer(hmiState, "mic_layer_8", "./../images/Microphone - 3-ai-8.png");
  addMicrophonesByLayer(hmiState, "mic_layer_7", "./../images/Microphone - 3-ai-7.png");
  addMicrophonesByLayer(hmiState, "mic_layer_6", "./../images/Microphone - 3-ai-6.png");
  addMicrophonesByLayer(hmiState, "mic_layer_5", "./../images/Microphone - 3-ai-5.png");
  addMicrophonesByLayer(hmiState, "mic_layer_4", "./../images/Microphone - 3-ai-4.png");
  addMicrophonesByLayer(hmiState, "mic_layer_3", "./../images/Microphone - 3-ai-3.png");
  addMicrophonesByLayer(hmiState, "mic_layer_2", "./../images/Microphone - 3-ai-2.png");
  addMicrophonesByLayer(hmiState, "mic_layer_1", "./../images/Microphone - 3-ai-1.png");

  addMicrophonesByHiddenLayer(hmiState, "mic_layer", "./../images/mic2.png");
}

export function enableMicAnimation(hmiState){
  var staticLayer = findMapLayerWithName(hmiState, "mic_layer");
  staticLayer.setVisible(false);

  for(var i = 1; i <= 9; i++){
    var nextLayer = findMapLayerWithName(hmiState, "mic_layer_" + i);
    nextLayer.setVisible(true);
  }

  stepMicAnimation(hmiState);
}

export function disableMicAnimation(hmiState){
  if(animTimeout){
    clearTimeout(animTimeout);
  }

  for(var i = 1; i <= 9; i++){
    var nextLayer = findMapLayerWithName(hmiState, "mic_layer_" + i);
    nextLayer.setVisible(false);
  }
}

function stepMicAnimation(hmiState) {
  var currentIndex = micAnimFrameIndex;
  micAnimFrameIndex = (micAnimFrameIndex % 9) + 1;

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

function createMapClickEvent(hmiState){
  hmiState.basemap.on("click", function (evt) {
    const feature = hmiState.basemap.forEachFeatureAtPixel(evt.pixel, function (feature) {
      return feature;
    });

    let active_content = $("#animal-popup-content");
    let default_content = $("#animal-default-content");
    if (feature){
      // console.log('feature: ', feature);
      stopAudioPlayback();

      active_content.show();
      default_content.hide();

      let values = feature.getProperties();
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
            document.getElementById("desc_confidence").innerText = values.animalLocConfidence + "%";
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
            document.getElementById("desc_confidence").innerText = values.animalLocConfidence + "%";
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
            document.getElementById("markup_confidence").innerHTML = values.animalConfidence + "%";
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
    } else {
      active_content.hide();
      default_content.show();
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
    //TODO also update vocalisation layer here.
  })
}

//TODO Implement this
function updateVocalizationEvents(hmiState){
  retrieveVocalizationEventsInTimeRange(hmiState.currentTime-5, hmiState.currentTime).then((res) => {
    updateVocalizationLayerFromLiveData(hmiState, res.data);
    //TODO also update vocalisation layer here.
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
      const featureToPurge = layer.getSource().getFeatureById(event.animalId);
      //console.log(featureToPurge);
      layer.getSource().removeFeature(featureToPurge);
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

    if(hmiState.liveEventCutoff > event.timestamp){
      let layerName = deriveLayerName(event.animalStatus, event.animalType);
      let layer = findMapLayerWithName(hmiState, layerName);
      const featureToPurge = layer.getSource().getFeatureById(event.eventId);
      //console.log(featureToPurge);
      layer.getSource().removeFeature(featureToPurge);
    }
    else{
      persistEvents[event.animalId] = event;
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
    let newDelay = (getUTC() - new Date((unix + (10*60*60)) * 1000)) + 10000;
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