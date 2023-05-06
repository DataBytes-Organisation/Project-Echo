"use strict";

import { getAudioTestString } from "./HMI-utils.js";
import { retrieveTruthEventsInTimeRange, retrieveVocalizationEventsInTimeRange, retrieveMicrophones } from "./routes.js";
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
  "insect" : "insects",
  "reptile" : "reptiles"
};

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

  addAllTruthFeatures(hmiState);
  addAllVocalizationFeatures(hmiState);

  createMapClickEvent(hmiState);

  retrieveMicrophones().then((res) => {
    
    updateMicrophoneLayer(hmiState, res.data);
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
  console.log(results);

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
    layer.getSource().removeFeature(featureToPurge);
  }

  addNewTruthFeatures(hmiState, updatedMovementEvents);
  addNewTruthFeatures(hmiState, newMovementEvents);
}

export function updateVocalizationLayerFromLiveData(hmiState, results){

  let newVocalizationEvents = [];

  for (let data of results) {
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
  movementEvent.timestamp = hmiState.currentTime;
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

  console.log(data);

  vocalizationEvent.timestamp = hmiState.currentTime;
  vocalizationEvent.eventTimestamp = data.timestamp;
  vocalizationEvent.eventId = data._id;
  
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



function playAudioString(audioDataString) {
  // Convert the binary audio data string into a typed array
  const audioData = new Uint8Array(
    atob(audioDataString)
      .split("")
      .map((char) => char.charCodeAt(0))
  );

  // Create an audio context
  const audioContext = new AudioContext();

  // Create a new audio buffer
  const numChannels = 1;
  const sampleRate = 44100;
  const bufferLength = audioData.length / 2;
  const audioBuffer = audioContext.createBuffer(
    numChannels,
    bufferLength,
    sampleRate
  );

  // Copy the binary audio data into the audio buffer
  audioBuffer.copyToChannel(new Float32Array(audioData.buffer), 0);

  // Create a new audio buffer source node and set its buffer
  const sourceNode = audioContext.createBufferSource();
  sourceNode.buffer = audioBuffer;

  // Connect the source node to the destination node of the audio context
  sourceNode.connect(audioContext.destination);

  // Start playing the audio
  sourceNode.start();
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
    let entry = hmiState.movementEvents[key];
    //console.log("True location found!:  ")
    var trueLocation = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([entry.locationLon,entry.locationLat])
      ),
        name: 'trueLocation_' + entry.speciesScientificName,
        animalType: entry.animalType,
        animalStatus: entry.animalStatus,
        animalSpecies: entry.speciesScientificName 
    });
      //console.log(entry.locationLon, " ", entry.locationLat)
    
    var trueIcon = new ol.style.Style({
        image: new ol.style.Icon({
          src: './../images/sim/' + getIconName(entry.animalStatus, entry.animalType),
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
    var trueLocation = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([entry.locationLon, entry.locationLat])
      ),
        name: 'trueLocation_' + entry.speciesScientificName,
        animalType: entry.animalType,
        animalStatus: entry.animalStatus,
        animalSpecies: entry.speciesScientificName 
    });
      //console.log(entry.locationLon, " ", entry.locationLat)
    
    var trueIcon = new ol.style.Style({
        image: new ol.style.Icon({
          src: './../images/sim/' + getIconName(entry.animalStatus, entry.animalType),
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
    var evtLocation = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([entry.locationLon,entry.locationLat])
      ),
        name: 'vocalisation_' + entry.speciesScientificName,
        animalType: entry.animalType,
        animalStatus: entry.animalStatus,
        animalSpecies: entry.speciesScientificName 
    });
      //console.log(entry.locationLon, " ", entry.locationLat)
    
    var icon = new ol.style.Style({
        image: new ol.style.Icon({
          src: './../images/vocalization/' + getIconName(entry.animalStatus, entry.animalType),
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
    var evtLocation = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([entry.locationLon,entry.locationLat])
      ),
        name: 'vocalisation_' + entry.speciesScientificName,
        animalType: entry.animalType,
        animalStatus: entry.animalStatus,
        animalSpecies: entry.speciesScientificName 
    });
      //console.log(entry.locationLon, " ", entry.locationLat)
    
    var icon = new ol.style.Style({
        image: new ol.style.Icon({
          src: './../images/vocalization/' + getIconName(entry.animalStatus, entry.animalType),
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

function addmicrophones(hmiState) {
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
        src: "./../images/mic2.png",
        anchor: [0.5, 1],
        scale: 0.75,
      }),
    });
    mic.setStyle(icon);
    mics.push(mic);
  });

  //console.log("mics: ", mics);
  let layer = findMapLayerWithName(hmiState, "mic_layer");
  let layerSource = layer.getSource();
  layerSource.addFeatures(mics);
  layer.getSource().changed();
  layer.changed();
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
    if (feature){
      let values = feature.getProperties();
      if (values.animalSpecies){
          var result = sample_data.find(({ common }) => common.toUpperCase() === values.animalSpecies.toUpperCase())
          if (result) {
            animal_data = result;
            document.getElementById("desc_name").innerText = result.common;
            document.getElementById("desc_species").innerText = result.species;
            document.getElementById("desc_summary").innerText = result.summary;
            document.getElementById("desc_img").src = "../../images/bio/" + result.common + ".png";
            let summary = document.getElementById("desc_details");
            summary.innerHTML = '';
            result.description.forEach(content => {
              var p = document.createElement('p');
              p.className = "desc_ul";
              p.innerText = content;
              summary.appendChild(p);
            })
            const toggleEvent = new CustomEvent('animalToggled', { 
              detail: {
                message: 'toggled'
              }
            });
            document.dispatchEvent(toggleEvent);
          }

      }
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
  animal_toggled = false;
}

function updateTruthEvents(hmiState){
  retrieveTruthEventsInTimeRange(hmiState.previousUpdateTime, hmiState.currentTime).then((res) => {
    updateAnimalMovementLayerFromLiveData(hmiState, res.data);
    //TODO also update vocalisation layer here.
  })
}

//TODO Implement this
function updateVocalizationEvents(hmiState){
  retrieveVocalizationEventsInTimeRange(hmiState.previousUpdateTime, hmiState.currentTime).then((res) => {
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

let simUpdateTimeout = null;

function queueSimUpdate(hmiState) {
  if(hmiState.liveMode){
    hmiState.currentTime = Math.floor((Date.now() - hmiState.timeOffset - hmiState.simUpdateDelay) / 1000);
    hmiState.liveEventCutoff = Math.floor((Date.now() - hmiState.timeOffset - hmiState.simUpdateDelay - hmiState.liveWindow) / 1000);
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