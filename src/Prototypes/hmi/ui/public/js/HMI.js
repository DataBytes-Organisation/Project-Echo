"use strict";

import { getAudioTestString } from "./HMI-utils.js";
import { retrieveTruthEventsInTimeRange } from "./routes.js";
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

var sample_data = data.data;

var animal_data = []

export var animal_toggled = false;

//For Demo purposes only
//This plays an awful quailty audio test string
document.addEventListener('click', function() {
  //playAudioString(getAudioTestString());  
});


export function initialiseHMI(hmiState) {
  console.log(`initialising`);
  createBasemap(hmiState);
  console.log("Get sample element from document: ", document.getElementById("menuPanel"))
  addTruthLayers(hmiState);
  addVocalisationLayers(hmiState);

  addAllTruthFeatures(hmiState);

  createMapClickEvent(hmiState);
  //addVocalizedFeatures(hmiState); !!! Remove and work on in next Update
  addmicrophones(hmiState);
  queueSimUpdate(hmiState);
  //simulateData(hmiState);
}

function updateFilters(){
  
}

export function updateAnimalMovementLayerFromPastData(hmiState, results){

  clearAllTruthLayers();

  hmiState.movementEvents = [];

  for (let data of results) {
    hmiState.movementEvents.push(convertJSONtoAnimalMovementEvent(hmiState, data));
  }

  addAllTruthFeatures(hmiState);
}

export function updateAnimalMovementLayerFromLiveData(hmiState, results){
  let newMovementEvents = [];

  for (let data of results) {
    let event = convertJSONtoAnimalMovementEvent(hmiState, data);
    hmiState.movementEvents.push(event);
    newMovementEvents.push(event);
  }

  addNewTruthFeatures(hmiState, newMovementEvents);
}

export function resetWildlifeLayers(hmiState){
  hmiState.vocalizationEvents = [];
  hmiState.movementEvents = [];
  clearAllVocalisationLayers();
  clearAllTruthLayers();
}

export function clearAllVocalisationLayers(){
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

export function convertJSONtoAnimalMovementEvent(hmiState, data){
    let movementEvent = {};

    movementEvent.animalId = data.animalId;
    movementEvent.eventId = "M_" + hmiState.truthEventId;

    hmiState.truthEventId = hmiState.truthEventId + 1;

    movementEvent.timestamp = hmiState.currentTime;
    movementEvent.speciesScientificName = data.species.toLowerCase();
    movementEvent.speciesIdentificationConfidence = 100.0;
    movementEvent.locationLat = data.animalTrueLLA[0];
    movementEvent.locationLon = data.animalTrueLLA[1];
    movementEvent.locationConfidence = 100.0;
    movementEvent.animalType = data.type.toLowerCase();
    movementEvent.animalStatus = matchStatus(data.status.toLowerCase());
    movementEvent.animalDiet = data.diet.toLowerCase();

    console.log("Movement Event:", movementEvent);

    return movementEvent;
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
  //console.log("Filter applied, updating vocalized layer visibility...");
  //console.log(filterState)
  /*vocalizedLayers.forEach((entry) => {
    if (filterState.includes(entry.values_.animalType) && 
        filterState.includes(entry.values_.animalStatus)) {
          //console.log("do something in here!");
          entry.getStyle().getImage().setOpacity(0.5);
          entry.changed()
    }
    else{
      entry.getStyle().getImage().setOpacity(0);
      entry.changed()
    }
  })*/

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

/* Add Vocalized Layers, work on in next branch
function addVocalizedLayers(hmiState) {
  console.log("addVocalizedLayers called.")
  console.log("voc locs", hmiState.vocalizationEvents);
  hmiState.vocalizationEvents.forEach((entry) => {
    console.log("Vocalization Found")
    var vocalization = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([entry.locationLon,entry.locationLat])
      ),
        name: 'vocalization',
        animalType: entry.animalType,
        animalStatus: entry.animalStatus
      });
      console.log(entry.locationLon, " ", entry.locationLat)
    
    var vocIcon = new ol.style.Style({
        image: new ol.style.Icon({
          src: `./../images/${entry.animalType+entry.animalStatus}.png`,
          opacity: 0.5,
          anchor: [0.5, 1],
          scale: 0.01,
          className: 'voc-icon'
        }),
    })
    vocalization.setStyle(vocIcon);
    vocalizedLayers.push(vocalization);
  })
  
  console.log("Vocalized Layers (vocals): ", vocalizedLayers);
  addVectorLayerTopDown(hmiState, "vocLayer");
  let layer = findMapLayerWithName(hmiState, "vocLayer");
  let layerSource = layer.getSource();
  layerSource.addFeatures(vocalizedLayers);

  //setting visibility
  //vocalizedLayers[0].getStyle().getImage().setOpacity(0.5);
  //vocalizedLayers[2].getStyle().getImage().setOpacity(0.5);
  //vocalizedLayers[1].on('click', () => {
 //   console.log("Element clicked");
  //});
}

//TESTING TO SEE IF CORRECTLY OUTPUT

function printTest(vocalizedLayers){

  vocalizedLayers.forEach((layer) => {
    console.log("Animal type: ",layer.values_.animalType)
    console.log("Animal status: ",layer.values_.animalStatus)
  })

}
*/

function addAllTruthFeatures(hmiState) {
  //console.log("addTruthLayers called.")
  //console.log("Truth locs", hmiState.movementEvents);
  for (let entry of hmiState.movementEvents) {
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
          src: `./../images/${entry.animalType+"-"+entry.animalStatus}.png`,
          anchor: [0.5, 1],
          scale: 0.75,
          className: 'true-icon'
        }),
    })
    trueLocation.setStyle(trueIcon);
    trueLocation.setId(entry.eventId);

    let layerName = deriveTruthLayerName(entry.animalStatus,entry.animalType);
    let layer = findMapLayerWithName(hmiState, layerName);
    let layerSource = layer.getSource();

    console.log(layerName);
    console.log("animal type: ", entry.speciesScientificName);

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
          src: `./../images/${entry.animalType+"-"+entry.animalStatus}.png`,
          anchor: [0.5, 1],
          scale: 0.75,
          className: 'true-icon'
        }),
    })
    trueLocation.setStyle(trueIcon);
    trueLocation.setId(entry.eventId);

    let layerName = deriveTruthLayerName(entry.animalStatus,entry.animalType);
    let layer = findMapLayerWithName(hmiState, layerName);
    let layerSource = layer.getSource();

    console.log(layerName);

    layerSource.addFeature(trueLocation);
    layer.getSource().changed();
    layer.changed();
  }
}

function addmicrophones(hmiState) {
  var mics = [];

  //console.log("locs", hmiState.microphoneLocations);
  hmiState.microphoneLocations.forEach((location) => {
    //console.log("Mic Found")
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
  addVectorLayerTopDown(hmiState, "mic_layer");
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

function purgeTruthEvents(hmiState){
  let persistEvents = [];
  //console.log("purging");

  //console.log(hmiState.liveEventCutoff);
  for(let event of hmiState.movementEvents){
    //console.log(event);

    if(hmiState.liveEventCutoff > event.timestamp){
      let layerName = deriveTruthLayerName(event.animalStatus, event.animalType);
      let layer = findMapLayerWithName(hmiState, layerName);
      const featureToPurge = layer.getSource().getFeatureById(event.eventId);
      //console.log(featureToPurge);
      layer.getSource().removeFeature(featureToPurge);
    }
    else{
      persistEvents.push(event);
    }
  }

  hmiState.movementEvents = persistEvents;
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
    updateTruthEvents(hmiState);
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