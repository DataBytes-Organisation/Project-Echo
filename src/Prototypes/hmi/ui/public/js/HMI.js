"use strict";

import { getAudioTestString } from "./HMI-utils.js";

var markups = ["elephant.png", "monkey.png", "tiger.png"];

var statuses = [
  "endangered",
  "vulnerable",
  "near-threatened",
  "normal",
  "invasive",
];

var animalTypes = ["mammal", "bird", "amphibian", "reptile", "insect"];

//For Demo purposes only
//This plays an awful quailty audio test string
document.addEventListener('click', function() {
  //playAudioString(getAudioTestString());  
});

export function initialiseHMI(hmiState) {
  console.log(`initialising`);

  createBasemap(hmiState);
  addWildlifeLayers(hmiState);

  addTruthLayers(hmiState);
  addDummyMarkers(hmiState);
  addmicrophones(hmiState);

  //simulateData(hmiState);
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

function addmicrophones(hmiState) {
  var mics = [];
  console.log("locs", hmiState.microphoneLocations);

  hmiState.microphoneLocations.forEach((location) => {
    // Add the marker into the array
    var mic = new ol.Feature({
      geometry: new ol.geom.Point(
        ol.proj.fromLonLat([location.lon, location.lat])
      ),
      name: "mic",
    });

    var icon = new ol.style.Style({
      image: new ol.style.Icon({
        src: "./../images/microphone.png",
        anchor: [0.5, 1],
        scale: 1,
      }),
    });

    mic.setStyle(icon);
    mics.push(mic);
  });

  console.log("mics: ", mics);

  addVectorLayerTopDown(hmiState, "micLayer");
  let layer = findMapLayerWithName(hmiState, "micLayer");
  let layerSource = layer.getSource();
  layerSource.addFeatures(mics);
}

function addDummyMarkers(hmiState) {
  //***Add some sample markers (WIP)***
  var markers = [];

  for (var i = 0; i < 10; i++) {
    // Compute a random icon and lon/lat position.
    var lon = hmiState.originLon + Math.random() * 0.1;
    var lat = hmiState.originLat + Math.random() * 0.1;

    // Add the marker into the array
    var mark = new ol.Feature({
      geometry: new ol.geom.Point(ol.proj.fromLonLat([lon, lat])),
      name: "marker" + i,
    });
    var icon = new ol.style.Style({
      image: new ol.style.Icon({
        src: "./../images/" + markups[i % 3],
        anchor: [0.5, 1],
        scale: 0.05,
      }),
    });
    mark.setStyle(icon);
    markers.push(mark);
  }

  console.log("markers: ", markers);

  addVectorLayerTopDown(hmiState, "markerLayer");
  let layer = findMapLayerWithName(hmiState, "markerLayer");
  let layerSource = layer.getSource();
  layerSource.addFeatures(markers);
}

function findMapLayerWithName(hmiState, name) {
  if (!hmiState.basemap) {
    console.log(`findMapLayerWithName: invalid basemap`);
    return null;
  } else {
    let mapLayers = hmiState.basemap.getLayers();

    for (let i = 0; i < mapLayers.getLength(); ++i) {
      let currentLayer = mapLayers.item(i);
      if (currentLayer.get("name") == name) {
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
    });

    if (zIndex != 0) {
      layer.setZIndex(zIndex);
    }

    hmiState.basemap.addLayer(layer);
  }
}

function addWildlifeLayers(hmiState) {
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
          maxZoom: 19,
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

function simulateData(hmiState) {}

let simUpdateTimeout = null;

function queueSimUpdate(hmiState) {
  if (simUpdateTimeout) {
    clearTimeout(simUpdateTimeout);
  }

  simUpdateTimeout = setTimeout(
    simulateData,
    hmiState.requestInterval,
    hmiState
  );
}