// Initialize the map
const map = new ol.Map({
  target: 'basemap',
  layers: [
    new ol.layer.Tile({
      source: new ol.source.OSM(), // Use OpenStreetMap for simplicity
    }),
  ],
  view: new ol.View({
    center: ol.proj.fromLonLat([133.7751, -25.2744]), // Australia coordinates
    zoom: 4, // Initial zoom level
  }),
});

// Create vector sources and layers for animal detections and survey sites
const animalSource = new ol.source.Vector();
const animalLayer = new ol.layer.Vector({
  source: animalSource,
});

const surveySource = new ol.source.Vector();
const surveyLayer = new ol.layer.Vector({
  source: surveySource,
});

// Add layers to the map
map.addLayer(animalLayer);
map.addLayer(surveyLayer);

// Function to add a marker for animal detection
function addAnimalMarker(lon, lat) {
  const marker = new ol.Feature({
    geometry: new ol.geom.Point(ol.proj.fromLonLat([lon, lat])),
  });

  // Style the marker (smaller red dot with black border)
  const markerStyle = new ol.style.Style({
    image: new ol.style.Circle({
      radius: 5, // Smaller radius for red dot
      fill: new ol.style.Fill({ color: 'red' }),
      stroke: new ol.style.Stroke({ color: 'black', width: 1 }),
    }),
  });

  marker.setStyle(markerStyle);
  animalSource.addFeature(marker);
}

// Function to add a marker for survey site
function addSurveyMarker(lon, lat) {
  const marker = new ol.Feature({
    geometry: new ol.geom.Point(ol.proj.fromLonLat([lon, lat])),
  });

  // Style the marker (smaller grey dot with black border)
  const markerStyle = new ol.style.Style({
    image: new ol.style.Circle({
      radius: 4, // Smaller radius for grey dot
      fill: new ol.style.Fill({ color: 'grey' }),
      stroke: new ol.style.Stroke({ color: 'black', width: 1 }),
    }),
  });

  marker.setStyle(markerStyle);
  surveySource.addFeature(marker);
}

// Sample data with more animal detection locations (red dots) and survey sites (grey dots)
const animalLocations = {
  swamp_antechinus: [[133.7751, -25.2744], [144.9631, -37.8136], [139.69, -20.01], [145.63, -29.76]],
  long_nosed_potoroo: [[151.2093, -33.8688], [145.77, -16.92], [148.95, -35.03], [149.13, -36.76]],
  broad_toothed_rat: [[150.891, -34.93], [146.85, -34.98], [147.28, -35.56]],
  southern_brown_bandicoot: [[146.45, -36.15], [145.12, -37.04], [144.78, -38.55]],
  new_holland_mouse: [[153.02, -27.47], [151.0, -26.78], [152.0, -28.30]],
  smoky_mouse: [[148.25, -37.65], [147.70, -37.50], [148.50, -36.90]],
};

const surveySites = [
  [134.0, -25.0], [145.0, -35.0], [151.0, -30.0], [150.5, -33.0], [153.5, -28.0],
  [135.5, -24.5], [143.5, -31.0], [140.0, -29.5], [144.5, -36.0], [142.8, -34.0]
];

// Function to filter and show selected animal markers
function filterAnimal(animalType) {
  // Clear existing animal markers
  animalSource.clear();

  // Get locations for the selected animal
  const locations = animalLocations[animalType] || [];

  // Add a marker for each location
  locations.forEach((loc) => {
    addAnimalMarker(loc[0], loc[1]);
  });
}

// Add all survey sites to the map (grey dots)
surveySites.forEach((loc) => {
  addSurveyMarker(loc[0], loc[1]);
});

// Reset map and clear all markers
function resetMap() {
  animalSource.clear();
  // Optionally, you can re-add survey sites if needed
}

// Function to simulate time-lapse update (placeholder for now)
function updateTimeLapse(value) {
  console.log(`Time-lapse updated: ${value}`);
}
