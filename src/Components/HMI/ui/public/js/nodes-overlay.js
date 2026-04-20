
let axios;

if (typeof window === 'undefined') {
  axios = require('axios');
} else {
  axios = window.axios;
}

let detectionSimulationInterval = null;
let currentHighlightedFeature = null;
let currentHighlightResetTimeout = null;
let nodeFeaturesCache = [];

function showNotification(message) {
  const notification = document.getElementById('notification');
  const text = document.getElementById('notification-text');
  const closeBtn = document.getElementById('notification-close');

  if (!notification || !text) {
    console.error('Notification elements not found');
    return;
  }

  text.innerText = message;
  notification.style.display = 'block';

  const hide = () => {
    notification.style.display = 'none';
  };

  if (closeBtn) {
    closeBtn.onclick = hide;
  }

  setTimeout(hide, 3000);
}

window.showNotification = showNotification;

function getNodeStyles(feature, highlighted = false) {
  const type = feature.get('type');
  const iconSrc =
    type === 'master'
      ? 'images/nodes/master-node.svg'
      : 'images/nodes/microchip-solid.svg';

  const baseColor = type === 'master' ? '#ff4444' : '#4CAF50';

  if (highlighted) {
    return [
      new ol.style.Style({
        image: new ol.style.Circle({
          radius: 28,
          fill: new ol.style.Fill({ color: '#FFD700' }),
          stroke: new ol.style.Stroke({
            color: '#fff',
            width: 3
          })
        })
      }),
      new ol.style.Style({
        image: new ol.style.Icon({
          src: iconSrc,
          scale: 0.05,
          anchor: [0.5, 0.5],
          anchorXUnits: 'fraction',
          anchorYUnits: 'fraction'
        })
      })
    ];
  }

  return [
    new ol.style.Style({
      image: new ol.style.Circle({
        radius: 23,
        fill: new ol.style.Fill({ color: baseColor }),
        stroke: new ol.style.Stroke({
          color: '#fff',
          width: 2
        })
      })
    }),
    new ol.style.Style({
      image: new ol.style.Icon({
        src: iconSrc,
        scale: 0.05,
        anchor: [0.5, 0.5],
        anchorXUnits: 'fraction',
        anchorYUnits: 'fraction'
      })
    })
  ];
}

function resetCurrentHighlightedNode() {
  if (currentHighlightResetTimeout) {
    clearTimeout(currentHighlightResetTimeout);
    currentHighlightResetTimeout = null;
  }

  if (currentHighlightedFeature) {
    currentHighlightedFeature.setStyle(getNodeStyles(currentHighlightedFeature, false));
    currentHighlightedFeature = null;
  }
}

function highlightNode(feature) {
  if (!feature) return;

  resetCurrentHighlightedNode();

  feature.setStyle(getNodeStyles(feature, true));
  currentHighlightedFeature = feature;

  currentHighlightResetTimeout = setTimeout(() => {
    if (currentHighlightedFeature) {
      currentHighlightedFeature.setStyle(getNodeStyles(currentHighlightedFeature, false));
      currentHighlightedFeature = null;
    }
  }, 2000);
}

function triggerSimulatedDetection(feature) {
  if (!feature) return;

  const nodeName = feature.get('name') || 'Unknown Node';
  highlightNode(feature);
  showNotification(`${nodeName} detected`);
  console.log('Simulated detection triggered for:', nodeName);
}

function startDetectionSimulation(intervalMs = 3000) {
  stopDetectionSimulation();

  if (!nodeFeaturesCache || nodeFeaturesCache.length === 0) {
    console.warn('No node features available for detection simulation.');
    return;
  }

  detectionSimulationInterval = setInterval(() => {
    const randomIndex = Math.floor(Math.random() * nodeFeaturesCache.length);
    const randomNodeFeature = nodeFeaturesCache[randomIndex];
    triggerSimulatedDetection(randomNodeFeature);
  }, intervalMs);

  console.log(`Detection simulation started with interval ${intervalMs}ms`);
}

function stopDetectionSimulation() {
  if (detectionSimulationInterval) {
    clearInterval(detectionSimulationInterval);
    detectionSimulationInterval = null;
  }

  resetCurrentHighlightedNode();
  console.log('Detection simulation stopped');
}

function getFallbackNodes() {
  return [
    {
      _id: 'node-1',
      name: 'Node A',
      type: 'master',
      model: 'Echo Master',
      location: { longitude: 143.509, latitude: -38.673 }
    },
    {
      _id: 'node-2',
      name: 'Node B',
      type: 'sensor',
      model: 'Echo Sensor',
      location: { longitude: 143.515, latitude: -38.678 }
    },
    {
      _id: 'node-3',
      name: 'Node C',
      type: 'sensor',
      model: 'Echo Sensor',
      location: { longitude: 143.520, latitude: -38.684 }
    }
  ];
}

async function addIoTNodesToMap(hmiState) {
  if (!hmiState.basemap) {
    console.error('Basemap not initialized');
    return;
  }

  try {
    let nodes = [];

    try {
      const response = await axios.get('http://127.0.0.1:8000/iot/nodes');
      nodes = response.data;
      console.log('Loaded nodes from backend API');
    } catch (apiError) {
      console.warn('Backend /iot/nodes not available. Using fallback nodes instead.');
      nodes = getFallbackNodes();
    }

    const iotLayer = new ol.layer.Vector({
      source: new ol.source.Vector(),
      style: function (feature) {
        if (feature.get('isNode')) {
          return getNodeStyles(feature, false);
        }
        return feature.getStyle();
      }
    });

    nodeFeaturesCache = [];

    nodes.forEach((node) => {
      const feature = new ol.Feature({
        geometry: new ol.geom.Point(
          ol.proj.fromLonLat([
            node.location.longitude,
            node.location.latitude
          ])
        ),
        lat: node.location.latitude,
        lon: node.location.longitude,
        type: node.type,
        name: node.name,
        model: node.model,
        isNode: true
      });

      feature.setId(node._id);
      feature.setStyle(getNodeStyles(feature, false));
      iotLayer.getSource().addFeature(feature);
      nodeFeaturesCache.push(feature);
    });

    hmiState.basemap.addLayer(iotLayer);

    const popupElement = document.createElement('div');
    popupElement.className = 'node-popup';

    const popup = new ol.Overlay({
      element: popupElement,
      positioning: 'bottom-center',
      stopEvent: false
    });

    hmiState.basemap.addOverlay(popup);

    hmiState.basemap.on('pointermove', function (evt) {
      const feature = hmiState.basemap.forEachFeatureAtPixel(
        evt.pixel,
        function (feature) {
          return feature;
        }
      );

      const element = popup.getElement();

      if (
        feature &&
        feature.get('isNode') &&
        feature.get('name') &&
        feature.get('type') &&
        feature.get('model')
      ) {
        const coordinates = feature.getGeometry().getCoordinates();
        popup.setPosition(coordinates);
        element.innerHTML = `
          <div class="node-popup">
            <strong>${feature.get('name')}</strong><br>
            Type: ${feature.get('type')}<br>
            Model: ${feature.get('model')}
          </div>
        `;
        element.style.display = 'block';
      } else {
        element.style.display = 'none';
      }
    });

    hmiState.basemap.on('singleclick', function (evt) {
      const feature = hmiState.basemap.forEachFeatureAtPixel(
        evt.pixel,
        function (feature) {
          return feature;
        }
      );

      if (feature && feature.get('isNode')) {
        triggerSimulatedDetection(feature);
      }
    });

    startDetectionSimulation(3000);

  } catch (error) {
    console.error('Error loading IoT nodes:', error);
  }
}
window.startDetectionSimulation = startDetectionSimulation;
window.stopDetectionSimulation = stopDetectionSimulation;

export { addIoTNodesToMap, startDetectionSimulation, stopDetectionSimulation };