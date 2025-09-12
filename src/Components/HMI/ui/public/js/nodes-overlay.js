let axios;

if (typeof window === 'undefined') {
  axios = require('axios');
} else {
  axios = window.axios;
}

// Function to add IoT nodes to the existing map
async function addIoTNodesToMap(hmiState) {
    if (!hmiState.basemap) {
        console.error('Basemap not initialized');
        return;
    }
    try {
        // Fetch nodes from the API
        const response = await axios.get('/iot/nodes');
        const nodes = response.data;
        
        // Create a new vector layer for IoT nodes
        const iotLayer = new ol.layer.Vector({
            source: new ol.source.Vector(),
            style: function(feature) {
                const type = feature.get('type');
                const iconSrc = type === 'master' ? 'images/nodes/master-node.svg' : 'images/nodes/microchip-solid.svg';
                const circleColor = type === 'master' ? '#ff4444' : '#4CAF50';

                return [
                    // Background circle
                    new ol.style.Style({
                        image: new ol.style.Circle({
                            radius: 23,
                            fill: new ol.style.Fill({ color: circleColor }),
                        })
                    }),
                    // Icon on top
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
        });

        // Add features for each node
        nodes.forEach(node => {
            const feature = new ol.Feature({
                geometry: new ol.geom.Point(ol.proj.fromLonLat([
                    node.location.longitude,
                    node.location.latitude
                ])),
                lat: node.location.latitude,
                lon: node.location.longitude,
                type: node.type,
                name: node.name,
                model: node.model,
                isNode: true
            });

            feature.setId(node._id);
            iotLayer.getSource().addFeature(feature);

            // If node has connections, draw lines
            if (node.connectedNodes && node.connectedNodes.length > 0) {
                node.connectedNodes.forEach(connectedId => {
                    const connectedNode = nodes.find(n => n._id === connectedId);
                    if (connectedNode) {
                        const lineFeature = new ol.Feature({
                            geometry: new ol.geom.LineString([
                                ol.proj.fromLonLat([node.location.longitude, node.location.latitude]),
                                ol.proj.fromLonLat([connectedNode.location.longitude, connectedNode.location.latitude])
                            ])
                        });

                        const lineStyle = new ol.style.Style({
                            stroke: new ol.style.Stroke({
                                color: '#3388ff',
                                width: 2,
                                lineDash: [5, 10]
                            })
                        });

                        lineFeature.setStyle(lineStyle);
                        iotLayer.getSource().addFeature(lineFeature);
                    }
                });
            }
        });

        // Add the layer to the map
        hmiState.basemap.addLayer(iotLayer);

        // Add popup for node information
        const popup = new ol.Overlay({
            element: document.createElement('div'),
            positioning: 'bottom-center',
            stopEvent: false
        });
        hmiState.basemap.addOverlay(popup);

        // Show node info on hover
        hmiState.basemap.on('pointermove', function(evt) {
            const feature = hmiState.basemap.forEachFeatureAtPixel(evt.pixel, function(feature) {
                return feature;
            });

            const element = popup.getElement();
            if (feature && feature.get('name') && feature.get('type') && feature.get('model')) {
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

    } catch (error) {
        console.error('Error loading IoT nodes:', error);
    }
}

// Export the function to be used in HMI.js
export { addIoTNodesToMap };
