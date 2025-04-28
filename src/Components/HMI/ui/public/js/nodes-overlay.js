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
                const iconHtml = type === 'master' ? '<i class="fas fa-broadcast-tower"></i>' : '<i class="fas fa-microphone"></i>';
                const color = type === 'master' ? '#ff4444' : '#4CAF50';
                
                return new ol.style.Style({
                    image: new ol.style.Icon({
                        src: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(
                            `<svg width="32" height="32" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
                                <rect width="32" height="32" fill="${color}" rx="16"/>
                                <text x="16" y="16" fill="white" font-family="FontAwesome" font-size="16" text-anchor="middle" dominant-baseline="central">
                                    ${type === 'master' ? '\uf519' : '\uf130'}
                                </text>
                            </svg>`
                        )}`
                    }),
                });
            }
        });

        // Add features for each node
        nodes.forEach(node => {
            const feature = new ol.Feature({
                geometry: new ol.geom.Point(ol.proj.fromLonLat([
                    node.location.longitude,
                    node.location.latitude
                ])),
                type: node.type,
                name: node.name,
                model: node.model
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
