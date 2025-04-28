// Initialize the map centered on the average node location
let map;
let markers = [];

// Custom icons for different node types
const masterIcon = L.icon({
    iconUrl: '/images/nodes/master-node.png',
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32]
});

const childIcon = L.icon({
    iconUrl: '/images/nodes/child-node.jpg',
    iconSize: [24, 24],
    iconAnchor: [12, 24],
    popupAnchor: [0, -24]
});

async function initMap() {
    // Create map centered on approximate location
    map = L.map('map').setView([-38.7789, 143.5705], 14);
    
    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    // Fetch and display nodes
    await loadNodes();
}

async function loadNodes() {
    try {
        const response = await fetch('http://ts-api-cont:9000/nodes');
        const nodes = await response.json();
        
        // Clear existing markers
        markers.forEach(marker => marker.remove());
        markers = [];

        // Add markers for each node
        nodes.forEach(node => {
            const icon = node.type === 'master' ? masterIcon : childIcon;
            const marker = L.marker([node.location.latitude, node.location.longitude], {
                icon: icon,
                title: node.name
            });

            // Create popup content
            const popupContent = `
                <div class="node-popup">
                    <h3>${node.name}</h3>
                    <p>Type: ${node.type}</p>
                    <p>Model: ${node.model}</p>
                    ${node.parentNode ? `<p>Parent: ${node.parentNode}</p>` : ''}
                    <p>Components: ${node.components.length}</p>
                </div>
            `;

                marker.bindPopup(popupContent);
            marker.addTo(map);
            markers.push(marker);

            // If nodes are connected, draw a line between them
            if (node.connectedNodes && node.connectedNodes.length > 0) {
                node.connectedNodes.forEach(connectedId => {
                    const connectedNode = nodes.find(n => n._id === connectedId);
                    if (connectedNode) {
                        const line = L.polyline([
                            [node.location.latitude, node.location.longitude],
                            [connectedNode.location.latitude, connectedNode.location.longitude]
                        ], {
                            color: '#3388ff',
                            weight: 2,
                            opacity: 0.6,
                            dashArray: '5, 10'
                        }).addTo(map);
                        markers.push(line);
                    }
                });
            }
        });
    } catch (error) {
        console.error('Error loading nodes:', error);
    }
}

// Initialize map when DOM is loaded
document.addEventListener('DOMContentLoaded', initMap);
