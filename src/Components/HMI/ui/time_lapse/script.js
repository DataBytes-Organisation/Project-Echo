document.getElementById("fetch-timelapse").addEventListener("click", () => {
    const start = document.getElementById("start").value;
    const stop = document.getElementById("stop").value;

    if (!start || !stop) {
        alert("Please select both start and stop times.");
        return;
    }

    fetch(`http://localhost:5000/time_lapse_api?start=${start}&stop=${stop}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const map = L.map('map').setView([0, 0], 2);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

                let index = 0;
                function animateEvents(events) {
                    if (index < events.length) {
                        const event = events[index];
                        if (event.latitude && event.longitude) {
                            L.marker([event.latitude, event.longitude])
                                .addTo(map)
                                .bindPopup(`<b>${event.Animal || "Unknown"}</b><br>${event.timestamp}`);
                        }
                        index++;
                        setTimeout(() => animateEvents(events), 500); // Delay for animation
                    }
                }

                animateEvents(data.events);
            } else {
                console.error("Error fetching events:", data.error);
                alert("Error fetching events. Please try again.");
            }
        })
        .catch(error => {
            console.error("Error:", error);
            alert("Failed to fetch events. Check your connection.");
        });
});
