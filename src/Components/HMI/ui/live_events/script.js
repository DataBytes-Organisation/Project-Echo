document.getElementById("fetch-event").addEventListener("click", () => {
    // Fetch the event data only when the button is clicked
    fetch('http://localhost:5000/live_events_api')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const display = document.getElementById("event-display");
            if (data.event === "No events available") {
                display.innerHTML = `<p>No events available. Try again later!</p>`;
            } else {
                let eventDetails = `<p><strong>Animal:</strong> ${data.Animal || "Unknown"}</p>`;
                Object.keys(data).forEach(key => {
                    if (key !== 'Animal') {
                        eventDetails += `<p><strong>${key}:</strong> ${data[key]}</p>`;
                    }
                });
                display.innerHTML = eventDetails;
            }
        })
        .catch(error => {
            console.error('Error fetching the event:', error); // Log error
            document.getElementById("event-display").innerHTML = `<p>Error fetching event. Please try again!</p>`;
        });
});
