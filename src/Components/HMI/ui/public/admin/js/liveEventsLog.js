// public/admin/js/liveEventsLog.js

window.addEventListener("DOMContentLoaded", () => {
  fetchLiveEvents();
  setInterval(fetchLiveEvents, 10000); 
});

function fetchLiveEvents() {
  const now = Math.floor(Date.now() / 1000);  
  const fiveMinsAgo = now - 300;  

  fetch(`http://localhost:9000/hmi/events_time?start=${fiveMinsAgo}&end=${now}`)
    .then((res) => res.json())
    .then((data) => {
      const container = document.getElementById("live-events-log");
      container.innerHTML = "";

      if (!Array.isArray(data) || data.length === 0) {
        container.innerHTML = "<p class='text-muted'>No recent events.</p>";
        return;
      }

      data.forEach((event) => {
        const timeStr = new Date(event.timestamp).toLocaleString();

        const entry = document.createElement("div");
        entry.className = "event-entry border rounded p-2 mb-2 bg-light text-dark";
        entry.innerHTML = `
          <strong>${timeStr}</strong> | 
          <span class="badge bg-info">${event.eventType || "Unknown Event"}</span> | 
          <span class="text-primary">${event.species || "Unknown Species"}</span>
        `;

        entry.onclick = () => {
          if (event.microphone_id) {
            window.open(`/map.html?highlightMic=${event.microphone_id}`, "_blank");
          } else {
            alert("Microphone ID not found for this event.");
          }
        };

        container.appendChild(entry);
      });
    })
    .catch((err) => {
      console.error("Failed to load live events:", err);
      const container = document.getElementById("live-events-log");
      container.innerHTML = "<p class='text-danger'>Error loading events.</p>";
    });
}
