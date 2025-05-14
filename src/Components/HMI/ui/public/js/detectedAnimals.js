// public/js/detectedAnimals.js

document.addEventListener("DOMContentLoaded", () => {
  fetchAnimals();
});

let currentAnimal = null;

function fetchAnimals() {
  fetch('/io/combined_animal_data.json')
    .then((res) => res.json())
    .then((data) => {
      const tbody = document.querySelector("#animals-table tbody");

      data.forEach((animal, index) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${animal["Common Name"] || animal["Animal"] || "Unknown"}</td>
          <td>${animal["timestamp"] || "N/A"}</td>
          <td>${animal["microphone_id"] || "N/A"}</td>
        `;
        tr.onclick = () => showPopup(animal, index);
        tbody.appendChild(tr);
      });
    })
    .catch((error) => {
      console.error("Failed to fetch animal data:", error);
    });
}


function showPopup(animal, index) {
  currentAnimal = animal;
  document.getElementById("animal-name").innerText = `${animal["Common Name"] || animal["Animal"] || "Animal"} Details`;
  document.getElementById("animal-time").innerText = `Detected at: ${animal["timestamp"] || "N/A"}`;
  document.getElementById("animal-mic").innerText = `Microphone ID: ${animal["microphone_id"] || "N/A"}`;
  document.getElementById("animal-popup").style.display = "block";

  document.getElementById("show-map").onclick = () => {
    const url = `/map.html?animalIndex=${index}`;
    window.location.href = url;
  };
}

function closePopup() {
  document.getElementById("animal-popup").style.display = "none";
}
