// public/js/detectedAnimals.js

// 初期化
document.addEventListener("DOMContentLoaded", () => {
    fetchAnimals();
  });
  
  let currentAnimal = null;
  
  function fetchAnimals() {
    fetch('/io/combined_animal_data.json')
      .then((res) => res.json())
      .then((data) => {
        const tbody = document.querySelector("#animals-table tbody");
        data.forEach((animal) => {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td>${animal.name}</td>
            <td>${animal.detected_time}</td>
            <td>${animal.microphone_id}</td>
          `;
          tr.onclick = () => showPopup(animal);
          tbody.appendChild(tr);
        });
      })
      .catch((error) => {
        console.error("Failed to fetch animal data:", error);
      });
  }
  
  function showPopup(animal) {
    currentAnimal = animal;
    document.getElementById("animal-name").innerText = `${animal.name} Details`;
    document.getElementById("animal-time").innerText = `Detected at: ${animal.detected_time}`;
    document.getElementById("animal-mic").innerText = `Microphone ID: ${animal.microphone_id}`;
    document.getElementById("animal-popup").style.display = "block";
  
    document.getElementById("show-map").onclick = () => {
      const url = `/map.html?animalId=${animal.id}&micId=${animal.microphone_id}`;
      window.location.href = url;
    };
  }
  
  function closePopup() {
    document.getElementById("animal-popup").style.display = "none";
  }
  