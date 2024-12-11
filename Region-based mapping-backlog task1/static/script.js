document.getElementById("loadDataButton").addEventListener("click", function () {
    fetch("/region-data")
        .then((response) => response.json())
        .then((data) => {
            const tableBody = document.querySelector("#regionTable tbody");
            tableBody.innerHTML = ""; // Clear previous data

            Object.keys(data).forEach((region) => {
                const row = document.createElement("tr");

                const regionCell = document.createElement("td");
                regionCell.textContent = region;
                row.appendChild(regionCell);

                const sightingsCell = document.createElement("td");
                sightingsCell.textContent = data[region].sightings;
                row.appendChild(sightingsCell);

                const vegetationCell = document.createElement("td");
                vegetationCell.textContent = data[region].vegetation_density;
                row.appendChild(vegetationCell);

                const movementCell = document.createElement("td");
                movementCell.textContent = data[region].movement_patterns;
                row.appendChild(movementCell);

                tableBody.appendChild(row);
            });
        })
        .catch((error) => {
            console.error("Error fetching region data:", error);
        });
});
