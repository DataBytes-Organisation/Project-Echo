document.getElementById("movementForm").addEventListener("submit", function(event) {
    event.preventDefault();

    // Gather form data
    const x = parseFloat(document.getElementById("x").value);
    const y = parseFloat(document.getElementById("y").value);
    const movement_factor = parseFloat(document.getElementById("movement_factor").value);
    const time = parseFloat(document.getElementById("time").value);

    // Send data to backend via POST
    fetch("/projected-movement", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ x, y, movement_factor, time }),
    })
    .then(response => response.json())
    .then(data => {
        // Display the result
        const output = `New X: ${data.new_x}, New Y: ${data.new_y}`;
        document.getElementById("output").textContent = output;
    })
    .catch(error => console.error("Error:", error));
});
