document.getElementById("user-request-form").addEventListener("submit", (event) => {
    event.preventDefault(); // Prevent form submission

    // Gather form data
    const formData = {
        username: document.getElementById("username").value,
        email: document.getElementById("email").value,
        animal: document.getElementById("animal").value,
        request_type: document.getElementById("request_type").value,
        details: document.getElementById("details").value,
    };

    // Send the data to the backend
    fetch("http://localhost:5000/submit_request", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
    })
        .then((response) => response.json())
        .then((data) => {
            const messageDiv = document.getElementById("response-message");
            if (data.success) {
                messageDiv.textContent = data.message;
                messageDiv.style.color = "green";
            } else {
                messageDiv.textContent = `Error: ${data.error}`;
                messageDiv.style.color = "red";
            }
        })
        .catch((error) => {
            console.error("Error:", error);
            const messageDiv = document.getElementById("response-message");
            messageDiv.textContent = "Failed to submit the request. Please try again.";
            messageDiv.style.color = "red";
        });
});
