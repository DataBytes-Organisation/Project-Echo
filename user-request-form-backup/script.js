document.getElementById("requestForm").addEventListener("submit", function(event){
    event.preventDefault();
  
    const form = event.target;
    const formData = new FormData(form);
    const statusMsg = document.getElementById("statusMsg");
  
    fetch("submit_request.php", {
      method: "POST",
      body: formData,
    })
    .then(response => response.text())
    .then(data => {
      if (data.includes("success")) {
        statusMsg.innerHTML = "✅ Request submitted successfully.";
        statusMsg.style.color = "green";
      } else {
        statusMsg.innerHTML = "❌ Error submitting request: " + data;
        statusMsg.style.color = "red";
      }
    })
    .catch(error => {
      statusMsg.innerHTML = "❌ Submission failed. " + error;
      statusMsg.style.color = "red";
    });
  });
  