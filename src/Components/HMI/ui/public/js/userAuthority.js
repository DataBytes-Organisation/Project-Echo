document.addEventListener("DOMContentLoaded", () => {
  const tableBody = document.querySelector("#userTable tbody");
  const status = document.getElementById("status");

  fetch("http://localhost:9000/hmi/users")
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }
      return response.json();
    })
    .then(users => {
      if (users.length === 0) {
        status.textContent = "No users found.";
        return;
      }

      users.forEach(user => {
        const row = document.createElement("tr");

        // username
        const usernameCell = document.createElement("td");
        usernameCell.textContent = user.username || "-";

       
        const roles = Array.isArray(user.role)
          ? user.role.map(r => r.name).filter(Boolean)
          : [];
        const roleCell = document.createElement("td");
        roleCell.textContent = roles.length > 0 ? roles.join(", ") : "-";

        // australia time
        const loginCell = document.createElement("td");
        loginCell.textContent = user.last_login
          ? new Date(user.last_login).toLocaleString("en-AU", {
              timeZone: "Australia/Melbourne"
            })
          : "N/A";

        row.appendChild(usernameCell);
        row.appendChild(roleCell);
        row.appendChild(loginCell);
        tableBody.appendChild(row);
      });
    })
    .catch(err => {
      console.error("Error fetching users:", err);
      status.textContent = "Failed to load user data.";
    });
});
