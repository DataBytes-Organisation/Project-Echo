document.addEventListener("DOMContentLoaded", () => {
  const tableBody = document.getElementById("iot-nodes-body");
  const typeFilter = document.getElementById("filter-type");
  const modelFilter = document.getElementById("filter-model");
  const searchInput = document.getElementById("filter-search");
  const detailsContainer = document.getElementById("node-details");
  const detailsContent = document.getElementById("node-details-content");

  let allNodes = [];
  let currentNodeId = null;

  async function fetchNodes() {
    try {
      const response = await fetch("/api/iot-nodes");
      const data = await response.json();
      allNodes = data;
      renderTable(allNodes);
    } catch (err) {
      console.error("Error fetching nodes:", err);
    }
  }

  function renderTable(nodes) {
    tableBody.innerHTML = "";
    nodes.forEach(node => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${node.id}</td>
        <td>${node.type || "—"}</td>
        <td>${node.model || "—"}</td>
        <td>${node.location}</td>
        <td>${node.status}</td>
      `;
      row.addEventListener("click", () => showDetails(node));
      tableBody.appendChild(row);
    });
  }

  function showDetails(node) {
    currentNodeId = node.id;

    detailsContent.innerHTML = `
      <form id="edit-node-form">
        <div class="mb-3">
          <label for="nodeName" class="form-label">Node Name</label>
          <input type="text" class="form-control" id="nodeName" value="${node.name || ''}" readonly>
        </div>

        <div class="mb-3">
          <label for="model" class="form-label">Model</label>
          <input type="text" class="form-control" id="model" value="${node.model || ''}" readonly>
        </div>

        <div class="mb-3">
          <label for="location" class="form-label">Location</label>
          <input type="text" class="form-control" id="location" value="${node.location || ''}" readonly>
        </div>

        <button type="button" id="editBtn" class="btn btn-primary">Edit</button>
        <button type="button" id="saveBtn" class="btn btn-success d-none">Save</button>
      </form>
    `;

    detailsContainer.classList.remove("d-none");

    document.getElementById("editBtn").addEventListener("click", () => {
      document.getElementById("nodeName").readOnly = false;
      document.getElementById("model").readOnly = false;
      document.getElementById("location").readOnly = false;

      document.getElementById("editBtn").classList.add("d-none");
      document.getElementById("saveBtn").classList.remove("d-none");
    });

    document.getElementById("saveBtn").addEventListener("click", async () => {
      const updatedNode = {
        name: document.getElementById("nodeName").value,
        model: document.getElementById("model").value,
        location: document.getElementById("location").value
      };

      try {
        const res = await fetch(`/api/nodes/${currentNodeId}`, {
          method: "PUT",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(updatedNode)
        });

        const result = await res.json();
        alert("Node updated successfully!");

        // Re-lock the fields
        document.getElementById("nodeName").readOnly = true;
        document.getElementById("model").readOnly = true;
        document.getElementById("location").readOnly = true;

        document.getElementById("editBtn").classList.remove("d-none");
        document.getElementById("saveBtn").classList.add("d-none");

        // Optionally refresh the full table
        fetchNodes();
      } catch (err) {
        console.error("Error updating node:", err);
        alert("Failed to update node.");
      }
    });
  }

  function applyFilters() {
    const type = typeFilter.value;
    const model = modelFilter.value;
    const search = searchInput.value.toLowerCase();

    const filtered = allNodes.filter(n =>
      (!type || n.type === type) &&
      (!model || n.model === model) &&
      (!search || n.id.toLowerCase().includes(search) || n.location.toLowerCase().includes(search))
    );

    renderTable(filtered);
  }

  typeFilter.addEventListener("change", applyFilters);
  modelFilter.addEventListener("change", applyFilters);
  searchInput.addEventListener("input", applyFilters);

  fetchNodes();
});
