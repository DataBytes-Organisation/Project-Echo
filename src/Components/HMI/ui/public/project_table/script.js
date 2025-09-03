const projects = [
  { id: 1, name: "Wildlife Tracker", status: "Active", description: "Tracks wildlife movement in real-time." },
  { id: 2, name: "Habitat Restoration", status: "Planning", description: "Restores natural habitat using AI models." },
  { id: 3, name: "Marine Monitor", status: "Completed", description: "Monitors marine ecosystems using sonar data." }
];

const projectBody = document.getElementById("projectBody");
const modal = document.getElementById("projectModal");
const modalTitle = document.getElementById("modalTitle");
const modalDetails = document.getElementById("modalDetails");

projects.forEach(project => {
  const row = document.createElement("tr");
  row.innerHTML = `
    <td>${project.id}</td>
    <td>${project.name}</td>
    <td>${project.status}</td>
  `;
  row.onclick = () => showDetails(project);
  projectBody.appendChild(row);
});

function showDetails(project) {
  modalTitle.textContent = project.name;
  modalDetails.textContent = project.description;
  modal.style.display = "flex";
}

function closeModal() {
  modal.style.display = "none";
}
