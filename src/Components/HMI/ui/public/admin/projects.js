$(document).ready(function () {
  // ================================================================
  // Layout Includes
  // ================================================================
  $("#sidebar").load("/admin/component/sidebar-component.html");
  $("#header").load("/admin/component/header-component.html");
  $("#footer").load("/admin/component/footer-component.html");

  // ================================================================
  // DATA SETUP
  // ================================================================
  // TODO (backend): Replace this with GET /api/projects and GET /api/ecologists
  const ecologists = [
    { id: 1, name: "Bob James" },
    { id: 2, name: "Joe blow" },
    { id: 3, name: "Drew Giessler" },
    { id: 4, name: "Fang" },
    { id: 5, name: "Lexi" }
  ];

  // TODO (backend): Remove defaultProjects once real API is connected
  const defaultProjects = [
    {
      id: 1,
      name: "Birds",
      description: "monitoring Birds",
      location: "Bird Place",
      status: "active",
      sensors: 12,
      ecologistIds: [1, 2]
    },
    {
      id: 2,
      name: "Bears",
      description: "A study of Bears",
      location: "melbourne",
      status: "active",
      sensors: 8,
      ecologistIds: [2, 3, 4]
    },
    {
      id: 3,
      name: "Fox",
      description: "Fox study",
      location: "gipsland",
      status: "planning",
      sensors: 5,
      ecologistIds: []
    }
  ];

  // Use in-memory data for now
  let projects = [...defaultProjects];
  let nextProjectId = projects.length + 1;

  // ================================================================
  // HELPERS
  // ================================================================

  function getEcologistNameById(id) {
    const match = ecologists.find(e => e.id === id);
    return match ? match.name : "Unknown";
  }

  function renderEcologistSelect() {
    const select = $("#projectEcologists");
    select.empty();

    ecologists.forEach(e => {
      select.append(`<option value="${e.id}">${e.name}</option>`);
    });

    $("#totalEcologistsCount").text(ecologists.length);
  }

  function renderStats() {
    const activeCount = projects.filter(p => p.status === "active").length;
    $("#activeProjectsCount").text(activeCount);

    const totalSensors = projects.reduce(
      (sum, p) => sum + (Number(p.sensors) || 0),
      0
    );
    $("#sensorsDeployedCount").text(totalSensors);
  }

  function renderProjects() {
    const container = $("#projectsList");
    container.empty();

    if (projects.length === 0) {
      container.append(`<div class="col-12 text-muted">No projects yet.</div>`);
      renderStats();
      return;
    }

    projects.forEach(project => {
      const badgeClass = project.status === "active"
        ? "wild-badge-active"
        : "wild-badge-planning";

      const ecologistNames = project.ecologistIds
        .map(getEcologistNameById)
        .join(", ");

      const card = `
        <div class="col-md-4">
          <div class="wild-card wild-project-card">
            <div class="wild-project-header">
              <h3 class="wild-project-title">${project.name}</h3>
              <span class="wild-badge ${badgeClass}">
                ${project.status.toUpperCase()}
              </span>
            </div>

            <p class="wild-project-desc">${project.description || "No description."}</p>

            <div class="wild-project-meta">
              <span>${project.location || "Location not set"}</span>
              <span>${project.ecologistIds.length} Ecologists</span>
              <span>${project.sensors} Sensors</span>
            </div>

            <div class="small text-muted">
              ${ecologistNames || "No ecologists assigned yet."}
            </div>

            <div class="wild-project-footer">
              <button 
                class="wild-btn wild-btn-outline wild-btn-sm manage-project-btn"
                data-project-id="${project.id}">
                Manage
              </button>
            </div>
          </div>
        </div>
      `;

      container.append(card);
    });

    renderStats();
  }

  function resetProjectForm() {
    $("#projectId").val("");
    $("#projectName").val("");
    $("#projectDescription").val("");
    $("#projectLocation").val("");
    $("#projectStatus").val("active");
    $("#projectSensors").val(0);
    $("#projectEcologists").val([]);
    $("#selectedEcologistsList").text("None selected yet.");
  }

  function openProjectModalForNew() {
    $("#projectModalLabel").text("Create New Project");
    resetProjectForm();
    $("#deleteProjectBtn").hide();
  }

  function openProjectModalForEdit(projectId) {
    const project = projects.find(p => p.id === projectId);
    if (!project) return;

    $("#projectModalLabel").text("Manage Project");
    $("#projectId").val(project.id);
    $("#projectName").val(project.name);
    $("#projectDescription").val(project.description);
    $("#projectLocation").val(project.location);
    $("#projectStatus").val(project.status);
    $("#projectSensors").val(project.sensors);
    $("#projectEcologists").val(project.ecologistIds.map(String));

    updateSelectedEcologistsPreview();
    $("#deleteProjectBtn").show();
  }

  function updateSelectedEcologistsPreview() {
    const ids = $("#projectEcologists").val() || [];
    if (ids.length === 0) {
      $("#selectedEcologistsList").text("None selected yet.");
      return;
    }

    const names = ids.map(id => getEcologistNameById(Number(id))).join(", ");
    $("#selectedEcologistsList").text(names);
  }

  function closeModal() {
    const modalElement = document.getElementById("projectModal");
    const modal = bootstrap.Modal.getInstance(modalElement);
    if (modal) modal.hide();
  }

  // ================================================================
  // EVENT BINDINGS
  // ================================================================

  $("#newProjectBtn").on("click", openProjectModalForNew);

  $("#projectEcologists").on("change", updateSelectedEcologistsPreview);

  $("#projectsList").on("click", ".manage-project-btn", function () {
    const id = Number($(this).data("project-id"));
    openProjectModalForEdit(id);

    const modal = new bootstrap.Modal(document.getElementById("projectModal"));
    modal.show();
  });

  $("#projectForm").on("submit", function (e) {
    e.preventDefault();

    const idVal = $("#projectId").val();
    const isNew = idVal === "";

    const data = {
      name: $("#projectName").val().trim(),
      description: $("#projectDescription").val().trim(),
      location: $("#projectLocation").val().trim(),
      status: $("#projectStatus").val(),
      sensors: Number($("#projectSensors").val()) || 0,
      ecologistIds: ($("#projectEcologists").val() || []).map(Number)
    };

    if (!data.name) {
      alert("Project name is required.");
      return;
    }

    if (isNew) {
      // TODO (backend): POST /api/projects
      data.id = nextProjectId++;
      projects.push(data);
    } else {
      // TODO (backend): PUT /api/projects/:id
      const id = Number(idVal);
      const index = projects.findIndex(p => p.id === id);
      if (index !== -1) {
        projects[index] = { id, ...data };
      }
    }

    closeModal();
    renderProjects();
  });

  $("#deleteProjectBtn").on("click", function () {
    const id = Number($("#projectId").val());
    if (!id) return;

    const confirmed = confirm("Delete this project? This cannot be undone.");
    if (!confirmed) return;

    // TODO (backend): DELETE /api/projects/:id  
    const index = projects.findIndex(p => p.id === id);
    if (index !== -1) projects.splice(index, 1);

    closeModal();
    renderProjects();
  });

  // ================================================================
  // INITIAL RENDER
  // ================================================================
  renderEcologistSelect();
  renderProjects();
});