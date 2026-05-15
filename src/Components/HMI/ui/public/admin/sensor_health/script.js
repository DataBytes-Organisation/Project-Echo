/* Project Echo — Sensor Health script.js
   Handles sidebar toggle, theme, actions and live sensor UI helpers
*/

const menuToggle = document.getElementById("menu-toggle");
const mobileBackdrop = document.getElementById("mobile-backdrop");

if (menuToggle) {
  menuToggle.addEventListener("click", () => {
    document.body.classList.toggle("sidebar-open");
  });
}

if (mobileBackdrop) {
  mobileBackdrop.addEventListener("click", () => {
    document.body.classList.remove("sidebar-open");
  });
}

const themeToggle = document.getElementById("theme-toggle");

function setTheme(darkMode) {
  if (darkMode) {
    document.documentElement.setAttribute("data-theme", "dark");
    themeToggle && (themeToggle.textContent = "☀️");
    localStorage.setItem("echo-theme", "dark");
  } else {
    document.documentElement.removeAttribute("data-theme");
    themeToggle && (themeToggle.textContent = "🌙");
    localStorage.setItem("echo-theme", "light");
  }
}

if (themeToggle) {
  const savedTheme = localStorage.getItem("echo-theme");
  setTheme(savedTheme === "dark");
  themeToggle.addEventListener("click", () => {
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    setTheme(!isDark);
  });
}

// ================================================================
// Shared admin page state helpers
// ================================================================
const pageState = window.createAdminPageState ? createAdminPageState() : null;
pageState?.resetPageState();

function showPageLoading() {
  pageState?.showLoading();
}

function hidePageLoading() {
  pageState?.hideLoading();
}

function showPageError(message) {
  pageState?.showError(message);
}

function hidePageError() {
  pageState?.hideError();
}

// ================================================================
// Inline page message helper (existing behavior kept)
// ================================================================
function showMessage(elementId, message, type = "success") {
  const el = document.getElementById(elementId);
  if (!el) return;
  const colors = {success: "var(--primary)", warning: "var(--warning)", danger: "var(--danger)"};
  el.style.color = colors[type] || colors.success;
  el.style.marginTop = "10px";
  el.style.fontWeight = "600";
  el.textContent = message;
  el.style.opacity = 0;
  setTimeout(() => {
    el.style.opacity = 1;
    el.style.transition = "opacity .4s";
  }, 30);
}

// ================================================================
// API helper
// ================================================================
async function apiFetch(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(text || `Request failed: ${response.status}`);
  }

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }

  return response.text();
}

function pillHtml(status) {
  const s = String(status || "").trim();
  let cls = "pill-warning";
  if (s === "Online" || s === "Success") cls = "pill-success";
  else if (s === "Offline" || s === "Failed") cls = "pill-danger";
  else if (
    s === "Warning" ||
    s === "Queued" ||
    s === "High CPU" ||
    s === "High RAM" ||
    s === "High Disk"
  ) {
    cls = "pill-warning";
  }
  return `<span class="pill ${cls}">${s || "Unknown"}</span>`;
}

function formatPercent(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "—";
  return `${num}%`;
}

function formatGps(gps) {
  if (!gps || typeof gps !== "object") return "—";
  const lat = gps.lat;
  const lon = gps.lon;
  if (lat === null || lat === undefined || lon === null || lon === undefined) {
    return "—";
  }
  return `${lat}, ${lon}`;
}

function formatUptime(seconds) {
  const total = Number(seconds);
  if (!Number.isFinite(total) || total < 0) return "—";
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const mins = Math.floor((total % 3600) / 60);
  if (days > 0) return `${days}d ${hours}h ${mins}m`;
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
}

// ================================================================
// Reboot sensors
// ================================================================
async function rebootSensors() {
  const sensorsRaw = document.getElementById("reboot-sensor")?.value.trim();
  const reason = document.getElementById("reboot-reason")?.value.trim();

  if (!sensorsRaw) {
    showMessage("reboot-message", "Please enter at least one sensor ID.", "warning");
    showPageError("Please enter at least one sensor ID.");
    return;
  }

  const sensors = sensorsRaw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  hidePageError();
  showPageLoading();

  try {
    for (const sensorId of sensors) {
      await apiFetch(`/sensors/${encodeURIComponent(sensorId)}/reboot`, {
        method: "POST",
        body: JSON.stringify({ reason: reason || null }),
      });
    }

    showMessage("reboot-message", `Reboot queued for ${sensors.join(", ")}.`, "success");
    await loadRecentRebootHistory();
  } catch (e) {
    showMessage("reboot-message", `Failed to queue reboot: ${e.message}`, "danger");
    showPageError(`Failed to queue reboot: ${e.message}`);
  } finally {
    hidePageLoading();
  }
}

// ================================================================
// Settings helpers
// ================================================================
function intervalLabelToSeconds(label) {
  const text = String(label || "").toLowerCase();
  if (text.includes("30")) return 30;
  if (text.includes("10")) return 600;
  if (text.includes("5")) return 300;
  return 60;
}

function secondsToIntervalLabel(seconds) {
  const s = Number(seconds);
  if (s === 30) return "30 seconds";
  if (s === 300) return "5 minutes";
  if (s === 600) return "10 minutes";
  return "1 minute";
}

// ================================================================
// Save settings
// ================================================================
async function saveSettings() {
  const intervalLabel = document.getElementById("record-interval")?.value;
  const sensitivity = document.getElementById("sensitivity")?.value;
  const battery = Number(document.getElementById("battery-threshold")?.value);

  hidePageError();
  showPageLoading();

  try {
    const payload = {
      recordIntervalSeconds: intervalLabelToSeconds(intervalLabel),
      sensitivity: sensitivity || "Medium",
      batteryThresholdPct: Number.isFinite(battery) ? battery : 25,
    };

    await apiFetch(`/sensors/__default__/settings`, {
      method: "PUT",
      body: JSON.stringify({ settings: payload }),
    });

    showMessage("settings-message", "Settings saved.", "success");
  } catch (e) {
    showMessage("settings-message", `Failed to save settings: ${e.message}`, "danger");
    showPageError(`Failed to save settings: ${e.message}`);
  } finally {
    hidePageLoading();
  }
}

// ================================================================
// Fake add project
// ================================================================
function fakeCreateProject() {
  const name = document.getElementById("project-name")?.value.trim();
  const loc = document.getElementById("project-location")?.value.trim();
  const sensors = document.getElementById("project-sensors")?.value.trim();

  if (!name || !loc) {
    showMessage("project-message", "Project name and location are required.", "warning");
    return;
  }

  showMessage(
    "project-message",
    `Project "${name}" created successfully. Assigned sensors: ${sensors || "None"}.`
  );
}

document.querySelectorAll(".card").forEach((card) => {
  card.style.opacity = 0;
  setTimeout(() => {
    card.style.opacity = 1;
    card.style.transition = "opacity .45s ease";
  }, 120);
});

const shell = document.querySelector(".dashboard-shell");
if (shell) {
  shell.style.opacity = 0;
  setTimeout(() => {
    shell.style.opacity = 1;
    shell.style.transition = "opacity .5s ease";
  }, 80);
}

// Ensure sidebar nav always contains expected links
function ensureSensorSidebar() {
  const sidebar = document.querySelector(".sidebar .nav-section");
  if (!sidebar) return;

  const required = [
    { href: "/admin/dashboard.html", icon: "▦", text: "Dashboard" },
    { href: "/admin/sensor-health.html", icon: "📡", text: "Sensor Health" },
    { href: "/admin/sensor_health/alerts.html", icon: "🔔", text: "Alerts" },
    { href: "/admin/sensor_health/reboot.html", icon: "⟲", text: "Reboot" },
    { href: "/admin/sensor_health/settings.html", icon: "⚙", text: "Settings" },
    { href: "/admin/sensor_health/add-project.html", icon: "➕", text: "Add a Project" }
  ];

  required.forEach((item) => {
    if (!sidebar.querySelector(`a[href="${item.href}"]`)) {
      const a = document.createElement("a");
      a.className = "nav-link";
      a.href = item.href;
      a.innerHTML = `<span class="nav-icon">${item.icon}</span><span class="nav-text">${item.text}</span>`;
      sidebar.appendChild(a);
    }
  });
}

document.addEventListener("DOMContentLoaded", function () {
  ensureSensorSidebar();
  const navSection = document.querySelector(".sidebar .nav-section");
  if (navSection) {
    const observer = new MutationObserver(() => ensureSensorSidebar());
    observer.observe(navSection, { childList: true });
  }
});

// ================================================================
// Sensor Health page
// ================================================================
async function loadSensorHealthPage() {
  const tbody = document.getElementById("sensor-overview-tbody");
  if (!tbody) return;

  const statusFilter = document.getElementById("sensor-status-filter");
  let lastItems = [];

  function render() {
    const statusVal = statusFilter?.value || "All";

    const filtered = lastItems.filter((item) => {
      if (statusVal !== "All" && item.status !== statusVal) return false;
      return true;
    });

    tbody.innerHTML = "";

    if (!filtered.length) {
      tbody.innerHTML = '<tr><td colspan="8">No live sensor data available.</td></tr>';
      return;
    }

    for (const item of filtered) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${item.sensorId || "—"}</td>
        <td>${pillHtml(item.status)}</td>
        <td>${formatPercent(item.cpu)}</td>
        <td>${formatPercent(item.ram)}</td>
        <td>${formatPercent(item.disk)}</td>
        <td>${formatUptime(item.uptime)}</td>
        <td>${formatGps(item.gps)}</td>
        <td>${item.lastAudio || "—"}</td>
      `;
      tbody.appendChild(tr);
    }
  }

  async function refresh() {
    hidePageError();
    showPageLoading();

    try {
      const data = await apiFetch("/sensors/updates");
      lastItems = Array.isArray(data.items) ? data.items : [];
      render();
    } catch (e) {
      tbody.innerHTML = `<tr><td colspan="8">Failed to load sensors: ${e.message}</td></tr>`;
    }
  }

  statusFilter?.addEventListener("change", render);

  await refresh();
  setInterval(refresh, 15000);
}

// ================================================================
// Alerts page
// ================================================================
async function loadAlertsPage() {
  const tbody = document.getElementById("alerts-tbody");
  if (!tbody) return;

  hidePageError();
  showPageLoading();

  try {
    const data = await apiFetch("/sensors/alerts");
    const items = Array.isArray(data.items) ? data.items : [];
    tbody.innerHTML = "";
    if (!items.length) {
      tbody.innerHTML = '<tr><td colspan="4">No active alerts.</td></tr>';
      return;
    }

    for (const alert of items) {
      const tr = document.createElement("tr");
      const sevPill = pillHtml(alert.issue);
      tr.innerHTML = `
        <td>${alert.sensorId || "—"}</td>
        <td>${sevPill}</td>
        <td>${alert.details || ""}</td>
        <td>${alert.lastAudioMinutesAgo !== undefined ? alert.lastAudioMinutesAgo : "—"}</td>
      `;
      tbody.appendChild(tr);
    }
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="4">Failed to load alerts: ${e.message}</td></tr>`;
    showPageError(`Failed to load alerts: ${e.message}`);
  } finally {
    hidePageLoading();
  }
}

// ================================================================
// Reboot history page
// ================================================================
async function loadRecentRebootHistory() {
  const tbody = document.getElementById("reboot-history-tbody");
  if (!tbody) return;

  hidePageError();
  showPageLoading();

  try {
    const data = await apiFetch("/sensors/reboots/recent?limit=50");
    const items = Array.isArray(data.items) ? data.items : [];
    tbody.innerHTML = "";
    if (!items.length) {
      tbody.innerHTML = '<tr><td colspan="3">No reboot history yet.</td></tr>';
      return;
    }

    for (const r of items) {
      const t = r.requestedAt ? new Date(r.requestedAt).toLocaleString() : "—";
      const status = String(r.status || "Queued");
      const cls = status.toLowerCase().includes("fail") ? "pill-danger" : status.toLowerCase().includes("success") ? "pill-success" : "pill-warning";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${r.sensorId}</td>
        <td><span class="pill ${cls}">${status}</span></td>
        <td>${t}</td>
      `;
      tbody.appendChild(tr);
    }
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="3">Failed to load reboot history: ${e.message}</td></tr>`;
    showPageError(`Failed to load reboot history: ${e.message}`);
  } finally {
    hidePageLoading();
  }
}

// ================================================================
// Settings page
// ================================================================
async function loadSettingsPage() {
  const interval = document.getElementById("record-interval");
  const sensitivity = document.getElementById("sensitivity");
  const battery = document.getElementById("battery-threshold");
  if (!interval || !sensitivity || !battery) return;

  hidePageError();
  showPageLoading();

  try {
    const data = await apiFetch("/sensors/__default__/settings");
    const settings = data.settings || {};
    const intervalLabel = secondsToIntervalLabel(settings.recordIntervalSeconds);

    interval.value = intervalLabel;
    sensitivity.value = settings.sensitivity || "Medium";
    battery.value = Number(settings.batteryThresholdPct || 25);
  } catch (e) {
    showMessage("settings-message", `Failed to load settings: ${e.message}`, "danger");
  }
}

// ================================================================
// Existing onclick compatibility
// ================================================================
window.fakeReboot = rebootSensors;
window.fakeSaveSettings = saveSettings;
window.fakeCreateProject = fakeCreateProject;

document.addEventListener("DOMContentLoaded", () => {
  loadSensorHealthPage();
  loadAlertsPage();
  loadRecentRebootHistory();
  loadSettingsPage();
});