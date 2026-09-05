/* Project Echo — Sensor Health script.js
   Sprint 2: UI enhancement and live status monitoring
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
    if (themeToggle) themeToggle.textContent = "☀️";
    localStorage.setItem("echo-theme", "dark");
  } else {
    document.documentElement.removeAttribute("data-theme");
    if (themeToggle) themeToggle.textContent = "🌙";
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
// Named distinctly: each admin page also declares its own `pageState`, and a
// duplicate top-level `const` in a classic script aborts this whole file.
const sensorHealthPageState = window.createAdminPageState ? window.createAdminPageState() : null;
sensorHealthPageState?.resetPageState();

function showPageLoading() {
  sensorHealthPageState?.showLoading();
}

function hidePageLoading() {
  sensorHealthPageState?.hideLoading();
}

function showPageError(message) {
  sensorHealthPageState?.showError(message);
}

function hidePageError() {
  sensorHealthPageState?.hideError();
}

// ================================================================
// Inline page message helper (existing behavior kept)
// ================================================================
function showMessage(elementId, message, type = "success") {
  const el = document.getElementById(elementId);
  if (!el) return;

  const colors = {
    success: "var(--primary)",
    warning: "var(--warning)",
    danger: "var(--danger)"
  };

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
  const { timeoutMs = 8000, headers, signal, ...rest } = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  try {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(headers || {}) },
      signal: controller.signal,
      ...rest,
    });

    const contentType = response.headers.get("content-type") || "";
    const isJson = contentType.includes("application/json");
    const payload = isJson
      ? await response.json().catch(() => null)
      : await response.text().catch(() => "");

    if (!response.ok) {
      const detail =
        payload && typeof payload === "object"
          ? payload.detail || payload.error || JSON.stringify(payload)
          : payload;
      const error = new Error(detail || `Request failed: ${response.status}`);
      error.status = response.status;
      throw error;
    }

    return payload;
  } catch (error) {
    if (error.name === "AbortError") {
      const timeoutError = new Error("Request timed out while loading sensor data");
      timeoutError.status = 408;
      throw timeoutError;
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function deviceDetailUrl(sensorId) {
  return `/admin/sensor_health/device-detail.html?sensorId=${encodeURIComponent(sensorId)}`;
}

function openDeviceDetail(sensorId) {
  if (!sensorId) return;
  window.location.href = deviceDetailUrl(sensorId);
}

// `Number(null)` is 0, not NaN, so a missing reading would otherwise render as a
// real value ("0%", "Just now"). Every formatter must reject null/"" first.
function toNumberOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatMinutesAgo(mins) {
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins} min ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function formatLastAudio(item) {
  if (!item || typeof item !== "object") return "—";
  if (typeof item.lastAudio === "string" && item.lastAudio && item.lastAudio !== "—") {
    return item.lastAudio;
  }

  const mins = toNumberOrNull(item.lastAudioMinutesAgo);
  if (mins !== null) return formatMinutesAgo(mins);

  if (item.lastAudioTs) {
    const parsed = new Date(item.lastAudioTs);
    if (!Number.isNaN(parsed.getTime())) return parsed.toLocaleString();
  }

  return "—";
}

function formatLastSeen(item) {
  const mins = toNumberOrNull(item?.lastSeenMinutesAgo);
  if (mins !== null) return formatMinutesAgo(mins);

  if (item?.lastSeen) {
    const parsed = new Date(item.lastSeen);
    if (!Number.isNaN(parsed.getTime())) return parsed.toLocaleString();
    return String(item.lastSeen);
  }

  return "Never reported";
}

function formatDateTime(value) {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleString();
}

function pillHtml(status, batteryPct) {
  const s = String(status || "").trim();

  if (typeof batteryPct === "number" && batteryPct < 20) {
    return `<span class="pill pill-warning">Low Battery</span>`;
  }

  if (s === "Online" || s === "Success") {
    return `<span class="pill pill-success">${s}</span>`;
  }

  if (s === "Offline" || s === "Failed") {
    return `<span class="pill pill-danger">${s}</span>`;
  }

  // A device that has never checked in is not a fault, so it must not read as one.
  if (!s || s === "Unknown") {
    return `<span class="pill pill-muted" title="This device has never sent a heartbeat">Unknown</span>`;
  }

  return `<span class="pill pill-warning">${s}</span>`;
}

function formatPercent(value) {
  const num = toNumberOrNull(value);
  if (num === null) return "—";
  return `${num}%`;
}

function formatBattery(value) {
  const num = toNumberOrNull(value);
  if (num === null) return "—";

  const lowClass = num < 20 ? "battery-low" : "";
  return `<span class="${lowClass}">${num}%</span>`;
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
  const total = toNumberOrNull(seconds);
  if (total === null || total < 0) return "—";

  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const mins = Math.floor((total % 3600) / 60);

  if (days > 0) return `${days}d ${hours}h ${mins}m`;
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
}

function updateLastUpdated(count) {
  const el = document.getElementById("last-updated-at");
  if (!el) return;

  const now = new Date();
  const suffix = count === undefined ? "" : ` · ${count} device${count === 1 ? "" : "s"}`;
  el.textContent = `Last updated at: ${now.toLocaleTimeString()}${suffix}`;
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
  const searchInput = document.getElementById("sensor-search-input");

  let lastItems = [];

  function render() {
    const statusVal = statusFilter?.value || "All";
    const searchVal = (searchInput?.value || "").trim().toLowerCase();

    const filtered = lastItems.filter((item) => {
      const sensorId = String(item.sensorId || "").toLowerCase();
      const batteryPct = toNumberOrNull(item.batteryPct);

      const matchesSearch = !searchVal || sensorId.includes(searchVal);

      let matchesStatus = true;
      if (statusVal === "Online") {
        matchesStatus = item.status === "Online";
      } else if (statusVal === "Offline") {
        matchesStatus = item.status === "Offline";
      } else if (statusVal === "Unknown") {
        matchesStatus = !item.status || item.status === "Unknown";
      } else if (statusVal === "Low Battery") {
        matchesStatus = batteryPct !== null && batteryPct < 20;
      }

      return matchesSearch && matchesStatus;
    });

    tbody.innerHTML = "";

    if (!filtered.length) {
      tbody.innerHTML = '<tr><td colspan="11">No sensors match the current search or filter.</td></tr>';
      return;
    }

    for (const item of filtered) {
      const tr = document.createElement("tr");
      const sensorId = item.sensorId || "";

      if (typeof item.batteryPct === "number" && item.batteryPct < 20) {
        tr.classList.add("sensor-row-low-battery");
      }

      tr.classList.add("sensor-row-clickable");
      tr.tabIndex = 0;
      tr.setAttribute("role", "link");
      tr.setAttribute("aria-label", `Open device detail for ${sensorId || "sensor"}`);

      const componentCount = toNumberOrNull(item.componentCount);

      tr.innerHTML = `
        <td>${escapeHtml(sensorId) || "—"}</td>
        <td>${escapeHtml(item.name || "—")}</td>
        <td>${pillHtml(item.status, item.batteryPct)}</td>
        <td>${escapeHtml(formatDeviceType(item.type))}</td>
        <td>${escapeHtml(item.model || "—")}</td>
        <td>${formatBattery(item.batteryPct)}</td>
        <td>${escapeHtml(formatTemperature(item.temperatureC))}</td>
        <td>${componentCount === null ? "—" : componentCount}</td>
        <td>${formatGps(item.gps)}</td>
        <td>${escapeHtml(formatLastAudio(item))}</td>
        <td>
          <a class="btn-secondary device-detail-link" href="${deviceDetailUrl(sensorId)}">View</a>
        </td>
      `;

      tr.addEventListener("click", (event) => {
        if (event.target.closest("a")) return;
        openDeviceDetail(sensorId);
      });
      tr.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openDeviceDetail(sensorId);
        }
      });

      tbody.appendChild(tr);
    }
  }

  // The seeded inventory rarely changes between polls, so a manual refresh can
  // look like it did nothing. Disabling the button and stamping the row count
  // gives the click visible feedback either way.
  async function refresh({ manual = false } = {}) {
    hidePageError();

    const sourceEl = document.getElementById("sensor-data-source");
    const refreshButton = document.getElementById("sensor-refresh-button");

    if (sourceEl) sourceEl.textContent = "Loading…";
    if (manual && refreshButton) {
      refreshButton.disabled = true;
      refreshButton.textContent = "Refreshing…";
    }

    try {
      const data = await apiFetch("/sensors/updates", { timeoutMs: 4000 });
      lastItems = Array.isArray(data?.items) ? data.items : [];
      if (sourceEl) {
        sourceEl.textContent = data?.source === "demo-fallback" ? "Demo fallback" : "Backend API";
      }
      updateLastUpdated(lastItems.length);
      render();
    } catch (e) {
      tbody.innerHTML = `<tr><td colspan="11">Failed to load sensors: ${escapeHtml(e.message)}</td></tr>`;
      showPageError(`Failed to load sensors: ${e.message}`);
      if (sourceEl) sourceEl.textContent = "Unavailable";
    } finally {
      if (manual && refreshButton) {
        refreshButton.disabled = false;
        refreshButton.textContent = "Refresh";
      }
    }
  }

  statusFilter?.addEventListener("change", render);
  searchInput?.addEventListener("input", render);
  document.getElementById("sensor-refresh-button")?.addEventListener("click", () => {
    refresh({ manual: true });
  });

  await refresh();
  setInterval(() => refresh(), 15000);
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
      const sensorId = alert.sensorId || "";
      const sevPill = pillHtml(alert.issue);
      const lastAudio = alert.lastAudioMinutesAgo !== undefined
        ? formatLastAudio({ lastAudioMinutesAgo: alert.lastAudioMinutesAgo, lastAudioTs: alert.lastAudioTs })
        : "—";

      tr.classList.add("sensor-row-clickable");
      tr.tabIndex = 0;
      tr.innerHTML = `
        <td>
          <a class="device-detail-link" href="${deviceDetailUrl(sensorId)}">${escapeHtml(sensorId) || "—"}</a>
        </td>
        <td>${sevPill}</td>
        <td>${escapeHtml(alert.details || "")}</td>
        <td>${escapeHtml(lastAudio)}</td>
      `;

      tbody.appendChild(tr);
    }

    if (typeof window.renderAlertsChart === "function") {
      window.renderAlertsChart(items);
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
      const cls = status.toLowerCase().includes("fail")
        ? "pill-danger"
        : status.toLowerCase().includes("success")
        ? "pill-success"
        : "pill-warning";

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
// Device detail workflow (FR-C2)
// ================================================================
function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value ?? "—";
}

function showDeviceEmptyState(message) {
  const empty = document.getElementById("device-empty-state");
  const root = document.getElementById("device-detail-root");
  const emptyMessage = document.getElementById("device-empty-message");
  if (emptyMessage && message) emptyMessage.textContent = message;
  if (empty) {
    empty.hidden = false;
    empty.classList.remove("d-none");
  }
  if (root) {
    root.hidden = true;
    root.classList.add("d-none");
  }
}

function showDeviceDetailRoot() {
  const empty = document.getElementById("device-empty-state");
  const root = document.getElementById("device-detail-root");
  if (empty) {
    empty.hidden = true;
    empty.classList.add("d-none");
  }
  if (root) {
    root.hidden = false;
    root.classList.remove("d-none");
  }
}

let deviceLocationMap = null;

function deviceMarkerStyle() {
  return new ol.style.Style({
    image: new ol.style.Circle({
      radius: 7,
      fill: new ol.style.Fill({ color: "#c8473c" }),
      stroke: new ol.style.Stroke({ color: "#ffffff", width: 2 }),
    }),
  });
}

// Rendered with the bundled OpenLayers build rather than an embedded map iframe:
// the HMI's Content-Security-Policy only allows frames from 'self', so a
// third-party iframe is blocked outright.
function renderDeviceLocation(gps) {
  const text = document.getElementById("device-location-text");
  const mapEl = document.getElementById("device-location-map");
  if (!text || !mapEl) return;

  const lat = toNumberOrNull(gps?.lat);
  const lon = toNumberOrNull(gps?.lon);

  if (lat === null || lon === null) {
    text.textContent = "This device has not reported a GPS position.";
    mapEl.hidden = true;
    mapEl.classList.add("d-none");
    return;
  }

  text.textContent = `Latitude ${lat}, longitude ${lon}`;
  mapEl.hidden = false;
  mapEl.classList.remove("d-none");

  if (typeof ol === "undefined") {
    text.textContent = `Latitude ${lat}, longitude ${lon} (map unavailable)`;
    mapEl.hidden = true;
    return;
  }

  const center = ol.proj.fromLonLat([lon, lat]);
  const marker = new ol.Feature({ geometry: new ol.geom.Point(center) });
  marker.setStyle(deviceMarkerStyle());

  if (deviceLocationMap) {
    deviceLocationMap.getView().setCenter(center);
    deviceLocationMap.updateSize();
    return;
  }

  deviceLocationMap = new ol.Map({
    target: mapEl,
    layers: [
      new ol.layer.Tile({ source: new ol.source.OSM() }),
      new ol.layer.Vector({ source: new ol.source.Vector({ features: [marker] }) }),
    ],
    view: new ol.View({ center, zoom: 13 }),
  });

  // The card is revealed immediately before this runs, so the canvas needs a
  // resize once the browser has settled the final layout.
  setTimeout(() => deviceLocationMap?.updateSize(), 0);
}

function renderDeviceHardware(detail) {
  const hardware = detail.hardware || {};
  setText("device-hw-type", formatDeviceType(detail.type));
  setText("device-hw-model", detail.model || "—");
  setText("device-hw-processor", hardware.processor || "—");
  setText("device-hw-clock", hardware.clockSpeed || "—");
  setText("device-hw-memory", hardware.memory || "—");
  setText("device-hw-storage", hardware.storage || "—");
}

function formatDeviceType(type) {
  if (!type) return "—";
  return String(type)
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function renderDevicePower(detail) {
  const power = detail.power || {};

  const health = toNumberOrNull(power.batteryHealthPct);
  setText(
    "device-health-battery-health",
    health === null
      ? "—"
      : `${health}%${power.batteryCapacity ? ` (${power.batteryCapacity})` : ""}`
  );

  setText("device-health-temperature", formatTemperature(detail.temperatureC));

  const output = toNumberOrNull(power.solarOutputW);
  const rated = toNumberOrNull(power.solarRatedW);
  const pct = toNumberOrNull(power.solarOutputPct);

  if (output !== null && rated !== null) {
    setText("device-health-solar", `${output} W of ${rated} W${pct === null ? "" : ` (${pct}%)`}`);
  } else if (output !== null) {
    setText("device-health-solar", `${output} W`);
  } else {
    setText("device-health-solar", "—");
  }
}

function formatTemperature(value) {
  const number = toNumberOrNull(value);
  return number === null ? "—" : `${number} °C`;
}

function renderDeviceComponents(components) {
  const tbody = document.getElementById("device-components-tbody");
  if (!tbody) return;

  const items = Array.isArray(components) ? components : [];
  if (!items.length) {
    tbody.innerHTML = '<tr><td colspan="4">This device does not report any components.</td></tr>';
    return;
  }

  tbody.innerHTML = "";
  for (const component of items) {
    const readings = Array.isArray(component.metrics) ? component.metrics : [];
    const readingText = readings.length
      ? readings.map((m) => `${m.label}: ${m.display}`).join(" · ")
      : "No readings reported";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(formatDeviceType(component.type))}</td>
      <td>${escapeHtml(formatDeviceType(component.category))}</td>
      <td>${escapeHtml(component.model || "—")}</td>
      <td>${escapeHtml(readingText)}</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderConnectedDevices(devices) {
  const container = document.getElementById("device-connected-list");
  if (!container) return;

  const items = Array.isArray(devices) ? devices : [];
  if (!items.length) {
    container.textContent = "This device has no recorded mesh neighbours.";
    return;
  }

  container.innerHTML = items
    .map((device) => {
      const label = device.name ? `${device.sensorId} · ${device.name}` : device.sensorId;
      if (!device.known) {
        return `<span class="device-connected-item device-connected-missing" title="Not present in the device inventory">${escapeHtml(label)} (unknown)</span>`;
      }
      return `<a class="device-connected-item" href="${deviceDetailUrl(device.sensorId)}">${escapeHtml(label)}</a>`;
    })
    .join("");
}

function renderLastAudio(detail) {
  const lastAudio = detail.lastAudio || null;
  const note = document.getElementById("device-audio-note");

  setText("device-audio-when", formatLastAudio(detail));
  setText("device-audio-species", lastAudio?.species || "—");
  setText(
    "device-audio-confidence",
    toNumberOrNull(lastAudio?.confidence) === null
      ? "—"
      : `${toNumberOrNull(lastAudio.confidence)}%`
  );
  setText(
    "device-audio-sample-rate",
    toNumberOrNull(lastAudio?.sampleRate) === null
      ? "—"
      : `${toNumberOrNull(lastAudio.sampleRate)} Hz`
  );

  if (note) {
    note.textContent = lastAudio
      ? ""
      : "No audio has been recorded from this device yet, so there is no recording metadata to show.";
  }
}

function renderTelemetryNote(detail) {
  const note = document.getElementById("device-telemetry-note");
  if (!note) return;

  // Distinguishes "no data exists yet" from "the device is genuinely unhealthy",
  // since live heartbeat ingestion is not connected to this view yet.
  note.textContent =
    detail.lastSeen == null
      ? "This device has never sent a heartbeat, so its status is Unknown rather than Offline. The readings shown here come from the device inventory."
      : "";
}

function renderAudioHistory(events) {
  const tbody = document.getElementById("device-audio-history-tbody");
  if (!tbody) return;

  const items = Array.isArray(events) ? events : [];
  if (!items.length) {
    tbody.innerHTML =
      '<tr><td colspan="4">No audio events recorded for this device yet.</td></tr>';
    return;
  }

  tbody.innerHTML = "";
  for (const event of items) {
    const tr = document.createElement("tr");
    const confidence = toNumberOrNull(event.confidence);
    const sampleRate = toNumberOrNull(event.sampleRate);
    tr.innerHTML = `
      <td>${escapeHtml(formatDateTime(event.timestamp))}</td>
      <td>${escapeHtml(event.species || "—")}</td>
      <td>${confidence === null ? "—" : `${confidence}%`}</td>
      <td>${sampleRate === null ? "—" : `${sampleRate} Hz`}</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderDeviceRebootHistory(items) {
  const tbody = document.getElementById("device-reboot-history-tbody");
  if (!tbody) return;

  const rows = Array.isArray(items) ? items : [];
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="3">No reboot history yet.</td></tr>';
    return;
  }

  tbody.innerHTML = "";
  for (const r of rows) {
    const status = String(r.status || "Queued");
    const cls = status.toLowerCase().includes("fail")
      ? "pill-danger"
      : status.toLowerCase().includes("success")
      ? "pill-success"
      : "pill-warning";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="pill ${cls}">${escapeHtml(status)}</span></td>
      <td>${escapeHtml(r.reason || "—")}</td>
      <td>${escapeHtml(formatDateTime(r.requestedAt))}</td>
    `;
    tbody.appendChild(tr);
  }
}

function applyDeviceSettingsForm(settings) {
  const interval = document.getElementById("device-record-interval");
  const sensitivity = document.getElementById("device-sensitivity");
  const battery = document.getElementById("device-battery-threshold");
  if (!interval || !sensitivity || !battery) return;

  interval.value = secondsToIntervalLabel(settings.recordIntervalSeconds);
  sensitivity.value = settings.sensitivity || "Medium";
  battery.value = Number(settings.batteryThresholdPct || 25);
}

async function loadDeviceRebootHistory(sensorId) {
  try {
    const data = await apiFetch(`/sensors/${encodeURIComponent(sensorId)}/reboots?limit=20`);
    renderDeviceRebootHistory(data.items);
  } catch (e) {
    const tbody = document.getElementById("device-reboot-history-tbody");
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="3">Failed to load reboot history: ${escapeHtml(e.message)}</td></tr>`;
    }
  }
}

async function rebootSelectedDevice(sensorId) {
  const reason = document.getElementById("device-reboot-reason")?.value.trim();
  if (!window.confirm(`Queue a reboot for ${sensorId}?`)) return;

  hidePageError();
  showPageLoading();
  try {
    await apiFetch(`/sensors/${encodeURIComponent(sensorId)}/reboot`, {
      method: "POST",
      body: JSON.stringify({ reason: reason || null }),
    });
    showMessage("device-reboot-message", `Reboot queued for ${sensorId}.`, "success");
    await loadDeviceRebootHistory(sensorId);
  } catch (e) {
    showMessage("device-reboot-message", `Failed to queue reboot: ${e.message}`, "danger");
    showPageError(`Failed to queue reboot: ${e.message}`);
  } finally {
    hidePageLoading();
  }
}

async function saveSelectedDeviceSettings(sensorId) {
  const intervalLabel = document.getElementById("device-record-interval")?.value;
  const sensitivity = document.getElementById("device-sensitivity")?.value;
  const battery = Number(document.getElementById("device-battery-threshold")?.value);

  hidePageError();
  showPageLoading();
  try {
    const payload = {
      recordIntervalSeconds: intervalLabelToSeconds(intervalLabel),
      sensitivity: sensitivity || "Medium",
      batteryThresholdPct: Number.isFinite(battery) ? battery : 25,
    };

    await apiFetch(`/sensors/${encodeURIComponent(sensorId)}/settings`, {
      method: "PUT",
      body: JSON.stringify({ settings: payload }),
    });
    showMessage("device-settings-message", "Device settings saved.", "success");
  } catch (e) {
    showMessage("device-settings-message", `Failed to save settings: ${e.message}`, "danger");
    showPageError(`Failed to save settings: ${e.message}`);
  } finally {
    hidePageLoading();
  }
}

async function loadDeviceDetailPage() {
  const root = document.getElementById("device-detail-root");
  const empty = document.getElementById("device-empty-state");
  if (!root && !empty) return;

  const params = new URLSearchParams(window.location.search);
  const sensorId = (params.get("sensorId") || params.get("id") || "").trim();

  if (!sensorId) {
    showDeviceEmptyState("No sensor was selected. Return to Sensor Health and choose a device.");
    return;
  }

  hidePageError();
  showPageLoading();

  try {
    const detail = await apiFetch(`/sensors/${encodeURIComponent(sensorId)}`);
    showDeviceDetailRoot();

    document.title = `${detail.sensorId || sensorId} – Device Detail`;
    setText("device-title", detail.name ? `${detail.sensorId} · ${detail.name}` : detail.sensorId || sensorId);

    const statusWrap = document.getElementById("device-status-wrap");
    if (statusWrap) statusWrap.innerHTML = pillHtml(detail.status, detail.batteryPct);

    const statusEl = document.getElementById("device-health-status");
    if (statusEl) statusEl.innerHTML = pillHtml(detail.status, detail.batteryPct);

    const batteryEl = document.getElementById("device-health-battery");
    if (batteryEl) batteryEl.innerHTML = formatBattery(detail.batteryPct);

    setText("device-health-last-seen", formatLastSeen(detail));
    setText("device-health-project", detail.project || "—");

    renderDevicePower(detail);
    renderDeviceHardware(detail);
    renderDeviceComponents(detail.components);
    renderConnectedDevices(detail.connectedDevices);
    renderTelemetryNote(detail);
    renderDeviceLocation(detail.gps);
    renderLastAudio(detail);
    renderAudioHistory(detail.recentAudio);

    try {
      const settingsData = await apiFetch(`/sensors/${encodeURIComponent(sensorId)}/settings`);
      applyDeviceSettingsForm(settingsData.settings || {});
    } catch (e) {
      showMessage("device-settings-message", `Failed to load settings: ${e.message}`, "danger");
    }

    await loadDeviceRebootHistory(sensorId);

    document.getElementById("device-reboot-button")?.addEventListener("click", () => {
      rebootSelectedDevice(sensorId);
    });
    document.getElementById("device-settings-button")?.addEventListener("click", () => {
      saveSelectedDeviceSettings(sensorId);
    });
  } catch (e) {
    if (e.status === 404) {
      showDeviceEmptyState(`Sensor '${sensorId}' was not found.`);
    } else {
      showDeviceEmptyState(`Could not load sensor details: ${e.message}`);
      showPageError(`Could not load sensor details: ${e.message}`);
    }
  } finally {
    hidePageLoading();
  }
}

window.fakeReboot = rebootSensors;
window.fakeSaveSettings = saveSettings;
window.fakeCreateProject = fakeCreateProject;

function bootSensorPages() {
  loadSensorHealthPage();
  loadAlertsPage();
  loadRecentRebootHistory();
  loadSettingsPage();
  loadDeviceDetailPage();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootSensorPages);
} else {
  bootSensorPages();
}