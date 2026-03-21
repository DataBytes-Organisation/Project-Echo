/* Project Echo — moved sensor health script.js (copied)
   Handles sidebar toggle, theme, fake actions and UI helpers
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

function showMessage(elementId, message, type = "success") {
  const el = document.getElementById(elementId);
  if (!el) return;
  const colors = { success: "var(--primary)", warning: "var(--warning)", danger: "var(--danger)" };
  el.style.color = colors[type] || colors.success;
  el.style.marginTop = "10px";
  el.style.fontWeight = "600";
  el.textContent = message;
  el.style.opacity = 0;
  setTimeout(() => { el.style.opacity = 1; el.style.transition = "opacity .4s"; }, 30);
}

async function apiFetch(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });

  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw new Error(text || `Request failed: ${response.status}`);
  }

  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return response.json();
  }
  return response.text();
}

function formatMinutesAgo(mins) {
  if (mins === null || typeof mins === 'undefined') return '—';
  const m = Number(mins);
  if (!Number.isFinite(m)) return '—';
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  const r = m % 60;
  return r ? `${h}h ${r}m ago` : `${h}h ago`;
}

function pillHtml(status) {
  const s = String(status || '').trim();
  let cls = 'pill-warning';
  if (s === 'Online') cls = 'pill-success';
  if (s === 'Offline') cls = 'pill-danger';
  return `<span class="pill ${cls}">${s || 'Unknown'}</span>`;
}

async function rebootSensors() {
  const sensorsRaw = document.getElementById("reboot-sensor")?.value.trim();
  const reason = document.getElementById("reboot-reason")?.value.trim();
  if (!sensorsRaw) {
    showMessage("reboot-message", "Please enter at least one sensor ID.", "warning");
    return;
  }

  const sensors = sensorsRaw
    .split(',')
    .map(s => s.trim())
    .filter(Boolean);

  try {
    const results = [];
    for (const sensorId of sensors) {
      const res = await apiFetch(`/sensors/${encodeURIComponent(sensorId)}/reboot`, {
        method: 'POST',
        body: JSON.stringify({ reason: reason || null }),
      });
      results.push(res);
    }
    showMessage("reboot-message", `Reboot queued for ${sensors.join(', ')}.`, "success");
    await loadRecentRebootHistory();
  } catch (e) {
    showMessage("reboot-message", `Failed to queue reboot: ${e.message}`, "danger");
  }
}

function intervalLabelToSeconds(label) {
  const text = String(label || '').toLowerCase();
  if (text.includes('30')) return 30;
  if (text.includes('10')) return 600;
  if (text.includes('5')) return 300;
  return 60;
}

function secondsToIntervalLabel(seconds) {
  const s = Number(seconds);
  if (s === 30) return '30 seconds';
  if (s === 300) return '5 minutes';
  if (s === 600) return '10 minutes';
  return '1 minute';
}

async function saveSettings() {
  const intervalLabel = document.getElementById("record-interval")?.value;
  const sensitivity = document.getElementById("sensitivity")?.value;
  const battery = Number(document.getElementById("battery-threshold")?.value);

  try {
    const payload = {
      recordIntervalSeconds: intervalLabelToSeconds(intervalLabel),
      sensitivity: sensitivity || 'Medium',
      batteryThresholdPct: Number.isFinite(battery) ? battery : 25,
    };

    await apiFetch(`/sensors/__default__/settings`, {
      method: 'PUT',
      body: JSON.stringify({ settings: payload }),
    });

    showMessage("settings-message", `Settings saved.`, "success");
  } catch (e) {
    showMessage("settings-message", `Failed to save settings: ${e.message}`, "danger");
  }
}

function fakeCreateProject() {
  const name = document.getElementById("project-name")?.value.trim();
  const loc = document.getElementById("project-location")?.value.trim();
  const sensors = document.getElementById("project-sensors")?.value.trim();
  if (!name || !loc) { showMessage("project-message", "Project name and location are required.", "warning"); return; }
  showMessage("project-message", `Project "${name}" created successfully. Assigned sensors: ${sensors || "None"}.`);
}

document.querySelectorAll(".card").forEach((card) => { card.style.opacity = 0; setTimeout(() => { card.style.opacity = 1; card.style.transition = "opacity .45s ease"; }, 120); });
const shell = document.querySelector(".dashboard-shell"); if (shell) { shell.style.opacity = 0; setTimeout(() => { shell.style.opacity = 1; shell.style.transition = "opacity .5s ease"; }, 80); }

// Ensure sidebar nav always contains expected links (guard against unexpected DOM changes)
function ensureSensorSidebar() {
  const sidebar = document.querySelector('.sidebar .nav-section');
  if (!sidebar) return;
  const required = [
    {href: '/admin/dashboard.html', icon: '▦', text: 'Dashboard'},
    {href: '/admin/sensor-health.html', icon: '📡', text: 'Sensor Health'},
    {href: '/admin/sensor_health/alerts.html', icon: '🔔', text: 'Alerts'},
    {href: '/admin/sensor_health/reboot.html', icon: '⟲', text: 'Reboot'},
    {href: '/admin/sensor_health/settings.html', icon: '⚙', text: 'Settings'},
    {href: '/admin/sensor_health/add-project.html', icon: '➕', text: 'Add a Project'}
  ];

  required.forEach(item => {
    if (!sidebar.querySelector(`a[href="${item.href}"]`)) {
      const a = document.createElement('a');
      a.className = 'nav-link';
      a.href = item.href;
      a.innerHTML = `<span class="nav-icon">${item.icon}</span><span class="nav-text">${item.text}</span>`;
      sidebar.appendChild(a);
    }
  });
}

document.addEventListener('DOMContentLoaded', function() {
  ensureSensorSidebar();
  const navSection = document.querySelector('.sidebar .nav-section');
  if (navSection) {
    const observer = new MutationObserver(() => ensureSensorSidebar());
    observer.observe(navSection, { childList: true });
  }
});

// ---- Page-specific data wiring ----
async function loadSensorHealthPage() {
  const tbody = document.getElementById('sensor-overview-tbody');
  if (!tbody) return;

  const statusFilter = document.getElementById('sensor-status-filter');
  const projectFilter = document.getElementById('sensor-project-filter');

  let lastItems = [];

  function render() {
    const statusVal = statusFilter?.value || 'All';
    const projectVal = projectFilter?.value || 'All Projects';

    const filtered = lastItems.filter(item => {
      if (statusVal !== 'All' && item.status !== statusVal) return false;
      if (projectVal !== 'All Projects' && (item.project || '—') !== projectVal) return false;
      return true;
    });

    tbody.innerHTML = '';
    if (!filtered.length) {
      tbody.innerHTML = '<tr><td colspan="5">No sensors match the current filter.</td></tr>';
      return;
    }

    for (const item of filtered) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${item.sensorId}</td>
        <td>${item.project || '—'}</td>
        <td>${pillHtml(item.status)}</td>
        <td>${typeof item.batteryPct === 'number' ? `${item.batteryPct}%` : '—'}</td>
        <td>${formatMinutesAgo(item.lastSeenMinutesAgo)}</td>
      `;
      tbody.appendChild(tr);
    }
  }

  async function refresh() {
    try {
      const data = await apiFetch('/sensors/updates');
      lastItems = Array.isArray(data.items) ? data.items : [];

      // Populate project filter options dynamically
      if (projectFilter) {
        const projects = Array.from(new Set(lastItems.map(i => i.project).filter(Boolean))).sort();
        const current = projectFilter.value;
        projectFilter.innerHTML = '<option>All Projects</option>' + projects.map(p => `<option>${p}</option>`).join('');
        if ([...projectFilter.options].some(o => o.value === current)) {
          projectFilter.value = current;
        }
      }

      render();
    } catch (e) {
      tbody.innerHTML = `<tr><td colspan="5">Failed to load sensors: ${e.message}</td></tr>`;
    }
  }

  statusFilter?.addEventListener('change', render);
  projectFilter?.addEventListener('change', render);

  await refresh();
  setInterval(refresh, 15000);
}

async function loadAlertsPage() {
  const tbody = document.getElementById('alerts-tbody');
  if (!tbody) return;
  try {
    const data = await apiFetch('/sensors/alerts');
    const items = Array.isArray(data.items) ? data.items : [];
    tbody.innerHTML = '';
    if (!items.length) {
      tbody.innerHTML = '<tr><td colspan="4">No active alerts.</td></tr>';
      return;
    }
    for (const alert of items) {
      const tr = document.createElement('tr');
      const sevPill = pillHtml(alert.issue);
      tr.innerHTML = `
        <td>${alert.sensorId}</td>
        <td>${sevPill}</td>
        <td>${alert.details || ''}</td>
        <td>${formatMinutesAgo(alert.lastAudioMinutesAgo)}</td>
      `;
      tbody.appendChild(tr);
    }
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="4">Failed to load alerts: ${e.message}</td></tr>`;
  }
}

async function loadRecentRebootHistory() {
  const tbody = document.getElementById('reboot-history-tbody');
  if (!tbody) return;
  try {
    const data = await apiFetch('/sensors/reboots/recent?limit=50');
    const items = Array.isArray(data.items) ? data.items : [];
    tbody.innerHTML = '';
    if (!items.length) {
      tbody.innerHTML = '<tr><td colspan="3">No reboot history yet.</td></tr>';
      return;
    }
    for (const r of items) {
      const t = r.requestedAt ? new Date(r.requestedAt).toLocaleString() : '—';
      const status = String(r.status || 'Queued');
      const cls = status.toLowerCase().includes('fail') ? 'pill-danger' : (status.toLowerCase().includes('success') ? 'pill-success' : 'pill-warning');
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${r.sensorId}</td>
        <td><span class="pill ${cls}">${status}</span></td>
        <td>${t}</td>
      `;
      tbody.appendChild(tr);
    }
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="3">Failed to load reboot history: ${e.message}</td></tr>`;
  }
}

async function loadSettingsPage() {
  const interval = document.getElementById('record-interval');
  const sensitivity = document.getElementById('sensitivity');
  const battery = document.getElementById('battery-threshold');
  if (!interval || !sensitivity || !battery) return;

  try {
    const data = await apiFetch('/sensors/__default__/settings');
    const settings = data.settings || {};
    const intervalLabel = secondsToIntervalLabel(settings.recordIntervalSeconds);

    // Set values if present
    interval.value = intervalLabel;
    sensitivity.value = settings.sensitivity || 'Medium';
    battery.value = Number(settings.batteryThresholdPct || 25);
  } catch (e) {
    showMessage('settings-message', `Failed to load settings: ${e.message}`, 'danger');
  }
}

// Keep existing onclick bindings working
window.fakeReboot = rebootSensors;
window.fakeSaveSettings = saveSettings;

document.addEventListener('DOMContentLoaded', () => {
  loadSensorHealthPage();
  loadAlertsPage();
  loadRecentRebootHistory();
  loadSettingsPage();
});
