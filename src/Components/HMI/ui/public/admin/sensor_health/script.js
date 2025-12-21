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

function fakeReboot() {
  const sensors = document.getElementById("reboot-sensor")?.value.trim();
  const reason = document.getElementById("reboot-reason")?.value.trim();
  if (!sensors) { showMessage("reboot-message", "Please enter at least one sensor ID.", "warning"); return; }
  showMessage("reboot-message", `Reboot queued for ${sensors}. Reason: ${reason || "No reason provided"}.`);
}

function fakeSaveSettings() {
  const interval = document.getElementById("record-interval")?.value;
  const sensitivity = document.getElementById("sensitivity")?.value;
  const battery = document.getElementById("battery-threshold")?.value;
  showMessage("settings-message", `Settings saved: ${interval}, ${sensitivity} sensitivity, battery alert at ${battery}%.`);
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
