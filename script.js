/* ============================================================
   PROJECT ECHO – SCRIPT.JS
   Handles:
   - Mobile sidebar toggle
   - Theme toggle (light/dark)
   - Fake actions (Reboot, Settings, Project creation)
   - UI message helpers
   ============================================================ */


/* ------------------------------------------------------------
   1) MOBILE SIDEBAR TOGGLE
   ------------------------------------------------------------ */

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



/* ------------------------------------------------------------
   2) THEME TOGGLE (LIGHT ↔ DARK)
   ------------------------------------------------------------ */

const themeToggle = document.getElementById("theme-toggle");

function setTheme(darkMode) {
  if (darkMode) {
    document.documentElement.setAttribute("data-theme", "dark");
    themeToggle.textContent = "☀️";
    localStorage.setItem("echo-theme", "dark");
  } else {
    document.documentElement.removeAttribute("data-theme");
    themeToggle.textContent = "🌙";
    localStorage.setItem("echo-theme", "light");
  }
}

if (themeToggle) {
  // Load saved theme
  const savedTheme = localStorage.getItem("echo-theme");
  setTheme(savedTheme === "dark");

  // Toggle on click
  themeToggle.addEventListener("click", () => {
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    setTheme(!isDark);
  });
}



/* ------------------------------------------------------------
   3) UI MESSAGE HELPER
   ------------------------------------------------------------ */

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

  // Fade animation
  el.style.opacity = 0;
  setTimeout(() => {
    el.style.opacity = 1;
    el.style.transition = "opacity .4s";
  }, 30);
}



/* ------------------------------------------------------------
   4) FAKE REBOOT ACTION
   ------------------------------------------------------------ */

function fakeReboot() {
  const sensors = document.getElementById("reboot-sensor")?.value.trim();
  const reason = document.getElementById("reboot-reason")?.value.trim();

  if (!sensors) {
    showMessage("reboot-message", "Please enter at least one sensor ID.", "warning");
    return;
  }

  showMessage(
    "reboot-message",
    `Reboot queued for ${sensors}. Reason: ${reason || "No reason provided"}.`
  );
}



/* ------------------------------------------------------------
   5) FAKE SAVE SETTINGS (Settings Page)
   ------------------------------------------------------------ */

function fakeSaveSettings() {
  const interval = document.getElementById("record-interval")?.value;
  const sensitivity = document.getElementById("sensitivity")?.value;
  const battery = document.getElementById("battery-threshold")?.value;

  showMessage(
    "settings-message",
    `Settings saved: ${interval}, ${sensitivity} sensitivity, battery alert at ${battery}%.`
  );
}



/* ------------------------------------------------------------
   6) FAKE CREATE PROJECT
   ------------------------------------------------------------ */

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



/* ------------------------------------------------------------
   7) GENERAL UI ENHANCEMENTS
   ------------------------------------------------------------ */

// Smooth fade-in for all cards
document.querySelectorAll(".card").forEach((card) => {
  card.style.opacity = 0;

  setTimeout(() => {
    card.style.opacity = 1;
    card.style.transition = "opacity .45s ease";
  }, 120);
});

// Smooth fade for dashboard-shell
const shell = document.querySelector(".dashboard-shell");
if (shell) {
  shell.style.opacity = 0;

  setTimeout(() => {
    shell.style.opacity = 1;
    shell.style.transition = "opacity .5s ease";
  }, 80);
}
