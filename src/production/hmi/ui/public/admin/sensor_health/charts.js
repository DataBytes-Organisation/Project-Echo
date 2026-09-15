/* Charts helper for sensor health (moved into /admin/sensor_health/) */

// same shared client as script.js on these pages - this was the last raw fetch
// left on the sensor health screens. Silent retry: the chart is decorative and
// already falls back to zeroes, so it shouldn't stack toasts on top of the ones
// loadAlertsPage() is already showing for the same endpoint.
import { retrieveSensorAlerts } from "/js/routes.js";

document.addEventListener("DOMContentLoaded", () => {
  drawAlertsChart();
});

async function drawAlertsChart() {
  const canvas = document.getElementById("alertsChart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const labels = ["Critical", "High", "Medium", "Low"];
  let values = [0, 0, 0, 0];

  try {
    const response = await retrieveSensorAlerts({ silent: true });
    const data = response.data;
    const items = Array.isArray(data.items) ? data.items : [];

async function drawAlertsChart() {
  const root = document.getElementById("alertsChart");
  if (!root) return;

  root.innerHTML = '<p class="card-subtitle">Loading alert summary…</p>';

  try {
    const data = await apiFetch("/sensors/alerts", { timeoutMs: 4000 });
    renderAlertsChart(Array.isArray(data.items) ? data.items : []);
  } catch (e) {
    console.warn("Alerts chart fallback in use:", e);
    renderAlertsChart([]);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  drawAlertsChart();
});

window.renderAlertsChart = renderAlertsChart;
