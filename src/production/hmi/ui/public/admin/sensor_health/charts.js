/* Alert severity bars for sensor health */

function renderAlertsChart(items) {
  const root = document.getElementById("alertsChart");
  if (!root) return;

  const counts = { Critical: 0, High: 0, Medium: 0, Low: 0 };
  for (const alert of Array.isArray(items) ? items : []) {
    const issue = String(alert.issue || "").toLowerCase();
    const sev = String(alert.severity || "");
    if (counts[sev] !== undefined) {
      counts[sev] += 1;
    } else if (issue.includes("offline")) {
      counts.Critical += 1;
    } else if (issue.includes("battery")) {
      counts.High += 1;
    } else {
      counts.Medium += 1;
    }
  }

  const rows = [
    { label: "Critical", value: counts.Critical, color: "#C8473C" },
    { label: "High", value: counts.High, color: "#D29B38" },
    { label: "Medium", value: counts.Medium, color: "#F59E0B" },
    { label: "Low", value: counts.Low, color: "#2F6E4F" },
  ];

  const total = rows.reduce((sum, row) => sum + row.value, 0);
  if (total === 0) {
    root.innerHTML = '<p class="card-subtitle">No active alerts.</p>';
    return;
  }

  const maxValue = Math.max(...rows.map((row) => row.value), 1);

  root.innerHTML = rows
    .map((row) => {
      const width = Math.max(6, Math.round((row.value / maxValue) * 100));
      return `
        <div class="alert-bar-row">
          <span class="alert-bar-label">${row.label}</span>
          <div class="alert-bar-track">
            <div class="alert-bar-fill" style="width:${width}%; background:${row.color};"></div>
          </div>
          <span class="alert-bar-value">${row.value}</span>
        </div>
      `;
    })
    .join("");
}

async function drawAlertsChart() {
  const root = document.getElementById("alertsChart");
  if (!root) return;

  root.innerHTML = '<p class="card-subtitle">Loading alert summary…</p>';

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 4000);
    const res = await fetch("/sensors/alerts", { signal: controller.signal });
    clearTimeout(timer);

    if (!res.ok) throw new Error(`Failed to load alerts: ${res.status}`);
    const data = await res.json();
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
