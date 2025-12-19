/* ============================================================
   PROJECT ECHO – CHARTS.JS
   Simple canvas-based charts (no external libraries)
   Currently renders:
   - Alerts severity bar chart  (alerts.html → <canvas id="alertsChart">)
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  drawAlertsChart();
});

/**
 * Draw a simple bar chart for alert severity.
 * Canvas: <canvas id="alertsChart">
 */
function drawAlertsChart() {
  const canvas = document.getElementById("alertsChart");
  if (!canvas) return; // not on this page

  const ctx = canvas.getContext("2d");

  // Data (you can adjust these values to match your narrative)
  const labels = ["Critical", "High", "Medium", "Low"];
  const values = [3, 5, 7, 2];

  // Colours aligned to your UI palette
  const colors = [
    "#C8473C", // Critical (danger)
    "#D29B38", // High (warning)
    "#F59E0B", // Medium (amber)
    "#2F6E4F"  // Low (green)
  ];

  // Canvas dimensions
  const width = canvas.width;
  const height = canvas.height;

  // Chart area padding
  const padding = 30;
  const bottomPadding = 40;
  const topPadding = 20;

  // Compute scaling
  const maxValue = Math.max(...values);
  const chartHeight = height - bottomPadding - topPadding;
  const scaleY = chartHeight / maxValue;

  const barWidth = 32;
  const gap = 26;

  // Clear canvas
  ctx.clearRect(0, 0, width, height);

  // Background (optional light grid line)
  ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
  ctx.lineWidth = 1;

  // Horizontal baseline
  ctx.beginPath();
  ctx.moveTo(padding, height - bottomPadding);
  ctx.lineTo(width - padding, height - bottomPadding);
  ctx.stroke();

  // Draw bars
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";

  values.forEach((value, i) => {
    const x = padding + i * (barWidth + gap) + barWidth / 2;
    const barHeight = value * scaleY;
    const y = height - bottomPadding - barHeight;

    // Bar
    ctx.fillStyle = colors[i];
    ctx.beginPath();
    ctx.roundRect(x - barWidth / 2, y, barWidth, barHeight, 6);
    ctx.fill();

    // Value label on top
    ctx.fillStyle = "#111827";
    ctx.fillText(value, x, y - 6);

    // X-axis label
    ctx.fillStyle = "#6F7B74";
    ctx.fillText(labels[i], x, height - bottomPadding + 18);
  });
}

/**
 * Polyfill for roundRect on older browsers.
 * (Most modern browsers support it, but this keeps it safe.)
 */
if (!CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    this.beginPath();
    this.moveTo(x + r, y);
    this.arcTo(x + w, y, x + w, y + h, r);
    this.arcTo(x + w, y + h, x, y + h, r);
    this.arcTo(x, y + h, x, y, r);
    this.arcTo(x, y, x + w, y, r);
    this.closePath();
    return this;
  };
}
