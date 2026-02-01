/* Charts helper for sensor health (moved into /admin/sensor_health/) */
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
    const res = await fetch('/sensors/alerts');
    const data = await res.json();
    const items = Array.isArray(data.items) ? data.items : [];
    const counts = { Critical: 0, High: 0, Medium: 0, Low: 0 };
    for (const a of items) {
      const sev = String(a.severity || 'Medium');
      if (counts[sev] !== undefined) counts[sev] += 1;
    }
    values = [counts.Critical, counts.High, counts.Medium, counts.Low];
  } catch (e) {
    // fall back to zeros
  }

  const colors = ["#C8473C","#D29B38","#F59E0B","#2F6E4F"];
  const width = canvas.width; const height = canvas.height; const padding = 30; const bottomPadding = 40; const topPadding = 20;
  const maxValue = Math.max(...values, 1); const chartHeight = height - bottomPadding - topPadding; const scaleY = chartHeight / maxValue;
  const barWidth = 32; const gap = 26;
  ctx.clearRect(0,0,width,height);
  ctx.strokeStyle = "rgba(148, 163, 184, 0.25)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(padding, height - bottomPadding); ctx.lineTo(width - padding, height - bottomPadding); ctx.stroke();
  ctx.font = "12px Inter, system-ui, sans-serif"; ctx.textAlign = "center";
  values.forEach((value,i) => { const x = padding + i * (barWidth + gap) + barWidth/2; const barHeight = value * scaleY; const y = height - bottomPadding - barHeight; ctx.fillStyle = colors[i]; ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x - barWidth/2, y, barWidth, barHeight, 6) : ctx.fillRect(x - barWidth/2, y, barWidth, barHeight); ctx.fill(); ctx.fillStyle = "#111827"; ctx.fillText(value, x, y - 6); ctx.fillStyle = "#6F7B74"; ctx.fillText(labels[i], x, height - bottomPadding + 18); });
}
if (!CanvasRenderingContext2D.prototype.roundRect) { CanvasRenderingContext2D.prototype.roundRect = function (x,y,w,h,r){ r = Math.min(r,w/2,h/2); this.beginPath(); this.moveTo(x + r, y); this.arcTo(x + w, y, x + w, y + h, r); this.arcTo(x + w, y + h, x, y + h, r); this.arcTo(x, y + h, x, y, r); this.arcTo(x, y, x + w, y, r); this.closePath(); return this; }; }
