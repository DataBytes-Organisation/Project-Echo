// HTML rendering for the admin Detection Review page.
//
// Kept separate from detection-review.js (which does the fetching and DOM
// wiring) so it has no browser dependencies and can be unit tested in Node.
//
// Species names, ids and timestamps all come from stored detections, and the
// Backend accepts and returns markup in those fields, so every dynamic value
// is HTML-escaped here before it is placed into a template. Nothing in this
// file should interpolate stored data into HTML without going through
// escapeHtml.
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.DetectionReviewRender = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // A result can be in the top-k just because k results are always returned,
  // even when nothing in the gallery is genuinely close (small gallery, or the
  // query itself sits in an uncertain part of the embedding space). Below this
  // score, treat the row as "weak" rather than a real precedent, so the admin
  // does not read it as equally trustworthy as a strong match.
  const WEAK_MATCH_THRESHOLD = 0.6;

  function escapeHtml(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function finiteNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : 0;
  }

  function detectionListItemHtml(detection) {
    const when = new Date(detection.timestamp).toLocaleString();
    return `
      <div class="card" data-id="${escapeHtml(detection._id)}">
        <div class="card-body p-2">
          <div class="fw-semibold">${escapeHtml(detection.species)}</div>
          <div class="text-muted small">${escapeHtml(when)} - ${escapeHtml(finiteNumber(detection.confidence).toFixed(1))}%</div>
        </div>
      </div>
    `;
  }

  function similarRowHtml({ species, similarity, isMatchOfTop, audioUrl }) {
    const score = finiteNumber(similarity);
    const pct = Math.max(0, Math.min(1, score)) * 100;
    const rowClass = isMatchOfTop ? "dr-match" : "dr-diff";
    const isWeak = score < WEAK_MATCH_THRESHOLD;
    const weakClass = isWeak ? "dr-row-weak" : "";
    const weakLabel = isWeak ? '<span class="dr-weak-label">weak, not a strong precedent</span>' : "";
    return `
      <div class="d-flex align-items-center gap-3 py-2 border-bottom ${weakClass}">
        <audio controls src="${escapeHtml(audioUrl)}" style="height:32px;width:180px;"></audio>
        <div style="flex:1;">
          <div class="${rowClass}">${escapeHtml(species)}${weakLabel}</div>
          <div class="dr-similarity-bar"><div style="width:${pct}%"></div></div>
        </div>
        <div class="text-muted" style="width:60px;text-align:right;">${score.toFixed(3)}</div>
      </div>
    `;
  }

  function badgesHtml(similar) {
    const badges = [];
    if (similar.ambiguous) {
      badges.push('<span class="badge bg-warning text-dark">Ambiguous: multiple species nearby</span>');
    }
    if (similar.novel) {
      badges.push('<span class="badge bg-danger">Novel: nothing similar found before</span>');
    }
    if (!similar.ambiguous && !similar.novel) {
      badges.push('<span class="badge bg-success">Consistent match</span>');
    }
    return badges.join("");
  }

  // rows: [{ species, similarity, isMatchOfTop, audioUrl }]
  function detailHtml({ detection, similar, queryAudioUrl, rows }) {
    const rowsHtml = rows.length === 0
      ? '<div class="dr-empty-state">No other detections with a stored embedding to compare against yet.</div>'
      : rows.map(similarRowHtml).join("");

    return `
      <h5>Reviewing: ${escapeHtml(detection.species)}</h5>
      <div class="d-flex align-items-center gap-3 mb-2">
        <audio controls src="${escapeHtml(queryAudioUrl)}"></audio>
        <span class="text-muted">confidence ${escapeHtml(finiteNumber(detection.confidence).toFixed(1))}%</span>
      </div>
      <div class="dr-badges mb-3">${badgesHtml(similar)}</div>
      <h6>Most similar past detections</h6>
      ${rowsHtml}
    `;
  }

  return {
    WEAK_MATCH_THRESHOLD,
    escapeHtml,
    detectionListItemHtml,
    similarRowHtml,
    badgesHtml,
    detailHtml,
  };
});
