// Detection Review admin page.
//
// Lists recent detections, and for whichever one is selected, calls the
// similar-detection retrieval feature (GET /detections/{id}/similar) and
// renders the results with playable audio, so an admin can verify an
// uncertain detection against precedent instead of trusting a confidence
// number alone. Both endpoints already exist on the Backend and are proxied
// through this HMI server's own /detections/* route (see server.js), so
// these are plain relative fetches, no separate backend host/CORS needed.

const REQUEST_TIMEOUT = 10000;

// A result can be in the top-k just because k results are always returned,
// even when nothing in the gallery is genuinely close (small gallery, or the
// query itself sits in an uncertain part of the embedding space). Below this
// score, treat the row as "weak" rather than a real precedent, so the admin
// does not read it as equally trustworthy as a strong match.
const WEAK_MATCH_THRESHOLD = 0.6;

async function fetchJson(url) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(`${url} returned status ${response.status}`);
    }
    return await response.json();
  } finally {
    clearTimeout(timeoutId);
  }
}

// The detections collection doesn't store a content-type for audioClip, so
// sniff it from the file's magic bytes rather than guessing one MIME type
// for every file (the demo data mixes .mp3 and .wav).
function sniffAudioMimeType(bytes) {
  if (bytes.length >= 12 && bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46) {
    return "audio/wav"; // "RIFF..."
  }
  if (bytes.length >= 3 && bytes[0] === 0x49 && bytes[1] === 0x44 && bytes[2] === 0x33) {
    return "audio/mpeg"; // ID3 tag
  }
  if (bytes.length >= 2 && bytes[0] === 0xff && (bytes[1] & 0xe0) === 0xe0) {
    return "audio/mpeg"; // raw MPEG frame sync
  }
  return "audio/mpeg";
}

function base64ToAudioUrl(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: sniffAudioMimeType(bytes) });
  return URL.createObjectURL(blob);
}

function speciesRowHtml(species, similarity, isMatchOfTop, audioUrl) {
  const pct = Math.max(0, Math.min(1, similarity)) * 100;
  const rowClass = isMatchOfTop ? "dr-match" : "dr-diff";
  const isWeak = similarity < WEAK_MATCH_THRESHOLD;
  const weakClass = isWeak ? "dr-row-weak" : "";
  const weakLabel = isWeak ? '<span class="dr-weak-label">weak, not a strong precedent</span>' : "";
  return `
    <div class="d-flex align-items-center gap-3 py-2 border-bottom ${weakClass}">
      <audio controls src="${audioUrl}" style="height:32px;width:180px;"></audio>
      <div style="flex:1;">
        <div class="${rowClass}">${species}${weakLabel}</div>
        <div class="dr-similarity-bar"><div style="width:${pct}%"></div></div>
      </div>
      <div class="text-muted" style="width:60px;text-align:right;">${similarity.toFixed(3)}</div>
    </div>
  `;
}

async function renderDetail(detection, pageState) {
  const detailEl = document.getElementById("detection-detail");
  pageState.showLoading();
  try {
    const similar = await fetchJson(`/detections/${detection._id}/similar?k=5`);

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

    const queryAudioUrl = base64ToAudioUrl(detection.audioClip);

    let rowsHtml = "";
    if (similar.results.length === 0) {
      rowsHtml = '<div class="dr-empty-state">No other detections with a stored embedding to compare against yet.</div>';
    } else {
      // Fetch each candidate's audio so it can be played inline for a real by-ear check.
      const candidateDetails = await Promise.all(
        similar.results.map((r) => fetchJson(`/detections/${r.detection_id}`))
      );
      rowsHtml = similar.results
        .map((r, i) => {
          const audioUrl = base64ToAudioUrl(candidateDetails[i].audioClip);
          const isMatch = r.species === detection.species;
          return speciesRowHtml(r.species, r.similarity, isMatch, audioUrl);
        })
        .join("");
    }

    detailEl.innerHTML = `
      <h5>Reviewing: ${detection.species}</h5>
      <div class="d-flex align-items-center gap-3 mb-2">
        <audio controls src="${queryAudioUrl}"></audio>
        <span class="text-muted">confidence ${Number(detection.confidence).toFixed(1)}%</span>
      </div>
      <div class="dr-badges mb-3">${badges.join("")}</div>
      <h6>Most similar past detections</h6>
      ${rowsHtml}
    `;
  } catch (error) {
    pageState.showError(`Could not load similar detections: ${error.message}`);
  } finally {
    pageState.hideLoading();
  }
}

async function init() {
  const pageState = window.createAdminPageState();
  const listEl = document.getElementById("detection-list");

  pageState.showLoading();
  try {
    const page = await fetchJson("/detections?page=1&page_size=20");
    const items = page.items || [];

    if (items.length === 0) {
      listEl.innerHTML = '<div class="dr-empty-state">No detections yet.</div>';
      return;
    }

    listEl.innerHTML = items
      .map(
        (d) => `
          <div class="card" data-id="${d._id}">
            <div class="card-body p-2">
              <div class="fw-semibold">${d.species}</div>
              <div class="text-muted small">${new Date(d.timestamp).toLocaleString()} - ${Number(d.confidence).toFixed(1)}%</div>
            </div>
          </div>
        `
      )
      .join("");

    listEl.querySelectorAll(".card").forEach((card) => {
      card.addEventListener("click", () => {
        listEl.querySelectorAll(".card").forEach((c) => c.classList.remove("active"));
        card.classList.add("active");
        const detection = items.find((d) => d._id === card.dataset.id);
        renderDetail(detection, pageState);
      });
    });
  } catch (error) {
    pageState.showError(`Could not load detections: ${error.message}`);
  } finally {
    pageState.hideLoading();
  }
}

document.addEventListener("DOMContentLoaded", init);
