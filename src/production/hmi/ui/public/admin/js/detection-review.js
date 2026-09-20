// Detection Review admin page.
//
// Lists recent detections, and for whichever one is selected, calls the
// similar-detection retrieval feature and renders the results with playable
// audio, so an admin can verify an uncertain detection against precedent
// instead of trusting a confidence number alone.
//
// All reads go through the HMI's own session-protected proxy
// (routes/detection-review.routes.js), not straight to the Backend, so they
// are plain same-origin fetches. HTML is built only by detection-review-render.js,
// which escapes every stored value; do not interpolate detection fields into
// innerHTML here.

const API_BASE = "/api/detection-review";
const REQUEST_TIMEOUT = 10000;

async function fetchJson(url) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      let message = `${url} returned status ${response.status}`;
      try {
        const payload = await response.json();
        if (payload && payload.error && payload.error.message) message = payload.error.message;
      } catch (_) {
        // Body was not JSON; keep the status-based message.
      }
      throw new Error(message);
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

let activeObjectUrls = [];

function base64ToAudioUrl(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: sniffAudioMimeType(bytes) });
  const url = URL.createObjectURL(blob);
  activeObjectUrls.push(url);
  return url;
}

function releaseObjectUrls() {
  activeObjectUrls.forEach((url) => URL.revokeObjectURL(url));
  activeObjectUrls = [];
}

let renderSequence = 0;

async function renderDetail(detection, pageState) {
  const render = window.DetectionReviewRender;
  const detailEl = document.getElementById("detection-detail");
  // If the admin clicks another detection while this one is still loading,
  // the older response must not overwrite the newer selection.
  const mySequence = (renderSequence += 1);
  pageState.showLoading();
  try {
    const similar = await fetchJson(`${API_BASE}/detections/${encodeURIComponent(detection._id)}/similar?k=5`);

    // Fetch each candidate's audio so it can be played inline for a real by-ear check.
    const candidateDetails = await Promise.all(
      similar.results.map((r) => fetchJson(`${API_BASE}/detections/${encodeURIComponent(r.detection_id)}`))
    );
    if (mySequence !== renderSequence) return;

    releaseObjectUrls();
    const queryAudioUrl = base64ToAudioUrl(detection.audioClip);
    const rows = similar.results.map((r, i) => ({
      species: r.species,
      similarity: r.similarity,
      isMatchOfTop: r.species === detection.species,
      audioUrl: base64ToAudioUrl(candidateDetails[i].audioClip),
    }));

    detailEl.innerHTML = render.detailHtml({ detection, similar, queryAudioUrl, rows });
  } catch (error) {
    if (mySequence !== renderSequence) return;
    pageState.showError(`Could not load similar detections: ${error.message}`);
  } finally {
    if (mySequence === renderSequence) pageState.hideLoading();
  }
}

async function init() {
  const render = window.DetectionReviewRender;
  const pageState = window.createAdminPageState();
  const listEl = document.getElementById("detection-list");

  pageState.showLoading();
  try {
    const page = await fetchJson(`${API_BASE}/detections?page=1&page_size=20`);
    const items = page.items || [];

    if (items.length === 0) {
      listEl.innerHTML = '<div class="dr-empty-state">No detections yet.</div>';
      return;
    }

    listEl.innerHTML = items.map(render.detectionListItemHtml).join("");

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
