const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatTimestamp(timestamp) {
  const date = new Date(timestamp);
  const day = String(date.getUTCDate()).padStart(2, "0");
  const month = MONTHS[date.getUTCMonth()];
  const year = date.getUTCFullYear();
  const hours = String(date.getUTCHours()).padStart(2, "0");
  const minutes = String(date.getUTCMinutes()).padStart(2, "0");
  return `${day} ${month} ${year}, ${hours}:${minutes} UTC`;
}

function formatConfidence(confidence) {
  return `${Number.isInteger(confidence) ? confidence : confidence.toFixed(1)}%`;
}

function renderQueueState(state) {
  if (state.status === "loading") {
    return `
      <div class="queue-state queue-state--loading" role="status" aria-live="polite">
        <p class="queue-state__title">Loading detection queue…</p>
        <div class="skeleton-stack" aria-hidden="true">
          <span class="skeleton-row"></span>
          <span class="skeleton-row"></span>
          <span class="skeleton-row"></span>
        </div>
      </div>`;
  }

  if (state.status === "empty") {
    return `
      <div class="queue-state" role="status">
        <p class="queue-state__title">No detections waiting for review</p>
        <p>This deterministic fixture set contains no queued records.</p>
      </div>`;
  }

  if (state.status === "failed") {
    return `
      <div class="queue-state queue-state--error" role="alert">
        <p class="queue-state__title">Detection queue unavailable</p>
        <p>${escapeHtml(state.errorMessage)}</p>
        <a class="text-link" href="?scenario=populated">Open valid fixtures</a>
      </div>`;
  }

  const items = state.records.map(record => {
    const isSelected = record.id === state.selectedId;
    const confidence = formatConfidence(record.confidence);
    const timestamp = formatTimestamp(record.timestamp);

    return `
      <li class="queue-list__item">
        <button
          type="button"
          class="queue-item"
          data-detection-id="${escapeHtml(record.id)}"
          aria-pressed="${isSelected}"
          aria-describedby="queue-keyboard-help"
          aria-keyshortcuts="ArrowUp ArrowDown Home End"
        >
          <span class="queue-item__topline">
            <strong>${escapeHtml(record.species)}</strong>
            <span class="confidence">${escapeHtml(confidence)}</span>
          </span>
          <span class="queue-item__meta">
            <span>${escapeHtml(record.sensorId)}</span>
            <time datetime="${escapeHtml(record.timestamp)}">${escapeHtml(timestamp)}</time>
          </span>
          <span class="queue-status">Pending review</span>
        </button>
      </li>`;
  }).join("");

  return `<ol class="queue-list">${items}</ol>`;
}

export function renderQueue(state) {
  const count = state.records.length;
  const countLabel = state.status === "loading"
    ? "Loading"
    : state.status === "failed"
      ? "Unavailable"
      : count === 1
        ? "1 record"
        : `${count} records`;
  const countAriaLabel = state.status === "loading"
    ? "Record count loading"
    : state.status === "failed"
      ? "Record count unavailable"
      : countLabel;

  return `
    <section class="panel queue-panel" aria-labelledby="queue-heading">
      <header class="panel-heading">
        <div>
          <h2 id="queue-heading">Detection queue</h2>
          <p>Records waiting for an ecological review.</p>
        </div>
        <span class="record-count" aria-label="${escapeHtml(countAriaLabel)}">${escapeHtml(countLabel)}</span>
      </header>
      <p class="sr-only" id="queue-keyboard-help">Use the Up and Down arrow keys, Home, or End to move through detection records.</p>
      ${renderQueueState(state)}
    </section>`;
}

function evidenceRow(label, value, modifier = "") {
  const className = modifier ? `evidence-list__value ${modifier}` : "evidence-list__value";
  return `
    <div class="evidence-list__row">
      <dt>${escapeHtml(label)}</dt>
      <dd class="${className}">${value}</dd>
    </div>`;
}

function evidenceEmptyCopy(stateStatus) {
  if (stateStatus === "loading") {
    return {
      title: "Evidence will appear after loading",
      description: "Waiting for validated detection records.",
    };
  }

  if (stateStatus === "empty") {
    return {
      title: "No evidence to inspect",
      description: "There are no queued detections in this fixture set.",
    };
  }

  if (stateStatus === "failed") {
    return {
      title: "Evidence unavailable",
      description: "Evidence cannot be shown until the fixture data is valid.",
    };
  }

  return {
    title: "Select a detection",
    description: "Choose a queue record to inspect its available evidence.",
  };
}

export function renderEvidence(record, stateStatus) {
  if (!record) {
    const copy = evidenceEmptyCopy(stateStatus);

    return `
      <section class="panel evidence-panel" aria-labelledby="evidence-heading">
        <header class="panel-heading">
          <div>
            <h2 id="evidence-heading">Evidence record</h2>
            <p>Model output and field context.</p>
          </div>
        </header>
        <div class="evidence-empty" role="status">
          <p class="queue-state__title">${copy.title}</p>
          <p>${copy.description}</p>
        </div>
      </section>`;
  }

  const [latitude, longitude, altitude] = record.animalEstLLA;
  const audioLabel = record.audioAvailable
    ? "Audio evidence available"
    : "No audio evidence attached";
  const audioModifier = record.audioAvailable
    ? "evidence-list__value--available"
    : "evidence-list__value--unavailable";

  return `
    <section class="panel evidence-panel" aria-labelledby="evidence-heading">
      <header class="panel-heading panel-heading--evidence">
        <div>
          <h2 id="evidence-heading">Evidence record</h2>
          <p>Model output and field context.</p>
        </div>
        <span class="record-id">${escapeHtml(record.id)}</span>
      </header>
      <div class="prediction-summary">
        <p>Species prediction</p>
        <strong>${escapeHtml(record.species)}</strong>
        <span>${escapeHtml(formatConfidence(record.confidence))} model confidence</span>
      </div>
      <dl class="evidence-list">
        ${evidenceRow("Predicted species", escapeHtml(record.species))}
        ${evidenceRow("Confidence", escapeHtml(formatConfidence(record.confidence)), "tabular")}
        ${evidenceRow("Timestamp", `<time datetime="${escapeHtml(record.timestamp)}">${escapeHtml(formatTimestamp(record.timestamp))}</time>`, "tabular")}
        ${evidenceRow("Sensor ID", `Sensor ${escapeHtml(record.sensorId)}`)}
        ${evidenceRow("Estimated location", `
          <span class="coordinate"><small>Lat</small> ${latitude.toFixed(4)}</span>
          <span class="coordinate"><small>Lon</small> ${longitude.toFixed(4)}</span>
          <span class="coordinate"><small>Alt</small> ${altitude} m</span>
        `, "location-value tabular")}
        ${evidenceRow("Location uncertainty", `± ${escapeHtml(record.animalLLAUncertainty)} m`, "tabular")}
        ${evidenceRow("Audio evidence", escapeHtml(audioLabel), audioModifier)}
      </dl>
    </section>`;
}

export function renderWorkbench(state, {
  workflowHtml = "",
  adjudicationQueueHtml = "",
} = {}) {
  const selectedRecord = state.status === "populated"
    ? state.records.find(record => record.id === state.selectedId) ?? null
    : null;

  return `
    ${adjudicationQueueHtml}
    <div class="workbench" data-queue-state="${escapeHtml(state.status)}">
      ${renderQueue(state)}
      ${renderEvidence(selectedRecord, state.status)}
      ${workflowHtml}
    </div>`;
}
