"use strict";

// ─────────────────────────────────────────────────────────────────────────────
// Audio test helper  (Sprint 1 — unchanged)
// ─────────────────────────────────────────────────────────────────────────────

export function getAudioTestString() {
  const sampleRate = 8000;
  const frequency = 440;
  const duration = 1;
  const numSamples = sampleRate * duration;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + numSamples * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, numSamples * 2, true);

  for (let i = 0; i < numSamples; i++) {
    const sample = Math.sin((2 * Math.PI * frequency * i) / sampleRate);
    view.setInt16(44 + i * 2, sample * 32767, true);
  }

  let binary = "";
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }

  return btoa(binary);
}

// ─────────────────────────────────────────────────────────────────────────────
// Style injection  (one call, idempotent)
// ─────────────────────────────────────────────────────────────────────────────

let _stylesInjected = false;

function _injectStyles() {
  if (_stylesInjected) return;
  _stylesInjected = true;

  const style = document.createElement("style");
  style.id = "hmi-utils-styles";

  style.textContent = `
    /* ── Toast container ───────────────────────────────────────────────── */
    #hmi-toast-container {
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 99999;
      display: flex;
      flex-direction: column-reverse;
      gap: 10px;
      pointer-events: none;
    }

    .hmi-toast {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      min-width: 260px;
      max-width: 380px;
      padding: 12px 16px;
      border-radius: 8px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.35);
      font-family: 'Segoe UI', Arial, sans-serif;
      font-size: 14px;
      line-height: 1.4;
      color: #fff;
      pointer-events: all;
      cursor: default;
      opacity: 0;
      transform: translateX(40px);
      transition: opacity 0.25s ease, transform 0.25s ease;
      position: relative;
      overflow: hidden;
    }

    .hmi-toast.hmi-toast--visible  { opacity: 1; transform: translateX(0); }
    .hmi-toast.hmi-toast--hiding   { opacity: 0; transform: translateX(40px); }

    .hmi-toast--success { background: #1e7e34; border-left: 4px solid #28a745; }
    .hmi-toast--error   { background: #921725; border-left: 4px solid #dc3545; }
    .hmi-toast--info    { background: #0d5c8a; border-left: 4px solid #17a2b8; }
    .hmi-toast--warning { background: #8a5a00; border-left: 4px solid #ffc107; }

    .hmi-toast__icon  { flex-shrink: 0; font-size: 18px; line-height: 1.2; }
    .hmi-toast__msg   { flex: 1; word-break: break-word; }

    .hmi-toast__close {
      flex-shrink: 0;
      background: none;
      border: none;
      color: rgba(255,255,255,0.75);
      font-size: 16px;
      line-height: 1;
      cursor: pointer;
      padding: 0 0 0 4px;
      align-self: flex-start;
      transition: color 0.15s;
    }
    .hmi-toast__close:hover { color: #fff; }

    .hmi-toast__progress {
      position: absolute;
      bottom: 0; left: 0;
      height: 3px;
      border-radius: 0 0 8px 8px;
      background: rgba(255,255,255,0.45);
      width: 100%;
      transform-origin: left;
    }

    /* ── Loading overlay ───────────────────────────────────────────────── */
    .hmi-shared-loading {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(255,255,255,0.82);
      z-index: 10;
      border-radius: inherit;
    }

    .hmi-shared-loading__box {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 10px;
      color: #444;
      font-family: 'Segoe UI', Arial, sans-serif;
      font-size: 14px;
    }

    .hmi-shared-loading__box p { margin: 0; }

    .hmi-shared-loading__spinner {
      width: 30px;
      height: 30px;
      border: 3px solid rgba(0,0,0,0.12);
      border-top-color: #17a2b8;
      border-radius: 50%;
      animation: hmi-spin 0.7s linear infinite;
    }

    @keyframes hmi-spin { to { transform: rotate(360deg); } }

    /* ── Retry state ───────────────────────────────────────────────────── */
    .hmi-shared-retry {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(255,255,255,0.9);
      z-index: 10;
      border-radius: inherit;
    }

    .hmi-shared-retry__box {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 12px;
      padding: 20px;
      text-align: center;
      font-family: 'Segoe UI', Arial, sans-serif;
      font-size: 14px;
      color: #444;
    }

    .hmi-shared-retry__box p { margin: 0; }

    .hmi-shared-retry__box button {
      padding: 7px 20px;
      background: #17a2b8;
      color: #fff;
      border: none;
      border-radius: 5px;
      font-size: 14px;
      font-family: inherit;
      cursor: pointer;
      transition: background 0.15s;
    }
    .hmi-shared-retry__box button:hover { background: #138496; }

    /* ── Page banner ───────────────────────────────────────────────────── */
    .hmi-page-banner {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 16px;
      font-family: 'Segoe UI', Arial, sans-serif;
      font-size: 14px;
      line-height: 1.4;
      border-bottom: 2px solid transparent;
      position: relative;
      z-index: 1000;
    }

    .hmi-page-banner--info    { background: #d1ecf1; color: #0c5460; border-color: #17a2b8; }
    .hmi-page-banner--success { background: #d4edda; color: #155724; border-color: #28a745; }
    .hmi-page-banner--warning { background: #fff3cd; color: #856404; border-color: #ffc107; }
    .hmi-page-banner--error   { background: #f8d7da; color: #721c24; border-color: #dc3545; }

    .hmi-page-banner__icon { flex-shrink: 0; font-size: 16px; }
    .hmi-page-banner__msg  { flex: 1; }

    .hmi-page-banner__close {
      flex-shrink: 0;
      background: none;
      border: none;
      font-size: 18px;
      line-height: 1;
      cursor: pointer;
      color: inherit;
      opacity: 0.6;
      padding: 0 2px;
      transition: opacity 0.15s;
    }
    .hmi-page-banner__close:hover { opacity: 1; }

    /* ── Confirm dialog ────────────────────────────────────────────────── */
    .hmi-confirm-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.45);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 99998;
    }

    .hmi-confirm-box {
      background: #fff;
      border-radius: 8px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.25);
      padding: 28px 32px;
      max-width: 400px;
      width: 90%;
      font-family: 'Segoe UI', Arial, sans-serif;
      text-align: center;
    }

    .hmi-confirm-box p {
      margin: 0 0 20px;
      font-size: 15px;
      line-height: 1.5;
      color: #333;
    }

    .hmi-confirm-actions {
      display: flex;
      gap: 10px;
      justify-content: center;
    }

    .hmi-confirm-actions button {
      padding: 8px 22px;
      border: none;
      border-radius: 5px;
      font-size: 14px;
      font-family: inherit;
      cursor: pointer;
      transition: background 0.15s;
    }

    .hmi-confirm-actions .hmi-confirm-btn--cancel {
      background: #e9ecef;
      color: #333;
    }
    .hmi-confirm-actions .hmi-confirm-btn--cancel:hover { background: #ced4da; }

    .hmi-confirm-actions .hmi-confirm-btn--ok {
      background: #dc3545;
      color: #fff;
    }
    .hmi-confirm-actions .hmi-confirm-btn--ok:hover { background: #c82333; }
  `;

  document.head.appendChild(style);
}

// ─────────────────────────────────────────────────────────────────────────────
// Toast  (Sprint 2 — unchanged except _injectToastStyles → _injectStyles)
// ─────────────────────────────────────────────────────────────────────────────

function _getToastContainer() {
  let container = document.getElementById("hmi-toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "hmi-toast-container";
    document.body.appendChild(container);
  }
  return container;
}

const _TOAST_ICONS = {
  success: "✔",
  error:   "✖",
  info:    "ℹ",
  warning: "⚠",
};

/**
 * Show a toast notification.
 *
 * @param {string} message
 * @param {'success'|'error'|'info'|'warning'} [type='info']
 * @param {number} [duration=4000]
 * @returns {Function} dismiss  — call to close the toast early.
 */
export function showToast(message, type = "info", duration = 4000) {
  if (!["success", "error", "info", "warning"].includes(type)) type = "info";

  _injectStyles();

  const container = _getToastContainer();
  const toast = document.createElement("div");

  toast.className = `hmi-toast hmi-toast--${type}`;
  toast.setAttribute("role", type === "error" || type === "warning" ? "alert" : "status");

  toast.innerHTML = `
    <span class="hmi-toast__icon" aria-hidden="true">${_TOAST_ICONS[type]}</span>
    <span class="hmi-toast__msg">${_escapeHtml(message)}</span>
    <button class="hmi-toast__close" aria-label="Dismiss notification">✕</button>
    <span class="hmi-toast__progress"></span>
  `;

  container.appendChild(toast);

  requestAnimationFrame(() => {
    requestAnimationFrame(() => toast.classList.add("hmi-toast--visible"));
  });

  const bar = toast.querySelector(".hmi-toast__progress");
  if (bar) {
    bar.style.transition = `transform ${duration}ms linear`;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { bar.style.transform = "scaleX(0)"; });
    });
  }

  let dismissed = false;

  function dismiss() {
    if (dismissed) return;
    dismissed = true;
    clearTimeout(autoTimer);
    toast.classList.remove("hmi-toast--visible");
    toast.classList.add("hmi-toast--hiding");
    toast.addEventListener("transitionend", () => toast.remove(), { once: true });
  }

  const autoTimer = setTimeout(dismiss, duration);

  const closeButton = toast.querySelector(".hmi-toast__close");
  if (closeButton) closeButton.addEventListener("click", dismiss);

  return dismiss;
}

// ─────────────────────────────────────────────────────────────────────────────
// API error formatter  (Sprint 2 — unchanged)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Convert a fetch/axios error or HTTP response into a user-safe string.
 *
 * @param {Error|Response|object} error
 * @param {string} [fallbackMessage]
 * @returns {string}
 */
export function getApiErrorMessage(error, fallbackMessage = "Something went wrong. Please try again.") {
  if (!error) return fallbackMessage;

  if (error.code === "ECONNABORTED") {
    return "Request timed out. Please check your connection and try again.";
  }

  if (error.response) {
    const status = error.response.status;
    if (status === 401 || status === 403) return "You are not authorised to access this data.";
    if (status === 404)  return "Requested data was not found.";
    if (status >= 500)   return "Server is currently unavailable. Please try again shortly.";
  }

  return error.message || fallbackMessage;
}

// ─────────────────────────────────────────────────────────────────────────────
// Loading state  (Sprint 2 — unchanged)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Show a loading spinner overlay inside a container element.
 *
 * @param {string} containerId  - The element's id (without #).
 * @param {string} [label='Loading...']
 */
export function showElementLoading(containerId, label = "Loading...") {
  const container = document.getElementById(containerId);
  if (!container) return;

  _injectStyles();

  if (getComputedStyle(container).position === "static") {
    container.style.position = "relative";
  }

  hideElementLoading(containerId);

  const loader = document.createElement("div");
  loader.id = `${containerId}-loading-state`;
  loader.className = "hmi-shared-loading";
  loader.innerHTML = `
    <div class="hmi-shared-loading__box">
      <div class="hmi-shared-loading__spinner"></div>
      <p>${_escapeHtml(label)}</p>
    </div>
  `;

  container.appendChild(loader);
}

/**
 * Remove the loading spinner from a container.
 *
 * @param {string} containerId
 */
export function hideElementLoading(containerId) {
  const loader = document.getElementById(`${containerId}-loading-state`);
  if (loader) loader.remove();
}

// ─────────────────────────────────────────────────────────────────────────────
// Retry state  (Sprint 2 — unchanged)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Show an error message with a Retry button inside a container.
 *
 * @param {string}   containerId
 * @param {string}   message
 * @param {Function} onRetry  - Called when the user clicks Retry.
 */
export function showRetryState(containerId, message, onRetry) {
  const container = document.getElementById(containerId);
  if (!container) return;

  _injectStyles();

  if (getComputedStyle(container).position === "static") {
    container.style.position = "relative";
  }

  hideRetryState(containerId);

  const retry = document.createElement("div");
  retry.id = `${containerId}-retry-state`;
  retry.className = "hmi-shared-retry";
  retry.innerHTML = `
    <div class="hmi-shared-retry__box">
      <p>${_escapeHtml(message)}</p>
      <button type="button">Retry</button>
    </div>
  `;

  container.appendChild(retry);

  const button = retry.querySelector("button");
  if (button) {
    button.addEventListener("click", () => {
      hideRetryState(containerId);
      if (typeof onRetry === "function") onRetry();
    });
  }
}

/**
 * Remove the retry state panel from a container.
 *
 * @param {string} containerId
 */
export function hideRetryState(containerId) {
  const retry = document.getElementById(`${containerId}-retry-state`);
  if (retry) retry.remove();
}

// ─────────────────────────────────────────────────────────────────────────────
// withRetry  (Task 7.3 — new)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Automatically retry an async action on failure.
 * Shows a warning toast between attempts and throws on final failure.
 *
 * @param {Function} action               - Async function to attempt.
 * @param {object}   [opts]
 * @param {number}   [opts.attempts=3]    - Maximum attempts.
 * @param {number}   [opts.delayMs=1500]  - Wait between attempts (ms).
 * @param {string}   [opts.retryMessage]  - Toast text shown on each retry.
 * @returns {Promise<*>}
 *
 * @example
 * const data = await withRetry(() => fetchNodeData(), { attempts: 3 });
 */
export async function withRetry(action, opts = {}) {
  const {
    attempts = 3,
    delayMs = 1500,
    retryMessage = "Retrying…",
  } = opts;

  for (let i = 1; i <= attempts; i++) {
    try {
      return await action();
    } catch (err) {
      if (i < attempts) {
        showToast(`${retryMessage} (attempt ${i}/${attempts})`, "warning");
        await _delay(delayMs);
      } else {
        throw err;
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Page-level banner  (Task 7.1 — new)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Show a persistent full-width status banner at the top of the page.
 * Only one banner per type at a time. Replaces itself if called again.
 *
 * @param {string}  message
 * @param {'info'|'success'|'warning'|'error'} [type='info']
 * @param {boolean} [dismissible=true]
 * @returns {HTMLElement}
 *
 * @example
 * showPageBanner("Critical alert active — check the alerts panel.", "error");
 */
export function showPageBanner(message, type = "info", dismissible = true) {
  if (!["info", "success", "warning", "error"].includes(type)) type = "info";

  _injectStyles();

  // Remove any existing banner of the same type
  const existing = document.getElementById(`hmi-page-banner--${type}`);
  if (existing) existing.remove();

  const banner = document.createElement("div");
  banner.id = `hmi-page-banner--${type}`;
  banner.className = `hmi-page-banner hmi-page-banner--${type}`;
  banner.setAttribute("role", type === "error" || type === "warning" ? "alert" : "status");

  const icons = { info: "ℹ", success: "✔", warning: "⚠", error: "✖" };

  banner.innerHTML = `
    <span class="hmi-page-banner__icon" aria-hidden="true">${icons[type]}</span>
    <span class="hmi-page-banner__msg">${_escapeHtml(message)}</span>
  `;

  if (dismissible) {
    const closeBtn = document.createElement("button");
    closeBtn.className = "hmi-page-banner__close";
    closeBtn.setAttribute("aria-label", "Dismiss");
    closeBtn.innerHTML = "&times;";
    closeBtn.addEventListener("click", () => hidePageBanner(type));
    banner.appendChild(closeBtn);
  }

  document.body.insertAdjacentElement("afterbegin", banner);
  return banner;
}

/**
 * Remove the page banner for a given type.
 *
 * @param {'info'|'success'|'warning'|'error'} [type='info']
 */
export function hidePageBanner(type = "info") {
  const banner = document.getElementById(`hmi-page-banner--${type}`);
  if (banner) banner.remove();
}

// ─────────────────────────────────────────────────────────────────────────────
// Confirmation dialog  (Task 7.3 — new)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Show a non-blocking confirmation dialog.
 * Returns a Promise that resolves to true (confirmed) or false (cancelled).
 * Supports Escape key and backdrop click to cancel.
 *
 * @param {string} message
 * @param {string} [confirmLabel='Confirm']
 * @param {string} [cancelLabel='Cancel']
 * @returns {Promise<boolean>}
 *
 * @example
 * if (await confirmDialog("Remove this device?")) { ... }
 */
export function confirmDialog(message, confirmLabel = "Confirm", cancelLabel = "Cancel") {
  _injectStyles();

  return new Promise((resolve) => {
    // Remove any leftover dialog first
    const old = document.getElementById("hmi-confirm-overlay");
    if (old) old.remove();

    const overlay = document.createElement("div");
    overlay.id = "hmi-confirm-overlay";
    overlay.className = "hmi-confirm-overlay";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Confirmation");

    overlay.innerHTML = `
      <div class="hmi-confirm-box">
        <p>${_escapeHtml(message)}</p>
        <div class="hmi-confirm-actions">
          <button type="button" class="hmi-confirm-btn--cancel">${_escapeHtml(cancelLabel)}</button>
          <button type="button" class="hmi-confirm-btn--ok">${_escapeHtml(confirmLabel)}</button>
        </div>
      </div>
    `;

    document.body.appendChild(overlay);

    // Move focus to the confirm button
    overlay.querySelector(".hmi-confirm-btn--ok").focus();

    function finish(result) {
      document.removeEventListener("keydown", onKey);
      overlay.remove();
      resolve(result);
    }

    overlay.querySelector(".hmi-confirm-btn--ok").addEventListener("click",     () => finish(true));
    overlay.querySelector(".hmi-confirm-btn--cancel").addEventListener("click", () => finish(false));

    // Backdrop click cancels
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) finish(false);
    });

    // Escape key cancels
    function onKey(e) {
      if (e.key === "Escape") finish(false);
    }
    document.addEventListener("keydown", onKey);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// In-flight request guard  (Task 7.3 — new)
// ─────────────────────────────────────────────────────────────────────────────

const _inFlight = new Set();

/**
 * Prevent duplicate concurrent calls for the same keyed action.
 * If the key is already running, shows a warning toast and returns null.
 *
 * @param {string}   key     - Unique identifier for this action.
 * @param {Function} action  - Async function to run.
 * @returns {Promise<*>|null}
 *
 * @example
 * saveBtn.addEventListener("click", () =>
 *   guardRequest("save-device", () => saveDevice(id))
 * );
 */
export async function guardRequest(key, action) {
  if (_inFlight.has(key)) {
    showToast("Request already in progress, please wait.", "warning");
    return null;
  }

  _inFlight.add(key);

  try {
    return await action();
  } finally {
    _inFlight.delete(key);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Debounce  (Task 7.4 — new)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Return a debounced version of a function.
 *
 * @param {Function} fn
 * @param {number}   [wait=300]
 * @returns {Function}
 *
 * @example
 * searchInput.addEventListener("input", debounce(handleSearch, 400));
 */
export function debounce(fn, wait = 300) {
  let timer;
  return function debounced(...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), wait);
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Form submit guard  (Task 7.4 — new)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Attach a safe submit handler to a form element.
 * Automatically: disables submit buttons during submission, shows a loading
 * label on the primary button, fires a success toast, and catches errors.
 *
 * @param {string}   formId              - The form element's id (without #).
 * @param {Function} handler             - async (FormData) => void
 * @param {object}   [opts]
 * @param {string}   [opts.loadingMsg]   - Button label while submitting.
 * @param {string}   [opts.successMsg]   - Toast text on success.
 *
 * @example
 * guardFormSubmit("settings-form", async (formData) => {
 *   const res = await fetch("/api/settings", { method: "POST", body: formData });
 *   if (!res.ok) throw new Error(res.statusText);
 * });
 */
export function guardFormSubmit(formId, handler, opts = {}) {
  const form = document.getElementById(formId);
  if (!form) {
    console.warn(`[HMI-utils] guardFormSubmit: form #${formId} not found.`);
    return;
  }

  const { loadingMsg = "Saving…", successMsg = "Saved successfully." } = opts;

  form.addEventListener("submit", async function (e) {
    e.preventDefault();

    const submitBtns = form.querySelectorAll('[type="submit"]');
    const primaryBtn = submitBtns[0] || null;
    const originalLabel = primaryBtn ? primaryBtn.textContent : "";

    submitBtns.forEach((b) => { b.disabled = true; });
    if (primaryBtn) primaryBtn.textContent = loadingMsg;

    try {
      await handler(new FormData(form));
      showToast(successMsg, "success");
    } catch (err) {
      showToast(getApiErrorMessage(err), "error");
    } finally {
      submitBtns.forEach((b) => { b.disabled = false; });
      if (primaryBtn) primaryBtn.textContent = originalLabel;
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal utilities
// ─────────────────────────────────────────────────────────────────────────────

function _escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function _delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
