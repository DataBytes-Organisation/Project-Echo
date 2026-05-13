"use strict";

export function getAudioTestString() {
  // Returns a short base-64 encoded WAV test tone (existing implementation)
  // Kept exactly as-is – do not modify.
  const sampleRate = 8000;
  const frequency  = 440;
  const duration   = 1;
  const numSamples = sampleRate * duration;
  const buffer     = new ArrayBuffer(44 + numSamples * 2);
  const view       = new DataView(buffer);

  function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeString(view,  0, 'RIFF');
  view.setUint32( 4, 36 + numSamples * 2, true);
  writeString(view,  8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16,          true);
  view.setUint16(20, 1,           true);
  view.setUint16(22, 1,           true);
  view.setUint32(24, sampleRate,  true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2,           true);
  view.setUint16(34, 16,          true);
  writeString(view, 36, 'data');
  view.setUint32(40, numSamples * 2, true);

  for (let i = 0; i < numSamples; i++) {
    const sample = Math.sin((2 * Math.PI * frequency * i) / sampleRate);
    view.setInt16(44 + i * 2, sample * 32767, true);
  }

  let binary = '';
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}





// Usage (from any module):
//   import { showToast } from "./HMI-utils.js";
//   showToast("Microphones loaded successfully", "success");
//   showToast("Connection timed out. Retrying…",  "error");
//   showToast("Simulator is in Animal Mode",       "info");
//
// Parameters:
//   message  {string}  – Human-readable text to display.
//   type     {string}  – "success" | "error" | "info"  (default: "info")
//   duration {number}  – Milliseconds before auto-dismiss  (default: 4000)


// Inject the required CSS once, the first time showToast is called.
let _toastStylesInjected = false;
function _injectToastStyles() {
  if (_toastStylesInjected) return;
  _toastStylesInjected = true;

  const style = document.createElement('style');
  style.id = 'hmi-toast-styles';
  style.textContent = `
    /* ── Toast container */
    #hmi-toast-container {
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 99999;
      display: flex;
      flex-direction: column-reverse;   /* newest toast appears on top */
      gap: 10px;
      pointer-events: none;             /* container itself is click-through */
    }

    /* ── Individual toast  */
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

      /* enter animation */
      opacity: 0;
      transform: translateX(40px);
      transition: opacity 0.25s ease, transform 0.25s ease;
    }

    .hmi-toast.hmi-toast--visible {
      opacity: 1;
      transform: translateX(0);
    }

    .hmi-toast.hmi-toast--hiding {
      opacity: 0;
      transform: translateX(40px);
    }

    /* ── Type colour themes  */
    .hmi-toast--success { background: #1e7e34; border-left: 4px solid #28a745; }
    .hmi-toast--error   { background: #921725; border-left: 4px solid #dc3545; }
    .hmi-toast--info    { background: #0d5c8a; border-left: 4px solid #17a2b8; }

    /* ── Icon  */
    .hmi-toast__icon {
      flex-shrink: 0;
      font-size: 18px;
      line-height: 1.2;
    }

    /* ── Message text  */
    .hmi-toast__msg {
      flex: 1;
      word-break: break-word;
    }

    /* ── Close button  */
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

    /* ── Progress bar (auto-dismiss countdown)  */
    .hmi-toast__progress {
      position: absolute;
      bottom: 0;
      left: 0;
      height: 3px;
      border-radius: 0 0 8px 8px;
      background: rgba(255,255,255,0.45);
      width: 100%;
      transform-origin: left;
    }
  `;
  document.head.appendChild(style);
}

// Lazily create / return the shared container element.
function _getToastContainer() {
  let container = document.getElementById('hmi-toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'hmi-toast-container';
    document.body.appendChild(container);
  }
  return container;
}

const _TOAST_ICONS = {
  success : '✔',
  error   : '✖',
  info    : 'ℹ',
};

/**
 * showToast – display a self-dismissing notification pop-up.
 *
 * @param {string} message   Text to show the user.
 * @param {string} [type]    "success" | "error" | "info"  (default "info")
 * @param {number} [duration] Milliseconds until auto-dismiss (default 4000).
 */
export function showToast(message, type = 'info', duration = 4000) {
  // Normalise type
  if (!['success', 'error', 'info'].includes(type)) type = 'info';

  _injectToastStyles();
  const container = _getToastContainer();

  // ── Build toast element 
  const toast = document.createElement('div');
  toast.className = `hmi-toast hmi-toast--${type}`;
  toast.setAttribute('role', type === 'error' ? 'alert' : 'status');
  toast.style.position = 'relative';          // needed for progress bar

  toast.innerHTML = `
    <span class="hmi-toast__icon" aria-hidden="true">${_TOAST_ICONS[type]}</span>
    <span class="hmi-toast__msg">${_escapeHtml(message)}</span>
    <button class="hmi-toast__close" aria-label="Dismiss notification">✕</button>
    <span class="hmi-toast__progress"></span>
  `;

  container.appendChild(toast);

  // ── Animate in 
  // rAF ensures the element is painted before the transition fires.
  requestAnimationFrame(() => {
    requestAnimationFrame(() => toast.classList.add('hmi-toast--visible'));
  });

  // ── Progress bar animation 
  const bar = toast.querySelector('.hmi-toast__progress');
  bar.style.transition = `transform ${duration}ms linear`;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => { bar.style.transform = 'scaleX(0)'; });
  });

  // ── Dismiss helper
  let dismissed = false;
  function dismiss() {
    if (dismissed) return;
    dismissed = true;
    clearTimeout(autoTimer);
    toast.classList.remove('hmi-toast--visible');
    toast.classList.add('hmi-toast--hiding');
    toast.addEventListener('transitionend', () => toast.remove(), { once: true });
  }

  // ── Auto-dismiss 
  const autoTimer = setTimeout(dismiss, duration);

  // ── Manual close button 
  toast.querySelector('.hmi-toast__close').addEventListener('click', dismiss);

  return dismiss; // caller can invoke to close early if needed
}

// Simple HTML-escape to prevent XSS in toast messages.
function _escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
