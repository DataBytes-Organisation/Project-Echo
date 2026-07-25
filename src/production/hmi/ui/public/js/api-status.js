// =============================================================
// API Status & Environment Detection
// Sprint 2 - Frontend API Environment Validation
// =============================================================

/**
 * Test if the API connection is working.
 * Returns true if reachable, false if not.
 */
async function testApiConnection() {
  try {
    const response = await fetch('/iot/nodes', { method: 'GET' });
    return response.ok;
  } catch (error) {
    console.error('API connection test failed:', error);
    return false;
  }
}

/**
 * Determine which environment the frontend is running in.
 * Returns: "Local Mode", "Live Mode", or "Fallback Mode"
 */
function getEnvironmentMode() {
  const host = window.location.hostname;

  if (host === 'localhost' || host === '127.0.0.1') {
    return 'Local Mode';
  } else if (host.match(/^\d+\.\d+\.\d+\.\d+$/) || host.includes('project-echo')) {
    return 'Live Mode';
  } else {
    return 'Fallback Mode';
  }
}

/**
 * Get a user-friendly status message based on environment and connection.
 */
function getStatusMessage(connected, mode) {
  if (mode === 'Fallback Mode') {
    return 'Live server URL not configured yet, running in local fallback mode.';
  }
  if (!connected) {
    return 'Cannot connect to backend API. Please check your configuration.';
  }
  return `Running in ${mode} — API connected successfully.`;
}

/**
 * Update the status display on the page.
 * Looks for an element with id="api-status-badge" to update.
 */
async function updateApiStatus() {
  const badge = document.getElementById('api-status-badge');
  if (!badge) return;

  const mode = getEnvironmentMode();
  const connected = await testApiConnection();
  const message = getStatusMessage(connected, mode);

  badge.textContent = message;
  badge.style.padding = '6px 12px';
  badge.style.borderRadius = '6px';
  badge.style.fontSize = '12px';
  badge.style.fontWeight = '500';

  if (connected && mode !== 'Fallback Mode') {
    badge.style.backgroundColor = '#d4edda';
    badge.style.color = '#155724';
  } else {
    badge.style.backgroundColor = '#f8d7da';
    badge.style.color = '#721c24';
  }
}

// Run once on page load, then refresh every 30 seconds
document.addEventListener('DOMContentLoaded', () => {
  updateApiStatus();
  setInterval(updateApiStatus, 30000);
});