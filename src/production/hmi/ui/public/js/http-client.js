/**
 * Shared HTTP helper for HMI browser pages and Node route proxies.
 * FR-D1: one common client instead of per-file AbortController/fetch copies.
 *
 * Browser: load via <script src="/js/http-client.js"> → window.EchoHttp
 * Node:    const { apiFetch, request } = require("../public/js/http-client.js");
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.EchoHttp = api;
    // Convenience for classic admin scripts that already call apiFetch(...).
    if (typeof root.apiFetch !== "function") {
      root.apiFetch = api.apiFetch;
    }
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const DEFAULT_TIMEOUT_MS = 8000;

  /**
   * Low-level request. Does not throw on HTTP error status codes.
   * Throws on timeout / network failure.
   *
   * @returns {Promise<{ status: number, ok: boolean, data: any }>}
   */
  async function request(url, options = {}) {
    const { timeoutMs = DEFAULT_TIMEOUT_MS, headers, signal, ...rest } = options;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    if (signal) {
      if (signal.aborted) controller.abort();
      else signal.addEventListener("abort", () => controller.abort(), { once: true });
    }

    try {
      const response = await fetch(url, {
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
          ...(headers || {}),
        },
        signal: controller.signal,
        ...rest,
      });

      const contentType = response.headers.get("content-type") || "";
      const data = contentType.includes("application/json")
        ? await response.json().catch(() => null)
        : await response.text().catch(() => "");

      return {
        status: response.status,
        ok: response.ok,
        data,
      };
    } catch (error) {
      if (error && error.name === "AbortError") {
        const timeoutError = new Error("Request timed out");
        timeoutError.status = 408;
        timeoutError.name = "AbortError";
        throw timeoutError;
      }
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Browser-facing helper: returns parsed payload, throws on non-2xx / timeout.
   */
  async function apiFetch(path, options = {}) {
    try {
      const result = await request(path, options);
      if (!result.ok) {
        const payload = result.data;
        const detail =
          payload && typeof payload === "object"
            ? payload.detail || payload.error || JSON.stringify(payload)
            : payload;
        const error = new Error(detail || `Request failed: ${result.status}`);
        error.status = result.status;
        throw error;
      }
      return result.data;
    } catch (error) {
      if (error && error.name === "AbortError" && error.status === 408) {
        const timeoutError = new Error("Request timed out while loading sensor data");
        timeoutError.status = 408;
        throw timeoutError;
      }
      throw error;
    }
  }

  return {
    DEFAULT_TIMEOUT_MS,
    request,
    apiFetch,
  };
});
