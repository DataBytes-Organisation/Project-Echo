import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { apiFetch, request } = require("./http-client.js");

function jsonResponse(status, data, ok = status >= 200 && status < 300) {
  return {
    ok,
    status,
    headers: { get: () => "application/json" },
    json: async () => data,
    text: async () => JSON.stringify(data),
  };
}

test("request returns status and data without throwing on HTTP errors", async (t) => {
  const previousFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = previousFetch;
  });

  globalThis.fetch = async () => jsonResponse(503, { detail: "unavailable" }, false);
  const result = await request("http://example.test/sensors/alerts", { timeoutMs: 1000 });
  assert.equal(result.status, 503);
  assert.equal(result.ok, false);
  assert.equal(result.data.detail, "unavailable");
});

test("apiFetch returns parsed JSON on success", async (t) => {
  const previousFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = previousFetch;
  });

  globalThis.fetch = async () => jsonResponse(200, { items: [], count: 0 });
  const data = await apiFetch("/sensors/alerts", { timeoutMs: 1000 });
  assert.deepEqual(data, { items: [], count: 0 });
});

test("apiFetch throws a status-bearing error on non-2xx", async (t) => {
  const previousFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = previousFetch;
  });

  globalThis.fetch = async () => jsonResponse(404, { detail: "missing" }, false);
  await assert.rejects(() => apiFetch("/sensors/nope"), (error) => {
    assert.equal(error.status, 404);
    assert.match(String(error.message), /missing/);
    return true;
  });
});

test("request times out through AbortController", async (t) => {
  const previousFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = previousFetch;
  });

  globalThis.fetch = (_url, init) =>
    new Promise((_resolve, reject) => {
      init.signal.addEventListener("abort", () => {
        const error = new Error("aborted");
        error.name = "AbortError";
        reject(error);
      });
    });

  await assert.rejects(() => request("/slow", { timeoutMs: 20 }), (error) => {
    assert.equal(error.status, 408);
    return true;
  });
});
