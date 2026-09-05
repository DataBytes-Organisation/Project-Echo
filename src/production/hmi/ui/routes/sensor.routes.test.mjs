import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const registerSensorRoutes = require("./sensor.routes.js");

function createRes() {
  return {
    statusCode: 200,
    body: undefined,
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(payload) {
      this.body = payload;
      return this;
    },
    send(payload) {
      this.body = payload;
      return this;
    },
    end() {
      return this;
    },
  };
}

function createApp() {
  const handlers = new Map();
  const app = {
    get(path, handler) {
      handlers.set(`GET ${path}`, handler);
    },
    put(path, handler) {
      handlers.set(`PUT ${path}`, handler);
    },
    post(path, handler) {
      handlers.set(`POST ${path}`, handler);
    },
  };
  registerSensorRoutes(app);
  return {
    async invoke(method, path, { params = {}, query = {}, body, originalUrl } = {}) {
      const handler = handlers.get(`${method} ${path}`);
      if (!handler) throw new Error(`No handler for ${method} ${path}`);
      const req = {
        method,
        params,
        query,
        body,
        originalUrl: originalUrl || path,
      };
      const res = createRes();
      await handler(req, res);
      return res;
    },
  };
}

function jsonResponse(status, data) {
  return {
    status,
    headers: { get: () => "application/json" },
    json: async () => data,
    text: async () => JSON.stringify(data),
  };
}

test("alertsFrom skips Online and Unknown, and raises Offline as Critical", () => {
  const alerts = registerSensorRoutes.alertsFrom([
    { sensorId: "a", status: "Online" },
    { sensorId: "b", status: "Unknown" },
    { sensorId: "c", status: "Offline", lastSeenMinutesAgo: 40 },
    { sensorId: "d", status: "Low Battery", batteryPct: 12 },
  ]);

  assert.equal(alerts.length, 2);
  assert.equal(alerts[0].sensorId, "c");
  assert.equal(alerts[0].severity, "Critical");
  assert.equal(alerts[1].sensorId, "d");
  assert.equal(alerts[1].severity, "High");
});

test("empty backend alerts list is trusted, not replaced with demo data", async (t) => {
  const previousFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = previousFetch;
  });

  globalThis.fetch = async () => jsonResponse(200, { items: [], count: 0 });

  const app = createApp();
  const res = await app.invoke("GET", "/sensors/alerts");

  assert.equal(res.statusCode, 200);
  assert.equal(res.body.source, "backend");
  assert.equal(res.body.count, 0);
  assert.deepEqual(res.body.items, []);
});

test("unreachable alerts backend falls back to demo sensors", async (t) => {
  const previousFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = previousFetch;
  });

  globalThis.fetch = async () => {
    throw new Error("backend down");
  };

  const app = createApp();
  const res = await app.invoke("GET", "/sensors/alerts");

  assert.equal(res.body.source, "demo-fallback");
  assert.ok(res.body.count > 0);
  assert.ok(res.body.items.some((item) => item.sensorId === "LIVE-002"));
});

test("non-2xx alerts response falls back to demo sensors", async (t) => {
  const previousFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = previousFetch;
  });

  globalThis.fetch = async () => jsonResponse(503, { detail: "unavailable" });

  const app = createApp();
  const res = await app.invoke("GET", "/sensors/alerts");

  assert.equal(res.body.source, "demo-fallback");
  assert.ok(res.body.count > 0);
});

test("sensor updates use backend items when the catalog is non-empty", async (t) => {
  const previousFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = previousFetch;
  });

  globalThis.fetch = async () =>
    jsonResponse(200, {
      items: [{ sensorId: "node_1", status: "Unknown", name: "Node Alpha" }],
      count: 1,
    });

  const app = createApp();
  const res = await app.invoke("GET", "/sensors/updates");

  assert.equal(res.body.source, "backend");
  assert.equal(res.body.count, 1);
  assert.equal(res.body.items[0].sensorId, "node_1");
});

test("empty backend catalog falls back to labelled demo sensors", async (t) => {
  const previousFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = previousFetch;
  });

  globalThis.fetch = async () => jsonResponse(200, { items: [], count: 0 });

  const app = createApp();
  const res = await app.invoke("GET", "/sensors/updates");

  assert.equal(res.body.source, "demo-fallback");
  assert.ok(res.body.items.some((item) => item.sensorId === "LIVE-001"));
  assert.equal(res.body.items[0].recentAudio, undefined);
});

test("sensor detail prefers a live backend payload", async (t) => {
  const previousFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = previousFetch;
  });

  globalThis.fetch = async () =>
    jsonResponse(200, {
      sensorId: "node_1",
      status: "Unknown",
      gps: { lat: -38.7789, lon: 143.5705 },
    });

  const app = createApp();
  const res = await app.invoke("GET", "/sensors/:sensorId", {
    params: { sensorId: "node_1" },
    originalUrl: "/sensors/node_1",
  });

  assert.equal(res.body.source, "backend");
  assert.equal(res.body.sensorId, "node_1");
  assert.equal(res.body.gps.lat, -38.7789);
});

test("unknown sensor id returns 404 when backend and demo both miss it", async (t) => {
  const previousFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = previousFetch;
  });

  globalThis.fetch = async () => jsonResponse(404, { detail: "not found" });

  const app = createApp();
  const res = await app.invoke("GET", "/sensors/:sensorId", {
    params: { sensorId: "does-not-exist" },
    originalUrl: "/sensors/does-not-exist",
  });

  assert.equal(res.statusCode, 404);
  assert.match(res.body.detail, /does-not-exist/);
});
