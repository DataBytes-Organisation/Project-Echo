const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { createCheckUserSession } = require("../middleware/session");

function load(relative, dependencies) {
  const module = { exports: {} };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, relative), "utf8"), {
    module, exports: module.exports, process: { env: { API_HOST: "backend.test" } }, console,
    require(name) {
      if (!(name in dependencies)) throw new Error(`Unexpected dependency: ${name}`);
      return dependencies[name];
    },
  });
  return module.exports;
}

function harness() {
  const redis = { isOpen: true, on() {}, async get() { return "session-jwt"; } };
  const middleware = load("../middleware/index.js", {
    "./verifySignup": {}, redis: { createClient: () => redis },
    "./session": { createCheckUserSession },
  });
  const calls = [];
  const http = { async get(url, options) { calls.push({ url, options }); return { data: [] }; } };
  const register = load("../routes/map.routes.js", {
    "../middleware": middleware, axios: http, dotenv: { config() {} },
  });
  const routes = new Map();
  const app = { use() {}, get(route, ...handlers) { routes.set(route, handlers); }, post() {}, put() {} };
  register(app);
  async function request(route, token) {
    const req = { path: route, session: token ? { token } : {} };
    const res = { statusCode: 200, status(code) { this.statusCode = code; return this; },
      json(body) { this.body = body; return this; }, send(body) { this.body = body; return this; },
      redirect(url) { this.statusCode = 302; this.location = url; } };
    let i = 0;
    const handlers = route === "/map" ? [middleware.checkUserSession, (_req, response) => response.send("map")] : routes.get(route);
    assert.ok(handlers, `${route} is registered`);
    const next = async () => { if (handlers[i]) return handlers[i++](req, res, next); };
    await next(); return res;
  }
  return { request, calls, http };
}

test("map redirects unauthenticated users despite public route configuration", async () => {
  const h = harness();
  const response = await h.request("/map");
  assert.equal(response.statusCode, 302);
  assert.equal(response.location, "/login");
  assert.equal((await h.request("/map", "session-jwt")).body, "map");
});

test("detection API rejects missing or mismatched sessions before Backend access", async () => {
  const h = harness();
  for (const token of [undefined, "another-jwt"]) {
    const response = await h.request("/api/detections", token);
    assert.equal(response.statusCode, 401);
    assert.match(JSON.stringify(response.body), /Authentication required/);
  }
  assert.equal(h.calls.length, 0);
});

test("authenticated detection read forwards JWT and returns live data unchanged", async () => {
  const h = harness();
  const live = [{ _id: "event-1", sourceType: "real", microphoneLLA: { latitude: -37, longitude: 144, altitude: 0 },
    species: "Magpie", confidence: 91.5, timestamp: "2026-08-06T10:30:00Z", sensorId: "esp32-001" }];
  h.http.get = async (url, options) => { h.calls.push({ url, options }); return { data: live }; };
  const response = await h.request("/api/detections", "session-jwt");
  assert.equal(response.body, live);
  assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0].url, "http://backend.test:9000/hmi/detections");
  assert.equal(h.calls[0].options.headers.Authorization, "Bearer session-jwt");
  assert.equal(h.calls[0].options.timeout, 10000);
});

test("empty reads stay empty and unknown failures stay generic and safe", async () => {
  const h = harness();
  assert.equal((await h.request("/api/detections", "session-jwt")).body.length, 0);
  h.http.get = async () => { throw { response: { status: 500, data: { error: "private token http://internal" } } }; };
  const response = await h.request("/api/detections", "session-jwt");
  assert.equal(response.statusCode, 502);
  assert.equal(typeof response.body.error.message, "string");
  assert.ok("details" in response.body.error);
  assert.doesNotMatch(JSON.stringify(response.body), /private|token|internal/);
});

test("upstream 401 tells the user to log in again", async () => {
  const h = harness();
  h.http.get = async () => { throw { response: { status: 401,
    data: { error: { code: "UNAUTHENTICATED", message: "Invalid token or expired token.", details: null } } } }; };
  const response = await h.request("/api/detections", "session-jwt");
  assert.equal(response.statusCode, 401);
  assert.match(response.body.error.message, /log in again/i);
});

test("upstream 403 with budget detail points to an administrator", async () => {
  const h = harness();
  for (const message of ["Budget is not configured for 'detections' (monthly_limit=0).",
    "Budget exceeded for 'detections'. Used 100000/100000 this month."]) {
    h.http.get = async () => { throw { response: { status: 403,
      data: { error: { code: "FORBIDDEN", message, details: null } } } }; };
    const response = await h.request("/api/detections", "session-jwt");
    assert.equal(response.statusCode, 403);
    assert.match(response.body.error.message, /administrator/i);
    assert.doesNotMatch(JSON.stringify(response.body), /monthly_limit|Used \d+|log in again/i);
  }
});

test("upstream 403 with auth failure tells the user to log in again", async () => {
  const h = harness();
  h.http.get = async () => { throw { response: { status: 403,
    data: { error: { code: "FORBIDDEN", message: "Invalid token or expired token.", details: null } } } }; };
  const response = await h.request("/api/detections", "session-jwt");
  assert.equal(response.statusCode, 401);
  assert.match(response.body.error.message, /log in again/i);
});

test("upstream 429 asks the user to wait before retrying", async () => {
  const h = harness();
  h.http.get = async () => { throw { response: { status: 429,
    data: { error: { code: "RATE_LIMIT_EXCEEDED", message: "Budget exceeded for 'detections'.", details: null } } } }; };
  const response = await h.request("/api/detections", "session-jwt");
  assert.equal(response.statusCode, 429);
  assert.match(response.body.error.message, /too many|wait/i);
});

test("upstream 503 reports the service as temporarily unavailable", async () => {
  const h = harness();
  h.http.get = async () => { throw { response: { status: 503,
    data: { error: { code: "SERVICE_UNAVAILABLE", message: "Service 'detections' is temporarily paused by admin.", details: null } } } }; };
  const response = await h.request("/api/detections", "session-jwt");
  assert.equal(response.statusCode, 503);
  assert.match(response.body.error.message, /unavailable|later|paused/i);
});

test("the static map alias is session-protected before Express serves files", async () => {
  const express = require("express");
  const app = express();
  app.use((req, _res, next) => { req.session = { token: req.headers["test-token"] }; next(); });
  const checkUserSession = createCheckUserSession({ isOpen: true, async get() { return "session-jwt"; } });
  // Execute the relevant production registrations in their original order.
  const registration = fs.readFileSync(path.join(__dirname, "../server.js"), "utf8")
    .split("\n").filter(line => line.includes('app.get("/index.html"') || line.includes("app.use(express.static")).join("\n");
  vm.runInNewContext(registration, { app, express, path, __dirname: path.join(__dirname, ".."), checkUserSession });
  const server = app.listen(0, "127.0.0.1");
  await new Promise(resolve => server.once("listening", resolve));
  try {
    const url = `http://127.0.0.1:${server.address().port}/index.html`;
    const anonymous = await fetch(url, { redirect: "manual" });
    assert.equal(anonymous.status, 302);
    assert.equal(anonymous.headers.get("location"), "/login");
    const authenticated = await fetch(url, { headers: { "test-token": "session-jwt" } });
    assert.equal(authenticated.status, 200);
    assert.match(await authenticated.text(), /id="basemap"/);
  } finally { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
});
