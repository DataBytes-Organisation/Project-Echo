const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { createCheckUserSession } = require("../middleware/session");

// Objects built inside the vm sandbox have a different Object.prototype, which
// makes deepStrictEqual fail on identical content; normalise before comparing.
const plain = (value) => JSON.parse(JSON.stringify(value));

const DETECTION_ID = "651f2a9f4d1f1b1c3e2a4567";
const BACKEND = "http://backend.test:9000";

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
  // Real session middleware, with only Redis replaced.
  const redis = { isOpen: true, on() {}, async get() { return "session-jwt"; } };
  const middleware = load("../middleware/index.js", {
    "./verifySignup": {}, redis: { createClient: () => redis },
    "./session": { createCheckUserSession },
  });

  const calls = [];
  const state = { response: { data: { ok: true } }, error: null };
  const axios = {
    async get(url, options) {
      calls.push({ url, options });
      if (state.error) throw state.error;
      return state.response;
    },
  };
  const register = load("../routes/detection-review.routes.js", {
    axios,
    "../middleware": middleware,
    "../services/apiClient": { API_BASE_URL: BACKEND },
  });

  const routes = new Map();
  register({ get(route, ...handlers) { routes.set(route, handlers); } });

  async function request(route, { token = "session-jwt", params = {}, query = {} } = {}) {
    const handlers = routes.get(route);
    assert.ok(handlers, `${route} is registered`);
    const req = { path: route, session: token ? { token } : {}, params, query };
    const res = {
      statusCode: 200,
      status(code) { this.statusCode = code; return this; },
      json(body) { this.body = body; return this; },
      send(body) { this.body = body; return this; },
      redirect(url) { this.statusCode = 302; this.location = url; return this; },
    };
    let i = 0;
    const next = async () => { if (handlers[i]) return handlers[i++](req, res, next); };
    await next();
    return res;
  }

  return { request, calls, state, routes };
}

const LIST = "/api/detection-review/detections";
const DETAIL = "/api/detection-review/detections/:id";
const SIMILAR = "/api/detection-review/detections/:id/similar";

test("registers only the three read routes the page needs, each behind the session check", () => {
  const h = harness();
  assert.deepEqual([...h.routes.keys()].sort(), [LIST, DETAIL, SIMILAR].sort());
  for (const handlers of h.routes.values()) {
    assert.equal(handlers.length, 2, "session guard plus handler");
  }
});

test("rejects a missing or mismatched session with JSON 401 before any Backend call", async () => {
  const h = harness();

  const missing = await h.request(LIST, { token: null });
  assert.equal(missing.statusCode, 401);
  assert.deepEqual(missing.body, { error: "Authentication required." });

  const wrong = await h.request(DETAIL, { token: "someone-elses-token", params: { id: DETECTION_ID } });
  assert.equal(wrong.statusCode, 401);

  assert.equal(h.calls.length, 0);
});

test("list forwards the session token and only clamped page and page_size", async () => {
  const h = harness();
  h.state.response = { data: { items: [], total: 0, page: 1, page_size: 100 } };

  const res = await h.request(LIST, { query: { page: "0", page_size: "9999", species: "injected", sourceType: "x" } });

  assert.equal(res.statusCode, 200);
  assert.deepEqual(res.body, h.state.response.data);
  assert.equal(h.calls[0].url, `${BACKEND}/detections`);
  assert.deepEqual(plain(h.calls[0].options.params), { page: 1, page_size: 100 });
  assert.equal(h.calls[0].options.headers.Authorization, "Bearer session-jwt");
});

test("list falls back to defaults for non-numeric paging values", async () => {
  const h = harness();
  await h.request(LIST, { query: { page: "abc", page_size: "" } });
  assert.deepEqual(plain(h.calls[0].options.params), { page: 1, page_size: 20 });
});

test("detail forwards a valid id and rejects anything that is not an ObjectId", async () => {
  const h = harness();

  const ok = await h.request(DETAIL, { params: { id: DETECTION_ID } });
  assert.equal(ok.statusCode, 200);
  assert.equal(h.calls[0].url, `${BACKEND}/detections/${DETECTION_ID}`);
  assert.equal(h.calls[0].options.headers.Authorization, "Bearer session-jwt");

  for (const bad of ["abc", "../sensors", `${DETECTION_ID}/similar`, "dashboard-summary", ""]) {
    const res = await h.request(DETAIL, { params: { id: bad } });
    assert.equal(res.statusCode, 400, `rejects ${JSON.stringify(bad)}`);
  }
  assert.equal(h.calls.length, 1, "no Backend call for rejected ids");
});

test("similar forwards a valid id with k defaulted and clamped", async () => {
  const h = harness();

  await h.request(SIMILAR, { params: { id: DETECTION_ID } });
  assert.equal(h.calls[0].url, `${BACKEND}/detections/${DETECTION_ID}/similar`);
  assert.deepEqual(plain(h.calls[0].options.params), { k: 5 });

  await h.request(SIMILAR, { params: { id: DETECTION_ID }, query: { k: "99" } });
  assert.deepEqual(plain(h.calls[1].options.params), { k: 20 });

  await h.request(SIMILAR, { params: { id: DETECTION_ID }, query: { k: "-3" } });
  assert.deepEqual(plain(h.calls[2].options.params), { k: 1 });

  const bad = await h.request(SIMILAR, { params: { id: "not-an-id" } });
  assert.equal(bad.statusCode, 400);
  assert.equal(h.calls.length, 3);
});

function upstreamError(status, message) {
  const error = new Error("upstream failed");
  error.response = { status, data: { error: { message } } };
  return error;
}

test("maps Backend failures to fixed safe messages without forwarding upstream text", async () => {
  const cases = [
    { upstream: upstreamError(404, "Detection not found: secret-internal-detail"), status: 404, code: "NOT_FOUND" },
    { upstream: upstreamError(422, "embedding malformed: secret-internal-detail"), status: 422, code: "UNPROCESSABLE" },
    { upstream: upstreamError(400, "secret-internal-detail"), status: 400, code: "BAD_REQUEST" },
    { upstream: upstreamError(401, "secret-internal-detail"), status: 401, code: "UNAUTHENTICATED" },
    { upstream: upstreamError(403, "Token invalid"), status: 401, code: "UNAUTHENTICATED" },
    { upstream: upstreamError(403, "Budget exceeded for service"), status: 403, code: "FORBIDDEN" },
    { upstream: upstreamError(429, "secret-internal-detail"), status: 429, code: "RATE_LIMITED" },
    { upstream: upstreamError(503, "secret-internal-detail"), status: 503, code: "SERVICE_UNAVAILABLE" },
    { upstream: upstreamError(500, "secret-internal-detail"), status: 502, code: "UPSTREAM_ERROR" },
    { upstream: new Error("connect ECONNREFUSED"), status: 502, code: "UPSTREAM_ERROR" },
  ];

  for (const { upstream, status, code } of cases) {
    const h = harness();
    h.state.error = upstream;
    const res = await h.request(SIMILAR, { params: { id: DETECTION_ID } });
    assert.equal(res.statusCode, status, `${upstream.response?.status ?? "network"} maps to ${status}`);
    assert.equal(res.body.error.code, code);
    assert.ok(!JSON.stringify(res.body).includes("secret-internal-detail"));
    assert.ok(!JSON.stringify(res.body).includes("ECONNREFUSED"));
  }
});
