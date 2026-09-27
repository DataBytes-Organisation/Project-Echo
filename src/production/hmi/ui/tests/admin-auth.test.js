const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { createCheckUserSession, tokensMatch, resolveLandingPath } = require("../middleware/session");

function loadMiddleware(storedToken = "session-jwt") {
  const redis = { isOpen: true, on() {}, async get() { return storedToken; } };
  const module = { exports: {} };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../middleware/index.js"), "utf8"), {
    module,
    exports: module.exports,
    process: { env: {} },
    console,
    require(name) {
      if (name === "./verifySignup") return {};
      if (name === "redis") return { createClient: () => redis };
      if (name === "./session") return { createCheckUserSession, tokensMatch, resolveLandingPath };
      throw new Error(`Unexpected dependency: ${name}`);
    },
  });
  return module.exports;
}

async function runCheck(checkUserSession, route, token) {
  const req = { path: route, session: token ? { token } : {} };
  const res = {
    statusCode: 200,
    status(code) { this.statusCode = code; return this; },
    json(body) { this.body = body; return this; },
    send(body) { this.body = body; return this; },
    redirect(url) { this.statusCode = 302; this.location = url; },
  };
  let nextCalled = false;
  await checkUserSession(req, res, () => { nextCalled = true; });
  return { res, nextCalled };
}

test("unauthenticated /admin-dashboard redirects to /login even when Redis holds an admin JWT", async () => {
  const { checkUserSession } = loadMiddleware("admin-jwt");
  const { res, nextCalled } = await runCheck(checkUserSession, "/admin-dashboard", undefined);
  assert.equal(nextCalled, false);
  assert.equal(res.statusCode, 302);
  assert.equal(res.location, "/login");
});

test("mismatched session cannot reach /admin-nodes", async () => {
  const { checkUserSession } = loadMiddleware("session-jwt");
  const { res, nextCalled } = await runCheck(checkUserSession, "/admin-nodes", "another-jwt");
  assert.equal(nextCalled, false);
  assert.equal(res.statusCode, 302);
  assert.equal(res.location, "/login");
});

test("matching session still reaches admin pages", async () => {
  const { checkUserSession } = loadMiddleware("session-jwt");
  const { nextCalled } = await runCheck(checkUserSession, "/admin-dashboard", "session-jwt");
  assert.equal(nextCalled, true);
});

test("landing redirect sends strangers to login, admins to dashboard, users to map", () => {
  assert.equal(resolveLandingPath(undefined, "session-jwt", "admin"), "/login");
  assert.equal(resolveLandingPath("another-jwt", "session-jwt", "admin"), "/login");
  assert.equal(resolveLandingPath("session-jwt", "session-jwt", "admin"), "/admin-dashboard");
  assert.equal(resolveLandingPath("session-jwt", "session-jwt", "Admin,User"), "/admin-dashboard");
  assert.equal(resolveLandingPath("session-jwt", "session-jwt", "user"), "/map");
  assert.equal(resolveLandingPath("session-jwt", "session-jwt", null), "/map");
});
