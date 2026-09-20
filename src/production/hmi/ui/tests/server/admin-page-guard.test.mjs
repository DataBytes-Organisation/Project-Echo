import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createApp } = require("../../server/app");

function emptyRedis() {
  return { isOpen: true, async connect() {}, async get() { return null; } };
}

async function withServer(app, fn) {
  const server = await new Promise((resolveServer) => {
    const listener = app.listen(0, "127.0.0.1", () => resolveServer(listener));
  });
  try {
    const { port } = server.address();
    await fn(`http://127.0.0.1:${port}`);
  } finally {
    await new Promise((resolveClose, rejectClose) => {
      server.close((error) => (error ? rejectClose(error) : resolveClose()));
    });
  }
}

test("unauthenticated /admin/*.html redirects to /login", async () => {
  const app = createApp({ redisClient: emptyRedis() });
  await withServer(app, async (baseUrl) => {
    for (const route of ["/admin/dashboard.html", "/admin/sensor-health.html"]) {
      const response = await fetch(`${baseUrl}${route}`, { redirect: "manual" });
      assert.equal(response.status, 302, `${route} should redirect`);
      assert.equal(response.headers.get("location"), "/login");
    }
  });
});

test("unauthenticated /pages/admin/*.html redirects to /login", async () => {
  const app = createApp({ redisClient: emptyRedis() });
  await withServer(app, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/pages/admin/dashboard.html`, { redirect: "manual" });
    assert.equal(response.status, 302);
    assert.equal(response.headers.get("location"), "/login");
  });
});

test("authenticated /admin/*.html still serves the page", async () => {
  const app = createApp({
    checkUserSession: (req, res, next) => next(),
    redisClient: emptyRedis(),
  });
  await withServer(app, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/admin/dashboard.html`, { redirect: "manual" });
    assert.equal(response.status, 200);
    assert.match(await response.text(), /<title[^>]*>/i);
  });
});
