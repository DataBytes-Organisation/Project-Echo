const assert = require("node:assert/strict");
const test = require("node:test");

async function withServer(app, fn) {
  const server = await new Promise((resolveServer) => {
    const listener = app.listen(0, "127.0.0.1", () => resolveServer(listener));
  });
  try {
    const { port } = server.address();
    await fn(`http://127.0.0.1:${port}`);
  } finally {
    await new Promise((resolveClose, rejectClose) => {
      server.close((error) => error ? rejectClose(error) : resolveClose());
    });
  }
}

test("createApp returns an Express app without listening or connecting", () => {
  const { createApp } = require("../../server/app");
  let connected = false;
  const app = createApp({
    redisClient: { connect: async () => { connected = true; } },
    databases: { connect: async () => { connected = true; } },
  });

  assert.equal(typeof app, "function");
  assert.equal(typeof app.listen, "function");
  assert.equal(connected, false);
});

test("notification read-all route is registered before notification id routes", async () => {
  const { createApp } = require("../../server/app");
  const calls = [];
  const notificationStore = {
    async listNotifications() {
      return { notifications: [{ id: "n1", read: false }], unreadCount: 1 };
    },
    async setNotificationFlags(ids, patch) {
      calls.push({ ids, patch });
    },
  };

  const app = createApp({
    requireApiSession: (req, res, next) => next(),
    notificationStore,
    redisClient: { isOpen: true, async connect() {}, async get() { return null; } },
  });

  await withServer(app, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/notifications/read-all`, { method: "PATCH" });
    assert.equal(response.status, 200);
    assert.deepEqual(calls, [{ ids: ["n1"], patch: { read: true } }]);
  });
});

test("sensor-specific routes run before the wildcard API proxy", async (t) => {
  const { createApp } = require("../../server/app");
  const previousFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = previousFetch;
  });

  const app = createApp({
    redisClient: { isOpen: true, async connect() {}, async get() { return null; } },
  });

  await withServer(app, async (baseUrl) => {
    globalThis.fetch = async (url, options) => {
      if (String(url).startsWith(baseUrl)) return previousFetch(url, options);
      return {
        status: 200,
        headers: { get: () => "application/json" },
        json: async () => ({ items: [{ sensorId: "node_1", status: "Online" }], count: 1 }),
        text: async () => "{}",
      };
    };

    const response = await fetch(`${baseUrl}/sensors/updates`);
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.source, "backend");
    assert.equal(body.items[0].sensorId, "node_1");
  });
});

test("/iot/nodes keeps the session guard before /iot/nodes/:nodeId", async () => {
  const { createApp } = require("../../server/app");
  const app = createApp({
    checkUserSession: (req, res) => res.status(401).json({ error: "Authentication required." }),
    redisClient: { isOpen: true, async connect() {}, async get() { return null; } },
  });

  await withServer(app, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/iot/nodes`);
    assert.equal(response.status, 401);
    assert.deepEqual(await response.json(), { error: "Authentication required." });
  });
});
