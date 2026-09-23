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

const storeItems = new Map([[1, { priceInCents: 100, name: "donation" }]]);

function checkoutApp({ stripeBehavior, clientUrl }) {
  const stripe = {
    checkout: {
      sessions: {
        create: stripeBehavior,
      },
    },
  };
  return createApp({
    redisClient: emptyRedis(),
    stripe,
    storeItems,
    donationClient: {},
    dbState: {},
    config: { apiHost: "localhost", apiPort: 9000, clientUrl },
  });
}

test("checkout failure returns a generic error, not the raw provider message", async () => {
  const app = checkoutApp({
    clientUrl: "https://hmi.example.test",
    stripeBehavior: async () => {
      throw new Error("Stripe secret key sk_test_123 is invalid");
    },
  });
  await withServer(app, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/create-checkout-session`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ items: [{ id: 1, quantity: 1 }] }),
    });
    assert.equal(response.status, 500);
    const body = await response.json();
    assert.equal(body.error, "Internal server error");
    assert.doesNotMatch(JSON.stringify(body), /sk_test_123/);
  });
});

test("checkout redirect URLs come from client configuration, not localhost:9000", async () => {
  let captured = null;
  const app = checkoutApp({
    clientUrl: "https://hmi.example.test",
    stripeBehavior: async (params) => {
      captured = params;
      return { url: "https://checkout.stripe.com/session" };
    },
  });
  await withServer(app, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/create-checkout-session`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ items: [{ id: 1, quantity: 1 }] }),
    });
    assert.equal(response.status, 200);
    assert.match(captured.success_url, /^https:\/\/hmi\.example\.test\//);
    assert.equal(captured.cancel_url, "https://hmi.example.test");
    assert.doesNotMatch(JSON.stringify(captured), /localhost:9000/);
  });
});

test("request_access failure returns a generic message, not the caught error", async () => {
  const app = createApp({
    redisClient: emptyRedis(),
    emailService: {
      guestSalt: () => "salt",
      genPass: () => "password1234",
      controller: {
        guestsignup: async () => {
          throw new Error("connect ECONNREFUSED 10.0.0.5:6379");
        },
      },
    },
  });
  await withServer(app, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/request_access`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ email: "guest@example.com" }),
    });
    assert.equal(response.status, 500);
    const body = await response.json();
    assert.equal(body.message, "An error occurred while sending the request access.");
    assert.doesNotMatch(JSON.stringify(body), /ECONNREFUSED/);
    assert.doesNotMatch(JSON.stringify(body), /6379/);
  });
});
