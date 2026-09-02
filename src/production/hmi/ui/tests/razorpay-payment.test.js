"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const {
  createOrder,
  createPaymentRateLimiter,
  handleWebhook,
  registerRazorpayBrowserRoutes,
  savePayment,
  withRazorpayCsp,
} = require("../routes/razorpay.routes");
const { createCheckUserSession } = require("../middleware/session");

const CHECKOUT_ORDER = {
  keyId: "rzp_test_key",
  orderId: "order_verified123",
  amount: 500,
  currency: "AUD",
};

function response() {
  return {
    statusCode: 200,
    body: undefined,
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(body) {
      this.body = body;
      return this;
    },
  };
}

function proxyClient(result = { status: 200, data: { status: "success" } }) {
  return {
    calls: [],
    async post(url, body, options) {
      this.calls.push({ url, body, options });
      if (result.error) throw result.error;
      return result;
    },
  };
}

function backendError(status, data) {
  const error = new Error("Backend request failed");
  error.response = { status, data };
  return error;
}

function routeHarness() {
  const routes = new Map();

  return {
    app: {
      post(path, ...handlers) {
        routes.set(path, handlers);
      },
    },
    async post(path, req) {
      const handlers = routes.get(path);
      const res = response();
      let index = 0;
      const next = async () => {
        const handler = handlers?.[index++];
        if (handler) return handler(req, res, next);
      };
      await next();
      return res;
    },
  };
}

test("createOrder forwards the amount and requester JWT to FastAPI once", async () => {
  const httpClient = proxyClient({ status: 201, data: CHECKOUT_ORDER });
  const res = response();

  await createOrder(
    { body: { amount: 500 }, session: { token: "requester-jwt" } },
    res,
    { httpClient, apiBaseUrl: "http://backend:9000" }
  );

  assert.equal(httpClient.calls.length, 1);
  assert.deepEqual(httpClient.calls[0], {
    url: "http://backend:9000/payments/razorpay/orders",
    body: { amount: 500 },
    options: {
      headers: { Authorization: "Bearer requester-jwt" },
      timeout: 10000,
    },
  });
  assert.equal(res.statusCode, 201);
  assert.deepEqual(res.body, CHECKOUT_ORDER);
});

test("savePayment forwards only checkout proof fields to FastAPI once", async () => {
  const httpClient = proxyClient();
  const res = response();
  const proof = {
    paymentId: "pay_verified123",
    orderId: "order_verified123",
    signature: "a".repeat(64),
    name: "Alice",
  };

  await savePayment(
    { body: { ...proof, amount: 999999, currency: "USD", email: "forged@example.com" } },
    res,
    { httpClient, apiBaseUrl: "http://backend:9000" }
  );

  assert.equal(httpClient.calls.length, 1);
  assert.deepEqual(httpClient.calls[0], {
    url: "http://backend:9000/payments/razorpay/verify",
    body: proof,
    options: { timeout: 21000 },
  });
  assert.equal(res.statusCode, 200);
  assert.deepEqual(res.body, { status: "success" });
});

test("handleWebhook forwards exact bytes and Razorpay signature once", async () => {
  const rawBody = Buffer.from('{  "event": "order.paid"  }');
  const httpClient = proxyClient({ status: 200, data: { status: "ignored" } });
  const res = response();

  await handleWebhook(
    {
      body: rawBody,
      get(name) {
        return name.toLowerCase() === "x-razorpay-signature" ? "a".repeat(64) : undefined;
      },
    },
    res,
    { httpClient, apiBaseUrl: "http://backend:9000" }
  );

  assert.equal(httpClient.calls.length, 1);
  assert.equal(httpClient.calls[0].body, rawBody);
  assert.deepEqual(httpClient.calls[0], {
    url: "http://backend:9000/payments/razorpay/webhook",
    body: rawBody,
    options: {
      headers: {
        "Content-Type": "application/json",
        "x-razorpay-signature": "a".repeat(64),
      },
      timeout: 10000,
    },
  });
  assert.equal(res.statusCode, 200);
  assert.deepEqual(res.body, { status: "ignored" });
});

test("payment proxies preserve known Backend errors", async () => {
  const cases = [
    [createOrder, { body: { amount: 101 }, session: { token: "requester-jwt" } }, 400, "Invalid donation amount."],
    [savePayment, { body: {} }, 400, "Payment could not be verified."],
    [handleWebhook, { body: Buffer.from("{}"), headers: {} }, 400, "Invalid webhook."],
  ];

  for (const [handler, req, status, message] of cases) {
    const httpClient = proxyClient({
      error: backendError(status, { error: message, detail: "internal stack" }),
    });
    const res = response();
    await handler(req, res, { httpClient, apiBaseUrl: "http://backend:9000" });
    assert.equal(httpClient.calls.length, 1);
    assert.equal(res.statusCode, status);
    assert.deepEqual(res.body, { error: message });
  }
});

test("payment proxies replace unreachable or malformed Backend responses safely", async (t) => {
  t.mock.method(console, "error", () => {});
  const cases = [
    [createOrder, { body: { amount: 500 }, session: { token: "requester-jwt" } }, "Payment service is unavailable."],
    [savePayment, { body: {} }, "Donation could not be recorded."],
    [handleWebhook, { body: Buffer.from("{}"), headers: {} }, "Donation could not be recorded."],
  ];

  for (const [handler, req, message] of cases) {
    for (const result of [
      { error: new Error("network down") },
      { status: 200, data: "not-json" },
      { error: backendError(500, { detail: "internal stack" }) },
    ]) {
      const res = response();
      await handler(req, res, {
        httpClient: proxyClient(result),
        apiBaseUrl: "http://backend:9000",
      });
      assert.equal(res.statusCode, 503);
      assert.deepEqual(res.body, { error: message });
    }
  }
});

test("withRazorpayCsp enables Checkout scripts, API connections, and payment frames", () => {
  const directives = withRazorpayCsp({
    scriptSrc: ["'self'"],
    scriptSrcElem: ["'self'"],
    connectSrc: ["'self'"],
    frameSrc: ["'self'"],
  });

  assert.deepEqual(directives.scriptSrc, ["'self'", "https://checkout.razorpay.com"]);
  assert.deepEqual(directives.scriptSrcElem, ["'self'", "https://checkout.razorpay.com"]);
  assert.deepEqual(directives.connectSrc, ["'self'", "https://api.razorpay.com", "https://*.razorpay.com"]);
  assert.deepEqual(directives.frameSrc, ["'self'", "https://api.razorpay.com", "https://*.razorpay.com"]);
});

test("payment rate limiters keep order and verification budgets separate", () => {
  let now = 1000;
  let orderCalls = 0;
  let verificationCalls = 0;
  const options = {
    limit: 2,
    windowMs: 1000,
    now: () => now,
  };
  const limitOrders = createPaymentRateLimiter(options);
  const limitVerification = createPaymentRateLimiter(options);
  const req = { ip: "192.0.2.1" };

  limitOrders(req, response(), () => { orderCalls += 1; });
  limitOrders(req, response(), () => { orderCalls += 1; });
  const blockedOrder = response();
  limitOrders(req, blockedOrder, () => { orderCalls += 1; });
  limitVerification(req, response(), () => { verificationCalls += 1; });
  limitVerification(req, response(), () => { verificationCalls += 1; });
  const blockedVerification = response();
  limitVerification(req, blockedVerification, () => { verificationCalls += 1; });

  assert.equal(orderCalls, 2);
  assert.equal(verificationCalls, 2);
  for (const blocked of [blockedOrder, blockedVerification]) {
    assert.equal(blocked.statusCode, 429);
    assert.deepEqual(blocked.body, { error: "Too many payment attempts. Please try again later." });
  }

  now = 2001;
  limitVerification(req, response(), () => { verificationCalls += 1; });
  assert.equal(verificationCalls, 3);
});

test("Razorpay browser routes rate-limit verification without requiring authentication", async () => {
  const harness = routeHarness();
  const httpClient = proxyClient();
  const checkUserSession = (req, res, next) => {
    if (req.session?.token !== "requester-jwt") {
      return res.status(401).json({ error: "Authentication required." });
    }
    return next();
  };
  registerRazorpayBrowserRoutes(harness.app, {
    apiBaseUrl: "http://backend:9000",
    checkUserSession,
    httpClient,
  });

  const unauthenticatedOrder = await harness.post(
    "/api/create-razorpay-order",
    { ip: "192.0.2.1", body: { amount: 500 } }
  );
  assert.equal(unauthenticatedOrder.statusCode, 401);

  for (let attempt = 0; attempt < 10; attempt += 1) {
    const order = await harness.post(
      "/api/create-razorpay-order",
      { ip: "192.0.2.1", body: { amount: 500 }, session: { token: "requester-jwt" } }
    );
    assert.equal(order.statusCode, 200);
  }
  const blockedOrder = await harness.post(
    "/api/create-razorpay-order",
    { ip: "192.0.2.1", body: { amount: 500 }, session: { token: "requester-jwt" } }
  );
  assert.equal(blockedOrder.statusCode, 429);

  for (let attempt = 0; attempt < 10; attempt += 1) {
    const verification = await harness.post(
      "/api/save-razorpay-payment",
      { ip: "192.0.2.1", body: {} }
    );
    assert.equal(verification.statusCode, 200);
  }
  const blockedVerification = await harness.post(
    "/api/save-razorpay-payment",
    { ip: "192.0.2.1", body: {} }
  );
  assert.equal(blockedVerification.statusCode, 429);
  assert.deepEqual(blockedVerification.body, {
    error: "Too many payment attempts. Please try again later.",
  });
});

test("order authentication rejects missing or mismatched requester sessions even when Redis has a JWT", async () => {
  const checkUserSession = createCheckUserSession({
    isOpen: true,
    async get() {
      return "requester-jwt";
    },
  });

  for (const session of [{}, { token: "another-user-jwt" }]) {
    const res = response();
    let nextCalls = 0;
    await checkUserSession(
      { path: "/api/create-razorpay-order", session },
      res,
      () => { nextCalls += 1; }
    );
    assert.equal(res.statusCode, 401);
    assert.deepEqual(res.body, { error: "Authentication required." });
    assert.equal(nextCalls, 0);
  }
});

test("order authentication accepts the requester session whose JWT matches Redis", async () => {
  const checkUserSession = createCheckUserSession({
    isOpen: true,
    async get() {
      return "requester-jwt";
    },
  });
  const res = response();
  let nextCalls = 0;

  await checkUserSession(
    { path: "/api/create-razorpay-order", session: { token: "requester-jwt" } },
    res,
    () => { nextCalls += 1; }
  );

  assert.equal(res.statusCode, 200);
  assert.equal(res.body, undefined);
  assert.equal(nextCalls, 1);
});
