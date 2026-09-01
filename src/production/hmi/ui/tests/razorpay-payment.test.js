"use strict";

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const test = require("node:test");

const {
  createOrder,
  createOrderRateLimiter,
  handleWebhook,
  savePayment,
  withRazorpayCsp,
} = require("../routes/razorpay.routes");

const CAPTURED_WEBHOOK = '{  "entity":"event","account_id":"acc_test123","event":"payment.captured","contains":["payment"],"payload":{"payment":{"entity":{"id":"pay_verified123","entity":"payment","amount":999999,"currency":"USD","status":"captured","order_id":"order_verified123","method":"card","email":"forged@example.com","contact":"+61000000000","created_at":1700000000,"captured":true}}},"created_at":1700000001  }';

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

function providerClient(overrides = {}) {
  const calls = [];
  const payment = {
    id: "pay_verified123",
    entity: "payment",
    amount: 500,
    currency: "AUD",
    status: "captured",
    order_id: "order_verified123",
    method: "card",
    email: "donor@example.com",
    contact: "+61000000000",
    created_at: 1700000000,
    ...overrides.payment,
  };
  const order = {
    id: "order_verified123",
    entity: "order",
    amount: 500,
    amount_paid: 500,
    amount_due: 0,
    currency: "AUD",
    receipt: "echo-donation-0123456789abcdef01234567",
    status: "paid",
    attempts: 1,
    notes: { purpose: "donation" },
    created_at: 1699999990,
    ...overrides.order,
  };

  return {
    calls,
    async post(url, body, options) {
      calls.push({ method: "post", url, body, options });
      if (overrides.postError) throw overrides.postError;
      return { data: order };
    },
    async get(url, options) {
      calls.push({ method: "get", url, options });
      if (overrides.getError) throw overrides.getError;
      return { data: url.includes("/payments/") ? payment : order };
    },
  };
}

function collection() {
  const records = new Map();
  return {
    records,
    updateCalls: 0,
    async findOne(query) {
      if (query._id) return records.get(query._id) || null;
      for (const record of records.values()) {
        if (record.paymentId === query.paymentId) return record;
      }
      return null;
    },
    async updateOne(filter, update) {
      this.updateCalls += 1;
      if (records.has(filter._id)) return { matchedCount: 1, upsertedCount: 0 };
      records.set(filter._id, { ...update.$setOnInsert });
      return { matchedCount: 0, upsertedCount: 1 };
    },
  };
}

function pendingOrders() {
  const orders = collection();
  orders.records.set("order_verified123", {
    _id: "order_verified123",
    amount: 500,
    currency: "AUD",
    purpose: "donation",
    created: 1699999990,
  });
  return orders;
}

function signature(secret = "test_secret") {
  return crypto
    .createHmac("sha256", secret)
    .update("order_verified123|pay_verified123")
    .digest("hex");
}

function webhookSignature(body, secret = "webhook_secret") {
  return crypto.createHmac("sha256", secret).update(body).digest("hex");
}

function webhookRequest(body, signature = webhookSignature(body)) {
  return {
    body: Buffer.from(body),
    get(name) {
      return name.toLowerCase() === "x-razorpay-signature" ? signature : undefined;
    },
  };
}

function paymentRequest(body = {}) {
  return {
    body: {
      paymentId: "pay_verified123",
      orderId: "order_verified123",
      signature: signature(),
      name: " Alice ",
      ...body,
    },
  };
}

test("createOrder rejects amounts outside the five donation choices without calling Razorpay", async () => {
  for (const amount of [undefined, null, "500", 0, 101, 500.5, 10000]) {
    const httpClient = providerClient();
    const res = response();

    await createOrder({ body: { amount } }, res, {
      httpClient,
      keyId: "rzp_test_key",
      keySecret: "test_secret",
      orders: collection(),
    });

    assert.equal(res.statusCode, 400);
    assert.deepEqual(res.body, { error: "Invalid donation amount." });
    assert.equal(httpClient.calls.length, 0);
  }
});

test("createOrder creates and retains each supported AUD donation order without exposing the secret", async () => {
  for (const amount of [100, 500, 1000, 2000, 5000]) {
    const httpClient = providerClient({ order: { amount, amount_paid: amount } });
    const orders = collection();
    const res = response();

    await createOrder({ body: { amount } }, res, {
      httpClient,
      keyId: "rzp_test_key",
      keySecret: "test_secret",
      orders,
    });

    assert.equal(res.statusCode, 201);
    assert.deepEqual(res.body, {
      keyId: "rzp_test_key",
      orderId: "order_verified123",
      amount,
      currency: "AUD",
    });
    assert.equal(JSON.stringify(res.body).includes("test_secret"), false);
    assert.deepEqual(orders.records.get("order_verified123"), {
      _id: "order_verified123",
      amount,
      currency: "AUD",
      purpose: "donation",
      created: 1699999990,
    });
    assert.equal(httpClient.calls.length, 1);
    assert.equal(httpClient.calls[0].url, "https://api.razorpay.com/v1/orders");
    assert.equal(httpClient.calls[0].body.amount, amount);
    assert.equal(httpClient.calls[0].body.currency, "AUD");
    assert.deepEqual(httpClient.calls[0].body.notes, { purpose: "donation" });
    assert.match(httpClient.calls[0].body.receipt, /^echo-donation-[a-f0-9]{24}$/);
    assert.ok(httpClient.calls[0].body.receipt.length <= 40);
    assert.deepEqual(httpClient.calls[0].options.auth, {
      username: "rzp_test_key",
      password: "test_secret",
    });
  }
});

test("createOrder reports missing configuration and provider failure without leaking details", async () => {
  const missingConfig = response();
  await createOrder({ body: { amount: 500 } }, missingConfig, {
    httpClient: providerClient(),
    keyId: "",
    keySecret: "",
    orders: collection(),
  });
  assert.equal(missingConfig.statusCode, 503);
  assert.deepEqual(missingConfig.body, { error: "Payment service is unavailable." });

  const providerFailure = response();
  await createOrder({ body: { amount: 500 } }, providerFailure, {
    httpClient: providerClient({ postError: new Error("secret provider response") }),
    keyId: "rzp_test_key",
    keySecret: "test_secret",
    orders: collection(),
  });
  assert.equal(providerFailure.statusCode, 502);
  assert.deepEqual(providerFailure.body, { error: "Payment service is unavailable." });
  assert.equal(JSON.stringify(providerFailure.body).includes("secret"), false);
});

test("createOrder does not return an order that could not be retained server-side", async () => {
  const orders = collection();
  orders.updateOne = async () => { throw new Error("database unavailable"); };
  const res = response();

  await createOrder({ body: { amount: 500 } }, res, {
    httpClient: providerClient(),
    keyId: "rzp_test_key",
    keySecret: "test_secret",
    orders,
  });

  assert.equal(res.statusCode, 503);
  assert.deepEqual(res.body, { error: "Payment service is unavailable." });
});

test("savePayment rejects the legacy browser-trusted payload before any provider or database call", async () => {
  const httpClient = providerClient();
  const donations = collection();
  const res = response();

  await savePayment(
    { body: { paymentId: "pay_fake", amount: 999999, currency: "AUD" } },
    res,
    donations,
    { httpClient, keyId: "rzp_test_key", keySecret: "test_secret", orders: pendingOrders() }
  );

  assert.equal(res.statusCode, 400);
  assert.deepEqual(res.body, { error: "Payment could not be verified." });
  assert.equal(httpClient.calls.length, 0);
  assert.equal(donations.updateCalls, 0);
});

test("savePayment rejects malformed identifiers and signatures before contacting Razorpay", async () => {
  const malformedBodies = [
    { paymentId: "../payment", orderId: "order_verified123", signature: signature() },
    { paymentId: "pay_verified123", orderId: "../order", signature: signature() },
    { paymentId: "pay_verified123", orderId: "order_verified123", signature: "not-a-signature" },
  ];

  for (const body of malformedBodies) {
    const httpClient = providerClient();
    const res = response();
    await savePayment({ body }, res, collection(), {
      httpClient,
      keyId: "rzp_test_key",
      keySecret: "test_secret",
      orders: pendingOrders(),
    });
    assert.equal(res.statusCode, 400);
    assert.equal(httpClient.calls.length, 0);
  }
});

test("savePayment rejects an invalid checkout signature before contacting Razorpay", async () => {
  const httpClient = providerClient();
  const donations = collection();
  const res = response();

  await savePayment(paymentRequest({ signature: "0".repeat(64) }), res, donations, {
    httpClient,
    keyId: "rzp_test_key",
    keySecret: "test_secret",
    orders: pendingOrders(),
  });

  assert.equal(res.statusCode, 400);
  assert.equal(httpClient.calls.length, 0);
  assert.equal(donations.updateCalls, 0);
});

test("savePayment rejects a callback for an order the server did not create", async () => {
  const httpClient = providerClient();
  const donations = collection();
  const res = response();

  await savePayment(paymentRequest(), res, donations, {
    httpClient,
    keyId: "rzp_test_key",
    keySecret: "test_secret",
    orders: collection(),
  });

  assert.equal(res.statusCode, 400);
  assert.deepEqual(res.body, { error: "Payment could not be verified." });
  assert.equal(httpClient.calls.length, 0);
  assert.equal(donations.updateCalls, 0);
});

test("savePayment rejects incomplete or mismatched provider records without writing", async () => {
  const cases = [
    { payment: { status: "authorized" } },
    { order: { status: "attempted" } },
    { order: { notes: {} } },
    { payment: { order_id: "order_other" } },
    { payment: { amount: 100 } },
    { payment: { currency: "INR" } },
    { order: { amount: 101 } },
  ];

  for (const providerOverride of cases) {
    const donations = collection();
    const res = response();
    await savePayment(paymentRequest(), res, donations, {
      httpClient: providerClient(providerOverride),
      keyId: "rzp_test_key",
      keySecret: "test_secret",
      orders: pendingOrders(),
    });
    assert.equal(res.statusCode, 400);
    assert.deepEqual(res.body, { error: "Payment could not be verified." });
    assert.equal(donations.updateCalls, 0);
  }
});

test("savePayment stores authoritative Razorpay fields and ignores forged browser financial fields", async () => {
  const donations = collection();
  const res = response();

  await savePayment(
    paymentRequest({ amount: 999999, currency: "USD", method: "Cash", status: "succeeded", email: "fake@example.com" }),
    res,
    donations,
    { httpClient: providerClient(), keyId: "rzp_test_key", keySecret: "test_secret", orders: pendingOrders() }
  );

  assert.equal(res.statusCode, 201);
  assert.deepEqual(res.body, { status: "success" });
  assert.deepEqual(donations.records.get("razorpay:pay_verified123"), {
    _id: "razorpay:pay_verified123",
    paymentId: "pay_verified123",
    orderId: "order_verified123",
    name: "Alice",
    email: "donor@example.com",
    amount: 5,
    currency: "aud",
    method: "card",
    status: "succeeded",
    created: 1700000000,
  });
});

test("savePayment treats historical and concurrent replay as success without another donation", async () => {
  const historical = collection();
  historical.records.set("legacy-id", { _id: "legacy-id", paymentId: "pay_verified123", amount: 5 });
  const historicalResponse = response();
  await savePayment(paymentRequest(), historicalResponse, historical, {
    httpClient: providerClient(),
    keyId: "rzp_test_key",
    keySecret: "test_secret",
    orders: pendingOrders(),
  });
  assert.equal(historicalResponse.statusCode, 200);
  assert.equal(historical.records.size, 1);

  const concurrent = collection();
  const first = response();
  const second = response();
  await Promise.all([
    savePayment(paymentRequest(), first, concurrent, {
      httpClient: providerClient(),
      keyId: "rzp_test_key",
      keySecret: "test_secret",
      orders: pendingOrders(),
    }),
    savePayment(paymentRequest(), second, concurrent, {
      httpClient: providerClient(),
      keyId: "rzp_test_key",
      keySecret: "test_secret",
      orders: pendingOrders(),
    }),
  ]);
  assert.equal(concurrent.records.size, 1);
  assert.deepEqual([first.statusCode, second.statusCode].sort(), [200, 201]);
});

test("savePayment reports provider and database failures without recording success", async () => {
  const providerFailure = response();
  await savePayment(paymentRequest(), providerFailure, collection(), {
    httpClient: providerClient({ getError: new Error("provider credentials leaked") }),
    keyId: "rzp_test_key",
    keySecret: "test_secret",
    orders: pendingOrders(),
  });
  assert.equal(providerFailure.statusCode, 502);
  assert.deepEqual(providerFailure.body, { error: "Payment service is unavailable." });

  const failingCollection = collection();
  failingCollection.updateOne = async () => { throw new Error("mongodb://internal-host"); };
  const databaseFailure = response();
  await savePayment(paymentRequest(), databaseFailure, failingCollection, {
    httpClient: providerClient(),
    keyId: "rzp_test_key",
    keySecret: "test_secret",
    orders: pendingOrders(),
  });
  assert.equal(databaseFailure.statusCode, 503);
  assert.deepEqual(databaseFailure.body, { error: "Donation could not be recorded." });
  assert.equal(JSON.stringify(databaseFailure.body).includes("mongodb"), false);
});

test("handleWebhook requires a configured webhook secret before processing", async () => {
  const httpClient = providerClient();
  const donations = collection();
  const res = response();

  await handleWebhook(webhookRequest(CAPTURED_WEBHOOK), res, donations, {
    httpClient,
    keyId: "rzp_test_key",
    keySecret: "test_secret",
    webhookSecret: "",
    orders: pendingOrders(),
  });

  assert.equal(res.statusCode, 503);
  assert.equal(httpClient.calls.length, 0);
  assert.equal(donations.updateCalls, 0);
});

test("handleWebhook rejects missing or non-matching signatures before processing", async () => {
  const signedBody = CAPTURED_WEBHOOK;
  const changedBody = CAPTURED_WEBHOOK.replace("999999", "999998");
  const httpClient = providerClient();
  const donations = collection();
  const missing = response();
  const res = response();
  const dependencies = {
    httpClient,
    keyId: "rzp_test_key",
    keySecret: "test_secret",
    webhookSecret: "webhook_secret",
    orders: pendingOrders(),
  };

  await handleWebhook(webhookRequest(signedBody, null), missing, donations, dependencies);

  await handleWebhook(
    webhookRequest(changedBody, webhookSignature(signedBody)),
    res,
    donations,
    dependencies
  );

  assert.equal(missing.statusCode, 400);
  assert.equal(res.statusCode, 400);
  assert.equal(httpClient.calls.length, 0);
  assert.equal(donations.updateCalls, 0);
});

test("handleWebhook rejects malformed signed JSON before provider or database calls", async () => {
  const body = '{"event":"payment.captured"';
  const httpClient = providerClient();
  const donations = collection();
  const res = response();

  await handleWebhook(webhookRequest(body), res, donations, {
    httpClient,
    keyId: "rzp_test_key",
    keySecret: "test_secret",
    webhookSecret: "webhook_secret",
    orders: pendingOrders(),
  });

  assert.equal(res.statusCode, 400);
  assert.equal(httpClient.calls.length, 0);
  assert.equal(donations.updateCalls, 0);
});

test("handleWebhook rejects malformed payment.captured identifiers without writes", async () => {
  const payload = JSON.parse(CAPTURED_WEBHOOK);
  payload.payload.payment.entity.id = "../payment";
  const body = JSON.stringify(payload);
  const httpClient = providerClient();
  const donations = collection();
  const res = response();

  await handleWebhook(webhookRequest(body), res, donations, {
    httpClient,
    keyId: "rzp_test_key",
    keySecret: "test_secret",
    webhookSecret: "webhook_secret",
    orders: pendingOrders(),
  });

  assert.equal(res.statusCode, 400);
  assert.equal(httpClient.calls.length, 0);
  assert.equal(donations.updateCalls, 0);
});

test("handleWebhook acknowledges a signed unrelated event without side effects", async () => {
  const body = '{"entity":"event","event":"payment.failed","payload":{}}';
  const httpClient = providerClient();
  const donations = collection();
  const res = response();

  await handleWebhook(webhookRequest(body), res, donations, {
    httpClient,
    keyId: "rzp_test_key",
    keySecret: "test_secret",
    webhookSecret: "webhook_secret",
    orders: pendingOrders(),
  });

  assert.equal(res.statusCode, 200);
  assert.deepEqual(res.body, { status: "ignored" });
  assert.equal(httpClient.calls.length, 0);
  assert.equal(donations.updateCalls, 0);
});

test("handleWebhook stores authoritative Razorpay records for payment.captured", async () => {
  const donations = collection();
  const res = response();

  await handleWebhook(webhookRequest(CAPTURED_WEBHOOK), res, donations, {
    httpClient: providerClient(),
    keyId: "rzp_test_key",
    keySecret: "test_secret",
    webhookSecret: "webhook_secret",
    orders: pendingOrders(),
  });

  assert.equal(res.statusCode, 201);
  assert.deepEqual(res.body, { status: "success" });
  assert.deepEqual(donations.records.get("razorpay:pay_verified123"), {
    _id: "razorpay:pay_verified123",
    paymentId: "pay_verified123",
    orderId: "order_verified123",
    name: "Anonymous",
    email: "donor@example.com",
    amount: 5,
    currency: "aud",
    method: "card",
    status: "succeeded",
    created: 1700000000,
  });
});

test("handleWebhook and savePayment concurrently persist one donation", async () => {
  const donations = collection();
  const webhookResponse = response();
  const checkoutResponse = response();
  const replayResponse = response();
  const orders = pendingOrders();

  await Promise.all([
    handleWebhook(webhookRequest(CAPTURED_WEBHOOK), webhookResponse, donations, {
      httpClient: providerClient(),
      keyId: "rzp_test_key",
      keySecret: "test_secret",
      webhookSecret: "webhook_secret",
      orders,
    }),
    savePayment(paymentRequest(), checkoutResponse, donations, {
      httpClient: providerClient(),
      keyId: "rzp_test_key",
      keySecret: "test_secret",
      orders,
    }),
  ]);

  await handleWebhook(webhookRequest(CAPTURED_WEBHOOK), replayResponse, donations, {
    httpClient: providerClient(),
    keyId: "rzp_test_key",
    keySecret: "test_secret",
    webhookSecret: "webhook_secret",
    orders,
  });

  assert.equal(donations.records.size, 1);
  assert.deepEqual([webhookResponse.statusCode, checkoutResponse.statusCode].sort(), [200, 201]);
  assert.equal(replayResponse.statusCode, 200);
});

test("handleWebhook returns transient provider and database failures for retry", async () => {
  const providerFailure = response();
  await handleWebhook(webhookRequest(CAPTURED_WEBHOOK), providerFailure, collection(), {
    httpClient: providerClient({ getError: new Error("provider unavailable") }),
    keyId: "rzp_test_key",
    keySecret: "test_secret",
    webhookSecret: "webhook_secret",
    orders: pendingOrders(),
  });
  assert.equal(providerFailure.statusCode, 502);

  const donations = collection();
  donations.findOne = async () => { throw new Error("database unavailable"); };
  const databaseFailure = response();
  await handleWebhook(webhookRequest(CAPTURED_WEBHOOK), databaseFailure, donations, {
    httpClient: providerClient(),
    keyId: "rzp_test_key",
    keySecret: "test_secret",
    webhookSecret: "webhook_secret",
    orders: pendingOrders(),
  });
  assert.equal(databaseFailure.statusCode, 503);
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

test("createOrderRateLimiter bounds order creation per client and resets after its window", () => {
  let now = 1000;
  let nextCalls = 0;
  const limiter = createOrderRateLimiter({
    limit: 2,
    windowMs: 1000,
    now: () => now,
  });
  const req = { ip: "192.0.2.1" };

  limiter(req, response(), () => { nextCalls += 1; });
  limiter(req, response(), () => { nextCalls += 1; });
  const blocked = response();
  limiter(req, blocked, () => { nextCalls += 1; });

  assert.equal(nextCalls, 2);
  assert.equal(blocked.statusCode, 429);
  assert.deepEqual(blocked.body, { error: "Too many payment attempts. Please try again later." });

  now = 2001;
  limiter(req, response(), () => { nextCalls += 1; });
  assert.equal(nextCalls, 3);
});
