"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { payment } = require("../public/paymentgateway");

const order = {
  keyId: "rzp_test_public",
  orderId: "order_verified123",
  amount: 500,
  currency: "AUD",
};

test("payment builds Checkout from the server-created order", () => {
  const options = payment("Alice", order, "alice@example.com");

  assert.equal(options.key, "rzp_test_public");
  assert.equal(options.order_id, "order_verified123");
  assert.equal(options.amount, 500);
  assert.equal(options.currency, "AUD");
  assert.equal(options.prefill.name, "Alice");
  assert.equal(options.prefill.email, "alice@example.com");
});

test("payment submits only the checkout proof and waits for server verification before success", async (t) => {
  let releaseResponse;
  const fetchCalls = [];
  const toasts = [];
  let settled = 0;

  const originalFetch = global.fetch;
  const originalWindow = global.window;
  t.after(() => {
    global.fetch = originalFetch;
    global.window = originalWindow;
  });

  global.fetch = async (url, options) => {
    fetchCalls.push({ url, options });
    return new Promise(resolve => { releaseResponse = resolve; });
  };
  global.window = { showToast: (message, type) => toasts.push({ message, type }) };

  const options = payment("Alice", order, "alice@example.com", () => { settled += 1; });
  const pending = options.handler({
    razorpay_payment_id: "pay_verified123",
    razorpay_order_id: "order_verified123",
    razorpay_signature: "a".repeat(64),
  });

  assert.deepEqual(toasts, []);
  assert.equal(fetchCalls.length, 1);
  assert.equal(fetchCalls[0].url, "/api/save-razorpay-payment");
  const body = JSON.parse(fetchCalls[0].options.body);
  assert.deepEqual(body, {
    paymentId: "pay_verified123",
    orderId: "order_verified123",
    signature: "a".repeat(64),
    name: "Alice",
  });
  for (const untrustedField of ["amount", "currency", "method", "status", "email"]) {
    assert.equal(Object.hasOwn(body, untrustedField), false);
  }

  releaseResponse({ ok: true, json: async () => ({ status: "success" }) });
  await pending;
  assert.deepEqual(toasts, [{ message: "Donation verified. Thank you!", type: "success" }]);
  assert.equal(settled, 1);
});

test("payment reports failed verification safely and always releases the donate button", async (t) => {
  const toasts = [];
  let settled = 0;
  const originalFetch = global.fetch;
  const originalWindow = global.window;
  t.after(() => {
    global.fetch = originalFetch;
    global.window = originalWindow;
  });

  global.fetch = async () => ({ ok: false, json: async () => ({ error: "internal provider response" }) });
  global.window = { showToast: (message, type) => toasts.push({ message, type }) };

  const options = payment("Alice", order, "alice@example.com", () => { settled += 1; });
  await options.handler({
    razorpay_payment_id: "pay_verified123",
    razorpay_order_id: "order_verified123",
    razorpay_signature: "a".repeat(64),
  });

  assert.deepEqual(toasts, [{ message: "Payment could not be verified. Please contact support.", type: "error" }]);
  assert.equal(settled, 1);
});

