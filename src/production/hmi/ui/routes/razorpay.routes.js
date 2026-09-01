"use strict";

const axios = require("axios");
const crypto = require("node:crypto");

const API_URL = "https://api.razorpay.com/v1";
const ALLOWED_AMOUNTS = new Set([100, 500, 1000, 2000, 5000]);
const PAYMENT_ID = /^pay_[A-Za-z0-9]+$/;
const ORDER_ID = /^order_[A-Za-z0-9]+$/;
const SIGNATURE = /^[a-f0-9]{64}$/i;

function withRazorpayCsp(directives) {
  return {
    ...directives,
    scriptSrc: [...directives.scriptSrc, "https://checkout.razorpay.com"],
    scriptSrcElem: [...directives.scriptSrcElem, "https://checkout.razorpay.com"],
    connectSrc: [...directives.connectSrc, "https://api.razorpay.com", "https://*.razorpay.com"],
    frameSrc: [...directives.frameSrc, "https://api.razorpay.com", "https://*.razorpay.com"],
  };
}

function createOrderRateLimiter(options = {}) {
  const limit = options.limit || 10;
  const windowMs = options.windowMs || 10 * 60 * 1000;
  const now = options.now || Date.now;
  const attempts = new Map();

  // ponytail: per-process/IP state; move to shared session-aware storage if HMI scales horizontally.
  return function limitOrderCreation(req, res, next) {
    const time = now();
    const client = req.ip || req.socket?.remoteAddress || "unknown";
    const current = attempts.get(client);

    if (!current || time - current.startedAt >= windowMs) {
      attempts.set(client, { count: 1, startedAt: time });
      return next();
    }
    if (current.count >= limit) {
      return res.status(429).json({ error: "Too many payment attempts. Please try again later." });
    }

    current.count += 1;
    return next();
  };
}

function credentials(dependencies) {
  return {
    keyId: Object.hasOwn(dependencies, "keyId")
      ? dependencies.keyId
      : process.env.RAZORPAY_KEY_ID,
    keySecret: Object.hasOwn(dependencies, "keySecret")
      ? dependencies.keySecret
      : process.env.RAZORPAY_KEY_SECRET,
  };
}

function requestOptions(keyId, keySecret) {
  return {
    auth: { username: keyId, password: keySecret },
    timeout: 10000,
  };
}

function validCredentials(keyId, keySecret) {
  return typeof keyId === "string" && keyId.trim() &&
    typeof keySecret === "string" && keySecret.trim();
}

function validSignature(orderId, paymentId, signature, keySecret) {
  if (!ORDER_ID.test(orderId) || !PAYMENT_ID.test(paymentId) || !SIGNATURE.test(signature)) {
    return false;
  }

  return validHmac(`${orderId}|${paymentId}`, signature, keySecret);
}

function validHmac(message, signature, secret) {
  if (!SIGNATURE.test(signature)) return false;

  const expected = crypto.createHmac("sha256", secret).update(message).digest();
  const received = Buffer.from(signature, "hex");
  return received.length === expected.length && crypto.timingSafeEqual(received, expected);
}

async function createOrder(req, res, dependencies = {}) {
  const amount = req.body?.amount;
  if (!Number.isInteger(amount) || !ALLOWED_AMOUNTS.has(amount)) {
    return res.status(400).json({ error: "Invalid donation amount." });
  }

  const { keyId, keySecret } = credentials(dependencies);
  if (!validCredentials(keyId, keySecret)) {
    return res.status(503).json({ error: "Payment service is unavailable." });
  }
  if (!dependencies.orders) {
    return res.status(503).json({ error: "Payment service is unavailable." });
  }

  const httpClient = dependencies.httpClient || axios;
  let order;
  try {
    const response = await httpClient.post(
      `${API_URL}/orders`,
      {
        amount,
        currency: "AUD",
        receipt: `echo-donation-${crypto.randomBytes(12).toString("hex")}`,
        notes: { purpose: "donation" },
      },
      requestOptions(keyId, keySecret)
    );
    order = response.data;
    if (!ORDER_ID.test(order?.id) || order.amount !== amount || order.currency !== "AUD") {
      throw new Error("Invalid Razorpay order response");
    }
  } catch (error) {
    console.error("Razorpay order creation failed.");
    return res.status(502).json({ error: "Payment service is unavailable." });
  }

  try {
    await dependencies.orders.updateOne(
      { _id: order.id },
      {
        $setOnInsert: {
          _id: order.id,
          amount: order.amount,
          currency: order.currency,
          purpose: "donation",
          created: order.created_at,
        },
      },
      { upsert: true }
    );
  } catch (error) {
    console.error("Razorpay order persistence failed.");
    return res.status(503).json({ error: "Payment service is unavailable." });
  }

  return res.status(201).json({
    keyId,
    orderId: order.id,
    amount: order.amount,
    currency: order.currency,
  });
}

function verifiedProviderRecords(payment, order, expectedOrder, paymentId) {
  return payment?.id === paymentId &&
    order?.id === expectedOrder._id &&
    payment.order_id === expectedOrder._id &&
    payment.status === "captured" &&
    order.status === "paid" &&
    order.notes?.purpose === "donation" &&
    expectedOrder.purpose === "donation" &&
    payment.currency === "AUD" &&
    order.currency === "AUD" &&
    expectedOrder.currency === "AUD" &&
    payment.amount === order.amount &&
    payment.amount === expectedOrder.amount &&
    ALLOWED_AMOUNTS.has(payment.amount) &&
    order.amount_paid === order.amount &&
    order.amount_due === 0 &&
    Number.isInteger(payment.created_at);
}

async function persistVerifiedPayment(paymentId, orderId, name, res, donations, dependencies) {
  const { keyId, keySecret } = credentials(dependencies);

  if (!validCredentials(keyId, keySecret)) {
    return res.status(503).json({ error: "Payment service is unavailable." });
  }
  if (!ORDER_ID.test(orderId) || !PAYMENT_ID.test(paymentId)) {
    return res.status(400).json({ error: "Payment could not be verified." });
  }
  if (!donations || !dependencies.orders) {
    return res.status(503).json({ error: "Donation could not be recorded." });
  }

  let expectedOrder;
  try {
    expectedOrder = await dependencies.orders.findOne({ _id: orderId });
  } catch (error) {
    console.error("Razorpay order lookup failed.");
    return res.status(503).json({ error: "Donation could not be recorded." });
  }
  if (!expectedOrder) {
    return res.status(400).json({ error: "Payment could not be verified." });
  }

  try {
    const existing = await donations.findOne({ paymentId });
    if (existing) return res.status(200).json({ status: "success" });
  } catch (error) {
    console.error("Donation lookup failed.");
    return res.status(503).json({ error: "Donation could not be recorded." });
  }

  const httpClient = dependencies.httpClient || axios;
  let payment;
  let order;
  try {
    const options = requestOptions(keyId, keySecret);
    const responses = await Promise.all([
      httpClient.get(`${API_URL}/payments/${paymentId}`, options),
      httpClient.get(`${API_URL}/orders/${orderId}`, options),
    ]);
    payment = responses[0].data;
    order = responses[1].data;
  } catch (error) {
    console.error("Razorpay payment verification failed.");
    return res.status(502).json({ error: "Payment service is unavailable." });
  }

  if (!verifiedProviderRecords(payment, order, expectedOrder, paymentId)) {
    return res.status(400).json({ error: "Payment could not be verified." });
  }

  const donation = {
    _id: `razorpay:${payment.id}`,
    paymentId: payment.id,
    orderId: order.id,
    name: typeof name === "string" && name.trim() ? name.trim().slice(0, 100) : "Anonymous",
    email: typeof payment.email === "string" && payment.email ? payment.email : "N/A",
    amount: payment.amount / 100,
    currency: payment.currency.toLowerCase(),
    method: typeof payment.method === "string" && payment.method ? payment.method : "unknown",
    status: "succeeded",
    created: payment.created_at,
  };

  try {
    const result = await donations.updateOne(
      { _id: donation._id },
      { $setOnInsert: donation },
      { upsert: true }
    );
    return res.status(result.upsertedCount === 1 ? 201 : 200).json({ status: "success" });
  } catch (error) {
    console.error("Donation persistence failed.");
    return res.status(503).json({ error: "Donation could not be recorded." });
  }
}

async function savePayment(req, res, donations, dependencies = {}) {
  const { paymentId, orderId, signature, name } = req.body || {};
  const { keyId, keySecret } = credentials(dependencies);

  if (!validCredentials(keyId, keySecret)) {
    return res.status(503).json({ error: "Payment service is unavailable." });
  }
  if (!validSignature(orderId, paymentId, signature, keySecret)) {
    return res.status(400).json({ error: "Payment could not be verified." });
  }

  return persistVerifiedPayment(paymentId, orderId, name, res, donations, dependencies);
}

async function handleWebhook(req, res, donations, dependencies = {}) {
  const webhookSecret = Object.hasOwn(dependencies, "webhookSecret")
    ? dependencies.webhookSecret
    : process.env.RAZORPAY_WEBHOOK_SECRET;
  if (!validCredentials("webhook", webhookSecret)) {
    return res.status(503).json({ error: "Payment service is unavailable." });
  }

  const rawBody = req.body;
  const signature = typeof req.get === "function"
    ? req.get("x-razorpay-signature")
    : req.headers?.["x-razorpay-signature"];
  if (!Buffer.isBuffer(rawBody) || !validHmac(rawBody, signature, webhookSecret)) {
    return res.status(400).json({ error: "Invalid webhook." });
  }

  let event;
  try {
    event = JSON.parse(rawBody.toString("utf8"));
  } catch (error) {
    return res.status(400).json({ error: "Invalid webhook." });
  }

  if (event?.event !== "payment.captured") {
    return res.status(200).json({ status: "ignored" });
  }

  const payment = event.payload?.payment?.entity;
  if (!PAYMENT_ID.test(payment?.id) || !ORDER_ID.test(payment?.order_id)) {
    return res.status(400).json({ error: "Invalid webhook." });
  }

  return persistVerifiedPayment(
    payment.id,
    payment.order_id,
    "Anonymous",
    res,
    donations,
    dependencies
  );
}

module.exports = {
  createOrder,
  createOrderRateLimiter,
  handleWebhook,
  savePayment,
  withRazorpayCsp,
};
