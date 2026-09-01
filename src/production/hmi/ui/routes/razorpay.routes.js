"use strict";

const axios = require("axios");


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


function backendUrl(dependencies, path) {
  if (typeof dependencies.apiBaseUrl !== "string" || !dependencies.apiBaseUrl) {
    throw new Error("Backend URL is unavailable");
  }
  return `${dependencies.apiBaseUrl.replace(/\/$/, "")}${path}`;
}


function isJsonObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}


function forwardResponse(res, response, safeError) {
  if (
    !Number.isInteger(response?.status) ||
    response.status < 200 ||
    response.status > 299 ||
    !isJsonObject(response.data)
  ) {
    return res.status(503).json({ error: safeError });
  }
  return res.status(response.status).json(response.data);
}


function forwardFailure(res, error, safeError) {
  const status = error?.response?.status;
  const body = error?.response?.data;
  const knownStatus = Number.isInteger(status) && status >= 400 && status <= 599;

  if (knownStatus && isJsonObject(body) && typeof body.error === "string") {
    return res.status(status).json({ error: body.error });
  }
  if (knownStatus && isJsonObject(body) && typeof body.status === "string") {
    return res.status(status).json({ status: body.status });
  }
  console.error("Payment Backend request failed.");
  return res.status(503).json({ error: safeError });
}


async function createOrder(req, res, dependencies = {}) {
  const httpClient = dependencies.httpClient || axios;
  try {
    const response = await httpClient.post(
      backendUrl(dependencies, "/payments/razorpay/orders"),
      { amount: req.body?.amount },
      {
        headers: { Authorization: `Bearer ${req.session?.token}` },
        timeout: 10000,
      }
    );
    return forwardResponse(res, response, "Payment service is unavailable.");
  } catch (error) {
    return forwardFailure(res, error, "Payment service is unavailable.");
  }
}


async function savePayment(req, res, dependencies = {}) {
  const httpClient = dependencies.httpClient || axios;
  const { paymentId, orderId, signature, name } = req.body || {};
  try {
    const response = await httpClient.post(
      backendUrl(dependencies, "/payments/razorpay/verify"),
      { paymentId, orderId, signature, name },
      { timeout: 10000 }
    );
    return forwardResponse(res, response, "Donation could not be recorded.");
  } catch (error) {
    return forwardFailure(res, error, "Donation could not be recorded.");
  }
}


async function handleWebhook(req, res, dependencies = {}) {
  const httpClient = dependencies.httpClient || axios;
  const signature = typeof req.get === "function"
    ? req.get("x-razorpay-signature")
    : req.headers?.["x-razorpay-signature"];
  try {
    const response = await httpClient.post(
      backendUrl(dependencies, "/payments/razorpay/webhook"),
      req.body,
      {
        headers: {
          "Content-Type": "application/json",
          "x-razorpay-signature": signature,
        },
        timeout: 10000,
      }
    );
    return forwardResponse(res, response, "Donation could not be recorded.");
  } catch (error) {
    return forwardFailure(res, error, "Donation could not be recorded.");
  }
}


module.exports = {
  createOrder,
  createOrderRateLimiter,
  handleWebhook,
  savePayment,
  withRazorpayCsp,
};
