"use strict";

// Authenticated read-only proxy for the admin Detection Review page.
//
// The page needs three Backend reads: the recent-detections list, one
// detection's detail (for its audio), and its similar-detection results. The
// HMI deliberately has no unauthenticated pass-through to the Backend (see the
// note above the /hmi proxy in server.js), so these go through the same
// session-protected path as routes/map.routes.js: the caller must hold the
// current session token, and that token is what gets forwarded upstream.
//
// Mounted under /api/ so an unauthenticated call gets a JSON 401 from
// checkUserSession instead of a redirect to the login page, which fetch()
// would follow and then fail trying to parse as JSON.

const axios = require("axios");
const { checkUserSession } = require("../middleware");
const apiClient = require("../services/apiClient");

const API_BASE_URL = apiClient.API_BASE_URL;
const REQUEST_TIMEOUT_MS = 10000;
const OBJECT_ID_PATTERN = /^[0-9a-fA-F]{24}$/;

function boundedInt(value, fallback, min, max) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

function errorBody(code, message) {
  return { error: { code, message, details: null } };
}

// Upstream error text is never forwarded to the browser; the status picks a
// fixed, safe message instead. Backend auth and budget failures share HTTP 403,
// so the upstream message is only used to tell them apart, same as
// routes/map.routes.js.
function sendUpstreamError(res, error) {
  const status = error.response?.status;
  const upstreamMessage = error.response?.data?.error?.message;
  const isBudget = typeof upstreamMessage === "string" && /budget/i.test(upstreamMessage);

  if (status === 401 || (status === 403 && !isBudget)) {
    return res.status(401).json(errorBody("UNAUTHENTICATED", "Your session has expired. Please log in again."));
  }
  if (status === 403) {
    return res.status(403).json(errorBody("FORBIDDEN", "Detection access is temporarily unavailable. Please contact an administrator."));
  }
  if (status === 400) {
    return res.status(400).json(errorBody("BAD_REQUEST", "The request could not be understood."));
  }
  if (status === 404) {
    return res.status(404).json(errorBody("NOT_FOUND", "Detection not found."));
  }
  if (status === 422) {
    return res.status(422).json(errorBody("UNPROCESSABLE", "This detection has no usable stored embedding to compare with."));
  }
  if (status === 429) {
    return res.status(429).json(errorBody("RATE_LIMITED", "Too many requests. Please wait a moment and try again."));
  }
  if (status === 503) {
    return res.status(503).json(errorBody("SERVICE_UNAVAILABLE", "Detection service is temporarily unavailable. Please try again later."));
  }
  return res.status(502).json(errorBody("UPSTREAM_ERROR", "Could not reach the detection service."));
}

async function forward(req, res, path, params) {
  try {
    const response = await axios.get(`${API_BASE_URL}${path}`, {
      params,
      headers: { Authorization: `Bearer ${req.session.token}` },
      timeout: REQUEST_TIMEOUT_MS,
    });
    return res.json(response.data);
  } catch (error) {
    return sendUpstreamError(res, error);
  }
}

module.exports = function registerDetectionReviewRoutes(app) {
  app.get("/api/detection-review/detections", checkUserSession, (req, res) => {
    // Only these two values ever reach the Backend, clamped to its own limits.
    const params = {
      page: boundedInt(req.query?.page, 1, 1, 10000),
      page_size: boundedInt(req.query?.page_size, 20, 1, 100),
    };
    return forward(req, res, "/detections", params);
  });

  app.get("/api/detection-review/detections/:id", checkUserSession, (req, res) => {
    if (!OBJECT_ID_PATTERN.test(req.params.id)) {
      return res.status(400).json(errorBody("BAD_REQUEST", "Invalid detection id."));
    }
    return forward(req, res, `/detections/${req.params.id}`);
  });

  app.get("/api/detection-review/detections/:id/similar", checkUserSession, (req, res) => {
    if (!OBJECT_ID_PATTERN.test(req.params.id)) {
      return res.status(400).json(errorBody("BAD_REQUEST", "Invalid detection id."));
    }
    const params = { k: boundedInt(req.query?.k, 5, 1, 20) };
    return forward(req, res, `/detections/${req.params.id}/similar`, params);
  });
};
