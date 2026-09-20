const { verifySignUp, client, checkUserSession } = require("../middleware");
const axios = require('axios');
const apiClient = require('../services/apiClient');
require('dotenv').config();
const MESSAGE_API_URL = '/hmi';
const API_BASE_URL = apiClient.API_BASE_URL;

module.exports = function(app) {
  app.get("/api/detections", checkUserSession, async (req, res) => {
    try {
      const { sourceType } = req.query || {};
      const params = { limit: 100 };
      // Forward only validated values; "all"/missing/unknown omits sourceType
      // so the backend default (all-behaviour) applies. Never forward "all".
      if (sourceType === "real" || sourceType === "simulator") params.sourceType = sourceType;
      const response = await axios.get(`${API_BASE_URL}/hmi/latest_events`, {
        params,
        headers: { Authorization: `Bearer ${req.session.token}` },
        timeout: 10000,
      });
      res.json(response.data);
    } catch (error) {
      // Backend failure classes share HTTP 403 (auth vs budget), so the
      // upstream message — never forwarded — picks the safe user message.
      // Coupling note: matches the two stable budget strings in
      // backend/app/services/budget.py ("Budget is not configured ...",
      // "Budget exceeded ..."). Auth failures mean the session is stale, so
      // they answer 401 and the UI prompts a re-login.
      const upstreamStatus = error.response?.status;
      const upstreamMessage = error.response?.data?.error?.message;
      const isBudget = typeof upstreamMessage === "string" && /budget/i.test(upstreamMessage);
      if (upstreamStatus === 401 || (upstreamStatus === 403 && !isBudget)) {
        return res.status(401).json({ error: {
          code: "UNAUTHENTICATED",
          message: "Your session has expired. Please log in again.",
          details: null,
        } });
      }
      if (upstreamStatus === 403) {
        return res.status(403).json({ error: {
          code: "FORBIDDEN",
          message: "Detection access is temporarily unavailable. Please contact an administrator.",
          details: null,
        } });
      }
      if (upstreamStatus === 429) {
        return res.status(429).json({ error: {
          code: "RATE_LIMITED",
          message: "Too many requests. Please wait a moment and try again.",
          details: null,
        } });
      }
      if (upstreamStatus === 503) {
        return res.status(503).json({ error: {
          code: "SERVICE_UNAVAILABLE",
          message: "Detection service is temporarily unavailable. Please try again later.",
          details: null,
        } });
      }
      return res.status(502).json({ error: {
        code: "UPSTREAM_ERROR",
        message: "Detections are currently unavailable.",
        details: null,
      } });
    }
  });

  app.get("/api/weather", checkUserSession, async (req, res) => {
    try {
      const { timestamp, lat, lon } = req.query || {};
      if (timestamp === undefined || lat === undefined || lon === undefined) {
        return res.status(400).json({ error: {
          code: "BAD_REQUEST",
          message: "Weather data is currently unavailable.",
          details: null,
        } });
      }
      const response = await axios.get(`${API_BASE_URL}/hmi/weather`, {
        params: { timestamp, lat, lon },
        headers: { Authorization: `Bearer ${req.session.token}` },
        timeout: 10000,
      });
      res.send(response.data);
    } catch (err) {
      if (!res.headersSent) res.status(502).json({ error: 'API unavailable' });
    }
  });

  app.use(function(req, res, next) {
    res.header("Access-Control-Allow-Headers", "Origin, Content-Type, Accept");
    next();
  });

  app.get(`/movement_time/:start/:end`, checkUserSession, async (req, res) => {
    try {
      const data = await apiClient.get(`${MESSAGE_API_URL}/movement_time`, {
        params: { start: req.params.start, end: req.params.end },
      });
      res.send(data || []);
    } catch (err) {
      if (!res.headersSent) apiClient.sendApiError(res, err);
    }
  });

  app.get(`/movement_time_daily/:start/:end`, checkUserSession, async (req, res) => {
    try {
      const data = await apiClient.get(`${MESSAGE_API_URL}/movement_time_daily`, {
        params: { start: req.params.start, end: req.params.end },
      });
      res.send(data || []);
    } catch (err) {
      if (!res.headersSent) apiClient.sendApiError(res, err);
    }
  });

  app.get(`/events_time/:start/:end`, checkUserSession, async (req, res) => {
    try {
      const data = await apiClient.get(`${MESSAGE_API_URL}/events_time`, {
        params: { start: req.params.start, end: req.params.end },
      });
      res.send(data || []);
    } catch (err) {
      if (!res.headersSent) apiClient.sendApiError(res, err);
    }
  });

  app.get(`/microphones`, checkUserSession, async (req, res) => {
    try {
      const data = await apiClient.get(`${MESSAGE_API_URL}/microphones`);
      res.send(data);
    } catch (err) {
      if (!res.headersSent) apiClient.sendApiError(res, err);
    }
  });

  app.get(`/audio/:id`, checkUserSession, async (req, res) => {
    try {
      const data = await apiClient.get(`${MESSAGE_API_URL}/audio`, { params: { id: req.params.id } });
      res.send(data);
    } catch (err) {
      if (!res.headersSent) apiClient.sendApiError(res, err);
    }
  });

  app.post(`/post_recording`, checkUserSession, async (req, res) => {
    try {
      const data = await apiClient.post(`${MESSAGE_API_URL}/post_recording`, req.body);
      res.send(data);
    } catch (err) {
      if (!res.headersSent) apiClient.sendApiError(res, err);
    }
  });

  app.post(`/sim_control/:control`, checkUserSession, async (req, res) => {
    try {
      const data = await apiClient.post(`${MESSAGE_API_URL}/sim_control`, undefined, {
        params: { control: req.params.control },
      });
      res.send(data);
    } catch (err) {
      if (!res.headersSent) apiClient.sendApiError(res, err);
    }
  });

  // Sensor Health routes (/sensors/*) are proxied to the Backend API in server.js.
  // Do not re-register them here — the first Express match would shadow live data.

  app.get(`/latest_movement`, checkUserSession, async (req, res) => {
    try {
      const data = await apiClient.get(`${MESSAGE_API_URL}/latest_movement`);
      res.send(data);
    } catch (err) {
      if (!res.headersSent) apiClient.sendApiError(res, err);
    }
  });
}
