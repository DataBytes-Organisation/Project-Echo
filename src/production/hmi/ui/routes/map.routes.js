const { verifySignUp, client } = require("../middleware");
const apiClient = require('../services/apiClient');
require('dotenv').config();
const MESSAGE_API_URL = '/hmi';

module.exports = function(app) {
  app.use(function(req, res, next) {
    res.header("Access-Control-Allow-Headers", "Origin, Content-Type, Accept");
    next();
  });

  app.get(`/movement_time/:start/:end`, async (req, res) => {
    try {
      const data = await apiClient.get(`${MESSAGE_API_URL}/movement_time`, {
        params: { start: req.params.start, end: req.params.end },
      });
      res.send(data || []);
    } catch (err) {
      if (!res.headersSent) apiClient.sendApiError(res, err);
    }
  });

  app.get(`/movement_time_daily/:start/:end`, async (req, res) => {
    try {
      const data = await apiClient.get(`${MESSAGE_API_URL}/movement_time_daily`, {
        params: { start: req.params.start, end: req.params.end },
      });
      res.send(data || []);
    } catch (err) {
      if (!res.headersSent) apiClient.sendApiError(res, err);
    }
  });

  app.get(`/events_time/:start/:end`, async (req, res) => {
    try {
      const data = await apiClient.get(`${MESSAGE_API_URL}/events_time`, {
        params: { start: req.params.start, end: req.params.end },
      });
      res.send(data || []);
    } catch (err) {
      if (!res.headersSent) apiClient.sendApiError(res, err);
    }
  });

  app.get(`/microphones`, async (req, res) => {
    try {
      const data = await apiClient.get(`${MESSAGE_API_URL}/microphones`);
      res.send(data);
    } catch (err) {
      if (!res.headersSent) apiClient.sendApiError(res, err);
    }
  });

  app.get(`/audio/:id`, async (req, res) => {
    try {
      const data = await apiClient.get(`${MESSAGE_API_URL}/audio`, { params: { id: req.params.id } });
      res.send(data);
    } catch (err) {
      if (!res.headersSent) apiClient.sendApiError(res, err);
    }
  });

  app.post(`/post_recording`, async (req, res) => {
    try {
      const data = await apiClient.post(`${MESSAGE_API_URL}/post_recording`, req.body);
      res.send(data);
    } catch (err) {
      if (!res.headersSent) apiClient.sendApiError(res, err);
    }
  });

  app.post(`/sim_control/:control`, async (req, res) => {
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

  app.get(`/latest_movement`, async (req, res) => {
    try {
      const data = await apiClient.get(`${MESSAGE_API_URL}/latest_movement`);
      res.send(data);
    } catch (err) {
      if (!res.headersSent) apiClient.sendApiError(res, err);
    }
  });
}