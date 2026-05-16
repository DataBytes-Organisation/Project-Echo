const { verifySignUp, client } = require("../middleware");
const axios = require('axios');
require('dotenv').config();
const API_BASE_URL = `http://${process.env.API_HOST || 'localhost'}:9001`;
const MESSAGE_API_URL = `${API_BASE_URL}/hmi`;

module.exports = function(app) {
  app.use(function(req, res, next) {
    res.header("Access-Control-Allow-Headers", "Origin, Content-Type, Accept");
    next();
  });

  app.get(`/movement_time/:start/:end`, async (req, res) => {
    try {
      const response = await axios.get(`${MESSAGE_API_URL}/movement_time?start=${req.params.start}&end=${req.params.end}`);
      res.send(response.data || []);
    } catch (err) {
      if (!res.headersSent) res.status(502).json({ error: 'API unavailable' });
    }
  });

  app.get(`/movement_time_daily/:start/:end`, async (req, res) => {
    try {
      const response = await axios.get(`${MESSAGE_API_URL}/movement_time_daily?start=${req.params.start}&end=${req.params.end}`);
      res.send(response.data || []);
    } catch (err) {
      if (!res.headersSent) res.status(502).json({ error: 'API unavailable' });
    }
  });

  app.get(`/events_time/:start/:end`, async (req, res) => {
    try {
      const response = await axios.get(`${MESSAGE_API_URL}/events_time?start=${req.params.start}&end=${req.params.end}`);
      res.send(response.data || []);
    } catch (err) {
      if (!res.headersSent) res.status(502).json({ error: 'API unavailable' });
    }
  });

  app.get(`/microphones`, async (req, res) => {
    try {
      const response = await axios.get(`${MESSAGE_API_URL}/microphones`);
      res.send(response.data);
    } catch (err) {
      if (!res.headersSent) res.status(502).json({ error: 'API unavailable' });
    }
  });

  app.get(`/audio/:id`, async (req, res) => {
    try {
      const response = await axios.get(`${MESSAGE_API_URL}/audio?id=${req.params.id}`);
      res.send(response.data);
    } catch (err) {
      if (!res.headersSent) res.status(502).json({ error: 'API unavailable' });
    }
  });

  app.post(`/post_recording`, async (req, res) => {
    try {
      const response = await axios.post(`${MESSAGE_API_URL}/post_recording`, req.body);
      res.send(response.data);
    } catch (err) {
      if (!res.headersSent) res.status(502).json({ error: 'API unavailable' });
    }
  });

  app.post(`/sim_control/:control`, async (req, res) => {
    try {
      const response = await axios.post(`${MESSAGE_API_URL}/sim_control?control=${req.params.control}`);
      res.send(response.data);
    } catch (err) {
      if (!res.headersSent) res.status(502).json({ error: 'API unavailable' });
    }
  });

  app.get(`/latest_movement`, async (req, res) => {
    try {
      const response = await axios.get(`${MESSAGE_API_URL}/latest_movement`);
      res.send(response.data);
    } catch (err) {
      if (!res.headersSent) res.status(502).json({ error: 'API unavailable' });
    }
  });
}