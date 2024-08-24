const express = require('express');
const router = express.Router();
const axios = require('axios');

const MESSAGE_API_URL = 'http://ts-api-cont:9000/hmi';

// Middleware to set headers
router.use((req, res, next) => {
  res.header(
    "Access-Control-Allow-Headers",
    "Origin, Content-Type, Accept"
  );
  next();
});

// Define routes
router.get(`/movement_time/:start/:end`, async (req, res) => {
  const start = req.params.start;
  const end = req.params.end;
  try {
    const response = await axios.get(`${MESSAGE_API_URL}/movement_time?start=${start}&end=${end}`);
    res.send(response.data.length ? response.data : []);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.get(`/events_time/:start/:end`, async (req, res) => {
  const start = req.params.start;
  const end = req.params.end;
  try {
    const response = await axios.get(`${MESSAGE_API_URL}/events_time?start=${start}&end=${end}`);
    res.send(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.get(`/microphones`, async (req, res) => {
  try {
    const response = await axios.get(`${MESSAGE_API_URL}/microphones`);
    res.send(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.get(`/audio/:id`, async (req, res) => {
  const id = req.params.id;
  try {
    const response = await axios.get(`${MESSAGE_API_URL}/audio?id=${id}`);
    res.send(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.post(`/post_recording`, async (req, res) => {
  try {
    const response = await axios.post(`${MESSAGE_API_URL}/post_recording`, req.body);
    res.send(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.post(`/sim_control/:control`, async (req, res) => {
  const control = req.params.control;
  try {
    const response = await axios.post(`${MESSAGE_API_URL}/sim_control?control=${control}`);
    res.send(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.get(`/latest_movement`, async (req, res) => {
  try {
    const response = await axios.get(`${MESSAGE_API_URL}/latest_movement`);
    res.send(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Export the router
module.exports = router;
