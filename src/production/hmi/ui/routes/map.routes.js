const { verifySignUp, client, checkUserSession } = require("../middleware");
const axios = require('axios');
require('dotenv').config();
const API_BASE_URL = `http://${process.env.API_HOST || 'localhost'}:9000`;
const MESSAGE_API_URL = `${API_BASE_URL}/hmi`;

// Module-scoped sensor state. Set by your MQTT handler elsewhere;
// declared here so the /sensors routes don't reference an undeclared global.
let latestSensorData = null;

module.exports = function(app) {
  app.get("/api/detections", checkUserSession, async (req, res) => {
    try {
      const response = await axios.get(`${MESSAGE_API_URL}/detections`, {
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

  // ---- Sensor Health backend (from sensor-dashboard branch) ----

  // Sprint 2 demo/test seed route
  app.get("/sensors/test-seed", (req, res) => {
    latestSensorData = {
      items: [
        {
          sensorId: "LIVE-001",
          status: "Online",
          batteryPct: 78,
          cpu: 37,
          ram: 48,
          disk: 62,
          uptime: 15432,
          gps: { lat: -38.1499, lon: 144.3617 },
          lastAudio: "audio_sensor_001.wav",
        },
        {
          sensorId: "LIVE-002",
          status: "Offline",
          batteryPct: 55,
          cpu: 0,
          ram: 0,
          disk: 0,
          uptime: 0,
          gps: { lat: -38.1605, lon: 144.3502 },
          lastAudio: "—",
        },
        {
          sensorId: "LIVE-003",
          status: "Online",
          batteryPct: 15,
          cpu: 52,
          ram: 63,
          disk: 70,
          uptime: 28765,
          gps: { lat: -38.1422, lon: 144.3728 },
          lastAudio: "audio_sensor_003.wav",
        },
      ],
    };

    res.json({
      success: true,
      message: "Test sensor payload seeded successfully.",
      latestSensorData,
    });
  });

  app.get("/sensors/updates", async (req, res) => {
    try {
      if (!latestSensorData) {
        return res.json({ items: [] });
      }

      // If Sprint 2 seeded/demo data exists, return directly
      if (Array.isArray(latestSensorData.items)) {
        return res.json({ items: latestSensorData.items });
      }

      // Otherwise treat it as live MQTT payload structure
      return res.json({
        items: [
          {
            sensorId: "LIVE-001",
            status: "Online",
            batteryPct: 18, // temporary fallback until real battery field is available
            cpu: latestSensorData.health_data?.cpu,
            ram: latestSensorData.health_data?.ram,
            disk: latestSensorData.health_data?.disk,
            uptime: latestSensorData.health_data?.uptime,
            gps: latestSensorData.gps_data,
            lastAudio: latestSensorData.savedAudio || null,
          },
        ],
      });
    } catch (error) {
      console.error("Error in /sensors/updates:", error.message);
      res.status(500).json({ error: "Failed to load sensor updates." });
    }
  });

  app.get("/sensors/alerts", async (req, res) => {
    try {
      if (!latestSensorData) {
        return res.json({ items: [] });
      }

      let sourceItems = [];

      if (Array.isArray(latestSensorData.items)) {
        sourceItems = latestSensorData.items;
      } else {
        sourceItems = [
          {
            sensorId: "LIVE-001",
            status: "Online",
            batteryPct: 18,
            cpu: latestSensorData.health_data?.cpu,
            ram: latestSensorData.health_data?.ram,
            disk: latestSensorData.health_data?.disk,
            uptime: latestSensorData.health_data?.uptime,
            gps: latestSensorData.gps_data,
            lastAudio: latestSensorData.savedAudio || null,
          },
        ];
      }

      const items = [];

      sourceItems.forEach((sensor) => {
        if (sensor.status === "Offline") {
          items.push({
            sensorId: sensor.sensorId,
            issue: "Offline",
            details: "Sensor is currently offline.",
          });
        }

        if (typeof sensor.batteryPct === "number" && sensor.batteryPct < 20) {
          items.push({
            sensorId: sensor.sensorId,
            issue: "Low Battery",
            details: `Battery level is ${sensor.batteryPct}%.`,
          });
        }

        if (typeof sensor.cpu === "number" && sensor.cpu > 90) {
          items.push({
            sensorId: sensor.sensorId,
            issue: "High CPU",
            details: `CPU usage ${sensor.cpu}%`,
          });
        }

        if (typeof sensor.ram === "number" && sensor.ram > 90) {
          items.push({
            sensorId: sensor.sensorId,
            issue: "High RAM",
            details: `RAM usage ${sensor.ram}%`,
          });
        }

        if (typeof sensor.disk === "number" && sensor.disk > 90) {
          items.push({
            sensorId: sensor.sensorId,
            issue: "High Disk",
            details: `Disk usage ${sensor.disk}%`,
          });
        }
      });

      res.json({ items });
    } catch (error) {
      console.error("Error in /sensors/alerts:", error.message);
      res.status(500).json({ error: "Failed to load sensor alerts." });
    }
  });

  let rebootHistory = [];

  app.get("/sensors/reboots/recent", async (req, res) => {
    try {
      const limit = Number(req.query.limit) || 50;
      res.json({ items: rebootHistory.slice(0, limit) });
    } catch (error) {
      console.error("Error in /sensors/reboots/recent:", error.message);
      res.status(500).json({ error: "Failed to load reboot history." });
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
