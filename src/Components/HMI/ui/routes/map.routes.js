const { verifySignUp, client } = require("../middleware");
const axios = require("axios");
const mqtt = require("mqtt");
const fs = require("fs");
const path = require("path");
require("dotenv").config();

const API_BASE_URL = `http://${process.env.API_HOST || "localhost"}:9000`;
const MESSAGE_API_URL = `${API_BASE_URL}/hmi`;

const MQTT_BROKER = "mqtt://broker.hivemq.com";
const MQTT_TOPIC = "iot/data/test";

let token;
let latestSensorData = null;

// Ensure audio output directory exists inside the HMI app
const audioDir = path.join(__dirname, "..", "audioLocal");
fs.mkdirSync(audioDir, { recursive: true });

const mqttClient = mqtt.connect(MQTT_BROKER);

mqttClient.on("connect", () => {
  console.log("MQTT Connected");
  mqttClient.subscribe(MQTT_TOPIC, (err) => {
    if (err) {
      console.error("MQTT subscribe error:", err);
    } else {
      console.log(`Subscribed to MQTT topic: ${MQTT_TOPIC}`);
    }
  });
});

mqttClient.on("error", (err) => {
  console.error("MQTT connection error:", err);
});

mqttClient.on("message", (topic, message) => {
  try {
    const data = JSON.parse(message.toString());

    if (data.audio_file) {
      const buffer = Buffer.from(data.audio_file, "base64");
      const fileName = `audio_${Date.now()}.wav`;
      const filePath = path.join(audioDir, fileName);

      fs.writeFileSync(filePath, buffer);
      data.savedAudio = fileName;
    }

    latestSensorData = data;
    console.log("Latest sensor payload updated from MQTT");
  } catch (err) {
    console.error("MQTT parse error:", err);
  }
});

module.exports = function (app) {
  app.use(function (req, res, next) {
    res.header("Access-Control-Allow-Headers", "Origin, Content-Type, Accept");
    next();
  });

  app.get(`/movement_time/:start/:end`, async (req, res, next) => {
    try {
      const start = req.params.start;
      const end = req.params.end;
      const response = await axios.get(
        `${MESSAGE_API_URL}/movement_time?start=${start}&end=${end}`
      );

      if (Object.keys(response.data).length === 0) {
        res.send([]);
      } else {
        res.send(response.data);
      }
    } catch (error) {
      console.error("Error fetching movement_time:", error.message);
      res.status(500).send({ error: "Failed to fetch movement data." });
    }
  });

  app.get(`/events_time/:start/:end`, async (req, res, next) => {
    try {
      const start = req.params.start;
      const end = req.params.end;
      const response = await axios.get(
        `${MESSAGE_API_URL}/events_time?start=${start}&end=${end}`
      );
      res.send(response.data);
      next();
    } catch (error) {
      console.error("Error fetching events_time:", error.message);
      res.status(500).send({ error: "Failed to fetch event data." });
    }
  });

  app.get(`/microphones`, async (req, res, next) => {
    try {
      const response = await axios.get(`${MESSAGE_API_URL}/microphones`);
      res.send(response.data);
      next();
    } catch (error) {
      console.error("Error fetching microphones:", error.message);
      res.status(500).send({ error: "Failed to fetch microphones." });
    }
  });

  app.get(`/audio/:id`, async (req, res, next) => {
    try {
      const id = req.params.id;
      const response = await axios.get(`${MESSAGE_API_URL}/audio?id=${id}`);
      res.send(response.data);
      next();
    } catch (error) {
      console.error("Error fetching audio:", error.message);
      res.status(500).send({ error: "Failed to fetch audio." });
    }
  });

  app.post(`/post_recording`, async (req, res, next) => {
    const data = req.body;
    try {
      const response = await axios.post(`${MESSAGE_API_URL}/post_recording`, data);
      console.log("Record response:", response.data);
      res.send(response.data);
      next();
    } catch (error) {
      console.error("Error posting recording:", error.message);
      res.status(500).send({ error: "Failed to post recording." });
    }
  });

  app.post(`/sim_control/:control`, async (req, res, next) => {
    try {
      const control = req.params.control;
      const response = await axios.post(
        `${MESSAGE_API_URL}/sim_control?control=${control}`
      );
      res.send(response.data);
      next();
    } catch (error) {
      console.error("Error posting sim control:", error.message);
      res.status(500).send({ error: "Failed to control simulation." });
    }
  });

  app.get(`/latest_movement`, async (req, res, next) => {
    try {
      const response = await axios.get(`${MESSAGE_API_URL}/latest_movement`);
      res.send(response.data);
      next();
    } catch (error) {
      console.error("Error fetching latest movement:", error.message);
      res.status(500).send({ error: "Failed to fetch latest movement." });
    }
  });

  app.get(`/movement_time_daily/:start/:end`, async (req, res, next) => {
    try {
      const start = req.params.start;
      const end = req.params.end;
      const response = await axios.get(
        `${MESSAGE_API_URL}/movement_time_daily?start=${start}&end=${end}`
      );
      res.send(response.data || []);
    } catch (error) {
      console.error("Error fetching movement_time_daily:", error.message);
      res.status(500).send({ error: "Failed to fetch daily movement data." });
    }
  });

  app.get("/sensors/test-seed", (req, res) => {
  latestSensorData = {
    health_data: {
      cpu: 37,
      ram: 48,
      disk: 62,
      uptime: 15432
    },
    gps_data: {
      lat: -38.1499,
      lon: 144.3617
    },
    savedAudio: "test_audio.wav"
  };

  res.json({
    success: true,
    message: "Test sensor payload seeded successfully.",
    latestSensorData
  });
});

  // Sensor Health live routes
  app.get("/sensors/updates", async (req, res) => {
    try {
      if (!latestSensorData) {
        return res.json({ items: [] });
      }

      res.json({
        items: [
          {
            sensorId: "LIVE-001",
            status: "Online",
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

      const items = [];

      if (latestSensorData.health_data?.cpu > 90) {
        items.push({
          sensorId: "LIVE-001",
          issue: "High CPU",
          details: `CPU usage ${latestSensorData.health_data.cpu}%`,
        });
      }

      if (latestSensorData.health_data?.ram > 90) {
        items.push({
          sensorId: "LIVE-001",
          issue: "High RAM",
          details: `RAM usage ${latestSensorData.health_data.ram}%`,
        });
      }

      if (latestSensorData.health_data?.disk > 90) {
        items.push({
          sensorId: "LIVE-001",
          issue: "High Disk",
          details: `Disk usage ${latestSensorData.health_data.disk}%`,
        });
      }

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
      res.json({
        items: rebootHistory.slice(0, limit),
      });
    } catch (error) {
      console.error("Error in /sensors/reboots/recent:", error.message);
      res.status(500).json({ error: "Failed to load reboot history." });
    }
  });

  app.post("/sensors/:sensorId/reboot", async (req, res) => {
    try {
      const sensorId = req.params.sensorId;

      const entry = {
        sensorId,
        status: "Queued",
        requestedAt: new Date().toISOString(),
      };

      rebootHistory.unshift(entry);

      res.json({
        success: true,
        message: `Reboot queued for ${sensorId}`,
        item: entry,
      });
    } catch (error) {
      console.error("Error in reboot route:", error.message);
      res.status(500).json({ error: "Failed to queue reboot." });
    }
  });

  let sensorSettings = {
    recordIntervalSeconds: 60,
    sensitivity: "Medium",
    batteryThresholdPct: 25,
  };

  app.get("/sensors/__default__/settings", async (req, res) => {
    try {
      res.json({ settings: sensorSettings });
    } catch (error) {
      console.error("Error in settings GET route:", error.message);
      res.status(500).json({ error: "Failed to load settings." });
    }
  });

  app.put("/sensors/__default__/settings", async (req, res) => {
    try {
      const incoming = req.body?.settings || {};

      sensorSettings = {
        ...sensorSettings,
        ...incoming,
      };

      res.json({
        success: true,
        settings: sensorSettings,
      });
    } catch (error) {
      console.error("Error in settings PUT route:", error.message);
      res.status(500).json({ error: "Failed to save settings." });
    }
  });
};