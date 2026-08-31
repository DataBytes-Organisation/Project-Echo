try {
  require("dotenv").config();
} catch (_) {
  // Environment variables may already be provided by the HMI server.
}

const API_BASE_URL = `http://${process.env.API_HOST || "localhost"}:9000`;
const BACKEND_TIMEOUT_MS = 3000;

const DEFAULT_SETTINGS = {
  recordIntervalSeconds: 60,
  sensitivity: "Medium",
  batteryThresholdPct: 25,
  onlineWindowMinutes: 5,
  degradedWindowMinutes: 15,
};

const settingsStore = {};
const rebootStore = [];

function minutesAgoIso(minutes) {
  return new Date(Date.now() - minutes * 60 * 1000).toISOString();
}

function getDemoSensors() {
  return [
    {
      sensorId: "LIVE-001",
      name: "Otway Ridge Mic",
      project: "Otway Deployment",
      status: "Online",
      batteryPct: 78,
      cpu: 37,
      ram: 48,
      disk: 62,
      uptime: 15432,
      gps: { lat: -38.1499, lon: 144.3617 },
      lastSeen: minutesAgoIso(2),
      lastSeenMinutesAgo: 2,
      lastAudioTs: minutesAgoIso(8),
      lastAudioMinutesAgo: 8,
      lastAudio: {
        timestamp: minutesAgoIso(8),
        species: "Pteropus poliocephalus",
        confidence: 91.2,
        sampleRate: 48000,
      },
      recentAudio: [
        {
          timestamp: minutesAgoIso(8),
          species: "Pteropus poliocephalus",
          confidence: 91.2,
          sampleRate: 48000,
        },
        {
          timestamp: minutesAgoIso(42),
          species: "Phascolarctos cinereus",
          confidence: 76.4,
          sampleRate: 48000,
        },
      ],
    },
    {
      sensorId: "LIVE-002",
      name: "Creek Line Node",
      project: "Otway Deployment",
      status: "Offline",
      batteryPct: 55,
      cpu: 0,
      ram: 0,
      disk: 0,
      uptime: 0,
      gps: { lat: -38.1605, lon: 144.3502 },
      lastSeen: minutesAgoIso(96),
      lastSeenMinutesAgo: 96,
      lastAudioTs: minutesAgoIso(180),
      lastAudioMinutesAgo: 180,
      lastAudio: {
        timestamp: minutesAgoIso(180),
        species: "Dasyurus maculatus",
        confidence: 64.1,
        sampleRate: 44100,
      },
      recentAudio: [
        {
          timestamp: minutesAgoIso(180),
          species: "Dasyurus maculatus",
          confidence: 64.1,
          sampleRate: 44100,
        },
      ],
    },
    {
      sensorId: "LIVE-003",
      name: "Canopy Edge Sensor",
      project: "Otway Deployment",
      status: "Low Battery",
      batteryPct: 15,
      cpu: 52,
      ram: 63,
      disk: 70,
      uptime: 28765,
      gps: { lat: -38.1422, lon: 144.3728 },
      lastSeen: minutesAgoIso(4),
      lastSeenMinutesAgo: 4,
      lastAudioTs: minutesAgoIso(11),
      lastAudioMinutesAgo: 11,
      lastAudio: {
        timestamp: minutesAgoIso(11),
        species: "Litoria aurea",
        confidence: 88.0,
        sampleRate: 48000,
      },
      recentAudio: [
        {
          timestamp: minutesAgoIso(11),
          species: "Litoria aurea",
          confidence: 88.0,
          sampleRate: 48000,
        },
        {
          timestamp: minutesAgoIso(27),
          species: "Litoria aurea",
          confidence: 81.5,
          sampleRate: 48000,
        },
      ],
    },
  ];
}

function toListItem(sensor) {
  const { recentAudio, lastAudio, ...rest } = sensor;
  return rest;
}

function findDemoSensor(sensorId) {
  return getDemoSensors().find((item) => item.sensorId === sensorId) || null;
}

function alertsFrom(items) {
  const alerts = [];
  for (const item of items) {
    if (!item?.sensorId || item.status === "Online") continue;

    if (item.status === "Offline") {
      alerts.push({
        sensorId: item.sensorId,
        severity: "Critical",
        issue: "Offline",
        details:
          item.lastSeenMinutesAgo == null
            ? "No contact"
            : `No contact for ${item.lastSeenMinutesAgo} minutes`,
        lastAudioTs: item.lastAudioTs,
        lastAudioMinutesAgo: item.lastAudioMinutesAgo,
      });
    } else if (item.status === "Low Battery" || (typeof item.batteryPct === "number" && item.batteryPct < 20)) {
      alerts.push({
        sensorId: item.sensorId,
        severity: "High",
        issue: "Low Battery",
        details: `Battery at ${item.batteryPct}%`,
        lastAudioTs: item.lastAudioTs,
        lastAudioMinutesAgo: item.lastAudioMinutesAgo,
      });
    } else if (item.status === "Degraded") {
      alerts.push({
        sensorId: item.sensorId,
        severity: "Medium",
        issue: "Degraded",
        details:
          item.lastSeenMinutesAgo == null
            ? "Irregular heartbeat"
            : `Last contact ${item.lastSeenMinutesAgo} minutes ago`,
        lastAudioTs: item.lastAudioTs,
        lastAudioMinutesAgo: item.lastAudioMinutesAgo,
      });
    }
  }
  return alerts;
}

async function tryBackend(req) {
  const url = new URL(req.originalUrl, API_BASE_URL);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), BACKEND_TIMEOUT_MS);

  try {
    const init = {
      method: req.method,
      signal: controller.signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
    };
    if (!["GET", "HEAD"].includes(String(req.method).toUpperCase()) && req.body) {
      init.body = JSON.stringify(req.body);
    }

    const response = await fetch(url, init);
    const contentType = response.headers.get("content-type") || "";
    const data = contentType.includes("application/json")
      ? await response.json().catch(() => null)
      : await response.text().catch(() => "");
    return { status: response.status, data };
  } finally {
    clearTimeout(timer);
  }
}

function sendBackend(res, response) {
  res.status(response.status);
  if (typeof response.data === "undefined") return res.end();
  return res.send(response.data);
}

async function loadSensorCatalog() {
  try {
    const response = await tryBackend({
      method: "GET",
      originalUrl: "/sensors/updates",
    });
    if (response.status < 400 && Array.isArray(response.data?.items) && response.data.items.length > 0) {
      return { items: response.data.items, source: "backend" };
    }
  } catch (error) {
    console.warn("Sensor catalog backend unavailable:", error.message);
  }

  return { items: getDemoSensors(), source: "demo-fallback" };
}

function asDetailPayload(sensor, source) {
  if (!sensor) return null;
  return {
    recentAudio: Array.isArray(sensor.recentAudio) ? sensor.recentAudio : [],
    ...sensor,
    source,
  };
}

module.exports = function (app) {
  app.get("/sensors/updates", async (_req, res) => {
    const catalog = await loadSensorCatalog();
    const items = catalog.source === "demo-fallback" ? catalog.items.map(toListItem) : catalog.items;
    return res.json({
      items,
      count: items.length,
      source: catalog.source,
    });
  });

  app.get("/sensors/alerts", async (req, res) => {
    try {
      const response = await tryBackend(req);
      // An empty list is a valid answer ("no active alerts"), so only fall back
      // when the Backend is genuinely unreachable or errored.
      if (response.status < 400 && Array.isArray(response.data?.items)) {
        return res.json({ ...response.data, source: "backend" });
      }
    } catch (error) {
      console.warn("Sensor alerts backend unavailable:", error.message);
    }

    const items = alertsFrom(getDemoSensors());
    return res.json({ items, count: items.length, source: "demo-fallback" });
  });

  app.get("/sensors/reboots/recent", async (req, res) => {
    try {
      const response = await tryBackend(req);
      if (response.status < 400 && Array.isArray(response.data?.items)) {
        return res.json({ ...response.data, source: "backend" });
      }
    } catch (error) {
      console.warn("Recent reboots backend unavailable:", error.message);
    }

    const limit = Number(req.query.limit) || 50;
    const items = rebootStore.slice(0, limit);
    return res.json({ items, count: items.length, source: "demo-fallback" });
  });

  app.get("/sensors/:sensorId/settings", async (req, res) => {
    const { sensorId } = req.params;
    try {
      const response = await tryBackend(req);
      if (response.status < 400 && response.data?.settings) {
        return sendBackend(res, response);
      }
    } catch (error) {
      console.warn("Sensor settings backend unavailable:", error.message);
    }

    const settings = {
      ...DEFAULT_SETTINGS,
      ...(settingsStore[sensorId] || settingsStore.__default__ || {}),
    };
    return res.json({ sensorId, settings, source: "demo-fallback" });
  });

  app.put("/sensors/:sensorId/settings", async (req, res) => {
    const { sensorId } = req.params;
    try {
      const response = await tryBackend(req);
      if (response.status < 400) {
        return sendBackend(res, response);
      }
    } catch (error) {
      console.warn("Saving sensor settings backend unavailable:", error.message);
    }

    const payload = req.body && typeof req.body === "object" ? req.body : {};
    const settingsUpdate = payload.settings && typeof payload.settings === "object" ? payload.settings : payload;
    settingsStore[sensorId] = {
      ...DEFAULT_SETTINGS,
      ...(settingsStore[sensorId] || {}),
      ...settingsUpdate,
    };
    return res.json({
      sensorId,
      settings: settingsStore[sensorId],
      source: "demo-fallback",
    });
  });

  app.post("/sensors/:sensorId/reboot", async (req, res) => {
    const { sensorId } = req.params;
    try {
      const response = await tryBackend(req);
      if (response.status < 400) {
        return sendBackend(res, response);
      }
    } catch (error) {
      console.warn("Sensor reboot backend unavailable:", error.message);
    }

    const reason =
      req.body && typeof req.body.reason === "string" && req.body.reason.trim()
        ? req.body.reason.trim()
        : null;
    const doc = {
      rebootId: `local-${Date.now()}`,
      sensorId,
      reason,
      status: "Queued",
      requestedAt: new Date().toISOString(),
    };
    rebootStore.unshift(doc);
    return res.json({ ...doc, source: "demo-fallback" });
  });

  app.get("/sensors/:sensorId/reboots", async (req, res) => {
    const { sensorId } = req.params;
    try {
      const response = await tryBackend(req);
      if (response.status < 400 && Array.isArray(response.data?.items)) {
        return sendBackend(res, response);
      }
    } catch (error) {
      console.warn("Sensor reboot history backend unavailable:", error.message);
    }

    const limit = Number(req.query.limit) || 50;
    const items = rebootStore.filter((item) => item.sensorId === sensorId).slice(0, limit);
    return res.json({ items, count: items.length, source: "demo-fallback" });
  });

  app.get("/sensors/:sensorId", async (req, res) => {
    const { sensorId } = req.params;
    try {
      const response = await tryBackend(req);
      if (response.status < 400 && response.data && response.data.sensorId) {
        return res.json({ ...response.data, source: "backend" });
      }
    } catch (error) {
      console.warn("Sensor detail backend unavailable:", error.message);
    }

    const catalog = await loadSensorCatalog();
    const fromCatalog = catalog.items.find((item) => item.sensorId === sensorId);
    if (fromCatalog) {
      return res.json(asDetailPayload(fromCatalog, catalog.source));
    }

    const demoSensor = findDemoSensor(sensorId);
    if (!demoSensor) {
      return res.status(404).json({ detail: `Sensor '${sensorId}' was not found` });
    }
    return res.json(asDetailPayload(demoSensor, "demo-fallback"));
  });
};
