/**
 * Sensor Health page tests that assert shown content, not private helpers.
 *
 * The page scripts are classic browser files, so we load them into a minimal
 * document stub, seed /sensors responses, and check the rendered table / detail
 * text a user would see.
 */

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const scriptPath = path.join(here, "script.js");
const chartsPath = path.join(here, "charts.js");
const httpClientPath = path.join(here, "..", "..", "js", "http-client.js");

class FakeClassList {
  constructor(el) {
    this.el = el;
    this._values = new Set();
  }
  add(...names) {
    for (const name of names) this._values.add(name);
    this.el.className = [...this._values].join(" ");
  }
  remove(...names) {
    for (const name of names) this._values.delete(name);
    this.el.className = [...this._values].join(" ");
  }
  contains(name) {
    return this._values.has(name);
  }
  toggle(name, force) {
    if (force === true) this.add(name);
    else if (force === false) this.remove(name);
    else if (this.contains(name)) this.remove(name);
    else this.add(name);
  }
}

class FakeElement {
  constructor(tagName = "div", ownerDocument = null) {
    this.tagName = String(tagName).toUpperCase();
    this.ownerDocument = ownerDocument;
    this.children = [];
    this.attributes = {};
    this.style = {};
    this.className = "";
    this.classList = new FakeClassList(this);
    this._innerHTML = "";
    this._textContent = "";
    this.value = "";
    this.hidden = false;
    this.disabled = false;
    this.tabIndex = 0;
    this.listeners = {};
    this.parentNode = null;
  }

  get innerHTML() {
    if (this.children.length) {
      return this.children
        .map((child) => {
          const attrs = Object.entries(child.attributes)
            .map(([key, value]) => ` ${key}="${value}"`)
            .join("");
          return `<${child.tagName.toLowerCase()}${attrs}>${child.innerHTML}</${child.tagName.toLowerCase()}>`;
        })
        .join("");
    }
    return this._innerHTML;
  }

  set innerHTML(value) {
    this._innerHTML = String(value ?? "");
    this.children = [];
  }

  get textContent() {
    if (this.children.length) {
      return this.children.map((child) => child.textContent).join("");
    }
    if (this._innerHTML && /</.test(this._innerHTML)) {
      return this._innerHTML.replace(/<[^>]+>/g, "");
    }
    return this._textContent || this._innerHTML;
  }

  set textContent(value) {
    this._textContent = String(value ?? "");
    this._innerHTML = this._textContent;
    this.children = [];
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name === "id") this.id = String(value);
  }

  removeAttribute(name) {
    delete this.attributes[name];
  }

  getAttribute(name) {
    return this.attributes[name] ?? null;
  }

  appendChild(child) {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  querySelector(selector) {
    if (selector === "a") {
      return this.children.find((child) => child.tagName === "A") || null;
    }
    return null;
  }

  querySelectorAll() {
    return [];
  }

  closest() {
    return null;
  }

  focus() {}

  addEventListener(type, handler) {
    (this.listeners[type] ||= []).push(handler);
  }

  removeEventListener(type, handler) {
    this.listeners[type] = (this.listeners[type] || []).filter((h) => h !== handler);
  }
}

function createElement(tagName, doc) {
  return new FakeElement(tagName, doc);
}

function buildDocument(ids) {
  const elements = new Map();
  const document = {
    readyState: "complete",
    documentElement: {
      getAttribute: () => null,
      setAttribute() {},
      removeAttribute() {},
    },
    body: createElement("body"),
    title: "",
    scripts: [],
    listeners: {},
    getElementById(id) {
      return elements.get(id) || null;
    },
    querySelector() {
      return null;
    },
    querySelectorAll() {
      return [];
    },
    createElement(tagName) {
      return createElement(tagName, document);
    },
    addEventListener(type, handler) {
      (this.listeners[type] ||= []).push(handler);
    },
  };

  for (const id of ids) {
    const el = createElement("div", document);
    el.id = id;
    if (id.endsWith("-tbody")) el.tagName = "TBODY";
    if (id.includes("filter") || id.includes("input") || id.includes("reason")) {
      el.tagName = "INPUT";
      el.value = id.includes("filter") ? "All" : "";
    }
    if (id.includes("button")) el.tagName = "BUTTON";
    elements.set(id, el);
  }

  return { document, elements };
}

function jsonResponse(data, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: () => "application/json" },
    json: async () => data,
    text: async () => JSON.stringify(data),
  };
}

async function settle() {
  for (let i = 0; i < 15; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

async function loadDashboardPage({ items, pathname = "/admin/sensor-health.html" }) {
  const { document, elements } = buildDocument([
    "sensor-overview-tbody",
    "sensor-status-filter",
    "sensor-search-input",
    "sensor-data-source",
    "sensor-refresh-button",
    "last-updated-at",
    "menu-toggle",
    "mobile-backdrop",
    "theme-toggle",
  ]);

  const fetchImpl = async (url) => {
    const path = String(url);
    if (path.includes("/sensors/updates")) {
      return jsonResponse({ items, count: items.length, source: "backend" });
    }
    if (path.includes("/sensors/alerts")) {
      return jsonResponse({ items: [], count: 0, source: "backend" });
    }
    if (path.includes("/sensors/") && path.includes("/settings")) {
      return jsonResponse({
        settings: {
          recordIntervalSeconds: 60,
          sensitivity: "Medium",
          batteryThresholdPct: 25,
        },
      });
    }
    if (path.includes("/sensors/") && path.includes("/reboots")) {
      return jsonResponse({ items: [], count: 0 });
    }
    return jsonResponse({ items: [], count: 0 });
  };

  const context = {
    document,
    localStorage: { getItem: () => null, setItem() {} },
    location: { search: "", href: pathname, pathname },
    setTimeout,
    clearTimeout,
    setInterval: () => 0,
    clearInterval() {},
    fetch: fetchImpl,
    AbortController,
    URLSearchParams,
    console: { log() {}, warn() {}, error() {} },
    confirm: () => false,
    createAdminPageState: () => ({
      resetPageState() {},
      showLoading() {},
      hideLoading() {},
      showError() {},
      hideError() {},
    }),
  };
  context.window = context;
  context.globalThis = context;

  const httpClient = await readFile(httpClientPath, "utf8");
  const source = await readFile(scriptPath, "utf8");
  vm.createContext(context);
  vm.runInContext("const pageState = createAdminPageState(); pageState.resetPageState();", context);
  vm.runInContext(httpClient, context, { filename: "http-client.js" });
  vm.runInContext(source, context, { filename: "script.js" });
  await settle();

  return { context, elements, document };
}

async function loadDeviceDetailPage({ sensorId, detail, pathname }) {
  const { document, elements } = buildDocument([
    "device-empty-state",
    "device-empty-message",
    "device-detail-root",
    "device-title",
    "device-status-wrap",
    "device-health-status",
    "device-health-battery",
    "device-health-battery-health",
    "device-health-temperature",
    "device-health-solar",
    "device-health-last-seen",
    "device-health-project",
    "device-telemetry-note",
    "device-hw-type",
    "device-hw-model",
    "device-hw-processor",
    "device-hw-clock",
    "device-hw-memory",
    "device-hw-storage",
    "device-components-tbody",
    "device-connected-list",
    "device-location-text",
    "device-location-map",
    "device-audio-when",
    "device-audio-species",
    "device-audio-confidence",
    "device-audio-sample-rate",
    "device-audio-note",
    "device-audio-history-tbody",
    "device-reboot-history-tbody",
    "device-reboot-reason",
    "device-reboot-button",
    "device-reboot-message",
    "device-record-interval",
    "device-sensitivity",
    "device-battery-threshold",
    "device-settings-button",
    "device-settings-message",
    "menu-toggle",
    "mobile-backdrop",
    "theme-toggle",
  ]);

  elements.get("device-empty-state").hidden = true;
  elements.get("device-detail-root").hidden = true;

  const fetchImpl = async (url) => {
    const path = String(url);
    if (path.includes(`/sensors/${encodeURIComponent(sensorId)}/settings`)) {
      return jsonResponse({
        settings: {
          recordIntervalSeconds: 60,
          sensitivity: "Medium",
          batteryThresholdPct: 25,
        },
      });
    }
    if (path.includes(`/sensors/${encodeURIComponent(sensorId)}/reboots`)) {
      return jsonResponse({ items: [], count: 0 });
    }
    if (path.includes(`/sensors/${encodeURIComponent(sensorId)}`)) {
      if (!detail) return jsonResponse({ detail: "not found" }, 404);
      return jsonResponse(detail);
    }
    return jsonResponse({ items: [], count: 0 });
  };

  const context = {
    document,
    localStorage: { getItem: () => null, setItem() {} },
    location: {
      search: `?sensorId=${encodeURIComponent(sensorId)}`,
      href: pathname || `/admin/sensor_health/device-detail.html?sensorId=${encodeURIComponent(sensorId)}`,
      pathname: pathname || "/admin/sensor_health/device-detail.html",
    },
    setTimeout,
    clearTimeout,
    setInterval: () => 0,
    clearInterval() {},
    fetch: fetchImpl,
    AbortController,
    URLSearchParams,
    console: { log() {}, warn() {}, error() {} },
    confirm: () => false,
    ol: undefined,
    createAdminPageState: () => ({
      resetPageState() {},
      showLoading() {},
      hideLoading() {},
      showError() {},
      hideError() {},
    }),
  };
  context.window = context;
  context.globalThis = context;

  const httpClient = await readFile(httpClientPath, "utf8");
  const source = await readFile(scriptPath, "utf8");
  vm.createContext(context);
  vm.runInContext("const pageState = createAdminPageState(); pageState.resetPageState();", context);
  vm.runInContext(httpClient, context, { filename: "http-client.js" });
  vm.runInContext(source, context, { filename: "script.js" });
  await settle();

  return { context, elements, document };
}

test("dashboard table shows Unknown and dashes for missing readings, not zero", async () => {
  const { elements } = await loadDashboardPage({
    items: [
      {
        sensorId: "node_1",
        name: "Node Alpha",
        status: "Unknown",
        type: "master",
        model: "RaspberryPi4",
        batteryPct: null,
        temperatureC: null,
        componentCount: 2,
        gps: { lat: -38.7789, lon: 143.5705 },
        lastAudioMinutesAgo: null,
        lastAudioTs: null,
      },
      {
        sensorId: "node_1_2",
        name: "Alpha Sub 2",
        status: "Unknown",
        type: "raspberry_pi",
        model: "RaspberryPi Zero",
        batteryPct: 75,
        temperatureC: 0,
        componentCount: 1,
        gps: { lat: -38.78, lon: 143.57 },
        lastAudioMinutesAgo: null,
      },
    ],
  });

  const html = elements.get("sensor-overview-tbody").innerHTML;

  assert.match(html, /pill-muted[^>]*>Unknown/);
  assert.match(html, /node_1/);
  assert.match(html, /Node Alpha/);
  assert.doesNotMatch(html, />0%</);
  assert.doesNotMatch(html, /Just now/);
  assert.match(html, /75%/);
  assert.match(html, /0 °C/);
  assert.match(html, /View/);
  assert.equal(elements.get("sensor-data-source").textContent, "Backend API");
});

test("dashboard marks Offline devices in danger styling", async () => {
  const { elements } = await loadDashboardPage({
    items: [
      {
        sensorId: "gone",
        name: "Stale Node",
        status: "Offline",
        type: "arduino",
        model: "Arduino Uno",
        batteryPct: null,
        temperatureC: null,
        componentCount: 0,
        gps: null,
        lastSeenMinutesAgo: 40,
      },
    ],
  });

  const html = elements.get("sensor-overview-tbody").innerHTML;
  assert.match(html, /pill-danger[^>]*>Offline/);
  assert.match(html, /gone/);
});

test("device detail shows Never reported and empty audio copy for unknown devices", async () => {
  const { elements } = await loadDeviceDetailPage({
    sensorId: "node_1_2",
    detail: {
      sensorId: "node_1_2",
      name: "Alpha Sub 2",
      status: "Unknown",
      batteryPct: 75,
      power: { batteryHealthPct: 92, batteryCapacity: "2000mAh" },
      temperatureC: null,
      hardware: { clockSpeed: "1GHz", memory: "512MB" },
      type: "raspberry_pi",
      model: "RaspberryPi Zero",
      components: [
        {
          type: "battery",
          category: "power",
          model: "LithiumPro 2000",
          metrics: [{ key: "currentCharge", label: "Charge", display: "75.0%" }],
        },
      ],
      connectedDevices: [],
      gps: { lat: -38.7789, lon: 143.5705 },
      lastSeen: null,
      lastSeenMinutesAgo: null,
      lastAudio: null,
      recentAudio: [],
      project: null,
    },
  });

  assert.equal(elements.get("device-detail-root").hidden, false);
  assert.equal(elements.get("device-empty-state").hidden, true);
  assert.match(elements.get("device-health-status").innerHTML, /Unknown/);
  assert.match(elements.get("device-health-battery").innerHTML, /75%/);
  assert.equal(elements.get("device-health-last-seen").textContent, "Never reported");
  assert.equal(elements.get("device-audio-when").textContent, "—");
  assert.match(
    elements.get("device-audio-note").textContent,
    /No audio has been recorded from this device yet/
  );
  assert.match(elements.get("device-components-tbody").innerHTML, /75\.0%/);
  assert.match(elements.get("device-location-text").textContent, /-38\.7789/);
  assert.equal(elements.get("device-location-map").innerHTML.includes("<iframe"), false);
});

test("invalid sensor id shows the empty state message", async () => {
  const { elements } = await loadDeviceDetailPage({
    sensorId: "does-not-exist",
    detail: null,
  });

  assert.equal(elements.get("device-empty-state").hidden, false);
  assert.equal(elements.get("device-detail-root").hidden, true);
  assert.match(elements.get("device-empty-message").textContent, /does-not-exist|not found|could not/i);
});

test("script still loads when the page already declared pageState", async () => {
  const { elements } = await loadDashboardPage({
    items: [
      {
        sensorId: "node_ok",
        name: "Ready",
        status: "Online",
        type: "master",
        model: "Pi",
        batteryPct: 50,
        temperatureC: 20,
        componentCount: 1,
        gps: { lat: 1, lon: 2 },
      },
    ],
  });

  // If the pageState collision returned, the table would never render rows.
  assert.match(elements.get("sensor-overview-tbody").innerHTML, /node_ok/);
  assert.match(elements.get("sensor-overview-tbody").innerHTML, /Online/);
});

test("shared http client is used instead of local fetch timeout copies", async () => {
  const script = await readFile(scriptPath, "utf8");
  const charts = await readFile(chartsPath, "utf8");
  assert.ok(!script.includes("new AbortController"));
  assert.ok(!charts.includes("new AbortController"));
  assert.ok(charts.includes('apiFetch("/sensors/alerts"'));
  assert.ok(!(await readFile(scriptPath, "utf8")).includes("<iframe"));
});
