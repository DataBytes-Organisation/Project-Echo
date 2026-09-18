const assert = require("node:assert/strict");
const test = require("node:test");

const { loadEnvironment } = require("../../server/config/environment");
const apiClient = require("../../server/services/apiClient");

test("resolveApiBaseUrl honors API_PORT and defaults to 9000", () => {
  assert.equal(
    apiClient.resolveApiBaseUrl({ API_HOST: "localhost", API_PORT: "9100" }),
    "http://localhost:9100"
  );
  assert.equal(apiClient.resolveApiBaseUrl({}), "http://localhost:9000");
  assert.equal(
    apiClient.resolveApiBaseUrl({ API_HOST: "api-service" }),
    "http://api-service:9000"
  );
});

test("loadEnvironment maps API_PORT onto apiPort", () => {
  assert.equal(loadEnvironment({ API_PORT: "9100" }).apiPort, 9100);
  assert.equal(loadEnvironment({}).apiPort, 9000);
});

function fakeApp() {
  const handlers = new Map();
  return {
    handlers,
    get(path, handler) {
      handlers.set(`GET ${path}`, handler);
    },
    put(path, handler) {
      handlers.set(`PUT ${path}`, handler);
    },
    post(path, handler) {
      handlers.set(`POST ${path}`, handler);
    },
  };
}

function stubHttpClient(captured) {
  const httpPath = require.resolve("../../public/shared/http/http-client.js");
  const original = require.cache[httpPath]?.exports;
  require.cache[httpPath] = {
    id: httpPath,
    filename: httpPath,
    loaded: true,
    exports: {
      request: async (url) => {
        captured.push(String(url));
        return { status: 200, data: { items: [{ sensorId: "node_1" }] } };
      },
    },
  };
  return () => {
    if (original === undefined) delete require.cache[httpPath];
    else require.cache[httpPath].exports = original;
  };
}

function freshSensorRoutes() {
  const sensorPath = require.resolve("../../server/routes/sensor.routes.js");
  delete require.cache[sensorPath];
  return require(sensorPath);
}

test("sensor routes use the injected apiBaseUrl (createApp config path)", async () => {
  const captured = [];
  const restoreHttp = stubHttpClient(captured);
  try {
    const registerSensorRoutes = freshSensorRoutes();
    const app = fakeApp();
    registerSensorRoutes(app, { apiBaseUrl: "http://localhost:9100" });
    const handler = app.handlers.get("GET /sensors/updates");
    const res = { json(body) { this.body = body; return this; } };
    await handler({}, res);
    assert.equal(res.body.source, "backend");
    assert.ok(captured[0].startsWith("http://localhost:9100"), captured[0]);
  } finally {
    restoreHttp();
  }
});

test("sensor routes default honors API_PORT from the environment", async () => {
  process.env.API_PORT = "9100";
  const captured = [];
  const restoreHttp = stubHttpClient(captured);
  try {
    const registerSensorRoutes = freshSensorRoutes();
    const app = fakeApp();
    registerSensorRoutes(app);
    const handler = app.handlers.get("GET /sensors/updates");
    const res = { json(body) { this.body = body; return this; } };
    await handler({}, res);
    assert.equal(res.body.source, "backend");
    assert.ok(captured[0].startsWith("http://localhost:9100"), captured[0]);
  } finally {
    restoreHttp();
    delete process.env.API_PORT;
  }
});

test("auth controller honors API_PORT from the environment", async () => {
  process.env.API_PORT = "9100";
  const axiosPath = require.resolve("axios");
  const originalAxios = require.cache[axiosPath]?.exports;
  let postedUrl = null;
  require.cache[axiosPath] = {
    id: axiosPath,
    filename: axiosPath,
    loaded: true,
    exports: {
      post: async (url) => {
        postedUrl = url;
        return { status: 201 };
      },
    },
  };
  try {
    const controllerPath = require.resolve("../../controller/auth.controller.js");
    delete require.cache[controllerPath];
    const controller = require("../../controller/auth.controller.js");
    const result = await controller.guestsignup({
      username: "guest",
      email: "guest@example.com",
      password: "password123",
      timestamp: 0,
    });
    assert.equal(result.status, "success");
    assert.ok(postedUrl.startsWith("http://localhost:9100/"), postedUrl);
  } finally {
    if (originalAxios === undefined) delete require.cache[axiosPath];
    else require.cache[axiosPath].exports = originalAxios;
    delete process.env.API_PORT;
  }
});
