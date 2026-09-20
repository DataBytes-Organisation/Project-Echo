import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { createRequire } from "node:module";

export const pageRoutes = [
  ["/login", "public/pages/auth/login.html", false],
  ["/login.html", "public/pages/auth/login.html", false],
  ["/verify-otp", "public/pages/auth/verify-otp.html", false],
  ["/verify-otp.html", "public/pages/auth/verify-otp.html", false],
  ["/forgotPassword", "public/pages/auth/reset-password.html", false],
  ["/resetPassword.html", "public/pages/auth/reset-password.html", false],
  ["/map", "public/pages/map/index.html", true],
  ["/index.html", "public/pages/map/index.html", true],
  ["/admin-dashboard", "public/pages/admin/dashboard.html", true],
  ["/admin-nodes", "public/pages/admin/admin-nodes.html", true],
  ["/admin-nodes-temp", "public/pages/admin/admin-nodes-temp.html", true],
  ["/admin-compute", "public/pages/admin/cloud-compute.html", true],
  ["/admin-api-explorer", "public/pages/admin/api-explorer.html", true],
  ["/admin-projects", "public/pages/admin/projects.html", true],
  ["/admin-profile", "public/pages/admin/profile.html", true],
  ["/admin-template", "public/pages/admin/template.html", true],
  ["/admin-donations", "public/pages/admin/donations.html", true],
  ["/requests", "public/pages/admin/admin-request.html", true],
  ["/notifications", "public/pages/admin/notifications.html", true],
  ["/admin/dashboard.html", "public/pages/admin/dashboard.html", true],
  ["/admin/admin-nodes.html", "public/pages/admin/admin-nodes.html", true],
  ["/admin/admin-nodes-temp.html", "public/pages/admin/admin-nodes-temp.html", true],
  ["/admin/cloud-compute.html", "public/pages/admin/cloud-compute.html", true],
  ["/admin/api-explorer.html", "public/pages/admin/api-explorer.html", true],
  ["/admin/projects.html", "public/pages/admin/projects.html", true],
  ["/admin/profile.html", "public/pages/admin/profile.html", true],
  ["/admin/template.html", "public/pages/admin/template.html", true],
  ["/admin/donations.html", "public/pages/admin/donations.html", true],
  ["/admin/admin-request.html", "public/pages/admin/admin-request.html", true],
  ["/admin/notifications.html", "public/pages/admin/notifications.html", true],
  ["/admin/detection-review.html", "public/pages/admin/detection-review.html", true],
  ["/admin/sensor-health.html", "public/pages/admin/sensor-health.html", true],
  ["/admin/feedback.html", "public/pages/admin/feedback.html", true],
  ["/admin/hmi-data-insights.html", "public/pages/admin/hmi-data-insights.html", true],
  ["/admin/sensor_health/alerts.html", "public/pages/admin/sensor_health/alerts.html", true],
  ["/admin/sensor_health/reboot.html", "public/pages/admin/sensor_health/reboot.html", true],
  ["/admin/sensor_health/settings.html", "public/pages/admin/sensor_health/settings.html", true],
  ["/admin/sensor_health/add-project.html", "public/pages/admin/sensor_health/add-project.html", true],
  ["/admin/sensor_health/device-detail.html", "public/pages/admin/sensor_health/device-detail.html", true],
];

const uiDir = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const require = createRequire(import.meta.url);
const { createApp } = require("../../server/app");

function markerFor(file) {
  const source = readFileSync(resolve(uiDir, file), "utf8");
  return source.match(/<title[^>]*>([^<]+)/i)?.[1]?.trim() || "<!DOCTYPE html>";
}

async function withServer(fn) {
  const app = createApp({
    redisClient: {
      isOpen: true,
      async connect() {},
      async get() { return null; },
    },
  });
  const server = await new Promise((resolveServer) => {
    const listener = app.listen(0, "127.0.0.1", () => resolveServer(listener));
  });
  try {
    const { port } = server.address();
    await fn(`http://127.0.0.1:${port}`);
  } finally {
    await new Promise((resolveClose, rejectClose) => {
      server.close((error) => error ? rejectClose(error) : resolveClose());
    });
  }
}

for (const [route, file, protectedRoute] of pageRoutes) {
  test(`page route ${route} keeps current access for ${file}`, async () => {
    assert.equal(existsSync(resolve(uiDir, file)), true, `${file} should exist on disk`);
    await withServer(async (baseUrl) => {
      const response = await fetch(`${baseUrl}${route}`, { redirect: "manual" });
      if (protectedRoute) {
        assert.equal(response.status, 302);
        assert.equal(response.headers.get("location"), "/login");
        return;
      }

      assert.equal(response.status, 200);
      assert.match(await response.text(), new RegExp(markerFor(file).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
    });
  });
}

test("/welcome is a redirect and not a sendFile mapping", async () => {
  await withServer(async (baseUrl) => {
    const response = await fetch(`${baseUrl}/welcome`, { redirect: "manual" });
    assert.equal(response.status, 302);
    assert.equal(response.headers.get("location"), "/login");
  });
});
