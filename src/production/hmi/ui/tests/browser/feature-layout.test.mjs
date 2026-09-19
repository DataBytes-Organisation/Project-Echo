import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";

const root = resolve(import.meta.dirname, "../..");
const expected = [
  "public/features/payments/razorpay-checkout.js",
  "public/features/audio/audio-recorder.js",
  "public/features/audio/spectrogram.js",
  "public/features/audio/spectrogram-workflow.js",
  "public/features/detections/real-detections.js",
  "public/features/detections/source-filter.js",
  "public/features/map/hmi-map.js",
  "public/features/map/nodes-overlay.js",
  "public/shared/ui/hmi-utils.js",
  "public/shared/http/routes.js",
  "public/shared/http/http-client.js",
  "public/shared/admin/admin-layout.js",
  "public/shared/admin/admin-page-state.js",
  "public/features/sensor-health/sensor-health.js",
  "public/features/sensor-health/charts.js",
  "public/features/sensor-health/sensor-health.css",
  "public/vendor/openlayers/ol.js",
  "public/vendor/openlayers/ol.js.map",
  "public/vendor/openlayers/ol.css",
  "public/vendor/openlayers/ol.css.map",
  "public/features/admin-dashboard/dashboard.js",
  "public/features/admin-dashboard/dashboard.css",
  "public/features/api-explorer/api-explorer.js",
  "public/features/cloud-compute/cloud-compute.js",
  "public/features/notifications/notifications.js",
  "public/features/notifications/notifications.css",
  "public/features/projects/projects.js",
  "public/features/projects/projects.css",
  "public/features/profile/profile.css",
  "public/vendor/admin/styles.min.css",
  "public/vendor/admin/app.min.js",
  "public/vendor/admin/sidebarmenu.js",
  "public/vendor/admin/tabler-icons/tabler-icons.css",
  "public/vendor/admin/tabler-icons/fonts/tabler-icons.woff2",
  "public/assets/images/admin/logos/favicon.png",
  "public/assets/images/admin/logos/logo.png",
  "public/assets/images/admin/profile/user-1.jpg",
];

test("isolated browser features live under public/features", async () => {
  await Promise.all(expected.map((path) => access(resolve(root, path))));
});

test("the map page imports feature paths", async () => {
  const html = await readFile(resolve(root, "public/pages/map/index.html"), "utf8");
  assert.match(html, /features\/payments\/razorpay-checkout\.js/);
  assert.match(html, /features\/audio\/audio-recorder\.js/);
  assert.match(html, /features\/map\/hmi-map\.js/);
  assert.match(html, /vendor\/openlayers\/ol\.js/);
});
