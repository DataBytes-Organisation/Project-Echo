import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";

const root = new URL("..", import.meta.url);

test("live WS bypass is gone: no direct-backend stream client", async () => {
  assert.equal(
    existsSync(new URL("../public/js/detection_stream_client.js", import.meta.url)),
    false,
    "public/js/detection_stream_client.js must be deleted"
  );
});

test("HMI map has no direct-backend stream wiring or local token read", async () => {
  const source = await readFile(new URL("../public/features/map/hmi-map.js", import.meta.url), "utf8");
  assert.doesNotMatch(source, /detection_stream_client/);
  assert.doesNotMatch(source, /connectDetectionStream/);
  assert.doesNotMatch(source, /startDetectionStream/);
  assert.doesNotMatch(source, /B1\.2 WS/);
  assert.doesNotMatch(source, /ECHO_API_WS_URL/);
});

test("stale detection pipeline asserting the removed /hmi/detections proxy is gone", async () => {
  assert.equal(
    existsSync(new URL("./detection-pipeline.mjs", import.meta.url)),
    false,
    "tests/detection-pipeline.mjs must be deleted"
  );
});

test("HMI server keeps no dead /hmi/detections proxy", async () => {
  const source = await readFile(new URL("../server.js", import.meta.url), "utf8");
  assert.doesNotMatch(source, /app\.all\('\/hmi\/detections'/);
});
