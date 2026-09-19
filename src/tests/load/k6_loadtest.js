// k6 load test against Project Echo's backend API - direct comparison against
// locustfile.py: same 4 real, safe, read-only, no-auth GET endpoints, same
// load shape (20 virtual users, 60 seconds), so results are comparable.
// See docs/team-guides/TDD_Guide.md for the full Locust-vs-k6 write-up and
// why /insights/* is excluded (a real bug, not a load-test artifact).
//
// Prerequisites: same as locustfile.py - Mongo+Redis up, backend running
// locally with MONGODB_URI/MAIL_*/PYTHONIOENCODING set (see the guide).
//
// Run:
//   k6 run src/tests/load/k6_loadtest.js

import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  vus: 20,
  duration: "60s",
};

const BASE_URL = __ENV.BASE_URL || "http://127.0.0.1:9000";

const endpoints = [
  "/public/public-test",
  "/hmi/microphones",
  "/iot/nodes",
  "/engine/animal_records",
];

// Weighted the same as locustfile.py's @task weights (3:2:2:1) by repeating
// entries, so the traffic mix matches for a fair comparison.
const weightedEndpoints = [
  ...Array(3).fill(endpoints[0]),
  ...Array(2).fill(endpoints[1]),
  ...Array(2).fill(endpoints[2]),
  ...Array(1).fill(endpoints[3]),
];

export default function () {
  const path = weightedEndpoints[Math.floor(Math.random() * weightedEndpoints.length)];
  const res = http.get(`${BASE_URL}${path}`, { tags: { name: path } });
  check(res, {
    "status is 200": (r) => r.status === 200,
  });
  sleep(Math.random() * 2 + 1); // 1-3s, matches locustfile.py's between(1, 3)
}
