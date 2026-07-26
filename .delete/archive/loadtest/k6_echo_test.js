import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "10s", target: 20 },
    { duration: "20s", target: 60 },
    { duration: "30s", target: 110 },
    { duration: "20s", target: 0 },
  ],
  thresholds: {
    http_req_failed: ["rate<0.03"],
    http_req_duration: ["p(95)<1200"],
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:9000";

const DETECTIONS = `${BASE_URL}/detections`;

function isControlledBlock(status) {
  return status === 429 || status === 503 || status === 403;
}

function extractId(json) {
  if (!json || typeof json !== "object") return null;
  return json._id || json.id || (json.data && (json.data._id || json.data.id)) || null;
}

function isoNowMinusMinutes(min) {
  const d = new Date(Date.now() - min * 60 * 1000);
  return d.toISOString();
}

export default function () {
  const q = Math.random();

  let url = `${DETECTIONS}?page=1&page_size=20`;
  if (q < 0.33) {
    url = `${DETECTIONS}?species=TestSpecies&page=1&page_size=20`;
  } else if (q < 0.66) {
    url = `${DETECTIONS}?start_time=${encodeURIComponent(isoNowMinusMinutes(60))}&end_time=${encodeURIComponent(
      new Date().toISOString()
    )}&page=1&page_size=20`;
  } else {
    url = `${DETECTIONS}?lat=-37.81&lon=144.96&radius_km=5&page=1&page_size=20`;
  }

  const listRes = http.get(url);
  check(listRes, {
    "list: status ok/controlled": (r) => r.status === 200 || isControlledBlock(r.status),
  });

  sleep(0.1);

  const createPayload = JSON.stringify({
    timestamp: new Date().toISOString(),
    species: "TestSpecies",
    confidence: 0.91,
  });

  const createRes = http.post(DETECTIONS, createPayload, {
    headers: { "Content-Type": "application/json" },
  });

  const createdOk = createRes.status === 200 || createRes.status === 201;
  check(createRes, {
    "create: status ok/controlled": (r) => createdOk || isControlledBlock(r.status),
  });

  if (!createdOk) {
    sleep(0.2);
    return;
  }

  let createdId = null;
  try {
    createdId = extractId(createRes.json());
  } catch (e) {
    createdId = null;
  }

  if (!createdId) {
    sleep(0.2);
    return;
  }

  const itemUrl = `${DETECTIONS}/${createdId}`;

  const getRes = http.get(itemUrl);
  check(getRes, {
    "get: status ok/controlled": (r) => r.status === 200 || isControlledBlock(r.status),
  });

  sleep(0.05);

  const patchPayload = JSON.stringify({
    confidence: 0.92,
  });

  const patchRes = http.patch(itemUrl, patchPayload, {
    headers: { "Content-Type": "application/json" },
  });

  check(patchRes, {
    "patch: status ok/controlled": (r) =>
      r.status === 200 || r.status === 204 || isControlledBlock(r.status),
  });

  sleep(0.05);

  const delRes = http.del(itemUrl);
  check(delRes, {
    "delete: status ok/controlled": (r) =>
      r.status === 200 || r.status === 204 || isControlledBlock(r.status),
  });

  sleep(0.1);

  const doPredict = Math.random() < 0.2;
  if (doPredict) {
    const predRes = http.post(
      `${DETECTIONS}/predict`,
      JSON.stringify({ detection_id: createdId, note: "k6 test" }),
      { headers: { "Content-Type": "application/json" } }
    );

    check(predRes, {
      "predict: status ok/controlled": (r) => r.status === 200 || isControlledBlock(r.status),
    });
  }
}
