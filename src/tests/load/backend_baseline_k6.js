import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://host.docker.internal:9000";

const VUS = Number(__ENV.VUS || 20);
const DURATION = __ENV.DURATION || "60s";

export const options = {
  vus: VUS,
  duration: DURATION,
};

// Per-workload metrics
const healthRequests = new Counter("health_requests");
const healthErrors = new Rate("health_errors");
const healthLatency = new Trend("health_latency", true);

const audioRequests = new Counter("audio_requests");
const audioErrors = new Rate("audio_errors");
const audioLatency = new Trend("audio_latency", true);

const queueRequests = new Counter("queue_requests");
const queueErrors = new Rate("queue_errors");
const queueLatency = new Trend("queue_latency", true);

function testHealth() {
  const res = http.get(`${BASE_URL}/`, {
    tags: {
      name: "health_liveness",
      workload: "health",
    },
  });

  const ok = check(res, {
    "health status is 200": (r) => r.status === 200,
  });

  healthRequests.add(1);
  healthErrors.add(!ok);
  healthLatency.add(res.timings.duration);
}

function testAudioUpload() {
  const uniqueFilename =
    `task3_loadtest_vu${__VU}_iter${__ITER}_${Date.now()}.wav`;

  // The current backend validates extension/content-type/size,
  // but does not decode the WAV contents.
  const formData = {
    file: http.file(
      "task3-load-test-audio",
      uniqueFilename,
      "audio/wav"
    ),
    user_id: "task3-load-test",
  };

  const res = http.post(
    `${BASE_URL}/api/audio/upload`,
    formData,
    {
      tags: {
        name: "audio_upload",
        workload: "audio",
      },
    }
  );

  const ok = check(res, {
    "audio upload status is 200": (r) => r.status === 200,
  });

  audioRequests.add(1);
  audioErrors.add(!ok);
  audioLatency.add(res.timings.duration);
}

function testQueueIngest() {
  const payload = JSON.stringify({
    timestamp: new Date().toISOString(),
    sensorId: `task3-loadtest-${__VU}`,
    microphoneLLA: [-37.8136, 144.9631, 10.0],
    animalEstLLA: [-37.8135, 144.9632, 10.0],
    animalTrueLLA: [-37.8134, 144.9633, 10.0],
    animalLLAUncertainty: 5.0,
    audioClip: `task3-loadtest-${__VU}-${__ITER}`,
    mode: "Recording_Mode",
    audioFile: `task3_loadtest_${__VU}_${__ITER}.wav`,
  });

  const res = http.post(
    `${BASE_URL}/hmi/post_recording`,
    payload,
    {
      headers: {
        "Content-Type": "application/json",
      },
      tags: {
        name: "queue_ingest",
        workload: "queue",
      },
    }
  );

  const ok = check(res, {
    "queue ingest status is 201": (r) => r.status === 201,
  });

  queueRequests.add(1);
  queueErrors.add(!ok);
  queueLatency.add(res.timings.duration);
}

export default function () {
  // Rotate deterministically so every workload receives comparable traffic.
  const workload = (__VU + __ITER) % 3;

  if (workload === 0) {
    testHealth();
  } else if (workload === 1) {
    testAudioUpload();
  } else {
    testQueueIngest();
  }

  // Match the existing Project Echo load-testing profile.
  sleep(Math.random() * 2 + 1);
}
