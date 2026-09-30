# Backend Load Testing Baseline

## Overview

This document records a repeatable Backend load-testing baseline for Project Echo.

The baseline was executed on 22 September 2026 against the local Docker development
stack at commit `8b88a3c6`, using both k6 and Locust.

The purpose of this baseline is to measure Backend throughput, latency and error
behaviour at 20, 50 and 100 concurrent virtual users and to provide a reference
point for future performance and scalability work.

## Scope

Three representative Backend workloads were exercised:

| Workload | Endpoint | Expected status | Purpose |
| --- | --- | ---: | --- |
| Liveness | `GET /` | 200 | Lightweight API liveness proxy |
| Audio ingestion | `POST /api/audio/upload` | 200 | Multipart audio upload, local storage and Mongo metadata persistence |
| Queue ingestion | `POST /hmi/post_recording` | 201 | Backend publication of a recording message to MQTT |

The repository does not currently expose a dedicated `/health` endpoint, so the
API root (`GET /`) was used as the liveness proxy.

The simulator was deliberately stopped during the queue-ingestion tests. This
keeps the measurement boundary at successful publication from the Backend to the
MQTT broker and prevents downstream simulator processing from affecting the
Backend baseline.

## Important Baseline Boundary

This branch was created from `main` before the Cloudflare R2 audio-ingestion
change was merged.

Therefore, the audio results in this document represent the existing local-disk
audio-upload implementation and should be treated as a **pre-R2 baseline**.

A future run after the R2 integration is merged can use the same workload to
measure the effect of remote object storage.

## Test Environment

- Backend: FastAPI / Uvicorn in `ts-api-cont`
- MongoDB: `ts-mongodb-cont`
- MQTT broker: `ts-mqtt-server-cont`
- Load generator: Docker-hosted k6 and Locust
- k6: 2.3.0
- Locust: 2.46.6
- Duration: 60 seconds per load level
- Load levels: 20, 50 and 100 concurrent users / VUs
- Inter-request wait: approximately 1-3 seconds
- Workload distribution: approximately one third liveness, one third audio
  ingestion and one third queue ingestion

A 1-user / 10-second smoke test was performed before the official runs to verify
all three request paths and expected HTTP status codes.

## k6 Results

| VUs | Requests | Throughput | Overall p95 | Health p95 | Audio p95 | Queue p95 | Error rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 601 | 9.60 req/s | 18.78 ms | 11.02 ms | 21.92 ms | 18.97 ms | 0.00% |
| 50 | 1,529 | 24.42 req/s | 23.64 ms | 13.00 ms | 27.50 ms | 34.04 ms | 0.00% |
| 100 | 3,012 | 48.40 req/s | 30.78 ms | 23.67 ms | 33.55 ms | 33.23 ms | 0.00% |

All k6 checks passed at every load level.

## Locust Results

| Users | Requests | Throughput | Overall p95 | Health p95 | Audio p95 | Queue p95 | Error rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 595 | 9.96 req/s | 18 ms | 7 ms | 14 ms | 190 ms | 0.00% |
| 50 | 1,499 | 25.08 req/s | 20 ms | 10 ms | 14 ms | 220 ms | 0.00% |
| 100 | 3,024 | 50.57 req/s | 26 ms | 11 ms | 450 ms | 19 ms | 0.00% |

No Locust request failed at any load level.

## Resource Snapshot

During the k6 runs, Backend CPU utilisation increased with load:

| VUs | Backend CPU |
| ---: | ---: |
| 20 | ~10.6% |
| 50 | ~16.6-17.4% |
| 100 | ~27.4% |

MongoDB CPU remained low and MQTT CPU remained below approximately 9% in the
captured snapshots.

These observations do not indicate resource saturation at 100 VUs.

## Findings

### 1. Throughput scales approximately linearly

Both tools show a close-to-linear increase in throughput as concurrency rises.

k6 increased from approximately 9.60 req/s at 20 VUs to 48.40 req/s at 100 VUs.

Locust increased from approximately 9.96 req/s at 20 users to 50.57 req/s at
100 users.

This is close to the expected five-fold increase when concurrency changes from
20 to 100.

### 2. No request failures were observed

Both k6 and Locust reported a 0.00% error rate at 20, 50 and 100 users.

Backend logs were also checked after the runs and contained no matching
500, 502, 503, ERROR, Traceback or Exception entries.

### 3. Overall latency remained low

k6 overall p95 increased from 18.78 ms at 20 VUs to 30.78 ms at 100 VUs.

Locust overall p95 increased from 18 ms to 26 ms over the same concurrency
range.

This increase is modest relative to the five-fold increase in concurrency.

### 4. Endpoint-specific tail-latency spikes were observed

Locust reported queue-ingestion p95 values of approximately 190 ms and 220 ms at
20 and 50 users respectively, while the 100-user run reported only 19 ms.

The 100-user Locust run also reported a 450 ms p95 for audio upload despite a
9 ms median.

These spikes were not reproduced consistently by k6 and do not increase
monotonically with load. They should therefore be treated as intermittent
tail-latency events rather than evidence of a stable capacity bottleneck.

Future performance runs should repeat each load level multiple times and compare
p95/p99 distributions before attributing them to a specific component.

## Bottleneck Assessment

No definitive capacity bottleneck was reached at or below 100 concurrent users.

Backend CPU increased most clearly with concurrency while MongoDB and MQTT
remained comparatively lightly utilised. If load is increased further, the
Backend API process is the first component that should be monitored for
saturation.

The queue and audio endpoints also deserve further tail-latency investigation
because Locust observed isolated high-percentile latency spikes.

## Limitations

- The tests were run on a local Docker development environment, not production
  infrastructure.
- `GET /` is a liveness proxy rather than a dedicated `/health` endpoint.
- Queue-ingestion measurements end at successful Backend-to-MQTT publication;
  downstream simulator processing is excluded.
- Audio ingestion uses the pre-R2 local-disk implementation.
- Audio test payloads are intentionally small. This baseline measures request
  handling/concurrency rather than large-file transfer bandwidth.
- Docker resource statistics are point-in-time snapshots rather than continuous
  monitoring.
- Each official load level was used as a baseline run rather than a statistical
  multi-run benchmark.

## Reproduction

### k6

docker run --rm `
    -v "${repo}:/work" `
    -e BASE_URL=http://host.docker.internal:9000 `
    -e VUS=20 `
    -e DURATION=60s `
    grafana/k6:latest `
    run /work/src/tests/load/backend_baseline_k6.js

Change VUS to 50 or 100 for the other load levels.

Locust
docker run --rm `
    -v "${repo}:/work" `
    locustio/locust:latest `
    -f /work/src/tests/load/backend_baseline_locust.py `
    --host http://host.docker.internal:9000 `
    --headless `
    -u 20 `
    -r 20 `
    -t 60s `
    --only-summary

Change -u and -r to 50 or 100 for the other load levels.

Follow-up

After the Cloudflare R2 audio-ingestion work is merged, repeat the same benchmark
against the R2-backed upload endpoint and compare the resulting throughput,
latency and error rate against this pre-R2 baseline.