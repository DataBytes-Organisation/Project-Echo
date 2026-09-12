# Project Echo Retrieval Stress-Test Results

## Test information

- **Task:** D1.1 E2E Load Testing
- **Tool:** Artillery 2.0.33
- **Endpoint:** `GET /iot/nodes`
- **Target:** `http://localhost:9000`
- **Environment:** Local Docker Compose deployment
- **Test duration:** Approximately 102 seconds
- **Total virtual users:** 4,200

## Load phases

1. Warm-up: 5 users per second for 20 seconds
2. Medium load: 20 users per second for 30 seconds
3. High load: 50 users per second for 30 seconds
4. Traffic spike: 100 users per second for 20 seconds

## Results

| Metric | Result |
|---|---:|
| Total requests | 4,200 |
| Successful HTTP 200 responses | 4,200 |
| Failed virtual users | 0 |
| Success rate | 100% |
| Mean response time | 4.5 ms |
| Median response time | 3 ms |
| P95 response time | 10.1 ms |
| P99 response time | 21.1 ms |
| Maximum response time | 79 ms |
| Average request rate | 47 requests/second |

## Performance analysis

The `GET /iot/nodes` endpoint remained stable throughout the stress workload. All 4,200 requests completed successfully, including the final traffic spike of 100 new virtual users per second.

No request failures, connection errors, or HTTP error responses occurred. Response latency also remained low throughout the test. The P95 latency was 10.1 ms and the P99 latency was 21.1 ms.

The test did not expose a clear bottleneck at the tested load level. A higher breaking-point test or continuous resource monitoring would be required to identify the endpoint's maximum sustainable throughput.

## Limitations

- The test was executed on a local development machine.
- Results may differ in staging or production environments.
- Docker resource snapshots were collected before and after the test rather than continuously.
- The workload currently covers retrieval only.
- A safe ingestion workload still needs to be added.
