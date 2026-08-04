# Project Echo Retrieval Baseline Results

## Test information

- **Task:** D1.1 E2E Load Testing
- **Tool:** Artillery 2.0.33
- **Endpoint:** `GET /iot/nodes`
- **Target:** `http://localhost:9000`
- **Environment:** Local Docker Compose deployment
- **Test duration:** Approximately 82 seconds
- **Total virtual users:** 490

## Load phases

1. Warm-up: 2 users per second for 20 seconds
2. Normal load: 5 users per second for 30 seconds
3. Increased load: 10 users per second for 30 seconds

## Results

| Metric | Result |
|---|---:|
| Total requests | 490 |
| Successful HTTP 200 responses | 490 |
| Failed virtual users | 0 |
| Success rate | 100% |
| Mean response time | 8.9 ms |
| Median response time | 6 ms |
| P95 response time | 21.1 ms |
| P99 response time | 50.9 ms |
| Maximum response time | 157 ms |
| Average request rate | 7 requests/second |

## Initial observation

The `GET /iot/nodes` retrieval endpoint remained stable during the baseline workload. All 490 requests completed successfully, and no virtual users failed. Response times remained low, although a small increase appeared during the highest-load phase.

This result provides a baseline for comparison with the later stress test and future performance improvements.
