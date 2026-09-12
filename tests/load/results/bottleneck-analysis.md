# D1.1 Bottleneck Analysis

## Summary

Baseline and stress workloads were executed against both retrieval and ingestion paths in the Project Echo API.

Retrieval testing used:

`GET /iot/nodes`

Ingestion testing used:

`POST /engine/event`

## Ingestion Issue Identified

The original `POST /detections` endpoint could not be tested because the local environment returned HTTP 403 when the detections budget was not configured.

The existing `POST /engine/event` endpoint was then evaluated as the ingestion route.

During the first manual request, the endpoint returned HTTP 500 because the Pydantic confidence value was represented as a Python Decimal, which PyMongo could not directly encode into BSON.

The ingestion route was updated to convert the confidence value to float before MongoDB insertion. After this change, the same request returned HTTP 201 and was successfully stored.

## Performance Results

The ingestion baseline processed 230 requests successfully.

The ingestion stress workload processed 890 requests successfully, with:

- Mean response time: 5.5 ms
- P95 response time: 10.1 ms
- P99 response time: 16.9 ms
- Maximum response time: 65 ms
- Failed requests: 0

Continuous Docker monitoring during the actual stress-test window showed the API reaching approximately 7.43% CPU and about 97 MiB memory.

No CPU, memory or database saturation was observed at the configured workload.

## Conclusion

No clear performance bottleneck was reached during the tested workloads. The API remained stable during both retrieval and ingestion testing.

The results provide a reproducible performance baseline for future comparison after changes such as Redis caching and other backend optimisations.

A larger sustained breaking-point workload would be required to determine the maximum throughput of the system.
