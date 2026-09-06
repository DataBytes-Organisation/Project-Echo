# Sprint 2 Resilience & Scalability Architecture Log

**Prepared by:** Mack Turley (Backend Leader)
**Scope:** Group C Tasks (C1, C3, C6)

## 1. Redis Namespacing Strategy
To ensure that background asynchronous queue operations do not interfere with the HMI session management or caching layers, we have implemented strict Redis namespacing via logical databases:
- **`db=0`**: Reserved for default operations, legacy caching, and HMI JWT/session handling (handled by HMI module).
- **`db=1`**: Exclusively allocated to the newly introduced `rq` background job queue (`queue.py`).
- **`db=2`**: Allocated for future high-frequency read caches (Task C2 implementation).

## 2. Graceful Shutdown Behavior
Instead of abruptly killing background operations when a SIGTERM is received (e.g. `docker compose stop echo_api`), the FastAPI backend now relies on the `@asynccontextmanager def lifespan(app)` context block.
- **Startup:** Establishes necessary connection pools.
- **Shutdown:** Explicitly triggers `.close()` on the global `pymongo.MongoClient` and the backend's `redis_conn`. This prevents zombie connections from leaking memory inside the backend Docker container across rapid iterative deployments.

## 3. Dependency Aggregation (/health/dependencies)
A new endpoint was introduced to aggregate the health of `MongoDB`, `Redis`, and `HiveMQ`.
- **Constraint:** Rather than importing heavy client objects exclusively for health checks, we use non-blocking `ping` commands and simple `socket.create_connection()` probes.
- **Constraint:** Connection strings and server internals are purposely excluded from the response payload to maintain internal boundary security.
