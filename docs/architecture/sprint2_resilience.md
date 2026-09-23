# Sprint 2 Resilience & Scalability Architecture Log

**Prepared by:** Mack Turley (Backend Leader)  
**Scope:** Group C Resilience & Infrastructure Tasks (C1.1, C3, C6, C18)  
**Cross-PR Coordination:** Aligned with PR #1048 (Jamie Zhi – C1.2 Asynchronous Worker Implementation)

---

## 1. Unified Asynchronous Queue Architecture (C1.1 & C1.2 Coordination)

To establish a single supported queue approach across Project Echo, the backend API (`EB/MT/Sprint2-Resilience`) and worker implementation (`#1048`) adhere to the following shared contract:

* **Queue Framework:** `rq` (Redis Queue)
* **Redis Logical Database Allocation:**
  * `db=0`: Reserved for HMI session management, token storage, and legacy caching.
  * `db=1`: Exclusively dedicated to the backend task queue (`echo-backend`).
  * `db=2`: Allocated for high-frequency read caching (Task C2).
* **Queue Name:** `echo-backend`
* **Connection:** `REDIS_URL=redis://echo-redis:6379/1` (same variable and default as #1048).
* **Worker:** a single Compose service, `echo_rq_worker`, running `rq worker echo-backend --with-scheduler --url redis://echo-redis:6379/1`. The `--with-scheduler` flag is required: retries use delay intervals, and without the scheduler a retried job sits in the `ScheduledJobRegistry` forever.
* **Canonical Application Workflow:** `app.queue.process_audio_ingest_job`
  * **Trigger:** Invoked asynchronously whenever audio files are uploaded via `POST /api/audio/upload`.
  * **Input Parameters:** `upload_id: str` (MongoDB ObjectId), `file_path: Optional[str]`, `filename: Optional[str]`.
  * **State Transitions:** Updates MongoDB `AudioUploads` record through lifecycle states: `processing` -> `completed` (or `retry_scheduled` -> `processing` ... -> `failed`). `processing_attempts` counts each run.
  * **File Check:** The job only sets `pipeline_status: "ready_for_inference"` after it confirms the audio file exists on storage and is non-empty. A missing or empty file fails the job permanently (`processing_status: "failed"`) and skips retries, because retrying cannot bring the file back.
  * **Output Contract:** JSON-serializable dictionary containing `{"status": "completed", "upload_id": "...", "filename": "...", "duration_seconds": float, "completed_at": "ISO-8601"}`.
  * **Timeout & Retry Policy:** `job_timeout = 180s`, `Retry(max=3, interval=[10, 30, 60])` (the #1048 schedule).
  * **Failure Handling:** Other errors (for example MongoDB being unavailable) are recorded as `retry_scheduled` while retries remain, then as `failed` on the last attempt. The exception is re-raised so RQ applies the retry policy and finally moves the job to the `FailedJobRegistry`.

---

## 2. Bounded Dependency Health Probes (`/health/dependencies`)

The health aggregation endpoint ensures all subsystem probes are non-blocking and strictly bounded to prevent cascading gateway timeouts during downstream outages:

* **MongoDB (Primary & User Databases):**
  * Probes both `EchoNet` (`client`) and `UserSample` (`Userclient`) using `admin.command('ping', maxTimeMS=settings.mongo_timeout_ms)` with a default 2000ms ceiling.
* **Redis Job Queue:**
  * Uses dedicated connection parameters: `socket_connect_timeout=2.0s` and `socket_timeout=2.0s`.
* **HiveMQ / MQTT Broker:**
  * Probed via non-blocking TCP socket connection with a 2.0s timeout limit.
* **Payload Structure:**
  * Returns round-trip latency (`latency_ms`) per component, error details on failure, and an aggregate system status (`UP`, `DEGRADED`, or `DOWN`).

---

## 3. Comprehensive Graceful Shutdown & Resource Cleanup

To prevent connection leaks, zombie sockets, and incomplete operations upon container restarts or SIGTERM signals, the FastAPI application lifecycle is managed via `@asynccontextmanager def lifespan(app: FastAPI)`:

* **Teardown Sequence:**
  1. **MQTT Client (`stop_mqtt_client()`):** Sends DISCONNECT to the broker, stops paho's reconnect attempts and joins its network-loop thread (`loop_stop()`). Connection state becomes `disconnected`.
  2. **Primary MongoDB Client (`client.close()`):** Closes all active connection pools to the `EchoNet` database.
  3. **User MongoDB Client (`Userclient.close()`):** Closes connection pools to `UserSample`.
  4. **Redis Queue Connection (`redis_conn.close()`):** Closes the connection pool to `echo-redis` (db=1).
  5. **Redis Cache Connection (`app.cache.close_client()`):** Closes the insights cache pool (db=2).
* **Error Isolation:** Each cleanup step is wrapped in isolated `try/except` blocks to guarantee that a failure in one client teardown does not abort the remaining resource cleanups.
