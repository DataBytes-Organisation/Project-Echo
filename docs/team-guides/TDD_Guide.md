# Testing & TDD Guide

Project Echo currently has no unified test tooling and does no TDD. This guide sets up
**pytest** (Backend + Engine unit/integration/e2e) and **Locust + k6** (load testing),
and shows how to run and extend each. Every command here was actually run against the
real codebase - full findings, bugs, and the detailed load-testing analysis are in
`Testing_Framework_Research_Report_Nolan_Nguyen.pdf` (same folder as this guide's source
task); this file is the short, practical how-to.

## 1. What Exists Today

| Area | Tests today | Notes |
|---|---|---|
| Backend | None | New in this guide |
| Engine | 44 tests (`unittest`), no runner config | Now unified under pytest, zero changes needed |
| HMI | `node --test`, 2 files | Unchanged, extended with 1 new test |
| Load/performance | None | New in this guide (Locust + k6) |
| CI | Docker container-health check only | Doesn't run any of the above yet |

## 2. Frameworks and Why

- **pytest** - discovers and runs Engine's existing `unittest.TestCase` files natively
  (zero migration cost), adds fixtures/coverage on top. Not Jest - wrong language for
  Backend/Engine.
- **Locust + k6** - both practically tested and compared (see the report for the full
  head-to-head), per direct request to evaluate these two specifically for load
  testing. Recommendation: **Locust**, since scripts stay in Python like the rest of
  the stack - k6 is an equally valid pick, especially for teams wanting more explicit
  failure signalling under stress (see section 4).
- **HMI stays on `node --test`** - already works, no migration benefit at this scale.

## 3. Setup

```powershell
pip install -r requirements-dev.txt   # pytest, pytest-cov, locust
```

**Backend needs its own virtual environment** - its `requirements.txt` pins
`pydantic<2.0`, which conflicts with other tools in a shared/base Python environment
(verified: this broke `gradio`/`streamlit` when tried).

```powershell
cd src/production/backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install pytest pytest-cov locust "httpx<0.28,>=0.23"
```

(`httpx<0.28` is required - the newest `httpx` breaks `TestClient` against this
project's pinned `starlette==0.36.3`.)

k6 is a separate binary, not a pip package:

```powershell
winget install k6
```

Root `pytest.ini` (new) - deliberately excludes `src/tests/unit/backend` and
`src/tests/integration/backend`, since Backend needs its own venv above and a plain
`pytest` run in another environment would fail on unrelated missing dependencies:

```ini
[pytest]
testpaths =
    src/tests/unit/engine
    src/tests/integration/engine_backend
    src/tests/pipeline
    src/production/engine
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

Root `.coveragerc` (new) - without it, `pytest-cov` counts test files themselves as
"covered", inflating the number:

```ini
[run]
omit = */test_*.py
       */.venv/*
       */node_modules/*
```

## 4. Running Everything

### Engine + repo-wide tests (pytest)

```powershell
python -m pytest --cov=src/production/engine --cov-report=term-missing
```
**Expect: 61 passed**, ~40% coverage on `echo_engine.py`.

### Backend unit test (TDD example)

```powershell
cd src/production/backend
.venv\Scripts\python.exe -m pytest ../../tests/unit/backend/test_errors.py -v
```
**Expect: 6 passed.** `test_locked_status_maps_to_locked_code` is a real red→green
example - it failed before `app/errors.py` gained a one-line fix (`423: "LOCKED"`).
That's the TDD loop: write the test for the behaviour you want, watch it fail for the
right reason, make the smallest change that passes it.

### Backend integration/e2e test (`TestClient`)

Needs MongoDB reachable even for a route that touches no data - `app/database.py` calls
`create_index(...)` unconditionally at import time.

```powershell
docker compose -f src/deployment/docker/docker-compose.yml up echo_store echo-redis -d
$env:MONGODB_URI = "mongodb://root:root_password@localhost:27017/EchoNet?authSource=admin"
$env:MAIL_STARTTLS = "true"; $env:MAIL_SSL_TLS = "false"

cd src/production/backend
.venv\Scripts\python.exe -m pytest ../../tests/integration/backend/test_public_routes.py -v
```
**Expect: 2 passed.**

### Backend load test (Locust and k6)

```powershell
$env:PYTHONIOENCODING = "utf-8"   # see section 5 - required to even start the server
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 9000
```

In another terminal:

```powershell
# Locust
.venv\Scripts\python.exe -m locust -f src/tests/load/locustfile.py `
    --host http://127.0.0.1:9000 --headless -u 20 -r 5 -t 60s

# k6
k6 run src/tests/load/k6_loadtest.js
```

**Baseline (20 users/VUs, 60s): both ~0% failures, ~5ms median, ~9.7 req/s** - the two
tools agree closely at this load.

**Pushed further (10 → 1000 concurrent users)**, the system is clean up to ~200 users,
then a real capacity ceiling appears from ~300 users on (single uvicorn worker's
connection backlog) - **and Locust and k6 report that same ceiling completely
differently**: k6 surfaces it as explicit connection failures (1.88% → 5.36% as load
rises); Locust shows 0.00% failures throughout, and only its latency exposes the
identical problem (median jumps from ~6ms to 130ms at 1000 users). **Takeaway: a
0%-failure summary alone isn't proof of headroom - check latency too.** Full per-level
data and the code-verified root cause are in the report PDF.

### HMI unit test

```powershell
cd src/production/hmi/ui
npm test
```
**Expect: 19 passed** (17 pre-existing + 2 new, including the first-ever test for
`middleware/verifySignup.js`).

## 5. Known Gotchas

Quick reference - full explanations in the report PDF:

- Backend needs its own venv (section 3).
- MongoDB: use `localhost` + `authSource=admin`, not the container's internal hostname.
- `PYTHONIOENCODING=utf-8` is required to start the backend outside a real terminal -
  `app/main.py` prints a ✅ that crashes on Windows' default console codepage otherwise.
- `GET /insights/overview` (and `/species`) 500 in any real deployment - a env-var-name
  mismatch (`MONGO_URI` vs the rest of the app's `MONGODB_URI`), not a testing artifact.
  Excluded from the load-test target list for this reason.
- A live Gmail app password is committed in `app/routers/sim.py` - unrelated to
  testing, found while getting the app to import cleanly. Needs rotating.

## 6. Writing Your Own Tests From Here

- **Unit test** (Python): `src/tests/unit/<area>/test_*.py`, following
  `test_errors.py` - import the module directly, no server needed.
- **Integration/e2e** (Backend route): follow `test_public_routes.py` with
  `TestClient` - check whether the route touches Mongo/Redis first.
- **Unit test** (HMI): follow `verifySignup.test.mjs` - mock `req`/`res`/`next` by
  hand, add the file to `package.json`'s `test` script.
- **Load test**: add a new `@task` (Locust) or entry in `endpoints` (k6) for a new
  route - check auth/write requirements first, same as above.
- General TDD sequence for any of the above: write the test for the behaviour you
  want, confirm it fails for the right reason, make the smallest change that passes it.

## 7. What's Still Missing

- CI doesn't run any of this yet - `.github/workflows/docker-image.yml` only builds
  images and checks containers start. Adding a `pytest` job and a `node --test` job is
  the natural next step; everything above already runs cleanly outside CI.
- Backend has 2 tests (depth over breadth for a first example) and HMI has 3 files -
  expanding either is "writing your own tests from here," not a new pattern.
- The `/insights/*` bug and the `app/main.py` startup-encoding bug are real and still
  open (section 5) - not fixed as part of this task.
