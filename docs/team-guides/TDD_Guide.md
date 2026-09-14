# Testing & TDD Guide

Project Echo currently has no unified test tooling and does not practice TDD. This
guide sets up **pytest** (Backend and Engine unit, integration, and e2e tests) and
**Locust plus k6** (load testing), and shows how to run and extend each one. Every
command here was actually run against the real codebase. The full findings, bugs, and
detailed load-testing analysis are in `Testing_Framework_Research_Report_Nolan_Nguyen.pdf`
(same folder as this guide's source task); this file is just the short, practical
how-to.

## 1. What Exists Today

| Area | Tests today | Notes |
|---|---|---|
| Backend | None | New in this guide |
| Engine | 44 tests (`unittest`), no runner config | Now unified under pytest, zero changes needed |
| HMI | `node --test`, 2 files | Unchanged, extended with 1 new test |
| Load/performance | None | New in this guide (Locust + k6) |
| CI | Docker container-health check only | Doesn't run any of the above yet |

## 2. Frameworks and Why

- **pytest**: discovers and runs Engine's existing `unittest.TestCase` files
  natively, so there is zero migration cost, and adds fixtures and coverage on top.
  Not Jest, since that is the wrong language for Backend and Engine.
- **Locust and k6**: both were practically tested and compared (see the report for
  the full head to head), following a direct request to evaluate these two
  specifically for load testing. Recommendation is **Locust**, since its scripts
  stay in Python like the rest of the stack. k6 is an equally valid pick though,
  especially for teams that want more explicit failure signalling under stress (see
  section 4).
- **HMI stays on `node --test`**, since it already works and there is no migration
  benefit at this scale.

## 3. Setup

```powershell
pip install -r requirements-dev.txt   # pytest, pytest-cov, locust
```

**Backend needs its own virtual environment**, because its `requirements.txt` pins
`pydantic<2.0`, which conflicts with other tools in a shared or base Python
environment. This was verified: it broke `gradio` and `streamlit` when tried.

```powershell
cd src/production/backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install pytest pytest-cov locust "httpx<0.28,>=0.23"
```

(`httpx<0.28` is required, because the newest `httpx` breaks `TestClient` against
this project's pinned `starlette==0.36.3`.)

k6 is a separate binary, not a pip package:

```powershell
winget install k6
```

Root `pytest.ini` (new) deliberately excludes `src/tests/unit/backend` and
`src/tests/integration/backend`. This is because Backend needs its own venv as
described above, and a plain `pytest` run in another environment would fail on
unrelated missing dependencies:

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

Root `.coveragerc` (new). Without it, `pytest-cov` counts the test files themselves
as covered, which inflates the number:

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
**Expect: 61 passed**, about 40% coverage on `echo_engine.py`.

### Backend unit test (TDD example)

```powershell
cd src/production/backend
.venv\Scripts\python.exe -m pytest ../../tests/unit/backend/test_errors.py -v
```
**Expect: 6 passed.** `test_locked_status_maps_to_locked_code` is a real
red-to-green example. It failed before `app/errors.py` got a one-line fix
(`423: "LOCKED"`). That is the TDD loop: write the test for the behaviour you want,
watch it fail for the right reason, then make the smallest change that passes it.

### Backend integration/e2e test (`TestClient`)

Needs MongoDB to be reachable even for a route that touches no data, because
`app/database.py` calls `create_index(...)` unconditionally at import time.

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
$env:PYTHONIOENCODING = "utf-8"   # see section 5, required to even start the server
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

**Baseline (20 users or VUs, 60 seconds): both show about 0% failures, about 5ms
median latency, and about 9.7 requests per second.** The two tools agree closely at
this load.

**Pushed further, from 10 up to 1000 concurrent users**, the system stays clean up
to about 200 users. From about 300 users on, a real capacity ceiling appears,
caused by a single uvicorn worker's connection backlog. **Locust and k6 report that
same ceiling in completely different ways.** k6 surfaces it as explicit connection
failures, rising from 1.88% to 5.36% as load increases. Locust shows 0.00% failures
throughout, and only its latency numbers expose the same problem, with median
latency jumping from about 6ms to 130ms at 1000 users. **Takeaway: a 0% failure
summary alone is not proof of headroom, so latency should be checked too.** Full
per-level data and the code-verified root cause are in the report PDF.

### HMI unit test

```powershell
cd src/production/hmi/ui
npm test
```
**Expect: 19 passed** (17 pre-existing plus 2 new, including the first-ever test
for `middleware/verifySignup.js`).

## 5. Known Gotchas

Quick reference. Full explanations are in the report PDF:

- Backend needs its own venv (section 3).
- MongoDB: use `localhost` together with `authSource=admin`, not the container's
  internal hostname.
- `PYTHONIOENCODING=utf-8` is required to start the backend outside a real
  terminal, because `app/main.py` prints a checkmark character that otherwise
  crashes on Windows' default console codepage.
- `GET /insights/overview` (and `/species`) return 500 errors in any real
  deployment. This is caused by an env-var name mismatch (`MONGO_URI` versus the
  rest of the app's `MONGODB_URI`), not a testing artifact. It was excluded from
  the load-test target list for this reason.
- A live Gmail app password is committed in `app/routers/sim.py`. This is
  unrelated to testing and was found while getting the app to import cleanly. It
  needs to be rotated.

## 6. Writing Your Own Tests From Here

- **Unit test** (Python): `src/tests/unit/<area>/test_*.py`, following the pattern
  in `test_errors.py`. Import the module directly, no server needed.
- **Integration/e2e** (Backend route): follow `test_public_routes.py` using
  `TestClient`. Check whether the route touches Mongo or Redis first.
- **Unit test** (HMI): follow `verifySignup.test.mjs`. Mock `req`, `res`, and
  `next` by hand, then add the file to `package.json`'s `test` script.
- **Load test**: add a new `@task` (Locust) or entry in `endpoints` (k6) for a new
  route. Check auth and write requirements first, same as above.
- General TDD sequence for any of the above: write the test for the behaviour you
  want, confirm it fails for the right reason, make the smallest change that
  passes it.

## 7. What's Still Missing

- CI does not run any of this yet. `.github/workflows/docker-image.yml` only
  builds images and checks that containers start. Adding a `pytest` job and a
  `node --test` job is the natural next step, since everything above already runs
  cleanly outside CI.
- Backend has 2 tests (depth over breadth for a first example) and HMI has 3
  files. Expanding either one is just "writing your own tests from here," not a
  new pattern.
- The `/insights/*` bug and the `app/main.py` startup-encoding bug are real and
  still open (see section 5). They were not fixed as part of this task.
