# Testing & TDD Guide

**Researched and written for**: Project Echo Engine team, Sprint 2 (Trimester 2, 2026),
per Krish Warnakulasuriya's request to research and practically test 1-2 testing
frameworks (unit, integration/e2e, load/performance) and produce a short TDD guide for
future students, since the project currently does no TDD.

Everything in this guide was actually run against the real codebase while writing it -
every example, every command, every number. Where something didn't work on the first
try, that's noted too, because those are usually the parts worth knowing.

## 1. What's Actually Here Today (before this guide)

| Area | What exists | Gap |
|---|---|---|
| Backend (`src/production/backend`) | No tests at all | - |
| Engine (`src/production/engine`) | 7 files, `unittest.TestCase`, "44 tests / 46% coverage" per `TESTING.md` | No `pytest.ini` anywhere; never wired into CI; the coverage workflow was manual and undocumented in requirements |
| HMI (`src/production/hmi/ui`) | `node --test` via `package.json`'s `test` script, 2 files | Works fine, just needed more coverage |
| Load/performance testing | Nothing, anywhere | - |
| CI (`.github/workflows/docker-image.yml`) | Builds all Docker images, checks 9 containers report "Up" | A container-health smoke check, not application testing - doesn't run pytest, `unittest`, or `node --test` at all |

## 2. Frameworks Chosen, and Why Not the Others

- **pytest** (Backend + Engine unit/integration/e2e). It discovers and runs
  `unittest.TestCase`-based files natively - Engine's existing 44 tests needed **zero
  changes** to run under it (verified below). Jest wasn't chosen for the Python side
  because it's a JavaScript framework; it wasn't needed for HMI either, since
  `node --test` (Node's built-in runner, zero extra dependency) already does the job.
- **Locust** for load/performance testing, over k6 and Artillery. Load test scripts are
  plain Python (`locustfile.py`), matching the rest of the team's stack - no new
  language/DSL to learn, no separate binary to install system-wide.
- **HMI stays on `node --test`.** It already works (2 passing test files before this
  guide); introducing Jest would be a real migration cost for no concrete benefit at
  this project's current scale.

## 3. Setup

From the repo root:

```powershell
pip install -r requirements-dev.txt   # pytest, pytest-cov, locust
```

**Backend specifically needs its own virtual environment - don't install its
requirements into a shared/base Python environment.** Found out the hard way: Backend's
`requirements.txt` pins `pydantic<2.0` (via `fastapi_mail`), while the same machine's
base environment had other tools (`gradio`, `streamlit`) that need `pydantic>=2.7`.
Installing Backend's requirements into the base env downgraded `pydantic` and broke
those other tools' imports outright - not a hypothetical, this actually happened and
had to be reverted mid-session. Instead:

```powershell
cd src/production/backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install pytest pytest-cov locust httpx
```

One more pin needed on top of Backend's `requirements.txt`: it pins
`starlette==0.36.3`, which is incompatible with the newest `httpx` (`TestClient`
breaks with `TypeError: Client.__init__() got an unexpected keyword argument 'app'`).
Fix:

```powershell
.venv\Scripts\python.exe -m pip install "httpx<0.28,>=0.23"
```

Engine's dependencies (TensorFlow, librosa, etc.) were already present in this
project's base environment and didn't need a separate venv for the tests below - but
if you're setting this up fresh, give Engine its own venv too, for the same reason.

Root `pytest.ini` (new):

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

`src/tests/unit/backend` and `src/tests/integration/backend` are deliberately **not**
in this list. Found out why the hard way: Backend needs its own venv (previous
section), and a plain `pytest` invocation using whatever environment happens to be
active would otherwise try to collect those two folders too and fail with unrelated
dependency errors (`email-validator version >= 2.0 required` was the one hit here).
Run Backend's tests explicitly through its own venv instead - shown below.

Root `.coveragerc` (new) - without this, `pytest-cov` counts the test files themselves
as "covered" source, inflating the number:

```ini
[run]
omit =
    */test_*.py
    */.venv/*
    */node_modules/*
```

## 4. Running Everything

### Engine + repo-wide unit/integration tests (pytest)

```powershell
python -m pytest --cov=src/production/engine --cov-report=term-missing
```

Result when this guide was written: **61 passed**, `echo_engine.py` at **40%**
coverage (close to `TESTING.md`'s manually-reported 46% - the small difference is
expected, since that number came from a different, undocumented ad-hoc run).

**A real bug this surfaced**: running the full suite together (not each file
individually) failed one test -
`test_heldout_baseline.py::test_summarize_heldout_predictions_uses_only_shared_rows` -
with `ModuleNotFoundError: No module named 'sklearn.metrics'; 'sklearn' is not a
package`, even though `sklearn` is installed. Root cause:
`test_iot_integration.py` stubs `sys.modules["sklearn"] = MagicMock()` at module import
time (to avoid needing real TensorFlow/librosa/etc. for its own tests), and pytest
imports every test file during collection **before running any of them** - so the stub
was already in place before the heldout test ever got a chance to do a real
`from sklearn.metrics import ...`. This is exactly the kind of bug that stays invisible
until tests are unified under one runner (each file passed fine in isolation). Fixed in
`test_iot_integration.py` by restoring `sys.modules["sklearn"]` immediately after the
one import that needed it mocked, rather than never restoring it. Lesson: **global
state changes (`sys.modules`, monkeypatching a singleton, etc.) need matching cleanup,
or they leak into whatever test happens to run in the same process afterward.**

### Backend unit test (TDD red-green example)

`src/tests/unit/backend/test_errors.py` - first-ever tests for `app/errors.py`'s
`error_body()` (pure function, no Mongo/Redis needed):

```powershell
cd src/production/backend
.venv\Scripts\python.exe -m pytest ../../tests/unit/backend/test_errors.py -v
```

This file **is** the TDD example. One of its tests,
`test_locked_status_maps_to_locked_code`, was written and run *before* the
corresponding code change - it failed (`assert 'REQUEST_FAILED' == 'LOCKED'`, since
HTTP 423 wasn't in `STATUS_CODES` yet), which is exactly the point: 423 isn't raised by
any route today, but a future account-lockout feature (the two-factor auth flow is
already in this codebase) plausibly could raise it, and the fix costs one line:

```python
# before
STATUS_CODES = {..., 422: "VALIDATION_ERROR", 429: "RATE_LIMIT_EXCEEDED", ...}
# after (red -> green)
STATUS_CODES = {..., 422: "VALIDATION_ERROR", 423: "LOCKED", 429: "RATE_LIMIT_EXCEEDED", ...}
```

That's the whole TDD loop: **write the test for the behaviour you want, watch it fail
for the right reason, make the smallest change that passes it.**

### Backend integration/e2e test (FastAPI's `TestClient`)

`src/tests/integration/backend/test_public_routes.py` tests a real route
(`GET /public/public-test`) through the real app - routing, middleware, response
handling - without a live server process, using `fastapi.testclient.TestClient`.

**This needed more than expected to actually run.** `app.main` cannot be imported at
all without a *reachable* MongoDB - `app/database.py` calls
`SensorSettings.create_index(...)` unconditionally at import time (not lazily on first
real query, which is what a plain `pymongo.MongoClient()` construction alone would be).
So even a route that touches no database at all still needs Mongo up just to import the
app. In order:

```powershell
# 1. Start Mongo + Redis
docker compose -f src/deployment/docker/docker-compose.yml up echo_store echo-redis -d

# 2. Point at Mongo from the host (not from inside a container) - the container's
#    own hostname (ts-mongodb-cont, the default in app/database.py) only resolves
#    on the Docker network, not from your host machine. Also needs `authSource=admin`
#    - the root user was created in the admin database, not EchoNet.
$env:MONGODB_URI = "mongodb://root:root_password@localhost:27017/EchoNet?authSource=admin"

# 3. fastapi_mail's ConnectionConfig needs these even though they're not passed
#    explicitly in code (app/routers/sim.py) - it reads them from the environment.
#    Matches docker-compose.yml's echo_api service.
$env:MAIL_STARTTLS = "true"
$env:MAIL_SSL_TLS = "false"

cd src/production/backend
.venv\Scripts\python.exe -m pytest ../../tests/integration/backend/test_public_routes.py -v
```

Result: **2 passed** (both the happy-path GET and a 405-on-POST routing check).

`GET /public/public-test` was chosen deliberately - checked `app/routers/public.py`
first and confirmed it doesn't touch Mongo/Redis at all (unlike its neighbour
`GET /public/filter-data`, which queries `Events` and would need real seeded data to
test meaningfully).

### Backend load test (Locust) - a real run, not just config

```powershell
# Same env vars as above, plus this one (see the "bugs found" section - without it,
# the server crashes on startup when its output isn't a real interactive terminal):
$env:PYTHONIOENCODING = "utf-8"

.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 9000
```

In another terminal:

```powershell
cd src/production/backend
.venv\Scripts\python.exe -m locust -f ../../tests/load/locustfile.py `
    --host http://127.0.0.1:9000 --headless -u 20 -r 5 -t 60s --csv=locust_results
```

Real results from this run (20 simulated users, 60 seconds, all 4 endpoints
combined):

| Metric | Value |
|---|---|
| Total requests | 578 |
| Failures | 0 (0.00%) |
| Median response time | 5 ms |
| 95th percentile | 12 ms |
| Max response time | 26 ms |
| Requests/sec | ~9.8 |

Per-endpoint breakdown (median response time): `/public/public-test` 2ms,
`/hmi/microphones` 5ms, `/iot/nodes` 5ms, `/engine/animal_records` 11ms (this one is
CSV-serialising every record on every request - the slowest of the four, worth knowing
if it ever needs to handle real traffic).

### HMI unit test (`node --test`)

First-ever test for `middleware/verifySignup.js`'s `confirmPassword` - a small, pure
Express middleware (compares two request-body fields, no DB) that had zero coverage
before this:

```powershell
cd src/production/hmi/ui
npm test
```

Result: **19 passed** (17 pre-existing + 2 new). Same TDD spirit as the backend
example - writing the *first* test for previously-untested code is itself the valuable
step, whether or not it's phrased as red-green.

**Node.js wasn't installed on the machine this guide was written on at all** - installed
via `winget install OpenJS.NodeJS.LTS` to actually verify the test runs, rather than
writing it blind. If you're setting this up fresh and don't have Node, that's the
fastest way to get it.

## 5. Real Bugs This Work Surfaced (beyond the sklearn one above)

Testing found these; none were fixed here beyond what's noted, since fixing them
wasn't this task's scope - flagging them for whoever owns that code:

- **`GET /insights/overview` (and likely `/insights/species`) 500s in any real
  deployment.** `app/routers/insights.py` reads `os.getenv("MONGO_URI")` - a
  *different* env var name than every other file in the backend, which uses
  `MONGODB_URI` - with no fallback default. Unset (which it is everywhere today,
  including `docker-compose.yml`), it connects to Mongo with zero credentials and
  fails with `command find requires authentication`. This is why those two routes are
  excluded from the Locust target list above.
- **`app/main.py` crashes on startup whenever its stdout isn't a real interactive
  terminal** (redirected to a file, launched by a process manager, etc.) - it prints a
  ✅ character unconditionally at import time
  (`export_openapi_to_file()`), and Windows' default console codepage (cp1252) can't
  encode it: `UnicodeEncodeError: 'charmap' codec can't encode character '✅'`.
  Workaround used throughout this guide: `PYTHONIOENCODING=utf-8`. (The exact same
  class of bug - an emoji hitting cp1252 under output redirection - was independently
  found this sprint in a different script, `run_validation_experiment.py`; worth a
  project-wide search for `print(f"✅` / similar before it bites someone else.)
- A hardcoded Gmail app password is committed in `app/routers/sim.py`
  (`MAIL_PASSWORD="oocr srvw ndoj bwte"`) - a real secret in source control, unrelated
  to testing but found while getting the app to import cleanly. Worth rotating and
  moving to an env var/secret store regardless of this guide.

## 6. Writing Your Own Tests From Here

- **Unit test** (Python): put it under `src/tests/unit/<area>/test_*.py`, following
  `test_errors.py`'s pattern - import the module under test directly, no server needed.
  Run just that file the same way shown above.
- **Integration/e2e test** (Backend route): follow `test_public_routes.py` - use
  `TestClient`, and check whether your route touches Mongo/Redis first (like the
  Locust section explains) before assuming it'll run without them.
- **Unit test** (HMI): follow `verifySignup.test.mjs` - mock `req`/`res`/`next` by
  hand (no extra mocking library needed for something this small), add the new file to
  `package.json`'s `test` script.
- **Load test**: add a new `@task` method to `EchoApiUser` in `locustfile.py` for a new
  endpoint - check first whether it needs auth or writes real data, same caveats as
  the integration test section.
- Whenever you're adding a test for code that has none yet (which is most of this
  codebase right now), the TDD sequence is the same one used twice above: write the
  test for the behaviour you want first, confirm it fails for the reason you expect
  (not a typo or import error), then write the smallest change that makes it pass.

## 7. What's Still Missing

- CI doesn't run any of this yet - `.github/workflows/docker-image.yml` only builds
  images and checks containers start. A follow-up (not done here, out of this task's
  scope) would add a `pytest` job and a `node --test` job, both straightforward given
  everything above already runs cleanly outside CI.
- Coverage is only wired up for Engine in this guide; Backend has 2 tests total (by
  design - depth over breadth for a first example) and HMI has 3 files. Expanding
  either is exactly "writing your own tests from here" above, not a new pattern.
- The two `/insights/*` bugs and the `app/main.py` startup-encoding bug are real and
  still open - not fixed as part of this task (see section 5).
