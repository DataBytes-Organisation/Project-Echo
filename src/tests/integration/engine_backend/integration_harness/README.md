# Engine-Backend Integration Test Harness

## Purpose

This Sprint 1 harness checks whether a Project Echo Engine detection can be sent to the Backend `POST /engine/event` endpoint. It also checks how the client reports invalid payload responses, timeouts, connection failures, Backend errors, and unexpected response codes.

The harness does not run model inference, MQTT, the simulator, or the HMI.

## Location and Structure

```text
src/tests/integration/engine_backend/integration_harness/
|-- client.py
|-- test_harness.py
|-- README.md
`-- fixtures/
    |-- valid_payloads/
    |   |-- valid_detection_01.json
    |   |-- valid_detection_02.json
    |   |-- valid_detection_03.json
    |   |-- valid_detection_04.json
    |   `-- valid_detection_05.json
    `-- invalid_payloads/
        |-- invalid_detection_01_missing_species.json
        |-- invalid_detection_02_confidence_out_of_range.json
        |-- invalid_detection_03_bad_location.json
        |-- invalid_detection_04_bad_timestamp.json
        `-- invalid_detection_05_empty_sensor_id.json
```

## Files

- `client.py` loads one JSON fixture and sends it to a configurable Backend URL with an explicit timeout.
- `test_harness.py` uses controlled HTTP mocks to test client response handling without Docker or MongoDB.
- `fixtures/valid_payloads/` contains five payloads expected to satisfy the current Backend schema.
- `fixtures/invalid_payloads/` contains five payloads that each break one current Backend validation rule.

## Current Backend Request Contract

The Sprint 1 executable contract is `EventSchema` in `src/production/backend/app/schemas.py`.

| Field | Current requirement |
|---|---|
| `timestamp` | Valid date and time |
| `sensorId` | Non-empty string |
| `species` | Non-empty string |
| `microphoneLLA` | Array containing exactly three numbers |
| `animalEstLLA` | Array containing exactly three numbers |
| `animalTrueLLA` | Array containing exactly three numbers |
| `animalLLAUncertainty` | Integer |
| `audioClip` | String |
| `confidence` | Number greater than 0 and less than 100 |
| `sampleRate` | Integer |

The expected successful response is HTTP `201 Created`.

An example valid request is:

```json
{
  "timestamp": "2026-08-06T10:15:30Z",
  "sensorId": "engine-test-sensor-01",
  "species": "Koala",
  "microphoneLLA": [-38.143, 144.361, 15.0],
  "animalEstLLA": [-38.142, 144.36, 15.0],
  "animalTrueLLA": [-38.142, 144.36, 15.0],
  "animalLLAUncertainty": 8,
  "audioClip": "VGVzdCBhdWRpbw==",
  "confidence": 96.42,
  "sampleRate": 48000
}
```

## Prerequisites

### Automated tests

- Python 3.10 or newer
- No third-party Python packages required
- Docker is not required

### Live Backend tests

- Docker Desktop running
- Project Echo Backend and MongoDB images built
- Local port `9000` available for the Backend

## Run the Automated Tests

Run this command from the Project Echo repository root:

```powershell
& C:\Users\maddi\AppData\Local\Microsoft\WindowsApps\python3.11.exe src\tests\integration\engine_backend\integration_harness\test_harness.py
```

Expected summary:

```text
Ran 6 tests
OK
```

These are simulated client tests. They mock HTTP responses and do not prove that the real Backend or MongoDB is running.

The successful-response test sends all five valid fixtures through the mocked client path. The invalid-response test sends all five invalid fixtures through a mocked HTTP `422` path.

## Start the Development Backend

From the repository root:

```powershell
cd src\deployment\docker
docker compose up -d echo_store echo_api
docker compose ps
cd ..\..\..
```

Confirm that the API documentation opens at `http://localhost:9000/docs` and displays `POST /engine/event`.

## Run One Live Valid Request

From the repository root:

```powershell
& C:\Users\maddi\AppData\Local\Microsoft\WindowsApps\python3.11.exe src\tests\integration\engine_backend\integration_harness\client.py
```

The client uses `fixtures/valid_payloads/valid_detection_01.json` by default and sends it to:

```text
http://localhost:9000/engine/event
```

Expected result:

```text
PASS: Backend returned HTTP 201
```

This live request should insert one test event into the development MongoDB database.

## Run One Live Invalid Request

From the repository root:

```powershell
& C:\Users\maddi\AppData\Local\Microsoft\WindowsApps\python3.11.exe src\tests\integration\engine_backend\integration_harness\client.py --payload src\tests\integration\engine_backend\integration_harness\fixtures\invalid_payloads\invalid_detection_01_missing_species.json
```

Expected result:

```text
FAIL: Backend returned HTTP 422
```

This failure is expected because the payload does not contain `species`.

## Optional Client Arguments

Use another Backend URL or timeout without editing source code:

```powershell
& C:\Users\maddi\AppData\Local\Microsoft\WindowsApps\python3.11.exe src\tests\integration\engine_backend\integration_harness\client.py --url http://localhost:9000/engine/event --timeout 5
```

### Select another fixture

Do not edit `client.py` to change the fixture. The path to `valid_detection_01.json` in `client.py` is only the default used when no `--payload` argument is supplied.

Select any other fixture by passing its complete path through `--payload`.

Valid case 2:

```powershell
& C:\Users\maddi\AppData\Local\Microsoft\WindowsApps\python3.11.exe src\tests\integration\engine_backend\integration_harness\client.py --payload src\tests\integration\engine_backend\integration_harness\fixtures\valid_payloads\valid_detection_02.json
```

Valid case 3:

```powershell
& C:\Users\maddi\AppData\Local\Microsoft\WindowsApps\python3.11.exe src\tests\integration\engine_backend\integration_harness\client.py --payload src\tests\integration\engine_backend\integration_harness\fixtures\valid_payloads\valid_detection_03.json
```

Valid case 4:

```powershell
& C:\Users\maddi\AppData\Local\Microsoft\WindowsApps\python3.11.exe src\tests\integration\engine_backend\integration_harness\client.py --payload src\tests\integration\engine_backend\integration_harness\fixtures\valid_payloads\valid_detection_04.json
```

Valid case 5:

```powershell
& C:\Users\maddi\AppData\Local\Microsoft\WindowsApps\python3.11.exe src\tests\integration\engine_backend\integration_harness\client.py --payload src\tests\integration\engine_backend\integration_harness\fixtures\valid_payloads\valid_detection_05.json
```

Invalid missing-species case:

```powershell
& C:\Users\maddi\AppData\Local\Microsoft\WindowsApps\python3.11.exe src\tests\integration\engine_backend\integration_harness\client.py --payload src\tests\integration\engine_backend\integration_harness\fixtures\invalid_payloads\invalid_detection_01_missing_species.json
```

Each live valid request creates a separate test record in the development MongoDB database. Running all five live cases is optional unless requested by the Engine or Backend lead.

## Fixture Coverage

### Valid payloads

| Fixture | Main variation |
|---|---|
| `valid_detection_01.json` | Koala, 48,000 Hz |
| `valid_detection_02.json` | Uperoleia mimula, 32,000 Hz |
| `valid_detection_03.json` | Sulphur-crested Cockatoo, timezone offset, 44,100 Hz |
| `valid_detection_04.json` | Australian Magpie, 22,050 Hz |
| `valid_detection_05.json` | Laughing Kookaburra, 16,000 Hz |

### Invalid payloads

| Fixture | Expected validation problem |
|---|---|
| `invalid_detection_01_missing_species.json` | Required `species` field is missing |
| `invalid_detection_02_confidence_out_of_range.json` | Confidence is exactly 100; current schema requires less than 100 |
| `invalid_detection_03_bad_location.json` | `microphoneLLA` contains two values instead of three |
| `invalid_detection_04_bad_timestamp.json` | Timestamp is not a valid date and time |
| `invalid_detection_05_empty_sensor_id.json` | `sensorId` is empty |

## Automated Validation Coverage

| Test group | Simulated expected result |
|---|---|
| Five valid payloads | HTTP `201` accepted |
| Five invalid payloads | HTTP `422` reported |
| Request timeout | Controlled timeout error |
| Backend unavailable | Controlled connection error |
| Backend server failure | HTTP `500` reported |
| Unexpected HTTP `200` | Contract failure reported |

The mocked tests verify client behaviour. At least one valid and one invalid request should also be tested against the real development Backend for Sprint 1 evidence.

## Live-Test Evidence to Record

After live testing, record:

- Test date
- Python version
- Backend URL
- Fixture filename
- Expected HTTP status
- Actual HTTP status
- Returned event ID for the valid request
- MongoDB storage confirmation
- Pass or fail result

Do not state that a live test passed until the request has actually been run against the development Backend.

## Current Interface Mismatches

1. `echo_engine_iot.py` reads `API_URL` from `echo_engine.json`, while `echo_engine.py` hardcodes the Backend URL.
2. The existing Engine sender does not set an explicit request timeout.
3. The existing Engine sender prints the response body but does not explicitly check for HTTP `201`.
4. The existing Engine sender does not catch timeout or connection errors locally.
5. The current Backend uses three-element arrays for locations; Anand's proposed interface uses named location objects.
6. The current Backend requires integer `animalLLAUncertainty`; the proposed interface shows a decimal value.
7. Proposed `status` and `error` fields are not part of the current stored-event schema.

## Sprint 2 Recommendations

1. Agree on one versioned Engine-Backend contract.
2. Clarify whether the inference response and stored detection event are separate contracts.
3. Standardise Backend URL configuration across Engine implementations.
4. Add an agreed production timeout, structured logging, and controlled error handling.
5. Decide whether real detections require `animalTrueLLA` and the complete Base64 audio clip.
6. Decide whether location uncertainty must support decimal values.
7. Evaluate bounded retries or queued delivery for temporary Backend outages.
8. Run an end-to-end test using a real model prediction.

## Troubleshooting

- `Payload file not found`: verify that the path includes `fixtures/valid_payloads/` or `fixtures/invalid_payloads/`.
- `Could not connect to Backend`: confirm Docker Desktop, `echo_store`, and `echo_api` are running.
- HTTP `422`: compare the fixture with `EventSchema` and read the response details.
- HTTP `500`: run `docker compose logs echo_api` and `docker compose logs echo_store` from `src/deployment/docker`.
- Timeout: confirm the Backend URL and check `http://localhost:9000/docs`.

Do not commit credentials, real sensitive recordings, generated `__pycache__` files, or unrelated model changes.
