"""
End-to-end test (C2.4): prove that a detection posted through the API
can actually be retrieved back out, using the real running Backend + MongoDB.

Why /detections and not /engine/event:
    The task guide's example posts to /engine/event and then reads it back
    from /detections. In this codebase those two write to two separate,
    disconnected MongoDB collections ('events' vs 'detections') -- see
    docs/architecture/Backend_Dependency_Map.md, section 2. Posting to one
    will never appear in the other, so that pairing can't be tested as an
    ingestion-to-retrieval loop here.

    /engine/event's own ingestion contract is already covered by
    src/tests/integration/engine_backend/integration_harness/ (5 valid +
    5 invalid payloads). That harness does not test retrieval at all. This
    file tests the retrieval half using a pair of endpoints -- POST
    /detections and GET /detections -- that genuinely share one collection.

Requires:
    - Docker Desktop running
    - echo_store and echo_api containers started:
        cd src/deployment/docker
        docker-compose up -d echo_store echo_api

Run with:
    pytest src/tests/integration/detections_api/test_ingestion_to_retrieval.py -v

Note: test_posted_detection_appears_in_list is marked xfail (expected to
fail) -- it caught a real, pre-existing bug in the list endpoint. See that
test's docstring/marker for the exact cause. This is not a bug in the test.
"""
import os
from datetime import datetime, timezone

import pytest
import requests

BACKEND_URL = os.getenv("ECHO_BACKEND_URL", "http://localhost:9000")


@pytest.fixture(scope="module", autouse=True)
def _require_live_backend():
    """Skip with a clear message if the Backend isn't reachable, instead of
    every test failing with a confusing connection-error traceback."""
    try:
        requests.get(f"{BACKEND_URL}/", timeout=3)
    except requests.exceptions.ConnectionError:
        pytest.skip(
            f"Backend not reachable at {BACKEND_URL}. Start it first: "
            "cd src/deployment/docker && docker-compose up -d echo_store echo_api"
        )


@pytest.fixture(scope="module", autouse=True)
def _ensure_detections_budget(_require_live_backend):
    """
    Depends on _require_live_backend (as a parameter, not just definition
    order) so that a skip there always runs first and this fixture's own
    request is never attempted against an unreachable Backend.

    /detections is protected by a monthly usage budget (app/services/budget.py).
    With no budget configured the limit defaults to zero and every request is
    rejected with 403 -- so this sets a generous limit first, the same way an
    admin would via POST /api/admin/budget/limits.
    """
    response = requests.post(
        f"{BACKEND_URL}/api/admin/budget/limits",
        json=[{"service": "detections", "monthly_limit": 100000}],
        timeout=5,
    )
    assert response.status_code == 200, (
        f"Could not configure the detections budget before testing: "
        f"{response.status_code} {response.text}"
    )


def _sample_detection_payload():
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sensorId": "e2e-test-sensor",
        "species": "Koala",
        "microphoneLLA": [-38.143, 144.361, 15.0],
        "animalEstLLA": [-38.142, 144.360, 15.0],
        "animalTrueLLA": [-38.142, 144.360, 15.0],
        "animalLLAUncertainty": 8,
        "audioClip": "VGVzdCBhdWRpbw==",
        "confidence": 91.5,
        "sampleRate": 48000,
    }


def test_post_then_get_detection_by_id():
    """The core end-to-end check: post a detection, then read that same one back."""
    payload = _sample_detection_payload()

    # --- Ingestion ---
    post_response = requests.post(f"{BACKEND_URL}/detections", json=payload, timeout=5)
    assert post_response.status_code == 200, (
        f"POST /detections failed: {post_response.status_code} {post_response.text}"
    )
    created = post_response.json()
    # NOTE: the API returns the Mongo id as "_id", not "id" -- confirmed by
    # inspecting a real response rather than assuming the field name.
    detection_id = created["_id"]
    assert created["species"] == payload["species"]
    assert created["sensorId"] == payload["sensorId"]

    # --- Retrieval by id ---
    get_response = requests.get(f"{BACKEND_URL}/detections/{detection_id}", timeout=5)
    assert get_response.status_code == 200, (
        f"GET /detections/{{id}} failed: {get_response.status_code} {get_response.text}"
    )
    fetched = get_response.json()
    assert fetched["_id"] == detection_id
    assert fetched["species"] == payload["species"]
    assert fetched["confidence"] == payload["confidence"]


@pytest.mark.xfail(
    reason=(
        "Known pre-existing bug, not caused by this test: app/schemas.py defines the "
        "Detection class twice. Python keeps only the second definition (line 369), "
        "which has a typo -- 'class config' instead of 'class Config' -- so Pydantic "
        "silently ignores its json_encoders = {ObjectId: str} setting. "
        "DetectionListResponses (line 398) has the identical typo. As a result GET "
        "/detections (the list endpoint) raises HTTP 500 whenever it has to serialize "
        "at least one result, even though GET /detections/{id} for a single item works "
        "fine. Remove this xfail once schemas.py is fixed to use 'class Config'."
    ),
    strict=True,
)
def test_posted_detection_appears_in_list():
    """Matches the task guide's "GET /detections, assert the recently posted
    event exists in the response array" step. Currently fails -- see reason above."""
    payload = _sample_detection_payload()

    post_response = requests.post(f"{BACKEND_URL}/detections", json=payload, timeout=5)
    assert post_response.status_code == 200
    detection_id = post_response.json()["_id"]

    list_response = requests.get(
        f"{BACKEND_URL}/detections",
        params={"species": payload["species"], "page_size": 100},
        timeout=5,
    )
    assert list_response.status_code == 200, (
        f"GET /detections failed: {list_response.status_code} {list_response.text}"
    )
    returned_ids = [item["_id"] for item in list_response.json()["items"]]
    assert detection_id in returned_ids, (
        "The detection just created did not appear in GET /detections results"
    )
