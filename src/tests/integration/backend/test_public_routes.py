"""Integration/e2e test for the backend's public router, using FastAPI's
TestClient - simulates a real HTTP request against the real app (routing,
middleware, response handling) inside the test process, no live server or
Docker needed.

GET /public/public-test is deliberately chosen: it's the one route in this
app that touches neither MongoDB nor Redis (see app/routers/public.py) -
pymongo's MongoClient doesn't connect eagerly at import time, but any route
that actually queries the database (e.g. GET /public/filter-data) would hang
or fail here without a live Mongo instance. See docs/team-guides/TDD_Guide.md
for which routes are safe to test this way vs which need the full
docker-compose stack (used by the Locust load test instead).

Note: importing app.main has a side effect - app/main.py calls
export_openapi_to_file() unconditionally at import time, which writes
backend/project-echo-openapi.json relative to the current working directory
(gitignored - see .gitignore).
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[4] / "src" / "production" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


class TestPublicRoutes:
    def test_public_test_route_returns_active_message(self):
        response = client.get("/public/public-test")
        assert response.status_code == 200
        assert response.json() == {"message": "Public router is active!"}

    def test_public_test_route_rejects_post(self):
        # Confirms only GET is wired for this route - a real routing check,
        # not something a unit test of a bare function could catch.
        response = client.post("/public/public-test")
        assert response.status_code == 405
