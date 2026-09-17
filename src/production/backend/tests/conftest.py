"""
C2.3 MongoDB mocking for the tests/ package.

`app.database` opens a real MongoClient as soon as it is imported. Pytest
imports every test module during collection, which happens *before* any
fixture runs. If another file does `from app.main import app` at module
level (for example `test_schema_validation.py`), the real client is created
first and a later monkeypatch cannot replace the already-bound collections.

This conftest therefore:
1. Sets required Settings / mail env vars before any app import.
2. Ensures an asyncio event loop exists before modules create asyncio.Lock().
3. Patches `pymongo.MongoClient` at import time so every collected test in
   this folder receives mongomock when `pytest tests/` runs as a suite.
"""

import asyncio
import os

# Required by app.config.Settings before app.main / app.database can import.
os.environ.setdefault(
    "MONGODB_URI",
    "mongodb://mock-user:mock-pass@localhost:27017/EchoNet",
)
os.environ.setdefault(
    "USER_MONGODB_URI",
    "mongodb://mock-user:mock-pass@localhost:27017/UserSample?authSource=admin",
)
os.environ.setdefault("JWT_SECRET", "c2.3-test-only-jwt-secret")
# Keep engine webhook open for local API tests unless a test opts in.
os.environ.setdefault("ENGINE_API_KEY", "")
# fastapi-mail ConnectionConfig rejects None; provide harmless placeholders.
os.environ.setdefault("MAIL_USERNAME", "c23-mock@example.com")
os.environ.setdefault("MAIL_PASSWORD", "c23-mock-password")
os.environ.setdefault("MAIL_FROM", "c23-mock@example.com")

# Python 3.9: asyncio.Lock() during import needs a current event loop.
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import mongomock
import pymongo
import pytest
from fastapi.testclient import TestClient

# Import-time patch: must run before test modules are collected/imported.
pymongo.MongoClient = mongomock.MongoClient


def pytest_configure(config):
    """Re-apply the patch early in the pytest lifecycle as a safety net."""
    pymongo.MongoClient = mongomock.MongoClient


@pytest.fixture
def mock_mongo_api():
    """Provide a TestClient wired to in-memory mongomock collections."""
    from app.database import Events
    from app.main import app

    assert isinstance(Events.database.client, mongomock.MongoClient), (
        "Expected mongomock, but app.database still has a real Mongo client. "
        "The C2.3 patch must run before any test module imports the app."
    )

    Events.delete_many({})
    with TestClient(app) as client:
        yield client, Events
    Events.delete_many({})
