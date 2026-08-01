import mongomock
import pymongo
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_mongo_api(monkeypatch):
    """Load FastAPI only after replacing PyMongo with an in-memory client."""
    monkeypatch.setattr(pymongo, "MongoClient", mongomock.MongoClient)

    from app.database import Events
    from app.main import app

    assert isinstance(Events.database.client, mongomock.MongoClient)

    Events.delete_many({})
    with TestClient(app) as client:
        yield client, Events
    Events.delete_many({})
