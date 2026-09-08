"""Real detection HTTP contracts with Mongo and security boundaries faked."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bson import BSON, ObjectId
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import ValidationError

# Database import creates indexes; prevent all network activity at that boundary.
with patch("pymongo.MongoClient", return_value=MagicMock()):
    from app import schemas, serializers, detections as service
    from app.routers import engine, hmi, detections
from app.errors import (
    StandardizeErrorResponseMiddleware, http_exception_handler,
    validation_exception_handler, unhandled_exception_handler,
)


PAYLOAD = {
    "sourceType": "real", "timestamp": "2026-08-06T10:30:00Z", "sensorId": "esp32-001",
    "species": "Magpie", "confidence": 91.5, "microphoneLLA": [-37.8136, 144.9631, 0],
    "animalEstLLA": [10, 20, 0], "animalTrueLLA": [30, 40, 0],
    "animalLLAUncertainty": 5, "audioClip": "", "sampleRate": 0,
}


class MemoryCollection:
    """Small Mongo boundary fake; reject unsupported stages instead of silently passing."""
    def __init__(self, collections):
        self.documents = []
        self.collections = collections

    def insert_one(self, payload):
        doc = copy.deepcopy(payload)
        doc.setdefault("_id", ObjectId())
        self.documents.append(doc)
        return SimpleNamespace(inserted_id=doc["_id"])

    @staticmethod
    def matches(doc, query):
        for key, value in query.items():
            actual = doc
            for part in key.split("."):
                actual = actual[int(part)] if isinstance(actual, list) else actual.get(part)
            if isinstance(value, dict):
                for op, expected in value.items():
                    if op == "$gte" and actual < expected:
                        return False
                    if op == "$lte" and actual > expected:
                        return False
            elif actual != value:
                return False
        return True

    def find_one(self, query):
        return next((copy.deepcopy(doc) for doc in self.documents if self.matches(doc, query)), None)

    def count_documents(self, query):
        return len([doc for doc in self.documents if self.matches(doc, query)])

    def find(self, query):
        docs = [doc for doc in self.documents if self.matches(doc, query)]
        class Cursor(list):
            def sort(self, key, order):
                return Cursor(sorted(self, key=lambda doc: doc[key], reverse=order < 0))
            def skip(self, count):
                return Cursor(self[count:])
            def limit(self, count):
                return Cursor(self[:count])
        return Cursor(docs)

    def aggregate(self, pipeline):
        docs = self.run(copy.deepcopy(self.documents), pipeline)
        if any(len(BSON.encode(doc)) > 16 * 1024 * 1024 for doc in docs):
            raise RuntimeError("Mongo aggregation result exceeds BSON document size limit")
        return docs

    def run(self, docs, pipeline):
        for stage in pipeline:
            op, value = next(iter(stage.items()))
            if op == "$unionWith":
                name = value if isinstance(value, str) else value["coll"]
                docs += copy.deepcopy(self.collections[name].documents)
            elif op == "$match":
                docs = [doc for doc in docs if self.matches(doc, value)]
            elif op == "$sort":
                for key, order in reversed(list(value.items())):
                    docs.sort(key=lambda doc: doc[key], reverse=order < 0)
            elif op == "$skip":
                docs = docs[value:]
            elif op == "$limit":
                docs = docs[:value]
            elif op == "$count":
                docs = [{value: len(docs)}] if docs else []
            elif op == "$facet":
                docs = [{key: self.run(copy.deepcopy(docs), stages) for key, stages in value.items()}]
            else:
                raise AssertionError(f"Unsupported fake Mongo stage: {op}")
        return docs


class RealDetectionTests(unittest.TestCase):
    def setUp(self):
        collections = {}
        collections["events"] = self.events = MemoryCollection(collections)
        collections["detections"] = self.detections = MemoryCollection(collections)
        for target, name, value in [
            (engine, "Events", self.events), (hmi, "Events", self.events),
            (service, "Detections", self.detections),
            (detections, "enforce_and_consume", lambda *args, **kwargs: None),
        ]:
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        if hasattr(service, "Events"):
            patcher = patch.object(service, "Events", self.events)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch("app.middleware.pause_guard.get_service_state", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        app = FastAPI()
        app.add_exception_handler(HTTPException, http_exception_handler)
        app.add_exception_handler(RequestValidationError, validation_exception_handler)
        app.add_exception_handler(Exception, unhandled_exception_handler)
        app.add_middleware(StandardizeErrorResponseMiddleware)
        app.include_router(engine.router, prefix="/engine")
        app.include_router(detections.router)
        app.include_router(hmi.router, prefix="/hmi")
        self.app = app
        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def assert_contract(self, record):
        for key in ("sourceType", "species", "confidence", "sensorId", "microphoneLLA"):
            self.assertEqual(record.get(key), PAYLOAD[key], key)
        self.assertIn("2026-08-06T10:30:00", record["timestamp"])
        self.assertIsInstance(record["microphoneLLA"], list)

    def test_engine_ingest_is_persisted_once_and_readable_only_through_authenticated_hmi(self):
        created = self.client.post("/engine/event", json=PAYLOAD)
        self.assertEqual(created.status_code, 201, created.text)
        self.assert_contract(created.json())
        self.assertEqual(len(self.events.documents), 1)
        self.assertEqual(len(self.detections.documents), 0)
        # Engine events must not leak through the unauthenticated detection API.
        self.assertIsNone(service.get_detection(created.json()["_id"]))
        listing = self.client.get("/detections")
        self.assertEqual(listing.status_code, 200, listing.text)
        self.assertEqual(listing.json()["total"], 0)
        self.assertEqual(listing.json()["items"], [])
        self.app.dependency_overrides[hmi.jwtBearer] = lambda: "session-jwt"
        listing = self.client.get("/hmi/detections", headers={"Authorization": "Bearer session-jwt"})
        self.assertEqual(listing.status_code, 200, listing.text)
        self.assert_contract(listing.json()[0])

    def test_source_and_real_microphone_validation_rejects_bad_input_before_persistence(self):
        invalid = [None, [], [1, 2], [1, 2, 3, 4], ["1", 2, 3], [True, 2, 3],
                   [float("nan"), 2, 3], [1, float("inf"), 3], [1, 2, float("inf")],
                   [-91, 2, 3], [91, 2, 3], [1, -181, 3], [1, 181, 3]]
        for lla in invalid:
            with self.subTest(lla=lla), self.assertRaises(ValidationError):
                schemas.EventSchema(**{**PAYLOAD, "microphoneLLA": lla})
        for source in ("simulated", "REAL", "untrusted"):
            with self.subTest(source=source), self.assertRaises(ValidationError):
                schemas.DetectionCreate(**{**PAYLOAD, "sourceType": source})
        for endpoint in ("/engine/event", "/detections"):
            for data in ({key: value for key, value in PAYLOAD.items() if key != "microphoneLLA"},
                         {**PAYLOAD, "microphoneLLA": [91, 0, 0]}):
                response = self.client.post(endpoint, json=data)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")
        self.assertEqual(self.events.documents, [])
        self.assertEqual(self.detections.documents, [])

    def test_manual_and_engine_records_stay_in_their_own_read_paths(self):
        response = self.client.post("/detections", json=PAYLOAD)
        self.assertEqual(response.status_code, 200, response.text)
        self.assert_contract(response.json())
        self.assertEqual(self.client.post("/engine/event", json=PAYLOAD).status_code, 201)
        listing = self.client.get("/detections").json()
        self.assertEqual(listing["total"], 1)
        self.assert_contract(listing["items"][0])
        self.app.dependency_overrides[hmi.jwtBearer] = lambda: "session-jwt"
        hmi_items = self.client.get("/hmi/detections").json()
        self.assertEqual(len(hmi_items), 1)
        self.assert_contract(hmi_items[0])

    def test_hmi_detection_read_requires_authentication_and_empty_is_not_seeded(self):
        response = self.client.get("/hmi/detections")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "FORBIDDEN")
        self.app.dependency_overrides[hmi.jwtBearer] = lambda: "session-jwt"
        self.assertEqual(self.client.get("/hmi/detections").json(), [])

    def test_legacy_hmi_serializer_preserves_real_source_and_structured_location(self):
        event = {**PAYLOAD, "_id": ObjectId()}
        for serializer in (serializers.eventEntity, serializers.eventSpeciesEntity):
            self.assert_contract(serializer(event))

    def test_real_coordinate_boundary_values_are_valid_floats(self):
        for lla in ([-90, -180, 0], [90, 180, -1], [0, 0, 10]):
            event = schemas.EventSchema(**{**PAYLOAD, "microphoneLLA": lla})
            self.assertTrue(all(isinstance(value, float) for value in event.microphoneLLA))

    def test_audio_bearing_page_does_not_exceed_mongo_document_limit(self):
        for _ in range(2):
            event = schemas.EventSchema(**{**PAYLOAD, "audioClip": "a" * (9 * 1024 * 1024)})
            self.events.insert_one(event.dict())
        self.app.dependency_overrides[hmi.jwtBearer] = lambda: "session-jwt"
        response = self.client.get("/hmi/detections")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()), 2)
        self.assertNotIn("audioClip", response.json()[0])

    def test_engine_to_backend_to_authenticated_hmi_marker_with_boundary_fakes(self):
        production = Path(__file__).resolve().parents[3]
        # Isolate Engine's existing heavy-dependency mocks from Backend imports.
        capture = subprocess.run([sys.executable, "-c", '''
import contextlib, io, json
with contextlib.redirect_stdout(io.StringIO()):
    from test_iot_integration import EchoEngine, _make_msg, _valid_payload
    from unittest.mock import patch
    engine = EchoEngine()
    engine.config['API_URL'] = 'http://backend.test/engine/event'
    with patch('echo_engine.requests.post') as post:
        engine.on_iot_message(None, None, _make_msg(_valid_payload(
            type='prediction', species='Magpie', confidence=91.5,
            timestamp='2026-08-06T10:30:00Z', sensor_id='esp32-001')))
    payload = post.call_args.kwargs['json']
print(json.dumps(payload))
'''], cwd=production / "engine", capture_output=True, text=True, check=True)
        response = self.client.post("/engine/event", json=json.loads(capture.stdout))
        self.assertEqual(response.status_code, 201, response.text)
        self.app.dependency_overrides[hmi.jwtBearer] = lambda: "session-jwt"
        read = self.client.get("/hmi/detections")
        self.assertEqual(read.status_code, 200, read.text)
        result = subprocess.run(["node", "tests/detection-pipeline.mjs"],
                                cwd=production / "hmi" / "ui", input=read.text,
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["sourceType"], "real")


if __name__ == "__main__":
    unittest.main()
