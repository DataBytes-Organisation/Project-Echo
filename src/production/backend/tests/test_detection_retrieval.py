from datetime import datetime, timezone

from bson import ObjectId

from app import detections as detections_service


class FakeCursor:
    def __init__(self, documents):
        self.documents = list(documents)

    def sort(self, field, direction):
        reverse = direction == -1
        self.documents.sort(
            key=lambda doc: doc[field],
            reverse=reverse,
        )
        return self

    def skip(self, count):
        self.documents = self.documents[count:]
        return self

    def limit(self, count):
        self.documents = self.documents[:count]
        return self

    def __iter__(self):
        return iter(self.documents)


class FakeDetectionsCollection:
    def __init__(self, documents):
        self.documents = documents
        self.last_query = None

    def count_documents(self, query):
        self.last_query = query
        return len(self._filter(query))

    def find(self, query):
        self.last_query = query
        return FakeCursor(self._filter(query))

    def _filter(self, query):
        results = self.documents

        if "species" in query:
            results = [
                doc for doc in results
                if doc["species"] == query["species"]
            ]

        if "timestamp" in query:
            timestamp_filter = query["timestamp"]

            if "$gte" in timestamp_filter:
                results = [
                    doc for doc in results
                    if doc["timestamp"] >= timestamp_filter["$gte"]
                ]

            if "$lte" in timestamp_filter:
                results = [
                    doc for doc in results
                    if doc["timestamp"] <= timestamp_filter["$lte"]
                ]

        if "microphoneLLA.0" in query:
            lat_filter = query["microphoneLLA.0"]
            results = [
                doc for doc in results
                if lat_filter["$gte"]
                <= doc["microphoneLLA"][0]
                <= lat_filter["$lte"]
            ]

        if "microphoneLLA.1" in query:
            lon_filter = query["microphoneLLA.1"]
            results = [
                doc for doc in results
                if lon_filter["$gte"]
                <= doc["microphoneLLA"][1]
                <= lon_filter["$lte"]
            ]

        return list(results)


def make_detection(species, sensor_id, timestamp, lat=-38.8081, lon=143.5913):
    return {
        "_id": ObjectId(),
        "timestamp": timestamp,
        "sensorId": sensor_id,
        "species": species,
        "microphoneLLA": [lat, lon, 10.0],
        "animalEstLLA": [-38.8082, 143.5929, 4.6],
        "animalTrueLLA": [-38.8082, 143.5929, 10.0],
        "animalLLAUncertainty": 0,
        "audioClip": "fixture-audio",
        "confidence": 90.0,
        "sampleRate": 32000,
    }


def test_list_detections_filters_by_species(monkeypatch):
    documents = [
        make_detection(
            "Uperoleia mimula",
            "sensor-1",
            datetime(2026, 8, 31, 3, 0, tzinfo=timezone.utc),
        ),
        make_detection(
            "Sus Scrofa",
            "sensor-2",
            datetime(2026, 8, 31, 2, 0, tzinfo=timezone.utc),
        ),
    ]

    fake_collection = FakeDetectionsCollection(documents)
    monkeypatch.setattr(
        detections_service,
        "Detections",
        fake_collection,
    )

    result = detections_service.list_detections(
        species="Uperoleia mimula",
        page=1,
        page_size=20,
    )

    assert result["total"] == 1
    assert len(result["items"]) == 1
    assert result["items"][0].species == "Uperoleia mimula"


def test_list_detections_filters_by_time_range(monkeypatch):
    documents = [
        make_detection(
            "Species A",
            "sensor-1",
            datetime(2026, 8, 31, 1, 0, tzinfo=timezone.utc),
        ),
        make_detection(
            "Species B",
            "sensor-2",
            datetime(2026, 8, 31, 2, 0, tzinfo=timezone.utc),
        ),
        make_detection(
            "Species C",
            "sensor-3",
            datetime(2026, 8, 31, 3, 0, tzinfo=timezone.utc),
        ),
    ]

    fake_collection = FakeDetectionsCollection(documents)
    monkeypatch.setattr(
        detections_service,
        "Detections",
        fake_collection,
    )

    result = detections_service.list_detections(
        start_time=datetime(2026, 8, 31, 1, 30, tzinfo=timezone.utc),
        end_time=datetime(2026, 8, 31, 2, 30, tzinfo=timezone.utc),
    )

    assert result["total"] == 1
    assert result["items"][0].sensorId == "sensor-2"


def test_list_detections_paginates_and_sorts_newest_first(monkeypatch):
    documents = [
        make_detection(
            "Species A",
            "sensor-1",
            datetime(2026, 8, 31, 1, 0, tzinfo=timezone.utc),
        ),
        make_detection(
            "Species B",
            "sensor-2",
            datetime(2026, 8, 31, 3, 0, tzinfo=timezone.utc),
        ),
        make_detection(
            "Species C",
            "sensor-3",
            datetime(2026, 8, 31, 2, 0, tzinfo=timezone.utc),
        ),
    ]

    fake_collection = FakeDetectionsCollection(documents)
    monkeypatch.setattr(
        detections_service,
        "Detections",
        fake_collection,
    )

    result = detections_service.list_detections(
        page=1,
        page_size=2,
    )

    assert result["total"] == 3
    assert len(result["items"]) == 2
    assert result["items"][0].sensorId == "sensor-2"
    assert result["items"][1].sensorId == "sensor-3"


def test_list_detections_returns_empty_result(monkeypatch):
    documents = [
        make_detection(
            "Species A",
            "sensor-1",
            datetime(2026, 8, 31, 1, 0, tzinfo=timezone.utc),
        )
    ]

    fake_collection = FakeDetectionsCollection(documents)
    monkeypatch.setattr(
        detections_service,
        "Detections",
        fake_collection,
    )

    result = detections_service.list_detections(
        species="Does Not Exist",
    )

    assert result["total"] == 0
    assert result["items"] == []


def test_list_detections_applies_location_filter(monkeypatch):
    documents = [
        make_detection(
            "Species A",
            "nearby",
            datetime(2026, 8, 31, 1, 0, tzinfo=timezone.utc),
            lat=-38.8081,
            lon=143.5913,
        ),
        make_detection(
            "Species B",
            "far-away",
            datetime(2026, 8, 31, 2, 0, tzinfo=timezone.utc),
            lat=-37.0,
            lon=145.0,
        ),
    ]

    fake_collection = FakeDetectionsCollection(documents)
    monkeypatch.setattr(
        detections_service,
        "Detections",
        fake_collection,
    )

    result = detections_service.list_detections(
        lat=-38.8081,
        lon=143.5913,
        radius_km=5,
    )

    assert result["total"] == 1
    assert result["items"][0].sensorId == "nearby"
