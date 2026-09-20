from datetime import datetime, timezone

import pytest
from bson import ObjectId

from app import detections as detections_service


class Cursor:
    def __init__(self, documents):
        self.documents = list(documents)

    def sort(self, fields):
        for field, direction in reversed(fields):
            self.documents.sort(key=lambda document: document[field], reverse=direction == -1)
        return self

    def limit(self, count):
        self.documents = self.documents[:count]
        return self

    def __iter__(self):
        return iter(self.documents)


class Collection:
    def __init__(self, documents):
        self.documents = documents

    def find(self, query):
        return Cursor([document for document in self.documents if self._matches(document, query)])

    def _matches(self, document, query):
        for key, condition in query.items():
            if key == "$and" and not all(self._matches(document, item) for item in condition):
                return False
            if key == "$or" and not any(self._matches(document, item) for item in condition):
                return False
            if key.startswith("$"):
                continue
            value = document[key]
            if isinstance(condition, dict):
                if "$lt" in condition and not value < condition["$lt"]:
                    return False
                if "$gte" in condition and not value >= condition["$gte"]:
                    return False
                if "$lte" in condition and not value <= condition["$lte"]:
                    return False
            elif value != condition:
                return False
        return True


def detection(index, timestamp, species="Species A"):
    return {
        "_id": ObjectId(f"0000000000000000000000{index:02x}"),
        "timestamp": timestamp,
        "sensorId": f"sensor-{index}",
        "species": species,
        "microphoneLLA": [-38.8, 143.5, 10.0],
        "animalEstLLA": [-38.8, 143.5, 10.0],
        "animalTrueLLA": [-38.8, 143.5, 10.0],
        "animalLLAUncertainty": 1,
        "audioClip": "fixture-audio",
        "confidence": 90.0,
        "sampleRate": 32000,
    }


def test_cursor_pagination_has_no_duplicates_or_gaps(monkeypatch):
    same_timestamp = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    documents = [
        detection(1, same_timestamp),
        detection(2, same_timestamp),
        detection(3, same_timestamp),
        detection(4, datetime(2026, 9, 1, 11, tzinfo=timezone.utc)),
        detection(5, datetime(2026, 9, 1, 10, tzinfo=timezone.utc)),
    ]
    monkeypatch.setattr(detections_service, "Detections", Collection(documents))

    first_page = detections_service.list_detections_cursor(limit=2)
    second_page = detections_service.list_detections_cursor(
        limit=2, cursor=first_page["next_cursor"]
    )
    third_page = detections_service.list_detections_cursor(
        limit=2, cursor=second_page["next_cursor"]
    )

    returned_ids = [
        item.id
        for page in (first_page, second_page, third_page)
        for item in page["items"]
    ]
    returned_keys = [
        (item.timestamp, item.id)
        for page in (first_page, second_page, third_page)
        for item in page["items"]
    ]
    assert returned_keys == sorted(returned_keys, reverse=True)
    assert len(returned_ids) == len(set(returned_ids)) == len(documents)
    assert third_page["next_cursor"] is None


def test_cursor_pagination_keeps_filters(monkeypatch):
    timestamp = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(
        detections_service,
        "Detections",
        Collection([detection(1, timestamp), detection(2, timestamp, "Species B")]),
    )

    result = detections_service.list_detections_cursor(species="Species B", limit=20)

    assert [item.species for item in result["items"]] == ["Species B"]


def test_cursor_rejects_invalid_values():
    with pytest.raises(ValueError, match="Invalid cursor"):
        detections_service._decode_cursor("not-a-cursor")
