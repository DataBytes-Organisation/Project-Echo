from datetime import datetime, timezone

from app.routers import insights


def test_parse_ts_accepts_datetime_and_iso_strings():
    dt = datetime(2026, 7, 29, 10, 0, 0, tzinfo=timezone.utc)

    assert insights._parse_ts(dt) == dt
    assert insights._parse_ts("2026-07-29T10:00:00Z") == dt
    assert insights._parse_ts("2026-07-29T10:00:00+00:00") == dt
    assert insights._parse_ts("not-a-date") is None
    assert insights._parse_ts(None) is None


def test_to_iso_string_handles_datetime_and_string_without_crashing():
    dt = datetime(2026, 7, 29, 10, 0, 0, tzinfo=timezone.utc)

    assert insights._to_iso_string(dt) == "2026-07-29T10:00:00Z"
    assert insights._to_iso_string("2026-07-29T11:00:00Z") == "2026-07-29T11:00:00Z"
    # Unsupported type must not raise (this was the PR review failure mode).
    assert insights._to_iso_string(12345) is None
    assert insights._to_iso_string(None) is None
    assert insights._to_iso_string("already-a-string-value") == "already-a-string-value"


def test_overview_returns_valid_response_for_datetime_and_string_timestamps(monkeypatch):
    class FakeCollection:
        def __init__(self, docs):
            self.docs = docs

        def count_documents(self, *_args, **_kwargs):
            return 0

        def aggregate(self, pipeline):
            # Execute the important parts of the pipeline in-memory so mixed
            # timestamp storage is covered without requiring a live MongoDB.
            rows = list(self.docs)
            for stage in pipeline:
                if "$addFields" in stage:
                    field = stage["$addFields"]["_normalizedTs"]
                    assert "$switch" in field
                    normalized = []
                    for row in rows:
                        ts = row.get("timestamp")
                        parsed = insights._parse_ts(ts)
                        copy = dict(row)
                        copy["_normalizedTs"] = parsed
                        normalized.append(copy)
                    rows = normalized
                elif "$match" in stage:
                    match = stage["$match"]
                    ts_filter = match.get("_normalizedTs", {})
                    filtered = []
                    for row in rows:
                        ts = row.get("_normalizedTs")
                        if ts is None:
                            continue
                        if "$gte" in ts_filter and ts < ts_filter["$gte"]:
                            continue
                        if "$lte" in ts_filter and ts > ts_filter["$lte"]:
                            continue
                        if "species" in match and row.get("species") != match["species"]:
                            continue
                        if "sensorId" in match and row.get("sensorId") != match["sensorId"]:
                            continue
                        filtered.append(row)
                    rows = filtered
                elif "$group" in stage:
                    if not rows:
                        return []
                    species = sorted({row.get("species") for row in rows if row.get("species")})
                    sensors = sorted({row.get("sensorId") for row in rows if row.get("sensorId")})
                    timestamps = [row["_normalizedTs"] for row in rows]
                    return [
                        {
                            "_id": None,
                            "detections": len(rows),
                            "species": species,
                            "sensors": sensors,
                            "minTs": min(timestamps),
                            "maxTs": max(timestamps),
                        }
                    ]
            return []

    mixed_events = FakeCollection(
        [
            {
                "timestamp": datetime(2026, 7, 29, 10, 0, 0, tzinfo=timezone.utc),
                "species": "Sus scrofa",
                "sensorId": "B2.3-SENSOR-A",
            },
            {
                "timestamp": "2026-07-29T11:00:00Z",
                "species": "Koala",
                "sensorId": "B2.3-SENSOR-B",
            },
            {
                # Unsupported timestamp must be skipped, not crash the request.
                "timestamp": {"bad": "shape"},
                "species": "IgnoreMe",
                "sensorId": "B2.3-SENSOR-C",
            },
        ]
    )

    monkeypatch.setattr(insights, "Events", mixed_events)
    monkeypatch.setattr(insights, "Microphones", FakeCollection([]))
    monkeypatch.setattr(insights, "Nodes", FakeCollection([]))

    unfiltered = insights.insights_overview()
    assert unfiltered["counts"]["detections"] == 2
    assert unfiltered["counts"]["uniqueSpecies"] == 2
    assert unfiltered["timeRange"]["start"] is not None
    assert unfiltered["timeRange"]["end"] is not None
    # Must be JSON-serializable strings, never raw datetime-only assumption crashes.
    assert isinstance(unfiltered["timeRange"]["start"], str)
    assert isinstance(unfiltered["timeRange"]["end"], str)

    filtered = insights.insights_overview(
        start="2026-07-29T10:30:00Z",
        end="2026-07-29T12:00:00Z",
        species="Koala",
        sensorId="B2.3-SENSOR-B",
    )
    assert filtered["counts"]["detections"] == 1
    assert filtered["counts"]["uniqueSpecies"] == 1
    assert filtered["counts"]["sensorsWithDetections"] == 1


def test_species_endpoint_accepts_filters_with_mixed_timestamps(monkeypatch):
    class FakeCollection:
        def __init__(self, docs):
            self.docs = docs

        def aggregate(self, pipeline):
            rows = [dict(doc) for doc in self.docs]
            for stage in pipeline:
                if "$addFields" in stage:
                    for row in rows:
                        row["_normalizedTs"] = insights._parse_ts(row.get("timestamp"))
                elif "$match" in stage:
                    match = stage["$match"]
                    ts_filter = match.get("_normalizedTs", {})
                    filtered = []
                    for row in rows:
                        ts = row.get("_normalizedTs")
                        if ts is None:
                            continue
                        if "$gte" in ts_filter and ts < ts_filter["$gte"]:
                            continue
                        if "$lte" in ts_filter and ts > ts_filter["$lte"]:
                            continue
                        species_match = match.get("species")
                        if isinstance(species_match, dict):
                            if not row.get("species"):
                                continue
                        elif species_match and row.get("species") != species_match:
                            continue
                        if "sensorId" in match and row.get("sensorId") != match["sensorId"]:
                            continue
                        filtered.append(row)
                    rows = filtered
                elif "$group" in stage:
                    grouped = {}
                    for row in rows:
                        key = row.get("species")
                        bucket = grouped.setdefault(
                            key,
                            {"species": key, "count": 0, "sum_confidence": 0.0},
                        )
                        bucket["count"] += 1
                        bucket["sum_confidence"] += float(row.get("confidence") or 0.0)
                    rows = [
                        {
                            "species": value["species"],
                            "count": value["count"],
                            "avg_confidence": value["sum_confidence"] / value["count"],
                        }
                        for value in grouped.values()
                    ]
                elif "$sort" in stage:
                    rows = sorted(rows, key=lambda item: item.get("count", 0), reverse=True)
                elif "$limit" in stage:
                    rows = rows[: stage["$limit"]]
            return rows

    mixed_events = FakeCollection(
        [
            {
                "timestamp": datetime(2026, 7, 29, 10, 0, 0, tzinfo=timezone.utc),
                "species": "Sus scrofa",
                "sensorId": "B2.3-SENSOR-A",
                "confidence": 90.0,
            },
            {
                "timestamp": "2026-07-29T11:00:00Z",
                "species": "Koala",
                "sensorId": "B2.3-SENSOR-B",
                "confidence": 88.0,
            },
        ]
    )

    monkeypatch.setattr(insights, "Events", mixed_events)

    result = insights.insights_species(
        start="2026-07-29T00:00:00Z",
        end="2026-07-30T00:00:00Z",
        limit=10,
    )
    species_names = {item["species"] for item in result["items"]}
    assert species_names == {"Sus scrofa", "Koala"}

    filtered = insights.insights_species(
        start="2026-07-29T10:30:00Z",
        end="2026-07-29T12:00:00Z",
        species="Koala",
        sensorId="B2.3-SENSOR-B",
        limit=10,
    )
    assert len(filtered["items"]) == 1
    assert filtered["items"][0]["species"] == "Koala"
